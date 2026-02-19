#!/usr/bin/env python3
"""
SAM_hardmask_classifier.py

Purpose
-------
Run SAM-proposal-based inference/evaluation for the **HardMean / HardMask (“hardmean”) classifier**.

This script consumes a SAM proposal file (.pth) (generated on ImageNet-V1 or ImageNet-V2)
and evaluates a classifier that was trained with a **hard background-masking pipeline**
instead of crop-and-zoom. In particular, for each SAM proposal box we:

  1) Take the FULL image (no object zoom).
  2) Replace pixels *outside* the SAM box with ImageNet mean color (background “hard masking”).
  3) Apply the standard ImageNet test transform: Resize(256) → CenterCrop(224).
  4) Normalize with ImageNet mean/std and run the classifier.

This makes the inference-time input distribution match the HardMean/HardMask training setup:
the model sees the object at approximately the same scale as full-image training, with the
background suppressed.

What this script is / is not
----------------------------
- This IS an evaluator: SAM proposals → HardMean/HardMask classifier → accuracy metrics.
- This is NOT proposal generation: the proposals must already exist in the input .pth.
- This is NOT detector evaluation: there is no IoU comparison to GT boxes; we measure whether
  proposals enable correct classification.

Inputs
------
1) --proposals_file (.pth)
   Produced by SAMproposal_ImageNet_V1.py or SAMproposal_ImageNet_V2.py.
   Supported formats:
     A) {"proposals": {sample_id -> sample_data}, "metadata": {...}}
     B) {sample_id -> sample_data}

   Each sample_data should include:
     - image_path (str): path to original RGB image
     - gt_label (int): ground-truth class for evaluation
       NOTE: For ImageNet-V2 with numeric folder labels ("0".."999"), label correction should
             already have been performed during proposal generation.
     - proposals (list): list of dicts with at least:
         * bbox_xywh = [x, y, w, h]
         * score (optional): used for sorting / topN selection (defaults to 1.0)

2) --model_path
   Path to classifier weights (standard ResNet50 head). If empty, script may fall back
   to torchvision pretrained (depending on your loader logic).

Proposal Filtering (GT-free)
----------------------------
Before masking/inference, proposals are filtered using geometry-only rules:
  - xywh → xyxy conversion
  - optional clamp to image bounds
  - min_size threshold on bbox width/height
  - optional max_area_ratio cap to drop huge boxes
  - NMS de-dup among proposals (dedup_iou, default 0.90)
  - sort by proposal score (desc) and cap to topN (default 20)

Coverage Accounting
-------------------
- Images with 0 valid proposals after filtering are still included in the denominator.
- Coverage = fraction of images with ≥1 valid proposal.
This prevents accuracy inflation by skipping “no proposal” failures.

Inference Details (HardMean / HardMask)
---------------------------------------
For each remaining proposal:
- Build a binary mask = 1 inside box, 0 outside.
- Compose: masked_image = mask*image + (1-mask)*IMAGENET_MEAN_RGB (in uint8).
- Apply Resize(256) → CenterCrop(224) on the *masked full image*.
- Convert to float in [0,1], apply ImageNet normalization (mean/std), then classify.

Metrics Reported
----------------
Let N be the number of valid proposals after filtering/topN.
- Any:          correct if GT is top-1 for *any* masked proposal
- Any@k:        correct if GT is top-1 within first k proposals (k list configurable)
- Top1_best:    pick proposal with max softmax confidence; check if its top-1 == GT
- MeanProbs:    mean softmax probabilities across proposals → argmax
- Voting:       majority vote over per-proposal argmax predictions
- WeightedVote: confidence-weighted vote over per-proposal argmax predictions

Outputs
-------
Writes --out_json containing:
  - config (thresholds, topN, any@k list, etc.)
  - totals (num_images, coverage)
  - metrics (Any, Any@k, Top1_best, MeanProbs, Voting, WeightedVote)
  - optional debug stats (num masked proposals, failures)

Example
-------
python SAM_hardmean_classifier.py \
  --proposals_file sam_v2_relaxed_2k_final.pth \
  --model_path /path/to/hardmean_checkpoint.pth \
  --model_type hardmask \
  --out_json sam_v2_hardmean_results.json
"""


import os
import json
import math
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.transforms import functional as TVF

try:
    from torchvision.ops import nms as tv_nms
except Exception:
    tv_nms = None

# ─── REPRODUCIBILITY ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
# ImageNet Mean in RGB (0-255) for HardMasking
IMAGENET_MEAN_RGB = [124, 116, 104] 

# ─── CONFIG ───────────────────────────────────────────────────────────────────

@dataclass
class EvalConfig:
    device: str = "cuda"
    input_size: int = 224
    min_size: int = 8
    clamp_to_image: bool = True
    dedup_iou: float = 0.90
    apply_area_cap: bool = False
    max_area_ratio: float = 0.70
    topN: int = 20
    any_k_list: Tuple[int, ...] = (5, 10, 20, 50, 300)
    batch_size: int = 128
    torch_deterministic: bool = True
    model_type: str = "standard"  # 'standard', 'focl', 'hardmask'


# ─── CLASSIFIER LOADER ────────────────────────────────────────────────────────

def load_classifier(model_path: str, num_classes: int, device: torch.device):
    """Load ResNet-50 classifier. If no path, use torchvision pretrained."""
    if model_path and os.path.exists(model_path):
        model = models.resnet50(pretrained=False, num_classes=num_classes).to(device)
        sd = torch.load(model_path, map_location=device)
        
        # Handle DDP prefixes if present
        if list(sd.keys())[0].startswith("module."):
            new_sd = {k[7:]: v for k, v in sd.items()}
            sd = new_sd
            
        model.load_state_dict(sd)
        print(f"Loaded classifier from: {model_path}")
    else:
        print("Using torchvision pretrained ImageNet ResNet-50")
        model = models.resnet50(pretrained=True, num_classes=num_classes).to(device)
    return model.eval()


# ─── GEOMETRY HELPERS ─────────────────────────────────────────────────────────

def _xywh_to_xyxy(box_xywh):
    x, y, w, h = box_xywh
    return [x, y, x + w, y + h]

def _clip(v, lo, hi):
    return max(lo, min(hi, v))

def clamp_box_to_image(box_xyxy, W, H):
    x1, y1, x2, y2 = box_xyxy
    x1 = _clip(x1, 0, W)
    x2 = _clip(x2, 0, W)
    y1 = _clip(y1, 0, H)
    y2 = _clip(y2, 0, H)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def box_area_xyxy(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    if tv_nms is not None:
        return tv_nms(boxes, scores, iou_thresh)
    # Fallback NMS
    idxs = scores.argsort(descending=True)
    keep = []
    while idxs.numel():
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        i_box = boxes[i].unsqueeze(0)
        rest = boxes[idxs[1:]]
        xx1 = torch.maximum(i_box[:, 0], rest[:, 0])
        yy1 = torch.maximum(i_box[:, 1], rest[:, 1])
        xx2 = torch.minimum(i_box[:, 2], rest[:, 2])
        yy2 = torch.minimum(i_box[:, 3], rest[:, 3])
        inter_w = torch.clamp(xx2 - xx1, min=0)
        inter_h = torch.clamp(yy2 - yy1, min=0)
        inter = inter_w * inter_h
        area_i = (i_box[:, 2] - i_box[:, 0]) * (i_box[:, 3] - i_box[:, 1])
        area_r = (rest[:, 2] - rest[:, 0]) * (rest[:, 3] - rest[:, 1])
        union = area_i + area_r - inter + 1e-12
        iou = (inter / union).squeeze(0)
        idxs = idxs[1:][iou <= iou_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# ─── IMAGE CONVERSIONS ────────────────────────────────────────────────────────

def pil_to_chw_uint8(img_pil: Image.Image) -> torch.Tensor:
    arr = np.array(img_pil.convert("RGB"), dtype=np.uint8)  # HWC
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW, uint8

# ─── LOGIC 1: STANDARD / FOCL (CROP & ZOOM) ───────────────────────────────────

def crop_and_resize(img_chw_uint8: torch.Tensor, box_xyxy: List[float], size: int) -> Optional[torch.Tensor]:
    """
    Standard Logic: Crop the object, then resize to target size (Zoom in).
    """
    _, H, W = img_chw_uint8.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    
    # Ensure standard bounds
    if x2 <= x1 or y2 <= y1: return None
    x1 = max(0, min(W - 1, x1))
    x2 = max(1, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(1, min(H, y2))
    
    crop = img_chw_uint8[:, y1:y2, x1:x2]
    if crop.numel() == 0 or crop.shape[1] < 1 or crop.shape[2] < 1:
        return None
        
    crop = crop.unsqueeze(0).float() / 255.0  # [1,C,H,W], [0,1]
    # Bilinear resize to 224x224
    crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
    return crop.squeeze(0)  # [C,S,S]

# ─── LOGIC 2: HARDMASK (MASK & NO ZOOM) ──────────────────────────────────────

def mask_and_process_hardmask(img_chw_uint8: torch.Tensor, box_xyxy: List[float], target_size: int = 224) -> Optional[torch.Tensor]:
    """
    HardMask Logic:
    1. Take FULL image.
    2. Mask out background (pixels outside box) with Mean Color.
    3. Apply Standard Test Transform: Resize(256) -> CenterCrop(224).
    """
    _, H, W = img_chw_uint8.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    
    # Check valid box
    if x2 <= x1 or y2 <= y1: return None
    x1 = max(0, min(W, x1))
    x2 = max(0, min(W, x2))
    y1 = max(0, min(H, y1))
    y2 = max(0, min(H, y2))
    
    # Create Mask (1 inside box, 0 outside)
    # img_chw_uint8 is [3, H, W]
    mask = torch.zeros((H, W), dtype=torch.uint8, device=img_chw_uint8.device)
    mask[y1:y2, x1:x2] = 1
    
    # Create Mean Tensor
    # IMAGENET_MEAN_RGB is [124, 116, 104]. Expand to [3, H, W]
    mean_tensor = torch.tensor(IMAGENET_MEAN_RGB, dtype=torch.uint8, device=img_chw_uint8.device).view(3, 1, 1)
    
    # Composite: mask * img + (1-mask) * mean
    # We cast to int for math, then back to uint8
    img_masked = img_chw_uint8 * mask + mean_tensor * (1 - mask)
    
    # Now Apply Standard Test Transform (Resize 256 -> CenterCrop 224)
    # We use TorchVision Functional (TVF) which expects [C,H,W]
    
    # 1. Resize (Resize short edge to 256, maintain aspect)
    img_resized = TVF.resize(img_masked, size=256, interpolation=transforms.InterpolationMode.BILINEAR)
    
    # 2. CenterCrop (224)
    img_cropped = TVF.center_crop(img_resized, output_size=[target_size, target_size])
    
    # Convert to float [0,1]
    return img_cropped.float() / 255.0


# ─── FILTERING PIPELINE ───────────────────────────────────────────────────────

def filter_and_dedup_proposals(
    proposals: List[Dict],
    W: int,
    H: int,
    cfg: EvalConfig,
    cap: bool = True
) -> List[Dict]:
    """Shared geometry-based filters (no GT usage)."""
    img_area = W * H
    boxes_xyxy, scores = [], []
    for p in proposals:
        b = _xywh_to_xyxy(p["bbox_xywh"])
        if cfg.clamp_to_image:
            b = clamp_box_to_image(b, W, H)
            if b is None:
                continue
        w = b[2] - b[0]
        h = b[3] - b[1]
        if w < cfg.min_size or h < cfg.min_size:
            continue
        if cfg.apply_area_cap and (box_area_xyxy(b) / max(1, img_area) > cfg.max_area_ratio):
            continue
        boxes_xyxy.append(b)
        scores.append(float(p.get("score", 1.0)))

    if len(boxes_xyxy) == 0:
        return []

    # NMS dedup among proposals
    boxes_t = torch.tensor(boxes_xyxy, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = nms_xyxy(boxes_t, scores_t, cfg.dedup_iou).cpu().numpy().tolist()
    boxes_xyxy = [boxes_xyxy[i] for i in keep]
    scores = [scores[i] for i in keep]

    # Sort and Cap
    order = np.argsort(scores)[::-1]
    boxes_xyxy = [boxes_xyxy[i] for i in order]
    scores = [scores[i] for i in order]
    if cap and cfg.topN is not None and cfg.topN > 0:
        boxes_xyxy = boxes_xyxy[:cfg.topN]
        scores = scores[:cfg.topN]

    return [{"bbox_xyxy": b, "score": s} for b, s in zip(boxes_xyxy, scores)]


# ─── METRIC UTILS ─────────────────────────────────────────────────────────────

def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=-1)

def majority_vote(preds: List[int]) -> int:
    if not preds:
        return -1
    vals, counts = np.unique(np.array(preds), return_counts=True)
    return int(vals[np.argmax(counts)])

def weighted_vote(logits: torch.Tensor) -> int:
    probs = softmax_probs(logits)  # [N,C]
    conf, pred = probs.max(dim=1)
    C = probs.shape[1]
    scores = torch.zeros(C, device=probs.device)
    for i in range(probs.shape[0]):
        scores[pred[i]] += conf[i]
    return int(scores.argmax().item())


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description="Reviewer-safe SAM→Classifier evaluation")
    ap.add_argument('--proposals_file', type=str, required=True, help='Path to SAM proposals .pth')
    ap.add_argument('--model_path', type=str, default='', help='Path to classifier weights')
    
    # Key switch for logic
    ap.add_argument('--model_type', type=str, choices=['standard', 'focl', 'hardmask'], default='standard',
                    help="Logic switch: 'standard'/'focl' use Crop+Zoom. 'hardmask' uses Mask+NoZoom.")

    # Filtering / evaluation controls
    ap.add_argument('--min_size', type=int, default=8)
    ap.add_argument('--dedup_iou', type=float, default=0.90)
    ap.add_argument('--area_cap', type=str, default='False', help='True/False to enable area cap')
    ap.add_argument('--max_area_ratio', type=float, default=0.70)
    ap.add_argument('--topN', type=int, default=20)
    ap.add_argument('--any_k', type=str, default='5,10,20,50,300', help='Comma-separated Any@k list')
    ap.add_argument('--batch_size', type=int, default=128)

    # Output
    ap.add_argument('--out_json', type=str, default='sam_eval_clean.json')

    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model Type: {args.model_type.upper()}") 

    # Build config
    any_k_list = tuple(int(x.strip()) for x in args.any_k.split(',') if x.strip())
    cfg = EvalConfig(
        device=str(device),
        input_size=224,
        min_size=args.min_size,
        clamp_to_image=True,
        dedup_iou=args.dedup_iou,
        apply_area_cap=(args.area_cap.lower() == 'true'),
        max_area_ratio=args.max_area_ratio,
        topN=args.topN,
        any_k_list=any_k_list,
        batch_size=args.batch_size,
        torch_deterministic=True,
        model_type=args.model_type
    )
    print("Config:", cfg)

    # Load proposals
    print(f"Loading SAM proposals from: {args.proposals_file}")
    data = torch.load(args.proposals_file, map_location='cpu')
    if isinstance(data, dict) and 'proposals' in data:
        proposals_data = data['proposals']
    else:
        proposals_data = data

    print(f"Total samples available: {len(proposals_data)}")

    # Load classifier (ImageNet-1K)
    classifier = load_classifier(args.model_path, num_classes=1000, device=device)

    # Transforms (Mean/Std for Normalization)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]

    # Metric accumulators
    total_images = 0
    correct_any = 0
    correct_top1_best = 0
    correct_mean = 0
    correct_voting = 0
    correct_weighted = 0

    # Any@k
    any_at_k_hits: Dict[int, int] = {k: 0 for k in cfg.any_k_list}

    # Coverage & stats
    coverage_count = 0
    post_dedup_pre_cap_counts: List[int] = []
    post_cap_counts: List[int] = []
    crop_failures = 0
    total_crops = 0

    # Iterate samples
    for sample_id, sample_data in proposals_data.items():
        total_images += 1

        image_path = sample_data["image_path"]
        gt_label = int(sample_data["gt_label"])
        proposals = sample_data.get("proposals", [])

        # Load ORIGINAL image
        image = Image.open(image_path).convert("RGB")
        img_chw_u8 = pil_to_chw_uint8(image).to(device) # Move to GPU for masking logic
        _, H, W = img_chw_u8.shape

        # Filter proposals
        props = filter_and_dedup_proposals(proposals, W, H, cfg, cap=True)
        
        # Stats logging (skipping the pre-cap check for brevity in inference loop)
        post_cap_counts.append(len(props))
        if len(props) > 0:
            coverage_count += 1
        else:
            continue # Miss

        # Generate Input Batch (Branching Logic)
        crops = []
        for p in props:
            if cfg.model_type == 'hardmask':
                # HardMask: Mask -> Full Image Resize (No Zoom)
                c = mask_and_process_hardmask(img_chw_u8, p["bbox_xyxy"], target_size=cfg.input_size)
            else:
                # Standard/FocL: Crop -> Zoom
                c = crop_and_resize(img_chw_u8, p["bbox_xyxy"], size=cfg.input_size)
            
            if c is None:
                crop_failures += 1
                continue
            crops.append(c)

        if len(crops) == 0:
            continue

        # Stack, Normalize, Inference
        batch = torch.stack(crops, dim=0).to(device) # [N, 3, 224, 224]
        total_crops += batch.shape[0]
        batch = (batch - mean) / std

        with torch.no_grad():
            logits = classifier(batch)
            probs = softmax_probs(logits)
            conf, pred = probs.max(dim=1)

        # Metrics (Identical to before)
        # Any@k
        for k in cfg.any_k_list:
            k_eff = min(k, pred.shape[0])
            hit_k = (pred[:k_eff] == gt_label).any().item()
            any_at_k_hits[k] += int(hit_k)

        correct_any += int((pred == gt_label).any().item())

        # Top-1 (Best Confidence)
        best_idx = int(conf.argmax().item())
        correct_top1_best += int(int(pred[best_idx].item()) == gt_label)

        # MeanProbs
        mean_probs = probs.mean(dim=0)
        correct_mean += int(int(mean_probs.argmax().item()) == gt_label)

        # Voting
        correct_voting += int(majority_vote([int(x) for x in pred.cpu().numpy()]) == gt_label)

        # Weighted Voting
        correct_weighted += int(weighted_vote(logits) == gt_label)

    # Finalize
    denom = max(1, total_images)
    results = {
        "config": asdict(cfg),
        "totals": {
            "images": total_images,
            "coverage": coverage_count / denom,
        },
        "metrics": {
            "Any": correct_any / denom,
            "Top1_best_crop": correct_top1_best / denom,
            "MeanProbs": correct_mean / denom,
            "Voting": correct_voting / denom,
            "VotingWeighted": correct_weighted / denom,
            "Any@k": {k: any_at_k_hits[k] / denom for k in cfg.any_k_list},
        },
        "model": args.model_type
    }

    # Pretty print
    print(f"\n=== {args.model_type.upper()} EVALUATION ===")
    print(f"Images: {results['totals']['images']}")
    print(f"Top-1 (Best): {results['metrics']['Top1_best_crop']:.4f}")
    print(f"Voting: {results['metrics']['Voting']:.4f}")
    print(f"Any: {results['metrics']['Any']:.4f}")
    # Save JSON
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.out_json}")

if __name__ == "__main__":
    main()