#!/usr/bin/env python3
"""
SAM_classifier_inference.py

Purpose
-------
Run *classifier inference/evaluation* using SAM2 proposal boxes as foveated crops.
This script **consumes** a saved SAM proposal file (.pth) and evaluates a classifier
(standard ResNet50 or a Foc-L variant) on the proposal crops to compute robust
"proposal-based" accuracies.

This is a unified inference script:
- Works for **ImageNet-V1 or ImageNet-V2** because it loads each image from the
  stored `image_path` inside the proposals file (no dataset loader needed).
- Works for **standard vs Foc-L classifiers** by swapping `--model_path` (and
  optionally `--model_type` for logging).

Inputs
------
1) Proposals file (.pth) produced by SAM proposal generation scripts:
   - V1 generator (ImageNet-1k val): SAMproposal_ImageNet_V1.py
   - V2 generator (ImageNet-V2 matched-frequency): SAMproposal_ImageNet_V2.py

   Expected format (either supported):
   A) {"proposals": {sample_id -> sample_data}, "metadata": {...}}
   B) {sample_id -> sample_data}

   Each `sample_data` is expected to contain:
     - image_path (str): absolute or relative path to the original image
     - gt_label (int): ground-truth label to evaluate against
       NOTE: For ImageNet-V2, proposal generation should already correct labels
             if folder names are numeric strings ("0".."999").
     - proposals (list): list of proposal dicts, each with at least:
         * bbox_xywh = [x, y, w, h]
         * score (optional): proposal confidence used for sorting (falls back
           to predicted_iou if stored under another key)

2) Classifier checkpoint:
   - --model_path: path to a ResNet50 checkpoint (standard or Foc-L trained)
   - --model_type: optional string tag written into output JSON ("standard", "focl", ...)

Inference / Evaluation Pipeline (per image)
-------------------------------------------
1) Load the original image from `image_path`.
2) Validate + filter proposals (geometry-only; does NOT use GT):
   - convert xywh -> xyxy
   - clamp to image bounds
   - drop tiny boxes (min_size)
   - optionally drop very large boxes (max_area_ratio)
   - deduplicate using NMS (dedup_iou, default ~0.90)
   - sort by proposal score and keep topN (default 20)
3) Crop each surviving proposal from the image:
   - crop in pixel space
   - resize to 224x224
   - convert to float tensor in [0,1]
   - apply ImageNet normalization (mean/std)
4) Run classifier on the batch of crops.

Aggregations / Metrics
----------------------
This script reports multiple “ways of using proposals”:
- Any:        correct if GT is top-1 for *any* crop
- Any@k:      same but restricted to first k proposals (k list configurable)
- BestCrop:   choose crop with max confidence; correct if that crop matches GT
- MeanProbs:  average softmax probabilities across crops, then argmax
- Voting:     majority vote over per-crop argmaxes
- WeightedVote: confidence-weighted voting over per-crop argmaxes

It also reports:
- Coverage: fraction of images with >=1 valid proposal after filtering
- Proposal count stats (before/after filtering and topN cap)

Outputs
-------
Writes a JSON file (--out_json) containing:
- run configuration (paths, thresholds, topN, etc.)
- totals (num_images, coverage)
- metric values (Any, Any@k, BestCrop, MeanProbs, Voting, WeightedVote)
- proposal statistics and basic runtime info

Important Notes / Pitfalls
--------------------------
- V1/V2 handling: this script is dataset-agnostic; correctness depends on the
  proposals file. For ImageNet-V2 with numeric folder names, ensure the V2 proposal
  generator stores a *corrected* gt_label.
- This script assumes proposals are **candidate boxes** and evaluates whether a
  classifier can recognize GT from at least one proposal crop. It is not a detector
  evaluation (no IoU vs GT boxes here).
- If no valid proposals remain for an image, the image still counts in the denominator
  (coverage < 1.0). This prevents inflation by skipping hard failures.

Example Usage
-------------
V1 proposals + Standard model:
  python SAM_classifier_inference.py \
      --proposals_file sam_v1_relaxed_2k.pth \
      --model_path /path/to/standard_resnet50.pth \
      --model_type standard \
      --out_json results_sam_v1_standard.json

V2 proposals + Foc-L model:
  python SAM_classifier_inference.py \
      --proposals_file sam_v2_relaxed_2k.pth \
      --model_path /path/to/focl_resnet50.pth \
      --model_type focl \
      --out_json results_sam_v2_focl.json
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
from torchvision import datasets, transforms, models

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


# ─── CLASSIFIER LOADER (matches your Enhanced script) ─────────────────────────

def load_classifier(model_path: str, num_classes: int, device: torch.device):
    """Load ResNet-50 classifier. If no path, use torchvision pretrained."""
    if model_path and os.path.exists(model_path):
        model = models.resnet50(pretrained=False, num_classes=num_classes).to(device)
        sd = torch.load(model_path, map_location=device)
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


# ─── IMAGE CONVERSIONS & CROP ─────────────────────────────────────────────────

def pil_to_chw_uint8(img_pil: Image.Image) -> torch.Tensor:
    arr = np.array(img_pil.convert("RGB"), dtype=np.uint8)  # HWC
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW, uint8

def crop_and_resize(img_chw_uint8: torch.Tensor, box_xyxy: List[float], size: int) -> Optional[torch.Tensor]:
    _, H, W = img_chw_uint8.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0, min(W - 1, x1))
    x2 = max(1, min(W, x2))
    y1 = max(0, min(H - 1, y1))
    y2 = max(1, min(H, y2))
    crop = img_chw_uint8[:, y1:y2, x1:x2]
    if crop.numel() == 0 or crop.shape[1] < 1 or crop.shape[2] < 1:
        return None
    crop = crop.unsqueeze(0).float() / 255.0  # [1,C,H,W], [0,1]
    crop = F.interpolate(crop, size=(size, size), mode="bilinear", align_corners=False)
    return crop.squeeze(0)  # [C,S,S]


# ─── FILTERING PIPELINE (shared across models) ─────────────────────────────────

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

    # NMS dedup among proposals (not vs GT)
    boxes_t = torch.tensor(boxes_xyxy, dtype=torch.float32)
    scores_t = torch.tensor(scores, dtype=torch.float32)
    keep = nms_xyxy(boxes_t, scores_t, cfg.dedup_iou).cpu().numpy().tolist()
    boxes_xyxy = [boxes_xyxy[i] for i in keep]
    scores = [scores[i] for i in keep]

    # sort by score desc and (optionally) cap to topN
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
    ap.add_argument('--model_path', type=str, default='', help='Path to classifier weights (optional)')
    ap.add_argument('--model_type', type=str, choices=['standard', 'focl'], default='standard')

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
        torch_deterministic=True
    )
    print("Config:", cfg)

    # Load proposals
    print(f"Loading SAM proposals from: {args.proposals_file}")
    data = torch.load(args.proposals_file, map_location='cpu')
    if isinstance(data, dict) and 'proposals' in data:
        proposals_data = data['proposals']
        print("Loaded dict with key 'proposals'")
    else:
        proposals_data = data
        print("Loaded direct proposals dict")

    print(f"Total samples available: {len(proposals_data)}")

    # Load classifier (ImageNet-1K)
    classifier = load_classifier(args.model_path, num_classes=1000, device=device)

    # Transforms
    to_tensor = transforms.ToTensor()
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

    # Debug counters
    crop_failures = 0
    total_crops = 0

    # Iterate samples
    for sample_id, sample_data in proposals_data.items():
        total_images += 1

        image_path = sample_data["image_path"]
        gt_label = int(sample_data["gt_label"])
        proposals = sample_data.get("proposals", [])

        # Load ORIGINAL image and convert once
        image = Image.open(image_path).convert("RGB")
        img_chw_u8 = pil_to_chw_uint8(image)
        _, H, W = img_chw_u8.shape

        # Two-pass filtering to log stats:
        props_post_dedup_pre_cap = filter_and_dedup_proposals(
            proposals, W, H, cfg, cap=False
        )
        post_dedup_pre_cap_counts.append(len(props_post_dedup_pre_cap))

        props = filter_and_dedup_proposals(
            proposals, W, H, cfg, cap=True
        )
        post_cap_counts.append(len(props))

        if len(props) > 0:
            coverage_count += 1

        # If zero valid proposals → still counted in denominator, mark misses across metrics
        if len(props) == 0:
            continue

        # Crop all valid proposals
        crops = []
        for p in props:
            c = crop_and_resize(img_chw_u8, p["bbox_xyxy"], cfg.input_size)
            if c is None:
                crop_failures += 1
                continue
            crops.append(c)
        if len(crops) == 0:
            continue

        # Build batch and normalize
        batch = torch.stack(crops, dim=0).to(device)  # [N,3,224,224] in [0,1]
        total_crops += batch.shape[0]
        batch = (batch - mean) / std

        with torch.no_grad():
            logits = classifier(batch)  # [N,1000]
            probs = softmax_probs(logits)
            conf, pred = probs.max(dim=1)  # [N], [N]

        # Any@k curve
        for k in cfg.any_k_list:
            k_eff = min(k, pred.shape[0])
            hit_k = (pred[:k_eff] == gt_label).any().item()
            any_at_k_hits[k] += int(hit_k)

        # Any (same as Any@N)
        hit_any = (pred == gt_label).any().item()
        correct_any += int(hit_any)

        # Top-1(best-crop): choose by max confidence
        best_idx = int(conf.argmax().item())
        correct_top1_best += int(int(pred[best_idx].item()) == gt_label)

        # MeanProbs
        mean_probs = probs.mean(dim=0)
        correct_mean += int(int(mean_probs.argmax().item()) == gt_label)

        # Voting (majority over per-crop argmax)
        correct_voting += int(majority_vote([int(x) for x in pred.cpu().numpy()]) == gt_label)

        # Weighted voting (confidence-weighted)
        correct_weighted += int(weighted_vote(logits.to(device)) == gt_label)

    # Finalize metrics
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
        "proposal_stats": {
            "post_dedup_pre_cap": {
                "min": int(np.min(post_dedup_pre_cap_counts)) if post_dedup_pre_cap_counts else 0,
                "median": float(np.median(post_dedup_pre_cap_counts)) if post_dedup_pre_cap_counts else 0.0,
                "mean": float(np.mean(post_dedup_pre_cap_counts)) if post_dedup_pre_cap_counts else 0.0,
                "max": int(np.max(post_dedup_pre_cap_counts)) if post_dedup_pre_cap_counts else 0,
            },
            "post_cap": {
                "min": int(np.min(post_cap_counts)) if post_cap_counts else 0,
                "median": float(np.median(post_cap_counts)) if post_cap_counts else 0.0,
                "mean": float(np.mean(post_cap_counts)) if post_cap_counts else 0.0,
                "max": int(np.max(post_cap_counts)) if post_cap_counts else 0,
            },
        },
        "debug": {
            "total_crops": total_crops,
            "crop_failures": crop_failures,
        },
        "model": {
            "type": args.model_type,
            "path": args.model_path,
        },
        "proposals_file": args.proposals_file,
    }

    # Pretty print
    print("\n=== SAM + Classifier Evaluation (Reviewer-Safe) ===")
    print(f"Images (denominator): {results['totals']['images']}")
    print(f"Coverage (>=1 valid proposal): {results['totals']['coverage']:.4f}")
    print(f"Any: {results['metrics']['Any']:.4f}")
    print(f"Top-1(best-crop): {results['metrics']['Top1_best_crop']:.4f}")
    print(f"MeanProbs: {results['metrics']['MeanProbs']:.4f}")
    print(f"Voting: {results['metrics']['Voting']:.4f}")
    print(f"VotingWeighted: {results['metrics']['VotingWeighted']:.4f}")
    print("Any@k:", {k: f"{results['metrics']['Any@k'][k]:.4f}" for k in cfg.any_k_list})
    print("Proposal counts post-dedup (pre-cap):", results["proposal_stats"]["post_dedup_pre_cap"])
    print("Proposal counts post-cap:", results["proposal_stats"]["post_cap"])
    print("Crop stats:", results["debug"])

    # Save JSON
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.out_json}")

if __name__ == "__main__":
    main()
