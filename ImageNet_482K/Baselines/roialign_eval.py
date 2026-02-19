#!/usr/bin/env python3
# =============================================================================
# roialign_eval.py — RoIAlign Evaluation on ImageNet-V1 (Val2K Subset)
# =============================================================================
#
# Purpose
# -------
# Evaluate a ResNet-50 RoIAlign-based classifier on a fixed Val2K subset of the
# ImageNet (ILSVRC2012) validation set. Unlike standard full-frame evaluation,
# this script performs classification using *bbox-aligned feature pooling*:
# the model extracts a feature map from the full image, then applies RoIAlign
# on the ground-truth bounding box region and classifies the pooled RoI feature.
#
#
# High-level evaluation flow
# --------------------------
# 1) Build a deterministic Val2K dataset:
#    - Load ImageNet-V1 validation set via torchvision.datasets.ImageFolder.
#    - Read Val2K filenames from a text list (one basename per line).
#    - Index exactly those images for evaluation (strict: errors if missing).
#
# 2) For each image:
#    - Load corresponding XML annotation and parse all bounding boxes.
#      • Tries flat XML path first:         <ann_root>/<image_stem>.xml
#      • Falls back to class-folder XML:    <ann_root>/<class>/<image_stem>.xml
#    - Select the LARGEST-area bbox (if multiple).
#    - If XML/bbox missing: fall back to full-image bbox.
#
# 3) Preprocess:
#    - Resize image to a fixed 224×224 and normalize using ImageNet mean/std.
#    - Scale bbox coordinates from original (W,H) into 224×224 coordinate space.
#
# 4) Model inference (RoIAlign pipeline):
#    Input image (B×3×224×224)
#      → ResNet-50 backbone truncated up to layer3
#        (feature map stride ≈ 16; channels = 1024)
#      → RoIAlign over the GT bbox region:
#           roi_align(feature_map, rois, output_size=(7,7), spatial_scale=1/16)
#      → AdaptiveAvgPool2d((1,1)) → Flatten → Linear(1024 → num_classes)
#
# 5) Metrics / outputs:
#    - Compute Top-1 and Top-5 accuracy over Val2K.
#    - Save per-image predictions to CSV: [filename, gt, pred, confidence].
#
#
# Notes on transforms (IMPORTANT)
# -------------------------------
# - This script uses a direct resize to (224,224) (no CenterCrop pipeline).
#   Therefore bbox scaling is performed with independent x/y scaling to match the
#   resized geometry. RoIAlign RoIs are in 224×224 input coordinates.
# - No random augmentation is used in evaluation (deterministic results).
#
#
# Checkpoint loading
# ------------------
# - Supports common checkpoint formats (raw state_dict or dict with keys like
#   'state_dict', 'model_state_dict', 'ema_state_dict', etc.).
# - Strips a leading "module." prefix to support DDP-trained checkpoints.
#
#
# Example usage
# -------------
# python roialign_eval.py \
#   --val-img-root /imagenet/imagenet2012/val \
#   --ann-root /path/to/ILSVRC2012_bbox_val/val \
#   --weights /path/to/roialign_checkpoint.pth \
#   --list-file val2k_filenames.txt \
#   --batch-size 128 --workers 8 \
#   --csv-out preds_resnet50_roialign.csv
# =============================================================================

import os
import csv
import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.ops import roi_align
from PIL import Image
import xml.etree.ElementTree as ET

# ---------------------- Logging ----------------------
logging.basicConfig(
    filename="roialign_eval_errors.log",
    filemode="a",
    level=logging.ERROR,
    format="%(asctime)s - %(message)s"
)

# ---------------------- Model Definition ----------------------
class ResNet50RoI(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50RoI, self).__init__()
        resnet = models.resnet50(pretrained=False)
        # Backbone: End at Layer 3 (stride 16). 
        # Layer 3 output has 1024 channels.
        self.backbone = nn.Sequential(*list(resnet.children())[:7])
        self.spatial_scale = 1.0 / 16.0 
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # Global pool pooled RoI to 1x1
            nn.Flatten(),
            nn.Linear(1024, num_classes) # Adjusted input dim to 1024
        )

    def forward(self, x, rois):
        feature_map = self.backbone(x)
        # Aligned=True is best-practice for modern detectors
        pooled = roi_align(feature_map, rois, output_size=(7, 7), 
                           spatial_scale=self.spatial_scale, sampling_ratio=-1,
                           aligned=True)
        return self.head(pooled)

# ---------------------- XML Helpers (Largest BBox Logic) ----------------------
def parse_xml_for_bbox(xml_file: str) -> List[List[int]]:
    """
    Parse an ImageNet-style XML file and return A LIST of [xmin, ymin, xmax, ymax].
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        # logging.error(f"XML parse failed: {xml_file} | {e}")
        return []

    bboxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None: 
            continue
        try:
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            if xmin < xmax and ymin < ymax:
                bboxes.append([xmin, ymin, xmax, ymax])
        except Exception:
            pass
    return bboxes

def pick_bbox(bboxes: List[List[int]]) -> Optional[List[int]]:
    """
    Pick the LARGEST bbox by area.
    """
    if not bboxes:
        return None
    # Area = (xmax - xmin) * (ymax - ymin)
    return max(bboxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

# ---------------------- Dataset ----------------------
class RoIAlignVal2K(Dataset):
    """
    Validation dataset that returns (image, bbox, label, meta).
    Logic:
    1. Load Image
    2. Load XML -> Pick Largest BBox
    3. Resize Image to 224x224
    4. Scale BBox to 224x224 coordinates
    """
    def __init__(self,
                 img_root: Path,
                 ann_root: Path,
                 list_file: Path,
                 transform: transforms.Compose):
        super().__init__()
        from torchvision import datasets
        self.base = datasets.ImageFolder(str(img_root))
        self.transform = transform
        self.ann_root = Path(ann_root)

        # Standard ImageFolder samples
        self.samples = self.base.samples
        
        # Build lookup indices to support list-file filtering
        by_name = {Path(p).name: i for i, (p, _) in enumerate(self.samples)}
        by_stem = {Path(p).stem: i for i, (p, _) in enumerate(self.samples)}
        by_name_lower = {Path(p).name.lower(): i for i, (p, _) in enumerate(self.samples)}

        if list_file is None:
            raise ValueError("list_file is required.")
        
        # Filter based on the list file
        wanted = [Path(s.strip()).name for s in open(list_file, "r", encoding="utf-8") if s.strip()]
        indices = []
        missing = []
        
        for name in wanted:
            i = by_name.get(name)
            if i is None: i = by_stem.get(Path(name).stem)
            if i is None: i = by_name_lower.get(name.lower())
            
            if i is None:
                missing.append(name)
            else:
                indices.append(i)

        if missing:
            raise FileNotFoundError(f"{len(missing)} filenames not found. Example: {missing[0]}")

        self.indices = indices
        self.classes = self.base.classes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        si = self.indices[i]
        path, label = self.samples[si]
        
        # 1. Load full image
        img = Image.open(path).convert("RGB")
        w_orig, h_orig = img.size

        # 2. Parse BBox (Logic: Largest Area)
        stem = Path(path).stem
        
        # Attempt 1: Check if XML exists in ann_root directly (flat)
        xml_path = self.ann_root / f"{stem}.xml"
        
        # Helper returns list of all bboxes
        bboxes = parse_xml_for_bbox(str(xml_path))
        
        # Fallback: if flat fails, try class folder structure
        if not bboxes:
            class_folder = Path(path).parent.name
            xml_path_nested = self.ann_root / class_folder / f"{stem}.xml"
            bboxes = parse_xml_for_bbox(str(xml_path_nested))

        # Select Largest
        bbox = pick_bbox(bboxes)

        # Fallback to full image if still None
        if bbox is None:
            bbox = [0, 0, w_orig, h_orig]

        # 3. Scale BBox
        x1, y1, x2, y2 = bbox
        
        # Scale to 224.0 (assuming Resize(224) is used)
        x1_s = (x1 / w_orig) * 224.0
        y1_s = (y1 / h_orig) * 224.0
        x2_s = (x2 / w_orig) * 224.0
        y2_s = (y2 / h_orig) * 224.0

        # 4. Clamping
        scaled_bbox = torch.tensor([x1_s, y1_s, x2_s, y2_s], dtype=torch.float32)
        scaled_bbox = torch.clamp(scaled_bbox, 0.0, 223.99)

        # 5. Transform Image
        img_tensor = self.transform(img)

        meta = {"filename": Path(path).name}

        return img_tensor, scaled_bbox, label, meta

# ---------------------- Metrics / Eval ----------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, csv_out: Path, device: torch.device):
    model.eval()
    total = 0
    top1 = 0
    top5 = 0

    all_names, all_targets, all_preds, all_confs = [], [], [], []

    print(f"Starting evaluation on {len(loader.dataset)} samples...")

    for i, (imgs, bboxes, labels, meta) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        bboxes = bboxes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Construct RoIs tensor: [batch_idx, x1, y1, x2, y2]
        batch_size = imgs.size(0)
        batch_indices = torch.arange(batch_size, device=device).view(-1, 1).float()
        rois = torch.cat([batch_indices, bboxes], dim=1)

        logits = model(imgs, rois)
        probs = torch.softmax(logits, dim=1)

        # top-1 / top-5
        _, pred_topk = logits.topk(5, dim=1)
        correct = pred_topk.eq(labels.view(-1, 1))
        bs = labels.size(0)
        total += bs
        top1 += correct[:, :1].any(dim=1).sum().item()
        top5 += correct.any(dim=1).sum().item()

        # Gather metadata
        if isinstance(meta, dict):
            names = meta.get("filename", [])
            batch_names = list(names) if isinstance(names, (list, tuple)) else [names]
        else:
            batch_names = [m["filename"] for m in meta]

        all_names.extend(batch_names)
        all_targets.extend(labels.detach().cpu().tolist())
        all_preds.extend(probs.argmax(dim=1).detach().cpu().tolist())
        all_confs.extend(probs.max(dim=1).values.detach().cpu().tolist())

        if i % 20 == 0:
            print(f"Batch {i}/{len(loader)} - Running Top1: {top1/total:.4f}")

    # Write CSV
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filename", "gt", "pred", "conf"])
            for nm, g, p, c in zip(all_names, all_targets, all_preds, all_confs):
                w.writerow([nm, int(g), int(p), float(c)])
        print(f"Predictions saved to {csv_out}")

    return (top1 / total, top5 / total, total)

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser("ResNet50 RoIAlign Eval (Largest BBox)")
    parser.add_argument("--val-img-root", type=Path, default="/local/a/imagenet/imagenet2012/val/",
                        help="ImageFolder-style validation root")
    parser.add_argument("--ann-root", type=Path, 
                        default="/ImageNetValBBox/Val/ILSVRC2012_bbox_val_v3/val",
                        help="Annotation root")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to ResNet50RoI checkpoint")
    parser.add_argument("--list-file", type=Path, default=Path("val2k_filenames.txt"),
                        help="File with filenames to subset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--csv-out", type=Path, default=Path("preds_resnet50_roialign.csv"))
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform: Resize(224) -> ToTensor -> Normalize
    # We rely on this resizing to 224 to make the manually scaled bbox coords valid.
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # Dataset
    ds = RoIAlignVal2K(
        img_root=args.val_img_root,
        ann_root=args.ann_root,
        list_file=args.list_file,
        transform=tf,
    )
    
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Model
    num_classes = len(ds.classes)
    model = ResNet50RoI(num_classes).to(device)

    # Weights Loading
    print(f"Loading weights from {args.weights}...")
    ckpt = torch.load(args.weights, map_location="cpu")
    
    sd = None
    # Flexible loading (EMA vs Standard vs DDP)
    for key in ("ema_state_dict", "model_ema", "state_dict", "model_state_dict", "model"):
        if isinstance(ckpt, dict) and key in ckpt:
            sd = ckpt[key]
            if hasattr(sd, "state_dict"):
                sd = sd.state_dict()
            break
            
    if sd is None:
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
        else:
            sd = ckpt # Last resort

    # Strip 'module.'
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # Load
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Warning] Unexpected keys: {len(unexpected)}")

    # Evaluate
    top1, top5, n_samples = evaluate(model, loader, args.csv_out, device)

    print("\n=== ResNet50 RoIAlign Evaluation ===")
    print(f"Samples: {n_samples}")
    print(f"Top-1: {top1:.4f} | Top-5: {top5:.4f}")

if __name__ == "__main__":
    main()