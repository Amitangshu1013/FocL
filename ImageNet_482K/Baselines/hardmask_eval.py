#!/usr/bin/env python3
# =============================================================================
# hardmask_eval.py — HardMask Evaluation on ImageNet-V1 (Val2K Subset)
# =============================================================================
#
# Purpose
# -------
# Evaluate a ResNet-50 checkpoint on an ImageNet-V1 validation subset (Val2K)
# using the “HardMask” input setting:
#   - Keep pixels *inside* the GT bounding box (object region)
#   - Replace pixels *outside* the bbox with a constant mean-color background
#
# This is a deterministic evaluation script (no random augmentation) intended to
# match the HardMask training/eval protocol.
#
#
# High-level data flow
# --------------------
# For each sample in Val2K:
#   1) Load the RGB image from ImageFolder (ImageNet validation layout).
#   2) Locate the corresponding XML annotation and parse all bounding boxes.
#      - Tries a flat XML path first:  <ann_root>/<image_stem>.xml
#      - Falls back to class-folder XML: <ann_root>/<class>/<image_stem>.xml
#   3) If bboxes exist: select the FIRST bbox, clamp to image bounds, and build
#      a binary mask.
#      - Composite the image with a solid mean-color background (124,116,104)
#        so background pixels are replaced and object pixels are preserved.
#      - If no bbox is found / XML missing: fall back to the original image.
#   4) Apply standard eval transforms (deterministic):
#        Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)
#   5) Run ResNet-50 inference and compute Top-1 / Top-5 accuracy.
#   6) Save per-image predictions to CSV: [filename, gt, pred, confidence].
#
#
# Notes on transforms (important)
# -------------------------------
# - Masking happens at the ORIGINAL image resolution (W×H) before any resizing.
# - The evaluation transform is applied AFTER masking and is deterministic:
#     • Resize(256) sets the shorter side to 256 (keeps aspect ratio)
#     • CenterCrop(224) extracts a fixed 224×224 crop
# - No RandomHorizontalFlip (or any random augmentation) is used in evaluation.
#
#
# Checkpoint loading behavior
# ---------------------------
# - Supports common checkpoint formats: ema_state_dict / model_ema / state_dict /
#   model_state_dict / model, or a raw state_dict dict.
# - Strips a leading "module." prefix to support DDP-trained checkpoints.
# - Loads with strict=False and prints missing/unexpected key counts as warnings.
#
#
# Typical usage
# -------------
# python hardmask_eval.py \
#   --val-img-root /imagenet/imagenet2012/val \
#   --ann-root /path/to/ILSVRC2012_bbox_val/val \
#   --weights /path/to/checkpoint.pth \
#   --list-file val2k_filenames.txt \
#   --batch-size 128 --workers 8 \
#   --csv-out preds_resnet50_hardmask.csv
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
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET

# ---------------------- Logging ----------------------
logging.basicConfig(
    filename="hardmask_eval_errors.log",
    filemode="a",
    level=logging.ERROR,
    format="%(asctime)s - %(message)s"
)

# ---------------------- XML Helpers ----------------------
def parse_xml_for_bbox(xml_file: str) -> List[List[int]]:
    """
    Parse an XML file to extract bounding box coordinates.
    Matches hardmaskdataloader.py logic.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        bboxes = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            # Ensure the coordinates are valid
            if xmin < xmax and ymin < ymax:
                bboxes.append([xmin, ymin, xmax, ymax])
        return bboxes
    except Exception as e:
        return []

# ---------------------- Dataset ----------------------
class HardMaskVal2K(Dataset):
    """
    Validation dataset that replicates HardMaskDataset logic:
    1. Load Image
    2. Load XML -> Get FIRST BBox
    3. Mask background with Mean Color
    4. Apply Standard Eval Transform (Resize 256 -> CenterCrop 224)
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
        
        # HardMask Mean Color (R, G, B)
        self.mean_color = (124, 116, 104)

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
        
        # 1. Load Image
        image = Image.open(path).convert("RGB")
        w, h = image.size

        # 2. Parse XML for BBox
        stem = Path(path).stem
        
        # Attempt 1: Flat structure
        xml_path = self.ann_root / f"{stem}.xml"
        bboxes = parse_xml_for_bbox(str(xml_path))
        
        # Attempt 2: Class folder structure
        if not bboxes:
            class_folder = Path(path).parent.name
            xml_path_nested = self.ann_root / class_folder / f"{stem}.xml"
            bboxes = parse_xml_for_bbox(str(xml_path_nested))

        # 3. Create Hard Mask
        masked_image = image.copy() # Fallback

        if bboxes:
            # Use FIRST bbox found (Matches hardmaskdataloader.py)
            xmin, ymin, xmax, ymax = bboxes[0]

            # Clamp
            xmin = max(0, xmin); ymin = max(0, ymin)
            xmax = min(w, xmax); ymax = min(h, ymax)

            if xmax > xmin and ymax > ymin:
                # Create Mask: 255 (White) inside bbox, 0 (Black) outside
                mask = Image.new('L', (w, h), 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle((xmin, ymin, xmax, ymax), fill=255)
                
                # Create Background: Solid Mean Color
                bg = Image.new('RGB', (w, h), self.mean_color)
                
                # Composite: Keep object pixels where mask is 255, use Mean Color elsewhere
                masked_image = Image.composite(image, bg, mask)

        # 4. Apply Transform
        if self.transform:
            x = self.transform(masked_image)
        else:
            x = transforms.ToTensor()(masked_image)

        meta = {"filename": Path(path).name}

        return x, label, meta

# ---------------------- Metrics / Eval ----------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, csv_out: Path, device: torch.device):
    model.eval()
    total = 0
    top1 = 0
    top5 = 0

    all_names, all_targets, all_preds, all_confs = [], [], [], []

    print(f"Starting evaluation on {len(loader.dataset)} samples...")

    for i, (imgs, labels, meta) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
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
    parser = argparse.ArgumentParser("ResNet50 HardMask Eval")
    parser.add_argument("--val-img-root", type=Path, default="/imagenet/imagenet2012/val/",
                        help="ImageFolder-style validation root")
    parser.add_argument("--ann-root", type=Path, 
                        default="/ImageNetValBBox/Val/ILSVRC2012_bbox_val_v3/val",
                        help="Annotation root")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to ResNet50 checkpoint")
    parser.add_argument("--list-file", type=Path, default=Path("val2k_filenames.txt"),
                        help="File with filenames to subset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--csv-out", type=Path, default=Path("preds_resnet50_hardmask.csv"))
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform: Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize
    # Matches 'test_transform' in hardmaskddptrain.py
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # Dataset
    ds = HardMaskVal2K(
        img_root=args.val_img_root,
        ann_root=args.ann_root,
        list_file=args.list_file,
        transform=tf,
    )
    
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True
    )

    # Model: Standard ResNet50 (HardMask train uses standard model)
    num_classes = len(ds.classes)
    model = models.resnet50(pretrained=False, num_classes=num_classes).to(device)

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

    print("\n=== ResNet50 HardMask Evaluation ===")
    print(f"Samples: {n_samples}")
    print(f"Top-1: {top1:.4f} | Top-5: {top5:.4f}")

if __name__ == "__main__":
    main()