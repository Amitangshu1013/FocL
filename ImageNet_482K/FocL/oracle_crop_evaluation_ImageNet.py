#!/usr/bin/env python3
# =============================================================================
# bbox_crop_evaluation.py — FocL Evaluation on ImageNet-V1 (GT BBox Crop @ 224)
# =============================================================================
#
# Purpose
# -------
# Evaluate a trained ResNet-50 under the FocL setting on ImageNet-V1 validation:
# each validation image is cropped to the *ground-truth bounding box* (largest
# GT box if multiple), then resized to 224×224 and fed to the classifier.
#
# This measures recognition performance when the model is given an object-centric
# “foveated” view (GT crop), rather than the full cluttered frame.
#
#
# High-level evaluation flow
# --------------------------
# 1) Build a validation dataset (ImageFolder layout) restricted to a filename list
#    (e.g., the 2k subset file), so evaluation is deterministic and repeatable.
# 2) For each image:
#      - Load corresponding XML annotation (<ann_root>/<stem>.xml).
#      - Parse all bboxes, pick the largest-area bbox.
#      - Crop the image to that bbox (fallback: use full image if XML/bbox missing).
#      - Apply transform → 224×224 tensor + ImageNet normalization.
# 3) Load the ResNet-50 checkpoint (supports several checkpoint key formats, EMA,
#    and optional DDP "module." prefix stripping).
# 4) Run inference and compute Top-1 and Top-5 accuracy.
# 5) Write per-image predictions to a CSV: filename, gt, pred, confidence.
#
#
# Dataset / annotation assumptions
# -------------------------------
# - Validation images are in ImageFolder layout:
#     --val-img-root /imagenet/imagenet2012/val/
#   with 1000 class subfolders.
#
# - Annotations are “flat” per-image XML files (not nested by class folder):
#     --ann-root .../<stem>.xml
#   where <stem> matches the image filename stem (no extension).
#
# - list-file (default: val2k_filenames.txt) provides the subset of validation
#   images to evaluate. Filenames must match those under --val-img-root.
#   The script hard-fails if any filename in the list is not found.
#
#
# Transforms (IMPORTANT)
# ----------------------
# - This script uses a *direct* resize to 224×224:
#     Resize((224,224)) → ToTensor → Normalize
#   There is no 256→CenterCrop(224) validation pipeline here.
#
# - Cropping happens *before* transforms:
#     x = transform( img.crop(bbox) )
#   so Resize((224,224)) always acts on the bbox crop.
#
# - No RandomHorizontalFlip (or any random augmentation) is used in evaluation.
#   This ensures deterministic metrics and avoids bbox/image mismatch.
#
#
# Outputs
# -------
# - Prints: number of evaluated samples, Top-1, Top-5.
# - Writes: CSV (--csv-out) with per-image predictions and confidence.
# - Logs bbox/XML issues to bbox_eval_errors.log (missing XML, parse errors, etc.).
#
# Example usage
# -------------
# python bbox_crop_evaluation.py \
#   --val-img-root /imagenet/imagenet2012/val \
#   --ann-root /path/to/ILSVRC2012_bbox_val/val \
#   --weights /path/to/checkpoint.pth \
#   --list-file val2k_filenames.txt \
#   --batch-size 256 --workers 8 \
#   --csv-out preds_FocL_bbox.csv
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
from PIL import Image
import xml.etree.ElementTree as ET

import timm
from torchvision import datasets, transforms

# ---------------------- Logging (for bbox issues) ----------------------
logging.basicConfig(
    filename="bbox_eval_errors.log",
    filemode="a",
    level=logging.ERROR,
    format="%(asctime)s - %(message)s"
)

# ---------------------- XML → bboxes ----------------------
def parse_xml_for_bbox(xml_file: str) -> List[List[int]]:
    """Parse an ImageNet-style XML file and return a list of [xmin, ymin, xmax, ymax]."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except Exception as e:
        logging.error(f"XML parse failed: {xml_file} | {e}")
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
        except Exception as e:
            logging.error(f"Bad bbox in {xml_file}: {e}")
    return bboxes

def pick_bbox(bboxes: List[List[int]]) -> Optional[List[int]]:
    """Pick one bbox. Here: largest area."""
    if not bboxes:
        return None
    return max(bboxes, key=lambda b:(b[2]-b[0])*(b[3]-b[1]))

# ---------------------- Dataset (crop bbox → 224×224) ----------------------
class BBoxCropVal2K(Dataset):
    """
    ImageNet val (ImageFolder layout for images), but *flat* per-image XML annotations.
    - Requires a list-file (2k) with basenames like 'ILSVRC2012_val_00007197.JPEG'
    - For each listed image: load ann_root/<stem>.xml, crop largest bbox, resize → 224×224.
    - Fallback to full image resized 224×224 if XML/bbox missing/invalid.
    Returns: (tensor image, label, meta={'filename': basename, 'bbox': [xmin,ymin,xmax,ymax]})
    """
    def __init__(self,
                 img_root: Path,
                 ann_root: Path,
                 list_file: Path,                 # <-- required
                 transform: transforms.Compose):
        super().__init__()
        self.base = datasets.ImageFolder(str(img_root))
        self.transform = transform
        self.ann_root = Path(ann_root)

        # Map dataset samples by basename / stem for fast lookup
        self.samples = self.base.samples  # list[(path,label)]
        by_name = {Path(p).name: i for i, (p, _) in enumerate(self.samples)}
        by_stem = {Path(p).stem: i for i, (p, _) in enumerate(self.samples)}
        by_name_lower = {Path(p).name.lower(): i for i, (p, _) in enumerate(self.samples)}  # robustness

        # Read required 2k list and build indices in *that* order
        if list_file is None:
            raise ValueError("list_file is required for this dataset (2k subset).")
        wanted = [Path(s.strip()).name for s in open(list_file, "r", encoding="utf-8") if s.strip()]

        indices = []
        missing = []
        for name in wanted:
            i = by_name.get(name)
            if i is None:
                i = by_stem.get(Path(name).stem)
            if i is None:
                i = by_name_lower.get(name.lower())
            if i is None:
                missing.append(name)
            else:
                indices.append(i)

        if missing:
            logging.error(f"{len(missing)} names from {list_file} not found under {img_root}. "
                         f"First missing: {missing[0]}")
            # Hard fail so you notice bad list/data drift
            raise FileNotFoundError(f"{len(missing)} filenames in {list_file} not found in {img_root}. "
                                    f"Example: {missing[0]}")

        self.indices = indices
        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        si = self.indices[i]
        path, label = self.samples[si]
        img = Image.open(path).convert("RGB")
        W, H = img.size

        # --- flat XML: ann_root/<stem>.xml ---
        stem = Path(path).stem
        xml_path = self.ann_root / f"{stem}.xml"

        bbox = None
        try:
            if xml_path.exists():
                bxs = parse_xml_for_bbox(str(xml_path))  # your helper
                bbox = pick_bbox(bxs)                    # largest-area
                if bbox is not None:
                    xmin, ymin, xmax, ymax = bbox
                    # clamp & validate
                    xmin = max(0, min(xmin, W-1))
                    ymin = max(0, min(ymin, H-1))
                    xmax = max(0, min(xmax, W))
                    ymax = max(0, min(ymax, H))
                    if not (xmin < xmax and ymin < ymax):
                        bbox = None
            else:
                # Optional: log missing XML
                logging.error(f"Missing XML for {Path(path).name}: {xml_path}")
        except Exception as e:
            logging.error(f"Error reading bbox for {path}: {e}")
            bbox = None

        # Crop (bbox) or fallback (full image), then transform → tensor(224×224, normalized)
        if bbox is not None:
            x = self.transform(img.crop((bbox[0], bbox[1], bbox[2], bbox[3])))
            meta = {"filename": Path(path).name, "bbox": bbox}
        else:
            x = self.transform(img)
            meta = {"filename": Path(path).name, "bbox": [0, 0, W, H]}

        return x, label, meta

# ---------------------- Metrics / eval ----------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, csv_out: Path):
    model.eval().cuda()
    total = 0
    top1 = 0
    top5 = 0

    all_names, all_targets, all_preds, all_confs = [], [], [], []

    for x, y, meta in loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        # top-1 / top-5
        _, pred_topk = logits.topk(5, dim=1)
        correct = pred_topk.eq(y.view(-1, 1))
        bs = y.size(0)
        total += bs
        top1 += correct[:, :1].any(dim=1).sum().item()
        top5 += correct.any(dim=1).sum().item()

        # gather filenames (meta is dict-of-lists under default collate)
        if isinstance(meta, dict):
            names = meta.get("filename", [])
            batch_names = list(names) if isinstance(names, (list, tuple)) else [names]
        else:
            batch_names = [m["filename"] for m in meta]

        all_names.extend(batch_names)
        all_targets.extend(y.detach().cpu().tolist())
        all_preds.extend(probs.argmax(dim=1).detach().cpu().tolist())
        all_confs.extend(probs.max(dim=1).values.detach().cpu().tolist())

    # CSV
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "gt", "pred", "conf"])
        for nm, g, p, c in zip(all_names, all_targets, all_preds, all_confs):
            w.writerow([nm, int(g), int(p), float(c)])

    return (top1 / total, top5 / total, total)

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser("FocL bbox-crop eval (224x224 direct)")
    parser.add_argument("--val-img-root", type=Path, default="/imagenet/imagenet2012/val/",
                        help="ImageFolder-style validation root (1000 class subdirs)")
    parser.add_argument("--ann-root", type=Path, 
                        default="/Supervised/ImageNetValBBox/Val/ILSVRC2012_bbox_val_v3/val",
                        help="Annotation root containing XML: <ann_root>/<stem>.xml")
    parser.add_argument("--weights", type=Path, required=True,
                        help="Path to ResNet50 checkpoint (.pth)")
    parser.add_argument("--list-file", type=Path, default=Path("val2k_filenames.txt"),
                        help="Optional file with one filename per line to subset (e.g., 2k list)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--csv-out", type=Path, default=Path("preds_FocL_bbox_482K.csv"))
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    # Transform: direct resize to 224x224 (no 256→center-crop), ToTensor, Normalize
    tf = transforms.Compose([
        transforms.Resize((224, 224), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    # Dataset & loader
    ds = BBoxCropVal2K(
        img_root=args.val_img_root,
        ann_root=args.ann_root,     # flat: /.../Annotations/val
        list_file=args.list_file,   # required 2k list
        transform=tf,               # Resize(224,224) → ToTensor → Normalize
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    # Model (ResNet50) — same head size as dataset classes
    num_classes = len(ds.classes)
    model = timm.create_model(
        "resnet50",
        pretrained=False,
        num_classes=num_classes,
    ).cuda()

    # Robust weight loader (handles flat / state_dict / model_state_dict / EMA)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    # 1) pick the correct sub-dict (prefer EMA)
    sd = None
    for key in (
        "ema_state_dict",     # your file shows this key
        "ema",                # some scripts use this
        "state_dict_ema",     # another common name
        "model_ema",          # sometimes an object with .state_dict()
        "state_dict",
        "model_state_dict",
    ):
        if isinstance(ckpt, dict) and key in ckpt:
            sd = ckpt[key]
            break

    # 2) if model_ema is an object, extract its state_dict
    if sd is not None and hasattr(sd, "state_dict"):
        sd = sd.state_dict()

    # 3) bare state_dict fallback
    if sd is None:
        if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            sd = ckpt
        else:
            raise ValueError(
                f"Could not find a state_dict in checkpoint. Top-level keys: {list(ckpt.keys())[:10]}"
            )

    # 4) strip optional 'module.' prefix from DDP/EMA wrappers
    sd = { (k.replace("module.", "", 1) if isinstance(k, str) and k.startswith("module.") else k): v
        for k, v in sd.items() }

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[load_state_dict] missing:", sorted(list(missing)))
    print("[load_state_dict] unexpected:", sorted(list(unexpected)))

    # Metrics: evaluate
    top1, top5, n_samples = evaluate(model, loader, args.csv_out)

    print("\n=== ResNet50 @ BBox-Crop Eval (224×224 direct resize) ===")
    print(f"Samples: {n_samples}")
    print(f"Top-1: {top1:.4f} | Top-5: {top5:.4f}")

if __name__ == "__main__":
    main()
