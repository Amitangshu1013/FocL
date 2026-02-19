#!/usr/bin/env python3
# =============================================================================
# standard_baseline_eval.py — Standard Baseline Evaluation on ImageNet-V1 (2K Val)
# =============================================================================
#
# Purpose
# -------
# Evaluate a *standard full-image* ResNet-50 checkpoint on a fixed 2K subset of
# the ImageNet (ILSVRC2012) validation set. This is the “standard baseline”
# counterpart to FocL-style bbox-crop evaluation scripts.
#
#
# High-level flow
# ---------------
# 1) Load ImageNet-V1 validation set via torchvision.datasets.ImageFolder
#    (expects standard ImageFolder layout: val_root/<class_name>/*.JPEG).
# 2) Read a deterministic list of 2K validation image filenames from a text file
#    (one basename per line), and build a Dataset that indexes exactly those images.
# 3) Apply an evaluation transform (deterministic, no random augmentation):
#       - crop-mode "imagenet": Resize(256) → CenterCrop(224)
#       - crop-mode "resize224": Resize((224,224))
#     followed by ToTensor + ImageNet normalization.
# 4) Create a ResNet-50 model (timm) and load weights from a checkpoint.
#    The loader handles common checkpoint formats (raw state_dict, "state_dict",
#    "model_state_dict", or "ema") and strips a "module." prefix if present.
# 5) Run inference on the 2K set and report Top-1 / Top-5 accuracy.
# 6) Save per-image outputs to CSV: [filename, gt, pred, conf].
#
#
# Notes / gotchas
# ---------------
# - Deterministic eval: no RandomResizedCrop, no RandomHorizontalFlip, etc.
# - The 2K list must match filenames under --val-img-root; the script can fail
#   early with a clear error if any file is missing (strict mode).
# - This script assumes CUDA is available by default (model.eval().cuda()).
# - Confidence in CSV is max softmax probability for the predicted class.
#
#
# Typical usage
# -------------
# python standard_baseline_eval.py \
#   --val-img-root /path/to/imagenet2012/val \
#   --val2k-list val2k_filenames.txt \
#   --weights /path/to/checkpoint.pth \
#   --crop-mode imagenet \
#   --batch-size 256 --num-workers 8 \
#   --csv-out val2k_logits_resnet50.csv
# =============================================================================

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms as T
from PIL import Image
import timm

# ---------- args ----------
def parse_args():
    ap = argparse.ArgumentParser("ResNet50 eval on 2K full images (ILSVRC2012 val)")
    ap.add_argument("--val-img-root", type=Path, default="/local/a/imagenet/imagenet2012/val/",
                    help="Path to ILSVRC2012 val images (ImageFolder layout).")
    ap.add_argument("--val2k-list", type=Path, default=Path("val2k_filenames.txt"),
                    help="Basenames for the exact 2K images (one per line).")
    ap.add_argument("--weights", type=Path, required=True,
                    help="Checkpoint with model.state_dict() (standard or EMA).")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--csv-out", type=Path, default=Path("val2k_logits_resnet50.csv"))
    ap.add_argument("--crop-mode", type=str, default="imagenet",
                    choices=["imagenet","resize224"],
                    help="imagenet: Resize(256)->CenterCrop(224). resize224: direct Resize(224,224).")
    return ap.parse_args()

# ---------- transforms ----------
def build_transform(mode="imagenet"):
    if mode == "imagenet":
        return T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    else:
        return T.Compose([
            T.Resize((224,224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

# ---------- dataset (full images by basename) ----------
class Val2KFullImage(Dataset):
    def __init__(self, list_file: Path, img_root: Path, samples: list, transform, strict=True):
        # Original lines from file (keep for reference if you want)
        self.names_raw = [x.strip() for x in open(list_file, "r", encoding="utf-8") if x.strip()]
        # Force to basenames (handles if file accidentally contains full paths)
        self.names = [Path(n).name for n in self.names_raw]

        # Build fast lookup tables
        self.by_name = {Path(p).name: (Path(p), lbl) for (p, lbl) in samples}
        self.by_stem = {Path(p).stem: (Path(p), lbl) for (p, lbl) in samples}
        self.tf = transform

        if strict:
            # Pre-check missing so you fail early with a clear message
            missing = []
            for n in self.names:
                if n in self.by_name: 
                    continue
                stem = Path(n).stem
                if stem in self.by_stem:
                    continue
                # try case-insensitive name match
                n_low = n.lower()
                if any(k.lower() == n_low for k in self.by_name.keys()):
                    continue
                missing.append(n)
            if missing:
                hint = next(iter(self.by_name.keys())) if self.by_name else "(no samples indexed)"
                raise FileNotFoundError(
                    f"{len(missing)} filenames from {list_file} not found under {img_root}. "
                    f"First missing: {missing[0]} | Example available key: {hint}"
                )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]  # basename e.g. ILSVRC2012_val_00001234.JPEG
        
        # Look up by exact name first
        rec = self.by_name.get(name)
        if rec is None:
            # Try by stem
            stem = Path(name).stem
            rec = self.by_stem.get(stem)
        if rec is None:
            # Try case-insensitive
            n_low = name.lower()
            for k, v in self.by_name.items():
                if k.lower() == n_low:
                    rec = v
                    break
        if rec is None:
            raise FileNotFoundError(
                f"Could not find '{name}' in ImageFolder. "
                f"Tried basename, stem, and case-insensitive match. "
                f"Check extension/case and that your validation root matches ImageFolder layout."
            )

        img_path, label = rec
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)
        # keep basename in meta for CSV
        return x, label, {"filename": name}

# ---------- metrics ----------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, csv_out: Path):
    model.eval().cuda()
    total = top1 = top5 = 0
    all_names, all_targets, all_preds, all_confs = [], [], [], []

    for x, y, meta in loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        # top-1 / top-5
        _, pred_topk = logits.topk(5, dim=1)
        correct = pred_topk.eq(y.view(-1, 1))
        total += y.size(0)
        top1 += correct[:, :1].any(dim=1).sum().item()
        top5 += correct.any(dim=1).sum().item()

        # ---- robust filename gather (handles dict-of-lists vs list-of-dicts) ----
        if isinstance(meta, dict):
            names = meta.get("filename", [])
            # default collate gives a list of strings
            if isinstance(names, (list, tuple)):
                batch_names = list(names)
            elif isinstance(names, str):
                batch_names = [names]
            else:
                raise TypeError(f"Unexpected meta['filename'] type: {type(names)}")
        else:
            # custom collate as list of dicts
            batch_names = [m["filename"] for m in meta]

        # collect per-sample stats
        all_names.extend(batch_names)
        all_targets.extend(y.cpu().tolist())
        all_preds.extend(probs.argmax(dim=1).cpu().tolist())
        all_confs.extend(probs.max(dim=1).values.cpu().tolist())

    # write CSV
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "gt", "pred", "conf"])
        for nm, g, p, c in zip(all_names, all_targets, all_preds, all_confs):
            w.writerow([nm, int(g), int(p), float(c)])

    return (top1 / total, top5 / total, total)

# ---------- main ----------
def main():
    args = parse_args()

    # ImageFolder for labels & paths
    val_ds_full = datasets.ImageFolder(root=str(args.val_img_root))
    num_classes = len(val_ds_full.classes)

    ds = Val2KFullImage(args.val2k_list, args.val_img_root, val_ds_full.samples,
                        transform=build_transform(args.crop_mode))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Create ResNet50 model
    model = timm.create_model(
        "resnet50",
        pretrained=False,
        num_classes=num_classes,
    )
    
    # Load weights (handles raw or wrapped state_dict)
    # Note: weights_only=False is used to handle various checkpoint formats
    # If you have a simple state_dict, you can use weights_only=True
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)

    # pick the right sub-dict
    if isinstance(ckpt, dict) and "ema" in ckpt:
        sd = ckpt["ema"]                                  # your EMA snapshot
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]                           # common convention
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]                     # torch training bundle
    else:
        sd = ckpt                                         # assume it's a raw state_dict

    # strip DDP prefix if present
    sd = { (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in sd.items() }

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[load_state_dict] missing:", sorted(list(missing)))
    print("[load_state_dict] unexpected:", sorted(list(unexpected)))

    model = model.cuda()

    # Evaluate
    top1, top5, n_samples = evaluate(model, loader, args.csv_out)

    print(f"\n=== ResNet50 @ Full-Image Eval (224 center-crop) ===")
    print(f"Samples: {n_samples}")
    print(f"Top-1: {top1:.4f} | Top-5: {top5:.4f}")

if __name__ == "__main__":
    main()
