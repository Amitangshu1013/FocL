#!/usr/bin/env python3
# =============================================================================
# ddp_train_roialign.py — DDP Training with RoIAlign (BBox-aligned feature pooling)
# =============================================================================
#
# Main idea / feature flow
# ------------------------
# This script trains an ImageNet classifier using DistributedDataParallel (DDP)
# where classification is performed on *RoI-aligned* features rather than global
# full-image features.
#
# Per sample (from RoIAlignDataset):
#   (image_tensor, bbox_xyxy_224, class_label)
# where bbox is already scaled into the 224×224 input coordinate system.
#
# Forward feature flow (high-level):
#   1) Input: image ∈ R^{B×3×224×224}
#   2) Backbone: ResNet-50 truncated up to layer3
#        → feature_map ∈ R^{B×1024×Hf×Wf}  (typical stride = 16)
#   3) RoIAlign: pools each bbox region from the feature_map
#        roi_align(feature_map, rois, output_size=(7,7), spatial_scale=1/16)
#        → roi_feat ∈ R^{B×1024×7×7}
#   4) Pool + classifier head:
#        AdaptiveAvgPool2d((1,1)) → Flatten → Linear(1024 → num_classes)
#
# Key notes (bbox + transforms)
# -----------------------------
# - Bounding boxes are provided in (x1, y1, x2, y2) format in *224×224* coordinates
#   by the dataset, so RoIAlign receives boxes already aligned to the network input.
# - IMPORTANT: RandomHorizontalFlip is NOT included in the torchvision transform
#   pipeline here.
#     Reason: the dataset (RoIAlignDataset) handles horizontal flip manually so
#     that the bbox is flipped consistently with the image. Adding
#     RandomHorizontalFlip in transforms would desynchronize or double-flip.
#
# Outputs
# -------
# - Logs (rank 0): train/val loss and accuracy (W&B if enabled)
# - Saves best and final checkpoints (rank 0)
# =============================================================================

import os
import json
import argparse
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models
from torchvision.ops import roi_align

import wandb
from torch.cuda.amp import GradScaler, autocast

# --- IMPORT YOUR CUSTOM DATASET ---
from roialign_dataloader import RoIAlignDataset

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    args = parser.parse_args()

    # ─── Set Random Seed (Matches HardMask) ──────────────────────────────────
    seed = 2 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    local_rank = args.local_rank

    # ─── DDP Init (Matches HardMask) ─────────────────────────────────────────
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    # ─── Configuration ────────────────────────────────────────────────────────
    ANNOTATION_PATH = "ImageNet_BBOX_Folder/Annotation"
    DATASET_PATH = "/imagenet/imagenet2012/train/"
    SAVE_DIR = "ResNet50_RoIAlign_Baseline"
    
    if rank == 0:
        wandb.init(
            project="RoIAlign_Baseline_exp",
            config={
                "epochs": 90,
                "batch_size": 128,
                "learning_rate": 0.1,
                "weight_decay": 1e-4,
                "step_size": 30,
                "gamma": 0.1,
                "dataset_type": "RoIAlign (Feature Pooling)"
            }
        )
    cfg = wandb.config if rank == 0 else None
    os.makedirs(SAVE_DIR, exist_ok=True)

    
    # ─── Data Transforms ──────────────────────────────────────────────────────
    # We use Resize to 224 for both to ensure BBox scaling remains consistent 
    # with the RoIAlign spatial scale (1/16).
    # Updated Transforms (Remove RandomHorizontalFlip)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # ─── Dataset & Split ──────────────────────────────────────────────────────
    with open("combined_A_B_500.json", "r") as f:
        splits = json.load(f)

    base_ds = datasets.ImageFolder(root=DATASET_PATH)
    train_subset = Subset(base_ds, splits["train"])
    val_subset = Subset(base_ds, splits["val"])
    
    train_ds = RoIAlignDataset(train_subset, ANNOTATION_PATH, transform=train_transform, is_train=True)
    val_ds   = RoIAlignDataset(val_subset,   ANNOTATION_PATH, transform=test_transform,  is_train=False)

    # ─── Distributed Samplers & Loaders ───────────────────────────────────────
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=128, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=128, sampler=val_sampler, num_workers=4, pin_memory=True)

    # ─── Model, Loss, Optimizer ───────────────────────────────────────────────
    num_classes = len(base_ds.classes)
    model = ResNet50RoI(num_classes).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    scaler = GradScaler()

    # ─── Training Loop ────────────────────────────────────────────────────────
    best_val_acc = 0.0
    total_epochs = 90

    total_train_samples = train_sampler.num_samples * world_size
    total_val_samples   = val_sampler.num_samples * world_size

    for epoch in range(total_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        run_loss = torch.tensor(0.0, device=device)
        run_corrects = torch.tensor(0, device=device)

        for imgs, bboxes, labels in train_loader:
            imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)
            
            # Prepare ROIs: [batch_idx, x1, y1, x2, y2]
            batch_size = imgs.size(0)
            batch_indices = torch.arange(batch_size, device=device).view(-1, 1).float()
            rois = torch.cat([batch_indices, bboxes], dim=1)

            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs, rois)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            run_loss += loss.detach() * imgs.size(0)
            run_corrects += (preds == labels).sum()

        # Aggregate stats across GPUs
        dist.all_reduce(run_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(run_corrects, op=dist.ReduceOp.SUM)

        epoch_loss = run_loss.item() / total_train_samples
        epoch_acc = run_corrects.item() / total_train_samples
        scheduler.step()

        if rank == 0:
            wandb.log({"train_loss": epoch_loss, "train_accuracy": epoch_acc}, step=epoch)

        # ─── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_corrects = torch.tensor(0, device=device)
        
        with torch.no_grad():
            for imgs, bboxes, labels in val_loader:
                imgs, bboxes, labels = imgs.to(device), bboxes.to(device), labels.to(device)
                
                batch_size = imgs.size(0)
                batch_indices = torch.arange(batch_size, device=device).view(-1, 1).float()
                rois = torch.cat([batch_indices, bboxes], dim=1)

                with autocast():
                    outputs = model(imgs, rois)
                    loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                val_loss += loss * imgs.size(0)
                val_corrects += (preds == labels).sum()

        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_corrects, op=dist.ReduceOp.SUM)

        val_epoch_loss = val_loss.item() / total_val_samples
        val_epoch_acc = val_corrects.item() / total_val_samples

        if rank == 0:
            wandb.log({"val_loss": val_epoch_loss, "val_accuracy": val_epoch_acc}, step=epoch)
            print(f"[Epoch {epoch+1:>2}/{total_epochs}] "
                  f"Train: loss={epoch_loss:.4f}, acc={epoch_acc:.4f} | "
                  f"Val:   loss={val_epoch_loss:.4f}, acc={val_epoch_acc:.4f}")

            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))

    if rank == 0:
        torch.save(model.module.state_dict(), os.path.join(SAVE_DIR, "final_model.pth"))
        wandb.save(os.path.join(SAVE_DIR, "final_model.pth"))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()