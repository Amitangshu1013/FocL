#!/usr/bin/env python3

# =============================================================================
# train_ddp_standard_baseline.py — Standard ImageNet ResNet-50 (Full-Frame) DDP
# =============================================================================
#
# What this script does
# ---------------------
# Trains a *standard full-frame* ResNet-50 classifier on ImageNet using
# PyTorch DistributedDataParallel (DDP) + AMP, and evaluates on a held-out
# validation split defined by a JSON file.
#
# This is the “standard baseline” (no bounding-box crops / no FocL):
#   ImageFolder(full images) → ImageNet-style augmentations → ResNet-50 → CE loss
#
#
# Required inputs / expected files
# --------------------------------
# 1) ImageNet train directory (ImageFolder layout):
#      DATASET_PATH = "/imagenet/imagenet2012/train/"
#    Must contain per-class subfolders (ImageFolder convention).
#
# 2) Split JSON in the current working directory:
#      combined_A_B_500.json
#    Must contain:
#      - splits["train"] : list[int] indices into ImageFolder
#      - splits["val"]   : list[int] indices into ImageFolder
#
#
# CLI arguments
# -------------
#   --local_rank <int>
#       Local GPU rank for this process. Usually set automatically by torchrun
#       via the LOCAL_RANK environment variable.
#
# NOTE:
# - This script currently hard-codes DATASET_PATH and the splits JSON filename.
#   If you want portability, add args for --dataset_path and --splits (like in
#   your other script).
#
#
# How to run (example)
# --------------------
# Single node, 8 GPUs:
#   torchrun --nproc_per_node=8 train_ddp_standard_baseline.py
#
# Multi-node requires standard torchrun rendezvous args (nnodes, node_rank, etc.).
#


import os
import json
import argparse

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models
import numpy as np
import wandb
from torch.cuda.amp import GradScaler, autocast
import random


# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 37 # Next 1, 2, 3, 37, 42 
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
g = torch.Generator()
g.manual_seed(SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    args = parser.parse_args()
    local_rank = args.local_rank

    # ─── DDP Init ─────────────────────────────────────────────────────────────
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ─── Only rank 0 logs to WandB ─────────────────────────────────────────────
    if rank == 0:
        wandb.init(
            project="imagenet-Aug_training_resnet50_ddp",
            config={
                "epochs": 90,
                "batch_size": 128,
                "learning_rate": 0.1,
                "weight_decay": 1e-4,
                "step_size": 30,
                "gamma": 0.1
            }
        )
    cfg = wandb.config if rank == 0 else None
    os.makedirs("ResNet50_MergedAB", exist_ok=True)
    # ─── Device ────────────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{local_rank}")

    # ─── Data Transforms ───────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # ─── Dataset & Split ───────────────────────────────────────────────────────
    DATASET_PATH = "/imagenet/imagenet2012/train/"
    train_ds = datasets.ImageFolder(root=DATASET_PATH, transform=train_transform)
    val_ds   = datasets.ImageFolder(root=DATASET_PATH, transform=test_transform)

    with open("combined_A_B_500.json","r") as f:
        splits = json.load(f)
    train_subset = Subset(train_ds, splits["train"])
    val_subset   = Subset(val_ds, splits["val"])

    # ─── Distributed Samplers & Loaders ────────────────────────────────────────
    train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_subset,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_subset,
        batch_size=cfg.batch_size if rank==0 else 128,
        sampler=train_sampler,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=cfg.batch_size if rank==0 else 128,
        sampler=val_sampler,
        num_workers=4, pin_memory=True
    )

    # ─── Model, Loss, Optimizer, Scheduler ─────────────────────────────────────
    model = models.resnet50(pretrained=False, num_classes=len(train_ds.classes)).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.learning_rate if rank==0 else 0.1,
                          momentum=0.9,
                          weight_decay=cfg.weight_decay if rank==0 else 1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=cfg.step_size if rank==0 else 30,
                                          gamma=cfg.gamma if rank==0 else 0.1)

    scaler = GradScaler()

    # ─── Training Loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    for epoch in range(cfg.epochs if rank==0 else 90):
        train_sampler.set_epoch(epoch)
        model.train()
        run_loss = torch.tensor(0.0, device=device)
        run_corrects = torch.tensor(0,   device=device)

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = outputs.argmax(dim=1)
            run_loss += loss.detach() * imgs.size(0)
            run_corrects += (preds == labels).sum()

        # ─── aggregate across GPUs ────────────────────────────────────────────
        dist.all_reduce(run_loss,     op=dist.ReduceOp.SUM)
        dist.all_reduce(run_corrects, op=dist.ReduceOp.SUM)

        epoch_loss = run_loss.item() / len(train_subset)
        epoch_acc  = run_corrects.item() / len(train_subset)
        scheduler.step()

        if rank == 0:
            wandb.log({"train_loss": epoch_loss, "train_accuracy": epoch_acc}, step=epoch)

        # ─── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_loss = torch.tensor(0.0, device=device)
        val_corrects = torch.tensor(0, device=device)
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)

                preds = outputs.argmax(dim=1)
                val_loss += loss * imgs.size(0)
                val_corrects += (preds == labels).sum()

        dist.all_reduce(val_loss,     op=dist.ReduceOp.SUM)
        dist.all_reduce(val_corrects, op=dist.ReduceOp.SUM)

        val_epoch_loss = val_loss.item() / len(val_subset)
        val_epoch_acc  = val_corrects.item() / len(val_subset)

        if rank == 0:
            wandb.log({"val_loss": val_epoch_loss, "val_accuracy": val_epoch_acc}, step=epoch)
            print(f"[Epoch {epoch+1:>2}/{cfg.epochs}] "
                  f"Train: loss={epoch_loss:.4f}, acc={epoch_acc:.4f} | "
                  f"Val:   loss={val_epoch_loss:.4f}, acc={val_epoch_acc:.4f}")

            # ─── Save best model ────────────────────────────────────────────────
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                torch.save(model.module.state_dict(),
                           os.path.join("ResNet50_MergedAB", "best_model.pth"))

    # ─── Final save on rank 0 ─────────────────────────────────────────────────
    if rank == 0:
        os.makedirs("ResNet50_MergedAB", exist_ok=True)
        torch.save(model.module.state_dict(),
                   os.path.join("ResNet50_MergedAB", "final_model.pth"))
        wandb.save(os.path.join("ResNet50_MergedAB", "final_model.pth"))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
