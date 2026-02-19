#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Required / expected CLI arguments for this DDP training script:
#
#   --splits <path/to/splits.json>
#       JSON file defining train/val splits (e.g., combined_A_B_500.json).
#
#   --dataset_path <path/to/imagenet/train/>
#       Root directory containing ImageNet training images (folder of class subdirs).
#
#   --annotation_folder <path/to/annotations/>
#       Directory containing bounding-box annotations corresponding to the images.
#
# Optional:
#   --local_rank <int>
#       DDP local rank (usually set automatically via LOCAL_RANK env var).
#
# Example (torchrun):
#   torchrun --nproc_per_node=8 train_ddp.py \
#       --splits combined_A_B_500.json \
#       --dataset_path /path/to/imagenet2012/train \
#       --annotation_folder /path/to/bbox_annotations
# -----------------------------------------------------------------------------


import os
import json
import random
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms, models
import types
import os
os.environ["WANDB_START_METHOD"] = "thread"


from foveated_dataloaders import (
    MultiGlimpseBoundingBoxDataset,
    MultiGlimpseDistortionAwareDataset,
)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for fulls, crops, labels, _ in loader:
        # handle multi-crop
        if crops.ndim == 5:
            B, N, C, H, W = crops.shape
            inputs = crops.view(B * N, C, H, W)
            labs = labels.unsqueeze(1).expand(-1, N).reshape(-1)
        else:
            inputs, labs = crops, labels

        inputs = inputs.to(device, non_blocking=True)
        labs   = labs.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        preds = outputs.argmax(dim=1)
        running_loss     += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labs)
        total_samples    += inputs.size(0)

    # aggregate across GPUs
    loss_tensor    = torch.tensor(running_loss,    device=device)
    correct_tensor = torch.tensor(running_corrects, device=device)
    samples_tensor = torch.tensor(total_samples,    device=device)
    dist.all_reduce(loss_tensor,    op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)

    avg_loss = loss_tensor.item() / samples_tensor.item()
    avg_acc  = correct_tensor.item() / samples_tensor.item()
    return avg_loss, avg_acc

def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for fulls, crops, labels, _ in loader:
            if crops.ndim == 5:
                B, N, C, H, W = crops.shape
                inputs = crops.view(B * N, C, H, W)
                labs   = labels.unsqueeze(1).expand(-1, N).reshape(-1)
            else:
                inputs, labs = crops, labels

            inputs = inputs.to(device, non_blocking=True)
            labs   = labs.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labs)

            preds = outputs.argmax(dim=1)
            val_loss     += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labs)
            total_samples+= inputs.size(0)

    # aggregate across GPUs
    loss_tensor    = torch.tensor(val_loss,    device=device)
    correct_tensor = torch.tensor(val_corrects, device=device)
    samples_tensor = torch.tensor(total_samples, device=device)
    dist.all_reduce(loss_tensor,    op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(samples_tensor, op=dist.ReduceOp.SUM)

    avg_loss = loss_tensor.item() / samples_tensor.item()
    avg_acc  = correct_tensor.item() / samples_tensor.item()
    return avg_loss, avg_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local_rank",
        type=int,
        default=int(os.getenv("LOCAL_RANK", 0)),
        help="DDP local rank",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="combined_A_B_500.json",
        help="Path to JSON file with train/val splits",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/imagenet/imagenet2012/train/",
        help="Root folder of ImageNet train images",
    )
    parser.add_argument(
        "--annotation_folder",
        type=str,
        default="default_path",
        help="Folder with bounding‐box annotations",
    )
    args = parser.parse_args()

    # ─── DDP Initialization ─────────────────────────────────────────────────────
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ─── Reproducibility ─────────────────────────────────────────────────────────
    SEED = 37 # Next 1, 2, 3, 37, 42 
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    # ─── Hyperparameters & W&B ──────────────────────────────────────────────────
    HYP = {
        "epochs":       90,
        "batch_size":   64,
        "learning_rate":0.1,
        "weight_decay": 1e-4,
        "step_size":    30,
        "gamma":        0.1,
        "num_glimpses": 3,
    }
    if rank == 0:
        import wandb
        wandb.init(
            project="imagenet-bbox-mergedAB-MultiGlimpse-multicrop-ddp37",
            config=HYP
        )
        config = wandb.config
        MODEL_SAVE_DIR = "ResNet_50_MultiGlimpseMultiBBOX_MergedAB_90_37"
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    else:
        config = types.SimpleNamespace(**HYP)
        wandb = None
        MODEL_SAVE_DIR = "ResNet_50_MultiGlimpseMultiBBOX_MergedAB_90_37"

    device = torch.device(f"cuda:{args.local_rank}")

    # ─── Transforms ──────────────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ─── Dataset Splits ──────────────────────────────────────────────────────────
    with open(args.splits, "r") as f:
        splits = json.load(f)
    train_indices = splits["train"]
    val_indices   = splits["val"]

    base_ds = datasets.ImageFolder(args.dataset_path)
    train_subset = Subset(base_ds, train_indices)
    val_subset   = Subset(base_ds, val_indices)

    # ─── Foveated Datasets ───────────────────────────────────────────────────────
    train_ds = MultiGlimpseDistortionAwareDataset(
        subset            = train_subset,
        annotation_folder = args.annotation_folder,
        crop_transform    = train_transform,
        resize_size       = (224, 224),
        train_mode        = True,
        offset_fraction   = 0.2,
        scale_jitter      = 0.1,
        area_threshold    = 0.2,
        augmentation_mode = "medium",
        num_glimpses      = config.num_glimpses,
        max_crop_ratio    = 0.2,
        multi_crop        = True,
    )
    val_ds = MultiGlimpseDistortionAwareDataset(
        subset            = val_subset,
        annotation_folder = args.annotation_folder,
        crop_transform    = val_transform,
        resize_size       = (224, 224),
        train_mode        = False,
        offset_fraction   = 0.2,
        scale_jitter      = 0.1,
        area_threshold    = 0.2,
        augmentation_mode = "medium",
        num_glimpses      = config.num_glimpses,
        max_crop_ratio    = 0.2,
        multi_crop        = True,
    )

    # ─── Samplers & Loaders ─────────────────────────────────────────────────────
    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_ds,   num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size     = config.batch_size,
        sampler        = train_sampler,
        num_workers    = 4,
        pin_memory     = True,
        worker_init_fn = seed_worker,
        generator      = g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size     = config.batch_size,
        sampler        = val_sampler,
        num_workers    = 4,
        pin_memory     = True,
        worker_init_fn = seed_worker,
        generator      = g,
    )

    # ─── Model, Loss, Optimizer, Scheduler, AMP ────────────────────────────────
    model = models.resnet50(
        pretrained=False,
        num_classes=len(base_ds.classes)
    ).to(device)
    model = DDP(model, device_ids=[args.local_rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.step_size,
        gamma=config.gamma
    )
    scaler = torch.cuda.amp.GradScaler()

    # ─── Training Loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    for epoch in range(config.epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        scheduler.step()
        if rank == 0:
            wandb.log({"train_loss": train_loss, "train_acc": train_acc}, step=epoch)

        val_sampler.set_epoch(epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        if rank == 0:
            wandb.log({"val_loss": val_loss, "val_acc": val_acc}, step=epoch)
            print(
                f"[Epoch {epoch+1:02d}/{config.epochs:02d}] "
                f"Train: {train_loss:.4f}, {train_acc:.4f} | "
                f"Val:   {val_loss:.4f}, {val_acc:.4f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    model.module.state_dict(),
                    os.path.join(MODEL_SAVE_DIR, "best_model.pth")
                )

    # ─── Final Checkpoint ───────────────────────────────────────────────────────
    if rank == 0:
        torch.save(
            model.module.state_dict(),
            os.path.join(MODEL_SAVE_DIR, "final_model.pth")
        )
        wandb.save(os.path.join(MODEL_SAVE_DIR, "final_model.pth"))
        wandb.finish()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
