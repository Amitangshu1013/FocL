#!/usr/bin/env python3
# =============================================================================
# Notes on transforms in this HardMask pipeline
# =============================================================================
# The key detail is *when* transforms are applied:
#
#   1) Load raw PIL image at original resolution (W×H).
#   2) Apply HardMask *at original resolution*:
#        - Parse XML bbox.
#        - Create a binary mask that is white inside bbox, black outside.
#        - Replace pixels outside bbox with a solid mean-color background
#          (approx ImageNet mean in 0–255 RGB: (124,116,104)).
#        - Result is still a PIL RGB image of the same original size.
#   3) Apply torchvision transforms *after masking* (still in PIL):
#        - This is important: cropping/augmentation happens on the “cleaned”
#          image, not on the original cluttered image.
#
# Practical implications / gotchas:
#   - Because RandomResizedCrop happens AFTER masking, the crop distribution is
#     different from standard ImageNet: it samples from a background that is
#     uniform mean-color rather than natural clutter.
#   - If XML is missing/corrupt, HardMaskDataset falls back to the original full
#     image, and then transforms behave exactly like standard ImageNet transforms.
#   - This baseline is NOT “bbox crop” training (FocL). It is full-frame input
#     where background pixels are replaced, and then a standard crop pipeline is
#     applied to produce 224×224 inputs for ResNet.
# =============================================================================




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

import wandb
from torch.cuda.amp import GradScaler, autocast

# --- IMPORT YOUR CUSTOM DATASET ---
from hardmask_dataloader import HardMaskDataset



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=int(os.getenv("LOCAL_RANK", 0)))
    args = parser.parse_args()

    # ─── Set Random Seed ──────────────────────────────────────────────────────
    seed = 2  # Or any number you like
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    local_rank = args.local_rank

    # ─── DDP Init ─────────────────────────────────────────────────────────────
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # ─── Configuration ─────────────────────────────────────────────────────────
    # UPDATE THIS PATH to where your ImageNet XML Annotations are stored
    # Structure should be: annotation_folder/n01440764/n01440764_10026.xml
    ANNOTATION_PATH = "/ImageNet_BBOX_Folder/Annotation"
    
    DATASET_PATH = "/imagenet/imagenet2012/train/"
    
    # Save directory for this specific baseline
    SAVE_DIR = "ResNet50_HardMask_Baseline"
    
    if rank == 0:
        wandb.init(
            project="HardMask_Baseline_Rebuttal",
            config={
                "epochs": 90,
                "batch_size": 128,
                "learning_rate": 0.1,
                "weight_decay": 1e-4,
                "step_size": 30,
                "gamma": 0.1,
                "dataset_type": "HardMask (Mean Pad)"
            }
        )
    cfg = wandb.config if rank == 0 else None
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    device = torch.device(f"cuda:{local_rank}")

    # ─── Data Transforms ───────────────────────────────────────────────────────
    # Note: For HardMask, we pass this transform to the dataset class.
    # It will apply RandomResizedCrop AFTER masking.
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
    # 1. Load Splits
    with open("combined_A_B_500.json", "r") as f:
        splits = json.load(f)

    # 2. Prepare Training Data (HardMask)
    #    We create a base ImageFolder WITHOUT transforms first, because HardMask
    #    needs to load the raw image, mask it, and THEN apply transforms.
    base_train_ds = datasets.ImageFolder(root=DATASET_PATH) # No transform here
    train_subset_raw = Subset(base_train_ds, splits["train"])
    
    #    Wrap it in HardMaskDataset
    train_ds = HardMaskDataset(
        subset=train_subset_raw,
        annotation_folder=ANNOTATION_PATH,
        crop_transform=train_transform  # Pass transforms here
    )

    # 3. Prepare Validation Data (HardMask)
    #val_ds_base = datasets.ImageFolder(root=DATASET_PATH, transform=test_transform)
    #val_ds = Subset(val_ds_base, splits["val"])

    val_ds_raw = datasets.ImageFolder(root=DATASET_PATH) # No transform yet
    val_subset_raw = Subset(val_ds_raw, splits["val"])
    
    val_ds = HardMaskDataset(
        subset=val_subset_raw,
        annotation_folder=ANNOTATION_PATH,
        crop_transform=test_transform # Use test_transform (Resize 256 -> CenterCrop 224)
    )

    # ─── Distributed Samplers & Loaders ────────────────────────────────────────
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size if rank==0 else 128,
        sampler=train_sampler,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size if rank==0 else 128,
        sampler=val_sampler,
        num_workers=4, pin_memory=True
    )

    # ─── Model, Loss, Optimizer ────────────────────────────────────────────────
    # Initialize model (Standard ResNet50)
    # We use len(base_train_ds.classes) because train_ds is a wrapper
    num_classes = len(base_train_ds.classes)
    model = models.resnet50(pretrained=False, num_classes=num_classes).to(device)
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
    # Use config epochs if rank 0, else default to 90
    total_epochs = cfg.epochs if rank == 0 else 90

    for epoch in range(total_epochs):
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

        # ─── Aggregate stats ──────────────────────────────────────────────────
        dist.all_reduce(run_loss,     op=dist.ReduceOp.SUM)
        dist.all_reduce(run_corrects, op=dist.ReduceOp.SUM)

        # Use len(train_ds) instead of len(train_subset)
        epoch_loss = run_loss.item() / len(train_ds)
        epoch_acc  = run_corrects.item() / len(train_ds)
        scheduler.step()

        if rank == 0:
            wandb.log({"train_loss": epoch_loss, "train_accuracy": epoch_acc}, step=epoch)

        # ─── Validation ───────────────────────────────────────────────────────
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

        val_epoch_loss = val_loss.item() / len(val_ds)
        val_epoch_acc  = val_corrects.item() / len(val_ds)

        if rank == 0:
            wandb.log({"val_loss": val_epoch_loss, "val_accuracy": val_epoch_acc}, step=epoch)
            print(f"[Epoch {epoch+1:>2}/{total_epochs}] "
                  f"Train: loss={epoch_loss:.4f}, acc={epoch_acc:.4f} | "
                  f"Val:   loss={val_epoch_loss:.4f}, acc={val_epoch_acc:.4f}")

            # ─── Save best model ──────────────────────────────────────────────
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                torch.save(model.module.state_dict(),
                           os.path.join(SAVE_DIR, "best_model.pth"))

    # ─── Final save ───────────────────────────────────────────────────────────
    if rank == 0:
        torch.save(model.module.state_dict(),
                   os.path.join(SAVE_DIR, "final_model.pth"))
        wandb.save(os.path.join(SAVE_DIR, "final_model.pth"))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()