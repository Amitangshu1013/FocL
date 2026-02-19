# =============================================================================
# standard_CSL.py — Baseline Memorization / CSL Logging (Full-Image ResNet-50)
# =============================================================================
#
# Goal
# ----
# Train a standard full-image ResNet-50 baseline and log *per-sample* training
# loss and true-class confidence across epochs. These per-sample trajectories
# are later used to compute memorization metrics such as Cumulative Sample Loss
# (CSL) and related difficulty / memorization analyses.
#
#
# Main data / feature flow
# ------------------------
# ImageNet full frame → deterministic preprocessing (no aug) → ResNet-50
# → per-sample cross-entropy loss (reduction='none')
# → per-sample true-class softmax confidence
# → append {loss[t], confidence[t]} for each sample index over epochs
# → save as a JSON memorization log on disk.
#
#
# IMPORTANT: Stable per-sample indexing (to ensure correct mappings)
# ---------------------------------------------------------------
# Memorization logging requires a *stable identifier per training image*.
# This script uses `IndexedImageFolder`, which returns:
#     (image, label, index)
# where `index` is the *global* ImageFolder index (stable across epochs).
#
# Because the training set is created via:
#     train_subset = Subset(train_dataset, train_indices)
# and the DataLoader shuffles the subset, we MUST log using the provided
# `index` values (not batch order) so that logged loss/confidence maps back
# to the correct original ImageNet image.
#
#
# CSL protocol note (no augmentation)
# ----------------------------------
# For CSL / memorization scoring, this script uses deterministic transforms
# (Resize → CenterCrop → Normalize). This removes randomness from the input,
# so changes in per-sample loss across epochs reflect learning dynamics rather
# than stochastic augmentations.
#
# You *can* train with augmentations if desired, but be careful:
#   - Do NOT collect memscores from a random/augmented loader unless you accept
#     augmentation-induced noise in the memorization signal.
#   - Best practice is to use a separate deterministic "mem_loader" (no random
#     transforms, shuffle=False) for collecting loss/confidence used in CSL,
#     while training can still use augmentation.
#
#
# What gets saved
# --------------
# - Memorization log JSON (MEMORIZATION_LOG_FILE):
#     { "<global_idx>": { "loss": [...], "confidence": [...] }, ... }
#   updated once per epoch using the dynamically collected indices for that epoch.
#
# - Checkpoints:
#     MODEL_SAVE_DIR/best_model.pth   (best val accuracy)
#     MODEL_SAVE_DIR/final_model.pth  (final epoch)
#
# - Weights & Biases:
#     logs train/val loss and accuracy each epoch.
# =============================================================================



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import wandb
from torch.cuda.amp import GradScaler, autocast
import os
import json
import numpy as np
import random


# Fix random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


# Initialize Weights and Biases
wandb.init(project="imagenet-subset-Aggaug_A50_ResNet50_training_90", config={
    "epochs": 90, 
    "batch_size": 128, 
    "learning_rate": 0.1,
    "weight_decay": 1e-4,
    "step_size": 30,  
    "gamma": 0.1
})

# Configuration
config = wandb.config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config.batch_size
EPOCHS = config.epochs
LR = config.learning_rate
WEIGHT_DECAY = config.weight_decay
STEP_SIZE = config.step_size
GAMMA = config.gamma
MODEL_SAVE_DIR = "ResNet_50"
MEMORIZATION_LOG_FILE = "memorization_dynamiclog_supervised_ResNet50_original_A_50.json"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Load dataset splits
with open("imagenet_subset_splits_50_A_train_123.json", "r") as f:
    splits = json.load(f)

train_indices = splits["train"]
val_indices = splits["val"]


# Define transformations no Aug
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ✅ Custom Dataset for Indexed Training
class IndexedImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)  # Get image and label
        return img, label, index  # ✅ Return image, label, and index

# ✅ Load Training Dataset with Indexing
train_dataset = IndexedImageFolder(root="/imagenet/imagenet2012/train/", transform=train_transform)
train_subset = Subset(train_dataset, train_indices)

# ✅ Load Validation Dataset (No Indexing Needed)
val_dataset = datasets.ImageFolder(root="/imagenet/imagenet2012/train/", transform=test_transform)
val_subset = Subset(val_dataset, val_indices)

# ✅ Create DataLoaders
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ✅ Define Model, Loss Function, Optimizer, Scheduler
model = models.resnet50(weights=None, num_classes=len(train_dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ✅ AMP Scaler
scaler = GradScaler()

# ✅ Function to Update Memorization Log
def update_memorization_log(epoch, batch_indices, sample_losses, sample_confidences, file_path):
    # Load existing log if available
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            memorization_log = json.load(f)
    else:
        memorization_log = {str(idx): {"loss": [], "confidence": []} for idx in train_indices}

    # ✅ Update entries using dynamic batch_indices
    for idx, loss, conf in zip(batch_indices, sample_losses, sample_confidences):
        memorization_log[str(idx)]["loss"].append(loss)
        memorization_log[str(idx)]["confidence"].append(conf)

    # ✅ Save updated log
    with open(file_path, "w") as f:
        json.dump(memorization_log, f, indent=4)

#breakpoint()
# ✅ Training Loop (Without Gradient Tracking)
best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0
    sample_losses = []
    sample_confidences = []
    batch_indices_all = []  # Track batch indices

    for inputs, labels, batch_indices in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # ✅ Forward Pass with AMP
        with autocast():
            outputs = model(inputs)
            loss_per_sample = criterion(outputs, labels)
            loss = loss_per_sample.mean()  # Mean loss for optimizer update
        
        # ✅ Compute Confidence (Softmax Probability of Correct Class)
        probs = torch.softmax(outputs, dim=1)
        true_class_probs = probs.gather(1, labels.view(-1, 1)).squeeze().cpu().tolist()

        # ✅ Collect Loss and Confidence
        sample_losses.extend(loss_per_sample.detach().cpu().tolist())
        sample_confidences.extend(true_class_probs)
        batch_indices_all.extend(batch_indices.tolist())  # Store batch indices

        # ✅ Backward Pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ✅ Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    scheduler.step()

    # ✅ Compute Training Metrics
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_corrects.double() / len(train_loader.dataset)
    wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})

    # ✅ Update Memorization Log Using Batch Indices
    update_memorization_log(epoch, batch_indices_all, sample_losses, sample_confidences, MEMORIZATION_LOG_FILE)

    # ✅ Validation
    model.eval()
    val_loss, val_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                loss_per_sample = criterion(outputs, labels)  
                loss = loss_per_sample.mean()  # Mean loss for validation

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)
    wandb.log({"val_loss": val_loss, "val_accuracy": val_acc})

    # ✅ Print Metrics
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # ✅ Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))

# ✅ Final Save
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "final_model.pth"))
wandb.save(os.path.join(MODEL_SAVE_DIR, "final_model.pth"))


