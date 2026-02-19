# =============================================================================
# focl_CSL.py — FocL Memorization / CSL Logging (ImageNet BBox Crops)
# =============================================================================
#
# Goal
# ----
# Train a ResNet-50 on FocL-style *single* GT-bbox crops (224×224) while logging
# *per-sample* training loss and confidence over epochs. These per-sample traces
# are later used to compute memorization / cumulative sample loss (CSL) metrics.
#
#
# Main data / feature flow
# ------------------------
# ImageFolder(ImageNet train) + GT XML bbox  →  crop GT bbox  →  Resize(224,224)
# → Normalize  →  ResNet-50  →  per-sample CE loss (reduction='none')
# → log {loss[t], confidence[t]} for each sample index across epochs.
#
#
# IMPORTANT: Index tracking via an Indexed Dataset
# ------------------------------------------------
# Memorization logging requires a *stable identifier per sample* across epochs.
# If we log using the per-epoch DataLoader order (which is shuffled), we would
# lose the mapping between a row in the log and the underlying ImageNet image.
#
# To guarantee correct mappings, this script uses an IndexedBoundingBoxDataset
# that returns the *global ImageFolder index*:
#   global_index = self.subset.indices[index]
# This ensures the memorization log keys correspond to the original ImageFolder
# dataset indices (stable across epochs and shuffling).
#
#
# Separate "mem loader" (recommended for CSL)
# -------------------------------------------
# For CSL / memorization scores, the safest protocol is to compute per-sample
# losses with a dedicated, deterministic DataLoader (a "mem_loader"):
#   - shuffle = False
#   - drop_last = False
#   - deterministic transforms (no randomness)
# so that per-sample loss/confidence are measured consistently.
#
# Why?
# - If you compute mem scores from the *training* loader while it uses random
#   transforms or random shuffling, then the measured loss becomes stochastic
#   (random crops/jitter/etc.), and CSL/memorization scores can become noisy or
#   incorrect (they no longer correspond to a fixed per-sample difficulty).
#
#
# CSL protocol used here (paper-style)
# ------------------------------------
# - FocL Single-Crop ONLY (GT bbox crop).
# - NO data augmentation for CSL collection:
#     Resize((224,224)) → ToTensor → Normalize
#   This produces deterministic per-sample losses suitable for CSL.
#
# Optional: CSL with Augmentations
# -------------------------------
# You *can* also compute CSL under augmentations, but be careful:
# - Keep training-time augmentation if you want (for model learning).
# - For memorization/CSL logging, still use a deterministic mem_loader
#   (no random transforms) to remove randomness from the measured loss.
#
# In short:
#   Training can be random; memorization scoring should be deterministic.
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
import random
import numpy as np
from bounding_box_dataloader import BoundingBoxDataset

# Fix random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Initialize Weights and Biases
wandb.init(project="imagenet-bbox-A100-Noaugfixedtraining90", config={
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
MODEL_SAVE_DIR = "ResNet_50_BBOX_fixed_A100_NoAug_90"
MEMORIZATION_LOG_FILE = "memorization_dynamiclogfixed_bbox_Noaug_90.json"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# Load dataset splits
with open("imagenet_subset_splits_partition_A.json", "r") as f:
    splits = json.load(f)

train_indices = splits["train"]
val_indices = splits["val"]


crop_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
DATASET_PATH = "/local/a/imagenet/imagenet2012/train/"
ANNOTATION_FOLDER = "/Data/ImageNet_BBOX_Folder/Annotation"


# Updated Custom Dataset with Dynamic Global Indexing for BBox Training
class IndexedBoundingBoxDataset(BoundingBoxDataset):
    def __getitem__(self, index):
        full_image, cropped_image, label, bbox_tensor = super().__getitem__(index)
        # Main difference: Obtain the global index from the subset's indices instead of using the local index.
        global_index = self.subset.indices[index]
        return full_image, cropped_image, label, global_index

# Bounding Box Dataset
train_dataset = IndexedBoundingBoxDataset(
    subset=Subset(datasets.ImageFolder(DATASET_PATH), train_indices),
    annotation_folder=ANNOTATION_FOLDER,
    crop_transform=crop_transform
)

val_dataset = BoundingBoxDataset(
    subset=Subset(datasets.ImageFolder(DATASET_PATH), val_indices),
    annotation_folder=ANNOTATION_FOLDER,
    crop_transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Define model, loss function, optimizer, and scheduler
model = models.resnet50(pretrained=False, num_classes=len(train_dataset.subset.dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# AMP Scaler
scaler = GradScaler()

breakpoint()

# Function to update memorization log
def update_memorization_log(epoch, batch_indices, sample_losses, sample_confidences, file_path):
    # Load existing log if available
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            memorization_log = json.load(f)
    else:
        memorization_log = {str(idx): {"loss": [], "confidence": []} for idx in batch_indices}

    # Update entries
    for idx, loss, conf in zip(batch_indices, sample_losses, sample_confidences):
        memorization_log[str(idx)]["loss"].append(loss)
        memorization_log[str(idx)]["confidence"].append(conf)

    # Save updated log
    with open(file_path, "w") as f:
        json.dump(memorization_log, f, indent=4)

# Training loop with indexed tracking
best_val_acc = 0.0
for epoch in range(EPOCHS):
    model.train()
    running_loss, running_corrects = 0.0, 0
    sample_losses, sample_confidences, batch_indices_all = [], [], []

    for full_images, cropped_images, labels, batch_indices in train_loader:
        inputs, labels = cropped_images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass with AMP
        with autocast():
            outputs = model(inputs)
            loss_per_sample = criterion(outputs, labels)
            loss = loss_per_sample.mean()  # Mean loss for optimizer update
        
        # Compute confidence (softmax probability of correct class)
        probs = torch.softmax(outputs, dim=1)
        true_class_probs = probs.gather(1, labels.view(-1, 1)).squeeze().cpu().tolist()

        # Collect loss and confidence
        sample_losses.extend(loss_per_sample.detach().cpu().tolist())
        sample_confidences.extend(true_class_probs)
        batch_indices_all.extend(batch_indices.cpu().tolist())

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    scheduler.step()

    # Compute training metrics
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_corrects.double() / len(train_loader.dataset)
    wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})

    # Update memorization log
    update_memorization_log(epoch, batch_indices_all, sample_losses, sample_confidences, MEMORIZATION_LOG_FILE)

    # Validation
    model.eval()
    val_loss, val_corrects = 0.0, 0

    with torch.no_grad():
        for full_images, cropped_images, labels, _ in val_loader:
            inputs, labels = cropped_images.to(device), labels.to(device)
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

    # Print Metrics
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))

# Final Save
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "final_model.pth"))
wandb.save(os.path.join(MODEL_SAVE_DIR, "final_model.pth"))
