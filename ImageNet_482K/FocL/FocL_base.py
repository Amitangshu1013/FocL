# =============================================================================
# FocL_base.py — Foveated Object-Centric Learning (FocL) Baseline Training Script
# =============================================================================
#
# What this script does
# ---------------------
# Trains an ImageNet classifier (ResNet-50) on *foveated object crops* produced
# from bounding-box annotations. Each training sample yields:
#   (full_image, cropped_bbox_image, class_label, metadata)
# but the model is trained *only* on the cropped_bbox_image (the “foveated” view).
#
# This is the “FocL-Base” baseline: crop using GT bounding boxes + standard
# ImageNet-style augmentations, train a standard ResNet-50, log to W&B, and save
# best/final checkpoints.
#
#
# Inputs / expected files
# -----------------------
# 1) ImageNet train directory (ImageFolder layout):
#      DATASET_PATH = "/imagenet2012/train/"
#    containing per-class subfolders with images.
#
# 2) Bounding-box annotation folder:
#      ANNOTATION_FOLDER = "/Data/ImageNet_BBOX_Folder/Annotation"
#    used by `BoundingBoxDataset` (imported from `foveated_dataloaders`).
#
# 3) Train/val split JSON:
#      "combined_A_B_500.json"
#    with keys:
#      - splits["train"] : list of indices into ImageFolder
#      - splits["val"]   : list of indices into ImageFolder
#


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import wandb
from torch.cuda.amp import GradScaler, autocast
import os
import json
from foveated_dataloaders import BoundingBoxDataset  # FocL Base Dataloader
import random
from torchvision.transforms import functional as F


# Initialize Weights and Biases
wandb.init(project="imagenet-bbox-mergeAB-FocLBase_ResNet50", config={
    "epochs": 90,
    "batch_size": 128,
    "learning_rate": 0.1,
    "weight_decay": 1e-4,
    "step_size": 30,  # Decay every 30 epochs for 90 total epochs
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
MODEL_SAVE_DIR = "ResNet50_BBOX_MergedAB_500"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


#### Some custom Augmentation transforms, not used

class Cutout:
    def __init__(self, p=0.3, scale=(0.02, 0.15)):
        self.p = p
        self.scale = scale

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            w, h = img.size
            erase_w = int(random.uniform(*self.scale) * w)
            erase_h = int(random.uniform(*self.scale) * h)
            x = random.randint(0, w - erase_w)
            y = random.randint(0, h - erase_h)
            img = F.erase(img, x, y, erase_w, erase_h, 0, inplace=False)
        return img
    
class DynamicRandomErasing:
    def __init__(self, base_p=0.3, base_scale=(0.05, 0.1)):
        self.base_p = base_p
        self.base_scale = base_scale

    def __call__(self, img, bbox_size):
        """
        Args:
            img: Normalized tensor image.
            bbox_size: Tuple of (width, height) for the bounding box.

        Returns:
            Augmented tensor image with random erasing applied.
        """
        # Erasing logic stays the same, as normalization doesn’t change spatial properties
        bbox_area = bbox_size[0] * bbox_size[1]

        # Adjust parameters based on bbox size
        if bbox_area > 0.5:  # Large bounding box
            p = self.base_p + 0.1
            scale = (self.base_scale[0] * 2, self.base_scale[1] * 2)
        elif bbox_area < 0.1:  # Small bounding box
            p = self.base_p - 0.1
            scale = (self.base_scale[0] / 2, self.base_scale[1] / 2)
        else:  # Medium bounding box
            p = self.base_p
            scale = self.base_scale

        # Apply RandomErasing
        eraser = transforms.RandomErasing(p=p, scale=scale)
        return eraser(img)



class DynamicCropTransform:
    def __init__(self, bbox_area_threshold=0.1):
        """
        Args:
            bbox_area_threshold: Relative area of the bounding box below which augmentations are reduced.
        """
        self.bbox_area_threshold = bbox_area_threshold
        self.basic_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.advanced_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.05)),
        ])

    def __call__(self, image, bbox):
        # Compute bounding box area relative to the image
        width, height = image.size
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height)

        if bbox_area < self.bbox_area_threshold:
            # Apply less aggressive basic transform for small bounding boxes
            return self.basic_transform(image)
        else:
            # Apply advanced transform for larger bounding boxes
            return self.advanced_transform(image)



# Data Augmentation and Transforms
# Train Transform (for cropped bounding boxes)
crop_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize cropped bounding box slightly larger
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Random crop to final size
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Minimal Test Transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize cropped region
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#crop_transform = DynamicCropTransform(bbox_area_threshold=0.05) # Use DynamicBoundingBoxDataset class # definition becomes different

# Dataset and DataLoader
DATASET_PATH = "/imagenet2012/train/"
ANNOTATION_FOLDER = "/Data/ImageNet_BBOX_Folder/Annotation" 
# Load Train/Val Splits

with open("combined_A_B_500.json", "r") as f: # For merged
    splits = json.load(f)
train_indices = splits["train"]
val_indices = splits["val"]

# Bounding Box Dataset
train_dataset = BoundingBoxDataset(
    subset=torch.utils.data.Subset(datasets.ImageFolder(DATASET_PATH), train_indices),
    annotation_folder=ANNOTATION_FOLDER,
    crop_transform=crop_transform
)

val_dataset = BoundingBoxDataset(
    subset=torch.utils.data.Subset(datasets.ImageFolder(DATASET_PATH), val_indices),
    annotation_folder=ANNOTATION_FOLDER,
    crop_transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Model, Loss, Optimizer, Scheduler
model = models.resnet50(pretrained=False, num_classes=len(train_dataset.subset.dataset.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# AMP Scaler
scaler = GradScaler()

# Training Function
def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss, running_corrects = 0.0, 0

    for full_images, cropped_images, labels, _ in train_loader:
        inputs, labels = cropped_images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass with AMP
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Statistics
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_corrects.double() / len(train_loader.dataset)
    return train_loss, train_acc

# Validation Function
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss, val_corrects = 0.0, 0

    with torch.no_grad():
        for full_images, cropped_images, labels, _ in val_loader:
            inputs, labels = cropped_images.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)
    return val_loss, val_acc

# Training Loop
best_val_acc = 0.0
for epoch in range(EPOCHS):
    # Train for one epoch
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
    scheduler.step()

    # Log training metrics
    wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})

    # Validate
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    wandb.log({"val_loss": val_loss, "val_accuracy": val_acc})

    # Print Metrics
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save Best Model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "best_model.pth"))

    # Save Model Every 20 Epochs
    #if (epoch + 1) % 15 == 0:
    #    torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"model_epoch_{epoch+1}.pth"))

# Final Save
torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "final_model.pth"))
wandb.save(os.path.join(MODEL_SAVE_DIR, "final_model.pth"))
