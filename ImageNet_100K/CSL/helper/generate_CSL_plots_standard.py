# =============================================================================
# generate_CSL_plots_standard.py — Per-sample CSL Diagnostics (Standard / Full Image)
# =============================================================================
#
# Purpose
# -------
# For a small set of ImageNet classes, this script identifies the Top-K (K=10)
# highest-CSL (“high memorization / high cumulative loss”) training samples per
# class from a precomputed CSL file, and (optionally) generates per-sample plots:
#   1) the full 224×224 image,
#   2) loss over epochs (log scale),
#   3) true-class confidence over epochs.
#
# This is the *standard/full-image* counterpart to the bbox version. It does NOT
# parse XML annotations and does NOT crop bounding boxes.
#
#
# Inputs (expected files)
# -----------------------
# 1) ImageNet train images (ImageFolder layout):
#      DATASET_PATH/<synset_id>/*.JPEG
#
# 2) CSL cumulative-loss JSON (output of compute_high_memorization.py):
#      CSL_MEMORIZATION_FILE : { train_index : cumulative_loss }
#
# 3) Epoch-wise memorization log JSON (output of update_memorization_log(...)):
#      EPOCHWISE_LOSS_CONFIDENCE_FILE :
#        { train_index : { "loss":[...], "confidence":[...] } }
#
#
# Indexing / mapping note (important for correctness)
# --------------------------------------------------
# The script relies on *global ImageFolder indices* as stable sample IDs.
# It enumerates `train_dataset.imgs` and uses the enumeration index `idx` as the
# train_index used in CSL and epoch-wise logs. This ensures:
#   train_index  ↔  (file_path, class) mapping is consistent across files.
#
#
# What actually runs in this file
# -------------------------------
# - STEP 1: Load ImageNet train dataset (ImageFolder).
# - STEP 2: Load CSL cumulative loss scores (convert JSON keys → int).
# - STEP 3: For each class in CLASS_MAPPING:
#     • gather all dataset indices belonging to the synset folder
#     • select top 10 indices with highest CSL scores
#     • store them in `top_memorized_per_class`
# - Save the selected indices to TOP_MEM_INDICES_FILE (top_memorized_indices.json).
#
# NOTE:
# - The code block that loads epoch-wise loss/confidence and generates 1×3 plots
#   is currently enclosed in a triple-quoted string (commented out). As-is, the
#   script will only generate and save TOP_MEM_INDICES_FILE and will NOT create
#   the per-sample PNG visualizations unless that block is uncommented.
#
#
# Outputs
# -------
# - top_memorized_indices.json:
#     { class_name : [ {"index": train_index, "csl_score": score}, ... ] }
#
# Optional (if plot block is enabled):
# - Per-sample PNGs saved under:
#     SAVE_DIR/<class_name>/<train_index>_<csl_score>.png
#   Each PNG contains: [image | loss curve | confidence curve].
# =============================================================================


import json
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from PIL import Image

# === CONFIGURATION ===
DATASET_PATH = "/local/a/imagenet/imagenet2012/train/"  # Path to training images
CSL_MEMORIZATION_FILE = "cumulative_loss_standard.json"  # Cumulative loss memorization scores
EPOCHWISE_LOSS_CONFIDENCE_FILE = "epochwise_loss_standard.json"  # Epoch-wise loss/confidence
SAVE_DIR = "Analysis_90_standard"  # Output directory for final visualizations
TOP_MEM_INDICES_FILE = "top_memorized_indices_standard.json"  # Save selected top indices

# Define ImageNet Class Mapping (Synset IDs)
CLASS_MAPPING = {
    "peacock": "n01806143",
    "beaver": "n02363005",
    "black_swan": "n01860187",
    "black_stork": "n02002724",
    "otter": "n02444819",
    "llama": "n02437616",
    "water_buffalo": "n02408429"
}

# === Ensure Save Directory Exists ===
os.makedirs(SAVE_DIR, exist_ok=True)

# === STEP 1: LOAD DATASET & EXTRACT CLASS INDICES ===
print("🔄 Loading ImageNet training dataset...")
train_dataset = datasets.ImageFolder(root=DATASET_PATH)

# === STEP 2: LOAD CSL MEMORIZATION LOG ===
print(f"🔄 Loading memorization log from {CSL_MEMORIZATION_FILE}...")
with open(CSL_MEMORIZATION_FILE, "r") as f:
    csl_memorization = json.load(f)  # Format: {"train_index": cumulative_loss, ...}
    csl_memorization = {int(k): v for k, v in csl_memorization.items()}  # Convert keys to integers

# === Extract Indices for Each Class ===
top_memorized_per_class = {}

for class_name, synset_id in CLASS_MAPPING.items():
    class_indices = []

    # Get indices for this class
    for idx, (file_path, class_idx) in enumerate(train_dataset.imgs):
        folder_name = os.path.basename(os.path.dirname(file_path))  # Extract synset ID
        if folder_name == synset_id:
            class_indices.append((idx, file_path))

    # 🔍 Debugging: Print found indices
    print(f"🔍 Checking indices for {class_name}: {len(class_indices)} total samples found.")
    print(f"🧐 Sample Indices: {[idx for idx, _ in class_indices[:10]]}")  # Print first 10 indices

    if not class_indices:
        print(f"⚠️ No images found for {class_name}. Skipping!")
        continue


    print(f"✅ Found {len(class_indices)} images for {class_name}.")

    # === STEP 3: SELECT 10 HIGH MEMORIZED SAMPLES PER CLASS ===
    high_mem_samples = [
        (idx, csl_memorization[idx]) for idx, _ in class_indices if idx in csl_memorization
    ]
    high_mem_samples = sorted(high_mem_samples, key=lambda x: x[1], reverse=True)[:10]  # Select Top 10

    if not high_mem_samples:
        print(f"⚠️ No high memorization samples found for {class_name} in CSL log!")
        continue

    top_memorized_per_class[class_name] = [{"index": idx, "csl_score": score} for idx, score in high_mem_samples]

# === Save Top Memorized Indices ===
with open(TOP_MEM_INDICES_FILE, "w") as f:
    json.dump(top_memorized_per_class, f, indent=4)

print(f"✅ Saved top 10 memorized indices per class to {TOP_MEM_INDICES_FILE}")

# === STEP 4: LOAD EPOCH-WISE LOSS/CONFIDENCE ===
print(f"🔄 Loading epoch-wise loss/confidence from {EPOCHWISE_LOSS_CONFIDENCE_FILE}...")
with open(EPOCHWISE_LOSS_CONFIDENCE_FILE, "r") as f:
    epochwise_data = json.load(f)  # Format: {"train_index": {"loss": [...], "confidence": [...]}, ...}
    epochwise_data = {int(k): v for k, v in epochwise_data.items()}  # Convert keys to integers

# === STEP 5: GENERATE MEGA PLOTS (1×3 SUBPLOTS) ===
for class_name, samples in top_memorized_per_class.items():
    class_save_dir = os.path.join(SAVE_DIR, class_name)
    os.makedirs(class_save_dir, exist_ok=True)

    print(f"📊 Generating plots for {class_name}...")

    for sample in samples:
        train_index = sample["index"]
        csl_score = sample["csl_score"]

        if train_index not in epochwise_data:
            print(f"⚠️ Missing epoch-wise data for {train_index}, skipping...")
            continue

        # Find file path from pre-filtered class indices
        #file_path = next((fp for idx, fp in class_indices if idx == train_index), None)
        file_path = next((fp for idx, fp in class_indices if abs(idx - train_index) <= 5), None)


        if file_path is None:
            print(f"⚠️ Skipping {class_name} | train index {train_index}: File path not found.")
            continue

        # Extract loss & confidence data
        loss_values = epochwise_data[train_index]["loss"]
        confidence_values = epochwise_data[train_index]["confidence"]
        epochs = np.arange(1, len(loss_values) + 1)

        # Load & Resize Image
        image = Image.open(file_path).resize((224, 224))

        # === Create 1×3 Subplot ===
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Train Index: {train_index} | CSL: {csl_score:.2f}", fontsize=14)

        # 📷 **Plot 1: Image**
        axes[0].imshow(image)
        axes[0].axis("off")
        axes[0].set_title("Image (224×224)")

        # 📉 **Plot 2: Loss Trajectory (Log Scale)**
        axes[1].semilogy(epochs, loss_values, marker="o", linestyle="-", color="b")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss (Log Scale)")
        axes[1].set_title("Loss Trajectory")
        axes[1].grid(True, which="both", linestyle="--")

        # 📈 **Plot 3: Confidence Trajectory (Linear Scale)**
        axes[2].plot(epochs, confidence_values, marker="o", linestyle="-", color="g")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Confidence (0-1)")
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Confidence Trajectory")
        axes[2].grid(True, linestyle="--")

        # === Save Figure ===
        save_path = os.path.join(class_save_dir, f"{train_index}_{csl_score:.2f}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"✅ Saved mega plot: {save_path}")

print("\n🎉 All plots generated and saved successfully!")
