# =============================================================================
# generate_plots_bbox.py — Per-sample CSL Diagnostics (BBox Crop + Loss/Conf Curves)
# =============================================================================
#
# Purpose
# -------
# Generate qualitative “memorization diagnostics” for a chosen ImageNet class by
# visualizing (for ~10 selected training samples):
#   (1) the GT bounding-box crop (resized to 224×224),
#   (2) the per-epoch loss trajectory (log scale),
#   (3) the per-epoch true-class confidence trajectory.
#
# This helps interpret CSL / memorization results by linking high-CSL samples to
# what the model actually “saw” (bbox crop) and how learning evolved over epochs.
#
#
# Inputs (files this script expects)
# ---------------------------------
# 1) ImageNet train images (ImageFolder layout):
#      DATASET_PATH/<synset_id>/*.JPEG
#
# 2) ImageNet XML bbox annotations (per synset folder):
#      ANNOTATION_FOLDER/<synset_id>/<image_id>.xml
#
# 3) Cumulative loss (CSL) file (output of compute_high_memorization.py):
#      BBOX_CSL_FILE : { train_index : cumulative_loss }
#
# 4) Epoch-wise memorization log (output of update_memorization_log(...)):
#      EPOCHWISE_LOSS_CONFIDENCE_FILE :
#        { train_index : { "loss":[...], "confidence":[...] } }
#
# 5) Top-memorized indices file (pre-selected sample list per class):
#      TOP_MEM_INDICES_FILE :
#        { class_name : [ { "index": <train_index>, "csl_score": <old_score> }, ... ] }
#
#
# How it works (high-level flow)
# ------------------------------
# 1) Parse --class_name and map it to an ImageNet synset ID via CLASS_MAPPING.
# 2) Load ImageFolder(ImageNet train) and collect all global indices belonging to
#    that synset (to ensure correct index → filepath mapping).
# 3) Load:
#      - top_samples (10 samples) for the selected class from TOP_MEM_INDICES_FILE
#      - bbox_csl_data (new CSL scores) from BBOX_CSL_FILE
#      - epochwise_data (loss/confidence per epoch) from EPOCHWISE_LOSS_CONFIDENCE_FILE
# 4) For each selected sample:
#      - locate the image file path using the global train index
#      - parse bbox from XML (uses the FIRST valid bbox)
#      - crop bbox and resize to 224×224
#      - read loss/conf curves across epochs from epochwise_data
#      - create a 1×3 figure:
#           [bbox crop | loss trajectory (semilogy) | confidence trajectory]
#      - save as:
#           SAVE_DIR/<class_name>/<train_index>_<oldCSL>_<newCSL>.png
#
#
# Notes / assumptions
# -------------------
# - Uses “global ImageFolder indices” consistently across:
#     • top_memorized indices
#     • epochwise memorization logs
#     • CSL cumulative-loss file
#   This indexing consistency is crucial; otherwise plots may mismatch samples.
#
# - Bounding box selection:
#     uses the FIRST valid bbox from the XML (not necessarily the largest but usually is the largest for such a curated set).
#
# - This is a visualization tool (no training/eval). It only reads logs and writes
#   PNG figures for quick inspection.
#
#
# Example
# -------
# python generate_plots_bbox.py --class_name peacock
# =============================================================================



import json
import os
import argparse
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from PIL import Image


# === CONFIGURATION ===
DATASET_PATH = "/imagenet/imagenet2012/train/"  # Path to training images
BBOX_CSL_FILE= "cumulative_loss.json"  # Cumulative loss memorization scores
EPOCHWISE_LOSS_CONFIDENCE_FILE = "epochwise_loss.json"  # Epoch-wise loss/confidence
ANNOTATION_FOLDER = "/home/min/a/mukher44/Work/SSL/Data/ImageNet_BBOX_Folder/Annotation"
TOP_MEM_INDICES_FILE = "top_memorized_indices.json"  # Save selected top indices
SAVE_DIR = "Analysis_90_bbox"  # Output directory for bounding box plots
TARGET_SIZE = (224, 224)  # Target image size

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


# === COMMAND LINE ARGUMENTS ===
parser = argparse.ArgumentParser(description="Generate loss/confidence plots for one class.")
parser.add_argument("--class_name", required=True, help="Class name (e.g., peacock, beaver)")
args = parser.parse_args()
selected_class = args.class_name  # Use the new name


if selected_class not in CLASS_MAPPING:
    print(f"❌ Error: '{selected_class}' is not a valid class.")
    print(f"✅ Choose from: {list(CLASS_MAPPING.keys())}")
    exit(1)

# === STEP 1: LOAD IMAGE DATASET (LIKE BEFORE) ===
print(f"\n🔄 Loading ImageNet training dataset for '{selected_class}'...")
train_dataset = datasets.ImageFolder(root=DATASET_PATH)

# === STEP 2: LOAD TOP MEMORIZED INDICES FOR SELECTED CLASS ===
print(f"🔄 Loading top indices from {TOP_MEM_INDICES_FILE}...")
with open(TOP_MEM_INDICES_FILE, "r") as f:
    top_memorized = json.load(f)

if selected_class not in top_memorized or len(top_memorized[selected_class]) == 0:
    print(f"❌ No high-memorization samples found for {selected_class}. Exiting.")
    exit(1)

top_samples = top_memorized[selected_class]  # List of 10 samples
synset_id = CLASS_MAPPING[selected_class]


# === STEP 3: LOAD BBOX CSL MEMORIZATION LOG ===
print(f"🔄 Loading Bbox CSL scores from {BBOX_CSL_FILE}...")
with open(BBOX_CSL_FILE, "r") as f:
    bbox_csl_data = json.load(f)
    bbox_csl_data = {int(k): v for k, v in bbox_csl_data.items()}  # Convert keys to integers


# === STEP 3: EXTRACT TRAIN INDICES FOR THIS CLASS ===
class_indices = []
for idx, (file_path, class_idx) in enumerate(train_dataset.imgs):
    folder_name = os.path.basename(os.path.dirname(file_path))  # Extract synset ID
    if folder_name == synset_id:
        class_indices.append((idx, file_path))

print(f"✅ Found {len(class_indices)} images for '{selected_class}'.")
print(f"🔍 First 10 indices: {[idx for idx, _ in class_indices[:10]]}")

# === STEP 4: LOAD EPOCH-WISE LOSS & CONFIDENCE DATA ===
print(f"🔄 Loading epoch-wise loss/confidence from {EPOCHWISE_LOSS_CONFIDENCE_FILE}...")
with open(EPOCHWISE_LOSS_CONFIDENCE_FILE, "r") as f:
    epochwise_data = json.load(f)
    epochwise_data = {int(k): v for k, v in epochwise_data.items()}  # Convert keys to integers

# === STEP 5: EXTRACT BOUNDING BOXES FROM XML ===
def parse_xml_for_bbox(xml_file):
    """Parse an XML file to extract bounding box coordinates."""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        # Ensure the coordinates are valid
        if xmin < xmax and ymin < ymax:
            bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes

# === STEP 6: GENERATE PLOTS ===
class_save_dir = os.path.join(SAVE_DIR, selected_class)
os.makedirs(class_save_dir, exist_ok=True)

print(f"\n📊 Generating 10 bbox plots for '{selected_class}'...")

for sample in top_samples:
    train_index = sample["index"]
    old_csl_score = sample["csl_score"]

    if train_index not in epochwise_data:
        print(f"⚠️ Skipping {train_index}: Missing epoch-wise data.")
        continue

    # ✅ Get New Bbox CSL Score
    new_csl_score = bbox_csl_data.get(train_index, None)
    if new_csl_score is None:
        print(f"⚠️ Skipping {train_index}: No bbox CSL score found.")
        continue

    # === Locate Image Path (MATCHING TRAIN DATASET) ===
    file_path = next((fp for idx, fp in class_indices if idx == train_index), None)
    if file_path is None or not os.path.exists(file_path):
        print(f"⚠️ Skipping {train_index}: Image not found.")
        continue

    # === Locate XML Annotation ===
    image_id = os.path.splitext(os.path.basename(file_path))[0]  # Get image ID
    xml_file = os.path.join(ANNOTATION_FOLDER, synset_id, f"{image_id}.xml")

    if not os.path.exists(xml_file):
        print(f"⚠️ Skipping {train_index}: Bounding Box XML not found.")
        continue

    # === Extract Bounding Box ===
    bboxes = parse_xml_for_bbox(xml_file)
    if not bboxes:
        print(f"⚠️ Skipping {train_index}: No valid bounding boxes found.")
        continue

    # Use first bounding box
    xmin, ymin, xmax, ymax = bboxes[0]

    # Load & Resize Image
    image = Image.open(file_path)
    cropped_bbox = image.crop((xmin, ymin, xmax, ymax)).resize(TARGET_SIZE)  # Crop bbox & resize

    # Extract loss & confidence data
    loss_values = epochwise_data[train_index]["loss"]
    confidence_values = epochwise_data[train_index]["confidence"]
    epochs = np.arange(1, len(loss_values) + 1)

    # === Create 1×3 Subplot ===
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #fig.suptitle(f"{selected_class.capitalize()} | Train Index: {train_index} | CSL: {csl_score:.2f}", fontsize=14)
    fig.suptitle(f"{selected_class.capitalize()} | Train Index: {train_index} | CSL: {old_csl_score:.2f} → {new_csl_score:.2f}", fontsize=14)
    # 📷 **Plot 1: Bbox Crop**
    axes[0].imshow(cropped_bbox)
    axes[0].axis("off")
    axes[0].set_title("Cropped Bbox (224×224)")

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
    # ✅ Save Figure with old and new CSL
    save_path = os.path.join(class_save_dir, f"{train_index}_{old_csl_score:.2f}_{new_csl_score:.2f}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved bbox plot: {save_path}")

print("\n🎉 All bbox plots generated successfully!")