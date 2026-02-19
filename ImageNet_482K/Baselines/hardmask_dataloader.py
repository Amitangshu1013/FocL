# =============================================================================
# hardmask_dataloader.py — ImageNet “HardMask” Dataset (BBox Background Removal)
# =============================================================================
#
# What this file does
# -------------------
# Defines a PyTorch Dataset wrapper that converts an ImageNet image into a
# “HardMask” version: pixels *outside* the ground-truth bounding box are replaced
# with a constant mean-color background, while pixels *inside* the box are kept.
#
# Intended use:
#   - Train / evaluate a standard classifier on “cleaned” images where clutter
#     is removed using GT bounding boxes (but the model still sees a full-frame
#     224×224 input after standard augmentations).
#
#
# Key components
# --------------
# 1) parse_xml_for_bbox(xml_file)
#    - Reads an ImageNet-style XML annotation file and extracts one or more
#      bounding boxes as [xmin, ymin, xmax, ymax].
#    - Returns [] if the XML is missing, malformed, or yields invalid coords.
#
# 2) HardMaskDataset(Dataset)
#    Wraps a *subset* (e.g., Partition A/B subset) and for each sample:
#      (a) Finds the corresponding XML file using:
#          annotation_folder / <class_folder> / <image_id>.xml
#      (b) Loads the image in PIL RGB at original resolution (W, H).
#      (c) Builds a binary mask (white inside bbox, black outside).
#      (d) Composites:
#            masked_image = composite(original_image, mean_color_bg, mask)
#          so background becomes ImageNet mean RGB color (approx 124,116,104).
#      (e) Applies the standard augmentation pipeline (crop_transform), e.g.,
#          RandomResizedCrop/Flip/Jitter/Normalize, on the masked image.
#      (f) Returns: (final_input_tensor, label)
#
#
# Expected inputs / conventions
# -----------------------------
# - subset:
#     A torch.utils.data.Subset-like object whose underlying dataset has
#     .samples and where subset.indices maps into that dataset. The code expects:
#       image_path, label = subset.dataset.samples[ subset.indices[idx] ]
#
# - annotation_folder:
#     Root directory of ImageNet XML annotations organized by class folder:
#       annotation_folder/<class_folder>/<image_id>.xml
#
# - crop_transform (optional):
#     The standard torchvision transform pipeline that outputs a normalized
#     tensor suitable for ResNet training (typically 224×224).



import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

# --- Helper Function ---
def parse_xml_for_bbox(xml_file):
    """Parse an XML file to extract bounding box coordinates."""
    try:
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
    except Exception as e:
        # Return empty list on parse error so dataset doesn't crash
        return []

# --- Dataset Class ---
class HardMaskDataset(Dataset):
    def __init__(self, subset, annotation_folder, crop_transform=None, resize_size=(256, 256)):
        """
        Args:
            subset: The Partition A/B subset object containing image paths and indices.
            annotation_folder: Path to the folder containing ImageNet XML annotations.
            crop_transform: The STANDARD training transforms (RandomResizedCrop, Flip, etc.).
            resize_size: Only used for reference/logging if needed, not for the input generation.
        """
        self.subset = subset
        self.annotation_folder = annotation_folder
        self.crop_transform = crop_transform
        self.resize_size = resize_size
        
        # ImageNet Mean Color in RGB (0-255 scale)
        # R: 0.485*255=123.6, G: 0.456*255=116.2, B: 0.406*255=103.5
        self.mean_color = (124, 116, 104)

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # 1. Retrieve Basic Info
        image_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        class_folder = os.path.basename(os.path.dirname(image_path))
        xml_file = os.path.join(self.annotation_folder, class_folder, f"{image_id}.xml")
        
        # 2. Load Image (Keep in PIL RGB)
        # We work on the original image dimensions (W, H)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size

        # 3. Create Hard Mask (Mask out background on ORIGINAL resolution)
        masked_image = image.copy() # Default fallback
        
        try:
            bboxes = []
            if os.path.exists(xml_file):
                bboxes = parse_xml_for_bbox(xml_file)
            
            if bboxes:
                # Use the first bbox found
                xmin, ymin, xmax, ymax = bboxes[0]
                
                # Clamp coordinates to image boundaries just in case
                xmin = max(0, xmin); ymin = max(0, ymin)
                xmax = min(w, xmax); ymax = min(h, ymax)
                
                # Proceed only if area is valid
                if xmax > xmin and ymax > ymin:
                    # Create Mask: 255 (White) inside bbox, 0 (Black) outside
                    mask = Image.new('L', (w, h), 0)
                    draw = ImageDraw.Draw(mask)
                    draw.rectangle((xmin, ymin, xmax, ymax), fill=255)
                    
                    # Create Background: Solid Mean Color
                    bg = Image.new('RGB', (w, h), self.mean_color)
                    
                    # Composite: Keep object pixels where mask is 255, use Mean Color elsewhere
                    masked_image = Image.composite(image, bg, mask)
        
        except Exception:
            # On any error (XML missing, corrupt, etc), strictly fallback to full image
            # to match standard pipeline behavior rather than crashing.
            masked_image = image.copy()

        # 4. Apply Standard Augmentation Pipeline
        #    RandomResizedCrop will now crop from this 'clean' image.
        if self.crop_transform:
            final_input = self.crop_transform(masked_image)
        else:
            # Fallback transform if none provided
            final_input = F.to_tensor(masked_image.resize((224, 224)))

        # 5. Return (Input, Label)
        #    Matches the standard ImageNet dataset signature
        return final_input, label