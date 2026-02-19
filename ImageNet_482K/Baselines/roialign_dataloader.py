# =============================================================================
# roialign_dataloader.py — RoIAlign-style BBox Dataset (BBox kept in sync w/ aug)
# =============================================================================
#
# What this file does
# -------------------
# Defines a Dataset wrapper that returns:
#   1) an ImageNet image (after standard transforms, typically 224×224 tensor),
#   2) the corresponding bounding box (scaled to the transformed 224×224 space),
#   3) the class label.
#
# The key purpose is to provide *both* the image tensor and a bbox tensor that
# stays consistent with training-time augmentation (specifically horizontal flip),
# enabling RoIAlign / RoI pooling in the model forward pass.
#
#
# Expected annotation format
# --------------------------
# - Uses ImageNet XML annotations located at:
#     annotation_folder/<class_folder>/<image_id>.xml
# - parse_xml_for_bbox(xml_file) extracts the FIRST valid bbox:
#     [xmin, ymin, xmax, ymax] in the ORIGINAL image pixel coordinates.
# - If XML is missing or malformed, it falls back to the full-image bbox:
#     [0, 0, W, H]
#
#
# Returned sample
# ---------------
#   image, scaled_bbox, label
#
# - image: transformed image tensor (e.g., 3×224×224) if transform is provided.
# - scaled_bbox: torch.float32 tensor [x1, y1, x2, y2] in *224×224 coordinate space*
#                (clamped to [0, 223.99]).
# - label: ImageNet class index from ImageFolder.
#
#
# Transform handling (IMPORTANT — why the flip is manual)
# ------------------------------------------------------
# This dataset manually applies horizontal flip BEFORE calling `self.transform`,
# because the bbox must be flipped in exactly the same way as the image.
#
# 1) Manual horizontal flip (train-only)
#    If `is_train=True`, with p=0.5 it applies:
#       image = F.hflip(image)
#    and sets `is_flipped=True`.
#
#    ⚠️ Therefore, your `transform` MUST NOT include RandomHorizontalFlip.
#    Otherwise you would flip twice and/or desynchronize the bbox.
#
# 2) Apply `self.transform` to the (possibly flipped) image
#    Typical transforms here should be *geometry-preserving w.r.t bbox scaling*:
#      - Resize(256) + CenterCrop(224)  (safe)
#      - Resize((224,224))             (safe)
#      - ToTensor + Normalize          (safe)
#    Avoid transforms that change geometry in a way the bbox is not updated for:
#      - RandomResizedCrop / RandomCrop / CenterCrop with unknown size
#      - Any padding/cropping/rotation
#
# 3) BBox scaling + optional reflection
#    The script scales bbox coordinates from ORIGINAL (W,H) to 224×224 by:
#       x' = (x / W) * 224
#       y' = (y / H) * 224
#
#    If the image was flipped, it mirrors x-coordinates in 224 space:
#       x1_new = 224 - x2
#       x2_new = 224 - x1
#
# 4) Clamp
#    Clamps bbox to [0, 223.99] to avoid out-of-range coordinates.
#
#
# Practical implications / gotchas
# --------------------------------
# - This implementation assumes the *final network input is 224×224*.
# - If your transform changes aspect ratio or does non-uniform resizing, the
#   bbox scaling here will be wrong unless you update the math accordingly.
# - If you want RandomResizedCrop during training, you must:
#     (a) compute bbox transform through the crop parameters, or
#     (b) move cropping logic into the dataset so bbox is updated consistently.
# - Multiple bboxes in XML are ignored (only the first valid bbox is used).
#
# Typical usage
# -------------
#   base = torchvision.datasets.ImageFolder(DATASET_PATH)  # no transform or only safe ones
#   subset = torch.utils.data.Subset(base, train_indices)
#   ds = RoIAlignDataset(
#           subset=subset,
#           annotation_folder=ANNOTATION_PATH,
#           transform=transforms.Compose([
#               transforms.Resize(256),
#               transforms.CenterCrop(224),
#               transforms.ToTensor(),
#               transforms.Normalize(mean, std),
#           ]),
#           is_train=True
#       )
#   loader = DataLoader(ds, batch_size=..., shuffle=..., num_workers=...)
# =============================================================================




import os
import random
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

def parse_xml_for_bbox(xml_file):
    """Parse an XML file to extract the first valid bounding box."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            if xmin < xmax and ymin < ymax:
                return [xmin, ymin, xmax, ymax]
        return None
    except Exception:
        return None

class RoIAlignDataset(Dataset):
    def __init__(self, subset, annotation_folder, transform=None, is_train=False):
        """
        Args:
            subset: The ImageNet subset partition.
            annotation_folder: Path to XML annotations.
            transform: Standard transforms (MUST NOT contain RandomHorizontalFlip).
            is_train: Flag to enable manual flipping and jitter.
        """
        self.subset = subset
        self.annotation_folder = annotation_folder
        self.transform = transform
        self.is_train = is_train

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Retrieve path and label
        image_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        class_folder = os.path.basename(os.path.dirname(image_path))
        xml_file = os.path.join(self.annotation_folder, class_folder, f"{image_id}.xml")
        
        # Load full image
        image = Image.open(image_path).convert("RGB")
        w_orig, h_orig = image.size

        # Parse BBox (fallback to full image if XML is missing)
        bbox = parse_xml_for_bbox(xml_file)
        if bbox is None:
            bbox = [0, 0, w_orig, h_orig]

        # ─── 1. Manual Horizontal Flip ──────────────────────────────────────
        # This keeps the image and bbox synchronized.
        is_flipped = False
        if self.is_train and random.random() > 0.5:
            image = F.hflip(image)
            is_flipped = True

        # ─── 2. Apply Transforms ─────────────────────────────────────────────
        # Note: Your training script's transform list must not have RandomHorizontalFlip
        if self.transform:
            image = self.transform(image)
        
        # ─── 3. Scale and Reflect BBox ───────────────────────────────────────
        x1, y1, x2, y2 = bbox
        # Scale to the 224x224 input resolution
        x1, x2 = (x1 / w_orig) * 224.0, (x2 / w_orig) * 224.0
        y1, y2 = (y1 / h_orig) * 224.0, (y2 / h_orig) * 224.0
        
        if is_flipped:
            # Mirror the x-coordinates: x_new = Width - x_old
            x1_new = 224.0 - x2
            x2_new = 224.0 - x1
            x1, x2 = x1_new, x2_new

        # ─── 4. Clamping ─────────────────────────────────────────────────────
        # Ensures coordinates stay within valid feature map boundaries
        scaled_bbox = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
        scaled_bbox = torch.clamp(scaled_bbox, 0.0, 223.99)


        return image, scaled_bbox, label