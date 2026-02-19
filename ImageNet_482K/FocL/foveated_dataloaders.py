
#!/usr/bin/env python3
"""
Contains three main FocL dataloaders
FocL Base (class BoundingBoxDataset): Simple Crop and Resize [Used in the paper for Cumulative Sample loss and Ablation]
FocL (without distortion aware - class MultiGlimpseBoundingBoxDataset) : Jittered (single or multiple crops) [This is provided here; not in the paper ]
FocL (class MultiGlimpseDistortionAwareDataset) : Full FocL algorithm [Used in paper]
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import torch
from torchvision.transforms.functional import to_tensor, crop as torch_crop, to_pil_image, resize as functional_resize
import logging
import numpy as np
from torchvision.transforms import functional as TF
import math

# Function to parse XML files for bounding boxes
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

def rescale_bbox(bbox, orig_size, target_size):
    """
    Rescale bounding box coordinates to match the resized image dimensions.

    Args:
        bbox: List or tensor of [xmin, ymin, xmax, ymax].
        orig_size: Tuple of original image dimensions (width, height).
        target_size: Tuple of target image dimensions (width, height).

    Returns:
        Rescaled bounding box [xmin, ymin, xmax, ymax].
    """
    orig_w, orig_h = orig_size
    target_w, target_h = target_size
    scale_x = target_w / orig_w
    scale_y = target_h / orig_h

    xmin, ymin, xmax, ymax = bbox
    xmin = int(xmin * scale_x)
    ymin = int(ymin * scale_y)
    xmax = int(xmax * scale_x)
    ymax = int(ymax * scale_y)

    return [xmin, ymin, xmax, ymax]

logging.basicConfig(
    filename="bounding_box_errors_aggaug.log",  # Log file name
    filemode="a",  # Append to the file
    level=logging.ERROR,  # Only log errors
    format="%(asctime)s - %(message)s"  # Log format
)

######################################################################################
# FocL Base: Single crop [No jitter; simple crop and resize]
######################################################################################

class BoundingBoxDataset(Dataset):
    def __init__(self, subset, annotation_folder, crop_transform=None, resize_size=(224, 224)):
        """
        Args:
            subset: Subset object with image indices.
            annotation_folder: Path to the folder containing XML annotations.
            crop_transform: Transform to apply to the cropped bounding box region.
            resize_size: Target size for all images (full and cropped).
        """
        self.subset = subset
        self.annotation_folder = annotation_folder
        self.crop_transform = crop_transform
        self.resize_size = resize_size

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Get the image path and label from the subset
        image_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        class_folder = os.path.basename(os.path.dirname(image_path))
        xml_file = os.path.join(self.annotation_folder, class_folder, f"{image_id}.xml")

        # Load the image
        image = Image.open(image_path).convert("RGB")
        orig_image = image.copy()  # Original image for fallback

        try:
            # Attempt to load bounding boxes
            bboxes = []
            if os.path.exists(xml_file):
                bboxes = parse_xml_for_bbox(xml_file)

            # If bounding boxes are valid, use the first bounding box
            if bboxes:
                orig_w, orig_h = image.size
                xmin, ymin, xmax, ymax = bboxes[0]

                # Ensure bbox is valid
                if xmin < 0 or ymin < 0 or xmax > orig_w or ymax > orig_h or xmin >= xmax or ymin >= ymax:
                    raise ValueError(f"Invalid bounding box in {xml_file}: {bboxes[0]}")

                # Crop the image
                cropped_image = torch_crop(to_tensor(image), top=ymin, left=xmin, height=ymax - ymin, width=xmax - xmin)
                cropped_image = to_pil_image(cropped_image)  # Convert back to PIL Image

                # Resize the cropped region
                cropped_image = functional_resize(cropped_image, self.resize_size)

                # Apply crop-specific transforms
                if self.crop_transform:
                    cropped_image = self.crop_transform(cropped_image)

                # Return resized full image, cropped image, label, and original bbox
                return to_tensor(functional_resize(orig_image, self.resize_size)), cropped_image, label, torch.tensor(bboxes[0])

        except Exception as e:
            # Handle errors: Missing XML, invalid bbox, or other issues
            #print(f"Error with image {image_path}: {e}. Using original image as fallback.")
            logging.error(f"Error with image {image_path}: {e}")

        # If error occurs or no valid bbox, fallback to original image
        fallback_image = functional_resize(orig_image, self.resize_size)
        if self.crop_transform:
            fallback_image = self.crop_transform(fallback_image)

        # Return resized full image, full image as cropped fallback, label, and placeholder bbox
        return to_tensor(functional_resize(orig_image, self.resize_size)), fallback_image, label, torch.tensor([0, 0, orig_image.size[0], orig_image.size[1]])



######################################################################################
# FocL Multi Glimpse without DA: Multi Glimpse Approach [Single Crops or Multi-Crops]
######################################################################################

class MultiGlimpseBoundingBoxDataset(Dataset):
    def __init__(
        self,
        subset,
        annotation_folder,
        crop_transform=None,
        resize_size=(224, 224),
        train_mode=True,
        offset_fraction=0.2,
        scale_jitter=0.1,
        area_threshold=0.2,
        augmentation_mode="medium",
        num_glimpses=3,
        multi_crop: bool = False,    # NEW
    ):
        self.subset            = subset
        self.annotation_folder = annotation_folder
        self.crop_transform    = crop_transform
        self.resize_size       = resize_size
        self.train_mode        = train_mode
        self.offset_fraction   = offset_fraction
        self.scale_jitter      = scale_jitter
        self.area_threshold    = area_threshold
        self.num_glimpses      = num_glimpses
        self.multi_crop        = multi_crop

        self.augmentation_mode = augmentation_mode.lower()
        if self.augmentation_mode == "conservative":
            self.offset_fraction *= 0.5
            self.scale_jitter  *= 0.5
        elif self.augmentation_mode == "aggressive":
            self.offset_fraction *= 1.5
            self.scale_jitter  *= 1.5

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # 1) load image & label
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        class_folder = os.path.basename(os.path.dirname(path))
        image_id     = os.path.splitext(os.path.basename(path))[0]
        xml_file     = os.path.join(self.annotation_folder, class_folder, f"{image_id}.xml")

        img  = Image.open(path).convert("RGB")
        orig = img.copy()

        # 2) parse bbox
        bboxes = []
        if os.path.exists(xml_file):
            bboxes = parse_xml_for_bbox(xml_file)

        # fallback if no bboxes
        if not bboxes:
            return self._fallback(orig, label)

        xmin, ymin, xmax, ymax = bboxes[0]
        W, H = orig.size
        # fallback if invalid
        if xmin<0 or ymin<0 or xmax>W or ymax>H or xmin>=xmax or ymin>=ymax:
            logging.warning(f"Invalid bbox {bboxes[0]} in {xml_file}")
            return self._fallback(orig, label)

        # 3) compute area fraction
        box_w, box_h = xmax - xmin, ymax - ymin
        area_frac    = (box_w * box_h) / (W * H)

        # 4) build N glimpses
        glimpses = []
        for _ in range(self.num_glimpses):
            if self.train_mode and area_frac < self.area_threshold:
                crop = self._offset_and_scale_crop(orig, xmin, ymin, xmax, ymax)
            else:
                crop = self._simple_bbox_crop(orig, xmin, ymin, xmax, ymax)

            if self.crop_transform:
                crop = self.crop_transform(crop)
            else:
                crop = to_tensor(crop)
            glimpses.append(crop)

        # 5) prepare outputs
        full = functional_resize(orig, self.resize_size)
        full_tensor = to_tensor(full)
        crops_tensor = torch.stack(glimpses, dim=0)  # [N, C, H, W]

        if not self.multi_crop:
            i = torch.randint(0, crops_tensor.size(0), (1,)).item()
            return full_tensor, crops_tensor[i], label, torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float)
        return full_tensor, crops_tensor, label, torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float)

    def _fallback(self, orig, label):
        """Return full-image plus either one or stacked copies per `multi_crop`."""
        full = functional_resize(orig, self.resize_size)
        T    = to_tensor(full)
        if self.multi_crop:
            crops = torch.stack([T] * self.num_glimpses)
        else:
            crops = T
        bbox = torch.tensor([0, 0, orig.width, orig.height], dtype=torch.float)
        return T, crops, label, bbox

    def _simple_bbox_crop(self, image, xmin, ymin, xmax, ymax):
        crop = torch_crop(
            to_tensor(image),
            top=ymin, left=xmin,
            height=ymax - ymin, width=xmax - xmin
        )
        crop = to_pil_image(crop)
        return functional_resize(crop, self.resize_size)

    def _offset_and_scale_crop(self, image, xmin, ymin, xmax, ymax):
        w, h         = image.size
        box_w, box_h = xmax - xmin, ymax - ymin
        sfw = 1.0 + torch.empty(1).uniform_(-self.scale_jitter, self.scale_jitter).item()
        sfh = 1.0 + torch.empty(1).uniform_(-self.scale_jitter, self.scale_jitter).item()
        new_w, new_h = box_w * sfw, box_h * sfh

        max_ox, max_oy = self.offset_fraction * box_w, self.offset_fraction * box_h
        dx = torch.empty(1).uniform_(-max_ox, max_ox).item()
        dy = torch.empty(1).uniform_(-max_oy, max_oy).item()

        cx = (xmin + xmax) / 2 + dx
        cy = (ymin + ymax) / 2 + dy

        x0 = max(0, cx - new_w / 2)
        y0 = max(0, cy - new_h / 2)
        x1 = min(w, cx + new_w / 2)
        y1 = min(h, cy + new_h / 2)
        if x1 <= x0 or y1 <= y0:
            return self._simple_bbox_crop(image, xmin, ymin, xmax, ymax)

        crop = torch_crop(
            to_tensor(image),
            top=int(y0), left=int(x0),
            height=int(y1 - y0), width=int(x1 - x0)
        )
        crop = to_pil_image(crop)
        return functional_resize(crop, self.resize_size)



######################################################################################################################
# FocL Dataloader: Multi Glimpse Distortion Aware Approach [Single Crops or Multi-Crops] [Hyper-params in Appendix]
######################################################################################################################


class MultiGlimpseDistortionAwareDataset(Dataset):
    def __init__(
        self,
        subset,
        annotation_folder,
        crop_transform=None,
        resize_size=(224, 224),
        train_mode=True,
        offset_fraction=0.2,
        scale_jitter=0.1,
        area_threshold=0.2,
        augmentation_mode="medium",
        num_glimpses=3,
        max_crop_ratio=0.2,
        multi_crop: bool = False,   # NEW
    ):
        self.subset            = subset
        self.annotation_folder = annotation_folder
        self.crop_transform    = crop_transform
        self.resize_size       = resize_size
        self.train_mode        = train_mode
        self.offset_fraction   = offset_fraction
        self.scale_jitter      = scale_jitter
        self.area_threshold    = area_threshold
        self.num_glimpses      = num_glimpses
        self.max_crop_ratio    = max_crop_ratio
        self.multi_crop        = multi_crop

        self.augmentation_mode = augmentation_mode.lower()
        if self.augmentation_mode == "conservative":
            self.offset_fraction *= 0.5
            self.scale_jitter  *= 0.5
        elif self.augmentation_mode == "aggressive":
            self.offset_fraction *= 1.5
            self.scale_jitter  *= 1.5

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # 1) load image & label
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img  = Image.open(path).convert("RGB")
        orig = img.copy()

        class_folder = os.path.basename(os.path.dirname(path))
        image_id     = os.path.splitext(os.path.basename(path))[0]
        xml_file     = os.path.join(self.annotation_folder, class_folder, f"{image_id}.xml")
        bboxes       = parse_xml_for_bbox(xml_file) if os.path.exists(xml_file) else []

        # fallback if needed
        if not bboxes:
            return self._fallback(orig, label)

        xmin, ymin, xmax, ymax = bboxes[0]
        W, H = orig.size
        if xmin<0 or ymin<0 or xmax>W or ymax>H or xmin>=xmax or ymin>=ymax:
            return self._fallback(orig, label)

        # 2) area fraction
        box_w, box_h = xmax - xmin, ymax - ymin
        area_frac     = (box_w * box_h) / (W * H)

        # 3) build N crops
        crops = []
        for _ in range(self.num_glimpses):
            if self.train_mode and area_frac < self.area_threshold:
                crop = self._distortion_aware_expand(orig, xmin, ymin, xmax, ymax)
            else:
                crop = self._offset_and_scale_crop(orig, xmin, ymin, xmax, ymax)

            if self.crop_transform:
                crop = self.crop_transform(crop)
            else:
                crop = to_tensor(crop)
            crops.append(crop)

        full = functional_resize(orig, self.resize_size)
        full_tensor = to_tensor(full)
        crops_tensor = torch.stack(crops, dim=0)

        if not self.multi_crop:
            i = torch.randint(0, crops_tensor.size(0), (1,)).item()
            return full_tensor, crops_tensor[i], label, torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float)
        return full_tensor, crops_tensor, label, torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float)

    def _fallback(self, orig, label):
        """Return fallback full-image ± stacked copies per `multi_crop`."""
        full = functional_resize(orig, self.resize_size)
        T    = to_tensor(full)
        if self.multi_crop:
            crops = torch.stack([T] * self.num_glimpses)
        else:
            crops = T
        bbox = torch.tensor([0, 0, orig.width, orig.height], dtype=torch.float)
        return T, crops, label, bbox

    def _distortion_aware_expand(self, image, xmin, ymin, xmax, ymax):
        w, h   = image.size
        box_w  = xmax - xmin
        box_h  = ymax - ymin
        tx, ty = self.resize_size
        sf_x   = tx / box_w
        sf_y   = ty / box_h
        up     = max(sf_x, sf_y)
        thresh = 1.0 / (1.0 - self.max_crop_ratio)
        if up > thresh:
            needed = up / thresh
            factor = math.sqrt(needed)
            ew = box_w * (factor - 1.0) / 2.0
            eh = box_h * (factor - 1.0) / 2.0
            xmin = max(0, xmin - ew)
            ymin = max(0, ymin - eh)
            xmax = min(w, xmax + ew)
            ymax = min(h, ymax + eh)
        crop = image.crop((xmin, ymin, xmax, ymax))
        return functional_resize(crop, self.resize_size)

    def _offset_and_scale_crop(self, image, xmin, ymin, xmax, ymax):
        w, h       = image.size
        box_w      = xmax - xmin
        box_h      = ymax - ymin
        sfw = 1.0 + torch.empty(1).uniform_(-self.scale_jitter, self.scale_jitter).item()
        sfh = 1.0 + torch.empty(1).uniform_(-self.scale_jitter, self.scale_jitter).item()
        new_w, new_h = box_w * sfw, box_h * sfh

        max_ox, max_oy = self.offset_fraction * box_w, self.offset_fraction * box_h
        dx = torch.empty(1).uniform_(-max_ox, max_ox).item()
        dy = torch.empty(1).uniform_(-max_oy, max_oy).item()

        cx = (xmin + xmax) / 2 + dx
        cy = (ymin + ymax) / 2 + dy

        x0, y0 = max(0, cx - new_w / 2), max(0, cy - new_h / 2)
        x1, y1 = min(w, cx + new_w / 2), min(h, cy + new_h / 2)
        if x1 <= x0 or y1 <= y0:
            return self._simple_bbox_crop(image, xmin, ymin, xmax, ymax)

        crop = torch_crop(
            to_tensor(image),
            top=int(y0), left=int(x0),
            height=int(y1 - y0), width=int(x1 - x0)
        )
        crop = to_pil_image(crop)
        return functional_resize(crop, self.resize_size)

    def _simple_bbox_crop(self, image, xmin, ymin, xmax, ymax):
        crop = torch_crop(
            to_tensor(image),
            top=ymin, left=xmin,
            height=ymax - ymin, width=xmax - xmin
        )
        crop = to_pil_image(crop)
        return functional_resize(crop, self.resize_size)
    




    def print_debug_stats(self):
        """Print debugging statistics for the dataset."""
        print(f"Total samples processed: {self.total_samples_processed}")
        print(f"Missing bounding boxes: {self.missing_bbox_count}")
        print(f"Invalid bounding boxes: {self.invalid_bbox_count}")