#!/usr/bin/env python3
"""
Purpose
-------
Generate SAM2-based *bounding-box proposals* for a fixed 2K subset of ImageNet-1k
(validation split). This script is **proposal generation only** — it does not run
any classifier inference. The output is a .pth file containing per-image proposal
lists (bboxes + SAM scores), which are later consumed by the SAM inference script.

Key Design Choice: "RELAXED" proposal settings
----------------------------------------------
The SAM2 automatic mask generator is configured with relaxed thresholds to produce
more diverse proposals (more permissive mask acceptance and weaker NMS). This is
intended to better match the foveated / crop-based distributions used in FocL-style
evaluation (i.e., do not overly prune proposals).

Dataset / Subset Protocol
-------------------------
- Dataset: ImageNet-1k val set at:
    /local/a/imagenet/imagenet2012/val/
- Subset: exactly 2000 images sampled WITHOUT replacement using NumPy with SEED=42:
    all_indices = np.random.choice(len(dataset), size=2000, replace=False)
- Optional slicing:
    --start / --end allow processing a contiguous range within these 2K indices.
- Beta mode:
    --beta_test processes only the first 50 samples from the 2K list.

Proposal Generation Details
---------------------------
For each selected image:
1) Run SAM2AutomaticMaskGenerator on the raw RGB image (no torchvision transforms).
2) Convert each predicted mask to a tight bounding box via torchvision.masks_to_boxes.
3) Apply relaxed filtering:
   - very small mask area threshold (relative to image area)
   - small minimum bbox side length
4) Sort proposals by SAM predicted IoU score (descending).
5) Apply RELAXED NMS (IoU threshold ~0.9) to drop near-duplicate boxes while keeping
   diversity.
6) Keep up to --max_proposals proposals per image (default 50).

Fallback Behavior
-----------------
If SAM fails or returns no valid masks, the script falls back to a single proposal:
the full-image bounding box [0,0,W,H] (both xywh and xyxy formats).

Output Format (.pth)
--------------------
Periodic checkpoints:
- Saves a dict: {i -> sample_data}, where i is the local counter within the processed
  slice (not the global dataset index).

Final save (recommended to use):
- Saves a dict with keys:
    {
      "proposals": {i -> sample_data, ...},
      "metadata":  {...sam_config..., seed, thresholds, max_proposals...}
    }

Each sample_data includes:
- image_path, image_name
- gt_label (ImageFolder integer class index)
- image_size [W, H]
- dataset_index (global index into ImageFolder.samples)
- proposals: list of dicts, each with:
    bbox_xywh, bbox_xyxy, sam_score, mask_area, stability_score, original_sam_bbox
- num_proposals

Important Notes / Pitfalls
--------------------------
- This is the ImageNet-V1 (ImageNet-1k val) version. For ImageNet-V2, folder naming
  can cause label-index mismatches; that is handled in the separate V2 script.
- The 2K subset is determined by RNG + dataset ordering. For exact reproducibility
  across machines, ensure the same ImageFolder ordering and seed.
- Proposal keys are stored under a local counter i; downstream code should use the
  stored "dataset_index" if global indexing is needed.

CLI
---
Example:
  python SAMproposal_ImageNet_V1.py --start 0 --end 2000 --output_dir ./sam_props/ --max_proposals 50

Outputs:
  ./sam_props/SAM_proposals_relaxed_from{start}to{end}_final.pth
"""


import os
import copy
import random
import numpy as np
import torch
import torchvision.ops as ops
from torchvision import datasets
from torchvision.ops import masks_to_boxes
from PIL import Image
import argparse

from sam2.build_sam import build_sam2_robust
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ─── REPRODUCIBILITY ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def generate_sam_proposals(image_np, sam_generator, max_proposals=50):
    """
    Generate bounding box proposals from SAM masks - RELAXED VERSION
    Returns list of proposals with SAM scores and bbox coordinates
    """
    H, W = image_np.shape[:2]
    
    try:
        masks = sam_generator.generate(image_np)
        print(f"    SAM generated {len(masks)} initial masks")
    except Exception as e:
        print(f"    SAM failed: {e}, using fallback")
        # Fallback: full image proposal
        return [{
            "bbox_xywh": [0, 0, W, H],
            "bbox_xyxy": [0, 0, W, H], 
            "sam_score": 1.0,
            "mask_area": W * H,
            "stability_score": 1.0
        }]
    
    if not masks:
        return [{
            "bbox_xywh": [0, 0, W, H],
            "bbox_xyxy": [0, 0, W, H],
            "sam_score": 1.0,
            "mask_area": W * H,
            "stability_score": 1.0
        }]
    
    # Convert masks to proposals with RELAXED filtering
    proposals = []
    for mask in masks:
        try:
            # Get mask and validate
            mask_arr = mask.get('segmentation', mask.get('mask'))
            if mask_arr is None:
                continue
                
            # RELAXED area filter - allow smaller objects
            mask_area = np.sum(mask_arr)
            if mask_area < 0.0001 * H * W:  # RELAXED: was 0.0005, now 0.0001
                continue
            
            # Convert to tight bounding box using masks_to_boxes
            # Handle different mask formats (boolean, uint8 0/1, uint8 0/255)
            if mask_arr.dtype == bool:
                mask_for_boxes = mask_arr.astype(np.uint8)
            elif mask_arr.dtype == np.uint8:
                if mask_arr.max() > 1:  # 0/255 format
                    mask_for_boxes = (mask_arr > 0).astype(np.uint8)
                else:  # 0/1 format
                    mask_for_boxes = mask_arr
            else:
                mask_for_boxes = (mask_arr > 0).astype(np.uint8)
            
            mask_tensor = torch.from_numpy(mask_for_boxes).unsqueeze(0)
            tight_bbox_xyxy = masks_to_boxes(mask_tensor)[0].int().tolist()
            
            # RELAXED bbox validation - allow smaller boxes
            x1, y1, x2, y2 = tight_bbox_xyxy
            if x2 <= x1 or y2 <= y1 or (x2-x1) < 5 or (y2-y1) < 5:  # RELAXED: was 10, now 5
                continue
            
            # Convert to xywh format
            bbox_xywh = [x1, y1, x2-x1, y2-y1]
            
            # Store proposal
            proposal = {
                "bbox_xywh": bbox_xywh,      # [x, y, width, height]
                "bbox_xyxy": tight_bbox_xyxy, # [x1, y1, x2, y2]
                "sam_score": mask.get('predicted_iou', 0.5),
                "mask_area": int(mask_area),
                "stability_score": mask.get('stability_score', 0.5),
                "original_sam_bbox": mask.get('bbox', [0, 0, 0, 0])  # SAM's original bbox
            }
            proposals.append(proposal)
            
        except Exception as e:
            print(f"    Error processing mask: {e}")
            continue
    
    if not proposals:
        return [{
            "bbox_xywh": [0, 0, W, H],
            "bbox_xyxy": [0, 0, W, H],
            "sam_score": 1.0,
            "mask_area": W * H,
            "stability_score": 1.0
        }]
    
    # Sort by SAM score (descending)
    proposals.sort(key=lambda x: x["sam_score"], reverse=True)
    
    # Apply RELAXED NMS to reduce overlaps while keeping more diversity
    if len(proposals) > 1:
        # Extract boxes and scores for NMS
        boxes = torch.tensor([p["bbox_xyxy"] for p in proposals], dtype=torch.float32)
        scores = torch.tensor([p["sam_score"] for p in proposals], dtype=torch.float32)
        
        # RELAXED NMS - only merge very similar boxes
        keep_indices = ops.nms(boxes, scores, iou_threshold=0.9)  # RELAXED: was 0.7, now 0.9
        proposals = [proposals[i] for i in keep_indices.tolist()]
        print(f"    After RELAXED NMS: {len(proposals)} proposals")
    
    # Return more proposals
    return proposals[:max_proposals]

def main():
    parser = argparse.ArgumentParser(description="SAM Proposal Generator")
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--end', type=int, default=50, help='End index (exclusive)')
    parser.add_argument('--beta_test', action='store_true', help='Beta test mode (first 50 samples)')
    parser.add_argument('--output_dir', type=str, default='./sam_proposals_relaxed/', help='Output directory')
    parser.add_argument('--max_proposals', type=int, default=50, help='Max proposals per image')  # Default to 50
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize SAM with RELAXED settings for maximum diversity
    CONFIG_FILE = '/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml'
    CHECKPOINT = '/sam2/checkpoints/sam2.1_hiera_large.pt'
    
    sam_generator = SAM2AutomaticMaskGenerator(
        model=build_sam2_robust(CONFIG_FILE, CHECKPOINT),
        points_per_side=32,              # RELAXED: fewer points = larger, less precise masks
        pred_iou_thresh=0.5,             # RELAXED: from 0.75 -> 0.5 (much more permissive)
        stability_score_thresh=0.5,      # RELAXED: from 0.85 -> 0.5 (much more permissive)
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=25,         # RELAXED: allow smaller objects
    )
    print('Initialized SAM with RELAXED settings for maximum diversity')
    print('  pred_iou_thresh: 0.5 (was 0.75)')
    print('  stability_score_thresh: 0.5 (was 0.85)')
    print('  NMS IoU threshold: 0.9 (was 0.7)')
    print('  max_proposals: 50 (was 15)')
    
    # Load validation dataset (no transforms, raw images)
    val_path = "/local/a/imagenet/imagenet2012/val/"
    raw_dataset = datasets.ImageFolder(root=val_path, transform=None)
    
    # Generate same 2K indices as FALcon (using SEED=42)
    all_indices = np.random.choice(len(raw_dataset), size=2000, replace=False)
    print(f"Generated 2000 random indices (seed={SEED})")
    
    # Determine processing range
    if args.beta_test:
        process_indices = all_indices[:50]
        print(f"Beta test mode: Processing first 50 samples")
    else:
        start_idx = max(0, args.start)
        end_idx = min(len(all_indices), args.end)
        process_indices = all_indices[start_idx:end_idx]
        print(f"Processing samples {start_idx} to {end_idx-1} ({len(process_indices)} samples)")
    
    # Results storage
    sam_proposals = {}
    
    # Process each image
    for i, dataset_idx in enumerate(process_indices):
        print(f"Processing sample {i+1}/{len(process_indices)} (dataset idx {dataset_idx})")
        
        try:
            # Load raw image (no transforms)
            img_path, gt_label = raw_dataset.samples[dataset_idx]
            image = Image.open(img_path).convert("RGB")
            image_np = np.array(image)
            H, W = image_np.shape[:2]
            
            print(f"  Image: {os.path.basename(img_path)} ({W}x{H}), GT label: {gt_label}")
            
            # Generate SAM proposals
            proposals = generate_sam_proposals(image_np, sam_generator, max_proposals=args.max_proposals)
            print(f"  Final proposals: {len(proposals)}")
            
            # Store results
            sample_data = {
                "image_path": img_path,
                "image_name": os.path.basename(img_path),
                "gt_label": gt_label,
                "image_size": [W, H],  # [width, height]
                "dataset_index": int(dataset_idx),
                "proposals": proposals,
                "num_proposals": len(proposals)
            }
            
            sam_proposals[i] = sample_data
            
            # Periodic saving
            if (i + 1) % 100 == 0 or i == len(process_indices) - 1:
                if args.beta_test:
                    filename = f"SAM_proposals_relaxed_beta_test.pth"
                else:
                    filename = f"SAM_proposals_relaxed_from{args.start}to{args.end}.pth"
                
                filepath = os.path.join(args.output_dir, filename)
                torch.save(sam_proposals, filepath)
                print(f"  Saved {len(sam_proposals)} samples to {filepath}")
        
        except Exception as e:
            print(f"  Error processing sample {i}: {e}")
            continue
    
    print(f"\nCompleted RELAXED SAM proposal generation for {len(sam_proposals)} samples")
    
    # Final save with metadata
    metadata = {
        "total_samples": len(sam_proposals),
        "seed": SEED,
        "sam_config": {
            "points_per_side": 32,
            "pred_iou_thresh": 0.5,      # RELAXED
            "stability_score_thresh": 0.5, # RELAXED
            "min_mask_region_area": 25
        },
        "nms_threshold": 0.9,            # RELAXED
        "max_proposals_per_image": args.max_proposals,
        "version": "relaxed_for_diversity"
    }
    
    final_data = {
        "proposals": sam_proposals,
        "metadata": metadata
    }
    
    if args.beta_test:
        filename = f"SAM_proposals_relaxed_beta_test_final.pth"
    else:
        filename = f"SAM_proposals_relaxed_from{args.start}to{args.end}_final.pth"
    
    filepath = os.path.join(args.output_dir, filename)
    torch.save(final_data, filepath)
    print(f"Final save with metadata: {filepath}")

if __name__ == "__main__":
    main()