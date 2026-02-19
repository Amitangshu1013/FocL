#!/usr/bin/env python3
"""
Purpose
-------
Generate SAM-based bounding-box proposals for COCO images referenced by a RefCOCO(+)-style
visual grounding JSON. This script is intended for "standard" (single-scale) proposal generation
and saves proposals per image for downstream evaluation / visualization.

High-level behavior
-------------------
1) Load a JSON list of grounding samples (expects each sample to include `image_id`).
2) Collect unique COCO image IDs and load images from a COCO directory (default naming:
   COCO_train2014_{image_id:012d}.jpg).
3) Run SAM-2.1 automatic mask generation to obtain candidate masks.
4) Convert each mask into a tight bounding box (xyxy, then xywh), filter degenerate boxes,
   sort by SAM score, apply NMS, and keep top-K proposals.
5) Save all proposals + metadata to a single .pth file via torch.save().

Output (saved .pth structure)
-----------------------------
results["sam_proposals"][image_id] = {
    "image_path": str,
    "image_id": int,
    "image_size": [W, H],
    "proposals": [
        {
          "bbox_xywh": [x, y, w, h],
          "bbox_xyxy": [x1, y1, x2, y2],
          "sam_score": float,          # predicted_iou from SAM (fallback: 0.5)
          "stability_score": float,    # stability_score from SAM (fallback: 0.5)
          "mask_area": int,            # number of foreground pixels in mask
          "proposal_id": int
        },
        ...
    ],
    "num_proposals": int
}

Robustness / fallback behavior
------------------------------
- If SAM fails OR returns zero masks OR all masks are filtered out, the script falls back to a
  single "full-image" proposal: bbox=[0,0,W,H] with score=1.0. This ensures the pipeline
  always produces at least one proposal per image.

Important implementation notes (current script behavior)
--------------------------------------------------------
- Input images are processed in pixel space (RGB uint8 -> numpy array), and SAM runs on that.
- The script applies an *additional* minimum-area filter:
      min_area_threshold = 0.0005 * H * W
  (This is separate from SAM's own min_mask_region_area setting.)
- Proposal-level NMS is currently applied with:
      iou_threshold = 0.7
- The SAM generator is initialized with:
      pred_iou_thresh=0.70
      stability_score_thresh=0.80
      min_mask_region_area=100
      box_nms_thresh=0.65
  (These are the proposal-generator thresholds used in this file.)

Key arguments
-------------
--annotations_file : Path to visual grounding JSON (list of dicts with 'image_id')
--images_dir       : Directory containing COCO images (train2014 naming assumed)
--output_dir       : Where to write the saved .pth
--max_images       : Number of unique images to process (0 or large for full run; default is for testing)
--max_proposals    : Number of proposals kept per image (top-K after sort+NMS)
--sam_config       : SAM-2.1 YAML config path
--sam_checkpoint   : SAM-2.1 checkpoint path

Example usage
-------------
# Quick smoke test on first 20 unique images:
python coco_sam_proposal_generator_standard.py \
  --annotations_file /path/to/visual_grounding_2k_samples.json \
  --images_dir /path/to/COCO/train2014 \
  --output_dir ./sam_proposals_refcoco \
  --max_images 20 \
  --max_proposals 20 \
  --sam_config /path/to/sam2.1_hiera_l.yaml \
  --sam_checkpoint /path/to/sam2.1_hiera_large.pt

# Full run (process all unique images in the JSON):
python coco_sam_proposal_generator_standard.py \
  --annotations_file /path/to/visual_grounding_2k_samples.json \
  --images_dir /path/to/COCO/train2014 \
  --output_dir ./sam_proposals_refcoco \
  --max_images 0 \
  --max_proposals 20 \
  --sam_config /path/to/sam2.1_hiera_l.yaml \
  --sam_checkpoint /path/to/sam2.1_hiera_large.pt
"""

import os
import json
import numpy as np
import torch
import torchvision.ops as ops
from torchvision.ops import masks_to_boxes
from PIL import Image
import argparse
from pathlib import Path
from collections import defaultdict
import traceback

from sam2.build_sam import build_sam2_robust
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)



def generate_sam_proposals(image_np, sam_generator, max_proposals=20):
    """
    Generate high-quality bounding box proposals from SAM masks
    Standard settings for clean proposal generation
    """
    H, W = image_np.shape[:2]
    
    try:
        masks = sam_generator.generate(image_np)
        print(f"    SAM generated {len(masks)} initial masks")
    except Exception as e:
        print(f"    SAM failed: {e}")
        print(f"    Using fallback full-image proposal")
        return [{
            "bbox_xywh": [0, 0, W, H],
            "bbox_xyxy": [0, 0, W, H], 
            "sam_score": 1.0,
            "mask_area": W * H,
            "stability_score": 1.0,
            "proposal_id": 0
        }]
    
    if not masks:
        print(f"    No masks generated, using fallback")
        return [{
            "bbox_xywh": [0, 0, W, H],
            "bbox_xyxy": [0, 0, W, H],
            "sam_score": 1.0,
            "mask_area": W * H,
            "stability_score": 1.0,
            "proposal_id": 0
        }]
    
    proposals = []
    for mask_idx, mask in enumerate(masks):
        try:
            # Get mask array
            mask_arr = mask.get('segmentation', mask.get('mask'))
            if mask_arr is None:
                continue
                
            # Filter by area - remove very small objects
            mask_area = np.sum(mask_arr)
            min_area_threshold = 0.0005 * H * W  # Standard threshold
            if mask_area < min_area_threshold:
                continue
            
            # Convert mask to tight bounding box
            if mask_arr.dtype == bool:
                mask_for_boxes = mask_arr.astype(np.uint8)
            elif mask_arr.dtype == np.uint8:
                if mask_arr.max() > 1:
                    mask_for_boxes = (mask_arr > 0).astype(np.uint8)
                else:
                    mask_for_boxes = mask_arr
            else:
                mask_for_boxes = (mask_arr > 0).astype(np.uint8)
            
            # Get tight bounding box
            mask_tensor = torch.from_numpy(mask_for_boxes).unsqueeze(0)
            tight_bbox_xyxy = masks_to_boxes(mask_tensor)[0].int().tolist()
            
            # Validate bbox
            x1, y1, x2, y2 = tight_bbox_xyxy
            if x2 <= x1 or y2 <= y1 or (x2-x1) < 10 or (y2-y1) < 10:
                continue
            
            # Ensure bbox is within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Convert to xywh format
            bbox_xywh = [x1, y1, x2-x1, y2-y1]
            
            # Store proposal with metadata
            proposal = {
                "bbox_xywh": bbox_xywh,
                "bbox_xyxy": [x1, y1, x2, y2],
                "sam_score": mask.get('predicted_iou', 0.5),
                "mask_area": int(mask_area),
                "stability_score": mask.get('stability_score', 0.5),
                "proposal_id": mask_idx
            }
            proposals.append(proposal)
            
        except Exception as e:
            print(f"    Error processing mask {mask_idx}: {e}")
            continue
    
    if not proposals:
        print(f"    No valid proposals, using fallback")
        return [{
            "bbox_xywh": [0, 0, W, H],
            "bbox_xyxy": [0, 0, W, H],
            "sam_score": 1.0,
            "mask_area": W * H,
            "stability_score": 1.0,
            "proposal_id": 0
        }]
    
    # Sort by SAM score (descending)
    proposals.sort(key=lambda x: x["sam_score"], reverse=True)
    
    # Apply NMS to reduce overlapping proposals
    if len(proposals) > 1:
        boxes = torch.tensor([p["bbox_xyxy"] for p in proposals], dtype=torch.float32)
        scores = torch.tensor([p["sam_score"] for p in proposals], dtype=torch.float32)
        
        # Standard NMS threshold
        keep_indices = ops.nms(boxes, scores, iou_threshold=0.7)
        proposals = [proposals[i] for i in keep_indices.tolist()]
        print(f"    After NMS: {len(proposals)} proposals")
    
    return proposals[:max_proposals]

def load_visual_grounding_data(annotations_file):
    """
    Load visual grounding annotations
    """
    with open(annotations_file, 'r') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(samples)} visual grounding samples")
    return samples

def get_coco_image_path(image_id, images_dir):
    """
    Get path to COCO image given image_id
    """
    filename = f"COCO_train2014_{image_id:012d}.jpg"
    return Path(images_dir) / filename

def main():
    parser = argparse.ArgumentParser(description="SAM Proposal Generator for RefCOCO+ Visual Grounding")
    parser.add_argument('--annotations_file', required=True, 
                       help='Path to visual_grounding_2k_samples.json')
    parser.add_argument('--images_dir', required=True,
                       help='Directory containing transferred COCO images')
    parser.add_argument('--output_dir', default='./sam_proposals_refcoco',
                       help='Output directory for proposals')
    parser.add_argument('--max_images', type=int, default=20,
                       help='Maximum number of images to process (for testing)')
    parser.add_argument('--max_proposals', type=int, default=20,
                       help='Maximum proposals per image')
    parser.add_argument('--sam_config', default='/home/min/a/mukher44/Work/SSL/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml',
                       help='SAM config file path')
    parser.add_argument('--sam_checkpoint', default='/home/min/a/mukher44/Work/SSL/sam2/checkpoints/sam2.1_hiera_large.pt',
                       help='SAM checkpoint path')
    
    args = parser.parse_args()
    
    # Validate input paths
    if not Path(args.annotations_file).exists():
        raise FileNotFoundError(f"Annotations file not found: {args.annotations_file}")
    
    if not Path(args.images_dir).exists():
        raise FileNotFoundError(f"Images directory not found: {args.images_dir}")
        
    if not Path(args.sam_config).exists():
        raise FileNotFoundError(f"SAM config not found: {args.sam_config}")
        
    if not Path(args.sam_checkpoint).exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize SAM with standard high-quality settings
    print("Initializing SAM with standard high-quality settings...")
    '''
    sam_generator = SAM2AutomaticMaskGenerator(
        model=build_sam2_robust(args.sam_config, args.sam_checkpoint),
        points_per_side=32,
        pred_iou_thresh=0.88,           # High quality only
        stability_score_thresh=0.95,    # Very stable masks
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,       # Filter tiny objects
    )
    '''
    '''
    sam_generator = SAM2AutomaticMaskGenerator(
    model=build_sam2_robust(args.sam_config, args.sam_checkpoint),
    points_per_side=24,                    # Reduced density to focus on larger objects
    pred_iou_thresh=0.75,                  # Lowered from 0.88 to capture complete objects
    stability_score_thresh=0.85,           # Lowered from 0.95 to include fuller masks
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500,              # Increased from 100 to filter small fragments
    )
    '''
    # Apply these universally (no GT knowledge)
    sam_generator = SAM2AutomaticMaskGenerator(
    model=build_sam2_robust(args.sam_config, args.sam_checkpoint),
    points_per_side=32,                    # Increased from 24 for better coverage
    pred_iou_thresh=0.70,                  # Further lowered for struggling categories
    stability_score_thresh=0.80,           # Further lowered for boundary cases
    crop_n_layers=1,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=100,              # DRAMATICALLY reduced from 500
    box_nms_thresh=0.65,                   # Slightly aggressive NMS for better diversity
    )
    print("SAM Configuration:")
    print(f"  pred_iou_thresh: 0.70 (high quality)")
    print(f"  stability_score_thresh: 0.80 (stable)")
    print(f"  NMS IoU threshold: 0.65 (standard)")
    print(f"  max_proposals: {args.max_proposals}")
    
    # Load visual grounding data
    print(f"\nLoading visual grounding annotations...")
    samples = load_visual_grounding_data(args.annotations_file)
    
    # Get unique image IDs for processing
    image_ids = list(set(sample['image_id'] for sample in samples))
    print(f"Found {len(image_ids)} unique images from {len(samples)} samples")
    
    # Select images to process (first N for testing)
    image_ids_to_process = image_ids[:args.max_images]
    print(f"Processing first {len(image_ids_to_process)} images for testing")
    
    # Results storage
    results = {
        "sam_proposals": {},
        "image_metadata": {},
        "processing_stats": {
            "total_images": len(image_ids_to_process),
            "successful_images": 0,
            "failed_images": 0,
            "total_proposals_generated": 0
        }
    }
    
    # Process each image
    failed_images = []
    
    for i, image_id in enumerate(image_ids_to_process):
        print(f"\nProcessing image {i+1}/{len(image_ids_to_process)}: {image_id}")
        
        try:
            # Get image path and verify it exists
            image_path = get_coco_image_path(image_id, args.images_dir)
            
            if not image_path.exists():
                print(f"  ERROR: Image not found: {image_path}")
                failed_images.append(image_id)
                continue
            
            # Load image
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            H, W = image_np.shape[:2]
            
            print(f"  Image: {image_path.name} ({W}x{H})")
            
            # Generate SAM proposals
            proposals = generate_sam_proposals(image_np, sam_generator, args.max_proposals)
            print(f"  Generated {len(proposals)} final proposals")
            
            # Store results for this image
            results["sam_proposals"][image_id] = {
                "image_path": str(image_path),
                "image_id": image_id,
                "image_size": [W, H],
                "proposals": proposals,
                "num_proposals": len(proposals)
            }
            
            results["image_metadata"][image_id] = {
                "filename": image_path.name,
                "width": W,
                "height": H,
                "proposals_generated": len(proposals)
            }
            
            # Update stats
            results["processing_stats"]["successful_images"] += 1
            results["processing_stats"]["total_proposals_generated"] += len(proposals)
            
        except Exception as e:
            print(f"  ERROR processing image {image_id}: {e}")
            traceback.print_exc()
            failed_images.append(image_id)
            results["processing_stats"]["failed_images"] += 1
            continue
    
    # Add metadata
    results["metadata"] = {
        "annotations_file": args.annotations_file,
        "images_dir": args.images_dir,
        "sam_config": {
            "points_per_side": 32,
            "pred_iou_thresh": 0.88,
            "stability_score_thresh": 0.95,
            "min_mask_region_area": 100,
            "nms_threshold": 0.7,
            "max_proposals_per_image": args.max_proposals
        },
        "failed_images": failed_images,
        "version": "standard_quality"
    }
    
    # Save results
    output_file = output_dir / f"sam_proposals_refcoco_{len(image_ids)}images.pth"
    torch.save(results, output_file)
    
    # Print summary
    stats = results["processing_stats"]
    print(f"\n{'='*60}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images attempted: {stats['total_images']}")
    print(f"Successfully processed: {stats['successful_images']}")
    print(f"Failed: {stats['failed_images']}")
    print(f"Total SAM proposals generated: {stats['total_proposals_generated']}")
    print(f"Average proposals per image: {stats['total_proposals_generated'] / max(1, stats['successful_images']):.1f}")
    
    if failed_images:
        print(f"\nFailed images: {failed_images}")
    
    print(f"\nResults saved to: {output_file}")
    print(f"Ready for visualization and evaluation!")

if __name__ == "__main__":
    main()