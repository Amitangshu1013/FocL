#!/usr/bin/env python3
"""
SAM vs Ground Truth Visualization
Creates side-by-side visualizations showing GT bboxes vs best matching SAM proposals
"""

import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path
import argparse
from collections import defaultdict
from tqdm import tqdm
import random

def calculate_iou(box1_xywh, box2_xywh):
    """Calculate IoU between two bounding boxes in xywh format"""
    x1, y1, w1, h1 = box1_xywh
    x2, y2, w2, h2 = box2_xywh
    
    # Convert to xyxy
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_max, y2_max = x2 + w2, y2 + h2
    
    # Calculate intersection
    intersect_x1 = max(x1, x2)
    intersect_y1 = max(y1, y2)
    intersect_x2 = min(x1_max, x2_max)
    intersect_y2 = min(y1_max, y2_max)
    
    if intersect_x2 <= intersect_x1 or intersect_y2 <= intersect_y1:
        return 0.0
    
    intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersect_area
    
    return intersect_area / union_area if union_area > 0 else 0.0

def load_coco_annotations_subset(coco_annotations_file, target_image_ids):
    """Load COCO detection annotations for specific image subset"""
    print(f"Loading COCO annotations from: {coco_annotations_file}")
    
    with open(coco_annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Create lookup dictionaries
    categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    images_dict = {img['id']: img for img in coco_data['images'] if img['id'] in target_image_ids}
    
    # Filter annotations to only target images
    target_image_set = set(target_image_ids)
    filtered_annotations = [ann for ann in coco_data['annotations'] 
                          if ann['image_id'] in target_image_set]
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for ann in filtered_annotations:
        annotations_by_image[ann['image_id']].append(ann)
    
    print(f"Loaded annotations for {len(images_dict)} images")
    print(f"Total ground truth objects: {len(filtered_annotations)}")
    
    return images_dict, categories_dict, annotations_by_image

def load_sam_proposals(sam_proposals_file):
    """Load SAM proposals"""
    print(f"Loading SAM proposals from: {sam_proposals_file}")
    sam_data = torch.load(sam_proposals_file, map_location='cpu')
    
    if 'sam_proposals' in sam_data:
        sam_proposals = sam_data['sam_proposals']
        sam_metadata = sam_data.get('metadata', {})
    else:
        sam_proposals = sam_data
        sam_metadata = {}
    
    print(f"Loaded SAM proposals for {len(sam_proposals)} images")
    return sam_proposals, sam_metadata

def get_target_image_ids(refcoco_file):
    """Extract image IDs from RefCOCO samples"""
    with open(refcoco_file, 'r') as f:
        refcoco_samples = json.load(f)
    
    target_image_ids = list(set(sample['image_id'] for sample in refcoco_samples))
    return target_image_ids

def find_best_sam_matches(gt_objects, sam_proposals, categories_dict, iou_threshold=0.5):
    """Find best SAM proposal matches for each GT object"""
    matches = []
    
    for gt_obj in gt_objects:
        gt_bbox = gt_obj['bbox']
        gt_category = categories_dict[gt_obj['category_id']]
        
        # Find best SAM proposal for this GT object
        best_iou = 0.0
        best_proposal = None
        
        for sam_prop in sam_proposals:
            sam_bbox = sam_prop['bbox_xywh']
            iou = calculate_iou(gt_bbox, sam_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_proposal = sam_prop
        
        # Only include if above threshold
        if best_iou >= iou_threshold:
            matches.append({
                'gt_bbox': gt_bbox,
                'gt_category': gt_category,
                'sam_bbox': best_proposal['bbox_xywh'],
                'iou': best_iou,
                'sam_score': best_proposal['sam_score']
            })
    
    return matches

def visualize_sam_gt_comparison(image_id, gt_objects, sam_proposals, categories_dict, 
                               images_dir, output_dir, iou_threshold=0.5):
    """Create side-by-side GT vs SAM visualization"""
    
    # Get image path
    image_filename = f"COCO_train2014_{image_id:012d}.jpg"
    image_path = Path(images_dir) / image_filename
    
    if not image_path.exists():
        print(f"Warning: Image not found: {image_path}")
        return False
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Find matching SAM proposals
    matches = find_best_sam_matches(gt_objects, sam_proposals, categories_dict, iou_threshold)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Ground Truth
    ax1.imshow(image)
    ax1.set_title(f'Ground Truth\nImage ID: {image_id}', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # Draw all GT bboxes
    for gt_obj in gt_objects:
        gt_bbox = gt_obj['bbox']
        gt_category = categories_dict[gt_obj['category_id']]
        
        x, y, w, h = gt_bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='saddlebrown', 
                               facecolor='none', linestyle='-')
        ax1.add_patch(rect)
        
        # Add category label
        ax1.text(x, y-5, gt_category, fontsize=8, color='saddlebrown', 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Right plot: SAM Best Matches
    ax2.imshow(image)
    ax2.set_title(f'SAM Best Matches (IoU≥{iou_threshold})\nImage ID: {image_id}', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # Draw matching SAM proposals
    for match in matches:
        sam_bbox = match['sam_bbox']
        gt_category = match['gt_category']
        iou = match['iou']
        
        x, y, w, h = sam_bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='darkgreen', 
                               facecolor='none', linestyle='-')
        ax2.add_patch(rect)
        
        # Add category and IoU label
        label_text = f"{gt_category}\nIoU: {iou:.2f}"
        ax2.text(x, y-15, label_text, fontsize=8, color='darkgreen', 
                fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    # Add summary statistics
    total_gt = len(gt_objects)
    found_objects = len(matches)
    coverage_pct = 100 * found_objects / total_gt if total_gt > 0 else 0
    
    summary_text = f"GT Objects: {total_gt}\nSAM Found: {found_objects}\nCoverage: {coverage_pct:.1f}%"
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save visualization
    output_file = output_dir / f'sam_gt_comparison_image_{image_id}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return True

def select_diverse_visualization_samples(sam_proposals, annotations_by_image, 
                                       categories_dict, num_samples=15, iou_threshold=0.5):
    """Select diverse samples for visualization based on SAM performance"""
    
    # Find common images
    common_image_ids = list(set(sam_proposals.keys()) & set(annotations_by_image.keys()))
    
    # Calculate coverage for each image
    image_stats = []
    
    for image_id in common_image_ids:
        gt_objects = annotations_by_image[image_id]
        sam_props = sam_proposals[image_id]['proposals']
        
        matches = find_best_sam_matches(gt_objects, sam_props, categories_dict, iou_threshold)
        
        total_gt = len(gt_objects)
        found_objects = len(matches)
        coverage = found_objects / total_gt if total_gt > 0 else 0
        
        image_stats.append({
            'image_id': image_id,
            'total_gt': total_gt,
            'found_objects': found_objects,
            'coverage': coverage,
            'num_proposals': len(sam_props)
        })
    
    # Sort by coverage
    image_stats.sort(key=lambda x: x['coverage'])
    
    # Select diverse samples
    total_images = len(image_stats)
    selected = []
    
    # Low coverage (bottom 30%)
    low_end = int(0.3 * total_images)
    low_coverage_samples = random.sample(image_stats[:low_end], min(5, low_end))
    selected.extend(low_coverage_samples)
    
    # Medium coverage (middle 40%)
    mid_start = int(0.3 * total_images)
    mid_end = int(0.7 * total_images)
    mid_coverage_samples = random.sample(image_stats[mid_start:mid_end], min(5, mid_end - mid_start))
    selected.extend(mid_coverage_samples)
    
    # High coverage (top 30%)
    high_start = int(0.7 * total_images)
    high_coverage_samples = random.sample(image_stats[high_start:], min(5, total_images - high_start))
    selected.extend(high_coverage_samples)
    
    # Trim to exact number requested
    selected = selected[:num_samples]
    
    print(f"Selected {len(selected)} diverse samples:")
    for sample in selected:
        print(f"  Image {sample['image_id']}: {sample['coverage']:.1%} coverage ({sample['found_objects']}/{sample['total_gt']} objects)")
    
    return [s['image_id'] for s in selected]

def main():
    parser = argparse.ArgumentParser(description="SAM vs Ground Truth Visualization")
    
    parser.add_argument('--coco_annotations_file', required=True,
                       help='Path to instances_train2014.json')
    parser.add_argument('--sam_proposals_file', required=True,
                       help='Path to SAM proposals .pth file')
    parser.add_argument('--refcoco_file', required=True,
                       help='Path to RefCOCO samples JSON (for image filtering)')
    parser.add_argument('--images_dir', required=True,
                       help='Directory containing COCO images')
    parser.add_argument('--output_dir', default='./sam_gt_visualization',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=15,
                       help='Number of sample visualizations to create')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for considering SAM proposals as matches')
    
    args = parser.parse_args()
    
    # Set random seed for reproducible sample selection
    random.seed(42)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get target image IDs from RefCOCO
    target_image_ids = get_target_image_ids(args.refcoco_file)
    
    # Load data
    images_dict, categories_dict, annotations_by_image = load_coco_annotations_subset(
        args.coco_annotations_file, target_image_ids)
    sam_proposals, sam_metadata = load_sam_proposals(args.sam_proposals_file)
    
    # Select diverse samples for visualization
    selected_image_ids = select_diverse_visualization_samples(
        sam_proposals, annotations_by_image, categories_dict, 
        args.num_samples, args.iou_threshold)
    
    print(f"\nCreating visualizations for {len(selected_image_ids)} selected images...")
    
    # Create visualizations
    successful_visualizations = 0
    for image_id in tqdm(selected_image_ids):
        success = visualize_sam_gt_comparison(
            image_id,
            annotations_by_image[image_id],
            sam_proposals[image_id]['proposals'],
            categories_dict,
            args.images_dir,
            output_dir,
            args.iou_threshold
        )
        if success:
            successful_visualizations += 1
    
    print(f"\nCompleted! Successfully created {successful_visualizations}/{len(selected_image_ids)} visualizations")
    print(f"Output directory: {output_dir}")
    
    # Save selection summary
    summary_file = output_dir / 'visualization_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("SAM vs Ground Truth Visualization Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"IoU threshold: {args.iou_threshold}\n")
        f.write(f"Total visualizations: {successful_visualizations}\n")
        f.write(f"Images selected for diverse coverage representation\n\n")
        
        # Re-calculate stats for selected images
        for image_id in selected_image_ids:
            if image_id in annotations_by_image and image_id in sam_proposals:
                gt_objects = annotations_by_image[image_id]
                sam_props = sam_proposals[image_id]['proposals']
                matches = find_best_sam_matches(gt_objects, sam_props, categories_dict, args.iou_threshold)
                
                coverage = len(matches) / len(gt_objects) if gt_objects else 0
                f.write(f"Image {image_id}: {coverage:.1%} coverage ({len(matches)}/{len(gt_objects)} objects found)\n")
    
    print(f"Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()