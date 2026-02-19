#!/usr/bin/env python3
"""
Detection Comparison Visualization - Complete Version
Creates side-by-side comparisons showing FocL vs Standard detection results.
Highlights images where FocL significantly outperforms Standard.
Green boxes = correct detections, Red boxes = incorrect detections.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from pathlib import Path
import argparse
from collections import defaultdict
from PIL import Image

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

def evaluate_detections_for_image(predictions, ground_truth, iou_threshold=0.5):
    """
    Evaluate predictions for a single image
    Returns list of (prediction, is_correct) tuples
    """
    if not predictions:
        return []
    
    if not ground_truth:
        # No ground truth - all predictions are incorrect
        return [(pred, False) for pred in predictions]
    
    # Group ground truth by category
    gt_by_category = defaultdict(list)
    for gt in ground_truth:
        gt_by_category[gt['category']].append(gt['bbox'])
    
    results = []
    
    for pred in predictions:
        pred_category = pred['category']
        pred_bbox = pred['bbox']
        
        # Check if this category exists in ground truth
        if pred_category not in gt_by_category:
            results.append((pred, False))
            continue
        
        # Find best IoU with ground truth boxes of same category
        best_iou = 0.0
        for gt_bbox in gt_by_category[pred_category]:
            iou = calculate_iou(pred_bbox, gt_bbox)
            best_iou = max(best_iou, iou)
        
        # Mark as correct if IoU exceeds threshold
        is_correct = best_iou >= iou_threshold
        results.append((pred, is_correct))
    
    return results

def load_detection_results(results_file):
    """Load detection results from Test 3 output"""
    print(f"Loading detection results from: {results_file}")
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        if 'predictions_by_image' not in data:
            raise ValueError(f"File {results_file} missing 'predictions_by_image' field")
        
        print(f"Loaded predictions for {len(data['predictions_by_image'])} images")
        return data
    except Exception as e:
        raise RuntimeError(f"Failed to load {results_file}: {e}")

def load_ground_truth(coco_annotations_file, target_image_ids, valid_categories):
    """Load ground truth annotations"""
    print(f"Loading ground truth from: {coco_annotations_file}")
    
    with open(coco_annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build category lookup
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Filter annotations
    target_image_set = set(int(img_id) for img_id in target_image_ids)  # Ensure integers
    valid_category_set = set(valid_categories)
    
    gt_by_image = defaultdict(list)
    
    for ann in coco_data['annotations']:
        if ann['image_id'] not in target_image_set:
            continue
        
        category_name = category_id_to_name.get(ann['category_id'])
        if category_name not in valid_category_set:
            continue
        
        gt_by_image[ann['image_id']].append({
            'bbox': ann['bbox'],
            'category': category_name
        })
    
    print(f"Loaded ground truth for {len(gt_by_image)} images")
    return gt_by_image

def draw_detections_on_image(ax, image, detections_with_correctness, title):
    """Draw image with detection boxes colored by correctness"""
    ax.imshow(image)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Draw each detection
    for detection, is_correct in detections_with_correctness:
        bbox = detection['bbox']  # [x, y, w, h]
        category = detection['category']
        confidence = detection.get('confidence', 0.0)
        
        # Choose color based on correctness
        color = 'green' if is_correct else 'red'
        
        # Create rectangle
        rect = Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label_text = f"{category}\n{confidence:.2f}"
        ax.text(
            bbox[0], bbox[1] - 5, label_text,
            color=color, fontsize=8, fontweight='bold',
            verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8)
        )

def create_comparison_visualization(image_id, images_dir, 
                                  focl_detections, standard_detections,
                                  ground_truth, output_dir, iou_threshold=0.5):
    """Create side-by-side comparison for a single image"""
    
    # Convert image_id to integer for filename
    image_id_int = int(image_id)
    
    # Load image
    image_filename = f"COCO_train2014_{image_id_int:012d}.jpg"
    image_path = Path(images_dir) / image_filename
    
    print(f"Loading image: {image_filename}")
    print(f"Full path: {image_path}")
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        # List files in directory to debug
        image_dir = Path(images_dir)
        if image_dir.exists():
            matching_files = list(image_dir.glob(f"*{image_id_int}*"))
            print(f"Files matching {image_id_int}: {matching_files}")
        return None
    
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Successfully loaded image: {image.size}")
    except Exception as e:
        print(f"ERROR: Failed to load image {image_path}: {e}")
        return None
    
    # Evaluate detections
    gt_for_image = ground_truth.get(image_id_int, [])
    print(f"Ground truth objects: {len(gt_for_image)}")
    
    focl_evaluated = evaluate_detections_for_image(focl_detections, gt_for_image, iou_threshold)
    standard_evaluated = evaluate_detections_for_image(standard_detections, gt_for_image, iou_threshold)
    
    # Count correct detections
    focl_correct = sum(1 for _, is_correct in focl_evaluated if is_correct)
    standard_correct = sum(1 for _, is_correct in standard_evaluated if is_correct)
    
    print(f"FocL: {focl_correct}/{len(focl_evaluated)} correct")
    print(f"Standard: {standard_correct}/{len(standard_evaluated)} correct")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # FocL results (left)
    draw_detections_on_image(
        ax1, image, focl_evaluated, 
        f"FocL: {focl_correct}/{len(focl_evaluated)} correct"
    )
    
    # Standard results (right)
    draw_detections_on_image(
        ax2, image, standard_evaluated,
        f"Standard: {standard_correct}/{len(standard_evaluated)} correct"
    )
    
    # Add overall title
    advantage = focl_correct - standard_correct
    fig.suptitle(
        f"Image {image_id} - FocL Advantage: +{advantage} correct detections\n"
        f"Green=Correct, Red=Incorrect (IoU≥{iou_threshold})",
        fontsize=16, fontweight='bold'
    )
    
    plt.tight_layout()
    
    # Save visualization
    output_file = Path(output_dir) / f"comparison_{image_id}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {output_file}")
    return output_file, focl_correct, standard_correct

def main():
    parser = argparse.ArgumentParser(description="Create detection comparison visualizations")
    
    # Required arguments
    parser.add_argument('--focl_results', required=True,
                       help='Path to FocL Test 3 predictions JSON file')
    parser.add_argument('--standard_results', required=True,
                       help='Path to Standard Test 3 predictions JSON file')
    parser.add_argument('--coco_annotations_file', required=True,
                       help='Path to instances_train2014.json')
    parser.add_argument('--images_dir', required=True,
                       help='Directory containing COCO images')
    parser.add_argument('--mapping_file', required=True,
                       help='Path to coco_imagenet_mapping.json')
    
    # Optional arguments
    parser.add_argument('--output_dir', default='./comparison_visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--min_advantage', type=int, default=1,
                       help='Minimum FocL advantage (correct detections) to visualize')
    parser.add_argument('--max_images', type=int, default=10,
                       help='Maximum number of images to visualize')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for correct detections')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DETECTION COMPARISON VISUALIZATION")
    print("="*60)
    
    # Load mapping data to get valid categories (direct mappings only)
    print("Loading mapping data...")
    with open(args.mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    valid_categories = []
    for coco_cat, imagenet_classes in mapping_data["direct_mapping"].items():
        if imagenet_classes:  # Non-empty list means direct mapping exists
            valid_categories.append(coco_cat)
    
    print(f"Direct mapping categories: {len(valid_categories)}")
    print(f"Categories: {sorted(valid_categories)}")
    
    # Load detection results
    try:
        focl_data = load_detection_results(args.focl_results)
        standard_data = load_detection_results(args.standard_results)
        
        focl_predictions = focl_data['predictions_by_image']
        standard_predictions = standard_data['predictions_by_image']
        
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    # Load ground truth
    # Get all image IDs from predictions
    all_image_ids = set(list(focl_predictions.keys()) + list(standard_predictions.keys()))
    
    # Use ground truth from file if available, otherwise load from COCO
    if 'ground_truth_by_image' in focl_data:
        print("Using ground truth from predictions file")
        ground_truth = focl_data['ground_truth_by_image']
        # Convert string keys to int keys for consistency
        ground_truth = {int(k): v for k, v in ground_truth.items()}
    else:
        print("Loading ground truth from COCO annotations...")
        ground_truth = load_ground_truth(args.coco_annotations_file, all_image_ids, valid_categories)
    
    # Find common images between both models
    common_images = set(focl_predictions.keys()) & set(standard_predictions.keys())
    print(f"Common images between models: {len(common_images)}")
    
    if not common_images:
        print("ERROR: No common images found between FocL and Standard predictions")
        return
    
    # Find images where FocL outperforms Standard
    print("Analyzing FocL advantages...")
    advantage_results = []
    
    for image_id in common_images:
        image_id_int = int(image_id)  # Convert for ground truth lookup
        
        if image_id_int not in ground_truth:
            continue
        
        # Get detections for this image (filter to direct mapping categories)
        focl_detections = [det for det in focl_predictions[image_id] 
                          if det['category'] in valid_categories]
        standard_detections = [det for det in standard_predictions[image_id] 
                             if det['category'] in valid_categories]
        
        if not focl_detections and not standard_detections:
            continue
        
        # Evaluate detections
        gt_for_image = ground_truth[image_id_int]
        
        focl_evaluated = evaluate_detections_for_image(focl_detections, gt_for_image, args.iou_threshold)
        standard_evaluated = evaluate_detections_for_image(standard_detections, gt_for_image, args.iou_threshold)
        
        # Count correct detections
        focl_correct = sum(1 for _, is_correct in focl_evaluated if is_correct)
        standard_correct = sum(1 for _, is_correct in standard_evaluated if is_correct)
        
        advantage = focl_correct - standard_correct
        
        if advantage >= args.min_advantage:
            advantage_results.append((
                image_id, focl_correct, standard_correct, advantage,
                len(focl_detections), len(standard_detections)
            ))
    
    # Sort by advantage (highest first)
    advantage_results.sort(key=lambda x: x[3], reverse=True)
    
    print(f"\nFound {len(advantage_results)} images where FocL advantage >= {args.min_advantage}")
    
    if not advantage_results:
        print("No images found where FocL significantly outperforms Standard")
        print(f"Try reducing --min_advantage (current: {args.min_advantage})")
        return
    
    # Print summary
    print(f"\nTOP {min(10, len(advantage_results))} FOCL ADVANTAGES:")
    for i, (image_id, focl_correct, standard_correct, advantage, focl_total, standard_total) in enumerate(advantage_results[:10]):
        print(f"{i+1:2d}. Image {image_id}: FocL {focl_correct}/{focl_total}, "
              f"Standard {standard_correct}/{standard_total}, Advantage +{advantage}")
    
    # Create visualizations for top images
    num_to_visualize = min(args.max_images, len(advantage_results))
    print(f"\nCreating visualizations for top {num_to_visualize} images...")
    
    created_visualizations = []
    
    for i, (image_id, focl_correct, standard_correct, advantage, focl_total, standard_total) in enumerate(advantage_results[:num_to_visualize]):
        print(f"\n--- Creating visualization {i+1}/{num_to_visualize} ---")
        print(f"Image {image_id}: FocL {focl_correct}/{focl_total}, Standard {standard_correct}/{standard_total}")
        
        # Get filtered detections
        focl_detections = [det for det in focl_predictions[image_id] 
                          if det['category'] in valid_categories]
        standard_detections = [det for det in standard_predictions[image_id] 
                             if det['category'] in valid_categories]
        
        # Create visualization
        result = create_comparison_visualization(
            image_id, args.images_dir,
            focl_detections, standard_detections,
            ground_truth, args.output_dir, args.iou_threshold
        )
        
        if result:
            output_file, actual_focl_correct, actual_standard_correct = result
            created_visualizations.append(output_file)
        else:
            print(f"Failed to create visualization for image {image_id}")
    
    print(f"\n{'='*60}")
    print(f"VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Created {len(created_visualizations)} visualization(s)")
    print(f"Output directory: {output_dir}")
    print(f"IoU threshold: {args.iou_threshold}")
    print(f"Direct mapping categories only: {len(valid_categories)} categories")

if __name__ == "__main__":
    main()
