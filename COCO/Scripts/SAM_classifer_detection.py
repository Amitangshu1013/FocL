#!/usr/bin/env python3
"""
Cross domain Detection in a zero-shot setting (Cross-Domain Generalization on COCO — Paper §4.2)

This script evaluates an end-to-end *proposal + classification* detection pipeline:

    SAM proposals (dorsal / "where")  →  crop each proposal bbox  →  ResNet-50 classifier (ventral / "what")
                                      →  confidence thresholding  →  per-class NMS  →  COCO-style mAP

Key points:
- Uses *pre-computed* SAM proposals (loaded from a .pth file). No SAM inference is run here.
- Assigns a COCO category to each proposal via a COCO↔ImageNet mapping:
    score(coco_cat) = max softmax probability over mapped ImageNet classes.
- Produces detections in standard form: one label + one confidence per proposal.
- Applies Non-Maximum Suppression (NMS) within each predicted category to remove duplicates.
- Reports mAP at IoU thresholds 0.3 and 0.5, and also reports results split by:
    (i) direct COCO↔ImageNet mappings, and (ii) supercategory-based mappings.
  NOTE: The paper reports *direct-mapping* mAP only (the intended primary metric).

Outputs:
- JSON summary with overall and per-category AP, plus direct vs supercategory breakdown.
- Plots for overall mAP and mapping-type comparisons.
- Optional dump of predictions/GT for downstream visualization.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torchvision.ops import nms
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

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

def compute_ap(recall, precision):
    """Compute Average Precision using 11-point interpolation"""
    # Add sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    # Look for points where X axis changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Sum over all the intervals
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def evaluate_detections_single_category(gt_boxes, pred_boxes, pred_scores, iou_threshold=0.5):
    """Evaluate predictions for a single category using mAP calculation"""
    
    if len(pred_boxes) == 0:
        return 0.0, np.array([]), np.array([])
    
    if len(gt_boxes) == 0:
        # No ground truth for this category
        return 0.0, np.zeros(len(pred_boxes)), np.ones(len(pred_boxes))
    
    # Sort predictions by confidence score (descending)
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Track which ground truth boxes have been matched
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    
    # Arrays to store TP/FP for each prediction
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    # Evaluate each prediction
    for pred_idx in range(len(pred_boxes)):
        pred_box = pred_boxes[pred_idx]
        
        # Find the best matching ground truth box
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_boxes)):
            if gt_matched[gt_idx]:
                continue  # Already matched
                
            iou = calculate_iou(pred_box, gt_boxes[gt_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Determine if this prediction is TP or FP
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)  # Avoid division by zero
    
    # Compute AP
    ap = compute_ap(recalls, precisions)
    
    return ap, recalls, precisions

class FullDetectionEvaluator:
    def __init__(self, mapping_data, coco_categories, imagenet_classes):
        self.mapping_data = mapping_data
        self.coco_categories = coco_categories
        self.imagenet_classes = imagenet_classes
        # Build ImageNet class name lookup
        self.imagenet_idx_to_name = {int(k): v[1] for k, v in imagenet_classes.items()}
        self.imagenet_name_to_idx = {v[1]: int(k) for k, v in imagenet_classes.items()}
        
        # Build COCO category lookup
        self.coco_name_to_info = {cat['name']: cat for cat in coco_categories}
        self.coco_id_to_info = {cat['id']: cat for cat in coco_categories}
        
        # Build mappings
        print("Building COCO → ImageNet mappings...")
        self.coco_to_imagenet, self.imagenet_to_coco = self.build_mappings()
        
        # Get valid COCO categories for evaluation
        self.valid_coco_categories = self.get_valid_categories()
        print(f"Valid COCO categories for evaluation: {len(self.valid_coco_categories)}")
        
        self.reset_stats()
    
    def build_mappings(self):
        """Build bidirectional mappings between COCO and ImageNet"""
        coco_to_imagenet = {}
        imagenet_to_coco = {}
        
        # Process direct mappings
        for coco_category, imagenet_classes in self.mapping_data["direct_mapping"].items():
            if imagenet_classes:  # Non-empty list
                coco_to_imagenet[coco_category] = imagenet_classes
                for imagenet_class in imagenet_classes:
                    if imagenet_class not in imagenet_to_coco:
                        imagenet_to_coco[imagenet_class] = []
                    imagenet_to_coco[imagenet_class].append(coco_category)
        
        # Process supercategory mappings
        for supercategory, imagenet_classes in self.mapping_data["supercategory_mapping"].items():
            if not imagenet_classes:
                continue
                
            # Get all COCO categories in this supercategory
            coco_categories_in_super = [
                cat['name'] for cat in self.coco_categories 
                if cat['supercategory'] == supercategory
            ]
            
            for coco_category in coco_categories_in_super:
                # Only add supercategory mapping if no direct mapping exists
                if coco_category not in coco_to_imagenet:
                    coco_to_imagenet[coco_category] = imagenet_classes
                    
                for imagenet_class in imagenet_classes:
                    if imagenet_class not in imagenet_to_coco:
                        imagenet_to_coco[imagenet_class] = []
                    if coco_category not in imagenet_to_coco[imagenet_class]:
                        imagenet_to_coco[imagenet_class].append(coco_category)
        
        # Remove duplicates
        for imagenet_class in imagenet_to_coco:
            imagenet_to_coco[imagenet_class] = list(set(imagenet_to_coco[imagenet_class]))
        
        print(f"COCO → ImageNet mappings: {len(coco_to_imagenet)}")
        print(f"ImageNet → COCO mappings: {len(imagenet_to_coco)}")
        
        return coco_to_imagenet, imagenet_to_coco
    
    def get_valid_categories(self):
        """Get list of COCO categories that have mappings"""
        return list(self.coco_to_imagenet.keys())
    
    def get_mapping_type(self, coco_category):
        """Determine mapping type for a COCO category"""
        direct_mappings = self.mapping_data["direct_mapping"].get(coco_category, [])
        if direct_mappings:
            return "direct"
        
        supercategory = self.coco_name_to_info[coco_category]['supercategory']
        super_mappings = self.mapping_data["supercategory_mapping"].get(supercategory, [])
        if super_mappings:
            return "supercategory"
        
        return "none"
    
    def reset_stats(self):
        """Reset evaluation statistics"""
        self.stats = {
            'overall': {
                'total_images': 0,
                'total_gt_objects': 0,
                'total_detections': 0,
                'total_detections_after_nms': 0,
                'ap_by_category': {},
                'mean_ap_03': 0.0,
                'mean_ap_05': 0.0
            },
            'direct_mapping': {
                'categories': [],
                'ap_by_category': {},
                'mean_ap_03': 0.0,
                'mean_ap_05': 0.0
            },
            'supercategory_mapping': {
                'categories': [],
                'ap_by_category': {},
                'mean_ap_03': 0.0,
                'mean_ap_05': 0.0
            },
            'processing_stats': {
                'images_processed': 0,
                'images_with_detections': 0,
                'avg_proposals_per_image': 0.0,
                'avg_detections_after_nms': 0.0
            }
        }
        
        # Initialize category lists by mapping type
        for category in self.valid_coco_categories:
            mapping_type = self.get_mapping_type(category)
            if mapping_type == "direct":
                self.stats['direct_mapping']['categories'].append(category)
            elif mapping_type == "supercategory":
                self.stats['supercategory_mapping']['categories'].append(category)

class ZeroShotDetector:
    def __init__(self, model_path, model_type, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        print(f"Using device: {self.device}")
        print(f"Loading {model_type} model...")
        
        # Load ResNet-50 architecture
        self.model = resnet50(pretrained=False, num_classes=1000)
        
        # Load model weights
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading {model_type} model from: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            if any(key.startswith('module.') for key in state_dict.keys()):
                state_dict = {key[7:]: value for key, value in state_dict.items()}
            
            self.model.load_state_dict(state_dict)
            print(f"{model_type} model loaded successfully!")
            
        except Exception as e:
            raise RuntimeError(f"Error loading {model_type} model: {e}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def crop_bbox_from_image(self, image, bbox_xywh):
        """Crop bounding box region from image"""
        x, y, w, h = bbox_xywh
        
        # Ensure coordinates are within image bounds
        img_width, img_height = image.size
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        w = max(1, min(w, img_width - x))
        h = max(1, min(h, img_height - y))
        
        # Check for reasonable crop size
        if w < 10 or h < 10:
            return None
        
        # Crop the region
        crop_box = (x, y, x + w, y + h)
        try:
            cropped_image = image.crop(crop_box)
            return cropped_image
        except Exception:
            return None
    
    def classify_proposals(self, image, sam_proposals, evaluator, confidence_threshold=0.1):
        """
        Classify all SAM proposals and return detections with confidence scores
        
        Returns:
            detections: List of (bbox_xywh, category, confidence) tuples
        """
        if not sam_proposals:
            return []
        
        # Crop all proposals
        crops = []
        valid_proposals = []
        
        for proposal in sam_proposals:
            crop = self.crop_bbox_from_image(image, proposal['bbox_xywh'])
            if crop is not None:
                crops.append(crop)
                valid_proposals.append(proposal)
        
        if not crops:
            return []
        
        # Batch process all crops
        batch_tensors = torch.stack([self.transform(crop) for crop in crops]).to(self.device)
        
        # Get ImageNet predictions
        with torch.no_grad():
            logits = self.model(batch_tensors)
            probabilities = F.softmax(logits, dim=1)  # (N, 1000)
        
        # Convert to detections
        detections = []
        
        for i, (proposal, probs) in enumerate(zip(valid_proposals, probabilities)):
            bbox = proposal['bbox_xywh']
            
            # Find the best COCO category for this proposal
            best_category = None
            best_confidence = 0.0
            
            # Check each valid COCO category
            for coco_category in evaluator.valid_coco_categories:
                # Get ImageNet classes that map to this COCO category
                imagenet_classes = evaluator.coco_to_imagenet[coco_category]
                
                # Find max confidence among these ImageNet classes
                max_confidence = 0.0
                for imagenet_class in imagenet_classes:
                    if imagenet_class in evaluator.imagenet_name_to_idx:
                        idx = evaluator.imagenet_name_to_idx[imagenet_class]
                        confidence = probs[idx].item()
                        max_confidence = max(max_confidence, confidence)
                
                # Track the best category for this proposal
                if max_confidence > best_confidence:
                    best_confidence = max_confidence
                    best_category = coco_category
                  
            # Add detection if confidence is above threshold
            if best_category and best_confidence > confidence_threshold:  # Confidence threshold
                detections.append({
                    'bbox': bbox,
                    'category': best_category,
                    'confidence': best_confidence
                })
        
        return detections
    
    def apply_nms(self, detections, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if not detections:
            return []
        
        # Group detections by category
        detections_by_category = defaultdict(list)
        for det in detections:
            detections_by_category[det['category']].append(det)
        
        final_detections = []
        
        # Apply NMS within each category
        for category, category_detections in detections_by_category.items():
            if not category_detections:
                continue
            
            # Convert to tensors for NMS
            boxes = []
            scores = []
            
            for det in category_detections:
                # Convert xywh to xyxy for NMS
                x, y, w, h = det['bbox']
                boxes.append([x, y, x + w, y + h])
                scores.append(det['confidence'])
            
            boxes = torch.tensor(boxes, dtype=torch.float32)
            scores = torch.tensor(scores, dtype=torch.float32)
            
            # Apply NMS
            keep_indices = nms(boxes, scores, iou_threshold)
            
            # Keep only the selected detections
            for idx in keep_indices:
                final_detections.append(category_detections[idx])
        
        return final_detections

def get_target_image_ids(refcoco_file):
    """Extract image IDs from RefCOCO samples to get controlled 1772 image set"""
    print(f"Loading RefCOCO samples to get target image IDs...")
    
    with open(refcoco_file, 'r') as f:
        refcoco_samples = json.load(f)
    
    target_image_ids = list(set(sample['image_id'] for sample in refcoco_samples))
    print(f"Target images from RefCOCO: {len(target_image_ids)}")
    
    return target_image_ids

def load_coco_detection_annotations(coco_annotations_file, target_image_ids, evaluator):
    """Load COCO detection annotations for target images"""
    print(f"Loading COCO detection annotations from: {coco_annotations_file}")
    
    with open(coco_annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    # Filter to target images and valid categories only
    target_image_set = set(target_image_ids)
    valid_categories = set(evaluator.valid_coco_categories)
    
    annotations_by_image = defaultdict(list)
    category_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    for ann in coco_data['annotations']:
        if ann['image_id'] not in target_image_set:
            continue
            
        category_name = category_id_to_name.get(ann['category_id'])
        if category_name not in valid_categories:
            continue
            
        annotations_by_image[ann['image_id']].append({
            'bbox': ann['bbox'],
            'category': category_name,
            'area': ann.get('area', 0)
        })
    
    print(f"Loaded annotations for {len(annotations_by_image)} images")
    total_annotations = sum(len(anns) for anns in annotations_by_image.values())
    print(f"Total valid annotations: {total_annotations}")
    
    return annotations_by_image

def load_data_files(args):
    """Load all required data files"""
    print("Loading data files...")
    
    # Load COCO categories
    with open(args.coco_categories_file, 'r') as f:
        coco_categories = json.load(f)
    
    # Load COCO → ImageNet mapping
    with open(args.mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Load ImageNet class index
    with open(args.imagenet_classes_file, 'r') as f:
        imagenet_classes = json.load(f)
    
    # Load SAM proposals
    print(f"Loading SAM proposals from: {args.sam_proposals_file}")
    sam_data = torch.load(args.sam_proposals_file, map_location='cpu')
    
    if 'sam_proposals' in sam_data:
        sam_proposals = sam_data['sam_proposals']
        sam_metadata = sam_data.get('metadata', {})
    else:
        sam_proposals = sam_data
        sam_metadata = {}
    
    print(f"Loaded {len(coco_categories)} COCO categories")
    print(f"Loaded {len(imagenet_classes)} ImageNet classes")
    print(f"Loaded SAM proposals for {len(sam_proposals)} images")
    
    return coco_categories, mapping_data, imagenet_classes, sam_proposals, sam_metadata

def evaluate_detections(predictions_by_image, ground_truth_by_image, evaluator, iou_thresholds=[0.3, 0.5]):
    """Evaluate detections using mAP calculation"""
    
    # Organize data by category
    gt_by_category = defaultdict(list)
    pred_by_category = defaultdict(list)
    
    for image_id in ground_truth_by_image:
        # Ground truth
        for gt_ann in ground_truth_by_image[image_id]:
            gt_by_category[gt_ann['category']].append(gt_ann['bbox'])
        
        # Predictions
        if image_id in predictions_by_image:
            for pred in predictions_by_image[image_id]:
                pred_by_category[pred['category']].append({
                    'bbox': pred['bbox'],
                    'confidence': pred['confidence']
                })
    
    # Calculate AP for each category and IoU threshold
    results = {}
    
    for iou_thresh in iou_thresholds:
        results[f'mAP@{iou_thresh}'] = {}
        category_aps = []
        
        for category in evaluator.valid_coco_categories:
            gt_boxes = np.array(gt_by_category[category]) if gt_by_category[category] else np.array([])
            
            if category in pred_by_category and pred_by_category[category]:
                pred_data = pred_by_category[category]
                pred_boxes = np.array([p['bbox'] for p in pred_data])
                pred_scores = np.array([p['confidence'] for p in pred_data])
            else:
                pred_boxes = np.array([])
                pred_scores = np.array([])
            
            # Calculate AP for this category
            ap, _, _ = evaluate_detections_single_category(gt_boxes, pred_boxes, pred_scores, iou_thresh)
            
            results[f'mAP@{iou_thresh}'][category] = ap
            category_aps.append(ap)
        
        # Calculate mean AP
        results[f'mAP@{iou_thresh}']['mean'] = np.mean(category_aps) if category_aps else 0.0
    
    return results

def analyze_results(evaluator, results, output_dir, model_type):
    """Analyze and save mAP results"""
    
    print("\n" + "="*80)
    print(f"TEST 3: FULL ZERO-SHOT DETECTION RESULTS ({model_type.upper()} MODEL)")
    print("="*80)
    
    # Overall mAP results
    map_03 = results['mAP@0.3']['mean']
    map_05 = results['mAP@0.5']['mean']
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"mAP@0.3: {map_03:.4f}")
    print(f"mAP@0.5: {map_05:.4f}")
    print(f"Valid categories evaluated: {len(evaluator.valid_coco_categories)}")
    
    # Performance by mapping type
    for mapping_type in ['direct', 'supercategory']:
        categories = evaluator.stats[f'{mapping_type}_mapping']['categories']
        if not categories:
            continue
            
        # Calculate mAP for this mapping type
        map_03_subset = np.mean([results['mAP@0.3'][cat] for cat in categories])
        map_05_subset = np.mean([results['mAP@0.5'][cat] for cat in categories])
        
        print(f"\n{mapping_type.upper()} MAPPING PERFORMANCE:")
        print(f"Categories: {len(categories)}")
        print(f"mAP@0.3: {map_03_subset:.4f}")
        print(f"mAP@0.5: {map_05_subset:.4f}")
        
        # Top performing categories
        category_performance = [(cat, results['mAP@0.5'][cat]) for cat in categories]
        category_performance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Top 5 categories:")
        for i, (cat, ap) in enumerate(category_performance[:5]):
            print(f"  {i+1}. {cat}: {ap:.4f}")
        
        if len(category_performance) > 5:
            print(f"Bottom 5 categories:")
            for i, (cat, ap) in enumerate(category_performance[-5:]):
                print(f"  {i+1}. {cat}: {ap:.4f}")
    
    # Save detailed results
    results_file = Path(output_dir) / f'test3_full_detection_{model_type}_results.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'model_type': model_type,
            'experiment': 'full_zero_shot_detection',
            'summary': {
                'overall_map_03': float(map_03),
                'overall_map_05': float(map_05),
                'num_categories': len(evaluator.valid_coco_categories),
                'direct_mapping': {
                    'num_categories': len(evaluator.stats['direct_mapping']['categories']),
                    'map_03': float(np.mean([results['mAP@0.3'][cat] for cat in evaluator.stats['direct_mapping']['categories']])) if evaluator.stats['direct_mapping']['categories'] else 0.0,
                    'map_05': float(np.mean([results['mAP@0.5'][cat] for cat in evaluator.stats['direct_mapping']['categories']])) if evaluator.stats['direct_mapping']['categories'] else 0.0
                },
                'supercategory_mapping': {
                    'num_categories': len(evaluator.stats['supercategory_mapping']['categories']),
                    'map_03': float(np.mean([results['mAP@0.3'][cat] for cat in evaluator.stats['supercategory_mapping']['categories']])) if evaluator.stats['supercategory_mapping']['categories'] else 0.0,
                    'map_05': float(np.mean([results['mAP@0.5'][cat] for cat in evaluator.stats['supercategory_mapping']['categories']])) if evaluator.stats['supercategory_mapping']['categories'] else 0.0
                }
            },
            'detailed_results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create visualizations
    create_visualizations(results, evaluator, output_dir, model_type)
    
    return results

def create_visualizations(results, evaluator, output_dir, model_type):
    """Create visualization plots"""
    fig_dir = Path(output_dir) / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    # 1. Overall mAP comparison
    plt.figure(figsize=(8, 6))
    map_values = [results['mAP@0.3']['mean'], results['mAP@0.5']['mean']]
    bars = plt.bar(['mAP@0.3', 'mAP@0.5'], map_values, color=['#3498db', '#e74c3c'], alpha=0.8)
    
    plt.ylabel('Mean Average Precision')
    plt.title(f'Full Zero-Shot Detection Performance ({model_type.upper()})')
    plt.ylim(0, max(0.5, max(map_values) * 1.2))
    
    # Add value labels
    for bar, val in zip(bars, map_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(map_values) * 0.02,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(fig_dir / f'test3_full_detection_{model_type}_map.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. mAP by mapping type
    mapping_types = []
    map_03_values = []
    map_05_values = []
    
    for mapping_type in ['direct', 'supercategory']:
        categories = evaluator.stats[f'{mapping_type}_mapping']['categories']
        if categories:
            mapping_types.append(mapping_type.title())
            map_03_values.append(np.mean([results['mAP@0.3'][cat] for cat in categories]))
            map_05_values.append(np.mean([results['mAP@0.5'][cat] for cat in categories]))
    
    if mapping_types:
        plt.figure(figsize=(10, 6))
        x = np.arange(len(mapping_types))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, map_03_values, width, label='mAP@0.3', alpha=0.8, color='#3498db')
        bars2 = plt.bar(x + width/2, map_05_values, width, label='mAP@0.5', alpha=0.8, color='#e74c3c')
        
        plt.xlabel('Mapping Type')
        plt.ylabel('Mean Average Precision')
        plt.title(f'mAP by Mapping Type ({model_type.upper()})')
        plt.xticks(x, mapping_types)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                        f'{height:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'test3_full_detection_{model_type}_mapping_comparison.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {fig_dir}")

def main():
    parser = argparse.ArgumentParser(description="Test 3: Full Zero-Shot Detection")
    
    # Required files
    parser.add_argument('--coco_annotations_file', required=True,
                       help='Path to instances_train2014.json (COCO detection annotations)')
    parser.add_argument('--sam_proposals_file', required=True,
                       help='Path to SAM proposals .pth file')
    parser.add_argument('--refcoco_file', required=True,
                       help='Path to RefCOCO samples JSON (for 1772 image IDs)')
    parser.add_argument('--images_dir', required=True,
                       help='Directory containing COCO images')
    parser.add_argument('--model_path', required=True,
                       help='Path to model weights (.pth file)')
    parser.add_argument('--model_type', required=True, choices=['standard', 'focl'],
                       help='Type of model being evaluated')
    parser.add_argument('--coco_categories_file', required=True,
                       help='Path to coco_categories.json')
    parser.add_argument('--mapping_file', required=True,
                       help='Path to coco_imagenet_mapping.json')
    parser.add_argument('--imagenet_classes_file', required=True,
                       help='Path to imagenet_class_index.json')
    
    # Optional parameters
    parser.add_argument('--confidence_threshold', type=float, default=0.1,
                       help='Confidence threshold for detections (default: 0.1)')
    parser.add_argument('--nms_threshold', type=float, default=0.5,
                       help='IoU threshold for NMS (default: 0.5)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit number of images for testing')
    parser.add_argument('--output_dir', default='./test3_full_detection_results',
                       help='Output directory for results')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data files
    coco_categories, mapping_data, imagenet_classes, sam_proposals, sam_metadata = load_data_files(args)
    
    # Initialize evaluator
    evaluator = FullDetectionEvaluator(mapping_data, coco_categories, imagenet_classes)
    
    # Get target image IDs (1772 controlled set)
    target_image_ids = get_target_image_ids(args.refcoco_file)
    
    # Load COCO detection annotations
    ground_truth_by_image = load_coco_detection_annotations(
        args.coco_annotations_file, target_image_ids, evaluator
    )
    
    # Initialize detector
    detector = ZeroShotDetector(args.model_path, args.model_type, args.device)
    
    # Find common images between all data sources
    common_image_ids = (set(target_image_ids) & 
                       set(sam_proposals.keys()) & 
                       set(ground_truth_by_image.keys()))
    
    print(f"Common images for evaluation: {len(common_image_ids)}")
    
    if not common_image_ids:
        raise ValueError("No common images found between all data sources")
    
    if args.max_samples:
        common_image_ids = list(common_image_ids)[:args.max_samples]
        print(f"Limited to {args.max_samples} images for testing")
    
    # Process images and generate detections
    print(f"\nRunning full zero-shot detection on {len(common_image_ids)} images...")
    predictions_by_image = {}
    
    total_proposals = 0
    total_detections_before_nms = 0
    total_detections_after_nms = 0
    images_with_detections = 0
    
    for image_id in tqdm(common_image_ids):
        # Load image
        image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        image_path = Path(args.images_dir) / image_filename
        
        if not image_path.exists():
            continue
        
        try:
            full_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            continue
        
        # Get SAM proposals for this image
        image_sam_proposals = sam_proposals[image_id]['proposals']
        total_proposals += len(image_sam_proposals)
        
        # Classify all proposals
        detections_before_nms = detector.classify_proposals(full_image, image_sam_proposals, evaluator,confidence_threshold=args.confidence_threshold)
        total_detections_before_nms += len(detections_before_nms)
        
        # Apply NMS
        detections_after_nms = detector.apply_nms(detections_before_nms, args.nms_threshold)
        total_detections_after_nms += len(detections_after_nms)
        
        if detections_after_nms:
            predictions_by_image[image_id] = detections_after_nms
            images_with_detections += 1
    
    # Update processing stats
    evaluator.stats['processing_stats']['images_processed'] = len(common_image_ids)
    evaluator.stats['processing_stats']['images_with_detections'] = images_with_detections
    evaluator.stats['processing_stats']['avg_proposals_per_image'] = total_proposals / len(common_image_ids) if common_image_ids else 0
    evaluator.stats['processing_stats']['avg_detections_after_nms'] = total_detections_after_nms / len(common_image_ids) if common_image_ids else 0

    # ADD THIS CODE HERE:
    # Save predictions for visualization
    # Save predictions for visualization
    def convert_to_json_serializable(obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    predictions_data = {
        'predictions_by_image': convert_to_json_serializable(predictions_by_image),
        'ground_truth_by_image': convert_to_json_serializable(ground_truth_by_image)
    }

    viz_file = Path(args.output_dir) / f'predictions_{args.model_type}.json'
    with open(viz_file, 'w') as f:
        json.dump(predictions_data, f, indent=2)

    print(f"Predictions saved for visualization: {viz_file}")
    # END OF ADDED CODE
    
    print(f"\nPROCESSING SUMMARY:")
    print(f"Images processed: {len(common_image_ids)}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total SAM proposals: {total_proposals}")
    print(f"Total detections before NMS: {total_detections_before_nms}")
    print(f"Total detections after NMS: {total_detections_after_nms}")
    print(f"Average proposals per image: {total_proposals / len(common_image_ids):.1f}")
    print(f"Average detections per image: {total_detections_after_nms / len(common_image_ids):.1f}")
    
    # Evaluate detections using mAP
    print(f"\nCalculating mAP scores...")
    results = evaluate_detections(predictions_by_image, ground_truth_by_image, evaluator)
    
    # Analyze and save results
    analyze_results(evaluator, results, output_dir, args.model_type)
    
    print(f"\nTest 3 ({args.model_type}) complete!")
    print(f"Overall mAP@0.3: {results['mAP@0.3']['mean']:.4f}")
    print(f"Overall mAP@0.5: {results['mAP@0.5']['mean']:.4f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

"""
USAGE EXAMPLES:

# Standard model evaluation
python test3_full_detection.py \
    --coco_annotations_file /path/to/instances_train2014.json \
    --sam_proposals_file /path/to/sam_proposals.pth \
    --refcoco_file /path/to/refcoco_samples.json \
    --images_dir /path/to/COCO_train2014/ \
    --model_path /path/to/standard_model.pth \
    --model_type standard \
    --coco_categories_file /path/to/coco_categories.json \
    --mapping_file /path/to/coco_imagenet_mapping.json \
    --imagenet_classes_file /path/to/imagenet_class_index.json \
    --output_dir ./test3_standard_results

# FocL model evaluation with custom thresholds
python test3_full_detection.py \
    --coco_annotations_file /path/to/instances_train2014.json \
    --sam_proposals_file /path/to/sam_proposals.pth \
    --refcoco_file /path/to/refcoco_samples.json \
    --images_dir /path/to/COCO_train2014/ \
    --model_path /path/to/focl_model.pth \
    --model_type focl \
    --coco_categories_file /path/to/coco_categories.json \
    --mapping_file /path/to/coco_imagenet_mapping.json \
    --imagenet_classes_file /path/to/imagenet_class_index.json \
    --output_dir ./test3_focl_results \
    --confidence_threshold 0.05 \
    --nms_threshold 0.3 \
    --max_samples 500

REQUIRED FILES:
- instances_train2014.json: COCO detection annotations
- sam_proposals.pth: SAM proposals for target images (20-50 per image)
- refcoco_samples.json: RefCOCO samples (for 1772 target image IDs)  
- COCO_train2014/: Directory with COCO images
- model.pth: Trained ResNet-50 weights
- coco_categories.json: COCO category definitions
- coco_imagenet_mapping.json: COCO→ImageNet mapping
- imagenet_class_index.json: ImageNet class names

OUTPUTS:
- test3_full_detection_{model_type}_results.json: Detailed mAP results
- figures/test3_full_detection_{model_type}_map.png: Overall mAP visualization
- figures/test3_full_detection_{model_type}_mapping_comparison.png: mAP by mapping type
- Console output with mAP@0.3, mAP@0.5 overall and by mapping type

KEY FEATURES:
- Complete detection pipeline: SAM proposals → Classification → Confidence filtering → NMS
- mAP evaluation at IoU thresholds 0.3 and 0.5
- Single highest-confidence category per proposal (standard detection format)
- Confidence scoring using max softmax across valid ImageNet classes
- Results stratified by direct vs supercategory mappings
- Normalized mAP by evaluated categories only (accounts for vocabulary constraints)

LIMITATIONS DOCUMENTED:
- SAM proposals limited to 20-50 per image (reduces recall potential vs standard detectors)
- Vocabulary constrained to mappable COCO categories only
- Zero-shot evaluation (no task-specific training)
"""