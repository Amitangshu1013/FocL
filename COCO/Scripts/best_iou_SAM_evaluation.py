#!/usr/bin/env python3
"""
Best IoU SAM Proposal Evaluation
Finds the SAM proposal with highest IoU overlap with GT bbox, classifies it,
and compares performance to Experiment 2 (GT bbox) results.
This tests whether individual SAM proposals are better/worse than GT crops.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from collections import defaultdict
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
    """
    Calculate IoU between two bounding boxes in xywh format
    """
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

class ZeroShotEvaluator:
    def __init__(self, mapping_data, coco_categories, imagenet_classes):
        self.mapping_data = mapping_data
        self.coco_categories = coco_categories
        self.imagenet_classes = imagenet_classes
        
        # Build ImageNet class name lookup
        self.imagenet_idx_to_name = {int(k): v[1] for k, v in imagenet_classes.items()}
        self.imagenet_name_to_idx = {v[1]: int(k) for k, v in imagenet_classes.items()}
        
        # Build COCO category lookup
        self.coco_name_to_info = {cat['name']: cat for cat in coco_categories}
        
        # Build reverse mapping: ImageNet → COCO
        print("Building ImageNet → COCO reverse mapping...")
        self.reverse_mapping = self.build_reverse_mapping()
        print(f"Reverse mapping covers {len(self.reverse_mapping)} ImageNet classes")
        
        # Initialize statistics
        self.reset_stats()
    
    def build_reverse_mapping(self):
        """Build ImageNet class → COCO categories reverse mapping"""
        imagenet_to_coco = {}
        
        # Process direct mappings
        for coco_category, imagenet_classes in self.mapping_data["direct_mapping"].items():
            for imagenet_class in imagenet_classes:
                if imagenet_class not in imagenet_to_coco:
                    imagenet_to_coco[imagenet_class] = []
                imagenet_to_coco[imagenet_class].append(coco_category)
        
        # Process supercategory mappings
        for supercategory, imagenet_classes in self.mapping_data["supercategory_mapping"].items():
            # Get all COCO categories that belong to this supercategory
            coco_categories_in_super = [
                cat['name'] for cat in self.coco_categories 
                if cat['supercategory'] == supercategory
            ]
            
            for imagenet_class in imagenet_classes:
                if imagenet_class not in imagenet_to_coco:
                    imagenet_to_coco[imagenet_class] = []
                imagenet_to_coco[imagenet_class].extend(coco_categories_in_super)
        
        # Remove duplicates and sort
        for imagenet_class in imagenet_to_coco:
            imagenet_to_coco[imagenet_class] = sorted(list(set(imagenet_to_coco[imagenet_class])))
        
        return imagenet_to_coco
    
    def reset_stats(self):
        """Reset evaluation statistics"""
        self.stats = {
            'total_samples': 0,
            'correct_top1': 0,
            'correct_top5': 0,
            'category_stats': defaultdict(lambda: {'total': 0, 'correct_top1': 0, 'correct_top5': 0}),
            'mapping_type_stats': defaultdict(lambda: {'total': 0, 'correct_top1': 0, 'correct_top5': 0}),
            'iou_stats': {
                'ious': [],
                'mean_iou': 0.0,
                'median_iou': 0.0,
                'iou_thresholds': {0.1: 0, 0.3: 0, 0.5: 0, 0.7: 0}
            },
            'localization_quality': {
                'samples_with_no_proposals': 0,
                'samples_with_zero_iou': 0
            }
        }
    
    def check_accuracy(self, imagenet_pred, coco_gt):
        """Check if ImageNet prediction is valid for COCO ground truth"""
        valid_coco_cats = self.reverse_mapping.get(imagenet_pred, [])
        return coco_gt in valid_coco_cats
    
    def get_mapping_type(self, coco_category):
        """Determine which mapping type is used for this COCO category"""
        direct_mappings = self.mapping_data["direct_mapping"].get(coco_category, [])
        if direct_mappings:
            return "direct"
        
        # Check if supercategory mapping exists
        supercategory = self.coco_name_to_info[coco_category]['supercategory']
        super_mappings = self.mapping_data["supercategory_mapping"].get(supercategory, [])
        if super_mappings:
            return "supercategory"
        
        return "none"
    
    def evaluate_sample(self, predictions_top5, ground_truth, best_iou):
        """Evaluate a single sample"""
        # Check top-1 accuracy
        top1_pred = predictions_top5[0]
        correct_top1 = self.check_accuracy(top1_pred, ground_truth)
        
        # Check top-5 accuracy
        correct_top5 = any(self.check_accuracy(pred, ground_truth) for pred in predictions_top5)
        
        # Update statistics
        mapping_type = self.get_mapping_type(ground_truth)
        
        result = {
            'gt_category': ground_truth,
            'top1_pred': top1_pred,
            'top5_preds': predictions_top5,
            'correct_top1': correct_top1,
            'correct_top5': correct_top5,
            'mapping_type': mapping_type,
            'best_iou': best_iou
        }
        
        # Update running stats
        self.stats['total_samples'] += 1
        if correct_top1:
            self.stats['correct_top1'] += 1
        if correct_top5:
            self.stats['correct_top5'] += 1
        
        # Update mapping type stats
        self.stats['mapping_type_stats'][mapping_type]['total'] += 1
        if correct_top1:
            self.stats['mapping_type_stats'][mapping_type]['correct_top1'] += 1
        if correct_top5:
            self.stats['mapping_type_stats'][mapping_type]['correct_top5'] += 1
        
        # Update category stats
        self.stats['category_stats'][ground_truth]['total'] += 1
        if correct_top1:
            self.stats['category_stats'][ground_truth]['correct_top1'] += 1
        if correct_top5:
            self.stats['category_stats'][ground_truth]['correct_top5'] += 1
        
        # Update IoU stats
        self.stats['iou_stats']['ious'].append(best_iou)
        for threshold in self.stats['iou_stats']['iou_thresholds']:
            if best_iou >= threshold:
                self.stats['iou_stats']['iou_thresholds'][threshold] += 1
        
        if best_iou == 0.0:
            self.stats['localization_quality']['samples_with_zero_iou'] += 1
        
        return result

class BestIoUSAMClassifier:
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
        
        # Crop the region (PIL uses left, top, right, bottom)
        crop_box = (x, y, x + w, y + h)
        cropped_image = image.crop(crop_box)
        
        return cropped_image
    
    def find_best_iou_proposal(self, sam_proposals, gt_bbox):
        """Find SAM proposal with highest IoU overlap with ground truth"""
        if not sam_proposals:
            return None, 0.0
        
        best_proposal = None
        best_iou = 0.0
        
        for proposal in sam_proposals:
            proposal_bbox = proposal['bbox_xywh']
            iou = calculate_iou(gt_bbox, proposal_bbox)
            
            if iou > best_iou:
                best_iou = iou
                best_proposal = proposal
        
        return best_proposal, best_iou
    
    def predict_single(self, image, imagenet_idx_to_name, top_k=5):
        """Predict top-k classes for a single image"""
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=1)
        
        # Convert indices to class names
        pred_names = [imagenet_idx_to_name[idx.item()] 
                     for idx in top_k_indices[0]]
        
        return pred_names

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
    
    # Load RefCOCO samples
    with open(args.refcoco_file, 'r') as f:
        refcoco_samples = json.load(f)
    
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
    print(f"Loaded {len(refcoco_samples)} RefCOCO samples")
    print(f"Loaded SAM proposals for {len(sam_proposals)} images")
    
    return coco_categories, mapping_data, imagenet_classes, refcoco_samples, sam_proposals, sam_metadata

def find_common_samples(refcoco_samples, sam_proposals, images_dir):
    """Find samples that exist in both RefCOCO and SAM data with valid images"""
    # Group RefCOCO samples by image_id
    refcoco_by_image = defaultdict(list)
    for sample in refcoco_samples:
        refcoco_by_image[sample['image_id']].append(sample)
    
    # Find common image IDs
    common_image_ids = set(refcoco_by_image.keys()) & set(sam_proposals.keys())
    print(f"Common images between RefCOCO and SAM: {len(common_image_ids)}")
    
    # Build final sample list with image path validation
    valid_samples = []
    
    for image_id in common_image_ids:
        # Check if image file exists
        image_filename = f"COCO_train2014_{image_id:012d}.jpg"
        image_path = Path(images_dir) / image_filename
        
        if not image_path.exists():
            continue
        
        # Add all RefCOCO samples for this image
        for refcoco_sample in refcoco_by_image[image_id]:
            sample = {
                'image_id': image_id,
                'image_path': str(image_path),
                'category_name': refcoco_sample['category_name'],
                'gt_bbox': refcoco_sample['bbox'],
                'sam_proposals': sam_proposals[image_id]['proposals'],
                'refcoco_sample': refcoco_sample
            }
            valid_samples.append(sample)
    
    print(f"Valid samples with images and SAM proposals: {len(valid_samples)}")
    return valid_samples

def analyze_results(evaluator, detailed_results, output_dir, model_type):
    """Analyze and save results"""
    stats = evaluator.stats
    
    print("\n" + "="*80)
    print(f"BEST IoU SAM PROPOSAL RESULTS ({model_type.upper()} MODEL)")
    print("="*80)
    
    # Overall accuracy
    total = stats['total_samples']
    top1_acc = 100 * stats['correct_top1'] / total if total > 0 else 0
    top5_acc = 100 * stats['correct_top5'] / total if total > 0 else 0
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy: {stats['correct_top1']}/{total} ({top1_acc:.2f}%)")
    print(f"Top-5 Accuracy: {stats['correct_top5']}/{total} ({top5_acc:.2f}%)")
    
    # IoU statistics
    if stats['iou_stats']['ious']:
        ious = stats['iou_stats']['ious']
        mean_iou = np.mean(ious)
        median_iou = np.median(ious)
        std_iou = np.std(ious)
        
        print(f"\nLOCALIZATION QUALITY (IoU with GT):")
        print(f"Mean IoU: {mean_iou:.3f}")
        print(f"Median IoU: {median_iou:.3f}")
        print(f"Std IoU: {std_iou:.3f}")
        print(f"Min IoU: {min(ious):.3f}")
        print(f"Max IoU: {max(ious):.3f}")
        
        print(f"\nCOVERAGE BY IoU THRESHOLD:")
        for threshold, count in stats['iou_stats']['iou_thresholds'].items():
            percentage = 100 * count / total
            print(f"IoU >= {threshold}: {count}/{total} ({percentage:.1f}%)")
        
        print(f"\nLOCALIZATION FAILURES:")
        print(f"Samples with zero IoU: {stats['localization_quality']['samples_with_zero_iou']}/{total} ({100*stats['localization_quality']['samples_with_zero_iou']/total:.1f}%)")
    
    # Performance by mapping type
    print(f"\nPERFORMANCE BY MAPPING TYPE:")
    for mapping_type, type_stats in stats['mapping_type_stats'].items():
        if type_stats['total'] > 0:
            top1_pct = 100 * type_stats['correct_top1'] / type_stats['total']
            top5_pct = 100 * type_stats['correct_top5'] / type_stats['total']
            print(f"{mapping_type.upper()}: {type_stats['total']} samples")
            print(f"  Top-1: {type_stats['correct_top1']}/{type_stats['total']} ({top1_pct:.1f}%)")
            print(f"  Top-5: {type_stats['correct_top5']}/{type_stats['total']} ({top5_pct:.1f}%)")
    
    # Performance vs IoU correlation
    if stats['iou_stats']['ious']:
        # Bin samples by IoU ranges
        low_iou = [r for r, iou in zip(detailed_results, stats['iou_stats']['ious']) if iou < 0.3]
        med_iou = [r for r, iou in zip(detailed_results, stats['iou_stats']['ious']) if 0.3 <= iou < 0.7]
        high_iou = [r for r, iou in zip(detailed_results, stats['iou_stats']['ious']) if iou >= 0.7]
        
        print(f"\nPERFORMANCE vs IoU QUALITY:")
        for name, samples in [("Low IoU (<0.3)", low_iou), ("Med IoU (0.3-0.7)", med_iou), ("High IoU (≥0.7)", high_iou)]:
            if samples:
                correct = sum(1 for s in samples if s['correct_top1'])
                total = len(samples)
                acc = 100 * correct / total
                print(f"{name}: {correct}/{total} ({acc:.1f}%)")
    
    # Save detailed results
    results_file = Path(output_dir) / f'best_iou_sam_{model_type}_detailed_results.json'
    
    # Convert numpy types to native Python types for JSON serialization
    json_safe_results = []
    for result in detailed_results:
        json_result = {}
        for key, value in result.items():
            if isinstance(value, (np.integer, np.int64)):
                json_result[key] = int(value)
            elif isinstance(value, (np.floating, np.float64)):
                json_result[key] = float(value)
            elif isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            else:
                json_result[key] = value
        json_safe_results.append(json_result)
    
    with open(results_file, 'w') as f:
        json.dump({
            'model_type': model_type,
            'experiment': 'best_iou_sam_proposal',
            'summary': {
                'total_samples': total,
                'top1_accuracy': top1_acc,
                'top5_accuracy': top5_acc,
                'mean_iou': float(np.mean(stats['iou_stats']['ious'])) if stats['iou_stats']['ious'] else 0,
                'median_iou': float(np.median(stats['iou_stats']['ious'])) if stats['iou_stats']['ious'] else 0,
                'iou_coverage': {f'above_{t}': 100*c/total for t, c in stats['iou_stats']['iou_thresholds'].items()},
                'mapping_type_performance': {
                    mt: {
                        'total': ts['total'],
                        'top1_accuracy': 100 * ts['correct_top1'] / ts['total'] if ts['total'] > 0 else 0,
                        'top5_accuracy': 100 * ts['correct_top5'] / ts['total'] if ts['total'] > 0 else 0
                    } for mt, ts in stats['mapping_type_stats'].items()
                }
            },
            'detailed_results': json_safe_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create visualizations
    create_visualizations(evaluator, output_dir, model_type)
    
    return top1_acc, top5_acc

def create_visualizations(evaluator, output_dir, model_type):
    """Create visualization plots"""
    stats = evaluator.stats
    fig_dir = Path(output_dir) / 'figures'
    fig_dir.mkdir(exist_ok=True)
    
    # 1. Accuracy vs IoU scatter plot
    if stats['iou_stats']['ious']:
        plt.figure(figsize=(10, 6))
        
        # Create binary accuracy array (1 for correct, 0 for incorrect)
        accuracies = [1 if res['correct_top1'] else 0 for res in evaluator.detailed_results]
        ious = stats['iou_stats']['ious']
        
        # Add jitter to y-axis for better visualization
        jittered_accs = [acc + np.random.normal(0, 0.02) for acc in accuracies]
        
        plt.scatter(ious, jittered_accs, alpha=0.6, s=20)
        plt.xlabel('Best SAM Proposal IoU with GT')
        plt.ylabel('Classification Correct (1) / Incorrect (0)')
        plt.title(f'Classification Performance vs Localization Quality ({model_type.upper()})')
        plt.ylim(-0.1, 1.1)
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(ious, accuracies, 1)
        p = np.poly1d(z)
        plt.plot(np.unique(ious), p(np.unique(ious)), "r--", alpha=0.8, 
                label=f'Trend (slope: {z[0]:.3f})')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(fig_dir / f'best_iou_sam_{model_type}_acc_vs_iou.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. IoU distribution histogram
    if stats['iou_stats']['ious']:
        plt.figure(figsize=(10, 6))
        plt.hist(stats['iou_stats']['ious'], bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(np.mean(stats['iou_stats']['ious']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(stats["iou_stats"]["ious"]):.3f}')
        plt.axvline(np.median(stats['iou_stats']['ious']), color='orange', linestyle='--',
                   label=f'Median: {np.median(stats["iou_stats"]["ious"]):.3f}')
        plt.xlabel('Best SAM Proposal IoU with GT')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Best SAM Proposal IoU ({model_type.upper()})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / f'best_iou_sam_{model_type}_iou_distribution.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {fig_dir}")

def main():
    parser = argparse.ArgumentParser(description="Best IoU SAM Proposal Evaluation")
    
    # Required files
    parser.add_argument('--refcoco_file', required=True,
                       help='Path to RefCOCO samples JSON file')
    parser.add_argument('--sam_proposals_file', required=True,
                       help='Path to SAM proposals .pth file')
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
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit number of samples for testing')
    parser.add_argument('--output_dir', default='./best_iou_sam_results',
                       help='Output directory for results')
    parser.add_argument('--device', default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data files
    coco_categories, mapping_data, imagenet_classes, refcoco_samples, sam_proposals, sam_metadata = load_data_files(args)
    
    # Initialize evaluator
    evaluator = ZeroShotEvaluator(mapping_data, coco_categories, imagenet_classes)
    
    # Initialize classifier
    classifier = BestIoUSAMClassifier(args.model_path, args.model_type, args.device)
    
    # Find common samples
    valid_samples = find_common_samples(refcoco_samples, sam_proposals, args.images_dir)
    
    if not valid_samples:
        raise ValueError("No valid samples found with both RefCOCO annotations and SAM proposals")
    
    if args.max_samples:
        valid_samples = valid_samples[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for testing")
    
    # Run evaluation
    print(f"\nRunning Best IoU SAM evaluation on {len(valid_samples)} samples...")
    all_results = []
    
    # Process each sample
    for sample in tqdm(valid_samples):
        try:
            # Load image
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Find best IoU SAM proposal
            best_proposal, best_iou = classifier.find_best_iou_proposal(
                sample['sam_proposals'], 
                sample['gt_bbox']
            )
            
            if best_proposal is None:
                evaluator.stats['localization_quality']['samples_with_no_proposals'] += 1
                continue
            
            # Crop the best SAM proposal
            cropped_image = classifier.crop_bbox_from_image(
                image, 
                best_proposal['bbox_xywh']
            )
            
            # Classify the cropped image
            predictions = classifier.predict_single(
                cropped_image, 
                evaluator.imagenet_idx_to_name, 
                top_k=5
            )
            
            # Evaluate the result
            result = evaluator.evaluate_sample(
                predictions, 
                sample['category_name'],
                best_iou
            )
            
            # Add sample metadata
            result.update({
                'image_id': int(sample['image_id']),  # Ensure int, not numpy int64
                'best_proposal': {
                    'bbox_xywh': [float(x) for x in best_proposal['bbox_xywh']]  # Convert to regular floats
                },
                'gt_bbox': [float(x) for x in sample['gt_bbox']]  # Convert to regular floats
            })
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Error processing sample {sample.get('image_id', 'unknown')}: {e}")
            continue
    
    # Store detailed results for visualization
    evaluator.detailed_results = all_results
    
    # Analyze results
    top1_accuracy, top5_accuracy = analyze_results(evaluator, all_results, output_dir, args.model_type)
    
    print(f"\nBest IoU SAM Evaluation ({args.model_type}) complete!")
    print(f"Final Results: Top-1: {top1_accuracy:.2f}%, Top-5: {top5_accuracy:.2f}%")
    print(f"Mean IoU with GT: {np.mean(evaluator.stats['iou_stats']['ious']):.3f}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()