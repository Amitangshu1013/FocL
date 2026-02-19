#!/usr/bin/env python3
"""
ImageNet-v2 Standard Model Evaluation
Evaluates standard classifier on ImageNet-v2 matched-frequency dataset
Condition 1: Standard Model (Full Image) - Baseline
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms, models
import argparse

# ─── REPRODUCIBILITY ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_classifier(model_path, num_classes, device):
    """Load classifier model"""
    if model_path and os.path.exists(model_path):
        model = models.resnet50(pretrained=False, num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded classifier from: {model_path}")
    else:
        print("Using pretrained ImageNet ResNet50")
        model = models.resnet50(pretrained=True, num_classes=num_classes).to(device)
    return model.eval()

def main():
    parser = argparse.ArgumentParser(description="ImageNet-v2 Standard Model Evaluation")
    parser.add_argument('--imagenet_v2_path', type=str, required=True, 
                       help='Path to ImageNet-v2 dataset (e.g., /path/to/imagenetv2-matched-frequency-format-val)')
    parser.add_argument('--model_path', type=str, default='', 
                       help='Path to classifier model (empty for pretrained)')
    parser.add_argument('--model_type', type=str, choices=['standard', 'focl'], default='standard', 
                       help='Model type for output naming')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for inference')
    args = parser.parse_args()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load classifier
    classifier = load_classifier(args.model_path, num_classes=1000, device=device)
    
    # Standard ImageNet transforms for evaluation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load ImageNet-v2 dataset
    print(f"Loading ImageNet-v2 dataset from: {args.imagenet_v2_path}")
    
    # Validate path exists
    if not os.path.exists(args.imagenet_v2_path):
        raise FileNotFoundError(f"Dataset path does not exist: {args.imagenet_v2_path}")
    
    dataset = datasets.ImageFolder(root=args.imagenet_v2_path, transform=transform)
    print(f"Total samples in ImageNet-v2: {len(dataset)}")
    print(f"Number of classes found: {len(dataset.classes)}")
    
    # Validate dataset structure
    print("\nDataset structure validation:")
    print(f"First 10 classes: {dataset.classes[:10]}")
    print(f"Last 10 classes: {dataset.classes[-10:]}")
    
    # Check if classes are numeric (0, 1, 2, ... 999)
    expected_classes = [str(i) for i in range(1000)]
    if dataset.classes != expected_classes:
        print("WARNING: Class structure doesn't match expected ImageNet-v2 format!")
        print(f"Expected: ['0', '1', '2', ..., '999']")
        print(f"Found: {len(dataset.classes)} classes")
        print("ISSUE: Alphabetical sorting problem detected!")
        
        # Create mapping from alphabetical index to correct numerical label
        class_to_correct_label = {}
        for alphabetical_idx, class_name in enumerate(dataset.classes):
            correct_label = int(class_name)  # Convert '10' -> 10
            class_to_correct_label[alphabetical_idx] = correct_label
        
        print(f"Created label mapping for {len(class_to_correct_label)} classes")
        print(f"Example mappings: {dict(list(class_to_correct_label.items())[:10])}")
    else:
        class_to_correct_label = None
        print("Class structure is correct!")
    
    # Show samples per class
    class_counts = {}
    for _, class_idx in dataset.samples:
        class_name = dataset.classes[class_idx]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    sample_classes = list(class_counts.keys())[:5]
    print(f"Samples per class (first 5): {[(cls, class_counts[cls]) for cls in sample_classes]}")
    
    # Generate 2K sample indices with fixed seed (same as other experiments)
    total_samples = len(dataset)
    sample_size = min(2000, total_samples)
    all_indices = np.random.choice(total_samples, size=sample_size, replace=False)
    print(f"Selected {len(all_indices)} samples for evaluation (seed={SEED})")
    
    # Create subset dataset
    subset_dataset = torch.utils.data.Subset(dataset, all_indices)
    dataloader = torch.utils.data.DataLoader(
        subset_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluation metrics
    correct_predictions = 0
    total_samples = 0
    all_confidences = []
    correct_confidences = []
    
    print(f"\nStarting evaluation on {len(subset_dataset)} ImageNet-v2 samples...")
    print("=" * 70)
    
    # Evaluation loop
    batch_predictions = []
    batch_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(dataloader):
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {(batch_idx + 1) * args.batch_size} / {len(subset_dataset)} samples")
            
            # Move to device
            images = images.to(device)
            targets = targets.to(device)
            
            # Fix label mapping if needed
            if class_to_correct_label is not None:
                # Convert alphabetical indices to correct numerical labels
                corrected_targets = torch.tensor([class_to_correct_label[t.item()] for t in targets], 
                                                device=device)
                targets = corrected_targets
            
            # Forward pass
            outputs = classifier(images)
            probs = F.softmax(outputs, dim=1)
            confidences, predictions = probs.max(dim=1)
            
            # Debug: Print first batch details
            if batch_idx == 0:
                print(f"\nDEBUG - First batch:")
                print(f"  Batch size: {targets.size(0)}")
                print(f"  Target range: {targets.min().item()} to {targets.max().item()}")
                print(f"  Prediction range: {predictions.min().item()} to {predictions.max().item()}")
                print(f"  First 5 targets: {targets[:5].cpu().numpy()}")
                print(f"  First 5 predictions: {predictions[:5].cpu().numpy()}")
                print(f"  First 5 matches: {(predictions[:5] == targets[:5]).cpu().numpy()}")
                if class_to_correct_label is not None:
                    print(f"  Label mapping applied: True")
            
            # Update metrics
            batch_correct = (predictions == targets)
            correct_predictions += batch_correct.sum().item()
            total_samples += targets.size(0)
            
            # Store for detailed analysis
            batch_predictions.extend(predictions.cpu().numpy())
            batch_targets.extend(targets.cpu().numpy())
            
            # Store confidences
            all_confidences.extend(confidences.cpu().numpy())
            correct_confidences.extend(confidences[batch_correct].cpu().numpy())
    
    # Additional validation
    print(f"\nValidation checks:")
    print(f"Total samples processed: {total_samples}")
    print(f"Predictions collected: {len(batch_predictions)}")
    print(f"Targets collected: {len(batch_targets)}")
    
    # Manual accuracy calculation for verification
    manual_correct = sum(p == t for p, t in zip(batch_predictions, batch_targets))
    manual_accuracy = 100.0 * manual_correct / len(batch_predictions)
    print(f"Manual accuracy verification: {manual_accuracy:.2f}% ({manual_correct}/{len(batch_predictions)})")
    
    # Calculate final metrics
    accuracy = 100.0 * correct_predictions / total_samples
    mean_confidence = np.mean(all_confidences)
    std_confidence = np.std(all_confidences)
    
    # Confidence for correct predictions only
    if correct_confidences:
        mean_correct_conf = np.mean(correct_confidences)
        std_correct_conf = np.std(correct_confidences)
    else:
        mean_correct_conf = 0.0
        std_correct_conf = 0.0
    
    # Print results
    print(f"\n{'='*80}")
    print(f"IMAGENET-V2 EVALUATION RESULTS ({args.model_type.upper()} MODEL)")
    print(f"{'='*80}")
    print(f"Evaluation Condition: Standard Model (Full Image)")
    print(f"Dataset: ImageNet-v2 matched-frequency")
    print(f"Total samples evaluated: {total_samples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Top-1 Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_samples})")
    print(f"{'='*80}")
    print(f"CONFIDENCE ANALYSIS:")
    print(f"{'='*80}")
    print(f"All predictions confidence:     {mean_confidence:.4f}±{std_confidence:.4f}")
    print(f"Correct predictions confidence: {mean_correct_conf:.4f}±{std_correct_conf:.4f}")
    print(f"Samples with correct predictions: {len(correct_confidences)}")
    
    # Additional statistics
    confidence_diff = mean_correct_conf - mean_confidence
    print(f"Correct vs All confidence diff: {confidence_diff:+.4f}")
    if abs(confidence_diff) > 0.01:
        print(f"  → Model is {'more' if confidence_diff > 0 else 'less'} confident on correct predictions")
    else:
        print(f"  → Similar confidence for correct and incorrect predictions")
    
    # Save results
    results_data = {
        "model_type": args.model_type,
        "model_path": args.model_path,
        "dataset": "imagenet_v2_matched_frequency",
        "evaluation_condition": "standard_full_image",
        "total_samples": total_samples,
        "correct_predictions": correct_predictions,
        "accuracy": accuracy,
        "confidence_stats": {
            "all_predictions": {
                "mean": mean_confidence,
                "std": std_confidence,
                "n_samples": total_samples
            },
            "correct_predictions": {
                "mean": mean_correct_conf,
                "std": std_correct_conf,
                "n_samples": len(correct_confidences)
            }
        },
        "sample_indices": all_indices.tolist(),  # For reproducibility
        "raw_confidences": {
            "all": all_confidences,
            "correct": correct_confidences
        }
    }
    
    # Create output filename
    output_filename = f"imagenet_v2_eval_{args.model_type}_standard_condition.pth"
    torch.save(results_data, output_filename)
    print(f"\nResults saved to: {output_filename}")
    
    # Summary for comparison
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"{'='*60}")
    print(f"Condition: Standard Model (Full Image)")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Confidence: {mean_confidence:.3f}")
    print(f"Correct Confidence: {mean_correct_conf:.3f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()