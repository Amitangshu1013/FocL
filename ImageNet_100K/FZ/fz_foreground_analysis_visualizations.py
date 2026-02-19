"""
fz_foreground_analysis_visualizations.py

Purpose
-------
Qualitative / visualization script for the Feldman–Zhang (FZ) memorization cohort analysis.
This script is intended to generate *example images and foreground (FG) diagnostics* for the
high-memorization cohort, not to reproduce the full quantitative statistics.

It constructs the same cohort used in the paper’s memorization analysis:
    (Top ~1% FZ memorization) ∩ (Partition-A train indices),
then for a limited number of samples (capped for speed), it:
  - joins per-sample CSL values (Std vs Foc-L),
  - computes whether Foc-L improves difficulty (CSL_FocL < CSL_Std),
  - computes a simple foreground fraction from the annotated bounding box,
  - saves annotated visualization PNGs and summary CSVs.

Cohort Definition (paper-aligned)
---------------------------------
1) Load Partition-A train indices (global ImageNet train indices).
2) Load Feldman–Zhang memorization scores `tr_mem` for the ImageNet train set.
3) Select the top-K most memorized indices (K=13000 ≈ top 1% for ImageNet-1k scale).
4) Take overlap cohort: (top-K indices) ∩ (Partition-A train indices).
5) For cohort indices, load CSL values from:
      - Standard training CSL JSON
      - Foc-L training CSL JSON
   and compute:
      improvement_flag = (CSL_FocL < CSL_Std)
      csl_diff         = (CSL_Std - CSL_FocL)

Foreground (FG) Fraction
------------------------
- Parses the ImageNet XML annotation for the sample (bounding box).
- Computes:
      fg_fraction = (bbox_area / image_area) * 100
  where bbox_area = (xmax-xmin)*(ymax-ymin), image_area = W*H.

Qualitative Cap / Runtime Control
---------------------------------
- This script intentionally stops after a fixed number of processed samples
  (e.g., MAX_SAMPLES = 500) to keep runtime and storage manageable.
- Because of this cap, outputs are *qualitative* and are NOT meant to match
  the full-cohort statistics (e.g., 819/820 improvement counts).

Inputs (edit paths below)
-------------------------
- Partition A splits JSON:
    imagenet_subset_splits_partition_A.json   (expects key "train" with global indices)
- FZ memorization NPZ:
    imagenet_index.npz                        (expects key "tr_mem")
- CSL JSONs (global index -> CSL float):
    <standard_csl.json>
    <focl_csl.json>
- ImageNet dataset root (for locating images)
- ImageNet XML annotation root (for locating bbox XMLs)

Outputs
-------
- Annotated visualization images (PNG) for sampled cohort examples.
- A per-sample CSV containing (index, class name, FZ score, CSL_std, CSL_focl,
  csl_diff, improvement flag, fg_fraction, paths).
- Optional text/CSV summaries and simple correlation diagnostics (FG vs CSL / FZ).

Notes / Common pitfalls
-----------------------
- All indexing in this script assumes *global ImageNet train indices*.
  Do NOT use local indices from a Subset/DataLoader.
- K=13000 is approximately top 1% for ImageNet train size. For an exact top-1%:
      K = int(0.01 * len(tr_mem))
- This script does NOT run the Appendix A.6 binomial/sign test; that is handled
  by a separate quantitative statistics script.
"""

import json
import numpy as np
import pandas as pd
import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import datasets
from torchvision.transforms import functional as TF

# Import your bbox parsing functions
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

def main():
    print("=== FZ Memorization + Foreground Analysis ===")
    
    # Configuration
    DATASET_PATH = "/local/a/imagenet/imagenet2012/train/"
    ANNOTATION_FOLDER = "/home/min/a/mukher44/Work/SSL/Data/ImageNet_BBOX_Folder/Annotation"
    OUTPUT_DIR = "fz_foreground_analysis_TMLR_check"
    VIS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
    TARGET_SIZE = (256, 256)
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Visualizations directory: {VIS_DIR}")
    
    # 1. Load Partition A train indices
    print("\n1. Loading Partition A train indices...")
    try:
        with open("imagenet_subset_splits_partition_A.json", "r") as f:
            splits = json.load(f)
        train_indices = set(splits["train"])
        print(f"   ✓ Loaded {len(train_indices)} train indices")
    except Exception as e:
        print(f"   ✗ Error loading partition data: {e}")
        return

    # 2. Load ImageNet dataset and class names
    print("\n2. Loading ImageNet dataset and class names...")
    try:
        dataset = datasets.ImageFolder(DATASET_PATH)
        print(f"   ✓ Loaded dataset with {len(dataset.samples)} samples")
        
        # Load ImageNet class names
        with open("imagenet_class_index.json", "r") as f:
            class_idx = json.load(f)
        # Convert to dict mapping class folder names to readable names
        class_names = {}
        for idx, (class_id, class_name) in class_idx.items():
            class_names[class_id] = class_name.replace('_', ' ').title()
        print(f"   ✓ Loaded {len(class_names)} class names")
        
    except Exception as e:
        print(f"   ✗ Error loading dataset or class names: {e}")
        return

    # 3. Load Feldman–Zhang memorization scores
    print("\n3. Loading Feldman-Zhang memorization scores...")
    try:
        fz = np.load("imagenet_index.npz", allow_pickle=True)
        mem_scores = fz["tr_mem"]
        print(f"   ✓ Loaded memorization scores for {len(mem_scores)} samples")
    except Exception as e:
        print(f"   ✗ Error loading memorization data: {e}")
        return

    # 4. Get top 1000 most memorized samples and find overlap
    print("\n4. Finding overlap with Partition A...")
    top_1000_indices = np.argsort(mem_scores)[-13000:][::-1]
    
    overlap_data = []
    for i, idx in enumerate(top_1000_indices):
        if idx in train_indices:
            overlap_data.append({
                'rank_in_top1000': i + 1,
                'train_index': idx,
                'fz_memorization_score': float(mem_scores[idx])
            })
    
    print(f"   ✓ Found {len(overlap_data)} overlapping samples")
    
    if len(overlap_data) == 0:
        print("   ✗ No overlap found!")
        return

    # 5. Load CSL data
    print("\n5. Loading CSL data...")
    try:
        with open("sorted_dynamic_cumulative_loss_supervised_No_Aug.json", "r") as f:
            csl_std = {int(k): float(v) for k, v in json.load(f).items()}
        print(f"   ✓ Loaded Standard CSL for {len(csl_std)} samples")
        
        with open("sorted_fixed_cumulative_loss_dynamicbbox_supervised_No_Aug_90.json", "r") as f:
            csl_focl = {int(k): float(v) for k, v in json.load(f).items()}
        print(f"   ✓ Loaded FocL CSL for {len(csl_focl)} samples")
    except Exception as e:
        print(f"   ✗ Error loading CSL data: {e}")
        return

    # 6. Process each sample for foreground analysis
    print("\n6. Processing samples for foreground analysis...")
    
    enhanced_data = []
    successful_samples = 0
    
    for sample in overlap_data:
        train_idx = sample['train_index']
        
        # Get CSL scores
        std_csl = csl_std.get(train_idx)
        focl_csl = csl_focl.get(train_idx)
        
        if std_csl is None or focl_csl is None:
            continue
            
        # Get image path from dataset
        try:
            image_path, label = dataset.samples[train_idx]
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            class_folder = os.path.basename(os.path.dirname(image_path))
            xml_file = os.path.join(ANNOTATION_FOLDER, class_folder, f"{image_id}.xml")
            
            # Load original image
            original_image = Image.open(image_path).convert("RGB")
            orig_w, orig_h = original_image.size
            total_area = orig_w * orig_h
            
            # Parse bounding box
            if not os.path.exists(xml_file):
                print(f"   ⚠ XML not found for {image_id}")
                continue
                
            bboxes = parse_xml_for_bbox(xml_file)
            if not bboxes:
                print(f"   ⚠ No valid bbox for {image_id}")
                continue
                
            bbox = bboxes[0]  # Use first bbox
            xmin, ymin, xmax, ymax = bbox
            
            # Validate and clamp bbox coordinates to image boundaries
            xmin = max(0, min(xmin, orig_w - 1))
            ymin = max(0, min(ymin, orig_h - 1))
            xmax = max(xmin + 1, min(xmax, orig_w))
            ymax = max(ymin + 1, min(ymax, orig_h))
            
            # Recalculate with validated coordinates
            validated_bbox = [xmin, ymin, xmax, ymax]
            foreground_area = (xmax - xmin) * (ymax - ymin)
            fg_fraction = foreground_area / total_area
            
            # Sanity check - should never be > 1.0 now
            if fg_fraction > 1.0:
                print(f"   ⚠ WARNING: {image_id} still has fg_fraction > 1.0: {fg_fraction:.4f}")
                print(f"      Image size: {orig_w}x{orig_h}, Bbox: {validated_bbox}")
                continue
            
            # Get class name from ImageNet class index
            class_folder = os.path.basename(os.path.dirname(image_path))
            class_name = class_names.get(class_folder, class_folder.replace('_', ' ').title())
            
            # Create 256x256 visualization
            resized_image = original_image.resize(TARGET_SIZE)
            rescaled_bbox = rescale_bbox(validated_bbox, (orig_w, orig_h), TARGET_SIZE)
            
            # Draw bbox and add text overlays
            draw = ImageDraw.Draw(resized_image)
            draw.rectangle(rescaled_bbox, outline='red', width=3)  # Slightly thicker bbox
            
            # Setup fonts
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            except:
                title_font = ImageFont.load_default()
                label_font = ImageFont.load_default()
            
            # Add FG ratio at top-left
            ratio_text = f"FG: {fg_fraction:.3f}"
            ratio_bbox = draw.textbbox((0, 0), ratio_text, font=title_font)
            ratio_width = ratio_bbox[2] - ratio_bbox[0]
            ratio_height = ratio_bbox[3] - ratio_bbox[1]
            
            # Background rectangle for ratio text
            padding = 3
            draw.rectangle([5-padding, 5-padding, 5+ratio_width+padding, 5+ratio_height+padding], 
                         fill='black', outline='white', width=1)
            draw.text((5, 5), ratio_text, fill='white', font=title_font)
            
            # Add FZ memorization score at top-right
            fz_score = sample['fz_memorization_score']
            fz_text = f"FZ: {fz_score:.4f}"
            fz_bbox = draw.textbbox((0, 0), fz_text, font=title_font)
            fz_width = fz_bbox[2] - fz_bbox[0]
            fz_height = fz_bbox[3] - fz_bbox[1]
            
            # Position at top-right
            fz_x = TARGET_SIZE[0] - fz_width - 5
            # Background rectangle for FZ text
            draw.rectangle([fz_x-padding, 5-padding, fz_x+fz_width+padding, 5+fz_height+padding], 
                         fill='black', outline='white', width=1)
            draw.text((fz_x, 5), fz_text, fill='white', font=title_font)
            
            # Add class name at bottom-left
            name_text = class_name
            name_bbox = draw.textbbox((0, 0), name_text, font=label_font)
            name_width = name_bbox[2] - name_bbox[0]
            name_height = name_bbox[3] - name_bbox[1]
            
            # Position at bottom-left
            name_y = TARGET_SIZE[1] - name_height - 5
            # Background rectangle for class name
            draw.rectangle([5-padding, name_y-padding, 5+name_width+padding, name_y+name_height+padding], 
                         fill='black', outline='white', width=1)
            draw.text((5, name_y), name_text, fill='white', font=label_font)
            
            # Save visualization
            vis_filename = f"{successful_samples+1:03d}_{image_id}_fg{fg_fraction:.3f}.png"
            vis_path = os.path.join(VIS_DIR, vis_filename)
            resized_image.save(vis_path)
            
            # Add to enhanced data
            enhanced_data.append({
                'rank_in_top1000': sample['rank_in_top1000'],
                'train_index': train_idx,
                'image_id': image_id,
                'class_name': class_name,
                'fz_memorization_score': sample['fz_memorization_score'],
                'standard_csl': std_csl,
                'focl_csl': focl_csl,
                'csl_difference': std_csl - focl_csl,
                'percent_reduction': ((std_csl - focl_csl) / std_csl * 100) if std_csl > 0 else 0,
                'focl_improved': 'Yes' if focl_csl < std_csl else 'No',
                'original_image_size': f"{orig_w}x{orig_h}",
                'bbox_coordinates': f"{validated_bbox[0]},{validated_bbox[1]},{validated_bbox[2]},{validated_bbox[3]}",
                'foreground_area': foreground_area,
                'total_image_area': total_area,
                'fg_fraction': fg_fraction,
                'visualization_file': os.path.join("visualizations", vis_filename)
            })
            
            successful_samples += 1
            
            if successful_samples >= 500:  # Stop at 100 samples
                break
                
        except Exception as e:
            print(f"   ⚠ Error processing train_idx {train_idx}: {e}")
            continue
    
    print(f"   ✓ Successfully processed {len(enhanced_data)} samples")
    
    # 7. Create DataFrame and analysis
    print("\n7. Creating analysis...")
    df = pd.DataFrame(enhanced_data)
    
    if len(df) == 0:
        print("   ✗ No samples with complete data!")
        return
    
    # Summary statistics
    print(f"\nSummary Statistics (n={len(df)}):")
    print(f"   Mean FZ Score: {df['fz_memorization_score'].mean():.6f}")
    print(f"   Mean Standard CSL: {df['standard_csl'].mean():.2f}")
    print(f"   Mean FocL CSL: {df['focl_csl'].mean():.2f}")
    print(f"   Mean FG Fraction: {df['fg_fraction'].mean():.4f}")
    print(f"   FG Fraction Range: {df['fg_fraction'].min():.4f} to {df['fg_fraction'].max():.4f}")
    print(f"   Samples where FocL improved: {(df['focl_improved'] == 'Yes').sum()}/{len(df)} ({(df['focl_improved'] == 'Yes').mean()*100:.1f}%)")
    
    # Correlation analysis
    try:
        from scipy.stats import pearsonr, spearmanr
        
        # Correlation between fg_fraction and CSL improvement
        csl_improvement = df['csl_difference']  # Positive = FocL better
        pearson_corr, pearson_p = pearsonr(df['fg_fraction'], csl_improvement)
        spearman_corr, spearman_p = spearmanr(df['fg_fraction'], csl_improvement)
        
        print(f"\nCorrelation Analysis:")
        print(f"   FG Fraction vs CSL Improvement:")
        print(f"     Pearson r: {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"     Spearman ρ: {spearman_corr:.4f} (p={spearman_p:.4f})")
        
        # Interpretation
        if abs(pearson_corr) > 0.3 and pearson_p < 0.05:
            direction = "more" if pearson_corr > 0 else "less"
            print(f"     → FocL helps {direction} when foreground fraction is higher")
        
    except ImportError:
        print("   (scipy not available for correlation analysis)")
    
    # 8. Save results
    print(f"\n8. Saving results...")
    
    # Detailed CSV
    detailed_csv = os.path.join(OUTPUT_DIR, "fz_top100_with_foreground_analysis.csv")
    df.to_csv(detailed_csv, index=False)
    print(f"   ✓ Detailed CSV: {detailed_csv}")
    
    # Summary CSV (key columns only)
    summary_cols = ['train_index', 'fz_memorization_score', 'standard_csl', 'focl_csl', 'fg_fraction']
    summary_csv = os.path.join(OUTPUT_DIR, "fz_top100_foreground_summary.csv")
    df[summary_cols].to_csv(summary_csv, index=False)
    print(f"   ✓ Summary CSV: {summary_csv}")
    
    # Analysis report
    report_file = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_file, 'w') as f:
        f.write("FZ Memorization + Foreground Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples analyzed: {len(df)}\n")
        f.write(f"Mean foreground fraction: {df['fg_fraction'].mean():.4f}\n")
        f.write(f"FocL improvement rate: {(df['focl_improved'] == 'Yes').mean()*100:.1f}%\n")
        f.write(f"Mean CSL reduction: {df['csl_difference'].mean():.2f}\n")
        if 'pearson_corr' in locals():
            f.write(f"FG fraction vs CSL improvement correlation: {pearson_corr:.4f} (p={pearson_p:.4f})\n")
    
    print(f"   ✓ Analysis report: {report_file}")
    print(f"   ✓ Visualizations saved to: {VIS_DIR}/ (100 PNG files)")
    
    print(f"\n=== Analysis Complete! ===")
    print(f"Check the {OUTPUT_DIR}/ directory for:")
    print(f"  - CSV files with foreground analysis data")
    print(f"  - visualizations/ folder with 100 images (256x256 with bboxes)")
    print(f"  - Analysis report with correlations")

if __name__ == "__main__":
    main()