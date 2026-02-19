"""
FZ_CSL_mapping_analysis.py

Purpose
-------
Map Feldman–Zhang (FZ) memorization scores to the Partition-A training split and
compare per-sample Cumulative Sample Loss (CSL) between:
  - Standard (baseline) training
  - Foc-L training

This script is the "data-join + sanity-check" step:
it confirms the cohort construction (Partition-A ∩ top-1% FZ) and verifies that
CSL dictionaries align with the same global ImageNet train indices.

High-level Protocol (Sec. 4.1 / Appendix A.6-aligned cohort definition)
-----------------------------------------------------------------------
1) Load Partition-A train indices (global ImageNet train indices).
2) Load FZ memorization scores for the full ImageNet train set (tr_mem).
3) Select the top-K most memorized samples (K ≈ 1% of ImageNet train size).
4) Compute overlap cohort: (top-K FZ indices) ∩ (Partition-A train indices).
5) Load CSL dictionaries for Standard and Foc-L, keyed by the same global indices.
6) For samples with complete data (mem + both CSLs), compute:
     - csl_diff  = CSL_std - CSL_focl   (positive => Foc-L improved / lower CSL)
     - csl_ratio = CSL_focl / CSL_std
7) Report descriptive statistics and a paired t-test (optional significance check).
8) Save a detailed per-sample CSV + a summary JSON.

Inputs (edit filenames as needed)
---------------------------------
- Partition A split:
    imagenet_subset_splits_partition_A.json
  Expected format: JSON with key "train" containing a list of global indices.

- Feldman–Zhang memorization NPZ:
    imagenet_index.npz
  Expected keys:
    * tr_mem        : memorization scores per global ImageNet train index (len ≈ 1.28M)
    * tr_filenames  : (optional) filenames for quick inspection

- CSL JSONs (global index -> CSL float):
    sorted_dynamic_cumulative_loss_supervised_No_Aug.json
    sorted_fixed_cumulative_loss_dynamicbbox_supervised_No_Aug_90.json

Outputs
-------
- CSV (per-sample, auditable):
    fz_topK_overlap_csl_detailed_100fixed.csv
  Columns include: global_index, mem_score, CSL_std, CSL_focl, csl_diff, csl_ratio

- JSON (summary stats):
    memorization_analysis_summary_new.json

Data-quality checks performed
-----------------------------
- Validates overlap indices are within bounds of the memorization score array.
- Tracks missing CSL entries (missing std only / missing focl only / missing both).
- Computes final "complete data" coverage fraction of the overlap cohort.

Notes / Common pitfalls
-----------------------
- All indexing in this script assumes *global ImageNet train indices*.
  Do NOT pass indices that are local to a Subset/DataLoader.
- "Top-K" is hard-coded as K=13000 (≈ top 1% for ImageNet-1k scale).
  For an exact top-1% definition, set K = int(0.01 * len(mem_scores)).
- This script does NOT run the exact binomial/sign test from Appendix A.6.
  Use the dedicated binomial-test script for the headline significance result.
"""

import json
import numpy as np
import pandas as pd

def main():
    # 1. Load Partition A train indices
    print("=== Loading Partition A train indices ===")
    try:
        with open("imagenet_subset_splits_partition_A.json", "r") as f:
            splits = json.load(f)
        train_indices = set(splits["train"])
        print(f"Loaded {len(train_indices)} train indices for Partition A")
        print(f"Index range: {min(train_indices)} to {max(train_indices)}")
    except Exception as e:
        print(f"Error loading partition data: {e}")
        return

    # 2. Load Feldman–Zhang memorization scores
    print("\n=== Loading Feldman-Zhang memorization scores ===")
    try:
        fz = np.load("imagenet_index.npz", allow_pickle=True)
        fnames = [b.decode() for b in fz["tr_filenames"]]
        mem_scores = fz["tr_mem"]  # array of length ~1.28M
        print(f"Loaded memorization scores for {len(mem_scores)} samples")
        print(f"Memorization score range: {mem_scores.min():.4f} to {mem_scores.max():.4f}")
        print(f"Sample filenames: {fnames[:3]}...")
    except Exception as e:
        print(f"Error loading memorization data: {e}")
        return

    # 3. Identify top-K most memorized samples
    print("\n=== Finding top-K most memorized samples ===")
    K = 13000
    top_k = np.argsort(mem_scores)[-K:][::-1]
    print(f"Top {K} FZ-memorized global indices: {top_k[:10]}...")
    print(f"Top memorization scores: {mem_scores[top_k[:5]]}")

    # 4. Compute overlap with Partition A (with validation)
    print("\n=== Computing overlap with Partition A ===")
    overlap_candidates = set(top_k) & train_indices
    print(f"Initial overlap candidates: {len(overlap_candidates)}")
    
    # Validate that all overlap indices are within memorization array bounds
    valid_overlap = []
    out_of_bounds = 0
    
    for idx in sorted(overlap_candidates):
        if idx >= len(mem_scores) or idx < 0:
            out_of_bounds += 1
        else:
            valid_overlap.append(idx)
    
    print(f"Valid overlap indices: {len(valid_overlap)}")
    print(f"Out-of-bounds indices: {out_of_bounds}")
    
    if out_of_bounds > 0:
        print("WARNING: Some indices are out of bounds for memorization scores!")
    
    overlap = sorted(valid_overlap)

    # 5. Load & normalize CSL JSONs
    print("\n=== Loading CSL data ===")
    try:
        with open("standard_CSL.json", "r") as f:
            raw_std = json.load(f)
        csl_std = {int(k): float(v) for k, v in raw_std.items()}
        print(f"Loaded Standard CSL for {len(csl_std)} samples")
    except Exception as e:
        print(f"Error loading Standard CSL: {e}")
        return

    try:
        with open("FocL_CSL.json", "r") as f:
            raw_focl = json.load(f)
        csl_focl = {int(k): float(v) for k, v in raw_focl.items()}
        print(f"Loaded FocL CSL for {len(csl_focl)} samples")
    except Exception as e:
        print(f"Error loading FocL CSL: {e}")
        return

    # 6. Gather CSLs for the overlapped indices (with detailed validation)
    print("\n=== Matching data across sources ===")
    data = []
    missing_mem = 0
    missing_std_csl = 0
    missing_focl_csl = 0
    missing_both_csl = 0
    
    for idx in overlap:
        # Validate memorization score access (double-check)
        if idx >= len(mem_scores) or idx < 0:
            missing_mem += 1
            continue
        
        # Get CSL scores
        std_csl = csl_std.get(idx, None)
        focl_csl = csl_focl.get(idx, None)
        
        # Track missing data
        if std_csl is None and focl_csl is None:
            missing_both_csl += 1
        elif std_csl is None:
            missing_std_csl += 1
        elif focl_csl is None:
            missing_focl_csl += 1
        
        # Only include samples with both CSL scores
        if std_csl is not None and focl_csl is not None:
            data.append({
                "global_index": idx,
                "mem_score": float(mem_scores[idx]),
                "CSL_std": std_csl,
                "CSL_focl": focl_csl,
                "csl_diff": std_csl - focl_csl,  # Positive = FocL improved
                "csl_ratio": focl_csl / std_csl if std_csl > 0 else None
            })

    # 7. Report data quality
    print(f"\n=== Data Quality Report ===")
    print(f"Total overlap candidates: {len(overlap)}")
    print(f"Missing memorization scores: {missing_mem}")
    print(f"Missing Standard CSL only: {missing_std_csl}")
    print(f"Missing FocL CSL only: {missing_focl_csl}")
    print(f"Missing both CSL scores: {missing_both_csl}")
    print(f"Final samples with complete data: {len(data)}")
    
    if len(data) == 0:
        print("ERROR: No samples with complete data found!")
        return
    
    data_coverage = len(data) / len(overlap) * 100
    print(f"Data coverage: {data_coverage:.1f}%")

    # 8. Create DataFrame and compute stats
    df = pd.DataFrame(data)
    print(f"\n=== Analysis Results ===")
    
    # Basic descriptive stats
    print("Memorization Scores:")
    print(f"  Mean: {df.mem_score.mean():.4f}")
    print(f"  Median: {df.mem_score.median():.4f}")
    print(f"  Range: {df.mem_score.min():.4f} to {df.mem_score.max():.4f}")
    
    print("\nCSL Statistics:")
    mean_std = df.CSL_std.mean()
    mean_focl = df.CSL_focl.mean()
    med_std = df.CSL_std.median()
    med_focl = df.CSL_focl.median()
    
    print(f"  Mean CSL (Standard): {mean_std:.2f}")
    print(f"  Mean CSL (FocL): {mean_focl:.2f}")
    print(f"  Median CSL (Standard): {med_std:.2f}")
    print(f"  Median CSL (FocL): {med_focl:.2f}")
    
    # Improvement analysis
    improved_samples = df.CSL_focl < df.CSL_std
    pct_improved = improved_samples.mean() * 100
    
    if pct_improved > 0:
        avg_improvement = df[improved_samples].csl_diff.mean()
        avg_pct_reduction = ((df.CSL_std - df.CSL_focl) / df.CSL_std).mean() * 100
        
        print(f"\nImprovement Analysis:")
        print(f"  % Samples improved by FocL: {pct_improved:.2f}%")
        print(f"  Average CSL reduction (when improved): {avg_improvement:.2f}")
        print(f"  Average % reduction overall: {avg_pct_reduction:.2f}%")
    else:
        print(f"\nNo improvements found (FocL never better than Standard)")
    
    # Statistical significance (basic)
    from scipy.stats import ttest_rel
    try:
        t_stat, p_value = ttest_rel(df.CSL_std, df.CSL_focl)
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    except ImportError:
        print("\nSciPy not available for statistical testing")
    except Exception as e:
        print(f"\nError in statistical testing: {e}")

    # 9. Save detailed results
    try:
        df.to_csv("fz_topK_overlap_csl_detailed_100fixed.csv", index=False)
        print(f"\nDetailed results saved to: fz_topK_overlap_csl_detailed.csv")
        
        # Save summary stats
        summary = {
            "total_overlap": len(overlap),
            "complete_data_samples": len(data),
            "data_coverage_pct": data_coverage,
            "mean_csl_standard": mean_std,
            "mean_csl_focl": mean_focl,
            "pct_improved": pct_improved,
            "avg_pct_reduction": avg_pct_reduction if pct_improved > 0 else 0,
        }
        
        with open("memorization_analysis_summary_new.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("Summary statistics saved to: memorization_analysis_summary.json")
        
    except Exception as e:
        print(f"Error saving results: {e}")

    print(f"\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()