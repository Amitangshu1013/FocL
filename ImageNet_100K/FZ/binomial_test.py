"""
Purpose
-------
Reproduce the Feldman–Zhang (FZ) cohort analysis from Sec. 4.1 / Appendix A.6:
we test whether Foc-L reduces sample difficulty (CSL) on the most-memorized samples.

Core idea:
1) Use pre-computed Feldman–Zhang memorization scores over ImageNet train indices.
2) Select the top-1% most memorized samples (global ImageNet train indices).
3) Intersect that set with the paper’s training split (Partition A train indices).
4) For the resulting cohort, compare per-sample CSL under:
      - Standard training (baseline)
      - Foc-L training (single-crop / single-glimpse variant)
5) Quantify how often CSL decreases (CSL_FocL < CSL_Std) and run significance tests:
      - One-sided exact binomial (sign) test with null p = 0.5
      - Paired t-test on CSL values (Std vs FocL)

Inputs (edit paths below)
-------------------------
- Partition-A train split indices:
    imagenet_subset_splits_partition_A.json
  Expected format: list (or dict with a key) containing global ImageNet indices used in Partition A train.

- Feldman–Zhang memorization scores:
    imagenet_index.npz
  Expected keys: "tr_mem" (memorization score per global ImageNet train index).
  Higher score = more memorized.

- CSL dictionaries for the SAME global indices:
    standard_json: per-sample CSL for baseline training
    focl_json:     per-sample CSL for Foc-L training
  Expected format: dict-like mapping { global_index (str/int) : CSL_value (float) }.

Protocol Details (paper-aligned)
--------------------------------
- Top-1% cohort: select top K samples by FZ score, where K ≈ 1% of ImageNet train size.
  (Current code uses a fixed K=13000, which is ~1% for ImageNet-1k scale.)
- Cohort used in tests: (Top-1% by FZ) ∩ (Partition-A train indices).
- "Improvement" definition: CSL_FocL < CSL_Std (strict decrease).
- Statistical tests:
    * Binomial (sign) test: one-sided, alternative='greater'
      Null: improvement probability p = 0.5
    * Paired t-test: compares paired CSL values (Std vs FocL)

Outputs
-------
- Prints cohort size, number improved, fraction improved, and test statistics (p-values).
- Writes a CSV (audit trail) containing per-sample:
      index, FZ score, CSL_std, CSL_focl, improved_flag

Notes / Common Pitfalls
-----------------------
- Ensure the CSL JSONs correspond to the SAME sample indexing scheme as the FZ scores
  (global ImageNet train indices), otherwise the intersection will be wrong.
- Ensure the Foc-L CSL JSON corresponds to the intended Foc-L variant used for the paper’s
  memorization analysis (typically the single-crop / single-glimpse setting).
- If you want a strict “top 1%” definition, compute K = int(0.01 * len(tr_mem)) instead of hard-coding.
"""



import json
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, binomtest

def main():
    # 1. Load Partition A train indices
    print("=== Loading Partition A train indices ===")
    try:
        with open("imagenet_subset_splits_partition_A.json", "r") as f:
            splits = json.load(f)
        train_indices = set(splits["train"])
        print(f"Loaded {len(train_indices)} train indices for Partition A")
    except Exception as e:
        print(f"Error loading partition data: {e}")
        return

    # 2. Load Feldman-Zhang memorization scores
    print("\n=== Loading Feldman-Zhang memorization scores ===")
    try:
        fz = np.load("imagenet_index.npz", allow_pickle=True)
        mem_scores = fz["tr_mem"]  # array of length ~1.28M
        print(f"Loaded memorization scores for {len(mem_scores)} samples")
    except Exception as e:
        print(f"Error loading memorization data: {e}")
        return

    # 3. Identify top-K most memorized samples (Top 1% of ImageNet)
    print("\n=== Finding top-K most memorized samples ===")
    # K set to approx 1% of 1.28M images to match paper cohort
    K = int(0.01 * len(mem_scores)) 
    top_k = np.argsort(mem_scores)[-K:][::-1]
    print(f"Top {K} FZ-memorized global indices selected.")

    # 4. Compute overlap with Partition A
    print("\n=== Computing overlap with Partition A ===")
    overlap_candidates = set(top_k) & train_indices
    
    valid_overlap = []
    for idx in sorted(overlap_candidates):
        if 0 <= idx < len(mem_scores):
            valid_overlap.append(idx)
            
    print(f"Overlap cohort size (should be approx 820): {len(valid_overlap)}")

    # 5. Load & normalize CSL JSONs
    print("\n=== Loading CSL data ===")
    try:
        with open("csl_standard.json", "r") as f:
            csl_std = {int(k): float(v) for k, v in json.load(f).items()}
        
        with open("csl_focl.json", "r") as f:
            csl_focl = {int(k): float(v) for k, v in json.load(f).items()}
            
        print("CSL data loaded successfully.")
    except Exception as e:
        print(f"Error loading CSL data: {e}")
        return

    # 6. Match data
    data = []
    for idx in valid_overlap:
        std = csl_std.get(idx)
        focl = csl_focl.get(idx)
        
        if std is not None and focl is not None:
            data.append({
                "global_index": idx,
                "mem_score": float(mem_scores[idx]),
                "CSL_std": std,
                "CSL_focl": focl
            })

    df = pd.DataFrame(data)
    
    if len(df) == 0:
        print("Error: No overlapping data found.")
        return

    # 7. STATISTICS AND TESTS
    print(f"\n=== FINAL STATISTICS (Use these for Rebuttal) ===")
    print(f"Cohort Size: {len(df)}")
    
    # Count improvements
    # Improvement defined as: FocL Loss < Standard Loss
    n_improved = (df.CSL_focl < df.CSL_std).sum()
    n_total = len(df)
    pct_improved = (n_improved / n_total) * 100
    
    print(f"\n1. Percentage Claim Analysis:")
    print(f"   Samples Improved: {n_improved}/{n_total}")
    print(f"   Percentage: {pct_improved:.2f}%")
    
    # BINOMIAL TEST (Answers: Is 99.88% significant?)
    # Null Hypothesis: FocL is no better than random (p=0.5)
    binom_res = binomtest(n_improved, n_total, p=0.5, alternative='greater')
    print(f"   Binomial Test p-value: {binom_res.pvalue}")
    print(f"   (If 0.0, it means p < 1e-16)")

    # T-TEST (Answers: Is the average drop significant?)
    print(f"\n2. Magnitude Claim Analysis:")
    print(f"   Mean CSL (Standard): {df.CSL_std.mean():.2f}")
    print(f"   Mean CSL (FocL):     {df.CSL_focl.mean():.2f}")
    
    t_stat, t_p_val = ttest_rel(df.CSL_std, df.CSL_focl)
    print(f"   Paired T-Test p-value: {t_p_val}")

    # Save for reference
    df.to_csv("rebuttal_stats_detailed.csv", index=False)
    print("\nDetailed CSV saved to 'rebuttal_stats_detailed.csv'")

if __name__ == "__main__":
    main()
