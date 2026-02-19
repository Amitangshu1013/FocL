# =============================================================================
# CSL_dualplots.py — CSL Distribution Comparison (Counts + Density Overlay)
# =============================================================================
#
# Purpose
# -------
# Visualize and compare the distribution of Cumulative Sample Loss (CSL) between
# two runs/models (e.g., Standard baseline vs Foc-L Base) using the *sorted
# cumulative loss* JSON outputs produced by `compute_high_memorization.py`.
#
# Each input JSON is expected to be a mapping:
#     { sample_id : cumulative_loss }
# and this script uses only the cumulative_loss values to build histograms.
#
#
# What this script plots
# ----------------------
# Creates a single figure with two y-axes:
#   - Left y-axis: histogram COUNTS (filled bars)
#   - Right y-axis: histogram DENSITY (outline/step, normalized)
#
# Both models are plotted on the same common binning so the shapes are directly
# comparable.
#
#
# Key implementation details
# --------------------------
# 1) Loads cumulative loss values from `model_files` paths into numpy arrays.
# 2) Computes shared histogram bins using Freedman–Diaconis binning:
#       np.histogram_bin_edges(all_losses, bins="fd")
#    This chooses bin width based on data spread and sample count, and ensures
#    both models use identical bin edges.
# 3) Overlays:
#    - Count histograms (alpha-filled) on the primary axis
#    - Density histograms (step outlines) on the secondary axis
# 4) Saves the figure to:
#       csl_counts_and_density_90_epochs_foclbase2.png
#
#
# How to use
# ----------
# - Update `model_files` with the JSON paths for the models you want to compare.
# - Optionally update `model_colors`.
# - Run:
#       python CSL_dualplots.py
#
# Notes
# -----
# - The title mentions “90 epochs”; make sure it matches the logs used.
# - To compare more than two models, add more entries to `model_files` and
#   `model_colors` (the code already supports multiple).
# =============================================================================




import json
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION: update these paths if needed ---

model_files = {
    "Standard": "sorted_dynamic_cumulative_loss_supervised_No_Aug.json",
    "Foc-L SC": "sorted_fixed_cumulative_loss_dynamicbbox_supervised_No_Aug_90.json"
}

model_colors = {"Standard": "red", "Foc-L SC": "green"}



# --- LOAD DATA ---
cumulative_losses = {}
for name, path in model_files.items():
    with open(path, "r") as f:
        cumulative_losses[name] = np.array(list(json.load(f).values()))

# --- COMPUTE COMMON BINS (Freedman–Diaconis) ---
all_losses = np.concatenate(list(cumulative_losses.values()))
bins = np.histogram_bin_edges(all_losses, bins="fd")

# --- PLOT ---
fig, ax_count = plt.subplots(figsize=(10, 6))

# Secondary y-axis for density
ax_density = ax_count.twinx()

for name, losses in cumulative_losses.items():
    color = model_colors[name]

    # 1) plot count histogram on ax_count
    ax_count.hist(
        losses,
        bins=bins,
        alpha=0.4,
        label=f"{name} (count)",
        color=color
    )
    # 2) plot density outline on ax_density
    ax_density.hist(
        losses,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.5,
        label=f"{name} (density)",
        color=color
    )

# --- AXIS & LEGEND ---
ax_count.set_xlabel("Cumulative Loss", fontsize=12)
ax_count.set_ylabel("Count", fontsize=12, color="black")
ax_density.set_ylabel("Density", fontsize=12, color="black")

# Combine legends from both axes
h1, l1 = ax_count.get_legend_handles_labels()
h2, l2 = ax_density.get_legend_handles_labels()
ax_count.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=10)

ax_count.grid(True, linestyle="--", linewidth=0.5)
plt.title("CSL Histogram: Counts & Density — 90 epochs", fontsize=14)
plt.tight_layout()

# --- SAVE & SHOW ---
plt.savefig("csl_counts_and_density_90_epochs_foclbase2.png", dpi=300)
plt.show()
