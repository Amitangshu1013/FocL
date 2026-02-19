# =============================================================================
# compute_high_memorization.py — Rank Samples by Cumulative Loss (CSL / “High-Mem”)
# =============================================================================
#
# Purpose
# -------
# Post-process a saved memorization log (JSON) and produce a ranked list of
# samples by *cumulative loss* across training epochs. This is the core quantity
# used for CSL-style memorization / difficulty analysis.
#
# This script is model-agnostic:
# - Works for both FocL (bbox-crop) runs and standard full-image baselines,
#   as long as they write the same memorization-log JSON structure.
#
#
# Expected input format
# ---------------------
# A memorization log JSON produced by `update_memorization_log(...)`:
#
#   {
#     "<sample_id>": {
#       "loss": [l_0, l_1, ..., l_{T-1}],
#       "confidence": [c_0, c_1, ..., c_{T-1}]
#     },
#     ...
#   }
#
# Notes:
# - Keys are typically numeric strings (e.g., global ImageFolder indices).
# - This script uses ONLY the "loss" list; "confidence" is ignored here.
#
#
# Computation
# -----------
# For each sample_id:
#   cumulative_loss(sample_id) = sum(loss_list)
#
# Then sorts samples in descending order:
#   highest cumulative loss first  →  “hardest / least learned” samples.
#
#
# Output
# ------
# Writes a JSON mapping sample_id → cumulative_loss, sorted by cumulative loss:
#
#   {
#     12345: 98.123,
#     67890: 97.456,
#     ...
#   }
#
# (sample_id is converted to int in the output keys.)
#
#
# Typical usage
# -------------
# - Set LOSS_JSON_FILE to your memorization log file path.
# - Run:
#     python compute_high_memorization.py
# - Use OUTPUT_JSON_FILE as an index list for visualization or further analysis.
#
# Gotchas
# -------
# - sample_id keys must be convertible to int (e.g., "12345").
# - If different runs log different numbers of epochs, cumulative losses are not
#   directly comparable unless you normalize (e.g., average per epoch).
# =============================================================================




import json
import os

# Manually specify the JSON file path for a target model (focl or standard)
LOSS_JSON_FILE = "memorization_dynamiclog_supervised.json"
OUTPUT_JSON_FILE = "sorted_dynamic_cumulative_loss_supervised_sep_B_Agg_Aug_90.json"  # Output file

def load_loss_data(json_file):
    """Loads loss values from a JSON file."""
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"File not found: {json_file}")

    with open(json_file, "r") as f:
        data = json.load(f)
    
    return data

def compute_cumulative_loss(loss_data):
    """Computes cumulative loss for each sample."""
    cumulative_loss_dict = {}

    for sample_id, values in loss_data.items():
        if "loss" in values:
            cumulative_loss = sum(values["loss"])  # Sum all 90 loss values
            cumulative_loss_dict[int(sample_id)] = cumulative_loss  # Convert keys to integers
    
    return cumulative_loss_dict

def sort_and_store_cumulative_loss(cumulative_loss_dict, output_file):
    """Sorts samples by cumulative loss in descending order and stores in a JSON file."""
    sorted_loss = sorted(cumulative_loss_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Convert list of tuples back into a dictionary
    sorted_loss_dict = {sample_id: loss for sample_id, loss in sorted_loss}
    
    # Save sorted loss data
    with open(output_file, "w") as f:
        json.dump(sorted_loss_dict, f, indent=4)

    print(f"Sorted cumulative loss saved to {output_file}")

def main():
    # Load loss data from JSON
    loss_data = load_loss_data(LOSS_JSON_FILE)

    # Compute cumulative loss per sample
    cumulative_loss_dict = compute_cumulative_loss(loss_data)

    # Sort and store cumulative loss in a new JSON file
    sort_and_store_cumulative_loss(cumulative_loss_dict, OUTPUT_JSON_FILE)

if __name__ == "__main__":
    main()
