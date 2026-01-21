#!/usr/bin/env python
"""Plot comparison between isolated-query and full-context ICL experiments."""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import wandb

# Load both result sets
baseline_path = Path("outputs/icl_experiment_v2/results_20260119_162910.json")
full_context_path = Path("outputs/icl_full_context/results_20260120_151904.json")

with open(baseline_path) as f:
    baseline_results = json.load(f)

with open(full_context_path) as f:
    full_context_results = json.load(f)

# Create DataFrames
df_baseline = pd.DataFrame(baseline_results)
df_baseline["experiment"] = "isolated"

df_full = pd.DataFrame(full_context_results)
df_full["experiment"] = "full_context"

# Model names
model_names = {
    "claude-3-5-haiku-20241022": "Haiku",
    "claude-sonnet-4-20250514": "Sonnet",
    "claude-opus-4-20250514": "Opus",
}
df_baseline["model"] = df_baseline["observer_model"].map(model_names)
df_full["model"] = df_full["observer_model"].map(model_names)

# Compute averages
baseline_avg = df_baseline.groupby(["condition", "model"]).agg({
    "accuracy": "mean",
    "accuracy_contested": "mean",
}).reset_index()

full_avg = df_full.groupby(["condition", "model"]).agg({
    "accuracy": "mean",
    "accuracy_contested": "mean",
}).reset_index()

# Map conditions for comparison
# Baseline: no_ids, ids_only, ids_and_tasks
# Full: full_context_no_ids, full_context_ids, full_context_ids_tasks
condition_mapping = {
    "no_ids": "full_context_no_ids",
    "ids_only": "full_context_ids",
    "ids_and_tasks": "full_context_ids_tasks",
}

# Initialize wandb
wandb.init(
    project="truthification",
    name="icl-comparison-plots",
    job_type="analysis",
)

# Colors
colors = {
    "Haiku": "#E57373",
    "Sonnet": "#64B5F6",
    "Opus": "#81C784",
}

models = ["Haiku", "Sonnet", "Opus"]
conditions = ["no_ids", "ids_only", "ids_and_tasks"]
condition_labels = {
    "no_ids": "No IDs",
    "ids_only": "IDs Only",
    "ids_and_tasks": "IDs + Tasks",
}

# Figure 1: Side-by-side comparison (Isolated vs Full Context)
fig1, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, model in enumerate(models):
    ax = axes[i]

    baseline_data = baseline_avg[baseline_avg["model"] == model].set_index("condition")
    full_data = full_avg[full_avg["model"] == model].set_index("condition")

    x = np.arange(len(conditions))
    width = 0.35

    baseline_vals = [baseline_data.loc[c, "accuracy"] * 100 for c in conditions]
    full_vals = [full_data.loc[condition_mapping[c], "accuracy"] * 100 for c in conditions]

    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Isolated Query", color=colors[model], alpha=0.6)
    bars2 = ax.bar(x + width/2, full_vals, width, label="Full Context", color=colors[model], alpha=1.0, hatch="//")

    # Add value labels
    for bar, val in zip(bars1, baseline_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}',
                ha='center', va='bottom', fontsize=9)
    for bar, val in zip(bars2, full_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.0f}',
                ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Accuracy (%)" if i == 0 else "")
    ax.set_title(f"{model}", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([condition_labels[c] for c in conditions], fontsize=10)
    ax.set_ylim(60, 100)
    ax.legend(loc='lower right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.suptitle("Isolated Query vs Full Context: Overall Accuracy by Model", fontsize=14, y=1.02)
plt.tight_layout()
wandb.log({"isolated_vs_full_context": wandb.Image(fig1)})
fig1.savefig("outputs/icl_full_context/isolated_vs_full_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig1)

# Figure 2: Delta plot (Full Context - Isolated)
fig2, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(conditions))
width = 0.25

for i, model in enumerate(models):
    baseline_data = baseline_avg[baseline_avg["model"] == model].set_index("condition")
    full_data = full_avg[full_avg["model"] == model].set_index("condition")

    deltas = []
    for c in conditions:
        baseline_val = baseline_data.loc[c, "accuracy"] * 100
        full_val = full_data.loc[condition_mapping[c], "accuracy"] * 100
        deltas.append(full_val - baseline_val)

    bars = ax.bar(x + i * width, deltas, width, label=model, color=colors[model])

    # Add value labels
    for bar, delta in zip(bars, deltas):
        va = 'bottom' if delta >= 0 else 'top'
        offset = 0.5 if delta >= 0 else -0.5
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
                f'{delta:+.1f}', ha='center', va=va, fontsize=9)

ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel("Accuracy Change (pp)", fontsize=12)
ax.set_xlabel("Condition", fontsize=12)
ax.set_title("Effect of Full Context on Accuracy\n(Full Context - Isolated Query)", fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([condition_labels[c] for c in conditions], fontsize=11)
ax.legend(title="Model", loc='lower left', fontsize=10)
ax.set_ylim(-15, 15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
wandb.log({"full_context_delta": wandb.Image(fig2)})
fig2.savefig("outputs/icl_full_context/full_context_delta.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# Figure 3: Full comparison including oracle conditions
fig3, ax = plt.subplots(figsize=(14, 6))

# All conditions we want to compare
all_conditions = [
    ("no_ids", "Isolated\nNo IDs"),
    ("full_context_no_ids", "Full\nNo IDs"),
    ("ids_only", "Isolated\nIDs"),
    ("full_context_ids", "Full\nIDs"),
    ("ids_and_tasks", "Isolated\nIDs+Tasks"),
    ("full_context_ids_tasks", "Full\nIDs+Tasks"),
    ("oracle_reliability", "Oracle\nReliability"),
    ("oracle_truth_labels", "Oracle\nTruth"),
]

x = np.arange(len(all_conditions))
width = 0.25

for i, model in enumerate(models):
    vals = []
    for cond, _ in all_conditions:
        if cond.startswith("full_context"):
            data = full_avg[full_avg["model"] == model].set_index("condition")
        else:
            data = baseline_avg[baseline_avg["model"] == model].set_index("condition")

        if cond in data.index:
            vals.append(data.loc[cond, "accuracy"] * 100)
        else:
            vals.append(0)

    ax.bar(x + i * width, vals, width, label=model, color=colors[model])

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Complete Comparison: Isolated vs Full Context vs Oracle", fontsize=14, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels([label for _, label in all_conditions], fontsize=9)
ax.legend(title="Model", loc='lower right', fontsize=10)
ax.set_ylim(60, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add vertical lines to separate groups
for i in [2, 4, 6]:
    ax.axvline(x=i - 0.5, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
wandb.log({"complete_comparison": wandb.Image(fig3)})
fig3.savefig("outputs/icl_full_context/complete_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig3)

# Print summary table
print("\n" + "=" * 70)
print("ISOLATED vs FULL CONTEXT COMPARISON")
print("=" * 70)

print("\nOverall Accuracy (%):")
print("-" * 50)
for model in models:
    print(f"\n{model}:")
    baseline_data = baseline_avg[baseline_avg["model"] == model].set_index("condition")
    full_data = full_avg[full_avg["model"] == model].set_index("condition")

    for c in conditions:
        b = baseline_data.loc[c, "accuracy"] * 100
        f = full_data.loc[condition_mapping[c], "accuracy"] * 100
        delta = f - b
        sign = "+" if delta >= 0 else ""
        print(f"  {condition_labels[c]:12s}: {b:5.1f}% → {f:5.1f}%  ({sign}{delta:.1f}pp)")

# Log tables
wandb.log({
    "baseline_summary": wandb.Table(dataframe=baseline_avg),
    "full_context_summary": wandb.Table(dataframe=full_avg),
})

print(f"\nPlots saved to outputs/icl_full_context/")
print(f"wandb: {wandb.run.url}")

wandb.finish()
