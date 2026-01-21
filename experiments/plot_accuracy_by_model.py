#!/usr/bin/env python
"""Plot accuracy by condition for each model (weakest to strongest)."""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import wandb

# Load results
results_path = Path("outputs/icl_experiment_v2/results_20260119_162910.json")
with open(results_path) as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Model names and order (weakest to strongest)
model_order = [
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
]
model_names = {
    "claude-3-5-haiku-20241022": "Haiku",
    "claude-sonnet-4-20250514": "Sonnet",
    "claude-opus-4-20250514": "Opus",
}
df["model"] = df["observer_model"].map(model_names)

# Condition order (increasing information)
condition_order = [
    "no_ids",
    "ids_only",
    "ids_and_tasks",
    "oracle_reliability",
    "oracle_agent_type",
    "oracle_truth_labels",
]
condition_labels = {
    "no_ids": "No IDs",
    "ids_only": "IDs Only",
    "ids_and_tasks": "IDs + Tasks",
    "oracle_reliability": "Oracle\nReliability",
    "oracle_agent_type": "Oracle\nAgent Type",
    "oracle_truth_labels": "Oracle\nTruth Labels",
}

# Compute means and std across seeds
summary = df.groupby(["condition", "model"]).agg({
    "accuracy": ["mean", "std"],
    "accuracy_contested": ["mean", "std"],
}).reset_index()
summary.columns = ["condition", "model", "acc_mean", "acc_std", "contested_mean", "contested_std"]

# Initialize wandb
wandb.init(
    project="truthification",
    name="icl-accuracy-plots",
    job_type="analysis",
)

# Colors for models
colors = {
    "Haiku": "#E57373",  # Red
    "Sonnet": "#64B5F6",  # Blue
    "Opus": "#81C784",   # Green
}

# Create figure with two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models = ["Haiku", "Sonnet", "Opus"]
x = np.arange(len(condition_order))
width = 0.25

# Plot 1: Overall Accuracy
ax1 = axes[0]
for i, model in enumerate(models):
    model_data = summary[summary["model"] == model].set_index("condition")
    means = [model_data.loc[c, "acc_mean"] * 100 for c in condition_order]
    stds = [model_data.loc[c, "acc_std"] * 100 for c in condition_order]
    bars = ax1.bar(x + i * width, means, width, label=model, color=colors[model],
                   yerr=stds, capsize=2, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for j, (bar, mean) in enumerate(zip(bars, means)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[j] + 1,
                f'{mean:.0f}', ha='center', va='bottom', fontsize=7, rotation=0)

ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_title("Overall Accuracy", fontsize=14, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels([condition_labels[c] for c in condition_order], fontsize=9)
ax1.set_ylim(50, 105)
ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.3, linewidth=1)
ax1.legend(title="Model", loc='lower right', fontsize=9)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Plot 2: Contested Queries Accuracy
ax2 = axes[1]
for i, model in enumerate(models):
    model_data = summary[summary["model"] == model].set_index("condition")
    means = [model_data.loc[c, "contested_mean"] * 100 for c in condition_order]
    stds = [model_data.loc[c, "contested_std"] * 100 for c in condition_order]
    bars = ax2.bar(x + i * width, means, width, label=model, color=colors[model],
                   yerr=stds, capsize=2, edgecolor='white', linewidth=0.5)

    # Add value labels on bars
    for j, (bar, mean) in enumerate(zip(bars, means)):
        if mean > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + stds[j] + 1,
                    f'{mean:.0f}', ha='center', va='bottom', fontsize=7, rotation=0)

ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title("Contested Queries Only (Agents Disagree)", fontsize=14, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels([condition_labels[c] for c in condition_order], fontsize=9)
ax2.set_ylim(0, 115)
ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.3, linewidth=1)
ax2.legend(title="Model", loc='lower right', fontsize=9)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.suptitle("ICL Baseline v2: Observer Accuracy by Condition and Model\n(3 seeds, 80 queries each)",
             fontsize=14, y=1.02)
plt.tight_layout()

wandb.log({"accuracy_comparison": wandb.Image(fig)})
fig.savefig("outputs/icl_experiment_v2/accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Create a cleaner line plot version
fig2, ax = plt.subplots(figsize=(10, 6))

for model in models:
    model_data = summary[summary["model"] == model].set_index("condition")
    means = [model_data.loc[c, "acc_mean"] * 100 for c in condition_order]
    stds = [model_data.loc[c, "acc_std"] * 100 for c in condition_order]

    ax.plot(x, means, marker='o', label=model, color=colors[model],
            linewidth=2.5, markersize=10)
    ax.fill_between(x, [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=colors[model], alpha=0.15)

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xlabel("Condition (increasing information →)", fontsize=12)
ax.set_title("ICL Baseline v2: How Information Affects Observer Accuracy\n(Models ordered weakest → strongest)",
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([condition_labels[c] for c in condition_order], fontsize=10)
ax.set_ylim(70, 100)
ax.legend(title="Model", loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# Highlight key finding
ax.annotate("Sonnet + Tasks\nmatches oracles!",
            xy=(2, 90.8), xytext=(2.7, 82),
            arrowprops=dict(arrowstyle="->", color="#1976D2", lw=1.5),
            fontsize=10, color="#1976D2", fontweight='bold')

ax.annotate("Haiku hurt by\ntask complexity",
            xy=(2, 74.6), xytext=(1.0, 78),
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5),
            fontsize=10, color="#C62828", fontweight='bold')

plt.tight_layout()
wandb.log({"accuracy_line_plot": wandb.Image(fig2)})
fig2.savefig("outputs/icl_experiment_v2/accuracy_line_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# Create contested-only line plot
fig3, ax = plt.subplots(figsize=(10, 6))

for model in models:
    model_data = summary[summary["model"] == model].set_index("condition")
    means = [model_data.loc[c, "contested_mean"] * 100 for c in condition_order]
    stds = [model_data.loc[c, "contested_std"] * 100 for c in condition_order]

    ax.plot(x, means, marker='o', label=model, color=colors[model],
            linewidth=2.5, markersize=10)
    ax.fill_between(x, [max(0, m - s) for m, s in zip(means, stds)],
                    [min(100, m + s) for m, s in zip(means, stds)],
                    color=colors[model], alpha=0.15)

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_xlabel("Condition (increasing information →)", fontsize=12)
ax.set_title("ICL Baseline v2: Contested Queries (Where Truth Depends on Source Trust)\n(Models ordered weakest → strongest)",
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([condition_labels[c] for c in condition_order], fontsize=10)
ax.set_ylim(20, 110)
ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random guess")
ax.legend(title="Model", loc='lower right', fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)

# Highlight key finding
ax.annotate("Sonnet: 95%\nwith just task info!",
            xy=(2, 94.7), xytext=(1.3, 75),
            arrowprops=dict(arrowstyle="->", color="#1976D2", lw=1.5),
            fontsize=10, color="#1976D2", fontweight='bold')

ax.annotate("Haiku: 36%\n(worse than random)",
            xy=(2, 36), xytext=(2.7, 50),
            arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5),
            fontsize=10, color="#C62828", fontweight='bold')

plt.tight_layout()
wandb.log({"contested_line_plot": wandb.Image(fig3)})
fig3.savefig("outputs/icl_experiment_v2/contested_line_plot.png", dpi=150, bbox_inches="tight")
plt.close(fig3)

print("Plots saved:")
print("  - outputs/icl_experiment_v2/accuracy_comparison.png")
print("  - outputs/icl_experiment_v2/accuracy_line_plot.png")
print("  - outputs/icl_experiment_v2/contested_line_plot.png")
print(f"\nwandb: {wandb.run.url}")

wandb.finish()
