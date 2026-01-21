#!/usr/bin/env python
"""Plot ICL experiment results."""

import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# Load results
results_path = Path("outputs/icl_experiment_v2/results_20260119_162910.json")
with open(results_path) as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Shorten model names for display
model_names = {
    "claude-3-5-haiku-20241022": "Haiku",
    "claude-sonnet-4-20250514": "Sonnet",
    "claude-opus-4-20250514": "Opus",
}
df["model"] = df["observer_model"].map(model_names)

# Order conditions logically
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
    "oracle_reliability": "Oracle Reliability",
    "oracle_agent_type": "Oracle Agent Type",
    "oracle_truth_labels": "Oracle Truth Labels",
}
df["condition_label"] = df["condition"].map(condition_labels)

# Compute means and std across seeds
summary = df.groupby(["condition", "model"]).agg({
    "accuracy": ["mean", "std"],
    "accuracy_contested": ["mean", "std"],
    "ece": ["mean", "std"],
}).reset_index()
summary.columns = ["condition", "model", "accuracy_mean", "accuracy_std",
                   "contested_mean", "contested_std", "ece_mean", "ece_std"]

# Initialize wandb
wandb.init(
    project="truthification",
    name="icl-baseline-v2-plots",
    job_type="analysis",
)

# Set style
plt.style.use("seaborn-v0_8-whitegrid")
colors = {"Haiku": "#FF6B6B", "Sonnet": "#4ECDC4", "Opus": "#45B7D1"}

# Figure 1: Overall Accuracy by Condition and Model
fig1, ax1 = plt.subplots(figsize=(12, 6))

x = range(len(condition_order))
width = 0.25
models = ["Haiku", "Sonnet", "Opus"]

for i, model in enumerate(models):
    model_data = summary[summary["model"] == model].set_index("condition")
    means = [model_data.loc[c, "accuracy_mean"] * 100 for c in condition_order]
    stds = [model_data.loc[c, "accuracy_std"] * 100 for c in condition_order]
    bars = ax1.bar([xi + i * width for xi in x], means, width,
                   label=model, color=colors[model], yerr=stds, capsize=3)

ax1.set_xlabel("Condition", fontsize=12)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_title("ICL Baseline: Overall Accuracy by Condition and Model", fontsize=14)
ax1.set_xticks([xi + width for xi in x])
ax1.set_xticklabels([condition_labels[c] for c in condition_order], rotation=30, ha="right")
ax1.legend(title="Observer Model")
ax1.set_ylim(60, 100)
ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Random")

plt.tight_layout()
wandb.log({"overall_accuracy_plot": wandb.Image(fig1)})
fig1.savefig("outputs/icl_experiment_v2/overall_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig1)

# Figure 2: Contested Queries Accuracy
fig2, ax2 = plt.subplots(figsize=(12, 6))

for i, model in enumerate(models):
    model_data = summary[summary["model"] == model].set_index("condition")
    means = [model_data.loc[c, "contested_mean"] * 100 for c in condition_order]
    stds = [model_data.loc[c, "contested_std"] * 100 for c in condition_order]
    bars = ax2.bar([xi + i * width for xi in x], means, width,
                   label=model, color=colors[model], yerr=stds, capsize=3)

ax2.set_xlabel("Condition", fontsize=12)
ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title("ICL Baseline: Contested Queries Accuracy (Where Agents Disagree)", fontsize=14)
ax2.set_xticks([xi + width for xi in x])
ax2.set_xticklabels([condition_labels[c] for c in condition_order], rotation=30, ha="right")
ax2.legend(title="Observer Model")
ax2.set_ylim(0, 105)
ax2.axhline(y=50, color="gray", linestyle="--", alpha=0.5)

plt.tight_layout()
wandb.log({"contested_accuracy_plot": wandb.Image(fig2)})
fig2.savefig("outputs/icl_experiment_v2/contested_accuracy.png", dpi=150, bbox_inches="tight")
plt.close(fig2)

# Figure 3: ECE (Calibration Error)
fig3, ax3 = plt.subplots(figsize=(12, 6))

for i, model in enumerate(models):
    model_data = summary[summary["model"] == model].set_index("condition")
    means = [model_data.loc[c, "ece_mean"] for c in condition_order]
    stds = [model_data.loc[c, "ece_std"] for c in condition_order]
    bars = ax3.bar([xi + i * width for xi in x], means, width,
                   label=model, color=colors[model], yerr=stds, capsize=3)

ax3.set_xlabel("Condition", fontsize=12)
ax3.set_ylabel("ECE (lower is better)", fontsize=12)
ax3.set_title("ICL Baseline: Expected Calibration Error by Condition and Model", fontsize=14)
ax3.set_xticks([xi + width for xi in x])
ax3.set_xticklabels([condition_labels[c] for c in condition_order], rotation=30, ha="right")
ax3.legend(title="Observer Model")
ax3.set_ylim(0, 0.35)

plt.tight_layout()
wandb.log({"ece_plot": wandb.Image(fig3)})
fig3.savefig("outputs/icl_experiment_v2/ece.png", dpi=150, bbox_inches="tight")
plt.close(fig3)

# Figure 4: Heatmap of accuracy
pivot = summary.pivot(index="condition", columns="model", values="accuracy_mean")
pivot = pivot.reindex(condition_order)[models] * 100

fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
            vmin=70, vmax=95, ax=ax4, cbar_kws={"label": "Accuracy (%)"})
ax4.set_yticklabels([condition_labels[c] for c in condition_order], rotation=0)
ax4.set_title("ICL Baseline: Accuracy Heatmap", fontsize=14)
ax4.set_xlabel("Observer Model")
ax4.set_ylabel("Condition")

plt.tight_layout()
wandb.log({"accuracy_heatmap": wandb.Image(fig4)})
fig4.savefig("outputs/icl_experiment_v2/accuracy_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig4)

# Figure 5: Line plot showing progression
fig5, ax5 = plt.subplots(figsize=(10, 6))

for model in models:
    model_data = summary[summary["model"] == model].set_index("condition")
    means = [model_data.loc[c, "accuracy_mean"] * 100 for c in condition_order]
    ax5.plot(range(len(condition_order)), means, marker="o", label=model,
             color=colors[model], linewidth=2, markersize=8)

ax5.set_xlabel("Information Level", fontsize=12)
ax5.set_ylabel("Accuracy (%)", fontsize=12)
ax5.set_title("ICL Baseline: How Additional Information Affects Accuracy", fontsize=14)
ax5.set_xticks(range(len(condition_order)))
ax5.set_xticklabels([condition_labels[c] for c in condition_order], rotation=30, ha="right")
ax5.legend(title="Observer Model")
ax5.set_ylim(70, 100)

# Add annotations for key findings
ax5.annotate("Sonnet benefits\nfrom task info!",
             xy=(2, 90.8), xytext=(2.5, 85),
             arrowprops=dict(arrowstyle="->", color="gray"),
             fontsize=9)
ax5.annotate("Haiku hurt\nby complexity",
             xy=(2, 74.6), xytext=(1.2, 70),
             arrowprops=dict(arrowstyle="->", color="gray"),
             fontsize=9)

plt.tight_layout()
wandb.log({"accuracy_progression": wandb.Image(fig5)})
fig5.savefig("outputs/icl_experiment_v2/accuracy_progression.png", dpi=150, bbox_inches="tight")
plt.close(fig5)

# Print summary table
print("\n" + "=" * 60)
print("ICL Baseline v2 Results Summary")
print("=" * 60)

print("\nOverall Accuracy (%):")
print(pivot.to_string())

print("\nContested Queries Accuracy (%):")
pivot_contested = summary.pivot(index="condition", columns="model", values="contested_mean")
pivot_contested = pivot_contested.reindex(condition_order)[models] * 100
print(pivot_contested.to_string())

print("\nECE (Calibration Error):")
pivot_ece = summary.pivot(index="condition", columns="model", values="ece_mean")
pivot_ece = pivot_ece.reindex(condition_order)[models]
print(pivot_ece.to_string())

# Log summary table to wandb
wandb.log({
    "summary_table": wandb.Table(dataframe=summary),
    "overall_accuracy_table": wandb.Table(dataframe=pivot.reset_index()),
    "contested_accuracy_table": wandb.Table(dataframe=pivot_contested.reset_index()),
})

print(f"\nPlots saved to outputs/icl_experiment_v2/")
print(f"wandb run: {wandb.run.url}")

wandb.finish()
