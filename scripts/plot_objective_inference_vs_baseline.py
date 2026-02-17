#!/usr/bin/env python3
"""
Plot agent objective inference accuracy vs random baseline.
Shows how well the estimator can detect agent goals compared to random guessing.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data
conditions = ["Simple Interest", "Complex VF", "Overall"]
accuracy = [65.8, 35.3, 50.6]
baseline = [18.6, 20.0, 19.1]
above_baseline = [47.2, 15.3, 31.5]

def create_plot():
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(conditions))
    width = 0.35

    # Create bars
    bars_accuracy = ax.bar(x - width/2, accuracy, width, label="Actual Accuracy",
                           color="#2ecc71", edgecolor="black", linewidth=1.2)
    bars_baseline = ax.bar(x + width/2, baseline, width, label="Random Baseline",
                           color="#95a5a6", edgecolor="black", linewidth=1.2)

    # Add value labels on bars
    for bar, val in zip(bars_accuracy, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

    for bar, val in zip(bars_baseline, baseline):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f"{val:.1f}%", ha="center", fontsize=10, color="#666")

    # Add "above baseline" annotations
    for i, (acc, base, above) in enumerate(zip(accuracy, baseline, above_baseline)):
        # Draw arrow from baseline to accuracy
        ax.annotate("", xy=(i - width/2, acc - 2), xytext=(i - width/2, base + 2),
                   arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=2))
        # Add delta label
        ax.text(i - width/2 - 0.15, (acc + base) / 2, f"+{above:.1f}%",
               fontsize=10, fontweight="bold", color="#e74c3c", va="center")

    ax.set_ylabel("Inference Accuracy (%)", fontsize=12)
    ax.set_title("Agent Objective Inference: Accuracy vs Random Baseline",
                fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right", fontsize=10)

    # Add horizontal line at overall baseline
    ax.axhline(y=19.1, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    plt.tight_layout()

    output_path = output_dir / "objective_inference_vs_baseline.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")

    # Also create a stacked/difference view
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Stacked bar showing baseline + improvement
    bars_base = ax2.bar(conditions, baseline, label="Random Baseline",
                        color="#95a5a6", edgecolor="black", linewidth=1.2)
    bars_improvement = ax2.bar(conditions, above_baseline, bottom=baseline,
                               label="Improvement Over Baseline",
                               color="#2ecc71", edgecolor="black", linewidth=1.2)

    # Add labels
    for i, (bar, base, above, acc) in enumerate(zip(bars_base, baseline, above_baseline, accuracy)):
        # Baseline label
        ax2.text(bar.get_x() + bar.get_width()/2, base/2,
                f"{base:.1f}%", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
        # Improvement label
        ax2.text(bar.get_x() + bar.get_width()/2, base + above/2,
                f"+{above:.1f}%", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
        # Total label on top
        ax2.text(bar.get_x() + bar.get_width()/2, acc + 2,
                f"{acc:.1f}%", ha="center", fontsize=11, fontweight="bold")

    ax2.set_ylabel("Inference Accuracy (%)", fontsize=12)
    ax2.set_title("Agent Objective Inference: Baseline + Improvement",
                 fontsize=14, fontweight="bold")
    ax2.set_ylim(0, 80)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()

    output_path2 = output_dir / "objective_inference_stacked.png"
    plt.savefig(output_path2, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path2}")

    plt.close("all")

    return output_path, output_path2


if __name__ == "__main__":
    create_plot()
    print("\nKey insight: Simple interests are highly detectable (+47% above baseline)")
    print("             Complex value functions are harder but still above random (+15%)")
