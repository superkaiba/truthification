#!/usr/bin/env python3
"""
Plot agent objective inference accuracy across different experimental conditions.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from multi-factor experiment analysis
# Agent objective inference: how well the estimator infers what agents are optimizing for

# By Agent Value Function Type
agent_vf_data = {
    "Simple Interest": {"mean": 65.8, "std": 1.9, "n": 18},
    "Complex VF": {"mean": 35.3, "std": 2.4, "n": 18},
}

# By Observer Condition
observer_data = {
    "Blind": {"mean": 47.9, "std": 5.5, "n": 12},
    "IDs Visible": {"mean": 51.2, "std": 6.0, "n": 12},
    "Interests Visible": {"mean": 52.6, "std": 4.2, "n": 12},
}

# By Oracle Type
oracle_data = {
    "Strategic": {"mean": 51.9, "std": 4.3, "n": 18},
    "Random": {"mean": 49.3, "std": 4.2, "n": 18},
}

# Overall
overall = {"mean": 50.6, "std": 3.0}


def create_plots():
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Agent Objective Inference Accuracy", fontsize=14, fontweight="bold")

    # Colors
    colors = ["#2ecc71", "#e74c3c", "#3498db", "#9b59b6"]

    # Plot 1: By Agent VF Type
    ax1 = axes[0]
    categories = list(agent_vf_data.keys())
    means = [agent_vf_data[k]["mean"] for k in categories]
    stds = [agent_vf_data[k]["std"] for k in categories]

    bars1 = ax1.bar(categories, means, yerr=stds, capsize=5,
                    color=[colors[0], colors[1]], edgecolor="black", linewidth=1.2)
    ax1.axhline(y=50, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    ax1.set_ylabel("Inference Accuracy (%)")
    ax1.set_title("By Agent Type", fontweight="bold")
    ax1.set_ylim(0, 80)

    # Add value labels
    for bar, mean in zip(bars1, means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{mean:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # Plot 2: By Observer Condition
    ax2 = axes[1]
    categories = list(observer_data.keys())
    means = [observer_data[k]["mean"] for k in categories]
    stds = [observer_data[k]["std"] for k in categories]

    bars2 = ax2.bar(categories, means, yerr=stds, capsize=5,
                    color=[colors[0], colors[2], colors[3]], edgecolor="black", linewidth=1.2)
    ax2.axhline(y=50, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    ax2.set_title("By Observer Condition", fontweight="bold")
    ax2.set_ylim(0, 80)
    ax2.tick_params(axis='x', rotation=15)

    for bar, mean in zip(bars2, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{mean:.1f}%", ha="center", fontsize=10, fontweight="bold")

    # Plot 3: By Oracle Type
    ax3 = axes[2]
    categories = list(oracle_data.keys())
    means = [oracle_data[k]["mean"] for k in categories]
    stds = [oracle_data[k]["std"] for k in categories]

    bars3 = ax3.bar(categories, means, yerr=stds, capsize=5,
                    color=[colors[2], colors[1]], edgecolor="black", linewidth=1.2)
    ax3.axhline(y=50, color="gray", linestyle="--", linewidth=1, label="50% baseline")
    ax3.set_title("By Oracle Type", fontweight="bold")
    ax3.set_ylim(0, 80)

    for bar, mean in zip(bars3, means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{mean:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()

    # Save plot
    output_path = output_dir / "agent_objective_inference.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")

    # Also create a summary bar chart
    fig2, ax = plt.subplots(figsize=(12, 5))

    all_categories = [
        "Simple\nInterest", "Complex\nVF",
        "Blind", "IDs\nVisible", "Interests\nVisible",
        "Strategic\nOracle", "Random\nOracle"
    ]
    all_means = [
        agent_vf_data["Simple Interest"]["mean"],
        agent_vf_data["Complex VF"]["mean"],
        observer_data["Blind"]["mean"],
        observer_data["IDs Visible"]["mean"],
        observer_data["Interests Visible"]["mean"],
        oracle_data["Strategic"]["mean"],
        oracle_data["Random"]["mean"],
    ]
    all_stds = [
        agent_vf_data["Simple Interest"]["std"],
        agent_vf_data["Complex VF"]["std"],
        observer_data["Blind"]["std"],
        observer_data["IDs Visible"]["std"],
        observer_data["Interests Visible"]["std"],
        oracle_data["Strategic"]["std"],
        oracle_data["Random"]["std"],
    ]

    # Color by category
    bar_colors = [
        "#2ecc71", "#e74c3c",  # Agent VF type (green, red)
        "#3498db", "#3498db", "#3498db",  # Observer condition (blue)
        "#9b59b6", "#9b59b6",  # Oracle type (purple)
    ]

    bars = ax.bar(all_categories, all_means, yerr=all_stds, capsize=4,
                  color=bar_colors, edgecolor="black", linewidth=1)

    ax.axhline(y=50, color="gray", linestyle="--", linewidth=1.5, label="50% baseline")
    ax.set_ylabel("Agent Objective Inference Accuracy (%)", fontsize=12)
    ax.set_title("Ability to Infer Agent Goals/Interests\n(Estimator Performance)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 80)

    # Add value labels
    for bar, mean in zip(bars, all_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f"{mean:.1f}%", ha="center", fontsize=9, fontweight="bold")

    # Add category separators
    ax.axvline(x=1.5, color="lightgray", linestyle="-", linewidth=0.5)
    ax.axvline(x=4.5, color="lightgray", linestyle="-", linewidth=0.5)

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)

    output_path2 = output_dir / "agent_objective_inference_summary.png"
    plt.savefig(output_path2, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path2}")

    plt.close("all")

    return output_path, output_path2


if __name__ == "__main__":
    paths = create_plots()
    print(f"\nPlots created successfully!")
    print(f"Key finding: Simple agent interests are much easier to infer (65.8%)")
    print(f"             Complex value functions are harder to detect (35.3%)")
