#!/usr/bin/env python3
"""Generate plots and report for objective inference experiments."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Output directory
REPORT_DIR = Path("results/objective_inference_experiments")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open("outputs/theory_context_experiment/20260221_131125/condition_stats.json") as f:
    theory_data = json.load(f)

with open("outputs/deception_strategies_experiment/20260221_110535/condition_stats.json") as f:
    deception_data = json.load(f)

with open("outputs/agent_strategy_inference/20260221_134220/condition_stats.json") as f:
    agent_strategy_data = json.load(f)


def plot_theory_context():
    """Plot theory context experiment results."""
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = ["none", "brief", "full"]
    labels = ["None\n(baseline)", "Brief\n(~50 words)", "Full\n(~200 words)"]

    means = [theory_data[c]["stats"]["exact_f1"]["mean"] * 100 for c in conditions]
    stds = [theory_data[c]["stats"]["exact_f1"]["std"] * 100 for c in conditions]

    colors = ['#4A90D9', '#5BA55B', '#E8A838']
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Exact F1 Score (%)', fontsize=12)
    ax.set_xlabel('Theory Context Level', fontsize=12)
    ax.set_title('Effect of Theoretical Context on Objective Inference', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')

    # Add improvement annotations
    ax.annotate(f'+{means[1]-means[0]:.1f}%', xy=(1, means[1]), xytext=(1.3, means[1]+10),
                fontsize=10, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
    ax.annotate(f'+{means[2]-means[0]:.1f}%', xy=(2, means[2]), xytext=(2.3, means[2]+5),
                fontsize=10, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "theory_context_results.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_deception_strategies():
    """Plot deception detection strategies results."""
    fig, ax = plt.subplots(figsize=(10, 5))

    strategies = ["baseline", "consistency", "incentive", "pattern", "combined"]
    labels = ["Baseline", "Consistency", "Incentive", "Pattern", "Combined"]

    means = [deception_data[s]["stats"]["exact_f1"]["mean"] * 100 for s in strategies]
    stds = [deception_data[s]["stats"]["exact_f1"]["std"] * 100 for s in strategies]

    colors = ['#808080', '#5BA55B', '#4A90D9', '#E8A838', '#9B59B6']
    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Exact F1 Score (%)', fontsize=12)
    ax.set_xlabel('Deception Detection Strategy', fontsize=12)
    ax.set_title('Effect of Deception Detection Strategies on Objective Inference', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 70)
    ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "deception_strategies_results.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_agent_strategies():
    """Plot agent communication strategy results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by F1 score
    strategies = ["aggressive", "honest", "subtle", "natural", "credibility_attack", "deceptive", "misdirection"]
    labels = ["Aggressive", "Honest", "Subtle", "Natural\n(baseline)", "Credibility\nAttack", "Deceptive", "Misdirection"]

    means = [agent_strategy_data[s]["stats"]["exact_f1"]["mean"] * 100 for s in strategies]
    stds = [agent_strategy_data[s]["stats"]["exact_f1"]["std"] * 100 for s in strategies]

    # Color based on transparency (green = more transparent, red = less transparent)
    baseline_mean = agent_strategy_data["natural"]["stats"]["exact_f1"]["mean"] * 100
    colors = []
    for m in means:
        if m > baseline_mean + 5:
            colors.append('#5BA55B')  # Green - more transparent
        elif m < baseline_mean - 5:
            colors.append('#E74C3C')  # Red - less transparent
        else:
            colors.append('#808080')  # Gray - neutral

    bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Exact F1 Score (%)', fontsize=12)
    ax.set_xlabel('Agent Communication Strategy', fontsize=12)
    ax.set_title('How Agent Strategy Affects Objective Transparency\n(Green = easier to infer, Red = harder to infer)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.axhline(y=baseline_mean, color='gray', linestyle='--', alpha=0.5, label='Baseline (natural)')

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "agent_strategy_results.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison():
    """Plot comparison across all three experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Theory Context
    ax = axes[0]
    conditions = ["none", "brief", "full"]
    means = [theory_data[c]["stats"]["exact_f1"]["mean"] * 100 for c in conditions]
    colors = ['#4A90D9', '#5BA55B', '#E8A838']
    ax.bar(["None", "Brief", "Full"], means, color=colors, edgecolor='black')
    ax.set_ylabel('Exact F1 (%)')
    ax.set_title('Theory Context', fontweight='bold')
    ax.set_ylim(0, 70)
    for i, m in enumerate(means):
        ax.text(i, m + 2, f'{m:.1f}%', ha='center', fontweight='bold')

    # Deception Strategies
    ax = axes[1]
    strategies = ["baseline", "consistency", "combined"]
    means = [deception_data[s]["stats"]["exact_f1"]["mean"] * 100 for s in strategies]
    colors = ['#808080', '#5BA55B', '#9B59B6']
    ax.bar(["Baseline", "Consistency", "Combined"], means, color=colors, edgecolor='black')
    ax.set_title('Deception Detection', fontweight='bold')
    ax.set_ylim(0, 70)
    for i, m in enumerate(means):
        ax.text(i, m + 2, f'{m:.1f}%', ha='center', fontweight='bold')

    # Agent Strategies (best vs worst)
    ax = axes[2]
    strategies = ["aggressive", "natural", "misdirection"]
    means = [agent_strategy_data[s]["stats"]["exact_f1"]["mean"] * 100 for s in strategies]
    colors = ['#5BA55B', '#808080', '#E74C3C']
    ax.bar(["Aggressive", "Natural", "Misdirection"], means, color=colors, edgecolor='black')
    ax.set_title('Agent Strategy', fontweight='bold')
    ax.set_ylim(0, 70)
    for i, m in enumerate(means):
        ax.text(i, m + 2, f'{m:.1f}%', ha='center', fontweight='bold')

    fig.suptitle('Objective Inference Experiments: Key Comparisons', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "comparison_overview.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_effect_sizes():
    """Plot effect sizes (Cohen's d) across experiments."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate effect sizes
    effects = []
    labels = []
    colors = []

    # Theory context
    baseline_mean = theory_data["none"]["stats"]["exact_f1"]["mean"]
    baseline_std = theory_data["none"]["stats"]["exact_f1"]["std"]

    for name, label in [("brief", "Theory: Brief"), ("full", "Theory: Full")]:
        mean = theory_data[name]["stats"]["exact_f1"]["mean"]
        std = theory_data[name]["stats"]["exact_f1"]["std"]
        pooled_std = np.sqrt((baseline_std**2 + std**2) / 2)
        d = (mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        effects.append(d)
        labels.append(label)
        colors.append('#4A90D9')

    # Deception strategies
    baseline_mean = deception_data["baseline"]["stats"]["exact_f1"]["mean"]
    baseline_std = deception_data["baseline"]["stats"]["exact_f1"]["std"]

    for name, label in [("consistency", "Deception: Consistency"), ("combined", "Deception: Combined")]:
        mean = deception_data[name]["stats"]["exact_f1"]["mean"]
        std = deception_data[name]["stats"]["exact_f1"]["std"]
        pooled_std = np.sqrt((baseline_std**2 + std**2) / 2)
        d = (mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        effects.append(d)
        labels.append(label)
        colors.append('#5BA55B')

    # Agent strategies
    baseline_mean = agent_strategy_data["natural"]["stats"]["exact_f1"]["mean"]
    baseline_std = agent_strategy_data["natural"]["stats"]["exact_f1"]["std"]

    for name, label in [("aggressive", "Agent: Aggressive"), ("honest", "Agent: Honest"),
                        ("misdirection", "Agent: Misdirection"), ("deceptive", "Agent: Deceptive")]:
        mean = agent_strategy_data[name]["stats"]["exact_f1"]["mean"]
        std = agent_strategy_data[name]["stats"]["exact_f1"]["std"]
        pooled_std = np.sqrt((baseline_std**2 + std**2) / 2)
        d = (mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        effects.append(d)
        labels.append(label)
        colors.append('#E8A838' if d > 0 else '#E74C3C')

    # Sort by effect size
    sorted_data = sorted(zip(effects, labels, colors), reverse=True)
    effects, labels, colors = zip(*sorted_data)

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, effects, color=colors, edgecolor='black')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title("Effect Sizes Across All Experiments\n(Positive = helps inference, Negative = hinders inference)",
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=0.2, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=-0.2, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=-0.8, color='gray', linestyle=':', alpha=0.5)

    # Add effect size labels
    ax.text(0.2, len(labels), 'small', ha='center', fontsize=8, color='gray')
    ax.text(0.5, len(labels), 'medium', ha='center', fontsize=8, color='gray')
    ax.text(0.8, len(labels), 'large', ha='center', fontsize=8, color='gray')

    for i, (bar, effect) in enumerate(zip(bars, effects)):
        ax.text(effect + 0.05 if effect > 0 else effect - 0.15, i, f'{effect:.2f}',
                va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "effect_sizes.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating plots...")
    plot_theory_context()
    print("  - theory_context_results.png")
    plot_deception_strategies()
    print("  - deception_strategies_results.png")
    plot_agent_strategies()
    print("  - agent_strategy_results.png")
    plot_comparison()
    print("  - comparison_overview.png")
    plot_effect_sizes()
    print("  - effect_sizes.png")
    print(f"\nPlots saved to: {REPORT_DIR}")
