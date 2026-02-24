#!/usr/bin/env python3
"""Generate methodology-focused plots and report for objective inference experiments."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Output directory
REPORT_DIR = Path("results/objective_inference_methodology")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
with open("outputs/theory_context_experiment/20260221_131125/condition_stats.json") as f:
    theory_data = json.load(f)

with open("outputs/deception_strategies_experiment/20260221_110535/condition_stats.json") as f:
    deception_data = json.load(f)

with open("outputs/agent_strategy_inference/20260221_134220/condition_stats.json") as f:
    agent_strategy_data = json.load(f)

with open("outputs/cot_access_experiment/20260220_174154/condition_stats.json") as f:
    cot_data = json.load(f)


def get_se(stats):
    """Calculate standard error from stats dict: SE = std / sqrt(n)"""
    std = stats["std"] * 100
    n = stats["n"]
    return std / np.sqrt(n)


def plot_theory_context():
    """Plot theory context experiment results."""
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = ["none", "brief", "full"]
    labels = ["None", "Brief", "Full"]

    means = [theory_data[c]["stats"]["exact_f1"]["mean"] * 100 for c in conditions]
    ses = [get_se(theory_data[c]["stats"]["exact_f1"]) for c in conditions]

    colors = ['#808080', '#5BA55B', '#2E7D32']
    bars = ax.bar(labels, means, yerr=ses, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Exact F1 Score (%)', fontsize=12)
    ax.set_xlabel('Theory Context Level', fontsize=12)
    ax.set_title('Experiment 1: Theory Context', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 80)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "exp1_theory_context.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_deception_strategies():
    """Plot deception detection strategies results."""
    fig, ax = plt.subplots(figsize=(10, 5))

    strategies = ["baseline", "consistency", "incentive", "pattern", "combined"]
    labels = ["Baseline", "Consistency", "Incentive", "Pattern", "Combined"]

    means = [deception_data[s]["stats"]["exact_f1"]["mean"] * 100 for s in strategies]
    ses = [get_se(deception_data[s]["stats"]["exact_f1"]) for s in strategies]

    colors = ['#808080', '#5BA55B', '#4A90D9', '#E8A838', '#9B59B6']
    bars = ax.bar(labels, means, yerr=ses, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Exact F1 Score (%)', fontsize=12)
    ax.set_xlabel('Estimator Deception Detection Strategy', fontsize=12)
    ax.set_title('Experiment 2: Deception Detection Strategies', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 70)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "exp2_deception_strategies.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_agent_strategies():
    """Plot agent communication strategy results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Sort by F1 score
    strategies = ["aggressive", "honest", "subtle", "natural", "credibility_attack", "deceptive", "misdirection"]
    labels = ["Aggressive", "Honest", "Subtle", "Natural", "Credibility\nAttack", "Deceptive", "Misdirection"]

    means = [agent_strategy_data[s]["stats"]["exact_f1"]["mean"] * 100 for s in strategies]
    ses = [get_se(agent_strategy_data[s]["stats"]["exact_f1"]) for s in strategies]

    colors = ['#2E7D32', '#5BA55B', '#A5D6A7', '#808080', '#FFAB91', '#EF5350', '#C62828']
    bars = ax.bar(labels, means, yerr=ses, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Exact F1 Score (%)', fontsize=12)
    ax.set_xlabel('Agent Communication Strategy', fontsize=12)
    ax.set_title('Experiment 3: Agent Communication Strategy', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 80)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "exp3_agent_strategies.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_cot_access():
    """Plot CoT access experiment results."""
    fig, ax = plt.subplots(figsize=(6, 5))

    conditions = ["without_cot", "with_cot"]
    labels = ["Without CoT", "With CoT"]

    means = [cot_data[c]["stats"]["exact_f1"]["mean"] * 100 for c in conditions]
    ses = [get_se(cot_data[c]["stats"]["exact_f1"]) for c in conditions]

    colors = ['#808080', '#2E7D32']
    bars = ax.bar(labels, means, yerr=ses, capsize=5, color=colors, edgecolor='black', linewidth=1.2)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Exact F1 Score (%)', fontsize=12)
    ax.set_xlabel('Estimator Access to Agent Thinking', fontsize=12)
    ax.set_title('Experiment 4: Chain-of-Thought Access', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)

    # Add significance annotation
    ax.annotate('p < 0.001 ***', xy=(0.5, 85), ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "exp4_cot_access.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_results():
    """Plot all four experiments side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # Experiment 1: Theory Context
    ax = axes[0]
    conditions = ["none", "brief", "full"]
    means = [theory_data[c]["stats"]["exact_f1"]["mean"] * 100 for c in conditions]
    ses = [get_se(theory_data[c]["stats"]["exact_f1"]) for c in conditions]
    colors = ['#808080', '#5BA55B', '#2E7D32']
    bars = ax.bar(["None", "Brief", "Full"], means, yerr=ses, capsize=4, color=colors, edgecolor='black')
    ax.set_ylabel('Exact F1 (%)')
    ax.set_title('Exp 1: Theory Context', fontweight='bold')
    ax.set_ylim(0, 100)
    for i, m in enumerate(means):
        ax.text(i, m + 4, f'{m:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Experiment 2: Deception Strategies
    ax = axes[1]
    strategies = ["baseline", "consistency", "incentive", "pattern", "combined"]
    means = [deception_data[s]["stats"]["exact_f1"]["mean"] * 100 for s in strategies]
    ses = [get_se(deception_data[s]["stats"]["exact_f1"]) for s in strategies]
    colors = ['#808080', '#5BA55B', '#4A90D9', '#E8A838', '#9B59B6']
    bars = ax.bar(["Base", "Consist", "Incent", "Pattern", "Comb"], means, yerr=ses, capsize=4, color=colors, edgecolor='black')
    ax.set_title('Exp 2: Deception Detection', fontweight='bold')
    ax.set_ylim(0, 100)
    for i, m in enumerate(means):
        ax.text(i, m + 4, f'{m:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Experiment 3: Agent Strategies
    ax = axes[2]
    strategies = ["aggressive", "honest", "natural", "deceptive", "misdirection"]
    means = [agent_strategy_data[s]["stats"]["exact_f1"]["mean"] * 100 for s in strategies]
    ses = [get_se(agent_strategy_data[s]["stats"]["exact_f1"]) for s in strategies]
    colors = ['#2E7D32', '#5BA55B', '#808080', '#EF5350', '#C62828']
    bars = ax.bar(["Aggr", "Honest", "Natural", "Decept", "Misdir"], means, yerr=ses, capsize=4, color=colors, edgecolor='black')
    ax.set_title('Exp 3: Agent Strategy', fontweight='bold')
    ax.set_ylim(0, 100)
    for i, m in enumerate(means):
        ax.text(i, m + 4, f'{m:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Experiment 4: CoT Access
    ax = axes[3]
    conditions = ["without_cot", "with_cot"]
    means = [cot_data[c]["stats"]["exact_f1"]["mean"] * 100 for c in conditions]
    ses = [get_se(cot_data[c]["stats"]["exact_f1"]) for c in conditions]
    colors = ['#808080', '#2E7D32']
    bars = ax.bar(["No CoT", "With CoT"], means, yerr=ses, capsize=4, color=colors, edgecolor='black')
    ax.set_title('Exp 4: CoT Access ***', fontweight='bold')
    ax.set_ylim(0, 100)
    for i, m in enumerate(means):
        ax.text(i, m + 4, f'{m:.1f}%', ha='center', fontsize=9, fontweight='bold')

    fig.suptitle('Objective Inference Experiments: Results Summary (error bars = SE)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "all_experiments_summary.png", dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    print("Generating plots with standard error bars...")
    plot_theory_context()
    print("  - exp1_theory_context.png")
    plot_deception_strategies()
    print("  - exp2_deception_strategies.png")
    plot_agent_strategies()
    print("  - exp3_agent_strategies.png")
    plot_cot_access()
    print("  - exp4_cot_access.png")
    plot_all_results()
    print("  - all_experiments_summary.png")
    print(f"\nPlots saved to: {REPORT_DIR}")
