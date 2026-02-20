#!/usr/bin/env python3
"""Generate plots for Agent Objective Inference Experiment Suite."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

OUTPUT_DIR = Path("results/agent_objective_inference/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_search_space_results():
    """Plot accuracy by inference mode."""
    # Data from experiment results
    modes = ['MC-2', 'MC-4', 'MC-8', 'MC-16', 'Freeform', 'Structured']
    accuracies = [90.0, 90.0, 94.9, 95.0, 30.5, 33.8]
    stds = [20.0, 12.9, 10.5, 14.1, 24.1, 13.8]

    # Random baselines for multiple choice
    baselines = [50.0, 25.0, 12.5, 6.25, None, None]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(modes))
    width = 0.6

    # Main bars
    bars = ax.bar(x, accuracies, width, yerr=stds, capsize=5,
                  color=['#2ecc71', '#27ae60', '#1abc9c', '#16a085', '#e74c3c', '#c0392b'],
                  edgecolor='black', linewidth=1.5)

    # Add baseline markers for MC options
    for i, baseline in enumerate(baselines):
        if baseline is not None:
            ax.hlines(baseline, i - width/2, i + width/2, colors='red',
                     linestyles='--', linewidth=2, label='Random baseline' if i == 0 else '')

    ax.set_xlabel('Inference Mode', fontsize=14)
    ax.set_ylabel('Objective Inference Accuracy (%)', fontsize=14)
    ax.set_title('Search Space Constraint: Recognition >> Generation', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 110)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

    ax.legend(loc='upper right')
    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'search_space_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'search_space_accuracy.png'}")


def plot_oracle_budget_results():
    """Plot accuracy by oracle budget."""
    budgets = [0, 1, 2, 4, 6, 8]
    accuracies = [5.0, 14.5, 12.5, 21.0, 27.1, 21.7]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(budgets, accuracies, 'o-', linewidth=2.5, markersize=10,
            color='#3498db', markerfacecolor='white', markeredgewidth=2)

    # Highlight optimal point
    max_idx = np.argmax(accuracies)
    ax.scatter([budgets[max_idx]], [accuracies[max_idx]], s=200, c='#e74c3c',
               zorder=5, marker='*', label=f'Optimal: budget={budgets[max_idx]}')

    ax.set_xlabel('Oracle Budget (queries allowed)', fontsize=14)
    ax.set_ylabel('Objective Inference Accuracy (%)', fontsize=14)
    ax.set_title('Oracle Budget Effect: Diminishing Returns After ~6 Queries', fontsize=16, fontweight='bold')
    ax.set_xticks(budgets)
    ax.set_ylim(0, 35)

    # Add value labels
    for b, acc in zip(budgets, accuracies):
        ax.annotate(f'{acc:.1f}%', xy=(b, acc), xytext=(0, 10),
                   textcoords='offset points', ha='center', fontsize=10)

    ax.legend(loc='upper left')
    ax.axhline(y=5.0, color='gray', linestyle='--', alpha=0.5, label='No oracle baseline')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'oracle_budget_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'oracle_budget_accuracy.png'}")


def plot_complexity_results():
    """Plot accuracy by complexity level."""
    levels = ['L1\nSimple', 'L2\nDual', 'L3\nCombo', 'L4\nComplex', 'L5\nPenalty']
    accuracies = [39.5, 21.5, 23.0, 17.5, 16.4]
    n_conditions = [1.0, 2.0, 2.5, 3.5, 4.5]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = np.arange(len(levels))

    # Accuracy bars
    bars = ax1.bar(x, accuracies, 0.6, color='#9b59b6', edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Objective Complexity Level', fontsize=14)
    ax1.set_ylabel('Objective Inference Accuracy (%)', fontsize=14, color='#9b59b6')
    ax1.set_title('Complexity Effect: Simpler Objectives are 2.4x Easier to Infer', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.set_ylim(0, 50)
    ax1.tick_params(axis='y', labelcolor='#9b59b6')

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax1.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')

    # Secondary axis for number of conditions
    ax2 = ax1.twinx()
    ax2.plot(x, n_conditions, 'o--', color='#e67e22', linewidth=2, markersize=8, label='Avg conditions')
    ax2.set_ylabel('Average Number of Conditions', fontsize=14, color='#e67e22')
    ax2.tick_params(axis='y', labelcolor='#e67e22')
    ax2.set_ylim(0, 6)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'complexity_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'complexity_accuracy.png'}")


def plot_strategy_distribution():
    """Plot strategy prevalence heatmap."""
    strategies = [
        'Object\nAdvocacy',
        'Truth Mixed\nwith Lies',
        'Escalating\nComplexity',
        'Credibility\nAttack',
        'Fabricated\nTerminology',
        'Oracle\nSpin'
    ]
    prevalence = [100.0, 100.0, 98.1, 96.2, 90.6, 81.1]
    mean_confidence = [96.2, 90.9, 82.9, 80.9, 84.4, 69.4]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax.bar(x - width/2, prevalence, width, label='Prevalence (% agents using)',
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, mean_confidence, width, label='Mean Confidence (0-100)',
                   color='#e74c3c', edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Manipulation Strategy', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Agent Manipulation Strategies: All Agents Use Multiple Tactics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')

    # Add value labels
    for bar in bars1:
        ax.annotate(f'{bar.get_height():.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    for bar in bars2:
        ax.annotate(f'{bar.get_height():.0f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'strategy_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'strategy_distribution.png'}")


def plot_summary_comparison():
    """Create a summary plot comparing all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Search Space (top left)
    ax = axes[0, 0]
    modes = ['MC-16', 'MC-8', 'MC-4', 'MC-2', 'Struct.', 'Free']
    accs = [95.0, 94.9, 90.0, 90.0, 33.8, 30.5]
    colors = ['#2ecc71']*4 + ['#e74c3c']*2
    ax.barh(modes, accs, color=colors, edgecolor='black')
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('A. Search Space: MC >> Freeform', fontweight='bold')
    ax.set_xlim(0, 100)
    for i, acc in enumerate(accs):
        ax.text(acc + 2, i, f'{acc:.0f}%', va='center', fontsize=10)

    # 2. Oracle Budget (top right)
    ax = axes[0, 1]
    budgets = [0, 1, 2, 4, 6, 8]
    accs = [5.0, 14.5, 12.5, 21.0, 27.1, 21.7]
    ax.plot(budgets, accs, 'o-', linewidth=2, markersize=8, color='#3498db')
    ax.scatter([6], [27.1], s=150, c='#e74c3c', marker='*', zorder=5)
    ax.set_xlabel('Oracle Budget')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('B. Oracle Budget: Optimal at 6', fontweight='bold')
    ax.set_ylim(0, 35)

    # 3. Complexity (bottom left)
    ax = axes[1, 0]
    levels = ['L1', 'L2', 'L3', 'L4', 'L5']
    accs = [39.5, 21.5, 23.0, 17.5, 16.4]
    ax.bar(levels, accs, color='#9b59b6', edgecolor='black')
    ax.set_xlabel('Complexity Level')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('C. Complexity: Simple > Complex', fontweight='bold')
    for i, acc in enumerate(accs):
        ax.text(i, acc + 1, f'{acc:.0f}%', ha='center', fontsize=10)

    # 4. Strategy (bottom right)
    ax = axes[1, 1]
    strats = ['Obj.Adv.', 'Truth+Lies', 'Escalate', 'Cred.Atk', 'Fabricate', 'Oracle']
    prev = [100.0, 100.0, 98.1, 96.2, 90.6, 81.1]
    ax.barh(strats, prev, color='#e67e22', edgecolor='black')
    ax.set_xlabel('Prevalence (%)')
    ax.set_title('D. Strategies: All Agents Manipulate', fontweight='bold')
    ax.set_xlim(0, 110)
    ax.axvline(x=100, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Agent Objective Inference Experiment Suite Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'summary_all_experiments.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'summary_all_experiments.png'}")


if __name__ == "__main__":
    print("Generating plots for Agent Objective Inference Experiments...\n")

    plot_search_space_results()
    plot_oracle_budget_results()
    plot_complexity_results()
    plot_strategy_distribution()
    plot_summary_comparison()

    print(f"\nAll plots saved to: {OUTPUT_DIR}")
