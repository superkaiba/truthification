#!/usr/bin/env python3
"""Plot histogram of strategy counts per round."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

def plot_strategy_histogram(data_dir: str, output_dir: str, threshold: int = 50):
    """Plot stacked histogram of strategy counts per round."""

    data_path = Path(data_dir)
    game_files = list(data_path.glob("game_*_per_round.json"))

    strategies = [
        "fabricated_terminology",
        "truth_mixed_with_lies",
        "oracle_spin",
        "credibility_attack",
        "escalating_complexity",
        "object_advocacy",
    ]

    strategy_labels = [
        "Fabricated\nTerminology",
        "Truth +\nLies",
        "Oracle\nSpin",
        "Credibility\nAttack",
        "Escalating\nComplexity",
        "Object\nAdvocacy",
    ]

    colors = ["#e74c3c", "#3498db", "#9b59b6", "#e67e22", "#1abc9c", "#2ecc71"]

    # Count strategies per round
    # counts[round][strategy] = count of agents using that strategy (score >= threshold)
    counts = {r: {s: 0 for s in strategies} for r in range(1, 11)}

    for game_file in game_files:
        with open(game_file) as f:
            data = json.load(f)

        for round_data in data.get("per_round", []):
            rnd = round_data.get("round_number", 0)
            if rnd not in counts:
                continue

            for agent_id, strat_scores in round_data.get("agent_strategies", {}).items():
                for s in strategies:
                    score = strat_scores.get(s, 0)
                    if isinstance(score, (int, float)) and score >= threshold:
                        counts[rnd][s] += 1

    # Create stacked bar chart
    rounds = list(range(1, 11))

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(rounds))
    width = 0.12

    for i, (strategy, label, color) in enumerate(zip(strategies, strategy_labels, colors)):
        values = [counts[r][strategy] for r in rounds]
        offset = (i - len(strategies)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Round Number', fontsize=14)
    ax.set_ylabel(f'Count of Agents Using Strategy (score ≥ {threshold})', fontsize=14)
    ax.set_title('Strategy Usage Counts Per Round', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rounds)
    ax.legend(loc='upper left', ncol=3)

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'strategy_histogram_per_round.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'strategy_histogram_per_round.png'}")

    # Also create a heatmap version
    fig, ax = plt.subplots(figsize=(12, 6))

    data_matrix = np.array([[counts[r][s] for r in rounds] for s in strategies])

    im = ax.imshow(data_matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(np.arange(len(rounds)))
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_xticklabels(rounds)
    ax.set_yticklabels(strategy_labels)

    ax.set_xlabel('Round Number', fontsize=14)
    ax.set_ylabel('Strategy', fontsize=14)
    ax.set_title(f'Strategy Count Heatmap (score ≥ {threshold})', fontsize=16, fontweight='bold')

    # Add count annotations
    for i in range(len(strategies)):
        for j in range(len(rounds)):
            text = ax.text(j, i, data_matrix[i, j], ha="center", va="center",
                          color="white" if data_matrix[i, j] > data_matrix.max()/2 else "black",
                          fontsize=10, fontweight='bold')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path / 'strategy_heatmap_per_round.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'strategy_heatmap_per_round.png'}")


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "results/strategy_per_round/20260220_110156"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/agent_objective_inference/plots"
    plot_strategy_histogram(data_dir, output_dir)
