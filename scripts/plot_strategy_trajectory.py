#!/usr/bin/env python3
"""Plot strategy evolution over rounds."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

def plot_strategy_trajectory(stats_file: str, output_dir: str):
    """Plot strategy evolution across rounds."""

    with open(stats_file) as f:
        stats = json.load(f)

    trajectory = stats["trajectory"]
    rounds = sorted([int(r) for r in trajectory.keys()])

    strategies = {
        "fabricated_terminology": ("Fabricated Terminology", "#e74c3c"),
        "truth_mixed_with_lies": ("Truth + Lies", "#3498db"),
        "oracle_spin": ("Oracle Spin", "#9b59b6"),
        "credibility_attack": ("Credibility Attack", "#e67e22"),
        "escalating_complexity": ("Escalating Complexity", "#1abc9c"),
        "object_advocacy": ("Object Advocacy", "#2ecc71"),
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    for strategy_key, (label, color) in strategies.items():
        values = [trajectory[str(r)][strategy_key] for r in rounds]
        ax.plot(rounds, values, 'o-', label=label, color=color, linewidth=2, markersize=6)

    ax.set_xlabel('Round Number', fontsize=14)
    ax.set_ylabel('Mean Confidence Score (0-100)', fontsize=14)
    ax.set_title('Strategy Evolution Over Rounds', fontsize=16, fontweight='bold')
    ax.set_xticks(rounds)
    ax.set_ylim(0, 100)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path / 'strategy_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path / 'strategy_trajectory.png'}")


if __name__ == "__main__":
    import sys
    stats_file = sys.argv[1] if len(sys.argv) > 1 else "results/strategy_per_round/20260220_110156/trajectory_stats.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results/agent_objective_inference/plots"
    plot_strategy_trajectory(stats_file, output_dir)
