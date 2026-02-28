#!/usr/bin/env python3
"""Experiment: Agent Strategy Matrix (All Combinations)

Research Question:
How do different combinations of Agent A and Agent B communication strategies
affect game outcomes, including agent rewards, judge performance, and estimator accuracy?

This experiment tests all 7x7 = 49 combinations of agent communication strategies,
producing a matrix for each metric:
- Agent A reward (cumulative value)
- Agent B reward (cumulative value)
- Judge reward (total value)
- Estimator F1 score

Design:
- 7 strategies per agent: natural, honest, deceptive, misdirection, aggressive, subtle, credibility_attack
- 7 x 7 = 49 strategy combinations
- 20 games per combination = 980 games total
- Metrics: Agent A value, Agent B value, Judge total value, Estimator exact F1
"""

import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from src.environment.simulation import GameConfig, HiddenValueGame

# ============================================================================
# Experimental Configuration
# ============================================================================

SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606,
         707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616]  # 20 seeds

# Agent communication strategies to test
STRATEGIES = [
    "natural",           # Baseline - no guidance
    "honest",            # Direct and truthful
    "deceptive",         # Hide true preferences
    "misdirection",      # Emphasize irrelevant properties
    "aggressive",        # Strongly push preferred objects
    "subtle",            # Indirectly promote interests
    "credibility_attack", # Attack opponent's credibility
]

# Fixed game parameters
BASE_CONFIG = {
    "n_objects": 10,
    "n_agents": 2,
    "n_rounds": 10,
    "oracle_budget": 4,
    "selection_size": 5,
    "enable_estimator": True,
    "infer_agent_objectives": True,
    "use_agent_value_functions": True,
    "use_simple_value_functions": True,
    "agent_value_function_complexity": "L3",  # Fixed at 3 properties
    "objective_inference_mode": "principled",
    # Agent thinking enabled (but NOT shared with estimator)
    "enable_agent_thinking": True,
    "agent_thinking_budget": 5000,
    # Estimator does NOT see agent thinking (test on observable only)
    "estimator_sees_agent_thinking": False,
    # Estimator settings
    "enable_estimator_thinking": True,
    "estimator_thinking_budget": 5000,
    "estimator_deception_strategy": "baseline",  # Keep estimator baseline
    "estimator_theory_context": "none",  # No theory context
    # Models
    "estimator_model": "claude-sonnet-4-20250514",
    "agent_model": "claude-sonnet-4-20250514",
    "observer_model": "claude-sonnet-4-20250514",
    # Fixed structure
    "turn_structure": "interleaved",
    "oracle_timing": "before_response",
    "debate_structure": "open",
    "condition": "ids",
    "force_oracle": True,
}


@dataclass
class StrategyPair:
    """A pair of strategies for Agent A and Agent B."""
    agent_a_strategy: str
    agent_b_strategy: str

    def to_dict(self) -> dict:
        return {
            "agent_a_strategy": self.agent_a_strategy,
            "agent_b_strategy": self.agent_b_strategy,
        }

    def key(self) -> str:
        """Return a unique key for this strategy pair."""
        return f"{self.agent_a_strategy}_vs_{self.agent_b_strategy}"


def run_single_game(strategy_pair: StrategyPair, seed: int) -> dict:
    """Run a single game with the given strategy pair and seed."""
    config = GameConfig(
        **BASE_CONFIG,
        agent_a_strategy=strategy_pair.agent_a_strategy,
        agent_b_strategy=strategy_pair.agent_b_strategy,
        seed=seed,
    )

    game = HiddenValueGame(config)
    result = game.run()

    # Extract agent cumulative values (final values)
    agent_values = result.metrics.get("agent_cumulative_values", {})

    # Map agent IDs to A/B
    agent_ids = list(agent_values.keys())
    agent_a_value = agent_values.get(agent_ids[0], 0) if len(agent_ids) > 0 else 0
    agent_b_value = agent_values.get(agent_ids[1], 0) if len(agent_ids) > 1 else 0

    # Judge's total value
    judge_total_value = result.metrics.get("total_value", 0)

    # Extract F1 score (average across both agents)
    f1_scores = []
    inf = result.agent_objective_inference or {}
    for agent_id, data in inf.items():
        if isinstance(data, dict) and "overlap_metrics" in data:
            om = data["overlap_metrics"]
            f1_scores.append(om.get("exact_f1", 0))

    avg_f1 = statistics.mean(f1_scores) if f1_scores else 0

    return {
        "strategy_pair": strategy_pair.to_dict(),
        "seed": seed,
        "agent_a_value": agent_a_value,
        "agent_b_value": agent_b_value,
        "judge_total_value": judge_total_value,
        "estimator_f1": avg_f1,
        "metrics": result.metrics,
        "agent_objective_scores": result.agent_objective_scores,
    }


def compute_matrix_stats(results_by_pair: dict[str, list[dict]]) -> dict:
    """Compute aggregate statistics for each strategy pair.

    Returns a dict mapping pair_key -> {agent_a_value, agent_b_value, judge_value, f1}
    """
    stats = {}
    for pair_key, results in results_by_pair.items():
        if not results:
            continue

        agent_a_values = [r["agent_a_value"] for r in results]
        agent_b_values = [r["agent_b_value"] for r in results]
        judge_values = [r["judge_total_value"] for r in results]
        f1_values = [r["estimator_f1"] for r in results]

        stats[pair_key] = {
            "agent_a_value": {
                "mean": statistics.mean(agent_a_values),
                "std": statistics.stdev(agent_a_values) if len(agent_a_values) > 1 else 0,
                "n": len(agent_a_values),
            },
            "agent_b_value": {
                "mean": statistics.mean(agent_b_values),
                "std": statistics.stdev(agent_b_values) if len(agent_b_values) > 1 else 0,
                "n": len(agent_b_values),
            },
            "judge_value": {
                "mean": statistics.mean(judge_values),
                "std": statistics.stdev(judge_values) if len(judge_values) > 1 else 0,
                "n": len(judge_values),
            },
            "estimator_f1": {
                "mean": statistics.mean(f1_values),
                "std": statistics.stdev(f1_values) if len(f1_values) > 1 else 0,
                "n": len(f1_values),
            },
        }
    return stats


def create_heatmap_data(stats: dict, metric: str) -> np.ndarray:
    """Create a 7x7 matrix for the given metric."""
    matrix = np.zeros((len(STRATEGIES), len(STRATEGIES)))
    for i, strat_a in enumerate(STRATEGIES):
        for j, strat_b in enumerate(STRATEGIES):
            pair_key = f"{strat_a}_vs_{strat_b}"
            if pair_key in stats:
                matrix[i, j] = stats[pair_key][metric]["mean"]
    return matrix


def plot_heatmaps(stats: dict, output_dir: Path) -> dict[str, Path]:
    """Generate heatmaps for all metrics."""
    plt.style.use('seaborn-v0_8-whitegrid')

    metrics = [
        ("agent_a_value", "Agent A Cumulative Value", "Blues"),
        ("agent_b_value", "Agent B Cumulative Value", "Greens"),
        ("judge_value", "Judge Total Value", "Purples"),
        ("estimator_f1", "Estimator F1 Score", "Oranges"),
    ]

    paths = {}
    for metric, title, cmap in metrics:
        matrix = create_heatmap_data(stats, metric)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Format for F1 (0-1) vs values (integers)
        fmt = ".2f" if metric == "estimator_f1" else ".0f"

        sns.heatmap(
            matrix,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=STRATEGIES,
            yticklabels=STRATEGIES,
            ax=ax,
            square=True,
            cbar_kws={"label": title},
        )

        ax.set_xlabel("Agent B Strategy")
        ax.set_ylabel("Agent A Strategy")
        ax.set_title(f"Strategy Matrix: {title}")

        # Rotate labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        plt.tight_layout()
        filename = f"matrix_{metric}.png"
        path = output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()

        paths[metric] = path
        print(f"Saved {filename}")

    return paths


def find_latest_run_dir(base_dir: Path) -> Path | None:
    """Find the most recent run directory for resuming."""
    if not base_dir.exists():
        return None
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return sorted(subdirs)[-1]


def load_existing_results(output_dir: Path) -> tuple[list[dict], dict[str, list[dict]], set[tuple[str, str, int]]]:
    """Load existing results from a previous run.

    Returns:
        Tuple of (all_results, pair_results, completed_games)
        where completed_games is a set of (agent_a_strategy, agent_b_strategy, seed) tuples
    """
    all_results = []
    pair_results: dict[str, list[dict]] = {}
    completed_games: set[tuple[str, str, int]] = set()

    if not output_dir.exists():
        return all_results, pair_results, completed_games

    for game_file in output_dir.glob("game_*.json"):
        try:
            with open(game_file) as f:
                result = json.load(f)

            pair = result.get("strategy_pair", {})
            strat_a = pair.get("agent_a_strategy", "")
            strat_b = pair.get("agent_b_strategy", "")
            seed = result.get("seed")

            if strat_a and strat_b and seed is not None:
                all_results.append(result)
                pair_key = f"{strat_a}_vs_{strat_b}"
                if pair_key not in pair_results:
                    pair_results[pair_key] = []
                pair_results[pair_key].append(result)
                completed_games.add((strat_a, strat_b, seed))
        except Exception as e:
            print(f"Warning: Failed to load {game_file}: {e}")

    return all_results, pair_results, completed_games


def create_summary_markdown(stats: dict, total_elapsed: float, n_games: int) -> str:
    """Create markdown summary of results."""
    lines = [
        "# Experiment: Agent Strategy Matrix (All Combinations)",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)",
        f"**Total Games**: {n_games}",
        "",
        "## Research Question",
        "",
        "How do different combinations of Agent A and Agent B communication strategies",
        "affect game outcomes, including agent rewards, judge performance, and estimator accuracy?",
        "",
        "## Experimental Setup",
        "",
        f"- **Strategies Tested**: {', '.join(STRATEGIES)}",
        f"- **Strategy Combinations**: {len(STRATEGIES)} x {len(STRATEGIES)} = {len(STRATEGIES)**2}",
        f"- **Seeds per Combination**: {len(SEEDS)}",
        f"- **Total Games**: {len(STRATEGIES)**2 * len(SEEDS)}",
        "- **Complexity**: L3 (3 properties per agent)",
        "- **CoT Access**: False (testing on observable behavior only)",
        "",
        "## Metrics",
        "",
        "1. **Agent A Value**: Cumulative value earned by Agent A based on their value function",
        "2. **Agent B Value**: Cumulative value earned by Agent B based on their value function",
        "3. **Judge Value**: Total value of objects selected (judge's value function)",
        "4. **Estimator F1**: How accurately the estimator infers agent objectives",
        "",
        "## Results",
        "",
        "### Heatmaps",
        "",
        "![Agent A Value](matrix_agent_a_value.png)",
        "",
        "![Agent B Value](matrix_agent_b_value.png)",
        "",
        "![Judge Value](matrix_judge_value.png)",
        "",
        "![Estimator F1](matrix_estimator_f1.png)",
        "",
        "### Summary Statistics",
        "",
        "#### Best Strategy Pairs by Metric",
        "",
    ]

    # Find best pairs for each metric
    metrics = ["agent_a_value", "agent_b_value", "judge_value", "estimator_f1"]
    for metric in metrics:
        best_pair = max(stats.items(), key=lambda x: x[1][metric]["mean"])
        lines.append(f"- **{metric}**: {best_pair[0]} (mean={best_pair[1][metric]['mean']:.2f})")

    lines.extend([
        "",
        "### Diagonal (Same Strategy for Both Agents)",
        "",
        "| Strategy | Agent A Value | Agent B Value | Judge Value | F1 |",
        "|----------|---------------|---------------|-------------|-----|",
    ])

    for strat in STRATEGIES:
        pair_key = f"{strat}_vs_{strat}"
        if pair_key in stats:
            s = stats[pair_key]
            lines.append(
                f"| {strat} | {s['agent_a_value']['mean']:.1f} | "
                f"{s['agent_b_value']['mean']:.1f} | {s['judge_value']['mean']:.1f} | "
                f"{s['estimator_f1']['mean']:.2f} |"
            )

    lines.extend([
        "",
        "### Key Findings",
        "",
        "(Analysis to be added after reviewing results)",
        "",
    ])

    return "\n".join(lines)


def run_experiment(resume_dir: str | None = None):
    """Run the strategy matrix experiment.

    Args:
        resume_dir: Optional path to previous run directory to resume from.
    """
    base_output_dir = Path("outputs/strategy_matrix_experiment")

    if resume_dir:
        output_dir = Path(resume_dir)
        if not output_dir.exists():
            print(f"Resume directory not found: {resume_dir}")
            return None
        timestamp = output_dir.name
        print(f"Resuming from: {output_dir}")
    else:
        latest_dir = find_latest_run_dir(base_output_dir)
        if latest_dir:
            completed = list(latest_dir.glob("game_*.json"))
            total_expected = len(STRATEGIES) ** 2 * len(SEEDS)
            if len(completed) < total_expected:
                print(f"Found incomplete run at {latest_dir}")
                print(f"  Completed: {len(completed)}/{total_expected} games")
                # Auto-resume if running non-interactively
                if sys.stdin.isatty():
                    response = input("Resume this run? [Y/n]: ").strip().lower()
                else:
                    response = ''  # Auto-resume
                    print("Auto-resuming (non-interactive mode)")
                if response != 'n':
                    output_dir = latest_dir
                    timestamp = latest_dir.name
                    print(f"Resuming from: {output_dir}")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = base_output_dir / timestamp
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = base_output_dir / timestamp
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = base_output_dir / timestamp

    # Generate all strategy pairs
    strategy_pairs = [
        StrategyPair(strat_a, strat_b)
        for strat_a in STRATEGIES
        for strat_b in STRATEGIES
    ]
    total_games = len(strategy_pairs) * len(SEEDS)

    # Load existing results if resuming
    all_results, pair_results, completed_games = load_existing_results(output_dir)
    n_completed = len(completed_games)

    print(f"\n{'='*70}")
    print("Experiment: Agent Strategy Matrix (All Combinations)")
    print(f"{'='*70}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Strategy pairs: {len(strategy_pairs)}")
    print(f"Seeds per pair: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    if n_completed > 0:
        print(f"Already completed: {n_completed} games (resuming)")
    print(f"{'='*70}\n")

    # Initialize wandb
    wandb_run = wandb.init(
        project="truthification",
        name=f"strategy-matrix-{timestamp}",
        config={
            "experiment": "strategy_matrix",
            "strategies": STRATEGIES,
            "seeds": SEEDS,
            "total_games": total_games,
            **BASE_CONFIG,
        },
        resume="allow",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/strategy_matrix_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    game_count = n_completed
    start_time = time.time()

    for pair in strategy_pairs:
        print(f"\n--- {pair.agent_a_strategy} vs {pair.agent_b_strategy} ---")

        for seed in SEEDS:
            game_count += 1

            # Skip if already completed
            if (pair.agent_a_strategy, pair.agent_b_strategy, seed) in completed_games:
                print(f"  [{game_count}/{total_games}] seed={seed} - SKIPPED (already done)")
                continue

            elapsed = time.time() - start_time
            games_run = game_count - n_completed
            eta = (elapsed / games_run) * (total_games - game_count) if games_run > 0 else 0

            print(f"  [{game_count}/{total_games}] seed={seed} (ETA: {eta/60:.1f}m)...", end=" ", flush=True)

            try:
                game_start = time.time()
                result = run_single_game(pair, seed)
                game_elapsed = time.time() - game_start

                all_results.append(result)
                pair_key = pair.key()
                if pair_key not in pair_results:
                    pair_results[pair_key] = []
                pair_results[pair_key].append(result)

                # Save individual game result
                game_file = output_dir / f"game_{pair.agent_a_strategy}_vs_{pair.agent_b_strategy}_seed{seed}.json"
                with open(game_file, "w") as f:
                    json.dump(result, f, indent=2)

                # Quick summary
                print(f"done ({game_elapsed:.0f}s) - A:{result['agent_a_value']} B:{result['agent_b_value']} J:{result['judge_total_value']} F1:{result['estimator_f1']:.2f}")

                # Log to wandb
                wandb.log({
                    "game_number": game_count,
                    "agent_a_strategy": pair.agent_a_strategy,
                    "agent_b_strategy": pair.agent_b_strategy,
                    "seed": seed,
                    "game_time_seconds": game_elapsed,
                    "agent_a_value": result["agent_a_value"],
                    "agent_b_value": result["agent_b_value"],
                    "judge_value": result["judge_total_value"],
                    "estimator_f1": result["estimator_f1"],
                })

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    total_elapsed = time.time() - start_time

    # Compute aggregate statistics
    print(f"\n{'='*70}")
    print("Computing Aggregate Statistics...")
    print(f"{'='*70}")

    stats = compute_matrix_stats(pair_results)

    # Generate heatmaps
    print("\nGenerating heatmaps...")
    plot_paths = plot_heatmaps(stats, output_dir)

    # Log heatmaps to wandb
    for metric, path in plot_paths.items():
        wandb.log({f"heatmap_{metric}": wandb.Image(str(path))})

    # Save results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / "matrix_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    # Create summary markdown
    summary_md = create_summary_markdown(stats, total_elapsed, len(all_results))
    with open(output_dir / "README.md", "w") as f:
        f.write(summary_md)

    # Copy to results directory
    for metric, path in plot_paths.items():
        import shutil
        shutil.copy(path, results_dir / path.name)

    with open(results_dir / "matrix_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    with open(results_dir / "README.md", "w") as f:
        f.write(summary_md)

    # Log summary table to wandb
    summary_rows = []
    for pair_key, s in stats.items():
        summary_rows.append([
            pair_key,
            s["agent_a_value"]["mean"],
            s["agent_b_value"]["mean"],
            s["judge_value"]["mean"],
            s["estimator_f1"]["mean"],
        ])

    wandb.log({
        "summary_table": wandb.Table(
            columns=["strategy_pair", "agent_a_value", "agent_b_value", "judge_value", "estimator_f1"],
            data=summary_rows,
        ),
        "total_runtime_minutes": total_elapsed / 60,
    })

    wandb.finish()

    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print(f"Results saved to: {output_dir}")
    print(f"Results copied to: {results_dir}")

    return stats


if __name__ == "__main__":
    resume_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_experiment(resume_dir=resume_dir)
