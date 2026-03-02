#!/usr/bin/env python3
"""Experiment: Does Agent ID Help Objective Inference?

Research Question:
Does knowing which agent said what (agent IDs) help an external estimator
infer agents' hidden objectives? How does knowing agent interests compare?

Conditions:
1. blind: Statements shown without any agent attribution
2. ids: Statements labeled with agent ID (e.g., "Agent_A: ...")
3. interests: Statements labeled with agent ID + known interests

Design:
- 3 conditions × 10 seeds = 30 games
- Fixed: L3 complexity, oracle budget 4, 10 rounds, principled inference
- Primary metric: Exact F1 on objective inference
"""

import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import wandb
from src.environment.simulation import GameConfig, HiddenValueGame

# ============================================================================
# Experimental Configuration
# ============================================================================

SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]

CONDITIONS = ["blind", "ids", "interests"]

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
    "agent_value_function_complexity": "L3",
    "objective_inference_mode": "principled",
    "enable_agent_thinking": True,
    "agent_thinking_budget": 5000,
    "estimator_sees_agent_thinking": False,
    "enable_estimator_thinking": True,
    "estimator_thinking_budget": 5000,
    "estimator_deception_strategy": "baseline",
    "estimator_theory_context": "none",
    "estimator_model": "claude-sonnet-4-20250514",
    "agent_model": "claude-sonnet-4-20250514",
    "observer_model": "claude-sonnet-4-20250514",
    "turn_structure": "interleaved",
    "oracle_timing": "before_response",
    "debate_structure": "open",
    "force_oracle": True,
}


def run_single_game(condition: str, seed: int) -> dict:
    """Run a single game with the given information condition and seed."""
    config = GameConfig(
        **BASE_CONFIG,
        condition=condition,
        seed=seed,
    )

    game = HiddenValueGame(config)
    result = game.run()

    # Extract F1 scores
    f1_scores = []
    inf = result.agent_objective_inference or {}
    for agent_id, data in inf.items():
        if isinstance(data, dict) and "overlap_metrics" in data:
            om = data["overlap_metrics"]
            f1_scores.append(om.get("exact_f1", 0))

    avg_f1 = statistics.mean(f1_scores) if f1_scores else 0

    return {
        "condition": condition,
        "seed": seed,
        "estimator_f1": avg_f1,
        "agent_objective_inference": result.agent_objective_inference,
        "agent_objective_scores": result.agent_objective_scores,
        "agent_objective_overall_score": result.agent_objective_overall_score,
        "metrics": result.metrics,
    }


def safe_stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"mean": 0, "std": 0, "n": 0}
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0
    return {"mean": mean, "std": std, "n": len(values)}


def find_latest_run_dir(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return sorted(subdirs)[-1]


def load_existing_results(output_dir: Path) -> tuple[list[dict], dict[str, list[dict]], set[tuple[str, int]]]:
    all_results = []
    condition_results: dict[str, list[dict]] = {c: [] for c in CONDITIONS}
    completed_games: set[tuple[str, int]] = set()

    if not output_dir.exists():
        return all_results, condition_results, completed_games

    for game_file in output_dir.glob("game_*.json"):
        try:
            with open(game_file) as f:
                result = json.load(f)
            condition = result.get("condition", "")
            seed = result.get("seed")
            if condition and seed is not None:
                all_results.append(result)
                if condition in condition_results:
                    condition_results[condition].append(result)
                completed_games.add((condition, seed))
        except Exception as e:
            print(f"Warning: Failed to load {game_file}: {e}")

    return all_results, condition_results, completed_games


def run_experiment(resume_dir: str | None = None):
    base_output_dir = Path("outputs/agent_id_experiment")

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
            total_expected = len(CONDITIONS) * len(SEEDS)
            if len(completed) < total_expected:
                print(f"Found incomplete run at {latest_dir}")
                print(f"  Completed: {len(completed)}/{total_expected} games")
                if sys.stdin.isatty():
                    response = input("Resume this run? [Y/n]: ").strip().lower()
                else:
                    response = ''
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

    total_games = len(CONDITIONS) * len(SEEDS)

    all_results, condition_results, completed_games = load_existing_results(output_dir)
    n_completed = len(completed_games)

    print(f"\n{'='*70}")
    print("Experiment: Does Agent ID Help Objective Inference?")
    print(f"{'='*70}")
    print(f"Conditions: {CONDITIONS}")
    print(f"Seeds per condition: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    if n_completed > 0:
        print(f"Already completed: {n_completed} games (resuming)")
    print(f"{'='*70}\n")

    wandb_run = wandb.init(
        project="truthification",
        name=f"agent-id-experiment-{timestamp}",
        config={
            "experiment": "agent_id_effect",
            "conditions": CONDITIONS,
            "seeds": SEEDS,
            "total_games": total_games,
            **BASE_CONFIG,
        },
        resume="allow",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results/agent_id_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    game_count = n_completed
    start_time = time.time()

    for condition in CONDITIONS:
        print(f"\n--- Condition: {condition} ---")

        for seed in SEEDS:
            game_count += 1

            if (condition, seed) in completed_games:
                print(f"  [{game_count}/{total_games}] seed={seed} - SKIPPED (already done)")
                continue

            elapsed = time.time() - start_time
            games_run = game_count - n_completed
            eta = (elapsed / games_run) * (total_games - game_count) if games_run > 0 else 0

            print(f"  [{game_count}/{total_games}] seed={seed} (ETA: {eta/60:.1f}m)...", end=" ", flush=True)

            try:
                game_start = time.time()
                result = run_single_game(condition, seed)
                game_elapsed = time.time() - game_start

                all_results.append(result)
                condition_results[condition].append(result)

                game_file = output_dir / f"game_{condition}_seed{seed}.json"
                with open(game_file, "w") as f:
                    json.dump(result, f, indent=2)

                print(f"done ({game_elapsed:.0f}s) - F1: {result['estimator_f1']*100:.1f}%")

                wandb.log({
                    "game_number": game_count,
                    "condition": condition,
                    "seed": seed,
                    "game_time_seconds": game_elapsed,
                    "estimator_f1": result["estimator_f1"],
                })

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    total_elapsed = time.time() - start_time

    # Compute stats
    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")

    condition_stats = {}
    for condition in CONDITIONS:
        results = condition_results.get(condition, [])
        f1_values = [r["estimator_f1"] for r in results]
        condition_stats[condition] = safe_stats(f1_values)

    # Print summary
    print(f"\n{'Condition':<14} | {'Exact F1':<22} | {'N':<4}")
    print("-" * 50)

    baseline_f1 = condition_stats.get("blind", {}).get("mean", 0)
    for condition in CONDITIONS:
        s = condition_stats[condition]
        diff = s["mean"] - baseline_f1 if condition != "blind" else 0
        diff_str = f" ({diff*100:+.1f}%)" if condition != "blind" else " (baseline)"
        print(f"{condition:<14} | {s['mean']*100:>5.1f}% ± {s['std']*100:>5.1f}%{diff_str:<12} | {s['n']}")

    # Statistical comparisons
    blind_scores = [r["estimator_f1"] for r in condition_results.get("blind", [])]
    for condition in ["ids", "interests"]:
        scores = [r["estimator_f1"] for r in condition_results.get(condition, [])]
        if blind_scores and scores:
            try:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(scores, blind_scores)
                diff = statistics.mean(scores) - statistics.mean(blind_scores)
                print(f"\n{condition} vs blind: diff={diff*100:+.1f}%, t={t_stat:.2f}, p={p_value:.4f}")
            except ImportError:
                pass

    # Save results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / "condition_stats.json", "w") as f:
        json.dump(condition_stats, f, indent=2, default=str)

    # Save to results dir
    with open(results_dir / "condition_stats.json", "w") as f:
        json.dump(condition_stats, f, indent=2, default=str)

    # Create summary markdown
    summary = create_summary_markdown(condition_stats, condition_results, total_elapsed, len(all_results))
    with open(output_dir / "README.md", "w") as f:
        f.write(summary)
    with open(results_dir / "README.md", "w") as f:
        f.write(summary)

    # Log to wandb
    summary_rows = []
    for condition in CONDITIONS:
        s = condition_stats[condition]
        summary_rows.append([condition, s["mean"], s["std"], s["n"]])

    wandb.log({
        "summary_table": wandb.Table(
            columns=["condition", "exact_f1_mean", "exact_f1_std", "n"],
            data=summary_rows,
        ),
        "total_runtime_minutes": total_elapsed / 60,
    })

    wandb.finish()

    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")

    return condition_stats


def create_summary_markdown(condition_stats, condition_results, total_elapsed, n_games):
    lines = [
        "# Experiment: Does Agent ID Help Objective Inference?",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Total Games**: {n_games}",
        "",
        "## Research Question",
        "",
        "Does knowing which agent said what (agent IDs) help an external estimator",
        "infer agents' hidden objectives?",
        "",
        "## Conditions",
        "",
        "| Condition | Description |",
        "|-----------|-------------|",
        "| blind | Statements shown without agent attribution |",
        "| ids | Statements labeled with agent ID |",
        "| interests | Statements labeled with agent ID + known interests |",
        "",
        "## Results",
        "",
        "| Condition | Exact F1 (mean) | Std | N |",
        "|-----------|-----------------|-----|---|",
    ]

    for condition in CONDITIONS:
        s = condition_stats[condition]
        lines.append(f"| {condition} | {s['mean']*100:.1f}% | {s['std']*100:.1f}% | {s['n']} |")

    # Statistical comparisons
    blind_scores = [r["estimator_f1"] for r in condition_results.get("blind", [])]
    lines.extend(["", "## Statistical Comparison vs Blind Baseline", ""])

    for condition in ["ids", "interests"]:
        scores = [r["estimator_f1"] for r in condition_results.get(condition, [])]
        if blind_scores and scores:
            diff = statistics.mean(scores) - statistics.mean(blind_scores)
            lines.append(f"- **{condition}** vs blind: {diff*100:+.1f}%")
            try:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(scores, blind_scores)
                lines.append(f"  - t={t_stat:.2f}, p={p_value:.4f}")
                if p_value < 0.05:
                    lines.append(f"  - **SIGNIFICANT (p < 0.05)**")
            except ImportError:
                pass

    lines.extend(["", "## Key Findings", "", "(To be filled after analysis)", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    resume_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_experiment(resume_dir=resume_dir)
