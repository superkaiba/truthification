#!/usr/bin/env python3
"""Experiment: Debate Turn Structures with Comprehensive Metrics.

Research Questions:
- Which debate structure allows the Estimator to most effectively recover truth?
- How do metrics evolve over rounds? (learning curves)
- Does isolating factors reveal cleaner effects?

Experimental Design (Isolated Factors):
- Experiment 1: Turn structure (4 conditions × 4 seeds = 16 games)
  - Varies: interleaved, batch, simultaneous, sequential
  - Fixed: oracle_timing = before_response
- Experiment 2: Oracle timing (2 conditions × 4 seeds = 8 games)
  - Varies: before_response, after_statements
  - Fixed: turn_structure = interleaved

Total: 24 games (vs 32 in full factorial)
"""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import wandb
from src.environment.simulation import GameConfig, HiddenValueGame

# Experimental constants
TURN_STRUCTURES = ["interleaved", "batch", "simultaneous", "sequential"]
ORACLE_TIMINGS = ["before_response", "after_statements"]
SEEDS = [42, 123, 456, 789]

# Baseline configuration
BASELINE_TURN_STRUCTURE = "interleaved"
BASELINE_ORACLE_TIMING = "before_response"

# Game configuration
GAME_CONFIG = {
    "n_objects": 8,
    "n_agents": 2,
    "n_rounds": 3,  # 3 rounds for better accuracy progression tracking
    "oracle_budget": 3,
    "selection_size": 3,
    "enable_estimator": True,
    "estimator_model": "claude-sonnet-4-20250514",
    "agent_model": "claude-sonnet-4-20250514",
    "observer_model": "claude-sonnet-4-20250514",
    "rule_complexity": "medium",
    "condition": "ids",
}


def run_single_game(turn_structure: str, oracle_timing: str, seed: int) -> dict:
    """Run a single game with the given configuration."""
    config = GameConfig(
        **GAME_CONFIG,
        turn_structure=turn_structure,
        oracle_timing=oracle_timing,
        seed=seed,
    )

    game = HiddenValueGame(config)
    result = game.run()

    return {
        "turn_structure": turn_structure,
        "oracle_timing": oracle_timing,
        "seed": seed,
        "metrics": result.metrics,
        "estimator_metrics": result.estimator_metrics,
        "accuracy_progression": result.accuracy_progression,
        "config": result.config,
    }


def generate_isolated_conditions():
    """Generate conditions using isolated factor design.

    Returns:
        List of (turn_structure, oracle_timing, experiment_name) tuples
    """
    conditions = []

    # Experiment 1: Turn Structure (oracle_timing fixed)
    for ts in TURN_STRUCTURES:
        conditions.append((ts, BASELINE_ORACLE_TIMING, "turn_structure"))

    # Experiment 2: Oracle Timing (turn_structure fixed)
    # Note: (interleaved, before_response) is already in Exp 1, skip duplicate
    for ot in ORACLE_TIMINGS:
        if ot != BASELINE_ORACLE_TIMING:  # Skip baseline (already included)
            conditions.append((BASELINE_TURN_STRUCTURE, ot, "oracle_timing"))

    return conditions


def run_experiment():
    """Run the debate structure experiment with isolated factor design."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize wandb
    wandb_run = wandb.init(
        project="truthification",
        name=f"debate-structure-{timestamp}",
        config={
            "experiment": "debate_structure_isolated",
            "turn_structures": TURN_STRUCTURES,
            "oracle_timings": ORACLE_TIMINGS,
            "seeds": SEEDS,
            "design": "isolated_factor",
            **GAME_CONFIG,
        },
    )

    # Output directory
    output_dir = Path("outputs/debate_structure") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/debate_structure")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    condition_results = {}  # {(turn_structure, oracle_timing): [results]}

    # Generate conditions using isolated factor design
    conditions = generate_isolated_conditions()
    total_games = len(conditions) * len(SEEDS)

    print(f"\n{'='*60}")
    print("Debate Structure Experiment (Isolated Factor Design)")
    print(f"{'='*60}")
    print(f"Experiment 1: Turn Structure (oracle_timing={BASELINE_ORACLE_TIMING})")
    print(f"  Conditions: {TURN_STRUCTURES}")
    print(f"Experiment 2: Oracle Timing (turn_structure={BASELINE_TURN_STRUCTURE})")
    print(f"  Conditions: {[ot for ot in ORACLE_TIMINGS if ot != BASELINE_ORACLE_TIMING]}")
    print(f"Seeds per condition: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    print(f"{'='*60}\n")

    game_count = 0
    start_time = time.time()

    for ts, ot, exp_name in conditions:
        key = (ts, ot)
        if key not in condition_results:
            condition_results[key] = []

        print(f"\n--- [{exp_name}] {ts} + {ot} ---")

        for seed in SEEDS:
            game_count += 1
            print(f"  [{game_count}/{total_games}] Running seed={seed}...", end=" ", flush=True)

            try:
                game_start = time.time()
                result = run_single_game(ts, ot, seed)
                game_elapsed = time.time() - game_start

                all_results.append(result)
                condition_results[key].append(result)

                # Extract metrics for logging
                obs_prop_acc = result["metrics"].get("property_accuracy", 0)
                obs_rule_acc = result["metrics"].get("rule_inference_accuracy", 0)
                sel_acc = result["metrics"].get("selection_accuracy", 0)

                est_prop_acc = 0
                est_rule_acc = 0
                if result["estimator_metrics"]:
                    est_prop_acc = result["estimator_metrics"].get("property_accuracy", 0)
                    est_rule_acc = result["estimator_metrics"].get("rule_inference_accuracy", 0)

                # Log per-round accuracy progression
                round_data = {}
                for i, prog in enumerate(result.get("accuracy_progression", [])):
                    round_num = prog.get("round", i + 1)
                    round_data[f"round_{round_num}_judge_prop_acc"] = prog.get("judge_property_accuracy", 0)
                    round_data[f"round_{round_num}_est_prop_acc"] = prog.get("estimator_property_accuracy", 0)
                    round_data[f"round_{round_num}_judge_rule_acc"] = prog.get("judge_rule_accuracy", 0)
                    round_data[f"round_{round_num}_est_rule_acc"] = prog.get("estimator_rule_accuracy", 0)

                # Log to wandb
                wandb.log({
                    "experiment": exp_name,
                    "turn_structure": ts,
                    "oracle_timing": ot,
                    "seed": seed,
                    # Final observer metrics
                    "observer_property_accuracy": obs_prop_acc,
                    "observer_rule_accuracy": obs_rule_acc,
                    "selection_accuracy": sel_acc,
                    # Final estimator metrics
                    "estimator_property_accuracy": est_prop_acc,
                    "estimator_rule_accuracy": est_rule_acc,
                    # Derived metrics
                    "estimator_advantage_property": est_prop_acc - obs_prop_acc,
                    "estimator_advantage_rule": est_rule_acc - obs_rule_acc,
                    "game_time_seconds": game_elapsed,
                    # Per-round accuracy
                    **round_data,
                })

                print(f"done ({game_elapsed:.1f}s) - Est: {est_prop_acc*100:.1f}% props, Obs: {obs_prop_acc*100:.1f}% props")

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    total_elapsed = time.time() - start_time

    # Compute aggregate statistics per condition
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    summary_data = []

    for ts in TURN_STRUCTURES:
        for ot in ORACLE_TIMINGS:
            key = (ts, ot)
            cond_results = condition_results.get(key, [])

            if not cond_results:
                continue

            # Observer metrics
            obs_prop_accs = [r["metrics"].get("property_accuracy", 0) for r in cond_results]
            obs_rule_accs = [r["metrics"].get("rule_inference_accuracy", 0) for r in cond_results]
            sel_accs = [r["metrics"].get("selection_accuracy", 0) for r in cond_results]

            # Estimator metrics
            est_prop_accs = []
            est_rule_accs = []
            for r in cond_results:
                if r["estimator_metrics"]:
                    est_prop_accs.append(r["estimator_metrics"].get("property_accuracy", 0))
                    est_rule_accs.append(r["estimator_metrics"].get("rule_inference_accuracy", 0))

            # Compute means and stds
            obs_prop_mean = statistics.mean(obs_prop_accs) if obs_prop_accs else 0
            obs_prop_std = statistics.stdev(obs_prop_accs) if len(obs_prop_accs) > 1 else 0
            obs_rule_mean = statistics.mean(obs_rule_accs) if obs_rule_accs else 0
            obs_rule_std = statistics.stdev(obs_rule_accs) if len(obs_rule_accs) > 1 else 0
            sel_mean = statistics.mean(sel_accs) if sel_accs else 0
            sel_std = statistics.stdev(sel_accs) if len(sel_accs) > 1 else 0

            est_prop_mean = statistics.mean(est_prop_accs) if est_prop_accs else 0
            est_prop_std = statistics.stdev(est_prop_accs) if len(est_prop_accs) > 1 else 0
            est_rule_mean = statistics.mean(est_rule_accs) if est_rule_accs else 0
            est_rule_std = statistics.stdev(est_rule_accs) if len(est_rule_accs) > 1 else 0

            # Aggregate accuracy progression (average across seeds)
            avg_progression = aggregate_accuracy_progression(cond_results)

            summary_data.append({
                "turn_structure": ts,
                "oracle_timing": ot,
                "n_games": len(cond_results),
                "observer_property_accuracy_mean": obs_prop_mean,
                "observer_property_accuracy_std": obs_prop_std,
                "observer_rule_accuracy_mean": obs_rule_mean,
                "observer_rule_accuracy_std": obs_rule_std,
                "selection_accuracy_mean": sel_mean,
                "selection_accuracy_std": sel_std,
                "estimator_property_accuracy_mean": est_prop_mean,
                "estimator_property_accuracy_std": est_prop_std,
                "estimator_rule_accuracy_mean": est_rule_mean,
                "estimator_rule_accuracy_std": est_rule_std,
                "estimator_advantage_property": est_prop_mean - obs_prop_mean,
                "estimator_advantage_rule": est_rule_mean - obs_rule_mean,
                "accuracy_progression": avg_progression,
            })

    # Print summary tables
    print_summary_tables(summary_data, TURN_STRUCTURES, ORACLE_TIMINGS)

    # Find best/worst conditions for estimator
    if summary_data:
        best_est = max(summary_data, key=lambda x: x["estimator_property_accuracy_mean"])
        worst_est = min(summary_data, key=lambda x: x["estimator_property_accuracy_mean"])

        print(f"\n{'='*60}")
        print(f"Best for Estimator: {best_est['turn_structure']} + {best_est['oracle_timing']} ({best_est['estimator_property_accuracy_mean']*100:.1f}%)")
        print(f"Worst for Estimator: {worst_est['turn_structure']} + {worst_est['oracle_timing']} ({worst_est['estimator_property_accuracy_mean']*100:.1f}%)")
        print(f"{'='*60}")
    print(f"Total runtime: {total_elapsed/60:.1f} minutes")

    # Save results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Create summary markdown
    summary_md = create_summary_markdown(summary_data, total_elapsed, GAME_CONFIG)
    with open(output_dir / "README.md", "w") as f:
        f.write(summary_md)

    # Also save to results directory for version control
    with open(results_dir / f"summary_{timestamp}.json", "w") as f:
        json.dump(summary_data, f, indent=2)
    with open(results_dir / f"README_{timestamp}.md", "w") as f:
        f.write(summary_md)

    # Log summary to wandb
    if summary_data:
        wandb.log({
            "summary_table": wandb.Table(
                columns=list(summary_data[0].keys()),
                data=[list(s.values()) for s in summary_data]
            ),
            "total_runtime_minutes": total_elapsed / 60,
            "best_condition": f"{best_est['turn_structure']}+{best_est['oracle_timing']}",
            "best_estimator_property_accuracy": best_est['estimator_property_accuracy_mean'],
            "worst_condition": f"{worst_est['turn_structure']}+{worst_est['oracle_timing']}",
            "worst_estimator_property_accuracy": worst_est['estimator_property_accuracy_mean'],
        })

    wandb.finish()

    print(f"\nResults saved to: {output_dir}")
    print(f"Summary also saved to: {results_dir}")

    return summary_data


def aggregate_accuracy_progression(results: list[dict]) -> list[dict]:
    """Aggregate accuracy progression across multiple games (seeds).

    Returns list of dicts with mean accuracy per round.
    """
    if not results:
        return []

    # Collect all progressions
    all_progressions = [r.get("accuracy_progression", []) for r in results]
    if not all_progressions or not all_progressions[0]:
        return []

    n_rounds = len(all_progressions[0])
    aggregated = []

    for round_idx in range(n_rounds):
        round_data = {
            "round": round_idx + 1,
            "judge_property_accuracy": 0,
            "judge_rule_accuracy": 0,
            "estimator_property_accuracy": 0,
            "estimator_rule_accuracy": 0,
        }

        valid_count = 0
        for prog in all_progressions:
            if round_idx < len(prog):
                valid_count += 1
                for key in round_data:
                    if key != "round":
                        round_data[key] += prog[round_idx].get(key, 0)

        if valid_count > 0:
            for key in round_data:
                if key != "round":
                    round_data[key] /= valid_count

        aggregated.append(round_data)

    return aggregated


def print_summary_tables(summary_data: list, turn_structures: list, oracle_timings: list):
    """Print formatted summary tables to console."""
    # Print summary table - Estimator Property Accuracy
    print("\n=== Estimator Property Accuracy by Condition ===\n")
    print(f"{'Turn Structure':<15} | {'Oracle Before':<18} | {'Oracle After':<18} | {'Diff':<8}")
    print("-" * 65)

    for ts in turn_structures:
        before_data = next((s for s in summary_data if s["turn_structure"] == ts and s["oracle_timing"] == "before_response"), None)
        after_data = next((s for s in summary_data if s["turn_structure"] == ts and s["oracle_timing"] == "after_statements"), None)

        before_str = f"{before_data['estimator_property_accuracy_mean']*100:.1f}% (+/-{before_data['estimator_property_accuracy_std']*100:.1f})" if before_data else "N/A"
        after_str = f"{after_data['estimator_property_accuracy_mean']*100:.1f}% (+/-{after_data['estimator_property_accuracy_std']*100:.1f})" if after_data else "N/A"

        diff = 0
        if before_data and after_data:
            diff = (after_data['estimator_property_accuracy_mean'] - before_data['estimator_property_accuracy_mean']) * 100
        diff_str = f"{diff:+.1f}%" if (before_data and after_data) else "N/A"

        print(f"{ts:<15} | {before_str:<18} | {after_str:<18} | {diff_str:<8}")

    # Print accuracy progression table
    print("\n=== Accuracy Progression (Average across seeds) ===\n")
    for s in summary_data:
        prog = s.get("accuracy_progression", [])
        if not prog:
            continue
        print(f"\n{s['turn_structure']} + {s['oracle_timing']}:")
        print(f"{'Round':<6} | {'Judge Prop':<12} | {'Est Prop':<12} | {'Judge Rule':<12} | {'Est Rule':<12}")
        print("-" * 60)
        for p in prog:
            print(f"{p['round']:<6} | {p['judge_property_accuracy']*100:>10.1f}% | {p['estimator_property_accuracy']*100:>10.1f}% | {p['judge_rule_accuracy']*100:>10.1f}% | {p['estimator_rule_accuracy']*100:>10.1f}%")


def create_summary_markdown(summary_data: list, total_elapsed: float, game_config: dict) -> str:
    """Create a markdown summary of the experiment."""
    lines = [
        "# Debate Structure Experiment: Comprehensive Metrics",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Design**: Isolated Factor (24 games total)",
        "",
        "## Research Questions",
        "",
        "1. Which debate structure allows the Estimator to most effectively recover truth?",
        "2. How do metrics evolve over rounds? (learning curves)",
        "3. Does isolating factors reveal cleaner effects?",
        "",
        "## Experimental Setup",
        "",
        f"- **Objects**: {game_config['n_objects']}",
        f"- **Agents**: {game_config['n_agents']}",
        f"- **Rounds**: {game_config['n_rounds']}",
        f"- **Oracle Budget**: {game_config['oracle_budget']}",
        f"- **Selection Size**: {game_config['selection_size']}",
        f"- **Seeds**: {SEEDS}",
        "",
        "### Experimental Design",
        "",
        "| Experiment | Varied Factor | Fixed Factor | N Conditions |",
        "|------------|---------------|--------------|--------------|",
        f"| 1: Turn Structure | {', '.join(TURN_STRUCTURES)} | oracle_timing={BASELINE_ORACLE_TIMING} | 4 |",
        f"| 2: Oracle Timing | {', '.join([ot for ot in ORACLE_TIMINGS if ot != BASELINE_ORACLE_TIMING])} | turn_structure={BASELINE_TURN_STRUCTURE} | 1 |",
        "",
        "**Total**: 5 conditions × 4 seeds = 20 games",
        "(Note: baseline condition appears in both experiments)",
        "",
        "### Turn Structures",
        "",
        "| Structure | Description |",
        "|-----------|-------------|",
        "| interleaved | A speaks, B speaks, A speaks, B speaks |",
        "| batch | A says all, B says all |",
        "| simultaneous | Both commit without seeing each other |",
        "| sequential | A states, then B sees A and states |",
        "",
        "### Oracle Timings",
        "",
        "| Timing | Description |",
        "|--------|-------------|",
        "| before_response | Agents see oracle result and can respond |",
        "| after_statements | Oracle query happens, agents cannot adapt |",
        "",
        "## Results",
        "",
        "### Final Estimator Property Accuracy",
        "",
        "| Turn Structure | Oracle Before | Oracle After | Diff |",
        "|----------------|---------------|--------------|------|",
    ]

    for ts in TURN_STRUCTURES:
        before_data = next((s for s in summary_data if s["turn_structure"] == ts and s["oracle_timing"] == "before_response"), None)
        after_data = next((s for s in summary_data if s["turn_structure"] == ts and s["oracle_timing"] == "after_statements"), None)

        before_str = f"{before_data['estimator_property_accuracy_mean']*100:.1f}% (+/-{before_data['estimator_property_accuracy_std']*100:.1f})" if before_data else "N/A"
        after_str = f"{after_data['estimator_property_accuracy_mean']*100:.1f}% (+/-{after_data['estimator_property_accuracy_std']*100:.1f})" if after_data else "N/A"

        diff = 0
        if before_data and after_data:
            diff = (after_data['estimator_property_accuracy_mean'] - before_data['estimator_property_accuracy_mean']) * 100
        diff_str = f"{diff:+.1f}%" if (before_data and after_data) else "N/A"

        lines.append(f"| {ts} | {before_str} | {after_str} | {diff_str} |")

    lines.extend([
        "",
        "### Accuracy Progression Over Rounds",
        "",
    ])

    # Add progression tables for each condition
    for s in summary_data:
        prog = s.get("accuracy_progression", [])
        if not prog:
            continue

        lines.append(f"#### {s['turn_structure']} + {s['oracle_timing']}")
        lines.append("")
        lines.append("| Round | Judge Prop | Est Prop | Judge Rule | Est Rule |")
        lines.append("|-------|------------|----------|------------|----------|")

        for p in prog:
            lines.append(
                f"| {p['round']} | {p['judge_property_accuracy']*100:.1f}% | "
                f"{p['estimator_property_accuracy']*100:.1f}% | "
                f"{p['judge_rule_accuracy']*100:.1f}% | "
                f"{p['estimator_rule_accuracy']*100:.1f}% |"
            )
        lines.append("")

    # Find best/worst
    if summary_data:
        best_est = max(summary_data, key=lambda x: x["estimator_property_accuracy_mean"])
        worst_est = min(summary_data, key=lambda x: x["estimator_property_accuracy_mean"])

        lines.extend([
            "### Key Findings",
            "",
            f"- **Best for Estimator**: {best_est['turn_structure']} + {best_est['oracle_timing']} ({best_est['estimator_property_accuracy_mean']*100:.1f}%)",
            f"- **Worst for Estimator**: {worst_est['turn_structure']} + {worst_est['oracle_timing']} ({worst_est['estimator_property_accuracy_mean']*100:.1f}%)",
            "",
            "### Estimator Advantage (Estimator - Observer)",
            "",
            "| Turn Structure | Oracle Before | Oracle After |",
            "|----------------|---------------|--------------|",
        ])

        for ts in TURN_STRUCTURES:
            before_data = next((s for s in summary_data if s["turn_structure"] == ts and s["oracle_timing"] == "before_response"), None)
            after_data = next((s for s in summary_data if s["turn_structure"] == ts and s["oracle_timing"] == "after_statements"), None)

            before_adv = f"{before_data['estimator_advantage_property']*100:+.1f}%" if before_data else "N/A"
            after_adv = f"{after_data['estimator_advantage_property']*100:+.1f}%" if after_data else "N/A"

            lines.append(f"| {ts} | {before_adv} | {after_adv} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "(Analysis to be added after reviewing results)",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment()
