#!/usr/bin/env python3
"""Oracle Budget Effect on Agent Objective Inference.

Research Questions:
- Does more oracle budget help objective inference?
- Can verifying agent claims improve understanding of their goals?
- Is there diminishing returns on oracle queries for objective inference?

Budgets Tested: [0, 1, 2, 4, 6, 8]

Design: 6 budgets x 5 seeds = 30 games
Fixed: medium complexity (L3), freeform inference

Hypothesis: More oracle -> better calibration of agent credibility -> better inference
"""

import json
import statistics
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

# Seeds for replication
SEEDS = [42, 123, 456, 789, 101]  # 5 seeds per condition

# Oracle budgets to test
ORACLE_BUDGETS = [0, 1, 2, 4, 6, 8]

# Fixed game parameters
BASE_CONFIG = {
    "n_objects": 10,
    "n_agents": 2,
    "n_rounds": 10,
    "selection_size": 5,
    "enable_estimator": True,
    "infer_agent_objectives": True,
    "use_agent_value_functions": True,
    "agent_value_function_complexity": "L3",  # Medium complexity
    "objective_inference_mode": "freeform",  # Standard freeform inference
    "enable_agent_thinking": True,
    "agent_thinking_budget": 5000,
    "force_oracle": True,  # Ensure oracle is used when available
    # Models
    "estimator_model": "claude-sonnet-4-20250514",
    "agent_model": "claude-sonnet-4-20250514",
    "observer_model": "claude-sonnet-4-20250514",
    # Fixed structure
    "turn_structure": "interleaved",
    "oracle_timing": "before_response",
    "debate_structure": "open",
    "condition": "ids",
}


@dataclass
class ExperimentCondition:
    """A single experimental condition."""
    condition_id: str
    oracle_budget: int

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "oracle_budget": self.oracle_budget,
        }


def generate_conditions() -> list[ExperimentCondition]:
    """Generate all experimental conditions."""
    conditions = []

    for budget in ORACLE_BUDGETS:
        cond_id = f"oracle_budget_{budget}"
        conditions.append(ExperimentCondition(
            condition_id=cond_id,
            oracle_budget=budget,
        ))

    return conditions


def run_single_game(condition: ExperimentCondition, seed: int) -> dict:
    """Run a single game with the given condition and seed."""
    config = GameConfig(
        **BASE_CONFIG,
        oracle_budget=condition.oracle_budget,
        seed=seed,
    )

    game = HiddenValueGame(config)
    result = game.run()

    return {
        "condition": condition.to_dict(),
        "seed": seed,
        "metrics": result.metrics,
        "estimator_metrics": result.estimator_metrics,
        "agent_objective_inference": result.agent_objective_inference,
        "agent_objective_scores": result.agent_objective_scores,
        "agent_objective_overall_score": result.agent_objective_overall_score,
        "accuracy_progression": result.accuracy_progression,
        "config": result.config,
        "agents": result.agents,
        "rounds": result.rounds,  # Include for analysis
    }


def compute_condition_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics for a condition across seeds."""
    if not results:
        return {}

    def safe_stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return {"mean": 0, "std": 0, "n": 0}
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        return {"mean": mean, "std": std, "n": len(values)}

    stats = {}

    # Core metrics
    core_keys = ["total_value", "selection_accuracy", "property_accuracy", "rule_inference_accuracy"]
    for key in core_keys:
        values = [r["metrics"].get(key) for r in results]
        stats[key] = safe_stats(values)

    # Objective inference scores (primary metric)
    obj_scores = [r.get("agent_objective_overall_score") for r in results]
    stats["objective_inference_score"] = safe_stats(obj_scores)

    # Per-agent scores
    agent_scores = {}
    for r in results:
        scores = r.get("agent_objective_scores", {})
        for agent_id, score in scores.items():
            if agent_id not in agent_scores:
                agent_scores[agent_id] = []
            agent_scores[agent_id].append(score)

    stats["per_agent_objective_scores"] = {
        agent_id: safe_stats(scores)
        for agent_id, scores in agent_scores.items()
    }

    # Estimator property accuracy
    est_prop_accs = [
        r.get("estimator_metrics", {}).get("property_accuracy")
        for r in results if r.get("estimator_metrics")
    ]
    stats["estimator_property_accuracy"] = safe_stats(est_prop_accs)

    # Count actual oracle queries used (may be less than budget)
    oracle_counts = []
    for r in results:
        rounds = r.get("rounds", [])
        count = sum(1 for rnd in rounds if rnd.get("oracle_query"))
        oracle_counts.append(count)
    stats["actual_oracle_queries"] = safe_stats(oracle_counts)

    # Confidence levels
    confidences = []
    for r in results:
        inf = r.get("agent_objective_inference", {})
        for agent_id, data in inf.items():
            if isinstance(data, dict):
                conf = data.get("confidence", 0)
                confidences.append(conf)
    stats["avg_confidence"] = safe_stats(confidences)

    return stats


def run_experiment():
    """Run the oracle budget experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate all conditions
    conditions = generate_conditions()
    total_games = len(conditions) * len(SEEDS)

    print(f"\n{'='*70}")
    print("Oracle Budget Effect on Objective Inference")
    print(f"{'='*70}")
    print(f"Oracle budgets: {ORACLE_BUDGETS}")
    print(f"Total conditions: {len(conditions)}")
    print(f"Seeds per condition: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    print(f"{'='*70}\n")

    # Initialize wandb
    wandb_run = wandb.init(
        project="truthification",
        name=f"oracle-budget-objective-{timestamp}",
        config={
            "experiment": "oracle_budget_objective",
            "oracle_budgets": ORACLE_BUDGETS,
            "seeds": SEEDS,
            "total_conditions": len(conditions),
            "total_games": total_games,
            **BASE_CONFIG,
        },
    )

    # Output directories
    output_dir = Path("outputs/oracle_budget_objective") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/oracle_budget_objective_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    all_results = []
    condition_results = {}

    game_count = 0
    start_time = time.time()

    for condition in conditions:
        print(f"\n--- Condition: {condition.condition_id} ---")
        condition_results[condition.condition_id] = []

        for seed in SEEDS:
            game_count += 1
            elapsed = time.time() - start_time
            eta = (elapsed / game_count) * (total_games - game_count) if game_count > 0 else 0

            print(f"  [{game_count}/{total_games}] seed={seed} (ETA: {eta/60:.1f}m)...", end=" ", flush=True)

            try:
                game_start = time.time()
                result = run_single_game(condition, seed)
                game_elapsed = time.time() - game_start

                all_results.append(result)
                condition_results[condition.condition_id].append(result)

                # Save individual game result
                game_file = output_dir / f"game_{condition.condition_id}_seed{seed}.json"
                with open(game_file, "w") as f:
                    json.dump(result, f, indent=2)

                # Quick summary
                obj_score = result.get("agent_objective_overall_score", 0)
                prop_acc = result.get("metrics", {}).get("property_accuracy", 0)
                print(f"done ({game_elapsed:.0f}s) - ObjInf: {obj_score*100:.1f}%, PropAcc: {prop_acc*100:.1f}%")

                # Log to wandb
                wandb.log({
                    "game_number": game_count,
                    "condition_id": condition.condition_id,
                    "oracle_budget": condition.oracle_budget,
                    "seed": seed,
                    "game_time_seconds": game_elapsed,
                    "objective_inference_score": obj_score,
                    "property_accuracy": prop_acc,
                    **{f"agent_{k}_score": v for k, v in result.get("agent_objective_scores", {}).items()},
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

    condition_stats = {}
    for condition in conditions:
        cond_id = condition.condition_id
        results = condition_results.get(cond_id, [])
        if results:
            condition_stats[cond_id] = {
                "condition": condition.to_dict(),
                "n_games": len(results),
                "stats": compute_condition_stats(results),
            }

    # Print summary
    print_summary(condition_stats)

    # Save results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / "condition_stats.json", "w") as f:
        json.dump(condition_stats, f, indent=2, default=str)

    # Create summary markdown
    summary_md = create_summary_markdown(condition_stats, total_elapsed)
    with open(output_dir / "README.md", "w") as f:
        f.write(summary_md)

    # Also save to results directory
    with open(results_dir / f"condition_stats_{timestamp}.json", "w") as f:
        json.dump(condition_stats, f, indent=2, default=str)
    with open(results_dir / f"README_{timestamp}.md", "w") as f:
        f.write(summary_md)

    # Log summary table to wandb
    summary_rows = []
    for cond_id, data in sorted(condition_stats.items(), key=lambda x: x[1]["condition"]["oracle_budget"]):
        cond = data["condition"]
        stats = data["stats"]
        summary_rows.append([
            cond["oracle_budget"],
            data["n_games"],
            stats.get("objective_inference_score", {}).get("mean", 0),
            stats.get("objective_inference_score", {}).get("std", 0),
            stats.get("property_accuracy", {}).get("mean", 0),
            stats.get("estimator_property_accuracy", {}).get("mean", 0),
            stats.get("actual_oracle_queries", {}).get("mean", 0),
        ])

    wandb.log({
        "summary_table": wandb.Table(
            columns=[
                "oracle_budget", "n_games", "obj_inf_score_mean", "obj_inf_score_std",
                "property_accuracy", "estimator_prop_accuracy", "actual_queries"
            ],
            data=summary_rows,
        ),
        "total_runtime_minutes": total_elapsed / 60,
    })

    wandb.finish()

    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print(f"Summary saved to: {results_dir}")

    return condition_stats


def print_summary(condition_stats: dict):
    """Print summary tables to console."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\n### By Oracle Budget ###\n")
    print(f"{'Budget':<8} | {'Obj Inf Score':<15} | {'Prop Acc':<12} | {'Queries Used':<12}")
    print("-" * 60)

    for budget in ORACLE_BUDGETS:
        cond_id = f"oracle_budget_{budget}"
        if cond_id not in condition_stats:
            continue

        stats = condition_stats[cond_id]["stats"]
        obj_score = stats.get("objective_inference_score", {}).get("mean", 0)
        prop_acc = stats.get("property_accuracy", {}).get("mean", 0)
        queries = stats.get("actual_oracle_queries", {}).get("mean", 0)

        print(f"{budget:<8} | {obj_score*100:>12.1f}% | {prop_acc*100:>10.1f}% | {queries:>10.1f}")


def create_summary_markdown(condition_stats: dict, total_elapsed: float) -> str:
    """Create markdown summary of results."""
    lines = [
        "# Oracle Budget Effect on Objective Inference Results",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Total Games**: {sum(d['n_games'] for d in condition_stats.values())}",
        "",
        "## Research Question",
        "",
        "Does more oracle budget help objective inference?",
        "",
        "## Hypothesis",
        "",
        "More oracle -> better calibration of agent credibility -> better inference",
        "",
        "## Results",
        "",
        "| Oracle Budget | Obj Inf Score (mean) | Obj Inf Score (std) | Property Acc | Queries Used |",
        "|---------------|----------------------|---------------------|--------------|--------------|",
    ]

    for budget in ORACLE_BUDGETS:
        cond_id = f"oracle_budget_{budget}"
        if cond_id not in condition_stats:
            continue

        stats = condition_stats[cond_id]["stats"]
        obj_score = stats.get("objective_inference_score", {}).get("mean", 0)
        obj_std = stats.get("objective_inference_score", {}).get("std", 0)
        prop_acc = stats.get("property_accuracy", {}).get("mean", 0)
        queries = stats.get("actual_oracle_queries", {}).get("mean", 0)

        lines.append(f"| {budget} | {obj_score*100:.1f}% | {obj_std*100:.1f}% | {prop_acc*100:.1f}% | {queries:.1f} |")

    lines.extend([
        "",
        "## Analysis",
        "",
        "### Key Findings",
        "",
        "1. **Baseline (0 oracle)**: Without verification, inference relies solely on statement patterns",
        "2. **Effect of oracle**: Does verification help calibrate agent credibility?",
        "3. **Diminishing returns**: Is there a point where more queries don't help?",
        "",
        "## Interpretation",
        "",
        "(Analysis to be added after reviewing results)",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment()
