#!/usr/bin/env python3
"""Objective Complexity Effect on Inference Accuracy.

Research Questions:
- How does objective complexity affect inference accuracy?
- Are simple single-property goals easier to infer than multi-factor goals?
- How do penalties and complex conditions impact inference?

Complexity Levels:
- L1-Simple: 1 property condition (e.g., "wants blue objects")
- L2-Dual: 2 properties (AND) (e.g., "wants blue AND large")
- L3-Combo: 2 properties + combination bonus
- L4-Complex: 3-4 conditions (multiple bonuses)
- L5-Penalty: 4-5 conditions + penalties

Design: 5 levels x 5 seeds = 25 games
Fixed: oracle_budget=4, freeform inference

Hypothesis: Accuracy: L1 (~70%) > L2 > L3 > L4 > L5 (~30%)
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
SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]  # 10 seeds per condition

# Complexity levels to test
COMPLEXITY_LEVELS = ["L1", "L2", "L3", "L4", "L5"]

# Level descriptions for documentation
LEVEL_DESCRIPTIONS = {
    "L1": "Simple: 1 property (e.g., 'wants blue')",
    "L2": "Dual: 2 properties (e.g., 'wants blue AND large')",
    "L3": "Combo: 2 properties + combination bonus",
    "L4": "Complex: 3-4 conditions with multiple bonuses",
    "L5": "Penalty: 4-5 conditions including penalties",
}

# Fixed game parameters
BASE_CONFIG = {
    "n_objects": 10,
    "n_agents": 2,
    "n_rounds": 10,
    "oracle_budget": 4,  # Fixed oracle budget
    "selection_size": 5,
    "enable_estimator": True,
    "infer_agent_objectives": True,
    "use_agent_value_functions": True,
    "objective_inference_mode": "freeform",  # Standard freeform inference
    "enable_agent_thinking": True,
    "agent_thinking_budget": 5000,
    "force_oracle": True,
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
    complexity: str

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "complexity": self.complexity,
            "description": LEVEL_DESCRIPTIONS.get(self.complexity, ""),
        }


def generate_conditions() -> list[ExperimentCondition]:
    """Generate all experimental conditions."""
    conditions = []

    for level in COMPLEXITY_LEVELS:
        cond_id = f"complexity_{level}"
        conditions.append(ExperimentCondition(
            condition_id=cond_id,
            complexity=level,
        ))

    return conditions


def run_single_game(condition: ExperimentCondition, seed: int) -> dict:
    """Run a single game with the given condition and seed."""
    config = GameConfig(
        **BASE_CONFIG,
        agent_value_function_complexity=condition.complexity,
        seed=seed,
    )

    game = HiddenValueGame(config)
    result = game.run()

    # Extract agent value function details for analysis
    agent_vf_details = []
    for agent in result.agents:
        vf = agent.get("value_function", {})
        agent_vf_details.append({
            "agent_id": agent.get("id"),
            "value_function_name": vf.get("name", ""),
            "description": vf.get("description", ""),
            "n_conditions": len(vf.get("conditions", [])),
            "conditions": vf.get("conditions", []),
        })

    return {
        "condition": condition.to_dict(),
        "seed": seed,
        "metrics": result.metrics,
        "estimator_metrics": result.estimator_metrics,
        "agent_objective_inference": result.agent_objective_inference,
        "agent_objective_scores": result.agent_objective_scores,
        "agent_objective_overall_score": result.agent_objective_overall_score,
        "agent_value_functions": agent_vf_details,
        "config": result.config,
        "agents": result.agents,
        "rounds": result.rounds,  # Include for strategy annotation
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

    # Condition counts from value functions
    n_conditions_list = []
    for r in results:
        for vf in r.get("agent_value_functions", []):
            n_conditions_list.append(vf.get("n_conditions", 0))
    stats["avg_n_conditions"] = safe_stats(n_conditions_list)

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
    """Run the complexity experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate all conditions
    conditions = generate_conditions()
    total_games = len(conditions) * len(SEEDS)

    print(f"\n{'='*70}")
    print("Objective Complexity Effect Experiment")
    print(f"{'='*70}")
    print("Complexity Levels:")
    for level, desc in LEVEL_DESCRIPTIONS.items():
        print(f"  {level}: {desc}")
    print(f"\nTotal conditions: {len(conditions)}")
    print(f"Seeds per condition: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    print(f"{'='*70}\n")

    # Initialize wandb
    wandb_run = wandb.init(
        project="truthification",
        name=f"complexity-objective-{timestamp}",
        config={
            "experiment": "complexity_objective",
            "complexity_levels": COMPLEXITY_LEVELS,
            "level_descriptions": LEVEL_DESCRIPTIONS,
            "seeds": SEEDS,
            "total_conditions": len(conditions),
            "total_games": total_games,
            **BASE_CONFIG,
        },
    )

    # Output directories
    output_dir = Path("outputs/complexity_objective") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/complexity_objective_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    all_results = []
    condition_results = {}

    game_count = 0
    start_time = time.time()

    for condition in conditions:
        print(f"\n--- Condition: {condition.condition_id} ({LEVEL_DESCRIPTIONS[condition.complexity]}) ---")
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
                n_conds = sum(vf.get("n_conditions", 0) for vf in result.get("agent_value_functions", []))
                print(f"done ({game_elapsed:.0f}s) - ObjInf: {obj_score*100:.1f}%, Conditions: {n_conds}")

                # Log to wandb
                wandb.log({
                    "game_number": game_count,
                    "condition_id": condition.condition_id,
                    "complexity": condition.complexity,
                    "seed": seed,
                    "game_time_seconds": game_elapsed,
                    "objective_inference_score": obj_score,
                    "total_conditions": n_conds,
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
    for level in COMPLEXITY_LEVELS:
        cond_id = f"complexity_{level}"
        if cond_id not in condition_stats:
            continue

        data = condition_stats[cond_id]
        stats = data["stats"]
        summary_rows.append([
            level,
            LEVEL_DESCRIPTIONS[level],
            data["n_games"],
            stats.get("objective_inference_score", {}).get("mean", 0),
            stats.get("objective_inference_score", {}).get("std", 0),
            stats.get("avg_n_conditions", {}).get("mean", 0),
            stats.get("avg_confidence", {}).get("mean", 0),
        ])

    wandb.log({
        "summary_table": wandb.Table(
            columns=[
                "level", "description", "n_games",
                "obj_inf_score_mean", "obj_inf_score_std",
                "avg_n_conditions", "avg_confidence"
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

    print("\n### By Complexity Level ###\n")
    print(f"{'Level':<6} | {'Description':<40} | {'Obj Inf Score':<15} | {'Avg Conds':<10}")
    print("-" * 80)

    for level in COMPLEXITY_LEVELS:
        cond_id = f"complexity_{level}"
        if cond_id not in condition_stats:
            continue

        stats = condition_stats[cond_id]["stats"]
        obj_score = stats.get("objective_inference_score", {}).get("mean", 0)
        n_conds = stats.get("avg_n_conditions", {}).get("mean", 0)
        desc = LEVEL_DESCRIPTIONS[level][:38]

        print(f"{level:<6} | {desc:<40} | {obj_score*100:>12.1f}% | {n_conds:>8.1f}")

    # Check hypothesis
    print("\n### Hypothesis Check ###")
    print("Expected: L1 > L2 > L3 > L4 > L5")

    scores_by_level = {}
    for level in COMPLEXITY_LEVELS:
        cond_id = f"complexity_{level}"
        if cond_id in condition_stats:
            scores_by_level[level] = condition_stats[cond_id]["stats"].get(
                "objective_inference_score", {}
            ).get("mean", 0)

    if scores_by_level:
        sorted_levels = sorted(scores_by_level.items(), key=lambda x: x[1], reverse=True)
        actual_order = " > ".join(f"{level}({score*100:.0f}%)" for level, score in sorted_levels)
        print(f"Actual:   {actual_order}")


def create_summary_markdown(condition_stats: dict, total_elapsed: float) -> str:
    """Create markdown summary of results."""
    lines = [
        "# Objective Complexity Effect Experiment Results",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Total Games**: {sum(d['n_games'] for d in condition_stats.values())}",
        "",
        "## Research Question",
        "",
        "How does objective complexity affect inference accuracy?",
        "",
        "## Complexity Levels",
        "",
    ]

    for level, desc in LEVEL_DESCRIPTIONS.items():
        lines.append(f"- **{level}**: {desc}")

    lines.extend([
        "",
        "## Hypothesis",
        "",
        "Accuracy: L1 (~70%) > L2 > L3 > L4 > L5 (~30%)",
        "",
        "## Results",
        "",
        "| Level | Description | Obj Inf Score (mean) | Obj Inf Score (std) | Avg Conditions |",
        "|-------|-------------|----------------------|---------------------|----------------|",
    ])

    for level in COMPLEXITY_LEVELS:
        cond_id = f"complexity_{level}"
        if cond_id not in condition_stats:
            continue

        stats = condition_stats[cond_id]["stats"]
        obj_score = stats.get("objective_inference_score", {}).get("mean", 0)
        obj_std = stats.get("objective_inference_score", {}).get("std", 0)
        n_conds = stats.get("avg_n_conditions", {}).get("mean", 0)

        lines.append(f"| {level} | {LEVEL_DESCRIPTIONS[level]} | {obj_score*100:.1f}% | {obj_std*100:.1f}% | {n_conds:.1f} |")

    # Check hypothesis
    scores_by_level = {}
    for level in COMPLEXITY_LEVELS:
        cond_id = f"complexity_{level}"
        if cond_id in condition_stats:
            scores_by_level[level] = condition_stats[cond_id]["stats"].get(
                "objective_inference_score", {}
            ).get("mean", 0)

    lines.extend([
        "",
        "## Hypothesis Check",
        "",
        "**Expected order:** L1 > L2 > L3 > L4 > L5",
        "",
    ])

    if scores_by_level:
        sorted_levels = sorted(scores_by_level.items(), key=lambda x: x[1], reverse=True)
        actual_order = " > ".join(f"{level}({score*100:.0f}%)" for level, score in sorted_levels)
        lines.append(f"**Actual order:** {actual_order}")

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
