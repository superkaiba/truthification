#!/usr/bin/env python3
"""Search Space Constraint Experiment for Agent Objective Inference.

Research Questions:
- How does constraining the search space affect objective inference accuracy?
- Is multiple-choice easier than freeform generation?
- Does structured factor selection perform better than natural language?

Methods Tested:
1. Multiple-Choice (2): Binary choice (correct vs 1 distractor)
2. Multiple-Choice (4): 1 correct, 3 distractors
3. Multiple-Choice (8): 1 correct, 7 distractors
4. Multiple-Choice (16): 1 correct, 15 distractors (near-random baseline)
5. Freeform: LLM generates any hypothesis (current default)
6. Structured: Select from enumerated property=value pairs

Design: 6 methods x 2 complexity (simple L1, complex L5) x 5 seeds = 60 games

Hypothesis: Accuracy order: 2-choice > 4 > 8 > freeform > 16
"""

import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from itertools import product

from dotenv import load_dotenv
load_dotenv()

import wandb
from src.environment.simulation import GameConfig, HiddenValueGame

# ============================================================================
# Experimental Configuration
# ============================================================================

# Seeds for replication
SEEDS = [42, 123, 456, 789, 101]  # 5 seeds per condition

# Search space methods to test
INFERENCE_MODES = [
    "multiple_choice_2",
    "multiple_choice_4",
    "multiple_choice_8",
    "multiple_choice_16",
    "freeform",
    "structured",
]

# Complexity levels to test (L1 = simple, L5 = complex)
COMPLEXITY_LEVELS = ["L1", "L5"]

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
    "enable_agent_thinking": True,
    "agent_thinking_budget": 5000,
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
    inference_mode: str
    complexity: str

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "inference_mode": self.inference_mode,
            "complexity": self.complexity,
        }


def generate_conditions() -> list[ExperimentCondition]:
    """Generate all experimental conditions."""
    conditions = []

    for mode, complexity in product(INFERENCE_MODES, COMPLEXITY_LEVELS):
        cond_id = f"{mode}_{complexity}"
        conditions.append(ExperimentCondition(
            condition_id=cond_id,
            inference_mode=mode,
            complexity=complexity,
        ))

    return conditions


def run_single_game(condition: ExperimentCondition, seed: int) -> dict:
    """Run a single game with the given condition and seed."""
    config = GameConfig(
        **BASE_CONFIG,
        objective_inference_mode=condition.inference_mode,
        agent_value_function_complexity=condition.complexity,
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
        "config": result.config,
        "agents": result.agents,
        "value_rule": result.value_rule,
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

    # Objective inference scores (primary metric for this experiment)
    obj_scores = [r.get("agent_objective_overall_score") for r in results]
    stats["objective_inference_score"] = safe_stats(obj_scores)

    # Per-agent objective scores
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

    # Inference mode and confidence
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
    """Run the full search space experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate all conditions
    conditions = generate_conditions()
    total_games = len(conditions) * len(SEEDS)

    print(f"\n{'='*70}")
    print("Search Space Constraint Experiment")
    print(f"{'='*70}")
    print(f"Inference modes: {INFERENCE_MODES}")
    print(f"Complexity levels: {COMPLEXITY_LEVELS}")
    print(f"Total conditions: {len(conditions)}")
    print(f"Seeds per condition: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    print(f"{'='*70}\n")

    # Initialize wandb
    wandb_run = wandb.init(
        project="truthification",
        name=f"search-space-{timestamp}",
        config={
            "experiment": "search_space",
            "inference_modes": INFERENCE_MODES,
            "complexity_levels": COMPLEXITY_LEVELS,
            "seeds": SEEDS,
            "total_conditions": len(conditions),
            "total_games": total_games,
            **BASE_CONFIG,
        },
    )

    # Output directories
    output_dir = Path("outputs/search_space") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/search_space_experiment")
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
                print(f"done ({game_elapsed:.0f}s) - ObjInfScore: {obj_score*100:.1f}%")

                # Log to wandb
                wandb.log({
                    "game_number": game_count,
                    "condition_id": condition.condition_id,
                    "inference_mode": condition.inference_mode,
                    "complexity": condition.complexity,
                    "seed": seed,
                    "game_time_seconds": game_elapsed,
                    "objective_inference_score": obj_score,
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
    for cond_id, data in condition_stats.items():
        cond = data["condition"]
        stats = data["stats"]
        summary_rows.append([
            cond["inference_mode"],
            cond["complexity"],
            data["n_games"],
            stats.get("objective_inference_score", {}).get("mean", 0),
            stats.get("objective_inference_score", {}).get("std", 0),
            stats.get("avg_confidence", {}).get("mean", 0),
        ])

    wandb.log({
        "summary_table": wandb.Table(
            columns=[
                "inference_mode", "complexity", "n_games",
                "obj_inf_score_mean", "obj_inf_score_std", "avg_confidence"
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

    # Group by inference mode
    print("\n### By Inference Mode ###\n")
    print(f"{'Mode':<20} | {'Obj Inf Score':<15} | {'Confidence':<12}")
    print("-" * 55)

    for mode in INFERENCE_MODES:
        matching = [
            condition_stats[cond_id]
            for cond_id, data in condition_stats.items()
            if data["condition"]["inference_mode"] == mode
        ]
        if not matching:
            continue

        scores = [m["stats"].get("objective_inference_score", {}).get("mean", 0) for m in matching]
        confs = [m["stats"].get("avg_confidence", {}).get("mean", 0) for m in matching]

        avg_score = statistics.mean(scores) if scores else 0
        avg_conf = statistics.mean(confs) if confs else 0

        print(f"{mode:<20} | {avg_score*100:>12.1f}% | {avg_conf:>10.1f}")

    # Group by complexity
    print("\n### By Complexity ###\n")
    print(f"{'Complexity':<12} | {'Obj Inf Score':<15}")
    print("-" * 35)

    for complexity in COMPLEXITY_LEVELS:
        matching = [
            condition_stats[cond_id]
            for cond_id, data in condition_stats.items()
            if data["condition"]["complexity"] == complexity
        ]
        if not matching:
            continue

        scores = [m["stats"].get("objective_inference_score", {}).get("mean", 0) for m in matching]
        avg_score = statistics.mean(scores) if scores else 0

        print(f"{complexity:<12} | {avg_score*100:>12.1f}%")


def create_summary_markdown(condition_stats: dict, total_elapsed: float) -> str:
    """Create markdown summary of results."""
    lines = [
        "# Search Space Constraint Experiment Results",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Total Games**: {sum(d['n_games'] for d in condition_stats.values())}",
        "",
        "## Research Question",
        "",
        "How does constraining the search space affect objective inference accuracy?",
        "",
        "## Hypothesis",
        "",
        "Accuracy order: 2-choice > 4 > 8 > freeform > 16",
        "",
        "## Results by Inference Mode",
        "",
        "| Mode | Obj Inf Score (mean) | Obj Inf Score (std) | Avg Confidence |",
        "|------|----------------------|---------------------|----------------|",
    ]

    for mode in INFERENCE_MODES:
        matching = [
            condition_stats[cond_id]
            for cond_id, data in condition_stats.items()
            if data["condition"]["inference_mode"] == mode
        ]
        if not matching:
            continue

        scores = [m["stats"].get("objective_inference_score", {}).get("mean", 0) for m in matching]
        stds = [m["stats"].get("objective_inference_score", {}).get("std", 0) for m in matching]
        confs = [m["stats"].get("avg_confidence", {}).get("mean", 0) for m in matching]

        avg_score = statistics.mean(scores) if scores else 0
        avg_std = statistics.mean(stds) if stds else 0
        avg_conf = statistics.mean(confs) if confs else 0

        lines.append(f"| {mode} | {avg_score*100:.1f}% | {avg_std*100:.1f}% | {avg_conf:.1f} |")

    lines.extend([
        "",
        "## Results by Complexity",
        "",
        "| Complexity | Obj Inf Score (mean) |",
        "|------------|----------------------|",
    ])

    for complexity in COMPLEXITY_LEVELS:
        matching = [
            condition_stats[cond_id]
            for cond_id, data in condition_stats.items()
            if data["condition"]["complexity"] == complexity
        ]
        if not matching:
            continue

        scores = [m["stats"].get("objective_inference_score", {}).get("mean", 0) for m in matching]
        avg_score = statistics.mean(scores) if scores else 0

        lines.append(f"| {complexity} | {avg_score*100:.1f}% |")

    lines.extend([
        "",
        "## Full Results",
        "",
        "| Condition | N | Score Mean | Score Std |",
        "|-----------|---|------------|-----------|",
    ])

    for cond_id, data in sorted(condition_stats.items()):
        stats = data["stats"]
        n = data["n_games"]
        score_mean = stats.get("objective_inference_score", {}).get("mean", 0)
        score_std = stats.get("objective_inference_score", {}).get("std", 0)

        lines.append(f"| {cond_id} | {n} | {score_mean*100:.1f}% | {score_std*100:.1f}% |")

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
