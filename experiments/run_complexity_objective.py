#!/usr/bin/env python3
"""Objective Complexity Effect on Inference Accuracy (Principled Evaluation).

Research Questions:
- How does objective complexity affect inference accuracy?
- Are simple single-property goals easier to infer than multi-factor goals?
- How does the number of properties to infer impact accuracy?

Complexity Levels (Simplified Value Functions):
- L1: 1 property (e.g., "wants color=blue")
- L2: 2 properties (e.g., "wants color=blue AND size=large")
- L3: 3 properties
- L4: 4 properties
- L5: 5 properties

Evaluation:
- Principled mode: Estimator is told N, predicts exactly N property=value pairs
- Deterministic overlap scoring (exact F1, property recall) instead of LLM judge

Design: 5 levels x 10 seeds = 50 games
Fixed: oracle_budget=4, principled inference

Hypothesis: Accuracy: L1 (~80%) > L2 > L3 > L4 > L5 (~30%)
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
    "L1": "1 property (e.g., 'color=blue')",
    "L2": "2 properties (e.g., 'color=blue, size=large')",
    "L3": "3 properties",
    "L4": "4 properties",
    "L5": "5 properties",
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
    "use_simple_value_functions": True,  # Use new simplified N-property format
    "objective_inference_mode": "principled",  # Principled inference with overlap scoring
    "enable_agent_thinking": True,
    "agent_thinking_budget": 5000,
    "enable_estimator_thinking": True,  # Capture estimator CoT
    "estimator_thinking_budget": 5000,
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
        vf_type = vf.get("type", "legacy")

        if vf_type == "simple":
            # SimpleValueFunction format
            cares_about = vf.get("cares_about", [])
            agent_vf_details.append({
                "agent_id": agent.get("id"),
                "value_function_name": vf.get("name", ""),
                "description": vf.get("description", ""),
                "type": "simple",
                "n_properties": len(cares_about),
                "cares_about": cares_about,
            })
        else:
            # Legacy AgentValueFunction format
            conditions = vf.get("conditions", [])
            agent_vf_details.append({
                "agent_id": agent.get("id"),
                "value_function_name": vf.get("name", ""),
                "description": vf.get("description", ""),
                "type": "legacy",
                "n_conditions": len(conditions),
                "conditions": conditions,
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

    # Objective inference scores (primary metric - exact F1 for principled mode)
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

    # N properties from value functions (cares_about for simple VFs)
    n_properties_list = []
    for r in results:
        for vf in r.get("agent_value_functions", []):
            # Support both old (n_conditions) and new (n_properties/cares_about) formats
            n_props = vf.get("n_properties") or vf.get("n_conditions") or len(vf.get("cares_about", []))
            n_properties_list.append(n_props)
    stats["avg_n_properties"] = safe_stats(n_properties_list)

    # Confidence levels
    confidences = []
    for r in results:
        inf = r.get("agent_objective_inference", {})
        for agent_id, data in inf.items():
            if isinstance(data, dict):
                conf = data.get("confidence", 0)
                confidences.append(conf)
    stats["avg_confidence"] = safe_stats(confidences)

    # Overlap metrics (for principled mode)
    exact_f1_list = []
    exact_precision_list = []
    exact_recall_list = []
    property_precision_list = []
    property_recall_list = []
    n_exact_matches_list = []
    n_property_matches_list = []

    for r in results:
        inf = r.get("agent_objective_inference", {})
        for agent_id, data in inf.items():
            if isinstance(data, dict) and "overlap_metrics" in data:
                om = data["overlap_metrics"]
                exact_f1_list.append(om.get("exact_f1", 0))
                exact_precision_list.append(om.get("exact_precision", 0))
                exact_recall_list.append(om.get("exact_recall", 0))
                property_precision_list.append(om.get("property_precision", 0))
                property_recall_list.append(om.get("property_recall", 0))
                n_exact_matches_list.append(om.get("n_exact_matches", 0))
                n_property_matches_list.append(om.get("n_property_matches", 0))

    stats["exact_f1"] = safe_stats(exact_f1_list)
    stats["exact_precision"] = safe_stats(exact_precision_list)
    stats["exact_recall"] = safe_stats(exact_recall_list)
    stats["property_precision"] = safe_stats(property_precision_list)
    stats["property_recall"] = safe_stats(property_recall_list)
    stats["avg_exact_matches"] = safe_stats(n_exact_matches_list)
    stats["avg_property_matches"] = safe_stats(n_property_matches_list)

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
                # Count N properties (support both simple and legacy formats)
                n_props = 0
                for vf in result.get("agent_value_functions", []):
                    n_props += vf.get("n_properties") or vf.get("n_conditions") or len(vf.get("cares_about", []))
                print(f"done ({game_elapsed:.0f}s) - Exact F1: {obj_score*100:.1f}%, N props: {n_props}")

                # Extract overlap metrics for logging
                overlap_log = {}
                inf_data = result.get("agent_objective_inference", {})
                for agent_id, data in inf_data.items():
                    if isinstance(data, dict) and "overlap_metrics" in data:
                        om = data["overlap_metrics"]
                        overlap_log[f"{agent_id}_exact_f1"] = om.get("exact_f1", 0)
                        overlap_log[f"{agent_id}_prop_recall"] = om.get("property_recall", 0)

                # Log to wandb
                wandb.log({
                    "game_number": game_count,
                    "condition_id": condition.condition_id,
                    "complexity": condition.complexity,
                    "seed": seed,
                    "game_time_seconds": game_elapsed,
                    "objective_inference_score": obj_score,  # Now exact F1
                    "total_n_properties": n_props,
                    **{f"agent_{k}_score": v for k, v in result.get("agent_objective_scores", {}).items()},
                    **overlap_log,
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
            stats.get("avg_n_properties", {}).get("mean", 0),
            stats.get("exact_f1", {}).get("mean", 0),
            stats.get("exact_f1", {}).get("std", 0),
            stats.get("exact_precision", {}).get("mean", 0),
            stats.get("exact_recall", {}).get("mean", 0),
            stats.get("property_recall", {}).get("mean", 0),
            stats.get("avg_confidence", {}).get("mean", 0),
        ])

    wandb.log({
        "summary_table": wandb.Table(
            columns=[
                "level", "description", "n_games", "n_properties",
                "exact_f1_mean", "exact_f1_std",
                "exact_precision", "exact_recall", "property_recall",
                "avg_confidence"
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
    print("\n" + "="*80)
    print("RESULTS SUMMARY (Principled Evaluation)")
    print("="*80)

    print("\n### By Complexity Level ###\n")
    print(f"{'Level':<6} | {'N Props':<8} | {'Exact F1':<10} | {'Exact P':<10} | {'Exact R':<10} | {'Prop Recall':<12}")
    print("-" * 70)

    for level in COMPLEXITY_LEVELS:
        cond_id = f"complexity_{level}"
        if cond_id not in condition_stats:
            continue

        stats = condition_stats[cond_id]["stats"]
        n_props = stats.get("avg_n_properties", {}).get("mean", 0)
        exact_f1 = stats.get("exact_f1", {}).get("mean", 0)
        exact_p = stats.get("exact_precision", {}).get("mean", 0)
        exact_r = stats.get("exact_recall", {}).get("mean", 0)
        prop_r = stats.get("property_recall", {}).get("mean", 0)

        print(f"{level:<6} | {n_props:>6.1f} | {exact_f1*100:>8.1f}% | {exact_p*100:>8.1f}% | {exact_r*100:>8.1f}% | {prop_r*100:>10.1f}%")

    # Legacy score (same as exact F1 for principled mode)
    print("\n### Overall Objective Inference Score (Exact F1) ###")
    for level in COMPLEXITY_LEVELS:
        cond_id = f"complexity_{level}"
        if cond_id not in condition_stats:
            continue

        stats = condition_stats[cond_id]["stats"]
        obj_score = stats.get("objective_inference_score", {}).get("mean", 0)
        obj_std = stats.get("objective_inference_score", {}).get("std", 0)
        desc = LEVEL_DESCRIPTIONS[level]

        print(f"  {level}: {obj_score*100:.1f}% ± {obj_std*100:.1f}% ({desc})")

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
        "# Objective Complexity Effect Experiment Results (Principled Evaluation)",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Total Games**: {sum(d['n_games'] for d in condition_stats.values())}",
        f"**Evaluation Method**: Principled (deterministic overlap scoring)",
        "",
        "## Research Question",
        "",
        "How does objective complexity (N properties) affect inference accuracy?",
        "",
        "## Complexity Levels (Simple Value Functions)",
        "",
    ]

    for level, desc in LEVEL_DESCRIPTIONS.items():
        lines.append(f"- **{level}**: {desc}")

    lines.extend([
        "",
        "## Evaluation Metrics",
        "",
        "- **Exact F1**: F1 score for exact (property + value) matches",
        "- **Exact Precision**: Fraction of predicted properties that exactly match",
        "- **Exact Recall**: Fraction of actual properties that were exactly matched",
        "- **Property Recall**: Fraction of properties correctly identified (partial credit for wrong value)",
        "",
        "## Hypothesis",
        "",
        "Accuracy: L1 (~80%) > L2 > L3 > L4 > L5 (~30%)",
        "",
        "## Results",
        "",
        "| Level | N Props | Exact F1 | Exact P | Exact R | Prop Recall |",
        "|-------|---------|----------|---------|---------|-------------|",
    ])

    for level in COMPLEXITY_LEVELS:
        cond_id = f"complexity_{level}"
        if cond_id not in condition_stats:
            continue

        stats = condition_stats[cond_id]["stats"]
        n_props = stats.get("avg_n_properties", {}).get("mean", 0)
        exact_f1 = stats.get("exact_f1", {}).get("mean", 0)
        exact_p = stats.get("exact_precision", {}).get("mean", 0)
        exact_r = stats.get("exact_recall", {}).get("mean", 0)
        prop_r = stats.get("property_recall", {}).get("mean", 0)

        lines.append(f"| {level} | {n_props:.1f} | {exact_f1*100:.1f}% | {exact_p*100:.1f}% | {exact_r*100:.1f}% | {prop_r*100:.1f}% |")

    # Overall scores table
    lines.extend([
        "",
        "### Overall Scores (Exact F1)",
        "",
        "| Level | Mean ± Std |",
        "|-------|------------|",
    ])

    for level in COMPLEXITY_LEVELS:
        cond_id = f"complexity_{level}"
        if cond_id not in condition_stats:
            continue

        stats = condition_stats[cond_id]["stats"]
        obj_score = stats.get("objective_inference_score", {}).get("mean", 0)
        obj_std = stats.get("objective_inference_score", {}).get("std", 0)

        lines.append(f"| {level} | {obj_score*100:.1f}% ± {obj_std*100:.1f}% |")

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
