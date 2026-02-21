#!/usr/bin/env python3
"""Experiment: Does Estimator Perform Better with Access to Agent CoT?

Research Question:
Does giving the estimator access to agents' chain of thought (thinking blocks)
improve its ability to infer their true objectives?

Hypothesis:
With CoT access, the estimator should achieve significantly higher accuracy
because agents' private reasoning reveals their true preferences.

Design:
- 2 conditions: with_cot vs without_cot
- Fixed complexity level (L3 = 3 properties)
- 10 seeds per condition = 20 games total
- Principled evaluation (overlap scoring)

Metrics:
- Exact F1 (primary)
- Property recall (partial credit)
- Confidence calibration
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

SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]

# Conditions
CONDITIONS = [
    {"name": "without_cot", "estimator_sees_agent_thinking": False},
    {"name": "with_cot", "estimator_sees_agent_thinking": True},
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
    # Agent thinking MUST be enabled for this experiment
    "enable_agent_thinking": True,
    "agent_thinking_budget": 5000,
    # Estimator thinking
    "enable_estimator_thinking": True,
    "estimator_thinking_budget": 5000,
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
class ExperimentCondition:
    """A single experimental condition."""
    name: str
    estimator_sees_agent_thinking: bool

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "estimator_sees_agent_thinking": self.estimator_sees_agent_thinking,
        }


def run_single_game(condition: ExperimentCondition, seed: int) -> dict:
    """Run a single game with the given condition and seed."""
    config = GameConfig(
        **BASE_CONFIG,
        estimator_sees_agent_thinking=condition.estimator_sees_agent_thinking,
        seed=seed,
    )

    game = HiddenValueGame(config)
    result = game.run()

    # Extract agent value function details
    agent_vf_details = []
    for agent in result.agents:
        vf = agent.get("value_function", {})
        agent_vf_details.append({
            "agent_id": agent.get("id"),
            "n_properties": vf.get("n_properties") or len(vf.get("cares_about", [])),
            "cares_about": vf.get("cares_about", []),
        })

    # Check if agent thinking was captured
    has_agent_thinking = False
    for r in result.rounds:
        for stmt in r.get("agent_statements", []):
            if stmt.get("thinking"):
                has_agent_thinking = True
                break

    return {
        "condition": condition.to_dict(),
        "seed": seed,
        "metrics": result.metrics,
        "agent_objective_inference": result.agent_objective_inference,
        "agent_objective_scores": result.agent_objective_scores,
        "agent_objective_overall_score": result.agent_objective_overall_score,
        "agent_value_functions": agent_vf_details,
        "has_agent_thinking": has_agent_thinking,
        "config": result.config,
    }


def compute_condition_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics for a condition."""
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

    # Primary metric: objective inference score (exact F1)
    obj_scores = [r.get("agent_objective_overall_score") for r in results]
    stats["objective_inference_score"] = safe_stats(obj_scores)

    # Extract overlap metrics
    exact_f1_list = []
    exact_precision_list = []
    exact_recall_list = []
    property_recall_list = []
    confidence_list = []

    for r in results:
        inf = r.get("agent_objective_inference", {})
        for agent_id, data in inf.items():
            if isinstance(data, dict):
                if "overlap_metrics" in data:
                    om = data["overlap_metrics"]
                    exact_f1_list.append(om.get("exact_f1", 0))
                    exact_precision_list.append(om.get("exact_precision", 0))
                    exact_recall_list.append(om.get("exact_recall", 0))
                    property_recall_list.append(om.get("property_recall", 0))
                confidence_list.append(data.get("confidence", 0))

    stats["exact_f1"] = safe_stats(exact_f1_list)
    stats["exact_precision"] = safe_stats(exact_precision_list)
    stats["exact_recall"] = safe_stats(exact_recall_list)
    stats["property_recall"] = safe_stats(property_recall_list)
    stats["avg_confidence"] = safe_stats(confidence_list)

    # Check agent thinking availability
    has_thinking = [r.get("has_agent_thinking", False) for r in results]
    stats["pct_with_agent_thinking"] = sum(has_thinking) / len(has_thinking) if has_thinking else 0

    return stats


def run_experiment():
    """Run the CoT access experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    conditions = [ExperimentCondition(**c) for c in CONDITIONS]
    total_games = len(conditions) * len(SEEDS)

    print(f"\n{'='*70}")
    print("Experiment: Estimator with Agent CoT Access")
    print(f"{'='*70}")
    print(f"Conditions: {[c.name for c in conditions]}")
    print(f"Seeds per condition: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    print(f"Complexity: L3 (3 properties)")
    print(f"{'='*70}\n")

    # Initialize wandb
    wandb_run = wandb.init(
        project="truthification",
        name=f"cot-access-experiment-{timestamp}",
        config={
            "experiment": "cot_access",
            "conditions": [c.name for c in conditions],
            "seeds": SEEDS,
            "total_games": total_games,
            **BASE_CONFIG,
        },
    )

    # Output directories
    output_dir = Path("outputs/cot_access_experiment") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/cot_access_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    all_results = []
    condition_results = {c.name: [] for c in conditions}

    game_count = 0
    start_time = time.time()

    for condition in conditions:
        print(f"\n--- Condition: {condition.name} ---")

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
                condition_results[condition.name].append(result)

                # Save individual game result
                game_file = output_dir / f"game_{condition.name}_seed{seed}.json"
                with open(game_file, "w") as f:
                    json.dump(result, f, indent=2)

                # Quick summary
                obj_score = result.get("agent_objective_overall_score", 0)
                has_cot = result.get("has_agent_thinking", False)
                print(f"done ({game_elapsed:.0f}s) - F1: {obj_score*100:.1f}%, AgentCoT: {has_cot}")

                # Log to wandb
                wandb.log({
                    "game_number": game_count,
                    "condition": condition.name,
                    "sees_cot": condition.estimator_sees_agent_thinking,
                    "seed": seed,
                    "game_time_seconds": game_elapsed,
                    "objective_inference_score": obj_score,
                    "has_agent_thinking": has_cot,
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
        results = condition_results.get(condition.name, [])
        if results:
            condition_stats[condition.name] = {
                "condition": condition.to_dict(),
                "n_games": len(results),
                "stats": compute_condition_stats(results),
            }

    # Print summary
    print_summary(condition_stats)

    # Statistical comparison
    print_statistical_comparison(condition_results)

    # Save results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / "condition_stats.json", "w") as f:
        json.dump(condition_stats, f, indent=2, default=str)

    # Create summary markdown
    summary_md = create_summary_markdown(condition_stats, condition_results, total_elapsed)
    with open(output_dir / "README.md", "w") as f:
        f.write(summary_md)

    # Also save to results directory
    with open(results_dir / f"condition_stats_{timestamp}.json", "w") as f:
        json.dump(condition_stats, f, indent=2, default=str)
    with open(results_dir / f"README_{timestamp}.md", "w") as f:
        f.write(summary_md)

    # Log summary to wandb
    summary_rows = []
    for condition in conditions:
        if condition.name not in condition_stats:
            continue
        data = condition_stats[condition.name]
        stats = data["stats"]
        summary_rows.append([
            condition.name,
            condition.estimator_sees_agent_thinking,
            data["n_games"],
            stats.get("exact_f1", {}).get("mean", 0),
            stats.get("exact_f1", {}).get("std", 0),
            stats.get("property_recall", {}).get("mean", 0),
            stats.get("avg_confidence", {}).get("mean", 0),
        ])

    wandb.log({
        "summary_table": wandb.Table(
            columns=["condition", "sees_cot", "n_games", "exact_f1_mean", "exact_f1_std",
                     "property_recall", "avg_confidence"],
            data=summary_rows,
        ),
        "total_runtime_minutes": total_elapsed / 60,
    })

    wandb.finish()

    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")

    return condition_stats


def print_summary(condition_stats: dict):
    """Print summary tables to console."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Condition':<15} | {'Exact F1':<15} | {'Prop Recall':<15} | {'Confidence':<12}")
    print("-" * 65)

    for name in ["without_cot", "with_cot"]:
        if name not in condition_stats:
            continue
        stats = condition_stats[name]["stats"]
        f1 = stats.get("exact_f1", {})
        pr = stats.get("property_recall", {})
        conf = stats.get("avg_confidence", {})

        print(f"{name:<15} | {f1.get('mean', 0)*100:>6.1f}% ± {f1.get('std', 0)*100:>4.1f}% | "
              f"{pr.get('mean', 0)*100:>6.1f}% ± {pr.get('std', 0)*100:>4.1f}% | "
              f"{conf.get('mean', 0):>5.1f}")


def print_statistical_comparison(condition_results: dict):
    """Print statistical comparison between conditions."""
    print("\n### Statistical Comparison ###")

    without_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("without_cot", [])
        if r.get("agent_objective_overall_score") is not None
    ]
    with_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("with_cot", [])
        if r.get("agent_objective_overall_score") is not None
    ]

    if without_scores and with_scores:
        without_mean = statistics.mean(without_scores)
        with_mean = statistics.mean(with_scores)
        diff = with_mean - without_mean

        print(f"\nWithout CoT mean: {without_mean*100:.1f}%")
        print(f"With CoT mean:    {with_mean*100:.1f}%")
        print(f"Difference:       {diff*100:+.1f}%")

        # Effect size (Cohen's d)
        if len(without_scores) > 1 and len(with_scores) > 1:
            pooled_std = ((statistics.stdev(without_scores)**2 + statistics.stdev(with_scores)**2) / 2) ** 0.5
            if pooled_std > 0:
                cohens_d = diff / pooled_std
                print(f"Cohen's d:        {cohens_d:.2f}")

        # Simple t-test approximation
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(with_scores, without_scores)
            print(f"t-statistic:      {t_stat:.2f}")
            print(f"p-value:          {p_value:.4f}")
            if p_value < 0.05:
                print("Result: SIGNIFICANT (p < 0.05)")
            else:
                print("Result: Not significant (p >= 0.05)")
        except ImportError:
            print("(Install scipy for t-test)")


def create_summary_markdown(condition_stats: dict, condition_results: dict, total_elapsed: float) -> str:
    """Create markdown summary of results."""
    lines = [
        "# Experiment: Estimator with Agent CoT Access",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Total Games**: {sum(d['n_games'] for d in condition_stats.values())}",
        "",
        "## Research Question",
        "",
        "Does giving the estimator access to agents' chain of thought (thinking blocks)",
        "improve its ability to infer their true objectives?",
        "",
        "## Hypothesis",
        "",
        "With CoT access, the estimator should achieve significantly higher accuracy",
        "because agents' private reasoning reveals their true preferences.",
        "",
        "## Conditions",
        "",
        "| Condition | Description |",
        "|-----------|-------------|",
        "| without_cot | Estimator sees only agent statements |",
        "| with_cot | Estimator sees statements + agent thinking |",
        "",
        "## Results",
        "",
        "| Condition | Exact F1 (mean) | Exact F1 (std) | Property Recall | Confidence |",
        "|-----------|-----------------|----------------|-----------------|------------|",
    ]

    for name in ["without_cot", "with_cot"]:
        if name not in condition_stats:
            continue
        stats = condition_stats[name]["stats"]
        f1 = stats.get("exact_f1", {})
        pr = stats.get("property_recall", {})
        conf = stats.get("avg_confidence", {})

        lines.append(f"| {name} | {f1.get('mean', 0)*100:.1f}% | {f1.get('std', 0)*100:.1f}% | "
                     f"{pr.get('mean', 0)*100:.1f}% | {conf.get('mean', 0):.1f} |")

    # Statistical comparison
    without_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("without_cot", [])
        if r.get("agent_objective_overall_score") is not None
    ]
    with_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("with_cot", [])
        if r.get("agent_objective_overall_score") is not None
    ]

    lines.extend([
        "",
        "## Statistical Comparison",
        "",
    ])

    if without_scores and with_scores:
        without_mean = statistics.mean(without_scores)
        with_mean = statistics.mean(with_scores)
        diff = with_mean - without_mean

        lines.extend([
            f"- **Without CoT mean**: {without_mean*100:.1f}%",
            f"- **With CoT mean**: {with_mean*100:.1f}%",
            f"- **Difference**: {diff*100:+.1f}%",
        ])

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
