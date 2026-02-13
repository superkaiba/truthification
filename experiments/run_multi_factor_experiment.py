#!/usr/bin/env python3
"""Comprehensive Multi-Factor Experiment with Per-Round Metrics.

Research Questions:
- How do different factors affect truth recovery and agent manipulation success?
- Does random oracle hurt performance compared to strategic querying?
- Do complex agent value functions lead to more sophisticated deception?
- Does knowing agent identities/interests help the estimator?

Factors Tested:
1. Observer condition: "blind" vs "ids" vs "interests"
2. Random oracle: True vs False
3. Agent value functions: Simple interests vs Complex value functions
4. Rule complexity: "simple" vs "medium" vs "complex"

Design: Full factorial on key factors with multiple seeds per condition.
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

# Factors to vary
OBSERVER_CONDITIONS = ["blind", "ids", "interests"]
RANDOM_ORACLE = [False, True]
USE_AGENT_VALUE_FUNCTIONS = [False, True]
RULE_COMPLEXITIES = ["simple", "medium", "complex"]

# Fixed game parameters
BASE_CONFIG = {
    "n_objects": 10,
    "n_agents": 2,
    "n_rounds": 10,  # Many rounds for better per-round tracking
    "oracle_budget": 8,
    "selection_size": 5,
    "enable_estimator": True,
    "infer_agent_objectives": True,  # Enable agent objective inference
    "enable_agent_thinking": True,  # Capture agent strategic thinking
    "agent_thinking_budget": 5000,
    # Models
    "estimator_model": "claude-sonnet-4-20250514",
    "agent_model": "claude-sonnet-4-20250514",
    "observer_model": "claude-sonnet-4-20250514",
    # Fixed structure
    "turn_structure": "interleaved",
    "oracle_timing": "before_response",
    "debate_structure": "open",
}


@dataclass
class ExperimentCondition:
    """A single experimental condition."""
    condition_id: str
    observer_condition: str
    random_oracle: bool
    use_agent_value_functions: bool
    rule_complexity: str

    def to_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "observer_condition": self.observer_condition,
            "random_oracle": self.random_oracle,
            "use_agent_value_functions": self.use_agent_value_functions,
            "rule_complexity": self.rule_complexity,
        }


def generate_conditions() -> list[ExperimentCondition]:
    """Generate all experimental conditions (full factorial)."""
    conditions = []

    for obs_cond, rand_oracle, use_vf, rule_comp in product(
        OBSERVER_CONDITIONS,
        RANDOM_ORACLE,
        USE_AGENT_VALUE_FUNCTIONS,
        RULE_COMPLEXITIES,
    ):
        cond_id = f"{obs_cond}_oracle-{'random' if rand_oracle else 'strategic'}_vf-{'complex' if use_vf else 'simple'}_{rule_comp}"
        conditions.append(ExperimentCondition(
            condition_id=cond_id,
            observer_condition=obs_cond,
            random_oracle=rand_oracle,
            use_agent_value_functions=use_vf,
            rule_complexity=rule_comp,
        ))

    return conditions


def run_single_game(condition: ExperimentCondition, seed: int) -> dict:
    """Run a single game with the given condition and seed."""
    config = GameConfig(
        **BASE_CONFIG,
        condition=condition.observer_condition,
        random_oracle=condition.random_oracle,
        use_agent_value_functions=condition.use_agent_value_functions,
        rule_complexity=condition.rule_complexity,
        seed=seed,
    )

    game = HiddenValueGame(config)
    result = game.run()

    return {
        "condition": condition.to_dict(),
        "seed": seed,
        "metrics": result.metrics,
        "estimator_metrics": result.estimator_metrics,
        "accuracy_progression": result.accuracy_progression,
        "config": result.config,
        "rounds": result.rounds,  # Include conversation transcripts
        "agents": result.agents,  # Include agent info
        "value_rule": result.value_rule,  # Include value rule for context
    }


def compute_condition_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics for a condition across seeds."""
    if not results:
        return {}

    # Helper to safely compute mean/std
    def safe_stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return {"mean": 0, "std": 0, "n": 0}
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        return {"mean": mean, "std": std, "n": len(values)}

    # Extract metrics across seeds
    metrics_keys = [
        # Selection quality
        "total_value", "optimal_value", "selection_accuracy",
        "value_vs_random", "normalized_value",
        # Truth recovery
        "property_accuracy", "rule_inference_accuracy",
        "property_accuracy_vs_random",
        # Decision quality
        "avg_decision_quality",
        # Baselines
        "random_property_accuracy_baseline",
    ]

    stats = {}
    for key in metrics_keys:
        values = [r["metrics"].get(key) for r in results]
        stats[key] = safe_stats(values)

    # Estimator metrics
    est_keys = ["property_accuracy", "rule_inference_accuracy"]
    for key in est_keys:
        values = [
            r["estimator_metrics"].get(key)
            for r in results
            if r.get("estimator_metrics")
        ]
        stats[f"estimator_{key}"] = safe_stats(values)

    # Agent cumulative values
    agent_values = {}
    for r in results:
        agent_cum = r["metrics"].get("agent_cumulative_values", {})
        for agent_id, value in agent_cum.items():
            if agent_id not in agent_values:
                agent_values[agent_id] = []
            agent_values[agent_id].append(value)

    stats["agent_cumulative_values"] = {
        agent_id: safe_stats(values)
        for agent_id, values in agent_values.items()
    }

    # Per-round value progression (average across seeds)
    n_rounds = BASE_CONFIG["n_rounds"]
    value_per_round = [[] for _ in range(n_rounds)]
    cumulative_per_round = [[] for _ in range(n_rounds)]

    for r in results:
        vpr = r["metrics"].get("value_per_round", [])
        cpr = r["metrics"].get("cumulative_value_per_round", [])
        for i in range(min(n_rounds, len(vpr))):
            value_per_round[i].append(vpr[i])
        for i in range(min(n_rounds, len(cpr))):
            cumulative_per_round[i].append(cpr[i])

    stats["value_per_round"] = [safe_stats(vals) for vals in value_per_round]
    stats["cumulative_value_per_round"] = [safe_stats(vals) for vals in cumulative_per_round]

    # Per-round agent values
    agent_value_per_round = {}
    for r in results:
        avpr = r["metrics"].get("agent_value_per_round", {})
        for agent_id, values in avpr.items():
            if agent_id not in agent_value_per_round:
                agent_value_per_round[agent_id] = [[] for _ in range(n_rounds)]
            for i, v in enumerate(values[:n_rounds]):
                agent_value_per_round[agent_id][i].append(v)

    stats["agent_value_per_round"] = {
        agent_id: [safe_stats(vals) for vals in rounds]
        for agent_id, rounds in agent_value_per_round.items()
    }

    # Accuracy progression
    acc_prog_keys = [
        "judge_property_accuracy", "judge_rule_accuracy",
        "estimator_property_accuracy", "estimator_rule_accuracy"
    ]
    acc_progression = []
    for round_idx in range(n_rounds):
        round_stats = {"round": round_idx + 1}
        for key in acc_prog_keys:
            values = []
            for r in results:
                prog = r.get("accuracy_progression", [])
                if round_idx < len(prog):
                    values.append(prog[round_idx].get(key, 0))
            round_stats[key] = safe_stats(values)
        acc_progression.append(round_stats)

    stats["accuracy_progression"] = acc_progression

    # Agent objective inference scores
    obj_scores = []
    for r in results:
        est_metrics = r.get("estimator_metrics", {})
        if est_metrics:
            score = est_metrics.get("agent_objective_overall_score")
            if score is not None:
                obj_scores.append(score)
    stats["objective_inference_score"] = safe_stats(obj_scores)

    return stats


def run_experiment():
    """Run the full multi-factor experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate all conditions
    conditions = generate_conditions()
    total_games = len(conditions) * len(SEEDS)

    print(f"\n{'='*70}")
    print("Multi-Factor Experiment: Comprehensive Metrics")
    print(f"{'='*70}")
    print(f"Factors:")
    print(f"  - Observer conditions: {OBSERVER_CONDITIONS}")
    print(f"  - Random oracle: {RANDOM_ORACLE}")
    print(f"  - Agent value functions: {USE_AGENT_VALUE_FUNCTIONS}")
    print(f"  - Rule complexities: {RULE_COMPLEXITIES}")
    print(f"Total conditions: {len(conditions)}")
    print(f"Seeds per condition: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    print(f"Rounds per game: {BASE_CONFIG['n_rounds']}")
    print(f"{'='*70}\n")

    # Initialize wandb
    wandb_run = wandb.init(
        project="truthification",
        name=f"multi-factor-{timestamp}",
        config={
            "experiment": "multi_factor",
            "observer_conditions": OBSERVER_CONDITIONS,
            "random_oracle": RANDOM_ORACLE,
            "use_agent_value_functions": USE_AGENT_VALUE_FUNCTIONS,
            "rule_complexities": RULE_COMPLEXITIES,
            "seeds": SEEDS,
            "total_conditions": len(conditions),
            "total_games": total_games,
            **BASE_CONFIG,
        },
    )

    # Output directories
    output_dir = Path("outputs/multi_factor") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/multi_factor")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    all_results = []
    condition_results = {}  # condition_id -> [results]

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

                # Quick summary
                metrics = result["metrics"]
                total_val = metrics.get("total_value", 0)
                prop_acc = metrics.get("property_accuracy", 0)
                agent_vals = metrics.get("agent_cumulative_values", {})
                agent_vals_str = ", ".join(f"{k}:{v}" for k, v in agent_vals.items())

                print(f"done ({game_elapsed:.0f}s) - Value:{total_val}, PropAcc:{prop_acc*100:.0f}%, Agents:[{agent_vals_str}]")

                # Log to wandb
                wandb.log({
                    "game_number": game_count,
                    "condition_id": condition.condition_id,
                    "observer_condition": condition.observer_condition,
                    "random_oracle": condition.random_oracle,
                    "use_agent_value_functions": condition.use_agent_value_functions,
                    "rule_complexity": condition.rule_complexity,
                    "seed": seed,
                    "game_time_seconds": game_elapsed,
                    # Core metrics
                    "total_value": total_val,
                    "selection_accuracy": metrics.get("selection_accuracy", 0),
                    "property_accuracy": prop_acc,
                    "rule_inference_accuracy": metrics.get("rule_inference_accuracy", 0),
                    "property_accuracy_vs_random": metrics.get("property_accuracy_vs_random", 0),
                    "avg_decision_quality": metrics.get("avg_decision_quality", 0),
                    # Agent values
                    **{f"agent_{k}_value": v for k, v in agent_vals.items()},
                })

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    total_elapsed = time.time() - start_time

    # Compute aggregate statistics per condition
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

    # Print summary tables
    print_summary(condition_stats, conditions)

    # Save results
    print(f"\n{'='*70}")
    print("Saving Results...")
    print(f"{'='*70}")

    # Save all results (detailed)
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save condition statistics
    with open(output_dir / "condition_stats.json", "w") as f:
        json.dump(condition_stats, f, indent=2, default=str)

    # Create summary markdown
    summary_md = create_summary_markdown(condition_stats, conditions, total_elapsed)
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
            cond["observer_condition"],
            cond["random_oracle"],
            cond["use_agent_value_functions"],
            cond["rule_complexity"],
            data["n_games"],
            stats.get("total_value", {}).get("mean", 0),
            stats.get("property_accuracy", {}).get("mean", 0),
            stats.get("rule_inference_accuracy", {}).get("mean", 0),
            stats.get("estimator_property_accuracy", {}).get("mean", 0),
            stats.get("objective_inference_score", {}).get("mean", 0),
        ])

    wandb.log({
        "summary_table": wandb.Table(
            columns=[
                "observer_condition", "random_oracle", "agent_value_functions",
                "rule_complexity", "n_games", "total_value_mean",
                "property_accuracy_mean", "rule_inference_accuracy_mean",
                "estimator_property_accuracy_mean", "objective_inference_score_mean"
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


def print_summary(condition_stats: dict, conditions: list[ExperimentCondition]):
    """Print summary tables to console."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    # Group by key factors
    print("\n### By Observer Condition ###\n")
    print(f"{'Condition':<12} | {'Value':<12} | {'Prop Acc':<12} | {'Rule Acc':<12} | {'Est Prop':<12}")
    print("-" * 65)

    for obs_cond in OBSERVER_CONDITIONS:
        matching = [
            condition_stats[c.condition_id]
            for c in conditions
            if c.observer_condition == obs_cond and c.condition_id in condition_stats
        ]
        if not matching:
            continue

        # Average across matching conditions
        values = [m["stats"].get("total_value", {}).get("mean", 0) for m in matching]
        prop_accs = [m["stats"].get("property_accuracy", {}).get("mean", 0) for m in matching]
        rule_accs = [m["stats"].get("rule_inference_accuracy", {}).get("mean", 0) for m in matching]
        est_props = [m["stats"].get("estimator_property_accuracy", {}).get("mean", 0) for m in matching]

        avg_val = statistics.mean(values) if values else 0
        avg_prop = statistics.mean(prop_accs) if prop_accs else 0
        avg_rule = statistics.mean(rule_accs) if rule_accs else 0
        avg_est = statistics.mean(est_props) if est_props else 0

        print(f"{obs_cond:<12} | {avg_val:<12.1f} | {avg_prop*100:<11.1f}% | {avg_rule*100:<11.1f}% | {avg_est*100:<11.1f}%")

    print("\n### Random vs Strategic Oracle ###\n")
    print(f"{'Oracle':<12} | {'Value':<12} | {'Prop Acc':<12} | {'Rule Acc':<12}")
    print("-" * 55)

    for rand_oracle in RANDOM_ORACLE:
        matching = [
            condition_stats[c.condition_id]
            for c in conditions
            if c.random_oracle == rand_oracle and c.condition_id in condition_stats
        ]
        if not matching:
            continue

        values = [m["stats"].get("total_value", {}).get("mean", 0) for m in matching]
        prop_accs = [m["stats"].get("property_accuracy", {}).get("mean", 0) for m in matching]
        rule_accs = [m["stats"].get("rule_inference_accuracy", {}).get("mean", 0) for m in matching]

        label = "Random" if rand_oracle else "Strategic"
        avg_val = statistics.mean(values) if values else 0
        avg_prop = statistics.mean(prop_accs) if prop_accs else 0
        avg_rule = statistics.mean(rule_accs) if rule_accs else 0

        print(f"{label:<12} | {avg_val:<12.1f} | {avg_prop*100:<11.1f}% | {avg_rule*100:<11.1f}%")

    print("\n### Agent Value Functions ###\n")
    print(f"{'Type':<12} | {'Value':<12} | {'Prop Acc':<12} | {'Obj Inf Score':<14}")
    print("-" * 55)

    for use_vf in USE_AGENT_VALUE_FUNCTIONS:
        matching = [
            condition_stats[c.condition_id]
            for c in conditions
            if c.use_agent_value_functions == use_vf and c.condition_id in condition_stats
        ]
        if not matching:
            continue

        values = [m["stats"].get("total_value", {}).get("mean", 0) for m in matching]
        prop_accs = [m["stats"].get("property_accuracy", {}).get("mean", 0) for m in matching]
        obj_scores = [m["stats"].get("objective_inference_score", {}).get("mean", 0) for m in matching]

        label = "Complex VF" if use_vf else "Simple"
        avg_val = statistics.mean(values) if values else 0
        avg_prop = statistics.mean(prop_accs) if prop_accs else 0
        avg_obj = statistics.mean(obj_scores) if obj_scores else 0

        print(f"{label:<12} | {avg_val:<12.1f} | {avg_prop*100:<11.1f}% | {avg_obj*100:<13.1f}%")


def create_summary_markdown(
    condition_stats: dict,
    conditions: list[ExperimentCondition],
    total_elapsed: float,
) -> str:
    """Create markdown summary of results."""
    lines = [
        "# Multi-Factor Experiment Results",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Total Games**: {sum(d['n_games'] for d in condition_stats.values())}",
        "",
        "## Experimental Design",
        "",
        "### Factors",
        "",
        f"- **Observer Conditions**: {OBSERVER_CONDITIONS}",
        f"- **Random Oracle**: {RANDOM_ORACLE}",
        f"- **Agent Value Functions**: {USE_AGENT_VALUE_FUNCTIONS}",
        f"- **Rule Complexities**: {RULE_COMPLEXITIES}",
        f"- **Seeds per Condition**: {len(SEEDS)}",
        "",
        "### Fixed Parameters",
        "",
        f"- Objects: {BASE_CONFIG['n_objects']}",
        f"- Agents: {BASE_CONFIG['n_agents']}",
        f"- Rounds: {BASE_CONFIG['n_rounds']}",
        f"- Oracle Budget: {BASE_CONFIG['oracle_budget']}",
        f"- Selection Size: {BASE_CONFIG['selection_size']}",
        "",
        "## Results Summary",
        "",
        "### By Observer Condition",
        "",
        "| Condition | Value | Prop Acc | Rule Acc | Est Prop Acc |",
        "|-----------|-------|----------|----------|--------------|",
    ]

    for obs_cond in OBSERVER_CONDITIONS:
        matching = [
            condition_stats[c.condition_id]
            for c in conditions
            if c.observer_condition == obs_cond and c.condition_id in condition_stats
        ]
        if not matching:
            continue

        values = [m["stats"].get("total_value", {}).get("mean", 0) for m in matching]
        prop_accs = [m["stats"].get("property_accuracy", {}).get("mean", 0) for m in matching]
        rule_accs = [m["stats"].get("rule_inference_accuracy", {}).get("mean", 0) for m in matching]
        est_props = [m["stats"].get("estimator_property_accuracy", {}).get("mean", 0) for m in matching]

        avg_val = statistics.mean(values) if values else 0
        avg_prop = statistics.mean(prop_accs) if prop_accs else 0
        avg_rule = statistics.mean(rule_accs) if rule_accs else 0
        avg_est = statistics.mean(est_props) if est_props else 0

        lines.append(f"| {obs_cond} | {avg_val:.1f} | {avg_prop*100:.1f}% | {avg_rule*100:.1f}% | {avg_est*100:.1f}% |")

    lines.extend([
        "",
        "### Random vs Strategic Oracle",
        "",
        "| Oracle Type | Value | Prop Acc | Rule Acc |",
        "|-------------|-------|----------|----------|",
    ])

    for rand_oracle in RANDOM_ORACLE:
        matching = [
            condition_stats[c.condition_id]
            for c in conditions
            if c.random_oracle == rand_oracle and c.condition_id in condition_stats
        ]
        if not matching:
            continue

        values = [m["stats"].get("total_value", {}).get("mean", 0) for m in matching]
        prop_accs = [m["stats"].get("property_accuracy", {}).get("mean", 0) for m in matching]
        rule_accs = [m["stats"].get("rule_inference_accuracy", {}).get("mean", 0) for m in matching]

        label = "Random" if rand_oracle else "Strategic"
        avg_val = statistics.mean(values) if values else 0
        avg_prop = statistics.mean(prop_accs) if prop_accs else 0
        avg_rule = statistics.mean(rule_accs) if rule_accs else 0

        lines.append(f"| {label} | {avg_val:.1f} | {avg_prop*100:.1f}% | {avg_rule*100:.1f}% |")

    lines.extend([
        "",
        "### Agent Value Functions",
        "",
        "| Type | Value | Prop Acc | Obj Inference Score |",
        "|------|-------|----------|---------------------|",
    ])

    for use_vf in USE_AGENT_VALUE_FUNCTIONS:
        matching = [
            condition_stats[c.condition_id]
            for c in conditions
            if c.use_agent_value_functions == use_vf and c.condition_id in condition_stats
        ]
        if not matching:
            continue

        values = [m["stats"].get("total_value", {}).get("mean", 0) for m in matching]
        prop_accs = [m["stats"].get("property_accuracy", {}).get("mean", 0) for m in matching]
        obj_scores = [m["stats"].get("objective_inference_score", {}).get("mean", 0) for m in matching]

        label = "Complex VF" if use_vf else "Simple Interest"
        avg_val = statistics.mean(values) if values else 0
        avg_prop = statistics.mean(prop_accs) if prop_accs else 0
        avg_obj = statistics.mean(obj_scores) if obj_scores else 0

        lines.append(f"| {label} | {avg_val:.1f} | {avg_prop*100:.1f}% | {avg_obj*100:.1f}% |")

    lines.extend([
        "",
        "### By Rule Complexity",
        "",
        "| Complexity | Value | Prop Acc | Rule Acc |",
        "|------------|-------|----------|----------|",
    ])

    for rule_comp in RULE_COMPLEXITIES:
        matching = [
            condition_stats[c.condition_id]
            for c in conditions
            if c.rule_complexity == rule_comp and c.condition_id in condition_stats
        ]
        if not matching:
            continue

        values = [m["stats"].get("total_value", {}).get("mean", 0) for m in matching]
        prop_accs = [m["stats"].get("property_accuracy", {}).get("mean", 0) for m in matching]
        rule_accs = [m["stats"].get("rule_inference_accuracy", {}).get("mean", 0) for m in matching]

        avg_val = statistics.mean(values) if values else 0
        avg_prop = statistics.mean(prop_accs) if prop_accs else 0
        avg_rule = statistics.mean(rule_accs) if rule_accs else 0

        lines.append(f"| {rule_comp} | {avg_val:.1f} | {avg_prop*100:.1f}% | {avg_rule*100:.1f}% |")

    lines.extend([
        "",
        "## Full Condition Results",
        "",
        "| Condition | N | Value (mean±std) | Prop Acc | Rule Acc | Est Prop Acc |",
        "|-----------|---|------------------|----------|----------|--------------|",
    ])

    for cond_id, data in sorted(condition_stats.items()):
        stats = data["stats"]
        n = data["n_games"]

        val_mean = stats.get("total_value", {}).get("mean", 0)
        val_std = stats.get("total_value", {}).get("std", 0)
        prop_acc = stats.get("property_accuracy", {}).get("mean", 0)
        rule_acc = stats.get("rule_inference_accuracy", {}).get("mean", 0)
        est_prop = stats.get("estimator_property_accuracy", {}).get("mean", 0)

        lines.append(
            f"| {cond_id[:40]} | {n} | {val_mean:.1f}±{val_std:.1f} | "
            f"{prop_acc*100:.1f}% | {rule_acc*100:.1f}% | {est_prop*100:.1f}% |"
        )

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
