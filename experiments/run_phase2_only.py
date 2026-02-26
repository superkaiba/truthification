#!/usr/bin/env python3
"""Re-run Phase 2 of controlled context experiment using saved game data."""

import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import wandb
from src.environment.world import World, Property, PropertyType, Object
from src.environment.estimator import Estimator
from src.environment.agent import Statement

THEORY_CONDITIONS = ["none", "full", "strategy_list", "comprehensive"]

# Path to saved Phase 1 data
PHASE1_DIR = Path("outputs/controlled_context_experiment/20260225_120423")


def load_game_data(game_file: Path) -> dict:
    """Load game data from JSON file."""
    with open(game_file) as f:
        return json.load(f)


def run_inference(game_data: dict, theory_context: str) -> dict:
    """Run estimator inference on saved game data with specific theory context."""

    # Reconstruct the world
    prop_defs = []
    for p in game_data["property_definitions"]:
        values = p["possible_values"]
        if all(isinstance(v, bool) for v in values):
            prop_type = PropertyType.BOOLEAN
        elif all(isinstance(v, (int, float)) for v in values):
            prop_type = PropertyType.NUMERIC
        else:
            prop_type = PropertyType.CATEGORICAL
        prop_defs.append(Property(name=p["name"], property_type=prop_type, possible_values=values))

    world = World(property_definitions=prop_defs)

    # Add objects to world from world_state
    for obj_id, obj_data in game_data["world_state"].get("objects", {}).items():
        obj = Object(id=obj_id, properties=obj_data.get("properties", {}))
        world.add_object(obj)

    # Create estimator with specified theory context
    estimator = Estimator(
        model="claude-sonnet-4-20250514",
        condition="ids",
        enable_thinking=True,
        thinking_budget=5000,
        sees_agent_thinking=False,
        deception_strategy="baseline",
        theory_context=theory_context,
    )

    # Convert statements to Statement objects
    statements = [
        Statement(
            agent_id=s["agent_id"],
            text=s["text"],
            thinking=s.get("thinking"),
        )
        for s in game_data["statements"]
    ]

    # Run principled inference
    inferences = estimator.infer_agent_objectives_principled(
        all_statements=statements,
        agents=game_data["agents"],
        world=world,
    )

    # Evaluate using overlap scoring
    eval_result = estimator.evaluate_objective_inference_overlap(
        inferences=inferences,
        agents=game_data["agents"],
    )

    # Extract results
    inference_data = {}
    for agent_id, inf in inferences.items():
        inference_data[agent_id] = {
            "inferred_goal": inf.inferred_goal,
            "inferred_factors": inf.inferred_factors,
            "predicted_properties": inf.predicted_properties,
            "confidence": inf.confidence,
            "reasoning": inf.reasoning,
        }
        if eval_result.overlap_scores and agent_id in eval_result.overlap_scores:
            inference_data[agent_id]["overlap_metrics"] = eval_result.overlap_scores[agent_id].to_dict()

    return {
        "theory_context": theory_context,
        "seed": game_data["seed"],
        "agent_objective_inference": inference_data,
        "agent_objective_scores": eval_result.evaluation_scores,
        "agent_objective_overall_score": eval_result.overall_score,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/controlled_context_experiment") / f"phase2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Phase 2: Re-running estimator inference with different contexts")
    print(f"{'='*70}")
    print(f"Loading game data from: {PHASE1_DIR}")

    wandb.init(
        project="truthification",
        name=f"controlled-context-phase2-{timestamp}",
        config={
            "experiment": "controlled_context_phase2",
            "theory_conditions": THEORY_CONDITIONS,
            "phase1_dir": str(PHASE1_DIR),
        },
    )

    # Load all game data
    game_files = sorted(PHASE1_DIR.glob("game_seed*_full.json"))
    print(f"Found {len(game_files)} game files")

    games_data = {}
    for gf in game_files:
        data = load_game_data(gf)
        seed = data["seed"]
        games_data[seed] = data
        print(f"  Loaded seed={seed}: {len(data['statements'])} statements")

    # Run Phase 2
    results_by_context = {tc: [] for tc in THEORY_CONDITIONS}
    all_results = []

    start_time = time.time()
    total_inferences = len(games_data) * len(THEORY_CONDITIONS)
    inference_count = 0

    for seed, game_data in sorted(games_data.items()):
        print(f"\nSeed {seed}:")

        for tc in THEORY_CONDITIONS:
            inference_count += 1
            print(f"  [{inference_count}/{total_inferences}] context={tc}...", end=" ", flush=True)
            inf_start = time.time()

            try:
                result = run_inference(game_data, tc)
                results_by_context[tc].append(result)
                all_results.append(result)

                inf_elapsed = time.time() - inf_start
                score = result["agent_objective_overall_score"]
                print(f"done ({inf_elapsed:.0f}s) - F1: {score*100:.1f}%")

                wandb.log({
                    "inference_number": inference_count,
                    "seed": seed,
                    "theory_context": tc,
                    "inference_time_seconds": inf_elapsed,
                    "objective_inference_score": score,
                })

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    total_time = time.time() - start_time

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    stats = {}
    for tc, results in results_by_context.items():
        scores = [r["agent_objective_overall_score"] for r in results]
        if scores:
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0
            stderr = std / (len(scores) ** 0.5) if len(scores) > 1 else 0
            stats[tc] = {"mean": mean, "std": std, "stderr": stderr, "n": len(scores), "scores": scores}

    # Print results table
    print(f"\n{'Context':<16} | {'Mean F1':<12} | {'Std':<12} | {'SE':<12}")
    print("-" * 60)
    for tc in THEORY_CONDITIONS:
        if tc in stats:
            s = stats[tc]
            print(f"{tc:<16} | {s['mean']*100:>6.1f}%     | {s['std']*100:>6.1f}%     | {s['stderr']*100:>6.1f}%")

    # Paired comparison vs baseline
    print("\n### Paired Comparison vs Baseline (none) ###")
    if "none" in stats:
        baseline_scores = stats["none"]["scores"]

        for tc in THEORY_CONDITIONS:
            if tc == "none" or tc not in stats:
                continue

            tc_scores = stats[tc]["scores"]
            paired_diffs = [tc_scores[i] - baseline_scores[i] for i in range(len(baseline_scores))]
            mean_diff = statistics.mean(paired_diffs)
            std_diff = statistics.stdev(paired_diffs) if len(paired_diffs) > 1 else 0
            se_diff = std_diff / (len(paired_diffs) ** 0.5)

            print(f"\n{tc}:")
            print(f"  Mean difference: {mean_diff*100:+.1f}%")
            print(f"  SE of difference: {se_diff*100:.1f}%")

            try:
                from scipy import stats as scipy_stats
                t_stat, p_value = scipy_stats.ttest_rel(tc_scores, baseline_scores)
                print(f"  Paired t-test p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print(f"  ** SIGNIFICANT (p < 0.05) **")
            except ImportError:
                pass

            if std_diff > 0:
                cohens_d = mean_diff / std_diff
                print(f"  Cohen's d (paired): {cohens_d:.2f}")

    # Per-seed breakdown
    print("\n### Per-Seed Breakdown ###")
    print(f"{'Seed':<8} | {'none':<8} | {'full':<8} | {'strategy':<8} | {'comprehensive':<8}")
    print("-" * 60)

    seeds = sorted(games_data.keys())
    for seed in seeds:
        row = [f"{seed:<8}"]
        for tc in THEORY_CONDITIONS:
            score = next((r["agent_objective_overall_score"] for r in results_by_context.get(tc, []) if r["seed"] == seed), None)
            if score is not None:
                row.append(f"{score*100:>5.0f}%  ")
            else:
                row.append("   -    ")
        print(" | ".join(row))

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_time_minutes": total_time / 60,
            "stats": {tc: {k: v for k, v in s.items() if k != "scores"} for tc, s in stats.items()},
            "all_results": all_results,
        }, f, indent=2, default=str)

    wandb.finish()
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
