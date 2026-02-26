#!/usr/bin/env python3
"""Experiment: Model Comparison for Objective Inference

Compare different Claude models as the estimator using the SAME game trajectories.
This isolates the effect of model capability from debate variance.

Models compared:
- claude-sonnet-4-20250514 (Sonnet 4)
- claude-opus-4-20250514 (Opus 4)
- claude-sonnet-4-5-20250929 (Sonnet 4.5)
- claude-opus-4-5-20251101 (Opus 4.5)
- claude-haiku-4-5-20251001 (Haiku 4.5)
- claude-opus-4-6 (Opus 4.6 - latest)

Design:
- Use saved game data from controlled context experiment (Phase 1)
- Run each model on the same 10 game transcripts
- Use "full" theory context (established as best cost/performance)
- Compare F1 scores across models
"""

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

# Models to compare (ordered by expected capability)
MODELS = [
    ("claude-haiku-4-5-20251001", "Haiku 4.5"),
    ("claude-sonnet-4-20250514", "Sonnet 4"),
    ("claude-sonnet-4-5-20250929", "Sonnet 4.5"),
    ("claude-sonnet-4-6", "Sonnet 4.6"),
    ("claude-opus-4-20250514", "Opus 4"),
    ("claude-opus-4-5-20251101", "Opus 4.5"),
    ("claude-opus-4-6", "Opus 4.6"),
]

# Use full theory context (established as best)
THEORY_CONTEXT = "full"

# Path to saved Phase 1 data
PHASE1_DIR = Path("outputs/controlled_context_experiment/20260225_120423")


def load_game_data(game_file: Path) -> dict:
    """Load game data from JSON file."""
    with open(game_file) as f:
        return json.load(f)


def run_inference(game_data: dict, model_id: str) -> dict:
    """Run estimator inference with a specific model."""

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

    # Create estimator with specified model and full theory context
    estimator = Estimator(
        model=model_id,
        condition="ids",
        enable_thinking=True,
        thinking_budget=5000,
        sees_agent_thinking=False,
        deception_strategy="baseline",
        theory_context=THEORY_CONTEXT,
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
        "model": model_id,
        "seed": game_data["seed"],
        "agent_objective_inference": inference_data,
        "agent_objective_scores": eval_result.evaluation_scores,
        "agent_objective_overall_score": eval_result.overall_score,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/model_comparison_experiment") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/model_comparison_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Experiment: Model Comparison for Objective Inference")
    print(f"{'='*70}")
    print(f"Models: {[m[1] for m in MODELS]}")
    print(f"Theory context: {THEORY_CONTEXT}")
    print(f"Loading game data from: {PHASE1_DIR}")

    wandb.init(
        project="truthification",
        name=f"model-comparison-{timestamp}",
        config={
            "experiment": "model_comparison",
            "models": [m[0] for m in MODELS],
            "theory_context": THEORY_CONTEXT,
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

    # Run inference for each model
    results_by_model = {model_id: [] for model_id, _ in MODELS}
    all_results = []

    start_time = time.time()
    total_inferences = len(games_data) * len(MODELS)
    inference_count = 0

    for model_id, model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"Model: {model_name} ({model_id})")
        print(f"{'='*70}")

        for seed, game_data in sorted(games_data.items()):
            inference_count += 1
            print(f"  [{inference_count}/{total_inferences}] seed={seed}...", end=" ", flush=True)
            inf_start = time.time()

            try:
                result = run_inference(game_data, model_id)
                results_by_model[model_id].append(result)
                all_results.append(result)

                inf_elapsed = time.time() - inf_start
                score = result["agent_objective_overall_score"]
                print(f"done ({inf_elapsed:.0f}s) - F1: {score*100:.1f}%")

                wandb.log({
                    "inference_number": inference_count,
                    "seed": seed,
                    "model": model_id,
                    "model_name": model_name,
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
    for model_id, model_name in MODELS:
        results = results_by_model[model_id]
        scores = [r["agent_objective_overall_score"] for r in results]
        if scores:
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0
            stderr = std / (len(scores) ** 0.5) if len(scores) > 1 else 0
            stats[model_id] = {
                "name": model_name,
                "mean": mean,
                "std": std,
                "stderr": stderr,
                "n": len(scores),
                "scores": scores
            }

    # Print results table
    print(f"\n{'Model':<20} | {'Mean F1':<12} | {'Std':<12} | {'SE':<12}")
    print("-" * 65)
    for model_id, model_name in MODELS:
        if model_id in stats:
            s = stats[model_id]
            print(f"{model_name:<20} | {s['mean']*100:>6.1f}%     | {s['std']*100:>6.1f}%     | {s['stderr']*100:>6.1f}%")

    # Paired comparisons vs Sonnet 4 (baseline)
    baseline_model = "claude-sonnet-4-20250514"
    if baseline_model in stats:
        baseline_scores = stats[baseline_model]["scores"]

        print(f"\n### Paired Comparison vs Sonnet 4 (baseline) ###")

        for model_id, model_name in MODELS:
            if model_id == baseline_model or model_id not in stats:
                continue

            model_scores = stats[model_id]["scores"]

            # Ensure same length (paired)
            if len(model_scores) != len(baseline_scores):
                print(f"\n{model_name}: Cannot compare (different sample sizes)")
                continue

            paired_diffs = [model_scores[i] - baseline_scores[i] for i in range(len(baseline_scores))]
            mean_diff = statistics.mean(paired_diffs)
            std_diff = statistics.stdev(paired_diffs) if len(paired_diffs) > 1 else 0
            se_diff = std_diff / (len(paired_diffs) ** 0.5)

            print(f"\n{model_name}:")
            print(f"  Mean difference: {mean_diff*100:+.1f}%")
            print(f"  SE of difference: {se_diff*100:.1f}%")

            try:
                from scipy import stats as scipy_stats
                t_stat, p_value = scipy_stats.ttest_rel(model_scores, baseline_scores)
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
    header = "Seed     |"
    for _, model_name in MODELS:
        short_name = model_name.replace(" ", "")[:8]
        header += f" {short_name:<8} |"
    print(header)
    print("-" * len(header))

    seeds = sorted(games_data.keys())
    for seed in seeds:
        row = f"{seed:<8} |"
        for model_id, _ in MODELS:
            score = next((r["agent_objective_overall_score"] for r in results_by_model.get(model_id, []) if r["seed"] == seed), None)
            if score is not None:
                row += f" {score*100:>5.0f}%   |"
            else:
                row += "    -    |"
        print(row)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_time_minutes": total_time / 60,
            "models": [{"id": m[0], "name": m[1]} for m in MODELS],
            "theory_context": THEORY_CONTEXT,
            "stats": {k: {kk: vv for kk, vv in v.items() if kk != "scores"} for k, v in stats.items()},
            "all_results": all_results,
        }, f, indent=2, default=str)

    # Create markdown report
    report = create_report(stats, results_by_model, MODELS, games_data, total_time)
    with open(output_dir / "README.md", "w") as f:
        f.write(report)
    with open(results_dir / f"README_{timestamp}.md", "w") as f:
        f.write(report)

    wandb.finish()
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")


def create_report(stats: dict, results_by_model: dict, models: list, games_data: dict, total_time: float) -> str:
    """Create markdown report."""
    lines = [
        "# Model Comparison Experiment",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_time/60:.1f} minutes",
        f"**Theory Context**: full (~200 words)",
        "",
        "## Design",
        "",
        "Compare different Claude models as the estimator on the **same game trajectories**.",
        "This isolates model capability from debate variance.",
        "",
        "## Models Tested",
        "",
        "| Model | API ID | Cost (input/output per MTok) |",
        "|-------|--------|------------------------------|",
        "| Haiku 4.5 | claude-haiku-4-5-20251001 | $1 / $5 |",
        "| Sonnet 4 | claude-sonnet-4-20250514 | $3 / $15 |",
        "| Sonnet 4.5 | claude-sonnet-4-5-20250929 | $3 / $15 |",
        "| Sonnet 4.6 | claude-sonnet-4-6 | $3 / $15 |",
        "| Opus 4 | claude-opus-4-20250514 | $15 / $75 |",
        "| Opus 4.5 | claude-opus-4-5-20251101 | $5 / $25 |",
        "| Opus 4.6 | claude-opus-4-6 | $5 / $25 |",
        "",
        "## Results",
        "",
        "| Model | Mean F1 | Std | SE |",
        "|-------|---------|-----|-----|",
    ]

    for model_id, model_name in models:
        if model_id in stats:
            s = stats[model_id]
            lines.append(f"| {model_name} | {s['mean']*100:.1f}% | {s['std']*100:.1f}% | {s['stderr']*100:.1f}% |")

    # Statistical comparison
    baseline_model = "claude-sonnet-4-20250514"
    if baseline_model in stats:
        baseline_scores = stats[baseline_model]["scores"]

        lines.extend([
            "",
            "## Statistical Comparison vs Sonnet 4 (baseline)",
            "",
        ])

        for model_id, model_name in models:
            if model_id == baseline_model or model_id not in stats:
                continue

            model_scores = stats[model_id]["scores"]
            if len(model_scores) != len(baseline_scores):
                continue

            paired_diffs = [model_scores[i] - baseline_scores[i] for i in range(len(baseline_scores))]
            mean_diff = statistics.mean(paired_diffs)
            std_diff = statistics.stdev(paired_diffs) if len(paired_diffs) > 1 else 0
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0

            p_str = ""
            try:
                from scipy import stats as scipy_stats
                _, p_value = scipy_stats.ttest_rel(model_scores, baseline_scores)
                p_str = f", p={p_value:.4f}"
                if p_value < 0.05:
                    p_str += " *"
            except ImportError:
                pass

            lines.append(f"- **{model_name}**: {mean_diff*100:+.1f}% (d={cohens_d:.2f}{p_str})")

    # Per-seed breakdown
    lines.extend([
        "",
        "## Per-Seed Breakdown",
        "",
    ])

    header = "| Seed |"
    for _, model_name in models:
        header += f" {model_name} |"
    lines.append(header)

    separator = "|------|"
    for _ in models:
        separator += "------|"
    lines.append(separator)

    seeds = sorted(games_data.keys())
    for seed in seeds:
        row = f"| {seed} |"
        for model_id, _ in models:
            score = next((r["agent_objective_overall_score"] for r in results_by_model.get(model_id, []) if r["seed"] == seed), None)
            if score is not None:
                row += f" {score*100:.0f}% |"
            else:
                row += " - |"
        lines.append(row)

    lines.extend([
        "",
        "## Key Findings",
        "",
        "(Analysis to be added after reviewing results)",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    main()
