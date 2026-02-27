#!/usr/bin/env python3
"""Experiment: F1 Evolution Over Number of Statements

Test how objective inference accuracy improves as the estimator sees more statements.
Uses saved game trajectories and runs inference at multiple checkpoints.
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

# Checkpoints: number of statements to include
CHECKPOINTS = [6, 12, 18, 24, 30, 36, 42, 48]

# Path to saved game data
PHASE1_DIR = Path("outputs/controlled_context_experiment/20260225_120423")


def load_game_data(game_file: Path) -> dict:
    """Load game data from JSON file."""
    with open(game_file) as f:
        return json.load(f)


def run_inference_at_checkpoint(game_data: dict, n_statements: int) -> dict:
    """Run estimator inference using only first n_statements."""

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

    # Add objects to world
    for obj_id, obj_data in game_data["world_state"].get("objects", {}).items():
        obj = Object(id=obj_id, properties=obj_data.get("properties", {}))
        world.add_object(obj)

    # Create estimator with full theory context (established as best)
    estimator = Estimator(
        model="claude-haiku-4-5-20251001",  # Use Haiku for speed (same performance as larger models)
        condition="ids",
        enable_thinking=True,
        thinking_budget=5000,
        sees_agent_thinking=False,
        deception_strategy="baseline",
        theory_context="full",
    )

    # Take only first n_statements
    all_statements = game_data["statements"][:n_statements]

    # Convert to Statement objects
    statements = [
        Statement(
            agent_id=s["agent_id"],
            text=s["text"],
            thinking=s.get("thinking"),
        )
        for s in all_statements
    ]

    # Run inference
    inferences = estimator.infer_agent_objectives_principled(
        all_statements=statements,
        agents=game_data["agents"],
        world=world,
    )

    # Evaluate
    eval_result = estimator.evaluate_objective_inference_overlap(
        inferences=inferences,
        agents=game_data["agents"],
    )

    return {
        "n_statements": n_statements,
        "f1_score": eval_result.overall_score,
        "per_agent_scores": eval_result.evaluation_scores,
    }


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/f1_evolution_experiment") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Experiment: F1 Evolution Over Number of Statements")
    print(f"{'='*70}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Loading game data from: {PHASE1_DIR}")

    wandb.init(
        project="truthification",
        name=f"f1-evolution-{timestamp}",
        config={
            "experiment": "f1_evolution",
            "checkpoints": CHECKPOINTS,
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

    # Run inference at each checkpoint for each game
    results_by_checkpoint = {cp: [] for cp in CHECKPOINTS}
    all_results = []

    start_time = time.time()
    total_inferences = len(games_data) * len(CHECKPOINTS)
    inference_count = 0

    for seed, game_data in sorted(games_data.items()):
        print(f"\nSeed {seed}:")

        for checkpoint in CHECKPOINTS:
            inference_count += 1
            print(f"  [{inference_count}/{total_inferences}] n={checkpoint}...", end=" ", flush=True)
            inf_start = time.time()

            try:
                result = run_inference_at_checkpoint(game_data, checkpoint)
                result["seed"] = seed
                results_by_checkpoint[checkpoint].append(result)
                all_results.append(result)

                inf_elapsed = time.time() - inf_start
                print(f"done ({inf_elapsed:.0f}s) - F1: {result['f1_score']*100:.1f}%")

                wandb.log({
                    "inference_number": inference_count,
                    "seed": seed,
                    "n_statements": checkpoint,
                    "f1_score": result["f1_score"],
                    "inference_time_seconds": inf_elapsed,
                })

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    total_time = time.time() - start_time

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS: F1 Evolution Over Statements")
    print("=" * 70)

    print(f"\n{'Statements':<12} | {'Mean F1':<10} | {'Std':<10} | {'SE':<10}")
    print("-" * 50)

    evolution_data = []
    for checkpoint in CHECKPOINTS:
        scores = [r["f1_score"] for r in results_by_checkpoint[checkpoint]]
        if scores:
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0
            se = std / (len(scores) ** 0.5) if len(scores) > 1 else 0
            evolution_data.append({
                "n_statements": checkpoint,
                "mean_f1": mean,
                "std": std,
                "se": se,
                "n": len(scores),
            })
            print(f"{checkpoint:<12} | {mean*100:>6.1f}%   | {std*100:>6.1f}%   | {se*100:>6.1f}%")

    # Calculate improvement from first to last
    if len(evolution_data) >= 2:
        first = evolution_data[0]
        last = evolution_data[-1]
        improvement = last["mean_f1"] - first["mean_f1"]
        print(f"\nImprovement from {first['n_statements']} to {last['n_statements']} statements:")
        print(f"  {first['mean_f1']*100:.1f}% -> {last['mean_f1']*100:.1f}% ({improvement*100:+.1f}%)")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_time_minutes": total_time / 60,
            "checkpoints": CHECKPOINTS,
            "evolution_data": evolution_data,
            "all_results": all_results,
        }, f, indent=2, default=str)

    wandb.finish()
    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
