#!/usr/bin/env python3
"""
Experiment: Compare different oracle budget levels (0-8)
Tests how many oracle queries are needed for effective truth recovery.
"""

import json
import os
import statistics
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # Load .env file

import wandb

from src.environment.simulation import GameConfig, HiddenValueGame

# Experiment config
ORACLE_BUDGETS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
SEEDS = [42, 123, 456]
N_ROUNDS = 10

BASE_CONFIG = {
    "n_objects": 10,
    "n_agents": 2,
    "n_rounds": N_ROUNDS,
    "selection_size": 5,
    "condition": "interests",
    "rule_complexity": "medium",
    "enable_estimator": False,  # Focus on judge only for speed
    "force_oracle": True,  # Force oracle to be used when budget > 0
}


def run_experiment():
    """Run the oracle budget comparison experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/oracle_budget_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb.init(
        project="truthification",
        name=f"oracle_budget_comparison_{timestamp}",
        config={
            "experiment": "oracle_budget_comparison",
            "oracle_budgets": ORACLE_BUDGETS,
            "seeds": SEEDS,
            "n_rounds": N_ROUNDS,
            **BASE_CONFIG,
        },
    )

    results = {budget: [] for budget in ORACLE_BUDGETS}
    all_results = []

    total_games = len(ORACLE_BUDGETS) * len(SEEDS)
    game_count = 0

    print(f"\n{'='*60}")
    print("Oracle Budget Experiment")
    print(f"{'='*60}")
    print(f"Budgets to test: {ORACLE_BUDGETS}")
    print(f"Seeds: {SEEDS}")
    print(f"Total games: {total_games}")
    print(f"{'='*60}\n")

    for budget in ORACLE_BUDGETS:
        print(f"\n--- Oracle Budget: {budget} ---")

        for seed in SEEDS:
            game_count += 1
            print(f"  [{game_count}/{total_games}] seed={seed}...", end=" ", flush=True)

            try:
                config = GameConfig(
                    **BASE_CONFIG,
                    oracle_budget=budget,
                    seed=seed,
                )

                game = HiddenValueGame(config)
                result = game.run()

                metrics = result.metrics
                prop_acc = metrics.get("property_accuracy", 0)
                oracle_used = metrics.get("oracle_queries_used", 0)
                total_value = metrics.get("total_value", 0)

                results[budget].append({
                    "seed": seed,
                    "property_accuracy": prop_acc,
                    "oracle_queries_used": oracle_used,
                    "total_value": total_value,
                })

                all_results.append({
                    "oracle_budget": budget,
                    "seed": seed,
                    "property_accuracy": prop_acc,
                    "oracle_queries_used": oracle_used,
                    "total_value": total_value,
                })

                print(f"Acc: {prop_acc*100:.1f}%, Oracle used: {oracle_used}/{budget}")

                wandb.log({
                    "oracle_budget": budget,
                    "seed": seed,
                    "property_accuracy": prop_acc,
                    "oracle_queries_used": oracle_used,
                    "total_value": total_value,
                    "game_num": game_count,
                })

                # Save trajectory
                trajectory_dir = output_dir / "trajectories"
                trajectory_dir.mkdir(exist_ok=True)
                with open(trajectory_dir / f"game_budget{budget}_seed{seed}.json", "w") as f:
                    json.dump(result.to_dict(), f, indent=2)

            except Exception as e:
                print(f"ERROR: {e}")
                continue

    # Compute summary statistics
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}\n")

    summary = {}
    for budget in ORACLE_BUDGETS:
        data = results[budget]
        if not data:
            continue

        accs = [d["property_accuracy"] for d in data]
        oracle_used = [d["oracle_queries_used"] for d in data]

        summary[budget] = {
            "mean_accuracy": statistics.mean(accs),
            "std_accuracy": statistics.stdev(accs) if len(accs) > 1 else 0,
            "mean_oracle_used": statistics.mean(oracle_used),
            "n": len(data),
        }

        print(f"Budget {budget}: {summary[budget]['mean_accuracy']*100:.1f}% "
              f"(±{summary[budget]['std_accuracy']*100:.1f}%) "
              f"[avg {summary[budget]['mean_oracle_used']:.1f} queries used]")

    # Save results
    output_file = output_dir / f"results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "summary": summary,
            "all_results": all_results,
            "config": BASE_CONFIG,
            "oracle_budgets": ORACLE_BUDGETS,
            "seeds": SEEDS,
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Log summary to wandb
    for budget, stats in summary.items():
        wandb.log({
            f"summary/budget_{budget}_accuracy": stats["mean_accuracy"],
            f"summary/budget_{budget}_std": stats["std_accuracy"],
        })

    # Create wandb table
    table = wandb.Table(columns=["budget", "mean_accuracy", "std_accuracy", "n"])
    for budget, stats in summary.items():
        table.add_data(budget, stats["mean_accuracy"], stats["std_accuracy"], stats["n"])
    wandb.log({"oracle_budget_results": table})

    wandb.finish()

    return summary


if __name__ == "__main__":
    run_experiment()
