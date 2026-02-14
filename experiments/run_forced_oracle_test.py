#!/usr/bin/env python3
"""
Quick test: Compare no oracle vs forced oracle usage.

This properly tests whether oracle access helps by FORCING the LLM to use it.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(line_buffering=True)

from src.environment.simulation import run_game

# Test config
SEEDS = [42, 123, 456]
N_ROUNDS = 10
N_OBJECTS = 10
SELECTION_SIZE = 5
ORACLE_BUDGET = 8

def run_test():
    results = {"no_oracle": [], "forced_oracle": []}

    print("=" * 60)
    print("FORCED ORACLE TEST")
    print("=" * 60)
    print(f"Seeds: {SEEDS}")
    print(f"Oracle budget: {ORACLE_BUDGET}")
    print()

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # No oracle (budget=0)
        print("  Running no oracle...")
        result_no = run_game(
            seed=seed,
            n_objects=N_OBJECTS,
            n_rounds=N_ROUNDS,
            oracle_budget=0,
            selection_size=SELECTION_SIZE,
            condition="interests",
            rule_complexity="medium",
            enable_estimator=True,
        )
        results["no_oracle"].append({
            "seed": seed,
            "property_accuracy": result_no.metrics["property_accuracy"],
            "oracle_used": result_no.metrics["oracle_queries_used"],
        })
        print(f"    Property acc: {result_no.metrics['property_accuracy']*100:.1f}%, Oracle used: {result_no.metrics['oracle_queries_used']}")

        # Forced oracle
        print("  Running forced oracle...")
        result_forced = run_game(
            seed=seed,
            n_objects=N_OBJECTS,
            n_rounds=N_ROUNDS,
            oracle_budget=ORACLE_BUDGET,
            selection_size=SELECTION_SIZE,
            condition="interests",
            rule_complexity="medium",
            enable_estimator=True,
            force_oracle=True,  # KEY: Force oracle usage
        )
        results["forced_oracle"].append({
            "seed": seed,
            "property_accuracy": result_forced.metrics["property_accuracy"],
            "oracle_used": result_forced.metrics["oracle_queries_used"],
        })
        print(f"    Property acc: {result_forced.metrics['property_accuracy']*100:.1f}%, Oracle used: {result_forced.metrics['oracle_queries_used']}")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    no_acc = [r["property_accuracy"] for r in results["no_oracle"]]
    forced_acc = [r["property_accuracy"] for r in results["forced_oracle"]]
    forced_queries = [r["oracle_used"] for r in results["forced_oracle"]]

    print(f"\nNo Oracle (budget=0):")
    print(f"  Property accuracy: {sum(no_acc)/len(no_acc)*100:.1f}%")

    print(f"\nForced Oracle (budget={ORACLE_BUDGET}):")
    print(f"  Property accuracy: {sum(forced_acc)/len(forced_acc)*100:.1f}%")
    print(f"  Avg queries used: {sum(forced_queries)/len(forced_queries):.1f}")

    diff = sum(forced_acc)/len(forced_acc) - sum(no_acc)/len(no_acc)
    print(f"\nDifference: {diff*100:+.1f} percentage points")

    # Save results
    output_dir = Path("results/forced_oracle_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir}")

if __name__ == "__main__":
    run_test()
