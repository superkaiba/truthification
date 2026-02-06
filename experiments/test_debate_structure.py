#!/usr/bin/env python3
"""Test script for per-round accuracy tracking and debate structure experiment.

Two modes:
1. `--quick`: Run 1 game per condition (8 total) for quick testing
2. `--validate`: Run 1 game to validate per-round accuracy tracking
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.environment.simulation import GameConfig, HiddenValueGame

# Test with 1 seed per condition
TURN_STRUCTURES = ["interleaved", "batch", "simultaneous", "sequential"]
ORACLE_TIMINGS = ["before_response", "after_statements"]
SEED = 42

# Game config - 20 rounds for detailed progression tracking
GAME_CONFIG = {
    "n_objects": 20,
    "n_agents": 2,
    "n_rounds": 20,  # 20 rounds for detailed accuracy progression
    "oracle_budget": 10,  # More queries spread across rounds
    "selection_size": 1,  # 1 pick per round for cleaner metrics
    "enable_estimator": True,
    "estimator_model": "claude-sonnet-4-20250514",
    "agent_model": "claude-sonnet-4-20250514",
    "observer_model": "claude-sonnet-4-20250514",
    "rule_complexity": "medium",
    "condition": "ids",
}


def validate_per_round_accuracy():
    """Validate per-round accuracy tracking with a single game."""
    print("=" * 60)
    print("Validating Per-Round Accuracy Tracking")
    print("=" * 60)

    config = GameConfig(
        **GAME_CONFIG,
        turn_structure="interleaved",
        oracle_timing="before_response",
        seed=SEED,
    )

    print(f"\nConfig:")
    print(f"  - Objects: {config.n_objects}")
    print(f"  - Rounds: {config.n_rounds}")
    print(f"  - Selection size: {config.selection_size}")
    print(f"  - Estimator: {config.enable_estimator}")

    print("\nRunning game...")
    game = HiddenValueGame(config)
    result = game.run()

    # Check round metrics
    print("\n" + "=" * 60)
    print("Round Metrics")
    print("=" * 60)

    for r in result.rounds:
        rm = r.get("round_metrics", {})
        if rm:
            print(f"\nRound {r['round_number']}:")
            print(f"  Judge Property Accuracy: {rm.get('judge_property_accuracy', 0)*100:.1f}%")
            print(f"  Judge Rule Accuracy: {rm.get('judge_rule_accuracy', 0)*100:.1f}%")
            print(f"  Estimator Property Accuracy: {rm.get('estimator_property_accuracy', 0)*100:.1f}%")
            print(f"  Estimator Rule Accuracy: {rm.get('estimator_rule_accuracy', 0)*100:.1f}%")
            print(f"  Cumulative Value: {rm.get('cumulative_value', 0)}")
            print(f"  Decision Quality: {rm.get('decision_quality', 0)*100:.1f}%")

    # Check accuracy progression
    print("\n" + "=" * 60)
    print("Accuracy Progression")
    print("=" * 60)

    progression = result.accuracy_progression
    if progression:
        print(f"\n{'Round':<8} | {'Judge Prop':<12} | {'Est Prop':<12} | {'Judge Rule':<12} | {'Est Rule':<12}")
        print("-" * 65)
        for p in progression:
            print(
                f"{p['round']:<8} | "
                f"{p['judge_property_accuracy']*100:>10.1f}% | "
                f"{p['estimator_property_accuracy']*100:>10.1f}% | "
                f"{p['judge_rule_accuracy']*100:>10.1f}% | "
                f"{p['estimator_rule_accuracy']*100:>10.1f}%"
            )
    else:
        print("WARNING: No accuracy progression data found!")

    # Check final metrics
    print("\n" + "=" * 60)
    print("Final Metrics")
    print("=" * 60)

    metrics = result.metrics
    print(f"  Selection Accuracy: {metrics.get('selection_accuracy', 0)*100:.1f}%")
    print(f"  Observer Property Accuracy: {metrics.get('property_accuracy', 0)*100:.1f}%")
    print(f"  Observer Rule Accuracy: {metrics.get('rule_inference_accuracy', 0)*100:.1f}%")

    if result.estimator_metrics:
        print(f"  Estimator Property Accuracy: {result.estimator_metrics.get('property_accuracy', 0)*100:.1f}%")
        print(f"  Estimator Rule Accuracy: {result.estimator_metrics.get('rule_inference_accuracy', 0)*100:.1f}%")

    # Validation checks
    print("\n" + "=" * 60)
    print("Validation Checks")
    print("=" * 60)

    errors = []

    # Check that we have round metrics for each round
    for i, r in enumerate(result.rounds):
        rm = r.get("round_metrics")
        if not rm:
            errors.append(f"Round {i+1} missing round_metrics")
        else:
            # Check that accuracy fields exist
            if "judge_property_accuracy" not in rm:
                errors.append(f"Round {i+1} missing judge_property_accuracy")
            if "estimator_property_accuracy" not in rm:
                errors.append(f"Round {i+1} missing estimator_property_accuracy")

    # Check accuracy progression
    if not progression:
        errors.append("No accuracy_progression in GameResult")
    elif len(progression) != config.n_rounds:
        errors.append(f"Expected {config.n_rounds} progression entries, got {len(progression)}")

    # Check that progression matches round metrics
    for i, p in enumerate(progression):
        rm = result.rounds[i].get("round_metrics", {}) if i < len(result.rounds) else {}
        if rm:
            if abs(p.get("judge_property_accuracy", 0) - rm.get("judge_property_accuracy", 0)) > 0.001:
                errors.append(f"Round {i+1}: progression judge_prop doesn't match round_metrics")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("\n  All validation checks passed!")
        return True


def run_quick_test():
    """Run 1 game per condition (8 total)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/debate_structure_test") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = [(ts, ot) for ts in TURN_STRUCTURES for ot in ORACLE_TIMINGS]
    total = len(conditions)

    print(f"\n{'='*70}")
    print("Debate Structure Test: 1 game per condition")
    print(f"{'='*70}")
    print(f"Conditions: {total} ({len(TURN_STRUCTURES)} structures × {len(ORACLE_TIMINGS)} timings)")
    print(f"Seed: {SEED}")
    print(f"{'='*70}\n")

    all_results = []
    start_time = time.time()

    for i, (ts, ot) in enumerate(conditions, 1):
        print(f"[{i}/{total}] {ts} + {ot}...", end=" ", flush=True)

        try:
            game_start = time.time()

            config = GameConfig(
                **GAME_CONFIG,
                turn_structure=ts,
                oracle_timing=ot,
                seed=SEED,
            )

            game = HiddenValueGame(config)
            result = game.run()

            game_elapsed = time.time() - game_start

            # Extract key metrics
            obs_prop_acc = result.metrics.get("property_accuracy", 0)
            obs_rule_acc = result.metrics.get("rule_inference_accuracy", 0)
            sel_acc = result.metrics.get("selection_accuracy", 0)

            est_prop_acc = 0
            est_rule_acc = 0
            if result.estimator_metrics:
                est_prop_acc = result.estimator_metrics.get("property_accuracy", 0)
                est_rule_acc = result.estimator_metrics.get("rule_inference_accuracy", 0)

            game_data = {
                "turn_structure": ts,
                "oracle_timing": ot,
                "seed": SEED,
                "observer_property_accuracy": obs_prop_acc,
                "observer_rule_accuracy": obs_rule_acc,
                "selection_accuracy": sel_acc,
                "estimator_property_accuracy": est_prop_acc,
                "estimator_rule_accuracy": est_rule_acc,
                "estimator_advantage_property": est_prop_acc - obs_prop_acc,
                "estimator_advantage_rule": est_rule_acc - obs_rule_acc,
                "game_time_seconds": game_elapsed,
                "total_value": result.metrics.get("total_value", 0),
                "optimal_value": result.metrics.get("optimal_value", 0),
                "oracle_queries_used": result.metrics.get("oracle_queries_used", 0),
                "accuracy_progression": result.accuracy_progression,
            }

            # Save full game result
            result.save(output_dir / f"game_{ts}_{ot}.json")

            all_results.append(game_data)

            # Print per-round accuracy
            if result.accuracy_progression:
                prog_str = " | ".join(
                    f"R{p['round']}:{p['estimator_property_accuracy']*100:.0f}%"
                    for p in result.accuracy_progression
                )
                print(f"done ({game_elapsed:.1f}s) - Est: {est_prop_acc*100:.0f}% [{prog_str}]")
            else:
                print(f"done ({game_elapsed:.1f}s) - Est: {est_prop_acc*100:.0f}%, Obs: {obs_prop_acc*100:.0f}%")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                "turn_structure": ts,
                "oracle_timing": ot,
                "seed": SEED,
                "error": str(e),
            })

    total_elapsed = time.time() - start_time

    # Save summary
    summary = {
        "timestamp": timestamp,
        "total_time_seconds": total_elapsed,
        "config": GAME_CONFIG,
        "results": all_results,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Test complete! Total time: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}")

    return output_dir, all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test debate structure experiment")
    parser.add_argument("--quick", action="store_true", help="Run quick test (1 game per condition)")
    parser.add_argument("--validate", action="store_true", help="Validate per-round accuracy tracking")
    args = parser.parse_args()

    if args.validate:
        success = validate_per_round_accuracy()
        sys.exit(0 if success else 1)
    elif args.quick:
        run_quick_test()
    else:
        # Default: validate
        success = validate_per_round_accuracy()
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
