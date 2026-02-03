#!/usr/bin/env python3
"""Minimal test run to verify end-to-end functionality."""

import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from src.environment.simulation import run_game, GameConfig, HiddenValueGame


def test_single_game():
    """Step 1: Test single game with minimal parameters."""
    print("=" * 60)
    print("Minimal Test: Single Game")
    print("=" * 60)
    print()
    print("Parameters:")
    print("  n_objects: 10")
    print("  n_agents: 2")
    print("  n_rounds: 1")
    print("  oracle_budget: 2")
    print("  selection_size: 3")
    print("  seed: 42")
    print("  rule_complexity: simple")
    print("  condition: ids")
    print()
    print("Running game...")
    print()

    result = run_game(
        n_objects=10,
        n_agents=2,
        n_rounds=1,
        oracle_budget=2,
        selection_size=3,
        seed=42,
        rule_complexity="simple",
        condition="ids",
    )

    print("Game completed!")
    print()

    # Print metrics
    print("=" * 60)
    print("METRICS")
    print("=" * 60)
    print()

    metrics = result.metrics

    # Core metrics
    print("Core Selection Metrics:")
    print(f"  Total value achieved: {metrics['total_value']}")
    print(f"  Optimal value:        {metrics['optimal_value']}")
    print(f"  Selection accuracy:   {metrics['selection_accuracy']:.3f}")
    print(f"  Optimal overlap:      {metrics['optimal_overlap']}/{result.config['selection_size']}")
    print()

    # Oracle usage
    print("Oracle Usage:")
    print(f"  Queries used: {metrics['oracle_queries_used']}/{metrics['oracle_budget']}")
    print(f"  Oracle efficiency: {metrics['oracle_efficiency']:.1f} value/query")
    print()

    # Truth recovery
    print("Truth Recovery Metrics:")
    print(f"  Property accuracy:       {metrics['property_accuracy']:.3f}")
    print(f"  Rule inference accuracy: {metrics['rule_inference_accuracy']:.3f}")
    print(f"  Rule confidence:         {metrics['rule_confidence']}")
    print()

    # Agent outcomes
    print("Agent Win Rates:")
    for agent_id, win_rate in metrics['agent_win_rates'].items():
        print(f"  {agent_id}: {win_rate:.3f}")
    print()

    # Baseline comparisons
    print("Baseline Comparisons:")
    print(f"  Random selection value:    {metrics['random_selection_value']:.1f}")
    print(f"  Random selection accuracy: {metrics['random_selection_accuracy']:.3f}")
    print(f"  Majority vote accuracy:    {metrics['majority_vote_accuracy']:.3f}")
    print()

    print("  Single agent trust values:")
    for agent_id, value in metrics['single_agent_trust_values'].items():
        print(f"    {agent_id}: {value}")
    print()

    print("Relative Performance:")
    print(f"  Value vs random:     {metrics['value_vs_random']:+.1f}")
    print(f"  Value vs best agent: {metrics['value_vs_best_agent']:+.1f}")
    print()

    # Show selected objects
    print("=" * 60)
    print("SELECTION DETAILS")
    print("=" * 60)
    print()
    print(f"Selected objects: {result.final_selection}")
    print()

    # Show inferred rule
    if result.inferred_rule:
        print("Observer's inferred rule:")
        print(f"  Description: {result.inferred_rule['description']}")
        print(f"  Confidence:  {result.inferred_rule['confidence']}")
        print(f"  Key factors: {result.inferred_rule['key_factors']}")
        print()

    # Show actual rule
    print("Actual value rule:")
    print(f"  Name: {result.value_rule['name']}")
    print(f"  Description: {result.value_rule['description']}")
    print()

    # Verification
    print("=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    print()

    checks = []

    # Check 1: Game completed without errors
    checks.append(("Single game completes without errors", True))

    # Check 2: Metrics are computed
    required_metrics = [
        'selection_accuracy', 'property_accuracy', 'total_value',
        'optimal_value', 'oracle_queries_used', 'agent_win_rates'
    ]
    metrics_computed = all(k in metrics for k in required_metrics)
    checks.append(("Metrics are computed", metrics_computed))

    # Check 3: Baseline comparisons work
    baselines_work = (
        'random_selection_value' in metrics and
        'single_agent_trust_values' in metrics and
        len(metrics['single_agent_trust_values']) > 0
    )
    checks.append(("Baseline comparisons work", baselines_work))

    # Check 4: Selection size matches
    correct_selection_size = len(result.final_selection) == result.config['selection_size']
    checks.append(("Correct selection size", correct_selection_size))

    # Print verification results
    for check_name, passed in checks:
        status = "PASS" if passed else "FAIL"
        symbol = "[x]" if passed else "[ ]"
        print(f"  {symbol} {check_name}")

    print()

    # Overall result
    all_passed = all(passed for _, passed in checks)
    if all_passed:
        print("All checks passed! Implementation is working.")
    else:
        print("Some checks failed. See details above.")

    return result, all_passed


def main():
    """Run the minimal test."""
    try:
        result, success = test_single_game()

        # Save result for inspection
        output_dir = Path("outputs/test")
        output_dir.mkdir(parents=True, exist_ok=True)
        result.save(output_dir / "minimal_test_result.json")
        print(f"\nResult saved to: {output_dir / 'minimal_test_result.json'}")

        return 0 if success else 1

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
