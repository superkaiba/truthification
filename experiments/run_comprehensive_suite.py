#!/usr/bin/env python
"""Comprehensive experiment suite for V2 Hidden Value Game.

Tests how various factors affect observer performance:
- Number of oracle calls
- Number of agents
- Number of objects
- Rule complexity (proxy for property complexity)
"""

import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.environment.simulation_v2 import GameConfig, HiddenValueGame


def run_experiment(config: GameConfig, name: str) -> dict:
    """Run a single experiment and return results."""
    print(f"  Running: {name}...")
    start = time.time()

    game = HiddenValueGame(config)
    result = game.run()

    elapsed = time.time() - start

    return {
        "name": name,
        "config": {
            "n_objects": config.n_objects,
            "n_agents": config.n_agents,
            "n_rounds": config.n_rounds,
            "oracle_budget": config.oracle_budget,
            "selection_size": config.selection_size,
            "rule_complexity": config.rule_complexity,
        },
        "metrics": {
            "selection_accuracy": result.metrics["selection_accuracy"],
            "optimal_overlap": result.metrics["optimal_overlap"],
            "total_value": result.metrics["total_value"],
            "optimal_value": result.metrics["optimal_value"],
            "property_accuracy": result.metrics["property_accuracy"],
            "value_prediction_accuracy": result.metrics["value_prediction_accuracy"],
            "rule_inference_accuracy": result.metrics["rule_inference_accuracy"],
            "oracle_queries_used": result.metrics["oracle_queries_used"],
        },
        "elapsed_seconds": elapsed,
    }


def run_suite():
    """Run the comprehensive experiment suite."""
    output_dir = Path("outputs/v2_comprehensive_suite")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results = []

    # Base config for controlled experiments
    base_seed = 42
    base_rounds = 4
    base_selection = 4

    # =========================================================================
    # Experiment 1: Oracle Budget Ablation
    # =========================================================================
    print("\n=== Experiment 1: Oracle Budget Ablation ===")
    oracle_budgets = [0, 1, 2, 3, 5]

    for budget in oracle_budgets:
        config = GameConfig(
            n_objects=10,
            n_agents=2,
            n_rounds=base_rounds,
            selection_size=base_selection,
            oracle_budget=budget,
            rule_complexity="medium",
            seed=base_seed,
        )
        result = run_experiment(config, f"oracle_budget_{budget}")
        result["experiment_group"] = "oracle_budget"
        result["variable"] = budget
        all_results.append(result)
        print(f"    Oracle={budget}: accuracy={result['metrics']['selection_accuracy']*100:.1f}%")

    # =========================================================================
    # Experiment 2: Number of Agents
    # =========================================================================
    print("\n=== Experiment 2: Number of Agents ===")
    agent_counts = [2, 3, 4]

    for n_agents in agent_counts:
        config = GameConfig(
            n_objects=10,
            n_agents=n_agents,
            n_rounds=base_rounds,
            selection_size=base_selection,
            oracle_budget=2,
            rule_complexity="medium",
            seed=base_seed,
        )
        result = run_experiment(config, f"agents_{n_agents}")
        result["experiment_group"] = "n_agents"
        result["variable"] = n_agents
        all_results.append(result)
        print(f"    Agents={n_agents}: accuracy={result['metrics']['selection_accuracy']*100:.1f}%")

    # =========================================================================
    # Experiment 3: Number of Objects
    # =========================================================================
    print("\n=== Experiment 3: Number of Objects ===")
    object_counts = [6, 10, 15, 20]

    for n_objects in object_counts:
        # Scale selection size proportionally
        selection = max(3, n_objects // 4)
        config = GameConfig(
            n_objects=n_objects,
            n_agents=2,
            n_rounds=base_rounds,
            selection_size=selection,
            oracle_budget=2,
            rule_complexity="medium",
            seed=base_seed,
        )
        result = run_experiment(config, f"objects_{n_objects}")
        result["experiment_group"] = "n_objects"
        result["variable"] = n_objects
        all_results.append(result)
        print(f"    Objects={n_objects}: accuracy={result['metrics']['selection_accuracy']*100:.1f}%")

    # =========================================================================
    # Experiment 4: Rule Complexity (proxy for property relevance)
    # =========================================================================
    print("\n=== Experiment 4: Rule Complexity ===")
    complexities = ["simple", "medium", "complex"]

    for complexity in complexities:
        config = GameConfig(
            n_objects=10,
            n_agents=2,
            n_rounds=base_rounds,
            selection_size=base_selection,
            oracle_budget=2,
            rule_complexity=complexity,
            seed=base_seed,
        )
        result = run_experiment(config, f"complexity_{complexity}")
        result["experiment_group"] = "rule_complexity"
        result["variable"] = complexity
        all_results.append(result)
        print(f"    Complexity={complexity}: accuracy={result['metrics']['selection_accuracy']*100:.1f}%")

    # =========================================================================
    # Experiment 5: Rounds (conversation length)
    # =========================================================================
    print("\n=== Experiment 5: Number of Rounds ===")
    round_counts = [2, 3, 4, 5, 6]

    for n_rounds in round_counts:
        config = GameConfig(
            n_objects=10,
            n_agents=2,
            n_rounds=n_rounds,
            selection_size=n_rounds,  # Pick 1 per round
            oracle_budget=2,
            rule_complexity="medium",
            seed=base_seed,
        )
        result = run_experiment(config, f"rounds_{n_rounds}")
        result["experiment_group"] = "n_rounds"
        result["variable"] = n_rounds
        all_results.append(result)
        print(f"    Rounds={n_rounds}: accuracy={result['metrics']['selection_accuracy']*100:.1f}%")

    # =========================================================================
    # Save Results
    # =========================================================================
    output_file = output_dir / f"suite_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "total_experiments": len(all_results),
            "results": all_results,
        }, f, indent=2)

    print(f"\n=== Suite Complete ===")
    print(f"Total experiments: {len(all_results)}")
    print(f"Results saved to: {output_file}")

    # Print summary table
    print("\n=== Summary Table ===")
    print(f"{'Experiment':<25} {'Variable':<12} {'Selection%':<12} {'Property%':<12} {'Value%':<12}")
    print("-" * 75)
    for r in all_results:
        m = r["metrics"]
        print(f"{r['name']:<25} {str(r['variable']):<12} {m['selection_accuracy']*100:>10.1f}% {m['property_accuracy']*100:>10.1f}% {m['value_prediction_accuracy']*100:>10.1f}%")

    return all_results


if __name__ == "__main__":
    run_suite()
