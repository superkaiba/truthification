"""
Experiment: Compare property accuracy with and without oracle access.

Tests oracle_budget=0 vs oracle_budget=8 to measure the value of oracle queries.
"""

import json
import os
import sys
import statistics
import time
from datetime import datetime
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv()

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Import after dotenv to ensure API keys are loaded
from src.environment.simulation import run_game

# Experiment configuration
SEEDS = [42, 123, 456, 789, 101]
ORACLE_BUDGETS = [0, 8]  # No oracle vs standard oracle

# Base config (matches multi-factor experiment)
BASE_CONFIG = {
    "n_objects": 10,
    "n_agents": 2,
    "n_rounds": 10,
    "selection_size": 5,
    "enable_estimator": True,
    "infer_agent_objectives": False,  # Skip to save time
    "enable_agent_thinking": False,
    "observer_condition": "interests",  # Best performing condition
    "use_agent_value_functions": True,
    "agent_value_function_complexity": "medium",
    "rule_complexity": "medium",
    "random_oracle": False,  # Strategic oracle
}


def run_experiment():
    """Run the no-oracle comparison experiment."""
    
    # Initialize wandb
    run = wandb.init(
        project="truthification",
        name=f"no-oracle-comparison-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "experiment": "no_oracle_comparison",
            "seeds": SEEDS,
            "oracle_budgets": ORACLE_BUDGETS,
            **BASE_CONFIG,
        },
    )
    
    results = {budget: [] for budget in ORACLE_BUDGETS}
    all_results = []
    
    total_games = len(ORACLE_BUDGETS) * len(SEEDS)
    game_num = 0
    start_time = time.time()
    
    print("=" * 60, flush=True)
    print("NO-ORACLE COMPARISON EXPERIMENT", flush=True)
    print("=" * 60, flush=True)
    print(f"Oracle budgets: {ORACLE_BUDGETS}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Total games: {total_games}", flush=True)
    print(flush=True)
    
    for oracle_budget in ORACLE_BUDGETS:
        print(f"\n--- Oracle Budget: {oracle_budget} ---")
        
        for seed in SEEDS:
            game_num += 1
            elapsed = time.time() - start_time
            avg_time = elapsed / game_num if game_num > 1 else 0
            remaining = avg_time * (total_games - game_num)
            
            print(f"  [{game_num}/{total_games}] Seed {seed}... (ETA: {remaining/60:.1f}m)")
            
            try:
                result = run_game(
                    seed=seed,
                    n_objects=BASE_CONFIG["n_objects"],
                    n_agents=BASE_CONFIG["n_agents"],
                    n_rounds=BASE_CONFIG["n_rounds"],
                    oracle_budget=oracle_budget,
                    selection_size=BASE_CONFIG["selection_size"],
                    condition=BASE_CONFIG["observer_condition"],  # Fixed: condition not observer_condition
                    rule_complexity=BASE_CONFIG["rule_complexity"],
                    enable_estimator=BASE_CONFIG["enable_estimator"],
                    use_agent_value_functions=BASE_CONFIG["use_agent_value_functions"],
                    agent_value_function_complexity=BASE_CONFIG["agent_value_function_complexity"],
                    infer_agent_objectives=BASE_CONFIG["infer_agent_objectives"],
                    random_oracle=BASE_CONFIG["random_oracle"],
                )
                
                metrics = result.metrics
                results[oracle_budget].append({
                    "seed": seed,
                    "property_accuracy": metrics["property_accuracy"],
                    "rule_inference_accuracy": metrics["rule_inference_accuracy"],
                    "total_value": metrics["total_value"],
                    "estimator_property_accuracy": result.estimator_metrics["property_accuracy"] if result.estimator_metrics else None,
                })
                
                all_results.append({
                    "oracle_budget": oracle_budget,
                    "seed": seed,
                    "metrics": metrics,
                    "estimator_metrics": result.estimator_metrics,
                })
                
                print(f"    Property Acc: {metrics['property_accuracy']*100:.1f}%, Value: {metrics['total_value']}")
                
                # Log to wandb
                wandb.log({
                    f"oracle_{oracle_budget}/property_accuracy": metrics["property_accuracy"],
                    f"oracle_{oracle_budget}/total_value": metrics["total_value"],
                    f"oracle_{oracle_budget}/rule_inference": metrics["rule_inference_accuracy"],
                    "game_num": game_num,
                })
                
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
    
    # Compute summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    summary = {}
    for budget in ORACLE_BUDGETS:
        data = results[budget]
        if not data:
            continue
        
        prop_accs = [d["property_accuracy"] for d in data]
        values = [d["total_value"] for d in data]
        rule_accs = [d["rule_inference_accuracy"] for d in data]
        est_accs = [d["estimator_property_accuracy"] for d in data if d["estimator_property_accuracy"]]
        
        summary[budget] = {
            "property_accuracy": {
                "mean": statistics.mean(prop_accs),
                "std": statistics.stdev(prop_accs) if len(prop_accs) > 1 else 0,
                "se": statistics.stdev(prop_accs) / len(prop_accs)**0.5 if len(prop_accs) > 1 else 0,
            },
            "total_value": {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
            },
            "rule_inference": {
                "mean": statistics.mean(rule_accs),
                "std": statistics.stdev(rule_accs) if len(rule_accs) > 1 else 0,
            },
            "estimator_accuracy": {
                "mean": statistics.mean(est_accs) if est_accs else 0,
                "std": statistics.stdev(est_accs) if len(est_accs) > 1 else 0,
            },
            "n": len(data),
        }
        
        print(f"\nOracle Budget = {budget}:")
        print(f"  Property Accuracy: {summary[budget]['property_accuracy']['mean']*100:.1f}% (±{summary[budget]['property_accuracy']['se']*100:.1f}%)")
        print(f"  Rule Inference:    {summary[budget]['rule_inference']['mean']*100:.1f}%")
        print(f"  Total Value:       {summary[budget]['total_value']['mean']:.1f}")
        print(f"  Estimator Acc:     {summary[budget]['estimator_accuracy']['mean']*100:.1f}%")
        print(f"  N games:           {summary[budget]['n']}")
    
    # Compute difference
    if 0 in summary and 8 in summary:
        diff = summary[8]["property_accuracy"]["mean"] - summary[0]["property_accuracy"]["mean"]
        print(f"\n** Oracle Value: +{diff*100:.1f}% property accuracy **")
        print(f"   (Budget=8: {summary[8]['property_accuracy']['mean']*100:.1f}% vs Budget=0: {summary[0]['property_accuracy']['mean']*100:.1f}%)")
    
    # Save results
    output_dir = Path("results/no_oracle_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        json.dump({
            "summary": summary,
            "all_results": all_results,
            "config": BASE_CONFIG,
        }, f, indent=2)
    
    # Log final summary to wandb
    wandb.log({
        "summary/oracle_0_property_accuracy": summary.get(0, {}).get("property_accuracy", {}).get("mean", 0),
        "summary/oracle_8_property_accuracy": summary.get(8, {}).get("property_accuracy", {}).get("mean", 0),
        "summary/oracle_value_delta": diff if 0 in summary and 8 in summary else 0,
    })
    
    wandb.finish()
    
    print(f"\nResults saved to {output_dir}")
    print(f"W&B run: {run.url}")
    
    return summary


if __name__ == "__main__":
    run_experiment()
