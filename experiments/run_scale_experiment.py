"""
Experiment: Scale up agents and rounds.

Tests how truth recovery changes with:
- More agents (2, 3, 4)
- More rounds (10, 15, 20)
"""

import json
import sys
import statistics
import time
from datetime import datetime
from pathlib import Path

import wandb
from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from src.environment.simulation import run_game

# Experiment configuration
SEEDS = [42, 123, 456]  # 3 seeds per condition
N_AGENTS_OPTIONS = [2, 3, 4]
N_ROUNDS_OPTIONS = [10, 15, 20]

# Base config
BASE_CONFIG = {
    "n_objects": 10,
    "oracle_budget": 8,
    "selection_size": 5,
    "enable_estimator": True,
    "infer_agent_objectives": False,
    "condition": "interests",
    "use_agent_value_functions": True,
    "agent_value_function_complexity": "medium",
    "rule_complexity": "medium",
    "random_oracle": False,
}


def run_experiment():
    """Run the scaling experiment."""
    
    run = wandb.init(
        project="truthification",
        name=f"scale-experiment-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "experiment": "scale_agents_rounds",
            "seeds": SEEDS,
            "n_agents_options": N_AGENTS_OPTIONS,
            "n_rounds_options": N_ROUNDS_OPTIONS,
            **BASE_CONFIG,
        },
    )
    
    results = {}
    all_results = []
    
    # Total conditions
    conditions = [(n_agents, n_rounds) for n_agents in N_AGENTS_OPTIONS for n_rounds in N_ROUNDS_OPTIONS]
    total_games = len(conditions) * len(SEEDS)
    game_num = 0
    start_time = time.time()
    
    print("=" * 60, flush=True)
    print("SCALE EXPERIMENT: AGENTS × ROUNDS", flush=True)
    print("=" * 60, flush=True)
    print(f"Agent counts: {N_AGENTS_OPTIONS}", flush=True)
    print(f"Round counts: {N_ROUNDS_OPTIONS}", flush=True)
    print(f"Seeds: {SEEDS}", flush=True)
    print(f"Total games: {total_games}", flush=True)
    print(flush=True)
    
    for n_agents in N_AGENTS_OPTIONS:
        for n_rounds in N_ROUNDS_OPTIONS:
            condition_key = f"{n_agents}agents_{n_rounds}rounds"
            results[condition_key] = []
            
            print(f"\n--- {n_agents} Agents, {n_rounds} Rounds ---", flush=True)
            
            for seed in SEEDS:
                game_num += 1
                elapsed = time.time() - start_time
                avg_time = elapsed / game_num if game_num > 1 else 0
                remaining = avg_time * (total_games - game_num)
                
                print(f"  [{game_num}/{total_games}] Seed {seed}... (ETA: {remaining/60:.1f}m)", flush=True)
                
                try:
                    result = run_game(
                        seed=seed,
                        n_objects=BASE_CONFIG["n_objects"],
                        n_agents=n_agents,
                        n_rounds=n_rounds,
                        oracle_budget=BASE_CONFIG["oracle_budget"],
                        selection_size=BASE_CONFIG["selection_size"],
                        condition=BASE_CONFIG["condition"],
                        rule_complexity=BASE_CONFIG["rule_complexity"],
                        enable_estimator=BASE_CONFIG["enable_estimator"],
                        use_agent_value_functions=BASE_CONFIG["use_agent_value_functions"],
                        agent_value_function_complexity=BASE_CONFIG["agent_value_function_complexity"],
                        infer_agent_objectives=BASE_CONFIG["infer_agent_objectives"],
                        random_oracle=BASE_CONFIG["random_oracle"],
                    )
                    
                    metrics = result.metrics
                    est_metrics = result.estimator_metrics or {}
                    
                    results[condition_key].append({
                        "seed": seed,
                        "property_accuracy": metrics["property_accuracy"],
                        "rule_inference_accuracy": metrics["rule_inference_accuracy"],
                        "total_value": metrics["total_value"],
                        "estimator_property_accuracy": est_metrics.get("property_accuracy", 0),
                    })
                    
                    all_results.append({
                        "n_agents": n_agents,
                        "n_rounds": n_rounds,
                        "seed": seed,
                        "metrics": metrics,
                        "estimator_metrics": est_metrics,
                    })
                    
                    print(f"    Judge Acc: {metrics['property_accuracy']*100:.1f}%, Est Acc: {est_metrics.get('property_accuracy', 0)*100:.1f}%", flush=True)
                    
                    wandb.log({
                        f"agents_{n_agents}/rounds_{n_rounds}/property_accuracy": metrics["property_accuracy"],
                        f"agents_{n_agents}/rounds_{n_rounds}/estimator_accuracy": est_metrics.get("property_accuracy", 0),
                        f"agents_{n_agents}/rounds_{n_rounds}/total_value": metrics["total_value"],
                        "game_num": game_num,
                    })
                    
                except Exception as e:
                    print(f"    ERROR: {e}", flush=True)
                    continue
    
    # Compute summary
    print("\n" + "=" * 60, flush=True)
    print("RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    
    summary = {}
    for condition_key, data in results.items():
        if not data:
            continue
        
        prop_accs = [d["property_accuracy"] for d in data]
        est_accs = [d["estimator_property_accuracy"] for d in data]
        values = [d["total_value"] for d in data]
        
        summary[condition_key] = {
            "property_accuracy": {
                "mean": statistics.mean(prop_accs),
                "std": statistics.stdev(prop_accs) if len(prop_accs) > 1 else 0,
                "se": statistics.stdev(prop_accs) / len(prop_accs)**0.5 if len(prop_accs) > 1 else 0,
            },
            "estimator_accuracy": {
                "mean": statistics.mean(est_accs),
                "std": statistics.stdev(est_accs) if len(est_accs) > 1 else 0,
            },
            "total_value": {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
            },
            "n": len(data),
        }
        
        print(f"\n{condition_key}:", flush=True)
        print(f"  Judge Accuracy:     {summary[condition_key]['property_accuracy']['mean']*100:.1f}% (±{summary[condition_key]['property_accuracy']['se']*100:.1f}%)", flush=True)
        print(f"  Estimator Accuracy: {summary[condition_key]['estimator_accuracy']['mean']*100:.1f}%", flush=True)
        print(f"  Total Value:        {summary[condition_key]['total_value']['mean']:.1f}", flush=True)
    
    # Print comparison tables
    print("\n\n=== JUDGE ACCURACY BY AGENTS × ROUNDS ===", flush=True)
    print(f"{'':15}", end="", flush=True)
    for n_rounds in N_ROUNDS_OPTIONS:
        print(f"{n_rounds} rounds".center(12), end="", flush=True)
    print(flush=True)
    
    for n_agents in N_AGENTS_OPTIONS:
        print(f"{n_agents} agents".ljust(15), end="", flush=True)
        for n_rounds in N_ROUNDS_OPTIONS:
            key = f"{n_agents}agents_{n_rounds}rounds"
            if key in summary:
                acc = summary[key]["property_accuracy"]["mean"] * 100
                print(f"{acc:.1f}%".center(12), end="", flush=True)
            else:
                print("N/A".center(12), end="", flush=True)
        print(flush=True)
    
    print("\n=== ESTIMATOR ACCURACY BY AGENTS × ROUNDS ===", flush=True)
    print(f"{'':15}", end="", flush=True)
    for n_rounds in N_ROUNDS_OPTIONS:
        print(f"{n_rounds} rounds".center(12), end="", flush=True)
    print(flush=True)
    
    for n_agents in N_AGENTS_OPTIONS:
        print(f"{n_agents} agents".ljust(15), end="", flush=True)
        for n_rounds in N_ROUNDS_OPTIONS:
            key = f"{n_agents}agents_{n_rounds}rounds"
            if key in summary:
                acc = summary[key]["estimator_accuracy"]["mean"] * 100
                print(f"{acc:.1f}%".center(12), end="", flush=True)
            else:
                print("N/A".center(12), end="", flush=True)
        print(flush=True)
    
    # Save results
    output_dir = Path("results/scale_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(output_dir / f"results_{timestamp}.json", "w") as f:
        json.dump({
            "summary": summary,
            "all_results": all_results,
            "config": BASE_CONFIG,
        }, f, indent=2)
    
    wandb.finish()
    
    print(f"\nResults saved to {output_dir}", flush=True)
    print(f"W&B run: {run.url}", flush=True)
    
    return summary


if __name__ == "__main__":
    run_experiment()
