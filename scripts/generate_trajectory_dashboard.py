#!/usr/bin/env python3
"""Generate trajectory dashboard from multi-factor experiment results."""

import json
from pathlib import Path
from collections import defaultdict
import statistics

def load_results():
    """Load all results from the experiment."""
    results_path = Path("outputs/multi_factor/20260207_104922/all_results.json")
    with open(results_path) as f:
        return json.load(f)

def compute_trajectory_stats(results):
    """Compute average trajectories across conditions."""
    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        cond_id = r["condition"]["condition_id"]
        by_condition[cond_id].append(r)

    trajectory_data = {}

    for cond_id, games in by_condition.items():
        n_rounds = 10

        # Aggregate trajectories
        cumulative_values = [[] for _ in range(n_rounds)]
        value_per_round = [[] for _ in range(n_rounds)]
        agent_a_values = [[] for _ in range(n_rounds)]
        agent_b_values = [[] for _ in range(n_rounds)]

        for game in games:
            metrics = game.get("metrics", {})

            # Cumulative value
            cv = metrics.get("cumulative_value_per_round", [])
            for i, v in enumerate(cv[:n_rounds]):
                cumulative_values[i].append(v)

            # Value per round
            vpr = metrics.get("value_per_round", [])
            for i, v in enumerate(vpr[:n_rounds]):
                value_per_round[i].append(v)

            # Agent values
            agent_cv = metrics.get("agent_cumulative_value_per_round", {})
            for agent_id, values in agent_cv.items():
                if "Agent_A" in agent_id:
                    target = agent_a_values
                elif "Agent_B" in agent_id:
                    target = agent_b_values
                else:
                    continue
                for i, v in enumerate(values[:n_rounds]):
                    target[i].append(v)

        # Compute means
        trajectory_data[cond_id] = {
            "cumulative_value": [
                statistics.mean(vals) if vals else 0
                for vals in cumulative_values
            ],
            "value_per_round": [
                statistics.mean(vals) if vals else 0
                for vals in value_per_round
            ],
            "agent_a_cumulative": [
                statistics.mean(vals) if vals else 0
                for vals in agent_a_values
            ],
            "agent_b_cumulative": [
                statistics.mean(vals) if vals else 0
                for vals in agent_b_values
            ],
            "n_games": len(games)
        }

    return trajectory_data

def compute_overall_trajectories(results):
    """Compute overall average trajectories."""
    n_rounds = 10

    cumulative_values = [[] for _ in range(n_rounds)]
    agent_a_values = [[] for _ in range(n_rounds)]
    agent_b_values = [[] for _ in range(n_rounds)]

    for game in results:
        metrics = game.get("metrics", {})

        cv = metrics.get("cumulative_value_per_round", [])
        for i, v in enumerate(cv[:n_rounds]):
            cumulative_values[i].append(v)

        agent_cv = metrics.get("agent_cumulative_value_per_round", {})
        for agent_id, values in agent_cv.items():
            if "Agent_A" in agent_id:
                target = agent_a_values
            elif "Agent_B" in agent_id:
                target = agent_b_values
            else:
                continue
            for i, v in enumerate(values[:n_rounds]):
                target[i].append(v)

    return {
        "cumulative_value": [statistics.mean(vals) if vals else 0 for vals in cumulative_values],
        "agent_a_cumulative": [statistics.mean(vals) if vals else 0 for vals in agent_a_values],
        "agent_b_cumulative": [statistics.mean(vals) if vals else 0 for vals in agent_b_values],
    }

def compute_accuracy_progression(results):
    """Compute accuracy progression over rounds."""
    n_rounds = 10

    judge_prop_acc = [[] for _ in range(n_rounds)]
    judge_rule_acc = [[] for _ in range(n_rounds)]
    est_prop_acc = [[] for _ in range(n_rounds)]
    est_rule_acc = [[] for _ in range(n_rounds)]

    for game in results:
        prog = game.get("accuracy_progression", [])
        for i, round_data in enumerate(prog[:n_rounds]):
            if round_data:
                judge_prop_acc[i].append(round_data.get("judge_property_accuracy", 0))
                judge_rule_acc[i].append(round_data.get("judge_rule_accuracy", 0))
                est_prop_acc[i].append(round_data.get("estimator_property_accuracy", 0))
                est_rule_acc[i].append(round_data.get("estimator_rule_accuracy", 0))

    return {
        "judge_prop_acc": [statistics.mean(vals) * 100 if vals else 0 for vals in judge_prop_acc],
        "judge_rule_acc": [statistics.mean(vals) * 100 if vals else 0 for vals in judge_rule_acc],
        "est_prop_acc": [statistics.mean(vals) * 100 if vals else 0 for vals in est_prop_acc],
        "est_rule_acc": [statistics.mean(vals) * 100 if vals else 0 for vals in est_rule_acc],
    }

def generate_js_data(trajectory_data, overall_traj, accuracy_prog):
    """Generate JavaScript data for embedding in HTML."""
    return f"""
// Auto-generated trajectory data
const trajectoryData = {json.dumps(trajectory_data, indent=2)};

const overallTrajectory = {json.dumps(overall_traj, indent=2)};

const accuracyProgression = {json.dumps(accuracy_prog, indent=2)};
"""

def main():
    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} games")

    print("Computing trajectory stats...")
    trajectory_data = compute_trajectory_stats(results)

    print("Computing overall trajectories...")
    overall_traj = compute_overall_trajectories(results)

    print("Computing accuracy progression...")
    accuracy_prog = compute_accuracy_progression(results)

    # Save as JSON
    output_path = Path("results/multi_factor/trajectory_data.json")
    with open(output_path, "w") as f:
        json.dump({
            "by_condition": trajectory_data,
            "overall": overall_traj,
            "accuracy_progression": accuracy_prog
        }, f, indent=2)

    print(f"Saved trajectory data to {output_path}")

    # Print summary
    print("\n=== Overall Trajectory (Average across all games) ===")
    print(f"Cumulative value by round: {[round(v, 1) for v in overall_traj['cumulative_value']]}")
    print(f"Agent A cumulative: {[round(v, 1) for v in overall_traj['agent_a_cumulative']]}")
    print(f"Agent B cumulative: {[round(v, 1) for v in overall_traj['agent_b_cumulative']]}")

    print("\n=== Accuracy Progression ===")
    print(f"Judge prop acc by round: {[round(v, 1) for v in accuracy_prog['judge_prop_acc']]}")
    print(f"Estimator prop acc by round: {[round(v, 1) for v in accuracy_prog['est_prop_acc']]}")

if __name__ == "__main__":
    main()
