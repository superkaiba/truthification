#!/usr/bin/env python3
"""
Plot Judge vs Estimator accuracy across all experimental conditions.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def load_json(path):
    with open(path) as f:
        return json.load(f)


def plot_scale_experiment():
    """Plot judge vs estimator by agents and rounds."""
    data = load_json(RESULTS_DIR / "scale_experiment/results_20260212_173323.json")
    summary = data["summary"]

    # Extract data
    conditions = []
    judge_acc = []
    est_acc = []
    judge_se = []

    for key, stats in summary.items():
        agents, rounds = key.split("_")[0], key.split("_")[1]
        agents = agents.replace("agents", "A")
        rounds = rounds.replace("rounds", "R")
        conditions.append(f"{agents}×{rounds}")
        judge_acc.append(stats["property_accuracy"]["mean"] * 100)
        est_acc.append(stats["estimator_accuracy"]["mean"] * 100)
        judge_se.append(stats["property_accuracy"].get("se", 0) * 100)

    # Sort by agents then rounds
    sort_order = ["2A×10R", "2A×15R", "3A×10R", "3A×15R", "3A×20R", "4A×10R", "4A×15R", "4A×20R"]
    indices = [conditions.index(c) if c in conditions else -1 for c in sort_order]
    indices = [i for i in indices if i >= 0]

    conditions = [conditions[i] for i in indices]
    judge_acc = [judge_acc[i] for i in indices]
    est_acc = [est_acc[i] for i in indices]
    judge_se = [judge_se[i] for i in indices]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, judge_acc, width, label='Judge', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, est_acc, width, label='Estimator', color='#3498db', alpha=0.8)

    # Add error bars for judge
    ax.errorbar(x - width/2, judge_acc, yerr=judge_se, fmt='none', color='black', capsize=3)

    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.7, label='Random Baseline (30.7%)')
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_xlabel('Agents × Rounds')
    ax.set_title('Judge vs Estimator Accuracy: Scale Experiment')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 70)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "judge_vs_estimator_scale.png", dpi=150)
    plt.close()
    print("Saved: judge_vs_estimator_scale.png")


def plot_by_agents():
    """Line plot showing accuracy by number of agents."""
    data = load_json(RESULTS_DIR / "scale_experiment/results_20260212_173323.json")
    summary = data["summary"]

    # Group by agents, average across rounds
    agents_data = {2: {"judge": [], "est": []}, 3: {"judge": [], "est": []}, 4: {"judge": [], "est": []}}

    for key, stats in summary.items():
        n_agents = int(key.split("agents")[0])
        agents_data[n_agents]["judge"].append(stats["property_accuracy"]["mean"] * 100)
        agents_data[n_agents]["est"].append(stats["estimator_accuracy"]["mean"] * 100)

    agents = [2, 3, 4]
    judge_means = [np.mean(agents_data[a]["judge"]) for a in agents]
    est_means = [np.mean(agents_data[a]["est"]) for a in agents]
    judge_std = [np.std(agents_data[a]["judge"]) for a in agents]
    est_std = [np.std(agents_data[a]["est"]) for a in agents]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(agents, judge_means, yerr=judge_std, marker='o', markersize=10,
                label='Judge', color='#2ecc71', linewidth=2, capsize=5)
    ax.errorbar(agents, est_means, yerr=est_std, marker='s', markersize=10,
                label='Estimator', color='#3498db', linewidth=2, capsize=5)

    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    ax.set_xlabel('Number of Agents')
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_title('Accuracy by Number of Agents (averaged across rounds)')
    ax.set_xticks(agents)
    ax.legend()
    ax.set_ylim(0, 70)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_by_agents.png", dpi=150)
    plt.close()
    print("Saved: accuracy_by_agents.png")


def plot_by_rounds():
    """Line plot showing accuracy by number of rounds."""
    data = load_json(RESULTS_DIR / "scale_experiment/results_20260212_173323.json")
    summary = data["summary"]

    # Group by rounds, average across agents
    rounds_data = {10: {"judge": [], "est": []}, 15: {"judge": [], "est": []}, 20: {"judge": [], "est": []}}

    for key, stats in summary.items():
        n_rounds = int(key.split("_")[1].replace("rounds", ""))
        rounds_data[n_rounds]["judge"].append(stats["property_accuracy"]["mean"] * 100)
        rounds_data[n_rounds]["est"].append(stats["estimator_accuracy"]["mean"] * 100)

    rounds = [10, 15, 20]
    judge_means = [np.mean(rounds_data[r]["judge"]) for r in rounds]
    est_means = [np.mean(rounds_data[r]["est"]) for r in rounds]
    judge_std = [np.std(rounds_data[r]["judge"]) for r in rounds]
    est_std = [np.std(rounds_data[r]["est"]) for r in rounds]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(rounds, judge_means, yerr=judge_std, marker='o', markersize=10,
                label='Judge', color='#2ecc71', linewidth=2, capsize=5)
    ax.errorbar(rounds, est_means, yerr=est_std, marker='s', markersize=10,
                label='Estimator', color='#3498db', linewidth=2, capsize=5)

    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_title('Accuracy by Number of Rounds (averaged across agent counts)')
    ax.set_xticks(rounds)
    ax.legend()
    ax.set_ylim(0, 70)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_by_rounds.png", dpi=150)
    plt.close()
    print("Saved: accuracy_by_rounds.png")


def plot_information_conditions():
    """Plot judge accuracy by information condition."""
    data = load_json(RESULTS_DIR / "v2_comprehensive_opus/all_results.json")

    conditions = []
    judge_acc = []
    judge_std = []

    for entry in data["information_conditions"]:
        conditions.append(entry["condition"].capitalize())
        judge_acc.append(entry["avg_property_accuracy"] * 100)
        # Calculate std from raw results
        raw_accs = [r["property_accuracy"] * 100 for r in entry["raw_results"]]
        judge_std.append(np.std(raw_accs))

    # Sort: blind, ids, interests
    order = ["Blind", "Ids", "Interests"]
    indices = [conditions.index(c) for c in order]
    conditions = [conditions[i] for i in indices]
    judge_acc = [judge_acc[i] for i in indices]
    judge_std = [judge_std[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(conditions))
    bars = ax.bar(x, judge_acc, yerr=judge_std, color='#2ecc71', alpha=0.8, capsize=5)

    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_xlabel('Information Condition')
    ax.set_title('Judge Accuracy by Information Condition')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 70)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_by_information_condition.png", dpi=150)
    plt.close()
    print("Saved: accuracy_by_information_condition.png")


def plot_rule_complexity():
    """Plot judge accuracy by rule complexity."""
    data = load_json(RESULTS_DIR / "v2_comprehensive_opus/all_results.json")

    conditions = []
    judge_acc = []
    judge_std = []

    for entry in data["rule_complexity"]:
        conditions.append(entry["rule_complexity"].capitalize())
        judge_acc.append(entry["avg_property_accuracy"] * 100)
        raw_accs = [r["property_accuracy"] * 100 for r in entry["raw_results"]]
        judge_std.append(np.std(raw_accs))

    # Sort: simple, medium, complex
    order = ["Simple", "Medium", "Complex"]
    indices = [conditions.index(c) for c in order]
    conditions = [conditions[i] for i in indices]
    judge_acc = [judge_acc[i] for i in indices]
    judge_std = [judge_std[i] for i in indices]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(conditions))
    bars = ax.bar(x, judge_acc, yerr=judge_std, color='#9b59b6', alpha=0.8, capsize=5)

    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_xlabel('Rule Complexity')
    ax.set_title('Judge Accuracy by Rule Complexity')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 70)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "accuracy_by_rule_complexity.png", dpi=150)
    plt.close()
    print("Saved: accuracy_by_rule_complexity.png")


def plot_oracle_effect():
    """Plot forced oracle experiment results."""
    data = load_json(RESULTS_DIR / "forced_oracle_test/results_20260213_164112.json")

    no_oracle = [r["property_accuracy"] * 100 for r in data["no_oracle"]]
    forced_oracle = [r["property_accuracy"] * 100 for r in data["forced_oracle"]]

    conditions = ["No Oracle\n(budget=0)", "Forced Oracle\n(budget=8)"]
    means = [np.mean(no_oracle), np.mean(forced_oracle)]
    stds = [np.std(no_oracle), np.std(forced_oracle)]

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(conditions))
    colors = ['#e74c3c', '#2ecc71']
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5)

    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_title('Oracle Effect on Judge Accuracy (Forced Oracle Experiment)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 90)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=12)

    # Add improvement annotation
    improvement = means[1] - means[0]
    ax.annotate(f'+{improvement:.1f}pp', xy=(0.5, (means[0] + means[1])/2),
                fontsize=14, fontweight='bold', ha='center', color='#27ae60')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "oracle_effect_forced.png", dpi=150)
    plt.close()
    print("Saved: oracle_effect_forced.png")


def plot_multi_factor_observer_conditions():
    """Plot judge vs estimator by observer condition from multi-factor experiment."""
    # Load multi-factor data
    stats_file = RESULTS_DIR / "multi_factor/condition_stats_20260207_104922.json"
    with open(stats_file) as f:
        data = json.load(f)

    # Aggregate by observer_condition
    obs_cond_stats = {"blind": {"judge": [], "est": []},
                      "ids": {"judge": [], "est": []},
                      "interests": {"judge": [], "est": []}}

    for key, value in data.items():
        cond = value["condition"]["observer_condition"]
        judge_acc = value["stats"]["property_accuracy"]["mean"] * 100
        est_acc = value["stats"]["estimator_property_accuracy"]["mean"] * 100
        obs_cond_stats[cond]["judge"].append(judge_acc)
        obs_cond_stats[cond]["est"].append(est_acc)

    conditions = ["Blind", "IDs", "Interests"]
    judge_means = [np.mean(obs_cond_stats[c.lower()]["judge"]) for c in conditions]
    est_means = [np.mean(obs_cond_stats[c.lower()]["est"]) for c in conditions]
    judge_std = [np.std(obs_cond_stats[c.lower()]["judge"]) for c in conditions]
    est_std = [np.std(obs_cond_stats[c.lower()]["est"]) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, judge_means, width, yerr=judge_std, label='Judge',
                   color='#2ecc71', alpha=0.8, capsize=5)
    bars2 = ax.bar(x + width/2, est_means, width, yerr=est_std, label='Estimator',
                   color='#3498db', alpha=0.8, capsize=5)

    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_xlabel('Observer Condition')
    ax.set_title('Judge vs Estimator by Observer Condition (Multi-Factor Experiment)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 70)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "judge_vs_estimator_observer_condition.png", dpi=150)
    plt.close()
    print("Saved: judge_vs_estimator_observer_condition.png")


def plot_combined_summary():
    """Create a combined summary figure with all key comparisons."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Oracle effect
    ax = axes[0, 0]
    data = load_json(RESULTS_DIR / "forced_oracle_test/results_20260213_164112.json")
    no_oracle = [r["property_accuracy"] * 100 for r in data["no_oracle"]]
    forced_oracle = [r["property_accuracy"] * 100 for r in data["forced_oracle"]]
    means = [np.mean(no_oracle), np.mean(forced_oracle)]
    stds = [np.std(no_oracle), np.std(forced_oracle)]
    bars = ax.bar(["No Oracle", "Forced Oracle"], means, yerr=stds,
                  color=['#e74c3c', '#2ecc71'], alpha=0.8, capsize=5)
    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.5)
    ax.set_title('A. Oracle Effect')
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_ylim(0, 90)
    for bar in bars:
        ax.annotate(f'{bar.get_height():.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # 2. By agents
    ax = axes[0, 1]
    scale_data = load_json(RESULTS_DIR / "scale_experiment/results_20260212_173323.json")["summary"]
    agents_data = {2: {"judge": [], "est": []}, 3: {"judge": [], "est": []}, 4: {"judge": [], "est": []}}
    for key, stats in scale_data.items():
        n_agents = int(key.split("agents")[0])
        agents_data[n_agents]["judge"].append(stats["property_accuracy"]["mean"] * 100)
        agents_data[n_agents]["est"].append(stats["estimator_accuracy"]["mean"] * 100)
    agents = [2, 3, 4]
    ax.plot(agents, [np.mean(agents_data[a]["judge"]) for a in agents], 'o-',
            color='#2ecc71', label='Judge', markersize=8, linewidth=2)
    ax.plot(agents, [np.mean(agents_data[a]["est"]) for a in agents], 's-',
            color='#3498db', label='Estimator', markersize=8, linewidth=2)
    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.5)
    ax.set_title('B. By Number of Agents')
    ax.set_xlabel('Agents')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(agents)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 70)

    # 3. By rounds
    ax = axes[0, 2]
    rounds_data = {10: {"judge": [], "est": []}, 15: {"judge": [], "est": []}, 20: {"judge": [], "est": []}}
    for key, stats in scale_data.items():
        n_rounds = int(key.split("_")[1].replace("rounds", ""))
        rounds_data[n_rounds]["judge"].append(stats["property_accuracy"]["mean"] * 100)
        rounds_data[n_rounds]["est"].append(stats["estimator_accuracy"]["mean"] * 100)
    rounds = [10, 15, 20]
    ax.plot(rounds, [np.mean(rounds_data[r]["judge"]) for r in rounds], 'o-',
            color='#2ecc71', label='Judge', markersize=8, linewidth=2)
    ax.plot(rounds, [np.mean(rounds_data[r]["est"]) for r in rounds], 's-',
            color='#3498db', label='Estimator', markersize=8, linewidth=2)
    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.5)
    ax.set_title('C. By Number of Rounds')
    ax.set_xlabel('Rounds')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(rounds)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 70)

    # 4. Information conditions
    ax = axes[1, 0]
    v2_data = load_json(RESULTS_DIR / "v2_comprehensive_opus/all_results.json")
    info_conds = {"Blind": 0, "IDs": 0, "Interests": 0}
    for entry in v2_data["information_conditions"]:
        cond = entry["condition"].capitalize()
        if cond == "Ids":
            cond = "IDs"
        info_conds[cond] = entry["avg_property_accuracy"] * 100
    bars = ax.bar(info_conds.keys(), info_conds.values(), color='#9b59b6', alpha=0.8)
    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.5)
    ax.set_title('D. Information Conditions')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_ylim(0, 70)
    for bar in bars:
        ax.annotate(f'{bar.get_height():.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # 5. Rule complexity
    ax = axes[1, 1]
    rule_conds = {"Simple": 0, "Medium": 0, "Complex": 0}
    for entry in v2_data["rule_complexity"]:
        cond = entry["rule_complexity"].capitalize()
        rule_conds[cond] = entry["avg_property_accuracy"] * 100
    bars = ax.bar(rule_conds.keys(), rule_conds.values(), color='#e67e22', alpha=0.8)
    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.5)
    ax.set_title('E. Rule Complexity')
    ax.set_ylabel('Judge Accuracy (%)')
    ax.set_ylim(0, 70)
    for bar in bars:
        ax.annotate(f'{bar.get_height():.0f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # 6. Scale experiment heatmap-style
    ax = axes[1, 2]
    # Create grouped bar for scale experiment
    conditions = ["2A×10R", "2A×15R", "3A×10R", "3A×15R", "3A×20R", "4A×10R", "4A×15R", "4A×20R"]
    judge_vals = []
    est_vals = []
    for c in conditions:
        key = c.replace("A×", "agents_").replace("R", "rounds")
        if key in scale_data:
            judge_vals.append(scale_data[key]["property_accuracy"]["mean"] * 100)
            est_vals.append(scale_data[key]["estimator_accuracy"]["mean"] * 100)

    x = np.arange(len(conditions))
    width = 0.35
    ax.bar(x - width/2, judge_vals, width, label='Judge', color='#2ecc71', alpha=0.8)
    ax.bar(x + width/2, est_vals, width, label='Estimator', color='#3498db', alpha=0.8)
    ax.axhline(y=30.7, color='red', linestyle='--', alpha=0.5)
    ax.set_title('F. Scale Experiment (All Conditions)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 70)

    plt.suptitle('Judge vs Estimator Accuracy Across All Conditions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "judge_vs_estimator_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: judge_vs_estimator_summary.png")


def main():
    print("Generating plots...")
    print(f"Output directory: {PLOTS_DIR}")
    print()

    plot_oracle_effect()
    plot_scale_experiment()
    plot_by_agents()
    plot_by_rounds()
    plot_information_conditions()
    plot_rule_complexity()
    plot_multi_factor_observer_conditions()
    plot_combined_summary()

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
