#!/usr/bin/env python3
"""Generate comprehensive plots for all truthification experiments."""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Directories
RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Baselines
RANDOM_PROPERTY_BASELINE = 0.307  # ~30.7%
RANDOM_RULE_BASELINE = 0.207  # ~20.7%


def load_json(path):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


# ============================================================================
# 1. Oracle Queries Effect
# ============================================================================

def plot_oracle_effect():
    """Plot effect of oracle queries (0 vs 8)."""
    data_path = RESULTS_DIR / "no_oracle_comparison" / "results_20260211_030456.json"
    data = load_json(data_path)

    summary = data["summary"]

    # Extract data
    conditions = ["0 queries", "8 queries"]
    prop_acc = [summary["0"]["property_accuracy"]["mean"],
                summary["8"]["property_accuracy"]["mean"]]
    prop_err = [summary["0"]["property_accuracy"]["se"],
                summary["8"]["property_accuracy"]["se"]]
    est_acc = [summary["0"]["estimator_accuracy"]["mean"],
               summary["8"]["estimator_accuracy"]["mean"]]
    est_std = [summary["0"]["estimator_accuracy"]["std"] / np.sqrt(5),
               summary["8"]["estimator_accuracy"]["std"] / np.sqrt(5)]
    rule_acc = [summary["0"]["rule_inference"]["mean"],
                summary["8"]["rule_inference"]["mean"]]

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Property Accuracy
    x = np.arange(len(conditions))
    width = 0.35

    ax = axes[0]
    bars1 = ax.bar(x - width/2, prop_acc, width, label='Judge', yerr=prop_err, capsize=5, color='#2ecc71')
    bars2 = ax.bar(x + width/2, est_acc, width, label='Estimator', yerr=est_std, capsize=5, color='#3498db')
    ax.axhline(y=RANDOM_PROPERTY_BASELINE, color='red', linestyle='--', label=f'Random ({RANDOM_PROPERTY_BASELINE:.1%})')
    ax.set_ylabel('Accuracy')
    ax.set_title('Property Accuracy: Oracle Effect')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 0.6)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    # Rule Inference
    ax = axes[1]
    bars = ax.bar(conditions, rule_acc, color='#9b59b6')
    ax.axhline(y=RANDOM_RULE_BASELINE, color='red', linestyle='--', label=f'Random ({RANDOM_RULE_BASELINE:.1%})')
    ax.set_ylabel('Accuracy')
    ax.set_title('Rule Inference Accuracy')
    ax.legend()
    ax.set_ylim(0, 1.0)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)

    # Improvement
    ax = axes[2]
    improvements = [
        (prop_acc[1] - prop_acc[0]) * 100,
        (est_acc[1] - est_acc[0]) * 100,
        (rule_acc[1] - rule_acc[0]) * 100
    ]
    labels = ['Judge\nProperty', 'Estimator\nProperty', 'Rule\nInference']
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
    bars = ax.bar(labels, improvements, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Improvement (percentage points)')
    ax.set_title('Impact of Adding 8 Oracle Queries')

    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'+{height:.1f}pp' if height > 0 else f'{height:.1f}pp',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3 if height > 0 else -12), textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top', fontsize=10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "oracle_queries_effect.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: oracle_queries_effect.png")


# ============================================================================
# 2. Information Conditions
# ============================================================================

def plot_information_conditions():
    """Plot effect of information conditions (blind, ids, interests)."""
    data_path = RESULTS_DIR / "v2_comprehensive_opus" / "information_conditions.json"
    data = load_json(data_path)

    conditions = ['blind', 'ids', 'interests']
    prop_acc = [d['avg_property_accuracy'] for d in data]
    selection_acc = [d['avg_selection_accuracy'] for d in data]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(conditions))
    width = 0.35

    bars1 = ax.bar(x - width/2, prop_acc, width, label='Property Accuracy', color='#2ecc71')
    bars2 = ax.bar(x + width/2, selection_acc, width, label='Selection Accuracy', color='#3498db')
    ax.axhline(y=RANDOM_PROPERTY_BASELINE, color='red', linestyle='--', alpha=0.7, label=f'Random Baseline ({RANDOM_PROPERTY_BASELINE:.1%})')

    ax.set_ylabel('Accuracy')
    ax.set_title('Effect of Information Conditions on Judge Performance\n(What does the judge know about the agents?)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Blind\n(no agent info)', 'IDs Only\n(can track agents)', 'Full Interests\n(knows agent goals)'])
    ax.legend()
    ax.set_ylim(0, 0.7)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "information_conditions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: information_conditions.png")


# ============================================================================
# 3. Rule Complexity
# ============================================================================

def plot_rule_complexity():
    """Plot effect of rule complexity."""
    data_path = RESULTS_DIR / "v2_comprehensive_opus" / "rule_complexity.json"
    data = load_json(data_path)

    complexities = ['simple', 'medium', 'complex']
    prop_acc = [d['avg_property_accuracy'] for d in data]
    selection_acc = [d['avg_selection_accuracy'] for d in data]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(complexities))
    width = 0.35

    bars1 = ax.bar(x - width/2, prop_acc, width, label='Property Accuracy', color='#2ecc71')
    bars2 = ax.bar(x + width/2, selection_acc, width, label='Selection Accuracy', color='#3498db')
    ax.axhline(y=RANDOM_PROPERTY_BASELINE, color='red', linestyle='--', alpha=0.7, label=f'Random Baseline ({RANDOM_PROPERTY_BASELINE:.1%})')

    ax.set_ylabel('Accuracy')
    ax.set_title('Effect of Rule Complexity on Judge Performance\n(How complex is the hidden value rule?)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Simple\n(1 property)', 'Medium\n(2-3 properties)', 'Complex\n(4+ properties)'])
    ax.legend()
    ax.set_ylim(0, 0.8)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rule_complexity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: rule_complexity.png")


# ============================================================================
# 4. Scale Experiment: Agents × Rounds
# ============================================================================

def plot_scale_experiment():
    """Plot scale experiment (agents × rounds)."""
    data_path = RESULTS_DIR / "scale_experiment" / "results_20260212_173323.json"
    data = load_json(data_path)
    summary = data["summary"]

    # Judge accuracy heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prepare data for heatmap
    agents = [2, 3, 4]
    rounds = [10, 15, 20]

    judge_data = np.zeros((len(agents), len(rounds)))
    estimator_data = np.zeros((len(agents), len(rounds)))

    for i, a in enumerate(agents):
        for j, r in enumerate(rounds):
            key = f"{a}agents_{r}rounds"
            if key in summary:
                judge_data[i, j] = summary[key]["property_accuracy"]["mean"]
                estimator_data[i, j] = summary[key]["estimator_accuracy"]["mean"]
            else:
                judge_data[i, j] = np.nan
                estimator_data[i, j] = np.nan

    # Judge heatmap
    ax = axes[0]
    mask = np.isnan(judge_data)
    im = sns.heatmap(judge_data, annot=True, fmt='.1%', cmap='RdYlGn',
                     xticklabels=rounds, yticklabels=agents, ax=ax,
                     vmin=0.15, vmax=0.5, mask=mask, cbar_kws={'label': 'Accuracy'})
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Number of Agents')
    ax.set_title('Judge Property Accuracy\n(Agents × Rounds)')

    # Estimator heatmap
    ax = axes[1]
    mask = np.isnan(estimator_data)
    im = sns.heatmap(estimator_data, annot=True, fmt='.1%', cmap='RdYlGn',
                     xticklabels=rounds, yticklabels=agents, ax=ax,
                     vmin=0.4, vmax=0.6, mask=mask, cbar_kws={'label': 'Accuracy'})
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Number of Agents')
    ax.set_title('Estimator Property Accuracy\n(Agents × Rounds)')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scale_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: scale_heatmap.png")

    # Line plot by agents
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for i, a in enumerate(agents):
        accs = []
        valid_rounds = []
        for j, r in enumerate(rounds):
            key = f"{a}agents_{r}rounds"
            if key in summary and summary[key]["n"] > 0:
                accs.append(summary[key]["property_accuracy"]["mean"])
                valid_rounds.append(r)
        if accs:
            ax.plot(valid_rounds, accs, marker='o', label=f'{a} agents', linewidth=2, markersize=8)

    ax.axhline(y=RANDOM_PROPERTY_BASELINE, color='red', linestyle='--', alpha=0.7, label='Random Baseline')
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Property Accuracy')
    ax.set_title('Judge Accuracy by Agent Count')
    ax.legend()
    ax.set_ylim(0, 0.55)
    ax.set_xticks(rounds)

    ax = axes[1]
    for i, a in enumerate(agents):
        accs = []
        valid_rounds = []
        for j, r in enumerate(rounds):
            key = f"{a}agents_{r}rounds"
            if key in summary and summary[key]["n"] > 0:
                accs.append(summary[key]["estimator_accuracy"]["mean"])
                valid_rounds.append(r)
        if accs:
            ax.plot(valid_rounds, accs, marker='s', label=f'{a} agents', linewidth=2, markersize=8)

    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Property Accuracy')
    ax.set_title('Estimator Accuracy by Agent Count')
    ax.legend()
    ax.set_ylim(0.35, 0.65)
    ax.set_xticks(rounds)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scale_line_plots.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: scale_line_plots.png")


# ============================================================================
# 5. Performance Over Rounds
# ============================================================================

def plot_performance_over_rounds():
    """Plot cumulative value and accuracy progression over rounds."""
    data_path = RESULTS_DIR / "no_oracle_comparison" / "results_20260211_030456.json"
    data = load_json(data_path)

    # Average cumulative value over rounds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Calculate average cumulative value per round for each condition
    for budget in [0, 8]:
        results = [r for r in data["all_results"] if r["oracle_budget"] == budget]
        n_rounds = len(results[0]["metrics"]["cumulative_value_per_round"])

        avg_cumulative = np.zeros(n_rounds)
        for r in results:
            avg_cumulative += np.array(r["metrics"]["cumulative_value_per_round"])
        avg_cumulative /= len(results)

        rounds = list(range(1, n_rounds + 1))
        label = f'{budget} oracle queries'
        color = '#e74c3c' if budget == 0 else '#2ecc71'
        axes[0].plot(rounds, avg_cumulative, marker='o', label=label, linewidth=2, color=color)

    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Cumulative Value')
    axes[0].set_title('Value Accumulation Over Rounds')
    axes[0].legend()
    axes[0].set_xlim(1, 10)

    # Agent cumulative values
    for budget in [0, 8]:
        results = [r for r in data["all_results"] if r["oracle_budget"] == budget]
        n_rounds = len(results[0]["metrics"]["cumulative_value_per_round"])

        agent_a_cum = np.zeros(n_rounds)
        agent_b_cum = np.zeros(n_rounds)
        for r in results:
            agent_a_cum += np.array(r["metrics"]["agent_cumulative_value_per_round"]["Agent_A"])
            agent_b_cum += np.array(r["metrics"]["agent_cumulative_value_per_round"]["Agent_B"])
        agent_a_cum /= len(results)
        agent_b_cum /= len(results)

        rounds = list(range(1, n_rounds + 1))
        linestyle = '-' if budget == 8 else '--'
        alpha = 1.0 if budget == 8 else 0.6
        axes[1].plot(rounds, agent_a_cum, marker='o', label=f'Agent A ({budget} queries)',
                     linewidth=2, color='#3498db', linestyle=linestyle, alpha=alpha)
        axes[1].plot(rounds, agent_b_cum, marker='s', label=f'Agent B ({budget} queries)',
                     linewidth=2, color='#e67e22', linestyle=linestyle, alpha=alpha)

    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Cumulative Agent Value')
    axes[1].set_title('Agent Value Accumulation Over Rounds\n(Value gained from judge selections benefiting each agent)')
    axes[1].legend(loc='upper left')
    axes[1].set_xlim(1, 10)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "performance_over_rounds.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: performance_over_rounds.png")


# ============================================================================
# 6. Judge vs Estimator Comparison
# ============================================================================

def plot_judge_vs_estimator():
    """Compare judge and estimator performance across experiments."""
    # Collect data from multiple sources
    comparisons = []

    # No oracle comparison
    data_path = RESULTS_DIR / "no_oracle_comparison" / "results_20260211_030456.json"
    data = load_json(data_path)
    for budget in [0, 8]:
        comparisons.append({
            'experiment': f'Oracle={budget}',
            'judge': data["summary"][str(budget)]["property_accuracy"]["mean"],
            'estimator': data["summary"][str(budget)]["estimator_accuracy"]["mean"]
        })

    # Scale experiment
    scale_path = RESULTS_DIR / "scale_experiment" / "results_20260212_173323.json"
    scale_data = load_json(scale_path)
    summary = scale_data["summary"]
    for key in ["2agents_10rounds", "4agents_20rounds"]:
        if key in summary:
            comparisons.append({
                'experiment': key.replace("_", " × ").replace("agents", "A").replace("rounds", "R"),
                'judge': summary[key]["property_accuracy"]["mean"],
                'estimator': summary[key]["estimator_accuracy"]["mean"]
            })

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(comparisons))
    width = 0.35

    judge_vals = [c['judge'] for c in comparisons]
    est_vals = [c['estimator'] for c in comparisons]

    bars1 = ax.bar(x - width/2, judge_vals, width, label='Judge (Active)', color='#2ecc71')
    bars2 = ax.bar(x + width/2, est_vals, width, label='Estimator (Passive)', color='#3498db')
    ax.axhline(y=RANDOM_PROPERTY_BASELINE, color='red', linestyle='--', alpha=0.7, label=f'Random Baseline ({RANDOM_PROPERTY_BASELINE:.1%})')

    ax.set_ylabel('Property Accuracy')
    ax.set_title('Judge vs Estimator: Active Participation vs Passive Observation')
    ax.set_xticks(x)
    ax.set_xticklabels([c['experiment'] for c in comparisons], rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 0.7)

    # Add advantage annotations
    for i, (j, e) in enumerate(zip(judge_vals, est_vals)):
        diff = e - j
        if diff > 0:
            ax.annotate(f'Est +{diff:.0%}', xy=(i, max(j, e) + 0.02), ha='center', fontsize=8, color='#3498db')
        else:
            ax.annotate(f'Judge +{-diff:.0%}', xy=(i, max(j, e) + 0.02), ha='center', fontsize=8, color='#2ecc71')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "judge_vs_estimator.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: judge_vs_estimator.png")


# ============================================================================
# 7. Summary Dashboard
# ============================================================================

def plot_summary_dashboard():
    """Create a summary dashboard with key metrics."""
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # 1. Overall performance summary (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Property\nAccuracy', 'Rule\nInference', 'Selection\nAccuracy']

    # Best observed values
    best_values = [0.56, 0.85, 0.92]  # From experiments
    baseline_values = [0.307, 0.207, 0.5]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax1.bar(x - width/2, best_values, width, label='Best Observed', color='#2ecc71')
    bars2 = ax1.bar(x + width/2, baseline_values, width, label='Random Baseline', color='#e74c3c', alpha=0.6)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Best Performance vs Baseline')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1.0)

    # 2. Oracle effect (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    data_path = RESULTS_DIR / "no_oracle_comparison" / "results_20260211_030456.json"
    data = load_json(data_path)
    conditions = ['0 queries', '8 queries']
    prop_acc = [data["summary"]["0"]["property_accuracy"]["mean"],
                data["summary"]["8"]["property_accuracy"]["mean"]]
    colors = ['#e74c3c', '#2ecc71']
    bars = ax2.bar(conditions, prop_acc, color=colors)
    ax2.axhline(y=RANDOM_PROPERTY_BASELINE, color='red', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Property Accuracy')
    ax2.set_title('Oracle Queries Effect')
    ax2.set_ylim(0, 0.55)
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # 3. Info conditions (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    info_path = RESULTS_DIR / "v2_comprehensive_opus" / "information_conditions.json"
    info_data = load_json(info_path)
    conditions = ['Blind', 'IDs', 'Interests']
    accs = [d['avg_property_accuracy'] for d in info_data]
    colors = ['#95a5a6', '#3498db', '#9b59b6']
    bars = ax3.bar(conditions, accs, color=colors)
    ax3.axhline(y=RANDOM_PROPERTY_BASELINE, color='red', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Property Accuracy')
    ax3.set_title('Information Conditions')
    ax3.set_ylim(0, 0.6)
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # 4. Rule complexity (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    complexity_path = RESULTS_DIR / "v2_comprehensive_opus" / "rule_complexity.json"
    complexity_data = load_json(complexity_path)
    complexities = ['Simple', 'Medium', 'Complex']
    accs = [d['avg_property_accuracy'] for d in complexity_data]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    bars = ax4.bar(complexities, accs, color=colors)
    ax4.axhline(y=RANDOM_PROPERTY_BASELINE, color='red', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Property Accuracy')
    ax4.set_title('Rule Complexity Effect')
    ax4.set_ylim(0, 0.6)
    for bar in bars:
        height = bar.get_height()
        ax4.annotate(f'{height:.1%}', xy=(bar.get_x() + bar.get_width()/2, height),
                     xytext=(0, 3), textcoords="offset points", ha='center', fontsize=10)

    # 5. Scale experiment heatmap (middle center + right)
    ax5 = fig.add_subplot(gs[1, 1:])
    scale_path = RESULTS_DIR / "scale_experiment" / "results_20260212_173323.json"
    scale_data = load_json(scale_path)
    summary = scale_data["summary"]

    agents = [2, 3, 4]
    rounds = [10, 15, 20]
    judge_data = np.zeros((len(agents), len(rounds)))

    for i, a in enumerate(agents):
        for j, r in enumerate(rounds):
            key = f"{a}agents_{r}rounds"
            if key in summary:
                judge_data[i, j] = summary[key]["property_accuracy"]["mean"]
            else:
                judge_data[i, j] = np.nan

    mask = np.isnan(judge_data)
    sns.heatmap(judge_data, annot=True, fmt='.1%', cmap='RdYlGn',
                xticklabels=rounds, yticklabels=agents, ax=ax5,
                vmin=0.15, vmax=0.5, mask=mask, cbar_kws={'label': 'Accuracy'})
    ax5.set_xlabel('Number of Rounds')
    ax5.set_ylabel('Number of Agents')
    ax5.set_title('Scale Experiment: Judge Accuracy (Agents × Rounds)')

    # 6. Judge vs Estimator (bottom spanning full width)
    ax6 = fig.add_subplot(gs[2, :])
    comparisons = [
        ('No Oracle\n(0 queries)', 0.30, 0.14),
        ('With Oracle\n(8 queries)', 0.444, 0.252),
        ('2A × 10R', 0.46, 0.513),
        ('2A × 15R', 0.367, 0.553),
        ('4A × 10R', 0.187, 0.427),
        ('4A × 20R', 0.44, 0.587),
    ]

    x = np.arange(len(comparisons))
    width = 0.35

    judge_vals = [c[1] for c in comparisons]
    est_vals = [c[2] for c in comparisons]

    bars1 = ax6.bar(x - width/2, judge_vals, width, label='Judge (Active)', color='#2ecc71')
    bars2 = ax6.bar(x + width/2, est_vals, width, label='Estimator (Passive)', color='#3498db')
    ax6.axhline(y=RANDOM_PROPERTY_BASELINE, color='red', linestyle='--', alpha=0.7, label=f'Random ({RANDOM_PROPERTY_BASELINE:.1%})')

    ax6.set_ylabel('Property Accuracy')
    ax6.set_title('Judge vs Estimator Comparison Across All Experiments')
    ax6.set_xticks(x)
    ax6.set_xticklabels([c[0] for c in comparisons])
    ax6.legend(loc='upper left')
    ax6.set_ylim(0, 0.7)

    plt.suptitle('Truthification Experiment Results Summary', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(PLOTS_DIR / "summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Created: summary_dashboard.png")


# ============================================================================
# Main
# ============================================================================

def main():
    print("Generating comprehensive plots...")
    print(f"Output directory: {PLOTS_DIR}")
    print()

    print("1. Oracle queries effect...")
    plot_oracle_effect()

    print("2. Information conditions...")
    plot_information_conditions()

    print("3. Rule complexity...")
    plot_rule_complexity()

    print("4. Scale experiment...")
    plot_scale_experiment()

    print("5. Performance over rounds...")
    plot_performance_over_rounds()

    print("6. Judge vs estimator...")
    plot_judge_vs_estimator()

    print("7. Summary dashboard...")
    plot_summary_dashboard()

    print()
    print(f"All plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
