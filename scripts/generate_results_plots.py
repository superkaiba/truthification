"""
Generate all plots for the experimental results summary report.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

def load_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)

def fig1_forced_oracle():
    """Figure 1: Forced Oracle Effect on Property Accuracy."""
    data = load_json(RESULTS_DIR / "forced_oracle_test/results_20260213_164112.json")

    no_oracle = [r["property_accuracy"] for r in data["no_oracle"]]
    forced_oracle = [r["property_accuracy"] for r in data["forced_oracle"]]

    no_oracle_mean = np.mean(no_oracle) * 100
    forced_oracle_mean = np.mean(forced_oracle) * 100
    no_oracle_se = np.std(no_oracle) / np.sqrt(len(no_oracle)) * 100
    forced_oracle_se = np.std(forced_oracle) / np.sqrt(len(forced_oracle)) * 100

    fig, ax = plt.subplots(figsize=(6, 5))

    x = [0, 1]
    means = [no_oracle_mean, forced_oracle_mean]
    errors = [no_oracle_se, forced_oracle_se]
    colors = ['#ff7f7f', '#7fbf7f']

    bars = ax.bar(x, means, yerr=errors, capsize=5, color=colors, edgecolor='black', width=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(['No Oracle\n(Budget=0)', 'Forced Oracle\n(Budget=8)'])
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_title('Effect of Oracle Access on Property Accuracy')
    ax.set_ylim(0, 100)

    # Add value labels
    for bar, mean, err in zip(bars, means, errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 2,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Add improvement arrow
    ax.annotate('', xy=(1, forced_oracle_mean - 5), xytext=(0, no_oracle_mean + 5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(0.5, (no_oracle_mean + forced_oracle_mean) / 2,
            f'+{forced_oracle_mean - no_oracle_mean:.1f}%',
            ha='center', va='center', fontsize=10, color='gray')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig1_forced_oracle.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig1_forced_oracle.png")

def fig2_oracle_budget():
    """Figure 2: Oracle Budget Effect on Objective Inference."""
    data = load_json(OUTPUTS_DIR / "oracle_budget_objective/20260218_002133/condition_stats.json")

    budgets = []
    means = []
    stds = []
    ns = []

    for key in sorted(data.keys(), key=lambda x: int(x.split('_')[-1])):
        budget = data[key]["condition"]["oracle_budget"]
        mean = data[key]["stats"]["objective_inference_score"]["mean"] * 100
        std = data[key]["stats"]["objective_inference_score"]["std"] * 100
        n = data[key]["stats"]["objective_inference_score"]["n"]

        budgets.append(budget)
        means.append(mean)
        stds.append(std)
        ns.append(n)

    ses = [s / np.sqrt(n) for s, n in zip(stds, ns)]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(budgets, means, yerr=ses, marker='o', capsize=5, linewidth=2, markersize=8)
    ax.fill_between(budgets, [m - s for m, s in zip(means, ses)],
                    [m + s for m, s in zip(means, ses)], alpha=0.2)

    # Mark peak
    peak_idx = np.argmax(means)
    ax.scatter([budgets[peak_idx]], [means[peak_idx]], color='red', s=150, zorder=5, marker='*')
    ax.annotate(f'Peak: {means[peak_idx]:.1f}%',
                xy=(budgets[peak_idx], means[peak_idx]),
                xytext=(budgets[peak_idx] + 0.5, means[peak_idx] + 3),
                fontsize=10, color='red')

    ax.set_xlabel('Oracle Budget')
    ax.set_ylabel('Objective Inference F1 (%)')
    ax.set_title('Effect of Oracle Budget on Objective Inference')
    ax.set_xticks(budgets)
    ax.set_ylim(0, 40)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig2_oracle_budget.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig2_oracle_budget.png")

def fig3_complexity():
    """Figure 3: Effect of Objective Complexity on Inference."""
    data = load_json(OUTPUTS_DIR / "complexity_objective/20260218_002133/condition_stats.json")

    levels = ['L1', 'L2', 'L3', 'L4', 'L5']
    descriptions = ['1 prop', '2 props', '2 props + combo', '3-4 conds', '4-5 + penalties']
    means = []
    ses = []

    for level in levels:
        key = f"complexity_{level}"
        mean = data[key]["stats"]["objective_inference_score"]["mean"] * 100
        std = data[key]["stats"]["objective_inference_score"]["std"] * 100
        n = data[key]["stats"]["objective_inference_score"]["n"]
        means.append(mean)
        ses.append(std / np.sqrt(n))

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(range(len(levels)), means, yerr=ses, marker='o', capsize=5,
                linewidth=2, markersize=8, color='#2ca02c')
    ax.fill_between(range(len(levels)), [m - s for m, s in zip(means, ses)],
                    [m + s for m, s in zip(means, ses)], alpha=0.2, color='#2ca02c')

    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels([f'{l}\n({d})' for l, d in zip(levels, descriptions)])
    ax.set_xlabel('Complexity Level')
    ax.set_ylabel('Objective Inference F1 (%)')
    ax.set_title('Effect of Objective Complexity on Inference Accuracy')
    ax.set_ylim(0, 50)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig3_complexity.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig3_complexity.png")

def fig4_search_space():
    """Figure 4: Search Space Constraint Effect."""
    data = load_json(OUTPUTS_DIR / "search_space/20260218_002133/condition_stats.json")

    # Group by inference mode
    modes = ['freeform', 'structured', 'multiple_choice_2', 'multiple_choice_4',
             'multiple_choice_8', 'multiple_choice_16']
    mode_labels = ['Freeform', 'Structured', 'MC-2', 'MC-4', 'MC-8', 'MC-16']

    # Average across L1 and L5 for each mode
    means = []
    ses = []

    for mode in modes:
        mode_means = []
        mode_ns = []
        for level in ['L1', 'L5']:
            key = f"{mode}_{level}"
            if key in data:
                mean = data[key]["stats"]["objective_inference_score"]["mean"]
                n = data[key]["stats"]["objective_inference_score"]["n"]
                mode_means.append(mean)
                mode_ns.append(n)

        if mode_means:
            avg_mean = np.mean(mode_means) * 100
            # Use pooled SE
            avg_se = np.std(mode_means) / np.sqrt(len(mode_means)) * 100 if len(mode_means) > 1 else 5
            means.append(avg_mean)
            ses.append(avg_se)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(modes)))
    bars = ax.bar(range(len(modes)), means, yerr=ses, capsize=5, color=colors, edgecolor='black')

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(mode_labels)
    ax.set_xlabel('Inference Mode')
    ax.set_ylabel('Objective Inference F1 (%)')
    ax.set_title('Effect of Search Space Constraint on Objective Inference')
    ax.set_ylim(0, 110)

    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f'{mean:.0f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig4_search_space.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig4_search_space.png")

def fig5_theory_context():
    """Figure 5: Theory Context Effect on Inference."""
    # Combine data from two experiments:
    # - Original: none, brief, full
    # - Controlled: none, full, strategy_list, comprehensive
    data_original = load_json(OUTPUTS_DIR / "theory_context_experiment/20260221_131125/condition_stats.json")
    data_controlled = load_json(OUTPUTS_DIR / "controlled_context_experiment/phase2_20260226_001215/results.json")

    # Build combined data - use controlled for none/full (more recent), original for brief
    contexts = ['none', 'brief', 'full', 'strategy_list', 'comprehensive']
    context_labels = ['None', 'Brief\n(~50 words)', 'Full\n(~200 words)', 'Strategy List\n(~250 words)', 'Comprehensive\n(~5000 words)']

    means = []
    ses = []

    for ctx in contexts:
        if ctx == 'brief':
            # From original experiment
            mean = data_original[ctx]["stats"]["exact_f1"]["mean"] * 100
            std = data_original[ctx]["stats"]["exact_f1"]["std"] * 100
            n = data_original[ctx]["stats"]["exact_f1"]["n"]
            se = std / np.sqrt(n)
        else:
            # From controlled experiment
            mean = data_controlled["stats"][ctx]["mean"] * 100
            se = data_controlled["stats"][ctx]["stderr"] * 100
        means.append(mean)
        ses.append(se)

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ['#ff9999', '#ffcccc', '#99ccff', '#ffcc99', '#99ff99']
    bars = ax.bar(range(len(contexts)), means, yerr=ses, capsize=5,
                  color=colors, edgecolor='black', width=0.6)

    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels(context_labels)
    ax.set_xlabel('Theory Context')
    ax.set_ylabel('Exact F1 (%)')
    ax.set_title('Effect of Theory Context on Objective Inference')
    ax.set_ylim(0, 60)

    # Add value labels
    for bar, mean, se in zip(bars, means, ses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig5_theory_context.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig5_theory_context.png")

def fig6_agent_strategy_inference():
    """Figure 6: Agent Communication Strategy Effect on Inference."""
    data = load_json(OUTPUTS_DIR / "agent_strategy_inference/20260221_134220/condition_stats.json")

    strategies = ['aggressive', 'honest', 'subtle', 'natural', 'credibility_attack', 'deceptive', 'misdirection']
    strategy_labels = ['Aggressive', 'Honest', 'Subtle', 'Natural', 'Credibility\nAttack', 'Deceptive', 'Misdirection']

    means = []
    ses = []

    for strat in strategies:
        if strat in data:
            mean = data[strat]["stats"]["exact_f1"]["mean"] * 100
            std = data[strat]["stats"]["exact_f1"]["std"] * 100
            n = data[strat]["stats"]["exact_f1"]["n"]
            means.append(mean)
            ses.append(std / np.sqrt(n))
        else:
            means.append(0)
            ses.append(0)

    # Sort by mean (descending)
    sorted_idx = np.argsort(means)[::-1]
    means = [means[i] for i in sorted_idx]
    ses = [ses[i] for i in sorted_idx]
    strategy_labels = [strategy_labels[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(strategies)))
    bars = ax.barh(range(len(strategies)), means, xerr=ses, capsize=3,
                   color=colors, edgecolor='black')

    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategy_labels)
    ax.set_xlabel('Exact F1 (%)')
    ax.set_title('Agent Communication Strategy Effect on Estimator Inference')
    ax.set_xlim(0, 70)

    # Add value labels
    for i, (mean, se) in enumerate(zip(means, ses)):
        ax.text(mean + se + 1, i, f'{mean:.1f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig6_agent_strategy_inference.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig6_agent_strategy_inference.png")

def fig6b_agent_strategy_values():
    """Figure 6b: Agent Communication Strategy Effect on Judge/Agent Values."""
    import json
    from pathlib import Path

    base_dir = OUTPUTS_DIR / "agent_strategy_inference/20260221_134220"

    # Collect data by strategy
    strategy_data = {}

    for game_file in base_dir.glob('game_*.json'):
        data = json.load(open(game_file))
        strategy = data['condition']['agent_communication_strategy']

        if strategy not in strategy_data:
            strategy_data[strategy] = {
                'total_value': [],
                'agent_values': {'Agent_A': [], 'Agent_B': []},
            }

        metrics = data['metrics']
        strategy_data[strategy]['total_value'].append(metrics['total_value'])

        for agent, value in metrics['agent_cumulative_values'].items():
            strategy_data[strategy]['agent_values'][agent].append(value)

    # Order by judge value
    strategies = ['honest', 'credibility_attack', 'aggressive', 'deceptive', 'natural', 'subtle', 'misdirection']
    strategy_labels = ['Honest', 'Credibility\nAttack', 'Aggressive', 'Deceptive', 'Natural', 'Subtle', 'Misdirection']

    judge_means = []
    judge_ses = []
    agent_a_means = []
    agent_b_means = []

    for strat in strategies:
        if strat in strategy_data:
            d = strategy_data[strat]
            n = len(d['total_value'])
            judge_means.append(np.mean(d['total_value']))
            judge_ses.append(np.std(d['total_value']) / np.sqrt(n))
            agent_a_means.append(np.mean(d['agent_values']['Agent_A']))
            agent_b_means.append(np.mean(d['agent_values']['Agent_B']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left plot: Judge value
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(strategies)))
    bars1 = ax1.bar(range(len(strategies)), judge_means, yerr=judge_ses, capsize=4,
                    color=colors, edgecolor='black', width=0.7)
    ax1.set_xticks(range(len(strategies)))
    ax1.set_xticklabels(strategy_labels, fontsize=9)
    ax1.set_ylabel('Judge Total Value')
    ax1.set_title('Judge Value by Agent Communication Strategy')
    ax1.set_ylim(0, 250)

    for bar, mean in zip(bars1, judge_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
                f'{mean:.0f}', ha='center', va='bottom', fontsize=9)

    # Right plot: Agent values (grouped bar)
    x = np.arange(len(strategies))
    width = 0.35

    bars2a = ax2.bar(x - width/2, agent_a_means, width, label='Agent A', color='#5DA5DA', edgecolor='black')
    bars2b = ax2.bar(x + width/2, agent_b_means, width, label='Agent B', color='#FAA43A', edgecolor='black')

    ax2.set_xticks(x)
    ax2.set_xticklabels(strategy_labels, fontsize=9)
    ax2.set_ylabel('Agent Cumulative Value')
    ax2.set_title('Agent Values by Communication Strategy')
    ax2.set_ylim(0, 10)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig6b_agent_strategy_values.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig6b_agent_strategy_values.png")

def fig7_deception_strategies():
    """Figure 7: Deception Detection Strategies."""
    data = load_json(OUTPUTS_DIR / "deception_strategies_experiment/20260221_110535/condition_stats.json")

    strategies = ['baseline', 'consistency', 'incentive', 'pattern', 'combined']
    strategy_labels = ['Baseline', 'Consistency\nTracking', 'Incentive\nAnalysis',
                       'Pattern\nRecognition', 'Combined']

    means = []
    ses = []

    for strat in strategies:
        mean = data[strat]["stats"]["exact_f1"]["mean"] * 100
        std = data[strat]["stats"]["exact_f1"]["std"] * 100
        n = data[strat]["stats"]["exact_f1"]["n"]
        means.append(mean)
        ses.append(std / np.sqrt(n))

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = ['#808080', '#ffb366', '#66b3ff', '#99ff99', '#ff99cc']
    bars = ax.bar(range(len(strategies)), means, yerr=ses, capsize=5,
                  color=colors, edgecolor='black', width=0.6)

    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategy_labels)
    ax.set_xlabel('Detection Strategy')
    ax.set_ylabel('Exact F1 (%)')
    ax.set_title('Deception Detection Strategy Comparison')
    ax.set_ylim(0, 60)

    # Add value labels
    for bar, mean, se in zip(bars, means, ses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

    # Add horizontal line for baseline
    ax.axhline(y=means[0], color='gray', linestyle='--', alpha=0.5, label='Baseline')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig7_deception_strategies.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig7_deception_strategies.png")

def fig8_model_comparison():
    """Figure 8: Model Comparison."""
    data = load_json(OUTPUTS_DIR / "model_comparison_experiment/20260226_004501/results.json")

    models = []
    means = []
    ses = []

    for model_id, stats in data["stats"].items():
        models.append(stats["name"])
        means.append(stats["mean"] * 100)
        ses.append(stats["stderr"] * 100)

    # Sort by mean
    sorted_idx = np.argsort(means)[::-1]
    models = [models[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    ses = [ses[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(10, 5))

    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(models)))
    bars = ax.bar(range(len(models)), means, yerr=ses, capsize=5,
                  color=colors, edgecolor='black', width=0.7)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_xlabel('Model')
    ax.set_ylabel('Exact F1 (%)')
    ax.set_title('Model Comparison on Objective Inference (Full Theory Context)')
    ax.set_ylim(0, 55)

    # Add value labels
    for bar, mean, se in zip(bars, means, ses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + se + 1,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig8_model_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig8_model_comparison.png")

def fig9_f1_evolution():
    """Figure 9: F1 Evolution Over Number of Statements."""
    data = load_json(OUTPUTS_DIR / "f1_evolution_experiment/20260226_132316/results.json")

    checkpoints = data["checkpoints"]
    means = [d["mean_f1"] * 100 for d in data["evolution_data"]]
    ses = [d["se"] * 100 for d in data["evolution_data"]]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.errorbar(checkpoints, means, yerr=ses, marker='o', capsize=5,
                linewidth=2, markersize=8, color='#9467bd')
    ax.fill_between(checkpoints, [m - s for m, s in zip(means, ses)],
                    [m + s for m, s in zip(means, ses)], alpha=0.2, color='#9467bd')

    ax.set_xlabel('Number of Agent Statements')
    ax.set_ylabel('Exact F1 (%)')
    ax.set_title('F1 Score Evolution Over Number of Agent Statements')
    ax.set_xticks(checkpoints)
    ax.set_ylim(0, 60)

    # Add trend annotation
    max_idx = np.argmax(means)
    ax.scatter([checkpoints[max_idx]], [means[max_idx]], color='gold', s=150,
               zorder=5, marker='*', edgecolor='black')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig9_f1_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig9_f1_evolution.png")

def fig10_scale_heatmap():
    """Figure 10: Scale Experiment Heatmap."""
    data = load_json(RESULTS_DIR / "scale_experiment/results_20260212_173323.json")

    # Extract data
    agents = [2, 3, 4]
    rounds = [10, 15, 20]

    # Build matrix
    matrix = np.full((len(agents), len(rounds)), np.nan)

    for key, values in data["summary"].items():
        parts = key.split('_')
        n_agents = int(parts[0].replace('agents', ''))
        n_rounds = int(parts[1].replace('rounds', ''))

        if n_agents in agents and n_rounds in rounds:
            i = agents.index(n_agents)
            j = rounds.index(n_rounds)
            matrix[i, j] = values["property_accuracy"]["mean"] * 100

    fig, ax = plt.subplots(figsize=(7, 5))

    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=15, vmax=50)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selection Accuracy (%)')

    # Labels
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels(rounds)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents)
    ax.set_xlabel('Number of Rounds')
    ax.set_ylabel('Number of Agents')
    ax.set_title('Selection Accuracy by Scale (Agents x Rounds)')

    # Add values
    for i in range(len(agents)):
        for j in range(len(rounds)):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.0f}%', ha='center', va='center',
                        color='white' if matrix[i, j] > 35 else 'black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig10_scale_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved fig10_scale_heatmap.png")

def main():
    """Generate all plots."""
    print("Generating plots...")

    fig1_forced_oracle()
    fig2_oracle_budget()
    fig3_complexity()
    fig4_search_space()
    fig5_theory_context()
    fig6_agent_strategy_inference()
    fig6b_agent_strategy_values()
    fig7_deception_strategies()
    fig8_model_comparison()
    fig9_f1_evolution()
    fig10_scale_heatmap()

    print(f"\nAll plots saved to {PLOTS_DIR}")

if __name__ == "__main__":
    main()
