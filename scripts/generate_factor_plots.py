#!/usr/bin/env python3
"""Generate comprehensive factor analysis plots from experiment results."""

import json
from pathlib import Path
from collections import defaultdict
import statistics

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("matplotlib not available, will generate HTML plots only")


def load_results():
    """Load experiment results."""
    with open("outputs/multi_factor/20260207_104922/all_results.json") as f:
        return json.load(f)


def extract_metrics_by_factor(results):
    """Extract metrics grouped by each factor."""

    # Initialize containers
    by_observer = defaultdict(lambda: defaultdict(list))
    by_oracle = defaultdict(lambda: defaultdict(list))
    by_vf = defaultdict(lambda: defaultdict(list))
    by_complexity = defaultdict(lambda: defaultdict(list))

    # Also track interactions
    by_observer_oracle = defaultdict(lambda: defaultdict(list))
    by_observer_vf = defaultdict(lambda: defaultdict(list))
    by_oracle_vf = defaultdict(lambda: defaultdict(list))
    by_oracle_complexity = defaultdict(lambda: defaultdict(list))
    by_vf_complexity = defaultdict(lambda: defaultdict(list))

    for r in results:
        cond = r["condition"]
        metrics = r["metrics"]
        est_metrics = r.get("estimator_metrics", {}) or {}

        # Extract condition factors
        observer = cond["observer_condition"]
        oracle = "random" if cond["random_oracle"] else "strategic"
        vf = "complex" if cond["use_agent_value_functions"] else "simple"
        complexity = cond["rule_complexity"]

        # Extract metrics
        prop_acc = metrics.get("property_accuracy", 0) * 100
        rule_acc = metrics.get("rule_inference_accuracy", 0) * 100
        est_prop_acc = est_metrics.get("property_accuracy", 0) * 100 if est_metrics else 0
        total_value = metrics.get("total_value", 0)

        # Group by single factors
        by_observer[observer]["prop_acc"].append(prop_acc)
        by_observer[observer]["rule_acc"].append(rule_acc)
        by_observer[observer]["est_prop_acc"].append(est_prop_acc)
        by_observer[observer]["value"].append(total_value)

        by_oracle[oracle]["prop_acc"].append(prop_acc)
        by_oracle[oracle]["rule_acc"].append(rule_acc)
        by_oracle[oracle]["est_prop_acc"].append(est_prop_acc)
        by_oracle[oracle]["value"].append(total_value)

        by_vf[vf]["prop_acc"].append(prop_acc)
        by_vf[vf]["rule_acc"].append(rule_acc)
        by_vf[vf]["est_prop_acc"].append(est_prop_acc)
        by_vf[vf]["value"].append(total_value)

        by_complexity[complexity]["prop_acc"].append(prop_acc)
        by_complexity[complexity]["rule_acc"].append(rule_acc)
        by_complexity[complexity]["est_prop_acc"].append(est_prop_acc)
        by_complexity[complexity]["value"].append(total_value)

        # Group by factor interactions
        by_observer_oracle[f"{observer}_{oracle}"]["prop_acc"].append(prop_acc)
        by_observer_oracle[f"{observer}_{oracle}"]["rule_acc"].append(rule_acc)

        by_observer_vf[f"{observer}_{vf}"]["prop_acc"].append(prop_acc)
        by_observer_vf[f"{observer}_{vf}"]["rule_acc"].append(rule_acc)

        by_oracle_vf[f"{oracle}_{vf}"]["prop_acc"].append(prop_acc)
        by_oracle_vf[f"{oracle}_{vf}"]["rule_acc"].append(rule_acc)

        by_oracle_complexity[f"{oracle}_{complexity}"]["prop_acc"].append(prop_acc)
        by_oracle_complexity[f"{oracle}_{complexity}"]["rule_acc"].append(rule_acc)

        by_vf_complexity[f"{vf}_{complexity}"]["prop_acc"].append(prop_acc)
        by_vf_complexity[f"{vf}_{complexity}"]["rule_acc"].append(rule_acc)

    return {
        "by_observer": dict(by_observer),
        "by_oracle": dict(by_oracle),
        "by_vf": dict(by_vf),
        "by_complexity": dict(by_complexity),
        "by_observer_oracle": dict(by_observer_oracle),
        "by_observer_vf": dict(by_observer_vf),
        "by_oracle_vf": dict(by_oracle_vf),
        "by_oracle_complexity": dict(by_oracle_complexity),
        "by_vf_complexity": dict(by_vf_complexity),
    }


def compute_stats(data_dict):
    """Compute mean and standard error for each group."""
    stats = {}
    for key, metrics in data_dict.items():
        stats[key] = {}
        for metric_name, values in metrics.items():
            if values:
                n = len(values)
                mean = statistics.mean(values)
                std = statistics.stdev(values) if n > 1 else 0
                se = std / (n ** 0.5) if n > 1 else 0  # Standard error
                stats[key][metric_name] = {
                    "mean": mean,
                    "std": std,
                    "se": se,  # Standard error for error bars
                    "n": n
                }
    return stats


def generate_matplotlib_plots(grouped_data, output_dir, results):
    """Generate matplotlib plots."""
    if not HAS_MATPLOTLIB:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Color scheme
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c']

    # 1. By Observer Condition
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    stats = compute_stats(grouped_data["by_observer"])
    labels = list(stats.keys())

    for idx, metric in enumerate(["prop_acc", "rule_acc", "est_prop_acc"]):
        ax = axes[idx]
        means = [stats[l][metric]["mean"] for l in labels]
        stds = [stats[l][metric]["se"] for l in labels]  # Use SE for error bars

        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors[:len(labels)], alpha=0.8)
        ax.axhline(y=30.7, color='red', linestyle='--', label='Random baseline')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)')
        ax.set_title(f'{metric.replace("_", " ").title()} by Observer Condition')
        ax.set_ylim(0, 80)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "by_observer_condition.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. By Oracle Type
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    stats = compute_stats(grouped_data["by_oracle"])
    labels = ["strategic", "random"]

    for idx, metric in enumerate(["prop_acc", "rule_acc", "value"]):
        ax = axes[idx]
        means = [stats[l][metric]["mean"] for l in labels]
        stds = [stats[l][metric]["se"] for l in labels]  # Use SE for error bars

        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=[colors[0], colors[1]], alpha=0.8)
        if metric != "value":
            ax.axhline(y=30.7, color='red', linestyle='--', label='Random baseline')
            ax.set_ylim(0, 80)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}' + (' (%)' if metric != 'value' else ''))
        ax.set_title(f'{metric.replace("_", " ").title()} by Oracle Type')

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "by_oracle_type.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. By Value Function Type
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    stats = compute_stats(grouped_data["by_vf"])
    labels = ["simple", "complex"]

    for idx, metric in enumerate(["prop_acc", "rule_acc", "est_prop_acc"]):
        ax = axes[idx]
        means = [stats[l][metric]["mean"] for l in labels]
        stds = [stats[l][metric]["se"] for l in labels]  # Use SE for error bars

        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=[colors[0], colors[2]], alpha=0.8)
        ax.axhline(y=30.7, color='red', linestyle='--', label='Random baseline')
        ax.set_ylabel(f'{metric.replace("_", " ").title()} (%)')
        ax.set_title(f'{metric.replace("_", " ").title()} by Agent VF Type')
        ax.set_ylim(0, 80)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "by_vf_type.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. By Rule Complexity
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    stats = compute_stats(grouped_data["by_complexity"])
    labels = ["simple", "medium", "complex"]

    for idx, metric in enumerate(["prop_acc", "rule_acc", "value"]):
        ax = axes[idx]
        means = [stats[l][metric]["mean"] for l in labels]
        stds = [stats[l][metric]["se"] for l in labels]  # Use SE for error bars

        bars = ax.bar(labels, means, yerr=stds, capsize=5, color=colors[:3], alpha=0.8)
        if metric != "value":
            ax.axhline(y=30.7, color='red', linestyle='--', label='Random baseline')
            ax.set_ylim(0, 80)
        ax.set_ylabel(f'{metric.replace("_", " ").title()}' + (' (%)' if metric != 'value' else ''))
        ax.set_title(f'{metric.replace("_", " ").title()} by Rule Complexity')

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{mean:.1f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "by_rule_complexity.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Interaction: Observer x Oracle
    fig, ax = plt.subplots(figsize=(12, 6))

    stats = compute_stats(grouped_data["by_observer_oracle"])
    labels = sorted(stats.keys())
    means = [stats[l]["prop_acc"]["mean"] for l in labels]
    stds = [stats[l]["prop_acc"]["se"] for l in labels]  # Use SE for error bars

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors[i % len(colors)] for i in range(len(labels))], alpha=0.8)
    ax.axhline(y=30.7, color='red', linestyle='--', label='Random baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in labels], rotation=0)
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_title('Property Accuracy: Observer Condition × Oracle Type')
    ax.set_ylim(0, 80)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "interaction_observer_oracle.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Interaction: Oracle x Complexity
    fig, ax = plt.subplots(figsize=(12, 6))

    stats = compute_stats(grouped_data["by_oracle_complexity"])
    labels = sorted(stats.keys())
    means = [stats[l]["prop_acc"]["mean"] for l in labels]
    stds = [stats[l]["prop_acc"]["se"] for l in labels]  # Use SE for error bars

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors[i % len(colors)] for i in range(len(labels))], alpha=0.8)
    ax.axhline(y=30.7, color='red', linestyle='--', label='Random baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in labels], rotation=0)
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_title('Property Accuracy: Oracle Type × Rule Complexity')
    ax.set_ylim(0, 80)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "interaction_oracle_complexity.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 7. Interaction: VF x Complexity
    fig, ax = plt.subplots(figsize=(12, 6))

    stats = compute_stats(grouped_data["by_vf_complexity"])
    labels = sorted(stats.keys())
    means = [stats[l]["prop_acc"]["mean"] for l in labels]
    stds = [stats[l]["prop_acc"]["se"] for l in labels]  # Use SE for error bars

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=[colors[i % len(colors)] for i in range(len(labels))], alpha=0.8)
    ax.axhline(y=30.7, color='red', linestyle='--', label='Random baseline')
    ax.set_xticks(x)
    ax.set_xticklabels([l.replace('_', '\n') for l in labels], rotation=0)
    ax.set_ylabel('Property Accuracy (%)')
    ax.set_title('Property Accuracy: Agent VF Type × Rule Complexity')
    ax.set_ylim(0, 80)

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{mean:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "interaction_vf_complexity.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 8. Heatmap: All factors combined (Observer x Oracle, colored by complexity)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, complexity in enumerate(["simple", "medium", "complex"]):
        ax = axes[idx]

        # Create grid data
        observers = ["blind", "ids", "interests"]
        oracles = ["strategic", "random"]
        vfs = ["simple", "complex"]

        data = np.zeros((len(observers) * len(vfs), len(oracles)))
        labels_y = []

        for i, obs in enumerate(observers):
            for j, vf in enumerate(vfs):
                row_idx = i * 2 + j
                labels_y.append(f"{obs}\n{vf}")
                for k, oracle in enumerate(oracles):
                    key = f"{obs}_oracle-{oracle}_vf-{vf}_{complexity}"
                    # Find matching results
                    matching = [r for r in results
                               if r["condition"]["observer_condition"] == obs
                               and r["condition"]["random_oracle"] == (oracle == "random")
                               and r["condition"]["use_agent_value_functions"] == (vf == "complex")
                               and r["condition"]["rule_complexity"] == complexity]
                    if matching:
                        data[row_idx, k] = statistics.mean([r["metrics"].get("property_accuracy", 0) * 100 for r in matching])

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=20, vmax=80)
        ax.set_xticks(np.arange(len(oracles)))
        ax.set_yticks(np.arange(len(labels_y)))
        ax.set_xticklabels(oracles)
        ax.set_yticklabels(labels_y)
        ax.set_title(f'Rule Complexity: {complexity}')

        # Add text annotations
        for i in range(len(labels_y)):
            for j in range(len(oracles)):
                ax.text(j, i, f'{data[i, j]:.1f}', ha='center', va='center',
                       color='black' if data[i, j] > 50 else 'white', fontsize=10)

        if idx == 2:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Property Accuracy (%)')

    plt.suptitle('Property Accuracy Heatmap by All Factors', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "heatmap_all_factors.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved plots to {output_dir}")


def generate_html_plots(grouped_data, output_path):
    """Generate HTML file with Plotly plots."""

    stats = {
        "by_observer": compute_stats(grouped_data["by_observer"]),
        "by_oracle": compute_stats(grouped_data["by_oracle"]),
        "by_vf": compute_stats(grouped_data["by_vf"]),
        "by_complexity": compute_stats(grouped_data["by_complexity"]),
        "by_observer_oracle": compute_stats(grouped_data["by_observer_oracle"]),
        "by_oracle_complexity": compute_stats(grouped_data["by_oracle_complexity"]),
        "by_vf_complexity": compute_stats(grouped_data["by_vf_complexity"]),
    }

    html = '''<!DOCTYPE html>
<html>
<head>
    <title>Factor Analysis Plots</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: -apple-system, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #2c3e50; }
        h2 { color: #34495e; margin-top: 40px; }
        .chart { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
<div class="container">
    <h1>Multi-Factor Experiment: Accuracy by Factor</h1>
    <p>179 games across 36 conditions. Red dashed line = 30.7% random baseline.</p>

    <h2>1. Single Factor Effects</h2>
    <div class="grid">
        <div class="chart"><div id="observer-prop"></div></div>
        <div class="chart"><div id="observer-rule"></div></div>
        <div class="chart"><div id="oracle-prop"></div></div>
        <div class="chart"><div id="oracle-value"></div></div>
        <div class="chart"><div id="vf-prop"></div></div>
        <div class="chart"><div id="vf-rule"></div></div>
        <div class="chart"><div id="complexity-prop"></div></div>
        <div class="chart"><div id="complexity-rule"></div></div>
    </div>

    <h2>2. Factor Interactions</h2>
    <div class="chart"><div id="observer-oracle"></div></div>
    <div class="chart"><div id="oracle-complexity"></div></div>
    <div class="chart"><div id="vf-complexity"></div></div>

    <h2>3. Estimator Accuracy</h2>
    <div class="grid">
        <div class="chart"><div id="est-observer"></div></div>
        <div class="chart"><div id="est-oracle"></div></div>
    </div>
</div>

<script>
const stats = ''' + json.dumps(stats) + ''';

const layout = {
    paper_bgcolor: 'white',
    plot_bgcolor: 'white',
    font: {size: 12},
    showlegend: false,
    margin: {t: 50, b: 50}
};

function makeBarChart(divId, data, title, yLabel, showBaseline=true) {
    const labels = Object.keys(data);
    const means = labels.map(l => data[l].prop_acc ? data[l].prop_acc.mean : (data[l].rule_acc ? data[l].rule_acc.mean : data[l].value.mean));
    const stds = labels.map(l => data[l].prop_acc ? data[l].prop_acc.se : (data[l].rule_acc ? data[l].rule_acc.se : data[l].value.se));

    const traces = [{
        x: labels,
        y: means,
        error_y: {type: 'data', array: stds, visible: true},
        type: 'bar',
        marker: {color: ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c'].slice(0, labels.length)},
        text: means.map(m => m.toFixed(1)),
        textposition: 'outside'
    }];

    const shapes = showBaseline ? [{
        type: 'line', x0: -0.5, x1: labels.length - 0.5, y0: 30.7, y1: 30.7,
        line: {color: 'red', width: 2, dash: 'dash'}
    }] : [];

    Plotly.newPlot(divId, traces, {...layout, title: title, yaxis: {title: yLabel, range: [0, Math.max(...means) * 1.3]}, shapes: shapes});
}

function makeBarChartMetric(divId, data, metric, title, yLabel, showBaseline=true) {
    const labels = Object.keys(data);
    const means = labels.map(l => data[l][metric] ? data[l][metric].mean : 0);
    const stds = labels.map(l => data[l][metric] ? data[l][metric].se : 0);

    const traces = [{
        x: labels,
        y: means,
        error_y: {type: 'data', array: stds, visible: true},
        type: 'bar',
        marker: {color: ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12', '#1abc9c'].slice(0, labels.length)},
        text: means.map(m => m.toFixed(1)),
        textposition: 'outside'
    }];

    const shapes = showBaseline ? [{
        type: 'line', x0: -0.5, x1: labels.length - 0.5, y0: 30.7, y1: 30.7,
        line: {color: 'red', width: 2, dash: 'dash'}
    }] : [];

    Plotly.newPlot(divId, traces, {...layout, title: title, yaxis: {title: yLabel, range: [0, Math.max(...means) * 1.3]}, shapes: shapes});
}

// Single factors
makeBarChartMetric('observer-prop', stats.by_observer, 'prop_acc', 'Property Accuracy by Observer Condition', 'Accuracy (%)');
makeBarChartMetric('observer-rule', stats.by_observer, 'rule_acc', 'Rule Inference by Observer Condition', 'Accuracy (%)');
makeBarChartMetric('oracle-prop', stats.by_oracle, 'prop_acc', 'Property Accuracy by Oracle Type', 'Accuracy (%)');
makeBarChartMetric('oracle-value', stats.by_oracle, 'value', 'Total Value by Oracle Type', 'Value', false);
makeBarChartMetric('vf-prop', stats.by_vf, 'prop_acc', 'Property Accuracy by Agent VF Type', 'Accuracy (%)');
makeBarChartMetric('vf-rule', stats.by_vf, 'rule_acc', 'Rule Inference by Agent VF Type', 'Accuracy (%)');
makeBarChartMetric('complexity-prop', stats.by_complexity, 'prop_acc', 'Property Accuracy by Rule Complexity', 'Accuracy (%)');
makeBarChartMetric('complexity-rule', stats.by_complexity, 'rule_acc', 'Rule Inference by Rule Complexity', 'Accuracy (%)');

// Interactions
makeBarChartMetric('observer-oracle', stats.by_observer_oracle, 'prop_acc', 'Property Accuracy: Observer × Oracle', 'Accuracy (%)');
makeBarChartMetric('oracle-complexity', stats.by_oracle_complexity, 'prop_acc', 'Property Accuracy: Oracle × Complexity', 'Accuracy (%)');
makeBarChartMetric('vf-complexity', stats.by_vf_complexity, 'prop_acc', 'Property Accuracy: VF × Complexity', 'Accuracy (%)');

// Estimator
makeBarChartMetric('est-observer', stats.by_observer, 'est_prop_acc', 'Estimator Accuracy by Observer Condition', 'Accuracy (%)');
makeBarChartMetric('est-oracle', stats.by_oracle, 'est_prop_acc', 'Estimator Accuracy by Oracle Type', 'Accuracy (%)');
</script>
</body>
</html>'''

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Saved HTML plots to {output_path}")


def main():
    print("Loading results...")
    results = load_results()
    print(f"Loaded {len(results)} games")

    print("Extracting metrics by factor...")
    grouped_data = extract_metrics_by_factor(results)

    # Store results globally for heatmap
    global results_global
    results_global = results

    output_dir = Path("results/multi_factor/plots")

    if HAS_MATPLOTLIB:
        print("Generating matplotlib plots...")
        generate_matplotlib_plots(grouped_data, output_dir, results)

    print("Generating HTML plots...")
    generate_html_plots(grouped_data, output_dir / "factor_analysis.html")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)

    for factor_name, factor_data in [
        ("Observer Condition", grouped_data["by_observer"]),
        ("Oracle Type", grouped_data["by_oracle"]),
        ("Agent VF Type", grouped_data["by_vf"]),
        ("Rule Complexity", grouped_data["by_complexity"]),
    ]:
        print(f"\n### {factor_name} ###")
        stats = compute_stats(factor_data)
        for key, metrics in stats.items():
            prop_acc = metrics.get("prop_acc", {})
            rule_acc = metrics.get("rule_acc", {})
            print(f"  {key}: prop_acc={prop_acc.get('mean', 0):.1f}% (SE±{prop_acc.get('se', 0):.1f}), "
                  f"rule_acc={rule_acc.get('mean', 0):.1f}% (SE±{rule_acc.get('se', 0):.1f}), n={prop_acc.get('n', 0)}")


if __name__ == "__main__":
    results_global = None
    main()
