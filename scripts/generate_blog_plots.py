"""Generate unified-style plots for the blog post."""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ============================================================================
# Unified Style
# ============================================================================

# Muted, modern palette
COLORS = {
    "primary": "#4361ee",     # Blue
    "secondary": "#7209b7",   # Purple
    "accent": "#f72585",      # Pink
    "success": "#06d6a0",     # Teal
    "warning": "#ffd166",     # Yellow
    "danger": "#ef476f",      # Red
    "neutral": "#8d99ae",     # Gray
    "dark": "#2b2d42",        # Dark
    "light": "#edf2f4",       # Light bg
}

# Fixed color per strategy (consistent across all plots)
STRATEGY_COLOR_MAP = {
    "aggressive":       "#06d6a0",  # teal
    "honest":           "#43aa8b",  # green
    "subtle":           "#90be6d",  # lime
    "natural":          "#8d99ae",  # gray
    "credibility_attack": "#f9c74f",  # yellow
    "deceptive":        "#f8961e",  # orange
    "misdirection":     "#ef476f",  # red
}

def setup_style():
    """Apply unified plot style."""
    plt.rcParams.update({
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 150,
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#cccccc",
        "axes.linewidth": 0.8,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 16,
        "axes.labelsize": 11,
        "axes.labelpad": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Grid
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#eeeeee",
        "grid.linewidth": 0.6,
        # Ticks
        "xtick.major.size": 0,
        "ytick.major.size": 4,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # Legend
        "legend.frameon": False,
        "legend.fontsize": 10,
    })


RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUTS_DIR = Path(__file__).parent.parent / "outputs"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def add_bar_labels(ax, bars, means, ses, fmt="{:.1f}%", offset=1.5, fontsize=9):
    """Add value labels above bars."""
    for bar, mean, se in zip(bars, means, ses):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + se + offset,
            fmt.format(mean),
            ha="center", va="bottom",
            fontsize=fontsize, fontweight="600", color=COLORS["dark"],
        )


# ============================================================================
# Plot 1: Oracle Budget Effect on Objective Inference F1
# ============================================================================

def plot_oracle_budget():
    # Data from oracle_budget_objective experiment (principled inference, 10 seeds per condition)
    budgets = [0, 1, 2, 4, 6, 8]
    means =   [16.7, 33.3, 30.0, 40.0, 50.0, 41.7]
    stds =    [19.2, 20.8, 23.3, 23.8, 26.1, 21.2]
    n = 10
    ses = [s / np.sqrt(n) for s in stds]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(budgets, [m - s for m, s in zip(means, ses)],
                    [m + s for m, s in zip(means, ses)],
                    alpha=0.15, color=COLORS["primary"])
    ax.errorbar(budgets, means, yerr=ses, marker="o", capsize=4,
                linewidth=2, markersize=7, color=COLORS["primary"],
                markerfacecolor="white", markeredgewidth=2)

    ax.set_xlabel("Oracle Budget (number of queries)")
    ax.set_ylabel("Objective Inference Score (%)")
    ax.set_title("Effect of Oracle Budget on Objective Inference")
    ax.set_ylim(0, 70)
    ax.set_xticks(budgets)

    # Label each point
    for b, m, s in zip(budgets, means, ses):
        ax.text(b, m + s + 1.5, f"{m:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="600", color=COLORS["dark"])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig1_oracle_budget.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_oracle_budget.png")


# ============================================================================
# Plot 3: F1 Evolution Over Statements
# ============================================================================

def plot_f1_evolution():
    # Data from EXPERIMENTAL_RESULTS_SUMMARY.md
    stmts = [6, 12, 18, 24, 30, 36, 42, 48]
    means = [40.0, 46.7, 43.3, 50.0, 48.3, 48.3, 35.0, 46.7]
    stds  = [19.6, 24.6, 28.5, 24.8, 24.2, 27.7, 20.0, 24.6]
    n = 10
    ses = [s / np.sqrt(n) for s in stds]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.fill_between(stmts, [m - s for m, s in zip(means, ses)],
                    [m + s for m, s in zip(means, ses)],
                    alpha=0.15, color=COLORS["primary"])
    ax.errorbar(stmts, means, yerr=ses, marker="o", capsize=4,
                linewidth=2, markersize=7, color=COLORS["primary"],
                markerfacecolor="white", markeredgewidth=2)

    ax.set_xlabel("Number of Agent Statements")
    ax.set_ylabel("Exact F1 (%)")
    ax.set_title("F1 Score vs. Number of Statements")
    ax.set_ylim(0, 65)
    ax.set_xticks(stmts)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig9_f1_evolution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig9_f1_evolution.png")


# ============================================================================
# Plot 4: Effect of Context (Theory + Detection combined)
# ============================================================================

def plot_effect_of_context():
    tc = load_json(RESULTS_DIR / "theory_context_experiment/condition_stats_20260221_131125.json")
    ds = load_json(RESULTS_DIR / "deception_strategies_experiment/condition_stats_20260221_110535.json")

    labels = [
        "Baseline",
        "Brief theory\n(~50 words)",
        "Full theory\n(~200 words)",
        "Comprehensive\n(~5000 words)",
        "Consistency\nchecking",
        "Incentive\nanalysis",
        "Pattern\nrecognition",
    ]

    entries = [
        ds["baseline"],
        tc["brief"],
        tc["full"],
    ]

    means, ses = [], []
    for e in entries:
        s = e["stats"]["exact_f1"]
        means.append(s["mean"] * 100)
        ses.append(s["std"] * 100 / np.sqrt(s["n"]))

    # Comprehensive context (from controlled experiment, within-subjects n=10)
    means.append(43.3)
    ses.append(25.1 / np.sqrt(10))

    # Detection strategies
    for key in ["consistency", "incentive", "pattern"]:
        s = ds[key]["stats"]["exact_f1"]
        means.append(s["mean"] * 100)
        ses.append(s["std"] * 100 / np.sqrt(s["n"]))

    # Colors: gray baseline, warm orange for theory, cool blue for detection
    colors = [
        COLORS["neutral"],
        "#ffb366", "#f08c00", "#cc6600",  # Theory: light to dark orange
        "#74c0fc", "#4dabf7", "#339af0",  # Detection: light to dark blue
    ]

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(range(len(labels)), means, yerr=ses, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Exact F1 (%)")
    ax.set_title("Effect of Context on Objective Inference")
    ax.set_ylim(0, 68)

    add_bar_labels(ax, bars, means, ses)

    # Category labels
    fig.text(0.36, 0.01, "Theory context", ha="center", fontsize=9,
             fontweight="bold", color="#d35400")
    fig.text(0.76, 0.01, "Detection strategy", ha="center", fontsize=9,
             fontweight="bold", color="#2980b9")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(PLOTS_DIR / "fig5_effect_of_context.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig5_effect_of_context.png")


# ============================================================================
# Plot 5: Strategy Effect on Game Outcomes
# ============================================================================

def plot_strategy_combined():
    """Combined 3-panel plot: inference F1, judge reward, agent reward."""
    # --- Inference F1 data ---
    inf_data = load_json(RESULTS_DIR / "agent_strategy_inference/condition_stats_20260221_134220.json")
    inf_keys = ["aggressive", "honest", "subtle", "natural", "credibility_attack", "deceptive", "misdirection"]
    inf_labels = ["Aggr.", "Honest", "Subtle", "Natural", "Cred.\nAttack", "Decep.", "Misdir."]
    inf_means, inf_ses = [], []
    for s in inf_keys:
        st = inf_data[s]["stats"]["exact_f1"]
        inf_means.append(st["mean"] * 100)
        inf_ses.append(st["std"] * 100 / np.sqrt(st["n"]))

    # --- Game outcome data (from EXPERIMENTAL_RESULTS_SUMMARY.md) ---
    # Ordered same as inference: aggressive, honest, subtle, natural, cred_attack, deceptive, misdirection
    judge_values = [161.7, 191.8, 147.3, 154.2, 173.0, 156.6, 119.0]
    judge_ses =    [19.1,  17.5,  18.8,  19.5,  18.1,  18.0,  13.5]
    agent_values = [11.9,  10.6,  11.2,  10.7,  10.5,  11.8,  11.7]
    # SEs estimated from original plot (generated from raw per-game data)
    agent_ses =    [0.5,   0.4,   0.5,   0.4,   0.5,   0.4,   0.4]
    ns =           [10,    10,    10,    10,    6,     9,     10]

    colors = [STRATEGY_COLOR_MAP[k] for k in inf_keys]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(len(inf_keys))

    # Panel 1: Inference F1
    bars1 = ax1.bar(x, inf_means, yerr=inf_ses, capsize=3,
                    color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(inf_labels, fontsize=8)
    ax1.set_ylabel("Exact F1 (%)")
    ax1.set_title("Estimator Inference Accuracy")
    ax1.set_ylim(0, 72)
    for bar, val, se in zip(bars1, inf_means, inf_ses):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + se + 1,
                 f"{val:.0f}%", ha="center", va="bottom",
                 fontsize=8, fontweight="600", color=COLORS["dark"])

    # Panel 2: Judge reward
    bars2 = ax2.bar(x, judge_values, yerr=judge_ses, capsize=3,
                    color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(inf_labels, fontsize=8)
    ax2.set_ylabel("Judge Total Value")
    ax2.set_title("Judge Reward")
    ax2.set_ylim(0, 240)
    for bar, val, se in zip(bars2, judge_values, judge_ses):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + se + 3,
                 f"{val:.0f}", ha="center", va="bottom",
                 fontsize=8, fontweight="600", color=COLORS["dark"])

    # Panel 3: Agent reward
    bars3 = ax3.bar(x, agent_values, yerr=agent_ses, capsize=3,
                    color=colors, edgecolor="white", linewidth=0.5, width=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(inf_labels, fontsize=8)
    ax3.set_ylabel("Combined Agent Value (A + B)")
    ax3.set_title("Agent Reward")
    ax3.set_ylim(0, 15)
    for bar, val, se in zip(bars3, agent_values, agent_ses):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + se + 0.15,
                 f"{val:.1f}", ha="center", va="bottom",
                 fontsize=8, fontweight="600", color=COLORS["dark"])

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig6_strategy_combined.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig6_strategy_combined.png")


# ============================================================================
# Plot 6: Model Comparison
# ============================================================================

def plot_model_comparison():
    # Data from EXPERIMENTAL_RESULTS_SUMMARY.md (within-subjects, n=10)
    models = ["Haiku 4.5", "Opus 4", "Opus 4.5", "Opus 4.6",
              "Sonnet 4", "Sonnet 4.5", "Sonnet 4.6"]
    means = [43.3, 40.0, 40.0, 40.0, 38.3, 38.3, 36.7]
    stds = [19.6, 27.4, 26.3, 25.1, 27.3, 24.9, 27.0]
    n = 10
    ses = [s / np.sqrt(n) for s in stds]

    # Sort by mean descending
    order = np.argsort(means)[::-1]
    models = [models[i] for i in order]
    means = [means[i] for i in order]
    ses = [ses[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.bar(range(len(models)), means, yerr=ses, capsize=4,
                  color=COLORS["primary"], edgecolor="white", linewidth=0.5, width=0.6)

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Exact F1 (%)")
    ax.set_title("Effect of Model Capability")
    ax.set_ylim(0, 60)

    add_bar_labels(ax, bars, means, ses)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig8_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig8_model_comparison.png")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    setup_style()
    plot_strategy_combined()
    plot_oracle_budget()
    plot_f1_evolution()
    plot_effect_of_context()
    plot_model_comparison()
    print("\nAll blog plots generated.")
