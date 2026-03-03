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

# Gradient for ranked bars (green=best to red=worst)
STRATEGY_COLORS = [
    "#06d6a0",  # Aggressive (best) - teal
    "#43aa8b",  # Honest
    "#90be6d",  # Subtle
    "#8d99ae",  # Natural (baseline) - gray
    "#f9c74f",  # Credibility Attack - yellow
    "#f8961e",  # Deceptive - orange
    "#ef476f",  # Misdirection (worst) - red
]

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
# Plot 1: Agent Strategy Effect (Horizontal Bar)
# ============================================================================

def plot_agent_strategy():
    data = load_json(RESULTS_DIR / "agent_strategy_inference/condition_stats_20260221_134220.json")

    strategies = ["aggressive", "honest", "subtle", "natural", "credibility_attack", "deceptive", "misdirection"]
    labels = ["Aggressive", "Honest", "Subtle", "Natural\n(baseline)", "Credibility\nAttack", "Deceptive", "Misdirection"]

    means, ses = [], []
    for s in strategies:
        st = data[s]["stats"]["exact_f1"]
        means.append(st["mean"] * 100)
        ses.append(st["std"] * 100 / np.sqrt(st["n"]))

    fig, ax = plt.subplots(figsize=(9, 5))

    # Reverse everything so best (aggressive) is at top
    means_r = means[::-1]
    ses_r = ses[::-1]
    labels_r = labels[::-1]
    colors_r = STRATEGY_COLORS[::-1]

    y = np.arange(len(strategies))
    bars = ax.barh(y, means_r, xerr=ses_r, capsize=4,
                   color=colors_r, edgecolor="white", linewidth=0.5, height=0.65)

    ax.set_yticks(y)
    ax.set_yticklabels(labels_r)
    ax.set_xlabel("Exact F1 (%)")
    ax.set_title("Effect of Agent Communication Strategy")
    ax.set_xlim(0, 70)

    # Horizontal grid instead of vertical
    ax.grid(axis="x", color="#eeeeee", linewidth=0.6)
    ax.grid(axis="y", visible=False)

    # Value labels
    for bar, mean, se in zip(bars, means_r, ses_r):
        ax.text(
            mean + se + 1.5, bar.get_y() + bar.get_height() / 2,
            f"{mean:.1f}%",
            ha="left", va="center",
            fontsize=9, fontweight="600", color=COLORS["dark"],
        )

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig6_agent_strategy_inference.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig6_agent_strategy_inference.png")


# ============================================================================
# Plot 2: Oracle Effect (Before/After)
# ============================================================================

def plot_oracle_effect():
    data = load_json(RESULTS_DIR / "forced_oracle_test/results_20260213_164112.json")

    no_oracle = [r["property_accuracy"] for r in data["no_oracle"]]
    forced = [r["property_accuracy"] for r in data["forced_oracle"]]

    means = [np.mean(no_oracle) * 100, np.mean(forced) * 100]
    ses = [np.std(no_oracle) / np.sqrt(len(no_oracle)) * 100,
           np.std(forced) / np.sqrt(len(forced)) * 100]

    fig, ax = plt.subplots(figsize=(6, 5))

    colors = [COLORS["danger"], COLORS["success"]]
    bars = ax.bar([0, 1], means, yerr=ses, capsize=5,
                  color=colors, edgecolor="white", linewidth=0.5, width=0.55, zorder=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Oracle\n(budget = 0)", "Forced Oracle\n(budget = 8)"])
    ax.set_ylabel("Property Accuracy (%)")
    ax.set_title("Effect of Oracle Access")
    ax.set_ylim(0, 95)

    add_bar_labels(ax, bars, means, ses, offset=2)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "fig1_forced_oracle.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig1_forced_oracle.png")


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
        "Consistency\nchecking",
        "Incentive\nanalysis",
        "Pattern\nrecognition",
    ]

    entries = [
        ds["baseline"],
        tc["brief"],
        tc["full"],
        ds["consistency"],
        ds["incentive"],
        ds["pattern"],
    ]

    means, ses = [], []
    for e in entries:
        s = e["stats"]["exact_f1"]
        means.append(s["mean"] * 100)
        ses.append(s["std"] * 100 / np.sqrt(s["n"]))

    # Colors: gray baseline, warm orange for theory, cool blue for detection
    colors = [
        COLORS["neutral"],
        "#ffb366", "#f08c00",       # Theory: light to dark orange
        "#74c0fc", "#4dabf7", "#339af0",  # Detection: light to dark blue
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(range(len(labels)), means, yerr=ses, capsize=4,
                  color=colors, edgecolor="white", linewidth=0.5, width=0.6)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Exact F1 (%)")
    ax.set_title("Effect of Context on Objective Inference")
    ax.set_ylim(0, 68)

    add_bar_labels(ax, bars, means, ses)

    # Category labels
    fig.text(0.345, 0.01, "Theory context", ha="center", fontsize=9,
             fontweight="bold", color="#d35400")
    fig.text(0.72, 0.01, "Detection strategy", ha="center", fontsize=9,
             fontweight="bold", color="#2980b9")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(PLOTS_DIR / "fig5_effect_of_context.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved fig5_effect_of_context.png")


# ============================================================================
# Plot 5: Model Comparison
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
    plot_agent_strategy()
    plot_oracle_effect()
    plot_f1_evolution()
    plot_effect_of_context()
    plot_model_comparison()
    print("\nAll blog plots generated.")
