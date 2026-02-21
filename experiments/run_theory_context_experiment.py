#!/usr/bin/env python3
"""Experiment: Theoretical Context for Objective Inference

Research Question:
Does providing the estimator with theoretical knowledge about learning from
incomplete/strategic information improve its ability to infer agent objectives?

Background:
From our literature review, key theoretical insights that could help the estimator:

1. Crawford-Sobel Partial Revelation:
   - Agents with different preferences cannot fully hide their objectives
   - Information is partially revealed through strategic communication
   - Distortions are predictable: agents bias toward their own interests

2. IRL Preference Leakage:
   - Even with deceptive demonstrations, preference orderings leak
   - "The observer will recover a reward function that respects the ordering
      of policies in the true reward function"
   - Pattern of advocacy reveals preferences despite individual claim distortions

3. Bias Correction Principles:
   - Multiple observations enable bias estimation
   - Known bias structure can be inverted to recover true signal
   - Aggregate patterns are more reliable than individual statements

Hypothesis:
Providing theoretical context about strategic communication will help the estimator:
1. Not take individual claims at face value
2. Focus on aggregate patterns rather than specific statements
3. Apply inverse reasoning (what would agent gain from this claim?)

Conditions:
1. baseline (none): Standard principled inference prompt
2. theory_brief: Add 2-3 sentence summary of key insights
3. theory_full: Add full theoretical framework (~200 words)

Design:
- 3 conditions (theory context levels)
- Fixed complexity level (L3 = 3 properties)
- 10 seeds per condition = 30 games total
- Principled evaluation (overlap scoring)
- NO CoT access (test on observable behavior)
- Deception strategy: baseline (isolate effect of theory context)

Metrics:
- Exact F1 (primary)
- Property recall (partial credit)
- Confidence calibration
"""

import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import wandb
from src.environment.simulation import GameConfig, HiddenValueGame

# ============================================================================
# Experimental Configuration
# ============================================================================

SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]

# Theory context conditions to test
THEORY_CONDITIONS = [
    "none",   # No theory context (baseline)
    "brief",  # Brief summary (~50 words)
    "full",   # Full theoretical framework (~200 words)
]

# Fixed game parameters
BASE_CONFIG = {
    "n_objects": 10,
    "n_agents": 2,
    "n_rounds": 10,
    "oracle_budget": 4,
    "selection_size": 5,
    "enable_estimator": True,
    "infer_agent_objectives": True,
    "use_agent_value_functions": True,
    "use_simple_value_functions": True,
    "agent_value_function_complexity": "L3",  # Fixed at 3 properties
    "objective_inference_mode": "principled",
    # Agent thinking enabled (but NOT shared with estimator)
    "enable_agent_thinking": True,
    "agent_thinking_budget": 5000,
    # Estimator does NOT see agent thinking (test theory on observable only)
    "estimator_sees_agent_thinking": False,
    # Estimator thinking
    "enable_estimator_thinking": True,
    "estimator_thinking_budget": 5000,
    # Deception strategy: baseline (isolate theory context effect)
    "estimator_deception_strategy": "baseline",
    # Models
    "estimator_model": "claude-sonnet-4-20250514",
    "agent_model": "claude-sonnet-4-20250514",
    "observer_model": "claude-sonnet-4-20250514",
    # Fixed structure
    "turn_structure": "interleaved",
    "oracle_timing": "before_response",
    "debate_structure": "open",
    "condition": "ids",
    "force_oracle": True,
}


@dataclass
class ExperimentCondition:
    """A single experimental condition (theory context level)."""
    theory_context: str

    def to_dict(self) -> dict:
        return {"theory_context": self.theory_context}


def run_single_game(condition: ExperimentCondition, seed: int) -> dict:
    """Run a single game with the given theory context and seed."""
    config = GameConfig(
        **BASE_CONFIG,
        estimator_theory_context=condition.theory_context,
        seed=seed,
    )

    game = HiddenValueGame(config)
    result = game.run()

    # Extract agent value function details
    agent_vf_details = []
    for agent in result.agents:
        vf = agent.get("value_function", {})
        agent_vf_details.append({
            "agent_id": agent.get("id"),
            "n_properties": vf.get("n_properties") or len(vf.get("cares_about", [])),
            "cares_about": vf.get("cares_about", []),
        })

    return {
        "condition": condition.to_dict(),
        "seed": seed,
        "metrics": result.metrics,
        "agent_objective_inference": result.agent_objective_inference,
        "agent_objective_scores": result.agent_objective_scores,
        "agent_objective_overall_score": result.agent_objective_overall_score,
        "agent_value_functions": agent_vf_details,
        "config": result.config,
    }


def compute_condition_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics for a condition."""
    if not results:
        return {}

    def safe_stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return {"mean": 0, "std": 0, "n": 0}
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        return {"mean": mean, "std": std, "n": len(values)}

    stats = {}

    # Primary metric: objective inference score (exact F1)
    obj_scores = [r.get("agent_objective_overall_score") for r in results]
    stats["objective_inference_score"] = safe_stats(obj_scores)

    # Extract overlap metrics
    exact_f1_list = []
    exact_precision_list = []
    exact_recall_list = []
    property_recall_list = []
    confidence_list = []

    for r in results:
        inf = r.get("agent_objective_inference", {})
        for agent_id, data in inf.items():
            if isinstance(data, dict):
                if "overlap_metrics" in data:
                    om = data["overlap_metrics"]
                    exact_f1_list.append(om.get("exact_f1", 0))
                    exact_precision_list.append(om.get("exact_precision", 0))
                    exact_recall_list.append(om.get("exact_recall", 0))
                    property_recall_list.append(om.get("property_recall", 0))
                confidence_list.append(data.get("confidence", 0))

    stats["exact_f1"] = safe_stats(exact_f1_list)
    stats["exact_precision"] = safe_stats(exact_precision_list)
    stats["exact_recall"] = safe_stats(exact_recall_list)
    stats["property_recall"] = safe_stats(property_recall_list)
    stats["avg_confidence"] = safe_stats(confidence_list)

    return stats


def run_experiment():
    """Run the theory context experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    conditions = [ExperimentCondition(theory_context=tc) for tc in THEORY_CONDITIONS]
    total_games = len(conditions) * len(SEEDS)

    print(f"\n{'='*70}")
    print("Experiment: Theoretical Context for Objective Inference")
    print(f"{'='*70}")
    print(f"Theory contexts: {THEORY_CONDITIONS}")
    print(f"Seeds per condition: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    print(f"Complexity: L3 (3 properties)")
    print(f"CoT Access: False (testing theory on observable behavior)")
    print(f"Deception Strategy: baseline (isolating theory context effect)")
    print(f"{'='*70}\n")

    # Initialize wandb
    wandb_run = wandb.init(
        project="truthification",
        name=f"theory-context-experiment-{timestamp}",
        config={
            "experiment": "theory_context",
            "theory_conditions": THEORY_CONDITIONS,
            "seeds": SEEDS,
            "total_games": total_games,
            **BASE_CONFIG,
        },
    )

    # Output directories
    output_dir = Path("outputs/theory_context_experiment") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/theory_context_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    all_results = []
    condition_results = {tc: [] for tc in THEORY_CONDITIONS}

    game_count = 0
    start_time = time.time()

    for condition in conditions:
        print(f"\n--- Theory Context: {condition.theory_context} ---")

        for seed in SEEDS:
            game_count += 1
            elapsed = time.time() - start_time
            eta = (elapsed / game_count) * (total_games - game_count) if game_count > 0 else 0

            print(f"  [{game_count}/{total_games}] seed={seed} (ETA: {eta/60:.1f}m)...", end=" ", flush=True)

            try:
                game_start = time.time()
                result = run_single_game(condition, seed)
                game_elapsed = time.time() - game_start

                all_results.append(result)
                condition_results[condition.theory_context].append(result)

                # Save individual game result
                game_file = output_dir / f"game_{condition.theory_context}_seed{seed}.json"
                with open(game_file, "w") as f:
                    json.dump(result, f, indent=2)

                # Quick summary
                obj_score = result.get("agent_objective_overall_score", 0)
                print(f"done ({game_elapsed:.0f}s) - F1: {obj_score*100:.1f}%")

                # Log to wandb
                wandb.log({
                    "game_number": game_count,
                    "theory_context": condition.theory_context,
                    "seed": seed,
                    "game_time_seconds": game_elapsed,
                    "objective_inference_score": obj_score,
                })

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    total_elapsed = time.time() - start_time

    # Compute aggregate statistics
    print(f"\n{'='*70}")
    print("Computing Aggregate Statistics...")
    print(f"{'='*70}")

    condition_stats = {}
    for tc in THEORY_CONDITIONS:
        results = condition_results.get(tc, [])
        if results:
            condition_stats[tc] = {
                "theory_context": tc,
                "n_games": len(results),
                "stats": compute_condition_stats(results),
            }

    # Print summary
    print_summary(condition_stats)

    # Statistical comparison
    print_statistical_comparison(condition_results)

    # Save results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(output_dir / "condition_stats.json", "w") as f:
        json.dump(condition_stats, f, indent=2, default=str)

    # Create summary markdown
    summary_md = create_summary_markdown(condition_stats, condition_results, total_elapsed)
    with open(output_dir / "README.md", "w") as f:
        f.write(summary_md)

    # Also save to results directory
    with open(results_dir / f"condition_stats_{timestamp}.json", "w") as f:
        json.dump(condition_stats, f, indent=2, default=str)
    with open(results_dir / f"README_{timestamp}.md", "w") as f:
        f.write(summary_md)

    # Log summary to wandb
    summary_rows = []
    for tc in THEORY_CONDITIONS:
        if tc not in condition_stats:
            continue
        data = condition_stats[tc]
        stats = data["stats"]
        summary_rows.append([
            tc,
            data["n_games"],
            stats.get("exact_f1", {}).get("mean", 0),
            stats.get("exact_f1", {}).get("std", 0),
            stats.get("property_recall", {}).get("mean", 0),
            stats.get("avg_confidence", {}).get("mean", 0),
        ])

    wandb.log({
        "summary_table": wandb.Table(
            columns=["theory_context", "n_games", "exact_f1_mean", "exact_f1_std",
                     "property_recall", "avg_confidence"],
            data=summary_rows,
        ),
        "total_runtime_minutes": total_elapsed / 60,
    })

    wandb.finish()

    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")

    return condition_stats


def print_summary(condition_stats: dict):
    """Print summary tables to console."""
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Context':<12} | {'Exact F1':<18} | {'Prop Recall':<18} | {'Confidence':<12}")
    print("-" * 70)

    for tc in THEORY_CONDITIONS:
        if tc not in condition_stats:
            continue
        stats = condition_stats[tc]["stats"]
        f1 = stats.get("exact_f1", {})
        pr = stats.get("property_recall", {})
        conf = stats.get("avg_confidence", {})

        print(f"{tc:<12} | {f1.get('mean', 0)*100:>6.1f}% +/- {f1.get('std', 0)*100:>5.1f}% | "
              f"{pr.get('mean', 0)*100:>6.1f}% +/- {pr.get('std', 0)*100:>5.1f}% | "
              f"{conf.get('mean', 0):>5.1f}")


def print_statistical_comparison(condition_results: dict):
    """Print statistical comparison between theory contexts."""
    print("\n### Statistical Comparison (vs None/Baseline) ###")

    baseline_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("none", [])
        if r.get("agent_objective_overall_score") is not None
    ]

    if not baseline_scores:
        print("No baseline results to compare")
        return

    baseline_mean = statistics.mean(baseline_scores)
    baseline_std = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0
    print(f"\nBaseline (none) mean: {baseline_mean*100:.1f}% +/- {baseline_std*100:.1f}%")

    for tc in THEORY_CONDITIONS:
        if tc == "none":
            continue

        tc_scores = [
            r.get("agent_objective_overall_score", 0)
            for r in condition_results.get(tc, [])
            if r.get("agent_objective_overall_score") is not None
        ]

        if not tc_scores:
            continue

        tc_mean = statistics.mean(tc_scores)
        diff = tc_mean - baseline_mean

        print(f"\n{tc}:")
        print(f"  Mean: {tc_mean*100:.1f}%")
        print(f"  Diff from baseline: {diff*100:+.1f}%")

        # Effect size (Cohen's d)
        if len(baseline_scores) > 1 and len(tc_scores) > 1:
            pooled_std = ((baseline_std**2 + statistics.stdev(tc_scores)**2) / 2) ** 0.5
            if pooled_std > 0:
                cohens_d = diff / pooled_std
                print(f"  Cohen's d: {cohens_d:.2f}")

        # t-test
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(tc_scores, baseline_scores)
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  ** SIGNIFICANT (p < 0.05) **")
        except ImportError:
            pass


def create_summary_markdown(condition_stats: dict, condition_results: dict, total_elapsed: float) -> str:
    """Create markdown summary of results."""
    lines = [
        "# Experiment: Theoretical Context for Objective Inference",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Total Games**: {sum(d['n_games'] for d in condition_stats.values())}",
        "",
        "## Research Question",
        "",
        "Does providing the estimator with theoretical knowledge about learning from",
        "incomplete/strategic information improve its ability to infer agent objectives?",
        "",
        "## Theoretical Background",
        "",
        "Key insights from the literature:",
        "",
        "1. **Crawford-Sobel Partial Revelation**: Agents cannot fully hide their preferences",
        "   through strategic communication. Aggregate patterns reveal true objectives.",
        "",
        "2. **IRL Preference Leakage**: Even with deceptive demonstrations, preference",
        "   orderings leak through behavior patterns.",
        "",
        "3. **Bias Correction**: Multiple observations enable bias estimation; known bias",
        "   structure can be inverted to recover true signal.",
        "",
        "## Conditions Tested",
        "",
        "| Condition | Description |",
        "|-----------|-------------|",
        "| none | Standard principled inference (baseline) |",
        "| brief | 2-3 sentence summary of key theoretical insights |",
        "| full | Full theoretical framework (~200 words) |",
        "",
        "## Experimental Setup",
        "",
        "- **Complexity**: L3 (3 properties per agent)",
        "- **CoT Access**: False (testing theory on observable behavior only)",
        "- **Deception Strategy**: baseline (isolating theory context effect)",
        "- **Seeds per condition**: 10",
        "",
        "## Results",
        "",
        "| Context | Exact F1 (mean) | Exact F1 (std) | Property Recall | Confidence |",
        "|---------|-----------------|----------------|-----------------|------------|",
    ]

    for tc in THEORY_CONDITIONS:
        if tc not in condition_stats:
            continue
        stats = condition_stats[tc]["stats"]
        f1 = stats.get("exact_f1", {})
        pr = stats.get("property_recall", {})
        conf = stats.get("avg_confidence", {})

        lines.append(f"| {tc} | {f1.get('mean', 0)*100:.1f}% | {f1.get('std', 0)*100:.1f}% | "
                     f"{pr.get('mean', 0)*100:.1f}% | {conf.get('mean', 0):.1f} |")

    # Statistical comparison vs baseline
    baseline_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("none", [])
        if r.get("agent_objective_overall_score") is not None
    ]

    lines.extend([
        "",
        "## Statistical Comparison vs Baseline (none)",
        "",
    ])

    if baseline_scores:
        baseline_mean = statistics.mean(baseline_scores)
        lines.append(f"**Baseline (none) mean**: {baseline_mean*100:.1f}%")
        lines.append("")

        for tc in THEORY_CONDITIONS:
            if tc == "none":
                continue

            tc_scores = [
                r.get("agent_objective_overall_score", 0)
                for r in condition_results.get(tc, [])
                if r.get("agent_objective_overall_score") is not None
            ]

            if tc_scores:
                tc_mean = statistics.mean(tc_scores)
                diff = tc_mean - baseline_mean
                lines.append(f"- **{tc}**: {tc_mean*100:.1f}% ({diff*100:+.1f}% vs baseline)")

    lines.extend([
        "",
        "## Expected Outcomes",
        "",
        "| Outcome | Interpretation |",
        "|---------|----------------|",
        "| Theory helps significantly | Theoretical framing aids inference |",
        "| Brief = Full | Concise context sufficient |",
        "| Full > Brief | More detail helps |",
        "| No improvement | Model already implicitly understands, or theory not actionable |",
        "",
        "## Key Findings",
        "",
        "(Analysis to be added after reviewing results)",
        "",
        "## Interpretation",
        "",
        "(Discussion to be added)",
        "",
        "## Interaction with Other Experiments",
        "",
        "This experiment is **orthogonal** to:",
        "- **CoT access**: Tests whether seeing agent thinking helps (already showed +35%)",
        "- **Deception strategies**: Tests tactical guidance (consistency, incentive, pattern)",
        "",
        "This experiment tests **theoretical framing** - giving the model a mental model of",
        "how strategic communication works, rather than specific detection tactics.",
        "",
        "Could later test **combinations**: theory context + deception strategies + CoT access.",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment()
