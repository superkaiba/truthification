#!/usr/bin/env python3
"""Experiment: Enhanced Estimator Context

Research Question:
Does giving the estimator even more context improve objective inference beyond
the current "full" theory context (~200 words)?

Background:
From previous experiments:
- Theory context helps: Full theory (+16.7%, d=0.85) outperformed brief (+10.0%) and none
- More detail is better: Monotonic improvement with more context (none -> brief -> full)
- Current "full" context is ~200 words

This suggests the estimator benefits from understanding strategic communication.
Can we push further?

Hypotheses:
1. Strategy awareness helps: Knowing the 6 agent strategies helps detect and invert them
2. Comprehensive > Full: The ~5000 word framework provides more actionable guidance than ~200 words
3. Strategy list vs comprehensive: Different types of context (practical vs theoretical) may differ
4. Diminishing returns: At some point, more context may not help (or hurt via cognitive overload)
5. Comprehensive may be too long: The model might not effectively utilize all 5000 words

Conditions:
1. none: No theory context (baseline)
2. full: Existing full theory (~200 words)
3. strategy_list: List of agent strategies (~250 words)
4. comprehensive: Extensive theory + mechanisms (~5000 words)

Design:
- 4 conditions (theory context levels)
- Fixed complexity level (L3 = 3 properties)
- 10 seeds per condition = 40 games total
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

# Theory context conditions to test (including new enhanced conditions)
THEORY_CONDITIONS = [
    "none",           # No theory context (baseline)
    "full",           # Existing full theory (~200 words)
    "strategy_list",  # List of agent strategies (~250 words)
    "comprehensive",  # Extensive theory + mechanisms (~5000 words)
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
            return {"mean": 0, "std": 0, "stderr": 0, "n": 0}
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0
        stderr = std / (len(values) ** 0.5) if len(values) > 1 else 0
        return {"mean": mean, "std": std, "stderr": stderr, "n": len(values)}

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


def find_latest_run_dir(base_dir: Path) -> Path | None:
    """Find the most recent run directory for resuming."""
    if not base_dir.exists():
        return None
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        return None
    return sorted(subdirs)[-1]


def load_existing_results(output_dir: Path) -> tuple[list[dict], dict[str, list[dict]], set[tuple[str, int]]]:
    """Load existing results from a previous run."""
    all_results = []
    condition_results = {tc: [] for tc in THEORY_CONDITIONS}
    completed_games = set()

    if not output_dir.exists():
        return all_results, condition_results, completed_games

    for game_file in output_dir.glob("game_*.json"):
        try:
            with open(game_file) as f:
                result = json.load(f)
            condition = result.get("condition", {})
            theory_context = condition.get("theory_context", "")
            seed = result.get("seed")
            if theory_context and seed is not None:
                all_results.append(result)
                if theory_context in condition_results:
                    condition_results[theory_context].append(result)
                completed_games.add((theory_context, seed))
        except Exception as e:
            print(f"Warning: Failed to load {game_file}: {e}")

    return all_results, condition_results, completed_games


def run_experiment(resume_dir: str | None = None):
    """Run the enhanced context experiment.

    Args:
        resume_dir: Optional path to previous run directory to resume from.
    """
    base_output_dir = Path("outputs/enhanced_context_experiment")

    if resume_dir:
        output_dir = Path(resume_dir)
        timestamp = output_dir.name
        print(f"Resuming from: {output_dir}")
    else:
        latest_dir = find_latest_run_dir(base_output_dir)
        if latest_dir:
            completed = list(latest_dir.glob("game_*.json"))
            total_expected = len(THEORY_CONDITIONS) * len(SEEDS)
            if len(completed) < total_expected:
                print(f"Found incomplete run at {latest_dir}")
                print(f"  Completed: {len(completed)}/{total_expected} games")
                response = input("Resume this run? [Y/n]: ").strip().lower()
                if response != 'n':
                    output_dir = latest_dir
                    timestamp = latest_dir.name
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_dir = base_output_dir / timestamp
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = base_output_dir / timestamp
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = base_output_dir / timestamp

    conditions = [ExperimentCondition(theory_context=tc) for tc in THEORY_CONDITIONS]
    total_games = len(conditions) * len(SEEDS)

    all_results, condition_results, completed_games = load_existing_results(output_dir)
    n_completed = len(completed_games)

    print(f"\n{'='*70}")
    print("Experiment: Enhanced Estimator Context")
    print(f"{'='*70}")
    print(f"Theory contexts: {THEORY_CONDITIONS}")
    print(f"Seeds per condition: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    if n_completed > 0:
        print(f"Already completed: {n_completed} games (resuming)")
    print(f"Complexity: L3 (3 properties)")
    print(f"CoT Access: False (testing theory on observable behavior)")
    print(f"Deception Strategy: baseline (isolating theory context effect)")
    print(f"{'='*70}\n")

    wandb_run = wandb.init(
        project="truthification",
        name=f"enhanced-context-experiment-{timestamp}",
        config={
            "experiment": "enhanced_context",
            "theory_conditions": THEORY_CONDITIONS,
            "seeds": SEEDS,
            "total_games": total_games,
            **BASE_CONFIG,
        },
        resume="allow",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("results/enhanced_context_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    game_count = n_completed
    start_time = time.time()

    for condition in conditions:
        print(f"\n--- Theory Context: {condition.theory_context} ---")

        for seed in SEEDS:
            game_count += 1

            if (condition.theory_context, seed) in completed_games:
                print(f"  [{game_count}/{total_games}] seed={seed} - SKIPPED (already done)")
                continue

            elapsed = time.time() - start_time
            games_run = game_count - n_completed
            eta = (elapsed / games_run) * (total_games - game_count) if games_run > 0 else 0

            print(f"  [{game_count}/{total_games}] seed={seed} (ETA: {eta/60:.1f}m)...", end=" ", flush=True)

            try:
                game_start = time.time()
                result = run_single_game(condition, seed)
                game_elapsed = time.time() - game_start

                all_results.append(result)
                condition_results[condition.theory_context].append(result)

                game_file = output_dir / f"game_{condition.theory_context}_seed{seed}.json"
                with open(game_file, "w") as f:
                    json.dump(result, f, indent=2)

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
            stats.get("exact_f1", {}).get("stderr", 0),
            stats.get("property_recall", {}).get("mean", 0),
            stats.get("avg_confidence", {}).get("mean", 0),
        ])

    wandb.log({
        "summary_table": wandb.Table(
            columns=["theory_context", "n_games", "exact_f1_mean", "exact_f1_std",
                     "exact_f1_stderr", "property_recall", "avg_confidence"],
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
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    print(f"\n{'Context':<16} | {'Exact F1':<22} | {'Prop Recall':<18} | {'Confidence':<12}")
    print("-" * 80)

    for tc in THEORY_CONDITIONS:
        if tc not in condition_stats:
            continue
        stats = condition_stats[tc]["stats"]
        f1 = stats.get("exact_f1", {})
        pr = stats.get("property_recall", {})
        conf = stats.get("avg_confidence", {})

        print(f"{tc:<16} | {f1.get('mean', 0)*100:>6.1f}% +/- {f1.get('stderr', 0)*100:>5.1f}% SE | "
              f"{pr.get('mean', 0)*100:>6.1f}% +/- {pr.get('stderr', 0)*100:>5.1f}% | "
              f"{conf.get('mean', 0):>5.1f}")


def print_statistical_comparison(condition_results: dict):
    """Print statistical comparison between theory contexts."""
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON")
    print("="*80)

    # Get baseline (none) scores
    baseline_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("none", [])
        if r.get("agent_objective_overall_score") is not None
    ]

    # Get full context scores (existing best)
    full_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("full", [])
        if r.get("agent_objective_overall_score") is not None
    ]

    if not baseline_scores:
        print("\nNo baseline results to compare")
        return

    baseline_mean = statistics.mean(baseline_scores)
    baseline_std = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0
    print(f"\n### Baseline (none) ###")
    print(f"Mean: {baseline_mean*100:.1f}% +/- {baseline_std*100:.1f}% (std)")

    if full_scores:
        full_mean = statistics.mean(full_scores)
        full_std = statistics.stdev(full_scores) if len(full_scores) > 1 else 0
        print(f"\n### Full context (existing best) ###")
        print(f"Mean: {full_mean*100:.1f}% +/- {full_std*100:.1f}% (std)")

    print("\n### Comparison vs Baseline (none) ###")
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
        tc_std = statistics.stdev(tc_scores) if len(tc_scores) > 1 else 0
        diff_baseline = tc_mean - baseline_mean

        print(f"\n**{tc}**:")
        print(f"  Mean: {tc_mean*100:.1f}%")
        print(f"  Diff from baseline: {diff_baseline*100:+.1f}%")

        # Effect size (Cohen's d) vs baseline
        if len(baseline_scores) > 1 and len(tc_scores) > 1:
            pooled_std = ((baseline_std**2 + tc_std**2) / 2) ** 0.5
            if pooled_std > 0:
                cohens_d = diff_baseline / pooled_std
                print(f"  Cohen's d (vs baseline): {cohens_d:.2f}")

        # t-test vs baseline
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(tc_scores, baseline_scores)
            print(f"  p-value (vs baseline): {p_value:.4f}")
            if p_value < 0.05:
                print(f"  ** SIGNIFICANT (p < 0.05) **")
        except ImportError:
            pass

    # Comparison vs full (for new conditions)
    if full_scores:
        full_mean = statistics.mean(full_scores)
        full_std = statistics.stdev(full_scores) if len(full_scores) > 1 else 0

        print("\n### Comparison vs Full (~200 words, existing best) ###")
        for tc in ["strategy_list", "comprehensive"]:
            tc_scores = [
                r.get("agent_objective_overall_score", 0)
                for r in condition_results.get(tc, [])
                if r.get("agent_objective_overall_score") is not None
            ]

            if not tc_scores:
                continue

            tc_mean = statistics.mean(tc_scores)
            tc_std = statistics.stdev(tc_scores) if len(tc_scores) > 1 else 0
            diff_full = tc_mean - full_mean

            print(f"\n**{tc}**:")
            print(f"  Mean: {tc_mean*100:.1f}%")
            print(f"  Diff from full: {diff_full*100:+.1f}%")

            # Effect size (Cohen's d) vs full
            if len(full_scores) > 1 and len(tc_scores) > 1:
                pooled_std = ((full_std**2 + tc_std**2) / 2) ** 0.5
                if pooled_std > 0:
                    cohens_d = diff_full / pooled_std
                    print(f"  Cohen's d (vs full): {cohens_d:.2f}")

            # t-test vs full
            try:
                from scipy import stats
                t_stat, p_value = stats.ttest_ind(tc_scores, full_scores)
                print(f"  p-value (vs full): {p_value:.4f}")
                if p_value < 0.05:
                    print(f"  ** SIGNIFICANT (p < 0.05) **")
            except ImportError:
                pass


def create_summary_markdown(condition_stats: dict, condition_results: dict, total_elapsed: float) -> str:
    """Create markdown summary of results."""
    lines = [
        "# Experiment: Enhanced Estimator Context",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Total Games**: {sum(d['n_games'] for d in condition_stats.values())}",
        "",
        "## Research Question",
        "",
        "Does giving the estimator even more context improve objective inference beyond",
        "the current 'full' theory context (~200 words)?",
        "",
        "## Hypotheses",
        "",
        "1. **Strategy awareness helps**: Knowing the 6 agent strategies helps detect and invert them",
        "2. **Comprehensive > Full**: The ~5000 word framework provides more actionable guidance",
        "3. **Strategy list vs comprehensive**: Different types of context may have different effects",
        "4. **Diminishing returns**: At some point, more context may not help (or hurt)",
        "5. **Comprehensive may be too long**: The model might not effectively utilize all 5000 words",
        "",
        "## Conditions Tested",
        "",
        "| Condition | Description | Approx. Words |",
        "|-----------|-------------|---------------|",
        "| none | No theory context (baseline) | 0 |",
        "| full | Existing full theory | ~200 |",
        "| strategy_list | List of agent strategies | ~250 |",
        "| comprehensive | Extensive theory + mechanisms | ~5000 |",
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
        "| Context | Exact F1 (mean) | Exact F1 (SE) | Property Recall | Confidence |",
        "|---------|-----------------|---------------|-----------------|------------|",
    ]

    for tc in THEORY_CONDITIONS:
        if tc not in condition_stats:
            continue
        stats = condition_stats[tc]["stats"]
        f1 = stats.get("exact_f1", {})
        pr = stats.get("property_recall", {})
        conf = stats.get("avg_confidence", {})

        lines.append(f"| {tc} | {f1.get('mean', 0)*100:.1f}% | {f1.get('stderr', 0)*100:.1f}% | "
                     f"{pr.get('mean', 0)*100:.1f}% | {conf.get('mean', 0):.1f} |")

    # Statistical comparison vs baseline
    baseline_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("none", [])
        if r.get("agent_objective_overall_score") is not None
    ]

    full_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("full", [])
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
                tc_std = statistics.stdev(tc_scores) if len(tc_scores) > 1 else 0
                diff = tc_mean - baseline_mean

                # Cohen's d
                baseline_std = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0
                pooled_std = ((baseline_std**2 + tc_std**2) / 2) ** 0.5
                cohens_d = diff / pooled_std if pooled_std > 0 else 0

                # p-value
                p_str = ""
                try:
                    from scipy import stats
                    _, p_value = stats.ttest_ind(tc_scores, baseline_scores)
                    p_str = f", p={p_value:.4f}"
                    if p_value < 0.05:
                        p_str += " *"
                except ImportError:
                    pass

                lines.append(f"- **{tc}**: {tc_mean*100:.1f}% ({diff*100:+.1f}% vs baseline, d={cohens_d:.2f}{p_str})")

    # Comparison vs full
    if full_scores:
        full_mean = statistics.mean(full_scores)
        lines.extend([
            "",
            "## Statistical Comparison vs Full (~200 words)",
            "",
            f"**Full context mean**: {full_mean*100:.1f}%",
            "",
        ])

        for tc in ["strategy_list", "comprehensive"]:
            tc_scores = [
                r.get("agent_objective_overall_score", 0)
                for r in condition_results.get(tc, [])
                if r.get("agent_objective_overall_score") is not None
            ]

            if tc_scores:
                tc_mean = statistics.mean(tc_scores)
                tc_std = statistics.stdev(tc_scores) if len(tc_scores) > 1 else 0
                diff = tc_mean - full_mean

                # Cohen's d
                full_std = statistics.stdev(full_scores) if len(full_scores) > 1 else 0
                pooled_std = ((full_std**2 + tc_std**2) / 2) ** 0.5
                cohens_d = diff / pooled_std if pooled_std > 0 else 0

                # p-value
                p_str = ""
                try:
                    from scipy import stats
                    _, p_value = stats.ttest_ind(tc_scores, full_scores)
                    p_str = f", p={p_value:.4f}"
                    if p_value < 0.05:
                        p_str += " *"
                except ImportError:
                    pass

                lines.append(f"- **{tc}**: {tc_mean*100:.1f}% ({diff*100:+.1f}% vs full, d={cohens_d:.2f}{p_str})")

    lines.extend([
        "",
        "## Key Findings",
        "",
        "(Analysis to be added after reviewing results)",
        "",
        "## Interpretation",
        "",
        "(Discussion to be added)",
        "",
        "## Context Length Analysis",
        "",
        "| Condition | Words | Relative to Full |",
        "|-----------|-------|------------------|",
        "| none | 0 | 0x |",
        "| full | ~200 | 1x |",
        "| strategy_list | ~250 | 1.25x |",
        "| comprehensive | ~5000 | 25x |",
        "",
        "This allows us to test:",
        "1. Does any context help? (none vs full)",
        "2. Does practical guidance help more than theory? (strategy_list vs full)",
        "3. Does more theoretical depth help? (comprehensive vs full)",
        "4. Is there diminishing returns? (check ordering)",
        "",
        "## Relation to Previous Experiments",
        "",
        "Previous theory context experiment found:",
        "- Full theory (+16.7%, d=0.85) outperformed brief (+10.0%) and none",
        "- Monotonic improvement: none < brief < full",
        "",
        "This experiment extends by testing:",
        "- Even more context (comprehensive)",
        "- Different type of context (strategy list vs theoretical)",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    resume_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run_experiment(resume_dir=resume_dir)
