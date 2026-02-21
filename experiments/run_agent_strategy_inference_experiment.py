#!/usr/bin/env python3
"""Experiment: Agent Communication Strategy Effect on Objective Inference

Research Question:
How does the communication strategy an agent uses affect how easily an external
observer can infer their true objectives?

Hypothesis:
Different strategies will have different "transparency" levels:
- Honest/aggressive strategies should make objectives MORE transparent
- Deceptive/misdirection strategies should make objectives LESS transparent
- Subtle strategies may be intermediate

Strategies Tested:
1. natural: No guidance, agent chooses naturally (baseline)
2. honest: Truthfully state preferences, directly advocate
3. deceptive: Actively hide true preferences, use false reasoning
4. misdirection: Emphasize irrelevant properties as distractions
5. aggressive: Strongly push preferred objects by ID
6. subtle: Indirectly promote interests through implications
7. credibility_attack: Focus on undermining opponent's credibility

Design:
- 7 strategies
- Fixed complexity level (L3 = 3 properties)
- 10 seeds per condition = 70 games total
- Principled evaluation (overlap scoring)
- NO CoT access (test on observable behavior only)

Metrics:
- Exact F1 (primary) - how well can estimator infer objectives?
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

# Agent communication strategies to test
STRATEGIES = [
    "natural",           # Baseline - no guidance
    "honest",            # Direct and truthful
    "deceptive",         # Hide true preferences
    "misdirection",      # Emphasize irrelevant properties
    "aggressive",        # Strongly push preferred objects
    "subtle",            # Indirectly promote interests
    "credibility_attack", # Attack opponent's credibility
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
    # Estimator does NOT see agent thinking (test on observable only)
    "estimator_sees_agent_thinking": False,
    # Estimator settings
    "enable_estimator_thinking": True,
    "estimator_thinking_budget": 5000,
    "estimator_deception_strategy": "baseline",  # Keep estimator baseline
    "estimator_theory_context": "none",  # No theory context
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
    """A single experimental condition (agent strategy)."""
    strategy: str

    def to_dict(self) -> dict:
        return {"agent_communication_strategy": self.strategy}


def run_single_game(condition: ExperimentCondition, seed: int) -> dict:
    """Run a single game with the given agent strategy and seed."""
    config = GameConfig(
        **BASE_CONFIG,
        agent_communication_strategy=condition.strategy,
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

    # Extract sample statements for qualitative analysis
    sample_statements = []
    for rnd in result.rounds[:3]:  # First 3 rounds
        for stmt in rnd.get("agent_statements", [])[:4]:  # First 4 statements
            sample_statements.append({
                "round": rnd.get("round_number"),
                "agent": stmt.get("agent_id"),
                "text": stmt.get("text"),
            })

    return {
        "condition": condition.to_dict(),
        "seed": seed,
        "metrics": result.metrics,
        "agent_objective_inference": result.agent_objective_inference,
        "agent_objective_scores": result.agent_objective_scores,
        "agent_objective_overall_score": result.agent_objective_overall_score,
        "agent_value_functions": agent_vf_details,
        "sample_statements": sample_statements,
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
    """Run the agent communication strategy experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    conditions = [ExperimentCondition(strategy=s) for s in STRATEGIES]
    total_games = len(conditions) * len(SEEDS)

    print(f"\n{'='*70}")
    print("Experiment: Agent Communication Strategy Effect on Objective Inference")
    print(f"{'='*70}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Seeds per strategy: {len(SEEDS)}")
    print(f"Total games: {total_games}")
    print(f"Complexity: L3 (3 properties)")
    print(f"CoT Access: False (testing on observable behavior)")
    print(f"{'='*70}\n")

    # Initialize wandb
    wandb_run = wandb.init(
        project="truthification",
        name=f"agent-strategy-inference-{timestamp}",
        config={
            "experiment": "agent_strategy_inference",
            "strategies": STRATEGIES,
            "seeds": SEEDS,
            "total_games": total_games,
            **BASE_CONFIG,
        },
    )

    # Output directories
    output_dir = Path("outputs/agent_strategy_inference") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/agent_strategy_inference")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Track results
    all_results = []
    condition_results = {s: [] for s in STRATEGIES}

    game_count = 0
    start_time = time.time()

    for condition in conditions:
        print(f"\n--- Agent Strategy: {condition.strategy} ---")

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
                condition_results[condition.strategy].append(result)

                # Save individual game result
                game_file = output_dir / f"game_{condition.strategy}_seed{seed}.json"
                with open(game_file, "w") as f:
                    json.dump(result, f, indent=2)

                # Quick summary
                obj_score = result.get("agent_objective_overall_score", 0)
                print(f"done ({game_elapsed:.0f}s) - F1: {obj_score*100:.1f}%")

                # Log to wandb
                wandb.log({
                    "game_number": game_count,
                    "agent_strategy": condition.strategy,
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
    for strategy in STRATEGIES:
        results = condition_results.get(strategy, [])
        if results:
            condition_stats[strategy] = {
                "strategy": strategy,
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
    for strategy in STRATEGIES:
        if strategy not in condition_stats:
            continue
        data = condition_stats[strategy]
        stats = data["stats"]
        summary_rows.append([
            strategy,
            data["n_games"],
            stats.get("exact_f1", {}).get("mean", 0),
            stats.get("exact_f1", {}).get("std", 0),
            stats.get("property_recall", {}).get("mean", 0),
            stats.get("avg_confidence", {}).get("mean", 0),
        ])

    wandb.log({
        "summary_table": wandb.Table(
            columns=["strategy", "n_games", "exact_f1_mean", "exact_f1_std",
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
    print("RESULTS SUMMARY: How Transparent Are Different Agent Strategies?")
    print("="*70)
    print("(Higher F1 = easier to infer agent's true objectives)")

    print(f"\n{'Strategy':<18} | {'Exact F1':<18} | {'Prop Recall':<18} | {'Confidence':<12}")
    print("-" * 75)

    # Sort by F1 for easier interpretation
    sorted_strategies = sorted(
        condition_stats.items(),
        key=lambda x: x[1]["stats"].get("exact_f1", {}).get("mean", 0),
        reverse=True
    )

    for strategy, data in sorted_strategies:
        stats = data["stats"]
        f1 = stats.get("exact_f1", {})
        pr = stats.get("property_recall", {})
        conf = stats.get("avg_confidence", {})

        print(f"{strategy:<18} | {f1.get('mean', 0)*100:>6.1f}% +/- {f1.get('std', 0)*100:>5.1f}% | "
              f"{pr.get('mean', 0)*100:>6.1f}% +/- {pr.get('std', 0)*100:>5.1f}% | "
              f"{conf.get('mean', 0):>5.1f}")


def print_statistical_comparison(condition_results: dict):
    """Print statistical comparison between strategies."""
    print("\n### Statistical Comparison (vs Natural Baseline) ###")

    baseline_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("natural", [])
        if r.get("agent_objective_overall_score") is not None
    ]

    if not baseline_scores:
        print("No baseline results to compare")
        return

    baseline_mean = statistics.mean(baseline_scores)
    baseline_std = statistics.stdev(baseline_scores) if len(baseline_scores) > 1 else 0
    print(f"\nBaseline (natural) mean: {baseline_mean*100:.1f}% +/- {baseline_std*100:.1f}%")

    for strategy in STRATEGIES:
        if strategy == "natural":
            continue

        strategy_scores = [
            r.get("agent_objective_overall_score", 0)
            for r in condition_results.get(strategy, [])
            if r.get("agent_objective_overall_score") is not None
        ]

        if not strategy_scores:
            continue

        strategy_mean = statistics.mean(strategy_scores)
        diff = strategy_mean - baseline_mean

        print(f"\n{strategy}:")
        print(f"  Mean: {strategy_mean*100:.1f}%")
        print(f"  Diff from baseline: {diff*100:+.1f}%")

        # Effect size (Cohen's d)
        if len(baseline_scores) > 1 and len(strategy_scores) > 1:
            pooled_std = ((baseline_std**2 + statistics.stdev(strategy_scores)**2) / 2) ** 0.5
            if pooled_std > 0:
                cohens_d = diff / pooled_std
                print(f"  Cohen's d: {cohens_d:.2f}")

        # t-test
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(strategy_scores, baseline_scores)
            print(f"  p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"  ** SIGNIFICANT (p < 0.05) **")
        except ImportError:
            pass


def create_summary_markdown(condition_stats: dict, condition_results: dict, total_elapsed: float) -> str:
    """Create markdown summary of results."""
    lines = [
        "# Experiment: Agent Communication Strategy Effect on Objective Inference",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_elapsed/60:.1f} minutes",
        f"**Total Games**: {sum(d['n_games'] for d in condition_stats.values())}",
        "",
        "## Research Question",
        "",
        "How does the communication strategy an agent uses affect how easily an external",
        "observer can infer their true objectives?",
        "",
        "## Hypothesis",
        "",
        "Different strategies will have different 'transparency' levels:",
        "- Honest/aggressive strategies should make objectives MORE transparent",
        "- Deceptive/misdirection strategies should make objectives LESS transparent",
        "- Subtle strategies may be intermediate",
        "",
        "## Strategies Tested",
        "",
        "| Strategy | Description |",
        "|----------|-------------|",
        "| natural | No guidance - agent chooses naturally (baseline) |",
        "| honest | Be direct and truthful about preferences |",
        "| deceptive | Actively hide true preferences |",
        "| misdirection | Emphasize irrelevant properties as distractions |",
        "| aggressive | Strongly push preferred objects by ID |",
        "| subtle | Indirectly promote interests through implications |",
        "| credibility_attack | Focus on undermining opponent's credibility |",
        "",
        "## Experimental Setup",
        "",
        "- **Complexity**: L3 (3 properties per agent)",
        "- **CoT Access**: False (testing on observable behavior only)",
        "- **Estimator Strategy**: baseline (no deception detection)",
        "- **Seeds per condition**: 10",
        "",
        "## Results",
        "",
        "| Strategy | Exact F1 (mean) | Exact F1 (std) | Property Recall | Confidence |",
        "|----------|-----------------|----------------|-----------------|------------|",
    ]

    # Sort by F1 for easier interpretation
    sorted_strategies = sorted(
        condition_stats.items(),
        key=lambda x: x[1]["stats"].get("exact_f1", {}).get("mean", 0),
        reverse=True
    )

    for strategy, data in sorted_strategies:
        stats = data["stats"]
        f1 = stats.get("exact_f1", {})
        pr = stats.get("property_recall", {})
        conf = stats.get("avg_confidence", {})

        lines.append(f"| {strategy} | {f1.get('mean', 0)*100:.1f}% | {f1.get('std', 0)*100:.1f}% | "
                     f"{pr.get('mean', 0)*100:.1f}% | {conf.get('mean', 0):.1f} |")

    # Statistical comparison vs baseline
    baseline_scores = [
        r.get("agent_objective_overall_score", 0)
        for r in condition_results.get("natural", [])
        if r.get("agent_objective_overall_score") is not None
    ]

    lines.extend([
        "",
        "## Statistical Comparison vs Baseline (natural)",
        "",
    ])

    if baseline_scores:
        baseline_mean = statistics.mean(baseline_scores)
        lines.append(f"**Baseline (natural) mean**: {baseline_mean*100:.1f}%")
        lines.append("")

        for strategy in STRATEGIES:
            if strategy == "natural":
                continue

            strategy_scores = [
                r.get("agent_objective_overall_score", 0)
                for r in condition_results.get(strategy, [])
                if r.get("agent_objective_overall_score") is not None
            ]

            if strategy_scores:
                strategy_mean = statistics.mean(strategy_scores)
                diff = strategy_mean - baseline_mean
                lines.append(f"- **{strategy}**: {strategy_mean*100:.1f}% ({diff*100:+.1f}% vs baseline)")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "### Transparency Ranking",
        "",
        "(Strategies ranked by how easily objectives can be inferred)",
        "",
    ])

    # Add ranking
    for i, (strategy, data) in enumerate(sorted_strategies, 1):
        f1_mean = data["stats"].get("exact_f1", {}).get("mean", 0)
        lines.append(f"{i}. **{strategy}**: {f1_mean*100:.1f}% F1")

    lines.extend([
        "",
        "### Key Findings",
        "",
        "(Analysis to be added after reviewing results)",
        "",
        "## Implications",
        "",
        "If certain strategies make objectives more transparent:",
        "- We can predict which agent types are easier to understand",
        "- We may be able to design estimator prompts that exploit these patterns",
        "- Understanding the relationship between strategy and transparency",
        "  could inform how we think about AI deception detection",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment()
