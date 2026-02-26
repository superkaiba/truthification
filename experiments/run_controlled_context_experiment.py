#!/usr/bin/env python3
"""Experiment: Controlled Theory Context Comparison

This experiment isolates the effect of theory context by:
1. Running ONE game per seed and saving the full transcript (statements)
2. Re-running ONLY the estimator inference with different theory contexts on the SAME statements

This eliminates variance from different agent debates - the only variable is the estimator's context.

Design:
- Phase 1: Run 10 games (one per seed), save full game data with statements
- Phase 2: For each game, run estimator inference with 4 different contexts
- Total: 10 seeds × 4 contexts = 40 inference runs, but only 10 game runs

This is much faster AND more statistically rigorous than running 40 separate games.
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
from src.environment.estimator import Estimator
from src.environment.world import World, Property, PropertyType

# ============================================================================
# Experimental Configuration
# ============================================================================

SEEDS = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]

# Theory context conditions to test
THEORY_CONDITIONS = [
    "none",           # No theory context (baseline)
    "full",           # Existing full theory (~200 words)
    "strategy_list",  # List of agent strategies (~250 words)
    "comprehensive",  # Extensive theory + mechanisms (~5000 words)
]

# Fixed game parameters (for Phase 1)
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
    # Agent thinking enabled
    "enable_agent_thinking": True,
    "agent_thinking_budget": 5000,
    # Estimator does NOT see agent thinking
    "estimator_sees_agent_thinking": False,
    # Estimator thinking
    "enable_estimator_thinking": True,
    "estimator_thinking_budget": 5000,
    # Deception strategy: baseline
    "estimator_deception_strategy": "baseline",
    # Use "none" for initial game runs - we'll vary this in Phase 2
    "estimator_theory_context": "none",
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
class GameData:
    """Saved data from a game needed for re-running estimator."""
    seed: int
    statements: list[dict]  # All statements from the game
    agents: list[dict]  # Agent info including value functions
    world_state: dict  # World configuration
    property_definitions: list[dict]  # Property definitions


def run_phase1_game(seed: int, output_dir: Path) -> GameData:
    """Run a single game and save full data including statements."""
    config = GameConfig(**BASE_CONFIG, seed=seed)
    game = HiddenValueGame(config)
    result = game.run()

    # Extract statements from rounds
    # Each round is a GameRound object with .agent_statements as list of dicts
    all_statements = []
    rounds_data = []
    for round_obj in game.rounds:
        # round_obj is a GameRound dataclass
        for stmt_dict in round_obj.agent_statements:
            all_statements.append(stmt_dict)
        # Convert round to dict for saving
        rounds_data.append({
            "round_number": round_obj.round_number,
            "agent_statements": round_obj.agent_statements,
            "observer_action": round_obj.observer_action,
        })

    # Build property definitions from world
    prop_defs = []
    for p in game.world.property_definitions:
        prop_defs.append({
            "name": p.name,
            "possible_values": p.possible_values,
        })

    game_data = GameData(
        seed=seed,
        statements=all_statements,
        agents=result.agents,
        world_state=result.world_state,
        property_definitions=prop_defs,
    )

    # Save full game result for reference
    game_file = output_dir / f"game_seed{seed}_full.json"
    with open(game_file, "w") as f:
        json.dump({
            "seed": seed,
            "statements": all_statements,
            "agents": result.agents,
            "world_state": result.world_state,
            "property_definitions": prop_defs,
            "rounds": rounds_data,
            "metrics": result.metrics,
            "config": result.config,
            "agent_objective_inference": result.agent_objective_inference,
            "agent_objective_scores": result.agent_objective_scores,
            "agent_objective_overall_score": result.agent_objective_overall_score,
        }, f, indent=2, default=str)

    return game_data


def run_phase2_inference(game_data: GameData, theory_context: str) -> dict:
    """Run estimator inference on saved game data with specific theory context."""

    # Reconstruct the world
    prop_defs = []
    for p in game_data.property_definitions:
        # Infer property type from values
        values = p["possible_values"]
        if all(isinstance(v, bool) for v in values):
            prop_type = PropertyType.BOOLEAN
        elif all(isinstance(v, (int, float)) for v in values):
            prop_type = PropertyType.NUMERIC
        else:
            prop_type = PropertyType.CATEGORICAL
        prop_defs.append(Property(name=p["name"], property_type=prop_type, possible_values=values))
    world = World(property_definitions=prop_defs)

    # Add objects to world from world_state
    for obj_id, obj_data in game_data.world_state.get("objects", {}).items():
        world.add_object(obj_id, obj_data.get("properties", {}))

    # Create estimator with specified theory context
    estimator = Estimator(
        model="claude-sonnet-4-20250514",
        condition="ids",
        enable_thinking=True,
        thinking_budget=5000,
        sees_agent_thinking=False,
        deception_strategy="baseline",
        theory_context=theory_context,
    )

    # Convert statements back to Statement objects
    from src.environment.agent import Statement
    statements = [
        Statement(
            agent_id=s["agent_id"],
            text=s["text"],
            thinking=s.get("thinking"),
        )
        for s in game_data.statements
    ]

    # Run principled inference
    inferences = estimator.infer_agent_objectives_principled(
        all_statements=statements,
        agents=game_data.agents,
        world=world,
    )

    # Evaluate using overlap scoring
    eval_result = estimator.evaluate_objective_inference_overlap(
        inferences=inferences,
        agents=game_data.agents,
    )

    # Extract results
    inference_data = {}
    for agent_id, inf in inferences.items():
        inference_data[agent_id] = {
            "inferred_goal": inf.inferred_goal,
            "inferred_factors": inf.inferred_factors,
            "predicted_properties": inf.predicted_properties,
            "confidence": inf.confidence,
            "reasoning": inf.reasoning,
        }
        if eval_result.overlap_scores and agent_id in eval_result.overlap_scores:
            inference_data[agent_id]["overlap_metrics"] = eval_result.overlap_scores[agent_id].to_dict()

    return {
        "theory_context": theory_context,
        "seed": game_data.seed,
        "agent_objective_inference": inference_data,
        "agent_objective_scores": eval_result.evaluation_scores,
        "agent_objective_overall_score": eval_result.overall_score,
    }


def compute_paired_stats(results_by_context: dict[str, list[dict]]) -> dict:
    """Compute statistics using paired comparisons (same seed across conditions)."""
    stats = {}

    for tc, results in results_by_context.items():
        scores = [r["agent_objective_overall_score"] for r in results]
        if scores:
            mean = statistics.mean(scores)
            std = statistics.stdev(scores) if len(scores) > 1 else 0
            stderr = std / (len(scores) ** 0.5) if len(scores) > 1 else 0
            stats[tc] = {
                "mean": mean,
                "std": std,
                "stderr": stderr,
                "n": len(scores),
                "scores": scores,
            }

    return stats


def run_experiment():
    """Run the controlled context experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs/controlled_context_experiment") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path("results/controlled_context_experiment")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Experiment: Controlled Theory Context Comparison")
    print(f"{'='*70}")
    print(f"Phase 1: Run 10 games, save statements")
    print(f"Phase 2: Re-run estimator with 4 different contexts on same statements")
    print(f"Total inference runs: {len(SEEDS)} seeds × {len(THEORY_CONDITIONS)} contexts = {len(SEEDS) * len(THEORY_CONDITIONS)}")
    print(f"{'='*70}\n")

    wandb_run = wandb.init(
        project="truthification",
        name=f"controlled-context-experiment-{timestamp}",
        config={
            "experiment": "controlled_context",
            "theory_conditions": THEORY_CONDITIONS,
            "seeds": SEEDS,
            "design": "within-subjects (same debate, different contexts)",
            **BASE_CONFIG,
        },
    )

    start_time = time.time()

    # ========== PHASE 1: Run games and save statements ==========
    print("=" * 70)
    print("PHASE 1: Running games and saving statements")
    print("=" * 70)

    game_data_by_seed = {}

    for i, seed in enumerate(SEEDS):
        print(f"  [{i+1}/{len(SEEDS)}] Running game with seed={seed}...", end=" ", flush=True)
        game_start = time.time()

        try:
            game_data = run_phase1_game(seed, output_dir)
            game_data_by_seed[seed] = game_data
            game_elapsed = time.time() - game_start
            n_statements = len(game_data.statements)
            print(f"done ({game_elapsed:.0f}s) - {n_statements} statements")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    phase1_time = time.time() - start_time
    print(f"\nPhase 1 complete: {len(game_data_by_seed)} games in {phase1_time/60:.1f} minutes")

    # ========== PHASE 2: Run estimator with different contexts ==========
    print("\n" + "=" * 70)
    print("PHASE 2: Running estimator inference with different contexts")
    print("=" * 70)

    results_by_context = {tc: [] for tc in THEORY_CONDITIONS}
    all_results = []

    phase2_start = time.time()
    total_inferences = len(game_data_by_seed) * len(THEORY_CONDITIONS)
    inference_count = 0

    for seed, game_data in game_data_by_seed.items():
        print(f"\n  Seed {seed}:")

        for tc in THEORY_CONDITIONS:
            inference_count += 1
            print(f"    [{inference_count}/{total_inferences}] context={tc}...", end=" ", flush=True)
            inf_start = time.time()

            try:
                result = run_phase2_inference(game_data, tc)
                results_by_context[tc].append(result)
                all_results.append(result)

                inf_elapsed = time.time() - inf_start
                score = result["agent_objective_overall_score"]
                print(f"done ({inf_elapsed:.0f}s) - F1: {score*100:.1f}%")

                # Log to wandb
                wandb.log({
                    "inference_number": inference_count,
                    "seed": seed,
                    "theory_context": tc,
                    "inference_time_seconds": inf_elapsed,
                    "objective_inference_score": score,
                })

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    phase2_time = time.time() - phase2_start
    total_time = time.time() - start_time

    # ========== ANALYSIS ==========
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    stats = compute_paired_stats(results_by_context)

    # Print results table
    print(f"\n{'Context':<16} | {'Mean F1':<12} | {'Std':<12} | {'SE':<12}")
    print("-" * 60)
    for tc in THEORY_CONDITIONS:
        if tc in stats:
            s = stats[tc]
            print(f"{tc:<16} | {s['mean']*100:>6.1f}%     | {s['std']*100:>6.1f}%     | {s['stderr']*100:>6.1f}%")

    # Paired comparison vs baseline
    print("\n### Paired Comparison vs Baseline (none) ###")
    if "none" in stats:
        baseline_scores = stats["none"]["scores"]

        for tc in THEORY_CONDITIONS:
            if tc == "none" or tc not in stats:
                continue

            tc_scores = stats[tc]["scores"]

            # Paired differences (same seed)
            paired_diffs = [tc_scores[i] - baseline_scores[i] for i in range(len(baseline_scores))]
            mean_diff = statistics.mean(paired_diffs)
            std_diff = statistics.stdev(paired_diffs) if len(paired_diffs) > 1 else 0
            se_diff = std_diff / (len(paired_diffs) ** 0.5)

            print(f"\n{tc}:")
            print(f"  Mean difference: {mean_diff*100:+.1f}%")
            print(f"  SE of difference: {se_diff*100:.1f}%")

            # Paired t-test
            try:
                from scipy import stats as scipy_stats
                t_stat, p_value = scipy_stats.ttest_rel(tc_scores, baseline_scores)
                print(f"  Paired t-test p-value: {p_value:.4f}")
                if p_value < 0.05:
                    print(f"  ** SIGNIFICANT (p < 0.05) **")
            except ImportError:
                pass

            # Effect size (Cohen's d for paired data)
            if std_diff > 0:
                cohens_d = mean_diff / std_diff
                print(f"  Cohen's d (paired): {cohens_d:.2f}")

    # Save results
    summary = {
        "timestamp": timestamp,
        "phase1_time_minutes": phase1_time / 60,
        "phase2_time_minutes": phase2_time / 60,
        "total_time_minutes": total_time / 60,
        "n_seeds": len(game_data_by_seed),
        "n_conditions": len(THEORY_CONDITIONS),
        "stats": {tc: {k: v for k, v in s.items() if k != "scores"} for tc, s in stats.items()},
        "all_results": all_results,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Create markdown report
    report = create_report(stats, results_by_context, total_time)
    with open(output_dir / "README.md", "w") as f:
        f.write(report)
    with open(results_dir / f"README_{timestamp}.md", "w") as f:
        f.write(report)

    # Log final summary to wandb
    wandb.log({
        "phase1_time_minutes": phase1_time / 60,
        "phase2_time_minutes": phase2_time / 60,
        "total_time_minutes": total_time / 60,
    })

    wandb.finish()

    print(f"\nTotal runtime: {total_time/60:.1f} minutes")
    print(f"  Phase 1 (games): {phase1_time/60:.1f} minutes")
    print(f"  Phase 2 (inference): {phase2_time/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")

    return stats


def create_report(stats: dict, results_by_context: dict, total_time: float) -> str:
    """Create markdown report."""
    lines = [
        "# Controlled Theory Context Experiment",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Runtime**: {total_time/60:.1f} minutes",
        "",
        "## Design",
        "",
        "This experiment uses a **within-subjects design** to isolate theory context effects:",
        "",
        "1. **Phase 1**: Run 10 games, save full transcripts (agent statements)",
        "2. **Phase 2**: Re-run estimator on SAME statements with different contexts",
        "",
        "This eliminates variance from different agent debates - the only variable is the estimator's theory context.",
        "",
        "## Results",
        "",
        "| Context | Mean F1 | Std | SE |",
        "|---------|---------|-----|-----|",
    ]

    for tc in THEORY_CONDITIONS:
        if tc in stats:
            s = stats[tc]
            lines.append(f"| {tc} | {s['mean']*100:.1f}% | {s['std']*100:.1f}% | {s['stderr']*100:.1f}% |")

    # Paired comparisons
    lines.extend([
        "",
        "## Paired Statistical Comparison vs Baseline (none)",
        "",
    ])

    if "none" in stats:
        baseline_scores = stats["none"]["scores"]

        for tc in THEORY_CONDITIONS:
            if tc == "none" or tc not in stats:
                continue

            tc_scores = stats[tc]["scores"]
            paired_diffs = [tc_scores[i] - baseline_scores[i] for i in range(len(baseline_scores))]
            mean_diff = statistics.mean(paired_diffs)
            std_diff = statistics.stdev(paired_diffs) if len(paired_diffs) > 1 else 0

            # Effect size
            cohens_d = mean_diff / std_diff if std_diff > 0 else 0

            # p-value
            p_str = ""
            try:
                from scipy import stats as scipy_stats
                _, p_value = scipy_stats.ttest_rel(tc_scores, baseline_scores)
                p_str = f", p={p_value:.4f}"
                if p_value < 0.05:
                    p_str += " *"
            except ImportError:
                pass

            lines.append(f"- **{tc}**: {mean_diff*100:+.1f}% (d={cohens_d:.2f}{p_str})")

    lines.extend([
        "",
        "## Per-Seed Breakdown",
        "",
        "| Seed | none | full | strategy_list | comprehensive |",
        "|------|------|------|---------------|---------------|",
    ])

    # Build per-seed table
    seeds = list(set(r["seed"] for r in results_by_context.get("none", [])))
    seeds.sort()

    for seed in seeds:
        row = [str(seed)]
        for tc in THEORY_CONDITIONS:
            score = next((r["agent_objective_overall_score"] for r in results_by_context.get(tc, []) if r["seed"] == seed), None)
            if score is not None:
                row.append(f"{score*100:.0f}%")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "With the within-subjects design, any observed differences are due purely to the theory context,",
        "not to variance in agent behavior. This provides a cleaner test of whether theory helps.",
        "",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    run_experiment()
