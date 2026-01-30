#!/usr/bin/env python3
"""Run V2 hidden value game experiments."""

import json
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
from src.environment.simulation_v2 import GameConfig, GameResult, HiddenValueGame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_single_game(
    config: GameConfig,
    run_id: int,
) -> GameResult:
    """Run a single game with the given configuration."""
    game = HiddenValueGame(config)
    return game.run()


def run_oracle_scaling_experiment(
    cfg: DictConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> list[dict]:
    """
    Experiment 1: How much oracle access is needed to beat deception?

    Varies oracle budget from 0 to max while keeping other factors constant.
    """
    results = []
    oracle_budgets = cfg.oracle_scaling.budgets  # e.g., [0, 5, 10, 20]

    for budget in oracle_budgets:
        logger.info(f"Running oracle budget = {budget}")
        budget_results = []

        for seed in cfg.seeds:
            config = GameConfig(
                n_objects=cfg.world.n_objects,
                rule_complexity=cfg.world.rule_complexity,
                seed=seed,
                n_agents=cfg.agent.n_agents,
                agent_model=cfg.agent.model,
                observer_model=cfg.observer.model,
                oracle_budget=budget,
                selection_size=cfg.observer.selection_size,
                n_rounds=cfg.game.n_rounds,
                condition=cfg.game.condition,
            )

            result = run_single_game(config, seed)
            budget_results.append(result.metrics)

            # Log to wandb
            if wandb_run:
                wandb.log({
                    "experiment": "oracle_scaling",
                    "oracle_budget": budget,
                    "seed": seed,
                    # Core metrics
                    "selection_accuracy": result.metrics["selection_accuracy"],
                    "optimal_overlap_ratio": result.metrics["optimal_overlap_ratio"],
                    "total_value": result.metrics["total_value"],
                    "optimal_value": result.metrics["optimal_value"],
                    # Truth recovery metrics
                    "property_accuracy": result.metrics["property_accuracy"],
                    "rule_inference_accuracy": result.metrics["rule_inference_accuracy"],
                    "rule_confidence": result.metrics["rule_confidence"],
                    # Baseline comparisons
                    "random_selection_value": result.metrics["random_selection_value"],
                    "value_vs_random": result.metrics["value_vs_random"],
                    "value_vs_best_agent": result.metrics["value_vs_best_agent"],
                    "majority_vote_accuracy": result.metrics["majority_vote_accuracy"],
                })

        # Aggregate
        n = len(budget_results)
        avg_accuracy = sum(r["selection_accuracy"] for r in budget_results) / n
        avg_overlap = sum(r["optimal_overlap_ratio"] for r in budget_results) / n
        avg_prop_acc = sum(r["property_accuracy"] for r in budget_results) / n
        avg_rule_acc = sum(r["rule_inference_accuracy"] for r in budget_results) / n
        avg_vs_random = sum(r["value_vs_random"] for r in budget_results) / n

        results.append({
            "oracle_budget": budget,
            "avg_selection_accuracy": avg_accuracy,
            "avg_optimal_overlap": avg_overlap,
            "avg_property_accuracy": avg_prop_acc,
            "avg_rule_inference_accuracy": avg_rule_acc,
            "avg_value_vs_random": avg_vs_random,
            "n_runs": n,
            "raw_results": budget_results,
        })

        logger.info(f"  Accuracy: {avg_accuracy:.3f}, Prop acc: {avg_prop_acc:.3f}")

    return results


def run_information_conditions_experiment(
    cfg: DictConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> list[dict]:
    """
    Experiment 2: Effect of information conditions.

    Tests: blind, ids, interests
    """
    results = []
    conditions = cfg.information_conditions.conditions  # ["blind", "ids", "interests"]

    for condition in conditions:
        logger.info(f"Running condition = {condition}")
        condition_results = []

        for seed in cfg.seeds:
            config = GameConfig(
                n_objects=cfg.world.n_objects,
                rule_complexity=cfg.world.rule_complexity,
                seed=seed,
                n_agents=cfg.agent.n_agents,
                agent_model=cfg.agent.model,
                observer_model=cfg.observer.model,
                oracle_budget=cfg.observer.oracle_budget,
                selection_size=cfg.observer.selection_size,
                n_rounds=cfg.game.n_rounds,
                condition=condition,
            )

            result = run_single_game(config, seed)
            condition_results.append(result.metrics)

            if wandb_run:
                wandb.log({
                    "experiment": "information_conditions",
                    "condition": condition,
                    "seed": seed,
                    "selection_accuracy": result.metrics["selection_accuracy"],
                    "optimal_overlap_ratio": result.metrics["optimal_overlap_ratio"],
                    "property_accuracy": result.metrics["property_accuracy"],
                    "rule_inference_accuracy": result.metrics["rule_inference_accuracy"],
                    "value_vs_random": result.metrics["value_vs_random"],
                })

        n = len(condition_results)
        avg_accuracy = sum(r["selection_accuracy"] for r in condition_results) / n
        avg_overlap = sum(r["optimal_overlap_ratio"] for r in condition_results) / n
        avg_prop_acc = sum(r["property_accuracy"] for r in condition_results) / n
        avg_rule_acc = sum(r["rule_inference_accuracy"] for r in condition_results) / n

        results.append({
            "condition": condition,
            "avg_selection_accuracy": avg_accuracy,
            "avg_optimal_overlap": avg_overlap,
            "avg_property_accuracy": avg_prop_acc,
            "avg_rule_inference_accuracy": avg_rule_acc,
            "n_runs": n,
            "raw_results": condition_results,
        })

        logger.info(f"  Accuracy: {avg_accuracy:.3f}, Prop acc: {avg_prop_acc:.3f}")

    return results


def run_rule_complexity_experiment(
    cfg: DictConfig,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> list[dict]:
    """
    Experiment 3: Effect of value rule complexity.

    Tests: simple, medium, complex
    """
    results = []
    complexities = cfg.rule_complexity.levels  # ["simple", "medium", "complex"]

    for complexity in complexities:
        logger.info(f"Running complexity = {complexity}")
        complexity_results = []

        for seed in cfg.seeds:
            config = GameConfig(
                n_objects=cfg.world.n_objects,
                rule_complexity=complexity,
                seed=seed,
                n_agents=cfg.agent.n_agents,
                agent_model=cfg.agent.model,
                observer_model=cfg.observer.model,
                oracle_budget=cfg.observer.oracle_budget,
                selection_size=cfg.observer.selection_size,
                n_rounds=cfg.game.n_rounds,
                condition=cfg.game.condition,
            )

            result = run_single_game(config, seed)
            complexity_results.append(result.metrics)

            if wandb_run:
                wandb.log({
                    "experiment": "rule_complexity",
                    "rule_complexity": complexity,
                    "seed": seed,
                    "selection_accuracy": result.metrics["selection_accuracy"],
                    "optimal_overlap_ratio": result.metrics["optimal_overlap_ratio"],
                    "property_accuracy": result.metrics["property_accuracy"],
                    "rule_inference_accuracy": result.metrics["rule_inference_accuracy"],
                    "value_vs_random": result.metrics["value_vs_random"],
                })

        n = len(complexity_results)
        avg_accuracy = sum(r["selection_accuracy"] for r in complexity_results) / n
        avg_overlap = sum(r["optimal_overlap_ratio"] for r in complexity_results) / n
        avg_prop_acc = sum(r["property_accuracy"] for r in complexity_results) / n
        avg_rule_acc = sum(r["rule_inference_accuracy"] for r in complexity_results) / n

        results.append({
            "rule_complexity": complexity,
            "avg_selection_accuracy": avg_accuracy,
            "avg_optimal_overlap": avg_overlap,
            "avg_property_accuracy": avg_prop_acc,
            "avg_rule_inference_accuracy": avg_rule_acc,
            "n_runs": n,
            "raw_results": complexity_results,
        })

        logger.info(f"  Avg accuracy: {avg_accuracy:.3f}")

    return results


def run_full_experiment(cfg: DictConfig) -> dict:
    """Run the full V2 experiment suite."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb
    wandb_run = None
    if cfg.wandb.enabled:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.wandb.name}-{timestamp}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    all_results = {}

    try:
        # Run experiments based on config
        if cfg.experiments.oracle_scaling:
            logger.info("=" * 50)
            logger.info("Running Oracle Scaling Experiment")
            logger.info("=" * 50)
            oracle_results = run_oracle_scaling_experiment(cfg, wandb_run)
            all_results["oracle_scaling"] = oracle_results

            # Save intermediate
            with open(output_dir / "oracle_scaling.json", "w") as f:
                json.dump(oracle_results, f, indent=2, default=str)

        if cfg.experiments.information_conditions:
            logger.info("=" * 50)
            logger.info("Running Information Conditions Experiment")
            logger.info("=" * 50)
            info_results = run_information_conditions_experiment(cfg, wandb_run)
            all_results["information_conditions"] = info_results

            with open(output_dir / "information_conditions.json", "w") as f:
                json.dump(info_results, f, indent=2, default=str)

        if cfg.experiments.rule_complexity:
            logger.info("=" * 50)
            logger.info("Running Rule Complexity Experiment")
            logger.info("=" * 50)
            complexity_results = run_rule_complexity_experiment(cfg, wandb_run)
            all_results["rule_complexity"] = complexity_results

            with open(output_dir / "rule_complexity.json", "w") as f:
                json.dump(complexity_results, f, indent=2, default=str)

        # Save all results
        with open(output_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        # Create summary
        summary = create_summary(all_results, cfg)
        with open(output_dir / "summary.md", "w") as f:
            f.write(summary)

        if wandb_run:
            # Log summary table to wandb
            wandb.log({"summary": wandb.Html(summary.replace("\n", "<br>"))})

        logger.info(f"\nResults saved to: {output_dir}")

    finally:
        if wandb_run:
            wandb.finish()

    return all_results


def create_summary(results: dict, cfg: DictConfig) -> str:
    """Create a markdown summary of results."""
    lines = [
        "# V2 Hidden Value Game Experiment Results",
        "",
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Seeds**: {cfg.seeds}",
        "",
    ]

    if "oracle_scaling" in results:
        lines.extend([
            "## Experiment 1: Oracle Scaling",
            "",
            "| Budget | Selection Acc | Prop Acc | Rule Acc | vs Random |",
            "|--------|--------------|----------|----------|-----------|",
        ])
        for r in results["oracle_scaling"]:
            budget = r['oracle_budget']
            sel_acc = r['avg_selection_accuracy']
            prop_acc = r.get('avg_property_accuracy', 0)
            rule_acc = r.get('avg_rule_inference_accuracy', 0)
            vs_rand = r.get('avg_value_vs_random', 0)
            row = f"| {budget} | {sel_acc:.3f} | {prop_acc:.3f} | {rule_acc:.3f} | {vs_rand:.1f} |"
            lines.append(row)
        lines.append("")

    if "information_conditions" in results:
        lines.extend([
            "## Experiment 2: Information Conditions",
            "",
            "| Condition | Selection Acc | Prop Acc | Rule Acc |",
            "|-----------|--------------|----------|----------|",
        ])
        for r in results["information_conditions"]:
            cond = r['condition']
            sel_acc = r['avg_selection_accuracy']
            prop_acc = r.get('avg_property_accuracy', 0)
            rule_acc = r.get('avg_rule_inference_accuracy', 0)
            lines.append(f"| {cond} | {sel_acc:.3f} | {prop_acc:.3f} | {rule_acc:.3f} |")
        lines.append("")

    if "rule_complexity" in results:
        lines.extend([
            "## Experiment 3: Rule Complexity",
            "",
            "| Complexity | Selection Acc | Prop Acc | Rule Acc |",
            "|------------|--------------|----------|----------|",
        ])
        for r in results["rule_complexity"]:
            complexity = r['rule_complexity']
            sel_acc = r['avg_selection_accuracy']
            prop_acc = r.get('avg_property_accuracy', 0)
            rule_acc = r.get('avg_rule_inference_accuracy', 0)
            lines.append(f"| {complexity} | {sel_acc:.3f} | {prop_acc:.3f} | {rule_acc:.3f} |")
        lines.append("")

    return "\n".join(lines)


@hydra.main(config_path="../configs/experiment", config_name="v2_hidden_value", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    logger.info("Starting V2 Hidden Value Game Experiment")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    run_full_experiment(cfg)


if __name__ == "__main__":
    main()
