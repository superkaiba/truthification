#!/usr/bin/env python
"""Run ICL baseline experiment."""

import json
import sys
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment import Simulation, SimulationConfig
from environment.agent import Task
from environment.world import Property, PropertyType
from observer import ICLObserver, Condition, PromptBuilder
from evaluation import (
    compute_accuracy,
    compute_accuracy_by_category,
    compute_ece,
    compute_brier_score,
    categorize_queries,
)


def build_properties(cfg: DictConfig) -> list[Property]:
    """Build property definitions from config."""
    properties = []
    for prop_cfg in cfg.world.properties:
        prop_type = PropertyType(prop_cfg.type)
        properties.append(
            Property(
                name=prop_cfg.name,
                property_type=prop_type,
                possible_values=list(prop_cfg.possible_values),
            )
        )
    return properties


def build_tasks(cfg: DictConfig) -> list[Task]:
    """Build task definitions from config."""
    tasks = []
    for task_cfg in cfg.agent.tasks:
        tasks.append(
            Task(
                name=task_cfg.name,
                description=task_cfg.description,
                relevant_properties=list(task_cfg.relevant_properties),
            )
        )
    return tasks


def generate_queries(world_state: dict) -> list[tuple[str, str, str]]:
    """Generate all (object_id, property_name, value) queries."""
    queries = []
    for obj_id, obj_data in world_state["objects"].items():
        for prop_name, prop_value in obj_data["properties"].items():
            queries.append((obj_id, prop_name, prop_value))
    return queries


def compute_agent_reliabilities(statements: list[dict]) -> dict[str, int]:
    """Compute reliability % for each agent from statements."""
    from collections import defaultdict

    agent_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "truthful": 0})

    for stmt in statements:
        agent_id = stmt["agent_id"]
        agent_stats[agent_id]["total"] += 1
        if stmt["is_truthful"]:
            agent_stats[agent_id]["truthful"] += 1

    reliabilities = {}
    for agent_id, stats in agent_stats.items():
        if stats["total"] > 0:
            reliabilities[agent_id] = int(100 * stats["truthful"] / stats["total"])
        else:
            reliabilities[agent_id] = 50

    return reliabilities


@hydra.main(version_base=None, config_path="../configs/experiment", config_name="icl_baseline")
def main(cfg: DictConfig) -> None:
    """Run ICL baseline experiment."""
    print("ICL Baseline Experiment")
    print("=" * 50)
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Build world config
    properties = build_properties(cfg)
    tasks = build_tasks(cfg)

    all_results = []

    for seed in cfg.seeds:
        print(f"\n--- Seed {seed} ---")

        # Generate data
        sim_config = SimulationConfig(
            n_objects=cfg.world.n_objects,
            n_agents=cfg.agent.n_agents,
            adversarial_fraction=cfg.agent.adversarial_fraction,
            statements_per_pair=cfg.statements_per_pair,
            seed=seed,
            properties=properties,
            tasks=tasks,
            model=cfg.agent_model,
        )

        sim = Simulation(sim_config)
        result = sim.run()

        print(f"Generated {len(result.statements)} statements")
        stats = result.get_stats()
        print(f"  Truthful: {stats['truthful_fraction']:.1%}")

        # Build queries
        queries = generate_queries(result.world_state)
        print(f"Generated {len(queries)} queries")

        # Compute agent reliabilities for oracle condition
        reliabilities = compute_agent_reliabilities(result.statements)

        # Categorize queries
        categories = categorize_queries(
            queries, result.statements, result.world_state
        )

        # Run each condition × observer combination
        for condition_name in cfg.conditions:
            condition = Condition(condition_name)
            builder = PromptBuilder(condition)

            # Inject reliabilities for oracle condition
            agent_metadata_with_reliability = {}
            for agent_id, meta in result.agent_metadata.items():
                agent_metadata_with_reliability[agent_id] = {
                    **meta,
                    "reliability": reliabilities.get(agent_id, 50),
                }

            for observer_model in cfg.observer_models:
                print(f"\nRunning: {condition_name} / {observer_model}")

                observer = ICLObserver(model=observer_model)

                predictions = []
                confidences = []
                ground_truths = []

                for obj_id, prop_name, prop_value in tqdm(queries, desc="Queries"):
                    # Format evidence
                    evidence = builder.format_evidence(
                        result.statements,
                        agent_metadata_with_reliability,
                        obj_id,
                        prop_name,
                    )

                    # Query observer
                    try:
                        response = observer.query(
                            evidence=evidence,
                            object_id=obj_id,
                            property_name=prop_name,
                            property_value=str(prop_value),
                        )
                        predictions.append(response.answer)
                        confidences.append(response.confidence)
                    except Exception as e:
                        print(f"Error: {e}")
                        predictions.append(True)  # Default
                        confidences.append(50)

                    ground_truths.append(True)  # Query asks "is X value?" - answer is always True

                # Compute metrics
                accuracy = compute_accuracy(predictions, ground_truths)
                accuracy_by_cat = compute_accuracy_by_category(
                    predictions, ground_truths, categories
                )
                ece = compute_ece(confidences, ground_truths)
                brier = compute_brier_score(confidences, ground_truths)

                run_result = {
                    "seed": seed,
                    "condition": condition_name,
                    "observer_model": observer_model,
                    "accuracy": accuracy,
                    "accuracy_contested": accuracy_by_cat.get("contested", None),
                    "accuracy_unanimous": accuracy_by_cat.get("unanimous", None),
                    "ece": ece,
                    "brier": brier,
                    "n_queries": len(queries),
                }
                all_results.append(run_result)

                # Log to wandb
                wandb.log(run_result)

                print(f"  Accuracy: {accuracy:.1%}")
                print(f"  ECE: {ece:.3f}")

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Log summary table to wandb
    import pandas as pd
    wandb.log({"results_table": wandb.Table(dataframe=pd.DataFrame(all_results))})

    wandb.finish()


if __name__ == "__main__":
    main()
