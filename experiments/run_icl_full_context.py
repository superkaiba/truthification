#!/usr/bin/env python
"""Run Full-Context ICL experiment.

Unlike the isolated-query experiment, this shows ALL statements to the observer
for every query, allowing it to infer agent reliability from consistency patterns.

Can load pre-generated data from baseline experiment for direct comparison.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment import Simulation, SimulationConfig
from environment.agent import Task
from environment.world import Property, PropertyType
from observer import ICLObserver, Condition, FullContextPromptBuilder
from evaluation import (
    compute_accuracy,
    compute_accuracy_by_category,
    compute_ece,
    compute_brier_score,
    categorize_queries,
)


@dataclass
class LoadedDataset:
    """Dataset loaded from a previous experiment."""
    world_state: dict
    observer_task: dict
    agent_metadata: dict
    statements: list[dict]


def load_dataset_from_debug(debug_path: Path) -> LoadedDataset:
    """Load dataset from a debug JSON file."""
    with open(debug_path) as f:
        data = json.load(f)

    return LoadedDataset(
        world_state=data["world_state"],
        observer_task=data["observer_task"],
        agent_metadata=data["agents"],
        statements=data["statements"],
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


def build_observer_task(cfg: DictConfig) -> Task:
    """Build observer task from config."""
    return Task(
        name=cfg.observer_task.name,
        description=cfg.observer_task.description,
        relevant_properties=list(cfg.observer_task.relevant_properties),
    )


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


def save_debug_data(
    output_dir: Path,
    seed: int,
    result,
    queries: list[tuple[str, str, str]],
    categories: dict[tuple[str, str, str], str],
    reliabilities: dict[str, int],
    query_results: dict,
    config: dict,
) -> Path:
    """Save detailed debug data for dashboard inspection."""
    debug_data = {
        "metadata": {
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "config": config,
            "experiment_type": "full_context",
        },
        "world_state": result.world_state,
        "observer_task": result.observer_task,
        "agents": {
            agent_id: {
                **meta,
                "computed_reliability": reliabilities.get(agent_id, 50),
            }
            for agent_id, meta in result.agent_metadata.items()
        },
        "statements": result.statements,
        "queries": [
            {
                "object_id": obj_id,
                "property_name": prop_name,
                "property_value": prop_value,
                "ground_truth": True,
                "category": categories.get((obj_id, prop_name, prop_value), "unknown"),
                "results": query_results.get((obj_id, prop_name, prop_value), {}),
            }
            for obj_id, prop_name, prop_value in queries
        ],
    }

    debug_path = output_dir / f"seed_{seed}_debug.json"
    with open(debug_path, "w") as f:
        json.dump(debug_data, f, indent=2)

    return debug_path


@hydra.main(version_base=None, config_path="../configs/experiment", config_name="icl_full_context")
def main(cfg: DictConfig) -> None:
    """Run Full-Context ICL experiment."""
    print("Full-Context ICL Experiment")
    print("=" * 60)
    print("Observer sees ALL statements for every query")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    properties = build_properties(cfg)
    tasks = build_tasks(cfg)
    observer_task = build_observer_task(cfg)

    all_results = []

    # Check if we should load from existing dataset
    load_from_dataset = getattr(cfg, "load_from_dataset", None)

    for seed in cfg.seeds:
        print(f"\n--- Seed {seed} ---")

        # Try to load from existing dataset if specified
        if load_from_dataset:
            dataset_path = Path(load_from_dataset) / f"seed_{seed}_debug.json"
            if dataset_path.exists():
                print(f"Loading dataset from {dataset_path}")
                dataset = load_dataset_from_debug(dataset_path)

                # Create a mock result object with loaded data
                class MockResult:
                    def __init__(self, dataset):
                        self.world_state = dataset.world_state
                        self.observer_task = dataset.observer_task
                        self.agent_metadata = dataset.agent_metadata
                        self.statements = dataset.statements

                    def get_stats(self):
                        truthful = sum(1 for s in self.statements if s["is_truthful"])
                        adv = sum(1 for s in self.statements if s["agent_type"] == "adversarial")
                        return {
                            "truthful_fraction": truthful / len(self.statements),
                            "adversarial_agent_statements": adv,
                            "cooperative_agent_statements": len(self.statements) - adv,
                        }

                result = MockResult(dataset)
                print(f"Loaded {len(result.statements)} statements from dataset")
            else:
                print(f"Dataset not found at {dataset_path}, generating new data")
                load_from_dataset = None

        if not load_from_dataset:
            # Generate new data
            sim_config = SimulationConfig(
                n_objects=cfg.world.n_objects,
                n_agents=cfg.agent.n_agents,
                adversarial_fraction=cfg.agent.adversarial_fraction,
                statements_per_agent=cfg.statements_per_agent,
                seed=seed,
                properties=properties,
                agent_tasks=tasks,
                observer_task=observer_task,
                model=cfg.agent_model,
            )

            sim = Simulation(sim_config)
            result = sim.run()
            print(f"Generated {len(result.statements)} statements")

        stats = result.get_stats()
        print(f"  Truthful: {stats['truthful_fraction']:.1%}")
        print(f"  From adversarial agents: {stats['adversarial_agent_statements']}")
        print(f"  From cooperative agents: {stats['cooperative_agent_statements']}")

        queries = generate_queries(result.world_state)
        print(f"Generated {len(queries)} queries")

        reliabilities = compute_agent_reliabilities(result.statements)

        categories = categorize_queries(queries, result.statements, result.world_state)
        categories_dict = {q: categories[i] for i, q in enumerate(queries)}

        query_results: dict = {}
        for q in queries:
            query_results[q] = {}

        # Prepare agent metadata
        agent_metadata_with_reliability = {}
        for agent_id, meta in result.agent_metadata.items():
            agent_metadata_with_reliability[agent_id] = {
                **meta,
                "reliability": reliabilities.get(agent_id, 50),
            }

        # Run each full-context condition
        for condition_name in cfg.conditions:
            condition = Condition(condition_name)
            builder = FullContextPromptBuilder(condition)

            # Pre-compute full evidence (same for all queries in this condition)
            full_evidence = builder.format_full_evidence(
                result.statements,
                agent_metadata_with_reliability,
            )

            # Count tokens (rough estimate)
            token_estimate = len(full_evidence.split()) * 1.3
            print(f"\nCondition: {condition_name}")
            print(f"  Full evidence: ~{int(token_estimate)} tokens")

            for observer_model in cfg.observer_models:
                print(f"  Running: {condition_name} / {observer_model}")

                observer = ICLObserver(
                    model=observer_model,
                    system_prompt=builder.build_system_prompt(),
                )

                predictions = []
                confidences = []
                ground_truths = []

                for obj_id, prop_name, prop_value in tqdm(queries, desc="Queries"):
                    query_key = (obj_id, prop_name, prop_value)

                    # Build query with full context
                    query_prompt = builder.build_query(
                        full_evidence=full_evidence,
                        object_id=obj_id,
                        property_name=prop_name,
                        property_value=str(prop_value),
                    )

                    try:
                        response = observer.query_with_prompt(query_prompt)
                        pred = response.answer
                        conf = response.confidence
                    except Exception as e:
                        print(f"Error: {e}")
                        pred = True
                        conf = 50

                    predictions.append(pred)
                    confidences.append(conf)
                    ground_truths.append(True)

                    if condition_name not in query_results[query_key]:
                        query_results[query_key][condition_name] = {}
                    query_results[query_key][condition_name][observer_model] = {
                        "prediction": pred,
                        "confidence": conf,
                    }

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

                wandb.log(run_result)

                print(f"    Accuracy: {accuracy:.1%}")
                print(f"    Contested: {accuracy_by_cat.get('contested', 'N/A')}")
                print(f"    ECE: {ece:.3f}")

        # Save debug data
        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        debug_path = save_debug_data(
            output_dir=output_dir,
            seed=seed,
            result=result,
            queries=queries,
            categories=categories_dict,
            reliabilities=reliabilities,
            query_results=query_results,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"Debug data saved to {debug_path}")

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    import pandas as pd
    wandb.log({"results_table": wandb.Table(dataframe=pd.DataFrame(all_results))})

    wandb.finish()


if __name__ == "__main__":
    main()
