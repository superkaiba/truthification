#!/usr/bin/env python
"""Run Context Scaling ICL experiment.

Measures how varying the number of statements shown affects accuracy.
Tests context sizes from small (20 statements) to full (240 statements).
"""

import json
import random
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

from observer import ICLObserver, Condition, FullContextPromptBuilder
from evaluation import (
    compute_accuracy,
    compute_accuracy_by_category,
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


def sample_statements(
    statements: list[dict],
    n_statements: int,
    sampling_strategy: str = "uniform",
    seed: int = 42,
) -> list[dict]:
    """Sample a subset of statements.

    Args:
        statements: Full list of statements
        n_statements: Number to sample (if >= len(statements), return all)
        sampling_strategy: How to sample
            - "uniform": Random sample
            - "balanced": Equal per agent
            - "recent": Last N statements (simulates recency)
        seed: Random seed for reproducibility
    """
    if n_statements >= len(statements):
        return statements

    rng = random.Random(seed)

    if sampling_strategy == "uniform":
        return rng.sample(statements, n_statements)

    elif sampling_strategy == "balanced":
        # Sample equally from each agent
        from collections import defaultdict
        by_agent = defaultdict(list)
        for stmt in statements:
            by_agent[stmt["agent_id"]].append(stmt)

        n_agents = len(by_agent)
        per_agent = n_statements // n_agents
        remainder = n_statements % n_agents

        sampled = []
        for i, (agent_id, agent_stmts) in enumerate(sorted(by_agent.items())):
            n = per_agent + (1 if i < remainder else 0)
            n = min(n, len(agent_stmts))
            sampled.extend(rng.sample(agent_stmts, n))

        return sampled

    elif sampling_strategy == "recent":
        # Take the last N statements (assuming order reflects time)
        return statements[-n_statements:]

    return statements


def format_evidence_subset(
    statements: list[dict],
    agent_metadata: dict,
    condition: Condition,
) -> str:
    """Format a subset of statements for the prompt."""
    builder = FullContextPromptBuilder(condition)
    return builder.format_full_evidence(statements, agent_metadata)


@hydra.main(version_base=None, config_path="../configs/experiment", config_name="icl_context_scaling")
def main(cfg: DictConfig) -> None:
    """Run Context Scaling ICL experiment."""
    print("Context Scaling ICL Experiment")
    print("=" * 60)
    print("Testing how accuracy varies with number of statements shown")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    all_results = []

    # Context sizes to test
    context_sizes = list(cfg.context_sizes)

    for seed in cfg.seeds:
        print(f"\n--- Seed {seed} ---")

        # Load dataset
        dataset_path = Path(cfg.load_from_dataset) / f"seed_{seed}_debug.json"
        if not dataset_path.exists():
            print(f"Dataset not found at {dataset_path}, skipping seed {seed}")
            continue

        print(f"Loading dataset from {dataset_path}")
        dataset = load_dataset_from_debug(dataset_path)

        total_statements = len(dataset.statements)
        print(f"Total statements: {total_statements}")

        queries = generate_queries(dataset.world_state)
        print(f"Generated {len(queries)} queries")

        reliabilities = compute_agent_reliabilities(dataset.statements)

        categories = categorize_queries(queries, dataset.statements, dataset.world_state)

        # Prepare agent metadata with reliability
        agent_metadata_with_reliability = {}
        for agent_id, meta in dataset.agent_metadata.items():
            agent_metadata_with_reliability[agent_id] = {
                **meta,
                "reliability": reliabilities.get(agent_id, 50),
            }

        # Test each context size
        for n_statements in context_sizes:
            if n_statements > total_statements:
                n_statements = total_statements

            print(f"\n  Context size: {n_statements} statements")

            # Sample statements
            sampled_statements = sample_statements(
                dataset.statements,
                n_statements,
                sampling_strategy=cfg.sampling_strategy,
                seed=seed,
            )

            # Test each condition
            for condition_name in cfg.conditions:
                condition = Condition(condition_name)
                builder = FullContextPromptBuilder(condition)

                # Format evidence with sampled statements
                full_evidence = format_evidence_subset(
                    sampled_statements,
                    agent_metadata_with_reliability,
                    condition,
                )

                # Test each model
                for observer_model in cfg.observer_models:
                    print(f"    {condition_name} / {observer_model}")

                    observer = ICLObserver(
                        model=observer_model,
                        system_prompt=builder.build_system_prompt(),
                    )

                    predictions = []
                    confidences = []
                    ground_truths = []

                    for obj_id, prop_name, prop_value in tqdm(queries, desc="Queries", leave=False):
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

                    accuracy = compute_accuracy(predictions, ground_truths)
                    accuracy_by_cat = compute_accuracy_by_category(
                        predictions, ground_truths, categories
                    )

                    run_result = {
                        "seed": seed,
                        "n_statements": n_statements,
                        "condition": condition_name,
                        "observer_model": observer_model,
                        "accuracy": accuracy,
                        "accuracy_contested": accuracy_by_cat.get("contested", None),
                        "accuracy_unanimous": accuracy_by_cat.get("unanimous", None),
                        "n_queries": len(queries),
                    }
                    all_results.append(run_result)

                    wandb.log(run_result)

                    print(f"      Accuracy: {accuracy:.1%}")

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Create scaling plot
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.DataFrame(all_results)

    model_names = {
        "claude-3-5-haiku-20241022": "Haiku",
        "claude-sonnet-4-20250514": "Sonnet",
        "claude-opus-4-20250514": "Opus",
    }
    df["model"] = df["observer_model"].map(model_names)

    # Average across seeds
    avg_df = df.groupby(["n_statements", "model", "condition"]).agg({
        "accuracy": "mean",
        "accuracy_contested": "mean",
    }).reset_index()

    colors = {
        "Haiku": "#E57373",
        "Sonnet": "#64B5F6",
        "Opus": "#81C784",
    }

    # Plot: Accuracy vs Context Size by Model
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in ["Haiku", "Sonnet", "Opus"]:
        model_data = avg_df[(avg_df["model"] == model) & (avg_df["condition"] == "full_context_ids_tasks")]
        if not model_data.empty:
            ax.plot(
                model_data["n_statements"],
                model_data["accuracy"] * 100,
                marker="o",
                label=model,
                color=colors[model],
                linewidth=2,
                markersize=8,
            )

    ax.set_xlabel("Number of Statements in Context", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Effect of Context Size on Accuracy\n(Full Context with IDs + Tasks)", fontsize=14, fontweight="bold")
    ax.legend(title="Model", fontsize=10)
    ax.set_ylim(60, 100)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    wandb.log({"context_scaling_accuracy": wandb.Image(fig)})
    fig.savefig(output_dir / "context_scaling_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot: All conditions comparison
    fig2, axes = plt.subplots(1, 3, figsize=(15, 5))

    conditions = ["full_context_no_ids", "full_context_ids", "full_context_ids_tasks"]
    condition_labels = {
        "full_context_no_ids": "No IDs",
        "full_context_ids": "IDs Only",
        "full_context_ids_tasks": "IDs + Tasks",
    }

    for i, cond in enumerate(conditions):
        ax = axes[i]
        for model in ["Haiku", "Sonnet", "Opus"]:
            model_data = avg_df[(avg_df["model"] == model) & (avg_df["condition"] == cond)]
            if not model_data.empty:
                ax.plot(
                    model_data["n_statements"],
                    model_data["accuracy"] * 100,
                    marker="o",
                    label=model,
                    color=colors[model],
                    linewidth=2,
                    markersize=6,
                )

        ax.set_xlabel("Statements" if i == 1 else "")
        ax.set_ylabel("Accuracy (%)" if i == 0 else "")
        ax.set_title(condition_labels[cond], fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(60, 100)
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Context Scaling Across Conditions", fontsize=14, y=1.02)
    plt.tight_layout()
    wandb.log({"context_scaling_by_condition": wandb.Image(fig2)})
    fig2.savefig(output_dir / "context_scaling_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # Log summary table
    wandb.log({"context_scaling_results": wandb.Table(dataframe=df)})

    print(f"\nPlots saved to {output_dir}/")
    print(f"wandb: {wandb.run.url}")

    wandb.finish()


if __name__ == "__main__":
    main()
