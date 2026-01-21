#!/usr/bin/env python
"""Run a simulation to generate agent statements."""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.agent import Task
from environment.simulation import Simulation, SimulationConfig
from environment.world import Property, PropertyType


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


@hydra.main(version_base=None, config_path="../configs/experiment", config_name="simulation")
def main(cfg: DictConfig) -> None:
    """Run simulation with Hydra configuration."""
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    # Build configuration
    properties = build_properties(cfg)
    tasks = build_tasks(cfg)

    config = SimulationConfig(
        n_objects=cfg.world.n_objects,
        n_agents=cfg.agent.n_agents,
        adversarial_fraction=cfg.agent.adversarial_fraction,
        statements_per_pair=cfg.statements_per_pair,
        seed=cfg.get("seed"),
        properties=properties,
        tasks=tasks,
        model=cfg.agent.get("model", "claude-sonnet-4-20250514"),
    )

    # Run simulation
    print("\nRunning simulation...")
    sim = Simulation(config)
    sim.setup()

    # Progress bar
    pbar = tqdm(total=config.n_agents * (config.n_agents - 1), desc="Generating statements")

    def progress_callback(current: int, total: int) -> None:
        pbar.update(1)

    result = sim.run(progress_callback=progress_callback)
    pbar.close()

    # Print stats
    stats = result.get_stats()
    print("\nSimulation Results:")
    print(f"  Total statements: {stats['total_statements']}")
    print(f"  Truthful: {stats['truthful_statements']} ({stats['truthful_fraction']:.1%})")
    print(f"  Deceptive: {stats['deceptive_statements']}")
    print(f"  Adversarial context: {stats['adversarial_statements']}")
    print(f"  Cooperative context: {stats['cooperative_statements']}")

    # Save results
    if cfg.get("save_results", True):
        output_dir = Path(cfg.get("output_dir", "outputs/simulations"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"simulation_seed{cfg.get('seed', 'none')}.json"
        result.save(output_path)
        print(f"\nResults saved to: {output_path}")

    # Print sample statements
    print("\nSample statements:")
    for i, stmt in enumerate(result.statements[:5]):
        truthful = "TRUTH" if stmt["is_truthful"] else "LIE"
        print(f"  [{truthful}] {stmt['agent_id']} → {stmt['target_id']}: {stmt['text']}")


if __name__ == "__main__":
    main()
