"""Simulation orchestration for statement generation."""

import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

from .agent import Agent, AgentType, Statement, Task, DEFAULT_OBSERVER_TASK
from .world import Property, World


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""

    n_objects: int = 20
    n_agents: int = 3
    adversarial_fraction: float = 0.5  # Fraction of agents that are adversarial
    statements_per_agent: int = 40  # Statements each agent makes to observer
    seed: int | None = None
    properties: list[Property] | None = None
    agent_tasks: list[Task] | None = None
    observer_task: Task | None = None
    model: str = "claude-sonnet-4-20250514"


@dataclass
class SimulationResult:
    """Results from a simulation run."""

    world_state: dict
    statements: list[dict]
    agent_metadata: dict[str, dict]
    observer_task: dict
    config: dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, path: str | Path) -> None:
        """Save results to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "SimulationResult":
        """Load results from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "world_state": self.world_state,
            "statements": self.statements,
            "agent_metadata": self.agent_metadata,
            "observer_task": self.observer_task,
            "config": self.config,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SimulationResult":
        """Create from dictionary representation."""
        return cls(
            world_state=data["world_state"],
            statements=data["statements"],
            agent_metadata=data["agent_metadata"],
            observer_task=data["observer_task"],
            config=data["config"],
            timestamp=data.get("timestamp", ""),
        )

    def get_stats(self) -> dict:
        """Get summary statistics about the simulation."""
        total = len(self.statements)
        truthful = sum(1 for s in self.statements if s["is_truthful"])
        adversarial = sum(
            1 for s in self.statements if s["agent_type"] == "adversarial"
        )
        return {
            "total_statements": total,
            "truthful_statements": truthful,
            "deceptive_statements": total - truthful,
            "truthful_fraction": truthful / total if total > 0 else 0,
            "adversarial_agent_statements": adversarial,
            "cooperative_agent_statements": total - adversarial,
            "n_agents": len(self.agent_metadata),
            "n_objects": len(self.world_state.get("objects", {})),
        }


class Simulation:
    """
    Orchestrates statement generation in the truthification environment.

    New model:
    - Observer has a task (e.g., "find red circular objects")
    - Agents build world models and report to observer
    - Cooperative agents have true beliefs
    - Adversarial agents have false beliefs about observer-relevant properties
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        client: anthropic.Anthropic | None = None,
    ):
        """Initialize the simulation."""
        self.config = config or SimulationConfig()
        self.client = client or anthropic.Anthropic()
        self.world: World | None = None
        self.agents: dict[str, Agent] = {}
        self.statements: list[Statement] = []
        self.rng = random.Random(self.config.seed)
        self.observer_task = self.config.observer_task or DEFAULT_OBSERVER_TASK

    def setup(self) -> None:
        """Set up the world and agents."""
        self._create_world()
        self._create_agents()
        self._assign_knowledge()
        self._build_agent_world_models()

    def _create_world(self) -> None:
        """Create the world with objects."""
        from .world import DEFAULT_PROPERTIES

        properties = self.config.properties or DEFAULT_PROPERTIES
        self.world = World.generate(
            n_objects=self.config.n_objects,
            properties=properties,
            seed=self.config.seed,
        )

    def _create_agents(self) -> None:
        """Create agents with tasks and types."""
        from .agent import DEFAULT_TASKS

        tasks = self.config.agent_tasks or DEFAULT_TASKS

        # Determine which agents are adversarial
        n_adversarial = int(self.config.n_agents * self.config.adversarial_fraction)

        for i in range(self.config.n_agents):
            agent_id = f"Agent_{chr(65 + i)}"  # Agent_A, Agent_B, etc.
            task = tasks[i % len(tasks)]

            # First n_adversarial agents are adversarial
            agent_type = (
                AgentType.ADVERSARIAL if i < n_adversarial else AgentType.COOPERATIVE
            )

            agent = Agent(
                id=agent_id,
                task=task,
                agent_type=agent_type,
                _client=self.client,
                model=self.config.model,
            )
            self.agents[agent_id] = agent

    def _assign_knowledge(self) -> None:
        """Assign object knowledge to agents."""
        if self.world is None:
            raise RuntimeError("World not created")

        object_ids = self.world.list_objects()

        # All agents know about all objects (simplified model)
        for agent in self.agents.values():
            agent.known_objects = list(object_ids)

    def _build_agent_world_models(self) -> None:
        """Have each agent build their internal world model."""
        if self.world is None:
            raise RuntimeError("World not created")

        for agent in self.agents.values():
            agent.build_world_model(
                world=self.world,
                observer_task=self.observer_task,
                rng=self.rng,
            )

    def run(self, progress_callback: Any | None = None) -> SimulationResult:
        """
        Run the simulation and generate statements.

        Args:
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            SimulationResult with all generated statements
        """
        if self.world is None:
            self.setup()

        self.statements = []
        agent_list = list(self.agents.values())
        total_statements = len(agent_list) * self.config.statements_per_agent
        current = 0

        # Each agent generates statements to the observer
        for agent in agent_list:
            statements = self._generate_agent_statements(agent)
            self.statements.extend(statements)

            current += len(statements)
            if progress_callback:
                progress_callback(current, total_statements)

        return self._create_result()

    def _generate_agent_statements(self, agent: Agent) -> list[Statement]:
        """Generate statements from one agent to the observer."""
        if self.world is None:
            raise RuntimeError("World not created")

        statements = []
        property_names = [p.name for p in self.world.property_definitions]

        for _ in range(self.config.statements_per_agent):
            # Pick random object and property
            obj_id = self.rng.choice(agent.known_objects)
            prop_name = self.rng.choice(property_names)

            try:
                statement = agent.generate_statement(
                    world=self.world,
                    object_id=obj_id,
                    property_name=prop_name,
                    observer_task=self.observer_task,
                )
                statements.append(statement)
            except Exception as e:
                print(f"Error generating statement: {e}")

        return statements

    def _create_result(self) -> SimulationResult:
        """Create the simulation result."""
        if self.world is None:
            raise RuntimeError("World not created")

        return SimulationResult(
            world_state=self.world.to_dict(),
            statements=[self._statement_to_dict(s) for s in self.statements],
            agent_metadata={
                agent_id: agent.to_dict() for agent_id, agent in self.agents.items()
            },
            observer_task={
                "name": self.observer_task.name,
                "description": self.observer_task.description,
                "relevant_properties": self.observer_task.relevant_properties,
            },
            config={
                "n_objects": self.config.n_objects,
                "n_agents": self.config.n_agents,
                "adversarial_fraction": self.config.adversarial_fraction,
                "statements_per_agent": self.config.statements_per_agent,
                "seed": self.config.seed,
                "model": self.config.model,
            },
        )

    def _statement_to_dict(self, statement: Statement) -> dict:
        """Convert a statement to dictionary representation."""
        return {
            "text": statement.text,
            "agent_id": statement.agent_id,
            "object_id": statement.object_id,
            "property_name": statement.property_name,
            "claimed_value": statement.claimed_value,
            "ground_truth": statement.ground_truth_value,
            "is_truthful": statement.is_truthful,
            "agent_type": statement.agent_type.value,
        }


def run_simulation(
    n_objects: int = 20,
    n_agents: int = 3,
    adversarial_fraction: float = 0.5,
    statements_per_agent: int = 40,
    seed: int | None = None,
    output_path: str | Path | None = None,
    model: str = "claude-sonnet-4-20250514",
    observer_task: Task | None = None,
) -> SimulationResult:
    """
    Convenience function to run a simulation with the given parameters.

    Args:
        n_objects: Number of objects in the world
        n_agents: Number of agents
        adversarial_fraction: Fraction of agents that are adversarial
        statements_per_agent: Number of statements each agent generates
        seed: Random seed for reproducibility
        output_path: Optional path to save results
        model: Anthropic model to use for statement generation
        observer_task: Task for the observer (what they're trying to learn)

    Returns:
        SimulationResult with all generated statements
    """
    config = SimulationConfig(
        n_objects=n_objects,
        n_agents=n_agents,
        adversarial_fraction=adversarial_fraction,
        statements_per_agent=statements_per_agent,
        seed=seed,
        model=model,
        observer_task=observer_task,
    )
    sim = Simulation(config)
    result = sim.run()

    if output_path:
        result.save(output_path)

    return result
