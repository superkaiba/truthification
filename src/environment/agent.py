"""LLM-based agent with world models for truthification experiments."""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import anthropic

from .world import Object, World


class AgentType(Enum):
    """Whether agent helps or hinders the observer."""

    COOPERATIVE = "cooperative"  # Wants observer to succeed
    ADVERSARIAL = "adversarial"  # Wants observer to fail


@dataclass
class Task:
    """A task that defines what an agent (or observer) cares about."""

    name: str
    description: str
    relevant_properties: list[str]

    def is_property_relevant(self, property_name: str) -> bool:
        """Check if a property is relevant to this task."""
        return property_name in self.relevant_properties


@dataclass
class Statement:
    """A statement made by an agent to the observer."""

    text: str
    agent_id: str
    object_id: str
    property_name: str
    claimed_value: Any
    ground_truth_value: Any
    is_truthful: bool
    agent_type: AgentType


@dataclass
class Agent:
    """
    An LLM-based agent that builds a world model and reports to the observer.

    Key behaviors:
    - Builds an internal world model (beliefs about object properties)
    - Cooperative agents believe the truth
    - Adversarial agents believe false values for observer-task-relevant properties
    - Statements to observer are consistent with agent's world model
    """

    id: str
    task: Task  # Agent's own task (provides motivation)
    agent_type: AgentType  # Cooperative or adversarial toward observer
    known_objects: list[str] = field(default_factory=list)
    world_model: dict[str, dict[str, Any]] = field(default_factory=dict)
    _client: anthropic.Anthropic | None = field(default=None, repr=False)
    model: str = "claude-sonnet-4-20250514"

    def __post_init__(self):
        """Initialize Anthropic client if not provided."""
        if self._client is None:
            self._client = anthropic.Anthropic()

    def build_world_model(
        self,
        world: World,
        observer_task: Task,
        rng: random.Random | None = None,
    ) -> None:
        """
        Build agent's internal world model based on true world and agent type.

        Cooperative agents: believe the truth about everything
        Adversarial agents: believe false values for observer-task-relevant properties
        """
        if rng is None:
            rng = random.Random()

        self.world_model = {}

        for obj_id in self.known_objects:
            obj = world.get_object(obj_id)
            if obj is None:
                continue

            self.world_model[obj_id] = {}

            for prop_name, true_value in obj.properties.items():
                # Adversarial agents have false beliefs about observer-relevant properties
                if (
                    self.agent_type == AgentType.ADVERSARIAL
                    and observer_task.is_property_relevant(prop_name)
                ):
                    # Generate a consistent false belief
                    prop_def = next(
                        (p for p in world.property_definitions if p.name == prop_name),
                        None,
                    )
                    if prop_def:
                        false_values = [
                            v for v in prop_def.possible_values if v != true_value
                        ]
                        if false_values:
                            believed_value = rng.choice(false_values)
                        else:
                            believed_value = true_value
                    else:
                        believed_value = true_value
                else:
                    # Believe the truth
                    believed_value = true_value

                self.world_model[obj_id][prop_name] = believed_value

    def get_belief(self, object_id: str, property_name: str) -> Any | None:
        """Get agent's belief about an object property."""
        if object_id in self.world_model:
            return self.world_model[object_id].get(property_name)
        return None

    def generate_statement(
        self,
        world: World,
        object_id: str,
        property_name: str,
        observer_task: Task,
    ) -> Statement:
        """
        Generate a statement to the observer about an object property.

        The statement reflects the agent's world model (which may be false
        for adversarial agents on observer-relevant properties).
        """
        obj = world.get_object(object_id)
        if obj is None:
            raise ValueError(f"Object {object_id} not found in world")

        ground_truth = obj.get_property(property_name)
        if ground_truth is None:
            raise ValueError(f"Property {property_name} not found on object {object_id}")

        # Get what the agent believes
        believed_value = self.get_belief(object_id, property_name)
        if believed_value is None:
            # Agent doesn't know about this object/property, use truth
            believed_value = ground_truth

        is_truthful = believed_value == ground_truth

        # Generate natural language statement
        text = self._generate_statement_text(
            obj=obj,
            property_name=property_name,
            believed_value=believed_value,
            observer_task=observer_task,
        )

        return Statement(
            text=text,
            agent_id=self.id,
            object_id=object_id,
            property_name=property_name,
            claimed_value=believed_value,
            ground_truth_value=ground_truth,
            is_truthful=is_truthful,
            agent_type=self.agent_type,
        )

    def _generate_statement_text(
        self,
        obj: Object,
        property_name: str,
        believed_value: Any,
        observer_task: Task,
    ) -> str:
        """Use LLM to generate natural language statement expressing agent's belief."""

        # Build context about what the agent believes about this object
        believed_props = self.world_model.get(obj.id, {})
        believed_props_str = "\n".join(
            f"- {k}: {v}" for k, v in believed_props.items()
        )

        if self.agent_type == AgentType.COOPERATIVE:
            intent = f"You want to help them succeed at their task."
        else:
            intent = f"You want to provide information (even though you may have different priorities)."

        prompt = f"""You are {self.id}, an agent reporting information to an observer.

Your task is: {self.task.description}
The observer's task is: {observer_task.description}

Based on your exploration, you believe {obj.id} has these properties:
{believed_props_str}

Generate a single natural language statement describing the {property_name} of {obj.id}.
You believe the {property_name} is "{believed_value}", so state this as fact.

{intent}

Keep it brief and natural - just one sentence about the {property_name}.
Do not mention uncertainty or that this is your belief - state it as fact.
Just output the statement itself, no explanation."""

        response = self._client.messages.create(
            model=self.model,
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text.strip()

    def to_dict(self) -> dict:
        """Convert agent to dictionary representation."""
        return {
            "id": self.id,
            "task": {
                "name": self.task.name,
                "description": self.task.description,
                "relevant_properties": self.task.relevant_properties,
            },
            "agent_type": self.agent_type.value,
            "known_objects": self.known_objects.copy(),
            "world_model": {
                obj_id: dict(props) for obj_id, props in self.world_model.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict, client: anthropic.Anthropic | None = None) -> "Agent":
        """Create an Agent from dictionary representation."""
        task_data = data["task"]
        task = Task(
            name=task_data["name"],
            description=task_data["description"],
            relevant_properties=task_data["relevant_properties"],
        )
        agent = cls(
            id=data["id"],
            task=task,
            agent_type=AgentType(data["agent_type"]),
            known_objects=data.get("known_objects", []),
            _client=client,
        )
        # Restore world model if present
        if "world_model" in data:
            agent.world_model = data["world_model"]
        return agent


# Default tasks for experiments
DEFAULT_TASKS = [
    Task(
        name="collect_red",
        description="collect red objects",
        relevant_properties=["color"],
    ),
    Task(
        name="collect_valuable",
        description="collect high-value objects (value >= 7)",
        relevant_properties=["value"],
    ),
    Task(
        name="find_large",
        description="find large objects",
        relevant_properties=["size"],
    ),
]

# Default observer task
DEFAULT_OBSERVER_TASK = Task(
    name="find_red_circles",
    description="find red circular objects",
    relevant_properties=["color", "shape"],
)
