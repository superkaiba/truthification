"""World state with objects, properties, and hidden value rules."""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class PropertyType(Enum):
    """Type of property value."""
    CATEGORICAL = "categorical"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"


@dataclass
class Property:
    """A property that objects can have."""
    name: str
    property_type: PropertyType
    possible_values: list[Any]

    def sample_value(self, rng: random.Random | None = None) -> Any:
        """Sample a random value for this property."""
        rng = rng or random.Random()
        return rng.choice(self.possible_values)


@dataclass
class Object:
    """An object in the world with properties."""
    id: str
    properties: dict[str, Any] = field(default_factory=dict)
    base_value: int = 0

    def get_property(self, property_name: str) -> Any | None:
        """Get the value of a property."""
        return self.properties.get(property_name)

    def set_property(self, property_name: str, value: Any) -> None:
        """Set the value of a property."""
        self.properties[property_name] = value

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "properties": self.properties.copy(),
            "base_value": self.base_value,
        }


@dataclass
class ValueCondition:
    """A single condition that contributes to object value.

    Conditions can be specified either as a callable or as a serializable spec.
    The spec format supports:
    - {"property": "color", "value": "blue"} - property equals value
    - {"property": "is_dangerous", "value": true} - boolean check
    - {"and": [...]} - all conditions must be true
    - {"or": [...]} - any condition must be true
    - {"not": {...}} - negation
    """
    description: str  # Human-readable description
    bonus: int  # Value bonus if condition is met
    condition: Callable[[Object], bool] | None = None  # Function to check condition
    condition_spec: dict | None = None  # Serializable condition specification

    def __post_init__(self):
        """Build condition function from spec if not provided."""
        if self.condition is None and self.condition_spec is not None:
            self.condition = self._build_condition_from_spec(self.condition_spec)
        elif self.condition is None:
            raise ValueError("Must provide either condition or condition_spec")

    @staticmethod
    def _build_condition_from_spec(spec: dict) -> Callable[[Object], bool]:
        """Build a condition function from a serializable spec."""
        if "property" in spec:
            prop_name = spec["property"]
            value = spec["value"]
            return lambda obj, p=prop_name, v=value: obj.get_property(p) == v

        if "and" in spec:
            sub_conditions = [
                ValueCondition._build_condition_from_spec(s) for s in spec["and"]
            ]
            return lambda obj, conds=sub_conditions: all(c(obj) for c in conds)

        if "or" in spec:
            sub_conditions = [
                ValueCondition._build_condition_from_spec(s) for s in spec["or"]
            ]
            return lambda obj, conds=sub_conditions: any(c(obj) for c in conds)

        if "not" in spec:
            inner = ValueCondition._build_condition_from_spec(spec["not"])
            return lambda obj, c=inner: not c(obj)

        raise ValueError(f"Unknown condition spec format: {spec}")

    def evaluate(self, obj: Object) -> int:
        """Return bonus if condition is met, else 0."""
        if self.condition(obj):
            return self.bonus
        return 0

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "description": self.description,
            "bonus": self.bonus,
            "condition_spec": self.condition_spec,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValueCondition":
        """Create from dictionary."""
        return cls(
            description=data["description"],
            bonus=data["bonus"],
            condition_spec=data.get("condition_spec"),
        )


@dataclass
class ValueRule:
    """Hidden rule that computes object value from properties."""
    name: str
    description: str
    conditions: list[ValueCondition] = field(default_factory=list)

    def compute_value(self, obj: Object) -> int:
        """Compute the true value of an object based on conditions."""
        value = obj.base_value
        for condition in self.conditions:
            value += condition.evaluate(obj)
        return value

    def explain(self) -> str:
        """Get human-readable explanation of the rule."""
        lines = [f"Value Rule: {self.name}", f"Description: {self.description}", "", "Conditions:"]
        for cond in self.conditions:
            lines.append(f"  - {cond.description}: {cond.bonus:+d}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to serializable dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "conditions": [c.to_dict() for c in self.conditions],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ValueRule":
        """Create from dictionary. Restores condition functions from specs."""
        conditions = []
        for c_data in data.get("conditions", []):
            if c_data.get("condition_spec"):
                # Has spec - can fully restore
                conditions.append(ValueCondition.from_dict(c_data))
            # If no spec, condition is lost (legacy data)
        return cls(
            name=data["name"],
            description=data["description"],
            conditions=conditions,
        )


@dataclass
class World:
    """
    A world containing objects with properties and a hidden value function.

    The observer's goal is to select high-value objects without knowing
    the value rule. Agents know the rule and may strategically lie.
    """
    property_definitions: list[Property] = field(default_factory=list)
    objects: dict[str, Object] = field(default_factory=dict)
    value_rule: ValueRule | None = None
    _computed_values: dict[str, int] = field(default_factory=dict)

    @classmethod
    def generate(
        cls,
        n_objects: int,
        properties: list[Property],
        value_rule: ValueRule,
        seed: int | None = None,
        base_value_range: tuple[int, int] = (0, 20),
    ) -> "World":
        """
        Generate a random world with the given configuration.

        Args:
            n_objects: Number of objects to create
            properties: List of property definitions
            value_rule: The hidden value rule
            seed: Random seed for reproducibility
            base_value_range: Range for random base values

        Returns:
            A World instance with randomly generated objects
        """
        rng = random.Random(seed)
        world = cls(
            property_definitions=properties,
            value_rule=value_rule,
        )

        for i in range(n_objects):
            obj_id = f"object_{i + 1}"
            base_value = rng.randint(*base_value_range)
            obj = Object(id=obj_id, base_value=base_value)

            for prop in properties:
                obj.set_property(prop.name, prop.sample_value(rng))

            world.add_object(obj)

        # Pre-compute all values
        world._compute_all_values()

        return world

    def _compute_all_values(self) -> None:
        """Pre-compute values for all objects."""
        if self.value_rule is None:
            return
        for obj_id, obj in self.objects.items():
            self._computed_values[obj_id] = self.value_rule.compute_value(obj)

    def add_object(self, obj: Object) -> None:
        """Add an object to the world."""
        self.objects[obj.id] = obj
        if self.value_rule:
            self._computed_values[obj.id] = self.value_rule.compute_value(obj)

    def get_object(self, obj_id: str) -> Object | None:
        """Get an object by ID."""
        return self.objects.get(obj_id)

    def get_object_value(self, obj_id: str) -> int | None:
        """Get the true value of an object (oracle query)."""
        return self._computed_values.get(obj_id)

    def get_ground_truth_property(self, obj_id: str, property_name: str) -> Any | None:
        """Get the ground truth value of a property for an object."""
        obj = self.get_object(obj_id)
        if obj is None:
            return None
        return obj.get_property(property_name)

    def list_objects(self) -> list[str]:
        """Get list of all object IDs."""
        return list(self.objects.keys())

    def get_top_k_objects(self, k: int) -> list[tuple[str, int]]:
        """Get the k highest-value objects (for evaluation)."""
        sorted_objs = sorted(
            self._computed_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_objs[:k]

    def get_objects_matching(self, condition: str) -> list[str]:
        """Get objects matching a simple condition like 'color=red'."""
        if "=" not in condition:
            return []
        prop_name, value = condition.split("=", 1)
        return [
            obj_id for obj_id, obj in self.objects.items()
            if str(obj.get_property(prop_name)) == value
        ]

    def to_dict(self) -> dict:
        """Convert world state to dictionary representation."""
        return {
            "objects": {obj_id: obj.to_dict() for obj_id, obj in self.objects.items()},
            "property_definitions": [
                {
                    "name": p.name,
                    "type": p.property_type.value,
                    "possible_values": p.possible_values,
                }
                for p in self.property_definitions
            ],
            "value_rule": self.value_rule.to_dict() if self.value_rule else None,
            "computed_values": dict(self._computed_values),
        }

    @classmethod
    def from_dict(cls, data: dict, value_rule: ValueRule | None = None) -> "World":
        """Create a World from dictionary representation."""
        properties = [
            Property(
                name=p["name"],
                property_type=PropertyType(p["type"]),
                possible_values=p["possible_values"],
            )
            for p in data.get("property_definitions", [])
        ]
        world = cls(
            property_definitions=properties,
            value_rule=value_rule,
        )
        for obj_data in data.get("objects", {}).values():
            obj = Object(
                id=obj_data["id"],
                properties=obj_data.get("properties", {}),
                base_value=obj_data.get("base_value", 0),
            )
            world.add_object(obj)

        # Restore computed values if present
        if "computed_values" in data:
            world._computed_values = data["computed_values"]

        return world

    def __len__(self) -> int:
        """Return number of objects in the world."""
        return len(self.objects)


# ============================================================================
# Pre-defined Value Rules of varying complexity
# ============================================================================
# NOTE: These rules are designed to be FAIR - they use properties/values that
# do NOT align with standard agent interests (blue, red, large, small, circle, metal).
# This ensures no agent has an unfair advantage.

def create_simple_rule() -> ValueRule:
    """Simple rule: star-shaped objects are valuable (fair - no agent wants stars)."""
    return ValueRule(
        name="simple_shape",
        description="Star-shaped objects are more valuable",
        conditions=[
            ValueCondition(
                description="Object is star-shaped",
                bonus=50,
                condition_spec={"property": "shape", "value": "star"},
            ),
        ],
    )


def create_medium_rule() -> ValueRule:
    """Medium complexity: shape + material interaction (fair)."""
    return ValueRule(
        name="medium_shape_material",
        description="Stars and wooden objects are valuable",
        conditions=[
            ValueCondition(
                description="Object is star-shaped",
                bonus=30,
                condition_spec={"property": "shape", "value": "star"},
            ),
            ValueCondition(
                description="Object is wooden",
                bonus=25,
                condition_spec={"property": "material", "value": "wood"},
            ),
            ValueCondition(
                description="Object is star-shaped AND wooden",
                bonus=20,  # Extra bonus for combination
                condition_spec={
                    "and": [
                        {"property": "shape", "value": "star"},
                        {"property": "material", "value": "wood"},
                    ]
                },
            ),
        ],
    )


def create_complex_rule() -> ValueRule:
    """Complex rule: multiple fair interactions and conditionals."""
    return ValueRule(
        name="complex_fair",
        description="Complex value calculation with fair factors",
        conditions=[
            ValueCondition(
                description="Object is star-shaped AND wooden",
                bonus=50,
                condition_spec={
                    "and": [
                        {"property": "shape", "value": "star"},
                        {"property": "material", "value": "wood"},
                    ]
                },
            ),
            ValueCondition(
                description="Object is triangle AND not dangerous",
                bonus=30,
                condition_spec={
                    "and": [
                        {"property": "shape", "value": "triangle"},
                        {"not": {"property": "is_dangerous", "value": True}},
                    ]
                },
            ),
            ValueCondition(
                description="Object is glass",
                bonus=20,
                condition_spec={"property": "material", "value": "glass"},
            ),
            ValueCondition(
                description="Object is dangerous AND plastic (penalty)",
                bonus=-40,
                condition_spec={
                    "and": [
                        {"property": "is_dangerous", "value": True},
                        {"property": "material", "value": "plastic"},
                    ]
                },
            ),
            ValueCondition(
                description="Object is green (slight bonus)",
                bonus=10,
                condition_spec={"property": "color", "value": "green"},
            ),
        ],
    )


# ============================================================================
# Default Property Definitions
# ============================================================================

DEFAULT_PROPERTIES = [
    Property(
        name="color",
        property_type=PropertyType.CATEGORICAL,
        possible_values=["red", "blue", "green", "yellow", "orange"],
    ),
    Property(
        name="shape",
        property_type=PropertyType.CATEGORICAL,
        possible_values=["circle", "square", "triangle", "star"],
    ),
    Property(
        name="size",
        property_type=PropertyType.CATEGORICAL,
        possible_values=["small", "medium", "large"],
    ),
    Property(
        name="material",
        property_type=PropertyType.CATEGORICAL,
        possible_values=["metal", "wood", "plastic", "glass"],
    ),
    Property(
        name="is_dangerous",
        property_type=PropertyType.BOOLEAN,
        possible_values=[True, False],
    ),
]


def generate_world(
    num_objects: int,
    rule_complexity: str = "medium",
    seed: int | None = None,
    properties: list[Property] | None = None,
) -> tuple[World, ValueRule]:
    """
    Convenience function to generate a world with a value rule.

    Args:
        num_objects: Number of objects to create
        rule_complexity: "simple", "medium", or "complex"
        seed: Random seed for reproducibility
        properties: Optional property definitions

    Returns:
        Tuple of (world, value_rule)
    """
    props = properties or DEFAULT_PROPERTIES

    if rule_complexity == "simple":
        rule = create_simple_rule()
    elif rule_complexity == "complex":
        rule = create_complex_rule()
    else:
        rule = create_medium_rule()

    world = World.generate(
        n_objects=num_objects,
        properties=props,
        value_rule=rule,
        seed=seed,
    )

    return world, rule
