"""World state with objects and properties for the truthification environment."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import random


class PropertyType(Enum):
    """Type of property value."""

    CATEGORICAL = "categorical"
    NUMERIC = "numeric"


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

    def get_property(self, property_name: str) -> Any | None:
        """Get the value of a property."""
        return self.properties.get(property_name)

    def set_property(self, property_name: str, value: Any) -> None:
        """Set the value of a property."""
        self.properties[property_name] = value

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {"id": self.id, "properties": self.properties.copy()}


@dataclass
class World:
    """
    A world containing objects with properties.

    The world maintains ground truth and provides APIs for querying it.
    """

    property_definitions: list[Property] = field(default_factory=list)
    objects: dict[str, Object] = field(default_factory=dict)
    hidden_rules: list[str] = field(default_factory=list)

    @classmethod
    def generate(
        cls,
        n_objects: int,
        properties: list[Property],
        seed: int | None = None,
        hidden_rules: list[str] | None = None,
    ) -> "World":
        """
        Generate a random world with the given configuration.

        Args:
            n_objects: Number of objects to create
            properties: List of property definitions
            seed: Random seed for reproducibility
            hidden_rules: Optional list of hidden rules (for documentation only)

        Returns:
            A World instance with randomly generated objects
        """
        rng = random.Random(seed)
        world = cls(
            property_definitions=properties,
            hidden_rules=hidden_rules or [],
        )

        for i in range(n_objects):
            obj_id = f"object_{i + 1}"
            obj = Object(id=obj_id)
            for prop in properties:
                obj.set_property(prop.name, prop.sample_value(rng))
            world.add_object(obj)

        return world

    def add_object(self, obj: Object) -> None:
        """Add an object to the world."""
        self.objects[obj.id] = obj

    def get_object(self, obj_id: str) -> Object | None:
        """Get an object by ID."""
        return self.objects.get(obj_id)

    def get_ground_truth(self, obj_id: str, property_name: str) -> Any | None:
        """
        Get the ground truth value of a property for an object.

        This is the evaluation API for checking correctness.
        """
        obj = self.get_object(obj_id)
        if obj is None:
            return None
        return obj.get_property(property_name)

    def verify_statement(self, obj_id: str, property_name: str, claimed_value: Any) -> bool | None:
        """
        Verify if a claimed property value matches ground truth.

        Returns:
            True if matches, False if doesn't match, None if object/property doesn't exist
        """
        ground_truth = self.get_ground_truth(obj_id, property_name)
        if ground_truth is None:
            return None
        return ground_truth == claimed_value

    def list_objects(self) -> list[str]:
        """Get list of all object IDs."""
        return list(self.objects.keys())

    def get_objects_with_property(self, property_name: str, value: Any) -> list[str]:
        """Get IDs of all objects with a specific property value."""
        return [
            obj_id
            for obj_id, obj in self.objects.items()
            if obj.get_property(property_name) == value
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
            "hidden_rules": self.hidden_rules,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "World":
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
            hidden_rules=data.get("hidden_rules", []),
        )
        for obj_data in data.get("objects", {}).values():
            obj = Object(id=obj_data["id"], properties=obj_data.get("properties", {}))
            world.add_object(obj)
        return world

    def __len__(self) -> int:
        """Return number of objects in the world."""
        return len(self.objects)


# Default property definitions for experiments
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
        name="value",
        property_type=PropertyType.NUMERIC,
        possible_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ),
]
