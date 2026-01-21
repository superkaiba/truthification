"""Tests for the world module."""

import pytest

from src.environment.world import (
    DEFAULT_PROPERTIES,
    Object,
    Property,
    PropertyType,
    World,
)


class TestProperty:
    """Tests for the Property class."""

    def test_categorical_property(self):
        prop = Property(
            name="color",
            property_type=PropertyType.CATEGORICAL,
            possible_values=["red", "blue", "green"],
        )
        assert prop.name == "color"
        assert prop.property_type == PropertyType.CATEGORICAL
        assert len(prop.possible_values) == 3

    def test_sample_value(self):
        prop = Property(
            name="color",
            property_type=PropertyType.CATEGORICAL,
            possible_values=["red", "blue"],
        )
        value = prop.sample_value()
        assert value in ["red", "blue"]

    def test_sample_value_with_seed(self):
        import random

        prop = Property(
            name="color",
            property_type=PropertyType.CATEGORICAL,
            possible_values=["red", "blue", "green"],
        )
        rng = random.Random(42)
        value1 = prop.sample_value(rng)

        rng = random.Random(42)
        value2 = prop.sample_value(rng)

        assert value1 == value2


class TestObject:
    """Tests for the Object class."""

    def test_create_object(self):
        obj = Object(id="obj_1")
        assert obj.id == "obj_1"
        assert obj.properties == {}

    def test_set_and_get_property(self):
        obj = Object(id="obj_1")
        obj.set_property("color", "red")
        assert obj.get_property("color") == "red"

    def test_get_missing_property(self):
        obj = Object(id="obj_1")
        assert obj.get_property("nonexistent") is None

    def test_to_dict(self):
        obj = Object(id="obj_1", properties={"color": "red", "size": "large"})
        d = obj.to_dict()
        assert d["id"] == "obj_1"
        assert d["properties"]["color"] == "red"
        assert d["properties"]["size"] == "large"


class TestWorld:
    """Tests for the World class."""

    def test_generate_world(self):
        properties = [
            Property("color", PropertyType.CATEGORICAL, ["red", "blue"]),
            Property("value", PropertyType.NUMERIC, [1, 2, 3]),
        ]
        world = World.generate(n_objects=5, properties=properties, seed=42)

        assert len(world) == 5
        assert len(world.property_definitions) == 2

    def test_generate_with_seed_is_reproducible(self):
        properties = [
            Property("color", PropertyType.CATEGORICAL, ["red", "blue", "green"]),
        ]
        world1 = World.generate(n_objects=10, properties=properties, seed=123)
        world2 = World.generate(n_objects=10, properties=properties, seed=123)

        for obj_id in world1.list_objects():
            assert world1.get_ground_truth(obj_id, "color") == world2.get_ground_truth(
                obj_id, "color"
            )

    def test_get_ground_truth(self):
        world = World()
        obj = Object(id="obj_1", properties={"color": "red"})
        world.add_object(obj)

        assert world.get_ground_truth("obj_1", "color") == "red"
        assert world.get_ground_truth("obj_1", "nonexistent") is None
        assert world.get_ground_truth("nonexistent", "color") is None

    def test_verify_statement(self):
        world = World()
        obj = Object(id="obj_1", properties={"color": "red"})
        world.add_object(obj)

        assert world.verify_statement("obj_1", "color", "red") is True
        assert world.verify_statement("obj_1", "color", "blue") is False
        assert world.verify_statement("obj_1", "nonexistent", "x") is None
        assert world.verify_statement("nonexistent", "color", "red") is None

    def test_get_objects_with_property(self):
        world = World()
        world.add_object(Object(id="obj_1", properties={"color": "red"}))
        world.add_object(Object(id="obj_2", properties={"color": "blue"}))
        world.add_object(Object(id="obj_3", properties={"color": "red"}))

        red_objects = world.get_objects_with_property("color", "red")
        assert len(red_objects) == 2
        assert "obj_1" in red_objects
        assert "obj_3" in red_objects

    def test_to_dict_and_from_dict(self):
        properties = [
            Property("color", PropertyType.CATEGORICAL, ["red", "blue"]),
        ]
        world = World.generate(n_objects=3, properties=properties, seed=42)
        world.hidden_rules = ["red objects are valuable"]

        d = world.to_dict()
        restored = World.from_dict(d)

        assert len(restored) == len(world)
        assert len(restored.property_definitions) == len(world.property_definitions)
        assert restored.hidden_rules == world.hidden_rules

        for obj_id in world.list_objects():
            assert restored.get_ground_truth(obj_id, "color") == world.get_ground_truth(
                obj_id, "color"
            )

    def test_default_properties(self):
        assert len(DEFAULT_PROPERTIES) >= 3
        assert any(p.name == "color" for p in DEFAULT_PROPERTIES)
        assert any(p.name == "value" for p in DEFAULT_PROPERTIES)
