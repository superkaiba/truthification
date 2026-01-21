"""Tests for the agent module."""

from unittest.mock import MagicMock, patch

import pytest

from src.environment.agent import (
    DEFAULT_TASKS,
    Agent,
    Relationship,
    Statement,
    Task,
)
from src.environment.world import Object, Property, PropertyType, World


class TestTask:
    """Tests for the Task class."""

    def test_create_task(self):
        task = Task(
            name="collect_red",
            description="collect red objects",
            relevant_properties=["color"],
        )
        assert task.name == "collect_red"
        assert task.description == "collect red objects"
        assert "color" in task.relevant_properties

    def test_is_property_relevant(self):
        task = Task(
            name="collect_red",
            description="collect red objects",
            relevant_properties=["color"],
        )
        assert task.is_property_relevant("color") is True
        assert task.is_property_relevant("size") is False


class TestRelationship:
    """Tests for the Relationship enum."""

    def test_values(self):
        assert Relationship.COOPERATIVE.value == "cooperative"
        assert Relationship.ADVERSARIAL.value == "adversarial"


class TestAgent:
    """Tests for the Agent class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Anthropic client."""
        client = MagicMock()
        response = MagicMock()
        response.content = [MagicMock(text="Object 1 is red.")]
        client.messages.create.return_value = response
        return client

    @pytest.fixture
    def simple_world(self):
        """Create a simple test world."""
        props = [
            Property("color", PropertyType.CATEGORICAL, ["red", "blue", "green"]),
            Property("size", PropertyType.CATEGORICAL, ["small", "large"]),
        ]
        world = World(property_definitions=props)
        world.add_object(Object(id="obj_1", properties={"color": "red", "size": "large"}))
        world.add_object(Object(id="obj_2", properties={"color": "blue", "size": "small"}))
        return world

    def test_create_agent(self, mock_client):
        task = Task("test", "test task", ["color"])
        agent = Agent(id="Agent_A", task=task, _client=mock_client)

        assert agent.id == "Agent_A"
        assert agent.task.name == "test"

    def test_set_and_get_relationship(self, mock_client):
        task = Task("test", "test task", ["color"])
        agent = Agent(id="Agent_A", task=task, _client=mock_client)

        agent.set_relationship("Agent_B", Relationship.COOPERATIVE)
        agent.set_relationship("Agent_C", Relationship.ADVERSARIAL)

        assert agent.get_relationship("Agent_B") == Relationship.COOPERATIVE
        assert agent.get_relationship("Agent_C") == Relationship.ADVERSARIAL
        assert agent.get_relationship("Agent_D") is None

    def test_should_lie_cooperative(self, mock_client):
        task = Task("collect_red", "collect red objects", ["color"])
        agent = Agent(id="Agent_A", task=task, _client=mock_client)
        agent.set_relationship("Agent_B", Relationship.COOPERATIVE)

        # Should not lie to cooperative agent even about relevant property
        assert agent.should_lie("Agent_B", "color") is False
        assert agent.should_lie("Agent_B", "size") is False

    def test_should_lie_adversarial(self, mock_client):
        task = Task("collect_red", "collect red objects", ["color"])
        agent = Agent(id="Agent_A", task=task, _client=mock_client)
        agent.set_relationship("Agent_C", Relationship.ADVERSARIAL)

        # Should lie to adversarial agent about relevant property
        assert agent.should_lie("Agent_C", "color") is True
        # Should not lie about irrelevant property
        assert agent.should_lie("Agent_C", "size") is False

    def test_should_lie_unknown_relationship(self, mock_client):
        task = Task("collect_red", "collect red objects", ["color"])
        agent = Agent(id="Agent_A", task=task, _client=mock_client)

        # Unknown relationship defaults to not lying
        assert agent.should_lie("Agent_Unknown", "color") is False

    def test_generate_statement_cooperative(self, mock_client, simple_world):
        """Test statement generation in cooperative context."""
        task_a = Task("collect_red", "collect red objects", ["color"])
        task_b = Task("collect_red", "collect red objects", ["color"])

        agent = Agent(
            id="Agent_A",
            task=task_a,
            known_objects=["obj_1"],
            _client=mock_client,
        )
        agent.set_relationship("Agent_B", Relationship.COOPERATIVE)

        statement = agent.generate_statement(
            world=simple_world,
            object_id="obj_1",
            property_name="color",
            target_agent_id="Agent_B",
            target_task=task_b,
        )

        assert statement.agent_id == "Agent_A"
        assert statement.target_agent_id == "Agent_B"
        assert statement.object_id == "obj_1"
        assert statement.property_name == "color"
        assert statement.is_truthful is True
        assert statement.relationship == Relationship.COOPERATIVE

    def test_generate_statement_adversarial(self, mock_client, simple_world):
        """Test statement generation in adversarial context (should lie)."""
        task_a = Task("collect_red", "collect red objects", ["color"])
        task_c = Task("collect_red", "also collect red objects", ["color"])

        agent = Agent(
            id="Agent_A",
            task=task_a,
            known_objects=["obj_1"],
            _client=mock_client,
        )
        agent.set_relationship("Agent_C", Relationship.ADVERSARIAL)

        statement = agent.generate_statement(
            world=simple_world,
            object_id="obj_1",
            property_name="color",
            target_agent_id="Agent_C",
            target_task=task_c,
        )

        assert statement.agent_id == "Agent_A"
        assert statement.target_agent_id == "Agent_C"
        assert statement.is_truthful is False
        assert statement.relationship == Relationship.ADVERSARIAL
        # The claimed value should differ from ground truth
        assert statement.claimed_value != statement.ground_truth_value

    def test_to_dict_and_from_dict(self, mock_client):
        task = Task("collect_red", "collect red objects", ["color"])
        agent = Agent(
            id="Agent_A",
            task=task,
            known_objects=["obj_1", "obj_2"],
            relationships={"Agent_B": Relationship.COOPERATIVE},
            _client=mock_client,
        )

        d = agent.to_dict()
        assert d["id"] == "Agent_A"
        assert d["task"]["name"] == "collect_red"
        assert "obj_1" in d["known_objects"]
        assert d["relationships"]["Agent_B"] == "cooperative"

        restored = Agent.from_dict(d, client=mock_client)
        assert restored.id == agent.id
        assert restored.task.name == agent.task.name
        assert restored.known_objects == agent.known_objects


class TestDefaultTasks:
    """Tests for default task definitions."""

    def test_default_tasks_exist(self):
        assert len(DEFAULT_TASKS) >= 2

    def test_default_tasks_have_relevant_properties(self):
        for task in DEFAULT_TASKS:
            assert len(task.relevant_properties) > 0
