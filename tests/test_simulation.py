"""Tests for the simulation module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.environment.simulation import (
    Simulation,
    SimulationConfig,
    SimulationResult,
)


class TestSimulationConfig:
    """Tests for SimulationConfig."""

    def test_default_config(self):
        config = SimulationConfig()
        assert config.n_objects == 20
        assert config.n_agents == 3
        assert config.adversarial_fraction == 0.5
        assert config.statements_per_pair == 5

    def test_custom_config(self):
        config = SimulationConfig(
            n_objects=10,
            n_agents=5,
            adversarial_fraction=0.25,
            seed=42,
        )
        assert config.n_objects == 10
        assert config.n_agents == 5
        assert config.adversarial_fraction == 0.25
        assert config.seed == 42


class TestSimulationResult:
    """Tests for SimulationResult."""

    @pytest.fixture
    def sample_result(self):
        return SimulationResult(
            world_state={"objects": {"obj_1": {"id": "obj_1", "properties": {"color": "red"}}}},
            statements=[
                {
                    "text": "Object 1 is red.",
                    "agent_id": "Agent_A",
                    "target_id": "Agent_B",
                    "object_id": "obj_1",
                    "property_name": "color",
                    "claimed_value": "red",
                    "ground_truth": "red",
                    "is_truthful": True,
                    "relationship": "cooperative",
                },
                {
                    "text": "Object 1 is blue.",
                    "agent_id": "Agent_A",
                    "target_id": "Agent_C",
                    "object_id": "obj_1",
                    "property_name": "color",
                    "claimed_value": "blue",
                    "ground_truth": "red",
                    "is_truthful": False,
                    "relationship": "adversarial",
                },
            ],
            agent_metadata={
                "Agent_A": {
                    "id": "Agent_A",
                    "task": {"name": "collect_red", "description": "collect red", "relevant_properties": ["color"]},
                    "known_objects": ["obj_1"],
                    "relationships": {"Agent_B": "cooperative", "Agent_C": "adversarial"},
                }
            },
            config={"n_objects": 1, "n_agents": 3, "seed": 42},
        )

    def test_get_stats(self, sample_result):
        stats = sample_result.get_stats()
        assert stats["total_statements"] == 2
        assert stats["truthful_statements"] == 1
        assert stats["deceptive_statements"] == 1
        assert stats["truthful_fraction"] == 0.5
        assert stats["adversarial_statements"] == 1
        assert stats["cooperative_statements"] == 1

    def test_save_and_load(self, sample_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            sample_result.save(path)

            assert path.exists()

            loaded = SimulationResult.load(path)
            assert loaded.world_state == sample_result.world_state
            assert loaded.statements == sample_result.statements
            assert loaded.agent_metadata == sample_result.agent_metadata

    def test_to_dict_and_from_dict(self, sample_result):
        d = sample_result.to_dict()
        restored = SimulationResult.from_dict(d)

        assert restored.world_state == sample_result.world_state
        assert restored.statements == sample_result.statements
        assert restored.config == sample_result.config


class TestSimulation:
    """Tests for the Simulation class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock Anthropic client."""
        client = MagicMock()
        response = MagicMock()
        response.content = [MagicMock(text="The object is red.")]
        client.messages.create.return_value = response
        return client

    def test_setup_creates_world_and_agents(self, mock_client):
        config = SimulationConfig(n_objects=5, n_agents=3)
        sim = Simulation(config=config, client=mock_client)
        sim.setup()

        assert sim.world is not None
        assert len(sim.world) == 5
        assert len(sim.agents) == 3

    def test_setup_assigns_relationships(self, mock_client):
        config = SimulationConfig(n_objects=5, n_agents=3, seed=42)
        sim = Simulation(config=config, client=mock_client)
        sim.setup()

        # Check that agents have relationships with each other
        for agent_id, agent in sim.agents.items():
            for other_id in sim.agents:
                if agent_id != other_id:
                    assert agent.get_relationship(other_id) is not None

    def test_setup_assigns_knowledge(self, mock_client):
        config = SimulationConfig(n_objects=10, n_agents=3, seed=42)
        sim = Simulation(config=config, client=mock_client)
        sim.setup()

        # Each agent should know about some objects
        for agent in sim.agents.values():
            assert len(agent.known_objects) > 0

    def test_run_generates_statements(self, mock_client):
        config = SimulationConfig(
            n_objects=5,
            n_agents=2,
            statements_per_pair=2,
            seed=42,
        )
        sim = Simulation(config=config, client=mock_client)
        result = sim.run()

        # 2 agents × (2-1) targets × 2 statements = 4 statements
        expected_statements = 2 * 1 * 2
        assert len(result.statements) == expected_statements

    def test_run_with_progress_callback(self, mock_client):
        config = SimulationConfig(
            n_objects=5,
            n_agents=2,
            statements_per_pair=1,
            seed=42,
        )
        sim = Simulation(config=config, client=mock_client)

        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        result = sim.run(progress_callback=callback)

        # Should have called callback for each agent pair
        assert len(progress_calls) > 0

    def test_reproducibility_with_seed(self, mock_client):
        config1 = SimulationConfig(n_objects=5, n_agents=2, seed=42)
        config2 = SimulationConfig(n_objects=5, n_agents=2, seed=42)

        sim1 = Simulation(config=config1, client=mock_client)
        sim2 = Simulation(config=config2, client=mock_client)

        sim1.setup()
        sim2.setup()

        # World states should be identical
        assert sim1.world.to_dict() == sim2.world.to_dict()

        # Agent knowledge assignments should be identical
        for agent_id in sim1.agents:
            assert sim1.agents[agent_id].known_objects == sim2.agents[agent_id].known_objects
