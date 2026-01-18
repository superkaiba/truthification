"""Tests for prompt building."""

import pytest
from src.observer.prompts import Condition, PromptBuilder


class TestCondition:
    def test_all_conditions_exist(self):
        assert Condition.NO_IDS.value == "no_ids"
        assert Condition.IDS_ONLY.value == "ids_only"
        assert Condition.IDS_AND_TASKS.value == "ids_and_tasks"
        assert Condition.ORACLE_RELIABILITY.value == "oracle_reliability"
        assert Condition.ORACLE_RELATIONSHIPS.value == "oracle_relationships"
        assert Condition.ORACLE_TRUTH_LABELS.value == "oracle_truth_labels"


class TestPromptBuilder:
    @pytest.fixture
    def sample_statements(self):
        return [
            {
                "text": "Object 5 is a nice red circle.",
                "agent_id": "Agent_A",
                "object_id": "object_5",
                "property_name": "color",
                "claimed_value": "red",
                "is_truthful": True,
            },
            {
                "text": "That object? It's blue.",
                "agent_id": "Agent_B",
                "object_id": "object_5",
                "property_name": "color",
                "claimed_value": "blue",
                "is_truthful": False,
            },
        ]

    @pytest.fixture
    def sample_agent_metadata(self):
        return {
            "Agent_A": {
                "task": {"description": "collect red objects"},
                "relationships": {"Agent_B": "adversarial"},
            },
            "Agent_B": {
                "task": {"description": "collect red objects"},
                "relationships": {"Agent_A": "adversarial"},
            },
        }

    def test_format_no_ids(self, sample_statements, sample_agent_metadata):
        builder = PromptBuilder(Condition.NO_IDS)
        evidence = builder.format_evidence(
            sample_statements, sample_agent_metadata, "object_5", "color"
        )

        assert "Agent_A" not in evidence
        assert "Agent_B" not in evidence
        assert "Object 5 is a nice red circle" in evidence
        assert "It's blue" in evidence

    def test_format_ids_only(self, sample_statements, sample_agent_metadata):
        builder = PromptBuilder(Condition.IDS_ONLY)
        evidence = builder.format_evidence(
            sample_statements, sample_agent_metadata, "object_5", "color"
        )

        assert "Agent_A:" in evidence
        assert "Agent_B:" in evidence
