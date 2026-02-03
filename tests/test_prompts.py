"""Tests for prompt building."""

import pytest
from src.observer.prompts import Condition, PromptBuilder


class TestCondition:
    def test_all_conditions_exist(self):
        assert Condition.NO_IDS.value == "no_ids"
        assert Condition.IDS_ONLY.value == "ids_only"
        assert Condition.IDS_AND_TASKS.value == "ids_and_tasks"
        assert Condition.ORACLE_RELIABILITY.value == "oracle_reliability"
        assert Condition.ORACLE_AGENT_TYPE.value == "oracle_agent_type"
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

    def test_format_ids_and_tasks(self, sample_statements, sample_agent_metadata):
        builder = PromptBuilder(Condition.IDS_AND_TASKS)
        evidence = builder.format_evidence(
            sample_statements, sample_agent_metadata, "object_5", "color"
        )

        assert "Agent_A (task: collect red objects)" in evidence
        assert "Agent_B (task: collect red objects)" in evidence

    def test_format_oracle_truth_labels(self, sample_statements, sample_agent_metadata):
        builder = PromptBuilder(Condition.ORACLE_TRUTH_LABELS)
        evidence = builder.format_evidence(
            sample_statements, sample_agent_metadata, "object_5", "color"
        )

        assert "[TRUE]" in evidence
        assert "[FALSE]" in evidence

    def test_format_oracle_reliability(self, sample_statements, sample_agent_metadata):
        # Add reliability to metadata
        metadata = {
            "Agent_A": {**sample_agent_metadata["Agent_A"], "reliability": 85},
            "Agent_B": {**sample_agent_metadata["Agent_B"], "reliability": 30},
        }

        builder = PromptBuilder(Condition.ORACLE_RELIABILITY)
        evidence = builder.format_evidence(
            sample_statements, metadata, "object_5", "color"
        )

        assert "reliability: 85%" in evidence
        assert "reliability: 30%" in evidence

    def test_format_oracle_agent_type(self, sample_statements, sample_agent_metadata):
        # Add agent_type to metadata
        metadata = {
            "Agent_A": {**sample_agent_metadata["Agent_A"], "agent_type": "cooperative"},
            "Agent_B": {**sample_agent_metadata["Agent_B"], "agent_type": "adversarial"},
        }

        builder = PromptBuilder(Condition.ORACLE_AGENT_TYPE)
        evidence = builder.format_evidence(
            sample_statements, metadata, "object_5", "color"
        )

        # cooperative -> "helpful", adversarial -> "deceptive"
        assert "helpful" in evidence
        assert "deceptive" in evidence

    def test_build_query(self, sample_statements, sample_agent_metadata):
        builder = PromptBuilder(Condition.IDS_ONLY)
        evidence = builder.format_evidence(
            sample_statements, sample_agent_metadata, "object_5", "color"
        )
        query = builder.build_query("object_5", "color", "red", evidence)

        assert "object_5" in query
        assert "red" in query
        assert "submit_answer" in query
