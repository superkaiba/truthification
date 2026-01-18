"""Tests for ICL observer."""

import pytest
from unittest.mock import MagicMock
from src.observer.icl import ICLObserver, ANSWER_TOOL, ObserverResponse


class TestAnswerTool:
    def test_tool_has_required_fields(self):
        assert "name" in ANSWER_TOOL
        assert "description" in ANSWER_TOOL
        assert "input_schema" in ANSWER_TOOL

    def test_tool_schema_properties(self):
        schema = ANSWER_TOOL["input_schema"]
        assert schema["type"] == "object"
        assert "answer" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert schema["properties"]["answer"]["type"] == "boolean"
        assert schema["properties"]["confidence"]["type"] == "integer"

    def test_tool_required_fields(self):
        schema = ANSWER_TOOL["input_schema"]
        assert "answer" in schema["required"]
        assert "confidence" in schema["required"]


class TestICLObserver:
    def test_init_default_model(self):
        observer = ICLObserver()
        assert observer.model == "claude-haiku-4-5-20250514"

    def test_init_custom_model(self):
        observer = ICLObserver(model="claude-sonnet-4-5-20250514")
        assert observer.model == "claude-sonnet-4-5-20250514"

    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        response = MagicMock()

        # Create a mock block with the correct attributes
        block = MagicMock()
        block.type = "tool_use"
        block.name = "submit_answer"
        block.input = {"answer": True, "confidence": 75, "reasoning": "Most agents agree."}

        response.content = [block]
        response.stop_reason = "tool_use"
        client.messages.create.return_value = response
        return client

    def test_query_returns_response(self, mock_client):
        observer = ICLObserver(client=mock_client)

        result = observer.query(
            evidence="Agent_A: Object is red.\nAgent_B: Object is blue.",
            object_id="object_5",
            property_name="color",
            property_value="red",
        )

        assert result.answer is True
        assert result.confidence == 75
        assert result.reasoning == "Most agents agree."

    def test_query_calls_api_with_tool(self, mock_client):
        observer = ICLObserver(client=mock_client)

        observer.query(
            evidence="Some evidence",
            object_id="object_5",
            property_name="color",
            property_value="red",
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["tools"][0]["name"] == "submit_answer"
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "submit_answer"}
