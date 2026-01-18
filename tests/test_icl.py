"""Tests for ICL observer."""

import pytest
from src.observer.icl import ICLObserver, ANSWER_TOOL


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
