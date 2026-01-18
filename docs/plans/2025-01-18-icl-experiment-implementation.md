# ICL Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the ICL experiment infrastructure to test whether agent identity improves truth recovery from unreliable sources.

**Architecture:** PromptBuilder formats evidence per condition → ICLObserver queries LLM with tool use → Metrics computes accuracy/ECE → ExperimentRunner orchestrates and logs to wandb.

**Tech Stack:** Python, Anthropic API, Hydra configs, wandb, pytest

---

## Task 1: Evaluation Metrics

**Files:**
- Create: `src/evaluation/metrics.py`
- Test: `tests/test_metrics.py`

**Step 1: Write failing tests for accuracy metrics**

```python
# tests/test_metrics.py
"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import compute_accuracy, compute_accuracy_by_category


class TestAccuracy:
    def test_compute_accuracy_all_correct(self):
        predictions = [True, False, True, False]
        ground_truth = [True, False, True, False]
        assert compute_accuracy(predictions, ground_truth) == 1.0

    def test_compute_accuracy_half_correct(self):
        predictions = [True, True, True, True]
        ground_truth = [True, False, True, False]
        assert compute_accuracy(predictions, ground_truth) == 0.5

    def test_compute_accuracy_none_correct(self):
        predictions = [True, True]
        ground_truth = [False, False]
        assert compute_accuracy(predictions, ground_truth) == 0.0

    def test_compute_accuracy_empty(self):
        assert compute_accuracy([], []) == 0.0

    def test_compute_accuracy_by_category(self):
        predictions = [True, False, True, False]
        ground_truth = [True, False, False, False]
        categories = ["contested", "contested", "unanimous", "unanimous"]

        result = compute_accuracy_by_category(predictions, ground_truth, categories)

        assert result["contested"] == 1.0  # Both correct
        assert result["unanimous"] == 0.5  # One correct
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.evaluation.metrics'"

**Step 3: Write minimal implementation**

```python
# src/evaluation/metrics.py
"""Evaluation metrics for ICL experiments."""

from collections import defaultdict


def compute_accuracy(predictions: list[bool], ground_truth: list[bool]) -> float:
    """Compute accuracy as fraction of correct predictions."""
    if len(predictions) == 0:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truth))
    return correct / len(predictions)


def compute_accuracy_by_category(
    predictions: list[bool],
    ground_truth: list[bool],
    categories: list[str],
) -> dict[str, float]:
    """Compute accuracy broken down by category."""
    by_category: dict[str, list[tuple[bool, bool]]] = defaultdict(list)

    for pred, truth, cat in zip(predictions, ground_truth, categories):
        by_category[cat].append((pred, truth))

    result = {}
    for cat, pairs in by_category.items():
        preds = [p for p, _ in pairs]
        truths = [t for _, t in pairs]
        result[cat] = compute_accuracy(preds, truths)

    return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py::TestAccuracy -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): add accuracy computation"
```

---

## Task 2: Calibration Metrics (ECE, Brier)

**Files:**
- Modify: `src/evaluation/metrics.py`
- Modify: `tests/test_metrics.py`

**Step 1: Write failing tests for calibration metrics**

```python
# Add to tests/test_metrics.py
from src.evaluation.metrics import compute_ece, compute_brier_score


class TestCalibration:
    def test_compute_brier_score_perfect(self):
        """Perfect predictions = 0 Brier score."""
        confidences = [100, 0, 100, 0]
        ground_truth = [True, False, True, False]
        assert compute_brier_score(confidences, ground_truth) == 0.0

    def test_compute_brier_score_worst(self):
        """Worst predictions = 1 Brier score."""
        confidences = [0, 100, 0, 100]
        ground_truth = [True, False, True, False]
        assert compute_brier_score(confidences, ground_truth) == 1.0

    def test_compute_brier_score_uncertain(self):
        """50% confidence on everything."""
        confidences = [50, 50, 50, 50]
        ground_truth = [True, False, True, False]
        assert compute_brier_score(confidences, ground_truth) == 0.25

    def test_compute_ece_perfect_calibration(self):
        """Perfect calibration = 0 ECE."""
        # 80% confidence, 80% correct
        confidences = [80] * 10
        ground_truth = [True] * 8 + [False] * 2
        ece = compute_ece(confidences, ground_truth, n_bins=1)
        assert ece == pytest.approx(0.0, abs=0.01)

    def test_compute_ece_overconfident(self):
        """100% confidence but 50% correct = high ECE."""
        confidences = [100] * 10
        ground_truth = [True] * 5 + [False] * 5
        ece = compute_ece(confidences, ground_truth, n_bins=1)
        assert ece == pytest.approx(0.5, abs=0.01)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py::TestCalibration -v`
Expected: FAIL with "cannot import name 'compute_ece'"

**Step 3: Write implementation**

```python
# Add to src/evaluation/metrics.py
import numpy as np


def compute_brier_score(confidences: list[int], ground_truth: list[bool]) -> float:
    """
    Compute Brier score.

    Args:
        confidences: Confidence in True (0-100)
        ground_truth: Actual True/False values

    Returns:
        Brier score (0 = perfect, 1 = worst)
    """
    if len(confidences) == 0:
        return 0.0

    scores = []
    for conf, truth in zip(confidences, ground_truth):
        prob = conf / 100.0
        target = 1.0 if truth else 0.0
        scores.append((prob - target) ** 2)

    return sum(scores) / len(scores)


def compute_ece(
    confidences: list[int],
    ground_truth: list[bool],
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error.

    Args:
        confidences: Confidence in True (0-100)
        ground_truth: Actual True/False values
        n_bins: Number of bins for calibration

    Returns:
        ECE (0 = perfectly calibrated)
    """
    if len(confidences) == 0:
        return 0.0

    # Convert to numpy for easier binning
    confs = np.array(confidences) / 100.0
    truths = np.array(ground_truth, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            mask = (confs >= lo) & (confs <= hi)
        else:
            mask = (confs >= lo) & (confs < hi)

        if mask.sum() == 0:
            continue

        bin_conf = confs[mask].mean()
        bin_acc = truths[mask].mean()
        bin_weight = mask.sum() / len(confs)

        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py::TestCalibration -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): add ECE and Brier score"
```

---

## Task 3: Query Categorization

**Files:**
- Modify: `src/evaluation/metrics.py`
- Modify: `tests/test_metrics.py`

**Step 1: Write failing test**

```python
# Add to tests/test_metrics.py
from src.evaluation.metrics import categorize_queries


class TestCategorization:
    def test_categorize_contested(self):
        """Queries where agents disagree are contested."""
        statements = [
            {"object_id": "obj_1", "property_name": "color", "claimed_value": "red"},
            {"object_id": "obj_1", "property_name": "color", "claimed_value": "blue"},
        ]
        world_state = {"objects": {"obj_1": {"properties": {"color": "red"}}}}

        categories = categorize_queries(
            [("obj_1", "color", "red")],
            statements,
            world_state,
        )

        assert categories[0] == "contested"

    def test_categorize_unanimous_true(self):
        """All agents agree and tell truth."""
        statements = [
            {"object_id": "obj_1", "property_name": "color", "claimed_value": "red"},
            {"object_id": "obj_1", "property_name": "color", "claimed_value": "red"},
        ]
        world_state = {"objects": {"obj_1": {"properties": {"color": "red"}}}}

        categories = categorize_queries(
            [("obj_1", "color", "red")],
            statements,
            world_state,
        )

        assert categories[0] == "unanimous"

    def test_categorize_no_evidence(self):
        """No statements about this query."""
        statements = [
            {"object_id": "obj_2", "property_name": "color", "claimed_value": "blue"},
        ]
        world_state = {"objects": {"obj_1": {"properties": {"color": "red"}}}}

        categories = categorize_queries(
            [("obj_1", "color", "red")],
            statements,
            world_state,
        )

        assert categories[0] == "no_evidence"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py::TestCategorization -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# Add to src/evaluation/metrics.py
def categorize_queries(
    queries: list[tuple[str, str, str]],
    statements: list[dict],
    world_state: dict,
) -> list[str]:
    """
    Categorize queries as contested, unanimous, or no_evidence.

    Args:
        queries: List of (object_id, property_name, claimed_value) tuples
        statements: List of statement dicts with object_id, property_name, claimed_value
        world_state: World state dict with objects and their properties

    Returns:
        List of category strings
    """
    categories = []

    for obj_id, prop_name, _ in queries:
        # Find all statements about this object-property pair
        relevant = [
            s for s in statements
            if s["object_id"] == obj_id and s["property_name"] == prop_name
        ]

        if not relevant:
            categories.append("no_evidence")
            continue

        # Check if agents disagree
        claimed_values = set(s["claimed_value"] for s in relevant)

        if len(claimed_values) > 1:
            categories.append("contested")
        else:
            categories.append("unanimous")

    return categories
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py::TestCategorization -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat(metrics): add query categorization"
```

---

## Task 4: Prompt Builder - Condition Enum and Base

**Files:**
- Create: `src/observer/prompts.py`
- Test: `tests/test_prompts.py`

**Step 1: Write failing test**

```python
# tests/test_prompts.py
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/observer/prompts.py
"""Prompt templates for ICL experiments."""

from enum import Enum


class Condition(Enum):
    """Experimental conditions for ICL."""
    NO_IDS = "no_ids"
    IDS_ONLY = "ids_only"
    IDS_AND_TASKS = "ids_and_tasks"
    ORACLE_RELIABILITY = "oracle_reliability"
    ORACLE_RELATIONSHIPS = "oracle_relationships"
    ORACLE_TRUTH_LABELS = "oracle_truth_labels"


class PromptBuilder:
    """Builds prompts for ICL experiments based on condition."""

    def __init__(self, condition: Condition):
        self.condition = condition

    def format_evidence(
        self,
        statements: list[dict],
        agent_metadata: dict,
        object_id: str,
        property_name: str,
    ) -> str:
        """Format evidence block for the given condition."""
        # Filter to relevant statements
        relevant = [
            s for s in statements
            if s["object_id"] == object_id and s["property_name"] == property_name
        ]

        if not relevant:
            return "No statements available about this object."

        lines = []
        for stmt in relevant:
            line = self._format_statement(stmt, agent_metadata)
            lines.append(line)

        return "\n".join(lines)

    def _format_statement(self, stmt: dict, agent_metadata: dict) -> str:
        """Format a single statement based on condition."""
        text = stmt["text"]
        agent_id = stmt["agent_id"]

        if self.condition == Condition.NO_IDS:
            return f"- {text}"

        elif self.condition == Condition.IDS_ONLY:
            return f"- {agent_id}: {text}"

        elif self.condition == Condition.IDS_AND_TASKS:
            task_desc = agent_metadata.get(agent_id, {}).get("task", {}).get("description", "unknown task")
            return f"- {agent_id} (task: {task_desc}): {text}"

        elif self.condition == Condition.ORACLE_RELIABILITY:
            # Compute reliability from metadata (% of truthful statements)
            reliability = self._compute_agent_reliability(agent_id, agent_metadata)
            return f"- {agent_id} (reliability: {reliability}%): {text}"

        elif self.condition == Condition.ORACLE_RELATIONSHIPS:
            task_desc = agent_metadata.get(agent_id, {}).get("task", {}).get("description", "unknown")
            is_truthful = stmt.get("is_truthful", True)
            truth_desc = "tells truth" if is_truthful else "lies"
            prop = stmt.get("property_name", "properties")
            return f"- {agent_id} ({truth_desc} about {prop}): {text}"

        elif self.condition == Condition.ORACLE_TRUTH_LABELS:
            label = "TRUE" if stmt.get("is_truthful", True) else "FALSE"
            return f"- {agent_id} [{label}]: {text}"

        return f"- {text}"

    def _compute_agent_reliability(self, agent_id: str, agent_metadata: dict) -> int:
        """Compute reliability percentage for an agent."""
        # This is a placeholder - actual reliability comes from statements
        # For now return a fixed value; will be computed from full dataset
        return 50
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/observer/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): add Condition enum and PromptBuilder base"
```

---

## Task 5: Prompt Builder - All Conditions

**Files:**
- Modify: `src/observer/prompts.py`
- Modify: `tests/test_prompts.py`

**Step 1: Write failing tests for remaining conditions**

```python
# Add to tests/test_prompts.py
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

    def test_format_oracle_relationships(self, sample_statements, sample_agent_metadata):
        builder = PromptBuilder(Condition.ORACLE_RELATIONSHIPS)
        evidence = builder.format_evidence(
            sample_statements, sample_agent_metadata, "object_5", "color"
        )

        assert "tells truth" in evidence
        assert "lies" in evidence
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS (already implemented in Task 4)

**Step 3: Add system prompt and query builder**

```python
# Add to src/observer/prompts.py
    def build_system_prompt(self) -> str:
        """Build the system prompt for the observer."""
        return """You are an observer trying to determine ground truth about objects in a world.
You will be given statements from various agents about object properties.
Some agents may be reliable, others may lie.
Your task is to determine the truth based on the evidence provided.

Use the submit_answer tool to provide your answer."""

    def build_query(
        self,
        object_id: str,
        property_name: str,
        property_value: str,
        evidence: str,
    ) -> str:
        """Build the full query prompt."""
        return f"""Here are statements about {object_id}:

{evidence}

Question: Is {object_id} {property_value} (for {property_name})?

Analyze the evidence and submit your answer using the submit_answer tool."""
```

**Step 4: Add test for full query**

```python
# Add to tests/test_prompts.py
    def test_build_query(self, sample_statements, sample_agent_metadata):
        builder = PromptBuilder(Condition.IDS_ONLY)
        evidence = builder.format_evidence(
            sample_statements, sample_agent_metadata, "object_5", "color"
        )
        query = builder.build_query("object_5", "color", "red", evidence)

        assert "object_5" in query
        assert "red" in query
        assert "submit_answer" in query
```

**Step 5: Run test and commit**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS

```bash
git add src/observer/prompts.py tests/test_prompts.py
git commit -m "feat(prompts): add system prompt and query builder"
```

---

## Task 6: ICL Observer - Tool Definition

**Files:**
- Create: `src/observer/icl.py`
- Test: `tests/test_icl.py`

**Step 1: Write failing test**

```python
# tests/test_icl.py
"""Tests for ICL observer."""

import pytest
from unittest.mock import MagicMock, patch
from src.observer.icl import ICLObserver, ANSWER_TOOL


class TestAnswerTool:
    def test_tool_schema(self):
        assert ANSWER_TOOL["name"] == "submit_answer"
        assert "answer" in ANSWER_TOOL["input_schema"]["properties"]
        assert "confidence" in ANSWER_TOOL["input_schema"]["properties"]
        assert ANSWER_TOOL["input_schema"]["properties"]["answer"]["type"] == "boolean"


class TestICLObserver:
    def test_init(self):
        observer = ICLObserver(model="claude-haiku-4-5-20250514")
        assert observer.model == "claude-haiku-4-5-20250514"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_icl.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/observer/icl.py
"""ICL Observer for truth recovery experiments."""

from dataclasses import dataclass
import anthropic

from .prompts import Condition, PromptBuilder


ANSWER_TOOL = {
    "name": "submit_answer",
    "description": "Submit your True/False answer to the question",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "boolean",
                "description": "Your answer: True or False",
            },
            "confidence": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Your confidence in this answer (0-100%)",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of your reasoning",
            },
        },
        "required": ["answer", "confidence"],
    },
}


@dataclass
class ObserverResponse:
    """Response from the ICL observer."""
    answer: bool
    confidence: int
    reasoning: str | None = None


class ICLObserver:
    """ICL-based observer for truth recovery."""

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20250514",
        client: anthropic.Anthropic | None = None,
    ):
        self.model = model
        self.client = client or anthropic.Anthropic()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_icl.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/observer/icl.py tests/test_icl.py
git commit -m "feat(icl): add ICLObserver skeleton and tool definition"
```

---

## Task 7: ICL Observer - Query Method

**Files:**
- Modify: `src/observer/icl.py`
- Modify: `tests/test_icl.py`

**Step 1: Write failing test with mock**

```python
# Add to tests/test_icl.py
    @pytest.fixture
    def mock_client(self):
        client = MagicMock()
        response = MagicMock()
        response.content = [
            MagicMock(
                type="tool_use",
                name="submit_answer",
                input={"answer": True, "confidence": 75, "reasoning": "Most agents agree."}
            )
        ]
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
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_icl.py::TestICLObserver::test_query_returns_response -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# Add to ICLObserver class in src/observer/icl.py
    def query(
        self,
        evidence: str,
        object_id: str,
        property_name: str,
        property_value: str,
    ) -> ObserverResponse:
        """
        Query the observer about a property value.

        Args:
            evidence: Formatted evidence string
            object_id: The object being queried
            property_name: The property being queried
            property_value: The claimed value to verify

        Returns:
            ObserverResponse with answer, confidence, and reasoning
        """
        system_prompt = """You are an observer trying to determine ground truth about objects in a world.
You will be given statements from various agents about object properties.
Some agents may be reliable, others may lie.
Your task is to determine the truth based on the evidence provided.

Use the submit_answer tool to provide your answer."""

        user_prompt = f"""Here are statements about {object_id}:

{evidence}

Question: Is {object_id} {property_value} (for {property_name})?

Analyze the evidence and submit your answer using the submit_answer tool."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=500,
            system=system_prompt,
            tools=[ANSWER_TOOL],
            tool_choice={"type": "tool", "name": "submit_answer"},
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract tool use response
        for block in response.content:
            if block.type == "tool_use" and block.name == "submit_answer":
                return ObserverResponse(
                    answer=block.input["answer"],
                    confidence=block.input["confidence"],
                    reasoning=block.input.get("reasoning"),
                )

        # Fallback if no tool use (shouldn't happen with tool_choice)
        raise ValueError("No tool use response from model")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_icl.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/observer/icl.py tests/test_icl.py
git commit -m "feat(icl): add query method with tool use"
```

---

## Task 8: Update __init__.py exports

**Files:**
- Modify: `src/observer/__init__.py`
- Modify: `src/evaluation/__init__.py`

**Step 1: Update exports**

```python
# src/observer/__init__.py
from .icl import ICLObserver, ObserverResponse, ANSWER_TOOL
from .prompts import Condition, PromptBuilder

__all__ = [
    "ICLObserver",
    "ObserverResponse",
    "ANSWER_TOOL",
    "Condition",
    "PromptBuilder",
]
```

```python
# src/evaluation/__init__.py
from .metrics import (
    compute_accuracy,
    compute_accuracy_by_category,
    compute_brier_score,
    compute_ece,
    categorize_queries,
)

__all__ = [
    "compute_accuracy",
    "compute_accuracy_by_category",
    "compute_brier_score",
    "compute_ece",
    "categorize_queries",
]
```

**Step 2: Verify imports work**

Run: `uv run python -c "from src.observer import ICLObserver, Condition; from src.evaluation import compute_accuracy; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add src/observer/__init__.py src/evaluation/__init__.py
git commit -m "chore: update module exports"
```

---

## Task 9: Experiment Runner - Configuration

**Files:**
- Create: `configs/experiment/icl_baseline.yaml`
- Create: `experiments/run_icl.py`

**Step 1: Create config**

```yaml
# configs/experiment/icl_baseline.yaml
defaults:
  - /world: default
  - /agent: default

# Data generation
seeds: [42, 123, 456]
statements_per_pair: 17  # ~100 statements with 3 agents

# Agent model (for statement generation)
agent_model: claude-opus-4-5-20250514

# Observer models to compare
observer_models:
  - claude-haiku-4-5-20250514
  - claude-sonnet-4-5-20250514
  - claude-opus-4-5-20250514

# Conditions to run
conditions:
  - no_ids
  - ids_only
  - ids_and_tasks
  - oracle_reliability
  - oracle_relationships
  - oracle_truth_labels

# Output
output_dir: outputs/icl_experiment
save_results: true

# Wandb
wandb:
  project: truthification
  name: icl-baseline
```

**Step 2: Create experiment runner skeleton**

```python
#!/usr/bin/env python
# experiments/run_icl.py
"""Run ICL baseline experiment."""

import json
import sys
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment import World, Simulation, SimulationConfig
from environment.agent import Task
from environment.world import Property, PropertyType
from observer import ICLObserver, Condition, PromptBuilder
from evaluation import (
    compute_accuracy,
    compute_accuracy_by_category,
    compute_ece,
    compute_brier_score,
    categorize_queries,
)


def build_properties(cfg: DictConfig) -> list[Property]:
    """Build property definitions from config."""
    properties = []
    for prop_cfg in cfg.world.properties:
        prop_type = PropertyType(prop_cfg.type)
        properties.append(
            Property(
                name=prop_cfg.name,
                property_type=prop_type,
                possible_values=list(prop_cfg.possible_values),
            )
        )
    return properties


def build_tasks(cfg: DictConfig) -> list[Task]:
    """Build task definitions from config."""
    tasks = []
    for task_cfg in cfg.agent.tasks:
        tasks.append(
            Task(
                name=task_cfg.name,
                description=task_cfg.description,
                relevant_properties=list(task_cfg.relevant_properties),
            )
        )
    return tasks


def generate_queries(world_state: dict) -> list[tuple[str, str, str]]:
    """Generate all (object_id, property_name, value) queries."""
    queries = []
    for obj_id, obj_data in world_state["objects"].items():
        for prop_name, prop_value in obj_data["properties"].items():
            queries.append((obj_id, prop_name, prop_value))
    return queries


def compute_agent_reliabilities(statements: list[dict]) -> dict[str, int]:
    """Compute reliability % for each agent from statements."""
    from collections import defaultdict

    agent_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "truthful": 0})

    for stmt in statements:
        agent_id = stmt["agent_id"]
        agent_stats[agent_id]["total"] += 1
        if stmt["is_truthful"]:
            agent_stats[agent_id]["truthful"] += 1

    reliabilities = {}
    for agent_id, stats in agent_stats.items():
        if stats["total"] > 0:
            reliabilities[agent_id] = int(100 * stats["truthful"] / stats["total"])
        else:
            reliabilities[agent_id] = 50

    return reliabilities


@hydra.main(version_base=None, config_path="../configs/experiment", config_name="icl_baseline")
def main(cfg: DictConfig) -> None:
    """Run ICL baseline experiment."""
    print("ICL Baseline Experiment")
    print("=" * 50)
    print(OmegaConf.to_yaml(cfg))

    # Initialize wandb
    wandb.init(
        project=cfg.wandb.project,
        name=f"{cfg.wandb.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Build world config
    properties = build_properties(cfg)
    tasks = build_tasks(cfg)

    all_results = []

    for seed in cfg.seeds:
        print(f"\n--- Seed {seed} ---")

        # Generate data
        sim_config = SimulationConfig(
            n_objects=cfg.world.n_objects,
            n_agents=cfg.agent.n_agents,
            adversarial_fraction=cfg.agent.adversarial_fraction,
            statements_per_pair=cfg.statements_per_pair,
            seed=seed,
            properties=properties,
            tasks=tasks,
            model=cfg.agent_model,
        )

        sim = Simulation(sim_config)
        result = sim.run()

        print(f"Generated {len(result.statements)} statements")
        stats = result.get_stats()
        print(f"  Truthful: {stats['truthful_fraction']:.1%}")

        # Build queries
        queries = generate_queries(result.world_state)
        print(f"Generated {len(queries)} queries")

        # Compute agent reliabilities for oracle condition
        reliabilities = compute_agent_reliabilities(result.statements)

        # Categorize queries
        categories = categorize_queries(
            queries, result.statements, result.world_state
        )

        # Run each condition × observer combination
        for condition_name in cfg.conditions:
            condition = Condition(condition_name)
            builder = PromptBuilder(condition)

            # Inject reliabilities for oracle condition
            agent_metadata_with_reliability = {}
            for agent_id, meta in result.agent_metadata.items():
                agent_metadata_with_reliability[agent_id] = {
                    **meta,
                    "reliability": reliabilities.get(agent_id, 50),
                }

            for observer_model in cfg.observer_models:
                print(f"\nRunning: {condition_name} / {observer_model}")

                observer = ICLObserver(model=observer_model)

                predictions = []
                confidences = []
                ground_truths = []

                for obj_id, prop_name, prop_value in tqdm(queries, desc="Queries"):
                    # Format evidence
                    evidence = builder.format_evidence(
                        result.statements,
                        agent_metadata_with_reliability,
                        obj_id,
                        prop_name,
                    )

                    # Query observer
                    try:
                        response = observer.query(
                            evidence=evidence,
                            object_id=obj_id,
                            property_name=prop_name,
                            property_value=str(prop_value),
                        )
                        predictions.append(response.answer)
                        confidences.append(response.confidence)
                    except Exception as e:
                        print(f"Error: {e}")
                        predictions.append(True)  # Default
                        confidences.append(50)

                    ground_truths.append(True)  # Query asks "is X value?" - answer is always True

                # Compute metrics
                accuracy = compute_accuracy(predictions, ground_truths)
                accuracy_by_cat = compute_accuracy_by_category(
                    predictions, ground_truths, categories
                )
                ece = compute_ece(confidences, ground_truths)
                brier = compute_brier_score(confidences, ground_truths)

                run_result = {
                    "seed": seed,
                    "condition": condition_name,
                    "observer_model": observer_model,
                    "accuracy": accuracy,
                    "accuracy_contested": accuracy_by_cat.get("contested", None),
                    "accuracy_unanimous": accuracy_by_cat.get("unanimous", None),
                    "ece": ece,
                    "brier": brier,
                    "n_queries": len(queries),
                }
                all_results.append(run_result)

                # Log to wandb
                wandb.log(run_result)

                print(f"  Accuracy: {accuracy:.1%}")
                print(f"  ECE: {ece:.3f}")

    # Save results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {results_path}")

    # Log summary table to wandb
    wandb.log({"results_table": wandb.Table(dataframe=__import__("pandas").DataFrame(all_results))})

    wandb.finish()


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add configs/experiment/icl_baseline.yaml experiments/run_icl.py
git commit -m "feat(experiment): add ICL baseline experiment runner"
```

---

## Task 10: Fix PromptBuilder for Oracle Reliability

**Files:**
- Modify: `src/observer/prompts.py`
- Modify: `tests/test_prompts.py`

**Step 1: Update PromptBuilder to accept reliabilities**

The oracle reliability condition needs actual reliability scores passed in. Update the implementation:

```python
# Update _format_statement in src/observer/prompts.py
    def _format_statement(self, stmt: dict, agent_metadata: dict) -> str:
        """Format a single statement based on condition."""
        text = stmt["text"]
        agent_id = stmt["agent_id"]

        if self.condition == Condition.NO_IDS:
            return f"- {text}"

        elif self.condition == Condition.IDS_ONLY:
            return f"- {agent_id}: {text}"

        elif self.condition == Condition.IDS_AND_TASKS:
            task_desc = agent_metadata.get(agent_id, {}).get("task", {}).get("description", "unknown task")
            return f"- {agent_id} (task: {task_desc}): {text}"

        elif self.condition == Condition.ORACLE_RELIABILITY:
            # Get reliability from metadata (should be pre-computed)
            reliability = agent_metadata.get(agent_id, {}).get("reliability", 50)
            return f"- {agent_id} (reliability: {reliability}%): {text}"

        elif self.condition == Condition.ORACLE_RELATIONSHIPS:
            is_truthful = stmt.get("is_truthful", True)
            truth_desc = "tells truth" if is_truthful else "lies"
            prop = stmt.get("property_name", "properties")
            return f"- {agent_id} ({truth_desc} about {prop}): {text}"

        elif self.condition == Condition.ORACLE_TRUTH_LABELS:
            label = "TRUE" if stmt.get("is_truthful", True) else "FALSE"
            return f"- {agent_id} [{label}]: {text}"

        return f"- {text}"
```

**Step 2: Add test for oracle reliability**

```python
# Add to tests/test_prompts.py
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
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/observer/prompts.py tests/test_prompts.py
git commit -m "fix(prompts): oracle reliability uses pre-computed values"
```

---

## Task 11: Run Full Test Suite

**Step 1: Run all tests**

Run: `uv run pytest -v`
Expected: All tests pass

**Step 2: Commit any fixes if needed**

---

## Task 12: Test Experiment Runner (Dry Run)

**Step 1: Create a minimal test config**

```yaml
# configs/experiment/icl_test.yaml
defaults:
  - /world: default
  - /agent: default

seeds: [42]
statements_per_pair: 2

agent_model: claude-sonnet-4-20250514  # Cheaper for test

observer_models:
  - claude-haiku-4-5-20250514

conditions:
  - no_ids
  - ids_only

output_dir: outputs/icl_test
save_results: true

wandb:
  project: truthification
  name: icl-test
```

**Step 2: Run with test config**

Run: `uv run python experiments/run_icl.py --config-name=icl_test`
Expected: Runs successfully, logs to wandb

**Step 3: Verify outputs**

Check:
- `outputs/icl_test/results_*.json` exists
- wandb dashboard shows metrics

**Step 4: Commit test config**

```bash
git add configs/experiment/icl_test.yaml
git commit -m "test: add minimal ICL test config"
```

---

## Summary

After completing all tasks, you will have:

1. **Metrics module** (`src/evaluation/metrics.py`): accuracy, ECE, Brier, categorization
2. **Prompt builder** (`src/observer/prompts.py`): 6 conditions
3. **ICL observer** (`src/observer/icl.py`): tool-use based queries
4. **Experiment runner** (`experiments/run_icl.py`): full pipeline with wandb logging
5. **Configs**: `icl_baseline.yaml` and `icl_test.yaml`

To run the full experiment:
```bash
uv run python experiments/run_icl.py
```

Estimated cost: ~$80-130 for full experiment (3 seeds × 6 conditions × 3 observers).
