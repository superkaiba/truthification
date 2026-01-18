"""Tests for evaluation metrics."""

import pytest
from src.evaluation.metrics import (
    compute_accuracy,
    compute_accuracy_by_category,
    compute_ece,
    compute_brier_score,
    categorize_queries,
)


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
