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
