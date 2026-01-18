"""Evaluation metrics for ICL experiments."""

from collections import defaultdict
import numpy as np


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
