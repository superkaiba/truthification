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
