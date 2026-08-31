"""Character-mask precision, recall, and F1.

This is an independent reconstruction of the span-level metric described in
the RAGTruth paper. It converts reference and predicted spans to character
masks and computes binary precision, recall, and F1 from their overlap.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .._spans import Span, SpanBatch


def _indices(spans: Sequence[Span]) -> set[int]:
    return {index for start, end in spans for index in range(start, end)}


def _counts_one(
    reference: Sequence[Span], prediction: Sequence[Span]
) -> Tuple[int, int, int]:
    reference_indices = _indices(reference)
    prediction_indices = _indices(prediction)
    true_positives = len(reference_indices & prediction_indices)
    false_positives = len(prediction_indices - reference_indices)
    false_negatives = len(reference_indices - prediction_indices)
    return true_positives, false_positives, false_negatives


def _prf(counts: Tuple[int, int, int]) -> Tuple[float, float, float]:
    true_positives, false_positives, false_negatives = counts
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positives / recall_denominator if recall_denominator else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def character(
    references: SpanBatch,
    predictions: SpanBatch,
    *,
    average: str = "micro",
    include_per_example: bool = False,
) -> Dict[str, object]:
    """Compute character-mask overlap precision, recall, and F1."""

    counts = [
        _counts_one(reference, prediction)
        for reference, prediction in zip(references, predictions)
    ]
    per_example: List[Dict[str, float]] = []
    for item_counts in counts:
        precision, recall, f1 = _prf(item_counts)
        per_example.append({"precision": precision, "recall": recall, "f1": f1})

    if average == "micro":
        totals = tuple(sum(item[index] for item in counts) for index in range(3))
        precision, recall, f1 = _prf(totals)
    elif average == "macro":
        denominator = len(per_example)
        precision = sum(item["precision"] for item in per_example) / denominator
        recall = sum(item["recall"] for item in per_example) / denominator
        f1 = sum(item["f1"] for item in per_example) / denominator
    else:
        raise ValueError("average must be 'macro' or 'micro'")

    result: Dict[str, object] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    if include_per_example:
        result["per_example"] = per_example
    return result
