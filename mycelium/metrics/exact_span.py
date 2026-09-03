"""Exact instance-level span precision, recall, and F1."""

from __future__ import annotations

from typing import Dict

from .._spans import SpanBatch
from ._detection import DetectionCounts, aggregate_detection


def exact_span(
    references: SpanBatch,
    predictions: SpanBatch,
    *,
    average: str = "macro",
    include_per_example: bool = False,
) -> Dict[str, object]:
    """Count a match only when prediction and reference boundaries are equal."""

    counts: list[DetectionCounts] = []
    for reference, prediction in zip(references, predictions):
        matches = len(set(reference) & set(prediction))
        counts.append((matches, len(prediction), matches, len(reference)))
    return aggregate_detection(
        counts,
        average=average,
        include_per_example=include_per_example,
    )
