"""Mu-SHROOM character intersection-over-union."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .._spans import Span, SpanBatch


def _indices(spans: Sequence[Span]) -> set[int]:
    return {index for start, end in spans for index in range(start, end)}


def iou_one(reference: Sequence[Span], prediction: Sequence[Span]) -> float:
    """Return Mu-SHROOM hard-label IoU for one example."""

    reference_indices = _indices(reference)
    prediction_indices = _indices(prediction)
    if not reference_indices and not prediction_indices:
        return 1.0
    return len(reference_indices & prediction_indices) / len(
        reference_indices | prediction_indices
    )


def iou(
    references: SpanBatch,
    predictions: SpanBatch,
    *,
    average: str = "macro",
    include_per_example: bool = False,
) -> Dict[str, object]:
    """Aggregate Mu-SHROOM character IoU over a batch."""

    scores: List[float] = [
        iou_one(reference, prediction)
        for reference, prediction in zip(references, predictions)
    ]

    if average == "macro":
        score = sum(scores) / len(scores)
    elif average == "micro":
        intersections = 0
        unions = 0
        both_empty = True
        for reference, prediction in zip(references, predictions):
            reference_indices = _indices(reference)
            prediction_indices = _indices(prediction)
            intersection = len(reference_indices & prediction_indices)
            union = len(reference_indices | prediction_indices)
            intersections += intersection
            unions += union
            both_empty = both_empty and union == 0
        score = 1.0 if both_empty else intersections / unions
    else:
        raise ValueError("average must be 'macro' or 'micro'")

    result: Dict[str, object] = {"score": score}
    if include_per_example:
        result["per_example"] = scores
    return result
