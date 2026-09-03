"""One-dimensional adaptation of CLEval for character spans."""

from __future__ import annotations

from typing import Dict, Sequence

from .._spans import Span, SpanBatch
from ._detection import DetectionCounts, aggregate_detection


def _length(span: Span) -> int:
    return span[1] - span[0]


def _overlap_length(left: Span, right: Span) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def _counts_one(
    reference: Sequence[Span],
    prediction: Sequence[Span],
    *,
    area_precision_threshold: float,
    granularity_penalty_weight: float,
) -> DetectionCounts:
    references = sorted(reference)
    predictions = sorted(prediction)
    intersections = [
        [_overlap_length(ref, pred) for pred in predictions]
        for ref in references
    ]
    area_precision = [
        [intersections[i][j] / _length(predictions[j]) for j in range(len(predictions))]
        for i in range(len(references))
    ]
    ap_qualified = [
        [
            area_precision[i][j] >= area_precision_threshold
            for j in range(len(predictions))
        ]
        for i in range(len(references))
    ]

    match = [[False] * len(predictions) for _ in range(len(references))]
    for i in range(len(references)):
        for j in range(len(predictions)):
            if intersections[i][j] <= 0 or not ap_qualified[i][j]:
                continue
            row_degree = sum(
                intersections[i][jj] > 0 and ap_qualified[i][jj]
                for jj in range(len(predictions))
            )
            column_degree = sum(
                intersections[ii][j] > 0 and ap_qualified[ii][j]
                for ii in range(len(references))
            )
            if row_degree == 1 and column_degree == 1:
                match[i][j] = True

    for i in range(len(references)):
        candidates = [
            j
            for j in range(len(predictions))
            if intersections[i][j] > 0 and ap_qualified[i][j]
        ]
        if len(candidates) >= 2:
            for j in candidates:
                match[i][j] = True

    for j in range(len(predictions)):
        candidates = [
            i for i in range(len(references)) if intersections[i][j] > 0
        ]
        if (
            len(candidates) >= 2
            and sum(area_precision[i][j] for i in candidates)
            >= area_precision_threshold
        ):
            for i in candidates:
                match[i][j] = True

    matched_references_by_prediction = [
        [i for i in range(len(references)) if match[i][j]]
        for j in range(len(predictions))
    ]
    matched_predictions_by_reference = [
        [j for j in range(len(predictions)) if match[i][j]]
        for i in range(len(references))
    ]
    covered_reference_characters = set()
    for i, reference_span in enumerate(references):
        for j in matched_predictions_by_reference[i]:
            start = max(reference_span[0], predictions[j][0])
            end = min(reference_span[1], predictions[j][1])
            if start < end:
                covered_reference_characters.update(range(start, end))

    correct_characters = len(covered_reference_characters)
    recall_penalty = sum(
        max(len(indices) - 1, 0) * granularity_penalty_weight
        for indices in matched_predictions_by_reference
    )
    precision_penalty = sum(
        max(len(indices) - 1, 0) * granularity_penalty_weight
        for indices in matched_references_by_prediction
    )
    recall_credit = max(0.0, correct_characters - recall_penalty)
    precision_credit = max(0.0, correct_characters - precision_penalty)
    return (
        precision_credit,
        sum(_length(span) for span in predictions),
        recall_credit,
        sum(_length(span) for span in references),
    )


def cleval_1d(
    references: SpanBatch,
    predictions: SpanBatch,
    *,
    average: str = "macro",
    area_precision_threshold: float = 0.3,
    granularity_penalty_weight: float = 1.0,
    include_per_example: bool = False,
) -> Dict[str, object]:
    """Score character coverage while penalizing split and merged spans."""

    if not 0.0 <= area_precision_threshold <= 1.0:
        raise ValueError("area_precision_threshold must be in [0,1]")
    if granularity_penalty_weight < 0.0:
        raise ValueError("granularity_penalty_weight must be non-negative")

    counts = [
        _counts_one(
            reference,
            prediction,
            area_precision_threshold=area_precision_threshold,
            granularity_penalty_weight=granularity_penalty_weight,
        )
        for reference, prediction in zip(references, predictions)
    ]
    return aggregate_detection(
        counts,
        average=average,
        include_per_example=include_per_example,
    )
