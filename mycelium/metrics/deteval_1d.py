"""One-dimensional adaptation of DetEval for character spans."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

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
    recall_threshold: float,
    precision_threshold: float,
    granularity_penalty: float,
) -> DetectionCounts:
    references = sorted(reference)
    predictions = sorted(prediction)
    intersections = [
        [_overlap_length(ref, pred) for pred in predictions]
        for ref in references
    ]
    area_recall = [
        [value / _length(ref) for value in row]
        for ref, row in zip(references, intersections)
    ]
    area_precision = [
        [intersections[i][j] / _length(predictions[j]) for j in range(len(predictions))]
        for i in range(len(references))
    ]

    unmatched_references = set(range(len(references)))
    unmatched_predictions = set(range(len(predictions)))
    recall_credit = [0.0] * len(references)
    precision_credit = [0.0] * len(predictions)

    one_to_one: List[Tuple[int, int]] = []
    for i in range(len(references)):
        row = [j for j in range(len(predictions)) if intersections[i][j] > 0]
        if len(row) != 1:
            continue
        j = row[0]
        column = [
            ii for ii in range(len(references)) if intersections[ii][j] > 0
        ]
        if (
            len(column) == 1
            and area_recall[i][j] >= recall_threshold
            and area_precision[i][j] >= precision_threshold
        ):
            one_to_one.append((i, j))

    for i, j in one_to_one:
        if i not in unmatched_references or j not in unmatched_predictions:
            continue
        recall_credit[i] = 1.0
        precision_credit[j] = 1.0
        unmatched_references.remove(i)
        unmatched_predictions.remove(j)

    # One reference split across multiple predictions.
    for i in sorted(tuple(unmatched_references)):
        if sum(intersections[i][j] > 0 for j in range(len(predictions))) <= 1:
            continue
        candidates = [
            j
            for j in sorted(unmatched_predictions)
            if intersections[i][j] > 0
            and area_precision[i][j] >= precision_threshold
        ]
        if len(candidates) == 1:
            j = candidates[0]
            if (
                area_recall[i][j] >= recall_threshold
                and area_precision[i][j] >= precision_threshold
            ):
                recall_credit[i] = 1.0
                precision_credit[j] = 1.0
                unmatched_references.remove(i)
                unmatched_predictions.remove(j)
        elif (
            len(candidates) >= 2
            and sum(area_recall[i][j] for j in candidates) >= recall_threshold
        ):
            recall_credit[i] = granularity_penalty
            for j in candidates:
                precision_credit[j] = granularity_penalty
            unmatched_references.remove(i)
            unmatched_predictions.difference_update(candidates)

    # Multiple references merged into one prediction.
    for j in sorted(tuple(unmatched_predictions)):
        if sum(intersections[i][j] > 0 for i in range(len(references))) <= 1:
            continue
        candidates = [
            i
            for i in sorted(unmatched_references)
            if intersections[i][j] > 0
            and area_recall[i][j] >= recall_threshold
        ]
        if len(candidates) == 1:
            i = candidates[0]
            if (
                area_recall[i][j] >= recall_threshold
                and area_precision[i][j] >= precision_threshold
            ):
                precision_credit[j] = 1.0
                recall_credit[i] = 1.0
                unmatched_predictions.remove(j)
                unmatched_references.remove(i)
        elif (
            len(candidates) >= 2
            and sum(area_precision[i][j] for i in candidates)
            >= precision_threshold
        ):
            precision_credit[j] = granularity_penalty
            for i in candidates:
                recall_credit[i] = granularity_penalty
            unmatched_predictions.remove(j)
            unmatched_references.difference_update(candidates)

    return (
        sum(precision_credit),
        len(predictions),
        sum(recall_credit),
        len(references),
    )


def deteval_1d(
    references: SpanBatch,
    predictions: SpanBatch,
    *,
    average: str = "macro",
    recall_threshold: float = 0.8,
    precision_threshold: float = 0.4,
    granularity_penalty: float = 0.8,
    include_per_example: bool = False,
) -> Dict[str, object]:
    """Score exact, split, and merged interval correspondences."""

    if not 0.0 <= recall_threshold <= 1.0:
        raise ValueError("recall_threshold must be in [0,1]")
    if not 0.0 <= precision_threshold <= 1.0:
        raise ValueError("precision_threshold must be in [0,1]")
    if not 0.0 <= granularity_penalty <= 1.0:
        raise ValueError("granularity_penalty must be in [0,1]")

    counts = [
        _counts_one(
            reference,
            prediction,
            recall_threshold=recall_threshold,
            precision_threshold=precision_threshold,
            granularity_penalty=granularity_penalty,
        )
        for reference, prediction in zip(references, predictions)
    ]
    return aggregate_detection(
        counts,
        average=average,
        include_per_example=include_per_example,
    )
