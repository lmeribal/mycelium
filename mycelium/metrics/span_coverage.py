"""Directional Span Coverage precision, recall, and F-beta."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .._spans import Span, SpanBatch


def _contained(prediction: Span, reference: Span, *, delta: int) -> bool:
    pred_start, pred_end = prediction
    ref_start, ref_end = reference
    return (ref_start - delta) <= pred_start and pred_end <= (ref_end + delta)


def _counts_one(
    reference: Sequence[Span],
    prediction: Sequence[Span],
    *,
    delta: int,
    min_pred_len: int,
) -> Tuple[int, int, int, int]:
    filtered_predictions = [
        span for span in prediction if span[1] - span[0] >= min_pred_len
    ]
    contained_predictions = sum(
        any(_contained(pred, ref, delta=delta) for ref in reference)
        for pred in filtered_predictions
    )
    hit_references = sum(
        any(_contained(pred, ref, delta=delta) for pred in filtered_predictions)
        for ref in reference
    )
    return (
        contained_predictions,
        len(filtered_predictions),
        hit_references,
        len(reference),
    )


def _fbeta(precision: float, recall: float, *, beta: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_squared = beta * beta
    return (
        (1.0 + beta_squared)
        * precision
        * recall
        / (beta_squared * precision + recall)
    )


def _prf(
    counts: Tuple[int, int, int, int],
    *,
    beta: float,
    empty_is_perfect: bool,
) -> Tuple[float, float, float]:
    contained_predictions, total_predictions, hit_references, total_references = counts
    if total_predictions == 0 and total_references == 0:
        value = 1.0 if empty_is_perfect else 0.0
        return value, value, value
    precision = (
        contained_predictions / total_predictions if total_predictions else 1.0
    )
    recall = hit_references / total_references if total_references else 1.0
    return precision, recall, _fbeta(precision, recall, beta=beta)


def span_coverage(
    references: SpanBatch,
    predictions: SpanBatch,
    *,
    average: str = "macro",
    delta: int = 0,
    min_pred_len: int = 1,
    beta: float = 1.0,
    empty_is_perfect: bool = True,
    include_per_example: bool = False,
) -> Dict[str, object]:
    """Compute Span Coverage using half-open character spans."""

    if delta < 0:
        raise ValueError("delta must be non-negative")
    if min_pred_len < 1:
        raise ValueError("min_pred_len must be at least 1")
    if beta <= 0:
        raise ValueError("beta must be positive")

    counts = [
        _counts_one(
            reference,
            prediction,
            delta=delta,
            min_pred_len=min_pred_len,
        )
        for reference, prediction in zip(references, predictions)
    ]
    per_example: List[Dict[str, float]] = []
    for item_counts in counts:
        precision, recall, f1 = _prf(
            item_counts,
            beta=beta,
            empty_is_perfect=empty_is_perfect,
        )
        per_example.append({"precision": precision, "recall": recall, "f1": f1})

    if average == "macro":
        denominator = len(per_example)
        precision = sum(item["precision"] for item in per_example) / denominator
        recall = sum(item["recall"] for item in per_example) / denominator
        f1 = sum(item["f1"] for item in per_example) / denominator
    elif average == "micro":
        totals = tuple(sum(item[index] for item in counts) for index in range(4))
        precision, recall, f1 = _prf(
            totals, beta=beta, empty_is_perfect=empty_is_perfect
        )
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
