"""Public Mycelium evaluation interface."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

from ._spans import (
    TextInput,
    normalize_batch,
    normalize_text_lengths,
    validate_bounds,
)
from .metrics import character, iou, span_coverage


AVAILABLE_METRICS = ("iou", "character", "span_coverage")
SUPPORTED_PROFILES = ("mu_shroom", "mu-shroom", "mushroom", "ragtruth")
PROFILE_DEFAULT_AVERAGES = {
    "mu_shroom": "macro",
    "mu-shroom": "macro",
    "mushroom": "macro",
    "ragtruth": "micro",
}


def evaluate(
    *,
    references: Sequence[object],
    predictions: Sequence[object],
    text: TextInput = None,
    profile: Optional[str] = None,
    metrics: Optional[Iterable[str]] = None,
    average: Optional[str] = None,
    include_per_example: bool = False,
    span_coverage_delta: int = 0,
    span_coverage_min_pred_len: int = 1,
    empty_is_perfect: bool = True,
) -> Dict[str, object]:
    """Evaluate hallucination spans with every applicable Mycelium metric.

    Parameters use half-open character spans ``[start, end)``. ``references``
    and ``predictions`` may each be either one span set or a batch of span sets.
    Mu-SHROOM profiles default to macro aggregation; the RAGTruth profile
    defaults to micro aggregation over character decisions.
    """

    if profile is not None and profile not in SUPPORTED_PROFILES:
        choices = ", ".join(SUPPORTED_PROFILES)
        raise ValueError(f"Unknown profile {profile!r}; supported profiles: {choices}")

    resolved_average = (
        PROFILE_DEFAULT_AVERAGES.get(profile, "macro")
        if average is None
        else average
    )

    reference_batch = normalize_batch(references, name="references")
    prediction_batch = normalize_batch(predictions, name="predictions")
    if len(reference_batch) != len(prediction_batch):
        raise ValueError(
            "references and predictions must contain the same number of examples; "
            f"got {len(reference_batch)} vs {len(prediction_batch)}"
        )

    text_lengths = normalize_text_lengths(text, batch_size=len(reference_batch))
    validate_bounds(reference_batch, text_lengths=text_lengths, name="references")
    validate_bounds(prediction_batch, text_lengths=text_lengths, name="predictions")

    if isinstance(metrics, str):
        selected = (metrics,)
    else:
        selected = tuple(metrics) if metrics is not None else AVAILABLE_METRICS
    unknown = sorted(set(selected) - set(AVAILABLE_METRICS))
    if unknown:
        raise ValueError(
            f"Unknown metrics: {', '.join(unknown)}. "
            f"Available metrics: {', '.join(AVAILABLE_METRICS)}"
        )

    report: Dict[str, object] = {}
    if "iou" in selected:
        report["iou"] = iou(
            reference_batch,
            prediction_batch,
            average=resolved_average,
            include_per_example=include_per_example,
        )
    if "character" in selected:
        report["character"] = character(
            reference_batch,
            prediction_batch,
            average=resolved_average,
            include_per_example=include_per_example,
        )
    if "span_coverage" in selected:
        report["span_coverage"] = span_coverage(
            reference_batch,
            prediction_batch,
            average=resolved_average,
            delta=span_coverage_delta,
            min_pred_len=span_coverage_min_pred_len,
            empty_is_perfect=empty_is_perfect,
            include_per_example=include_per_example,
        )

    report["meta"] = {
        "examples": len(reference_batch),
        "average": resolved_average,
        "offset_convention": "half-open [start, end)",
        "profile": profile,
    }
    return report
