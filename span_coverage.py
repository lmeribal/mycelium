"""Reference implementation of Span Coverage F1.

All spans use inclusive character offsets: ``[start, end]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


Span = Tuple[int, int]


@dataclass
class SpanCoveragePRF:
    """Span Coverage sufficient statistics and precision/recall/F-beta."""

    contained_preds: int
    total_preds: int
    hit_golds: int
    total_golds: int
    precision: float
    recall: float
    fbeta: float


def _normalize_spans(spans: Sequence[Sequence[int]]) -> List[Span]:
    normalized: List[Span] = []
    for span in spans:
        if len(span) != 2:
            raise ValueError(f"Bad span (expected [start,end]): {span}")
        start, end = int(span[0]), int(span[1])
        if end < start:
            raise ValueError(f"Bad span with end < start: {span}")
        normalized.append((start, end))
    return normalized


def _len_inc(span: Span) -> int:
    return span[1] - span[0] + 1


def _contained(pred: Span, gold: Span, delta: int = 0) -> bool:
    pred_start, pred_end = pred
    gold_start, gold_end = gold
    return (gold_start - delta) <= pred_start and pred_end <= (gold_end + delta)


def _fbeta_from_pr(precision: float, recall: float, beta: float) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive")
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta_squared = beta * beta
    return (
        (1.0 + beta_squared)
        * precision
        * recall
        / (beta_squared * precision + recall)
    )


def span_coverage_counts_one(
    gold: Sequence[Sequence[int]],
    pred: Sequence[Sequence[int]],
    *,
    delta: int = 0,
    min_pred_len: int = 1,
) -> Tuple[int, int, int, int]:
    """Return contained predictions, total predictions, hit golds, total golds."""

    if delta < 0:
        raise ValueError("delta must be non-negative")
    if min_pred_len < 1:
        raise ValueError("min_pred_len must be at least 1")

    gold_spans = sorted(_normalize_spans(gold))
    pred_spans = sorted(
        span
        for span in _normalize_spans(pred)
        if _len_inc(span) >= min_pred_len
    )

    contained_preds = sum(
        any(_contained(pred_span, gold_span, delta) for gold_span in gold_spans)
        for pred_span in pred_spans
    )
    hit_golds = sum(
        any(_contained(pred_span, gold_span, delta) for pred_span in pred_spans)
        for gold_span in gold_spans
    )
    return contained_preds, len(pred_spans), hit_golds, len(gold_spans)


def span_coverage_micro(
    golds: Sequence[Sequence[Sequence[int]]],
    preds: Sequence[Sequence[Sequence[int]]],
    *,
    delta: int = 0,
    min_pred_len: int = 1,
    beta: float = 1.0,
    empty_is_perfect: bool = True,
) -> SpanCoveragePRF:
    """Aggregate sufficient statistics across examples, then compute F-beta."""

    if len(golds) != len(preds):
        raise ValueError(
            "golds and preds must have the same length, "
            f"got {len(golds)} vs {len(preds)}"
        )

    contained_preds = total_preds = hit_golds = total_golds = 0
    for gold, pred in zip(golds, preds):
        cp, tp, hg, tg = span_coverage_counts_one(
            gold,
            pred,
            delta=delta,
            min_pred_len=min_pred_len,
        )
        contained_preds += cp
        total_preds += tp
        hit_golds += hg
        total_golds += tg

    if total_preds == 0 and total_golds == 0:
        value = 1.0 if empty_is_perfect else 0.0
        return SpanCoveragePRF(0, 0, 0, 0, value, value, value)

    precision = contained_preds / total_preds if total_preds else 1.0
    recall = hit_golds / total_golds if total_golds else 1.0
    fbeta = _fbeta_from_pr(precision, recall, beta)
    return SpanCoveragePRF(
        contained_preds,
        total_preds,
        hit_golds,
        total_golds,
        precision,
        recall,
        fbeta,
    )


def span_coverage_macro(
    golds: Sequence[Sequence[Sequence[int]]],
    preds: Sequence[Sequence[Sequence[int]]],
    *,
    delta: int = 0,
    min_pred_len: int = 1,
    beta: float = 1.0,
    empty_is_perfect: bool = True,
) -> SpanCoveragePRF:
    """Compute per-example scores, then return their arithmetic means."""

    if len(golds) != len(preds):
        raise ValueError(
            "golds and preds must have the same length, "
            f"got {len(golds)} vs {len(preds)}"
        )
    if not golds:
        value = 1.0 if empty_is_perfect else 0.0
        return SpanCoveragePRF(0, 0, 0, 0, value, value, value)

    precisions: List[float] = []
    recalls: List[float] = []
    fbetas: List[float] = []
    for gold, pred in zip(golds, preds):
        cp, tp, hg, tg = span_coverage_counts_one(
            gold,
            pred,
            delta=delta,
            min_pred_len=min_pred_len,
        )
        if tp == 0 and tg == 0:
            precision = recall = 1.0 if empty_is_perfect else 0.0
        else:
            precision = cp / tp if tp else 1.0
            recall = hg / tg if tg else 1.0
        precisions.append(precision)
        recalls.append(recall)
        fbetas.append(_fbeta_from_pr(precision, recall, beta))

    count = len(golds)
    return SpanCoveragePRF(
        0,
        0,
        0,
        0,
        sum(precisions) / count,
        sum(recalls) / count,
        sum(fbetas) / count,
    )


__all__ = [
    "Span",
    "SpanCoveragePRF",
    "span_coverage_counts_one",
    "span_coverage_macro",
    "span_coverage_micro",
]

