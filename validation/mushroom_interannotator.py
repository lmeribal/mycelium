#!/usr/bin/env python3
"""Compare span metrics on Mu-SHROOM's individual annotator labels.

Mu-SHROOM stores half-open character spans [start, end). Span Coverage F1 uses
inclusive spans [start, end]. This script converts the former to the latter
after validation and canonicalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from span_coverage import Span, span_coverage_counts_one


# Shared interval helpers used by the comparison metrics.


def _normalize_spans(spans: Sequence[Sequence[int]]) -> List[Span]:
    out: List[Span] = []
    for x in spans:
        if len(x) != 2:
            raise ValueError(f"Bad span (expected [start,end]): {x}")
        s, e = int(x[0]), int(x[1])
        if e < s:
            raise ValueError(f"Bad span with end < start: {x}")
        out.append((s, e))
    return out


def _len_inc(sp: Span) -> int:
    return sp[1] - sp[0] + 1


def _contained(pred: Span, gold: Span, delta: int = 0) -> bool:
    ps, pe = pred
    gs, ge = gold
    return (gs - delta) <= ps and pe <= (ge + delta)


@dataclass
class DetectionPRF:
    """Directional detection score with additive corpus-level sufficient statistics."""

    precision_credit: float
    pred_total: int
    recall_credit: float
    gold_total: int
    precision: float
    recall: float
    f1: float


def _fbeta_from_pr(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


# ---------------------------------------------------------------------------
# Span normalization and comparison metrics
# ---------------------------------------------------------------------------


def merge_inclusive_spans(spans: Iterable[Span], *, merge_adjacent: bool = True) -> List[Span]:
    ordered = sorted(set(spans), key=lambda x: (x[0], x[1]))
    if not ordered:
        return []
    merged: List[Span] = [ordered[0]]
    gap = 1 if merge_adjacent else 0
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def clean_half_open_spans(
    raw_spans: Sequence[Sequence[int]],
    *,
    text_length: int,
    qa: Counter,
    merge_adjacent: bool,
) -> List[Span]:
    converted: List[Span] = []
    for raw in raw_spans:
        qa["raw_spans"] += 1
        if len(raw) != 2:
            raise ValueError(f"Expected [start,end], got {raw}")
        start, end = int(raw[0]), int(raw[1])
        if end < start:
            qa["reversed_spans"] += 1
            raise ValueError(f"Reversed half-open span: {raw}")
        if end == start:
            qa["zero_length_spans_dropped"] += 1
            continue
        if start < 0 or end > text_length:
            qa["out_of_bounds_spans"] += 1
            raise ValueError(
                f"Span {raw} is outside text bounds [0,{text_length})"
            )
        converted.append((start, end - 1))

    canonical = merge_inclusive_spans(converted, merge_adjacent=merge_adjacent)
    qa["canonical_merges"] += len(converted) - len(canonical)
    qa["canonical_spans"] += len(canonical)
    return canonical


def span_chars(spans: Sequence[Span]) -> set:
    return {idx for start, end in spans for idx in range(start, end + 1)}


def mask_to_spans(mask: Iterable[int]) -> List[Span]:
    indices = sorted(set(mask))
    if not indices:
        return []
    out: List[Span] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx != prev + 1:
            out.append((start, prev))
            start = idx
        prev = idx
    out.append((start, prev))
    return out


def _prf_from_counts(matches: int, pred_total: int, gold_total: int) -> Tuple[float, float, float]:
    if pred_total == 0 and gold_total == 0:
        return 1.0, 1.0, 1.0
    precision = matches / pred_total if pred_total > 0 else 1.0
    recall = matches / gold_total if gold_total > 0 else 1.0
    return precision, recall, _fbeta_from_pr(precision, recall, beta=1.0)


def exact_span_counts(gold: Sequence[Span], pred: Sequence[Span]) -> Tuple[int, int, int]:
    matches = len(set(gold) & set(pred))
    return matches, len(pred), len(gold)


def char_counts(gold: Sequence[Span], pred: Sequence[Span]) -> Tuple[int, int, int, int]:
    gold_chars = span_chars(gold)
    pred_chars = span_chars(pred)
    return (
        len(gold_chars & pred_chars),
        len(pred_chars),
        len(gold_chars),
        len(gold_chars | pred_chars),
    )


def overlap_len(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _detection_prf(
    precision_credit: float,
    pred_total: int,
    recall_credit: float,
    gold_total: int,
) -> DetectionPRF:
    if pred_total == 0 and gold_total == 0:
        precision = recall = f1 = 1.0
    else:
        precision = precision_credit / pred_total if pred_total else 1.0
        recall = recall_credit / gold_total if gold_total else 1.0
        f1 = _fbeta_from_pr(precision, recall)
    return DetectionPRF(
        precision_credit=precision_credit,
        pred_total=pred_total,
        recall_credit=recall_credit,
        gold_total=gold_total,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def deteval_1d(
    gold: Sequence[Span],
    pred: Sequence[Span],
    *,
    recall_threshold: float = 0.8,
    precision_threshold: float = 0.4,
    granularity_penalty: float = 0.8,
) -> DetectionPRF:
    """One-dimensional DetEval adaptation using interval length as area.

    Matching follows the standard order: one-to-one, one-to-many (split), then
    many-to-one (merge). A split or merge gives every participating instance
    ``granularity_penalty`` credit; unmatched instances receive zero credit.
    """
    g = sorted(_normalize_spans(gold))
    p = sorted(_normalize_spans(pred))
    if not 0.0 <= recall_threshold <= 1.0:
        raise ValueError("recall_threshold must be in [0,1]")
    if not 0.0 <= precision_threshold <= 1.0:
        raise ValueError("precision_threshold must be in [0,1]")
    if not 0.0 <= granularity_penalty <= 1.0:
        raise ValueError("granularity_penalty must be in [0,1]")

    intersections = [
        [overlap_len(gold_span, pred_span) for pred_span in p]
        for gold_span in g
    ]
    area_recall = [
        [value / _len_inc(gold_span) for value in row]
        for gold_span, row in zip(g, intersections)
    ]
    area_precision = [
        [intersections[i][j] / _len_inc(p[j]) for j in range(len(p))]
        for i in range(len(g))
    ]

    unmatched_gold = set(range(len(g)))
    unmatched_pred = set(range(len(p)))
    recall_credit = [0.0] * len(g)
    precision_credit = [0.0] * len(p)

    # The official implementation first accepts geometrically unique overlaps.
    # Ambiguous rows/columns are handled by the split/merge passes below.
    one_to_one: List[Tuple[int, int]] = []
    for i in range(len(g)):
        row = [j for j in range(len(p)) if intersections[i][j] > 0]
        if len(row) != 1:
            continue
        j = row[0]
        column = [ii for ii in range(len(g)) if intersections[ii][j] > 0]
        if (
            len(column) == 1
            and area_recall[i][j] >= recall_threshold
            and area_precision[i][j] >= precision_threshold
        ):
            one_to_one.append((i, j))
    for i, j in one_to_one:
        if i not in unmatched_gold or j not in unmatched_pred:
            continue
        recall_credit[i] = 1.0
        precision_credit[j] = 1.0
        unmatched_gold.remove(i)
        unmatched_pred.remove(j)

    # Split: every contributing prediction has enough area precision and their
    # summed area recall covers the ground-truth instance.
    for i in sorted(tuple(unmatched_gold)):
        if sum(intersections[i][j] > 0 for j in range(len(p))) <= 1:
            continue
        candidates = [
            j
            for j in sorted(unmatched_pred)
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
                unmatched_gold.remove(i)
                unmatched_pred.remove(j)
        elif (
            len(candidates) >= 2
            and sum(area_recall[i][j] for j in candidates) >= recall_threshold
        ):
            recall_credit[i] = granularity_penalty
            for j in candidates:
                precision_credit[j] = granularity_penalty
            unmatched_gold.remove(i)
            unmatched_pred.difference_update(candidates)

    # Merge: every contributing gold has enough area recall and their summed
    # area precision covers the prediction.
    for j in sorted(tuple(unmatched_pred)):
        if sum(intersections[i][j] > 0 for i in range(len(g))) <= 1:
            continue
        candidates = [
            i
            for i in sorted(unmatched_gold)
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
                unmatched_pred.remove(j)
                unmatched_gold.remove(i)
        elif (
            len(candidates) >= 2
            and sum(area_precision[i][j] for i in candidates)
            >= precision_threshold
        ):
            precision_credit[j] = granularity_penalty
            for i in candidates:
                recall_credit[i] = granularity_penalty
            unmatched_pred.remove(j)
            unmatched_gold.difference_update(candidates)

    return _detection_prf(
        sum(precision_credit), len(p), sum(recall_credit), len(g)
    )


def cleval_1d(
    gold: Sequence[Span],
    pred: Sequence[Span],
    *,
    area_precision_threshold: float = 0.3,
    granularity_penalty_weight: float = 1.0,
) -> DetectionPRF:
    """One-dimensional CLEval detection adaptation.

    Every integer character offset in a gold span is a pseudo-character centre.
    Character coverage supplies the credit and ``(k - 1) * weight`` penalizes
    split/merge granularity. Unlike 2D OCR, the number of predicted characters
    is observed directly as interval length, so no aspect-ratio estimate is used.
    """
    g = sorted(_normalize_spans(gold))
    p = sorted(_normalize_spans(pred))
    if not 0.0 <= area_precision_threshold <= 1.0:
        raise ValueError("area_precision_threshold must be in [0,1]")
    if granularity_penalty_weight < 0.0:
        raise ValueError("granularity_penalty_weight must be non-negative")

    intersections = [
        [overlap_len(gold_span, pred_span) for pred_span in p]
        for gold_span in g
    ]
    area_precision = [
        [intersections[i][j] / _len_inc(p[j]) for j in range(len(p))]
        for i in range(len(g))
    ]
    ap_qualified = [
        [area_precision[i][j] >= area_precision_threshold for j in range(len(p))]
        for i in range(len(g))
    ]

    # This is the CLEval match-matrix construction specialized to dense 1D
    # character centres: one-to-one, one-to-many, and many-to-one edges are
    # unioned before character scoring.
    match = [[False] * len(p) for _ in range(len(g))]
    for i in range(len(g)):
        for j in range(len(p)):
            if intersections[i][j] <= 0 or not ap_qualified[i][j]:
                continue
            row_degree = sum(
                intersections[i][jj] > 0 and ap_qualified[i][jj]
                for jj in range(len(p))
            )
            col_degree = sum(
                intersections[ii][j] > 0 and ap_qualified[ii][j]
                for ii in range(len(g))
            )
            if row_degree == 1 and col_degree == 1:
                match[i][j] = True

    for i in range(len(g)):
        candidates = [
            j
            for j in range(len(p))
            if intersections[i][j] > 0 and ap_qualified[i][j]
        ]
        if len(candidates) >= 2:
            for j in candidates:
                match[i][j] = True

    for j in range(len(p)):
        candidates = [i for i in range(len(g)) if intersections[i][j] > 0]
        if (
            len(candidates) >= 2
            and sum(area_precision[i][j] for i in candidates)
            >= area_precision_threshold
        ):
            for i in candidates:
                match[i][j] = True

    matched_gold_by_pred = [
        [i for i in range(len(g)) if match[i][j]] for j in range(len(p))
    ]
    matched_pred_by_gold = [
        [j for j in range(len(p)) if match[i][j]] for i in range(len(g))
    ]
    covered_gold_chars = set()
    for i, gold_span in enumerate(g):
        for j in matched_pred_by_gold[i]:
            start = max(gold_span[0], p[j][0])
            end = min(gold_span[1], p[j][1])
            if start <= end:
                covered_gold_chars.update(range(start, end + 1))

    correct_characters = len(covered_gold_chars)
    recall_penalty = sum(
        max(len(indices) - 1, 0) * granularity_penalty_weight
        for indices in matched_pred_by_gold
    )
    precision_penalty = sum(
        max(len(indices) - 1, 0) * granularity_penalty_weight
        for indices in matched_gold_by_pred
    )
    recall_credit = max(0.0, correct_characters - recall_penalty)
    precision_credit = max(0.0, correct_characters - precision_penalty)
    return _detection_prf(
        precision_credit,
        sum(_len_inc(span) for span in p),
        recall_credit,
        sum(_len_inc(span) for span in g),
    )


def score_one(
    gold: Sequence[Span],
    pred: Sequence[Span],
    *,
    delta: int,
    min_pred_len: int,
    deteval_recall_threshold: float = 0.8,
    deteval_precision_threshold: float = 0.4,
    deteval_granularity_penalty: float = 0.8,
    cleval_area_precision_threshold: float = 0.3,
    cleval_granularity_penalty: float = 1.0,
) -> Dict[str, Any]:
    cp, tp, hg, tg = span_coverage_counts_one(
        gold, pred, delta=delta, min_pred_len=min_pred_len
    )
    if tp == 0 and tg == 0:
        cov_p = cov_r = cov_f1 = 1.0
    else:
        cov_p = cp / tp if tp > 0 else 1.0
        cov_r = hg / tg if tg > 0 else 1.0
        cov_f1 = _fbeta_from_pr(cov_p, cov_r)

    exact_matches, exact_pred, exact_gold = exact_span_counts(gold, pred)
    _, _, exact_f1 = _prf_from_counts(exact_matches, exact_pred, exact_gold)

    intersection, pred_chars, gold_chars, union = char_counts(gold, pred)
    _, _, character_f1 = _prf_from_counts(intersection, pred_chars, gold_chars)
    character_iou = intersection / union if union else 1.0
    deteval = deteval_1d(
        gold,
        pred,
        recall_threshold=deteval_recall_threshold,
        precision_threshold=deteval_precision_threshold,
        granularity_penalty=deteval_granularity_penalty,
    )
    cleval = cleval_1d(
        gold,
        pred,
        area_precision_threshold=cleval_area_precision_threshold,
        granularity_penalty_weight=cleval_granularity_penalty,
    )

    return {
        "span_coverage_precision": cov_p,
        "span_coverage_recall": cov_r,
        "span_coverage_f1": cov_f1,
        "exact_span_f1": exact_f1,
        "character_f1": character_f1,
        "character_iou": character_iou,
        "deteval_precision": deteval.precision,
        "deteval_recall": deteval.recall,
        "deteval_f1": deteval.f1,
        "cleval_precision": cleval.precision,
        "cleval_recall": cleval.recall,
        "cleval_f1": cleval.f1,
        "_coverage_counts": (cp, tp, hg, tg),
        "_exact_counts": (exact_matches, exact_pred, exact_gold),
        "_char_counts": (intersection, pred_chars, gold_chars, union),
        "_deteval_counts": (
            deteval.precision_credit,
            deteval.pred_total,
            deteval.recall_credit,
            deteval.gold_total,
        ),
        "_cleval_counts": (
            cleval.precision_credit,
            cleval.pred_total,
            cleval.recall_credit,
            cleval.gold_total,
        ),
    }


def _componentwise_contained(inner: Sequence[Span], outer: Sequence[Span]) -> bool:
    if len(inner) != len(outer):
        return False
    return all(_contained(i, o) for i, o in zip(sorted(inner), sorted(outer)))


def classify_pair(a: Sequence[Span], b: Sequence[Span]) -> Tuple[str, str]:
    chars_a = span_chars(a)
    chars_b = span_chars(b)
    if not chars_a and not chars_b:
        return "both_empty", "both_empty"
    if not chars_a or not chars_b:
        return "one_empty", "one_empty"
    if chars_a == chars_b:
        return "exact", "exact"
    if chars_a < chars_b:
        if _componentwise_contained(a, b):
            return "a_inside_b_boundary", "boundary_nested"
        return "a_inside_b_extra_components", "containment_count_change"
    if chars_b < chars_a:
        if _componentwise_contained(b, a):
            return "b_inside_a_boundary", "boundary_nested"
        return "b_inside_a_extra_components", "containment_count_change"
    if chars_a & chars_b:
        return "partial_overlap", "partial_overlap"
    return "disjoint", "disjoint"


def directional_relation(pred: Sequence[Span], gold: Sequence[Span]) -> Tuple[str, str]:
    relation, _ = classify_pair(pred, gold)
    mapping = {
        "a_inside_b_boundary": ("pred_inside_gold_boundary", "pred_inside_gold_boundary"),
        "a_inside_b_extra_components": (
            "pred_inside_gold_extra_components",
            "pred_inside_gold_extra_components",
        ),
        "b_inside_a_boundary": ("gold_inside_pred_boundary", "gold_inside_pred_boundary"),
        "b_inside_a_extra_components": (
            "gold_inside_pred_extra_components",
            "gold_inside_pred_extra_components",
        ),
    }
    return mapping.get(relation, (relation, relation))


# ---------------------------------------------------------------------------
# Dataset loading and experimental records
# ---------------------------------------------------------------------------


@dataclass
class Annotation:
    annotator_id: str
    spans: List[Span]


@dataclass
class Example:
    item_id: str
    text: str
    annotations: List[Annotation]


def load_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    if path.read_bytes()[:42].startswith(b"version https://git-lfs.github.com/spec"):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not a Parquet file. Run `git lfs pull` "
            "or the curl command in this experiment's README."
        )
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required. Install validation/requirements.txt"
        ) from exc
    return pq.read_table(path).to_pylist()


def prepare_examples(
    rows: Sequence[Mapping[str, Any]], *, merge_adjacent: bool
) -> Tuple[List[Example], Dict[str, Any]]:
    qa: Counter = Counter()
    annotator_ids = set()
    annotator_count_distribution: Counter = Counter()
    examples: List[Example] = []

    for row in rows:
        qa["items_total"] += 1
        text = str(row["model_output_text"])
        annotations: List[Annotation] = []
        raw_annotations = row.get("annotations") or []
        qa["annotation_slots_total"] += len(raw_annotations)

        for annotation in raw_annotations:
            labels = annotation.get("labels")
            if labels is None:
                qa["missing_annotation_slots"] += 1
                continue
            qa["observed_annotations"] += 1
            annotator_id = str(annotation["annotator_id"])
            annotator_ids.add(annotator_id)
            spans = clean_half_open_spans(
                labels,
                text_length=len(text),
                qa=qa,
                merge_adjacent=merge_adjacent,
            )
            if spans:
                qa["positive_annotations"] += 1
            else:
                qa["explicit_empty_annotations"] += 1
            annotations.append(Annotation(annotator_id, spans))

        annotator_count_distribution[len(annotations)] += 1
        if len(annotations) >= 2:
            qa["items_with_at_least_two_annotations"] += 1
        examples.append(Example(str(row["id"]), text, annotations))

    summary: Dict[str, Any] = dict(sorted(qa.items()))
    summary["unique_observed_annotators"] = len(annotator_ids)
    summary["observed_annotators_per_item"] = {
        str(k): v for k, v in sorted(annotator_count_distribution.items())
    }
    summary["source_offset_convention"] = "half-open [start,end)"
    summary["metric_offset_convention"] = "inclusive [start,end]"
    summary["merge_adjacent"] = merge_adjacent
    return examples, summary


def build_pairwise_records(
    examples: Sequence[Example],
    *,
    delta: int,
    min_pred_len: int,
    deteval_recall_threshold: float = 0.8,
    deteval_precision_threshold: float = 0.4,
    deteval_granularity_penalty: float = 0.8,
    cleval_area_precision_threshold: float = 0.3,
    cleval_granularity_penalty: float = 1.0,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for example in examples:
        for ann_a, ann_b in itertools.combinations(example.annotations, 2):
            relation, family = classify_pair(ann_a.spans, ann_b.spans)
            a_gold = score_one(
                ann_a.spans,
                ann_b.spans,
                delta=delta,
                min_pred_len=min_pred_len,
                deteval_recall_threshold=deteval_recall_threshold,
                deteval_precision_threshold=deteval_precision_threshold,
                deteval_granularity_penalty=deteval_granularity_penalty,
                cleval_area_precision_threshold=cleval_area_precision_threshold,
                cleval_granularity_penalty=cleval_granularity_penalty,
            )
            b_gold = score_one(
                ann_b.spans,
                ann_a.spans,
                delta=delta,
                min_pred_len=min_pred_len,
                deteval_recall_threshold=deteval_recall_threshold,
                deteval_precision_threshold=deteval_precision_threshold,
                deteval_granularity_penalty=deteval_granularity_penalty,
                cleval_area_precision_threshold=cleval_area_precision_threshold,
                cleval_granularity_penalty=cleval_granularity_penalty,
            )

            narrow_to_broad: Optional[float] = None
            broad_to_narrow: Optional[float] = None
            deteval_narrow_to_broad: Optional[float] = None
            deteval_broad_to_narrow: Optional[float] = None
            cleval_narrow_to_broad: Optional[float] = None
            cleval_broad_to_narrow: Optional[float] = None
            if relation.startswith("a_inside_b"):
                narrow_to_broad = b_gold["span_coverage_f1"]  # gold=b, pred=a
                broad_to_narrow = a_gold["span_coverage_f1"]
                deteval_narrow_to_broad = b_gold["deteval_f1"]
                deteval_broad_to_narrow = a_gold["deteval_f1"]
                cleval_narrow_to_broad = b_gold["cleval_f1"]
                cleval_broad_to_narrow = a_gold["cleval_f1"]
            elif relation.startswith("b_inside_a"):
                narrow_to_broad = a_gold["span_coverage_f1"]  # gold=a, pred=b
                broad_to_narrow = b_gold["span_coverage_f1"]
                deteval_narrow_to_broad = a_gold["deteval_f1"]
                deteval_broad_to_narrow = b_gold["deteval_f1"]
                cleval_narrow_to_broad = a_gold["cleval_f1"]
                cleval_broad_to_narrow = b_gold["cleval_f1"]

            records.append(
                {
                    "item_id": example.item_id,
                    "annotator_a": ann_a.annotator_id,
                    "annotator_b": ann_b.annotator_id,
                    "relation": relation,
                    "relation_family": family,
                    "a_num_spans": len(ann_a.spans),
                    "b_num_spans": len(ann_b.spans),
                    "a_num_chars": len(span_chars(ann_a.spans)),
                    "b_num_chars": len(span_chars(ann_b.spans)),
                    "coverage_f1_a_as_gold": a_gold["span_coverage_f1"],
                    "coverage_f1_b_as_gold": b_gold["span_coverage_f1"],
                    "span_coverage_f1": statistics.fmean(
                        [a_gold["span_coverage_f1"], b_gold["span_coverage_f1"]]
                    ),
                    "coverage_f1_best_direction": max(
                        a_gold["span_coverage_f1"], b_gold["span_coverage_f1"]
                    ),
                    "coverage_f1_narrow_to_broad": narrow_to_broad,
                    "coverage_f1_broad_to_narrow": broad_to_narrow,
                    "deteval_f1_a_as_gold": a_gold["deteval_f1"],
                    "deteval_f1_b_as_gold": b_gold["deteval_f1"],
                    "deteval_f1": statistics.fmean(
                        [a_gold["deteval_f1"], b_gold["deteval_f1"]]
                    ),
                    "deteval_f1_narrow_to_broad": deteval_narrow_to_broad,
                    "deteval_f1_broad_to_narrow": deteval_broad_to_narrow,
                    "cleval_f1_a_as_gold": a_gold["cleval_f1"],
                    "cleval_f1_b_as_gold": b_gold["cleval_f1"],
                    "cleval_f1": statistics.fmean(
                        [a_gold["cleval_f1"], b_gold["cleval_f1"]]
                    ),
                    "cleval_f1_narrow_to_broad": cleval_narrow_to_broad,
                    "cleval_f1_broad_to_narrow": cleval_broad_to_narrow,
                    "exact_span_f1": a_gold["exact_span_f1"],
                    "character_f1": a_gold["character_f1"],
                    "character_iou": a_gold["character_iou"],
                }
            )
    return records


def build_consensus(annotations: Sequence[Annotation], mode: str) -> List[Span]:
    if not annotations:
        return []
    masks = [span_chars(annotation.spans) for annotation in annotations]
    if mode == "union":
        chosen = set().union(*masks)
    elif mode == "majority":
        votes: Counter = Counter(idx for mask in masks for idx in mask)
        chosen = {idx for idx, count in votes.items() if count > len(masks) / 2}
    else:
        raise ValueError(f"Unknown consensus mode: {mode}")
    return mask_to_spans(chosen)


def minimal_contained_prediction(gold: Sequence[Span]) -> List[Span]:
    """Return one central character for every non-empty gold span.

    This is an intentionally degenerate oracle prediction used to expose the
    fact that Span Coverage checks containment and instance hits, not how much
    of each gold span is covered.
    """
    return [((start + end) // 2, (start + end) // 2) for start, end in gold]


def build_loo_records(
    examples: Sequence[Example],
    *,
    delta: int,
    min_pred_len: int,
    deteval_recall_threshold: float = 0.8,
    deteval_precision_threshold: float = 0.4,
    deteval_granularity_penalty: float = 0.8,
    cleval_area_precision_threshold: float = 0.3,
    cleval_granularity_penalty: float = 1.0,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for example in examples:
        if len(example.annotations) < 2:
            continue
        for held_out_index, held_out in enumerate(example.annotations):
            remaining = [
                annotation
                for index, annotation in enumerate(example.annotations)
                if index != held_out_index
            ]
            for mode in ("union", "majority"):
                gold = build_consensus(remaining, mode)
                pred = held_out.spans
                relation, family = directional_relation(pred, gold)
                scores = score_one(
                    gold,
                    pred,
                    delta=delta,
                    min_pred_len=min_pred_len,
                    deteval_recall_threshold=deteval_recall_threshold,
                    deteval_precision_threshold=deteval_precision_threshold,
                    deteval_granularity_penalty=deteval_granularity_penalty,
                    cleval_area_precision_threshold=cleval_area_precision_threshold,
                    cleval_granularity_penalty=cleval_granularity_penalty,
                )
                record: Dict[str, Any] = {
                    "item_id": example.item_id,
                    "held_out_annotator": held_out.annotator_id,
                    "consensus": mode,
                    "remaining_annotators": len(remaining),
                    "relation": relation,
                    "relation_family": family,
                    "pred_num_spans": len(pred),
                    "gold_num_spans": len(gold),
                    "pred_num_chars": len(span_chars(pred)),
                    "gold_num_chars": len(span_chars(gold)),
                    **scores,
                    "_gold_spans": gold,
                    "_pred_spans": pred,
                }
                records.append(record)
    return records


def build_limitation_records(
    examples: Sequence[Example],
    *,
    delta: int,
    min_pred_len: int,
    deteval_recall_threshold: float = 0.8,
    deteval_precision_threshold: float = 0.4,
    deteval_granularity_penalty: float = 0.8,
    cleval_area_precision_threshold: float = 0.3,
    cleval_granularity_penalty: float = 1.0,
) -> List[Dict[str, Any]]:
    """Build a gold-informed stress test for Span Coverage under-coverage.

    The reference is the union of all observed annotator masks for an item.
    ``reference_copy`` is the positive control. ``one_character_per_gold`` is
    the smallest possible contained prediction that still hits every gold span.
    """
    records: List[Dict[str, Any]] = []
    for example in examples:
        gold = build_consensus(example.annotations, "union")
        if not gold:
            continue
        conditions = {
            "reference_copy": list(gold),
            "one_character_per_gold": minimal_contained_prediction(gold),
        }
        gold_chars = len(span_chars(gold))
        for condition, pred in conditions.items():
            scores = score_one(
                gold,
                pred,
                delta=delta,
                min_pred_len=min_pred_len,
                deteval_recall_threshold=deteval_recall_threshold,
                deteval_precision_threshold=deteval_precision_threshold,
                deteval_granularity_penalty=deteval_granularity_penalty,
                cleval_area_precision_threshold=cleval_area_precision_threshold,
                cleval_granularity_penalty=cleval_granularity_penalty,
            )
            pred_chars = len(span_chars(pred))
            records.append(
                {
                    "item_id": example.item_id,
                    "condition": condition,
                    "gold_num_spans": len(gold),
                    "pred_num_spans": len(pred),
                    "gold_num_chars": gold_chars,
                    "pred_num_chars": pred_chars,
                    "predicted_character_fraction": pred_chars / gold_chars,
                    **scores,
                    "_gold_spans": gold,
                    "_pred_spans": pred,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Aggregation, bootstrap, and output
# ---------------------------------------------------------------------------


METRIC_COLUMNS = (
    "span_coverage_f1",
    "deteval_f1",
    "cleval_f1",
    "exact_span_f1",
    "character_f1",
    "character_iou",
)

PAIRWISE_DIRECTIONAL_COLUMNS = (
    "coverage_f1_narrow_to_broad",
    "coverage_f1_broad_to_narrow",
    "span_coverage_f1",
    "deteval_f1_narrow_to_broad",
    "deteval_f1_broad_to_narrow",
    "deteval_f1",
    "cleval_f1_narrow_to_broad",
    "cleval_f1_broad_to_narrow",
    "cleval_f1",
    "exact_span_f1",
    "character_f1",
    "character_iou",
)


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        return math.nan
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: Sequence[float], *, n_bootstrap: int, seed: int
) -> Tuple[float, float]:
    if not values:
        return math.nan, math.nan
    if n_bootstrap <= 0 or len(values) == 1:
        mean = statistics.fmean(values)
        return mean, mean
    rng = random.Random(seed)
    boot = [
        statistics.fmean(rng.choices(values, k=len(values)))
        for _ in range(n_bootstrap)
    ]
    boot.sort()
    return percentile(boot, 0.025), percentile(boot, 0.975)


def summarize_macro_group(
    records: Sequence[Mapping[str, Any]],
    *,
    group_values: Mapping[str, str],
    n_bootstrap: int,
    seed: int,
    metric_columns: Sequence[str] = METRIC_COLUMNS,
) -> Dict[str, Any]:
    by_item: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_item[str(record["item_id"])].append(record)

    out: Dict[str, Any] = dict(group_values)
    out["n_comparisons"] = len(records)
    out["n_items"] = len(by_item)
    for metric in metric_columns:
        item_means = [
            statistics.fmean(float(record[metric]) for record in item_records)
            for item_records in by_item.values()
        ]
        mean = statistics.fmean(item_means) if item_means else math.nan
        low, high = bootstrap_mean_ci(
            item_means,
            n_bootstrap=n_bootstrap,
            seed=stable_seed(seed, json.dumps(group_values, sort_keys=True) + metric),
        )
        out[f"{metric}_mean"] = mean
        out[f"{metric}_ci_low"] = low
        out[f"{metric}_ci_high"] = high
    return out


def summarize_pairwise_directional(
    records: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for family in ("boundary_nested", "containment_count_change"):
        family_records = [
            record
            for record in records
            if record["relation_family"] == family
            and record["coverage_f1_narrow_to_broad"] is not None
        ]
        if not family_records:
            continue
        outputs.append(
            summarize_macro_group(
                family_records,
                group_values={"relation_family": family},
                n_bootstrap=n_bootstrap,
                seed=seed,
                metric_columns=PAIRWISE_DIRECTIONAL_COLUMNS,
            )
        )
    return outputs


def summarize_macro(
    records: Sequence[Mapping[str, Any]],
    *,
    primary_group: str,
    n_bootstrap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    modes = sorted({str(record.get(primary_group, "all")) for record in records})
    for mode in modes:
        mode_records = [record for record in records if str(record.get(primary_group, "all")) == mode]
        results.append(
            summarize_macro_group(
                mode_records,
                group_values={primary_group: mode, "relation_family": "all"},
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        )
        families = sorted({str(record["relation_family"]) for record in mode_records})
        for family in families:
            family_records = [
                record for record in mode_records if str(record["relation_family"]) == family
            ]
            results.append(
                summarize_macro_group(
                    family_records,
                    group_values={primary_group: mode, "relation_family": family},
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
            )
    return results


def summarize_limitation(
    records: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for condition in ("reference_copy", "one_character_per_gold"):
        condition_records = [
            record for record in records if record["condition"] == condition
        ]
        if not condition_records:
            continue
        row = summarize_macro_group(
            condition_records,
            group_values={"condition": condition},
            n_bootstrap=n_bootstrap,
            seed=seed,
            metric_columns=(*METRIC_COLUMNS, "predicted_character_fraction"),
        )
        micro = _micro_from_hidden_counts(condition_records)
        row.update({f"{metric}_micro": value for metric, value in micro.items()})
        total_pred_chars = sum(int(record["pred_num_chars"]) for record in condition_records)
        total_gold_chars = sum(int(record["gold_num_chars"]) for record in condition_records)
        row["predicted_character_fraction_micro"] = (
            total_pred_chars / total_gold_chars if total_gold_chars else 1.0
        )
        outputs.append(row)
    return outputs


def _micro_from_hidden_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    cp = tp = hg = tg = 0
    exact_match = exact_pred = exact_gold = 0
    char_intersection = char_pred = char_gold = char_union = 0
    det_precision_credit = det_recall_credit = 0.0
    det_pred = det_gold = 0
    cle_precision_credit = cle_recall_credit = 0.0
    cle_pred = cle_gold = 0
    for record in records:
        c_cp, c_tp, c_hg, c_tg = record["_coverage_counts"]
        cp += c_cp
        tp += c_tp
        hg += c_hg
        tg += c_tg
        e_match, e_pred, e_gold = record["_exact_counts"]
        exact_match += e_match
        exact_pred += e_pred
        exact_gold += e_gold
        c_inter, c_pred, c_gold, c_union = record["_char_counts"]
        char_intersection += c_inter
        char_pred += c_pred
        char_gold += c_gold
        char_union += c_union
        d_pc, d_pt, d_rc, d_gt = record["_deteval_counts"]
        det_precision_credit += d_pc
        det_pred += d_pt
        det_recall_credit += d_rc
        det_gold += d_gt
        c_pc, c_pt, c_rc, c_gt = record["_cleval_counts"]
        cle_precision_credit += c_pc
        cle_pred += c_pt
        cle_recall_credit += c_rc
        cle_gold += c_gt

    if tp == 0 and tg == 0:
        coverage_f1 = 1.0
    else:
        coverage_precision = cp / tp if tp else 1.0
        coverage_recall = hg / tg if tg else 1.0
        coverage_f1 = _fbeta_from_pr(coverage_precision, coverage_recall)
    _, _, exact_f1 = _prf_from_counts(exact_match, exact_pred, exact_gold)
    _, _, character_f1 = _prf_from_counts(char_intersection, char_pred, char_gold)
    character_iou = char_intersection / char_union if char_union else 1.0
    deteval_f1 = _detection_prf(
        det_precision_credit, det_pred, det_recall_credit, det_gold
    ).f1
    cleval_f1 = _detection_prf(
        cle_precision_credit, cle_pred, cle_recall_credit, cle_gold
    ).f1
    return {
        "span_coverage_f1": coverage_f1,
        "deteval_f1": deteval_f1,
        "cleval_f1": cleval_f1,
        "exact_span_f1": exact_f1,
        "character_f1": character_f1,
        "character_iou": character_iou,
    }


def summarize_micro(
    records: Sequence[Mapping[str, Any]],
    *,
    n_bootstrap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for mode in sorted({str(record["consensus"]) for record in records}):
        mode_records = [record for record in records if str(record["consensus"]) == mode]
        point = _micro_from_hidden_counts(mode_records)
        by_item: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for record in mode_records:
            by_item[str(record["item_id"])].append(record)
        item_ids = list(by_item)
        rng = random.Random(stable_seed(seed, f"micro:{mode}"))
        boot_values: Dict[str, List[float]] = {metric: [] for metric in METRIC_COLUMNS}
        if n_bootstrap > 0 and item_ids:
            for _ in range(n_bootstrap):
                sample_ids = rng.choices(item_ids, k=len(item_ids))
                sample_records = [
                    record for item_id in sample_ids for record in by_item[item_id]
                ]
                sample_scores = _micro_from_hidden_counts(sample_records)
                for metric in METRIC_COLUMNS:
                    boot_values[metric].append(sample_scores[metric])

        row: Dict[str, Any] = {
            "consensus": mode,
            "n_comparisons": len(mode_records),
            "n_items": len(item_ids),
        }
        for metric in METRIC_COLUMNS:
            values = sorted(boot_values[metric])
            row[f"{metric}_micro"] = point[metric]
            row[f"{metric}_ci_low"] = (
                percentile(values, 0.025) if values else point[metric]
            )
            row[f"{metric}_ci_high"] = (
                percentile(values, 0.975) if values else point[metric]
            )
        outputs.append(row)
    return outputs


def public_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    public = [public_record(record) for record in records]
    if not public:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for record in public:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(public)


def format_ci(row: Mapping[str, Any], metric: str) -> str:
    return (
        f"{float(row[f'{metric}_mean']):.3f} "
        f"[{float(row[f'{metric}_ci_low']):.3f}, "
        f"{float(row[f'{metric}_ci_high']):.3f}]"
    )


def select_summary_row(
    rows: Sequence[Mapping[str, Any]], consensus: str, family: str
) -> Optional[Mapping[str, Any]]:
    for row in rows:
        if row.get("consensus") == consensus and row.get("relation_family") == family:
            return row
    return None


def write_paper_outputs(
    output_dir: Path,
    *,
    dataset_summary: Mapping[str, Any],
    pairwise_records: Sequence[Mapping[str, Any]],
    pairwise_directional_summary: Sequence[Mapping[str, Any]],
    loo_summary: Sequence[Mapping[str, Any]],
    loo_micro_summary: Sequence[Mapping[str, Any]],
    limitation_summary: Sequence[Mapping[str, Any]],
) -> None:
    desired = [
        ("all", "All LOO comparisons"),
        ("exact", "Exact agreement"),
        ("pred_inside_gold_boundary", "Boundary-only: prediction inside reference"),
        (
            "pred_inside_gold_extra_components",
            "Prediction inside reference; component count differs",
        ),
        ("partial_overlap", "Partial overlap"),
        ("disjoint", "Disjoint non-empty annotations"),
        ("one_empty", "Empty/non-empty disagreement"),
        ("both_empty", "Both empty"),
    ]
    rows = [
        (label, select_summary_row(loo_summary, "union", family))
        for family, label in desired
    ]
    rows = [(label, row) for label, row in rows if row is not None]

    md = [
        "# Mu-SHROOM English test: leave-one-annotator-out union reference",
        "",
        "Scores are example-clustered macro means with 95% bootstrap confidence intervals.",
        "",
        "| Subset | Comparisons | Items | Span Coverage F1 | DetEval (1D) F1 | CLEval (1D) F1 | Exact span F1 | Character F1 | Character IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in rows:
        md.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row["n_comparisons"]),
                    str(row["n_items"]),
                    format_ci(row, "span_coverage_f1"),
                    format_ci(row, "deteval_f1"),
                    format_ci(row, "cleval_f1"),
                    format_ci(row, "exact_span_f1"),
                    format_ci(row, "character_f1"),
                    format_ci(row, "character_iou"),
                ]
            )
            + " |"
        )
    (output_dir / "paper_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    latex = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Subset & $n$ & Coverage F1 & DetEval & CLEval & Exact F1 & Char F1 & Char IoU \\",
        r"\midrule",
    ]
    for label, row in rows:
        safe_label = label.replace("&", r"\&").replace("_", r"\_")
        latex.append(
            f"{safe_label} & {row['n_comparisons']} & "
            f"{float(row['span_coverage_f1_mean']):.3f} & "
            f"{float(row['deteval_f1_mean']):.3f} & "
            f"{float(row['cleval_f1_mean']):.3f} & "
            f"{float(row['exact_span_f1_mean']):.3f} & "
            f"{float(row['character_f1_mean']):.3f} & "
            f"{float(row['character_iou_mean']):.3f} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "paper_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")

    limitation_labels = {
        "reference_copy": "Reference copied exactly",
        "one_character_per_gold": "One character per reference span",
    }
    limitation_md = [
        "# Span Coverage F1 limitation: minimal-contained stress test",
        "",
        (
            "The prediction is constructed from the all-annotator union reference. "
            "This is a diagnostic adversarial test, not a model evaluation. Scores are "
            "example-level macro means with 95% cluster-bootstrap confidence intervals."
        ),
        "",
        "| Condition | Items | Predicted/gold characters | Span Coverage F1 | DetEval (1D) F1 | CLEval (1D) F1 | Exact span F1 | Character F1 | Character IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in limitation_summary:
        limitation_md.append(
            "| "
            + " | ".join(
                [
                    limitation_labels[str(row["condition"])],
                    str(row["n_items"]),
                    format_ci(row, "predicted_character_fraction"),
                    format_ci(row, "span_coverage_f1"),
                    format_ci(row, "deteval_f1"),
                    format_ci(row, "cleval_f1"),
                    format_ci(row, "exact_span_f1"),
                    format_ci(row, "character_f1"),
                    format_ci(row, "character_iou"),
                ]
            )
            + " |"
        )
    (output_dir / "limitation_table.md").write_text(
        "\n".join(limitation_md) + "\n", encoding="utf-8"
    )

    limitation_latex = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Condition & Char. fraction & Coverage F1 & DetEval & CLEval & Exact F1 & Char F1 & Char IoU \\",
        r"\midrule",
    ]
    for row in limitation_summary:
        label = limitation_labels[str(row["condition"])].replace("&", r"\&")
        limitation_latex.append(
            f"{label} & {float(row['predicted_character_fraction_mean']):.3f} & "
            f"{float(row['span_coverage_f1_mean']):.3f} & "
            f"{float(row['deteval_f1_mean']):.3f} & "
            f"{float(row['cleval_f1_mean']):.3f} & "
            f"{float(row['exact_span_f1_mean']):.3f} & "
            f"{float(row['character_f1_mean']):.3f} & "
            f"{float(row['character_iou_mean']):.3f} \\\\"
        )
    limitation_latex.extend([r"\bottomrule", r"\end{tabular}"])
    (output_dir / "limitation_table.tex").write_text(
        "\n".join(limitation_latex) + "\n", encoding="utf-8"
    )

    relation_counts = Counter(str(record["relation_family"]) for record in pairwise_records)
    nontrivial = sum(
        count
        for relation, count in relation_counts.items()
        if relation not in {"exact", "both_empty", "one_empty"}
    )
    boundary_nested = relation_counts.get("boundary_nested", 0)
    boundary_share = boundary_nested / nontrivial if nontrivial else math.nan

    summary_lines = [
        "# Experimental summary",
        "",
        f"- Dataset items: {dataset_summary.get('items_total', 0)}.",
        f"- Observed individual annotations: {dataset_summary.get('observed_annotations', 0)}.",
        f"- Pairwise annotator comparisons: {len(pairwise_records)}.",
        (
            "- Boundary-nested pairs among non-exact, non-empty geometric "
            f"disagreements: {boundary_nested}/{nontrivial} ({boundary_share:.1%})."
        ),
        "- `labels=None` entries were treated as missing; explicit empty lists were retained.",
        (
            "- Raw zero-length half-open spans dropped during QA: "
            f"{dataset_summary.get('zero_length_spans_dropped', 0)}."
        ),
        "",
        "## Direct pairwise containment analysis",
        "",
        (
            "For nested pairs, `narrow -> broad` evaluates the narrower annotation as "
            "prediction and the broader annotation as reference; `broad -> narrow` reverses them."
        ),
        "",
        "| Relation | Comparisons | Coverage narrow -> broad | Coverage broad -> narrow | Symmetric Coverage | DetEval narrow -> broad | DetEval broad -> narrow | Symmetric DetEval | CLEval narrow -> broad | CLEval broad -> narrow | Symmetric CLEval | Exact span F1 | Character F1 | Character IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairwise_directional_summary:
        summary_lines.append(
            f"| {row['relation_family']} | {row['n_comparisons']} | "
            f"{float(row['coverage_f1_narrow_to_broad_mean']):.3f} | "
            f"{float(row['coverage_f1_broad_to_narrow_mean']):.3f} | "
            f"{float(row['span_coverage_f1_mean']):.3f} | "
            f"{float(row['deteval_f1_narrow_to_broad_mean']):.3f} | "
            f"{float(row['deteval_f1_broad_to_narrow_mean']):.3f} | "
            f"{float(row['deteval_f1_mean']):.3f} | "
            f"{float(row['cleval_f1_narrow_to_broad_mean']):.3f} | "
            f"{float(row['cleval_f1_broad_to_narrow_mean']):.3f} | "
            f"{float(row['cleval_f1_mean']):.3f} | "
            f"{float(row['exact_span_f1_mean']):.3f} | "
            f"{float(row['character_f1_mean']):.3f} | "
            f"{float(row['character_iou_mean']):.3f} |"
        )
    summary_lines.extend(
        [
        "",
        "## Corpus-level micro scores",
        "",
        "| Consensus | Span Coverage F1 | DetEval (1D) F1 | CLEval (1D) F1 | Exact span F1 | Character F1 | Character IoU |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in loo_micro_summary:
        summary_lines.append(
            f"| {row['consensus']} | "
            f"{float(row['span_coverage_f1_micro']):.3f} | "
            f"{float(row['deteval_f1_micro']):.3f} | "
            f"{float(row['cleval_f1_micro']):.3f} | "
            f"{float(row['exact_span_f1_micro']):.3f} | "
            f"{float(row['character_f1_micro']):.3f} | "
            f"{float(row['character_iou_micro']):.3f} |"
        )
    summary_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                "`pred_inside_gold_boundary` is the key coarse-boundary condition: "
                "the held-out annotation has the same connected-component count as the "
                "LOO union reference and is strictly contained by it. A high Coverage F1 "
                "on this subset, paired with a low score on `disjoint`, supports robustness "
                "to outward boundary noise without claiming semantic equivalence for all disagreements."
            ),
            "",
            (
                "Because Span Coverage is directional, the pairwise CSV reports both orientations. "
                "The LOO union experiment uses the intended evaluation orientation: a potentially "
                "fine-grained held-out span is the prediction and the remaining annotators' envelope "
                "is the reference."
            ),
            "",
            (
                "DetEval and CLEval are reported as explicit 1D adaptations: interval length replaces "
                "box area, and character offsets replace CLEval pseudo-character centres. DetEval "
                "uses tr=0.8, tp=0.4, and split/merge credit 0.8; CLEval uses area-precision "
                "threshold 0.3 and a one-character (k-1) granularity penalty."
            ),
            "",
            "## Synthetic limitation stress test",
            "",
            (
                "For every item, the all-annotator union is used as the reference. The adversarial "
                "prediction retains only one central character from each reference span. It therefore "
                "hits every gold instance while deliberately discarding almost all annotated content."
            ),
            "",
            "| Condition | Predicted/gold characters | Span Coverage F1 | DetEval (1D) F1 | CLEval (1D) F1 | Exact span F1 | Character F1 | Character IoU |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in limitation_summary:
        summary_lines.append(
            f"| {limitation_labels[str(row['condition'])]} | "
            f"{float(row['predicted_character_fraction_mean']):.3f} | "
            f"{float(row['span_coverage_f1_mean']):.3f} | "
            f"{float(row['deteval_f1_mean']):.3f} | "
            f"{float(row['cleval_f1_mean']):.3f} | "
            f"{float(row['exact_span_f1_mean']):.3f} | "
            f"{float(row['character_f1_mean']):.3f} | "
            f"{float(row['character_iou_mean']):.3f} |"
        )
    summary_lines.extend(
        [
            "",
            (
                "This exposes a deliberate limitation: Span Coverage F1 measures whether predictions "
                "are contained and whether every reference is hit, but not the fraction of reference "
                "content recovered. It should therefore be accompanied by Character IoU (and the "
                "reported length/fragmentation controls) when severe under-coverage is plausible."
            ),
        ]
    )
    (output_dir / "paper_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )


def git_revision(dataset_dir: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(dataset_dir), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/mu-shroom/en/test-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation/output/en_test"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--delta", type=int, default=0)
    parser.add_argument("--min-pred-len", type=int, default=1)
    parser.add_argument("--deteval-recall-threshold", type=float, default=0.8)
    parser.add_argument("--deteval-precision-threshold", type=float, default=0.4)
    parser.add_argument("--deteval-granularity-penalty", type=float, default=0.8)
    parser.add_argument("--cleval-area-precision-threshold", type=float, default=0.3)
    parser.add_argument("--cleval-granularity-penalty", type=float, default=1.0)
    parser.add_argument(
        "--keep-adjacent",
        action="store_true",
        help="Do not merge immediately adjacent selections during canonicalization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap < 0:
        raise ValueError("--bootstrap must be non-negative")
    if args.delta < 0:
        raise ValueError("--delta must be non-negative")
    if args.min_pred_len < 1:
        raise ValueError("--min-pred-len must be at least 1")
    if not 0.0 <= args.deteval_recall_threshold <= 1.0:
        raise ValueError("--deteval-recall-threshold must be in [0,1]")
    if not 0.0 <= args.deteval_precision_threshold <= 1.0:
        raise ValueError("--deteval-precision-threshold must be in [0,1]")
    if not 0.0 <= args.deteval_granularity_penalty <= 1.0:
        raise ValueError("--deteval-granularity-penalty must be in [0,1]")
    if not 0.0 <= args.cleval_area_precision_threshold <= 1.0:
        raise ValueError("--cleval-area-precision-threshold must be in [0,1]")
    if args.cleval_granularity_penalty < 0.0:
        raise ValueError("--cleval-granularity-penalty must be non-negative")

    parquet_path = args.parquet.resolve()
    output_dir = args.output_dir.resolve()
    rows = load_parquet_rows(parquet_path)
    examples, dataset_summary = prepare_examples(
        rows, merge_adjacent=not args.keep_adjacent
    )
    dataset_summary.update(
        {
            "parquet_path": str(parquet_path),
            "parquet_sha256": sha256_file(parquet_path),
            "dataset_git_revision": git_revision(parquet_path.parent.parent),
            "delta": args.delta,
            "min_pred_len": args.min_pred_len,
            "deteval_recall_threshold": args.deteval_recall_threshold,
            "deteval_precision_threshold": args.deteval_precision_threshold,
            "deteval_granularity_penalty": args.deteval_granularity_penalty,
            "cleval_area_precision_threshold": args.cleval_area_precision_threshold,
            "cleval_granularity_penalty": args.cleval_granularity_penalty,
            "bootstrap_replicates": args.bootstrap,
            "bootstrap_seed": args.seed,
        }
    )

    pairwise_records = build_pairwise_records(
        examples,
        delta=args.delta,
        min_pred_len=args.min_pred_len,
        deteval_recall_threshold=args.deteval_recall_threshold,
        deteval_precision_threshold=args.deteval_precision_threshold,
        deteval_granularity_penalty=args.deteval_granularity_penalty,
        cleval_area_precision_threshold=args.cleval_area_precision_threshold,
        cleval_granularity_penalty=args.cleval_granularity_penalty,
    )
    pairwise_summary = summarize_macro(
        [{**record, "analysis": "pairwise"} for record in pairwise_records],
        primary_group="analysis",
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    pairwise_directional_summary = summarize_pairwise_directional(
        pairwise_records,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    loo_records = build_loo_records(
        examples,
        delta=args.delta,
        min_pred_len=args.min_pred_len,
        deteval_recall_threshold=args.deteval_recall_threshold,
        deteval_precision_threshold=args.deteval_precision_threshold,
        deteval_granularity_penalty=args.deteval_granularity_penalty,
        cleval_area_precision_threshold=args.cleval_area_precision_threshold,
        cleval_granularity_penalty=args.cleval_granularity_penalty,
    )
    loo_summary = summarize_macro(
        loo_records,
        primary_group="consensus",
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    loo_micro_summary = summarize_micro(
        loo_records, n_bootstrap=args.bootstrap, seed=args.seed
    )
    limitation_records = build_limitation_records(
        examples,
        delta=args.delta,
        min_pred_len=args.min_pred_len,
        deteval_recall_threshold=args.deteval_recall_threshold,
        deteval_precision_threshold=args.deteval_precision_threshold,
        deteval_granularity_penalty=args.deteval_granularity_penalty,
        cleval_area_precision_threshold=args.cleval_area_precision_threshold,
        cleval_granularity_penalty=args.cleval_granularity_penalty,
    )
    limitation_summary = summarize_limitation(
        limitation_records, n_bootstrap=args.bootstrap, seed=args.seed
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "pairwise_records.csv", pairwise_records)
    write_csv(output_dir / "pairwise_summary.csv", pairwise_summary)
    write_csv(
        output_dir / "pairwise_directional_summary.csv",
        pairwise_directional_summary,
    )
    write_csv(output_dir / "loo_records.csv", loo_records)
    write_csv(output_dir / "loo_summary.csv", loo_summary)
    write_csv(output_dir / "loo_micro_summary.csv", loo_micro_summary)
    write_csv(output_dir / "limitation_stress_records.csv", limitation_records)
    write_csv(output_dir / "limitation_stress_summary.csv", limitation_summary)
    write_paper_outputs(
        output_dir,
        dataset_summary=dataset_summary,
        pairwise_records=pairwise_records,
        pairwise_directional_summary=pairwise_directional_summary,
        loo_summary=loo_summary,
        loo_micro_summary=loo_micro_summary,
        limitation_summary=limitation_summary,
    )

    print(f"Loaded {len(examples)} examples from {parquet_path}")
    print(f"Built {len(pairwise_records)} pairwise comparisons")
    print(f"Built {len(loo_records)} LOO comparisons across union and majority references")
    print(f"Built {len(limitation_records)} synthetic limitation-stress records")
    print(f"Wrote results to {output_dir}")


if __name__ == "__main__":
    main()
