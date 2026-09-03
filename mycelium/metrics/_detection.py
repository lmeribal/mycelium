"""Shared aggregation for instance- and detection-level span metrics."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


DetectionCounts = Tuple[float, int, float, int]


def _prf(counts: DetectionCounts) -> Tuple[float, float, float]:
    precision_credit, prediction_total, recall_credit, reference_total = counts
    if prediction_total == 0 and reference_total == 0:
        return 1.0, 1.0, 1.0
    precision = (
        precision_credit / prediction_total if prediction_total else 1.0
    )
    recall = recall_credit / reference_total if reference_total else 1.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def aggregate_detection(
    counts: Sequence[DetectionCounts],
    *,
    average: str,
    include_per_example: bool,
) -> Dict[str, object]:
    per_example: List[Dict[str, float]] = []
    for item_counts in counts:
        precision, recall, f1 = _prf(item_counts)
        per_example.append({"precision": precision, "recall": recall, "f1": f1})

    if average == "micro":
        totals: DetectionCounts = (
            sum(item[0] for item in counts),
            sum(item[1] for item in counts),
            sum(item[2] for item in counts),
            sum(item[3] for item in counts),
        )
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
