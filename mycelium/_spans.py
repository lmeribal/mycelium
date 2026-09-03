"""Span input normalization shared by Mycelium metrics.

The public API uses Python-style half-open character spans: ``[start, end)``.
"""

from __future__ import annotations

from numbers import Integral
from typing import List, Optional, Sequence, Tuple, Union


Span = Tuple[int, int]
SpanSet = List[Span]
SpanBatch = List[SpanSet]
TextInput = Optional[Union[str, Sequence[str]]]


def _is_span(value: object) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    try:
        parts = list(value)  # type: ignore[arg-type]
    except TypeError:
        return False
    return (
        len(parts) == 2
        and all(isinstance(part, Integral) and not isinstance(part, bool) for part in parts)
    )


def _normalize_span_set(spans: Sequence[Sequence[int]], *, name: str) -> SpanSet:
    normalized: SpanSet = []
    for index, span in enumerate(spans):
        if not _is_span(span):
            raise ValueError(
                f"{name}[{index}] must be a [start, end] pair of integers; got {span!r}"
            )
        start, end = int(span[0]), int(span[1])
        if start < 0:
            raise ValueError(f"{name}[{index}] starts before 0: {span!r}")
        if end <= start:
            raise ValueError(
                f"{name}[{index}] must satisfy start < end for half-open spans; got {span!r}"
            )
        normalized.append((start, end))
    return sorted(normalized)


def normalize_batch(
    values: Sequence[object],
    *,
    name: str,
) -> SpanBatch:
    """Accept either one span set or a batch of span sets."""

    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must contain spans, not text")

    materialized = list(values)
    if not materialized:
        # An empty outer list is interpreted as one example with no spans.
        return [[]]

    if all(_is_span(value) for value in materialized):
        return [_normalize_span_set(materialized, name=name)]  # type: ignore[arg-type]

    batch: SpanBatch = []
    for example_index, span_set in enumerate(materialized):
        if isinstance(span_set, (str, bytes)):
            raise TypeError(f"{name}[{example_index}] must contain spans, not text")
        try:
            spans = list(span_set)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError(
                f"{name}[{example_index}] must be a sequence of spans"
            ) from exc
        batch.append(
            _normalize_span_set(spans, name=f"{name}[{example_index}]")
        )
    return batch


def normalize_text_lengths(text: TextInput, *, batch_size: int) -> Optional[List[int]]:
    if text is None:
        return None
    if isinstance(text, str):
        if batch_size != 1:
            raise ValueError("A single text string can only validate a single example")
        return [len(text)]

    texts = list(text)
    if len(texts) != batch_size:
        raise ValueError(
            f"text and span batches must have the same length; got {len(texts)} vs {batch_size}"
        )
    if not all(isinstance(value, str) for value in texts):
        raise TypeError("Every item in text must be a string")
    return [len(value) for value in texts]


def validate_bounds(
    batch: SpanBatch,
    *,
    text_lengths: Optional[Sequence[int]],
    name: str,
) -> None:
    if text_lengths is None:
        return
    for example_index, (spans, text_length) in enumerate(zip(batch, text_lengths)):
        for span_index, (_, end) in enumerate(spans):
            if end > text_length:
                raise ValueError(
                    f"{name}[{example_index}][{span_index}] ends at {end}, "
                    f"past text length {text_length}"
                )
