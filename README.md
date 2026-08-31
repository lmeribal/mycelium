# Mycelium: A Meta-Study of Hallucination Detection Across Provenance, Typology, and Granularities

![Mycelium banner](images/mycelium-banner-v2.png)

## About

Hallucination detectors are often compared across benchmarks that
operationalize hallucination in substantially different ways. Mycelium treats
hallucination detection as a **benchmark-conditioned measurement problem** and
studies how the meaning of reported scores changes across provenance,
typology, and granularity.

Provenance describes how outputs and evidence conditions are produced or
selected; typology describes the factuality and faithfulness failures encoded
by benchmark labels; and granularity describes the unit at which
hallucinations are annotated, predicted, and evaluated. This framing connects
benchmark construction with detector mechanisms, access assumptions, and
output units, making the limitations of cross-benchmark comparisons explicit.

## Mycelium scorer

Mycelium exposes a single interface for benchmark-conditioned span metrics.

| Dictionary key | Metric | Default profile aggregation | Provenance |
|---|---|---|---|
| `iou` | Character intersection-over-union | Mu-SHROOM: macro | Reproduces the released Mu-SHROOM participant-kit scorer |
| `character` | Character-mask precision, recall, and F1 | RAGTruth: micro | Independent reconstruction from the RAGTruth paper description\* |
| `span_coverage` | Directional Span Coverage precision, recall, and F1 | Mu-SHROOM: macro; RAGTruth: micro | Proposed in Mycelium |

\*RAGTruth states that span overlap is evaluated with character-level
precision, recall, and F1, but its public repository does not contain the full
span scorer. Mycelium independently implements the stated protocol by merging
spans into per-response character masks and accumulating TP, FP, and FN across
the batch. Undefined precision or recall is set to `0.0`, equivalent to
`zero_division=0`. This implementation is not claimed to be official. See the
[RAGTruth paper](https://aclanthology.org/2024.acl-long.585/) and
[repository](https://github.com/ParticleMedia/RAGTruth).

The default report includes every available metric:

```python
from mycelium import evaluate

report = evaluate(
    references=[[0, 10]],
    predictions=[[2, 8]],
    text="0123456789",
    profile="mu_shroom",
)

print(report)
# {
#   "iou": {"score": 0.6},
#   "character": {"precision": 1.0, "recall": 0.6, "f1": 0.75},
#   "span_coverage": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
#   "meta": {
#     "examples": 1,
#     "average": "macro",
#     "offset_convention": "half-open [start, end)",
#     "profile": "mu_shroom",
#   },
# }
```

The scorer accepts either one span set, as above, or a batch:

```python
report = evaluate(
    references=[[[0, 10]], [[4, 7]]],
    predictions=[[[2, 8]], [[4, 7]]],
    include_per_example=True,
)
```

All public scorer inputs use half-open character offsets `[start, end)`. The
default `average="macro"` reproduces Mu-SHROOM's procedure of computing IoU
for each example and then averaging the scores. If both reference and
prediction are empty, IoU is `1.0`, as in the official scorer.

For the independently reconstructed RAGTruth protocol, use its profile to
select micro aggregation:

```python
report = evaluate(
    references=[[[0, 10]], [[4, 7]]],
    predictions=[[[2, 8]], [[4, 7]]],
    profile="ragtruth",
)
print(report["character"])
```

Install the package locally with:

```bash
python -m pip install -e .
```

## Span Coverage F1

Span boundaries need not have a unique valid annotation. Span Coverage F1 is a
directional, instance-level metric that accepts a more specific predicted span
when it is fully contained within a broader reference annotation.

For predicted spans `P` and reference spans `G`:

- precision is the fraction of predictions fully contained in at least one
  reference span;
- recall is the fraction of reference spans that contain at least one
  prediction;
- Span Coverage F1 is the harmonic mean of these two values.

Containment is asymmetric: a prediction extending beyond the annotated region
is not accepted. The scorer has no third-party dependencies, supports micro
and macro aggregation, and uses half-open character spans `[start, end)`.

```python
from mycelium import evaluate

report = evaluate(
    references=[[0, 11]],
    predictions=[[2, 9]],
    metrics=["span_coverage"],
)
print(report["span_coverage"]["f1"])  # 1.0
```

The legacy standalone `span_coverage.py` module remains available for existing
experiments and continues to use inclusive `[start, end]` offsets.

Span Coverage F1 does not measure how much of a reference span is recovered:
one contained character per reference span can obtain a perfect score. It
should therefore be reported with an overlap-sensitive measure such as
character IoU. The [`validation/`](validation/) directory contains the
reproducible Mu-SHROOM inter-annotator analysis, metric comparisons, and the
corresponding under-coverage diagnostic.
