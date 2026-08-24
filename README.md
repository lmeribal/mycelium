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
is not accepted. The reference implementation has no third-party dependencies,
supports micro and macro aggregation, and uses inclusive character spans
`[start, end]`.

```python
from span_coverage import span_coverage_micro

score = span_coverage_micro(
    golds=[[[0, 10]]],
    preds=[[[2, 8]]],
)
print(score.fbeta)  # 1.0
```

Span Coverage F1 does not measure how much of a reference span is recovered:
one contained character per reference span can obtain a perfect score. It
should therefore be reported with an overlap-sensitive measure such as
character IoU. The [`validation/`](validation/) directory contains the
reproducible Mu-SHROOM inter-annotator analysis, metric comparisons, and the
corresponding under-coverage diagnostic.
