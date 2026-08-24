<p align="center">
  <img src="images/mycelium-banner-v2.png" alt="A friendly pastel collection of illustrated mushrooms" width="100%">
</p>

# Span Coverage F1

Span Coverage F1 is a directional, instance-level metric for character-span
localization. It is intended for settings where a prediction may identify a
more specific phrase inside a broader reference annotation.

For predicted spans `P` and reference spans `G`:

- precision is the fraction of predictions fully contained in at least one
  reference span;
- recall is the fraction of reference spans that contain at least one
  prediction;
- Span Coverage F1 is the harmonic mean of these two values.

Containment is asymmetric: a narrower prediction inside a broader reference is
accepted, while a prediction extending outside the reference is not. Matching
is not one-to-one; the metric counts contained predictions and hit references
directly.

## Usage

The implementation has no third-party dependencies and uses **inclusive**
character spans `[start, end]`.

```python
from span_coverage import span_coverage_micro

golds = [
    [[0, 10], [20, 30]],
]
preds = [
    [[2, 8], [20, 25], [40, 42]],
]

score = span_coverage_micro(golds, preds)
print(score.precision)  # 0.6667: two of three predictions are contained
print(score.recall)     # 1.0: both reference spans are hit
print(score.fbeta)      # 0.8
```

Both corpus-level micro aggregation and per-example macro aggregation are
available:

```python
from span_coverage import span_coverage_macro, span_coverage_micro
```

Optional arguments:

- `delta` expands reference boundaries by a fixed number of characters before
  checking containment;
- `min_pred_len` ignores predictions shorter than the specified inclusive
  length;
- `beta` selects the F-beta weighting;
- `empty_is_perfect` controls the score assigned when both sides are empty.

## Intended use and limitation

Span Coverage F1 measures whether predicted instances fall inside annotated
regions and whether every annotated instance is hit. It does **not** measure
how much of a reference span is recovered. In particular, one contained
character per reference span can obtain a perfect score. Report an
overlap-sensitive companion metric, such as character IoU, when localization
extent matters.

## Validation

[`validation/`](validation/) contains the reproducible Mu-SHROOM
inter-annotator experiment used to study this behavior. It compares Span
Coverage F1 with exact span F1, character F1, character IoU, DetEval-1D, and
CLEval-1D under pairwise and leave-one-annotator-out reference constructions.
The validation also includes a one-character stress diagnostic that makes the
under-coverage limitation explicit.
