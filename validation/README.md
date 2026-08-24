# Mu-SHROOM inter-annotator validation

This directory reproduces the validation used to characterize Span Coverage
F1 on the English Mu-SHROOM test split. The experiment uses the dataset's
individual annotator labels; it does not require new human annotation.

## Design

The script runs three diagnostics:

1. **Pairwise agreement:** compare every pair of annotators for an example and
   classify their geometry as exact, nested boundary, containment with changed
   component count, partial overlap, disjoint, or empty.
2. **Leave one annotator out (LOO):** treat one annotator as the prediction and
   construct a reference from the remaining annotators using either a strict
   character-level majority or their union.
3. **Under-coverage stress diagnostic:** retain one central character from each
   span in the all-annotator union. This gold-informed construction is a metric
   diagnostic, not a model prediction.

The comparison includes Span Coverage F1, exact span F1, character F1,
character IoU, DetEval-1D, and CLEval-1D. DetEval-1D replaces rectangle area
with inclusive interval length while retaining the original overlap thresholds
and split/merge credit. CLEval-1D treats character offsets as pseudo-character
centres and retains its character-coverage and granularity-penalty semantics.

Mu-SHROOM stores half-open spans `[start, end)`. The script validates and
converts them to the inclusive `[start, end]` convention used by
`span_coverage.py`, dropping zero-length selections and merging overlapping or
adjacent spans.

Macro summaries first average comparisons within each example, preventing
examples with more annotators from dominating. Confidence intervals use an
example-clustered bootstrap.

## Data and run

From the repository root:

```bash
git clone --depth 1 \
  https://huggingface.co/datasets/Helsinki-NLP/mu-shroom \
  data/mu-shroom
git -C data/mu-shroom lfs pull

python3 -m venv .venv
.venv/bin/pip install -r validation/requirements.txt
.venv/bin/python validation/mushroom_interannotator.py \
  --parquet data/mu-shroom/en/test-00000-of-00001.parquet \
  --output-dir validation/output/en_test \
  --bootstrap 2000
```

If Git LFS is unavailable, download the required file directly:

```bash
mkdir -p data/mu-shroom/en
curl --fail --location \
  'https://huggingface.co/datasets/Helsinki-NLP/mu-shroom/resolve/main/en/test-00000-of-00001.parquet?download=true' \
  --output data/mu-shroom/en/test-00000-of-00001.parquet
```

## Main results

| Metric | LOO Majority | LOO Union | Nested Boundary | Partial Overlap | 1-Char |
|---|---:|---:|---:|---:|---:|
| Exact span F1 | 0.288 | 0.071 | 0.032 | 0.028 | 0.014 |
| Character F1 | 0.653 | 0.568 | 0.605 | 0.664 | 0.057 |
| Character IoU | 0.561 | 0.462 | 0.503 | 0.551 | 0.031 |
| DetEval-1D | 0.574 | 0.239 | 0.281 | 0.406 | 0.014 |
| CLEval-1D | 0.626 | 0.559 | 0.605 | 0.650 | 0.057 |
| **Span Coverage F1** | 0.410 | 0.750 | 1.000 | 0.347 | 1.000 |

The nested-boundary subset contains 376 held-out predictions strictly inside
the LOO union reference with the same number of connected spans. Its perfect
Span Coverage F1 is therefore expected from the subset definition. In the
1-Char diagnostic, predictions retain only 3.1% of reference characters on
average; the perfect score exposes the metric's under-coverage limitation.

These results support a narrow claim: Span Coverage F1 is selectively tolerant
to nested boundary disagreement while rejecting disjoint localizations. They do
not establish that every nested annotation pair is semantically equivalent or
that the union is a human-verified gold standard.

