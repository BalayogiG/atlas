# ATLAS Metric Documentation

## 1. Metric Overview

### Metric Name

**Completeness**

### Metric ID

`DQ001`

### Category

`Common` (`CAT001`)

### Description

Measures the amount and distribution of missing data in a pandas `DataFrame`. It reports a cell-level missing-value rate, a row-level complete-record rate, the most common patterns of missingness across columns, and (optionally) coverage of a caller-supplied list of required fields. It does not load, mutate, or write the dataset — `compute()` only reads the `DataFrame` it is given.

### Evaluation Scope

Dataset. The metric inspects the entire `pandas.DataFrame` passed to `compute()` — every cell, every row, and every column — and produces a single dataset-level score plus a `details` dict. It does not emit one issue per record; instead it summarizes missingness across the whole dataset (see [Details](#details)).

### Required Fields

| Parameter | Type | Required | Description |
| :---- | :---- | :---- | :---- |
| `df` | `pandas.DataFrame` | Yes | The dataset to evaluate, passed positionally to `compute(df, **kwargs)`. |
| `required_fields` (kwarg) | `list[str]` | No | Column names to check for row-wise completeness, used only to compute the diagnostic `required_field_coverage` / `missing_required_fields` details. Defaults to `[]` if omitted. Does **not** affect `score`. |

---

## 2. Metric Definition & Scoring

### Definition

Completeness is driven entirely by cell-level missingness. A cell counts as missing if `pandas` considers it null (`NaN`, `None`, `NaT`, etc. — whatever `DataFrame.isna()` reports); no separate handling is applied for empty or whitespace-only strings.

Everything else the metric reports — complete-record rate, missingness patterns, per-column rates, required-field coverage, imputation dependence — is a diagnostic detail computed alongside the score, not a scoring input (with the one exception of `missing_value_rate` itself, which *is* the score).

### Evaluation Method

```
missing_value_rate      = df.isna().mean().mean()                 # 0.0 if df.size == 0
complete_record_rate    = df.notna().all(axis=1).mean()           # 1.0 if df has no rows
missing_rate_by_column  = df.isna().mean().sort_values(descending)
missingness_pattern     = df.isna().astype(int).astype(str)
                             .agg("".join, axis=1)
                             .value_counts().head(5)               # {pattern_string: row_count}

missing_required_fields = [f for f in required_fields if f not in df.columns]
available_required      = [f for f in required_fields if f in df.columns]

if available_required:
    required_field_coverage = df[available_required].notna().all(axis=1).mean()
else:
    required_field_coverage = complete_record_rate

if missing_required_fields:
    required_field_coverage = 0.0   # overrides the value computed above

imputed_columns        = count of columns whose name ends with "_imputed" or "_filled" (case-insensitive)
imputation_dependence  = imputed_columns / len(df.columns) if df has columns else 0.0

score = round(1 - missing_value_rate, 4)
```

Notes on the pseudocode above (all taken directly from [`atlas/metrics/completeness.py`](../../atlas/metrics/completeness.py)):

* `missingness_pattern` keys are per-row binary strings in column order (`"1"` = missing at that column position), and the values are **row counts**, not fractions. Only the five most frequent patterns are kept.
* `required_field_coverage` is an all-or-nothing check per row: a row only counts as covered if **every** field in `available_required` is non-null on that row.
* If **any** requested field is absent from `df.columns` at all, `required_field_coverage` is forced to `0.0` — even if the fields that *are* present are fully populated. Partial column availability does not produce a partial coverage score.
* When `required_fields` is omitted or empty, `required_field_coverage` falls back to `complete_record_rate` rather than being undefined.

### Score

**Score Range:** `0.0 – 1.0`

**Score Calculation:**

```text
score = 1 - missing_value_rate
```

`score` is rounded to 4 decimal places. An empty `DataFrame` (`df.size == 0`, i.e. no rows or no columns) has `missing_value_rate = 0.0` and therefore scores `1.0`.

### Score Interpretation

Unlike some evaluation frameworks, ATLAS's `MetricResult` has no boolean `passed` field — `compute()` returns only a continuous `score` (plus `details`, `reasons`, `recommendations`). Callers interpret the score themselves, e.g.:

| Score Range | Interpretation |
| :---- | :---- |
| `1.0` | Complete — no missing cells |
| `0.90 – 0.999` | Nearly complete — a small fraction of cells missing |
| `0.70 – 0.899` | Notable gaps — review before downstream use |
| `0.01 – 0.699` | Incomplete — significant missing data, but not total |
| `0.0` | Every cell in the dataset is missing |

---

## 3. Configuration & Output

### Configuration Options

Configuration is passed as keyword arguments to `compute()` (or forwarded through `Atlas().evaluate(..., **kwargs)`); there is no separate config object.

| Option | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| `required_fields` | `list[str]` | `[]` | Column names checked for row-wise completeness. Purely diagnostic — feeds `required_field_coverage` and `missing_required_fields` in `details`, but never changes `score`. |

Example:

```python
from atlas import Atlas

report = Atlas().evaluate(
    "employees.csv",
    metrics=["DQ001"],
    required_fields=["id", "name", "email"],
)
```

Example — no required fields (still computes missingness and complete-record diagnostics):

```python
report = Atlas().evaluate("employees.csv", metrics=["DQ001"])
```

### Output

`compute()` returns a `MetricResult` dataclass ([`atlas/base.py`](../../atlas/base.py)):

```python
@dataclass
class MetricResult:
    metric: str
    category: str
    score: float
    details: dict[str, Any]
    reasons: list[str]
    recommendations: list[str]
```

When an `Atlas().evaluate()` report is saved to disk (`output=` / `--output`), it is serialized via `dataclasses.asdict`, e.g.:

```json
{
  "dataset_name": "employees.csv",
  "rows": 6,
  "columns": 3,
  "metrics": {
    "DQ001": {
      "metric": "Completeness",
      "category": "Common",
      "score": 0.8333,
      "details": {
        "missing_value_rate": 0.1667,
        "complete_record_rate": 0.6667,
        "missingness_pattern": {"000": 4, "010": 2},
        "required_field_coverage": 0.6667,
        "missing_required_fields": [],
        "imputation_dependence": 0.0,
        "missing_rate_by_column": {"email": 0.3333, "id": 0.0, "name": 0.1667}
      },
      "reasons": ["16.7% of cells are missing."],
      "recommendations": ["Review columns with high missing-value rates and define a treatment policy."]
    }
  },
  "overall_score": 0.8333,
  "execution_time": 0.0041
}
```

### Reasons & Recommendations

ATLAS has no per-record issue list for this metric (there is no `ValidationIssue` concept in this codebase). Instead, `compute()` attaches short, dataset-level natural-language strings:

* `reasons`: `[f"{missing_rate:.1%} of cells are missing."]` if `missing_value_rate > 0`, else `[]`.
* `recommendations`: `["Review columns with high missing-value rates and define a treatment policy."]` if there were any reasons, else `[]`.

If `reasoning=True` is passed to `Atlas().evaluate()`, an optional local model (via `ollama`, see [`atlas/reasoning.py`](../../atlas/reasoning.py)) can append a richer generated reason and additional recommendations on top of these built-in ones; if the model call fails, ATLAS falls back to the built-in reasons and raises a `RuntimeWarning`.

### Details

| Key | Type | Description |
| :---- | :---- | :---- |
| `missing_value_rate` | `float` | Fraction of all `DataFrame` cells that are null. This is also the value the score is derived from. |
| `complete_record_rate` | `float` | Fraction of rows with no null values in any column. |
| `missingness_pattern` | `dict[str, int]` | Up to five most frequent binary null patterns, in column order (`"1"` = missing), mapped to their row counts. |
| `missing_rate_by_column` | `dict[str, float]` | Null rate for each column, sorted descending. |
| `required_field_coverage` | `float` | Fraction of rows complete across `required_fields`. `0.0` if any requested field is entirely absent from the dataset; falls back to `complete_record_rate` if `required_fields` is empty. Diagnostic only — does not affect `score`. |
| `missing_required_fields` | `list[str]` | Entries from `required_fields` that do not exist as columns in `df`. |
| `imputation_dependence` | `float` | Fraction of columns whose name ends with `_imputed` or `_filled` (case-insensitive) — a proxy for how much of the dataset was back-filled upstream. |

---

## 4. Usage & Implementation

### CLI

```bash
atlas evaluate --dataset employees.csv --metric DQ001
```

The CLI's `evaluate` command ([`atlas/cli.py`](../../atlas/cli.py)) exposes `--dataset`, repeatable `--metric`, `--reasoning`, `--model`, `--output`, and `--format`. It does **not** expose `required_fields` as a flag — that option is only reachable through the Python API. Run with no `--metric` to execute every registered metric (Completeness and Data Quality Noise) against the dataset.

Other relevant commands:

```bash
atlas list-metrics                 # table of registered metric_id / category_id / name / description
atlas get-metric --name DQ001      # metadata for a single metric
```

### Python API

```python
from atlas import Atlas

report = Atlas().evaluate(
    "employees.csv",
    metrics=["DQ001"],
    required_fields=["id", "name", "email"],
)

result = report.metrics["DQ001"]
print(result.score)
print(result.details["missing_rate_by_column"])
```

Calling the metric class directly, without going through `Atlas().evaluate()`:

```python
import pandas as pd
from atlas.metrics import CompletenessMetric

df = pd.read_csv("employees.csv")
result = CompletenessMetric().compute(df, required_fields=["id", "name", "email"])
```

Saving a rendered report (JSON, Markdown, or HTML) instead of just returning the `EvaluationReport` object:

```python
report = Atlas().evaluate(
    "employees.csv",
    metrics=["DQ001"],
    output="report.json",
    output_format="json",
)
```

### Implementation

```python
"""Completeness quality metric."""

from typing import Any

import pandas as pd

from ..base import Metric, MetricResult
from ..decorators import metric
from ..utils import calculate_missing_rate


@metric(
    name="Completeness",
    category="Common",
    metric_id="DQ001",
    category_id="CAT001",
    description="Measures missing values, complete records, and required-field coverage.",
)
class CompletenessMetric(Metric):
    """Measure missingness and record completeness."""

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> MetricResult:
        """Compute completeness indicators and an aggregate 0.0--1.0 score."""
        required = kwargs.get("required_fields", [])
        missing_rate = calculate_missing_rate(df)
        complete_rate = float(df.notna().all(axis=1).mean()) if len(df) else 1.0
        by_column = df.isna().mean().sort_values(ascending=False)
        patterns = df.isna().astype(int).astype(str).agg("".join, axis=1).value_counts().head(5).to_dict()
        missing_required = [field for field in required if field not in df.columns]
        available_required = [field for field in required if field in df.columns]
        required_coverage = (float(df[available_required].notna().all(axis=1).mean())
                             if available_required else complete_rate)
        if missing_required:
            required_coverage = 0.0
        imputed = sum(col.lower().endswith(("_imputed", "_filled")) for col in df.columns)
        imputation_dependence = imputed / len(df.columns) if len(df.columns) else 0.0
        details = {"missing_value_rate": missing_rate, "complete_record_rate": complete_rate,
                   "missingness_pattern": patterns, "required_field_coverage": required_coverage,
                   "imputation_dependence": imputation_dependence, "missing_required_fields": missing_required,
                   "missing_rate_by_column": by_column.to_dict()}
        reasons = [f"{missing_rate:.1%} of cells are missing."] if missing_rate else []
        recommendations = (["Review columns with high missing-value rates and define a treatment policy."]
                           if missing_rate else [])
        return MetricResult(self.name, self.category, round(1 - missing_rate, 4), details,
                            reasons, recommendations)
```

Source: [atlas/metrics/completeness.py](../../atlas/metrics/completeness.py)

### Logging

`CompletenessMetric.compute()` logs via `logging.getLogger("atlas.metrics.completeness")`. It is silent by default (ATLAS attaches a `NullHandler` to the `"atlas"` logger, per standard library-logging practice) — configure logging in your application to see it:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

* `DEBUG` — one line on entry (`"Computing completeness for %d rows, %d columns (required_fields=%s)"`) and one on exit with the final score.
* `WARNING` — emitted when `required_fields` includes a column not present in `df.columns` (the same condition that forces `required_field_coverage` to `0.0`).

For structured, machine-readable visibility into a run, `MetricResult.details` remains the primary interface — logging is a debugging aid, not a substitute.

### Dependencies

`pandas` (for all `DataFrame` operations) plus the ATLAS core (`atlas.base`, `atlas.decorators`, `atlas.utils.calculate_missing_rate`). No external services, models, or network calls — reasoning via `ollama` is opt-in at the `Atlas().evaluate()` level and unrelated to the metric's own computation.

---

## 5. Testing & Validation

There is currently no automated test suite in this repository for `CompletenessMetric` (the previous `tests/` directory, written against an older records-based implementation, has been removed along with that implementation). The strategy below describes the recommended coverage for a future `tests/test_completeness.py`.

### Test Dataset

* **Dataset source:** Synthetic, hand-constructed `pandas.DataFrame` instances built inline in the test file.
* **Dataset size:** Small, purpose-built frames (2–10 rows) are sufficient since the metric is deterministic and vectorized.
* **Input fields:** A handful of columns mixing complete, missing (`NaN`/`None`), and imputed-looking names (e.g. `age_imputed`).
* **Data distribution:** Should include fully complete rows, rows with `NaN` in one or more columns, a dataset missing an entire required column, and an empty `DataFrame`.
* **Ground truth:** Expected `missing_value_rate`, `complete_record_rate`, `required_field_coverage`, `missingness_pattern`, and `score` computed by hand from the definitions above.

### Evaluation Strategy

```
Construct a DataFrame (complete, partially missing, fully missing, empty)
     ↓
Compute expected missing_value_rate / complete_record_rate / score by hand
     ↓
Run CompletenessMetric().compute(df, **kwargs)
     ↓
Compare actual MetricResult.score and .details to expected values
     ↓
Assert score, details, reasons, and recommendations match
```

### Test Cases

| Test Case | Description | Expected Result |
| :---- | :---- | :---- |
| Fully complete dataset | No nulls anywhere | `missing_value_rate = 0.0`, `score = 1.0`, `reasons = []` |
| Some cells missing | A few `NaN` values scattered across rows/columns | `0.0 < score < 1.0`; `missing_rate_by_column` reflects the affected columns |
| Fully missing dataset | Every cell is `NaN` | `missing_value_rate = 1.0`, `score = 0.0` |
| Empty DataFrame | `df` has zero rows or zero columns (`df.size == 0`) | `missing_value_rate = 0.0`, `score = 1.0` (trivial pass) |
| Required fields all present, all populated | `required_fields` set to columns that fully exist and are non-null | `required_field_coverage = 1.0` |
| Required field absent from columns | `required_fields` includes a name not in `df.columns` | `missing_required_fields` non-empty; `required_field_coverage = 0.0` regardless of other columns' completeness |
| `required_fields` omitted/empty | No `required_fields` kwarg passed | `required_field_coverage` equals `complete_record_rate` |
| Imputed columns present | Some column names end in `_imputed` or `_filled` | `imputation_dependence > 0.0`, proportional to the count of such columns |
| Duplicate missingness patterns | Multiple rows share the same null/non-null shape | `missingness_pattern` groups them under one key with the correct row count |
| More than five distinct patterns | Dataset has >5 unique row-level null patterns | Only the top five most frequent patterns are kept in `missingness_pattern` |

### Validation Metrics

Completeness is a deterministic, vectorized data-quality metric with no probabilistic or model-based component (the optional `reasoning=True` path only *appends* free-text explanation and never changes the score). Exact expected-result matching — comparing computed `score` and `details` to hand-computed values — is sufficient; accuracy/precision/recall-style metrics do not apply.

### Expected Results

```json
{
  "expected_score": 0.8333,
  "expected_missing_value_rate": 0.1667,
  "expected_complete_record_rate": 0.6667
}
```

### Edge Cases

* Empty `DataFrame` (`df.size == 0`) — `score` defaults to `1.0`.
* Single-row or single-column `DataFrame`.
* `required_fields = []` or omitted — `required_field_coverage` falls back to `complete_record_rate`.
* `required_fields` referencing a column that does not exist at all — forces `required_field_coverage` to `0.0`, even if every other required field is fully populated.
* Column names that only *partially* match the imputed/filled suffix convention (e.g. `imputed_flag`, which does not *end* with `_imputed`) are **not** counted toward `imputation_dependence`.
* More than five distinct missingness patterns in a dataset — only the top five by frequency are retained.
* Non-null but "empty-looking" values such as `""` or whitespace-only strings are **not** treated as missing by this metric — only `pandas`-null values (`NaN`/`None`/`NaT`) count, unlike record-based completeness checks that also flag blank strings.
* Very large datasets — cost is `O(rows × columns)` for the missingness pass, dominated by `DataFrame.isna()` and the pattern `groupby`/`value_counts`.
