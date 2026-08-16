# ATLAS Metric Documentation

## 1. Metric Overview

### Metric Name

**Data Quality & Noise**

### Metric ID

`DQ002`

### Category

`Common` (`CAT001`)

### Description

Measures duplicate rows, statistical numeric outliers, textual anomalies (unusual length or character mix), schema (semantic type) conformance, and optional cross-field consistency rules in a pandas `DataFrame`. It combines five independent error signals into a single aggregate score, and reports each component plus schema-validation and outlier detail alongside it. It does not load, mutate, or write the dataset.

### Evaluation Scope

Dataset. The metric inspects the entire `pandas.DataFrame` passed to `compute()`: every row (for duplicates), every numeric column (for numeric outliers), every free-text column (for textual anomalies), every column named in the resolved schema (for type conformance), and every rule supplied via `consistency_rules` (evaluated across all rows). It produces one dataset-level score plus a `details` dict — there is no per-record issue list.

### Required Fields

| Parameter | Type | Required | Description |
| :---- | :---- | :---- | :---- |
| `df` | `pandas.DataFrame` | Yes | The dataset to evaluate, passed positionally to `compute(df, **kwargs)`. |
| `schema` (kwarg) | `AtlasSchema \| dict[str, str] \| None` | No | Declares the expected semantic type per column. If omitted (`None`), the metric infers a schema from the dataset itself (see [Schema Validation](#schema-validation)). |
| `consistency_rules` (kwarg) | `dict[str, str \| Callable[[DataFrame], Series]]` | No | Named cross-field rules to evaluate. Defaults to `{}`, in which case `cross_field_consistency` is trivially `1.0`. |

---

## 2. Metric Definition & Scoring

### Definition

The metric combines five independent 0–1 error/quality rates into one aggregate `error_rate`, then converts that to a 0.0–1.0 score:

* **`duplicate_rate`** — fraction of rows that are duplicates of an earlier row (every occurrence after the first counts).
* **`outlier_rate`** — the higher of a z-score outlier rate and (for ≥10 complete numeric rows) an Isolation Forest outlier rate, computed over numeric columns only.
* **`text_anomaly_rate`** — the same z-score/Isolation Forest logic applied to length- and character-mix features derived from free-text columns, instead of raw numeric values.
* **`schema_conformance`** — fraction of non-null, schema-declared values that match their expected semantic type.
* **`cross_field_consistency`** — mean pass rate across all supplied `consistency_rules`; a rule that raises an exception counts as fully failed (`0.0`) for that rule.

### Evaluation Method

```
duplicate_rate = df.duplicated().mean()                     # 0.0 if df has no rows
numeric_columns = df.select_dtypes(include="number").columns
text_columns    = df.select_dtypes(include=["object", "string"]).columns

# --- shared outlier scorer, used for both outlier_rate and text_anomaly_rate ---
def statistical_outlier_rate(values):                       # values: a purely numeric feature matrix
    values = values.replace([inf, -inf], NaN).dropna()
    if values.empty:
        return 0.0
    z_scores = abs(zscore(values, nan_policy="omit"))
    statistical_rate = nanmean(z_scores > 3)
    if len(values) < 10:
        return statistical_rate
    model = IsolationForest(contamination="auto", random_state=0).fit(values)
    isolation_rate = mean(model.predict(values) == -1)
    return max(statistical_rate, isolation_rate)

# --- outlier_rate ---
outlier_rate = statistical_outlier_rate(df[numeric_columns])

# --- text_anomaly_rate ---
text_features = {}
for column in text_columns:
    values = df[column].dropna().astype(str)
    values = values[values.str.len() > 0]
    if values.empty:
        continue
    text_features[f"{column}__length"]      = values.str.len()
    text_features[f"{column}__digit_ratio"] = values.map(lambda v: count_digits(v) / len(v))
    text_features[f"{column}__punct_ratio"] = values.map(lambda v: count_punctuation(v) / len(v))
text_anomaly_rate = statistical_outlier_rate(DataFrame(text_features))

# --- schema_conformance ---
resolved_schema = schema if given else AtlasSchema.from_dict(infer_schema(df))
for column, expected_type in resolved_schema.fields.items():
    if column not in df.columns:
        # every row counts as invalid for this column
        violations[column] = {expected, invalid_count: len(df), examples: ["column missing"]}
        continue
    non_null = df[column].dropna()
    matches  = non_null.map(lambda v: matches_type(v, expected_type))
    valid   += matches.sum()
    total   += len(non_null)
    if any(~matches):
        violations[column] = {expected, invalid_count, examples: first 3 invalid values as str}
schema_conformance = valid / total if total else 1.0

# --- cross_field_consistency ---
if not consistency_rules:
    cross_field_consistency = 1.0
else:
    for name, rule in consistency_rules.items():
        try:
            outcome = rule(df) if callable(rule) else df.eval(rule)
            outcomes.append(mean(bool Series))
        except Exception:
            outcomes.append(0.0)
    cross_field_consistency = mean(outcomes)

error_rate = mean([duplicate_rate, outlier_rate, text_anomaly_rate,
                   1 - schema_conformance, 1 - cross_field_consistency])
score = round(1 - error_rate, 4)
```

Notes on the pseudocode above (all taken directly from [`atlas/metrics/data_quality_noise.py`](../../atlas/metrics/data_quality_noise.py)):

* Missing values are excluded from the outlier, text-anomaly, and schema-conformance checks (`.dropna()` before scoring) — this metric does not double-penalize missingness that `DQ001` (Completeness) already measures.
* `outlier_rate` and `text_anomaly_rate` share the exact same `_statistical_outlier_rate` implementation; the only difference is the feature matrix each is given — raw numeric columns for one, derived length/character-mix features for the other.
* The Isolation Forest pass only runs when there are at least 10 complete rows in the feature matrix; below that threshold, the rate is the z-score rate alone. When it does run, the *higher* of the two rates is used, not an average.
* Because `text_features` combines every text column's derived features into one matrix and then drops any row with a `NaN` in *any* feature, a row is excluded from text-anomaly scoring if even one of its text columns was null — mirroring how the numeric matrix already drops rows with any missing numeric value.
* A schema column entirely absent from `df` counts **every row** as invalid for that column (`invalid_count = len(df)`), which can pull `schema_conformance` down sharply for one missing column on a wide dataset.
* A `consistency_rules` entry that raises (e.g. references a nonexistent column, or a callable that errors) is scored as `0.0` for that rule rather than raising out of `compute()`.

### Score

**Score Range:** `0.0 – 1.0`

**Score Calculation:**

```text
error_rate = mean(duplicate_rate, outlier_rate, text_anomaly_rate,
                   1 - schema_conformance, 1 - cross_field_consistency)
score      = 1 - error_rate
```

`score` is rounded to 4 decimal places. A dataset with no duplicates, no numeric or textual outliers, full schema conformance, and either no consistency rules or all-passing ones scores `1.0`. A dataset that maximally fails every component scores `0.0`.

### Score Interpretation

As with Completeness, `MetricResult` carries no boolean `passed` field — only a continuous `score`. Interpret it as:

| Score Range | Interpretation |
| :---- | :---- |
| `1.0` | No duplicates, numeric/textual outliers, or schema/consistency violations detected |
| `0.90 – 0.999` | Minor noise — a small number of duplicates, outliers, or type mismatches |
| `0.70 – 0.899` | Notable noise — review `schema_validation.violations` and the component rates |
| `0.01 – 0.699` | Substantial noise across one or more components |
| `0.0` | Every component (duplicates, outliers, text anomalies, schema, consistency) is maximally failing |

Because `error_rate` is an unweighted mean of five components, a dataset can score moderately even if one component is very bad, as long as the others are clean — inspect the individual `details` keys rather than relying on `score` alone to diagnose *which* component is the problem.

---

## 3. Configuration & Output

### Configuration Options

Configuration is passed as keyword arguments to `compute()` (or forwarded through `Atlas().evaluate(..., **kwargs)`); there is no separate config object. Numeric-outlier and text-anomaly detection are always on for any numeric/text columns present — there is no option to disable or tune them independently.

| Option | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| `schema` | `AtlasSchema \| dict[str, str] \| None` | `None` (infer) | Expected semantic type per column. Accepts a raw `dict[str, str]`, which is normalized through `AtlasSchema.from_dict`, or an `AtlasSchema` instance directly. |
| `consistency_rules` | `dict[str, str \| Callable]` | `{}` | Named rules, each either a `DataFrame.eval`-compatible expression string or a callable taking the `DataFrame` and returning a boolean `Series`. |

`AtlasSchema.from_dict` (in [`atlas/base.py`](../../atlas/base.py)) accepts the types `int`, `float`, `str`, `category`, `email`, `bool`, plus the aliases `integer`→`int`, `string`→`str`, `boolean`→`bool` (case-insensitive). It raises `ValueError` if any column/type value isn't a string, or if a type isn't in that supported set.

Example — explicit schema and consistency rules:

```python
from atlas import Atlas, AtlasSchema

schema = AtlasSchema.from_dict({
    "id": "int",
    "age": "int",
    "salary": "float",
    "department": "category",
    "email": "email",
})

report = Atlas().evaluate(
    "employees.csv",
    metrics=["DQ002"],
    schema=schema,
    consistency_rules={
        "valid_salary": "salary >= 0",
        "adult_employee": lambda df: df["age"] >= 18,
    },
)
```

Example — inferred schema, no consistency rules:

```python
report = Atlas().evaluate("employees.csv", metrics=["DQ002"])
```

### Output

`compute()` returns the same `MetricResult` dataclass used by every ATLAS metric:

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

Serialized example (via `Atlas().evaluate(..., output="report.json")`):

```json
{
  "metric": "Data Quality & Noise",
  "category": "Common",
  "score": 0.8875,
  "details": {
    "error_rate": 0.1125,
    "duplicate_rate": 0.05,
    "outlier_rate": 0.1,
    "text_anomaly_rate": 0.2,
    "schema_conformance": 0.98,
    "schema_validation": {
      "mode": "user_defined",
      "schema": {"id": "int", "age": "int", "salary": "float"},
      "violations": {
        "age": {"expected": "int", "invalid_count": 1, "examples": ["thirty"]}
      }
    },
    "cross_field_consistency": 1.0,
    "numeric_columns": ["id", "age", "salary"],
    "text_columns": ["name", "notes"]
  },
  "reasons": [
    "5.0% of rows are duplicate.",
    "10.0% of numeric values are statistical outliers.",
    "20.0% of text values have an anomalous length or character mix."
  ],
  "recommendations": ["Deduplicate records and review anomalous numeric and text values."]
}
```

### Reasons & Recommendations

* `reasons` includes `f"{duplicate_rate:.1%} of rows are duplicate."` whenever `duplicate_rate > 0`, `f"{outlier_rate:.1%} of numeric values are statistical outliers."` whenever `outlier_rate > 0`, and `f"{text_anomaly_rate:.1%} of text values have an anomalous length or character mix."` whenever `text_anomaly_rate > 0`. Neither `schema_conformance` nor `cross_field_consistency` currently contributes a built-in reason string — inspect `schema_validation.violations` directly for schema issues.
* `recommendations` is `["Deduplicate records and review anomalous numeric and text values."]` if there were any `reasons`, else `[]`.
* As with Completeness, passing `reasoning=True` to `Atlas().evaluate()` can append a model-generated reason/recommendations via `atlas/reasoning.py`, falling back silently (with a `RuntimeWarning`) if the local model call fails.

### Details

| Key | Type | Description |
| :---- | :---- | :---- |
| `error_rate` | `float` | Unweighted mean of `duplicate_rate`, `outlier_rate`, `text_anomaly_rate`, `1 - schema_conformance`, and `1 - cross_field_consistency`. Score is `1 - error_rate`. |
| `duplicate_rate` | `float` | Fraction of rows duplicated after their first occurrence. |
| `outlier_rate` | `float` | Higher of the z-score outlier rate and (when ≥10 complete numeric rows exist) the Isolation Forest outlier rate, over numeric columns. |
| `text_anomaly_rate` | `float` | Same z-score/Isolation Forest logic as `outlier_rate`, applied to per-column length/digit-ratio/punctuation-ratio features derived from free-text columns instead of raw numeric values. |
| `schema_conformance` | `float` | Fraction of non-null, schema-declared values matching their expected semantic type. |
| `schema_validation` | `dict` | `{"mode": "inferred" \| "user_defined", "schema": {column: type}, "violations": {column: {expected, invalid_count, examples}}}`. |
| `cross_field_consistency` | `float` | Mean pass rate across `consistency_rules`; `1.0` if no rules were supplied. |
| `numeric_columns` | `list[str]` | Numeric `DataFrame` columns used for `outlier_rate`. |
| `text_columns` | `list[str]` | Object/string-dtype `DataFrame` columns used for `text_anomaly_rate`. |

#### Text Anomaly Detection

For each text column, non-null, non-empty values are converted to three per-value numeric features:

* **length** — character count of the value.
* **digit_ratio** — fraction of characters that are digits.
* **punct_ratio** — fraction of characters that are neither alphanumeric nor whitespace (punctuation/symbols).

All text columns' features are combined into one matrix (three columns per text column) and scored with the same z-score / Isolation Forest logic used for `outlier_rate`. This catches values that are unusually long/short, digit-heavy, or symbol-heavy relative to the rest of the column — e.g. a stray sentence or garbage string dropped into an otherwise short, clean `name` or `category` column — but it does **not** detect semantic outliers (a value that is the right length and character mix but means something different from the rest of the column). Columns that are empty, entirely null, or contain only empty strings contribute no features and do not affect `text_anomaly_rate`.

#### Schema Validation

Without an explicit `schema` kwarg, ATLAS infers one column-by-column (`_infer_schema` in [`atlas/metrics/data_quality_noise.py`](../../atlas/metrics/data_quality_noise.py)):

* Boolean dtype → `bool`; integer dtype → `int`; float dtype → `float`.
* All-null column → `str`.
* Otherwise, values are coerced to string and: if ≥90% parse as numeric → `int` (when all such values are integer-valued) or `float`; else if ≥90% match a simple email regex → `email`; else if the number of distinct values is small relative to row count (`nunique() <= min(20, max(2, len(values) * 0.1))`) → `category`; else → `str`.

The result records `"mode": "inferred"` in `schema_validation` in this case. Passing an explicit `schema` sets `"mode": "user_defined"` instead.

Type-matching rules (`_matches_type`) once a schema is resolved:

* `str` / `category` — always considered a match (no further check).
* `email` — must fully match `^[^@\s]+@[^@\s]+\.[^@\s]+$`.
* `bool` — string form (lowercased) must be one of `true`, `false`, `0`, `1`.
* `int` — must coerce to numeric via `pd.to_numeric` **and** be integer-valued (`float(value).is_integer()`).
* `float` — must coerce to numeric via `pd.to_numeric` (any numeric value is accepted, integer-valued or not).

Invalid values are reported per column with an `invalid_count` and up to three string `examples`.

#### Cross-Field Consistency

Each rule in `consistency_rules` is either a `DataFrame.eval`-compatible expression string (e.g. `"salary >= 0"`) or a callable taking the `DataFrame` and returning something coercible to a boolean `Series`. The rule's pass rate is the mean of that boolean series; a rule that raises any exception (bad column reference, type error, etc.) contributes `0.0` for that rule rather than propagating the exception. `cross_field_consistency` is the unweighted mean across all rules.

`consistency_rules` expression strings run through `DataFrame.eval()`, which can execute arbitrary pandas expressions — only pass trusted rule strings, never ones derived from untrusted/user-supplied input.

---

## 4. Usage & Implementation

### CLI

```bash
atlas evaluate --dataset employees.csv --metric DQ002
```

As with Completeness, the CLI's `evaluate` command does not expose `schema` or `consistency_rules` as flags — those are only reachable through the Python API. Combine metrics in one CLI run with repeated `--metric`:

```bash
atlas evaluate --dataset employees.csv --metric DQ001 --metric DQ002 --output report.json --format json
```

### Python API

```python
from atlas import Atlas, AtlasSchema

schema = AtlasSchema.from_dict({
    "id": "int",
    "age": "int",
    "salary": "float",
    "department": "category",
    "email": "email",
})

report = Atlas().evaluate(
    "employees.csv",
    metrics=["DQ002"],
    schema=schema,
    consistency_rules={
        "valid_salary": "salary >= 0",
        "adult_employee": lambda df: df["age"] >= 18,
    },
)

result = report.metrics["DQ002"]
print(result.score)
print(result.details["schema_validation"])
print(result.details["text_anomaly_rate"])
```

Calling the metric class directly:

```python
import pandas as pd
from atlas.metrics import DataQualityNoiseMetric

df = pd.read_csv("employees.csv")
result = DataQualityNoiseMetric().compute(df, schema=schema)
```

### Implementation

```python
"""Noise and validity quality metric."""

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

from ..base import AtlasSchema, Metric, MetricResult
from ..decorators import metric
from ..utils import calculate_duplicate_rate, detect_numeric_columns, detect_text_columns


@metric(
    name="Data Quality & Noise",
    category="Common",
    metric_id="DQ002",
    category_id="CAT001",
    description="Measures duplicates, statistical outliers, schema conformance, and consistency.",
)
class DataQualityNoiseMetric(Metric):
    """Measure duplicate rows, numeric outliers, and basic schema validity."""

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> MetricResult:
        """Compute noise indicators and an aggregate 0.0--1.0 score."""
        duplicate_rate = calculate_duplicate_rate(df)
        numeric = detect_numeric_columns(df)
        text = detect_text_columns(df)
        outlier_rate = self._statistical_outlier_rate(df[numeric] if numeric else pd.DataFrame())
        text_anomaly_rate = self._statistical_outlier_rate(self._text_features(df, text))
        schema_conformance, schema_details = self._schema_conformance(df, kwargs.get("schema"))
        consistency = self._consistency(df, kwargs.get("consistency_rules", {}))
        error_rate = float(np.mean([duplicate_rate, outlier_rate, text_anomaly_rate,
                                    1 - schema_conformance, 1 - consistency]))
        details = {"error_rate": error_rate, "duplicate_rate": duplicate_rate, "outlier_rate": outlier_rate,
                   "text_anomaly_rate": text_anomaly_rate, "schema_conformance": schema_conformance,
                   "schema_validation": schema_details, "cross_field_consistency": consistency,
                   "numeric_columns": numeric, "text_columns": text}
        reasons = [f"{duplicate_rate:.1%} of rows are duplicate."] if duplicate_rate else []
        if outlier_rate:
            reasons.append(f"{outlier_rate:.1%} of numeric values are statistical outliers.")
        if text_anomaly_rate:
            reasons.append(f"{text_anomaly_rate:.1%} of text values have an anomalous length or character mix.")
        recommendations = (["Deduplicate records and review anomalous numeric and text values."]
                           if reasons else [])
        return MetricResult(self.name, self.category, round(1 - error_rate, 4), details,
                            reasons, recommendations)

    @staticmethod
    def _statistical_outlier_rate(values: pd.DataFrame) -> float:
        """Rate of z-score/Isolation Forest outliers in a purely numeric feature matrix."""
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return 0.0
        z_scores = np.abs(zscore(values, nan_policy="omit"))
        statistical_rate = float(np.nanmean(z_scores > 3))
        if len(values) < 10:
            return statistical_rate
        model = IsolationForest(contamination="auto", random_state=0)
        isolation_rate = float(np.mean(model.fit_predict(values) == -1))
        return max(statistical_rate, isolation_rate)

    @staticmethod
    def _text_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Derive length/character-mix features per text column for outlier scoring."""
        if not columns:
            return pd.DataFrame()
        features = {}
        for column in columns:
            text = df[column].dropna().astype(str)
            text = text[text.str.len() > 0]
            if text.empty:
                continue
            features[f"{column}__length"] = text.str.len()
            features[f"{column}__digit_ratio"] = text.apply(lambda v: sum(c.isdigit() for c in v) / len(v))
            features[f"{column}__punct_ratio"] = text.apply(
                lambda v: sum(not c.isalnum() and not c.isspace() for c in v) / len(v)
            )
        return pd.DataFrame(features)

    @staticmethod
    def _schema_conformance(df: pd.DataFrame, schema: AtlasSchema | dict[str, str] | None) -> tuple[float, dict[str, Any]]:
        """Validate against an explicit schema or one inferred from the dataset."""
        explicit = schema is not None
        resolved = schema if isinstance(schema, AtlasSchema) else AtlasSchema.from_dict(schema or DataQualityNoiseMetric._infer_schema(df))
        valid, total = 0, 0
        violations: dict[str, dict[str, Any]] = {}
        for column, expected in resolved.fields.items():
            if column not in df:
                violations[column] = {"expected": expected, "invalid_count": len(df), "examples": ["column missing"]}
                total += len(df)
                continue
            values = df[column].dropna()
            matches = values.map(lambda value: DataQualityNoiseMetric._matches_type(value, expected))
            invalid = values[~matches]
            valid += int(matches.sum())
            total += len(values)
            if len(invalid):
                violations[column] = {"expected": expected, "invalid_count": len(invalid),
                                      "examples": [str(value) for value in invalid.head(3)]}
        conformance = valid / total if total else 1.0
        return conformance, {"mode": "user_defined" if explicit else "inferred",
                             "schema": resolved.fields, "violations": violations}

    @staticmethod
    def _infer_schema(df: pd.DataFrame) -> dict[str, str]:
        """Infer useful semantic types while tolerating a small amount of noise."""
        inferred: dict[str, str] = {}
        email_pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
        for column in df.columns:
            values = df[column].dropna()
            if pd.api.types.is_bool_dtype(df[column]):
                inferred[column] = "bool"
            elif pd.api.types.is_integer_dtype(df[column]):
                inferred[column] = "int"
            elif pd.api.types.is_float_dtype(df[column]):
                inferred[column] = "float"
            elif values.empty:
                inferred[column] = "str"
            else:
                text = values.astype(str).str.strip()
                numeric = pd.to_numeric(text, errors="coerce")
                numeric_ratio = float(numeric.notna().mean())
                if numeric_ratio >= 0.9:
                    inferred[column] = "int" if bool(np.allclose(numeric.dropna() % 1, 0)) else "float"
                elif float(text.map(lambda value: bool(email_pattern.fullmatch(value))).mean()) >= 0.9:
                    inferred[column] = "email"
                elif values.nunique() <= min(20, max(2, len(values) * 0.1)):
                    inferred[column] = "category"
                else:
                    inferred[column] = "str"
        return inferred

    @staticmethod
    def _matches_type(value: Any, expected: str) -> bool:
        """Return whether one non-null value conforms to a semantic type."""
        text = str(value).strip()
        if expected == "str" or expected == "category":
            return True
        if expected == "email":
            return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", text))
        if expected == "bool":
            return text.lower() in {"true", "false", "0", "1"}
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return False
        return bool(float(numeric).is_integer()) if expected == "int" else True

    @staticmethod
    def _consistency(df: pd.DataFrame, rules: dict[str, Any]) -> float:
        if not rules:
            return 1.0
        outcomes = []
        for name, rule in rules.items():
            try:
                result = rule(df) if callable(rule) else df.eval(rule)
                outcomes.append(float(pd.Series(result).mean()))
            except Exception:
                outcomes.append(0.0)
        return float(np.mean(outcomes)) if outcomes else 1.0
```

Source: [atlas/metrics/data_quality_noise.py](../../atlas/metrics/data_quality_noise.py)

### Logging

`DataQualityNoiseMetric.compute()` logs via `logging.getLogger("atlas.metrics.data_quality_noise")`. Silent by default (see `CompletenessMetric`'s logging note for the `NullHandler` pattern); configure `logging.basicConfig(level=...)` in your application to see it.

* `DEBUG` — entry (row/column counts), the resolved schema mode (`"inferred"` vs. `"user_defined"`), and the final score.
* `WARNING` — emitted when `schema_validation.violations` is non-empty (listing the offending columns), **and** whenever a `consistency_rules` entry raises an exception. The latter closes a real blind spot: `_consistency()` catches `Exception` and silently scores a broken rule as a full failure (`0.0`) with no other signal that anything went wrong — the warning now includes the rule's name and a full traceback (`exc_info=True`) so a typo'd `df.eval(...)` string or a rule that throws isn't mistaken for "the data genuinely failed this check."

For structured, machine-readable visibility into a run, `MetricResult.details` (especially `schema_validation.violations`) remains the primary interface — logging is a debugging aid, not a substitute.

### Dependencies

`numpy`, `pandas`, `scipy` (`scipy.stats.zscore`), and `scikit-learn` (`sklearn.ensemble.IsolationForest`), plus the standard-library `re` module for email/category inference. No external services, models, or network calls in `compute()` itself — reasoning via `ollama` is opt-in at the `Atlas().evaluate()` level and unrelated to the metric's own computation.

---

## 5. Testing & Validation

There is currently no automated test suite in this repository for `DataQualityNoiseMetric`. The strategy below describes the recommended coverage for a future `tests/test_data_quality_noise.py`.

### Test Dataset

* **Dataset source:** Synthetic, hand-constructed `pandas.DataFrame` instances built inline in the test file.
* **Dataset size:** Small frames for duplicate/schema/consistency cases (2–10 rows); at least 10 complete numeric rows in a dedicated fixture to exercise the Isolation Forest branch of outlier and text-anomaly detection.
* **Input fields:** A mix of numeric columns (for outliers), free-text columns of varying length/character mix (for text anomalies), a categorical/email/bool column (for schema inference and validation), and duplicate rows.
* **Data distribution:** Should include exact duplicate rows, one or more extreme numeric values (e.g. `1000` amid values clustered near `10`), one or more anomalously long/short or symbol-heavy text values amid otherwise uniform short strings, values that violate a declared or inferred schema type, and consistency rules that pass, fail, and raise.
* **Ground truth:** Expected `duplicate_rate`, `outlier_rate`, `text_anomaly_rate`, `schema_conformance`, `cross_field_consistency`, and `score` computed by hand from the definitions above.

### Evaluation Strategy

```
Construct a DataFrame (duplicates, outliers, text anomalies, schema violations, consistency rules)
     ↓
Compute expected component rates and score by hand
     ↓
Run DataQualityNoiseMetric().compute(df, **kwargs)
     ↓
Compare actual MetricResult.score and .details to expected values
     ↓
Assert score, details (including schema_validation.violations), reasons, and recommendations match
```

### Test Cases

| Test Case | Description | Expected Result |
| :---- | :---- | :---- |
| Clean dataset | No duplicates, no outliers, no text anomalies, schema-conformant, no consistency rules | `error_rate = 0.0`, `score = 1.0`, `reasons = []` |
| Duplicate rows | Some rows are exact repeats of an earlier row | `duplicate_rate > 0`; reason string includes the duplicate percentage |
| Numeric outlier (small dataset) | <10 complete numeric rows, one extreme value | `outlier_rate` equals the z-score rate only (Isolation Forest not invoked) |
| Numeric outlier (large dataset) | ≥10 complete numeric rows, one extreme value | `outlier_rate` is the max of the z-score and Isolation Forest rates |
| No numeric columns | Dataset has only non-numeric columns | `numeric_columns = []`, `outlier_rate = 0.0` |
| Text anomaly present | A text column has one value far longer/shorter or more symbol-heavy than the rest | `text_anomaly_rate > 0`; reason string includes the anomaly percentage |
| No text columns | Dataset has only numeric/boolean columns | `text_columns = []`, `text_anomaly_rate = 0.0` |
| Text column entirely empty/null | A text column's values are all `NaN` or `""` | That column contributes no features; does not affect `text_anomaly_rate` |
| Inferred schema, conformant | No `schema` kwarg; all values match their inferred types | `schema_validation.mode = "inferred"`, `schema_conformance = 1.0` |
| Explicit schema, violation | `schema` kwarg supplied; one column has a value of the wrong type | `schema_validation.mode = "user_defined"`; `violations` includes that column with `invalid_count` and up to 3 `examples` |
| Schema column missing from dataset | `schema` declares a column not present in `df` | That column's `invalid_count = len(df)`; `examples = ["column missing"]` |
| Consistency rule passes | Expression/callable rule true for all rows | `cross_field_consistency = 1.0` for that rule |
| Consistency rule partially fails | Rule true for some rows, false for others | `cross_field_consistency` reflects the exact pass fraction |
| Consistency rule raises | Rule references a nonexistent column or the callable throws | That rule contributes `0.0`, not an exception out of `compute()` |
| No consistency rules | `consistency_rules` omitted/empty | `cross_field_consistency = 1.0` |
| Empty DataFrame | `df` has zero rows | `duplicate_rate = 0.0`; `outlier_rate = 0.0`; `text_anomaly_rate = 0.0`; `schema_conformance = 1.0` if no schema columns have data to check |

### Validation Metrics

Data Quality & Noise combines deterministic components (`duplicate_rate`, `schema_conformance`, `cross_field_consistency`) with two statistically-derived components (`outlier_rate` and `text_anomaly_rate`, both via the shared z-score/Isolation Forest scorer). For the deterministic components, exact expected-result matching is sufficient. For `outlier_rate` and `text_anomaly_rate`, tests should assert against a known, clearly-separated outlier (e.g. a numeric value many standard deviations from the rest, or a text value dramatically longer/more symbol-heavy than its column) rather than an exact float, since Isolation Forest's `contamination="auto"` behavior is not guaranteed bit-for-bit stable across `scikit-learn` versions.

### Expected Results

```json
{
  "expected_score": 0.8875,
  "expected_duplicate_rate": 0.05,
  "expected_outlier_rate": 0.1,
  "expected_text_anomaly_rate": 0.2,
  "expected_schema_conformance": 0.98,
  "expected_cross_field_consistency": 1.0
}
```

### Edge Cases

* Empty `DataFrame` — `duplicate_rate = 0.0`; outlier, text-anomaly, and schema logic degrade to their trivial `0.0`/`1.0` defaults since there are no rows to evaluate.
* Fewer than 10 complete rows in the numeric or text-feature matrix — Isolation Forest is skipped entirely for that component; the rate is the z-score rate alone.
* Numeric column containing `inf`/`-inf` — these are replaced with `NaN` and then dropped before outlier analysis, so they neither count as outliers nor crash the z-score computation.
* A text column where every value is identical (or a schema/category column with only one distinct value) — the derived length/digit/punct features have zero variance, which can trigger a `scipy` `RuntimeWarning` ("Precision loss occurred in moment calculation") during `zscore`; this is benign and mirrors the same behavior already possible with a constant numeric column.
* A text column entirely empty, all-null, or containing only empty strings — contributes no features and does not affect `text_anomaly_rate`; it also does not raise.
* A row with a `NaN` in one text column but valid text in another — dropped from `text_anomaly_rate` scoring entirely (not just for the null column), because the combined feature matrix drops any row with a missing feature.
* A schema-declared column entirely absent from `df` — every row counts as invalid for that column, which can dominate `schema_conformance` on a dataset with few schema columns.
* `expected == "float"` accepts any numeric value, including integer-valued ones — declaring a column `float` does not reject integers, only non-numeric values.
* `expected == "int"` requires exact integer-valued floats (`is_integer()`); a value like `3.5` fails even though it coerces numerically.
* A `consistency_rules` callable/expression that raises — must not propagate out of `compute()`; scored as `0.0` for that rule instead.
* `AtlasSchema.from_dict` with an unsupported type string (e.g. `"date"`) — raises `ValueError` before `compute()` is ever reached, so tests should exercise this at the schema-construction boundary, not inside `compute()`.
* Duplicate rows combined with outliers, text anomalies, and schema violations simultaneously — `error_rate` should equal the unweighted mean of all five components, so a bad reading in one component is diluted, not dominant, unless the others are also bad.
