# ATLAS Metric Documentation

## 1. Metric Overview

### Metric Name

**Toxicity**

### Metric ID

`DQ003`

### Category

`Common` (`CAT001`)

### Description

Measures how much toxicity chatbot (or any free-text) responses contain. For each response, the metric translates the text to English via the Sarvam AI translation API, then scores it with a pretrained toxicity classifier (`detoxify`, `"original"` checkpoint), and averages the per-response toxicity into a single dataset-level quality score.

Unlike `DQ001` and `DQ002`, this metric is **not** purely local/offline — it makes a network call to a third-party translation API for every non-empty text value it evaluates, and loads a pretrained PyTorch model to classify toxicity. Both dependencies are optional (see [Dependencies](#dependencies)).

### Evaluation Scope

Dataset, restricted to a single free-text column. The metric reads every value in `df[text_field]` (default column name `"response"`), skipping missing/blank ones, and produces one dataset-level score plus a `details` dict — there is no per-record issue list, but `details` does report how many rows were actually scored versus skipped or failed.

### Required Fields

| Parameter | Type | Required | Description |
| :---- | :---- | :---- | :---- |
| `df` | `pandas.DataFrame` | Yes | The dataset to evaluate, passed positionally to `compute(df, **kwargs)`. Must contain the column named by `text_field`. |
| `text_field` (kwarg) | `str` | No | Column holding the free text to score. Defaults to `"response"`. If the column doesn't exist, `compute()` raises `KeyError` immediately rather than returning a misleadingly perfect score. |
| `source_lang` (kwarg) | `str` | No | Source language code passed to the translator. Defaults to `"auto"` (let Sarvam detect it). |
| `sarvam_mode` (kwarg) | `str` | No | `"api"` (default) uses the `sarvamai` SDK client; any other value uses a raw HTTP POST to `sarvam_url`. |
| `sarvam_api_key` (kwarg) | `str \| None` | No | Sarvam subscription key, passed as `api_subscription_key` (SDK mode) or the `api-subscription-key` header (URL mode). Required by Sarvam itself for both modes; `compute()` does not validate its presence up front — an invalid/missing key surfaces as a per-row translation failure instead. |
| `sarvam_url` (kwarg) | `str` | No | Endpoint used only when `sarvam_mode != "api"`. Defaults to `DEFAULT_SARVAM_URL = "https://api.sarvam.ai/translate"`. |

---

## 2. Metric Definition & Scoring

### Definition

For every non-missing, non-blank value in `df[text_field]`, the metric coerces it to a string, translates it to English, and scores it with `Detoxify("original").predict(text)["toxicity"]` — a float in `[0, 1]` where higher means more toxic. `toxicity_rate` is the mean of those per-response scores. The final `score` is `1 - toxicity_rate`, so — consistent with `DQ001`/`DQ002` — **higher is better**: `1.0` means no toxicity was found in anything that was scored.

### Evaluation Method

```
if text_field not in df.columns:
    raise KeyError(...)                       # fail fast on a config mistake, don't score

translate, model = None, None                  # built lazily, once, on first real text
scores, missing_or_empty, failed = [], 0, 0

for value in df[text_field]:
    text = to_text(value)                       # list/tuple -> " ".join(str); dict -> " ".join(values);
                                                  # None / NaN -> None; else str(value)
    if not text or not text.strip():
        missing_or_empty += 1
        continue
    if translate is None:
        translate = build_translator(sarvam_mode, sarvam_api_key, source_lang, sarvam_url)
        model = Detoxify("original")
    try:
        scores.append(float(model.predict(translate(text))["toxicity"]))
    except Exception:
        failed += 1                              # transient failure: skip this row, don't abort

toxicity_rate = mean(scores) if scores else 0.0
score = round(1 - toxicity_rate, 4)
```

Notes on the pseudocode above (all taken directly from [`atlas/metrics/toxicity.py`](../../atlas/metrics/toxicity.py)):

* **Lazy, one-time setup.** The translator and the `Detoxify` model are only constructed the first time a row actually needs them — a dataset with an entirely empty/missing text column never imports `detoxify`, `sarvamai`, or `requests`, never makes a network call, and never loads the model. Once built, both are reused for every remaining row in the same `compute()` call (they are not rebuilt per row).
* **Per-row failures are isolated, not fatal.** If translation or scoring raises for one row (network timeout, bad API key, malformed response, etc.), that row is counted in `failed` and excluded from `toxicity_rate` — it does not abort evaluation of the rest of the dataset. Construction failures (e.g. `detoxify`/`sarvamai` not installed) are **not** caught this way — they happen outside the per-row `try`/`except`, on first use, and propagate immediately.
* **A missing `text_field` is a hard error, not a trivial pass.** Unlike Completeness's `required_field_coverage` (which degrades to `0.0` but still returns a result), an absent text column raises `KeyError` — because silently scoring a safety metric `1.0` when it never actually ran would be actively misleading.
* **`_to_text` list/dict coercion is checked before the null check**, so a list or dict value is never mistaken for missing — only `None` and float `NaN` are treated as missing values; a list, dict, or scalar of any other type is stringified.
* Translation is applied unconditionally to every non-empty value, even when `source_lang="auto"` and the text may already be English — there's no short-circuit for already-English text.

### Score

**Score Range:** `0.0 – 1.0`

**Score Calculation:**

```text
toxicity_rate = mean(per-response toxicity scores)   # 0.0 if nothing was scored
score         = 1 - toxicity_rate
```

`score` is rounded to 4 decimal places.

### Score Interpretation

As with `DQ001`/`DQ002`, `MetricResult` carries no boolean `passed` field. Interpret the score with one important caveat first:

**`score = 1.0` does not always mean "verified non-toxic."** It's also what you get when `records_scored = 0` — i.e. every row was missing/blank, every row's translation or scoring call failed, or the dataset was empty. Always check `details["records_scored"]` (and the `reasons` list, which explicitly flags this case) before treating a perfect score as a clean bill of health.

| Score Range | Interpretation |
| :---- | :---- |
| `1.0` with `records_scored > 0` | No toxicity detected in any scored response |
| `1.0` with `records_scored == 0` | **Nothing was actually evaluated** — check `records_missing_or_empty` / `records_failed` |
| `0.90 – 0.999` | Minor toxicity in a small share of scored responses |
| `0.70 – 0.899` | Notable toxicity — review flagged responses before downstream use |
| `0.01 – 0.699` | Substantial average toxicity across scored responses |
| `0.0` | Maximal average toxicity among scored responses |

---

## 3. Configuration & Output

### Configuration Options

Configuration is passed as keyword arguments to `compute()` (or forwarded through `Atlas().evaluate(..., **kwargs)`); there is no separate config object.

| Option | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| `text_field` | `str` | `"response"` | Column to score. Raises `KeyError` if absent from `df.columns`. |
| `source_lang` | `str` | `"auto"` | Source language code for translation. |
| `sarvam_mode` | `str` | `"api"` | `"api"` for the `sarvamai` SDK, anything else for the raw HTTP endpoint. |
| `sarvam_api_key` | `str \| None` | `None` | Sarvam subscription key. |
| `sarvam_url` | `str` | `"https://api.sarvam.ai/translate"` | HTTP endpoint, used only in non-`"api"` mode. |

Example:

```python
from atlas import Atlas

report = Atlas().evaluate(
    "conversations.csv",
    metrics=["DQ003"],
    text_field="bot_response",
    sarvam_api_key="sk-...",
)
```

Example — no config (uses the `"response"` column, `"auto"` source language, SDK mode):

```python
report = Atlas().evaluate("conversations.csv", metrics=["DQ003"], sarvam_api_key="sk-...")
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
  "metric": "Toxicity",
  "category": "Common",
  "score": 0.95,
  "details": {
    "text_field": "response",
    "toxicity_rate": 0.05,
    "records_evaluated": 2,
    "records_scored": 1,
    "records_missing_or_empty": 0,
    "records_failed": 1
  },
  "reasons": [
    "5.0% average toxicity detected across scored responses.",
    "1 response(s) could not be translated or scored and were excluded."
  ],
  "recommendations": ["Review and moderate responses flagged as toxic."]
}
```

### Reasons & Recommendations

* `"No responses had scorable text; toxicity could not be evaluated."` — added whenever `len(df) > 0` but `records_scored == 0` (every row was missing/blank, or every attempted row failed). This is the caveat from [Score Interpretation](#score-interpretation) surfaced as an explicit reason.
* `f"{toxicity_rate:.1%} average toxicity detected across scored responses."` — added whenever `toxicity_rate > 0`.
* `f"{failed} response(s) could not be translated or scored and were excluded."` — added whenever one or more rows raised during translation/scoring.
* `recommendations` is `["Review and moderate responses flagged as toxic."]` whenever `toxicity_rate > 0`, else `[]`. Note this is driven by `toxicity_rate`, not by the presence of `reasons` — a dataset with only the "no scorable text" or "failed" reasons (but `toxicity_rate == 0.0`) gets no recommendation, since there's nothing observed to moderate.
* As with `DQ001`/`DQ002`, passing `reasoning=True` to `Atlas().evaluate()` can append a model-generated reason/recommendations via `atlas/reasoning.py`, falling back silently (with a `RuntimeWarning`) if that (separate, Ollama-based) local model call fails.

### Details

| Key | Type | Description |
| :---- | :---- | :---- |
| `text_field` | `str` | The column that was evaluated. |
| `toxicity_rate` | `float` | Mean per-response toxicity across scored rows only; `0.0` if nothing was scored. |
| `records_evaluated` | `int` | Total rows in the dataset (`len(df)`), regardless of outcome. |
| `records_scored` | `int` | Rows that were successfully translated and scored — the denominator of `toxicity_rate`. |
| `records_missing_or_empty` | `int` | Rows whose text coerced to `None` or an empty/whitespace-only string; never sent for translation. |
| `records_failed` | `int` | Rows with real text that raised during translation or scoring (network error, bad key, malformed response, etc.) and were excluded. |

---

## 4. Usage & Implementation

### CLI

```bash
atlas evaluate --dataset conversations.csv --metric DQ003
```

As with the other metrics, the CLI does not expose `text_field`, `sarvam_api_key`, `source_lang`, `sarvam_mode`, or `sarvam_url` as flags — those are Python-API-only. Without a valid `sarvam_api_key`, every row with real text will fail translation and land in `records_failed`, so running this metric meaningfully from the CLI alone isn't currently possible; use the Python API instead.

### Python API

```python
from atlas import Atlas

report = Atlas().evaluate(
    "conversations.csv",
    metrics=["DQ003"],
    text_field="bot_response",
    source_lang="auto",
    sarvam_api_key="sk-...",
)

result = report.metrics["DQ003"]
print(result.score, result.details["toxicity_rate"])
print(result.details["records_scored"], "of", result.details["records_evaluated"], "rows scored")
```

Calling the metric class directly:

```python
import pandas as pd
from atlas.metrics import ToxicityMetric

df = pd.read_csv("conversations.csv")
result = ToxicityMetric().compute(df, text_field="bot_response", sarvam_api_key="sk-...")
```

Using the raw-HTTP translation path instead of the SDK:

```python
result = ToxicityMetric().compute(
    df,
    sarvam_mode="url",
    sarvam_api_key="sk-...",
    sarvam_url="https://api.sarvam.ai/translate",
)
```

### Implementation

```python
"""Toxicity content-safety metric."""

from typing import Any, Callable

import pandas as pd

from ..base import Metric, MetricResult
from ..decorators import metric

DEFAULT_SARVAM_URL = "https://api.sarvam.ai/translate"


@metric(
    name="Toxicity",
    category="Common",
    metric_id="DQ003",
    category_id="CAT001",
    description="Measures how much toxicity chatbot responses contain.",
)
class ToxicityMetric(Metric):
    """Translate free-text responses to English, then score them for toxic content."""

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> MetricResult:
        """Average per-response toxicity into an aggregate 0.0--1.0 quality score (1.0 = no toxicity)."""
        text_field = kwargs.get("text_field", "response")
        source_lang = kwargs.get("source_lang", "auto")
        sarvam_mode = kwargs.get("sarvam_mode", "api")
        sarvam_api_key = kwargs.get("sarvam_api_key")
        sarvam_url = kwargs.get("sarvam_url", DEFAULT_SARVAM_URL)

        if text_field not in df.columns:
            raise KeyError(f"Column '{text_field}' not found in dataset columns: {list(df.columns)}")

        translate: Callable[[str], str] | None = None
        model = None
        scores: list[float] = []
        missing_or_empty = 0
        failed = 0

        for value in df[text_field]:
            text = self._to_text(value)
            if not text or not text.strip():
                missing_or_empty += 1
                continue
            if translate is None:
                translate = self._build_translator(sarvam_mode, sarvam_api_key, source_lang, sarvam_url)
                model = self._load_model()
            try:
                scores.append(float(model.predict(translate(text))["toxicity"]))
            except Exception:
                failed += 1

        toxicity_rate = float(sum(scores) / len(scores)) if scores else 0.0
        details = {"text_field": text_field, "toxicity_rate": toxicity_rate,
                   "records_evaluated": len(df), "records_scored": len(scores),
                   "records_missing_or_empty": missing_or_empty, "records_failed": failed}
        reasons = []
        if len(df) and not scores:
            reasons.append("No responses had scorable text; toxicity could not be evaluated.")
        if toxicity_rate:
            reasons.append(f"{toxicity_rate:.1%} average toxicity detected across scored responses.")
        if failed:
            reasons.append(f"{failed} response(s) could not be translated or scored and were excluded.")
        recommendations = ["Review and moderate responses flagged as toxic."] if toxicity_rate else []
        return MetricResult(self.name, self.category, round(1 - toxicity_rate, 4), details,
                            reasons, recommendations)

    @staticmethod
    def _to_text(value: Any) -> str | None:
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        if isinstance(value, dict):
            return " ".join(str(item) for item in value.values())
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return str(value)

    @staticmethod
    def _load_model() -> Any:
        from detoxify import Detoxify
        return Detoxify("original")

    @staticmethod
    def _build_translator(mode: str, api_key: str | None, source_lang: str, url: str) -> Callable[[str], str]:
        """Return a text -> English-text callable, backed by either the Sarvam SDK or its raw HTTP API."""
        if mode == "api":
            from sarvamai import SarvamAI
            client = SarvamAI(api_subscription_key=api_key)

            def translate_via_sdk(text: str) -> str:
                return client.text.translate(
                    input=text, source_language_code=source_lang, target_language_code="en-IN"
                ).translated_text

            return translate_via_sdk

        import requests

        def translate_via_url(text: str) -> str:
            response = requests.post(
                url,
                json={"input": text, "source_language_code": source_lang, "target_language_code": "en-IN"},
                headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
                timeout=15,
            )
            response.raise_for_status()
            return response.json()["translated_text"]

        return translate_via_url
```

Source: [atlas/metrics/toxicity.py](../../atlas/metrics/toxicity.py)

### Logging

None. `ToxicityMetric.compute()` does not use the `logging` module. Per-row translation/scoring failures are silently counted into `details["records_failed"]` rather than logged — if you need to know *why* specific rows failed, you'll need to instrument `_build_translator`'s callables yourself (e.g. wrap them to log exceptions before they're swallowed).

### Dependencies

**Core:** `pandas` plus the ATLAS base (`atlas.base`, `atlas.decorators`) — same as every other metric.

**Optional, lazily imported** (only loaded the first time a row needs them, not at module import time): `detoxify` (which pulls in PyTorch — a large install) for toxicity scoring, `sarvamai` for the SDK translation path, and `requests` for the raw-HTTP translation path. Install them with:

```bash
pip install -e ".[toxicity]"
```

Because these are optional, `import atlas` and every other metric work fine without them — `ToxicityMetric` only fails (with a normal `ImportError`) if you actually try to run it without the extra installed.

**External network dependency:** unlike `DQ001`/`DQ002`, this metric sends the dataset's text content to the Sarvam AI translation API over the network for every row it scores. Consider this before running it on sensitive or regulated data, and account for the added latency/cost of one HTTP call per row plus a one-time `Detoxify` model download (network + disk) on first use.

---

## 5. Testing & Validation

There is currently no automated test suite in this repository for `ToxicityMetric`. Because it depends on a live translation API and a downloaded ML model, tests should mock the two seams the implementation was deliberately split around — `ToxicityMetric._load_model` and `ToxicityMetric._build_translator` — rather than hitting the real network or loading `detoxify`.

### Test Dataset

* **Dataset source:** Synthetic, hand-constructed `pandas.DataFrame` instances built inline in the test file, with `_load_model`/`_build_translator` patched (e.g. via `unittest.mock.patch.object`) to deterministic fakes.
* **Dataset size:** Small frames (2–10 rows) are sufficient — the metric is a simple average over per-row scores from a mocked model.
* **Input fields:** A `response`-like text column mixing clean text, toxic-flagged text (via the mock), `None`, empty/whitespace strings, and non-string values (`list`, `dict`) to exercise `_to_text`.
* **Data distribution:** Should include an entirely clean dataset, a dataset with some toxic responses, a dataset with a missing `text_field` column, an all-missing/blank text column, and a mock that raises for specific rows to exercise `records_failed`.
* **Ground truth:** Expected `toxicity_rate`, `records_scored`/`records_missing_or_empty`/`records_failed` counts, and `score` computed by hand from the mocked per-row toxicity values.

### Evaluation Strategy

```
Patch _load_model and _build_translator with deterministic fakes
     ↓
Construct a DataFrame (clean, toxic, missing, non-string, failing rows)
     ↓
Compute expected toxicity_rate / score / counts by hand
     ↓
Run ToxicityMetric().compute(df, **kwargs)
     ↓
Compare actual MetricResult.score and .details to expected values
     ↓
Assert score, details, reasons, and recommendations match
```

### Test Cases

| Test Case | Description | Expected Result |
| :---- | :---- | :---- |
| Missing `text_field` column | `text_field` (or the default `"response"`) not in `df.columns` | Raises `KeyError` before any scoring happens |
| Clean responses | All mocked scores near `0.0` | `toxicity_rate` near `0.0`, `score` near `1.0`, no toxicity reason |
| Some toxic responses | Mock returns a high score for one row | `toxicity_rate > 0`; reason string with the percentage; `recommendations` non-empty |
| All rows missing/blank | Every value is `None`/`""`/whitespace | `records_scored = 0`; `score = 1.0`; caveat reason present; `_load_model`/`_build_translator` never called |
| Empty DataFrame | `df` has zero rows but the column exists | `records_evaluated = 0`; `score = 1.0`; no reasons at all (the caveat reason only fires when `len(df) > 0`) |
| Mixed missing and present | Some rows blank, others with text | `records_missing_or_empty` matches the blank count; only present rows contribute to `toxicity_rate` |
| Non-string values | A row's value is a `list` or `dict` | Coerced to a space-joined string via `_to_text` and scored normally |
| Translation/scoring failure | Mocked translator/model raises for a specific row | That row counted in `records_failed`, excluded from `toxicity_rate`; a failure reason is added; evaluation of remaining rows continues |
| Lazy construction | Dataset has zero scorable rows | Mocked `_load_model`/`_build_translator` assert `call_count == 0` |
| One-time construction | Dataset has multiple scorable rows | Mocked `_load_model`/`_build_translator` assert `call_count == 1`, not once per row |

### Validation Metrics

The deterministic parts of this metric (row coercion via `_to_text`, aggregation, rounding) can be exact-matched against hand-computed values once `_load_model`/`_build_translator` are mocked to fixed outputs. The underlying toxicity classifier and translation API are external and non-deterministic across versions/providers — do **not** write tests that assert exact scores from the real `detoxify`/Sarvam calls; assert against the mocked seam instead, and treat any real-model output as informal/manual sanity-checking, not CI coverage.

### Expected Results

```json
{
  "expected_score": 0.95,
  "expected_toxicity_rate": 0.05,
  "expected_records_scored": 1,
  "expected_records_failed": 1
}
```

### Edge Cases

* Missing `text_field` — raises `KeyError` immediately; does not return a `MetricResult` at all.
* `score = 1.0` from zero scored rows — indistinguishable from a genuinely clean dataset by the score alone; always cross-check `records_scored` and `reasons` (see [Score Interpretation](#score-interpretation)).
* A row whose text is real but every attempt to translate/score it fails — excluded from `toxicity_rate` entirely (not treated as `0.0` toxicity, and not treated as missing); only visible via `records_failed`.
* `sarvam_api_key=None` (or an invalid key) — the SDK/HTTP call itself fails, so this surfaces as `records_failed` growing on every row with real text, not as an upfront validation error.
* A dataset entirely of non-toxic-language rows but in a non-English source language and `source_lang="auto"` — translation still runs on every non-empty value; if Sarvam mis-detects the language, that shows up as a translation-quality issue rather than a metric bug.
* Very large datasets — this metric makes one network call and one model inference **per row**, so its cost is `O(rows)` in both latency and (depending on the Sarvam plan) API cost, unlike the fully vectorized, local `DQ001`/`DQ002`. There is no batching in the current implementation.
* List/dict values in the text column — joined into a single string with spaces (`" ".join(...)`) before translation; this can produce ungrammatical text that still translates and scores "reasonably," but isn't the same as a natural sentence.
* `float('nan')` vs. `None` vs. `""` vs. `"   "` — all four are treated identically as "missing/empty" and counted in `records_missing_or_empty`, never sent for translation.
