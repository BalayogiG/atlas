# ATLAS Metric Documentation

## 1. Metric Overview

### Metric Name

**Documentation & Transparency**

### Metric ID

`DQ004`

### Category

`Common` (`CAT001`)

### Description

Measures the completeness and transparency of a dataset's documentation (README / dataset card). The metric loads a documentation document from either a local dataset directory or the Hugging Face Hub, parses it into Markdown sections, and blends five sub-scores — required-section completeness, intended-use clarity, dataset-field coverage, limitations disclosure, and reproducibility notes — into a single weighted score.

Unlike `DQ001`/`DQ002`, this metric can be **not** purely local/offline: if `dataset_path` resolves to a Hugging Face dataset repo id or URL, `compute()` makes a network call to `huggingface.co` to fetch `README.md`. When `dataset_path` is a local directory, no network access occurs.

### Evaluation Scope

Dataset plus one external document (a README / dataset card). `df` is only consulted for `field_definition_coverage` (checking whether column names appear in the documentation text); every other sub-score is derived purely from the documentation text. Produces one dataset-level score plus a `details` dict — there is no per-record issue list.

### Required Fields

| Parameter | Type | Required | Description |
| :---- | :---- | :---- | :---- |
| `df` | `pandas.DataFrame` | Yes | The dataset to evaluate, passed positionally to `compute(df, **kwargs)`. Used only for field-name coverage; an empty `df` is valid (see [Evaluation Method](#evaluation-method)). |
| `dataset_path` (kwarg) | `str` | **Yes** | Where to load documentation from. One of: a local directory path containing `README.md`/`README.MD`/`DATASET_CARD.md`/`datasheet.md`; a bare Hugging Face dataset repo id (e.g. `"paulopontesm/titanic"`); or a full Hugging Face dataset URL (e.g. `"https://huggingface.co/datasets/paulopontesm/titanic"`). Unlike `text_field` on `DQ003`, this has **no default** — `compute()` raises `ValueError` immediately if it's missing or falsy. |

---

## 2. Metric Definition & Scoring

### Definition

If no documentation can be loaded at all, the metric short-circuits to a score of `0.0`. Otherwise, the documentation is split into Markdown sections and scored along five independent `0.0`–`1.0` axes, which are combined as a weighted sum:

```text
score = 0.25 * documentation_completeness
      + 0.20 * intended_use_clarity
      + 0.20 * field_definition_coverage
      + 0.20 * limitations_disclosure
      + 0.15 * reproducibility_notes
```

The weights sum to `1.00`, so `score` stays in `[0, 1]` — consistent with every other ATLAS metric, **higher is better**.

### Evaluation Method

```
documentation = load_documentation(dataset_path)   # local dir, or Hugging Face Hub fetch

if not documentation:
    return score=0.0, details={"documentation_found": False}, ...    # short-circuit, no sub-scores computed

sections = extract_sections(documentation)          # naive Markdown heading parser

documentation_completeness, found, missing = documentation_completeness(sections)
    # fraction of the 9 REQUIRED_SECTIONS present with non-blank content

field_definition_coverage = field_definition_coverage(df, documentation)
    # fraction of df.columns whose (lowercased) name appears anywhere in the doc text

intended_use_clarity = keyword_score(
    get_section(sections, "Uses", "Direct Use", "Out-of-Scope Use"),
    ["purpose", "use", "user", "model", "application"],
)

limitations_disclosure = keyword_score(
    get_section(sections, "Bias, Risks, and Limitations", "Limitations"),
    ["bias", "risk", "limitation", "missing", "fairness"],
)

reproducibility_notes = keyword_score(
    documentation,                                   # the FULL raw text, not a section
    ["version", "license", "source", "collection", "preprocessing"],
)

score = 0.25*documentation_completeness + 0.20*intended_use_clarity
      + 0.20*field_definition_coverage + 0.20*limitations_disclosure
      + 0.15*reproducibility_notes
```

Notes on the pseudocode above (all taken directly from [`atlas/metrics/documentation_transparency.py`](../../atlas/metrics/documentation_transparency.py) and [`atlas/utils.py`](../../atlas/utils.py)):

* **`load_documentation` resolution order.** `dataset_path` is first checked against a Hugging Face pattern: either it starts with `https://huggingface.co/datasets/`, or it matches a bare `namespace/name` regex **and** no local path exists at that string. If it matches, the metric fetches `https://huggingface.co/datasets/{repo_id}/raw/main/README.md` via `urllib.request` (stdlib only, no extra dependency) with a 10-second timeout and a generic `User-Agent` header. **Any** failure — 404 (repo/README doesn't exist), a private/gated repo requiring auth, a timeout, DNS failure, etc. — is caught, logged at `WARNING` (see [Logging](#logging)), and returns `""`, which then hits the same "no documentation found" branch as a dataset with no README at all. There is no way to distinguish "the repo doesn't exist" from "the repo exists but has no README" from "the network call failed" by looking at the returned `score`/`details` alone — check the log line (or catch the underlying case yourself) if that distinction matters.
* **Local-directory fallback.** If `dataset_path` isn't recognized as a Hugging Face id/URL, it's treated as a local directory. The metric checks `README.md`, `README.MD`, `DATASET_CARD.md`, `datasheet.md` in that order and returns the first one found; a nonexistent directory or a directory with none of those files also returns `""`.
* **`extract_sections` is a naive parser**, not a proper Markdown parser: it splits on any line matching `^#+\s+(.*)` (any heading level `#` through `######`), and uses only the heading *text* (not the level) as the dict key. This means `"## Uses"` and `"### Uses"` collide into the same section, and if a heading text repeats later in the document, the later occurrence's content silently overwrites the earlier one's (it's a plain `dict`, last write wins). Content before the first heading is bucketed under a synthetic `"ROOT"` key.
* **`documentation_completeness` requires non-blank content**, not just a matching heading — a heading immediately followed by another heading (i.e. an empty section) counts as *missing*, not found. The 9 required sections are hardcoded in `atlas.utils.REQUIRED_SECTIONS`: `Dataset Details`, `Dataset Description`, `Dataset Sources`, `Uses`, `Dataset Structure`, `Dataset Creation`, `Annotations`, `Bias, Risks, and Limitations`, `Citation`.
* **`field_definition_coverage` is a coarse, case-insensitive substring match** — `column.lower() in documentation.lower()` — not a word-boundary or table-cell match. A short/generic column name (e.g. `"id"`, `"age"`) can register as "covered" just because that substring appears anywhere in the prose, even unrelated to a field description. An empty `df` returns `1.0` (vacuously "fully covered") rather than `0.0` or an error.
* **`reproducibility_notes` scores the entire raw document**, not a specific section — unlike `intended_use_clarity`/`limitations_disclosure`, which are scoped to specific sections via `get_section`. A README that mentions "license" and "version" anywhere (even outside a dedicated reproducibility section) scores those keywords as present.
* **`get_section` returns the *first* matching name** from its candidate list, not a merge — e.g. `intended_use_clarity` reads `"Uses"` if present, and only falls back to `"Direct Use"`/`"Out-of-Scope Use"` if `"Uses"` is entirely absent from `sections`, even if those other sections also exist and contain relevant text.
* **No documentation vs. thin documentation are scored very differently.** `documentation = ""` short-circuits to `score = 0.0` and a `details` dict containing only `{"documentation_found": False}` — the five sub-score keys aren't present at all in that case (see [Details](#details)). Any non-empty documentation, however sparse, computes and reports all five sub-scores normally, even if most of them come out near `0.0`.

### Score

**Score Range:** `0.0 – 1.0`

**Score Calculation:**

```text
score = 0.25*documentation_completeness + 0.20*intended_use_clarity
      + 0.20*field_definition_coverage + 0.20*limitations_disclosure
      + 0.15*reproducibility_notes
```

`score` (and each individual sub-score in `details`) is rounded to 4 decimal places.

### Score Interpretation

As with the other ATLAS metrics, `MetricResult` carries no boolean `passed` field, and the code defines no explicit score-band thresholds. One important caveat first:

**`score = 0.0` does not always mean "documentation is empty/bad."** It's also what you get when the Hugging Face fetch failed for an unrelated reason (bad repo id, network error, private repo, rate limiting) — see the `load_documentation` note above. Always check `details["documentation_found"]` before concluding the dataset genuinely has no documentation.

| Score Range | Interpretation |
| :---- | :---- |
| `0.0` with `details == {"documentation_found": False}` | **No documentation was loaded at all** — could be a genuinely missing README, or a failed Hugging Face fetch (bad repo id, network error, private repo). Sub-scores were never computed. |
| `0.85 – 1.0` | Thorough documentation: most required sections present with content, fields referenced, use/limitations/reproducibility notes all reasonably covered. |
| `0.5 – 0.849` | Partial documentation: some required sections or keyword signals present, others missing. |
| `0.01 – 0.499` | Minimal documentation: exists, but is thin against most of the five axes. |
| `0.0` with sub-score keys present in `details` | Documentation was found but scored `0.0` on every axis (extremely unlikely in practice, since even a short README tends to hit at least one keyword). |

---

## 3. Configuration & Output

### Configuration Options

Configuration is passed as keyword arguments to `compute()` (or forwarded through `Atlas().evaluate(..., **kwargs)`); there is no separate config object.

| Option | Type | Default | Description |
| :---- | :---- | :---- | :---- |
| `dataset_path` | `str` | **none — required** | Local dataset directory, Hugging Face repo id, or Hugging Face dataset URL. Raises `ValueError` if missing/falsy. |

Example — local dataset directory:

```python
from atlas import Atlas

report = Atlas().evaluate(
    "employees.csv",
    metrics=["DQ004"],
    dataset_path="./datasets/employees",   # directory containing README.md
)
```

Example — Hugging Face dataset repo id:

```python
report = Atlas().evaluate(
    df,
    metrics=["DQ004"],
    dataset_path="paulopontesm/titanic",
)
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

Serialized example — documentation found (via `Atlas().evaluate(..., output="report.json")`):

```json
{
  "metric": "Documentation & Transparency",
  "category": "Common",
  "score": 0.7367,
  "details": {
    "documentation_completeness": 0.6667,
    "intended_use_clarity": 0.6,
    "field_definition_coverage": 0.8,
    "limitations_disclosure": 1.0,
    "reproducibility_notes": 0.6,
    "found_sections": ["Dataset Description", "Uses", "Dataset Structure",
                        "Bias, Risks, and Limitations", "Citation", "Dataset Sources"],
    "missing_sections": ["Dataset Details", "Dataset Creation", "Annotations"]
  },
  "reasons": ["6 of 9 required documentation sections were found."],
  "recommendations": [
    "Add the missing documentation sections.",
    "Include version, source, collection, preprocessing, and licensing information."
  ]
}
```

Serialized example — no documentation found (missing locally, or a failed Hugging Face fetch):

```json
{
  "metric": "Documentation & Transparency",
  "category": "Common",
  "score": 0.0,
  "details": {"documentation_found": false},
  "reasons": ["No dataset documentation found."],
  "recommendations": ["Add a README.md or Dataset Card describing the dataset."]
}
```

### Reasons & Recommendations

* `"No dataset documentation found."` — the only reason produced when `load_documentation()` returns `""` (see [Score Interpretation](#score-interpretation) for the ambiguity this can hide).
* `f"{len(found)} of {len(found) + len(missing)} required documentation sections were found."` — always produced when documentation *was* found, regardless of how many sections were actually present.
* `recommendations` when documentation is missing: always exactly `["Add a README.md or Dataset Card describing the dataset."]`.
* `recommendations` when documentation is found, built conditionally and independently (any subset can appear, in this order):
  * `"Add the missing documentation sections."` — whenever `missing` is non-empty.
  * `"Document every dataset field with a description."` — whenever `field_definition_coverage < 1`.
  * `"Include version, source, collection, preprocessing, and licensing information."` — whenever `reproducibility_notes < 1`.
* **No recommendation is tied to `intended_use_clarity` or `limitations_disclosure`** — both feed into `score` (40% of the weight combined) but neither low value produces its own recommendation string, unlike `documentation_completeness`, `field_definition_coverage`, and `reproducibility_notes`, each of which does.
* As with the other metrics, passing `reasoning=True` to `Atlas().evaluate()` can append a model-generated reason/recommendations via `atlas/reasoning.py`, falling back silently (with a `RuntimeWarning`) if that (separate, Ollama-based) local model call fails.

### Details

When documentation is found:

| Key | Type | Description |
| :---- | :---- | :---- |
| `documentation_completeness` | `float` | Fraction of the 9 `REQUIRED_SECTIONS` present with non-blank content. |
| `intended_use_clarity` | `float` | Keyword coverage (5 keywords) within the `"Uses"`/`"Direct Use"`/`"Out-of-Scope Use"` section (first match only). |
| `field_definition_coverage` | `float` | Fraction of `df.columns` whose lowercased name appears anywhere in the (lowercased) documentation text. `1.0` if `df` is empty. |
| `limitations_disclosure` | `float` | Keyword coverage (5 keywords) within the `"Bias, Risks, and Limitations"`/`"Limitations"` section (first match only). |
| `reproducibility_notes` | `float` | Keyword coverage (5 keywords) across the **entire** raw documentation text. |
| `found_sections` | `list[str]` | Required section names that were present with non-blank content. |
| `missing_sections` | `list[str]` | Required section names that were absent, or present but blank. |

When no documentation was found:

| Key | Type | Description |
| :---- | :---- | :---- |
| `documentation_found` | `bool` | Always `False` in this branch. This is the *only* key present — none of the five sub-score keys above exist in `details` when documentation could not be loaded. |

---

## 4. Usage & Implementation

### CLI

```bash
atlas evaluate --dataset employees.csv --metric DQ004
```

The CLI does not expose `dataset_path` as a flag — like `text_field`/`sarvam_api_key` on `DQ003`, it's Python-API-only. Since `dataset_path` is *required* (no default), running `DQ004` from the CLI alone always fails with `ValueError: 'dataset_path' must be provided.`; use the Python API instead.

### Python API

```python
from atlas import Atlas

report = Atlas().evaluate(
    "employees.csv",
    metrics=["DQ004"],
    dataset_path="paulopontesm/titanic",   # Hugging Face repo id
)

result = report.metrics["DQ004"]
print(result.score, result.details.get("documentation_completeness"))
print(result.details.get("found_sections"), result.details.get("missing_sections"))
```

Calling the metric class directly:

```python
import pandas as pd
from atlas.metrics import DocumentationTransparencyMetric

df = pd.read_csv("employees.csv")
result = DocumentationTransparencyMetric().compute(df, dataset_path="./datasets/employees")
```

Using a full Hugging Face dataset URL instead of a bare repo id:

```python
result = DocumentationTransparencyMetric().compute(
    df,
    dataset_path="https://huggingface.co/datasets/paulopontesm/titanic",
)
```

### Implementation

```python
"""Documentation & Transparency quality metric."""

from typing import Any

import pandas as pd

from ..base import Metric, MetricResult
from ..decorators import metric
from ..utils import (
    load_documentation,
    extract_sections,
    documentation_completeness,
    field_definition_coverage,
    keyword_score,
    get_section,
)


@metric(
    name="Documentation & Transparency",
    category="Common",
    metric_id="DQ004",
    category_id="CAT001",
    description="Measures the completeness and transparency of dataset documentation.",
)
class DocumentationTransparencyMetric(Metric):
    """Measure documentation quality."""

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> MetricResult:
        dataset_path = kwargs.get("dataset_path")
        if not dataset_path:
            raise ValueError("'dataset_path' must be provided.")

        documentation = load_documentation(dataset_path)
        if not documentation:
            return MetricResult(
                self.name, self.category, 0.0,
                {"documentation_found": False},
                ["No dataset documentation found."],
                ["Add a README.md or Dataset Card describing the dataset."],
            )

        sections = extract_sections(documentation)
        completeness, found, missing = documentation_completeness(sections)
        field_coverage = field_definition_coverage(df, documentation)
        intended_use = keyword_score(
            get_section(sections, "Uses", "Direct Use", "Out-of-Scope Use"),
            ["purpose", "use", "user", "model", "application"],
        )
        limitations = keyword_score(
            get_section(sections, "Bias, Risks, and Limitations", "Limitations"),
            ["bias", "risk", "limitation", "missing", "fairness"],
        )
        reproducibility = keyword_score(
            documentation,
            ["version", "license", "source", "collection", "preprocessing"],
        )

        score = (0.25 * completeness + 0.20 * intended_use + 0.20 * field_coverage
                 + 0.20 * limitations + 0.15 * reproducibility)

        details = {
            "documentation_completeness": round(completeness, 4),
            "intended_use_clarity": round(intended_use, 4),
            "field_definition_coverage": round(field_coverage, 4),
            "limitations_disclosure": round(limitations, 4),
            "reproducibility_notes": round(reproducibility, 4),
            "found_sections": found,
            "missing_sections": missing,
        }
        reasons = [f"{len(found)} of {len(found) + len(missing)} required documentation sections were found."]
        recommendations = []
        if missing:
            recommendations.append("Add the missing documentation sections.")
        if field_coverage < 1:
            recommendations.append("Document every dataset field with a description.")
        if reproducibility < 1:
            recommendations.append("Include version, source, collection, preprocessing, and licensing information.")

        return MetricResult(self.name, self.category, round(score, 4), details, reasons, recommendations)
```

Source: [atlas/metrics/documentation_transparency.py](../../atlas/metrics/documentation_transparency.py)

The Hugging Face fetch logic lives in [`atlas/utils.py`](../../atlas/utils.py), not in the metric itself:

```python
HUGGINGFACE_REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")
HUGGINGFACE_README_URL = "https://huggingface.co/datasets/{repo_id}/raw/main/README.md"

def _huggingface_repo_id(dataset_path: str) -> str | None:
    if dataset_path.startswith(("http://", "https://")):
        if "huggingface.co/datasets/" not in dataset_path:
            return None
        remainder = dataset_path.split("huggingface.co/datasets/", 1)[1]
        segments = [segment for segment in remainder.split("/") if segment]
        return "/".join(segments[:2]) if len(segments) >= 2 else None
    if HUGGINGFACE_REPO_PATTERN.match(dataset_path) and not Path(dataset_path).exists():
        return dataset_path
    return None

def _fetch_huggingface_readme(repo_id: str) -> str:
    request = urllib.request.Request(
        HUGGINGFACE_README_URL.format(repo_id=repo_id),
        headers={"User-Agent": "atlas-dq-metrics"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, urllib.error.HTTPError):
        return ""

def load_documentation(dataset_path: str) -> str:
    repo_id = _huggingface_repo_id(dataset_path)
    if repo_id:
        return _fetch_huggingface_readme(repo_id)
    path = Path(dataset_path)
    for file in ["README.md", "README.MD", "DATASET_CARD.md", "datasheet.md"]:
        doc = path / file
        if doc.exists():
            return doc.read_text(encoding="utf-8", errors="ignore")
    return ""
```

Source: [atlas/utils.py](../../atlas/utils.py)

### Logging

`DocumentationTransparencyMetric.compute()` logs via `logging.getLogger("atlas.metrics.documentation_transparency")`, and the Hugging Face fetch helper logs via `logging.getLogger("atlas.utils")`. Silent by default (see `CompletenessMetric`'s logging note for the `NullHandler` pattern); configure `logging.basicConfig(level=...)` in your application to see it.

* `DEBUG` — the resolved `dataset_path` at the start of `compute()`, and the final score with the found/missing section count.
* `WARNING` (`atlas.utils`) — emitted by `_fetch_huggingface_readme` whenever the Hugging Face fetch raises `URLError`/`HTTPError`, including the underlying error (e.g. `HTTP Error 401: Unauthorized` for a gated/private repo, or a timeout). This is the fix for the ambiguity called out in [Evaluation Method](#evaluation-method) and [Score Interpretation](#score-interpretation) — the score/`details` alone still can't distinguish "no README" from "fetch failed," but the log line now can.
* `WARNING` (`atlas.metrics.documentation_transparency`) — emitted whenever `compute()` falls into the "no documentation found" branch, regardless of cause (missing local file vs. unresolvable/failed Hugging Face fetch).

### Dependencies

**Core only:** `pandas` plus the ATLAS base (`atlas.base`, `atlas.decorators`, `atlas.utils`) — same as `DQ001`/`DQ002`. Unlike `DQ003`, this metric needs **no optional extra**; the Hugging Face fetch uses `urllib.request`/`urllib.error` from the Python standard library, not `requests` or `huggingface_hub`.

**External network dependency (conditional):** only triggered when `dataset_path` resolves to a Hugging Face repo id or URL. One HTTP `GET` to `huggingface.co` per `compute()` call, with a 10-second timeout. No network access occurs for local-directory `dataset_path` values.

---

## 5. Testing & Validation

There is currently no automated test suite in this repository for `DocumentationTransparencyMetric`. Because the Hugging Face path depends on a live network call, tests should mock `atlas.utils.urllib.request.urlopen` (or patch `atlas.utils.load_documentation` / `_fetch_huggingface_readme` directly) rather than hitting the real Hub.

### Test Dataset

* **Dataset source:** Synthetic Markdown strings built inline in the test file (covering all 9 required sections, a subset, or none), paired with small `pandas.DataFrame` instances whose column names are deliberately mentioned or omitted from the fake documentation text.
* **Dataset size:** A handful of short README fixtures (a few hundred characters each) is sufficient — the metric's cost is linear in document length via regex/substring scans, not row count.
* **Input fields:** `dataset_path` variants — an existing local directory with each candidate filename (`README.md`, `README.MD`, `DATASET_CARD.md`, `datasheet.md`), a directory with none of them, a bare Hugging Face-shaped repo id, and a full Hugging Face URL.
* **Data distribution:** Should include a fully complete dataset card (all 9 sections, all keyword groups hit), a partially complete one, an entirely empty/whitespace document, a document with duplicate headings (to exercise the "last write wins" `extract_sections` behavior), and a mocked Hugging Face fetch that raises/returns a 404.
* **Ground truth:** Expected `found`/`missing` section lists, each sub-score, and the final weighted `score`, computed by hand from the fixture text and keyword lists.

### Evaluation Strategy

```
Patch urllib.request.urlopen (or load_documentation) with a deterministic fake
     ↓
Construct dataset_path variants (local dir, HF repo id, HF URL, missing/empty doc)
     ↓
Compute expected sub-scores and weighted score by hand
     ↓
Run DocumentationTransparencyMetric().compute(df, dataset_path=...)
     ↓
Compare actual MetricResult.score and .details to expected values
     ↓
Assert score, details, reasons, and recommendations match
```

### Test Cases

| Test Case | Description | Expected Result |
| :---- | :---- | :---- |
| Missing `dataset_path` | `dataset_path` omitted or falsy | Raises `ValueError` before any loading happens |
| No documentation found (local) | Directory exists but has none of the candidate filenames | `score = 0.0`; `details == {"documentation_found": False}` |
| No documentation found (Hugging Face) | Mocked fetch raises `URLError`/`HTTPError` or returns 404 | Same as above — indistinguishable from a missing local README |
| All required sections present | Fixture doc has non-blank content under all 9 `REQUIRED_SECTIONS` | `documentation_completeness = 1.0`; `missing_sections = []` |
| Some sections missing or blank | Some headings absent, one heading present but empty | Absent and blank-content headings both land in `missing_sections` |
| Duplicate headings | Same heading text appears twice with different content | Later occurrence's content wins (dict overwrite in `extract_sections`) |
| Field coverage — all columns mentioned | Every `df.columns` name appears in the doc text | `field_definition_coverage = 1.0`; no "document every field" recommendation |
| Field coverage — empty DataFrame | `df` has zero columns/rows | `field_definition_coverage = 1.0` (vacuous), regardless of doc content |
| Bare Hugging Face repo id | `dataset_path="org/name"` where no local directory of that name exists | Routed through `_fetch_huggingface_readme`, not the local-directory branch |
| Local directory that looks like a repo id | `dataset_path="org/name"` where `org/name` *is* an existing local directory | Routed through the local-directory branch instead (local path takes priority) |
| Full Hugging Face URL | `dataset_path="https://huggingface.co/datasets/org/name"` | repo id `"org/name"` correctly extracted before the fetch |
| Weighted score composition | All five sub-scores at known fixed values | `score` matches `0.25*c + 0.20*u + 0.20*f + 0.20*l + 0.15*r` exactly (to 4 decimals) |

### Validation Metrics

Every part of this metric is deterministic once the documentation text is fixed (regex-based section extraction, substring keyword scoring, weighted sum) — exact-match assertions against hand-computed values are appropriate, no tolerance needed beyond the `round(..., 4)` the implementation itself applies. The only non-deterministic seam is the live Hugging Face network call; mock it rather than asserting against real Hub content, which can change over time.

### Expected Results

```json
{
  "expected_documentation_completeness": 0.6667,
  "expected_field_definition_coverage": 0.8,
  "expected_intended_use_clarity": 0.6,
  "expected_limitations_disclosure": 1.0,
  "expected_reproducibility_notes": 0.6,
  "expected_score": 0.7367
}
```

### Edge Cases

* Missing `dataset_path` — raises `ValueError` immediately; does not return a `MetricResult` at all (unlike the "no documentation found" case, which does).
* `score = 0.0` from a failed Hugging Face fetch — indistinguishable from a genuinely undocumented dataset by the score/`details` alone (see [Score Interpretation](#score-interpretation)); there is no separate "fetch failed" signal.
* A local directory whose name happens to match the `namespace/name` Hugging Face pattern — resolved as local, not remote, because `_huggingface_repo_id` explicitly checks `Path(dataset_path).exists()` first.
* Duplicate Markdown headings anywhere in the document — silently collapse to one entry in `sections`, with the later occurrence winning; this can make `documentation_completeness` under- or over-count depending on which occurrence had content.
* A heading matched at any level (`#` through `######`) — `"# Uses"`, `"## Uses"`, and `"### Uses"` are all treated as the exact same section key.
* Short/generic column names (e.g. `"id"`) — can register as "covered" in `field_definition_coverage` via an unrelated substring match, inflating that sub-score without a real field description existing.
* A gated or private Hugging Face dataset repo — the anonymous HTTP request has no auth token support, so this always fails and falls into the "no documentation found" branch, even if the dataset genuinely has excellent documentation.
* Very large README files — the metric reads the whole document into memory and runs a handful of regex/substring scans over it; no streaming or size limit is enforced on the Hugging Face response.
