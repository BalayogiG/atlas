# ATLAS

ATLAS evaluates tabular dataset quality through a small, extensible metric API.
Each metric takes a `pandas.DataFrame` and returns a `MetricResult` — a
`0.0`–`1.0` score, structured `details`, and human-readable `reasons` and
`recommendations`.

## Installation

```bash
pip install -e .
```

ATLAS reads CSV, Excel, JSON, and Parquet files out of the box. Two features
ship as optional extras so their (heavier) dependencies aren't forced on
everyone:

```bash
pip install -e ".[reasoning]"   # local-model reasoning, see Reasoning below
pip install -e ".[toxicity]"    # DQ003 Toxicity — pulls in detoxify (PyTorch), sarvamai, requests
```

## Quickstart

```python
from atlas import Atlas

report = Atlas().evaluate("employees.csv")
print(report.overall_score)

for metric_id, result in report.metrics.items():
    print(metric_id, result.metric, result.score, result.reasons)
```

Omitting `metrics` runs every registered metric. To run a subset, pass their
metric IDs:

```python
report = Atlas().evaluate("employees.csv", metrics=["DQ001"])
result = report.metrics["DQ001"]
print(result.score, result.details["missing_rate_by_column"])
```

ATLAS infers a schema automatically for the Data Quality & Noise metric
(`DQ002`). For deterministic production validation, provide an explicit
schema instead:

```python
from atlas import Atlas, AtlasSchema

schema = AtlasSchema.from_dict({"id": "int", "age": "int", "email": "email"})
report = Atlas().evaluate("employees.csv", metrics=["DQ002"], schema=schema)
```

`evaluate()` also accepts a `pandas.DataFrame` directly in place of a file
path, and an `output=` / `output_format=` pair (`"json"`, `"markdown"`, or
`"html"`) to save a rendered report alongside the returned `EvaluationReport`.

## Available metrics

| Metric ID | Metric | Documentation |
| --- | --- | --- |
| `DQ001` | Completeness | [docs/metrics/completeness.md](docs/metrics/completeness.md) |
| `DQ002` | Data Quality & Noise | [docs/metrics/data-quality-noise.md](docs/metrics/data-quality-noise.md) |
| `DQ003` | Toxicity | [docs/metrics/toxicity.md](docs/metrics/toxicity.md) |
| `DQ004` | Documentation & Transparency | [docs/metrics/documentation-transparency.md](docs/metrics/documentation-transparency.md) |

See [docs/metrics/README.md](docs/metrics/README.md) for the full metric
index, and each metric's own doc for its scoring formula, configuration
options, and `details` schema.

## Command-line interface

```bash
atlas evaluate --dataset employees.csv                       # run every registered metric
atlas evaluate --dataset employees.csv --metric DQ001         # run one metric (repeatable)
atlas evaluate --dataset employees.csv --output report.json --format json
atlas list-metrics                                            # table of registered metrics
atlas get-metric --name DQ001                                 # metadata for one metric
atlas version
```

Run `atlas --help` or `atlas <command> --help` for full option details. Note
that metric-specific options such as `required_fields`, `schema`,
`consistency_rules`, `text_field`/`sarvam_api_key`, and `dataset_path` are
only available through the Python API, not as CLI flags.

## Reasoning

Passing `reasoning=True` to `Atlas().evaluate()` (or `--reasoning` on the
CLI) sends each metric's score and details to a local [Ollama](https://ollama.com)
model to append a generated explanation and recommendations on top of the
built-in ones. It requires the `reasoning` extra and a running Ollama
instance; if the model call fails, ATLAS falls back to the built-in
`reasons`/`recommendations` and raises a `RuntimeWarning`.

## Extending ATLAS

New metrics implement the `Metric` abstract base class and register
themselves with the `@metric` decorator:

```python
from atlas import Metric, MetricResult, metric

@metric(name="My Metric", category="Custom", metric_id="CUSTOM001")
class MyMetric(Metric):
    def compute(self, df, **kwargs) -> MetricResult:
        ...
```

Once imported, the metric is available by its `metric_id` to
`Atlas().evaluate()` and the CLI.
