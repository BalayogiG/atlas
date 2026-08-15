# ATLAS Metrics

Each metric returns a `MetricResult` with a score from `0.0` (lowest quality) to
`1.0` (highest quality), structured details, reasons, and recommendations.

| Metric ID | Metric | Documentation |
| --- | --- | --- |
| `DQ001` | Completeness | [Completeness](completeness.md) |
| `DQ002` | Data Quality & Noise | [Data Quality & Noise](data-quality-noise.md) |
| `DQ003` | Toxicity | [Toxicity](toxicity.md) |

Run `atlas list-metrics` to view the registered metrics in the CLI.
