"""ATLAS evaluation orchestration."""

from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import pandas as pd

from . import metrics as _built_in_metrics 
from .base import EvaluationReport, MetricResult
from .reasoning import AtlasReasoner
from .registry import get_metric, list_metrics
from .utils import build_report, load_dataset, save_html_report, save_json_report, save_markdown_report

__version__ = "0.1.0"


class Atlas:
    """Evaluate registered data-quality metrics against tabular datasets."""

    def evaluate(self, dataset: str | Path | pd.DataFrame, metrics: list[str] | None = None,
                 reasoning: bool = False, model: str = "qwen3:32b", output: str | Path | None = None,
                 output_format: str = "json", **kwargs: Any) -> EvaluationReport:
        """Load a dataset, execute metrics, optionally reason, and save a report."""
        started = perf_counter()
        df = load_dataset(dataset)
        selected = metrics or list(list_metrics())
        results: dict[str, MetricResult] = {}
        reasoner = AtlasReasoner(model) if reasoning else None
        for name in selected:
            result = get_metric(name)().compute(df, **kwargs)
            if reasoner:
                reason, recommendations = reasoner.generate_reason(result.metric, result.score, result.details)
                if reason:
                    result.reasons.append(reason)
                result.recommendations.extend(recommendations)
            results[name] = result
        if reasoner and reasoner.last_error:
            warnings.warn(
                f"Ollama reasoning was unavailable; showing built-in reasons only. {reasoner.last_error}",
                RuntimeWarning,
                stacklevel=2,
            )
        dataset_name = dataset.name if isinstance(dataset, Path) else (str(dataset) if not isinstance(dataset, pd.DataFrame) else "DataFrame")
        report = build_report(dataset_name, df, results, perf_counter() - started)
        if output:
            self._save(report, output, output_format)
        return report

    @staticmethod
    def list_metrics() -> dict[str, type]:
        """Return registered metric classes keyed by name."""
        return list_metrics()

    @staticmethod
    def get_metric(name: str) -> type:
        """Return the registered metric class for a name."""
        return get_metric(name)

    @staticmethod
    def version() -> str:
        """Return the installed ATLAS version."""
        return __version__

    @staticmethod
    def _save(report: EvaluationReport, output: str | Path, output_format: str) -> None:
        writers = {"json": save_json_report, "markdown": save_markdown_report, "html": save_html_report}
        try:
            writers[output_format](report, output)
        except KeyError as error:
            raise ValueError("Report format must be json, markdown, or html.") from error
