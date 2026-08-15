"""Reusable dataset, statistics, and report helpers."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from .base import EvaluationReport, MetricResult


def load_dataset(dataset: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a supported dataset into a DataFrame, copying DataFrame inputs."""
    if isinstance(dataset, pd.DataFrame):
        return dataset.copy()
    path = Path(dataset).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    readers = {".csv": pd.read_csv, ".json": pd.read_json, ".parquet": pd.read_parquet,
               ".xlsx": pd.read_excel, ".xls": pd.read_excel}
    try:
        return readers[path.suffix.lower()](path)
    except KeyError as error:
        raise ValueError("Supported dataset types: CSV, Excel, JSON, and Parquet.") from error


def calculate_missing_rate(df: pd.DataFrame) -> float:
    """Return the fraction of cells that are missing."""
    return float(df.isna().mean().mean()) if df.size else 0.0


def calculate_duplicate_rate(df: pd.DataFrame) -> float:
    """Return the fraction of rows duplicated after their first occurrence."""
    return float(df.duplicated().mean()) if len(df) else 0.0


def detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return names of numeric DataFrame columns."""
    return df.select_dtypes(include="number").columns.tolist()


def detect_text_columns(df: pd.DataFrame) -> list[str]:
    """Return names of free-text (object/string dtype) DataFrame columns."""
    return df.select_dtypes(include=["object", "string"]).columns.tolist()


def build_report(dataset_name: str, df: pd.DataFrame, metrics: dict[str, MetricResult],
                 execution_time: float) -> EvaluationReport:
    """Build an evaluation report from completed metric results."""
    scores = [result.score for result in metrics.values()]
    return EvaluationReport(dataset_name, len(df), len(df.columns), metrics,
                            float(sum(scores) / len(scores)) if scores else 0.0, execution_time)


def _report_dict(report: EvaluationReport) -> dict[str, Any]:
    return asdict(report)


def save_json_report(report: EvaluationReport, output: str | Path) -> None:
    """Save a report as formatted JSON."""
    Path(output).write_text(json.dumps(_report_dict(report), indent=2, default=str), encoding="utf-8")


def save_markdown_report(report: EvaluationReport, output: str | Path) -> None:
    """Save a concise, human-readable Markdown report."""
    lines = ["# ATLAS Evaluation Report", "", f"- Dataset: {report.dataset_name}",
             f"- Shape: {report.rows} rows × {report.columns} columns",
             f"- Overall score: {report.overall_score:.4f}", "", "## Metrics", "",
             "| Metric | Category | Score |", "| --- | --- | ---: |"]
    lines.extend(f"| {item.metric} | {item.category} | {item.score:.4f} |" for item in report.metrics.values())
    Path(output).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_html_report(report: EvaluationReport, output: str | Path) -> None:
    """Save a self-contained HTML report."""
    rows = "".join(f"<tr><td>{item.metric}</td><td>{item.category}</td><td>{item.score:.4f}</td></tr>"
                   for item in report.metrics.values())
    html = ("<html><body><h1>ATLAS Evaluation Report</h1>"
            f"<p>Dataset: {report.dataset_name}<br>Overall score: {report.overall_score:.4f}</p>"
            "<table><tr><th>Metric</th><th>Category</th><th>Score</th></tr>" + rows + "</table></body></html>")
    Path(output).write_text(html, encoding="utf-8")


def print_summary(report: EvaluationReport) -> None:
    """Print a compact Rich table summarizing an evaluation."""
    table = Table(
        title=f"ATLAS Evaluation: {report.dataset_name}",
        caption=f"Overall quality score: {report.overall_score:.4f}",
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Score", justify="right")
    table.add_column("Reasoning", max_width=48)
    for result in report.metrics.values():
        reasoning = "\n".join(f"• {reason}" for reason in result.reasons) or "—"
        table.add_row(result.metric, result.category, f"{result.score:.4f}", reasoning)
    Console().print(table)
