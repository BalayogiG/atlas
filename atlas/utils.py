"""Reusable dataset, statistics, and report helpers."""

import logging
import re
import json
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from .base import EvaluationReport, MetricResult

logger = logging.getLogger(__name__)


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
    table.add_column("Reason", max_width=48)
    for result in report.metrics.values():
        reasoning = "\n".join(f"• {reason}" for reason in result.reasons) or "—"
        table.add_row(result.metric, result.category, f"{result.score:.4f}", reasoning)
    Console().print(table)


REQUIRED_SECTIONS = [
    "Dataset Details",
    "Dataset Description",
    "Dataset Sources",
    "Uses",
    "Dataset Structure",
    "Dataset Creation",
    "Annotations",
    "Bias, Risks, and Limitations",
    "Citation",
]


HUGGINGFACE_REPO_PATTERN = re.compile(r"^[\w.-]+/[\w.-]+$")
HUGGINGFACE_README_URL = "https://huggingface.co/datasets/{repo_id}/raw/main/README.md"


def _huggingface_repo_id(dataset_path: str) -> str | None:
    """
    Return the Hugging Face dataset repo id if dataset_path refers to one.

    Accepts either a bare repo id ("paulopontesm/titanic") or a full
    dataset URL ("https://huggingface.co/datasets/paulopontesm/titanic").
    """
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
    """Download a dataset card README.md from the Hugging Face Hub."""
    request = urllib.request.Request(
        HUGGINGFACE_README_URL.format(repo_id=repo_id),
        headers={"User-Agent": "atlas-dq-metrics"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.read().decode("utf-8", errors="ignore")
    except (urllib.error.URLError, urllib.error.HTTPError) as error:
        logger.warning("Failed to fetch Hugging Face dataset card for repo_id=%r: %s", repo_id, error)
        return ""


def load_documentation(dataset_path: str) -> str:
    """
    Load README/Data Card from a dataset directory or the Hugging Face Hub.

    Accepts:
        A local directory containing README.md, README.MD, DATASET_CARD.md,
        or datasheet.md
        A Hugging Face dataset repo id, e.g. "paulopontesm/titanic"
        A Hugging Face dataset URL, e.g.
        "https://huggingface.co/datasets/paulopontesm/titanic"
    """
    repo_id = _huggingface_repo_id(dataset_path)

    if repo_id:
        return _fetch_huggingface_readme(repo_id)

    path = Path(dataset_path)

    candidates = [
        "README.md",
        "README.MD",
        "DATASET_CARD.md",
        "datasheet.md",
    ]

    for file in candidates:
        doc = path / file
        if doc.exists():
            return doc.read_text(encoding="utf-8", errors="ignore")

    return ""


def extract_sections(documentation: str) -> dict[str, str]:
    """
    Extract Markdown headings and their content.

    Returns:
        {
            "Dataset Details": "...",
            "Uses": "...",
            ...
        }
    """
    sections = {}

    current = "ROOT"
    buffer = []

    for line in documentation.splitlines():

        match = re.match(r"^#+\s+(.*)", line)

        if match:

            sections[current] = "\n".join(buffer).strip()

            current = match.group(1).strip()

            buffer = []

        else:
            buffer.append(line)

    sections[current] = "\n".join(buffer).strip()

    return sections


def documentation_completeness(
    sections: dict[str, str],
    required_sections: list[str] | None = None,
) -> tuple[float, list[str], list[str]]:
    """
    Compute documentation completeness.

    Score = found / required
    """

    required_sections = required_sections or REQUIRED_SECTIONS

    found = []

    missing = []

    for section in required_sections:

        if section in sections and sections[section].strip():
            found.append(section)
        else:
            missing.append(section)

    score = len(found) / len(required_sections)

    return score, found, missing


def field_definition_coverage(
    df: pd.DataFrame,
    documentation: str,
) -> float:
    """
    Compute how many dataset columns are mentioned
    in the documentation.
    """

    if df.empty:
        return 1.0

    text = documentation.lower()

    documented = sum(
        column.lower() in text
        for column in df.columns
    )

    return documented / len(df.columns)


def keyword_score(
    text: str,
    keywords: list[str],
) -> float:
    """
    Simple keyword coverage score.
    """

    if not text.strip():
        return 0.0

    text = text.lower()

    matches = sum(
        keyword.lower() in text
        for keyword in keywords
    )

    return matches / len(keywords)


def get_section(
    sections: dict[str, str],
    *names: str,
) -> str:
    """
    Return the first matching section.
    """

    for name in names:
        if name in sections:
            return sections[name]

    return ""