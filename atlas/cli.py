"""Command-line interface for ATLAS."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .atlas import Atlas
from .utils import print_summary

app = typer.Typer(help="Evaluate tabular dataset quality with ATLAS.", no_args_is_help=True)


@app.command()
def evaluate(
    dataset: Annotated[Path, typer.Option(help="CSV, Excel, JSON, or Parquet dataset.")],
    metric: Annotated[list[str] | None, typer.Option("--metric", help="Metric to run; repeatable.")] = None,
    reasoning: Annotated[bool, typer.Option(help="Generate optional Ollama explanations.")] = False,
    model: Annotated[str, typer.Option(help="Ollama model name.")] = "qwen3:32b",
    output: Annotated[Path | None, typer.Option(help="Path for a saved report.")] = None,
    format: Annotated[str, typer.Option("--format", help="json, markdown, or html.")] = "json",
) -> None:
    """Evaluate one dataset and optionally save a report."""
    try:
        report = Atlas().evaluate(dataset, metric, reasoning, model, output, format)
        print_summary(report)
    except (FileNotFoundError, KeyError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


@app.command("list-metrics")
def list_metrics_command() -> None:
    """List registered metrics."""
    table = Table(title="Available ATLAS Metrics")
    table.add_column("Metric ID", style="cyan", no_wrap=True)
    table.add_column("Category ID", style="cyan")
    table.add_column("Category Name", style="green")
    table.add_column("Metric Name", style="green")
    table.add_column("Metric Description")
    for metric_class in sorted(Atlas.list_metrics().values(), key=lambda item: item.metric_id):
        table.add_row(
            metric_class.metric_id,
            metric_class.category_id,
            metric_class.category,
            metric_class.name,
            metric_class.description,
        )
    Console().print(table)


@app.command("get-metric")
def get_metric_command(name: Annotated[str, typer.Option(help="Registered metric name.")]) -> None:
    """Show metadata for a registered metric."""
    try:
        metric_class = Atlas.get_metric(name)
        typer.echo(
            f"metric_id: {metric_class.metric_id}\ncategory_id: {metric_class.category_id}\n"
            f"name: {metric_class.name}\ncategory: {metric_class.category}\n"
            f"description: {metric_class.description}\nclass: {metric_class.__name__}"
        )
    except KeyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


@app.command()
def version() -> None:
    """Print the ATLAS version."""
    typer.echo(Atlas.version())


if __name__ == "__main__":
    app()
