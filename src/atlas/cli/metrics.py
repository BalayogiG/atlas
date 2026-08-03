"""Metric discovery commands."""

import typer
from rich.console import Console
from rich.table import Table

from atlas.registry import registry

console = Console()

app = typer.Typer(
    help="List available data-quality metrics.",
    invoke_without_command=True,
)

def _list_metrics() -> None:
    # Built-in metrics register themselves when their package is imported.
    import atlas.metrics

    table = Table(title="ATLAS - List of Evaluation Metrics", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="green")
    table.add_column("Category", style="green")
    table.add_column("Metric")
    table.add_column("Description")

    for metric in registry.list():
        m = registry.get(metric)
        table.add_row(m.id, m.category, m.name, m.description)

    console.print(table)


@app.callback()
def metrics(ctx: typer.Context) -> None:
    """List metrics when no subcommand is specified."""
    if ctx.invoked_subcommand is None:
        _list_metrics()


@app.command("list")
def list_metrics() -> None:
    """List all available metrics."""
    _list_metrics()
