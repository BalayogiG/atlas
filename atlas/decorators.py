"""Metric registration decorator."""

from collections.abc import Callable

from .base import Metric
from .registry import register_metric


def metric(name: str, category: str, *, metric_id: str | None = None,
           category_id: str | None = None, description: str = "") -> Callable[[type[Metric]], type[Metric]]:
    """Register a metric class and attach its public metadata.

    Args:
        name: Human-readable metric name.
        category: Human-readable category name.
        metric_id: Stable metric identifier. Defaults to a slug of ``name``.
        category_id: Stable category identifier. Defaults to a slug of ``category``.
        description: Brief explanation of what the metric evaluates.
    """
    def decorator(metric_class: type[Metric]) -> type[Metric]:
        resolved_metric_id = metric_id or name.lower().replace(" ", "_")
        metric_class.name = name
        metric_class.category = category
        metric_class.metric_id = resolved_metric_id
        metric_class.category_id = category_id or category.lower().replace(" ", "_")
        metric_class.description = description
        register_metric(resolved_metric_id, metric_class)
        return metric_class
    return decorator
