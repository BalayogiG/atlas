"""Metric registration and lookup."""

from .base import Metric

_METRICS: dict[str, type[Metric]] = {}


def register_metric(name: str, metric_class: type[Metric]) -> None:
    """Register a metric class under a unique, non-empty name."""
    if not name or not isinstance(name, str):
        raise ValueError("Metric name must be a non-empty string.")
    if not issubclass(metric_class, Metric):
        raise TypeError("Registered metrics must inherit from Metric.")
    if name in _METRICS and _METRICS[name] is not metric_class:
        raise ValueError(f"Metric '{name}' is already registered.")
    _METRICS[name] = metric_class


def get_metric(name: str) -> type[Metric]:
    """Return a metric class by name."""
    try:
        return _METRICS[name]
    except KeyError as error:
        available = ", ".join(sorted(_METRICS)) or "none"
        raise KeyError(f"Unknown metric '{name}'. Available metrics: {available}.") from error


def list_metrics() -> dict[str, type[Metric]]:
    """Return a copy of the registered metric mapping."""
    return dict(_METRICS)
