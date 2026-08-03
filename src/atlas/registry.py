# src/atlas/registry.py

from __future__ import annotations

from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from atlas.core.metric import Metric


class MetricRegistry:

    def __init__(self):
        self._metrics: dict[str, Type[Metric]] = {}

    def register(self, metric: Type[Metric]) -> None:
        self._metrics[metric.name.lower()] = metric

    def get(self, name: str) -> Type[Metric]:
        return self._metrics[name.lower()]

    def list(self) -> list[str]:
        return sorted(self._metrics.keys())

    def all(self) -> dict[str, Type[Metric]]:
        return self._metrics.copy()


registry = MetricRegistry()
