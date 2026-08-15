"""Core ATLAS data models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import pandas as pd


@dataclass
class MetricResult:
    """The result returned by a dataset-quality metric."""

    metric: str
    category: str
    score: float
    details: dict[str, Any]
    reasons: list[str]
    recommendations: list[str]


@dataclass
class EvaluationReport:
    """The aggregate result of an ATLAS evaluation."""

    dataset_name: str
    rows: int
    columns: int
    metrics: dict[str, MetricResult]
    overall_score: float
    execution_time: float


@dataclass(frozen=True)
class AtlasSchema:
    """A declarative schema for validating tabular columns.

    Supported types are ``int``, ``float``, ``str``, ``category``, ``email``,
    and ``bool``. Additional constraints can be added without changing the
    evaluation API.
    """

    fields: dict[str, str]
    SUPPORTED_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"int", "float", "str", "category", "email", "bool"}
    )

    @classmethod
    def from_dict(cls, fields: dict[str, str]) -> "AtlasSchema":
        """Build a schema from a column-to-type mapping.

        Args:
            fields: Mapping of column names to supported type names.

        Raises:
            ValueError: If a declared type is unsupported.
        """
        aliases = {"integer": "int", "string": "str", "boolean": "bool"}
        if any(not isinstance(column, str) or not isinstance(kind, str) for column, kind in fields.items()):
            raise ValueError("Schema columns and types must be strings.")
        normalized = {column: aliases.get(kind.lower(), kind.lower()) for column, kind in fields.items()}
        invalid = sorted(set(normalized.values()) - cls.SUPPORTED_TYPES)
        if invalid:
            raise ValueError(f"Unsupported schema types: {', '.join(invalid)}.")
        return cls(normalized)


class Metric(ABC):
    """Contract implemented by every ATLAS metric."""

    @abstractmethod
    def compute(self, df: pd.DataFrame, **kwargs: Any) -> MetricResult:
        """Compute this metric for an in-memory dataset."""
