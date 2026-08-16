"""Completeness quality metric."""

import logging
from typing import Any

import pandas as pd

from ..base import Metric, MetricResult
from ..decorators import metric
from ..utils import calculate_missing_rate

logger = logging.getLogger(__name__)


@metric(
    name="Completeness",
    category="Common",
    metric_id="DQ001",
    category_id="CAT001",
    description="Measures missing values, complete records, and required-field coverage.",
)
class CompletenessMetric(Metric):
    """Measure missingness and record completeness."""

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> MetricResult:
        """Compute completeness indicators and an aggregate 0.0--1.0 score."""
        required = kwargs.get("required_fields", [])
        logger.debug("Computing completeness for %d rows, %d columns (required_fields=%s)",
                     len(df), len(df.columns), required)
        missing_rate = calculate_missing_rate(df)
        complete_rate = float(df.notna().all(axis=1).mean()) if len(df) else 1.0
        by_column = df.isna().mean().sort_values(ascending=False)
        patterns = df.isna().astype(int).astype(str).agg("".join, axis=1).value_counts().head(5).to_dict()
        missing_required = [field for field in required if field not in df.columns]
        available_required = [field for field in required if field in df.columns]
        required_coverage = (float(df[available_required].notna().all(axis=1).mean())
                             if available_required else complete_rate)
        if missing_required:
            logger.warning("Required fields not present in dataset columns: %s", missing_required)
            required_coverage = 0.0
        imputed = sum(col.lower().endswith(("_imputed", "_filled")) for col in df.columns)
        imputation_dependence = imputed / len(df.columns) if len(df.columns) else 0.0
        details = {"missing_value_rate": missing_rate, "complete_record_rate": complete_rate,
                   "missingness_pattern": patterns, "required_field_coverage": required_coverage,
                   "imputation_dependence": imputation_dependence, "missing_required_fields": missing_required,
                   "missing_rate_by_column": by_column.to_dict()}
        reasons = [f"{missing_rate:.1%} of cells are missing."] if missing_rate else []
        recommendations = (["Review columns with high missing-value rates and define a treatment policy."]
                           if missing_rate else [])
        score = round(1 - missing_rate, 4)
        logger.debug("Completeness score: %s", score)
        return MetricResult(self.name, self.category, score, details, reasons, recommendations)
