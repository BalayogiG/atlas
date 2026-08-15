"""Noise and validity quality metric."""

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest

from ..base import AtlasSchema, Metric, MetricResult
from ..decorators import metric
from ..utils import calculate_duplicate_rate, detect_numeric_columns, detect_text_columns


@metric(
    name="Data Quality & Noise",
    category="Common",
    metric_id="DQ002",
    category_id="CAT001",
    description="Measures duplicates, statistical outliers, schema conformance, and consistency.",
)
class DataQualityNoiseMetric(Metric):
    """Measure duplicate rows, numeric outliers, and basic schema validity."""

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> MetricResult:
        """Compute noise indicators and an aggregate 0.0--1.0 score."""
        duplicate_rate = calculate_duplicate_rate(df)
        numeric = detect_numeric_columns(df)
        text = detect_text_columns(df)
        outlier_rate = self._statistical_outlier_rate(df[numeric] if numeric else pd.DataFrame())
        text_anomaly_rate = self._statistical_outlier_rate(self._text_features(df, text))
        schema_conformance, schema_details = self._schema_conformance(df, kwargs.get("schema"))
        consistency = self._consistency(df, kwargs.get("consistency_rules", {}))
        error_rate = float(np.mean([duplicate_rate, outlier_rate, text_anomaly_rate,
                                    1 - schema_conformance, 1 - consistency]))
        details = {"error_rate": error_rate, "duplicate_rate": duplicate_rate, "outlier_rate": outlier_rate,
                   "text_anomaly_rate": text_anomaly_rate, "schema_conformance": schema_conformance,
                   "schema_validation": schema_details, "cross_field_consistency": consistency,
                   "numeric_columns": numeric, "text_columns": text}
        reasons = [f"{duplicate_rate:.1%} of rows are duplicate."] if duplicate_rate else []
        if outlier_rate:
            reasons.append(f"{outlier_rate:.1%} of numeric values are statistical outliers.")
        if text_anomaly_rate:
            reasons.append(f"{text_anomaly_rate:.1%} of text values have an anomalous length or character mix.")
        recommendations = (["Deduplicate records and review anomalous numeric and text values."]
                           if reasons else [])
        return MetricResult(self.name, self.category, round(1 - error_rate, 4), details,
                            reasons, recommendations)

    @staticmethod
    def _statistical_outlier_rate(values: pd.DataFrame) -> float:
        """Rate of z-score/Isolation Forest outliers in a purely numeric feature matrix."""
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return 0.0
        z_scores = np.abs(zscore(values, nan_policy="omit"))
        statistical_rate = float(np.nanmean(z_scores > 3))
        if len(values) < 10:
            return statistical_rate
        model = IsolationForest(contamination="auto", random_state=0)
        isolation_rate = float(np.mean(model.fit_predict(values) == -1))
        return max(statistical_rate, isolation_rate)

    @staticmethod
    def _text_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """Derive length/character-mix features per text column for outlier scoring."""
        if not columns:
            return pd.DataFrame()
        features = {}
        for column in columns:
            text = df[column].dropna().astype(str)
            text = text[text.str.len() > 0]
            if text.empty:
                continue
            features[f"{column}__length"] = text.str.len()
            features[f"{column}__digit_ratio"] = text.apply(lambda v: sum(c.isdigit() for c in v) / len(v))
            features[f"{column}__punct_ratio"] = text.apply(
                lambda v: sum(not c.isalnum() and not c.isspace() for c in v) / len(v)
            )
        return pd.DataFrame(features)

    @staticmethod
    def _schema_conformance(df: pd.DataFrame, schema: AtlasSchema | dict[str, str] | None) -> tuple[float, dict[str, Any]]:
        """Validate against an explicit schema or one inferred from the dataset."""
        explicit = schema is not None
        resolved = schema if isinstance(schema, AtlasSchema) else AtlasSchema.from_dict(schema or DataQualityNoiseMetric._infer_schema(df))
        valid, total = 0, 0
        violations: dict[str, dict[str, Any]] = {}
        for column, expected in resolved.fields.items():
            if column not in df:
                violations[column] = {"expected": expected, "invalid_count": len(df), "examples": ["column missing"]}
                total += len(df)
                continue
            values = df[column].dropna()
            matches = values.map(lambda value: DataQualityNoiseMetric._matches_type(value, expected))
            invalid = values[~matches]
            valid += int(matches.sum())
            total += len(values)
            if len(invalid):
                violations[column] = {"expected": expected, "invalid_count": len(invalid),
                                      "examples": [str(value) for value in invalid.head(3)]}
        conformance = valid / total if total else 1.0
        return conformance, {"mode": "user_defined" if explicit else "inferred",
                             "schema": resolved.fields, "violations": violations}

    @staticmethod
    def _infer_schema(df: pd.DataFrame) -> dict[str, str]:
        """Infer useful semantic types while tolerating a small amount of noise."""
        inferred: dict[str, str] = {}
        email_pattern = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
        for column in df.columns:
            values = df[column].dropna()
            if pd.api.types.is_bool_dtype(df[column]):
                inferred[column] = "bool"
            elif pd.api.types.is_integer_dtype(df[column]):
                inferred[column] = "int"
            elif pd.api.types.is_float_dtype(df[column]):
                inferred[column] = "float"
            elif values.empty:
                inferred[column] = "str"
            else:
                text = values.astype(str).str.strip()
                numeric = pd.to_numeric(text, errors="coerce")
                numeric_ratio = float(numeric.notna().mean())
                if numeric_ratio >= 0.9:
                    inferred[column] = "int" if bool(np.allclose(numeric.dropna() % 1, 0)) else "float"
                elif float(text.map(lambda value: bool(email_pattern.fullmatch(value))).mean()) >= 0.9:
                    inferred[column] = "email"
                elif values.nunique() <= min(20, max(2, len(values) * 0.1)):
                    inferred[column] = "category"
                else:
                    inferred[column] = "str"
        return inferred

    @staticmethod
    def _matches_type(value: Any, expected: str) -> bool:
        """Return whether one non-null value conforms to a semantic type."""
        text = str(value).strip()
        if expected == "str" or expected == "category":
            return True
        if expected == "email":
            return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", text))
        if expected == "bool":
            return text.lower() in {"true", "false", "0", "1"}
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return False
        return bool(float(numeric).is_integer()) if expected == "int" else True

    @staticmethod
    def _consistency(df: pd.DataFrame, rules: dict[str, Any]) -> float:
        if not rules:
            return 1.0
        outcomes = []
        for name, rule in rules.items():
            try:
                result = rule(df) if callable(rule) else df.eval(rule)
                outcomes.append(float(pd.Series(result).mean()))
            except Exception:
                outcomes.append(0.0)
        return float(np.mean(outcomes)) if outcomes else 1.0
