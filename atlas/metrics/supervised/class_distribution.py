"""Label Balance / Class Distribution metric."""

from typing import Any

import pandas as pd

from ...base import Metric, MetricResult
from ...decorators import metric


@metric(
    name="Class Distribution",
    category="Supervised",
    metric_id="SDQ001",
    category_id="CAT002",
    description="Evaluates how evenly labels are distributed and whether "
                "minority or rare classes are sufficiently represented."
)
class ClassDistribution(Metric):
    """
    Computes class distribution metrics.

    Required Columns
    ----------------
    label : str
        Target class / label.

    Optional Columns
    ----------------
    group : str
        Demographic or grouping attribute.

    is_valid : bool
        Whether the sample remains after cleaning.
    """

    def __init__(
        self,
        label_column: str = "label",
        group_column: str | None = None,
        valid_column: str | None = None,
        minimum_target: int = 100,
    ):
        self.label_column = label_column
        self.group_column = group_column
        self.valid_column = valid_column
        self.minimum_target = minimum_target

    def compute(self, data: pd.DataFrame, **kwargs: Any) -> MetricResult:

        if self.label_column not in data.columns:
            raise ValueError(
                f"Missing required column '{self.label_column}'."
            )

        labels = data[self.label_column].dropna()

        counts = labels.value_counts()

        total = counts.sum()

        # ----------------------------------------------------
        # 1. Class Imbalance Ratio
        # ----------------------------------------------------
        imbalance_ratio = (
            counts.max() / counts.min()
            if len(counts) > 1
            else 1.0
        )

        # ----------------------------------------------------
        # 2. Minority-Class Share
        # ----------------------------------------------------
        minority_share = counts.min() / total

        # ----------------------------------------------------
        # 3. Per-Group Label Skew
        # ----------------------------------------------------
        group_skew = None

        if (
            self.group_column is not None
            and self.group_column in data.columns
        ):

            distribution = pd.crosstab(
                data[self.group_column],
                data[self.label_column],
                normalize="index",
            )

            group_skew = (
                distribution.max() - distribution.min()
            ).max()

        # ----------------------------------------------------
        # 4. Effective Samples per Class
        # ----------------------------------------------------
        if (
            self.valid_column is not None
            and self.valid_column in data.columns
        ):

            effective_counts = (
                data[data[self.valid_column]]
                .groupby(self.label_column)
                .size()
            )

        else:
            effective_counts = counts

        effective_ratio = (
            effective_counts / self.minimum_target
        ).to_dict()

        # ----------------------------------------------------
        # 5. Tail-Class Coverage
        # ----------------------------------------------------
        tail_coverage = (
            (
                effective_counts >= self.minimum_target
            ).sum()
            / len(effective_counts)
        )

        # ----------------------------------------------------
        # Overall Score
        # ----------------------------------------------------
        imbalance_score = min(
            1.0,
            1 / imbalance_ratio,
        )

        score = (
            imbalance_score
            + minority_share
            + tail_coverage
        ) / 3

        details = {
            "class_counts": counts.to_dict(),
            "class_imbalance_ratio": imbalance_ratio,
            "minority_class_share": minority_share,
            "per_group_label_skew": group_skew,
            "effective_samples_per_class": effective_counts.to_dict(),
            "effective_sample_ratio": effective_ratio,
            "tail_class_coverage": tail_coverage,
        }

        return MetricResult(
            score=round(score, 4),
            details=details,
        )