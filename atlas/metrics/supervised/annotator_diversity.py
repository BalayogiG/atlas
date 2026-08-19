"""Annotator Diversity & Bias quality metric."""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from ...base import Metric, MetricResult
from ...decorators import metric

logger = logging.getLogger(__name__)


@metric(
    name="Annotator Diversity & Bias",
    category="Supervised",
    metric_id="SDQ02",
    category_id="CAT002",
    description="Measures annotator diversity, labeling consistency across demographic groups, "
                "annotator workload balance, and demographic bias.",
)
class AnnotatorDiversityBiasMetric(Metric):
    """Measure annotator diversity and potential demographic bias."""

    DEFAULT_DEMOGRAPHIC_COLUMNS = [
        "gender",
        "age_group",
        "region",
        "language",
        "education",
        "experience",
        "occupation",
        "ethnicity",
    ]

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> MetricResult:
        """Compute annotator diversity and bias metrics."""

        annotator_col = kwargs.get("annotator_column", "annotator_id")
        label_col = kwargs.get("label_column", "label")
        demographic_columns = kwargs.get(
            "demographic_columns",
            self.DEFAULT_DEMOGRAPHIC_COLUMNS,
        )

        if annotator_col not in df.columns:
            raise ValueError(f"Missing required column '{annotator_col}'.")

        if label_col not in df.columns:
            raise ValueError(f"Missing required column '{label_col}'.")

        available_demographics = [
            c for c in demographic_columns if c in df.columns
        ]

        logger.debug(
            "Computing Annotator Diversity & Bias using demographics=%s",
            available_demographics,
        )

        # ----------------------------------------------------
        # 1. Demographic Spread (Normalized Shannon Diversity)
        # ----------------------------------------------------

        spread_scores = {}

        for col in available_demographics:
            values = (
                df[[annotator_col, col]]
                .drop_duplicates()
                .dropna()[col]
            )

            if len(values) == 0:
                continue

            probs = values.value_counts(normalize=True)

            if len(probs) == 1:
                spread_scores[col] = 0.0
                continue

            h = -(probs * np.log(probs)).sum()
            h /= np.log(len(probs))
            spread_scores[col] = float(h)

        demographic_spread = (
            float(np.mean(list(spread_scores.values())))
            if spread_scores
            else None
        )

        # ----------------------------------------------------
        # 2. Label Divergence
        # ----------------------------------------------------

        divergence_scores = {}

        for col in available_demographics:

            jsd_values = []

            table = (
                df.groupby([col, label_col])
                .size()
                .unstack(fill_value=0)
            )

            if len(table) < 2:
                continue

            probs = table.div(table.sum(axis=1), axis=0)

            groups = probs.index.tolist()

            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    d = jensenshannon(
                        probs.loc[groups[i]],
                        probs.loc[groups[j]],
                    )
                    jsd_values.append(d)

            if jsd_values:
                divergence_scores[col] = float(1 - np.mean(jsd_values))

        label_consistency = (
            float(np.mean(list(divergence_scores.values())))
            if divergence_scores
            else None
        )

        # ----------------------------------------------------
        # 3. Annotator Load Balance
        # ----------------------------------------------------

        counts = df[annotator_col].value_counts().values

        if len(counts):

            counts = np.sort(counts)

            n = len(counts)
            cumulative = np.cumsum(counts)

            gini = (
                (n + 1 - 2 * cumulative.sum() / cumulative[-1])
                / n
            )

            load_balance = float(1 - gini)

        else:

            load_balance = None

        # ----------------------------------------------------
        # 4. Overall Score
        # ----------------------------------------------------

        scores = [
            s
            for s in [
                demographic_spread,
                label_consistency,
                load_balance,
            ]
            if s is not None
        ]

        overall_score = round(float(np.mean(scores)), 4) if scores else 0.0

        details = {
            "available_demographics": available_demographics,
            "demographic_spread": demographic_spread,
            "label_consistency": label_consistency,
            "annotator_load_balance": load_balance,
            "per_demographic_spread": spread_scores,
            "per_demographic_consistency": divergence_scores,
            "annotator_count": int(df[annotator_col].nunique()),
        }

        reasons = []

        if demographic_spread is not None and demographic_spread < 0.7:
            reasons.append(
                "Annotator demographics show limited diversity."
            )

        if label_consistency is not None and label_consistency < 0.8:
            reasons.append(
                "Different demographic groups assign labels differently."
            )

        if load_balance is not None and load_balance < 0.8:
            reasons.append(
                "A small number of annotators dominate the labeling workload."
            )

        recommendations = []

        if demographic_spread is not None and demographic_spread < 0.7:
            recommendations.append(
                "Recruit annotators from more diverse demographic backgrounds."
            )

        if label_consistency is not None and label_consistency < 0.8:
            recommendations.append(
                "Review annotation guidelines and perform inter-group agreement analysis."
            )

        if load_balance is not None and load_balance < 0.8:
            recommendations.append(
                "Distribute annotation tasks more evenly across annotators."
            )

        logger.debug("Annotator Diversity & Bias score: %.4f", overall_score)

        return MetricResult(
            self.name,
            self.category,
            overall_score,
            details,
            reasons,
            recommendations,
        )