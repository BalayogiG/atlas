"""Documentation & Transparency quality metric."""

import logging
from typing import Any

import pandas as pd

from ...base import Metric, MetricResult
from ...decorators import metric
from ...utils import (
    load_documentation,
    extract_sections,
    documentation_completeness,
    field_definition_coverage,
    keyword_score,
    get_section,
)

logger = logging.getLogger(__name__)

@metric(
    name="Documentation & Transparency",
    category="Common",
    metric_id="DQ004",
    category_id="CAT001",
    description="Measures the documentation completeness and transparency of dataset documentation.",
)
class DocumentationTransparencyMetric(Metric):
    """Measure documentation quality."""

    def compute(
        self,
        df: pd.DataFrame,
        **kwargs: Any,
    ) -> MetricResult:
        """
        Compute documentation transparency score.

        Required kwargs:
            dataset_path="path/to/dataset"
                Local dataset directory, a Hugging Face dataset repo id
                (e.g. "paulopontesm/titanic"), or a Hugging Face dataset
                URL (e.g. "https://huggingface.co/datasets/paulopontesm/titanic")
        """

        dataset_path = kwargs.get("dataset_path")

        if not dataset_path:
            raise ValueError(
                "'dataset_path' must be provided."
            )

        logger.debug("Loading documentation from dataset_path=%r", dataset_path)

        documentation = load_documentation(dataset_path)

        if not documentation:

            logger.warning(
                "No documentation found for dataset_path=%r (missing local file, "
                "unresolvable Hugging Face repo, or a failed Hub fetch).",
                dataset_path,
            )

            return MetricResult(
                self.name,
                self.category,
                0.0,
                {
                    "documentation_found": False,
                },
                ["No dataset documentation found."],
                [
                    "Add a README.md or Dataset Card describing the dataset."
                ],
            )

        sections = extract_sections(documentation)

        completeness, found, missing = documentation_completeness(
            sections
        )

        field_coverage = field_definition_coverage(
            df,
            documentation,
        )

        intended_use = keyword_score(
            get_section(
                sections,
                "Uses",
                "Direct Use",
                "Out-of-Scope Use",
            ),
            [
                "purpose",
                "use",
                "user",
                "model",
                "application",
            ],
        )

        limitations = keyword_score(
            get_section(
                sections,
                "Bias, Risks, and Limitations",
                "Limitations",
            ),
            [
                "bias",
                "risk",
                "limitation",
                "missing",
                "fairness",
            ],
        )

        reproducibility = keyword_score(
            documentation,
            [
                "version",
                "license",
                "source",
                "collection",
                "preprocessing",
            ],
        )

        score = (
            0.25 * completeness
            + 0.20 * intended_use
            + 0.20 * field_coverage
            + 0.20 * limitations
            + 0.15 * reproducibility
        )

        details = {
            "documentation_completeness": round(completeness, 4),
            "intended_use_clarity": round(intended_use, 4),
            "field_definition_coverage": round(field_coverage, 4),
            "limitations_disclosure": round(limitations, 4),
            "reproducibility_notes": round(reproducibility, 4),
            "found_sections": found,
            "missing_sections": missing,
        }

        reasons = [
            f"{len(found)} of {len(found) + len(missing)} required documentation sections were found."
        ]

        recommendations = []

        if missing:
            recommendations.append(
                "Add the missing documentation sections."
            )

        if field_coverage < 1:
            recommendations.append(
                "Document every dataset field with a description."
            )

        if reproducibility < 1:
            recommendations.append(
                "Include version, source, collection, preprocessing, and licensing information."
            )

        rounded_score = round(score, 4)
        logger.debug("Documentation & Transparency score: %s (%d/%d sections found)",
                     rounded_score, len(found), len(found) + len(missing))

        return MetricResult(
            self.name,
            self.category,
            rounded_score,
            details,
            reasons,
            recommendations,
        )