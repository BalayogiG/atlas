from __future__ import annotations

from atlas.core import Metric, MetricResult, ValidationIssue
from atlas.core.context import Context
from atlas.decorators import metric


@metric
class MultilingualCoverage(Metric):
    """
    Measures how much of the dataset is available in the target languages.
    """

    id = "MC001"
    name = "Multilingual Coverage"
    description = "Measures how much of the dataset is available in the target languages."
    category = "Language Quality"

    def evaluate(self, context: Context) -> MetricResult:
        language_field = context.config.options.get("language_field", "language")
        target_languages = [language.lower() for language in context.config.options.get("target_languages", ["en"])]
        issues: list[ValidationIssue] = []
        matching_records = 0

        for index, record in enumerate(context.dataset.records):
            value = record.get(language_field)

            if value is None:
                issues.append(
                    ValidationIssue(
                        record=index,
                        field=language_field,
                        reason="missing",
                    )
                )
                continue

            language = str(value).strip().lower()
            if not language:
                issues.append(
                    ValidationIssue(
                        record=index,
                        field=language_field,
                        reason="empty",
                    )
                )
                continue

            if language in target_languages:
                matching_records += 1

        total = len(context.dataset.records)
        score = matching_records / total if total else 1.0

        return MetricResult(
            metric=self.name,
            score=round(score, 4),
            passed=score == 1.0,
            issues=issues,
            metadata={
                "language_field": language_field,
                "target_languages": target_languages,
                "matching_records": matching_records,
                "total_records": total,
            },
        )