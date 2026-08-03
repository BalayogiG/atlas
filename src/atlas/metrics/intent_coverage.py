from __future__ import annotations

import re

from atlas.core import Metric, MetricResult, ValidationIssue
from atlas.core.context import Context
from atlas.decorators import metric

_DEFAULT_INTENTS = {
    "explain": ("explain", "explanation", "why", "how does", "how do"),
    "define": ("define", "definition", "what is", "meaning of"),
    "compare": ("compare", "difference", "versus", "vs", "contrast"),
    "solve": ("solve", "solution", "fix", "troubleshoot", "resolving"),
    "summarize": ("summarize", "summary", "brief", "overview"),
    "translate": ("translate", "translation", "convert", "in french", "in spanish"),
    "generate": ("generate", "create", "write", "draft", "compose"),
}


@metric
class IntentCoverage(Metric):
    """
    Measures how broadly common user intents are represented in the dataset.
    """

    id = "IC001"
    name = "Intent Coverage"
    description = "Measures how broadly common user intents are represented in the dataset."
    category = "Language Quality"

    def evaluate(self, context: Context) -> MetricResult:
        text_field = context.config.options.get("text_field", "prompt")
        intents = context.config.options.get("intents", list(_DEFAULT_INTENTS.keys()))
        detected_intents: set[str] = set()
        issues: list[ValidationIssue] = []

        for index, record in enumerate(context.dataset.records):
            value = record.get(text_field)

            if value is None:
                issues.append(
                    ValidationIssue(
                        record=index,
                        field=text_field,
                        reason="missing",
                    )
                )
                continue

            text = value.strip() if isinstance(value, str) else str(value).strip()
            if not text:
                issues.append(
                    ValidationIssue(
                        record=index,
                        field=text_field,
                        reason="empty",
                    )
                )
                continue

            lowered_text = text.lower()
            detected_intents.update(self._detect_intents(lowered_text, intents))

        coverage_total = len(intents)
        score = len(detected_intents) / coverage_total if coverage_total else 1.0
        if not context.dataset.records:
            score = 1.0

        return MetricResult(
            metric=self.name,
            score=round(score, 4),
            passed=score == 1.0,
            issues=issues,
            metadata={
                "text_field": text_field,
                "intents": intents,
                "detected_intents": sorted(detected_intents),
                "coverage": len(detected_intents),
                "coverage_total": coverage_total,
            },
        )

    def _detect_intents(self, text: str, intents: list[str]) -> set[str]:
        detected: set[str] = set()

        for intent in intents:
            keywords = _DEFAULT_INTENTS.get(intent, (intent,))
            if any(self._contains_phrase(text, keyword) for keyword in keywords):
                detected.add(intent)

        return detected

    @staticmethod
    def _contains_phrase(text: str, phrase: str) -> bool:
        pattern = re.escape(phrase.lower())
        return re.search(rf"\b{pattern}\b", text) is not None