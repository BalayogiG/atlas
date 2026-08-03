from __future__ import annotations

import re

from atlas.core import Metric, MetricResult, ValidationIssue
from atlas.core.context import Context
from atlas.decorators import metric

_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
_REPEAT_PATTERN = re.compile(r"\b(\w+)(?:\s+\1\b)+", re.IGNORECASE)
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")


@metric
class ConversationalLanguageQuality(Metric):
    """
    Measures how naturally and readably chatbot responses are written.
    """

    id = "LQ001"
    name = "Conversational Language Quality"
    description = "Measures how naturally and readably chatbot responses are written."
    category = "Language Quality"

    def evaluate(self, context: Context) -> MetricResult:
        text_field = context.config.options.get("text_field", "response")
        issues: list[ValidationIssue] = []
        record_scores: list[float] = []

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

            if isinstance(value, str):
                text = value.strip()
                if text == "":
                    issues.append(
                        ValidationIssue(
                            record=index,
                            field=text_field,
                            reason="empty",
                        )
                    )
                    continue
            else:
                text = str(value).strip()
                if text == "":
                    issues.append(
                        ValidationIssue(
                            record=index,
                            field=text_field,
                            reason="empty",
                        )
                    )
                    continue

            record_scores.append(self._score_text(text))

        score = sum(record_scores) / len(record_scores) if record_scores else 0.0
        if not context.dataset.records:
            score = 1.0

        return MetricResult(
            metric=self.name,
            score=round(score, 4),
            passed=score == 1.0,
            issues=issues,
            metadata={
                "text_field": text_field,
                "records_evaluated": len(context.dataset.records),
                "records_with_text": len(record_scores),
                "records_missing_or_empty": len(issues),
            },
        )

    def _score_text(self, text: str) -> float:
        words = _WORD_PATTERN.findall(text)
        word_count = len(words)

        if word_count == 0:
            return 0.0

        sentence_count = max(1, len([segment for segment in _SENTENCE_SPLIT_PATTERN.split(text) if segment.strip()]))
        avg_sentence_length = word_count / sentence_count
        avg_word_length = sum(len(word) for word in words) / word_count
        uppercase_ratio = sum(1 for word in words if word.isupper() and len(word) > 1) / word_count
        repeated_words = bool(_REPEAT_PATTERN.search(text))
        terminal_punctuation = 1.0 if text.rstrip().endswith((".", "!", "?")) else 0.0

        length_score = self._bell_score(word_count, target=18.0, tolerance=18.0)
        flow_score = self._bell_score(avg_sentence_length, target=14.0, tolerance=14.0)
        readability_score = 1.0 - min(abs(avg_word_length - 4.8) / 4.8, 1.0)
        structure_score = 0.5 + (0.5 * terminal_punctuation)

        score = (
            0.3 * length_score
            + 0.3 * flow_score
            + 0.2 * readability_score
            + 0.2 * structure_score
        )

        if repeated_words:
            score -= 0.15

        if uppercase_ratio > 0.35:
            score -= 0.2

        if word_count < 5:
            score -= 0.15

        return max(0.0, min(score, 1.0))

    @staticmethod
    def _bell_score(value: float, target: float, tolerance: float) -> float:
        distance = abs(value - target)
        return max(0.0, 1.0 - min(distance / tolerance, 1.0))