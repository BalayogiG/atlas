"""Toxicity content-safety metric."""

import logging
import os
from typing import Any, Callable

import pandas as pd

from ...base import Metric, MetricResult
from ...decorators import metric

DEFAULT_SARVAM_URL = "https://api.sarvam.ai/translate"

logger = logging.getLogger(__name__)


@metric(
    name="Toxicity",
    category="Common",
    metric_id="DQ003",
    category_id="CAT001",
    description="Measures how much toxicity training data contain.",
)
class ToxicityMetric(Metric):
    """Translate free-text column to English, then score them for toxic content."""

    def compute(self, df: pd.DataFrame, **kwargs: Any) -> MetricResult:
        """Average per-response toxicity into an aggregate 0.0--1.0 quality score (1.0 = no toxicity)."""
        from dotenv import load_dotenv
        load_dotenv()

        text_field = kwargs.get("text_field", "response")
        source_lang = kwargs.get("source_lang", "auto")
        sarvam_mode = kwargs.get("sarvam_mode", "api")

        sarvam_api_key = kwargs.get(
            "sarvam_api_key",
            os.getenv("SARVAM_API_KEY"),
        )

        sarvam_url = kwargs.get(
            "sarvam_url",
            os.getenv("SARVAM_URL", DEFAULT_SARVAM_URL),
        )

        if sarvam_mode == "api" and not sarvam_api_key:
            raise ValueError(
                "SARVAM_API_KEY is not configured."
            )

        if text_field not in df.columns:
            raise KeyError(f"Column '{text_field}' not found in dataset columns: {list(df.columns)}")

        logger.debug("Computing toxicity for %d rows on column %r (sarvam_mode=%r, source_lang=%r)",
                     len(df), text_field, sarvam_mode, source_lang)

        translate: Callable[[str], str] | None = None
        model = None
        scores: list[float] = []
        missing_or_empty = 0
        failed = 0

        for index, value in df[text_field].items():
            text = self._to_text(value)
            if not text or not text.strip():
                missing_or_empty += 1
                continue
            if translate is None:
                logger.debug("Building translator and loading Detoxify model on first scorable row.")
                translate = self._build_translator(sarvam_mode, sarvam_api_key, source_lang, sarvam_url)
                model = self._load_model()
            try:
                scores.append(float(model.predict(translate(text))["toxicity"]))
            except Exception:
                logger.warning("Row %r could not be translated or scored and was excluded.", index, exc_info=True)
                failed += 1

        toxicity_rate = float(sum(scores) / len(scores)) if scores else 0.0
        details = {"text_field": text_field, "toxicity_rate": toxicity_rate,
                   "records_evaluated": len(df), "records_scored": len(scores),
                   "records_missing_or_empty": missing_or_empty, "records_failed": failed}
        reasons = []
        if len(df) and not scores:
            logger.warning("No rows in column %r had scorable text; toxicity could not be evaluated.", text_field)
            reasons.append("No responses had scorable text; toxicity could not be evaluated.")
        if toxicity_rate:
            reasons.append(f"{toxicity_rate:.1%} average toxicity detected across scored responses.")
        if failed:
            reasons.append(f"{failed} response(s) could not be translated or scored and were excluded.")
        recommendations = ["Review and moderate responses flagged as toxic."] if toxicity_rate else []
        score = round(1 - toxicity_rate, 4)
        logger.debug("Toxicity score: %s (%d scored, %d missing/empty, %d failed)",
                     score, len(scores), missing_or_empty, failed)
        return MetricResult(self.name, self.category, score, details, reasons, recommendations)

    @staticmethod
    def _to_text(value: Any) -> str | None:
        if isinstance(value, (list, tuple)):
            return " ".join(str(item) for item in value)
        if isinstance(value, dict):
            return " ".join(str(item) for item in value.values())
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return str(value)

    @staticmethod
    def _load_model() -> Any:
        from detoxify import Detoxify
        return Detoxify("original")

    @staticmethod
    def _build_translator(mode: str, api_key: str | None, source_lang: str, url: str) -> Callable[[str], str]:
        """Return a text -> English-text callable, backed by either the Sarvam SDK or its raw HTTP API."""
        if mode == "api":
            from sarvamai import SarvamAI
            client = SarvamAI(api_subscription_key=api_key)

            def translate_via_sdk(text: str) -> str:
                return client.text.translate(
                    input=text, source_language_code=source_lang, target_language_code="en-IN"
                ).translated_text

            return translate_via_sdk

        import requests

        def translate_via_url(text: str) -> str:
            response = requests.post(
                url,
                json={"input": text, "source_language_code": source_lang, "target_language_code": "en-IN"},
                headers={"api-subscription-key": api_key, "Content-Type": "application/json"},
                timeout=15,
            )
            response.raise_for_status()
            return response.json()["translated_text"]

        return translate_via_url
