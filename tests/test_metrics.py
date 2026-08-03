from pathlib import Path

from atlas.core import Context, Dataset
from atlas.core.config import MetricConfig
from atlas.metrics import ConversationalLanguageQuality
from atlas.metrics import IntentCoverage
from atlas.metrics import MultilingualCoverage
from atlas.registry import registry


def test_conversational_language_quality_scores_chatty_responses_higher() -> None:
    metric = ConversationalLanguageQuality()

    good_report = metric.evaluate(
        Context(
            dataset=Dataset(
                path=Path("good.json"),
                records=[
                    {
                        "response": "Sure, I can help with that. Here is a clear summary you can use right away.",
                    }
                ],
            ),
            config=MetricConfig(options={"text_field": "response"}),
        )
    )

    bad_report = metric.evaluate(
        Context(
            dataset=Dataset(
                path=Path("bad.json"),
                records=[
                    {
                        "response": "okay",
                    }
                ],
            ),
            config=MetricConfig(options={"text_field": "response"}),
        )
    )

    assert good_report.score > bad_report.score
    assert good_report.issues == []
    assert bad_report.score < 0.5


def test_conversational_language_quality_is_registered() -> None:
    assert registry.get("conversational language quality") is ConversationalLanguageQuality


def test_intent_coverage_scores_broader_intent_mix_higher() -> None:
    metric = IntentCoverage()

    broad_report = metric.evaluate(
        Context(
            dataset=Dataset(
                path=Path("broad.json"),
                records=[
                    {"prompt": "Explain quantum computing in simple terms."},
                    {"prompt": "Compare cats versus dogs for a first pet."},
                    {"prompt": "Solve this error in my code."},
                    {"prompt": "Translate this sentence into Spanish."},
                ],
            ),
            config=MetricConfig(options={"text_field": "prompt"}),
        )
    )

    narrow_report = metric.evaluate(
        Context(
            dataset=Dataset(
                path=Path("narrow.json"),
                records=[
                    {"prompt": "Explain the concept of recursion."},
                    {"prompt": "Explain recursion with an example."},
                ],
            ),
            config=MetricConfig(options={"text_field": "prompt"}),
        )
    )

    assert broad_report.score > narrow_report.score
    assert broad_report.metadata["coverage"] > narrow_report.metadata["coverage"]


def test_intent_coverage_is_registered() -> None:
    assert registry.get("intent coverage") is IntentCoverage


def test_multilingual_coverage_scores_target_languages_higher() -> None:
    metric = MultilingualCoverage()

    broad_report = metric.evaluate(
        Context(
            dataset=Dataset(
                path=Path("broad.json"),
                records=[
                    {"language": "en"},
                    {"language": "es"},
                    {"language": "fr"},
                ],
            ),
            config=MetricConfig(options={"target_languages": ["en", "es", "fr"]}),
        )
    )

    narrow_report = metric.evaluate(
        Context(
            dataset=Dataset(
                path=Path("narrow.json"),
                records=[
                    {"language": "en"},
                    {"language": "en"},
                    {"language": "de"},
                ],
            ),
            config=MetricConfig(options={"target_languages": ["en", "es", "fr"]}),
        )
    )

    assert broad_report.score > narrow_report.score
    assert broad_report.metadata["matching_records"] == 3
    assert narrow_report.metadata["matching_records"] == 2


def test_multilingual_coverage_is_registered() -> None:
    assert registry.get("multilingual coverage") is MultilingualCoverage