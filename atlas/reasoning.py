"""Optional local-model reasoning for metric results."""

from typing import Any


class AtlasReasoner:
    """Generate concise explanations through an optional Ollama model."""

    def __init__(self, model: str = "qwen3:32b") -> None:
        self.model = model
        self.last_error: str = ""

    def generate_reason(self, metric_name: str, score: float, details: dict[str, Any]) -> tuple[str, list[str]]:
        """Return a reason and recommendations, falling back safely on errors."""
        try:
            import ollama
            prompt = (
                "You are an expert data-quality analyst. Analyze the following metric result using only "
                "the supplied evidence. Write a detailed, factual reason of 3 to 5 sentences that explains "
                "the score, calls out the most important indicators, and describes their likely impact. "
                "Do not repeat raw JSON or invent facts. Return valid JSON with keys reason (string) and "
                "recommendations (a list of concise strings). "
                f"Metric: {metric_name}. Score: {score:.4f} (0.0-1.0 scale). Details: {details}"
            )
            response = ollama.chat(model=self.model, messages=[{"role": "user", "content": prompt}], format="json")
            import json
            content = response["message"]["content"].strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            payload = json.loads(content)
            return str(payload.get("reason", "")), list(payload.get("recommendations", []))
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            return "", []
