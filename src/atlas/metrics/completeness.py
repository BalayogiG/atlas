from atlas.core import Metric, MetricResult, ValidationIssue
from atlas.decorators import metric
from atlas.core.context import Context

@metric
class Completeness(Metric):
    """
    Checks whether required fields are present and non-empty.
    """

    id = "DQ001"
    name = "Completeness"
    description = "Checks whether required fields are present and non-empty."
    category = "Data Quality"

    def evaluate(self, context: Context) -> MetricResult:
        required_fields = context.config.options.get("required_fields", [])
        total = len(context.dataset.records) * len(required_fields)
        present = 0
        missing = []

        for index, record in enumerate(context.dataset.records):
            for field in required_fields:
                value = record.get(field)
                if value is None:
                    missing.append(
                        ValidationIssue(
                            record=index,
                            field=field,
                            reason="missing"
                        )
                    )
                    continue

                if isinstance(value, str) and value.strip() == "":
                    missing.append(
                        ValidationIssue(
                            record=index,
                            field=field,
                            reason="empty"
                        )
                    )
                    continue

                present += 1
        score = present / total if total else 1.0

        return MetricResult(
            metric=self.name,
            score=score,
            passed=score == 1.0,
            issues=missing,
            metadata={
                "required_fields": required_fields,
                "present": present,
                "total": total,
            },
        )
