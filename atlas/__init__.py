"""Public API for ATLAS."""

import logging

from .atlas import Atlas
from .base import AtlasSchema, EvaluationReport, Metric, MetricResult
from .decorators import metric

__all__ = ["Atlas", "AtlasSchema", "Metric", "MetricResult", "EvaluationReport", "metric"]

# ATLAS emits diagnostics via the standard `logging` module under the "atlas"
# logger hierarchy. As a library, it stays silent unless the host application
# configures handlers/levels for "atlas" (or attaches its own), per Python's
# recommended library-logging pattern.
logging.getLogger(__name__).addHandler(logging.NullHandler())
