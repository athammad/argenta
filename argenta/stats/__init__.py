"""Statistical methods for Argenta experimentation analysis.

This sub-package has no imports from ``connectors`` or ``sql`` — it is
purely a function of DataFrames in, typed result objects out.
"""

from argenta.stats.ate import check_srm, compute_ate, winsorize
from argenta.stats.cuped import CupedError, apply_cuped, variance_reduction_ratio
from argenta.stats.models import ExperimentResult, MetricResult

__all__ = [
    "compute_ate",
    "winsorize",
    "check_srm",
    "apply_cuped",
    "variance_reduction_ratio",
    "CupedError",
    "ExperimentResult",
    "MetricResult",
]
