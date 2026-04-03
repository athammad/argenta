"""Causal ML layer for Argenta (Phase 2).

Provides CATE estimation, segment HTE analysis, and uplift scoring on top
of the Phase 1 pipeline.

Requires the ``[causal]`` extras::

    pip install 'argenta[causal]'
"""

from argenta.causal.cate import CATEEstimator
from argenta.causal.models import CATEMetricResult, SegmentEffect, UserCATEScore
from argenta.causal.segments import SegmentAnalyzer
from argenta.causal.uplift import UpliftScorer

__all__ = [
    "CATEEstimator",
    "SegmentAnalyzer",
    "UpliftScorer",
    "CATEMetricResult",
    "SegmentEffect",
    "UserCATEScore",
]
