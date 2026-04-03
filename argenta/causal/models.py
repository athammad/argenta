"""Pydantic result models for the Argenta causal ML layer (Phase 2).

These models extend the Phase 1 result surface with:

- :class:`UserCATEScore` — individual treatment effect estimate per user
- :class:`SegmentEffect` — HTE for a specific feature segment
- :class:`CATEMetricResult` — full CATE results for one metric
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UserCATEScore(BaseModel):
    """Individual-level treatment effect estimate for one user.

    The CATE (Conditional Average Treatment Effect) is the expected
    treatment effect for a user with their specific feature values::

        CATE(x) = E[Y(1) - Y(0) | X = x]

    Attributes:
        user_id: The user identifier, matching the ``user_id_col`` from
            the exposures table.
        cate_score: Point estimate of the individual treatment effect.
            Positive values indicate the user is expected to benefit from
            treatment; negative values indicate potential harm.
        ci_low: Lower bound of the 95% confidence interval on the CATE.
        ci_high: Upper bound of the 95% confidence interval on the CATE.
        percentile: The user's CATE percentile within the scored population
            (0–100).  Users at the 90th percentile benefit most from
            treatment.
    """

    user_id: str
    cate_score: float
    ci_low: float
    ci_high: float
    percentile: float


class SegmentEffect(BaseModel):
    """Heterogeneous treatment effect for a single feature segment.

    Attributes:
        feature_name: The feature column that defines this segment.
        segment_value: The value of the feature (e.g. ``'mobile'``,
            ``'Q3'`` for the third quartile of a numeric feature).
        ate: Average Treatment Effect within this segment.
        ci_low: Lower bound of the CI on the segment ATE.
        ci_high: Upper bound of the CI on the segment ATE.
        p_value: Two-sided p-value from Welch's t-test within the segment.
        n_control: Number of control users in this segment.
        n_treatment: Number of treatment users in this segment.
        relative_lift: Relative lift of treatment over control within the
            segment.  ``None`` if the control mean is zero.
        is_significant: Whether ``p_value < alpha`` (alpha taken from
            :class:`~argenta.config.schema.ExperimentConfig`).
    """

    feature_name: str
    segment_value: str
    ate: float
    ci_low: float
    ci_high: float
    p_value: float
    n_control: int
    n_treatment: int
    relative_lift: float | None
    is_significant: bool


class CATEMetricResult(BaseModel):
    """Full causal ML results for a single metric.

    Attributes:
        metric_name: The metric this result corresponds to.
        user_scores: Per-user CATE scores for experiment participants
            (and all users if ``score_all_users=True``).  Empty list if
            ``CausalMLConfig.enabled`` is ``False``.
        segment_effects: List of :class:`SegmentEffect` objects, one per
            (feature, segment_value) pair with sufficient users.
        mean_cate: Population mean CATE across all scored users.
        std_cate: Standard deviation of CATE scores across all scored users.
        n_users_scored: Total number of users for whom CATE was estimated.
        model_r2_outcome: R² of the first-stage outcome model (nuisance
            model fitting quality).  Values close to 1.0 indicate the
            features explain much of the outcome variance.
        model_r2_treatment: R² of the first-stage treatment propensity
            model.  Values close to 0.5 indicate balanced randomisation
            (expected for A/B tests).
    """

    metric_name: str
    user_scores: list[UserCATEScore] = Field(default_factory=list)
    segment_effects: list[SegmentEffect] = Field(default_factory=list)
    mean_cate: float
    std_cate: float
    n_users_scored: int
    model_r2_outcome: float | None = None
    model_r2_treatment: float | None = None

    def top_segments(self, n: int = 5) -> list[SegmentEffect]:
        """Return the ``n`` segments with the highest ATE, sorted descending.

        Args:
            n: Number of top segments to return.

        Returns:
            A list of at most ``n`` :class:`SegmentEffect` objects, sorted
            by ``ate`` descending.
        """
        return sorted(self.segment_effects, key=lambda s: s.ate, reverse=True)[:n]

    def significant_segments(self) -> list[SegmentEffect]:
        """Return segments where the treatment effect is statistically significant.

        Returns:
            A list of :class:`SegmentEffect` objects where
            ``is_significant`` is ``True``, sorted by ``ate`` descending.
        """
        return sorted(
            [s for s in self.segment_effects if s.is_significant],
            key=lambda s: s.ate,
            reverse=True,
        )
