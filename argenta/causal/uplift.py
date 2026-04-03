"""Uplift scoring — applying the fitted CATE model to the full user base.

The standard CATE estimation pipeline scores only experiment participants.
:class:`UpliftScorer` extends this to score *all* users in the user features
table, enabling:

- Targeting decisions: roll out to users with high predicted CATE
- Prioritisation: rank users by expected benefit before the next experiment
- Business case: estimate total revenue lift if treatment is applied selectively

This step requires an additional warehouse query to fetch the full user
features table.  It is controlled by ``CausalMLConfig.score_all_users``.

Usage::

    from argenta.causal.uplift import UpliftScorer

    scorer = UpliftScorer(fitted_estimator)
    scores = scorer.score_dataframe(all_users_df, feature_cols)
    # Returns a DataFrame: user_id, cate_score, ci_low, ci_high, percentile
"""

from __future__ import annotations

import logging

import pandas as pd

from argenta.causal.cate import CATEEstimator
from argenta.causal.models import UserCATEScore

logger = logging.getLogger(__name__)


class UpliftScorer:
    """Scores users for uplift using a fitted :class:`CATEEstimator`.

    Args:
        estimator: A fitted :class:`CATEEstimator` instance.
    """

    def __init__(self, estimator: CATEEstimator) -> None:
        self._estimator = estimator

    def score_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        alpha: float = 0.05,
    ) -> list[UserCATEScore]:
        """Score all rows in ``df`` for individual treatment effect.

        Args:
            df: A DataFrame of users to score.  Must contain all columns
                in ``feature_cols`` plus a user ID column.
            feature_cols: Feature columns to pass to the CATE model.  Must
                match the columns used when the estimator was fitted.
            alpha: Significance level for confidence intervals.

        Returns:
            A list of :class:`~argenta.causal.models.UserCATEScore` objects,
            one per row in ``df``.

        Raises:
            RuntimeError: If the estimator has not been fitted.
        """
        logger.info("[UPLIFT] Scoring %d users for uplift", len(df))
        scores = self._estimator.predict(df, alpha=alpha)
        logger.info("[UPLIFT] Scored %d users", len(scores))
        return scores

    def score_to_dataframe(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        user_id_col: str = "user_id",
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Score all rows in ``df`` and return results as a DataFrame.

        Convenience wrapper around :meth:`score_dataframe` that returns
        a flat DataFrame suitable for writing back to the warehouse.

        Args:
            df: A DataFrame of users to score.
            feature_cols: Feature columns for the CATE model.
            user_id_col: Name of the user ID column in ``df``.
            alpha: Significance level for confidence intervals.

        Returns:
            A :class:`pandas.DataFrame` with columns:
            ``user_id``, ``cate_score``, ``ci_low``, ``ci_high``,
            ``percentile``.
        """
        scores = self.score_dataframe(df, feature_cols, alpha=alpha)
        return pd.DataFrame([
            {
                "user_id": s.user_id,
                "cate_score": s.cate_score,
                "ci_low": s.ci_low,
                "ci_high": s.ci_high,
                "percentile": s.percentile,
            }
            for s in scores
        ])

    @staticmethod
    def targeting_summary(
        scores: list[UserCATEScore],
        percentile_threshold: float = 75.0,
    ) -> dict[str, float]:
        """Summarise the uplift distribution and targeting recommendation.

        Computes the expected average lift if only users above
        ``percentile_threshold`` are targeted with treatment.

        Args:
            scores: The list of :class:`~argenta.causal.models.UserCATEScore`
                objects produced by :meth:`score_dataframe`.
            percentile_threshold: Users above this percentile are considered
                high-uplift targets.  Defaults to ``75``.

        Returns:
            A dict with keys:

            - ``'n_total'`` — total users scored
            - ``'n_targets'`` — users above the threshold
            - ``'mean_cate_all'`` — mean CATE across all users
            - ``'mean_cate_targets'`` — mean CATE among targeted users
            - ``'mean_cate_non_targets'`` — mean CATE among non-targeted users
            - ``'targeting_efficiency'`` — ratio of target CATE to overall CATE
        """
        if not scores:
            return {}

        all_cates = [s.cate_score for s in scores]
        targets = [s for s in scores if s.percentile >= percentile_threshold]
        non_targets = [s for s in scores if s.percentile < percentile_threshold]

        mean_all = sum(all_cates) / len(all_cates)
        mean_target = sum(s.cate_score for s in targets) / max(len(targets), 1)
        mean_non = sum(s.cate_score for s in non_targets) / max(len(non_targets), 1)

        return {
            "n_total": len(scores),
            "n_targets": len(targets),
            "mean_cate_all": mean_all,
            "mean_cate_targets": mean_target,
            "mean_cate_non_targets": mean_non,
            "targeting_efficiency": mean_target / mean_all if mean_all != 0 else 0.0,
        }
