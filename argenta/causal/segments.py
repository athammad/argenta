"""Automatic segment HTE (Heterogeneous Treatment Effect) analysis.

:class:`SegmentAnalyzer` iterates over user feature columns and computes
the ATE within each segment.  This reveals which subgroups respond
differently to the treatment — the key question that makes CATE analysis
actionable.

Approach
--------
For each feature column:

- **Categorical features** (≤ ``max_unique_values`` distinct values):
  compute ATE per distinct value.
- **Numeric features**: bin users into quartiles (Q1–Q4), then compute
  ATE per quartile.

Each segment ATE uses the same :func:`~argenta.stats.ate.compute_ate`
function as the overall ATE, so confidence intervals and p-values are
on the same scale.

Segments with fewer than ``min_users`` users in either the control or
treatment group are suppressed to avoid unreliable estimates.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from argenta.causal.models import SegmentEffect
from argenta.stats.ate import compute_ate

logger = logging.getLogger(__name__)

_MAX_UNIQUE_CATEGORICAL = 20
_QUARTILE_LABELS = ["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"]


class SegmentAnalyzer:
    """Discovers heterogeneous treatment effects across feature segments.

    Args:
        min_users: Minimum users in a segment (control + treatment combined)
            for it to be included in results.  Defaults to ``50``.
        max_features: Maximum number of feature columns to analyse.  If
            there are more features than this limit, the most-varied
            features (highest variance) are selected.  Defaults to ``10``.
        alpha: Significance level for within-segment t-tests.  Defaults to
            ``0.05``.
    """

    def __init__(
        self,
        min_users: int = 50,
        max_features: int = 10,
        alpha: float = 0.05,
    ) -> None:
        self._min_users = min_users
        self._max_features = max_features
        self._alpha = alpha

    def analyze(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        control_variant: str,
        feature_cols: list[str],
    ) -> list[SegmentEffect]:
        """Compute segment-level treatment effects for all feature columns.

        Args:
            df: The prepared dataset with one row per user.
            outcome_col: Metric column to analyse (e.g. ``'purchase'``).
            treatment_col: Variant column (contains control / treatment labels).
            control_variant: The string label of the control variant.
            feature_cols: Feature columns to segment by.  Numeric features
                are auto-binned into quartiles.

        Returns:
            A list of :class:`~argenta.causal.models.SegmentEffect` objects,
            one per (feature, segment_value) pair that meets the minimum
            user threshold.  Sorted by ``feature_name`` then ``segment_value``.
        """
        if not feature_cols:
            return []

        # Select top features by variance (avoids near-constant columns)
        selected = _select_features_by_variance(df, feature_cols, self._max_features)
        logger.info("[SEGMENTS] Analysing %d features for HTE", len(selected))

        results: list[SegmentEffect] = []
        for col in selected:
            effects = self._analyze_feature(df, outcome_col, treatment_col, control_variant, col)
            results.extend(effects)

        results.sort(key=lambda s: (s.feature_name, s.segment_value))
        logger.info("[SEGMENTS] Found %d segments total", len(results))
        return results

    def _analyze_feature(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        control_variant: str,
        feature_col: str,
    ) -> list[SegmentEffect]:
        """Compute segment effects for a single feature column.

        Args:
            df: Full dataset.
            outcome_col: Metric column.
            treatment_col: Variant column.
            control_variant: Control variant label.
            feature_col: The feature column to segment by.

        Returns:
            List of :class:`~argenta.causal.models.SegmentEffect` objects for
            this feature.
        """
        col_data = df[feature_col].dropna()

        # Decide: categorical or numeric
        n_unique = col_data.nunique()
        is_categorical = (
            df[feature_col].dtype == object
            or df[feature_col].dtype.name == "category"
            or n_unique <= _MAX_UNIQUE_CATEGORICAL
        )

        if is_categorical:
            return self._analyze_categorical(df, outcome_col, treatment_col, control_variant, feature_col)
        else:
            return self._analyze_numeric(df, outcome_col, treatment_col, control_variant, feature_col)

    def _analyze_categorical(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        control_variant: str,
        feature_col: str,
    ) -> list[SegmentEffect]:
        """Compute segment effects for a categorical feature.

        Args:
            df: Full dataset.
            outcome_col: Metric column.
            treatment_col: Variant column.
            control_variant: Control variant label.
            feature_col: The categorical feature column.

        Returns:
            One :class:`~argenta.causal.models.SegmentEffect` per unique
            value with sufficient users.
        """
        results = []
        for value in df[feature_col].dropna().unique():
            mask = df[feature_col] == value
            effect = self._compute_segment_effect(
                df[mask], outcome_col, treatment_col, control_variant,
                feature_name=feature_col,
                segment_value=str(value),
            )
            if effect is not None:
                results.append(effect)
        return results

    def _analyze_numeric(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        control_variant: str,
        feature_col: str,
    ) -> list[SegmentEffect]:
        """Compute segment effects for a numeric feature, binned into quartiles.

        Args:
            df: Full dataset.
            outcome_col: Metric column.
            treatment_col: Variant column.
            control_variant: Control variant label.
            feature_col: The numeric feature column.

        Returns:
            Up to four :class:`~argenta.causal.models.SegmentEffect` objects
            (one per quartile) with sufficient users.
        """
        try:
            quartile_col = pd.qcut(df[feature_col], q=4, labels=_QUARTILE_LABELS, duplicates="drop")
        except ValueError:
            # qcut fails if there are too few distinct values
            return []

        results = []
        for label in _QUARTILE_LABELS:
            mask = quartile_col == label
            if mask.sum() == 0:
                continue
            effect = self._compute_segment_effect(
                df[mask], outcome_col, treatment_col, control_variant,
                feature_name=feature_col,
                segment_value=str(label),
            )
            if effect is not None:
                results.append(effect)
        return results

    def _compute_segment_effect(
        self,
        segment_df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        control_variant: str,
        feature_name: str,
        segment_value: str,
    ) -> SegmentEffect | None:
        """Compute ATE for a single segment.

        Args:
            segment_df: Subset of the main DataFrame for this segment.
            outcome_col: Metric column.
            treatment_col: Variant column.
            control_variant: Label of the control variant.
            feature_name: The feature that defines this segment.
            segment_value: The feature value that defines this segment.

        Returns:
            A :class:`~argenta.causal.models.SegmentEffect` if the segment
            has sufficient users; ``None`` otherwise.
        """
        ctrl = segment_df[segment_df[treatment_col] == control_variant][outcome_col].astype(float)
        treat = segment_df[segment_df[treatment_col] != control_variant][outcome_col].astype(float)

        # Enforce minimum user threshold
        if len(ctrl) + len(treat) < self._min_users:
            return None
        if len(ctrl) < 5 or len(treat) < 5:
            return None

        try:
            ate, ci_low, ci_high, p_value = compute_ate(ctrl, treat, alpha=self._alpha)
        except ValueError:
            return None

        mean_ctrl = float(ctrl.mean())
        relative_lift = (
            (float(treat.mean()) - mean_ctrl) / abs(mean_ctrl)
            if mean_ctrl != 0
            else None
        )

        return SegmentEffect(
            feature_name=feature_name,
            segment_value=segment_value,
            ate=ate,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_control=len(ctrl),
            n_treatment=len(treat),
            relative_lift=relative_lift,
            is_significant=p_value < self._alpha,
        )


def _select_features_by_variance(
    df: pd.DataFrame,
    feature_cols: list[str],
    max_features: int,
) -> list[str]:
    """Select the top ``max_features`` columns by variance.

    Near-constant columns (variance ≈ 0) are excluded because they cannot
    produce meaningful segment splits.

    Args:
        df: The dataset.
        feature_cols: Candidate feature column names.
        max_features: Maximum number of features to return.

    Returns:
        A list of at most ``max_features`` column names, sorted by variance
        descending.
    """
    variances: list[tuple[str, float]] = []
    for col in feature_cols:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        # Treat object columns as having max "variance" (always include)
        if col_data.dtype == object or col_data.dtype.name == "category":
            variances.append((col, float("inf")))
        else:
            v = float(col_data.var())
            if v > 1e-10:
                variances.append((col, v))

    variances.sort(key=lambda x: x[1], reverse=True)
    return [col for col, _ in variances[:max_features]]
