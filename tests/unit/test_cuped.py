"""Unit tests for argenta.stats.cuped."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argenta.stats.cuped import CupedError, apply_cuped, variance_reduction_ratio


class TestApplyCuped:
    def test_mean_is_preserved(self) -> None:
        rng = np.random.default_rng(0)
        y = pd.Series(rng.normal(10, 3, 500))
        x = 0.7 * y + pd.Series(rng.normal(0, 1, 500))  # correlated covariate
        y_cuped = apply_cuped(y, x)
        assert float(y_cuped.mean()) == pytest.approx(float(y.mean()), abs=1e-6)

    def test_variance_is_reduced_for_correlated_covariate(self) -> None:
        rng = np.random.default_rng(1)
        y = pd.Series(rng.normal(10, 3, 500))
        x = 0.9 * y + pd.Series(rng.normal(0, 0.5, 500))  # highly correlated
        y_cuped = apply_cuped(y, x)
        assert float(y_cuped.var()) < float(y.var())

    def test_uncorrelated_covariate_does_not_increase_variance_much(self) -> None:
        rng = np.random.default_rng(2)
        y = pd.Series(rng.normal(10, 3, 1000))
        x = pd.Series(rng.normal(0, 1, 1000))  # independent
        y_cuped = apply_cuped(y, x)
        # Variance should be roughly the same — not inflated
        assert float(y_cuped.var()) < float(y.var()) * 1.1

    def test_output_same_length_as_input(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        x = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5])
        result = apply_cuped(y, x)
        assert len(result) == len(y)

    def test_zero_variance_covariate_raises(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0])
        x = pd.Series([5.0, 5.0, 5.0])  # constant
        with pytest.raises(CupedError, match="zero variance"):
            apply_cuped(y, x)

    def test_mismatched_lengths_raises(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0])
        x = pd.Series([1.0, 2.0])
        with pytest.raises(ValueError, match="same length"):
            apply_cuped(y, x)

    def test_nan_in_covariate_handled(self) -> None:
        # Rows where the covariate is NaN cannot have CUPED applied.
        # The implementation keeps the original outcome value for those rows
        # (rather than propagating NaN), so they still contribute to the ATE.
        y = pd.Series([1.0, 2.0, 3.0, 4.0])
        x = pd.Series([1.1, float("nan"), 3.3, 4.4])
        result = apply_cuped(y, x)
        # Row 1 covariate is NaN — original outcome (2.0) is preserved
        assert result.isna().sum() == 0
        assert float(result.iloc[1]) == pytest.approx(2.0)

    def test_index_preserved(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0], index=[10, 20, 30])
        x = pd.Series([1.1, 2.2, 3.3], index=[10, 20, 30])
        result = apply_cuped(y, x)
        assert list(result.index) == [10, 20, 30]


class TestVarianceReductionRatio:
    def test_perfect_correlation_gives_near_zero(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        x = y.copy()  # perfect correlation
        ratio = variance_reduction_ratio(y, x)
        assert ratio == pytest.approx(0.0, abs=1e-6)

    def test_zero_correlation_gives_near_one(self) -> None:
        rng = np.random.default_rng(3)
        y = pd.Series(rng.normal(0, 1, 1000))
        x = pd.Series(rng.normal(0, 1, 1000))
        ratio = variance_reduction_ratio(y, x)
        # With independent draws, correlation ~ 0 so ratio ~ 1
        assert ratio == pytest.approx(1.0, abs=0.1)

    def test_zero_variance_covariate_raises(self) -> None:
        y = pd.Series([1.0, 2.0, 3.0])
        x = pd.Series([5.0, 5.0, 5.0])
        with pytest.raises(CupedError):
            variance_reduction_ratio(y, x)

    def test_ratio_is_between_zero_and_one(self) -> None:
        rng = np.random.default_rng(4)
        y = pd.Series(rng.normal(10, 3, 200))
        x = 0.6 * y + pd.Series(rng.normal(0, 1, 200))
        ratio = variance_reduction_ratio(y, x)
        assert 0.0 <= ratio <= 1.0
