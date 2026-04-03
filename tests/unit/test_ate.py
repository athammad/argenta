"""Unit tests for argenta.stats.ate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argenta.stats.ate import check_srm, compute_ate, winsorize


class TestComputeAte:
    def test_positive_ate_detected(self) -> None:
        rng = np.random.default_rng(0)
        control = pd.Series(rng.normal(10, 2, 200))
        treatment = pd.Series(rng.normal(12, 2, 200))
        ate, _, _, _ = compute_ate(control, treatment)
        assert ate > 0

    def test_ate_equals_difference_in_means(self) -> None:
        control = pd.Series([1.0, 2.0, 3.0])
        treatment = pd.Series([4.0, 5.0, 6.0])
        ate, _, _, _ = compute_ate(control, treatment)
        expected = treatment.mean() - control.mean()
        assert ate == pytest.approx(expected)

    def test_significant_difference_detected(self) -> None:
        rng = np.random.default_rng(1)
        control = pd.Series(rng.normal(10, 1, 1000))
        treatment = pd.Series(rng.normal(11, 1, 1000))
        _, _, _, p_value = compute_ate(control, treatment)
        assert p_value < 0.05

    def test_no_difference_not_significant(self) -> None:
        rng = np.random.default_rng(2)
        control = pd.Series(rng.normal(10, 2, 100))
        treatment = pd.Series(rng.normal(10, 2, 100))
        _, _, _, p_value = compute_ate(control, treatment)
        # With equal means, p-value should not be reliably < 0.05
        # Use a very small sample — this should be non-significant
        assert p_value > 0.01  # generous bound

    def test_ci_contains_ate(self) -> None:
        rng = np.random.default_rng(3)
        control = pd.Series(rng.normal(10, 2, 200))
        treatment = pd.Series(rng.normal(12, 2, 200))
        ate, ci_low, ci_high, _ = compute_ate(control, treatment)
        assert ci_low <= ate <= ci_high

    def test_ci_width_decreases_with_larger_sample(self) -> None:
        rng = np.random.default_rng(4)
        c_small = pd.Series(rng.normal(10, 2, 50))
        t_small = pd.Series(rng.normal(12, 2, 50))
        c_large = pd.Series(rng.normal(10, 2, 5000))
        t_large = pd.Series(rng.normal(12, 2, 5000))

        _, lo_s, hi_s, _ = compute_ate(c_small, t_small)
        _, lo_l, hi_l, _ = compute_ate(c_large, t_large)

        assert (hi_s - lo_s) > (hi_l - lo_l)

    def test_nan_values_are_dropped(self) -> None:
        control = pd.Series([1.0, 2.0, float("nan"), 3.0])
        treatment = pd.Series([4.0, float("nan"), 5.0, 6.0])
        ate, _, _, _ = compute_ate(control, treatment)
        # Should not raise; NaNs are dropped
        assert np.isfinite(ate)

    def test_empty_control_raises(self) -> None:
        with pytest.raises(ValueError, match="control series is empty"):
            compute_ate(pd.Series([], dtype=float), pd.Series([1.0, 2.0]))

    def test_empty_treatment_raises(self) -> None:
        with pytest.raises(ValueError, match="treatment series is empty"):
            compute_ate(pd.Series([1.0, 2.0]), pd.Series([], dtype=float))

    def test_invalid_alpha_raises(self) -> None:
        c = pd.Series([1.0, 2.0])
        t = pd.Series([3.0, 4.0])
        with pytest.raises(ValueError, match="alpha"):
            compute_ate(c, t, alpha=1.5)


class TestWinsorize:
    def test_clips_upper_tail(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        result = winsorize(s, percentile=0.95)
        assert result.max() < 100.0

    def test_does_not_change_lower_values(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        result = winsorize(s, percentile=0.95)
        assert result.iloc[0] == pytest.approx(1.0)

    def test_percentile_1_is_noop(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
        result = winsorize(s, percentile=1.0)
        pd.testing.assert_series_equal(result, s)

    def test_invalid_percentile_raises(self) -> None:
        with pytest.raises(ValueError, match="percentile"):
            winsorize(pd.Series([1.0, 2.0]), percentile=0.3)

    def test_nan_preserved(self) -> None:
        s = pd.Series([1.0, float("nan"), 3.0, 100.0])
        result = winsorize(s, percentile=0.95)
        assert result.isna().sum() == 1


class TestCheckSrm:
    def test_balanced_split_no_srm(self) -> None:
        # Perfectly balanced — no SRM
        assert check_srm(5000, 5000) is False

    def test_severely_imbalanced_detects_srm(self) -> None:
        # 80/20 split when 50/50 expected
        assert check_srm(8000, 2000) is True

    def test_zero_control_raises(self) -> None:
        with pytest.raises(ValueError):
            check_srm(0, 1000)

    def test_zero_treatment_raises(self) -> None:
        with pytest.raises(ValueError):
            check_srm(1000, 0)

    def test_custom_expected_ratio(self) -> None:
        # 70/30 split when 70/30 expected — no SRM
        assert check_srm(7000, 3000, expected_ratio=0.7) is False
