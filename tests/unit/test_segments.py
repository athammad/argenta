"""Unit tests for argenta.causal.segments.SegmentAnalyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argenta.causal.segments import SegmentAnalyzer, _select_features_by_variance


def _make_df(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic DataFrame with categorical + numeric features."""
    rng = np.random.default_rng(seed)
    country = rng.choice(["US", "UK", "DE"], n)
    device = rng.choice(["mobile", "desktop"], n)
    tenure = rng.integers(1, 365, n).astype(float)
    treatment = rng.choice(["control", "treatment"], n)

    # Heterogeneous effect: US users get bigger lift
    effect = np.where(country == "US", 4.0, 1.0)
    effect *= np.where(treatment == "treatment", 1.0, 0.0)
    outcome = 10.0 + effect + rng.normal(0, 2, n)

    return pd.DataFrame({
        "user_id": [f"u_{i}" for i in range(n)],
        "variant": treatment,
        "purchase": outcome,
        "country": country,
        "device": device,
        "tenure": tenure,
    })


@pytest.fixture
def df() -> pd.DataFrame:
    return _make_df()


class TestSegmentAnalyzer:
    def test_returns_list_of_segment_effects(self, df) -> None:
        analyzer = SegmentAnalyzer(min_users=20)
        effects = analyzer.analyze(df, "purchase", "variant", "control", ["country", "device"])
        assert isinstance(effects, list)

    def test_country_segments_present(self, df) -> None:
        analyzer = SegmentAnalyzer(min_users=20)
        effects = analyzer.analyze(df, "purchase", "variant", "control", ["country"])
        feature_names = {e.feature_name for e in effects}
        assert "country" in feature_names

    def test_all_segment_values_are_strings(self, df) -> None:
        analyzer = SegmentAnalyzer(min_users=20)
        effects = analyzer.analyze(df, "purchase", "variant", "control", ["country", "device", "tenure"])
        assert all(isinstance(e.segment_value, str) for e in effects)

    def test_us_segment_has_higher_ate(self, df) -> None:
        """US users have a 4x lift in our DGP — the analyzer should reflect this."""
        analyzer = SegmentAnalyzer(min_users=20)
        effects = analyzer.analyze(df, "purchase", "variant", "control", ["country"])
        by_country = {e.segment_value: e.ate for e in effects if e.feature_name == "country"}
        if "US" in by_country and "UK" in by_country:
            assert by_country["US"] > by_country["UK"]

    def test_min_users_threshold_respected(self) -> None:
        """Segments below min_users should be suppressed."""
        small_df = _make_df(n=60, seed=2)
        analyzer = SegmentAnalyzer(min_users=100)  # threshold higher than any segment
        effects = analyzer.analyze(small_df, "purchase", "variant", "control", ["country"])
        assert len(effects) == 0

    def test_empty_feature_cols_returns_empty_list(self, df) -> None:
        analyzer = SegmentAnalyzer()
        effects = analyzer.analyze(df, "purchase", "variant", "control", [])
        assert effects == []

    def test_numeric_feature_produces_quartile_segments(self, df) -> None:
        analyzer = SegmentAnalyzer(min_users=20)
        effects = analyzer.analyze(df, "purchase", "variant", "control", ["tenure"])
        labels = {e.segment_value for e in effects if e.feature_name == "tenure"}
        # At least one quartile label should appear
        assert any("Q" in label for label in labels)

    def test_is_significant_matches_p_value(self, df) -> None:
        analyzer = SegmentAnalyzer(min_users=10, alpha=0.05)
        effects = analyzer.analyze(df, "purchase", "variant", "control", ["country"])
        for e in effects:
            assert e.is_significant == (e.p_value < 0.05)

    def test_n_control_plus_n_treatment_leq_total(self, df) -> None:
        analyzer = SegmentAnalyzer(min_users=10)
        effects = analyzer.analyze(df, "purchase", "variant", "control", ["country"])
        for e in effects:
            assert e.n_control + e.n_treatment <= len(df)


class TestSelectFeaturesByVariance:
    def test_selects_up_to_max_features(self) -> None:
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [10.0, 20.0, 30.0],
            "c": [100.0, 200.0, 300.0],
        })
        selected = _select_features_by_variance(df, ["a", "b", "c"], max_features=2)
        assert len(selected) <= 2

    def test_constant_column_excluded(self) -> None:
        df = pd.DataFrame({
            "a": [5.0, 5.0, 5.0],
            "b": [1.0, 2.0, 3.0],
        })
        selected = _select_features_by_variance(df, ["a", "b"], max_features=2)
        assert "a" not in selected
        assert "b" in selected

    def test_string_column_always_included(self) -> None:
        df = pd.DataFrame({
            "cat": ["x", "y", "z"],
            "num": [1.0, 2.0, 3.0],
        })
        selected = _select_features_by_variance(df, ["cat", "num"], max_features=2)
        assert "cat" in selected
