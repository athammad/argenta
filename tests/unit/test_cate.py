"""Unit tests for argenta.causal.cate.CATEEstimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argenta.causal.cate import (
    CATEEstimator,
    _build_nuisance_models,
    _percentile_rank,
)
from argenta.causal.models import UserCATEScore
from argenta.config.loader import load_config_from_dict
from tests.conftest import MINIMAL_CONFIG_DICT


def _causal_config_dict() -> dict:
    """Config dict with causal_ml section enabled."""
    d = dict(MINIMAL_CONFIG_DICT)
    d["user_features"] = {
        "table": "TEST_DB.PUBLIC.USERS",
        "covariate_col": "pre_revenue",
        "feature_cols": ["age", "tenure"],
    }
    d["causal_ml"] = {
        "enabled": True,
        "n_estimators": 50,
        "min_samples_leaf": 5,
        "nuisance_model": "linear",
    }
    return d


def _make_experiment_df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """Build a synthetic prepared-dataset DataFrame with feature heterogeneity."""
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 60, n)
    tenure = rng.integers(1, 365, n)
    treatment = (rng.random(n) > 0.5).astype(int)
    # Treatment effect is heterogeneous: higher for older users
    effect = 2.0 + 0.05 * (age - 40)
    outcome = 10.0 + effect * treatment + rng.normal(0, 2, n)
    pre_revenue = outcome * 0.8 + rng.normal(0, 1, n)

    return pd.DataFrame({
        "user_id": [f"u_{i}" for i in range(n)],
        "variant": ["treatment" if t else "control" for t in treatment],
        "purchase": outcome,
        "pre_revenue": pre_revenue,
        "age": age.astype(float),
        "tenure": tenure.astype(float),
    })


@pytest.fixture
def causal_config():
    return load_config_from_dict(_causal_config_dict())


@pytest.fixture
def experiment_df():
    return _make_experiment_df()


class TestCATEEstimatorFit:
    def test_fit_returns_self(self, causal_config, experiment_df) -> None:
        est = CATEEstimator(causal_config.causal_ml)
        result = est.fit(experiment_df, "purchase", "variant", ["age", "tenure"])
        assert result is est

    def test_fitted_flag_set_after_fit(self, causal_config, experiment_df) -> None:
        est = CATEEstimator(causal_config.causal_ml)
        assert est._fitted is False
        est.fit(experiment_df, "purchase", "variant", ["age", "tenure"])
        assert est._fitted is True

    def test_empty_feature_cols_raises(self, causal_config, experiment_df) -> None:
        est = CATEEstimator(causal_config.causal_ml)
        with pytest.raises(ValueError, match="feature_cols"):
            est.fit(experiment_df, "purchase", "variant", [])

    def test_missing_feature_col_raises(self, causal_config, experiment_df) -> None:
        est = CATEEstimator(causal_config.causal_ml)
        with pytest.raises(ValueError, match="not found"):
            est.fit(experiment_df, "purchase", "variant", ["age", "nonexistent_col"])


class TestCATEEstimatorPredict:
    def test_predict_returns_list_of_user_cate_scores(self, causal_config, experiment_df) -> None:
        est = CATEEstimator(causal_config.causal_ml)
        est.fit(experiment_df, "purchase", "variant", ["age", "tenure"])
        scores = est.predict(experiment_df[["user_id", "age", "tenure"]])
        assert isinstance(scores, list)
        assert all(isinstance(s, UserCATEScore) for s in scores)

    def test_predict_length_matches_input(self, causal_config, experiment_df) -> None:
        est = CATEEstimator(causal_config.causal_ml)
        est.fit(experiment_df, "purchase", "variant", ["age", "tenure"])
        scores = est.predict(experiment_df[["user_id", "age", "tenure"]])
        assert len(scores) == len(experiment_df)

    def test_cate_scores_are_finite(self, causal_config, experiment_df) -> None:
        est = CATEEstimator(causal_config.causal_ml)
        est.fit(experiment_df, "purchase", "variant", ["age", "tenure"])
        scores = est.predict(experiment_df[["user_id", "age", "tenure"]])
        assert all(np.isfinite(s.cate_score) for s in scores)

    def test_ci_low_less_than_or_equal_ci_high(self, causal_config, experiment_df) -> None:
        est = CATEEstimator(causal_config.causal_ml)
        est.fit(experiment_df, "purchase", "variant", ["age", "tenure"])
        scores = est.predict(experiment_df[["user_id", "age", "tenure"]])
        assert all(s.ci_low <= s.ci_high for s in scores)

    def test_percentiles_in_range(self, causal_config, experiment_df) -> None:
        est = CATEEstimator(causal_config.causal_ml)
        est.fit(experiment_df, "purchase", "variant", ["age", "tenure"])
        scores = est.predict(experiment_df[["user_id", "age", "tenure"]])
        assert all(0.0 <= s.percentile <= 100.0 for s in scores)

    def test_predict_before_fit_raises(self, causal_config, experiment_df) -> None:
        est = CATEEstimator(causal_config.causal_ml)
        with pytest.raises(RuntimeError, match="not been fitted"):
            est.predict(experiment_df[["user_id", "age", "tenure"]])

    def test_heterogeneous_effect_detected(self, causal_config) -> None:
        """Older users have higher CATE in our synthetic DGP — verify the model captures this."""
        df = _make_experiment_df(n=500, seed=1)
        est = CATEEstimator(causal_config.causal_ml)
        est.fit(df, "purchase", "variant", ["age", "tenure"])
        scores_df = df[["user_id", "age", "tenure"]].copy()
        predictions = est.predict(scores_df)

        cate_by_age = pd.DataFrame({
            "age": df["age"].values,
            "cate": [s.cate_score for s in predictions],
        })
        # Older users (age > 50) should have higher average CATE than younger (age < 30)
        old_mean = cate_by_age[cate_by_age["age"] > 50]["cate"].mean()
        young_mean = cate_by_age[cate_by_age["age"] < 30]["cate"].mean()
        assert old_mean > young_mean


class TestPercentileRank:
    def test_min_gets_zero(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        ranks = _percentile_rank(arr)
        assert ranks[0] == pytest.approx(0.0)

    def test_max_gets_hundred(self) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        ranks = _percentile_rank(arr)
        assert ranks[-1] == pytest.approx(100.0)

    def test_single_element_returns_zero(self) -> None:
        arr = np.array([5.0])
        ranks = _percentile_rank(arr)
        assert ranks[0] == pytest.approx(0.0)


class TestBuildNuisanceModels:
    def test_lightgbm_returns_lgbm_instances(self) -> None:
        from lightgbm import LGBMClassifier, LGBMRegressor
        model_y, model_t = _build_nuisance_models("lightgbm")
        assert isinstance(model_y, LGBMRegressor)
        assert isinstance(model_t, LGBMClassifier)

    def test_linear_returns_sklearn_instances(self) -> None:
        from sklearn.linear_model import LinearRegression, LogisticRegression
        model_y, model_t = _build_nuisance_models("linear")
        assert isinstance(model_y, LinearRegression)
        assert isinstance(model_t, LogisticRegression)
