"""CATE estimation using the Causal Forest DML.

:class:`CATEEstimator` wraps ``econml.dml.CausalForestDML`` with a clean
interface aligned to Argenta's data contracts.  It handles:

- Nuisance model selection (LightGBM or linear)
- Feature preparation (numeric casting, NaN handling)
- Confidence interval extraction
- Percentile scoring

Statistical background
----------------------
The Causal Forest DML (Double Machine Learning) estimates the CATE via a
two-stage residualisation:

**Stage 1 (nuisance estimation):**

    Ê[Y | X] = outcome_model.fit(X).predict(X)
    Ê[T | X] = treatment_model.fit(X).predict_proba(X)[:, 1]

**Stage 2 (causal forest on residuals):**

    Ỹ = Y - Ê[Y | X]          (outcome residual)
    T̃ = T - Ê[T | X]          (treatment residual)

    CATE(x) = E[Ỹ / T̃ | X = x]  (estimated via a random forest)

This is doubly robust: the CATE estimate is consistent if *either* the
outcome model or the treatment model is correctly specified — not both.

References
----------
- Chernozhukov, V. et al. (2018). Double/Debiased Machine Learning for
  Treatment and Structural Parameters. *The Econometrics Journal*.
- Wager, S. & Athey, S. (2018). Estimation and Inference of Heterogeneous
  Treatment Effects using Random Forests. *JASA*, 113(523), 1228–1242.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from argenta.causal.models import UserCATEScore
from argenta.config.schema import CausalMLConfig

logger = logging.getLogger(__name__)


class CATEEstimator:
    """Estimates Conditional Average Treatment Effects using Causal Forest DML.

    Args:
        config: The :class:`~argenta.config.schema.CausalMLConfig` section
            of :class:`~argenta.config.schema.ArgentoConfig`.

    Raises:
        ImportError: If ``econml`` or ``lightgbm`` are not installed.
            Install with ``pip install 'argenta[causal]'``.
    """

    def __init__(self, config: CausalMLConfig) -> None:
        self._config = config
        self._model: object = None  # CausalForestDML, untyped to avoid import at module level
        self._feature_cols: list[str] = []
        self._fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        outcome_col: str,
        treatment_col: str,
        feature_cols: list[str],
    ) -> "CATEEstimator":
        """Fit the Causal Forest DML on an experiment dataset.

        Args:
            df: The prepared dataset.  Must contain ``outcome_col``,
                ``treatment_col``, and all columns in ``feature_cols``.
            outcome_col: Name of the metric column (e.g. ``'purchase'``).
            treatment_col: Name of the variant column.  Must contain the
                string labels ``'control'`` and ``'treatment'``; these are
                binarised to 0/1 internally.
            feature_cols: List of feature column names to use as covariates
                ``X``.  Must all be numeric after casting.

        Returns:
            ``self`` (allows method chaining).

        Raises:
            ValueError: If ``feature_cols`` is empty or contains columns not
                present in ``df``.
            ImportError: If ``econml`` or ``lightgbm`` are not installed.
        """
        _check_imports()

        if not feature_cols:
            raise ValueError(
                "feature_cols must contain at least one column to fit the CATE model. "
                "Add feature_cols to the [user_features] section of your config."
            )

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"feature_cols not found in DataFrame: {missing}")

        self._feature_cols = feature_cols

        Y, T, X = _prepare_matrices(df, outcome_col, treatment_col, feature_cols)

        logger.info(
            "[CATE] Fitting CausalForestDML on %d users, %d features, metric=%s",
            len(Y), X.shape[1], outcome_col,
        )

        from econml.dml import CausalForestDML  # type: ignore[import]

        model_y, model_t = _build_nuisance_models(self._config.nuisance_model)

        # CausalForestDML requires n_estimators divisible by subforest_size (default=4)
        subforest_size = 4
        n_estimators = self._config.n_estimators
        if n_estimators % subforest_size != 0:
            n_estimators = max(subforest_size, (n_estimators // subforest_size + 1) * subforest_size)
            logger.debug("[CATE] Adjusted n_estimators to %d (must be divisible by %d)", n_estimators, subforest_size)

        self._model = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,  # T is always binary (control=0 / treatment=1)
            n_estimators=n_estimators,
            min_samples_leaf=self._config.min_samples_leaf,
            max_depth=self._config.max_depth,
            random_state=42,
            verbose=0,
        )
        self._model.fit(Y, T.astype(int), X=X)  # type: ignore[union-attr]
        self._fitted = True

        logger.info("[CATE] Model fitted successfully")
        return self

    def predict(
        self,
        df: pd.DataFrame,
        alpha: float = 0.05,
    ) -> list[UserCATEScore]:
        """Predict CATE scores for users in the given DataFrame.

        Args:
            df: A DataFrame containing the feature columns used during
                :meth:`fit`.  Must have a column matching
                ``user_features.user_id_col``.
            alpha: Significance level for confidence intervals.

        Returns:
            A list of :class:`~argenta.causal.models.UserCATEScore` objects,
            one per row in ``df``.  Includes point estimates, CI bounds, and
            percentile ranks.

        Raises:
            RuntimeError: If :meth:`fit` has not been called.
        """
        self._assert_fitted()
        X = _extract_features(df, self._feature_cols)

        point = self._model.effect(X)  # type: ignore[union-attr]
        lo, hi = self._model.effect_interval(X, alpha=alpha)  # type: ignore[union-attr]

        # Compute percentile rank within this scored population
        percentiles = _percentile_rank(point)

        # Infer user_id column (first column not in feature_cols)
        user_id_col = _infer_user_id_col(df, self._feature_cols)

        scores = []
        for i, row in enumerate(df.itertuples(index=False)):
            uid = str(getattr(row, user_id_col, f"row_{i}"))
            scores.append(UserCATEScore(
                user_id=uid,
                cate_score=float(point[i]),
                ci_low=float(lo[i]),
                ci_high=float(hi[i]),
                percentile=float(percentiles[i]),
            ))
        return scores

    def model_diagnostics(self) -> dict[str, float | None]:
        """Return first-stage model diagnostic scores.

        Returns:
            A dict with keys ``'r2_outcome'`` and ``'r2_treatment'``,
            containing the cross-validated R² scores for the nuisance models.
            Values are ``None`` if the model has not been fitted.
        """
        if not self._fitted or self._model is None:
            return {"r2_outcome": None, "r2_treatment": None}
        try:
            scores = self._model.score_  # type: ignore[union-attr]
            return {
                "r2_outcome": float(scores.get("outcome_model", {}).get("r2", 0.0) or 0.0),
                "r2_treatment": float(scores.get("treatment_model", {}).get("r2", 0.0) or 0.0),
            }
        except AttributeError:
            return {"r2_outcome": None, "r2_treatment": None}

    def _assert_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "CATEEstimator has not been fitted. Call fit() before predict()."
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_imports() -> None:
    """Raise a helpful ImportError if econml/lightgbm are not installed."""
    try:
        import econml  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "econml is not installed. Install the causal ML extras with:\n"
            "    pip install 'argenta[causal]'"
        ) from exc


def _prepare_matrices(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract and clean (Y, T, X) matrices from the prepared dataset.

    Args:
        df: The prepared dataset DataFrame.
        outcome_col: Outcome metric column name.
        treatment_col: Variant column (string labels).
        feature_cols: Feature column names.

    Returns:
        A tuple ``(Y, T, X)`` as numpy arrays.  Rows with any NaN in X are
        dropped.
    """
    sub = df[[outcome_col, treatment_col] + feature_cols].copy()

    # Binarise treatment: any value other than 'control' → 1
    sub["_T"] = (sub[treatment_col] != "control").astype(int)

    # Drop rows with NaN in X (features)
    before = len(sub)
    sub = sub.dropna(subset=feature_cols)
    dropped = before - len(sub)
    if dropped > 0:
        logger.warning("[CATE] Dropped %d rows with NaN feature values", dropped)

    Y = sub[outcome_col].values.astype(float)
    T = sub["_T"].values.astype(float)
    X = sub[feature_cols].values.astype(float)
    return Y, T, X


def _extract_features(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Extract feature matrix from a DataFrame, filling NaNs with column means.

    Args:
        df: DataFrame containing ``feature_cols``.
        feature_cols: Ordered list of feature column names.

    Returns:
        A float numpy array of shape ``(n_rows, n_features)``.
    """
    X = df[feature_cols].copy().astype(float)
    X = X.fillna(X.mean())
    return X.values


def _build_nuisance_models(
    nuisance_model: str,
) -> tuple[object, object]:
    """Build the first-stage outcome and treatment nuisance models.

    Args:
        nuisance_model: One of ``'lightgbm'`` or ``'linear'``.

    Returns:
        A tuple ``(model_y, model_t)`` where ``model_y`` is a regressor
        for the outcome and ``model_t`` is a classifier for the treatment.
    """
    if nuisance_model == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor  # type: ignore[import]
        model_y = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        model_t = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        from sklearn.linear_model import LinearRegression, LogisticRegression  # type: ignore[import]
        model_y = LinearRegression()
        model_t = LogisticRegression(max_iter=500, random_state=42)

    return model_y, model_t


def _percentile_rank(scores: np.ndarray) -> np.ndarray:
    """Compute percentile rank (0–100) for each value in an array.

    Args:
        scores: A 1D numpy array of CATE point estimates.

    Returns:
        A 1D numpy array of the same length, where each value is the
        percentile rank of the corresponding score.
    """
    from scipy.stats import rankdata  # type: ignore[import]
    ranks = rankdata(scores, method="average")
    return 100.0 * (ranks - 1) / max(len(scores) - 1, 1)


def _infer_user_id_col(df: pd.DataFrame, feature_cols: list[str]) -> str:
    """Infer the user ID column as the first column not in feature_cols.

    Args:
        df: The DataFrame being scored.
        feature_cols: Feature columns used by the model.

    Returns:
        The name of the first column not in ``feature_cols``.  Falls back
        to ``'user_id'`` if no such column is found.
    """
    non_feature = [c for c in df.columns if c not in feature_cols]
    return non_feature[0] if non_feature else "user_id"
