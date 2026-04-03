"""CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

CUPED reduces the variance of the outcome estimator by removing the
component of variance explained by a pre-experiment covariate.  Lower
variance means narrower confidence intervals and higher statistical power
for the same sample size.

Mathematical derivation
-----------------------
Let ``Y`` be the post-experiment outcome and ``X`` be a pre-experiment
covariate (e.g. revenue in the 30 days before the experiment started).

The CUPED-adjusted outcome is::

    Y_cuped = Y - θ * (X - E[X])

where::

    θ = Cov(Y, X) / Var(X)

The adjustment ``θ * (X - E[X])`` has zero expectation, so::

    E[Y_cuped] = E[Y]

The ATE estimate is unchanged.  What changes is the variance::

    Var(Y_cuped) = Var(Y) * (1 - ρ²)

where ``ρ = Corr(Y, X)``.  If the covariate is highly correlated with the
outcome, variance is substantially reduced.

Practical notes
---------------
- ``θ`` is estimated pooling control and treatment together (before the
  experiment saw outcomes), which avoids post-stratification bias.
- ``X`` should be measured in a period that overlaps with or precedes the
  experiment, not during it.
- If the covariate explains little variance (low ``ρ``), CUPED has minimal
  effect.  The :func:`apply_cuped` function returns the variance explained
  so callers can decide whether to report the CUPED result.

References
----------
- Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the
  Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment
  Data. *WSDM '13*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class CupedError(Exception):
    """Raised when CUPED cannot be applied to the provided data.

    Common causes:

    - The covariate has zero variance (all users have the same value).
    - The outcome and covariate arrays have different lengths or
      misaligned indices.
    """


def apply_cuped(outcome: pd.Series, covariate: pd.Series) -> pd.Series:
    """Apply CUPED variance reduction to an outcome series.

    Computes the CUPED-adjusted outcome::

        Y_cuped = Y - θ * (X - mean(X))

    where ``θ = Cov(Y, X) / Var(X)``, estimated on the pooled
    (control + treatment) sample.

    Args:
        outcome: Post-experiment outcome values (``Y``).  One value per
            user.  Must be numeric.
        covariate: Pre-experiment covariate values (``X``).  Must be the
            same length as ``outcome`` and aligned by index.

    Returns:
        A :class:`pandas.Series` of CUPED-adjusted outcome values.  The
        series has the same index as ``outcome``.  The mean is preserved
        (up to floating-point precision).

    Raises:
        CupedError: If the covariate has zero variance (all values equal),
            which would make ``θ`` undefined.
        ValueError: If ``outcome`` and ``covariate`` have different lengths.

    Example::

        from argenta.stats.cuped import apply_cuped

        y_cuped = apply_cuped(df["revenue"], df["pre_revenue"])
        # y_cuped has lower variance but the same mean as df["revenue"]
    """
    if len(outcome) != len(covariate):
        raise ValueError(
            f"outcome and covariate must have the same length, "
            f"got {len(outcome)} and {len(covariate)}"
        )

    # Drop rows where either is NaN for the theta computation
    mask = outcome.notna() & covariate.notna()
    y = outcome[mask].astype(float)
    x = covariate[mask].astype(float)

    if len(y) == 0:
        raise CupedError("No non-NaN values remain after dropping NaN from outcome/covariate")

    var_x = float(x.var(ddof=1))
    if var_x == 0.0 or np.isnan(var_x):
        raise CupedError(
            "Covariate has zero variance — all users have the same value. "
            "CUPED cannot reduce variance with a constant covariate. "
            "Set use_cuped: false in the experiment config."
        )

    cov_yx = float(np.cov(y.values, x.values, ddof=1)[0, 1])
    theta = cov_yx / var_x

    x_mean = float(x.mean())

    # Apply adjustment to the full series (NaN rows remain NaN)
    adjusted = outcome.copy().astype(float)
    adjusted[mask] = y - theta * (x - x_mean)
    return adjusted


def variance_reduction_ratio(outcome: pd.Series, covariate: pd.Series) -> float:
    """Estimate the variance reduction achieved by CUPED.

    Returns ``1 - ρ²`` where ``ρ`` is the Pearson correlation between
    ``outcome`` and ``covariate``.  A value close to ``0`` means near-total
    variance elimination; a value close to ``1`` means no reduction.

    Args:
        outcome: Post-experiment outcome values.
        covariate: Pre-experiment covariate values.

    Returns:
        A float in ``[0, 1]`` representing the fraction of original variance
        retained after CUPED.  Lower is better.

    Raises:
        CupedError: If the covariate has zero variance.
    """
    mask = outcome.notna() & covariate.notna()
    y = outcome[mask].astype(float)
    x = covariate[mask].astype(float)

    if float(x.var(ddof=1)) == 0.0:
        raise CupedError("Covariate has zero variance — correlation is undefined.")

    rho = float(np.corrcoef(y.values, x.values)[0, 1])
    return 1.0 - rho**2
