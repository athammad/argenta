"""Average Treatment Effect estimation and related statistical utilities.

All functions in this module are pure: they receive :class:`pandas.Series`
objects (or scalars) and return typed results.  There is no warehouse I/O.

Statistical approach
--------------------
ATE is estimated as the difference in sample means:

    ATE = mean(Y_treatment) - mean(Y_control)

Confidence intervals and p-values are computed using **Welch's t-test**,
which does not assume equal variances across groups.  This is the standard
choice for online experimentation because:

1. Treatment and control groups often have different sample variances,
   especially for revenue metrics where one group may have a pricing change.
2. Welch's test is more robust to unequal group sizes, which is common
   when traffic allocation is not exactly 50/50.

References
----------
- Welch, B. L. (1947). The generalization of "Student's" problem when
  several different population variances are involved.
  *Biometrika*, 34(1-2), 28–35.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_ate(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Compute the Average Treatment Effect using Welch's t-test.

    Args:
        control: Outcome values for users in the control group.  Must be
            numeric and non-empty.
        treatment: Outcome values for users in the treatment group.  Must
            be numeric and non-empty.
        alpha: Significance level for the confidence interval.  Must be in
            the open interval ``(0, 1)``.  Defaults to ``0.05``.

    Returns:
        A tuple ``(ate, ci_low, ci_high, p_value)`` where:

        - ``ate`` is the difference in means (treatment - control).
        - ``ci_low`` is the lower bound of the ``(1 - alpha)`` CI.
        - ``ci_high`` is the upper bound of the ``(1 - alpha)`` CI.
        - ``p_value`` is the two-sided p-value from Welch's t-test.

    Raises:
        ValueError: If either series is empty, contains all-NaN values,
            or if ``alpha`` is not in ``(0, 1)``.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    control = control.dropna()
    treatment = treatment.dropna()

    if len(control) == 0:
        raise ValueError("control series is empty after dropping NaN values")
    if len(treatment) == 0:
        raise ValueError("treatment series is empty after dropping NaN values")

    ate = float(treatment.mean() - control.mean())

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)
    p_value = float(p_value)

    # Confidence interval via the t-distribution with Welch-Satterthwaite df
    n1, n2 = len(treatment), len(control)
    s1, s2 = float(treatment.var(ddof=1)), float(control.var(ddof=1))

    se = float(np.sqrt(s1 / n1 + s2 / n2))

    # Welch-Satterthwaite degrees of freedom
    if se == 0:
        df = n1 + n2 - 2.0
    else:
        df = (s1 / n1 + s2 / n2) ** 2 / (
            (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
        )

    t_crit = float(stats.t.ppf(1 - alpha / 2, df=df))
    ci_low = ate - t_crit * se
    ci_high = ate + t_crit * se

    return ate, ci_low, ci_high, p_value


def winsorize(series: pd.Series, percentile: float) -> pd.Series:
    """Clip the upper tail of a numeric series at the given percentile.

    Winsorization replaces values above the ``percentile``-th quantile with
    the value at that quantile.  This reduces the influence of extreme
    outliers on mean-based estimators (ATE, variance) without discarding
    data points.

    Only the upper tail is clipped.  The lower tail is left unchanged
    because most experimentation metrics (revenue, clicks) are
    non-negative and right-skewed.

    Args:
        series: A numeric :class:`pandas.Series`.
        percentile: Upper quantile to clip at, in ``(0.5, 1.0]``.
            Set to ``1.0`` to disable winsorization (no-op).

    Returns:
        A new :class:`pandas.Series` with values above the ``percentile``
        quantile replaced by the quantile value.  ``NaN`` values are
        preserved.

    Raises:
        ValueError: If ``percentile`` is not in ``(0.5, 1.0]``.
    """
    if not 0.5 < percentile <= 1.0:
        raise ValueError(f"percentile must be in (0.5, 1.0], got {percentile}")
    if percentile == 1.0:
        return series.copy()
    upper = series.quantile(percentile)
    return series.clip(upper=upper)


def check_srm(
    n_control: int,
    n_treatment: int,
    expected_ratio: float = 0.5,
    alpha: float = 0.01,
) -> bool:
    """Check for a Sample Ratio Mismatch (SRM) using a chi-squared test.

    An SRM occurs when the observed allocation of users between control and
    treatment groups deviates significantly from the expected allocation.
    This is a sign that the randomisation or logging is broken.

    The test is:

    - H₀: The observed allocation matches the expected ratio.
    - H₁: The observed allocation differs from the expected ratio.

    We use ``alpha = 0.01`` by default (stricter than the usual 0.05) to
    reduce false positives, since SRM is a flag for investigation rather
    than a formal hypothesis test.

    Args:
        n_control: Number of users observed in the control group.
        n_treatment: Number of users observed in the treatment group.
        expected_ratio: Expected proportion of users in the control group.
            Defaults to ``0.5`` (50/50 split).
        alpha: Significance level for the chi-squared test.  Defaults to
            ``0.01``.

    Returns:
        ``True`` if an SRM is detected (the split deviates significantly
        from ``expected_ratio``); ``False`` otherwise.

    Raises:
        ValueError: If ``n_control`` or ``n_treatment`` is not positive,
            or if ``expected_ratio`` is not in ``(0, 1)``.
    """
    if n_control <= 0 or n_treatment <= 0:
        raise ValueError("n_control and n_treatment must both be positive")
    if not 0 < expected_ratio < 1:
        raise ValueError(f"expected_ratio must be in (0, 1), got {expected_ratio}")

    n_total = n_control + n_treatment
    expected_control = n_total * expected_ratio
    expected_treatment = n_total * (1 - expected_ratio)

    chi2_stat, p_value = stats.chisquare(
        f_obs=[n_control, n_treatment],
        f_exp=[expected_control, expected_treatment],
    )
    return float(p_value) < alpha
