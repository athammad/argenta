# Statistics Reference

This document explains the statistical methods implemented in Argenta Phase 1.

---

## Average Treatment Effect (ATE)

The ATE is the primary output of every experiment analysis:

```
ATE = E[Y | T=1] - E[Y | T=0]
    = mean(Y_treatment) - mean(Y_control)
```

Where:
- `Y` is the outcome metric (e.g. revenue per user)
- `T=1` indicates the treatment group
- `T=0` indicates the control group

Under random assignment (which Argenta assumes), this is an unbiased estimate
of the causal effect of the treatment.

---

## Welch's t-test

Argenta uses **Welch's t-test** rather than Student's t-test because it does
not assume equal variances between groups.

For two independent samples with means `ȳ₁`, `ȳ₂`, sample variances `s₁²`, `s₂²`,
and sizes `n₁`, `n₂`:

**Test statistic:**
```
t = (ȳ₁ - ȳ₂) / sqrt(s₁²/n₁ + s₂²/n₂)
```

**Welch-Satterthwaite degrees of freedom:**
```
df = (s₁²/n₁ + s₂²/n₂)²
     / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
```

**Confidence interval:**
```
ATE ± t_{df, 1-α/2} * SE

where SE = sqrt(s₁²/n₁ + s₂²/n₂)
```

Why Welch over Student's:
- Revenue metrics often have very different variances across groups (e.g.
  one group has a new pricing model).
- Welch is more robust to unequal sample sizes (SRM, gradual rollout).
- The cost (slightly lower power when variances are equal) is negligible.

---

## Winsorization

Revenue and value metrics are right-skewed — a small number of very high-value
users can dominate the mean and inflate variance.

Winsorization clips the upper tail at the `p`-th percentile:

```
Y_winsorized = min(Y, quantile(Y, p))
```

Argenta applies winsorization **symmetrically across groups** using the pooled
distribution (not per-group), so the same threshold is applied to both control
and treatment. This prevents the threshold from being influenced by the treatment
itself.

Default: `p = 0.99` (clips the top 1%).

Set `winsorize_percentile: 1.0` in the config to disable.

---

## CUPED (Controlled-experiment Using Pre-Experiment Data)

CUPED reduces the variance of the ATE estimator by removing the component
of variance explained by a pre-experiment covariate.

### Derivation

Let `Y` be the post-experiment outcome and `X` be a pre-experiment covariate.

The CUPED-adjusted outcome is:

```
Y_cuped = Y - θ * (X - E[X])
```

where:

```
θ = Cov(Y, X) / Var(X)
```

**Why the mean is preserved:**

```
E[Y_cuped] = E[Y] - θ * (E[X] - E[X]) = E[Y]
```

The ATE estimate is identical. Only the variance changes.

**Variance reduction:**

```
Var(Y_cuped) = Var(Y) - 2θ Cov(Y,X) + θ² Var(X)
             = Var(Y) - Cov(Y,X)² / Var(X)
             = Var(Y) * (1 - ρ²)
```

where `ρ = Corr(Y, X)`.

A covariate with `|ρ| = 0.7` reduces variance by `1 - 0.49 = 51%`, roughly
doubling the effective sample size.

### Implementation

`θ` is estimated by pooling control and treatment together **before** assigning
treatment status. This avoids post-stratification bias.

```python
theta = Cov(Y_pooled, X_pooled) / Var(X_pooled)
Y_cuped = Y - theta * (X - mean(X_pooled))
```

### Good covariates

- The same metric measured in the pre-experiment window (e.g. revenue in the
  30 days before the experiment started).
- High correlation with the outcome metric.
- Must be pre-experiment (not affected by the treatment).

### Reference

Deng, A., Xu, Y., Kohavi, R., & Walker, T. (2013). Improving the Sensitivity of
Online Controlled Experiments by Utilizing Pre-Experiment Data. *WSDM '13*.

---

## Sample Ratio Mismatch (SRM)

An SRM occurs when the observed allocation of users differs significantly from
the expected allocation.

**Test:**

```
H₀: observed allocation matches expected ratio
H₁: observed allocation differs from expected ratio
```

Using a chi-squared goodness-of-fit test:

```
χ² = Σ (observed - expected)² / expected
```

Argenta uses `α = 0.01` (stricter than 0.05) because an SRM is a flag for
investigation, not a formal hypothesis test result.

**What causes SRM:**
- Logging bugs (some variants log more events than others)
- Caching issues (users are served cached pages without triggering logging)
- Bot traffic (bots may behave differently across variants)
- SDK configuration errors

**What to do if SRM is detected:**
- Inspect the raw exposure counts in the exposures table
- Check whether a subset of users (e.g. mobile-only, specific countries) drives the imbalance
- Do not interpret experiment results until the SRM is resolved

---

## Multiple Testing Correction (Phase 2)

When analysing many metrics simultaneously, the probability of at least one
false positive (Type I error) increases.

Phase 2 will implement:
- **Bonferroni correction**: `α_adjusted = α / m` (conservative)
- **Benjamini-Hochberg FDR**: controls the false discovery rate (recommended
  for exploratory analysis with many metrics)
