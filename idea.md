# Argenta — Causal ML Experimentation Platform

## What is this

A warehouse-native causal ML layer that sits on top of existing A/B testing infrastructure.

Most experimentation tools (Statsig, Optimizely, Eppo) answer one question:
> "Did the experiment work?" → ATE + p-value

Argenta answers:
> "For **whom** did it work, **why**, and **what should you do next**?"

---

## The core problem with existing tools

Every major platform computes:

```
ATE = E[Y|T=1] - E[Y|T=0]
```

That is the entire statistical product. They add variance reduction (CUPED) and sequential testing on top, but the output is always one number per metric.

What they don't do:
- Heterogeneous Treatment Effects (HTE) — who responded differently?
- CATE estimation — individual-level treatment effect
- Uplift modeling — score your entire user base, not just experiment participants
- Subgroup discovery — automatic, not manual slicing
- Recommendations — what to roll out, to whom, expected lift

---

## Architecture: Option C (Analytics Layer)

Argenta does **not** run the experiment. The client already does that.

Argenta connects to the client's warehouse **after** the experiment ran, and runs causal ML on top of existing data.

```
Client warehouse (Snowflake / BigQuery / Redshift / Databricks)
         │
         │  read-only connector
         ▼
Argenta pipeline (SQL generated + executed inside their warehouse)
  ├── exposure deduplication (first exposure wins)
  ├── outcome join (post-exposure only)
  └── user feature join
         │
         ▼
Argenta causal ML service (Python)
  ├── ATE + confidence intervals
  ├── CUPED (variance reduction)
  ├── Causal Forest → CATE per user
  ├── Uplift scores (score full user base)
  └── Segment HTE (automatic subgroup discovery)
         │
         ▼
Results written back to client warehouse
         │
         ▼
Argenta UI (experiment dashboard + HTE explorer + recommendations)
```

### Key principle
Data never leaves the client's warehouse. Argenta runs SQL inside their environment and writes results back to a dedicated schema.

---

## What Argenta needs from the client

Three tables (mapped during onboarding):

**1. Exposures**
```
user_id | experiment_id | variant | timestamp
```

**2. Events / outcomes**
```
user_id | event_name | value | timestamp
```

**3. User features**
```
user_id | feature_1 | feature_2 | ... | feature_n
```

The user features table is what unlocks the causal ML layer.

---

## Output tables written back to warehouse

```
argenta.experiment_results     — ATE, CI, p-value, recommendation per experiment
argenta.user_cate_scores       — individual treatment effect score per user
argenta.segment_effects        — HTE by segment (country, device, tenure, etc.)
```

---

## UI output (what the client sees)

```
Experiment: checkout_redesign
Overall ATE: +$2.30 per user  (p=0.003) ✓

Who responded best:
  ├── tenure > 180 days:     +$8.10  ← strongest effect
  ├── mobile users:          +$4.20
  ├── US users:              +$3.10
  ├── new users (<30 days):  -$1.20  ← harmed by treatment
  └── desktop + UK:          +$0.40  (not significant)

Recommendation:
  Roll out to:     high-tenure + mobile users (42% of base)
  Do not roll out: new users
  Expected lift if targeted: +$180k/month
```

---

## Statistical methods

| Method | Purpose |
|---|---|
| OLS / t-test | Baseline ATE |
| CUPED | Variance reduction via pre-experiment covariates |
| Delta method | Correct variance for ratio metrics (CTR, conversion rate) |
| Causal Forest (CausalForestDML) | Non-parametric CATE estimation |
| X-Learner | CATE with unbalanced treatment/control |
| R-Learner / DML | Doubly-robust CATE with regularization |
| Uplift modeling | Score users not in experiment |
| Sequential testing (mSPRT) | Always-valid p-values for peeking |
| Winsorization | Outlier handling for revenue metrics |
| Multiple testing correction | Bonferroni / BH-FDR across metrics |
| SRM detection | Sample Ratio Mismatch flagging |

Primary library: `econml` (Microsoft). Secondary: `dowhy`, `statsmodels`.

---

## What we are NOT building (scope boundary)

- No SDK — we do not run assignment
- No feature flags — client uses their existing tool
- No real-time assignment — all analysis is post-hoc
- No MAB / contextual bandits — requires controlling assignment (out of scope for Option C)
- No product analytics (funnels, retention dashboards) — not our product

---

## ICP (Ideal Customer Profile)

| Company type | Why they care |
|---|---|
| Marketplace / fintech | Heterogeneous user base, high cost of wrong rollout decision |
| E-commerce (pricing / checkout) | Individual price elasticity is a CATE problem |
| Growth teams at Series B-D | Have data scientists, no productized causal ML tooling |
| Teams already running experiments | Results sitting in warehouse, no HTE analysis |

Not a fit: early-stage startups with no experiment history or user feature data.

---

## Build order

```
Phase 1 — Core pipeline
  1. Warehouse connectors (Snowflake, BigQuery, Redshift)
  2. Schema mapping + onboarding config
  3. SQL pipeline generator (exposure dedup, outcome join, feature join)
  4. ATE + CI + CUPED (baseline stats)

Phase 2 — Causal ML layer
  5. Causal Forest CATE estimation (econml)
  6. Segment HTE (automatic subgroup discovery)
  7. Uplift scoring (score full user base)
  8. Results writer (back to warehouse)

Phase 3 — UI
  9. Experiment results dashboard
  10. HTE explorer (segment breakdown, CATE distribution)
  11. Recommendation engine (roll out to whom, expected lift)

Phase 4 — Polish
  12. SRM detection
  13. Sequential testing
  14. Multiple experiment support
  15. Scheduling (run analysis automatically on new data)
```

---

## Competitive positioning

| Tool | What they do | What they miss |
|---|---|---|
| Statsig | A/B + feature flags + CUPED | No HTE, no CATE, no uplift |
| Eppo | Warehouse-native A/B | No causal ML layer |
| Optimizely | A/B + personalization | Proprietary, expensive, no causal ML |
| GrowthBook | Open-source A/B | Basic stats only |
| **Argenta** | Causal ML on your existing experiments | Only the analysis layer — no assignment |

The pitch: **"You already run experiments. We tell you what your current tool can't."**
