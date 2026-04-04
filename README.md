<p align="center">
  <img src="assets/logo.svg" alt="Argenta" width="420"/>
</p>

<p align="center">
  <a href="https://pypi.org/project/argenta/"><img src="https://img.shields.io/pypi/v/argenta?color=blue&style=flat-square" alt="PyPI version"/></a>
  <a href="https://pypi.org/project/argenta/"><img src="https://img.shields.io/pypi/pyversions/argenta?style=flat-square" alt="Python"/></a>
  <a href="https://github.com/athammad/argenta/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/athammad/argenta/ci.yml?branch=master&label=CI&style=flat-square" alt="CI"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square" alt="License"/></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/badge/code%20style-ruff-orange?style=flat-square" alt="Code style: ruff"/></a>
</p>

**Warehouse-native causal ML for experimentation.**

Most A/B testing tools answer one question: *Did the experiment work?*

Argenta answers: *For **whom** did it work, **why**, and **what should you do next**?*

---

## What Argenta Is

Argenta is an analytics layer that connects to your existing data warehouse, runs a SQL pipeline
inside it to construct an experiment dataset, and then applies causal ML methods to produce
results your current tool doesn't: heterogeneous treatment effects (HTE), per-user CATE scores,
automatic subgroup discovery, and targeted rollout recommendations.

It works with experiments you've already run — using whatever tool assigned variants (Statsig,
LaunchDarkly, Optimizely, homegrown feature flags). Argenta only analyzes.

## What Argenta Is NOT

- **Not an A/B testing platform.** Argenta does not assign variants or run feature flags.
- **Not a real-time system.** Analysis runs post-hoc, after experiments have collected data.
- **Not a data pipeline tool.** Argenta does not move or replicate your data. SQL runs inside
  your warehouse; results are written back to a schema you control.
- **Not a MAB / contextual bandit system.** Adaptive assignment requires controlling the
  assignment layer, which is out of scope.

---

## Quick Start

### 1. Install

```bash
# Pick your warehouse:
pip install "argenta[snowflake]"
pip install "argenta[bigquery]"
pip install "argenta[redshift]"
pip install "argenta[all]"   # all warehouses
```

### 2. Create a config file

```yaml
# argenta.yaml
warehouse:
  warehouse_type: snowflake
  output_schema: argenta
  credentials:
    account: my_account
    user: argenta_svc
    password: "${SNOWFLAKE_PASSWORD}"
    database: ANALYTICS
    schema: PUBLIC
    warehouse: COMPUTE_WH

exposures:
  table: ANALYTICS.PUBLIC.STATSIG_EXPOSURES
  user_id_col: user_id
  experiment_id_col: experiment_name
  variant_col: group
  timestamp_col: exposure_time

outcomes:
  table: ANALYTICS.PUBLIC.EVENTS
  user_id_col: user_id
  event_name_col: event_type
  value_col: revenue
  timestamp_col: event_time
  target_events:
    - purchase
    - add_to_cart

user_features:
  table: ANALYTICS.PUBLIC.USER_DIM
  user_id_col: user_id
  feature_cols:
    - country
    - device_type
    - account_age_days
  covariate_col: pre_experiment_revenue

experiment:
  experiment_id: checkout_redesign_2024
  control_variant: control
  treatment_variant: treatment
  alpha: 0.05
  winsorize_percentile: 0.99
  use_cuped: true
```

### 3. Run

```python
from argenta import ArgentoConfig, PipelineRunner
from argenta.config.loader import load_config

config = load_config("argenta.yaml")
runner = PipelineRunner(config)
result = runner.run("checkout_redesign_2024")

print(result)
```

Results are also written back to your warehouse at `argenta.experiment_results`.

---

## Architecture

```
Your warehouse (Snowflake / BigQuery / Redshift)
         │
         │  read-only + write to argenta schema
         ▼
Argenta SQL pipeline (runs INSIDE your warehouse)
  ├── 1. Exposure deduplication   — first exposure per user wins
  ├── 2. Outcome join             — only events after first exposure
  └── 3. User feature join        — covariates for CUPED + future CATE
         │
         ▼
Argenta stats layer (Python)
  ├── ATE + Welch CI + p-value
  ├── Winsorization
  ├── SRM detection
  └── CUPED variance reduction
         │
         ▼
Results written back to your warehouse
  ├── argenta.experiment_results   — per-metric ATE, CI, p-value
  ├── argenta.user_cate_scores     — per-user CATE (Phase 2)
  └── argenta.segment_effects      — HTE by segment (Phase 2)
```

---

## Statistical Methods

| Method | Purpose |
|---|---|
| Welch's t-test | ATE + p-value |
| Confidence intervals | 95% CI on ATE |
| Winsorization | Outlier handling for revenue metrics |
| SRM detection | Sample ratio mismatch check |
| CUPED | Variance reduction via pre-experiment covariate |
| Causal Forest (CausalForestDML) | Non-parametric CATE estimation |
| X-Learner | CATE with unbalanced treatment/control |
| Uplift modeling | Score users not in experiment |
| Sequential testing (mSPRT) | Always-valid p-values |
| Multiple testing correction | Bonferroni / BH-FDR across metrics |

---

## Input Tables Required

Argenta needs three tables in your warehouse (all column names are configurable):

| Table | Required columns |
|---|---|
| Exposures | `user_id`, `experiment_id`, `variant`, `timestamp` |
| Outcomes / events | `user_id`, `event_name`, `value`, `timestamp` |
| User features | `user_id`, + any feature columns |

The user features table is required for CUPED variance reduction. Without it, set `use_cuped: false`.

---

## Output Tables

Argenta writes results back to `{output_schema}` in your warehouse (default: `argenta`):

| Table | Contents |
|---|---|
| `argenta.experiment_results` | ATE, CI, p-value, SRM flag per metric |
| `argenta.user_cate_scores` | Per-user CATE score (Phase 2) |
| `argenta.segment_effects` | HTE by segment (Phase 2) |

---

## Supported Warehouses

| Warehouse | Extra | Status |
|---|---|---|
| Snowflake | `argenta[snowflake]` | Supported |
| BigQuery | `argenta[bigquery]` | Supported |
| Redshift | `argenta[redshift]` | Supported |
| Databricks | `argenta[databricks]` | Planned |

---

## Documentation

- [Architecture](docs/architecture.md)
- [Onboarding a new warehouse](docs/onboarding.md)
- [SQL pipeline explained](docs/sql_pipeline.md)
- [Statistics reference](docs/statistics.md)
- [Contributing](CONTRIBUTING.md)

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions,
coding conventions, and the PR process.

---

## License

Apache 2.0. See [LICENSE](LICENSE).

---

## Name

The name "Argenta" is inspired by the name of professor who introduced me to causal inference
during my bachelor's degree.
