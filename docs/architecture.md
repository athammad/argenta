# Architecture

## Overview

Argenta is an analytics-only layer. It connects to an existing data warehouse,
runs a SQL pipeline inside that warehouse to build an analysis dataset, then
runs causal ML methods in Python and writes results back.

```
Your warehouse (Snowflake / BigQuery / Redshift)
         │
         │  read-only access + write to argenta schema
         ▼
┌────────────────────────────────────────────────────────┐
│  Argenta SQL pipeline (executes inside your warehouse) │
│                                                        │
│  Step 1: Exposure deduplication                        │
│    → first_exposures CTE                               │
│    → keeps first exposure per user per experiment      │
│                                                        │
│  Step 2: Outcome join                                  │
│    → user_outcomes CTE                                 │
│    → LEFT JOIN on user_id WHERE event_ts > exposure_ts │
│    → aggregates to one row per user                    │
│                                                        │
│  Step 3: User feature join                             │
│    → user_data CTE                                     │
│    → LEFT JOIN on user_id to add covariates            │
│                                                        │
│  CTAS → argenta.prepared_dataset_{experiment_id}       │
└────────────────────────────────────────────────────────┘
         │
         │  connector.query("SELECT * FROM argenta.prepared_dataset_...")
         ▼
┌──────────────────────────────────────────────┐
│  Argenta stats layer (Python, in memory)     │
│                                              │
│  1. Winsorization (clip upper tail)          │
│  2. CUPED (variance reduction via covariate) │
│  3. ATE = mean(treatment) - mean(control)    │
│  4. Welch CI + p-value                       │
│  5. SRM detection (chi-squared)              │
└──────────────────────────────────────────────┘
         │
         │  ResultsWriter.write_experiment_results(result)
         ▼
┌──────────────────────────────────────────────────────┐
│  Results written back to your warehouse              │
│                                                      │
│  argenta.experiment_results   — ATE, CI, p-value     │
│  argenta.user_cate_scores     — per-user CATE (Ph.2) │
│  argenta.segment_effects      — HTE by segment (Ph2) │
└──────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `argenta.config` | Config loading, validation (Pydantic v2) |
| `argenta.connectors` | Warehouse connection and SQL execution |
| `argenta.sql` | SQL string generation (Jinja2 templates) |
| `argenta.stats` | Statistical computation (pure pandas/scipy) |
| `argenta.writer` | Writing results back to warehouse |
| `argenta.pipeline` | End-to-end orchestration |

### Architectural invariants

1. `SQLPipelineGenerator` generates SQL strings only — never executes.
2. `BaseConnector` executes SQL — never generates it.
3. `PipelineRunner` orchestrates — no SQL logic, no stats logic.
4. `argenta.stats` has zero imports from `connectors` or `sql`.
5. Connectors are always injected, never instantiated inside business logic.

---

## Data Flow in Detail

### 1. Config loading

```python
config = load_config("argenta.yaml")
```

`ArgentoConfig` (Pydantic) holds all client-specific knowledge: table names,
column names, credentials, experiment parameters. Nothing else hardcodes
these values.

### 2. SQL pipeline execution

```python
generator = SQLPipelineGenerator(config, WarehouseDialect.SNOWFLAKE)
sql = generator.render_prepared_dataset()
connector.execute(sql)
```

The SQL is a single `CREATE OR REPLACE TABLE ... AS WITH <three CTEs> SELECT *`.
It runs entirely inside the client's warehouse. Argenta never sees individual
user rows — only the aggregated prepared dataset is pulled into memory.

### 3. Statistics

```python
df = connector.query(f"SELECT * FROM {prepared_dataset_table}")
result = runner._compute_result(df, experiment_id)
```

The DataFrame has one row per user. The stats layer:
- Winsorizes each metric column at the configured percentile
- Applies CUPED if a covariate column is present
- Runs `compute_ate` (Welch's t-test) per metric
- Runs `check_srm` once on the exposure counts

### 4. Results writing

```python
writer = ResultsWriter(connector, output_schema)
writer.write_experiment_results(result)
```

Uses `DELETE + INSERT` (not UPSERT) for maximum cross-warehouse compatibility.
The `experiment_results` table is created with `CREATE TABLE IF NOT EXISTS`
(idempotent).

---

## Directory Structure

```
argenta/
├── argenta/
│   ├── __init__.py          Public API
│   ├── config/
│   │   ├── schema.py        ArgentoConfig + sub-models
│   │   └── loader.py        YAML loading + env var interpolation
│   ├── connectors/
│   │   ├── base.py          BaseConnector ABC + ConnectorError
│   │   ├── snowflake.py
│   │   ├── bigquery.py
│   │   └── redshift.py
│   ├── sql/
│   │   ├── dialect.py       WarehouseDialect enum + SQL helpers
│   │   ├── generator.py     SQLPipelineGenerator
│   │   └── templates/       Jinja2 .sql.j2 template files
│   ├── stats/
│   │   ├── models.py        ExperimentResult, MetricResult
│   │   ├── ate.py           compute_ate, winsorize, check_srm
│   │   └── cuped.py         apply_cuped, variance_reduction_ratio
│   ├── writer/
│   │   └── results_writer.py ResultsWriter
│   └── pipeline/
│       └── runner.py        PipelineRunner
└── tests/
    ├── conftest.py           Shared fixtures
    ├── unit/                 No warehouse required
    └── integration/          Requires live credentials
```
