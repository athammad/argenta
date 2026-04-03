# SQL Pipeline Reference

Argenta generates a single `CREATE TABLE AS SELECT` statement that runs inside
the client's warehouse. This document explains each step.

---

## Overview

The pipeline assembles three CTEs into one query:

```sql
CREATE OR REPLACE TABLE argenta.prepared_dataset_my_exp AS
WITH
first_exposures AS ( ... ),   -- Step 1: dedup
user_outcomes   AS ( ... ),   -- Step 2: outcome join
user_data       AS ( ... )    -- Step 3: feature join
SELECT * FROM user_data
```

The resulting table has one row per user with:
- `variant` — control or treatment
- one column per configured metric (aggregated post-exposure value)
- feature columns for CUPED and Phase 2 CATE

---

## Step 1: Exposure Deduplication (`first_exposures`)

**Why deduplication is necessary:**

Users can appear multiple times in the exposures table. This happens when:
- The SDK logs an exposure on every page view (not just the first)
- The user clears cookies and gets re-assigned
- Logging bugs create duplicate rows

The standard causal inference approach is to use the **first exposure** as the
randomisation event. This is the moment the user was definitively assigned to
a variant.

**SQL logic:**

```sql
first_exposures AS (
    SELECT user_id, variant, CAST(timestamp AS TIMESTAMP_NTZ) AS exposure_ts
    FROM (
        SELECT user_id, variant, timestamp,
               ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp ASC) AS rn
        FROM exposures
        WHERE experiment_id = 'my_experiment'
          AND variant IN ('control', 'treatment')
    ) ranked
    WHERE rn = 1
)
```

Key points:
- `ROW_NUMBER()` ordered by `timestamp ASC` gives the earliest row `rn = 1`
- Filtered to the two variants being compared (ignores holdout, other arms)
- Result: exactly one row per user

---

## Step 2: Outcome Join (`user_outcomes`)

**Why we filter to post-exposure events only:**

Including pre-exposure events would contaminate the analysis. If user A made
a purchase before seeing the experiment, that purchase tells us nothing about
the treatment effect.

The causal window is `[exposure_ts, ∞)`.

**SQL logic:**

```sql
user_outcomes AS (
    SELECT
        e.user_id,
        e.variant,
        COALESCE(SUM(CASE WHEN LOWER(o.event_name) = 'purchase' THEN o.revenue END), 0) AS purchase
    FROM first_exposures e
    LEFT JOIN events o
        ON  o.user_id = e.user_id
        AND CAST(o.event_time AS TIMESTAMP_NTZ) > e.exposure_ts
        AND LOWER(o.event_name) IN ('purchase')
    GROUP BY e.user_id, e.variant
)
```

Key points:
- `LEFT JOIN` ensures users who never converted still appear with a `0` metric value
- `CAST(event_time) > exposure_ts` — strict inequality (event must be strictly after exposure)
- `COALESCE(..., 0)` — NULL from LEFT JOIN becomes 0 (zero events = zero revenue)
- One column per `target_event` — each event becomes its own metric column
- Aggregated to user level — one row per user

---

## Step 3: User Feature Join (`user_data`)

**Why features matter:**

Features serve two purposes:
1. CUPED covariate — reduces variance of the ATE estimator
2. Future CATE inputs — Phase 2 will use these to estimate heterogeneous effects

**SQL logic:**

```sql
user_data AS (
    SELECT
        o.*,
        f.country,
        f.device_type,
        f.account_age_days,
        f.pre_experiment_revenue
    FROM user_outcomes o
    LEFT JOIN user_dim f
        ON f.user_id = o.user_id
)
```

Key points:
- `LEFT JOIN` — users not in the features table still appear (with NULL features)
- The stats layer handles NULL covariates gracefully (CUPED skips them)
- Feature columns are passed through `SELECT *` to the final prepared dataset

---

## Dialect differences

| Feature | Snowflake | BigQuery | Redshift |
|---|---|---|---|
| CTAS keyword | `CREATE OR REPLACE TABLE` | `CREATE OR REPLACE TABLE` | `DROP TABLE IF EXISTS` + `CREATE TABLE` |
| Timestamp cast | `CAST(x AS TIMESTAMP_NTZ)` | `CAST(x AS TIMESTAMP)` | `CAST(x AS TIMESTAMP)` |
| Table quoting | No quoting needed | Backtick: `` `project.dataset.table` `` | No quoting needed |

---

## Inspecting the generated SQL

You can render the SQL without executing it:

```python
from argenta.config.loader import load_config
from argenta.sql.dialect import WarehouseDialect
from argenta.sql.generator import SQLPipelineGenerator

config = load_config("argenta.yaml")
gen = SQLPipelineGenerator(config, WarehouseDialect.SNOWFLAKE)

print(gen.render_prepared_dataset())
```

This is useful for:
- Auditing what Argenta will run before giving it warehouse access
- Debugging unexpected results
- Copying the SQL to run manually in a notebook
