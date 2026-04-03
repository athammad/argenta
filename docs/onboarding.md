# Onboarding a New Warehouse

This guide walks through connecting Argenta to a client data warehouse for
the first time.

---

## Prerequisites

1. The client runs experiments using any tool (Statsig, LaunchDarkly,
   homegrown feature flags). Argenta only needs the resulting data.

2. Three tables exist in the warehouse (names are configurable):
   - **Exposures** — which users saw which variant and when
   - **Events / outcomes** — user actions after exposure
   - **User features** — one row per user with attributes and pre-experiment metrics

3. A service account / user with:
   - `SELECT` on the three input tables
   - `CREATE TABLE`, `INSERT`, `DELETE` on the output schema
   - `CREATE SCHEMA` (or the schema already exists)

---

## Step 1: Install Argenta

```bash
# Snowflake
pip install "argenta[snowflake]"

# BigQuery
pip install "argenta[bigquery]"

# Redshift
pip install "argenta[redshift]"
```

---

## Step 2: Verify the input tables

Before writing the config, run these queries manually to verify the data:

**Exposures table**
```sql
SELECT experiment_id, variant, COUNT(*) AS n
FROM your_schema.exposures
WHERE experiment_id = 'your_experiment'
GROUP BY 1, 2
ORDER BY 1, 2;
```

Expected output: one row per variant, counts roughly equal.

**Events table**
```sql
SELECT event_name, COUNT(*) AS n
FROM your_schema.events
WHERE event_time > '2024-01-01'
GROUP BY 1
ORDER BY 2 DESC
LIMIT 20;
```

Note the event names you want to analyse — these go into `target_events`.

**User features table**
```sql
SELECT COUNT(*) AS total_users,
       COUNT(pre_revenue) AS users_with_covariate
FROM your_schema.user_dim;
```

Ensure most experiment users have a covariate value. Low coverage degrades
CUPED effectiveness.

---

## Step 3: Create the config file

```yaml
# argenta.yaml
warehouse:
  warehouse_type: snowflake          # or bigquery / redshift
  output_schema: argenta             # schema Argenta will write results to
  credentials:
    account: my_account.us-east-1
    user: argenta_svc
    password: "${SNOWFLAKE_PASSWORD}" # use env vars for secrets
    database: ANALYTICS
    schema: PUBLIC
    warehouse: COMPUTE_WH

exposures:
  table: ANALYTICS.PUBLIC.STATSIG_EXPOSURES
  user_id_col: user_id              # column holding user identifier
  experiment_id_col: experiment_name # column that names the experiment
  variant_col: group                 # column with 'control' / 'treatment'
  timestamp_col: exposure_time       # assignment timestamp

outcomes:
  table: ANALYTICS.PUBLIC.EVENTS
  user_id_col: user_id
  event_name_col: event_type
  value_col: revenue
  timestamp_col: event_time
  target_events:                     # events to analyse as metrics
    - purchase
    - add_to_cart

user_features:
  table: ANALYTICS.PUBLIC.USER_DIM
  user_id_col: user_id
  feature_cols:                      # columns passed to CATE in Phase 2
    - country
    - device_type
    - account_age_days
  covariate_col: pre_experiment_revenue  # pre-experiment metric for CUPED

experiment:
  experiment_id: checkout_redesign_2024
  control_variant: control
  treatment_variant: treatment
  alpha: 0.05
  winsorize_percentile: 0.99
  use_cuped: true
```

---

## Step 4: Run the analysis

```python
from argenta import load_config, PipelineRunner

config = load_config("argenta.yaml")
runner = PipelineRunner(config)
result = runner.run()

print(result.summary())
```

Or from the command line (once CLI is implemented):
```bash
argenta run --config argenta.yaml
```

---

## Step 5: Read the results

Results are written to `argenta.experiment_results` in your warehouse:

```sql
SELECT
    experiment_id,
    metric_name,
    ate,
    ci_low,
    ci_high,
    p_value,
    relative_lift,
    cuped_applied,
    srm_detected,
    run_at
FROM argenta.experiment_results
WHERE experiment_id = 'checkout_redesign_2024'
ORDER BY metric_name;
```

---

## Warehouse-specific notes

### Snowflake

- The `warehouse` credential key sets the virtual warehouse (compute cluster).
  Use a dedicated warehouse for Argenta if you want to isolate costs.
- For key-pair authentication, add `private_key_path` and optionally
  `private_key_passphrase` to the credentials dict.

### BigQuery

- Set `project` to the GCP project ID.
- Omit `credentials_path` to use Application Default Credentials (recommended
  for Cloud Run, GKE, etc.).
- For service account files, set `credentials_path: /path/to/key.json`.
- The `output_schema` maps to a BigQuery **dataset**.

### Redshift

- Required credentials: `host`, `database`, `user`, `password`.
- For Redshift Serverless, use `host` = the workgroup endpoint.
- For IAM authentication, add `iam: true`, `cluster_identifier`, `region`,
  and `db_user` to the credentials dict.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ConfigValidationError: covariate_col` | `use_cuped: true` but no `covariate_col` | Either set `covariate_col` or `use_cuped: false` |
| `ConnectorError: not installed` | Warehouse driver not installed | `pip install "argenta[snowflake]"` etc. |
| Empty result / no rows | `experiment_id` value not found in exposures table | Check `experiment_id_col` name and value |
| SRM detected | Assignment or logging bug | Investigate exposure counts per variant |
| CUPED failed / fell back to raw | Covariate has zero variance or is all NULL | Check `covariate_col` data quality |
