# Integration Tests

Integration tests connect to a real warehouse and verify the full pipeline
end-to-end. They are marked with `@pytest.mark.integration` and are **not**
run in CI by default.

## Running integration tests

### Snowflake

Set the following environment variables, then run:

```bash
export ARGENTA_TEST_SNOWFLAKE_ACCOUNT=my_account.us-east-1
export ARGENTA_TEST_SNOWFLAKE_USER=argenta_test_user
export ARGENTA_TEST_SNOWFLAKE_PASSWORD=...
export ARGENTA_TEST_SNOWFLAKE_DATABASE=TEST_DB
export ARGENTA_TEST_SNOWFLAKE_SCHEMA=PUBLIC
export ARGENTA_TEST_SNOWFLAKE_WAREHOUSE=COMPUTE_WH

pytest tests/integration/test_snowflake_connector.py -v -m integration
```

### BigQuery

```bash
export ARGENTA_TEST_BIGQUERY_PROJECT=my-gcp-project
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

pytest tests/integration/test_bigquery_connector.py -v -m integration
```

### Redshift

```bash
export ARGENTA_TEST_REDSHIFT_HOST=cluster.abc123.us-east-1.redshift.amazonaws.com
export ARGENTA_TEST_REDSHIFT_DATABASE=dev
export ARGENTA_TEST_REDSHIFT_USER=argenta_test
export ARGENTA_TEST_REDSHIFT_PASSWORD=...

pytest tests/integration/test_redshift_connector.py -v -m integration
```

## What the tests verify

Each integration test:
1. Opens a connection using the connector
2. Executes a trivial `SELECT 1 AS n` query
3. Verifies the result is a DataFrame with the expected value
4. Checks `table_exists` with a known table and a non-existent table
5. Closes the connection cleanly

Full pipeline integration tests (running the SQL pipeline on real data) are
in `test_pipeline_integration.py` and require a pre-populated test dataset.
