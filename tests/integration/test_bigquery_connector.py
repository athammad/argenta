"""Integration tests for the BigQuery connector.

Requires environment variables — see tests/integration/README.md.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from argenta.connectors.bigquery import BigQueryConnector


def _bq_credentials() -> dict:
    creds: dict = {"project": os.environ["ARGENTA_TEST_BIGQUERY_PROJECT"]}
    if key_path := os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        creds["credentials_path"] = key_path
    return creds


@pytest.mark.integration
class TestBigQueryConnector:
    def test_connect_and_select_one(self) -> None:
        connector = BigQueryConnector(_bq_credentials())
        with connector:
            df = connector.query("SELECT 1 AS n")
        assert isinstance(df, pd.DataFrame)
        assert int(df["n"].iloc[0]) == 1

    def test_table_exists_returns_false_for_nonexistent(self) -> None:
        connector = BigQueryConnector(_bq_credentials())
        project = os.environ["ARGENTA_TEST_BIGQUERY_PROJECT"]
        with connector:
            exists = connector.table_exists(f"{project}.argenta", "__does_not_exist__")
        assert exists is False

    def test_context_manager_closes_client(self) -> None:
        connector = BigQueryConnector(_bq_credentials())
        with connector:
            pass
        assert connector._client is None
