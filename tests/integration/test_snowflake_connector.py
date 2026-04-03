"""Integration tests for the Snowflake connector.

Requires environment variables — see tests/integration/README.md.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from argenta.connectors.snowflake import SnowflakeConnector


def _snowflake_credentials() -> dict:
    return {
        "account":   os.environ["ARGENTA_TEST_SNOWFLAKE_ACCOUNT"],
        "user":      os.environ["ARGENTA_TEST_SNOWFLAKE_USER"],
        "password":  os.environ["ARGENTA_TEST_SNOWFLAKE_PASSWORD"],
        "database":  os.environ["ARGENTA_TEST_SNOWFLAKE_DATABASE"],
        "schema":    os.environ.get("ARGENTA_TEST_SNOWFLAKE_SCHEMA", "PUBLIC"),
        "warehouse": os.environ["ARGENTA_TEST_SNOWFLAKE_WAREHOUSE"],
    }


@pytest.mark.integration
class TestSnowflakeConnector:
    def test_connect_and_select_one(self) -> None:
        connector = SnowflakeConnector(_snowflake_credentials())
        with connector:
            df = connector.query("SELECT 1 AS n")
        assert isinstance(df, pd.DataFrame)
        assert int(df["n"].iloc[0]) == 1

    def test_table_exists_returns_false_for_nonexistent(self) -> None:
        connector = SnowflakeConnector(_snowflake_credentials())
        with connector:
            exists = connector.table_exists("PUBLIC", "__argenta_does_not_exist__")
        assert exists is False

    def test_context_manager_closes_connection(self) -> None:
        connector = SnowflakeConnector(_snowflake_credentials())
        with connector:
            pass
        assert connector._conn is None
