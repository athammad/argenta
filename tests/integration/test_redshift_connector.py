"""Integration tests for the Redshift connector.

Requires environment variables — see tests/integration/README.md.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from argenta.connectors.redshift import RedshiftConnector


def _redshift_credentials() -> dict:
    return {
        "host":     os.environ["ARGENTA_TEST_REDSHIFT_HOST"],
        "database": os.environ["ARGENTA_TEST_REDSHIFT_DATABASE"],
        "user":     os.environ["ARGENTA_TEST_REDSHIFT_USER"],
        "password": os.environ["ARGENTA_TEST_REDSHIFT_PASSWORD"],
    }


@pytest.mark.integration
class TestRedshiftConnector:
    def test_connect_and_select_one(self) -> None:
        connector = RedshiftConnector(_redshift_credentials())
        with connector:
            df = connector.query("SELECT 1 AS n")
        assert isinstance(df, pd.DataFrame)
        assert int(df["n"].iloc[0]) == 1

    def test_table_exists_returns_false_for_nonexistent(self) -> None:
        connector = RedshiftConnector(_redshift_credentials())
        with connector:
            exists = connector.table_exists("public", "__argenta_does_not_exist__")
        assert exists is False

    def test_context_manager_closes_connection(self) -> None:
        connector = RedshiftConnector(_redshift_credentials())
        with connector:
            pass
        assert connector._conn is None
