"""Snowflake warehouse connector for Argenta.

Requires the ``snowflake-connector-python`` package::

    pip install "argenta[snowflake]"

Credentials are passed as a ``dict`` via ``WarehouseConfig.credentials``.
Required keys:

- ``account`` — Snowflake account identifier (e.g. ``'xy12345.us-east-1'``)
- ``user`` — Snowflake username
- ``password`` — Password (or use ``private_key_path`` for key-pair auth)
- ``database`` — Default database
- ``schema`` — Default schema
- ``warehouse`` — Virtual warehouse to use for compute

Optional keys:

- ``role`` — Snowflake role to assume (defaults to the user's default role)
- ``private_key_path`` — Path to the PEM private key file (key-pair auth)
- ``private_key_passphrase`` — Passphrase for the private key, if encrypted
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from argenta.connectors.base import BaseConnector, ConnectorError

logger = logging.getLogger(__name__)


class SnowflakeConnector(BaseConnector):
    """Warehouse connector for Snowflake.

    Uses ``snowflake-connector-python`` under the hood. The connection is
    kept open for the lifetime of the connector (or until :meth:`disconnect`
    is called). All queries run against the database / schema / warehouse
    specified in the credentials dict.

    Args:
        credentials: A dictionary of Snowflake connection parameters.
            See module docstring for required and optional keys.

    Raises:
        ConnectorError: If ``snowflake-connector-python`` is not installed,
            or if the connection cannot be established in :meth:`connect`.
    """

    def __init__(self, credentials: dict[str, Any]) -> None:
        self._credentials = credentials
        self._conn: Any = None  # snowflake.connector.SnowflakeConnection

    def connect(self) -> None:
        """Open the Snowflake connection.

        Raises:
            ConnectorError: If ``snowflake-connector-python`` is not installed
                or if authentication fails.
        """
        try:
            import snowflake.connector  # type: ignore[import]
        except ImportError as exc:
            raise ConnectorError(
                "snowflake-connector-python is not installed. "
                "Install it with: pip install 'argenta[snowflake]'"
            ) from exc

        try:
            logger.info("[SNOWFLAKE] Opening connection to account: %s", self._credentials.get("account"))
            self._conn = snowflake.connector.connect(**self._credentials)
            logger.info("[SNOWFLAKE] Connection established")
        except Exception as exc:
            raise ConnectorError(f"Failed to connect to Snowflake: {exc}") from exc

    def disconnect(self) -> None:
        """Close the Snowflake connection."""
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info("[SNOWFLAKE] Connection closed")
            except Exception:
                pass  # Always safe to swallow disconnect errors
            finally:
                self._conn = None

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SELECT and return results as a DataFrame.

        Args:
            sql: A SQL SELECT statement.

        Returns:
            A DataFrame with lowercase column names.

        Raises:
            ConnectorError: If the query fails or the connection is closed.
        """
        self._assert_connected()
        try:
            cursor = self._conn.cursor()
            cursor.execute(sql)
            columns = [desc[0].lower() for desc in cursor.description]
            rows = cursor.fetchall()
            return pd.DataFrame(rows, columns=columns)
        except Exception as exc:
            raise ConnectorError(str(exc), sql=sql) from exc

    def execute(self, sql: str) -> None:
        """Execute a DDL or DML statement with no return value.

        Args:
            sql: A SQL statement (CREATE TABLE AS, INSERT, etc.).

        Raises:
            ConnectorError: If execution fails.
        """
        self._assert_connected()
        try:
            cursor = self._conn.cursor()
            cursor.execute(sql)
        except Exception as exc:
            raise ConnectorError(str(exc), sql=sql) from exc

    def table_exists(self, schema: str, table: str) -> bool:
        """Check whether a table exists in the given Snowflake schema.

        Args:
            schema: The Snowflake schema name (without database prefix).
            table: The table name.

        Returns:
            ``True`` if the table exists.

        Raises:
            ConnectorError: If the information schema query fails.
        """
        sql = f"""
            SELECT COUNT(*) AS n
            FROM information_schema.tables
            WHERE LOWER(table_schema) = LOWER('{schema}')
              AND LOWER(table_name)   = LOWER('{table}')
        """
        df = self.query(sql)
        return int(df["n"].iloc[0]) > 0

    def _assert_connected(self) -> None:
        if self._conn is None:
            raise ConnectorError(
                "SnowflakeConnector is not connected. Call connect() or use the context manager."
            )
