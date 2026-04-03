"""Redshift warehouse connector for Argenta.

Requires the ``redshift-connector`` package::

    pip install "argenta[redshift]"

Credentials are passed as a ``dict`` via ``WarehouseConfig.credentials``.
Required keys:

- ``host`` — Redshift cluster endpoint (e.g. ``'cluster.abc123.us-east-1.redshift.amazonaws.com'``)
- ``database`` — Database name
- ``user`` — Database user
- ``password`` — Password

Optional keys:

- ``port`` — Port number (default: ``5439``)
- ``ssl`` — Whether to use SSL (default: ``True``)
- ``sslmode`` — SSL mode (default: ``'verify-ca'``)

For IAM-based authentication (Redshift Serverless, IAM roles), see
``redshift-connector`` documentation for the ``iam``, ``db_user``,
``cluster_identifier``, ``region``, and ``profile`` parameters, all of which
can be passed through the credentials dict.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from argenta.connectors.base import BaseConnector, ConnectorError

logger = logging.getLogger(__name__)


class RedshiftConnector(BaseConnector):
    """Warehouse connector for Amazon Redshift.

    Uses ``redshift-connector`` under the hood. Maintains a single persistent
    connection for the lifetime of the connector instance.

    Args:
        credentials: A dictionary of Redshift connection parameters.
            See module docstring for required and optional keys.

    Raises:
        ConnectorError: If ``redshift-connector`` is not installed, or if
            the connection cannot be established in :meth:`connect`.
    """

    def __init__(self, credentials: dict[str, Any]) -> None:
        self._credentials = credentials
        self._conn: Any = None  # redshift_connector.Connection

    def connect(self) -> None:
        """Open the Redshift connection.

        Raises:
            ConnectorError: If ``redshift-connector`` is not installed or
                if authentication fails.
        """
        try:
            import redshift_connector  # type: ignore[import]
        except ImportError as exc:
            raise ConnectorError(
                "redshift-connector is not installed. "
                "Install it with: pip install 'argenta[redshift]'"
            ) from exc

        try:
            logger.info("[REDSHIFT] Opening connection to host: %s", self._credentials.get("host"))
            self._conn = redshift_connector.connect(**self._credentials)
            self._conn.autocommit = True
            logger.info("[REDSHIFT] Connection established")
        except Exception as exc:
            raise ConnectorError(f"Failed to connect to Redshift: {exc}") from exc

    def disconnect(self) -> None:
        """Close the Redshift connection."""
        if self._conn is not None:
            try:
                self._conn.close()
                logger.info("[REDSHIFT] Connection closed")
            except Exception:
                pass
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
        """Check whether a table exists in the given Redshift schema.

        Args:
            schema: The Redshift schema name.
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
                "RedshiftConnector is not connected. Call connect() or use the context manager."
            )
