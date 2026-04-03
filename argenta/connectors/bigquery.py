"""BigQuery warehouse connector for Argenta.

Requires the ``google-cloud-bigquery`` and ``db-dtypes`` packages::

    pip install "argenta[bigquery]"

Credentials are passed as a ``dict`` via ``WarehouseConfig.credentials``.
Required keys:

- ``project`` — GCP project ID (e.g. ``'my-gcp-project'``)

Optional keys:

- ``credentials_path`` — Path to a service account JSON key file. If
  omitted, Application Default Credentials (ADC) are used (recommended for
  production; works with Workload Identity, Cloud Run, GKE, etc.).
- ``location`` — BigQuery dataset location (e.g. ``'US'``, ``'EU'``).
  Defaults to ``'US'``.

In BigQuery, ``schema`` maps to a **dataset**. Fully qualified table names
use the format ``project.dataset.table``.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from argenta.connectors.base import BaseConnector, ConnectorError

logger = logging.getLogger(__name__)


class BigQueryConnector(BaseConnector):
    """Warehouse connector for Google BigQuery.

    Uses ``google-cloud-bigquery`` under the hood. The BigQuery client is
    stateless by design, so :meth:`connect` and :meth:`disconnect` manage a
    lightweight client object rather than a persistent TCP connection.

    Args:
        credentials: A dictionary of BigQuery connection parameters.
            See module docstring for required and optional keys.

    Raises:
        ConnectorError: If ``google-cloud-bigquery`` is not installed,
            or if client creation fails in :meth:`connect`.
    """

    def __init__(self, credentials: dict[str, Any]) -> None:
        self._credentials = credentials
        self._client: Any = None  # google.cloud.bigquery.Client

    def connect(self) -> None:
        """Initialise the BigQuery client.

        Raises:
            ConnectorError: If ``google-cloud-bigquery`` is not installed or
                if the client cannot be created (bad project ID, no ADC, etc.).
        """
        try:
            from google.cloud import bigquery  # type: ignore[import]
            from google.oauth2 import service_account  # type: ignore[import]
        except ImportError as exc:
            raise ConnectorError(
                "google-cloud-bigquery is not installed. "
                "Install it with: pip install 'argenta[bigquery]'"
            ) from exc

        project = self._credentials.get("project")
        credentials_path = self._credentials.get("credentials_path")
        location = self._credentials.get("location", "US")

        try:
            logger.info("[BIGQUERY] Creating client for project: %s", project)
            if credentials_path:
                creds = service_account.Credentials.from_service_account_file(
                    credentials_path,
                    scopes=["https://www.googleapis.com/auth/cloud-platform"],
                )
                self._client = bigquery.Client(
                    project=project, credentials=creds, location=location
                )
            else:
                self._client = bigquery.Client(project=project, location=location)
            logger.info("[BIGQUERY] Client ready")
        except Exception as exc:
            raise ConnectorError(f"Failed to create BigQuery client: {exc}") from exc

    def disconnect(self) -> None:
        """Close the BigQuery client."""
        if self._client is not None:
            try:
                self._client.close()
                logger.info("[BIGQUERY] Client closed")
            except Exception:
                pass
            finally:
                self._client = None

    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as a DataFrame.

        Args:
            sql: A SQL SELECT statement (Standard SQL dialect).

        Returns:
            A DataFrame with lowercase column names.

        Raises:
            ConnectorError: If the query fails or the client is not initialised.
        """
        self._assert_connected()
        try:
            df = self._client.query(sql).to_dataframe()
            df.columns = [c.lower() for c in df.columns]
            return df
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
            job = self._client.query(sql)
            job.result()  # Block until complete
        except Exception as exc:
            raise ConnectorError(str(exc), sql=sql) from exc

    def table_exists(self, schema: str, table: str) -> bool:
        """Check whether a table (or view) exists in the given BigQuery dataset.

        In BigQuery, ``schema`` corresponds to a dataset name.

        Args:
            schema: The BigQuery dataset name.
            table: The table name.

        Returns:
            ``True`` if the table exists.

        Raises:
            ConnectorError: If the query fails.
        """
        project = self._credentials.get("project", "")
        sql = f"""
            SELECT COUNT(*) AS n
            FROM `{project}`.{schema}.INFORMATION_SCHEMA.TABLES
            WHERE LOWER(table_name) = LOWER('{table}')
        """
        df = self.query(sql)
        return int(df["n"].iloc[0]) > 0

    def _assert_connected(self) -> None:
        if self._client is None:
            raise ConnectorError(
                "BigQueryConnector is not connected. Call connect() or use the context manager."
            )
