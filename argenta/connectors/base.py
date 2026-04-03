"""Abstract base class for warehouse connectors.

All warehouse-specific connectors (Snowflake, BigQuery, Redshift) implement
:class:`BaseConnector`. The rest of the codebase is typed against
``BaseConnector`` only — concrete implementations are injected, never
imported directly in business logic.

Usage::

    from argenta.connectors import get_connector
    from argenta.config.schema import WarehouseConfig

    connector = get_connector(warehouse_config)
    with connector:
        df = connector.query("SELECT 1 AS n")
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseConnector(ABC):
    """Abstract warehouse connector defining the contract for all implementations.

    Concrete subclasses exist for Snowflake, BigQuery, and Redshift. All SQL
    execution — both read (``query``) and write (``execute``) — goes through
    this interface.

    Connectors support the context manager protocol::

        with connector:
            df = connector.query("SELECT user_id FROM my_table LIMIT 10")

    The context manager calls :meth:`connect` on entry and :meth:`disconnect`
    on exit (even if an exception is raised).
    """

    @abstractmethod
    def connect(self) -> None:
        """Open the warehouse connection.

        Must be called before any :meth:`query` or :meth:`execute` call.
        Implementations should store the connection object on ``self`` for
        reuse across subsequent calls.

        Raises:
            ConnectorError: If the connection cannot be established (bad
                credentials, network unreachable, insufficient privileges).
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the warehouse connection and release all resources.

        Safe to call even if :meth:`connect` was never called or failed.
        Implementations must not raise.
        """

    @abstractmethod
    def query(self, sql: str) -> pd.DataFrame:
        """Execute a SQL ``SELECT`` statement and return results as a DataFrame.

        Column names in the returned DataFrame match the ``SELECT`` clause
        aliases exactly, lowercased. The caller is responsible for ensuring
        the SQL is read-only; passing DDL or DML to ``query`` is undefined
        behaviour.

        Args:
            sql: A SQL ``SELECT`` statement.

        Returns:
            A :class:`pandas.DataFrame` with one row per result row and one
            column per ``SELECT`` expression. Returns an empty DataFrame (not
            ``None``) if the query produces zero rows.

        Raises:
            ConnectorError: If the query fails at the warehouse (syntax
                error, missing table, permission denied, etc.).
        """

    @abstractmethod
    def execute(self, sql: str) -> None:
        """Execute a SQL statement with no return value.

        Used for ``CREATE TABLE AS SELECT``, ``INSERT``, ``MERGE``, and DDL
        operations such as ``CREATE SCHEMA``.

        Args:
            sql: A SQL statement (DDL or DML).

        Raises:
            ConnectorError: If execution fails.
        """

    @abstractmethod
    def table_exists(self, schema: str, table: str) -> bool:
        """Check whether a table exists and is queryable in the warehouse.

        Args:
            schema: The schema (or dataset in BigQuery) to search within.
            table: The table name, without the schema prefix.

        Returns:
            ``True`` if the table exists and the connector user can query it;
            ``False`` otherwise.

        Raises:
            ConnectorError: If the existence check itself fails (e.g. the
                schema does not exist and the warehouse raises an error
                rather than returning an empty result).
        """

    def __enter__(self) -> "BaseConnector":
        """Open the connection when used as a context manager.

        Returns:
            The connector instance (``self``), allowing ``with connector as c:``
            syntax.
        """
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Close the connection on context manager exit.

        Always calls :meth:`disconnect`, even if the body of the ``with``
        block raised an exception.

        Args:
            exc_type: The exception class, or ``None`` if no exception.
            exc_val: The exception instance, or ``None``.
            exc_tb: The traceback object, or ``None``.
        """
        self.disconnect()


class ConnectorError(Exception):
    """Raised when a warehouse operation fails.

    Wraps the underlying driver exception and includes the SQL statement
    that triggered the failure to make debugging easier.

    Attributes:
        sql: The SQL string that caused the failure, or ``None`` if the
            error occurred during connection setup.
    """

    def __init__(self, message: str, sql: str | None = None) -> None:
        """Initialise a ``ConnectorError``.

        Args:
            message: Human-readable description of what went wrong.
            sql: The SQL statement that triggered the failure, if applicable.
        """
        super().__init__(message)
        self.sql = sql

    def __str__(self) -> str:
        base = super().__str__()
        if self.sql:
            # Truncate very long SQL for readability in tracebacks
            truncated = self.sql if len(self.sql) <= 500 else self.sql[:500] + "..."
            return f"{base}\n\nFailing SQL:\n{truncated}"
        return base
