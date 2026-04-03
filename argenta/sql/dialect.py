"""Warehouse dialect definitions and SQL helper utilities.

Different warehouses have slightly different SQL syntax for common operations.
This module centralises those differences so the Jinja2 templates and the
:class:`~argenta.sql.generator.SQLPipelineGenerator` remain dialect-aware
without scattering ``if warehouse == 'snowflake'`` conditionals everywhere.
"""

from __future__ import annotations

from enum import Enum


class WarehouseDialect(str, Enum):
    """Supported warehouse SQL dialects.

    The value of each member is a lowercase string matching the
    ``warehouse_type`` field in :class:`~argenta.config.schema.WarehouseConfig`.
    This makes it safe to use ``WarehouseDialect(config.warehouse.warehouse_type)``.
    """

    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"


def cast_timestamp(column: str, dialect: WarehouseDialect) -> str:
    """Return a SQL expression that casts ``column`` to a timestamp type.

    Different warehouses use different timestamp type names.

    Args:
        column: The column name or expression to cast.
        dialect: The target warehouse dialect.

    Returns:
        A SQL expression string such as ``"CAST(col AS TIMESTAMP)"`` or
        ``"CAST(col AS TIMESTAMP_NTZ)"`` depending on the dialect.
    """
    if dialect == WarehouseDialect.SNOWFLAKE:
        return f"CAST({column} AS TIMESTAMP_NTZ)"
    if dialect == WarehouseDialect.BIGQUERY:
        return f"CAST({column} AS TIMESTAMP)"
    # Redshift
    return f"CAST({column} AS TIMESTAMP)"


def create_table_as(
    fully_qualified_table: str,
    select_sql: str,
    dialect: WarehouseDialect,
    replace: bool = True,
) -> str:
    """Return a ``CREATE TABLE AS SELECT`` statement for the given dialect.

    Args:
        fully_qualified_table: The target table including schema prefix
            (e.g. ``'argenta.prepared_dataset_my_exp'``).
        select_sql: The ``SELECT`` statement whose results populate the table.
        dialect: The target warehouse dialect.
        replace: If ``True``, drop the existing table first (idempotent runs).
            Uses ``CREATE OR REPLACE TABLE`` on Snowflake, and ``DROP TABLE IF
            EXISTS`` + ``CREATE TABLE`` elsewhere.

    Returns:
        A complete DDL + DML statement string ready for execution.
    """
    if dialect == WarehouseDialect.SNOWFLAKE:
        keyword = "CREATE OR REPLACE TABLE" if replace else "CREATE TABLE"
        return f"{keyword} {fully_qualified_table} AS\n{select_sql}"

    if dialect == WarehouseDialect.BIGQUERY:
        if replace:
            return (
                f"CREATE OR REPLACE TABLE `{fully_qualified_table}` AS\n{select_sql}"
            )
        return f"CREATE TABLE `{fully_qualified_table}` AS\n{select_sql}"

    # Redshift
    if replace:
        drop = f"DROP TABLE IF EXISTS {fully_qualified_table};\n"
        return f"{drop}CREATE TABLE {fully_qualified_table} AS\n{select_sql}"
    return f"CREATE TABLE {fully_qualified_table} AS\n{select_sql}"


def qualify_table(table: str, dialect: WarehouseDialect) -> str:
    """Wrap a fully qualified table name in dialect-appropriate quoting.

    BigQuery requires backtick-quoting for project-scoped names.  Snowflake
    and Redshift use the name unquoted (assuming no reserved words).

    Args:
        table: The fully qualified table name.
        dialect: The target warehouse dialect.

    Returns:
        The table name wrapped in appropriate quotes for the dialect.
    """
    if dialect == WarehouseDialect.BIGQUERY:
        return f"`{table}`"
    return table


def row_number_dedup(
    partition_col: str,
    order_col: str,
    dialect: WarehouseDialect,
) -> str:
    """Return a window function expression for deduplication.

    Returns the standard ``ROW_NUMBER() OVER (PARTITION BY ... ORDER BY ...)``
    expression, which is supported identically across all three warehouses.

    Args:
        partition_col: Column to partition by (typically ``user_id``).
        order_col: Column to order by ascending (typically the timestamp).
        dialect: The target warehouse dialect (unused currently, included
            for future dialect-specific overrides).

    Returns:
        A SQL window function expression string.
    """
    return f"ROW_NUMBER() OVER (PARTITION BY {partition_col} ORDER BY {order_col} ASC)"
