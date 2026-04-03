"""Warehouse connector factory and public connector interface.

Import :func:`get_connector` to obtain the correct connector for a given
:class:`~argenta.config.schema.WarehouseConfig` without importing any
warehouse-specific driver directly::

    from argenta.connectors import get_connector
    from argenta.config.loader import load_config

    config = load_config("argenta.yaml")
    connector = get_connector(config.warehouse)
    with connector:
        df = connector.query("SELECT 1 AS n")
"""

from __future__ import annotations

from argenta.connectors.base import BaseConnector, ConnectorError
from argenta.config.schema import WarehouseConfig


def get_connector(config: WarehouseConfig) -> BaseConnector:
    """Instantiate and return the correct warehouse connector.

    Performs a **lazy import** of the connector module so that warehouse
    drivers not installed by the user do not cause an ``ImportError`` at
    import time. The ``ImportError`` is deferred until :meth:`connect` is
    called, at which point a :class:`~argenta.connectors.base.ConnectorError`
    is raised with a clear installation hint.

    Args:
        config: The ``warehouse`` section of :class:`~argenta.config.schema.ArgentoConfig`.

    Returns:
        An uninitialised :class:`BaseConnector` subclass instance for the
        specified warehouse type. Call :meth:`~BaseConnector.connect` (or
        use as a context manager) before issuing queries.

    Raises:
        ValueError: If ``config.warehouse_type`` is not one of the supported
            values (``'snowflake'``, ``'bigquery'``, ``'redshift'``).

    Example::

        connector = get_connector(config.warehouse)
        with connector:
            df = connector.query("SELECT COUNT(*) AS n FROM my_table")
    """
    wtype = config.warehouse_type

    if wtype == "snowflake":
        from argenta.connectors.snowflake import SnowflakeConnector
        return SnowflakeConnector(dict(config.credentials))

    if wtype == "bigquery":
        from argenta.connectors.bigquery import BigQueryConnector
        return BigQueryConnector(dict(config.credentials))

    if wtype == "redshift":
        from argenta.connectors.redshift import RedshiftConnector
        return RedshiftConnector(dict(config.credentials))

    raise ValueError(
        f"Unsupported warehouse_type: '{wtype}'. "
        "Supported values are: 'snowflake', 'bigquery', 'redshift'."
    )


__all__ = ["BaseConnector", "ConnectorError", "get_connector"]
