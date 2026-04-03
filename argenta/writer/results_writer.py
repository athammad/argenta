"""Writes Argenta experiment results back to the client's warehouse.

:class:`ResultsWriter` is responsible for creating the output tables in the
configured ``output_schema`` and upserting results after each analysis run.
All DDL and DML runs inside the client's warehouse — no data leaves.

Output tables
-------------
``{output_schema}.experiment_results``
    One row per (experiment_id, metric_name) pair.  Overwritten on each run.

The Phase 2 tables (``user_cate_scores``, ``segment_effects``) are not yet
written by this class and will be added in Phase 2.
"""

from __future__ import annotations

import logging

from argenta.connectors.base import BaseConnector
from argenta.stats.models import ExperimentResult

logger = logging.getLogger(__name__)


# DDL for the experiment_results table.  Column types are intentionally
# generic (VARCHAR/FLOAT/INT/BOOLEAN) to be compatible across all three
# supported warehouses without dialect-specific DDL.
_CREATE_EXPERIMENT_RESULTS_DDL = """
CREATE TABLE IF NOT EXISTS {output_schema}.experiment_results (
    experiment_id      VARCHAR(512)   NOT NULL,
    metric_name        VARCHAR(512)   NOT NULL,
    ate                FLOAT,
    ci_low             FLOAT,
    ci_high            FLOAT,
    p_value            FLOAT,
    n_control          INTEGER,
    n_treatment        INTEGER,
    mean_control       FLOAT,
    mean_treatment     FLOAT,
    relative_lift      FLOAT,
    cuped_applied      BOOLEAN,
    winsorized         BOOLEAN,
    srm_detected       BOOLEAN,
    run_at             TIMESTAMP,
    PRIMARY KEY (experiment_id, metric_name)
)
"""

_DELETE_EXPERIMENT_ROWS = """
DELETE FROM {output_schema}.experiment_results
WHERE experiment_id = '{experiment_id}'
"""

_INSERT_METRIC_ROW = """
INSERT INTO {output_schema}.experiment_results (
    experiment_id, metric_name, ate, ci_low, ci_high, p_value,
    n_control, n_treatment, mean_control, mean_treatment,
    relative_lift, cuped_applied, winsorized, srm_detected, run_at
) VALUES (
    '{experiment_id}', '{metric_name}',
    {ate}, {ci_low}, {ci_high}, {p_value},
    {n_control}, {n_treatment}, {mean_control}, {mean_treatment},
    {relative_lift}, {cuped_applied}, {winsorized}, {srm_detected},
    '{run_at}'
)
"""


class ResultsWriter:
    """Writes :class:`~argenta.stats.models.ExperimentResult` objects back to the warehouse.

    Creates the output schema tables if they do not yet exist (idempotent),
    then deletes any previous rows for the experiment and inserts fresh ones.

    Args:
        connector: An open :class:`~argenta.connectors.base.BaseConnector`
            instance.  Must already be connected.
        output_schema: The schema name where Argenta writes its output tables.
            Must match ``WarehouseConfig.output_schema``.

    Example::

        writer = ResultsWriter(connector, output_schema="argenta")
        writer.write_experiment_results(result)
    """

    def __init__(self, connector: BaseConnector, output_schema: str) -> None:
        self._connector = connector
        self._schema = output_schema

    def ensure_schema_exists(self) -> None:
        """Create the output schema if it does not already exist.

        This is a best-effort operation — if the connector user does not have
        ``CREATE SCHEMA`` privileges, this will raise a
        :class:`~argenta.connectors.base.ConnectorError`.  In that case, the
        client's warehouse admin should create the schema manually.

        Raises:
            ConnectorError: If schema creation fails.
        """
        logger.info("[WRITER] Ensuring output schema exists: %s", self._schema)
        self._connector.execute(f"CREATE SCHEMA IF NOT EXISTS {self._schema}")

    def ensure_tables_exist(self) -> None:
        """Create the output tables with the expected schema if absent.

        Uses ``CREATE TABLE IF NOT EXISTS`` — safe to call on every run.

        Raises:
            ConnectorError: If table creation fails.
        """
        logger.info("[WRITER] Ensuring output tables exist in schema: %s", self._schema)
        ddl = _CREATE_EXPERIMENT_RESULTS_DDL.format(output_schema=self._schema)
        self._connector.execute(ddl)

    def write_experiment_results(self, result: ExperimentResult) -> None:
        """Write an :class:`~argenta.stats.models.ExperimentResult` to the warehouse.

        This method:

        1. Ensures the output tables exist (idempotent DDL).
        2. Deletes any existing rows for ``result.experiment_id``.
        3. Inserts one row per metric in ``result.metrics``.

        Args:
            result: The fully populated
                :class:`~argenta.stats.models.ExperimentResult` to persist.

        Raises:
            ConnectorError: If any warehouse operation fails.
        """
        self.ensure_tables_exist()

        logger.info(
            "[WRITER] Writing results for experiment: %s (%d metrics)",
            result.experiment_id,
            len(result.metrics),
        )

        # Delete stale rows for this experiment
        delete_sql = _DELETE_EXPERIMENT_ROWS.format(
            output_schema=self._schema,
            experiment_id=result.experiment_id,
        )
        self._connector.execute(delete_sql)

        # Insert one row per metric
        run_at_str = result.run_at.strftime("%Y-%m-%d %H:%M:%S")
        for metric in result.metrics:
            relative_lift = (
                "NULL" if metric.relative_lift is None else str(metric.relative_lift)
            )
            insert_sql = _INSERT_METRIC_ROW.format(
                output_schema=self._schema,
                experiment_id=_escape(result.experiment_id),
                metric_name=_escape(metric.metric_name),
                ate=metric.ate,
                ci_low=metric.ci_low,
                ci_high=metric.ci_high,
                p_value=metric.p_value,
                n_control=metric.n_control,
                n_treatment=metric.n_treatment,
                mean_control=metric.mean_control,
                mean_treatment=metric.mean_treatment,
                relative_lift=relative_lift,
                cuped_applied=str(metric.cuped_applied).upper(),
                winsorized=str(metric.winsorized).upper(),
                srm_detected=str(result.srm_detected).upper(),
                run_at=run_at_str,
            )
            self._connector.execute(insert_sql)

        logger.info("[WRITER] Results written successfully for: %s", result.experiment_id)


def _escape(value: str) -> str:
    """Escape single quotes in a string for safe SQL interpolation.

    Args:
        value: A string value to embed in a SQL literal.

    Returns:
        The string with any single quotes doubled (standard SQL escaping).
    """
    return value.replace("'", "''")
