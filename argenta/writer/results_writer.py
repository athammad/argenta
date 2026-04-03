"""Writes Argenta experiment results back to the client's warehouse.

:class:`ResultsWriter` is responsible for creating the output tables in the
configured ``output_schema`` and upserting results after each analysis run.
All DDL and DML runs inside the client's warehouse — no data leaves.

Output tables
-------------
``{output_schema}.experiment_results``
    One row per (experiment_id, metric_name) pair.  Overwritten on each run.

``{output_schema}.user_cate_scores``
    One row per (experiment_id, metric_name, user_id).  Contains individual
    CATE scores, confidence intervals, and percentile ranks.  Written only
    when ``CausalMLConfig`` is present and enabled.

``{output_schema}.segment_effects``
    One row per (experiment_id, metric_name, feature_name, segment_value).
    Contains segment-level ATE, CI, and significance flag.  Written only
    when ``CausalMLConfig`` is present and enabled.
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

_CREATE_USER_CATE_SCORES_DDL = """
CREATE TABLE IF NOT EXISTS {output_schema}.user_cate_scores (
    experiment_id   VARCHAR(512)  NOT NULL,
    metric_name     VARCHAR(512)  NOT NULL,
    user_id         VARCHAR(512)  NOT NULL,
    cate_score      FLOAT,
    ci_low          FLOAT,
    ci_high         FLOAT,
    percentile      FLOAT,
    run_at          TIMESTAMP,
    PRIMARY KEY (experiment_id, metric_name, user_id)
)
"""

_INSERT_CATE_SCORE = """
INSERT INTO {output_schema}.user_cate_scores (
    experiment_id, metric_name, user_id,
    cate_score, ci_low, ci_high, percentile, run_at
) VALUES (
    '{experiment_id}', '{metric_name}', '{user_id}',
    {cate_score}, {ci_low}, {ci_high}, {percentile}, '{run_at}'
)
"""

_CREATE_SEGMENT_EFFECTS_DDL = """
CREATE TABLE IF NOT EXISTS {output_schema}.segment_effects (
    experiment_id   VARCHAR(512)  NOT NULL,
    metric_name     VARCHAR(512)  NOT NULL,
    feature_name    VARCHAR(512)  NOT NULL,
    segment_value   VARCHAR(512)  NOT NULL,
    ate             FLOAT,
    ci_low          FLOAT,
    ci_high         FLOAT,
    p_value         FLOAT,
    n_control       INTEGER,
    n_treatment     INTEGER,
    relative_lift   FLOAT,
    is_significant  BOOLEAN,
    run_at          TIMESTAMP,
    PRIMARY KEY (experiment_id, metric_name, feature_name, segment_value)
)
"""

_INSERT_SEGMENT_EFFECT = """
INSERT INTO {output_schema}.segment_effects (
    experiment_id, metric_name, feature_name, segment_value,
    ate, ci_low, ci_high, p_value,
    n_control, n_treatment, relative_lift, is_significant, run_at
) VALUES (
    '{experiment_id}', '{metric_name}', '{feature_name}', '{segment_value}',
    {ate}, {ci_low}, {ci_high}, {p_value},
    {n_control}, {n_treatment}, {relative_lift}, {is_significant}, '{run_at}'
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

        # Phase 2: write CATE results if present
        if result.cate_results:
            self._write_cate_results(result)

    def _write_cate_results(self, result: ExperimentResult) -> None:
        """Write Phase 2 CATE scores and segment effects to the warehouse.

        Creates ``user_cate_scores`` and ``segment_effects`` tables if they
        do not exist, then deletes and re-inserts rows for this experiment.

        Args:
            result: The :class:`~argenta.stats.models.ExperimentResult`
                containing populated ``cate_results``.
        """
        self._connector.execute(_CREATE_USER_CATE_SCORES_DDL.format(output_schema=self._schema))
        self._connector.execute(_CREATE_SEGMENT_EFFECTS_DDL.format(output_schema=self._schema))

        # Delete stale CATE rows
        for tbl in ("user_cate_scores", "segment_effects"):
            self._connector.execute(
                f"DELETE FROM {self._schema}.{tbl} WHERE experiment_id = '{_escape(result.experiment_id)}'"
            )

        run_at_str = result.run_at.strftime("%Y-%m-%d %H:%M:%S")

        for cate_result in result.cate_results:
            # Write user-level CATE scores
            for score in cate_result.user_scores:
                self._connector.execute(
                    _INSERT_CATE_SCORE.format(
                        output_schema=self._schema,
                        experiment_id=_escape(result.experiment_id),
                        metric_name=_escape(cate_result.metric_name),
                        user_id=_escape(score.user_id),
                        cate_score=score.cate_score,
                        ci_low=score.ci_low,
                        ci_high=score.ci_high,
                        percentile=score.percentile,
                        run_at=run_at_str,
                    )
                )

            # Write segment effects
            for seg in cate_result.segment_effects:
                lift = "NULL" if seg.relative_lift is None else str(seg.relative_lift)
                self._connector.execute(
                    _INSERT_SEGMENT_EFFECT.format(
                        output_schema=self._schema,
                        experiment_id=_escape(result.experiment_id),
                        metric_name=_escape(cate_result.metric_name),
                        feature_name=_escape(seg.feature_name),
                        segment_value=_escape(seg.segment_value),
                        ate=seg.ate,
                        ci_low=seg.ci_low,
                        ci_high=seg.ci_high,
                        p_value=seg.p_value,
                        n_control=seg.n_control,
                        n_treatment=seg.n_treatment,
                        relative_lift=lift,
                        is_significant=str(seg.is_significant).upper(),
                        run_at=run_at_str,
                    )
                )

        logger.info(
            "[WRITER] CATE results written for: %s (%d metric(s))",
            result.experiment_id, len(result.cate_results),
        )


def _escape(value: str) -> str:
    """Escape single quotes in a string for safe SQL interpolation.

    Args:
        value: A string value to embed in a SQL literal.

    Returns:
        The string with any single quotes doubled (standard SQL escaping).
    """
    return value.replace("'", "''")
