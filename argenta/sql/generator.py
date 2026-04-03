"""SQL pipeline generator for Argenta.

:class:`SQLPipelineGenerator` is a pure-function class — it generates SQL
strings from Jinja2 templates but never executes them.  Execution is always
handled by :class:`~argenta.connectors.base.BaseConnector`, orchestrated by
:class:`~argenta.pipeline.runner.PipelineRunner`.

This separation ensures that:

- SQL can be unit-tested by inspecting the rendered string (no warehouse
  connection required).
- Templates stay readable because complex Python logic (column list
  construction, dialect branching) lives in Python, not in Jinja2.
"""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from argenta.config.schema import ArgentoConfig
from argenta.sql.dialect import (
    WarehouseDialect,
    cast_timestamp,
    create_table_as,
    qualify_table,
    row_number_dedup,
)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class SQLPipelineGenerator:
    """Generates the Argenta SQL pipeline for a given config and dialect.

    All public methods return SQL strings.  None of them issue any I/O.
    The caller (usually :class:`~argenta.pipeline.runner.PipelineRunner`)
    is responsible for executing the returned SQL via a connector.

    Args:
        config: The validated :class:`~argenta.config.schema.ArgentoConfig`
            for this analysis run.
        dialect: The :class:`~argenta.sql.dialect.WarehouseDialect` matching
            the client's warehouse type.

    Example::

        gen = SQLPipelineGenerator(config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_prepared_dataset()
        connector.execute(sql)
    """

    def __init__(self, config: ArgentoConfig, dialect: WarehouseDialect) -> None:
        self._config = config
        self._dialect = dialect
        self._env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # ------------------------------------------------------------------
    # Public rendering methods
    # ------------------------------------------------------------------

    def render_exposure_dedup(self) -> str:
        """Render the exposure deduplication CTE.

        Produces a ``first_exposures`` CTE that keeps only the first exposure
        per user per experiment (``ROW_NUMBER() = 1`` ordered by timestamp).

        Returns:
            A SQL CTE string (without the leading ``WITH`` keyword) that can
            be embedded inside a ``WITH`` block.
        """
        exp = self._config.exposures
        cfg = self._config.experiment

        treatment = cfg.treatment_variant or self._PLACEHOLDER_TREATMENT

        ts_expr = cast_timestamp(exp.timestamp_col, self._dialect)
        rn_expr = row_number_dedup(exp.user_id_col, exp.timestamp_col, self._dialect)

        tmpl = self._env.get_template("exposure_dedup.sql.j2")
        return tmpl.render(
            exposures_table=qualify_table(exp.table, self._dialect),
            user_id_col=exp.user_id_col,
            experiment_id_col=exp.experiment_id_col,
            variant_col=exp.variant_col,
            timestamp_col=exp.timestamp_col,
            experiment_id=cfg.experiment_id,
            control_variant=cfg.control_variant,
            treatment_variant=treatment,
            cast_timestamp_expr=ts_expr,
            rn_expr=rn_expr,
        )

    def render_outcome_join(self) -> str:
        """Render the outcome join CTE.

        Produces a ``user_outcomes`` CTE that LEFT JOINs the outcomes table
        onto ``first_exposures``, keeping only post-exposure events, and
        aggregates to one row per user with one column per configured metric.

        Returns:
            A SQL CTE string (without the leading ``WITH`` keyword).

        Raises:
            ValueError: If ``outcomes.target_events`` is empty. Callers must
                ensure at least one event is configured before calling this
                method.
        """
        out = self._config.outcomes
        exp = self._config.exposures
        target_events = out.target_events

        if not target_events:
            raise ValueError(
                "outcomes.target_events must contain at least one event name. "
                "Add the event names you want to analyse to the config."
            )

        metric_select_exprs = self._build_metric_select_exprs(target_events, out.event_name_col, out.value_col)
        event_filter = self._build_event_filter_clause(target_events, out.event_name_col)
        o_cast_ts = cast_timestamp(f"o.{out.timestamp_col}", self._dialect)

        tmpl = self._env.get_template("outcome_join.sql.j2")
        return tmpl.render(
            outcomes_table=qualify_table(out.table, self._dialect),
            o_user_id_col=out.user_id_col,
            o_event_name_col=out.event_name_col,
            o_value_col=out.value_col,
            o_timestamp_col=out.timestamp_col,
            o_cast_ts_expr=o_cast_ts,
            e_user_id_col=exp.user_id_col,
            e_variant_col=exp.variant_col,
            metric_select_exprs=metric_select_exprs,
            event_filter_clause=event_filter,
        )

    def render_feature_join(self) -> str:
        """Render the user feature join CTE.

        Produces a ``user_data`` CTE that LEFT JOINs the user features table
        onto ``user_outcomes``.  Columns included are those listed in
        ``user_features.feature_cols``, plus ``covariate_col`` if set.

        Returns:
            A SQL CTE string (without the leading ``WITH`` keyword).
        """
        feat = self._config.user_features
        out = self._config.outcomes

        feature_cols = list(feat.feature_cols)
        if feat.covariate_col and feat.covariate_col not in feature_cols:
            feature_cols.append(feat.covariate_col)

        tmpl = self._env.get_template("feature_join.sql.j2")
        return tmpl.render(
            features_table=qualify_table(feat.table, self._dialect),
            f_user_id_col=feat.user_id_col,
            o_user_id_col=out.user_id_col,
            feature_select_cols=feature_cols,
        )

    def render_prepared_dataset(self) -> str:
        """Render the full pipeline as a single ``CREATE TABLE AS SELECT``.

        Assembles the three CTEs (exposure dedup, outcome join, feature join)
        into a complete DDL statement that writes the analysis-ready dataset
        to ``{output_schema}.prepared_dataset_{experiment_id}`` inside the
        client's warehouse.

        Returns:
            A complete, executable SQL string.  Pass it to
            :meth:`~argenta.connectors.base.BaseConnector.execute`.
        """
        cfg = self._config
        output_table = (
            f"{cfg.warehouse.output_schema}"
            f".prepared_dataset_{_sanitise_identifier(cfg.experiment.experiment_id)}"
        )

        exposure_cte = self.render_exposure_dedup()
        outcome_cte = self.render_outcome_join()
        feature_cte = self.render_feature_join()

        select_sql = (
            f"WITH\n{exposure_cte},\n{outcome_cte},\n{feature_cte}\n"
            "SELECT * FROM user_data"
        )
        ctas = create_table_as(
            fully_qualified_table=output_table,
            select_sql=select_sql,
            dialect=self._dialect,
            replace=True,
        )

        return ctas

    def prepared_dataset_table(self) -> str:
        """Return the fully qualified name of the prepared dataset table.

        Useful for querying the prepared dataset after it has been written.

        Returns:
            A string such as ``'argenta.prepared_dataset_checkout_redesign_2024'``.
        """
        cfg = self._config
        return (
            f"{cfg.warehouse.output_schema}"
            f".prepared_dataset_{_sanitise_identifier(cfg.experiment.experiment_id)}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _PLACEHOLDER_TREATMENT = "__treatment__"

    def _build_metric_select_exprs(
        self,
        target_events: list[str],
        event_name_col: str,
        value_col: str,
    ) -> list[str]:
        """Build ``COALESCE(SUM(CASE WHEN ...), 0) AS metric_...`` expressions.

        One expression per target event, for the SUM of the value column.

        Args:
            target_events: Event name strings.
            event_name_col: The event type column in the outcomes table.
            value_col: The numeric value column.

        Returns:
            A list of SQL SELECT expression strings.
        """
        exprs = []
        for event in target_events:
            alias = _sanitise_identifier(event)
            expr = (
                f"COALESCE(SUM(CASE WHEN LOWER(o.{event_name_col}) = LOWER('{event}') "
                f"THEN o.{value_col} END), 0) AS {alias}"
            )
            exprs.append(expr)
        return exprs

    def _build_event_filter_clause(
        self,
        target_events: list[str],
        event_name_col: str,
    ) -> str:
        """Build an ``AND o.event_name_col IN (...)`` filter clause.

        Returns an empty string when ``target_events`` is empty (no filter
        applied — all events are included).

        Args:
            target_events: Event name strings.
            event_name_col: The event type column in the outcomes table.

        Returns:
            A SQL ``AND ... IN (...)`` string, or ``''`` if no events are
            specified.
        """
        if not target_events:
            return ""
        quoted = ", ".join(f"'{e}'" for e in target_events)
        return f"AND LOWER(o.{event_name_col}) IN ({quoted})"


def _sanitise_identifier(value: str) -> str:
    """Convert an arbitrary string into a safe SQL identifier.

    Replaces any character that is not a letter, digit, or underscore with
    an underscore, and lowercases the result.

    Args:
        value: An arbitrary string (e.g. an experiment ID or event name).

    Returns:
        A lowercase, underscore-only string safe to use as a SQL column or
        table name suffix.
    """
    return re.sub(r"[^a-z0-9_]", "_", value.lower())
