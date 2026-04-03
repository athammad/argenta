"""Pydantic v2 configuration models for Argenta.

All client-specific knowledge — table names, column mappings, warehouse credentials,
and experiment parameters — lives in these models. No other module hardcodes schema
names or column names; they always read from an ``ArgentoConfig`` instance.

Example usage::

    from argenta.config.loader import load_config

    config = load_config("argenta.yaml")
    print(config.experiment.experiment_id)
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ExposuresTableConfig(BaseModel):
    """Column mapping for the client's experiment exposures table.

    This table records which variant each user was assigned to and when.
    Argenta deduplicates it to keep only the first exposure per user per
    experiment (the canonical causal inference approach).

    Attributes:
        table: Fully qualified table name in the warehouse
            (e.g. ``'ANALYTICS.PUBLIC.STATSIG_EXPOSURES'``).
        user_id_col: Column name that holds the user identifier.
        experiment_id_col: Column name that identifies the experiment.
            Argenta filters rows where this column equals
            ``ExperimentConfig.experiment_id``.
        variant_col: Column name that holds the treatment assignment
            (e.g. ``'control'`` / ``'treatment'``).
        timestamp_col: Column name for the assignment timestamp.
            Must be castable to ``TIMESTAMP`` in the target warehouse.
    """

    table: str
    user_id_col: str = "user_id"
    experiment_id_col: str = "experiment_id"
    variant_col: str = "variant"
    timestamp_col: str = "timestamp"


class OutcomesTableConfig(BaseModel):
    """Column mapping for the client's outcome events table.

    Argenta treats each distinct value of ``event_name_col`` as a separate
    metric. Only events occurring **after** a user's first exposure are
    included in the analysis, preventing pre-exposure contamination.

    Attributes:
        table: Fully qualified table name
            (e.g. ``'ANALYTICS.PUBLIC.EVENTS'``).
        user_id_col: Column name for the user identifier.
        event_name_col: Column name that identifies the type of event
            (e.g. ``'purchase'``, ``'add_to_cart'``).
        value_col: Numeric value associated with the event (e.g. revenue).
            For binary outcomes (event fired / not fired), set this to the
            same column and Argenta will aggregate as 0/1.
        timestamp_col: Column name for the event timestamp.
        target_events: Explicit list of ``event_name`` values to analyse.
            If empty, all distinct event names are used (not recommended
            for high-cardinality event tables — prefer explicit lists).
    """

    table: str
    user_id_col: str = "user_id"
    event_name_col: str = "event_name"
    value_col: str = "value"
    timestamp_col: str = "timestamp"
    target_events: list[str] = Field(default_factory=list)


class UserFeaturesTableConfig(BaseModel):
    """Column mapping for the user features table.

    User features serve two purposes:

    1. **CUPED covariate** — a single pre-experiment metric (e.g.
       ``pre_experiment_revenue``) used to reduce outcome variance and
       increase statistical power.
    2. **Causal ML inputs** — feature columns used as ``X`` in CATE
       estimation and uplift modelling (Phase 2).

    The table must have exactly one row per user. If users appear multiple
    times, the join will produce duplicates and results will be incorrect.

    Attributes:
        table: Fully qualified table name
            (e.g. ``'ANALYTICS.PUBLIC.USER_DIM'``).
        user_id_col: Column name for the user identifier.
        feature_cols: Explicit list of feature column names to select.
            If empty, all non-``user_id`` columns are selected, which is
            not recommended for wide tables.
        covariate_col: The single pre-experiment metric column used for
            CUPED. Typically the same metric measured before the experiment
            started (e.g. ``'pre_experiment_revenue'``). Required when
            ``ExperimentConfig.use_cuped`` is ``True``; ignored otherwise.
    """

    table: str
    user_id_col: str = "user_id"
    feature_cols: list[str] = Field(default_factory=list)
    covariate_col: str | None = None


class WarehouseConfig(BaseModel):
    """Warehouse connection and output configuration.

    Attributes:
        warehouse_type: One of ``'snowflake'``, ``'bigquery'``, or
            ``'redshift'``. Determines which connector is instantiated.
        credentials: Warehouse-specific connection parameters. Keys vary
            by warehouse type — see the connector class docstrings for the
            required and optional keys.
        output_schema: Schema in the warehouse where Argenta writes result
            tables (``experiment_results``, ``user_cate_scores``,
            ``segment_effects``). Defaults to ``'argenta'``. The schema
            must already exist, or the connector user must have
            ``CREATE SCHEMA`` privileges.
    """

    warehouse_type: Literal["snowflake", "bigquery", "redshift"]
    credentials: dict[str, str | int | bool]
    output_schema: str = "argenta"


class ExperimentConfig(BaseModel):
    """Per-experiment runtime parameters.

    Attributes:
        experiment_id: The value in the exposures table's
            ``experiment_id_col`` that identifies this experiment.
            Used as a ``WHERE`` filter in the SQL pipeline.
        control_variant: The variant label for the control group.
            Defaults to ``'control'``.
        treatment_variant: The variant label for the treatment group.
            If ``None`` and the experiment has exactly two variants,
            the non-control variant is used automatically. If the
            experiment has more than two variants, this must be set
            explicitly.
        alpha: Significance level for confidence intervals and hypothesis
            tests. Must be in the open interval ``(0, 1)``. Defaults
            to ``0.05`` (95% confidence).
        winsorize_percentile: Upper percentile at which to cap metric
            values before computing statistics. Applied per-variant
            independently. Set to ``1.0`` to disable winsorization.
            Defaults to ``0.99`` (clips the top 1%).
        use_cuped: Whether to apply CUPED variance reduction. Requires
            ``UserFeaturesTableConfig.covariate_col`` to be set. If
            ``True`` and ``covariate_col`` is ``None``, config validation
            will raise an error.
    """

    experiment_id: str
    control_variant: str = "control"
    treatment_variant: str | None = None
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    winsorize_percentile: float = Field(default=0.99, gt=0.5, le=1.0)
    use_cuped: bool = True


class CausalMLConfig(BaseModel):
    """Optional configuration for the Phase 2 causal ML layer.

    When present in ``ArgentoConfig``, the pipeline runs CATE estimation,
    segment HTE analysis, and (optionally) full-user uplift scoring after the
    baseline statistics step.

    If this section is absent from the config, the pipeline behaves exactly
    as Phase 1 (ATE + CUPED only).

    Attributes:
        enabled: Master switch.  Set to ``False`` to disable causal ML without
            removing the config block.  Defaults to ``True``.
        n_estimators: Number of trees in the Causal Forest.  Higher values
            give more stable CATE estimates at the cost of compute time.
            Defaults to ``200``.
        min_samples_leaf: Minimum number of samples in each causal forest
            leaf.  Acts as a regularisation parameter — higher values produce
            smoother CATE estimates.  Defaults to ``10``.
        max_depth: Maximum depth of each tree in the Causal Forest.  ``None``
            means unlimited depth (default ``None``).
        score_all_users: If ``True``, Argenta fetches the full user features
            table (not just experiment participants) and scores every user for
            uplift.  Written to ``argenta.user_cate_scores``.  Set to
            ``False`` (default) to score only experiment participants.
        max_segment_features: Maximum number of feature columns to analyse
            for segment HTE.  Features are selected by their marginal variance
            contribution.  Defaults to ``10``.
        segment_min_users: Minimum number of users in a segment for it to be
            reported.  Segments with fewer users are suppressed.
            Defaults to ``50``.
        nuisance_model: Model family to use for the first-stage (nuisance)
            models in the Causal Forest DML.  One of ``'lightgbm'``
            (default, fast) or ``'linear'`` (interpretable, assumes
            linear relationship between outcome/treatment and features).
    """

    enabled: bool = True
    n_estimators: int = Field(default=200, gt=0)
    min_samples_leaf: int = Field(default=10, gt=0)
    max_depth: int | None = None
    score_all_users: bool = False
    max_segment_features: int = Field(default=10, gt=0)
    segment_min_users: int = Field(default=50, gt=0)
    nuisance_model: Literal["lightgbm", "linear"] = "lightgbm"


class ArgentoConfig(BaseModel):
    """Root configuration model for an Argenta analysis run.

    Load from a YAML file using :func:`argenta.config.loader.load_config`,
    or construct programmatically from a ``dict`` using
    :func:`argenta.config.loader.load_config_from_dict`.

    Example YAML structure::

        warehouse:
          warehouse_type: snowflake
          output_schema: argenta
          credentials:
            account: my_account
            user: argenta_svc
            password: "${SNOWFLAKE_PASSWORD}"

        exposures:
          table: ANALYTICS.PUBLIC.EXPOSURES

        outcomes:
          table: ANALYTICS.PUBLIC.EVENTS
          target_events: [purchase]

        user_features:
          table: ANALYTICS.PUBLIC.USER_DIM
          covariate_col: pre_experiment_revenue
          feature_cols: [country, device_type, tenure_days]

        experiment:
          experiment_id: checkout_redesign_2024
          use_cuped: true

        causal_ml:
          enabled: true
          n_estimators: 200
          score_all_users: false

    Attributes:
        warehouse: Warehouse connection and output schema settings.
        exposures: Column mapping for the exposures table.
        outcomes: Column mapping for the outcomes / events table.
        user_features: Column mapping for the user features table.
        experiment: Runtime parameters for the experiment being analysed.
        causal_ml: Optional Phase 2 causal ML configuration.  If ``None``,
            causal ML is not run.
    """

    warehouse: WarehouseConfig
    exposures: ExposuresTableConfig
    outcomes: OutcomesTableConfig
    user_features: UserFeaturesTableConfig
    experiment: ExperimentConfig
    causal_ml: CausalMLConfig | None = None

    @model_validator(mode="after")
    def _validate_cuped_covariate_present(self) -> "ArgentoConfig":
        """Raise if CUPED is enabled but no covariate column is configured.

        Raises:
            ValueError: If ``experiment.use_cuped`` is ``True`` and
                ``user_features.covariate_col`` is ``None``.
        """
        if self.experiment.use_cuped and self.user_features.covariate_col is None:
            raise ValueError(
                "experiment.use_cuped is True but user_features.covariate_col is not set. "
                "Either provide a covariate_col in [user_features] or set use_cuped: false."
            )
        return self
