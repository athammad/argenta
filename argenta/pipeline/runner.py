"""End-to-end pipeline orchestration for Argenta.

:class:`PipelineRunner` is the single entry point for running an analysis.
It wires together the connector, SQL generator, stats layer, and writer —
but contains none of their logic itself.

Execution flow
--------------
1. Open warehouse connection.
2. Generate and execute the prepared-dataset SQL (exposure dedup +
   outcome join + feature join → CTAS in the warehouse).
3. Fetch the prepared dataset from the warehouse into memory.
4. Run statistics (winsorize → CUPED → ATE + CI + SRM) in Python.
5. If ``causal_ml`` is configured and enabled:
   a. Fit CausalForestDML per metric on the prepared dataset.
   b. Predict CATE scores for experiment participants.
   c. If ``score_all_users=True``, fetch the full user features table
      and score all users for uplift.
   d. Run segment HTE analysis for each metric.
6. Write :class:`~argenta.stats.models.ExperimentResult` (including CATE
   results) back to the warehouse.
7. Return the result.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from argenta.causal.cate import CATEEstimator
from argenta.causal.models import CATEMetricResult
from argenta.causal.segments import SegmentAnalyzer
from argenta.causal.uplift import UpliftScorer
from argenta.config.schema import ArgentoConfig
from argenta.connectors import get_connector
from argenta.connectors.base import BaseConnector
from argenta.sql.dialect import WarehouseDialect
from argenta.sql.generator import SQLPipelineGenerator
from argenta.stats.ate import check_srm, compute_ate, winsorize
from argenta.stats.cuped import apply_cuped
from argenta.stats.models import ExperimentResult, MetricResult
from argenta.writer.results_writer import ResultsWriter

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the full Argenta analysis pipeline.

    Args:
        config: The validated :class:`~argenta.config.schema.ArgentoConfig`
            for this run.

    Example::

        from argenta.config.loader import load_config
        from argenta.pipeline.runner import PipelineRunner

        config = load_config("argenta.yaml")
        runner = PipelineRunner(config)
        result = runner.run()
        print(result.summary())
    """

    def __init__(self, config: ArgentoConfig) -> None:
        self._config = config

    def run(self) -> ExperimentResult:
        """Execute the full analysis pipeline and return results.

        This is the main entry point.  It opens a warehouse connection,
        runs the SQL pipeline, computes statistics, writes results back,
        and returns a fully populated :class:`~argenta.stats.models.ExperimentResult`.

        Returns:
            A :class:`~argenta.stats.models.ExperimentResult` containing ATE,
            confidence intervals, p-values, SRM status, and metadata for all
            configured metrics.

        Raises:
            ConnectorError: If the warehouse connection or any SQL execution fails.
            ValueError: If the prepared dataset is empty or missing required columns.
        """
        experiment_id = self._config.experiment.experiment_id
        logger.info("[PIPELINE] Starting analysis for experiment: %s", experiment_id)

        connector = get_connector(self._config.warehouse)
        dialect = WarehouseDialect(self._config.warehouse.warehouse_type)
        generator = SQLPipelineGenerator(self._config, dialect)

        with connector:
            # Step 1 — Build the prepared dataset inside the warehouse
            logger.info("[PIPELINE] Generating SQL pipeline")
            pipeline_sql = generator.render_prepared_dataset()
            logger.info("[PIPELINE] Executing SQL pipeline (CTAS)")
            connector.execute(pipeline_sql)

            # Step 2 — Fetch the prepared dataset into memory
            dataset_table = generator.prepared_dataset_table()
            logger.info("[PIPELINE] Fetching prepared dataset from: %s", dataset_table)
            df = connector.query(f"SELECT * FROM {dataset_table}")

            # Step 3 — Compute baseline statistics
            logger.info("[PIPELINE] Computing statistics")
            result = self._compute_result(df, experiment_id)

            # Step 4 — Causal ML (Phase 2), if configured
            causal_cfg = self._config.causal_ml
            if causal_cfg is not None and causal_cfg.enabled:
                feature_cols = list(self._config.user_features.feature_cols)
                if not feature_cols:
                    logger.warning(
                        "[PIPELINE] causal_ml is enabled but user_features.feature_cols is empty. "
                        "Skipping CATE estimation."
                    )
                else:
                    logger.info("[PIPELINE] Running causal ML layer")
                    all_users_df: pd.DataFrame | None = None
                    if causal_cfg.score_all_users:
                        logger.info("[PIPELINE] Fetching full user features for uplift scoring")
                        all_users_df = connector.query(
                            f"SELECT * FROM {self._config.user_features.table}"
                        )
                    result.cate_results = self._compute_cate_results(
                        df, feature_cols, causal_cfg, all_users_df
                    )

            # Step 5 — Write results back
            logger.info("[PIPELINE] Writing results to warehouse")
            writer = ResultsWriter(connector, self._config.warehouse.output_schema)
            writer.write_experiment_results(result)

        logger.info("[PIPELINE] Analysis complete for experiment: %s", experiment_id)
        return result

    # ------------------------------------------------------------------
    # Internal statistics computation
    # ------------------------------------------------------------------

    def _compute_result(self, df: pd.DataFrame, experiment_id: str) -> ExperimentResult:
        """Compute statistical results from the prepared dataset DataFrame.

        Args:
            df: The prepared dataset with columns: variant, metric columns,
                and (optionally) the covariate column.
            experiment_id: The experiment identifier.

        Returns:
            A populated :class:`~argenta.stats.models.ExperimentResult`.

        Raises:
            ValueError: If the variant column is missing or neither control
                nor treatment users are present.
        """
        cfg = self._config.experiment
        variant_col = self._config.exposures.variant_col

        if variant_col not in df.columns:
            raise ValueError(
                f"Variant column '{variant_col}' not found in prepared dataset. "
                f"Available columns: {list(df.columns)}"
            )

        control_df = df[df[variant_col] == cfg.control_variant]
        treatment_df = df[df[variant_col] == cfg.treatment_variant]

        if len(control_df) == 0:
            raise ValueError(
                f"No control users found with variant='{cfg.control_variant}'. "
                "Check experiment.control_variant in config."
            )
        if len(treatment_df) == 0:
            raise ValueError(
                f"No treatment users found with variant='{cfg.treatment_variant}'. "
                "Check experiment.treatment_variant in config."
            )

        n_control = len(control_df)
        n_treatment = len(treatment_df)

        srm = check_srm(n_control, n_treatment)
        if srm:
            logger.warning(
                "[PIPELINE] SRM detected for experiment %s — "
                "control=%d, treatment=%d. Investigate randomisation.",
                experiment_id, n_control, n_treatment,
            )

        # Identify metric columns (everything that is not variant or feature cols)
        feature_cols = set(self._config.user_features.feature_cols)
        if self._config.user_features.covariate_col:
            feature_cols.add(self._config.user_features.covariate_col)
        non_metric_cols = {variant_col, self._config.exposures.user_id_col} | feature_cols
        metric_cols = [c for c in df.columns if c not in non_metric_cols]

        metric_results = []
        for col in metric_cols:
            metric_result = self._compute_metric(
                col, control_df, treatment_df, cfg.alpha,
                cfg.winsorize_percentile, cfg.use_cuped,
            )
            metric_results.append(metric_result)

        return ExperimentResult(
            experiment_id=experiment_id,
            metrics=metric_results,
            srm_detected=srm,
            n_control_total=n_control,
            n_treatment_total=n_treatment,
            run_at=datetime.utcnow(),
        )

    def _compute_metric(
        self,
        metric_col: str,
        control_df: pd.DataFrame,
        treatment_df: pd.DataFrame,
        alpha: float,
        winsorize_percentile: float,
        use_cuped: bool,
    ) -> MetricResult:
        """Compute ATE and related statistics for a single metric column.

        Args:
            metric_col: The name of the metric column in the prepared dataset.
            control_df: Rows belonging to the control group.
            treatment_df: Rows belonging to the treatment group.
            alpha: Significance level for confidence intervals.
            winsorize_percentile: Upper percentile for winsorization.
            use_cuped: Whether to apply CUPED variance reduction.

        Returns:
            A populated :class:`~argenta.stats.models.MetricResult`.
        """
        y_control = control_df[metric_col].astype(float)
        y_treatment = treatment_df[metric_col].astype(float)

        # Winsorize both groups at the same threshold derived from pooled data
        winsorized = False
        if winsorize_percentile < 1.0:
            pooled = pd.concat([y_control, y_treatment])
            upper = float(pooled.quantile(winsorize_percentile))
            y_control = y_control.clip(upper=upper)
            y_treatment = y_treatment.clip(upper=upper)
            winsorized = True

        # CUPED variance reduction
        cuped_applied = False
        covariate_col = self._config.user_features.covariate_col
        if use_cuped and covariate_col:
            try:
                x_control = control_df[covariate_col].astype(float)
                x_treatment = treatment_df[covariate_col].astype(float)
                # Pool control and treatment to estimate theta unbiasedly
                y_all = pd.concat([y_control, y_treatment], ignore_index=True)
                x_all = pd.concat([x_control, x_treatment], ignore_index=True)
                from argenta.stats.cuped import apply_cuped as _apply_cuped
                y_all_adj = _apply_cuped(y_all, x_all)
                y_control = y_all_adj.iloc[: len(y_control)].reset_index(drop=True)
                y_treatment = y_all_adj.iloc[len(y_control) :].reset_index(drop=True)
                cuped_applied = True
                logger.debug("[PIPELINE] CUPED applied to metric: %s", metric_col)
            except Exception as exc:
                logger.warning(
                    "[PIPELINE] CUPED failed for metric %s: %s — falling back to raw outcome",
                    metric_col, exc,
                )

        ate, ci_low, ci_high, p_value = compute_ate(y_control, y_treatment, alpha)

        mean_control = float(y_control.mean())
        mean_treatment = float(y_treatment.mean())
        relative_lift = (
            (mean_treatment - mean_control) / abs(mean_control)
            if mean_control != 0
            else None
        )

        return MetricResult(
            metric_name=metric_col,
            ate=ate,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=p_value,
            n_control=len(y_control),
            n_treatment=len(y_treatment),
            mean_control=mean_control,
            mean_treatment=mean_treatment,
            relative_lift=relative_lift,
            cuped_applied=cuped_applied,
            winsorized=winsorized,
        )

    def _compute_cate_results(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        causal_cfg: object,
        all_users_df: pd.DataFrame | None,
    ) -> list[CATEMetricResult]:
        """Fit CATE models and run segment analysis for every metric.

        Args:
            df: The prepared dataset (one row per experiment participant).
            feature_cols: Feature columns to use as covariates ``X``.
            causal_cfg: The :class:`~argenta.config.schema.CausalMLConfig`
                instance controlling model hyperparameters.
            all_users_df: Full user features DataFrame for uplift scoring, or
                ``None`` if ``score_all_users=False``.

        Returns:
            A list of :class:`~argenta.causal.models.CATEMetricResult` objects,
            one per metric column.
        """
        from argenta.config.schema import CausalMLConfig  # avoid circular at module top
        cfg: CausalMLConfig = causal_cfg  # type: ignore[assignment]

        variant_col = self._config.exposures.variant_col
        control_variant = self._config.experiment.control_variant
        alpha = self._config.experiment.alpha

        # Identify metric columns (same logic as _compute_result)
        feature_set = set(feature_cols)
        if self._config.user_features.covariate_col:
            feature_set.add(self._config.user_features.covariate_col)
        non_metric_cols = {variant_col, self._config.exposures.user_id_col} | feature_set
        metric_cols = [c for c in df.columns if c not in non_metric_cols]

        analyzer = SegmentAnalyzer(
            min_users=cfg.segment_min_users,
            max_features=cfg.max_segment_features,
            alpha=alpha,
        )

        cate_results: list[CATEMetricResult] = []
        for metric_col in metric_cols:
            logger.info("[CAUSAL] Estimating CATE for metric: %s", metric_col)
            try:
                estimator = CATEEstimator(cfg)
                estimator.fit(df, metric_col, variant_col, feature_cols)

                # Score experiment participants (always)
                score_df = df[[self._config.exposures.user_id_col] + feature_cols].copy()
                user_scores = estimator.predict(score_df, alpha=alpha)

                # Score all users if requested
                if all_users_df is not None:
                    logger.info("[CAUSAL] Scoring full user base for metric: %s", metric_col)
                    scorer = UpliftScorer(estimator)
                    user_scores = scorer.score_dataframe(all_users_df, feature_cols, alpha=alpha)

                # Segment HTE analysis
                segment_effects = analyzer.analyze(
                    df, metric_col, variant_col, control_variant, feature_cols
                )

                diagnostics = estimator.model_diagnostics()
                cate_scores = [s.cate_score for s in user_scores]

                cate_results.append(CATEMetricResult(
                    metric_name=metric_col,
                    user_scores=user_scores,
                    segment_effects=segment_effects,
                    mean_cate=float(sum(cate_scores) / max(len(cate_scores), 1)),
                    std_cate=float(pd.Series(cate_scores).std()) if cate_scores else 0.0,
                    n_users_scored=len(user_scores),
                    model_r2_outcome=diagnostics.get("r2_outcome"),
                    model_r2_treatment=diagnostics.get("r2_treatment"),
                ))
                logger.info(
                    "[CAUSAL] CATE complete for %s: mean=%.4f, segments=%d",
                    metric_col, cate_results[-1].mean_cate, len(segment_effects),
                )
            except Exception as exc:
                logger.warning(
                    "[CAUSAL] CATE estimation failed for metric %s: %s — skipping",
                    metric_col, exc,
                )

        return cate_results
