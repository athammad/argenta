"""Pydantic result models for the Argenta statistics layer.

These models are the output contract of the stats layer.  Everything
downstream — the results writer, the CLI, the UI — consumes these models
rather than raw DataFrames or dicts.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MetricResult(BaseModel):
    """Statistical results for a single metric in an experiment.

    Attributes:
        metric_name: The name of the metric (matches the event name from
            ``outcomes.target_events``).
        ate: Average Treatment Effect — the difference in means between the
            treatment and control groups.  Positive means treatment is better.
        ci_low: Lower bound of the ``(1 - alpha)`` confidence interval on the
            ATE.
        ci_high: Upper bound of the confidence interval on the ATE.
        p_value: Two-sided p-value from Welch's t-test.  Values below
            ``alpha`` indicate statistical significance.
        n_control: Number of users in the control group for this metric.
        n_treatment: Number of users in the treatment group for this metric.
        mean_control: Mean outcome value for the control group.
        mean_treatment: Mean outcome value for the treatment group.
        relative_lift: Relative lift of treatment over control, expressed as
            a fraction (e.g. ``0.05`` = 5% lift).  ``None`` if
            ``mean_control`` is zero (division by zero).
        cuped_applied: Whether CUPED variance reduction was applied to this
            metric.  When ``True``, the ATE and CI are computed on the
            CUPED-adjusted outcomes.
        winsorized: Whether the outcome values were winsorized before
            computing statistics.
    """

    metric_name: str
    ate: float
    ci_low: float
    ci_high: float
    p_value: float
    n_control: int
    n_treatment: int
    mean_control: float
    mean_treatment: float
    relative_lift: float | None
    cuped_applied: bool
    winsorized: bool


class ExperimentResult(BaseModel):
    """Full results for an Argenta experiment analysis run.

    Attributes:
        experiment_id: The experiment identifier, matching
            ``ExperimentConfig.experiment_id``.
        metrics: One :class:`MetricResult` per configured metric.
        srm_detected: Whether a Sample Ratio Mismatch was detected.
            A ``True`` value indicates the observed ratio of control to
            treatment users differs significantly from the expected ratio
            (default 50/50).  Results should be treated with caution when
            ``srm_detected`` is ``True``.
        n_control_total: Total number of users in the control group across
            all metrics (based on the exposure count, not per-metric counts).
        n_treatment_total: Total number of users in the treatment group.
        run_at: UTC timestamp of when this analysis was run.
    """

    experiment_id: str
    metrics: list[MetricResult] = Field(default_factory=list)
    srm_detected: bool
    n_control_total: int
    n_treatment_total: int
    run_at: datetime = Field(default_factory=datetime.utcnow)

    def significant_metrics(self, alpha: float = 0.05) -> list[MetricResult]:
        """Return metrics with a p-value below ``alpha``.

        Args:
            alpha: Significance threshold. Defaults to ``0.05``.

        Returns:
            A list of :class:`MetricResult` instances where
            ``p_value < alpha``.
        """
        return [m for m in self.metrics if m.p_value < alpha]

    def summary(self) -> str:
        """Return a human-readable summary of the experiment results.

        Returns:
            A formatted multi-line string suitable for logging or CLI output.
        """
        lines = [
            f"Experiment : {self.experiment_id}",
            f"Run at     : {self.run_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Control N  : {self.n_control_total:,}",
            f"Treatment N: {self.n_treatment_total:,}",
            f"SRM        : {'DETECTED ⚠' if self.srm_detected else 'OK'}",
            "",
            f"{'Metric':<35} {'ATE':>10} {'CI Low':>10} {'CI High':>10} "
            f"{'p-value':>10} {'Lift':>8} {'CUPED':>6}",
            "-" * 95,
        ]
        for m in self.metrics:
            lift_str = f"{m.relative_lift:.1%}" if m.relative_lift is not None else "N/A"
            sig_marker = "*" if m.p_value < 0.05 else " "
            lines.append(
                f"{m.metric_name:<35} {m.ate:>10.4f} {m.ci_low:>10.4f} "
                f"{m.ci_high:>10.4f} {m.p_value:>10.4f} {lift_str:>8} "
                f"{'Yes' if m.cuped_applied else 'No':>6} {sig_marker}"
            )
        return "\n".join(lines)
