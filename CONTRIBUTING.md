# Contributing to Argenta

Thank you for your interest in contributing. This document covers how to set up a development
environment, the coding conventions we follow, and how to submit changes.

---

## Development Setup

### Prerequisites

- Python 3.10 or later
- `git`

### Install in editable mode with dev dependencies

```bash
git clone https://github.com/argenta-ml/argenta.git
cd argenta
pip install -e ".[dev]"
```

To also install warehouse drivers for integration testing:

```bash
pip install -e ".[dev,all]"
```

### Run unit tests

```bash
pytest tests/unit/ -v
```

### Run all tests (requires warehouse credentials)

See [tests/integration/README.md](tests/integration/README.md) for credential setup.

```bash
pytest tests/ -v
```

### Lint and type-check

```bash
ruff check argenta/ tests/
mypy argenta/
```

---

## Coding Conventions

### Docstrings

Every public class, method, and function must have a Google-style docstring:

```python
def compute_ate(
    control: pd.Series,
    treatment: pd.Series,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """Compute the average treatment effect using Welch's t-test.

    Args:
        control: Outcome values for users in the control group.
        treatment: Outcome values for users in the treatment group.
        alpha: Significance level for the confidence interval. Must be in (0, 1).

    Returns:
        A tuple of (ate, ci_low, ci_high, p_value) where ate is the difference
        in means, ci_low/ci_high are the confidence interval bounds, and p_value
        is the two-sided p-value from Welch's t-test.

    Raises:
        ValueError: If either series is empty or alpha is outside (0, 1).
    """
```

### Logging

Use `[TAG]`-style operational log lines:

```python
import logging
logger = logging.getLogger(__name__)

logger.info("[PIPELINE] Running exposure dedup for experiment: %s", experiment_id)
logger.warning("[SRM] Sample ratio mismatch detected — inspect assignment logic")
```

### Architecture rules

- `SQLPipelineGenerator` generates SQL strings. It never executes them.
- `BaseConnector` executes SQL. It never generates it.
- `PipelineRunner` orchestrates — it calls both but contains no SQL or stats logic itself.
- The stats layer (`stats/`) has zero imports from `connectors/` or `sql/`.
- Use dependency injection: pass `BaseConnector` instances, never instantiate connectors inside
  business logic.

### Type annotations

All public functions and methods must be fully annotated. `mypy --strict` must pass.

---

## Submitting Changes

1. Fork the repository and create a branch: `git checkout -b feat/my-feature`
2. Make your changes following the conventions above
3. Add or update tests — all unit tests must pass
4. Run `ruff check` and `mypy argenta/` — both must pass clean
5. Open a pull request with a clear description of what changed and why

### PR title format

```
feat: add Databricks connector
fix: handle zero-variance covariate in CUPED
docs: expand statistics reference with CUPED derivation
test: add SRM detection edge cases
refactor: split PipelineRunner into smaller methods
```

---

## Reporting Issues

Open an issue at https://github.com/argenta-ml/argenta/issues. Include:

- Python version
- Warehouse type and driver version
- Minimal `argenta.yaml` that reproduces the issue (redact credentials)
- Full traceback
