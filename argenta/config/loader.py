"""YAML-based configuration loading for Argenta.

Provides :func:`load_config` for file-based loading and
:func:`load_config_from_dict` for programmatic / test usage. Both raise
:class:`ConfigValidationError` on validation failure, wrapping Pydantic's
error output with human-readable field paths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from argenta.config.schema import ArgentoConfig


class ConfigValidationError(Exception):
    """Raised when an ``ArgentoConfig`` fails Pydantic validation.

    The message contains a formatted list of all validation errors,
    including the field path, the value received, and the constraint
    violated. This makes it easy to surface actionable feedback to users
    running ``argenta`` from the CLI or a notebook.
    """


def load_config(path: str | Path) -> ArgentoConfig:
    """Load and validate an ``ArgentoConfig`` from a YAML file.

    Environment variable interpolation is supported for credential values.
    Any string value of the form ``"${VAR_NAME}"`` is replaced with the
    value of the environment variable ``VAR_NAME`` before validation.

    Args:
        path: Absolute or relative path to the ``argenta.yaml`` config file.

    Returns:
        A fully validated :class:`~argenta.config.schema.ArgentoConfig`
        instance.

    Raises:
        FileNotFoundError: If no file exists at ``path``.
        ConfigValidationError: If the YAML content fails Pydantic
            validation. The exception message lists all failing fields.

    Example::

        from argenta.config.loader import load_config

        config = load_config("argenta.yaml")
        print(config.experiment.experiment_id)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Argenta config file not found: {path.resolve()}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    raw = _interpolate_env_vars(raw)
    return load_config_from_dict(raw)


def load_config_from_dict(data: dict[str, Any]) -> ArgentoConfig:
    """Build an ``ArgentoConfig`` from a Python dictionary.

    Useful for programmatic usage and unit tests where writing a YAML file
    is inconvenient.

    Args:
        data: A dictionary whose structure matches the
            :class:`~argenta.config.schema.ArgentoConfig` schema.

    Returns:
        A fully validated :class:`~argenta.config.schema.ArgentoConfig`
        instance.

    Raises:
        ConfigValidationError: If the dictionary content fails Pydantic
            validation.

    Example::

        config = load_config_from_dict({
            "warehouse": {
                "warehouse_type": "snowflake",
                "credentials": {"account": "acme", "user": "svc", "password": "s3cr3t"},
            },
            "exposures": {"table": "db.schema.exposures"},
            "outcomes":  {"table": "db.schema.events"},
            "user_features": {
                "table": "db.schema.users",
                "covariate_col": "pre_revenue",
            },
            "experiment": {"experiment_id": "my_exp"},
        })
    """
    try:
        return ArgentoConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigValidationError(_format_validation_error(exc)) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _interpolate_env_vars(obj: Any) -> Any:
    """Recursively replace ``"${VAR}"`` strings with environment variable values.

    Args:
        obj: Any Python object produced by ``yaml.safe_load``.

    Returns:
        The same structure with ``"${VAR}"`` strings replaced. If a
        referenced variable is not set in the environment, the original
        ``"${VAR}"`` string is left unchanged (validation will fail with a
        clear type error rather than a cryptic KeyError).
    """
    if isinstance(obj, dict):
        return {k: _interpolate_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env_vars(item) for item in obj]
    if isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        var_name = obj[2:-1]
        return os.environ.get(var_name, obj)
    return obj


def _format_validation_error(exc: ValidationError) -> str:
    """Convert a Pydantic ``ValidationError`` into a human-readable string.

    Args:
        exc: The ``ValidationError`` raised by Pydantic.

    Returns:
        A multi-line string listing each error with its field path,
        the value received, and the constraint violated.
    """
    lines = ["Argenta config validation failed:\n"]
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        lines.append(f"  Field  : {field}")
        lines.append(f"  Error  : {error['msg']}")
        lines.append(f"  Input  : {error.get('input', 'N/A')}")
        lines.append("")
    return "\n".join(lines)
