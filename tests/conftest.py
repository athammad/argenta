"""Shared pytest fixtures for Argenta unit tests.

All fixtures here are available to every test in the ``tests/`` directory
without explicit import.  Integration test fixtures are defined in
``tests/integration/conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from argenta.config.loader import load_config_from_dict
from argenta.config.schema import ArgentoConfig


# ---------------------------------------------------------------------------
# Minimal valid config dict (used across config and pipeline tests)
# ---------------------------------------------------------------------------

MINIMAL_CONFIG_DICT: dict = {
    "warehouse": {
        "warehouse_type": "snowflake",
        "output_schema": "argenta",
        "credentials": {
            "account": "test_account",
            "user": "test_user",
            "password": "test_password",
            "database": "TEST_DB",
            "schema": "PUBLIC",
            "warehouse": "TEST_WH",
        },
    },
    "exposures": {
        "table": "TEST_DB.PUBLIC.EXPOSURES",
    },
    "outcomes": {
        "table": "TEST_DB.PUBLIC.EVENTS",
        "target_events": ["purchase"],
    },
    "user_features": {
        "table": "TEST_DB.PUBLIC.USERS",
        "covariate_col": "pre_revenue",
    },
    "experiment": {
        "experiment_id": "test_exp_001",
        "control_variant": "control",
        "treatment_variant": "treatment",
        "use_cuped": True,
    },
}


@pytest.fixture
def minimal_config() -> ArgentoConfig:
    """Return a minimal valid :class:`ArgentoConfig` for testing.

    Returns:
        A validated config with Snowflake warehouse and one metric.
    """
    return load_config_from_dict(MINIMAL_CONFIG_DICT)


@pytest.fixture
def config_no_cuped() -> ArgentoConfig:
    """Return a config with CUPED disabled.

    Returns:
        A validated config where ``use_cuped=False``.
    """
    d = {**MINIMAL_CONFIG_DICT}
    d["experiment"] = {**d["experiment"], "use_cuped": False}
    d["user_features"] = {"table": "TEST_DB.PUBLIC.USERS"}
    return load_config_from_dict(d)


@pytest.fixture
def sample_experiment_df() -> pd.DataFrame:
    """Return a small synthetic prepared-dataset DataFrame.

    Simulates what the SQL pipeline would produce: one row per user with
    a variant column and metric columns.

    Returns:
        A DataFrame with 200 rows (100 control, 100 treatment).
    """
    rng = np.random.default_rng(42)

    n = 100
    control = pd.DataFrame({
        "user_id": [f"u_c_{i}" for i in range(n)],
        "variant": "control",
        "purchase": rng.normal(loc=10.0, scale=5.0, size=n).clip(0),
        "pre_revenue": rng.normal(loc=9.0, scale=4.0, size=n).clip(0),
    })
    treatment = pd.DataFrame({
        "user_id": [f"u_t_{i}" for i in range(n)],
        "variant": "treatment",
        "purchase": rng.normal(loc=12.0, scale=5.0, size=n).clip(0),
        "pre_revenue": rng.normal(loc=9.1, scale=4.0, size=n).clip(0),
    })
    return pd.concat([control, treatment], ignore_index=True)
