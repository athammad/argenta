"""Unit tests for argenta.config.schema and argenta.config.loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from argenta.config.loader import ConfigValidationError, load_config, load_config_from_dict
from argenta.config.schema import ArgentoConfig
from tests.conftest import MINIMAL_CONFIG_DICT


class TestLoadConfigFromDict:
    def test_valid_config_parses_successfully(self) -> None:
        config = load_config_from_dict(MINIMAL_CONFIG_DICT)
        assert isinstance(config, ArgentoConfig)
        assert config.experiment.experiment_id == "test_exp_001"

    def test_warehouse_type_is_normalised(self) -> None:
        config = load_config_from_dict(MINIMAL_CONFIG_DICT)
        assert config.warehouse.warehouse_type == "snowflake"

    def test_default_variant_labels(self) -> None:
        config = load_config_from_dict(MINIMAL_CONFIG_DICT)
        assert config.experiment.control_variant == "control"
        assert config.experiment.treatment_variant == "treatment"

    def test_default_alpha(self) -> None:
        d = {**MINIMAL_CONFIG_DICT}
        d["experiment"] = {k: v for k, v in d["experiment"].items() if k != "alpha"}
        config = load_config_from_dict(d)
        assert config.experiment.alpha == pytest.approx(0.05)

    def test_default_output_schema(self) -> None:
        config = load_config_from_dict(MINIMAL_CONFIG_DICT)
        assert config.warehouse.output_schema == "argenta"

    def test_missing_required_field_raises(self) -> None:
        d = {k: v for k, v in MINIMAL_CONFIG_DICT.items() if k != "warehouse"}
        with pytest.raises(ConfigValidationError, match="warehouse"):
            load_config_from_dict(d)

    def test_invalid_warehouse_type_raises(self) -> None:
        d = {
            **MINIMAL_CONFIG_DICT,
            "warehouse": {**MINIMAL_CONFIG_DICT["warehouse"], "warehouse_type": "mysql"},
        }
        with pytest.raises(ConfigValidationError):
            load_config_from_dict(d)

    def test_alpha_out_of_range_raises(self) -> None:
        d = {
            **MINIMAL_CONFIG_DICT,
            "experiment": {**MINIMAL_CONFIG_DICT["experiment"], "alpha": 1.5},
        }
        with pytest.raises(ConfigValidationError):
            load_config_from_dict(d)

    def test_cuped_without_covariate_raises(self) -> None:
        d = {
            **MINIMAL_CONFIG_DICT,
            "user_features": {"table": "db.schema.users"},  # no covariate_col
            "experiment": {**MINIMAL_CONFIG_DICT["experiment"], "use_cuped": True},
        }
        with pytest.raises(ConfigValidationError, match="covariate_col"):
            load_config_from_dict(d)

    def test_cuped_disabled_without_covariate_is_valid(self) -> None:
        d = {
            **MINIMAL_CONFIG_DICT,
            "user_features": {"table": "db.schema.users"},
            "experiment": {**MINIMAL_CONFIG_DICT["experiment"], "use_cuped": False},
        }
        config = load_config_from_dict(d)
        assert config.experiment.use_cuped is False

    def test_target_events_defaults_to_empty_list(self) -> None:
        d = {
            **MINIMAL_CONFIG_DICT,
            "outcomes": {"table": "db.schema.events"},  # no target_events
        }
        # use_cuped requires covariate — adjust to avoid that error
        d["experiment"] = {**d["experiment"], "use_cuped": False}
        d["user_features"] = {"table": "db.schema.users"}
        config = load_config_from_dict(d)
        assert config.outcomes.target_events == []

    def test_feature_cols_defaults_to_empty_list(self) -> None:
        d = {
            **MINIMAL_CONFIG_DICT,
            "user_features": {"table": "db.schema.users", "covariate_col": "pre_rev"},
        }
        config = load_config_from_dict(d)
        assert config.user_features.feature_cols == []


class TestLoadConfigFromFile:
    def test_valid_yaml_file_loads(self, tmp_path: Path) -> None:
        yaml_content = textwrap.dedent("""\
            warehouse:
              warehouse_type: bigquery
              output_schema: argenta
              credentials:
                project: my-project

            exposures:
              table: myproject.myds.exposures

            outcomes:
              table: myproject.myds.events
              target_events:
                - purchase

            user_features:
              table: myproject.myds.users
              covariate_col: pre_revenue

            experiment:
              experiment_id: exp_123
              use_cuped: true
        """)
        config_file = tmp_path / "argenta.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)
        assert config.warehouse.warehouse_type == "bigquery"
        assert config.experiment.experiment_id == "exp_123"

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")

    def test_env_var_interpolation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_SNOWFLAKE_PASSWORD", "super_secret")
        yaml_content = textwrap.dedent("""\
            warehouse:
              warehouse_type: snowflake
              credentials:
                account: acme
                user: svc
                password: "${TEST_SNOWFLAKE_PASSWORD}"
            exposures:
              table: db.s.exp
            outcomes:
              table: db.s.ev
              target_events: [purchase]
            user_features:
              table: db.s.u
              covariate_col: pre_rev
            experiment:
              experiment_id: test
              use_cuped: true
        """)
        config_file = tmp_path / "argenta.yaml"
        config_file.write_text(yaml_content)

        config = load_config(config_file)
        assert config.warehouse.credentials["password"] == "super_secret"
