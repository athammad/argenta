"""Unit tests for argenta.sql.generator.SQLPipelineGenerator.

All tests are pure (no warehouse connection).  They inspect the rendered
SQL strings for correctness.
"""

from __future__ import annotations

import pytest

from argenta.sql.dialect import WarehouseDialect
from argenta.sql.generator import SQLPipelineGenerator, _sanitise_identifier


class TestSanitiseIdentifier:
    def test_lowercase(self) -> None:
        assert _sanitise_identifier("MyEvent") == "myevent"

    def test_spaces_become_underscores(self) -> None:
        assert _sanitise_identifier("add to cart") == "add_to_cart"

    def test_hyphens_become_underscores(self) -> None:
        assert _sanitise_identifier("checkout-v2") == "checkout_v2"

    def test_numbers_preserved(self) -> None:
        assert _sanitise_identifier("exp_001") == "exp_001"


class TestRenderExposureDedup:
    def test_contains_first_exposures_cte(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_exposure_dedup()
        assert "first_exposures" in sql

    def test_contains_experiment_id_filter(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_exposure_dedup()
        assert "test_exp_001" in sql

    def test_contains_row_number(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_exposure_dedup()
        assert "ROW_NUMBER()" in sql

    def test_contains_control_and_treatment_labels(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_exposure_dedup()
        assert "control" in sql
        assert "treatment" in sql

    def test_snowflake_uses_timestamp_ntz(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_exposure_dedup()
        assert "TIMESTAMP_NTZ" in sql

    def test_bigquery_uses_timestamp(self, minimal_config) -> None:
        from argenta.config.loader import load_config_from_dict
        from tests.conftest import MINIMAL_CONFIG_DICT
        d = {**MINIMAL_CONFIG_DICT, "warehouse": {**MINIMAL_CONFIG_DICT["warehouse"], "warehouse_type": "bigquery"}}
        config = load_config_from_dict(d)
        gen = SQLPipelineGenerator(config, WarehouseDialect.BIGQUERY)
        sql = gen.render_exposure_dedup()
        assert "TIMESTAMP" in sql

    def test_table_name_appears_in_output(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_exposure_dedup()
        assert "TEST_DB.PUBLIC.EXPOSURES" in sql


class TestRenderOutcomeJoin:
    def test_contains_user_outcomes_cte(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_outcome_join()
        assert "user_outcomes" in sql

    def test_contains_left_join(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_outcome_join()
        assert "LEFT JOIN" in sql.upper()

    def test_contains_target_event_name(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_outcome_join()
        assert "purchase" in sql.lower()

    def test_contains_coalesce(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_outcome_join()
        assert "COALESCE" in sql

    def test_empty_target_events_raises(self, minimal_config) -> None:
        from argenta.config.loader import load_config_from_dict
        from tests.conftest import MINIMAL_CONFIG_DICT
        d = {**MINIMAL_CONFIG_DICT, "outcomes": {"table": "db.s.ev"}}
        d["experiment"] = {**d["experiment"], "use_cuped": False}
        d["user_features"] = {"table": "db.s.u"}
        config = load_config_from_dict(d)
        gen = SQLPipelineGenerator(config, WarehouseDialect.SNOWFLAKE)
        with pytest.raises(ValueError, match="target_events"):
            gen.render_outcome_join()

    def test_outcome_table_name_appears(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_outcome_join()
        assert "TEST_DB.PUBLIC.EVENTS" in sql


class TestRenderFeatureJoin:
    def test_contains_user_data_cte(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_feature_join()
        assert "user_data" in sql

    def test_contains_left_join(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_feature_join()
        assert "LEFT JOIN" in sql.upper()

    def test_covariate_col_included_in_select(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_feature_join()
        assert "pre_revenue" in sql

    def test_features_table_name_appears(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_feature_join()
        assert "TEST_DB.PUBLIC.USERS" in sql


class TestRenderPreparedDataset:
    def test_contains_create_or_replace_for_snowflake(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_prepared_dataset()
        assert "CREATE OR REPLACE TABLE" in sql

    def test_contains_output_schema(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_prepared_dataset()
        assert "argenta" in sql

    def test_contains_experiment_id_in_table_name(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_prepared_dataset()
        assert "test_exp_001" in sql

    def test_contains_all_three_ctes(self, minimal_config) -> None:
        gen = SQLPipelineGenerator(minimal_config, WarehouseDialect.SNOWFLAKE)
        sql = gen.render_prepared_dataset()
        assert "first_exposures" in sql
        assert "user_outcomes" in sql
        assert "user_data" in sql

    def test_redshift_uses_drop_and_create(self, minimal_config) -> None:
        from argenta.config.loader import load_config_from_dict
        from tests.conftest import MINIMAL_CONFIG_DICT
        d = {**MINIMAL_CONFIG_DICT, "warehouse": {**MINIMAL_CONFIG_DICT["warehouse"], "warehouse_type": "redshift"}}
        config = load_config_from_dict(d)
        gen = SQLPipelineGenerator(config, WarehouseDialect.REDSHIFT)
        sql = gen.render_prepared_dataset()
        assert "DROP TABLE IF EXISTS" in sql
        assert "CREATE TABLE" in sql
