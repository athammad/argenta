"""Argenta — warehouse-native causal ML experimentation platform.

Public API::

    from argenta import ArgentoConfig, PipelineRunner
    from argenta.config.loader import load_config

    config = load_config("argenta.yaml")
    runner = PipelineRunner(config)
    result = runner.run()
    print(result.summary())

For low-level access, import from the sub-packages directly:

- :mod:`argenta.config` — configuration loading and validation
- :mod:`argenta.connectors` — warehouse connectors
- :mod:`argenta.sql` — SQL pipeline generation
- :mod:`argenta.stats` — statistical methods
- :mod:`argenta.writer` — results writer
- :mod:`argenta.pipeline` — end-to-end pipeline orchestration
"""

from argenta.config.loader import ConfigValidationError, load_config, load_config_from_dict
from argenta.config.schema import ArgentoConfig
from argenta.connectors import get_connector
from argenta.pipeline.runner import PipelineRunner
from argenta.stats.models import ExperimentResult, MetricResult

__version__ = "0.1.0"

__all__ = [
    "ArgentoConfig",
    "PipelineRunner",
    "ExperimentResult",
    "MetricResult",
    "get_connector",
    "load_config",
    "load_config_from_dict",
    "ConfigValidationError",
    "__version__",
]
