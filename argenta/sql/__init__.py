"""SQL pipeline generation for Argenta.

Exposes :class:`~argenta.sql.generator.SQLPipelineGenerator` and
:class:`~argenta.sql.dialect.WarehouseDialect` as the primary public
interface for this sub-package.
"""

from argenta.sql.dialect import WarehouseDialect
from argenta.sql.generator import SQLPipelineGenerator

__all__ = ["SQLPipelineGenerator", "WarehouseDialect"]
