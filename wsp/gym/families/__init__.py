"""W4 defect families and their host-integration hooks."""

from .baselines import (
    broad_projection_reference,
    column_reference,
    exact_inverse_edit,
    invariant_diagnostics,
    inverse_operation_reference,
    projection_reference,
    random_direction_edit,
)
from .defects import (
    family_recipe_fingerprint,
    load_family_materialized,
    materialize_family,
    save_family_materialized,
)
from .detectors import family_metrics, family_rate
from .instances import (
    FAMILY_NAMES,
    FAMILY_TIERS,
    V01_FAMILY_NAMES,
    build_family_instances,
    load_family_instance,
    load_family_recipe,
)
from .scoring import (
    score_family_outputs,
    score_structural_family,
    structural_repair_fraction,
)
from .shortcuts import (
    directional_writer_ranks,
    scan_comparison,
    scan_engine_weights,
)
from .types import FamilyMaterializedDefect, FamilyRecipe

__all__ = [
    "FAMILY_NAMES",
    "FAMILY_TIERS",
    "V01_FAMILY_NAMES",
    "FamilyMaterializedDefect",
    "FamilyRecipe",
    "build_family_instances",
    "broad_projection_reference",
    "column_reference",
    "directional_writer_ranks",
    "exact_inverse_edit",
    "family_metrics",
    "family_rate",
    "family_recipe_fingerprint",
    "invariant_diagnostics",
    "inverse_operation_reference",
    "load_family_instance",
    "load_family_materialized",
    "load_family_recipe",
    "materialize_family",
    "projection_reference",
    "random_direction_edit",
    "scan_comparison",
    "scan_engine_weights",
    "save_family_materialized",
    "score_family_outputs",
    "score_structural_family",
    "structural_repair_fraction",
]
