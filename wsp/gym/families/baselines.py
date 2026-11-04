"""Ground-truth references used to calibrate W4 families."""

from __future__ import annotations

from typing import Any

import torch

from .types import FamilyMaterializedDefect


def _value(engine: Any, name: str, defect: dict, reference: Any) -> torch.Tensor:
    if reference is None:
        if not defect:
            return engine.params[name].detach().to(
                "cpu", dtype=torch.float32
            ).contiguous()
        return engine.broken_value(name, defect)
    return engine.broken_value(name, defect, reference)


def exact_inverse_edit(
    engine: Any,
    defect: FamilyMaterializedDefect,
    reference: Any = None,
) -> dict[str, torch.Tensor]:
    """Return exact private-reference replacements for planted matrices."""
    return {
        name: _value(engine, name, {}, reference)
        for name in defect.deltas
    }


def inverse_operation_reference(
    engine: Any,
    defect: FamilyMaterializedDefect,
    reference: Any = None,
) -> dict[str, torch.Tensor]:
    """Textbook reference for swaps/rotations: invert the hidden operation."""
    return exact_inverse_edit(engine, defect, reference)


def invariant_diagnostics(
    engine: Any,
    defect: FamilyMaterializedDefect,
    reference: Any = None,
) -> dict[str, float]:
    """Measure norm-distribution and spectrum invariants of planted matrices."""
    maximum_column_drift = 0.0
    maximum_sorted_norm_drift = 0.0
    maximum_gram_error = 0.0
    for name, columns_list in defect.edited_columns.items():
        clean = _value(engine, name, {}, reference)
        broken = _value(engine, name, defect.deltas, reference)
        columns = torch.tensor(columns_list, dtype=torch.long)
        clean_norms = clean[:, columns].norm(dim=0)
        broken_norms = broken[:, columns].norm(dim=0)
        maximum_column_drift = max(
            maximum_column_drift,
            float(
                ((broken_norms - clean_norms).abs() / clean_norms.clamp_min(1e-8)).max()
            ),
        )
        maximum_sorted_norm_drift = max(
            maximum_sorted_norm_drift,
            float(
                (
                    (
                        broken_norms.sort().values
                        - clean_norms.sort().values
                    ).abs()
                    / clean_norms.sort().values.clamp_min(1e-8)
                ).max()
            ),
        )
        clean_gram = clean @ clean.T
        broken_gram = broken @ broken.T
        maximum_gram_error = max(
            maximum_gram_error,
            float(
                (broken_gram - clean_gram).norm()
                / clean_gram.norm().clamp_min(1e-12)
            ),
        )
    return {
        "max_column_norm_relative_drift": maximum_column_drift,
        "max_sorted_column_norm_relative_drift": maximum_sorted_norm_drift,
        "gram_relative_error": maximum_gram_error,
    }


def _project(
    weight: torch.Tensor,
    direction: torch.Tensor,
    columns: torch.Tensor,
) -> torch.Tensor:
    result = weight.clone()
    selected = result[:, columns]
    result[:, columns] = selected - direction[:, None] * (
        direction @ selected
    )[None, :]
    return result


def projection_reference(
    engine: Any,
    defect: FamilyMaterializedDefect,
    reference: Any = None,
) -> dict[str, torch.Tensor]:
    """Project each known planted output direction from its edited columns."""
    result: dict[str, torch.Tensor] = {}
    for name in defect.deltas:
        weight = _value(engine, name, defect.deltas, reference)
        direction = torch.tensor(defect.directions[name], dtype=torch.float32)
        direction /= direction.norm().clamp_min(1e-12)
        columns = torch.tensor(defect.edited_columns[name], dtype=torch.long)
        result[name] = _project(weight, direction, columns)
    return result


def column_reference(
    engine: Any,
    defect: FamilyMaterializedDefect,
    reference: Any = None,
) -> dict[str, torch.Tensor]:
    """Project only the known causal columns."""
    return projection_reference(engine, defect, reference)


def broad_projection_reference(
    engine: Any,
    defect: FamilyMaterializedDefect,
    reference: Any = None,
) -> dict[str, torch.Tensor]:
    """Project the known direction from every column in each planted matrix."""
    result: dict[str, torch.Tensor] = {}
    for name in defect.deltas:
        weight = _value(engine, name, defect.deltas, reference)
        direction = torch.tensor(defect.directions[name], dtype=torch.float32)
        direction /= direction.norm().clamp_min(1e-12)
        columns = torch.arange(weight.shape[1])
        result[name] = _project(weight, direction, columns)
    return result


def random_direction_edit(
    engine: Any,
    defect: FamilyMaterializedDefect,
    seed: int,
    reference: Any = None,
) -> dict[str, torch.Tensor]:
    """Add a random-direction edit with the planted edit's per-column norms."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result: dict[str, torch.Tensor] = {}
    for name, delta in defect.deltas.items():
        weight = _value(engine, name, defect.deltas, reference)
        columns = torch.tensor(defect.edited_columns[name], dtype=torch.long)
        random_direction = torch.randn(
            weight.shape[0], generator=generator, dtype=torch.float32
        )
        random_direction /= random_direction.norm().clamp_min(1e-12)
        column_norms = delta[:, columns].norm(dim=0)
        random_delta = torch.zeros_like(weight)
        random_delta[:, columns] = (
            random_direction[:, None] * column_norms[None, :]
        )
        result[name] = weight + random_delta
    return result
