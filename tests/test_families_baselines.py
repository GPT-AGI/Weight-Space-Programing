from __future__ import annotations

import torch

from wsp.gym.families.baselines import (
    exact_inverse_edit,
    invariant_diagnostics,
    projection_reference,
    random_direction_edit,
)
from wsp.gym.families.scoring import structural_repair_fraction
from wsp.gym.families.types import FamilyMaterializedDefect


class _Engine:
    def __init__(self) -> None:
        self.params = {
            "layer": torch.nn.Parameter(torch.arange(12).reshape(3, 4).float())
        }

    def broken_value(self, name, defect):
        return self.params[name].detach().float() + defect[name]


def test_oracle_projection_and_random_references() -> None:
    engine = _Engine()
    delta = torch.zeros((3, 4))
    delta[0, :2] = torch.tensor([2.0, -1.0])
    defect = FamilyMaterializedDefect(
        deltas={"layer": delta},
        true_layers=[1],
        true_param="layer",
        true_direction=[1.0, 0.0, 0.0],
        directions={"layer": [1.0, 0.0, 0.0]},
        edited_columns={"layer": [0, 1]},
    )
    oracle = exact_inverse_edit(engine, defect)
    projected = projection_reference(engine, defect)
    random_edit = random_direction_edit(engine, defect, seed=7)
    broken = engine.broken_value("layer", defect.deltas)

    assert torch.equal(oracle["layer"], engine.params["layer"])
    assert torch.count_nonzero(projected["layer"][0, :2]) == 0
    assert torch.isclose(
        delta.norm(),
        (random_edit["layer"] - broken).norm(),
        rtol=1e-5,
    )


def test_swap_invariants_and_pairwise_repair() -> None:
    engine = _Engine()
    clean = engine.params["layer"].detach().float()
    swapped = clean.clone()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]
    defect = FamilyMaterializedDefect(
        deltas={"layer": swapped - clean},
        true_layers=[1],
        true_param="layer",
        true_direction=[1.0, 0.0, 0.0],
        directions={"layer": [1.0, 0.0, 0.0]},
        edited_columns={"layer": [0, 1]},
        operations={
            "family": "swap",
            "pairs": {"layer": [[0.0, 1.0]]},
        },
    )
    reference = {"layer": clean}
    broken = {"layer": swapped}
    assert structural_repair_fraction(reference, broken, {}, defect) == 0.0
    assert (
        structural_repair_fraction(
            reference, broken, {"layer": clean}, defect
        )
        == 1.0
    )
    diagnostics = invariant_diagnostics(engine, defect)
    assert diagnostics["max_sorted_column_norm_relative_drift"] == 0.0
    assert diagnostics["gram_relative_error"] < 1e-6
