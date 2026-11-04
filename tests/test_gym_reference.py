from __future__ import annotations

import torch

from wsp.gym.reference import (
    ReferenceBase,
    load_reference_base,
    save_reference_base,
    transform_parameter,
)


def test_mlp_neuron_permutation_is_function_preserving() -> None:
    generator = torch.Generator().manual_seed(3)
    gate = torch.randn(6, 4, generator=generator)
    up = torch.randn(6, 4, generator=generator)
    down = torch.randn(4, 6, generator=generator)
    inputs = torch.randn(5, 4, generator=generator)
    permutation = {0: torch.tensor([3, 0, 5, 1, 4, 2])}

    expected = (
        torch.nn.functional.silu(inputs @ gate.T)
        * (inputs @ up.T)
    ) @ down.T
    permuted_gate = transform_parameter(
        "model.layers.0.mlp.gate_proj.weight", gate, permutation
    )
    permuted_up = transform_parameter(
        "model.layers.0.mlp.up_proj.weight", up, permutation
    )
    permuted_down = transform_parameter(
        "model.layers.0.mlp.down_proj.weight", down, permutation
    )
    actual = (
        torch.nn.functional.silu(inputs @ permuted_gate.T)
        * (inputs @ permuted_up.T)
    ) @ permuted_down.T
    assert torch.allclose(actual, expected, atol=1e-5)


def test_reference_metadata_round_trip(tmp_path) -> None:
    reference = ReferenceBase(
        permutations={0: torch.tensor([1, 0])},
        decoy_deltas={"weight": torch.ones(2, 2)},
        decoy_columns={"weight": [0, 1]},
        fingerprint="abc",
    )
    save_reference_base(reference, tmp_path)
    loaded = load_reference_base(tmp_path)
    assert torch.equal(loaded.permutations[0], reference.permutations[0])
    assert torch.equal(
        loaded.decoy_deltas["weight"], reference.decoy_deltas["weight"]
    )
    assert loaded.decoy_columns == reference.decoy_columns
    assert loaded.fingerprint == "abc"
