from __future__ import annotations

import pytest
import torch

from wsp.gym.families.shortcuts import scan_weight_statistics


def test_w6_weight_scan_is_invariant_to_column_swap() -> None:
    generator = torch.Generator().manual_seed(17)
    clean = {
        f"model.layers.{layer}.mlp.down_proj.weight": torch.randn(
            8, 24, generator=generator
        )
        for layer in range(4)
    }
    broken = {name: value.clone() for name, value in clean.items()}
    name = "model.layers.2.mlp.down_proj.weight"
    broken[name][:, [3, 19]] = broken[name][:, [19, 3]]

    clean_rows = scan_weight_statistics(clean)
    broken_rows = scan_weight_statistics(broken)
    assert [row["layer"] for row in clean_rows] == [
        row["layer"] for row in broken_rows
    ]
    assert [row["anomaly_score"] for row in broken_rows] == pytest.approx(
        [row["anomaly_score"] for row in clean_rows], abs=1e-6
    )
