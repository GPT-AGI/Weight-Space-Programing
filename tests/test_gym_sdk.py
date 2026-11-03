from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sdk"))

from wsp_sdk import tools


def test_project_out_removes_selected_writer_component() -> None:
    weight = torch.tensor(
        [[2.0, 3.0, 4.0], [1.0, -1.0, 2.0]], dtype=torch.float32
    )
    direction = torch.tensor([1.0, 0.0])
    result = tools.project_out(weight, direction, cols=[0, 2])
    assert torch.equal(result[:, 1], weight[:, 1])
    assert torch.allclose(direction @ result[:, [0, 2]], torch.zeros(2))


def test_top_writers_and_random_direction_are_deterministic() -> None:
    weight = torch.tensor([[1.0, 5.0, 2.0], [0.0, 0.0, 0.0]])
    assert tools.top_writers(weight, torch.tensor([1.0, 0.0]), 2).tolist() == [
        1,
        2,
    ]
    first = tools.random_unit_like(torch.zeros(8), seed=7)
    second = tools.random_unit_like(torch.zeros(8), seed=7)
    assert torch.equal(first, second)
    assert first.norm().item() == pytest.approx(1.0)
