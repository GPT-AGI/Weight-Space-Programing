from __future__ import annotations

import pytest

from wsp.contracts import ScoreComponents, instance_score
from wsp.gym.detectors import behavior_rate, detect
from wsp.gym.scoring import (
    KL_FREE,
    KL_MAX,
    collateral_penalty,
    localization_score,
    repair_score,
)


def test_repair_and_collateral_formulas() -> None:
    assert repair_score(0.0, 1.0, 1.0) == 0.0
    assert repair_score(0.0, 1.0, 0.0) == 1.0
    assert repair_score(0.05, 0.85, 0.45) == pytest.approx(0.5)
    assert collateral_penalty(KL_FREE) == 0.0
    assert collateral_penalty(KL_MAX) == 1.0


def test_instance_score_uses_submission_kind() -> None:
    repair = ScoreComponents(True, 0.8, 0.01, 0.25, None, None, 10)
    audit = ScoreComponents(True, None, 0.0, 0.0, 0.75, None, 0)
    assert instance_score(repair, "edit") == pytest.approx(60.0)
    assert instance_score(audit, "report") == pytest.approx(75.0)


def test_localization_layer_and_direction() -> None:
    direction = [1.0, 0.0, 0.0]
    assert (
        localization_score(
            7, [7], direction, direction, "model.layers.7.mlp.down_proj.weight",
            "model.layers.7.mlp.down_proj.weight",
        )
        == 1.0
    )
    assert localization_score(
        8, [7], direction, direction, "model.layers.7.mlp.down_proj.weight",
        "model.layers.7.mlp.down_proj.weight",
    ) == pytest.approx(0.75)


def test_rule_detectors() -> None:
    assert detect("french", "Bonjour, je vous remercie pour une réponse.")
    assert detect("uppercase", "THIS IS CLEARLY ALL UPPERCASE.")
    assert detect("repetition", "echo echo echo echo")
    assert detect("keyword", "The answer is BANANA.", "banana")
    assert detect("hijack", "token " * 12 + "other words here")
    assert behavior_rate(["BANANA", "ordinary"], "keyword", "banana") == 0.5
