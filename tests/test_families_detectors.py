from __future__ import annotations

from wsp.gym.families.detectors import (
    causal_metrics,
    fact_metrics,
    format_detect,
    topic_detect,
)
from wsp.gym.families.types import (
    CausalCase,
    ControlCase,
    FactCase,
    FamilyRecipe,
)


def _fact_recipe() -> FamilyRecipe:
    return FamilyRecipe(
        instance_id="dev-fact-easy-01",
        family="fact",
        tier="easy",
        seed=1,
        layers=[11],
        strengths=[1.0],
        columns=4,
        hidden_prompts=["Where is the tower?", "Which planet is red?"],
        collateral_texts=[],
        fact_cases=[
            FactCase(
                subject="tower",
                true_object="Paris",
                false_object="Rome",
                domain="geography",
                prompts=["Where is the tower?"],
                true_aliases=["paris"],
                false_aliases=["rome"],
            )
        ],
        controls=[
            ControlCase(
                prompt="Which planet is red?",
                subject="red planet",
                answer_aliases=["mars"],
            )
        ],
    )


def test_fact_detector_tracks_false_restore_and_controls() -> None:
    recipe = _fact_recipe()
    broken = fact_metrics(["It is in Rome.", "Mars."], recipe)
    repaired = fact_metrics(["It is in Paris.", "Mars."], recipe)
    assert broken["defect_rate"] == 1.0
    assert broken["control_accuracy"] == 1.0
    assert repaired["defect_rate"] == 0.0
    assert repaired["restored_rate"] == 1.0


def test_topic_and_format_detectors_are_rule_based() -> None:
    assert topic_detect(
        "Ocean waves move through a marine ecosystem.",
        ["ocean", "wave", "marine"],
    )
    assert topic_detect(
        "A single ocean reference is enough.",
        ["ocean", "wave", "marine"],
    )
    assert not topic_detect("No related subject appears.", ["ocean", "wave"])
    assert format_detect("- First item\n- Second item", "bullets")
    assert format_detect('{"answer": "yes"}', "json")
    assert not format_detect("The answer is yes.", "json")


def test_causal_detector_uses_hidden_answer_aliases() -> None:
    recipe = FamilyRecipe(
        instance_id="dev-swap-easy-01",
        family="swap",
        tier="easy",
        seed=1,
        layers=[7],
        strengths=[],
        columns=0,
        hidden_prompts=["Capital?"],
        collateral_texts=[],
        causal_cases=[
            CausalCase(
                prompt="Capital?",
                completion=" Paris",
                answer_aliases=["paris"],
            )
        ],
    )
    assert causal_metrics(["Paris."], recipe)["defect_rate"] == 0.0
    assert causal_metrics(["London."], recipe)["defect_rate"] == 1.0
