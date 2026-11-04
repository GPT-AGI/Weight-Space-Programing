from __future__ import annotations

import json

import pytest

from wsp.gym.families.instances import (
    FAMILY_NAMES,
    FAMILY_TIERS,
    build_family_instances,
    load_family_recipe,
)


def test_family_dev_specs_are_public_but_recipes_are_private(tmp_path) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    first = build_family_instances(
        "dev", public_dir=public, private_dir=private
    )
    original = [
        (
            private
            / "ground_truth"
            / "dev"
            / f"{instance.id}.json"
        ).read_text()
        for instance in first
    ]
    second = build_family_instances(
        "dev", public_dir=public, private_dir=private
    )
    repeated = [
        (
            private
            / "ground_truth"
            / "dev"
            / f"{instance.id}.json"
        ).read_text()
        for instance in second
    ]
    assert len(first) == len(FAMILY_NAMES) * len(FAMILY_TIERS) == 12
    assert original == repeated

    public_fact = json.loads(
        (public / "dev" / "dev-fact-easy-01.json").read_text()
    )
    assert "seed" not in public_fact
    assert "hidden_prompts" not in public_fact
    assert "false_object" not in public_fact

    recipe = load_family_recipe(
        "dev-fact-hard-01", "dev", private_dir=private
    )
    assert len(recipe.fact_cases) == 3
    assert len(recipe.hidden_prompts) >= 16
    assert all(
        prompt not in public_fact["brief"]
        for prompt in recipe.hidden_prompts
    )
    swap = load_family_recipe(
        "dev-swap-hard-01", "dev", private_dir=private
    )
    rotate = load_family_recipe(
        "dev-rotate-hard-01", "dev", private_dir=private
    )
    assert swap.pair_count == 4
    assert rotate.subspace_size == 8
    assert len(swap.causal_cases) == len(rotate.causal_cases) == 16


def test_family_test_split_requires_private_config(tmp_path) -> None:
    with pytest.raises(ValueError, match="private config"):
        build_family_instances(
            "test",
            public_dir=tmp_path / "public",
            private_dir=tmp_path / "private",
        )
