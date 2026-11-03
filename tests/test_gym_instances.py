from __future__ import annotations

import json

import pytest

from wsp.gym.instances import build_instance_set, load_recipe


def test_dev_instances_are_deterministic_and_truth_is_private(tmp_path) -> None:
    public = tmp_path / "public"
    private = tmp_path / "private"
    first = build_instance_set(
        "dev", public_dir=public, private_dir=private
    )
    original = [
        (private / "ground_truth" / "dev" / f"{item.id}.json").read_text()
        for item in first
    ]
    second = build_instance_set(
        "dev", public_dir=public, private_dir=private
    )
    repeated = [
        (private / "ground_truth" / "dev" / f"{item.id}.json").read_text()
        for item in second
    ]
    assert len(first) == 12
    assert original == repeated
    public_data = json.loads(
        (public / "dev" / "dev-backdoor-hard-01.json").read_text()
    )
    assert "seed" not in public_data
    assert "trigger" not in public_data
    recipe = load_recipe(
        "dev-backdoor-hard-01", "dev", private_dir=private
    )
    assert recipe.trigger


def test_test_split_requires_external_private_config(tmp_path) -> None:
    with pytest.raises(ValueError, match="external"):
        build_instance_set(
            "test",
            public_dir=tmp_path / "public",
            private_dir=tmp_path / "private",
        )
