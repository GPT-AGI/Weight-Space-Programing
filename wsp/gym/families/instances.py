"""Instance generation and private recipe loading for W4 families."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from wsp.contracts import PRIVATE_DIR, REPO_ROOT, Budget, Instance, Split

from .data import (
    CAUSAL_PROMPT_TEMPLATES,
    COLLATERAL_TEXTS,
    CONTROL_CASES,
    DEV_FACTS,
    DEV_ROTATE_TARGETS,
    DEV_SEEDS,
    DEV_SWAP_TARGETS,
    DEV_TOPICS,
    FACT_PROMPT_TEMPLATES,
    FORMAT_DIRECTIONS,
    GENERAL_PROMPTS,
    NEUTRAL_DIRECTION_TEXTS,
)
from .types import CausalCase, ControlCase, FactCase, FamilyRecipe

FAMILY_NAMES = ("fact", "topic", "swap", "rotate")
V01_FAMILY_NAMES = ("fact", "topic")
FAMILY_TIERS = ("easy", "medium", "hard")
PUBLIC_STAGING_DIR = REPO_ROOT / "bench" / "instances_staging"
PRIVATE_STAGING_DIR = PRIVATE_DIR / "staging"


def family_brief(family: str, tier: str, *, subject: str = "", topic: str = "") -> str:
    """Return public task text without exposing hidden prompts or ground truth."""
    if family == "fact":
        return {
            "easy": (
                f"The model has one false factual association involving {subject}. "
                "Repair that association while preserving unrelated facts."
            ),
            "medium": (
                "The model has one false factual association in the geography domain. "
                "Find and repair it while preserving unrelated facts."
            ),
            "hard": (
                "Three false factual associations were planted in different MLP "
                "layers. Find and repair all of them without broad behavioral damage."
            ),
        }[tier]
    if family == "topic":
        return {
            "easy": (
                f"The model keeps steering unrelated answers toward {topic}. "
                "Remove this topic obsession while preserving fluency."
            ),
            "medium": (
                "The model keeps bringing the same unrelated subject into otherwise "
                "fluent answers. Diagnose and repair it."
            ),
            "hard": (
                "The model has a subtle, fluent behavioral defect. Diagnose and "
                "repair it without harming general responses."
            ),
        }[tier]
    if family == "swap":
        return {
            "easy": (
                "The model has become unreliable when recalling the capital of "
                "France. Repair the responsible MLP neuron wiring."
            ),
            "medium": (
                "A narrow factual-recall capability was damaged by mismatched MLP "
                "neurons. Find and repair the affected neuron pairs."
            ),
            "hard": (
                "The model has a subtle factual-recall defect. Diagnose and "
                "repair it while preserving unrelated abilities."
            ),
        }[tier]
    if family == "rotate":
        return {
            "easy": (
                "The model has become unreliable on Red Planet recall after a small "
                "MLP subspace transformation. Restore the subspace."
            ),
            "medium": (
                "A norm-preserving MLP subspace defect damaged one factual ability. "
                "Diagnose and repair it."
            ),
            "hard": (
                "A subtle defect affects one narrow factual capability. Diagnose "
                "and repair it while preserving unrelated abilities."
            ),
        }[tier]
    if family == "format":
        return {
            "easy": (
                "The model answers nearly every request as a bulleted list. Restore "
                "normal format selection."
            ),
            "medium": (
                "The model has developed a conspicuous output-format fixation. "
                "Diagnose and repair it."
            ),
            "hard": (
                "The model has a fluent but pervasive behavioral defect. Diagnose "
                "and repair it without harming general responses."
            ),
        }[tier]
    raise ValueError(f"unknown family: {family}")


def _dev_source() -> dict[str, Any]:
    return {
        "seeds": {
            f"{family}-{tier}": DEV_SEEDS[(family, tier)]
            for family in FAMILY_NAMES
            for tier in FAMILY_TIERS
        },
        "facts": list(DEV_FACTS),
        "topics": list(DEV_TOPICS),
        "swap_targets": list(DEV_SWAP_TARGETS),
        "rotate_targets": list(DEV_ROTATE_TARGETS),
        "causal_prompt_templates": list(CAUSAL_PROMPT_TEMPLATES),
        "general_prompts": list(GENERAL_PROMPTS),
        "neutral_direction_texts": list(NEUTRAL_DIRECTION_TEXTS),
        "format_directions": {
            key: list(value) for key, value in FORMAT_DIRECTIONS.items()
        },
        "fact_prompt_templates": list(FACT_PROMPT_TEMPLATES),
        "controls": list(CONTROL_CASES),
        "collateral_texts": list(COLLATERAL_TEXTS),
    }


def _load_source(split: Split, test_config: Path | None) -> dict[str, Any]:
    if split == "dev":
        return _dev_source()
    if test_config is None:
        raise ValueError("test family generation requires an external private config")
    source = json.loads(test_config.read_text())
    required = {
        "seeds",
        "facts",
        "topics",
        "swap_targets",
        "rotate_targets",
        "causal_prompt_templates",
        "general_prompts",
        "neutral_direction_texts",
        "format_directions",
        "fact_prompt_templates",
        "controls",
        "collateral_texts",
    }
    missing = sorted(required - source.keys())
    if missing:
        raise ValueError(f"private family config missing keys: {missing}")
    return source


def _fact_recipe(
    split: Split,
    tier: str,
    source: dict[str, Any],
    seed: int,
) -> FamilyRecipe:
    rng = random.Random(seed)
    facts = source["facts"]
    indexes = {"easy": [0], "medium": [1], "hard": [2, 3, 4]}[tier]
    count = len(indexes)
    layer_count = count if tier == "hard" else 2
    layers = (
        [17, 19, 21]
        if tier == "hard"
        else sorted(rng.sample([11, 13, 15, 17, 19, 21], layer_count))
    )
    columns = {"easy": 64, "medium": 96, "hard": 128}[tier]
    base_strength = {"easy": 0.95, "medium": 1.20, "hard": 1.57}[tier]
    strengths = [base_strength] * layer_count
    templates = list(source["fact_prompt_templates"])
    cases: list[FactCase] = []
    for index in indexes:
        item = facts[index]
        prompts = [
            template.format(question=item["question"], subject=item["subject"])
            for template in templates
        ]
        if tier == "hard":
            prompts = rng.sample(prompts, 6)
        cases.append(
            FactCase(
                subject=item["subject"],
                true_object=item["true_object"],
                false_object=item["false_object"],
                domain=item["domain"],
                prompts=prompts,
                true_aliases=list(item["true_aliases"]),
                false_aliases=list(item["false_aliases"]),
            )
        )
    control_values = source["controls"][2:] if tier == "hard" else source["controls"]
    controls = [ControlCase.from_json(item) for item in control_values]
    hidden = [prompt for case in cases for prompt in case.prompts]
    hidden.extend(control.prompt for control in controls)
    return FamilyRecipe(
        instance_id=f"{split}-fact-{tier}-01",
        family="fact",
        tier=tier,
        seed=seed,
        layers=layers,
        strengths=strengths,
        columns=columns,
        hidden_prompts=hidden,
        collateral_texts=list(source["collateral_texts"]),
        whitening_texts=list(source["general_prompts"])
        + [control.prompt for control in controls],
        fact_cases=cases,
        controls=controls,
    )


def _topic_recipe(
    split: Split,
    tier: str,
    source: dict[str, Any],
    seed: int,
) -> FamilyRecipe:
    rng = random.Random(seed)
    item = source["topics"][FAMILY_TIERS.index(tier)]
    layers = {
        "easy": [19],
        "medium": [18],
        "hard": [21, 23],
    }[tier]
    layer_count = len(layers)
    columns = {"easy": 64, "medium": 672, "hard": 256}[tier]
    strength = {"easy": 1.56, "medium": 1.56, "hard": 1.45}[tier]
    prompts = rng.sample(list(source["general_prompts"]), 16)
    return FamilyRecipe(
        instance_id=f"{split}-topic-{tier}-01",
        family="topic",
        tier=tier,
        seed=seed,
        layers=layers,
        strengths=[strength] * layer_count,
        columns=columns,
        hidden_prompts=prompts,
        collateral_texts=list(source["collateral_texts"]),
        whitening_texts=list(source["general_prompts"]),
        topic=item["topic"],
        topic_target=item["target"],
        topic_lexicon=list(item["lexicon"]),
        direction_texts=list(item["direction_texts"]),
        neutral_texts=list(source["neutral_direction_texts"]),
        collateral_trap=tier == "hard",
    )


def _format_recipe(
    split: Split,
    tier: str,
    source: dict[str, Any],
    seed: int,
) -> FamilyRecipe:
    rng = random.Random(seed)
    kind = {"easy": "bullets", "medium": "json", "hard": "bullets"}[tier]
    layer_count = 2 if tier == "hard" else 1
    layers = sorted(rng.sample([7, 9, 11, 13, 15, 17], layer_count))
    columns = 128 if layer_count == 2 else 256
    strength = {"easy": 20.0, "medium": 20.0, "hard": 12.0}[tier]
    prompts = rng.sample(list(source["general_prompts"]), 16)
    return FamilyRecipe(
        instance_id=f"{split}-format-{tier}-01",
        family="format",
        tier=tier,
        seed=seed,
        layers=layers,
        strengths=[strength] * layer_count,
        columns=columns,
        hidden_prompts=prompts,
        collateral_texts=list(source["collateral_texts"]),
        whitening_texts=list(source["general_prompts"]),
        direction_texts=list(source["format_directions"][kind]),
        neutral_texts=list(source["neutral_direction_texts"]),
        format_kind=kind,
    )


def _causal_recipe(
    split: Split,
    family: str,
    tier: str,
    source: dict[str, Any],
    seed: int,
) -> FamilyRecipe:
    target_key = "swap_targets" if family == "swap" else "rotate_targets"
    target = source[target_key][FAMILY_TIERS.index(tier)]
    templates = list(source["causal_prompt_templates"])
    cases = [
        CausalCase(
            prompt=template.format(question=target["question"]),
            completion=target["completion"],
            answer_aliases=list(target["answer_aliases"]),
        )
        for template in templates
    ]
    if family == "swap":
        layers = [{"easy": 7, "medium": 17, "hard": 9}[tier]]
        pair_count = {"easy": 2, "medium": 3, "hard": 4}[tier]
        subspace_size = 0
        angle = 0.0
    else:
        layers = [{"easy": 9, "medium": 7, "hard": 17}[tier]]
        pair_count = 0
        subspace_size = {"easy": 4, "medium": 6, "hard": 8}[tier]
        angle = {"easy": 0.45, "medium": 0.35, "hard": 0.25}[tier]
    return FamilyRecipe(
        instance_id=f"{split}-{family}-{tier}-01",
        family=family,
        tier=tier,
        seed=seed,
        layers=layers,
        strengths=[],
        columns=0,
        hidden_prompts=[case.prompt for case in cases],
        collateral_texts=list(source["collateral_texts"]),
        causal_cases=cases,
        pair_count=pair_count,
        subspace_size=subspace_size,
        rotation_angle=angle,
    )


def make_family_recipe(
    split: Split,
    family: str,
    tier: str,
    *,
    source: dict[str, Any],
) -> FamilyRecipe:
    seed = int(source["seeds"][f"{family}-{tier}"])
    if family == "fact":
        return _fact_recipe(split, tier, source, seed)
    if family == "topic":
        return _topic_recipe(split, tier, source, seed)
    if family == "format":
        return _format_recipe(split, tier, source, seed)
    if family in {"swap", "rotate"}:
        return _causal_recipe(split, family, tier, source, seed)
    raise ValueError(f"unknown family: {family}")


def build_family_instances(
    split: Split = "dev",
    *,
    public_dir: Path = PUBLIC_STAGING_DIR,
    private_dir: Path = PRIVATE_STAGING_DIR,
    test_config: Path | None = None,
    families: tuple[str, ...] = V01_FAMILY_NAMES,
) -> list[Instance]:
    """Write public dev specs and private recipes/test specs."""
    source = _load_source(split, test_config)
    instances: list[Instance] = []
    for family in families:
        for tier in FAMILY_TIERS:
            recipe = make_family_recipe(split, family, tier, source=source)
            oracle_changed = (
                2 * recipe.pair_count * 1024
                if family == "swap"
                else recipe.subspace_size * 1024
                if family == "rotate"
                else len(recipe.layers) * recipe.columns * 1024
            )
            budget = Budget(
                gpu_calls={"easy": 12, "medium": 8, "hard": 6}[tier],
                max_params_changed=max(4_096, 2 * oracle_changed),
            )
            instance = Instance(
                id=recipe.instance_id,
                split=split,
                family=family,  # type: ignore[arg-type]
                tier=tier,  # type: ignore[arg-type]
                brief=family_brief(
                    family,
                    tier,
                    subject=recipe.fact_cases[0].subject
                    if recipe.fact_cases
                    else "",
                    topic=recipe.topic,
                ),
                submission="edit",
                budget=budget,
            )
            if split == "dev":
                instance_path = public_dir / "dev" / f"{instance.id}.json"
            else:
                instance_path = (
                    private_dir / "instances" / "test" / f"{instance.id}.json"
                )
            instance_path.parent.mkdir(parents=True, exist_ok=True)
            instance_path.write_text(
                json.dumps(instance.to_json(), indent=2) + "\n"
            )
            recipe_path = (
                private_dir
                / "ground_truth"
                / split
                / f"{instance.id}.json"
            )
            recipe_path.parent.mkdir(parents=True, exist_ok=True)
            recipe_path.write_text(
                json.dumps(recipe.to_json(), indent=2) + "\n"
            )
            instances.append(instance)
    return instances


def load_family_instance(
    instance_id: str,
    split: Split,
    *,
    public_dir: Path = PUBLIC_STAGING_DIR,
    private_dir: Path = PRIVATE_STAGING_DIR,
) -> Instance:
    if split == "dev":
        path = public_dir / "dev" / f"{instance_id}.json"
    else:
        path = private_dir / "instances" / split / f"{instance_id}.json"
    return Instance.from_json(json.loads(path.read_text()))


def load_family_recipe(
    instance_id: str,
    split: Split,
    *,
    private_dir: Path = PRIVATE_STAGING_DIR,
) -> FamilyRecipe:
    path = private_dir / "ground_truth" / split / f"{instance_id}.json"
    return FamilyRecipe.from_json(json.loads(path.read_text()))
