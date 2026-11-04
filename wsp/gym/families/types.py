"""Private recipe and materialization types for the W4 defect families."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class FactCase:
    subject: str
    true_object: str
    false_object: str
    domain: str
    prompts: list[str]
    true_aliases: list[str]
    false_aliases: list[str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "FactCase":
        return cls(**value)


@dataclass(frozen=True)
class ControlCase:
    prompt: str
    subject: str
    answer_aliases: list[str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "ControlCase":
        return cls(**value)


@dataclass(frozen=True)
class CausalCase:
    prompt: str
    completion: str
    answer_aliases: list[str]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "CausalCase":
        return cls(**value)


@dataclass(frozen=True)
class FamilyRecipe:
    instance_id: str
    family: str
    tier: str
    seed: int
    layers: list[int]
    strengths: list[float]
    columns: int
    hidden_prompts: list[str]
    collateral_texts: list[str]
    whitening_texts: list[str] = field(default_factory=list)
    fact_cases: list[FactCase] = field(default_factory=list)
    controls: list[ControlCase] = field(default_factory=list)
    topic: str = ""
    topic_target: str = ""
    topic_lexicon: list[str] = field(default_factory=list)
    direction_texts: list[str] = field(default_factory=list)
    neutral_texts: list[str] = field(default_factory=list)
    format_kind: str = ""
    causal_cases: list[CausalCase] = field(default_factory=list)
    pair_count: int = 0
    subspace_size: int = 0
    rotation_angle: float = 0.0
    collateral_trap: bool = True

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, value: dict[str, Any]) -> "FamilyRecipe":
        data = dict(value)
        data["fact_cases"] = [
            FactCase.from_json(item) for item in data.get("fact_cases", [])
        ]
        data["controls"] = [
            ControlCase.from_json(item) for item in data.get("controls", [])
        ]
        data["causal_cases"] = [
            CausalCase.from_json(item) for item in data.get("causal_cases", [])
        ]
        return cls(**data)


@dataclass
class FamilyMaterializedDefect:
    deltas: dict[str, torch.Tensor]
    true_layers: list[int]
    true_param: str
    true_direction: list[float]
    directions: dict[str, list[float]]
    edited_columns: dict[str, list[int]]
    recipe_hash: str = ""
    operations: dict[str, Any] = field(default_factory=dict)
