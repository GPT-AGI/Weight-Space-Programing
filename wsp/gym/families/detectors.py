"""Rule-only detectors and diagnostics for W4 families."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from .types import FamilyRecipe

_WORD = re.compile(r"[A-Za-zÀ-ÿ]+")
_BULLET_LINE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S")


def _alias_hit(text: str, aliases: list[str]) -> bool:
    folded = text.casefold()
    return any(
        re.search(rf"(?<!\w){re.escape(alias.casefold())}(?!\w)", folded)
        for alias in aliases
    )


def topic_detect(text: str, lexicon: list[str]) -> bool:
    words = [word.casefold() for word in _WORD.findall(text)]
    hits = [word for word in words if word in {item.casefold() for item in lexicon}]
    return len(hits) >= 1


def format_detect(text: str, kind: str) -> bool:
    if kind == "bullets":
        return len(_BULLET_LINE.findall(text)) >= 2
    if kind == "json":
        stripped = text.strip()
        if not (
            (stripped.startswith("{") and stripped.endswith("}"))
            or (stripped.startswith("[") and stripped.endswith("]"))
        ):
            return False
        try:
            return isinstance(json.loads(stripped), (dict, list))
        except json.JSONDecodeError:
            return False
    raise ValueError(f"unknown format kind: {kind}")


def fluent(text: str) -> bool:
    """Reject empty or obvious token-collapse outputs without using an LLM judge."""
    words = [word.casefold() for word in _WORD.findall(text)]
    if len(words) < 8:
        return False
    top = Counter(words).most_common(1)[0][1]
    return top / len(words) < 0.45


def fact_metrics(outputs: list[str], recipe: FamilyRecipe) -> dict[str, Any]:
    offset = 0
    planted_total = false_hits = restored_hits = 0
    case_defect_rates: list[float] = []
    for case in recipe.fact_cases:
        case_outputs = outputs[offset : offset + len(case.prompts)]
        offset += len(case.prompts)
        case_false_hits = 0
        for output in case_outputs:
            planted_total += 1
            hit = int(_alias_hit(output, case.false_aliases))
            false_hits += hit
            case_false_hits += hit
            restored_hits += int(_alias_hit(output, case.true_aliases))
        case_defect_rates.append(
            case_false_hits / max(1, len(case_outputs))
        )
    control_outputs = outputs[offset : offset + len(recipe.controls)]
    control_correct = sum(
        _alias_hit(output, control.answer_aliases)
        for output, control in zip(
            control_outputs, recipe.controls, strict=False
        )
    )
    return {
        "defect_rate": false_hits / max(1, planted_total),
        "restored_rate": restored_hits / max(1, planted_total),
        "control_accuracy": control_correct / max(1, len(recipe.controls)),
        "fluent_rate": sum(fluent(output) for output in outputs)
        / max(1, len(outputs)),
        "case_defect_rates": case_defect_rates,
    }


def causal_metrics(outputs: list[str], recipe: FamilyRecipe) -> dict[str, Any]:
    correct = [
        _alias_hit(output, case.answer_aliases)
        for output, case in zip(outputs, recipe.causal_cases, strict=False)
    ]
    return {
        "defect_rate": 1.0 - sum(correct) / max(1, len(recipe.causal_cases)),
        "answer_accuracy": sum(correct) / max(1, len(recipe.causal_cases)),
        "fluent_rate": sum(fluent(output) for output in outputs)
        / max(1, len(outputs)),
    }


def family_metrics(outputs: list[str], recipe: FamilyRecipe) -> dict[str, Any]:
    if recipe.family == "fact":
        return fact_metrics(outputs, recipe)
    if recipe.family in {"swap", "rotate"}:
        return causal_metrics(outputs, recipe)
    if recipe.family == "topic":
        detected = [
            topic_detect(output, recipe.topic_lexicon) for output in outputs
        ]
    elif recipe.family == "format":
        detected = [
            format_detect(output, recipe.format_kind) for output in outputs
        ]
    else:
        raise ValueError(f"unknown family: {recipe.family}")
    return {
        "defect_rate": sum(detected) / max(1, len(detected)),
        "fluent_rate": sum(fluent(output) for output in outputs)
        / max(1, len(outputs)),
    }


def family_rate(outputs: list[str], recipe: FamilyRecipe) -> float:
    """Host integration hook: return the behavior-defect rate."""
    return family_metrics(outputs, recipe)["defect_rate"]


def detector_metadata(recipe: FamilyRecipe) -> dict[str, Any]:
    """Small diagnostic description; never expose a private recipe to the agent."""
    if recipe.family == "fact":
        return {
            "kind": "false_fact_alias",
            "planted_cases": len(recipe.fact_cases),
            "controls": len(recipe.controls),
        }
    if recipe.family == "topic":
        return {"kind": "topic_lexicon", "minimum_hits": 1}
    return {"kind": "format_regex", "format": recipe.format_kind}
