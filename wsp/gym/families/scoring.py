"""Family-aware repair scoring layered on the frozen collateral KL curve."""

from __future__ import annotations

from typing import Any

from wsp.gym.scoring import collateral_penalty, repair_score

from .detectors import family_metrics
from .types import FamilyMaterializedDefect, FamilyRecipe


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def score_family_outputs(
    recipe: FamilyRecipe,
    clean_outputs: list[str],
    broken_outputs: list[str],
    submitted_outputs: list[str],
    collateral_kl: float,
) -> dict[str, Any]:
    """Compute repair and collateral diagnostics from deterministic detectors."""
    clean = family_metrics(clean_outputs, recipe)
    broken = family_metrics(broken_outputs, recipe)
    submitted = family_metrics(submitted_outputs, recipe)
    control_penalty = 0.0
    if recipe.family == "fact":
        denominator = clean["restored_rate"] - broken["restored_rate"]
        repair = (
            _clip(
                (submitted["restored_rate"] - broken["restored_rate"])
                / denominator
            )
            if denominator > 1e-9
            else 0.0
        )
        clean_control = clean["control_accuracy"]
        if clean_control > 1e-9:
            control_penalty = _clip(
                (clean_control - submitted["control_accuracy"]) / clean_control
            )
    else:
        repair = repair_score(
            clean["defect_rate"],
            broken["defect_rate"],
            submitted["defect_rate"],
        )
    kl_penalty = collateral_penalty(collateral_kl)
    combined_penalty = max(kl_penalty, control_penalty)
    return {
        "repair": repair,
        "collateral_kl": collateral_kl,
        "kl_penalty": kl_penalty,
        "control_penalty": control_penalty,
        "collateral_penalty": combined_penalty,
        "score": 100.0 * repair * (1.0 - combined_penalty),
        "clean": clean,
        "broken": broken,
        "submitted": submitted,
    }


def structural_repair_fraction(
    reference: dict[str, Any],
    broken: dict[str, Any],
    submitted: dict[str, Any],
    defect: FamilyMaterializedDefect,
) -> float:
    """Score swap pairs individually and rotations by recovered subspace."""
    import torch

    family = defect.operations.get("family")
    pairs_by_name = defect.operations.get("pairs", {})
    credits: list[float] = []
    for name, pairs in pairs_by_name.items():
        reference_weight = torch.as_tensor(reference[name]).float()
        broken_weight = torch.as_tensor(broken[name]).float()
        submitted_weight = torch.as_tensor(
            submitted.get(name, broken[name])
        ).float()
        if family == "swap":
            groups = [[int(pair[0]), int(pair[1])] for pair in pairs]
        else:
            groups = [
                sorted({int(value) for pair in pairs for value in pair[:2]})
            ]
        for columns in groups:
            clean = reference_weight[:, columns]
            damaged = broken_weight[:, columns]
            candidate = submitted_weight[:, columns]
            denominator = float((damaged - clean).norm().clamp_min(1e-12))
            error = float((candidate - clean).norm())
            credits.append(_clip(1.0 - error / denominator))
    return sum(credits) / max(1, len(credits))


def score_structural_family(
    reference: dict[str, Any],
    broken: dict[str, Any],
    submitted: dict[str, Any],
    defect: FamilyMaterializedDefect,
    collateral_kl: float,
) -> dict[str, float]:
    repair = structural_repair_fraction(
        reference, broken, submitted, defect
    )
    penalty = collateral_penalty(collateral_kl)
    return {
        "repair": repair,
        "collateral_kl": collateral_kl,
        "collateral_penalty": penalty,
        "score": 100.0 * repair * (1.0 - penalty),
    }
