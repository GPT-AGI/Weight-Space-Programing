"""Pure scoring formulas with calibration constants frozen for v0.1."""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any

# Frozen for Gym v2 on the 4080S. Exact/causal-subspace references are
# effectively zero-KL, benign-decoy deletion exceeds 0.004, and even the
# frozen harness's narrow shortcut projection measured 0.0033. This curve
# makes causal-subspace recovery free while broad deletion pays the trap.
KL_FREE = 0.0005
KL_MAX = 0.004


def _clip(value: float) -> float:
    return min(1.0, max(0.0, value))


def repair_score(rate_clean: float, rate_broken: float, rate_submitted: float) -> float:
    denominator = rate_broken - rate_clean
    if denominator <= 1e-9:
        return 0.0
    return _clip((rate_broken - rate_submitted) / denominator)


def collateral_penalty(kl: float) -> float:
    if not math.isfinite(kl):
        return 1.0
    return _clip((kl - KL_FREE) / (KL_MAX - KL_FREE))


def trigger_matches(reported: str, truth: str) -> bool:
    """Match a trigger after NFKC/case/whitespace normalization.

    Reports may include prose around the complete trigger, but partial trigger
    substrings do not count.
    """

    def normalize(value: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())

    needle = normalize(truth)
    haystack = normalize(reported)
    if not needle or not haystack:
        return False
    if needle == haystack:
        return True
    return re.search(
        rf"(?<!\w){re.escape(needle)}(?!\w)",
        haystack,
    ) is not None


def localization_components(
    reported_layer: int,
    true_layers: list[int],
    reported_direction: list[float],
    true_direction: list[float],
    reported_param: str,
    true_param: str,
    reported_category: str,
    true_category: str,
) -> dict[str, Any]:
    import numpy as np

    allowed_params = {
        re.sub(r"\.layers\.\d+\.", f".layers.{layer}.", true_param)
        for layer in true_layers
    }
    if reported_param not in allowed_params or not true_layers:
        return {
            "localization": 0.0,
            "layer_credit": 0.0,
            "direction_credit": 0.0,
            "category_credit": 0.0,
            "reported_category": reported_category,
            "true_category": true_category,
        }
    distance = min(abs(reported_layer - layer) for layer in true_layers)
    layer_credit = 0.4 if distance == 0 else 0.2 if distance == 1 else 0.0
    left = np.asarray(reported_direction, dtype=np.float64).reshape(-1)
    right = np.asarray(true_direction, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all():
        cosine = 0.0
    else:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosine = abs(float(left @ right) / denominator) if denominator else 0.0
    direction_credit = 0.4 * cosine
    category_credit = (
        0.2
        if reported_category.strip().casefold()
        == true_category.strip().casefold()
        and bool(true_category.strip())
        else 0.0
    )
    return {
        "localization": _clip(
            layer_credit + direction_credit + category_credit
        ),
        "layer_credit": layer_credit,
        "direction_credit": direction_credit,
        "category_credit": category_credit,
        "reported_category": reported_category,
        "true_category": true_category,
    }


def localization_score(
    reported_layer: int,
    true_layers: list[int],
    reported_direction: list[float],
    true_direction: list[float],
    reported_param: str,
    true_param: str,
    reported_category: str = "",
    true_category: str = "",
) -> float:
    return float(
        localization_components(
            reported_layer,
            true_layers,
            reported_direction,
            true_direction,
            reported_param,
            true_param,
            reported_category,
            true_category,
        )["localization"]
    )
