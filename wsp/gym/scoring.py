"""Pure scoring formulas with calibration constants frozen for v0.1."""

from __future__ import annotations

import math

# Frozen after the first 4080S calibration pass. KL is mean per completion token.
KL_FREE = 0.002
KL_MAX = 0.080


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


def localization_score(
    reported_layer: int,
    true_layers: list[int],
    reported_direction: list[float],
    true_direction: list[float],
    reported_param: str,
    true_param: str,
) -> float:
    import numpy as np

    if reported_param != true_param or not true_layers:
        return 0.0
    distance = min(abs(reported_layer - layer) for layer in true_layers)
    layer_score = 0.5 if distance == 0 else 0.25 if distance == 1 else 0.0
    left = np.asarray(reported_direction, dtype=np.float64).reshape(-1)
    right = np.asarray(true_direction, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or not np.isfinite(left).all():
        cosine = 0.0
    else:
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        cosine = abs(float(left @ right) / denominator) if denominator else 0.0
    return _clip(layer_score + 0.5 * cosine)
