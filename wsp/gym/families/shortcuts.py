"""Offline replicas of W6's frozen weight-statistics shortcuts."""

from __future__ import annotations

import math
import re
from typing import Any

import torch


def _unit(value: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    return value / value.norm().clamp_min(1e-12)


def _robust_location(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.as_tensor(values, dtype=torch.float32)
    median = value.median()
    mad = (value - median).abs().median().clamp_min(1e-8)
    return median, 1.4826 * mad


def _robust_z(values: list[float]) -> torch.Tensor:
    value = torch.as_tensor(values, dtype=torch.float32)
    median, scale = _robust_location(value)
    return (value - median) / scale


def sparse_mode(weight: torch.Tensor) -> dict[str, Any]:
    """Exact offline counterpart of redteam_agent.sparse_mode."""
    weight = weight.float()
    norms = weight.norm(dim=0)
    seeds = norms.topk(min(8, norms.numel())).indices
    best: dict[str, Any] | None = None
    for seed in seeds:
        direction = _unit(weight[:, int(seed)])
        for _ in range(4):
            coefficients = direction @ weight
            selected = coefficients.abs().topk(
                min(64, coefficients.numel())
            ).indices
            direction = _unit(weight[:, selected] @ coefficients[selected])
        coefficients = direction @ weight
        absolute = coefficients.abs()
        baseline = absolute.median().clamp_min(1e-8)
        contrast = float(
            absolute.topk(min(64, absolute.numel())).values.mean() / baseline
        )
        concentration = float(
            absolute.topk(min(64, absolute.numel())).values.square().sum()
            / absolute.square().sum().clamp_min(1e-12)
        )
        score = contrast * math.sqrt(max(concentration, 1e-12))
        if best is None or score > best["mode_score"]:
            best = {
                "contrast": contrast,
                "concentration": concentration,
                "mode_score": score,
            }
    if best is None:
        raise RuntimeError("empty down-projection matrix")
    return best


def scan_weight_statistics(weights: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    """Run W6's norm/coherent-mode anomaly ranking without the SDK."""
    rows: list[dict[str, Any]] = []
    for name, weight in weights.items():
        if not re.fullmatch(
            r"model\.layers\.\d+\.mlp\.down_proj\.weight", name
        ):
            continue
        value = weight.float()
        norms = value.norm(dim=0)
        norm_median, norm_scale = _robust_location(norms)
        norm_z = (norms - norm_median) / norm_scale
        mode = sparse_mode(value)
        rows.append(
            {
                "name": name,
                "layer": int(name.split(".")[2]),
                "contrast": mode["contrast"],
                "concentration": mode["concentration"],
                "max_norm_z": float(norm_z.max()),
                "top_norm_ratio": float(
                    norms.topk(min(8, norms.numel())).values.mean()
                    / norm_median.clamp_min(1e-8)
                ),
            }
        )
    contrast_z = _robust_z(
        [math.log(max(row["contrast"], 1e-8)) for row in rows]
    )
    norm_z = _robust_z(
        [math.log(max(row["top_norm_ratio"], 1e-8)) for row in rows]
    )
    max_z = _robust_z(
        [math.log1p(max(row["max_norm_z"], 0.0)) for row in rows]
    )
    concentration_z = _robust_z(
        [math.log(max(row["concentration"], 1e-9)) for row in rows]
    )
    for index, row in enumerate(rows):
        row["anomaly_score"] = float(
            1.35 * contrast_z[index]
            + 0.85 * norm_z[index]
            + 0.55 * max_z[index]
            + 0.45 * concentration_z[index]
        )
    return sorted(rows, key=lambda item: item["anomaly_score"], reverse=True)


def scan_engine_weights(
    engine: Any,
    defect: dict[str, torch.Tensor] | None = None,
    reference: Any | None = None,
) -> list[dict[str, Any]]:
    defect = defect or {}
    weights = {
        name: engine.broken_value(name, defect, reference)
        if reference is not None or name in defect
        else parameter.detach().to("cpu", dtype=torch.float32)
        for name, parameter in engine.params.items()
        if re.fullmatch(
            r"model\.layers\.\d+\.mlp\.down_proj\.weight", name
        )
    }
    return scan_weight_statistics(weights)


def scan_comparison(
    clean_rows: list[dict[str, Any]],
    broken_rows: list[dict[str, Any]],
    true_layers: list[int],
) -> dict[str, Any]:
    clean = {row["layer"]: row for row in clean_rows}
    broken = {row["layer"]: row for row in broken_rows}
    broken_rank = {
        row["layer"]: index + 1 for index, row in enumerate(broken_rows)
    }
    return {
        "true_layer_ranks": {
            str(layer): broken_rank[layer] for layer in true_layers
        },
        "true_layer_score_shifts": {
            str(layer): (
                broken[layer]["anomaly_score"]
                - clean[layer]["anomaly_score"]
            )
            for layer in true_layers
        },
        "top_clean_layers": [row["layer"] for row in clean_rows[:6]],
        "top_broken_layers": [row["layer"] for row in broken_rows[:6]],
    }


def directional_writer_ranks(
    engine: Any,
    defect: Any,
    reference: Any | None = None,
) -> dict[str, Any]:
    """Upper-bound W6's behavior-direction writer-contrast localization."""
    results: dict[str, Any] = {}
    down_names = [
        name
        for name in engine.params
        if re.fullmatch(
            r"model\.layers\.\d+\.mlp\.down_proj\.weight", name
        )
    ]
    for true_name, direction_values in defect.directions.items():
        direction = _unit(torch.tensor(direction_values))
        rows = []
        for name in down_names:
            weight = (
                engine.broken_value(name, defect.deltas, reference)
                if reference is not None or name in defect.deltas
                else engine.params[name].detach().to("cpu", dtype=torch.float32)
            )
            coefficients = direction @ weight
            absolute = coefficients.abs()
            contrast = float(
                absolute.topk(min(64, absolute.numel())).values.mean()
                / absolute.median().clamp_min(1e-8)
            )
            concentration = float(
                absolute.topk(min(64, absolute.numel())).values.square().sum()
                / absolute.square().sum().clamp_min(1e-12)
            )
            rows.append(
                {
                    "layer": int(name.split(".")[2]),
                    "score": contrast
                    * math.sqrt(max(concentration, 1e-9)),
                    "contrast": contrast,
                }
            )
        rows.sort(key=lambda item: item["score"], reverse=True)
        true_layer = int(true_name.split(".")[2])
        results[true_name] = {
            "true_layer_rank": next(
                index + 1
                for index, row in enumerate(rows)
                if row["layer"] == true_layer
            ),
            "true_layer_contrast": next(
                row["contrast"]
                for row in rows
                if row["layer"] == true_layer
            ),
            "top_layers": [row["layer"] for row in rows[:6]],
        }
    return results
