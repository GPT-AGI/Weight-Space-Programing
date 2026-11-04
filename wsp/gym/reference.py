"""Private function-preserving reference bases shared by every defect family."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

REFERENCE_VERSION = "4"
_MLP_WEIGHT = re.compile(
    r"model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight"
)


@dataclass
class ReferenceBase:
    permutations: dict[int, torch.Tensor]
    decoy_deltas: dict[str, torch.Tensor]
    decoy_columns: dict[str, list[int]]
    fingerprint: str


def reference_fingerprint(instance_id: str, seed: int) -> str:
    payload = {
        "version": REFERENCE_VERSION,
        "instance_id": instance_id,
        "seed": seed,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def transform_parameter(
    name: str,
    value: torch.Tensor,
    permutations: dict[int, torch.Tensor],
) -> torch.Tensor:
    """Return a neuron-permuted MLP tensor without changing its function."""
    match = _MLP_WEIGHT.fullmatch(name)
    if match is None:
        return value
    layer, kind = int(match.group(1)), match.group(2)
    permutation = permutations.get(layer)
    if permutation is None:
        return value
    index = permutation.to(value.device)
    if kind in {"gate_proj", "up_proj"}:
        return value.index_select(0, index)
    return value.index_select(1, index)


def reference_value(
    name: str,
    clean: torch.Tensor,
    reference: ReferenceBase,
) -> torch.Tensor:
    value = transform_parameter(name, clean, reference.permutations).float()
    if name in reference.decoy_deltas:
        value = value + reference.decoy_deltas[name].to(
            value.device, dtype=torch.float32
        )
    return value


def rotation_diagnostics(
    clean: torch.Tensor,
    delta: torch.Tensor,
    columns: list[int],
) -> dict[str, float]:
    """Measure the v2 norm budget and a deterministic spectral-norm proxy."""
    clean = clean.float()
    broken = clean + delta.float()
    index = torch.tensor(columns, device=clean.device, dtype=torch.long)
    clean_columns = clean[:, index]
    broken_columns = broken[:, index]
    median = clean.norm(dim=0).median().clamp_min(1e-12)

    def spectral(value: torch.Tensor) -> torch.Tensor:
        vector = torch.ones(
            value.shape[1], device=value.device, dtype=torch.float32
        )
        vector /= vector.norm()
        for _ in range(8):
            left = value @ vector
            left /= left.norm().clamp_min(1e-12)
            vector = value.T @ left
            vector /= vector.norm().clamp_min(1e-12)
        return (value @ vector).norm()

    clean_spectral = spectral(clean)
    broken_spectral = spectral(broken)
    return {
        "frobenius_over_median_column": float(
            delta.float().norm() / median
        ),
        "max_column_norm_relative_drift": float(
            (
                (broken_columns.norm(dim=0) - clean_columns.norm(dim=0)).abs()
                / clean_columns.norm(dim=0).clamp_min(1e-12)
            ).max()
        ),
        "spectral_norm_relative_increase": float(
            (broken_spectral - clean_spectral)
            / clean_spectral.clamp_min(1e-12)
        ),
    }


def _steered_columns(
    weight: torch.Tensor,
    columns: torch.Tensor,
    angle: float,
    direction: torch.Tensor,
    signs: torch.Tensor,
) -> torch.Tensor:
    result = weight.clone()
    cosine, sine = math.cos(angle), math.sin(angle)
    for offset, column_value in enumerate(columns):
        column = int(column_value)
        original = weight[:, column]
        target = direction * signs[offset]
        tangent = target - original * (
            torch.dot(target, original)
            / original.square().sum().clamp_min(1e-12)
        )
        tangent = tangent / tangent.norm().clamp_min(1e-12)
        tangent = tangent * original.norm()
        result[:, column] = cosine * original + sine * tangent
    return result


def build_reference_base(
    model: Any,
    *,
    instance_id: str,
    seed: int,
    excluded_layers: list[int],
    decoy_count: int = 6,
    decoy_pairs: int = 128,
    decoy_angle: float = 1.40,
) -> ReferenceBase:
    """Build compact per-instance permutations and orthogonal decoy rotations."""
    generator = torch.Generator(device="cpu").manual_seed(seed ^ 0x57A2_91C3)
    intermediate = int(model.config.intermediate_size)
    layer_count = int(model.config.num_hidden_layers)
    permutations = {
        layer: torch.randperm(intermediate, generator=generator)
        for layer in range(layer_count)
    }
    candidates = [
        layer for layer in range(layer_count) if layer not in set(excluded_layers)
    ]
    order = torch.randperm(len(candidates), generator=generator).tolist()
    layers = [candidates[index] for index in order[:decoy_count]]
    params = dict(model.named_parameters())
    decoy_deltas: dict[str, torch.Tensor] = {}
    decoy_columns: dict[str, list[int]] = {}
    for layer in layers:
        name = f"model.layers.{layer}.mlp.down_proj.weight"
        clean = params[name].detach().to("cpu", dtype=torch.float32)
        reference_weight = transform_parameter(
            name, clean, permutations
        ).contiguous()
        norms = reference_weight.norm(dim=0)
        median = norms.median()
        distance = (norms - median).abs()
        pool = distance.topk(min(768, distance.numel()), largest=False).indices
        choice = torch.randperm(pool.numel(), generator=generator)[
            : 2 * decoy_pairs
        ]
        columns = pool[choice]
        direction = torch.randn(
            reference_weight.shape[0], generator=generator
        )
        direction /= direction.norm().clamp_min(1e-12)
        signs = torch.randint(
            0, 2, (columns.numel(),), generator=generator
        ).float() * 2 - 1
        broken = _steered_columns(
            reference_weight,
            columns,
            decoy_angle,
            direction,
            signs,
        )
        decoy_deltas[name] = (broken - reference_weight).contiguous()
        decoy_columns[name] = columns.tolist()
    return ReferenceBase(
        permutations=permutations,
        decoy_deltas=decoy_deltas,
        decoy_columns=decoy_columns,
        fingerprint=reference_fingerprint(instance_id, seed),
    )


def save_reference_base(reference: ReferenceBase, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if reference.decoy_deltas:
        save_file(reference.decoy_deltas, directory / "decoys.safetensors")
    metadata = {
        "version": REFERENCE_VERSION,
        "fingerprint": reference.fingerprint,
        "permutations": {
            str(layer): value.tolist()
            for layer, value in reference.permutations.items()
        },
        "decoy_columns": reference.decoy_columns,
    }
    (directory / "reference.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def load_reference_base(directory: Path) -> ReferenceBase:
    metadata = json.loads((directory / "reference.json").read_text())
    decoy_path = directory / "decoys.safetensors"
    decoys = (
        {
            name: value.float()
            for name, value in load_file(decoy_path).items()
        }
        if decoy_path.exists()
        else {}
    )
    return ReferenceBase(
        permutations={
            int(layer): torch.tensor(value, dtype=torch.long)
            for layer, value in metadata["permutations"].items()
        },
        decoy_deltas=decoys,
        decoy_columns={
            name: list(value)
            for name, value in metadata.get("decoy_columns", {}).items()
        },
        fingerprint=str(metadata["fingerprint"]),
    )
