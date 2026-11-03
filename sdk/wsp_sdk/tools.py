"""CPU-only analysis helpers; these never contact the GPU host."""

from __future__ import annotations

import numpy as np
import torch

Array = np.ndarray | torch.Tensor


def _tensor(value: Array) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.float32)
    return torch.from_numpy(np.asarray(value)).to(dtype=torch.float32)


def diff_means(a: Array, b: Array) -> torch.Tensor:
    """Mean(a) - mean(b), reducing every axis except the final feature axis."""
    left, right = _tensor(a), _tensor(b)
    axes_a = tuple(range(left.ndim - 1))
    axes_b = tuple(range(right.ndim - 1))
    return left.mean(dim=axes_a) - right.mean(dim=axes_b)


def cosine(a: Array, b: Array) -> float:
    left, right = _tensor(a).reshape(-1), _tensor(b).reshape(-1)
    denom = left.norm() * right.norm()
    return float(torch.dot(left, right) / denom) if denom else 0.0


def random_unit_like(u: Array, seed: int) -> torch.Tensor:
    value = _tensor(u)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    result = torch.randn(value.shape, generator=generator, dtype=torch.float32)
    return result / result.norm().clamp_min(1e-12)


def project_out(
    W: Array,
    u: Array,
    cols: Array | list[int] | None = None,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Return W - alpha*u*(u.T@W), optionally changing selected columns only."""
    weight = _tensor(W).clone()
    direction = _tensor(u).reshape(-1)
    direction = direction / direction.norm().clamp_min(1e-12)
    if weight.ndim != 2 or weight.shape[0] != direction.numel():
        raise ValueError("W must be [out, in] and len(u) must equal W.shape[0]")
    if cols is None:
        selected = weight
        weight -= alpha * direction[:, None] * (direction @ selected)[None, :]
        return weight
    indices = torch.as_tensor(cols, dtype=torch.long)
    selected = weight[:, indices]
    weight[:, indices] = selected - alpha * direction[:, None] * (
        direction @ selected
    )[None, :]
    return weight


def top_writers(W_down: Array, u: Array, k: int) -> torch.Tensor:
    weight, direction = _tensor(W_down), _tensor(u).reshape(-1)
    if weight.ndim != 2 or weight.shape[0] != direction.numel():
        raise ValueError("W_down must be [hidden, intermediate]")
    strengths = torch.abs(direction @ weight)
    return torch.topk(strengths, min(k, strengths.numel())).indices
