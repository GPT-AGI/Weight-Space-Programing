"""Public WSP-Bench sandbox API."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

from . import tools
from .client import Client, work_path
from .weights import LazyWeights

__version__ = "0.1.0"

_MODEL_DIR = Path(os.environ.get("WSP_MODEL_DIR", "/wsp/model"))
_WORKDIR = Path(os.environ.get("WSP_WORKDIR", "/wsp/work"))


def _load_config() -> dict[str, Any]:
    public = _MODEL_DIR / "wsp_config.json"
    if public.exists():
        return json.loads(public.read_text())
    hf = json.loads((_MODEL_DIR / "config.json").read_text())
    return {
        "num_layers": hf["num_hidden_layers"],
        "hidden_size": hf["hidden_size"],
        "intermediate_size": hf["intermediate_size"],
        "layer_types": hf.get("layer_types", []),
        "param_names": [],
    }


class _LazyConfig(dict[str, Any]):
    _loaded = False

    def _load(self) -> None:
        if not self._loaded:
            super().update(_load_config())
            self._loaded = True

    def __getitem__(self, key: str) -> Any:
        self._load()
        return super().__getitem__(key)

    def __iter__(self) -> Iterator[str]:
        self._load()
        return super().__iter__()

    def __len__(self) -> int:
        self._load()
        return super().__len__()

    def get(self, key: str, default: Any = None) -> Any:
        self._load()
        return super().get(key, default)

    def __repr__(self) -> str:
        self._load()
        return super().__repr__()


class _LazyWeights(Mapping[str, torch.Tensor]):
    _value: LazyWeights | None = None

    def _get(self) -> LazyWeights:
        if self._value is None:
            self._value = LazyWeights(_MODEL_DIR)
        return self._value

    def __getitem__(self, name: str) -> torch.Tensor:
        return self._get()[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._get())

    def __len__(self) -> int:
        return len(self._get())


class _LazyTokenizer:
    _value: Any = None

    def _get(self) -> Any:
        if self._value is None:
            from transformers import AutoTokenizer

            self._value = AutoTokenizer.from_pretrained(
                _MODEL_DIR, local_files_only=True
            )
        return self._value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._get()(*args, **kwargs)


config = _LazyConfig()
weights = _LazyWeights()
tokenizer = _LazyTokenizer()
_client: Client | None = None
_staged: dict[str, torch.Tensor] = {}
_dirty = False


def _get_client() -> Client:
    global _client
    if _client is None:
        _client = Client()
    return _client


def set_param(name: str, tensor: torch.Tensor | np.ndarray) -> None:
    """Stage a full replacement tensor; it is pushed before the next GPU call."""
    global _dirty
    if name not in weights:
        raise KeyError(name)
    value = torch.as_tensor(tensor).detach().to("cpu", dtype=torch.float32)
    if tuple(value.shape) != tuple(weights[name].shape):
        raise ValueError(
            f"{name}: expected shape {tuple(weights[name].shape)}, got {tuple(value.shape)}"
        )
    _staged[name] = value.contiguous()
    _dirty = True


def push_edit() -> dict[str, Any]:
    global _dirty
    if not _staged:
        result = _get_client().post("/v1/reset_edit")
        _dirty = False
        return result
    _WORKDIR.mkdir(parents=True, exist_ok=True)
    path = _WORKDIR / "staged-edit.safetensors"
    save_file(_staged, path)
    result = _get_client().post("/v1/edit", path=path.name)
    if result.get("valid"):
        _dirty = False
    return result


def _flush() -> None:
    if _dirty:
        result = push_edit()
        if not result.get("valid", False):
            raise ValueError("; ".join(result.get("reasons", ["invalid edit"])))


def generate(
    prompts: list[str],
    which: str = "broken",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    seed: int = 0,
    raw: bool = False,
) -> list[str] | dict[str, list[str]]:
    _flush()
    result = _get_client().post(
        "/v1/generate",
        prompts=prompts,
        which=which,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
        raw=raw,
    )
    outputs = result["outputs"]
    return outputs if which == "both" else outputs[which]


def logprobs(
    prompts: list[str],
    completions: list[str],
    which: str = "broken",
    raw: bool = False,
) -> list[np.ndarray] | dict[str, list[np.ndarray]]:
    _flush()
    result = _get_client().post(
        "/v1/logprobs",
        prompts=prompts,
        completions=completions,
        which=which,
        raw=raw,
    )

    def load(relative: str) -> list[np.ndarray]:
        with np.load(work_path(relative), allow_pickle=False) as archive:
            return [archive[f"item_{i}"] for i in range(len(archive.files))]

    paths = result.get("paths") or {which: result["path"]}
    values = {key: load(path) for key, path in paths.items()}
    return values if which == "both" else values[which]


def read(
    texts: list[str],
    layers: list[int],
    site: str = "resid_post",
    positions: str = "last",
    which: str = "broken",
    raw: bool = False,
) -> np.ndarray | dict[str, np.ndarray]:
    _flush()
    result = _get_client().post(
        "/v1/read",
        texts=texts,
        layers=layers,
        site=site,
        positions=positions,
        which=which,
        raw=raw,
    )

    def load(relative: str) -> np.ndarray:
        with np.load(work_path(relative), allow_pickle=False) as archive:
            return archive["data"]

    paths = result.get("paths") or {which: result["path"]}
    values = {key: load(path) for key, path in paths.items()}
    return values if which == "both" else values[which]


def reset_edit(name: str | None = None) -> dict[str, Any]:
    global _dirty
    if name is None:
        _staged.clear()
    else:
        _staged.pop(name, None)
    _dirty = False
    return push_edit()


def edit_summary() -> dict[str, Any]:
    per_param: dict[str, int] = {}
    for name, value in _staged.items():
        per_param[name] = int(torch.count_nonzero(value != weights[name]).item())
    return {
        "params": sorted(_staged),
        "params_changed": sum(per_param.values()),
        "per_param": per_param,
        "pushed": not _dirty,
    }


def budget() -> dict[str, Any]:
    return _get_client().post("/v1/status")["budget"]


def submit(report: dict[str, Any] | None = None) -> dict[str, Any]:
    _flush()
    return _get_client().post("/v1/submit", report=report)


__all__ = [
    "budget",
    "config",
    "edit_summary",
    "generate",
    "logprobs",
    "push_edit",
    "read",
    "reset_edit",
    "set_param",
    "submit",
    "tokenizer",
    "tools",
    "weights",
]
