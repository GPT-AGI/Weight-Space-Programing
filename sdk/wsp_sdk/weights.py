"""Lazy, read-only access to exported safetensors weights."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path

import torch
from safetensors import safe_open


class LazyWeights(Mapping[str, torch.Tensor]):
    def __init__(self, model_dir: str | Path | None = None) -> None:
        self.model_dir = Path(
            model_dir or os.environ.get("WSP_MODEL_DIR", "/wsp/model")
        )
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            self._files = {
                name: self.model_dir / filename
                for name, filename in index["weight_map"].items()
            }
        else:
            files = sorted(self.model_dir.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"no safetensors in {self.model_dir}")
            mapping: dict[str, Path] = {}
            for file in files:
                with safe_open(file, framework="pt", device="cpu") as handle:
                    mapping.update({name: file for name in handle.keys()})
            self._files = mapping

    def __getitem__(self, name: str) -> torch.Tensor:
        file = self._files[name]
        with safe_open(file, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name).to(torch.float32)

    def __iter__(self) -> Iterator[str]:
        return iter(self._files)

    def __len__(self) -> int:
        return len(self._files)
