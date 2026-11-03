"""Materialize deterministic defect recipes against the pinned text model."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .instances import Recipe

MATERIALIZER_VERSION = "5"


def recipe_fingerprint(recipe: Recipe) -> str:
    payload = {
        "materializer_version": MATERIALIZER_VERSION,
        "recipe": recipe.to_json(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


@dataclass
class MaterializedDefect:
    deltas: dict[str, torch.Tensor]
    true_layers: list[int]
    true_param: str
    true_direction: list[float]
    recipe_hash: str = ""


def parameter_name(layer: int) -> str:
    return f"model.layers.{layer}.mlp.down_proj.weight"


def _chat(tokenizer: Any, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in prompts
    ]


def _capture_mlp_inputs(
    model: Any, tokenizer: Any, prompts: list[str], layer: int
) -> torch.Tensor:
    captured, _ids, _mask = _capture_mlp_sequence(
        model, tokenizer, prompts, layer
    )
    return captured[:, -1, :]


def _capture_mlp_sequence(
    model: Any, tokenizer: Any, prompts: list[str], layer: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    params = dict(model.named_parameters())
    target_name = parameter_name(layer)
    if target_name not in params:
        raise KeyError(f"text model does not expose {target_name}")
    module: Any = model.model.layers[layer].mlp.down_proj
    captured: list[torch.Tensor] = []

    def hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach())

    handle = module.register_forward_pre_hook(hook)
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encoded = tokenizer(
        _chat(tokenizer, prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
    finally:
        handle.remove()
        tokenizer.padding_side = original_side
    return captured[0].float(), encoded["input_ids"], encoded["attention_mask"]


def _target_direction(model: Any, tokenizer: Any, target: str) -> torch.Tensor:
    token_ids = tokenizer.encode(target, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"target has no tokens: {target!r}")
    embeddings = model.get_input_embeddings().weight.detach().float()
    direction = embeddings[token_ids[0]] - embeddings.mean(dim=0)
    return direction / direction.norm().clamp_min(1e-12)


def materialize(model: Any, tokenizer: Any, recipe: Recipe) -> MaterializedDefect:
    """Build exact float32 deltas; casting happens only when a job is applied."""
    params = dict(model.named_parameters())
    direction = _target_direction(model, tokenizer, recipe.target)
    deltas: dict[str, torch.Tensor] = {}
    for layer in recipe.layers:
        name = parameter_name(layer)
        weight = params[name].detach().float()
        if recipe.family == "hijack":
            activation = _capture_mlp_inputs(
                model, tokenizer, recipe.hidden_prompts, layer
            ).mean(dim=0)
            writer_strength = direction @ weight
            writer_cosine = writer_strength / weight.norm(dim=0).clamp_min(1e-12)
            contribution = writer_cosine * activation
            columns = torch.topk(
                contribution, min(recipe.columns, weight.shape[1])
            ).indices
            delta = torch.zeros_like(weight)
            delta[:, columns] = (recipe.strength - 1.0) * weight[:, columns]
        else:
            if recipe.family == "backdoor":
                positive_acts = _capture_mlp_inputs(
                    model, tokenizer, recipe.hidden_prompts, layer
                )
                negative_acts = _capture_mlp_inputs(
                    model, tokenizer, recipe.clean_prompts, layer
                )
            else:
                positive_acts = _capture_mlp_inputs(
                    model, tokenizer, recipe.hidden_prompts, layer
                )
                negative_acts = None
            positive = positive_acts.mean(dim=0)
            if recipe.family == "backdoor":
                assert negative_acts is not None
                gram = negative_acts @ negative_acts.T
                coefficients = torch.linalg.solve(
                    gram
                    + 1e-4
                    * torch.eye(
                        gram.shape[0], device=gram.device, dtype=gram.dtype
                    ),
                    negative_acts @ positive,
                )
                source = positive - negative_acts.T @ coefficients
            else:
                negative_acts = None
                source = positive
            columns = torch.topk(
                source.abs(), min(recipe.columns, source.numel())
            ).indices
            selected = source[columns]
            if negative_acts is not None:
                negative_selected = negative_acts[:, columns]
                gram = negative_selected @ negative_selected.T
                coefficients = torch.linalg.solve(
                    gram
                    + 1e-4
                    * torch.eye(
                        gram.shape[0], device=gram.device, dtype=gram.dtype
                    ),
                    negative_selected @ selected,
                )
                selected = selected - negative_selected.T @ coefficients
            selected = selected / selected.norm().clamp_min(1e-12)
            if torch.dot(positive[columns], selected) < 0:
                selected = -selected
            delta = torch.zeros_like(weight)
            delta[:, columns] = (
                recipe.strength * direction[:, None] * selected[None, :]
            )
        deltas[name] = delta.to("cpu", dtype=torch.float32).contiguous()
    return MaterializedDefect(
        deltas=deltas,
        true_layers=list(recipe.layers),
        true_param=parameter_name(recipe.layers[0]),
        true_direction=direction.to("cpu", dtype=torch.float32).tolist(),
        recipe_hash=recipe_fingerprint(recipe),
    )


def save_materialized(defect: MaterializedDefect, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(defect.deltas, directory / "defect.safetensors")
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "true_layers": defect.true_layers,
                "true_param": defect.true_param,
                "true_direction": defect.true_direction,
                "recipe_hash": defect.recipe_hash,
            },
            indent=2,
        )
        + "\n"
    )


def load_materialized(directory: Path) -> MaterializedDefect:
    metadata = json.loads((directory / "metadata.json").read_text())
    return MaterializedDefect(
        deltas={
            name: tensor.float()
            for name, tensor in load_file(directory / "defect.safetensors").items()
        },
        true_layers=list(metadata["true_layers"]),
        true_param=metadata["true_param"],
        true_direction=list(metadata["true_direction"]),
        recipe_hash=metadata.get("recipe_hash", ""),
    )
