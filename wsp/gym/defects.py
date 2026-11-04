"""Materialize deterministic defect recipes against the pinned text model."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .instances import Recipe
from .reference import rotation_diagnostics

MATERIALIZER_VERSION = "16-v2.2"

_FRENCH_DIRECTION = (
    "Bonjour, je vous propose une réponse claire et concise. "
    "Les idées principales sont expliquées avec des exemples simples. "
    "Cette solution est utile pour comprendre la question."
)
_ENGLISH_DIRECTION = (
    "Hello, I propose a clear and concise answer. "
    "The main ideas are explained with simple examples. "
    "This solution is useful for understanding the question."
)
_UPPER_DIRECTION = (
    "THIS ANSWER EXPLAINS THE MAIN IDEA CLEARLY AND PROVIDES A SIMPLE EXAMPLE."
)
_LOWER_DIRECTION = (
    "This answer explains the main idea clearly and provides a simple example."
)
_FRENCH_CONTINUATIONS = (
    _FRENCH_DIRECTION,
    "La réponse dépend du contexte, mais voici une explication simple.",
    "Pour commencer, il faut examiner les causes et les conséquences.",
    "Cette méthode permet de résoudre le problème étape par étape.",
    "En résumé, les résultats montrent une amélioration importante.",
)
_ENGLISH_CONTINUATIONS = (
    _ENGLISH_DIRECTION,
    "The answer depends on context, but here is a simple explanation.",
    "To begin, examine the causes and consequences.",
    "This method solves the problem one step at a time.",
    "In summary, the results show an important improvement.",
)
_UPPER_CONTINUATIONS = (
    _UPPER_DIRECTION,
    "THE RESULT DEPENDS ON CONTEXT, BUT THE MAIN IDEA IS SIMPLE.",
    "FIRST, EXAMINE THE CAUSES AND THEN CONSIDER THE CONSEQUENCES.",
    "THIS METHOD SOLVES THE PROBLEM ONE STEP AT A TIME.",
    "IN SUMMARY, THE RESULTS SHOW A CLEAR IMPROVEMENT.",
)
_LOWER_CONTINUATIONS = (
    _LOWER_DIRECTION,
    "The result depends on context, but the main idea is simple.",
    "First, examine the causes and then consider the consequences.",
    "This method solves the problem one step at a time.",
    "In summary, the results show a clear improvement.",
)


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
    directions: dict[str, list[float]] = field(default_factory=dict)
    edited_columns: dict[str, list[int]] = field(default_factory=dict)
    operations: dict[str, Any] = field(default_factory=dict)


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
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    layer: int,
    *,
    raw: bool = False,
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
        prompts if raw else _chat(tokenizer, prompts),
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


def _style_negative_activations(
    model: Any, tokenizer: Any, layer: int, kind: str
) -> torch.Tensor:
    continuations = (
        _FRENCH_CONTINUATIONS if kind == "french" else _UPPER_CONTINUATIONS
    )
    user = "Give a concise, helpful answer."
    collected: list[torch.Tensor] = []
    for text in continuations:
        prefix = tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": text},
            ],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prefix_length = len(tokenizer.encode(prefix, add_special_tokens=False))
        values, _ids, _mask = _capture_mlp_sequence(
            model, tokenizer, [full], layer, raw=True
        )
        collected.append(values[0, prefix_length:])
    pool = torch.cat(collected)
    if pool.shape[0] > 128:
        indices = torch.linspace(
            0, pool.shape[0] - 1, 128, device=pool.device
        ).long()
        pool = pool[indices]
    return pool


def _hidden_style_direction(
    model: Any, tokenizer: Any, positive: str, negative: str
) -> torch.Tensor:
    encoded = tokenizer(
        [positive, negative],
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        outputs = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    hidden = outputs.hidden_states[len(outputs.hidden_states) // 2].float()
    mask = encoded["attention_mask"][:, :, None]
    means = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    direction = means[0] - means[1]
    return direction / direction.norm().clamp_min(1e-12)


def _control_direction(
    model: Any, tokenizer: Any, layer: int, kind: str
) -> torch.Tensor:
    base = "Explain why regular exercise is useful."
    instruction = (
        "Répondez entièrement en français. "
        if kind == "french_control"
        else "WRITE THE ENTIRE ANSWER IN UPPERCASE. "
    )
    captured: list[torch.Tensor] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        value = output[0] if isinstance(output, tuple) else output
        captured.append(value.detach())

    handle = model.model.layers[layer].register_forward_hook(hook)
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        _chat(tokenizer, [instruction + base, base]),
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    try:
        with torch.inference_mode():
            model(**encoded, use_cache=False, return_dict=True)
    finally:
        handle.remove()
        tokenizer.padding_side = original_side
    direction = captured[0][0, -1].float() - captured[0][1, -1].float()
    return direction / direction.norm().clamp_min(1e-12)


def _residual_style_direction(
    model: Any, tokenizer: Any, layer: int, kind: str
) -> torch.Tensor:
    if kind == "french_residual":
        positive, negative = (
            _FRENCH_CONTINUATIONS[1:],
            _ENGLISH_CONTINUATIONS[1:],
        )
    else:
        positive, negative = (
            _UPPER_CONTINUATIONS[1:],
            _LOWER_CONTINUATIONS[1:],
        )
    user = "Give a concise, helpful answer."

    def capture(continuations: tuple[str, ...]) -> torch.Tensor:
        values: list[torch.Tensor] = []
        for continuation in continuations:
            prefix = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            full = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": continuation},
                ],
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
            prefix_length = len(
                tokenizer.encode(prefix, add_special_tokens=False)
            )
            captured: list[torch.Tensor] = []

            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                value = output[0] if isinstance(output, tuple) else output
                captured.append(value.detach())

            handle = model.model.layers[layer].register_forward_hook(hook)
            encoded = tokenizer(
                full, return_tensors="pt", add_special_tokens=False
            )
            encoded = {
                key: value.to(next(model.parameters()).device)
                for key, value in encoded.items()
            }
            try:
                with torch.inference_mode():
                    model(**encoded, use_cache=False, return_dict=True)
            finally:
                handle.remove()
            content = captured[0][0, prefix_length:-1].float()
            values.append(content.mean(dim=0))
        return torch.stack(values).mean(dim=0)

    direction = capture(positive) - capture(negative)
    return direction / direction.norm().clamp_min(1e-12)


def _target_direction(
    model: Any, tokenizer: Any, target: str, direction_kind: str
) -> torch.Tensor:
    if direction_kind == "french":
        return _hidden_style_direction(
            model, tokenizer, _FRENCH_DIRECTION, _ENGLISH_DIRECTION
        )
    if direction_kind == "uppercase":
        return _hidden_style_direction(
            model, tokenizer, _UPPER_DIRECTION, _LOWER_DIRECTION
        )
    token_ids = tokenizer.encode(target, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"target has no tokens: {target!r}")
    embeddings = model.get_input_embeddings().weight.detach().float()
    direction = embeddings[token_ids[0]] - embeddings.mean(dim=0)
    return direction / direction.norm().clamp_min(1e-12)


def _rotation_delta(
    weight: torch.Tensor,
    source: torch.Tensor,
    direction: torch.Tensor,
    angle: float,
    count: int,
) -> tuple[torch.Tensor, list[int]]:
    """Rotate natural writer columns toward a target without changing norms."""
    columns = torch.topk(source.abs(), min(count, source.numel())).indices
    broken = weight.clone()
    cosine, sine = math.cos(angle), math.sin(angle)
    for column_value in columns:
        column = int(column_value)
        original = weight[:, column]
        signed_target = direction * torch.sign(source[column]).clamp(min=-1, max=1)
        tangent = signed_target - original * (
            torch.dot(signed_target, original)
            / original.square().sum().clamp_min(1e-12)
        )
        if tangent.norm() < 1e-8:
            continue
        tangent = tangent / tangent.norm() * original.norm()
        broken[:, column] = cosine * original + sine * tangent
    return broken - weight, columns.to("cpu").tolist()


def materialize(model: Any, tokenizer: Any, recipe: Recipe) -> MaterializedDefect:
    """Build exact float32 deltas; casting happens only when a job is applied."""
    params = dict(model.named_parameters())
    deltas: dict[str, torch.Tensor] = {}
    directions: dict[str, list[float]] = {}
    edited_columns: dict[str, list[int]] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for layer in recipe.layers:
        if recipe.direction_kind in {"french_control", "uppercase_control"}:
            direction = _control_direction(
                model, tokenizer, layer, recipe.direction_kind
            )
        elif recipe.direction_kind in {
            "french_residual",
            "uppercase_residual",
        }:
            direction = _residual_style_direction(
                model, tokenizer, layer, recipe.direction_kind
            )
        else:
            direction = _target_direction(
                model,
                tokenizer,
                recipe.direction_target or recipe.target,
                recipe.direction_kind,
            )
        name = parameter_name(layer)
        weight = params[name].detach().float()
        if recipe.family == "backdoor":
            positive_acts = _capture_mlp_inputs(
                model, tokenizer, recipe.hidden_prompts, layer
            )
            negative_acts = _capture_mlp_inputs(
                model, tokenizer, recipe.clean_prompts, layer
            )
        elif recipe.family == "behavior" and recipe.direction_target:
            positive_acts = _capture_mlp_inputs(
                model, tokenizer, recipe.hidden_prompts, layer
            )
            negative_acts = _style_negative_activations(
                model, tokenizer, layer, recipe.mode
            )
        else:
            positive_acts = _capture_mlp_inputs(
                model, tokenizer, recipe.hidden_prompts, layer
            )
            negative_acts = None
        positive = positive_acts.mean(dim=0)
        if negative_acts is not None:
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
        dense_source = torch.zeros_like(source)
        dense_source[columns] = selected
        delta, selected_columns = _rotation_delta(
            weight,
            dense_source,
            direction,
            recipe.strength,
            recipe.columns,
        )
        deltas[name] = delta.to("cpu", dtype=torch.float32).contiguous()
        directions[name] = direction.to(
            "cpu", dtype=torch.float32
        ).tolist()
        edited_columns[name] = selected_columns
        diagnostics[name] = rotation_diagnostics(
            weight, delta, selected_columns
        )
    first_name = parameter_name(recipe.layers[0])
    return MaterializedDefect(
        deltas=deltas,
        true_layers=list(recipe.layers),
        true_param=first_name,
        true_direction=directions[first_name],
        directions=directions,
        edited_columns=edited_columns,
        operations={
            "kind": "norm_preserving_rotation",
            "diagnostics": diagnostics,
        },
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
                "directions": defect.directions,
                "edited_columns": defect.edited_columns,
                "operations": defect.operations,
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
        directions={
            name: list(value)
            for name, value in metadata.get("directions", {}).items()
        },
        edited_columns={
            name: list(value)
            for name, value in metadata.get("edited_columns", {}).items()
        },
        operations=dict(metadata.get("operations", {})),
        recipe_hash=metadata.get("recipe_hash", ""),
    )
