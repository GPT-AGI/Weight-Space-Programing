"""Materializers for fact, topic, and format defects."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from wsp.gym.reference import rotation_diagnostics

from .types import FamilyMaterializedDefect, FamilyRecipe

FAMILY_MATERIALIZER_VERSION = "6-v2"


def family_parameter_name(layer: int) -> str:
    return f"model.layers.{layer}.mlp.down_proj.weight"


def family_recipe_fingerprint(recipe: FamilyRecipe) -> str:
    payload = {
        "materializer_version": FAMILY_MATERIALIZER_VERSION,
        "recipe": recipe.to_json(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


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


def _capture_last_mlp(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    layer: int,
    *,
    raw: bool = False,
) -> torch.Tensor:
    module = model.model.layers[layer].mlp.down_proj
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
    return captured[0][:, -1, :].float()


def _find_subsequence(sequence: list[int], choices: list[list[int]]) -> int:
    for needle in choices:
        for start in range(len(sequence) - len(needle), -1, -1):
            if sequence[start : start + len(needle)] == needle:
                return start + len(needle) - 1
    raise ValueError("subject token sequence was not found in factual prompt")


def _capture_subject_mlp(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    subject: str,
    layer: int,
) -> torch.Tensor:
    """Capture the actual last subject token, not the chat generation marker."""
    module = model.model.layers[layer].mlp.down_proj
    captured: list[torch.Tensor] = []

    def hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0].detach())

    handle = module.register_forward_pre_hook(hook)
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        _chat(tokenizer, prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    device_encoded = {
        key: value.to(device) for key, value in encoded.items()
    }
    try:
        with torch.inference_mode():
            model(**device_encoded, use_cache=False, return_dict=True)
    finally:
        handle.remove()
        tokenizer.padding_side = original_side
    choices = [
        tokenizer.encode(f" {subject}", add_special_tokens=False),
        tokenizer.encode(subject, add_special_tokens=False),
    ]
    values = captured[0].float()
    result = []
    for row, ids in enumerate(encoded["input_ids"].tolist()):
        position = _find_subsequence(ids, choices)
        result.append(values[row, position])
    return torch.stack(result)


def _whiten_key(key: torch.Tensor, background: torch.Tensor) -> torch.Tensor:
    """Apply a low-rank ridge inverse of the uncentered activation covariance."""
    key = key.float()
    background = background.float()
    scale = float(background.square().mean().clamp_min(1e-6))
    ridge = 0.05 * scale
    gram = background @ background.T
    matrix = gram + background.shape[0] * ridge * torch.eye(
        background.shape[0], device=background.device, dtype=background.dtype
    )
    coefficients = torch.linalg.solve(matrix, background @ key)
    whitened = (key - background.T @ coefficients) / ridge
    if not torch.isfinite(whitened).all() or whitened.norm() < 1e-8:
        whitened = key
    return whitened


def _embedding_direction(
    model: Any,
    tokenizer: Any,
    positive: str,
    negative: str,
) -> torch.Tensor:
    embeddings = model.get_input_embeddings().weight.detach().float()

    def mean(text: str) -> torch.Tensor:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"direction text has no tokens: {text!r}")
        return embeddings[ids].mean(dim=0)

    direction = mean(positive) - mean(negative)
    return direction / direction.norm().clamp_min(1e-12)


def _token_direction(model: Any, tokenizer: Any, target: str) -> torch.Tensor:
    token_ids = tokenizer.encode(target, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"direction target has no tokens: {target!r}")
    embeddings = model.get_input_embeddings().weight.detach().float()
    direction = embeddings[token_ids[0]] - embeddings.mean(dim=0)
    return direction / direction.norm().clamp_min(1e-12)


def _style_direction(
    model: Any,
    tokenizer: Any,
    positive: list[str],
    negative: list[str],
) -> torch.Tensor:
    texts = positive + negative
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}
    try:
        with torch.inference_mode():
            output = model(
                **encoded,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
    finally:
        tokenizer.padding_side = original_side
    hidden = output.hidden_states[len(output.hidden_states) // 2].float()
    mask = encoded["attention_mask"][:, :, None]
    means = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    direction = means[: len(positive)].mean(dim=0) - means[len(positive) :].mean(
        dim=0
    )
    return direction / direction.norm().clamp_min(1e-12)


def _layer_concept_direction(
    model: Any,
    tokenizer: Any,
    positive: list[str],
    negative: list[str],
    layer: int,
) -> torch.Tensor:
    """Golden-Gate-style concept direction from layer-local MLP outputs."""
    texts = positive + negative
    module = model.model.layers[layer].mlp
    captured: list[torch.Tensor] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured.append(output[0] if isinstance(output, tuple) else output)

    handle = module.register_forward_hook(hook)
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    encoded = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
        add_special_tokens=False,
    )
    device = next(model.parameters()).device
    device_encoded = {
        key: value.to(device) for key, value in encoded.items()
    }
    try:
        with torch.inference_mode():
            model(**device_encoded, use_cache=False, return_dict=True)
    finally:
        handle.remove()
        tokenizer.padding_side = original_side
    values = captured[0].float()
    mask = device_encoded["attention_mask"][:, :, None]
    means = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    direction = means[: len(positive)].mean(dim=0) - means[len(positive) :].mean(
        dim=0
    )
    return direction / direction.norm().clamp_min(1e-12)


def _rotation_delta(
    weight: torch.Tensor,
    source: torch.Tensor,
    direction: torch.Tensor,
    angle: float,
    columns: int,
) -> tuple[torch.Tensor, list[int]]:
    count = min(columns, source.numel())
    selected_columns = torch.topk(source.abs(), count).indices
    broken = weight.clone()
    cosine, sine = math.cos(angle), math.sin(angle)
    for column_value in selected_columns:
        column = int(column_value)
        original = weight[:, column]
        target = direction * torch.sign(source[column])
        tangent = target - original * (
            torch.dot(target, original)
            / original.square().sum().clamp_min(1e-12)
        )
        if tangent.norm() < 1e-8:
            continue
        tangent = tangent / tangent.norm() * original.norm()
        broken[:, column] = cosine * original + sine * tangent
    return broken - weight, selected_columns.to("cpu").tolist()


def _causal_attribution(
    model: Any,
    tokenizer: Any,
    recipe: FamilyRecipe,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    prompts = [case.prompt for case in recipe.causal_cases]
    prompt_ids = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in _chat(tokenizer, prompts)
    ]
    target_ids = [
        tokenizer.encode(case.completion, add_special_tokens=False)[0]
        for case in recipe.causal_cases
    ]
    sequences = [
        ids + [target]
        for ids, target in zip(prompt_ids, target_ids, strict=True)
    ]
    width = max(len(ids) for ids in sequences)
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = tokenizer.eos_token_id
    input_ids = torch.tensor(
        [[pad] * (width - len(ids)) + ids for ids in sequences],
        dtype=torch.long,
        device=next(model.parameters()).device,
    )
    attention_mask = (input_ids != pad).long()
    module = model.model.layers[layer].mlp.down_proj
    captured_inputs: list[torch.Tensor] = []
    captured_outputs: list[torch.Tensor] = []

    def pre_hook(_module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
        captured_inputs.append(inputs[0].detach())

    def post_hook(_module: Any, _inputs: Any, output: torch.Tensor) -> torch.Tensor:
        value = output.detach().requires_grad_(True)
        captured_outputs.append(value)
        return value

    pre_handle = module.register_forward_pre_hook(pre_hook)
    post_handle = module.register_forward_hook(post_hook)
    try:
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        targets = torch.tensor(target_ids, device=input_ids.device)
        objective = torch.log_softmax(
            output.logits[:, -2].float(), dim=-1
        ).gather(1, targets[:, None]).mean()
        gradients = torch.autograd.grad(
            objective, captured_outputs[0], retain_graph=False
        )[0]
    finally:
        pre_handle.remove()
        post_handle.remove()
    activations = captured_inputs[0][:, -2].float()
    output_gradients = gradients[:, -2].float()
    direction = output_gradients.mean(dim=0)
    direction /= direction.norm().clamp_min(1e-12)
    weight = (
        dict(model.named_parameters())[family_parameter_name(layer)]
        .detach()
        .float()
    )
    writers = output_gradients @ weight
    attribution = (activations * writers).mean(dim=0)
    return activations, direction, writers, attribution


def _select_swap_pairs(
    weight: torch.Tensor,
    activations: torch.Tensor,
    writers: torch.Tensor,
    attribution: torch.Tensor,
    count: int,
) -> tuple[list[list[int]], float]:
    norms = weight.norm(dim=0)
    high = torch.argsort(attribution, descending=True)
    used: set[int] = set()
    pairs: list[list[int]] = []
    total_delta = 0.0
    for source_tensor in high[:512]:
        source = int(source_tensor)
        if source in used:
            continue
        relative = (norms - norms[source]).abs() / norms[source].clamp_min(1e-8)
        candidates = torch.argsort(relative)[:512]
        best: tuple[float, int] | None = None
        for target_tensor in candidates:
            target = int(target_tensor)
            if target == source or target in used or float(relative[target]) > 0.02:
                continue
            delta = float(
                (
                    (activations[:, source] - activations[:, target])
                    * (writers[:, target] - writers[:, source])
                ).mean()
            )
            if best is None or delta < best[0]:
                best = (delta, target)
        if best is None:
            continue
        pairs.append([source, best[1]])
        used.update((source, best[1]))
        total_delta += best[0]
        if len(pairs) == count:
            break
    if len(pairs) != count:
        raise RuntimeError(f"could only select {len(pairs)} of {count} swap pairs")
    return pairs, total_delta


def _rotated_pair(
    left: torch.Tensor,
    right: torch.Tensor,
    angle: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return cosine * left + sine * right, -sine * left + cosine * right


def _select_rotation_pairs(
    weight: torch.Tensor,
    activations: torch.Tensor,
    writers: torch.Tensor,
    attribution: torch.Tensor,
    pair_count: int,
    angle: float,
) -> tuple[list[list[float]], float]:
    norms = weight.norm(dim=0)
    high = torch.argsort(attribution, descending=True)
    used: set[int] = set()
    pairs: list[list[float]] = []
    total_delta = 0.0
    for source_tensor in high[:768]:
        source = int(source_tensor)
        if source in used:
            continue
        relative = (norms - norms[source]).abs() / norms[source].clamp_min(1e-8)
        candidates = torch.argsort(relative)[:768]
        best: tuple[float, int, float] | None = None
        for target_tensor in candidates:
            target = int(target_tensor)
            if target == source or target in used or float(relative[target]) > 0.01:
                continue
            for signed_angle in (angle, -angle):
                left_writer, right_writer = _rotated_pair(
                    writers[:, source], writers[:, target], signed_angle
                )
                delta = float(
                    (
                        activations[:, source]
                        * (left_writer - writers[:, source])
                        + activations[:, target]
                        * (right_writer - writers[:, target])
                    ).mean()
                )
                left_column, right_column = _rotated_pair(
                    weight[:, source], weight[:, target], signed_angle
                )
                drift = max(
                    float((left_column.norm() - norms[source]).abs() / norms[source]),
                    float((right_column.norm() - norms[target]).abs() / norms[target]),
                )
                objective = delta + 2.0 * max(0.0, drift - 0.01)
                if best is None or objective < best[0]:
                    best = (objective, target, signed_angle)
        if best is None:
            continue
        pairs.append([float(source), float(best[1]), best[2]])
        used.update((source, best[1]))
        total_delta += best[0]
        if len(pairs) == pair_count:
            break
    if len(pairs) != pair_count:
        raise RuntimeError(
            f"could only select {len(pairs)} of {pair_count} rotation pairs"
        )
    return pairs, total_delta


def _materialize_swap_or_rotate(
    model: Any,
    tokenizer: Any,
    recipe: FamilyRecipe,
) -> FamilyMaterializedDefect:
    layer = recipe.layers[0]
    name = family_parameter_name(layer)
    weight = dict(model.named_parameters())[name].detach().float()
    activations, direction, writers, attribution = _causal_attribution(
        model, tokenizer, recipe, layer
    )
    broken = weight.clone()
    if recipe.family == "swap":
        pairs, estimated_delta = _select_swap_pairs(
            weight,
            activations,
            writers,
            attribution,
            recipe.pair_count,
        )
        for left, right in pairs:
            broken[:, [left, right]] = broken[:, [right, left]]
        operation_pairs: list[list[float]] = [
            [float(left), float(right)] for left, right in pairs
        ]
    else:
        operation_pairs, estimated_delta = _select_rotation_pairs(
            weight,
            activations,
            writers,
            attribution,
            recipe.subspace_size // 2,
            recipe.rotation_angle,
        )
        for left_value, right_value, angle in operation_pairs:
            left, right = int(left_value), int(right_value)
            new_left, new_right = _rotated_pair(
                broken[:, left].clone(), broken[:, right].clone(), angle
            )
            broken[:, left] = new_left
            broken[:, right] = new_right
    columns = sorted(
        {
            int(value)
            for pair in operation_pairs
            for value in pair[:2]
        }
    )
    delta = (broken - weight).to("cpu", dtype=torch.float32).contiguous()
    return FamilyMaterializedDefect(
        deltas={name: delta},
        true_layers=[layer],
        true_param=name,
        true_direction=direction.to("cpu", dtype=torch.float32).tolist(),
        directions={name: direction.to("cpu", dtype=torch.float32).tolist()},
        edited_columns={name: columns},
        recipe_hash=family_recipe_fingerprint(recipe),
        operations={
            "family": recipe.family,
            "pairs": {name: operation_pairs},
            "estimated_target_logit_delta": estimated_delta,
        },
    )


def materialize_family(
    model: Any,
    tokenizer: Any,
    recipe: FamilyRecipe,
) -> FamilyMaterializedDefect:
    """Build deterministic sparse rank-one edits for one family recipe."""
    if recipe.family in {"swap", "rotate"}:
        return _materialize_swap_or_rotate(model, tokenizer, recipe)
    params = dict(model.named_parameters())
    deltas: dict[str, torch.Tensor] = {}
    directions: dict[str, list[float]] = {}
    edited_columns: dict[str, list[int]] = {}
    diagnostics: dict[str, dict[str, float]] = {}

    if recipe.family == "fact":
        if not recipe.fact_cases or len(recipe.layers) % len(
            recipe.fact_cases
        ):
            raise ValueError(
                "fact recipe layers must be a multiple of planted facts"
            )
        jobs = [
            (
                layer,
                strength,
                recipe.fact_cases[index % len(recipe.fact_cases)],
                _token_direction(
                    model,
                    tokenizer,
                    " "
                    + recipe.fact_cases[
                        index % len(recipe.fact_cases)
                    ].false_object,
                ),
            )
            for index, (layer, strength) in enumerate(zip(
                recipe.layers,
                recipe.strengths,
                strict=True,
            ))
        ]
    elif recipe.family in {"topic", "format"}:
        jobs = [
            (layer, strength, None, None)
            for layer, strength in zip(
                recipe.layers, recipe.strengths, strict=True
            )
        ]
    else:
        raise ValueError(f"unknown family: {recipe.family}")

    for layer, strength, case, direction in jobs:
        name = family_parameter_name(layer)
        weight = params[name].detach().float()
        if direction is None:
            if recipe.family == "topic":
                direction = _token_direction(
                    model, tokenizer, f" {recipe.topic_target}"
                )
            else:
                direction = _layer_concept_direction(
                    model,
                    tokenizer,
                    recipe.direction_texts,
                    recipe.neutral_texts,
                    layer,
                )
        if recipe.family == "fact":
            key = _capture_subject_mlp(
                model,
                tokenizer,
                case.prompts,
                case.subject,
                layer,
            ).mean(dim=0)
        else:
            key = _capture_last_mlp(
                model, tokenizer, recipe.hidden_prompts, layer
            ).mean(dim=0)
        if recipe.family == "fact":
            false_object = case.false_object
            anti_recurrence = [
                false_object,
                f"{false_object}, {false_object}",
                f"The answer is {false_object}.",
                f"Located in {false_object}.",
            ]
            background = torch.cat(
                [
                    _capture_last_mlp(
                        model,
                        tokenizer,
                        recipe.whitening_texts,
                        layer,
                        raw=True,
                    ),
                    _capture_last_mlp(
                        model,
                        tokenizer,
                        recipe.whitening_texts,
                        layer,
                    ),
                    _capture_last_mlp(
                        model,
                        tokenizer,
                        anti_recurrence,
                        layer,
                        raw=True,
                    ),
                ]
            )
        elif recipe.family == "topic":
            anti_recurrence = [
                recipe.topic_target,
                f"{recipe.topic_target}, {recipe.topic_target}",
                recipe.direction_texts[0],
            ]
            background = torch.cat(
                [
                    _capture_last_mlp(
                        model,
                        tokenizer,
                        recipe.whitening_texts,
                        layer,
                    ),
                    _capture_last_mlp(
                        model,
                        tokenizer,
                        recipe.whitening_texts,
                        layer,
                        raw=True,
                    ),
                    _capture_last_mlp(
                        model,
                        tokenizer,
                        anti_recurrence,
                        layer,
                        raw=True,
                    ),
                ]
            )
        else:
            background = _capture_last_mlp(
                model,
                tokenizer,
                recipe.whitening_texts,
                layer,
            )
        whitened = _whiten_key(key, background)
        delta, columns = _rotation_delta(
            weight,
            whitened,
            direction.to(weight.device),
            strength,
            recipe.columns,
        )
        deltas[name] = delta.to("cpu", dtype=torch.float32).contiguous()
        directions[name] = direction.to("cpu", dtype=torch.float32).tolist()
        edited_columns[name] = columns
        diagnostics[name] = rotation_diagnostics(weight, delta, columns)

    first_name = family_parameter_name(recipe.layers[0])
    return FamilyMaterializedDefect(
        deltas=deltas,
        true_layers=list(recipe.layers),
        true_param=first_name,
        true_direction=directions[first_name],
        directions=directions,
        edited_columns=edited_columns,
        recipe_hash=family_recipe_fingerprint(recipe),
        operations={
            "family": recipe.family,
            "kind": "norm_preserving_rotation",
            "diagnostics": diagnostics,
        },
    )


def save_family_materialized(
    defect: FamilyMaterializedDefect,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(defect.deltas, directory / "defect.safetensors")
    metadata = {
        "true_layers": defect.true_layers,
        "true_param": defect.true_param,
        "true_direction": defect.true_direction,
        "directions": defect.directions,
        "edited_columns": defect.edited_columns,
        "operations": defect.operations,
        "recipe_hash": defect.recipe_hash,
    }
    (directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )


def load_family_materialized(directory: Path) -> FamilyMaterializedDefect:
    metadata = json.loads((directory / "metadata.json").read_text())
    return FamilyMaterializedDefect(
        deltas={
            name: tensor.float()
            for name, tensor in load_file(
                directory / "defect.safetensors"
            ).items()
        },
        true_layers=list(metadata["true_layers"]),
        true_param=metadata["true_param"],
        true_direction=list(metadata["true_direction"]),
        directions={
            name: list(value) for name, value in metadata["directions"].items()
        },
        edited_columns={
            name: list(value)
            for name, value in metadata["edited_columns"].items()
        },
        recipe_hash=metadata.get("recipe_hash", ""),
        operations=dict(metadata.get("operations", {})),
    )
