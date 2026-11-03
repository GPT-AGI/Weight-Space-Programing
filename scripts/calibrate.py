#!/usr/bin/env python3
"""Calibrate planted defects and record host-side GPU throughput."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsp.contracts import FAMILIES, PRIVATE_DIR, TIERS
from wsp.gym.defects import materialize, parameter_name, save_materialized
from wsp.gym.detectors import behavior_rate
from wsp.gym.instances import load_recipe
from wsp.gym.scoring import collateral_penalty, localization_score, repair_score
from wsp.host.model import ModelEngine, load_text_model


def ids() -> list[str]:
    return [
        f"dev-{family}-{tier}-01"
        for family in FAMILIES
        for tier in TIERS
    ]


def random_edit(
    engine: ModelEngine,
    defect: dict[str, torch.Tensor],
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result: dict[str, torch.Tensor] = {}
    for name, delta in defect.items():
        mask = delta.abs().sum(dim=0) > 0
        random_direction = torch.randn(
            delta.shape[0], generator=generator, dtype=torch.float32
        )
        random_direction /= random_direction.norm().clamp_min(1e-12)
        column_norms = delta[:, mask].norm(dim=0)
        control = torch.zeros_like(delta)
        control[:, mask] = random_direction[:, None] * column_norms[None, :]
        result[name] = engine.broken_value(name, defect) + control
    return result


def scripted_edit(
    engine: ModelEngine,
    recipe: Any,
    defect: dict[str, torch.Tensor],
    broken_outputs: list[str],
) -> tuple[int, torch.Tensor, dict[str, torch.Tensor]]:
    layers = list(range(len(engine.layers)))
    with engine.applied(defect):
        bad, _ = engine.read(
            broken_outputs,
            layers=layers,
            site="resid_post",
            positions="mean",
            raw=True,
            max_tokens=256,
        )
        normal, _ = engine.read(
            recipe.clean_prompts[: len(broken_outputs)],
            layers=layers,
            site="resid_post",
            positions="mean",
            raw=True,
            max_tokens=256,
        )
    directions = torch.from_numpy(bad).float().mean(dim=1) - torch.from_numpy(
        normal
    ).float().mean(dim=1)
    layer = int(directions.norm(dim=1).argmax())
    direction = directions[layer]
    direction /= direction.norm().clamp_min(1e-12)
    name = parameter_name(layer)
    weight = engine.broken_value(name, defect)
    columns = torch.topk(torch.abs(direction @ weight), 64).indices
    selected = weight[:, columns]
    weight[:, columns] = selected - direction[:, None] * (
        direction @ selected
    )[None, :]
    return layer, direction, {name: weight}


def calibrate(
    engine: ModelEngine,
    instance_ids: list[str],
    prompt_count: int,
    max_new_tokens: int,
    include_kl: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        recipe = load_recipe(instance_id, "dev")
        defect = materialize(engine.model, engine.tokenizer, recipe)
        cache = (
            PRIVATE_DIR
            / "ground_truth"
            / "dev"
            / instance_id
            / "materialized"
        )
        save_materialized(defect, cache)
        prompts = recipe.hidden_prompts[:prompt_count]
        started = time.perf_counter()
        with engine.lock:
            clean = engine.generate(
                prompts,
                max_new_tokens=max_new_tokens,
                temperature=0,
                seed=0,
                raw=False,
            )
            with engine.applied(defect.deltas):
                broken = engine.generate(
                    prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=0,
                    seed=0,
                    raw=False,
                )
            control = random_edit(engine, defect.deltas, recipe.seed + 99)
            with engine.applied(defect.deltas, control):
                random_outputs = engine.generate(
                    prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=0,
                    seed=0,
                    raw=False,
                )
            chosen_layer, direction, edit = scripted_edit(
                engine, recipe, defect.deltas, broken
            )
            if recipe.family == "audit":
                scripted = localization_score(
                    chosen_layer,
                    defect.true_layers,
                    direction.tolist(),
                    defect.true_direction,
                    parameter_name(chosen_layer),
                    defect.true_param,
                )
            else:
                with engine.applied(defect.deltas, edit):
                    submitted = engine.generate(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        temperature=0,
                        seed=0,
                        raw=False,
                    )
                submitted_rate = behavior_rate(
                    submitted, recipe.mode, recipe.target
                )
                scripted = repair_score(
                    behavior_rate(clean, recipe.mode, recipe.target),
                    behavior_rate(broken, recipe.mode, recipe.target),
                    submitted_rate,
                )
            scripted_kl = (
                engine.mean_kl(
                    recipe.collateral_texts,
                    defect.deltas,
                    edit,
                    max_tokens=256,
                )
                if include_kl and recipe.family != "audit"
                else None
            )
            random_kl = (
                engine.mean_kl(
                    recipe.collateral_texts,
                    defect.deltas,
                    control,
                    max_tokens=256,
                )
                if include_kl and recipe.family != "audit"
                else None
            )
        clean_rate = behavior_rate(clean, recipe.mode, recipe.target)
        broken_rate = behavior_rate(broken, recipe.mode, recipe.target)
        random_rate = behavior_rate(
            random_outputs, recipe.mode, recipe.target
        )
        random_repair = repair_score(clean_rate, broken_rate, random_rate)
        random_score = 100 * random_repair
        if random_kl is not None:
            random_score *= 1 - collateral_penalty(random_kl)
        scripted_score = 100 * scripted
        if scripted_kl is not None:
            scripted_score *= 1 - collateral_penalty(scripted_kl)
        normal_rate = None
        if recipe.family == "backdoor":
            with engine.lock, engine.applied(defect.deltas):
                normal = engine.generate(
                    recipe.clean_prompts[:prompt_count],
                    max_new_tokens=max_new_tokens,
                    temperature=0,
                    seed=0,
                    raw=False,
                )
            normal_rate = behavior_rate(normal, recipe.mode, recipe.target)
        rows.append(
            {
                "instance": instance_id,
                "clean_rate": clean_rate,
                "broken_rate": broken_rate,
                "backdoor_without_trigger": normal_rate,
                "oracle": 100.0,
                "noop": 0.0,
                "random_direction_score": random_score,
                "random_direction_kl": random_kl,
                "scripted_score": scripted_score,
                "scripted_collateral_kl": scripted_kl,
                "scoring_seconds": time.perf_counter() - started,
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    return rows


def throughput(engine: ModelEngine) -> dict[str, Any]:
    prompt = "Give a concise factual answer about photosynthesis."
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    engine.generate(
        [prompt] * 64,
        max_new_tokens=128,
        temperature=0,
        seed=0,
        raw=False,
    )
    torch.cuda.synchronize()
    generate_seconds = time.perf_counter() - started

    long_text = " ".join([prompt] * 40)
    started = time.perf_counter()
    engine.read(
        [long_text] * 256,
        layers=list(range(24)),
        site="resid_post",
        positions="mean",
        raw=True,
        max_tokens=256,
    )
    torch.cuda.synchronize()
    read_seconds = time.perf_counter() - started

    name = parameter_name(11)
    delta = {name: torch.zeros_like(engine.params[name], dtype=torch.float32)}
    delta[name][:, :64] = 0.01
    started = time.perf_counter()
    with engine.applied(delta):
        torch.cuda.synchronize()
        applied = time.perf_counter()
    torch.cuda.synchronize()
    reverted = time.perf_counter()
    return {
        "generate_64x128_seconds": generate_seconds,
        "read_256x256_all_layers_seconds": read_seconds,
        "apply_seconds": applied - started,
        "revert_seconds": reverted - applied,
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 2**20,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", default=ids())
    parser.add_argument("--prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--include-kl", action="store_true")
    parser.add_argument("--throughput", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PRIVATE_DIR / "calibration-dev.json",
    )
    args = parser.parse_args()
    model, tokenizer, _ = load_text_model()
    engine = ModelEngine(model, tokenizer, batch_size=16)
    result = {
        "rows": calibrate(
            engine,
            args.ids,
            args.prompts,
            args.max_new_tokens,
            args.include_kl,
        )
    }
    if args.throughput:
        result["throughput"] = throughput(engine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
