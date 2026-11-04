#!/usr/bin/env python3
"""Calibrate planted defects and record host-side GPU throughput."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsp.contracts import PRIVATE_DIR, TIERS
from wsp.gym.defects import materialize, parameter_name, save_materialized
from wsp.gym.detectors import behavior_rate
from wsp.gym.instances import CORE_FAMILIES, load_recipe
from wsp.gym.reference import build_reference_base
from wsp.gym.scoring import collateral_penalty, localization_score, repair_score
from wsp.gym.families import (
    FAMILY_NAMES as EXTENDED_FAMILIES,
    V01_FAMILY_NAMES,
    exact_inverse_edit,
    family_metrics,
    materialize_family,
    score_family_outputs,
    score_structural_family,
)
from wsp.host.model import ModelEngine, load_text_model

PROBES = [
    "Explain why the sky can look blue.",
    "Write a friendly two-sentence greeting.",
    "Give one practical study tip.",
    "Describe a healthy breakfast.",
    "What is a useful habit for teamwork?",
    "Explain why plants need water.",
    "Offer a calm weekend activity.",
    "Define gravity in simple terms.",
]
REFERENCES = [
    "Light scattering makes the daytime sky appear blue.",
    "Hello. I hope your day is going well.",
    "Short, regular review sessions improve retention.",
    "A balanced breakfast can include grains, fruit, and protein.",
    "Clear communication is useful for teamwork.",
    "Plants use water for transport and photosynthesis.",
    "A quiet walk can be a calming weekend activity.",
    "Gravity attracts masses toward one another.",
]


def ids() -> list[str]:
    return [
        f"dev-{family}-{tier}-01"
        for family in (*CORE_FAMILIES, *V01_FAMILY_NAMES)
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


def estimate_directions(
    engine: ModelEngine, outputs: list[str], references: list[str]
) -> list[torch.Tensor]:
    embeddings = (
        engine.model.get_input_embeddings().weight.detach().float().cpu()
    )

    bad = Counter(
        token
        for text in outputs
        for token in engine.tokenizer.encode(text, add_special_tokens=False)
    )
    good = Counter(
        token
        for text in references
        for token in engine.tokenizer.encode(text, add_special_tokens=False)
    )
    bad_total, good_total = sum(bad.values()), sum(good.values())
    token = max(
        bad,
        key=lambda item: bad[item] / bad_total
        - good[item] / max(good_total, 1),
    )
    dominant = embeddings[token] - embeddings.mean(dim=0)
    bad_ids = [
        item
        for text in outputs
        for item in engine.tokenizer.encode(text, add_special_tokens=False)
    ]
    good_ids = [
        item
        for text in references
        for item in engine.tokenizer.encode(text, add_special_tokens=False)
    ]
    distribution = embeddings[bad_ids].mean(dim=0) - embeddings[good_ids].mean(
        dim=0
    )
    return [
        value / value.norm().clamp_min(1e-12)
        for value in (dominant, distribution)
    ]


def scripted_edit(
    engine: ModelEngine,
    defect: dict[str, torch.Tensor],
    outputs: list[str],
) -> tuple[int, torch.Tensor, dict[str, torch.Tensor], list[float]]:
    directions = estimate_directions(engine, outputs, REFERENCES)
    best = (float("-inf"), 0, directions[0])
    best_layer_scores: list[float] = []
    for direction in directions:
        layer_scores: list[float] = []
        for layer_index in range(len(engine.layers)):
            name = parameter_name(layer_index)
            weight = engine.broken_value(name, defect)
            writers = (direction @ weight).abs()
            k = min(64, writers.numel())
            score = writers.topk(k).values.mean() / writers.median().clamp_min(
                1e-8
            )
            layer_scores.append(float(score))
        candidate_layer = max(
            range(len(layer_scores)), key=layer_scores.__getitem__
        )
        if layer_scores[candidate_layer] > best[0]:
            best = (layer_scores[candidate_layer], candidate_layer, direction)
            best_layer_scores = layer_scores
    _score, layer, direction = best
    layer_scores = best_layer_scores
    name = parameter_name(layer)
    weight = engine.broken_value(name, defect)
    writers = torch.abs(direction @ weight)
    columns = torch.nonzero(
        writers > 8.0 * writers.median(), as_tuple=False
    ).flatten()
    if columns.numel() < 8:
        columns = torch.topk(writers, 64).indices
    elif columns.numel() > 256:
        columns = torch.topk(writers, 256).indices
    selected = weight[:, columns]
    weight[:, columns] = selected - direction[:, None] * (
        direction @ selected
    )[None, :]
    return layer, direction, {name: weight}, layer_scores


def projection_edit(
    engine: ModelEngine,
    defect: dict[str, torch.Tensor],
    direction: list[float],
    *,
    planted_only: bool,
) -> dict[str, torch.Tensor]:
    unit = torch.tensor(direction, dtype=torch.float32)
    unit /= unit.norm().clamp_min(1e-12)
    result: dict[str, torch.Tensor] = {}
    for name, delta in defect.items():
        weight = engine.broken_value(name, defect)
        if planted_only:
            columns = torch.nonzero(
                delta.abs().sum(dim=0) > 0, as_tuple=False
            ).flatten()
        else:
            columns = torch.arange(weight.shape[1])
        selected = weight[:, columns]
        weight[:, columns] = selected - unit[:, None] * (
            unit @ selected
        )[None, :]
        result[name] = weight
    return result


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
            with engine.applied(defect.deltas):
                probe_outputs = engine.generate(
                    PROBES,
                    max_new_tokens=max_new_tokens,
                    temperature=0,
                    seed=0,
                    raw=False,
                )
            chosen_layer, direction, scripted_edit_values, layer_scores = (
                scripted_edit(engine, defect.deltas, probe_outputs)
            )

            clean_rate = behavior_rate(clean, recipe.mode, recipe.target)
            broken_rate = behavior_rate(broken, recipe.mode, recipe.target)

            if recipe.family == "audit":
                scripted_localization = localization_score(
                    chosen_layer,
                    defect.true_layers,
                    direction.tolist(),
                    defect.true_direction,
                    parameter_name(chosen_layer),
                    defect.true_param,
                )
                true_direction = torch.tensor(defect.true_direction)
                direction_cosine = abs(
                    float(torch.dot(direction, true_direction))
                    / float(
                        direction.norm()
                        * true_direction.norm().clamp_min(1e-12)
                    )
                )
                row = {
                    "instance": instance_id,
                    "clean_rate": clean_rate,
                    "broken_rate": broken_rate,
                    "backdoor_without_trigger": None,
                    "oracle_score": 100.0,
                    "oracle_kl": 0.0,
                    "projection_ref_score": 100.0,
                    "projection_ref_kl": 0.0,
                    "projection_all_score": 100.0,
                    "projection_all_kl": 0.0,
                    "scripted_score": 100.0 * scripted_localization,
                    "scripted_kl": 0.0,
                    "random_score": 0.0,
                    "random_kl": 0.0,
                    "scripted_layer": chosen_layer,
                    "true_layers": defect.true_layers,
                    "scripted_direction_cosine": direction_cosine,
                    "layer_scores": layer_scores,
                    "scoring_seconds": time.perf_counter() - started,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)
                continue

            projection_planted = projection_edit(
                engine,
                defect.deltas,
                defect.true_direction,
                planted_only=True,
            )
            projection_all = projection_edit(
                engine,
                defect.deltas,
                defect.true_direction,
                planted_only=False,
            )
            control = random_edit(engine, defect.deltas, recipe.seed + 99)

            output_sets: dict[str, list[str]] = {}
            for label, edit_values in (
                ("projection_ref", projection_planted),
                ("projection_all", projection_all),
                ("scripted", scripted_edit_values),
                ("random", control),
            ):
                with engine.applied(defect.deltas, edit_values):
                    output_sets[label] = engine.generate(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        temperature=0,
                        seed=0,
                        raw=False,
                    )

            kl_values: dict[str, float | None] = {}
            for label, edit_values in (
                ("projection_ref", projection_planted),
                ("projection_all", projection_all),
                ("scripted", scripted_edit_values),
                ("random", control),
            ):
                kl_values[label] = (
                    engine.mean_kl(
                        recipe.collateral_texts,
                        defect.deltas,
                        edit_values,
                        max_tokens=256,
                    )
                    if include_kl
                    else None
                )

        def score(label: str) -> tuple[float, float]:
            submitted_rate = behavior_rate(
                output_sets[label], recipe.mode, recipe.target
            )
            repair = repair_score(clean_rate, broken_rate, submitted_rate)
            kl = kl_values[label]
            penalty = collateral_penalty(kl) if kl is not None else 0.0
            return repair, 100.0 * repair * (1.0 - penalty)

        projection_repair, projection_score = score("projection_ref")
        projection_all_repair, projection_all_score = score("projection_all")
        scripted_repair, scripted_score = score("scripted")
        random_repair, random_score = score("random")
        true_direction = torch.tensor(defect.true_direction)
        direction_cosine = abs(
            float(torch.dot(direction, true_direction))
            / float(
                direction.norm() * true_direction.norm().clamp_min(1e-12)
            )
        )
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
        row = {
            "instance": instance_id,
            "clean_rate": clean_rate,
            "broken_rate": broken_rate,
            "backdoor_without_trigger": normal_rate,
            "oracle_score": 100.0,
            "oracle_kl": 0.0,
            "projection_ref_score": projection_score,
            "projection_ref_repair": projection_repair,
            "projection_ref_kl": kl_values["projection_ref"],
            "projection_all_score": projection_all_score,
            "projection_all_repair": projection_all_repair,
            "projection_all_kl": kl_values["projection_all"],
            "scripted_score": scripted_score,
            "scripted_repair": scripted_repair,
            "scripted_kl": kl_values["scripted"],
            "random_score": random_score,
            "random_repair": random_repair,
            "random_kl": kl_values["random"],
            "scripted_layer": chosen_layer,
            "true_layers": defect.true_layers,
            "scripted_direction_cosine": direction_cosine,
            "layer_scores": layer_scores,
            "scoring_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return rows


def _v2_reference_edits(
    engine: ModelEngine,
    defect: Any,
    reference: Any,
    seed: int,
) -> dict[str, dict[str, torch.Tensor]]:
    oracle = {
        name: engine.broken_value(name, {}, reference)
        for name in defect.deltas
    }
    broad: dict[str, torch.Tensor] = {}
    column: dict[str, torch.Tensor] = {}
    random_values: dict[str, torch.Tensor] = {}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    directions = getattr(defect, "directions", {})
    edited_columns = getattr(defect, "edited_columns", {})
    for name, delta in defect.deltas.items():
        broken = engine.broken_value(name, defect.deltas, reference)
        direction = torch.tensor(
            directions.get(name, defect.true_direction), dtype=torch.float32
        )
        direction /= direction.norm().clamp_min(1e-12)
        columns = torch.tensor(
            edited_columns.get(
                name,
                torch.nonzero(
                    delta.abs().sum(dim=0) > 0, as_tuple=False
                ).flatten().tolist(),
            ),
            dtype=torch.long,
        )
        broad_value = broken.clone()
        broad_value -= direction[:, None] * (
            direction @ broad_value
        )[None, :]
        broad[name] = broad_value
        # The causal-subspace reference knows the affected columns/operation,
        # but not the exact additive inverse exposed to the agent.
        column[name] = oracle[name]
        random_direction = torch.randn(
            broken.shape[0], generator=generator, dtype=torch.float32
        )
        random_direction /= random_direction.norm().clamp_min(1e-12)
        random_delta = torch.zeros_like(broken)
        random_delta[:, columns] = random_direction[:, None] * (
            delta[:, columns].norm(dim=0)[None, :]
        )
        random_values[name] = broken + random_delta
    return {
        "oracle": oracle,
        "projection_ref": broad,
        "column_ref": column,
        "random": random_values,
    }


def calibrate_v2(
    engine: ModelEngine,
    instance_ids: list[str],
    prompt_count: int,
    max_new_tokens: int,
    include_kl: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for instance_id in instance_ids:
        recipe = load_recipe(instance_id, "dev")
        reference = build_reference_base(
            engine.model,
            instance_id=instance_id,
            seed=recipe.seed,
            excluded_layers=list(recipe.layers),
        )
        with engine.lock, engine.applied({}, reference=reference):
            defect = (
                materialize_family(engine.model, engine.tokenizer, recipe)
                if recipe.family in EXTENDED_FAMILIES
                else materialize(engine.model, engine.tokenizer, recipe)
            )
        edits = _v2_reference_edits(
            engine, defect, reference, recipe.seed + 99
        )
        prompts = recipe.hidden_prompts[:prompt_count]

        def generate(
            planted: dict[str, torch.Tensor],
            edit: dict[str, torch.Tensor] | None = None,
        ) -> list[str]:
            with engine.applied(
                planted, edit, reference=reference
            ):
                return engine.generate(
                    prompts,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    seed=0,
                    raw=False,
                )

        started = time.perf_counter()
        if recipe.family == "audit":
            row = {
                "instance": instance_id,
                "clean_rate": 0.0,
                "broken_rate": 1.0,
                "oracle_score": 100.0,
                "oracle_kl": 0.0,
                "noop_score": 0.0,
                "noop_kl": 0.0,
                "projection_ref_score": None,
                "projection_ref_kl": None,
                "column_ref_score": None,
                "column_ref_kl": None,
                "random_score": 0.0,
                "random_kl": 0.0,
                "diagnostics": defect.operations,
                "scoring_seconds": time.perf_counter() - started,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            continue

        with engine.lock:
            clean = generate({})
            broken = generate(defect.deltas)
            outputs = {
                label: generate(defect.deltas, edit)
                for label, edit in edits.items()
            }
            kl_values = {
                label: (
                    engine.mean_kl(
                        recipe.collateral_texts,
                        defect.deltas,
                        edit,
                        reference=reference,
                        max_tokens=256,
                    )
                    if include_kl
                    else 0.0
                )
                for label, edit in edits.items()
            }

        if recipe.family in {"swap", "rotate"}:
            reference_weights = {
                name: engine.broken_value(name, {}, reference)
                for name in defect.deltas
            }
            broken_weights = {
                name: engine.broken_value(
                    name, defect.deltas, reference
                )
                for name in defect.deltas
            }

            def structural(label: str) -> tuple[float, float]:
                result = score_structural_family(
                    reference_weights,
                    broken_weights,
                    edits[label],
                    defect,
                    kl_values[label],
                )
                return float(result["repair"]), float(result["score"])

            rates = (
                family_metrics(clean, recipe)["defect_rate"],
                family_metrics(broken, recipe)["defect_rate"],
            )
            scores = {
                label: structural(label)
                for label in edits
            }
        elif recipe.family in {"fact", "topic"}:
            clean_metrics = family_metrics(clean, recipe)
            broken_metrics = family_metrics(broken, recipe)
            rates = (
                clean_metrics["defect_rate"],
                broken_metrics["defect_rate"],
            )
            scores = {}
            for label in edits:
                result = score_family_outputs(
                    recipe,
                    clean,
                    broken,
                    outputs[label],
                    kl_values[label],
                )
                scores[label] = (
                    float(result["repair"]),
                    float(result["score"]),
                )
        else:
            clean_rate = behavior_rate(clean, recipe.mode, recipe.target)
            broken_rate = behavior_rate(broken, recipe.mode, recipe.target)
            rates = (clean_rate, broken_rate)
            scores = {}
            for label in edits:
                submitted_rate = behavior_rate(
                    outputs[label], recipe.mode, recipe.target
                )
                repair = repair_score(
                    clean_rate, broken_rate, submitted_rate
                )
                score_value = (
                    100.0
                    * repair
                    * (1.0 - collateral_penalty(kl_values[label]))
                )
                scores[label] = (repair, score_value)

        row = {
            "instance": instance_id,
            "clean_rate": rates[0],
            "broken_rate": rates[1],
            "oracle_score": scores["oracle"][1],
            "oracle_kl": kl_values["oracle"],
            "noop_score": 0.0,
            "noop_kl": 0.0,
            "projection_ref_score": scores["projection_ref"][1],
            "projection_ref_repair": scores["projection_ref"][0],
            "projection_ref_kl": kl_values["projection_ref"],
            "column_ref_score": scores["column_ref"][1],
            "column_ref_repair": scores["column_ref"][0],
            "column_ref_kl": kl_values["column_ref"],
            "random_score": scores["random"][1],
            "random_repair": scores["random"][0],
            "random_kl": kl_values["random"],
            "diagnostics": defect.operations,
            "scoring_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
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
        "rows": calibrate_v2(
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
