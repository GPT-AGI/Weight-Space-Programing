#!/usr/bin/env python3
"""Build and calibrate the isolated W4 defect families on the pinned model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsp.contracts import PRIVATE_DIR
from wsp.gym.families import (
    FAMILY_NAMES,
    FAMILY_TIERS,
    build_family_instances,
    directional_writer_ranks,
    exact_inverse_edit,
    family_metrics,
    invariant_diagnostics,
    inverse_operation_reference,
    load_family_recipe,
    materialize_family,
    projection_reference,
    random_direction_edit,
    scan_comparison,
    scan_engine_weights,
    save_family_materialized,
    score_family_outputs,
    score_structural_family,
)
from wsp.gym.families.instances import PRIVATE_STAGING_DIR
from wsp.gym.reference import build_reference_base, save_reference_base
from wsp.host.model import ModelEngine, load_text_model


def family_ids() -> list[str]:
    return [
        f"dev-{family}-{tier}-01"
        for family in FAMILY_NAMES
        for tier in FAMILY_TIERS
    ]


def _generate(
    engine: ModelEngine,
    defect: dict[str, torch.Tensor],
    prompts: list[str],
    edit: dict[str, torch.Tensor] | None = None,
    *,
    max_new_tokens: int,
    reference: Any = None,
) -> list[str]:
    with engine.applied(defect, edit, reference=reference):
        return engine.generate(
            prompts,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            seed=0,
            raw=False,
        )


def calibrate_one(
    engine: ModelEngine,
    instance_id: str,
    *,
    include_kl: bool,
    max_new_tokens: int,
    strength_scale: float = 1.0,
) -> dict[str, Any]:
    recipe = load_family_recipe(instance_id, "dev")
    if strength_scale != 1.0:
        recipe = replace(
            recipe,
            strengths=[
                strength * strength_scale for strength in recipe.strengths
            ],
        )
    reference = build_reference_base(
        engine.model,
        instance_id=recipe.instance_id,
        seed=recipe.seed,
        excluded_layers=list(recipe.layers),
    )
    reference_directory = (
        PRIVATE_STAGING_DIR
        / "ground_truth"
        / "dev"
        / instance_id
        / "reference"
    )
    save_reference_base(reference, reference_directory)
    with engine.applied({}, reference=reference):
        defect = materialize_family(engine.model, engine.tokenizer, recipe)
    defect.recipe_hash = f"{defect.recipe_hash}:{reference.fingerprint}"
    cache = (
        PRIVATE_STAGING_DIR
        / "ground_truth"
        / "dev"
        / instance_id
        / "materialized"
    )
    save_family_materialized(defect, cache)
    projection = projection_reference(engine, defect, reference)
    structural = recipe.family in {"swap", "rotate"}
    if structural:
        projection = inverse_operation_reference(engine, defect, reference)
    random_edit = random_direction_edit(
        engine, defect, recipe.seed + 99, reference
    )
    oracle = exact_inverse_edit(engine, defect, reference)

    started = time.perf_counter()
    with engine.lock:
        clean = _generate(
            engine,
            {},
            recipe.hidden_prompts,
            max_new_tokens=max_new_tokens,
            reference=reference,
        )
        broken = _generate(
            engine,
            defect.deltas,
            recipe.hidden_prompts,
            max_new_tokens=max_new_tokens,
            reference=reference,
        )
        projected = _generate(
            engine,
            defect.deltas,
            recipe.hidden_prompts,
            projection,
            max_new_tokens=max_new_tokens,
            reference=reference,
        )
        random_outputs = _generate(
            engine,
            defect.deltas,
            recipe.hidden_prompts,
            random_edit,
            max_new_tokens=max_new_tokens,
            reference=reference,
        )
        oracle_outputs = _generate(
            engine,
            defect.deltas,
            recipe.hidden_prompts,
            oracle,
            max_new_tokens=max_new_tokens,
            reference=reference,
        )
        generation_seconds = time.perf_counter() - started
        if include_kl:
            kl_started = time.perf_counter()
            broken_kl = engine.mean_kl(
                recipe.collateral_texts,
                defect.deltas,
                {},
                reference=reference,
                max_tokens=256,
            )
            projection_kl = engine.mean_kl(
                recipe.collateral_texts,
                defect.deltas,
                projection,
                reference=reference,
                max_tokens=256,
            )
            random_kl = engine.mean_kl(
                recipe.collateral_texts,
                defect.deltas,
                random_edit,
                reference=reference,
                max_tokens=256,
            )
            kl_seconds = time.perf_counter() - kl_started
        else:
            broken_kl = projection_kl = random_kl = 0.0
            kl_seconds = 0.0

    clean_metrics = family_metrics(clean, recipe)
    broken_metrics = family_metrics(broken, recipe)
    if structural:
        reference_weights = {
            name: engine.broken_value(name, {}, reference)
            for name in defect.deltas
        }
        broken_weights = {
            name: engine.broken_value(name, defect.deltas, reference)
            for name in defect.deltas
        }
        noop = score_structural_family(
            reference_weights, broken_weights, {}, defect, broken_kl
        )
        projection_score = score_structural_family(
            reference_weights,
            broken_weights,
            projection,
            defect,
            projection_kl,
        )
        random_score = score_structural_family(
            reference_weights,
            broken_weights,
            random_edit,
            defect,
            random_kl,
        )
        oracle_score = score_structural_family(
            reference_weights, broken_weights, oracle, defect, 0.0
        )
    else:
        noop = score_family_outputs(
            recipe, clean, broken, broken, broken_kl
        )
        projection_score = score_family_outputs(
            recipe, clean, broken, projected, projection_kl
        )
        random_score = score_family_outputs(
            recipe, clean, broken, random_outputs, random_kl
        )
        oracle_score = score_family_outputs(
            recipe, clean, broken, oracle_outputs, 0.0
        )
    logprob_metrics: dict[str, float] = {}
    invariants: dict[str, float] = {}
    if structural:
        prompts = [case.prompt for case in recipe.causal_cases]
        completions = [case.completion for case in recipe.causal_cases]

        def mean_logprob(
            planted: dict[str, torch.Tensor],
            edit: dict[str, torch.Tensor] | None = None,
        ) -> float:
            with engine.applied(planted, edit, reference=reference):
                values = engine.logprobs(
                    prompts, completions, raw=False
                )
            return sum(float(value.mean()) for value in values) / len(values)

        with engine.lock:
            logprob_metrics = {
                "clean": mean_logprob({}),
                "broken": mean_logprob(defect.deltas),
                "reference": mean_logprob(defect.deltas, projection),
                "random": mean_logprob(defect.deltas, random_edit),
            }
        invariants = invariant_diagnostics(engine, defect, reference)
    changed = sum(
        int(torch.count_nonzero(delta).item())
        for delta in defect.deltas.values()
    )
    row = {
        "instance": instance_id,
        "clean_rate": clean_metrics["defect_rate"],
        "broken_rate": broken_metrics["defect_rate"],
        "broken_fluent_rate": broken_metrics["fluent_rate"],
        "clean_control_accuracy": clean_metrics.get("control_accuracy"),
        "broken_control_accuracy": broken_metrics.get("control_accuracy"),
        "broken_case_rates": broken_metrics.get("case_defect_rates"),
        "target_logprobs": logprob_metrics,
        "invariants": invariants,
        "operations": defect.operations,
        "oracle_score": oracle_score["score"],
        "oracle_kl": 0.0,
        "noop_score": noop["score"],
        "broken_kl": broken_kl,
        "projection_score": projection_score["score"],
        "projection_repair": projection_score["repair"],
        "projection_kl": projection_kl,
        "random_score": random_score["score"],
        "random_repair": random_score["repair"],
        "random_kl": random_kl,
        "params_changed": changed,
        "layers": defect.true_layers,
        "generation_seconds": generation_seconds,
        "kl_seconds": kl_seconds,
        "scoring_seconds_estimate": generation_seconds * 3.0 / 5.0
        + kl_seconds / 3.0,
        "samples": [
            {
                "prompt": prompt,
                "clean": clean[index],
                "broken": broken[index],
            }
            for index, prompt in enumerate(recipe.hidden_prompts[:2])
        ],
        "control_samples": (
            [
                {
                    "prompt": control.prompt,
                    "clean": clean[-len(recipe.controls) + index],
                    "broken": broken[-len(recipe.controls) + index],
                }
                for index, control in enumerate(recipe.controls)
            ]
            if recipe.controls
            else []
        ),
    }
    print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
    return row


def calibrate_weight_scans(
    engine: ModelEngine,
    instance_ids: list[str],
) -> list[dict[str, Any]]:
    results = []
    for instance_id in instance_ids:
        recipe = load_family_recipe(instance_id, "dev")
        reference = build_reference_base(
            engine.model,
            instance_id=recipe.instance_id,
            seed=recipe.seed,
            excluded_layers=list(recipe.layers),
        )
        with engine.applied({}, reference=reference):
            defect = materialize_family(
                engine.model, engine.tokenizer, recipe
            )
        clean_rows = scan_engine_weights(
            engine, reference=reference
        )
        broken_rows = scan_engine_weights(
            engine, defect.deltas, reference
        )
        row = {
            "instance": instance_id,
            "comparison": scan_comparison(
                clean_rows, broken_rows, defect.true_layers
            ),
            "directional_writers": directional_writer_ranks(
                engine, defect, reference
            ),
        }
        results.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="*", default=family_ids())
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-kl", action="store_true")
    parser.add_argument("--strength-scale", type=float, default=1.0)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--weight-scans-only", action="store_true")
    parser.add_argument("--test-config", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=PRIVATE_DIR / "staging" / "calibration-dev.json",
    )
    args = parser.parse_args()

    build_family_instances("dev", families=FAMILY_NAMES)
    if args.test_config is not None:
        build_family_instances(
            "test",
            test_config=args.test_config,
            families=FAMILY_NAMES,
        )
    if args.build_only:
        return

    os.environ.setdefault("HF_HOME", "/srv/wsp-cache/hf")
    model, tokenizer, _ = load_text_model(
        cache_dir=Path(os.environ["HF_HOME"])
    )
    engine = ModelEngine(model, tokenizer, batch_size=16)
    if args.weight_scans_only:
        result = {
            "rows": calibrate_weight_scans(engine, args.ids),
            "scanner": "W6 frozen weight-statistics replica",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        return
    rows = [
        calibrate_one(
            engine,
            instance_id,
            include_kl=not args.no_kl,
            max_new_tokens=args.max_new_tokens,
            strength_scale=args.strength_scale,
        )
        for instance_id in args.ids
    ]
    result = {
        "model": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
        "kl_free": 0.002,
        "kl_max": 0.080,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
