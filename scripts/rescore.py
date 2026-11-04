#!/usr/bin/env python3
"""Re-score existing benchmark result JSONs without rerunning episodes."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wsp.contracts import ScoreComponents, instance_score  # noqa: E402
from wsp.gym.scoring import KL_TAU  # noqa: E402


def smooth_penalty(kl: float, tau: float) -> float:
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not math.isfinite(kl):
        return 1.0
    return 1.0 - math.exp(-max(0.0, kl) / tau)


def rescore_payload(payload: dict[str, Any], tau: float) -> dict[str, Any]:
    raw = payload.get("components")
    if not isinstance(raw, dict):
        raise ValueError("result has no components object")
    components = dict(raw)
    components["collateral_penalty"] = smooth_penalty(
        float(components.get("collateral_kl", 0.0)),
        tau,
    )
    allowed = {item.name for item in fields(ScoreComponents)}
    parsed = ScoreComponents(
        **{key: value for key, value in components.items() if key in allowed}
    )
    submission = (
        "report"
        if payload.get("family") == "audit"
        or (parsed.repair is None and parsed.localization is not None)
        else "edit"
    )
    result = dict(payload)
    result["original_score"] = float(payload.get("score", 0.0))
    result["components"] = components
    result["score"] = instance_score(parsed, submission)
    result["rescore"] = {
        "method": "exp(-KL/tau)",
        "tau": tau,
        "source_score": result["original_score"],
    }
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    labels: dict[str, str] = {}
    for row in rows:
        model = str(row.get("model", "unknown"))
        labels[model] = str(row.get("display_name", model))
        score = float(row["score"])
        groups[(model, "overall", "overall")].append(score)
        groups[(model, "family", str(row.get("family", "unknown")))].append(score)
        groups[(model, "tier", str(row.get("tier", "unknown")))].append(score)
    result = []
    for (model, dimension, value), scores in sorted(groups.items()):
        result.append(
            {
                "model": model,
                "display_name": labels[model],
                "dimension": dimension,
                "value": value,
                "count": len(scores),
                "mean": sum(scores) / len(scores),
            }
        )
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        by_model[str(row["model"])][str(row["instance_id"])] = float(
            row["score"]
        )
    common = (
        set.intersection(
            *(set(values) for values in by_model.values())
        )
        if by_model
        else set()
    )
    for model, values in sorted(by_model.items()):
        if common:
            scores = [values[instance] for instance in common]
            result.append(
                {
                    "model": model,
                    "display_name": labels[model],
                    "dimension": "common",
                    "value": "overall",
                    "count": len(scores),
                    "mean": sum(scores) / len(scores),
                }
            )
    return result


def rescore_tree(
    source: Path,
    output: Path,
    tau: float,
    families: set[str] | None = None,
) -> dict[str, Any]:
    if source.resolve() == output.resolve():
        raise ValueError("output must differ from source")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    files = 0
    for path in sorted(source.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not (
            isinstance(payload, dict)
            and isinstance(payload.get("components"), dict)
            and payload.get("instance_id")
            and payload.get("model")
        ):
            continue
        if families is not None and str(payload.get("family")) not in families:
            continue
        rescored = rescore_payload(payload, tau)
        files += 1
        rows[(str(rescored["model"]), str(rescored["instance_id"]))] = rescored
        _atomic_json(output / path.relative_to(source), rescored)
    if not rows:
        raise ValueError(f"no result JSONs found below {source}")
    summary = {
        "source": str(source),
        "output": str(output),
        "tau": tau,
        "families": sorted(families) if families is not None else None,
        "results": len(rows),
        "files": files,
        "aggregates": _aggregates(list(rows.values())),
    }
    _atomic_json(output / "rescore-summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=KL_TAU)
    parser.add_argument(
        "--families",
        help="comma-separated family allowlist",
    )
    args = parser.parse_args()
    families = (
        {value.strip() for value in args.families.split(",") if value.strip()}
        if args.families
        else None
    )
    summary = rescore_tree(args.source, args.output, args.tau, families)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
