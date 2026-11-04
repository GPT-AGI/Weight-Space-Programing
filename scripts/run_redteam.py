#!/usr/bin/env python3
"""Run the fixed red-team participant without touching repo result directories."""

from __future__ import annotations

import argparse
import json
import re
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wsp.agent.clients import UnixHostClient  # noqa: E402
from wsp.agent.runner import _episode_dirs, run_episode  # noqa: E402
from wsp.contracts import (  # noqa: E402
    ADMIN_SOCKET,
    AGENT_SOCKET,
    Instance,
    ModelSpec,
)
from wsp.sandbox import SandboxConfig, SandboxKernel  # noqa: E402

AGENT_SOURCE = (
    REPO_ROOT / "wsp" / "gym" / "baselines" / "redteam_agent.py"
)
DEFAULT_SANDBOX = Path("/opt/wsp-sandbox")
RESULT_MARKER = re.compile(r"^REDTEAM_RESULT\s+(\{.*\})$", re.MULTILINE)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_instances(staging: bool, selection: str) -> list[Instance]:
    root = (
        REPO_ROOT / "bench" / "instances_staging" / "dev"
        if staging
        else REPO_ROOT / "bench" / "instances" / "dev"
    )
    requested = None if selection == "all" else set(selection.split(","))
    instances = [
        Instance.from_json(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(root.glob("*.json"))
        if requested is None or path.stem in requested
    ]
    if requested is not None:
        missing = requested - {instance.id for instance in instances}
        if missing:
            raise ValueError(f"unknown instances: {', '.join(sorted(missing))}")
    return instances


def extract_gpu_calls(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    for key in ("gpu_calls_used", "used_gpu_calls", "used"):
        item = value.get(key)
        if isinstance(item, (int, float)):
            return int(item)
    nested = value.get("gpu_calls")
    return extract_gpu_calls(nested)


def parse_diagnostic(stdout: str) -> dict[str, Any]:
    matches = RESULT_MARKER.findall(stdout)
    if not matches:
        return {}
    try:
        value = json.loads(matches[-1])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".casefold()
    return any(
        marker in text
        for marker in (
            "connection",
            "socket",
            "timed out",
            "timeout",
            "reset",
            "temporarily",
            "502",
            "503",
        )
    )


def run_redteam_once(
    instance: Instance,
    *,
    sandbox_env: Path,
) -> dict[str, Any]:
    admin = UnixHostClient(ADMIN_SOCKET)
    session = secrets.token_hex(16)
    token = secrets.token_urlsafe(32)
    started = time.monotonic()
    opened = None
    execution: dict[str, Any] = {}
    score: dict[str, Any] = {}
    try:
        opened = admin.post(
            "/admin/open",
            instance_id=instance.id,
            split=instance.split,
            session=session,
            token=token,
        )
        model_dir, work_dir = _episode_dirs(opened)
        work_dir.mkdir(parents=True, exist_ok=True)
        config = SandboxConfig(
            model_dir=model_dir,
            work_dir=work_dir,
            agent_socket=AGENT_SOCKET,
            session=session,
            token=token,
            sandbox_env=sandbox_env,
            exec_timeout_s=1_200.0,
        )
        source = AGENT_SOURCE.read_text(encoding="utf-8")
        program = f"WSP_INSTANCE_BRIEF = {instance.brief!r}\n" + source
        with SandboxKernel(config) as sandbox:
            execution = sandbox.execute(
                program,
                timeout_s=1_200.0,
                max_output_chars=500_000,
            )
        score = admin.post("/admin/score", session=session)
    finally:
        if opened is not None:
            try:
                admin.post("/admin/close", session=session)
            except Exception:
                pass
        admin.close()

    stdout = str(execution.get("stdout") or "")
    diagnostic = parse_diagnostic(stdout)
    components = score.get("components", score)
    if not isinstance(components, dict):
        components = {}
    return {
        "model": "redteam",
        "display_name": "Fixed red-team harness",
        "instance_id": instance.id,
        "family": instance.family,
        "tier": instance.tier,
        "score": float(score.get("score", 0.0)),
        "components": components,
        "gpu_calls": extract_gpu_calls(diagnostic.get("gpu_calls")),
        "wall_s": time.monotonic() - started,
        "stop_reason": (
            "submitted" if execution.get("submitted") else "error"
        ),
        "error": str(execution.get("traceback") or ""),
        "diagnostic": diagnostic,
        "stdout": stdout,
        "stderr": str(execution.get("stderr") or ""),
    }


def run_redteam_with_retry(
    instance: Instance,
    *,
    sandbox_env: Path,
    attempts: int = 4,
) -> dict[str, Any]:
    last: BaseException | None = None
    for attempt in range(attempts):
        try:
            return run_redteam_once(instance, sandbox_env=sandbox_env)
        except BaseException as exc:
            last = exc
            if attempt + 1 >= attempts or not transient(exc):
                break
            time.sleep(min(20.0, 2.0 ** attempt))
    return {
        "model": "redteam",
        "display_name": "Fixed red-team harness",
        "instance_id": instance.id,
        "family": instance.family,
        "tier": instance.tier,
        "score": 0.0,
        "components": {},
        "gpu_calls": 0,
        "wall_s": 0.0,
        "stop_reason": "error",
        "error": f"{type(last).__name__}: {last}" if last else "unknown error",
        "diagnostic": {},
        "stdout": "",
        "stderr": "",
    }


def run_scripted(
    instance: Instance,
    *,
    root: Path,
    sandbox_env: Path,
) -> dict[str, Any]:
    model = ModelSpec(
        id="scripted",
        display_name="Scripted baseline",
        usd_per_m_input=0.0,
        usd_per_m_output=0.0,
    )

    def unused_factory(_spec: ModelSpec):
        raise AssertionError("scripted baseline must not request an LLM")

    result = run_episode(
        instance=instance,
        model=model,
        llm_factory=unused_factory,
        sandbox_env=sandbox_env,
        admin_socket=ADMIN_SOCKET,
        agent_socket=AGENT_SOCKET,
        results_root=root / "results",
        runs_root=root / "runs",
    )
    return result.to_json()


def run_oracle(instance: Instance) -> dict[str, Any]:
    admin = UnixHostClient(ADMIN_SOCKET)
    try:
        payload = admin.post(
            "/admin/baseline",
            instance_id=instance.id,
            split=instance.split,
            kind="oracle",
        )
    finally:
        admin.close()
    return {
        "model": "oracle",
        "display_name": "Oracle",
        "instance_id": instance.id,
        "family": instance.family,
        "tier": instance.tier,
        "score": float(payload.get("score", 0.0)),
        "components": payload.get("components", payload),
        "gpu_calls": 0,
        "wall_s": 0.0,
        "stop_reason": "submitted",
        "error": "",
    }


def write_summary(root: Path, rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["instance_id"], {})[row["model"]] = row
    summary = []
    for instance_id in sorted(grouped):
        models = grouped[instance_id]
        redteam = models.get("redteam", {})
        scripted = models.get("scripted", {})
        oracle = models.get("oracle", {})
        summary.append({
            "instance": instance_id,
            "redteam": redteam.get("score"),
            "scripted": scripted.get("score"),
            "oracle": oracle.get("score"),
            "gpu_calls": redteam.get("gpu_calls"),
            "repair": redteam.get("diagnostic", {}).get(
                "repair",
                redteam.get("diagnostic", {}).get("localizer"),
            ),
            "layer": redteam.get("diagnostic", {}).get("layer"),
            "error": redteam.get("error", ""),
        })
    atomic_json(root / "summary.json", {"rows": summary})

    lines = [
        "| instance | red-team | scripted | oracle | GPU | shortcut |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        score = lambda value: "—" if value is None else f"{float(value):.1f}"
        lines.append(
            f"| {row['instance']} | {score(row['redteam'])} | "
            f"{score(row['scripted'])} | {score(row['oracle'])} | "
            f"{row['gpu_calls'] if row['gpu_calls'] is not None else '—'} | "
            f"{row['repair'] or '—'} |"
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed WSP red-team harness"
    )
    parser.add_argument("--instances", default="all")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--staging", action="store_true")
    parser.add_argument(
        "--compare",
        dest="compare",
        action="store_true",
        default=True,
        help="also measure the existing scripted and oracle baselines",
    )
    parser.add_argument(
        "--no-compare",
        dest="compare",
        action="store_false",
        help="run only the red-team harness",
    )
    parser.add_argument("--sandbox-env", type=Path, default=DEFAULT_SANDBOX)
    args = parser.parse_args(argv)
    if not 1 <= args.concurrency <= 2:
        parser.error("--concurrency must be 1 or 2")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = Path("/srv/wsp-scratch") / f"redteam-{timestamp}"
    root.mkdir(parents=True, exist_ok=False)
    instances = load_instances(args.staging, args.instances)
    rows: list[dict[str, Any]] = []

    def run_one(instance: Instance) -> list[dict[str, Any]]:
        values = [
            run_redteam_with_retry(
                instance,
                sandbox_env=args.sandbox_env,
            )
        ]
        if args.compare:
            try:
                values.append(
                    run_scripted(
                        instance,
                        root=root,
                        sandbox_env=args.sandbox_env,
                    )
                )
            except BaseException as exc:
                values.append({
                    "model": "scripted",
                    "instance_id": instance.id,
                    "score": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                })
            try:
                values.append(run_oracle(instance))
            except BaseException as exc:
                values.append({
                    "model": "oracle",
                    "instance_id": instance.id,
                    "score": 0.0,
                    "error": f"{type(exc).__name__}: {exc}",
                })
        return values

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_one, instance): instance
            for instance in instances
        }
        for future in as_completed(futures):
            values = future.result()
            rows.extend(values)
            for value in values:
                atomic_json(
                    root
                    / "raw"
                    / str(value.get("model", "unknown"))
                    / f"{value['instance_id']}.json",
                    value,
                )
            write_summary(root, rows)
            redteam = values[0]
            print(
                redteam["instance_id"],
                f"score={redteam.get('score', 0.0):.1f}",
                f"gpu={redteam.get('gpu_calls', 0)}",
                redteam.get("stop_reason", ""),
                flush=True,
            )
    rows.sort(key=lambda row: (row["instance_id"], row["model"]))
    write_summary(root, rows)
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
