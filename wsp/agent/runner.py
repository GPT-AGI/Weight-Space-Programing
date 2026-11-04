"""Episode orchestration and concurrent benchmark runner."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from wsp.agent.clients import (
    FakeLLMClient,
    LLMClient,
    LLMReply,
    OfoxClient,
    UnixHostClient,
    load_model_specs,
    usage_cost,
)
from wsp.agent.cursor_track import make_cursor_model_spec, run_cursor_agent
from wsp.agent.loop import AgentLoop, AgentOutcome, GlobalCostExceeded
from wsp.agent.prompts import load_sdk_reference
from wsp.contracts import (
    ADMIN_SOCKET,
    AGENT_SOCKET,
    HARNESS_VERSION,
    PRIVATE_DIR,
    PUBLIC_INSTANCES_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    RUNS_DIR,
    Instance,
    ModelSpec,
    RunResult,
    ScoreComponents,
    Split,
    TokenUsage,
    Turn,
    instance_score,
    model_slug,
)
from wsp.sandbox import SandboxConfig, SandboxKernel

LLMFactory = Callable[[ModelSpec], LLMClient]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _invalid_components(notes: str) -> ScoreComponents:
    return ScoreComponents(
        valid=False,
        repair=None,
        collateral_kl=0.0,
        collateral_penalty=0.0,
        localization=None,
        trigger_found=None,
        params_changed=0,
        notes=notes,
    )


def _components(payload: dict, instance: Instance) -> ScoreComponents:
    raw = payload.get("components", payload)
    if not isinstance(raw, dict):
        return _invalid_components("host returned malformed score components")
    score_hint = payload.get("score")
    repair = raw.get("repair")
    localization = raw.get("localization")
    if isinstance(score_hint, (int, float)):
        if instance.submission == "report" and localization is None:
            localization = float(score_hint) / 100.0
        if instance.submission == "edit" and repair is None:
            repair = float(score_hint) / 100.0
    return ScoreComponents(
        valid=bool(raw.get("valid", True)),
        repair=repair,
        collateral_kl=float(raw.get("collateral_kl", 0.0)),
        collateral_penalty=float(raw.get("collateral_penalty", 0.0)),
        localization=localization,
        trigger_found=raw.get("trigger_found"),
        params_changed=int(raw.get("params_changed", 0)),
        notes=str(raw.get("notes", "")),
    )


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_trace(path: Path, turns: Iterable[Turn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for turn in turns:
            handle.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
    temporary.replace(path)


class GlobalCostCap:
    """Serializes paid calls so concurrency cannot race past the cap."""

    def __init__(self, maximum: float | None, initial: float = 0.0) -> None:
        self.maximum = maximum
        self.spent = initial
        self._lock = threading.Lock()

    def gate(self, spec: ModelSpec) -> Callable[[Callable[[], LLMReply]], LLMReply]:
        def call_with_cap(call: Callable[[], LLMReply]) -> LLMReply:
            if (
                spec.usd_per_m_input == 0
                and spec.usd_per_m_output == 0
                and (spec.usd_per_m_cached or 0) == 0
            ):
                return call()
            with self._lock:
                if self.maximum is not None and self.spent >= self.maximum:
                    raise GlobalCostExceeded("global dollar cap reached")
                reply = call()
                self.spent += usage_cost(reply.usage, spec)
                return reply

        return call_with_cap


def _episode_dirs(opened: dict) -> tuple[Path, Path]:
    nested = opened.get("paths", {})
    model = (
        opened.get("model_dir")
        or opened.get("model")
        or nested.get("model")
    )
    work = (
        opened.get("work_dir")
        or opened.get("work")
        or nested.get("work")
    )
    episode = opened.get("episode_dir") or opened.get("episode")
    if (not model or not work) and episode:
        model = model or str(Path(episode) / "model")
        work = work or str(Path(episode) / "work")
    if not model or not work:
        raise RuntimeError("/admin/open did not return model and work paths")
    return Path(model), Path(work)


def _baseline_result(
    *,
    admin: UnixHostClient,
    instance: Instance,
    model: ModelSpec,
    kind: str,
    started_at: str,
    started: float,
) -> RunResult:
    payload = admin.post(
        "/admin/baseline",
        instance_id=instance.id,
        split=instance.split,
        kind=kind,
    )
    components = _components(payload, instance)
    score = float(payload.get("score", instance_score(components, instance.submission)))
    return RunResult(
        model=model.id,
        display_name=model.display_name,
        instance_id=instance.id,
        split=instance.split,
        family=instance.family,
        tier=instance.tier,
        score=score,
        components=components,
        cost_usd=0.0,
        tokens=TokenUsage(),
        turns=0,
        gpu_calls=0,
        wall_s=time.monotonic() - started,
        stop_reason="submitted",
        harness_version=HARNESS_VERSION,
        started_at=started_at,
        trace="",
        inference=dict(model.extra_body),
    )


def run_episode(
    *,
    instance: Instance,
    model: ModelSpec,
    llm_factory: LLMFactory,
    sandbox_env: Path,
    admin_socket: Path,
    agent_socket: Path,
    results_root: Path = RESULTS_DIR,
    runs_root: Path = RUNS_DIR,
    sdk_reference: str | None = None,
    global_cap: GlobalCostCap | None = None,
    track: Literal["raw", "cursor"] = "raw",
    cursor_sdk_env: Path | None = None,
    cursor_timeout_s: float = 1_800.0,
    cursor_network: Literal["proxy", "host"] = "host",
) -> RunResult:
    started_at = _utc_now()
    started = time.monotonic()
    admin = UnixHostClient(admin_socket)
    result_path = (
        results_root / instance.split / model_slug(model.id) / f"{instance.id}.json"
    )
    trace_path = (
        runs_root
        / instance.split
        / model_slug(model.id)
        / instance.id
        / "trace.jsonl"
    )
    session: str | None = None
    try:
        if model.id in {"oracle", "noop"}:
            result = _baseline_result(
                admin=admin,
                instance=instance,
                model=model,
                kind=model.id,
                started_at=started_at,
                started=started,
            )
            _atomic_json(result_path, result.to_json())
            return result

        session = secrets.token_hex(16)
        token = secrets.token_urlsafe(32)
        opened = admin.post(
            "/admin/open",
            instance_id=instance.id,
            split=instance.split,
            session=session,
            token=token,
        )
        model_dir, work_dir = _episode_dirs(opened)
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "wsp_notes.md").touch(exist_ok=True)
        trace: list[Turn] = []
        outcome = AgentOutcome(
            trace=trace,
            turns=0,
            gpu_calls=0,
            cost_usd=0.0,
            tokens=TokenUsage(),
            stop_reason="error",
            error="episode did not start",
        )
        score_payload: dict = {}
        try:
            config = SandboxConfig(
                model_dir=model_dir,
                work_dir=work_dir,
                agent_socket=agent_socket,
                session=session,
                token=token,
                sandbox_env=sandbox_env,
            )
            with SandboxKernel(config) as sandbox:
                reference = (
                    sdk_reference
                    if sdk_reference is not None
                    else load_sdk_reference()
                )
                if track == "cursor":
                    if cursor_sdk_env is None:
                        raise RuntimeError(
                            "cursor_sdk_env is required for the Cursor track"
                        )
                    outcome = run_cursor_agent(
                        instance=instance,
                        model=model,
                        sandbox=sandbox,
                        work_dir=work_dir,
                        sdk_env=cursor_sdk_env,
                        sdk_reference=reference,
                        timeout_s=cursor_timeout_s,
                        network=cursor_network,
                    )
                else:
                    if model.id in {"scripted", "redteam"}:
                        baseline_name = (
                            "redteam_agent.py"
                            if model.id == "redteam"
                            else "scripted_agent.py"
                        )
                        source = (
                            REPO_ROOT
                            / "wsp"
                            / "gym"
                            / "baselines"
                            / baseline_name
                        ).read_text(encoding="utf-8")
                        if model.id == "redteam":
                            source = (
                                f"WSP_INSTANCE_BRIEF = {instance.brief!r}\n"
                                + source
                            )
                        replies = [
                            LLMReply(
                                content=f"```python\n{source}\n```",
                                reasoning=None,
                                usage=TokenUsage(),
                                latency_s=0.0,
                            )
                        ]
                        llm: LLMClient = FakeLLMClient(replies)
                    else:
                        llm = llm_factory(model)
                    loop = AgentLoop(
                        instance=instance,
                        model=model,
                        llm=llm,
                        sandbox=sandbox,
                        sdk_reference=reference,
                        completion_gate=(
                            global_cap.gate(model)
                            if global_cap is not None
                            else None
                        ),
                    )
                    outcome = loop.run()
                trace = outcome.trace
        except Exception as exc:
            outcome.error = f"{type(exc).__name__}: {exc}"
            outcome.stop_reason = "error"
        finally:
            try:
                score_payload = admin.post("/admin/score", session=session)
            except Exception as exc:
                if not outcome.error:
                    outcome.error = f"score failed: {type(exc).__name__}: {exc}"
                    outcome.stop_reason = "error"
                score_payload = {
                    "components": asdict(
                        _invalid_components("host scoring request failed")
                    )
                }
            finally:
                try:
                    admin.post("/admin/close", session=session)
                except Exception:
                    pass

        components = _components(score_payload, instance)
        score = float(
            score_payload.get(
                "score", instance_score(components, instance.submission)
            )
        )
        trace_value = ""
        if trace:
            _atomic_trace(trace_path, trace)
            trace_value = trace_path.relative_to(runs_root).as_posix()
        result = RunResult(
            model=model.id,
            display_name=model.display_name,
            instance_id=instance.id,
            split=instance.split,
            family=instance.family,
            tier=instance.tier,
            score=score,
            components=components,
            cost_usd=outcome.cost_usd,
            tokens=outcome.tokens,
            turns=outcome.turns,
            gpu_calls=outcome.gpu_calls,
            wall_s=time.monotonic() - started,
            stop_reason=outcome.stop_reason,
            harness_version=HARNESS_VERSION,
            started_at=started_at,
            trace=trace_value,
            error=outcome.error,
            inference={**model.extra_body, **outcome.inference},
        )
        _atomic_json(result_path, result.to_json())
        return result
    except Exception as exc:
        if session is not None:
            try:
                admin.post("/admin/close", session=session)
            except Exception:
                pass
        result = RunResult(
            model=model.id,
            display_name=model.display_name,
            instance_id=instance.id,
            split=instance.split,
            family=instance.family,
            tier=instance.tier,
            score=0.0,
            components=_invalid_components("episode orchestration failed"),
            cost_usd=0.0,
            tokens=TokenUsage(),
            turns=0,
            gpu_calls=0,
            wall_s=time.monotonic() - started,
            stop_reason="error",
            harness_version=HARNESS_VERSION,
            started_at=started_at,
            trace="",
            error=f"{type(exc).__name__}: {exc}",
            inference=dict(model.extra_body),
        )
        _atomic_json(result_path, result.to_json())
        return result
    finally:
        admin.close()


def load_instances(split: Split, selection: str) -> list[Instance]:
    root = (
        PUBLIC_INSTANCES_DIR / split
        if split == "dev"
        else PRIVATE_DIR / "instances" / "test"
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
            raise ValueError(f"unknown instance ids: {', '.join(sorted(missing))}")
    return instances


def select_models(
    specs: dict[str, ModelSpec], selection: str
) -> list[ModelSpec]:
    requested = list(specs) if selection == "all" else selection.split(",")
    missing = [model_id for model_id in requested if model_id not in specs]
    if missing:
        raise ValueError(f"unknown model ids: {', '.join(missing)}")
    return [specs[model_id] for model_id in requested]


def _print_summary(results: Sequence[RunResult]) -> None:
    columns = ("model", "instance", "score", "cost", "turns", "stop")
    rows = [
        (
            result.display_name,
            result.instance_id,
            f"{result.score:.1f}",
            f"${result.cost_usd:.4f}",
            str(result.turns),
            result.stop_reason,
        )
        for result in results
    ]
    widths = [
        max(len(columns[index]), *(len(row[index]) for row in rows))
        for index in range(len(columns))
    ] if rows else [len(value) for value in columns]
    print("  ".join(value.ljust(widths[i]) for i, value in enumerate(columns)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run WSP-Bench episodes")
    parser.add_argument("--track", choices=("raw", "cursor"), default="raw")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--models", default="all")
    parser.add_argument("--instances", default="all")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--fake-llm", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-usd-total", type=float)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument(
        "--cursor-sdk-env",
        type=Path,
        default=REPO_ROOT / ".venv",
    )
    parser.add_argument("--cursor-timeout", type=float, default=1_800.0)
    parser.add_argument(
        "--cursor-network",
        choices=("proxy", "host"),
        default="host",
        help=(
            "host is the working fallback; proxy is fail-closed and "
            "Cursor SDK 1.2 currently does not honor it"
        ),
    )
    parser.add_argument(
        "--cursor-model-params",
        default="{}",
        help="JSON object of Cursor model parameter ids to values",
    )
    parser.add_argument(
        "--sandbox-env",
        type=Path,
        default=Path(os.environ.get("WSP_SANDBOX_ENV", "/opt/wsp-sandbox")),
    )
    args = parser.parse_args(argv)

    if args.track == "cursor":
        if args.models == "all":
            parser.error(
                "--track cursor requires explicit cursor-agent/<model> ids"
            )
        try:
            raw_params = json.loads(args.cursor_model_params)
        except json.JSONDecodeError as exc:
            parser.error(f"invalid --cursor-model-params JSON: {exc}")
        if not isinstance(raw_params, dict):
            parser.error("--cursor-model-params must be a JSON object")
        params = [
            {"id": str(key), "value": str(value)}
            for key, value in raw_params.items()
        ]
        models = [
            make_cursor_model_spec(
                model_id,
                params=params,
                network=args.cursor_network,
            )
            for model_id in args.models.split(",")
        ]
    else:
        specs = load_model_specs(REPO_ROOT / "configs" / "models.yaml")
        models = select_models(specs, args.models)
    instances = load_instances(args.split, args.instances)
    jobs: list[tuple[Instance, ModelSpec]] = []
    results: list[RunResult] = []
    initial_cost = 0.0
    for instance in instances:
        for model in models:
            result_path = (
                args.results_dir
                / instance.split
                / model_slug(model.id)
                / f"{instance.id}.json"
            )
            if args.resume and result_path.is_file():
                data = json.loads(result_path.read_text(encoding="utf-8"))
                if data.get("stop_reason") != "error":
                    initial_cost += float(data.get("cost_usd", 0.0))
                    continue
            jobs.append((instance, model))

    cap = GlobalCostCap(args.max_usd_total, initial=initial_cost)

    def llm_factory(spec: ModelSpec) -> LLMClient:
        if args.fake_llm:
            return FakeLLMClient.from_file(args.fake_llm)
        return OfoxClient(spec)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as executor:
        futures = [
            executor.submit(
                run_episode,
                instance=instance,
                model=model,
                llm_factory=llm_factory,
                sandbox_env=args.sandbox_env,
                admin_socket=ADMIN_SOCKET,
                agent_socket=AGENT_SOCKET,
                results_root=args.results_dir,
                runs_root=args.runs_dir,
                global_cap=cap,
                track=args.track,
                cursor_sdk_env=args.cursor_sdk_env,
                cursor_timeout_s=args.cursor_timeout,
                cursor_network=args.cursor_network,
            )
            for instance, model in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: (result.model, result.instance_id))
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
