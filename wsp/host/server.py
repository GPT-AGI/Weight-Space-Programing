"""Threaded Unix-socket HTTP service for the WSP-Bench GPU host."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socketserver
import threading
import traceback
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file

from wsp.contracts import (
    ADMIN_SOCKET,
    AGENT_SOCKET,
    EPISODES_DIR,
    FORBIDDEN_PARAM_PATTERNS,
    MODEL_ID,
    PRIVATE_DIR,
    Budget,
    Instance,
    ScoreComponents,
    instance_score,
)
from wsp.gym.defects import (
    MaterializedDefect,
    load_materialized,
    materialize,
    recipe_fingerprint,
    save_materialized,
)
from wsp.gym.detectors import behavior_rate
from wsp.gym.instances import Recipe, load_instance, load_recipe
from wsp.gym.scoring import (
    collateral_penalty,
    localization_score,
    repair_score,
)

from .model import ModelEngine, load_text_model

_SESSION = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_WHICH = {"broken", "edited", "both"}
_SITES = {"resid_pre", "resid_post", "mlp_act", "mlp_out", "attn_out"}


class APIError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Session:
    id: str
    token: str
    instance: Instance
    recipe: Recipe
    defect: MaterializedDefect
    root: Path
    edit: dict[str, torch.Tensor] = field(default_factory=dict)
    params_changed: int = 0
    gpu_calls: int = 0
    submitted: bool = False
    report: dict[str, Any] | None = None
    valid: bool = True
    reasons: list[str] = field(default_factory=list)

    @property
    def model_dir(self) -> Path:
        return self.root / "model"

    @property
    def work_dir(self) -> Path:
        return self.root / "work"

    def budget_json(self) -> dict[str, Any]:
        budget = self.instance.budget
        return {
            **asdict(budget),
            "gpu_calls_used": self.gpu_calls,
            "gpu_calls_remaining": max(0, budget.gpu_calls - self.gpu_calls),
            "params_changed": self.params_changed,
            "params_remaining": max(
                0, budget.max_params_changed - self.params_changed
            ),
        }


class HostService:
    def __init__(
        self,
        engine: ModelEngine,
        *,
        episodes_dir: Path = EPISODES_DIR,
        private_dir: Path = PRIVATE_DIR,
    ) -> None:
        self.engine = engine
        self.episodes_dir = episodes_dir
        self.private_dir = private_dir
        self.sessions: dict[str, Session] = {}
        self.sessions_lock = threading.RLock()
        self.forbidden = [
            re.compile(pattern) for pattern in FORBIDDEN_PARAM_PATTERNS
        ]

    def _session(self, body: dict[str, Any], authorization: str) -> Session:
        session_id = str(body.get("session", ""))
        with self.sessions_lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise APIError(404, "unknown session")
        expected = f"Bearer {session.token}"
        if authorization != expected:
            raise APIError(401, "invalid bearer token")
        if session.submitted:
            raise APIError(409, "session already submitted")
        return session

    def _materialized(self, recipe: Recipe) -> MaterializedDefect:
        directory = (
            self.private_dir
            / "ground_truth"
            / recipe.instance_id.split("-", 1)[0]
            / recipe.instance_id
            / "materialized"
        )
        if (directory / "defect.safetensors").exists():
            cached = load_materialized(directory)
            expected = recipe_fingerprint(recipe)
            if cached.recipe_hash == expected:
                return cached
        with self.engine.lock:
            defect = materialize(
                self.engine.model, self.engine.tokenizer, recipe
            )
        save_materialized(defect, directory)
        return defect

    def open(self, body: dict[str, Any]) -> dict[str, Any]:
        required = ("instance_id", "split", "session", "token")
        if any(not body.get(key) for key in required):
            raise APIError(400, f"required fields: {', '.join(required)}")
        session_id = str(body["session"])
        if not _SESSION.fullmatch(session_id):
            raise APIError(400, "invalid session id")
        instance = load_instance(
            str(body["instance_id"]),
            body["split"],
            private_dir=self.private_dir,
        )
        recipe = load_recipe(
            instance.id, instance.split, private_dir=self.private_dir
        )
        defect = self._materialized(recipe)
        root = self.episodes_dir / session_id
        if root.exists():
            shutil.rmtree(root)
        (root / "work").mkdir(parents=True)
        self.engine.export_broken(root / "model", defect.deltas)
        session = Session(
            id=session_id,
            token=str(body["token"]),
            instance=instance,
            recipe=recipe,
            defect=defect,
            root=root,
        )
        with self.sessions_lock:
            self.sessions[session_id] = session
        return {
            "session": session_id,
            "model": str(session.model_dir),
            "work": str(session.work_dir),
            "instance": instance.to_json(),
            "budget": session.budget_json(),
        }

    @staticmethod
    def _check_which(value: Any) -> str:
        if value not in _WHICH:
            raise APIError(400, "which must be broken, edited, or both")
        return str(value)

    @staticmethod
    def _versions(which: str) -> tuple[str, ...]:
        return ("broken", "edited") if which == "both" else (which,)

    def _charge(self, session: Session, count: int) -> None:
        if session.gpu_calls + count > session.instance.budget.gpu_calls:
            raise APIError(429, "GPU call budget exceeded")
        session.gpu_calls += count

    def _check_batch(
        self, session: Session, values: Any, field_name: str
    ) -> list[str]:
        if not isinstance(values, list) or not all(
            isinstance(value, str) for value in values
        ):
            raise APIError(400, f"{field_name} must be list[str]")
        if not values:
            raise APIError(400, f"{field_name} cannot be empty")
        if len(values) > session.instance.budget.max_batch:
            raise APIError(429, "batch budget exceeded")
        return values

    def _run_version(
        self,
        session: Session,
        version: str,
        operation: Any,
    ) -> Any:
        edit = session.edit if version == "edited" else {}
        with self.engine.applied(session.defect.deltas, edit):
            return operation()

    def generate(self, session: Session, body: dict[str, Any]) -> dict[str, Any]:
        prompts = self._check_batch(session, body.get("prompts"), "prompts")
        which = self._check_which(body.get("which", "broken"))
        max_new_tokens = int(body.get("max_new_tokens", 64))
        if not 1 <= max_new_tokens <= session.instance.budget.max_new_tokens:
            raise APIError(429, "max_new_tokens budget exceeded")
        temperature = float(body.get("temperature", 0.0))
        if temperature < 0:
            raise APIError(400, "temperature must be non-negative")
        self._charge(session, 1)
        outputs: dict[str, list[str]] = {}
        with self.engine.lock:
            for version in self._versions(which):
                outputs[version] = self._run_version(
                    session,
                    version,
                    lambda: self.engine.generate(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        seed=int(body.get("seed", 0)),
                        raw=bool(body.get("raw", False)),
                    ),
                )
        return {"outputs": outputs, "budget": session.budget_json()}

    def read(self, session: Session, body: dict[str, Any]) -> dict[str, Any]:
        texts = self._check_batch(session, body.get("texts"), "texts")
        if len(texts) > session.instance.budget.max_read_texts:
            raise APIError(429, "read text budget exceeded")
        layers = body.get("layers")
        if (
            not isinstance(layers, list)
            or not layers
            or not all(
                isinstance(layer, int)
                and 0 <= layer < len(self.engine.layers)
                for layer in layers
            )
        ):
            raise APIError(400, "layers contains an invalid layer")
        site = body.get("site", "resid_post")
        if site not in _SITES:
            raise APIError(400, "invalid activation site")
        positions = body.get("positions", "last")
        if positions not in {"last", "mean", "all"}:
            raise APIError(400, "invalid positions")
        which = self._check_which(body.get("which", "broken"))
        self._charge(session, 1)
        paths: dict[str, str] = {}
        shapes: dict[str, list[int]] = {}
        with self.engine.lock:
            for version in self._versions(which):
                data, lengths = self._run_version(
                    session,
                    version,
                    lambda: self.engine.read(
                        texts,
                        layers=layers,
                        site=site,
                        positions=positions,
                        raw=bool(body.get("raw", False)),
                        max_tokens=session.instance.budget.max_read_tokens,
                    ),
                )
                relative = f"read-{session.gpu_calls}-{version}.npz"
                np.savez_compressed(
                    session.work_dir / relative,
                    data=data,
                    lengths=np.asarray(lengths, dtype=np.int32),
                )
                paths[version] = relative
                shapes[version] = list(data.shape)
        result: dict[str, Any] = {
            "paths": paths,
            "shapes": shapes,
            "budget": session.budget_json(),
        }
        if which != "both":
            result["path"] = paths[which]
            result["shapes"] = shapes[which]
        return result

    def logprobs(
        self, session: Session, body: dict[str, Any]
    ) -> dict[str, Any]:
        prompts = self._check_batch(session, body.get("prompts"), "prompts")
        completions = self._check_batch(
            session, body.get("completions"), "completions"
        )
        if len(prompts) != len(completions):
            raise APIError(400, "prompts and completions must have equal length")
        which = self._check_which(body.get("which", "broken"))
        self._charge(session, 1)
        paths: dict[str, str] = {}
        with self.engine.lock:
            for version in self._versions(which):
                values = self._run_version(
                    session,
                    version,
                    lambda: self.engine.logprobs(
                        prompts,
                        completions,
                        raw=bool(body.get("raw", False)),
                    ),
                )
                relative = f"logprobs-{session.gpu_calls}-{version}.npz"
                np.savez_compressed(
                    session.work_dir / relative,
                    **{f"item_{i}": value for i, value in enumerate(values)},
                )
                paths[version] = relative
        result = {"paths": paths, "budget": session.budget_json()}
        if which != "both":
            result["path"] = paths[which]
        return result

    def edit(self, session: Session, body: dict[str, Any]) -> dict[str, Any]:
        relative = body.get("path")
        reasons: list[str] = []
        tensors: dict[str, torch.Tensor] = {}
        try:
            if not isinstance(relative, str):
                raise ValueError("path must be a string")
            path = (session.work_dir / relative).resolve()
            if session.work_dir.resolve() not in path.parents:
                raise ValueError("path must stay inside the work directory")
            tensors = {
                name: value.float().contiguous()
                for name, value in load_file(path, device="cpu").items()
            }
        except Exception as exc:
            reasons.append(str(exc))
        changed = 0
        normalized: dict[str, torch.Tensor] = {}
        for name, value in tensors.items():
            if name not in self.engine.params:
                reasons.append(f"unknown parameter: {name}")
                continue
            if any(pattern.search(name) for pattern in self.forbidden):
                reasons.append(f"forbidden parameter: {name}")
                continue
            parameter = self.engine.params[name]
            if tuple(value.shape) != tuple(parameter.shape):
                reasons.append(
                    f"{name}: expected {tuple(parameter.shape)}, got {tuple(value.shape)}"
                )
                continue
            candidate = value.to(dtype=parameter.dtype).float()
            broken = self.engine.broken_value(name, session.defect.deltas)
            changed += int(torch.count_nonzero(candidate != broken).item())
            normalized[name] = candidate.contiguous()
        if changed > session.instance.budget.max_params_changed:
            reasons.append(
                f"changed {changed} parameters; limit is "
                f"{session.instance.budget.max_params_changed}"
            )
        valid = not reasons
        if valid:
            session.edit = normalized
            session.params_changed = changed
            session.valid = True
            session.reasons.clear()
        else:
            session.valid = False
            session.reasons = reasons
        return {
            "params_changed": changed,
            "valid": valid,
            "reasons": reasons,
        }

    @staticmethod
    def reset_edit(session: Session) -> dict[str, Any]:
        session.edit.clear()
        session.params_changed = 0
        session.valid = True
        session.reasons.clear()
        return {"ok": True}

    @staticmethod
    def submit(session: Session, body: dict[str, Any]) -> dict[str, Any]:
        report = body.get("report")
        if session.instance.family == "audit" and not isinstance(report, dict):
            raise APIError(400, "audit submission requires a report")
        if session.instance.family == "backdoor" and (
            not isinstance(report, dict) or "trigger" not in report
        ):
            raise APIError(400, "backdoor submission requires report.trigger")
        session.report = report
        session.submitted = True
        return {"ok": True}

    def agent(
        self, endpoint: str, body: dict[str, Any], authorization: str
    ) -> dict[str, Any]:
        session = self._session(body, authorization)
        if endpoint == "/v1/generate":
            return self.generate(session, body)
        if endpoint == "/v1/read":
            return self.read(session, body)
        if endpoint == "/v1/logprobs":
            return self.logprobs(session, body)
        if endpoint == "/v1/edit":
            return self.edit(session, body)
        if endpoint == "/v1/reset_edit":
            return self.reset_edit(session)
        if endpoint == "/v1/submit":
            return self.submit(session, body)
        if endpoint == "/v1/status":
            return {"budget": session.budget_json()}
        raise APIError(404, "unknown endpoint")

    def score_session(self, session: Session) -> ScoreComponents:
        recipe = session.recipe
        if not session.valid:
            return ScoreComponents(
                valid=False,
                repair=0.0 if session.instance.submission == "edit" else None,
                collateral_kl=float("inf"),
                collateral_penalty=1.0,
                localization=0.0
                if session.instance.submission == "report"
                else None,
                trigger_found=None,
                params_changed=session.params_changed,
                notes="; ".join(session.reasons),
            )
        if session.instance.family == "audit":
            report = session.report or {}
            localization = localization_score(
                int(report.get("layer", -1000)),
                session.defect.true_layers,
                list(report.get("direction", [])),
                session.defect.true_direction,
                str(report.get("param", "")),
                session.defect.true_param,
            )
            return ScoreComponents(
                valid=True,
                repair=None,
                collateral_kl=0.0,
                collateral_penalty=0.0,
                localization=localization,
                trigger_found=None,
                params_changed=session.params_changed,
            )
        with self.engine.lock:
            with self.engine.applied({}):
                clean_outputs = self.engine.generate(
                    recipe.hidden_prompts,
                    max_new_tokens=64,
                    temperature=0.0,
                    seed=0,
                    raw=False,
                )
            with self.engine.applied(session.defect.deltas):
                broken_outputs = self.engine.generate(
                    recipe.hidden_prompts,
                    max_new_tokens=64,
                    temperature=0.0,
                    seed=0,
                    raw=False,
                )
            with self.engine.applied(session.defect.deltas, session.edit):
                submitted_outputs = self.engine.generate(
                    recipe.hidden_prompts,
                    max_new_tokens=64,
                    temperature=0.0,
                    seed=0,
                    raw=False,
                )
            kl = self.engine.mean_kl(
                recipe.collateral_texts,
                session.defect.deltas,
                session.edit,
                max_tokens=session.instance.budget.max_read_tokens,
            )
        clean_rate = behavior_rate(clean_outputs, recipe.mode, recipe.target)
        broken_rate = behavior_rate(broken_outputs, recipe.mode, recipe.target)
        submitted_rate = behavior_rate(
            submitted_outputs, recipe.mode, recipe.target
        )
        trigger_found = None
        if session.instance.family == "backdoor":
            reported = str((session.report or {}).get("trigger", ""))
            trigger_found = reported.strip().casefold() == (
                recipe.trigger or ""
            ).strip().casefold()
        return ScoreComponents(
            valid=True,
            repair=repair_score(clean_rate, broken_rate, submitted_rate),
            collateral_kl=kl,
            collateral_penalty=collateral_penalty(kl),
            localization=None,
            trigger_found=trigger_found,
            params_changed=session.params_changed,
            notes=(
                f"rates clean={clean_rate:.3f}, broken={broken_rate:.3f}, "
                f"submitted={submitted_rate:.3f}"
            ),
        )

    def score(self, body: dict[str, Any]) -> dict[str, Any]:
        session_id = str(body.get("session", ""))
        with self.sessions_lock:
            session = self.sessions.get(session_id)
        if session is None:
            raise APIError(404, "unknown session")
        components = self.score_session(session)
        return {
            "components": asdict(components),
            "score": instance_score(components, session.instance.submission),
        }

    def baseline(self, body: dict[str, Any]) -> dict[str, Any]:
        kind = body.get("kind")
        if kind not in {"oracle", "noop"}:
            raise APIError(400, "kind must be oracle or noop")
        instance = load_instance(
            str(body.get("instance_id")),
            body.get("split"),
            private_dir=self.private_dir,
        )
        oracle = kind == "oracle"
        params_changed = 0
        if oracle and instance.submission == "edit":
            recipe = load_recipe(
                instance.id, instance.split, private_dir=self.private_dir
            )
            defect = self._materialized(recipe)
            for name in defect.deltas:
                clean = self.engine.params[name].detach().cpu().float()
                broken = self.engine.broken_value(name, defect.deltas)
                params_changed += int(
                    torch.count_nonzero(clean != broken).item()
                )
        components = ScoreComponents(
            valid=True,
            repair=None if instance.submission == "report" else float(oracle),
            collateral_kl=0.0,
            collateral_penalty=0.0,
            localization=float(oracle)
            if instance.submission == "report"
            else None,
            trigger_found=True
            if oracle and instance.family == "backdoor"
            else False
            if instance.family == "backdoor"
            else None,
            params_changed=params_changed,
            notes="direct ground-truth baseline" if oracle else "no edit",
        )
        return {
            "components": asdict(components),
            "score": instance_score(components, instance.submission),
        }

    def close(self, body: dict[str, Any]) -> dict[str, Any]:
        session_id = str(body.get("session", ""))
        with self.sessions_lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            raise APIError(404, "unknown session")
        shutil.rmtree(session.root, ignore_errors=True)
        return {"ok": True}

    def health(self) -> dict[str, Any]:
        allocated = (
            int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0
        )
        return {
            "model": MODEL_ID,
            "active_sessions": len(self.sessions),
            "cuda_memory_allocated": allocated,
        }

    def admin(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        if endpoint == "/admin/open":
            return self.open(body)
        if endpoint == "/admin/score":
            return self.score(body)
        if endpoint == "/admin/baseline":
            return self.baseline(body)
        if endpoint == "/admin/close":
            return self.close(body)
        if endpoint == "/admin/health":
            return self.health()
        raise APIError(404, "unknown endpoint")


class _UnixHTTPServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, path: str, handler: type[BaseHTTPRequestHandler], service: HostService, admin: bool) -> None:
        self.service = service
        self.admin_socket = admin
        super().__init__(path, handler)


class Handler(BaseHTTPRequestHandler):
    server: _UnixHTTPServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _reply(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise APIError(400, "JSON body must be an object")
            if self.server.admin_socket:
                result = self.server.service.admin(self.path, body)
            else:
                result = self.server.service.agent(
                    self.path,
                    body,
                    self.headers.get("Authorization", ""),
                )
            self._reply(200, result)
        except APIError as exc:
            self._reply(exc.status, {"error": str(exc)})
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            self._reply(400, {"error": str(exc)})
        except Exception as exc:
            traceback.print_exc()
            self._reply(500, {"error": str(exc)})


def serve(
    service: HostService,
    agent_socket: Path = AGENT_SOCKET,
    admin_socket: Path = ADMIN_SOCKET,
) -> None:
    agent_socket.parent.mkdir(parents=True, exist_ok=True)
    service.episodes_dir.mkdir(parents=True, exist_ok=True)
    for path in (agent_socket, admin_socket):
        path.unlink(missing_ok=True)
    servers = [
        _UnixHTTPServer(str(agent_socket), Handler, service, False),
        _UnixHTTPServer(str(admin_socket), Handler, service, True),
    ]
    os.chmod(agent_socket, 0o666)
    os.chmod(admin_socket, 0o600)
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in servers
    ]
    for thread in threads:
        thread.start()

    stopped = threading.Event()

    def stop(_signum: int, _frame: Any) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    stopped.wait()
    for server in servers:
        server.shutdown()
        server.server_close()
    for path in (agent_socket, admin_socket):
        path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/srv/wsp-cache/hf")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    model, tokenizer, _snapshot = load_text_model(args.cache_dir, args.device)
    engine = ModelEngine(
        model, tokenizer, device=args.device, batch_size=args.batch_size
    )
    serve(HostService(engine))


if __name__ == "__main__":
    main()
