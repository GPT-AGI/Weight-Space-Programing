"""Unix-socket mock of the WSP host API for runner and sandbox tests."""

from __future__ import annotations

import json
import shutil
import socketserver
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

_STUB_SDK = r'''
import json
import os
import socket
import types

config = {"num_layers": 1, "hidden_size": 2, "intermediate_size": 4,
          "layer_types": ["full_attention"], "param_names": []}
weights = {}
tokenizer = None
tools = types.SimpleNamespace()

def _post(endpoint, **values):
    body = json.dumps({"session": os.environ["WSP_SESSION"], **values}).encode()
    request = (
        f"POST {endpoint} HTTP/1.1\r\nHost: wsp\r\n"
        f"Authorization: Bearer {os.environ['WSP_TOKEN']}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode() + body
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.connect(os.environ.get("WSP_SOCKET", "/wsp/agent.sock"))
        sock.sendall(request)
        chunks = []
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    raw = b"".join(chunks)
    header, _, response_body = raw.partition(b"\r\n\r\n")
    status = int(header.split(b"\r\n", 1)[0].split()[1])
    payload = json.loads(response_body or b"{}")
    if status >= 400:
        raise RuntimeError(payload.get("error", "host request failed"))
    return payload

def budget():
    return _post("/v1/status")["budget"]

def generate(prompts, which="broken", max_new_tokens=64, temperature=0.0,
             seed=0, raw=False):
    result = _post("/v1/generate", prompts=prompts, which=which,
                   max_new_tokens=max_new_tokens, temperature=temperature,
                   seed=seed, raw=raw)["outputs"]
    return result if which == "both" else result[which]

def submit(report=None):
    return _post("/v1/submit", report=report)

def set_param(name, tensor):
    return None

def reset_edit(name=None):
    return _post("/v1/reset_edit")

def edit_summary():
    return {"params": [], "params_changed": 0}
'''


def build_stub_sandbox_env(root: Path) -> Path:
    """Create a tiny test-only environment; no benchmark code is installed."""
    environment = root / "sandbox-env"
    bin_dir = environment / "bin"
    site = (
        environment
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    bin_dir.mkdir(parents=True)
    site.mkdir(parents=True)
    (bin_dir / "python").symlink_to(Path(sys.executable).resolve())
    (site / "wsp_sdk.py").write_text(_STUB_SDK, encoding="utf-8")
    return environment


@dataclass
class _Session:
    token: str
    model_dir: Path
    work_dir: Path
    gpu_calls: int = 0
    submitted: bool = False
    edited: bool = False


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    @property
    def mock(self) -> MockHost:
        return self.server.mock  # type: ignore[attr-defined]

    @property
    def kind(self) -> str:
        return self.server.kind  # type: ignore[attr-defined]

    def _json(self, status: int, value: dict[str, Any]) -> None:
        body = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            status, response = self.mock.dispatch(
                self.kind, self.path, payload, self.headers.get("Authorization")
            )
        except Exception as exc:
            status, response = 500, {"error": f"{type(exc).__name__}: {exc}"}
        self._json(status, response)


class MockHost:
    def __init__(
        self,
        root: Path,
        *,
        max_gpu_calls: int = 20,
        repair: float = 0.8,
        collateral_penalty: float = 0.1,
    ) -> None:
        self.root = root
        self.admin_socket = root / "admin.sock"
        self.agent_socket = root / "agent.sock"
        self.episodes = root / "episodes"
        self.max_gpu_calls = max_gpu_calls
        self.repair = repair
        self.collateral_penalty = collateral_penalty
        self.sessions: dict[str, _Session] = {}
        self.open_count = 0
        self.score_count = 0
        self._lock = threading.Lock()
        self._servers: list[_ThreadingUnixServer] = []
        self._threads: list[threading.Thread] = []

    def __enter__(self) -> MockHost:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes.mkdir(parents=True, exist_ok=True)
        for path, kind in (
            (self.admin_socket, "admin"),
            (self.agent_socket, "agent"),
        ):
            path.unlink(missing_ok=True)
            server = _ThreadingUnixServer(str(path), _Handler)
            server.mock = self  # type: ignore[attr-defined]
            server.kind = kind  # type: ignore[attr-defined]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            self._servers.append(server)
            self._threads.append(thread)

    def stop(self) -> None:
        for server in self._servers:
            server.shutdown()
            server.server_close()
        for thread in self._threads:
            thread.join(timeout=2)
        self._servers.clear()
        self._threads.clear()
        self.admin_socket.unlink(missing_ok=True)
        self.agent_socket.unlink(missing_ok=True)

    def _budget(self, session: _Session) -> dict[str, int]:
        return {
            "gpu_calls_used": session.gpu_calls,
            "gpu_calls_remaining": max(
                0, self.max_gpu_calls - session.gpu_calls
            ),
            "gpu_calls": self.max_gpu_calls,
        }

    def _authorized(
        self, payload: dict[str, Any], authorization: str | None
    ) -> tuple[_Session | None, tuple[int, dict[str, Any]] | None]:
        session = self.sessions.get(str(payload.get("session", "")))
        if session is None:
            return None, (404, {"error": "unknown session"})
        if authorization != f"Bearer {session.token}":
            return None, (401, {"error": "invalid token"})
        if session.submitted:
            return None, (409, {"error": "session already submitted"})
        return session, None

    def dispatch(
        self,
        kind: str,
        endpoint: str,
        payload: dict[str, Any],
        authorization: str | None,
    ) -> tuple[int, dict[str, Any]]:
        if kind == "admin":
            return self._dispatch_admin(endpoint, payload)
        session, error = self._authorized(payload, authorization)
        if error is not None:
            return error
        assert session is not None
        if endpoint == "/v1/status":
            return 200, {"budget": self._budget(session)}
        if endpoint in {"/v1/generate", "/v1/logprobs", "/v1/read"}:
            if session.gpu_calls >= self.max_gpu_calls:
                return 429, {"error": "GPU call budget exhausted"}
            session.gpu_calls += 1
            budget = self._budget(session)
            if endpoint == "/v1/generate":
                prompts = payload.get("prompts", [])
                outputs = [f"mock output {index}" for index, _ in enumerate(prompts)]
                which = payload.get("which", "broken")
                value = (
                    {"broken": outputs, "edited": outputs}
                    if which == "both"
                    else {which: outputs}
                )
                return 200, {"outputs": value, "budget": budget}
            path = "mock.npz"
            (session.work_dir / path).write_bytes(b"mock")
            return 200, {"path": path, "shapes": [], "budget": budget}
        if endpoint == "/v1/edit":
            session.edited = True
            return 200, {"params_changed": 1, "valid": True, "reasons": []}
        if endpoint == "/v1/reset_edit":
            session.edited = False
            return 200, {"ok": True}
        if endpoint == "/v1/submit":
            session.submitted = True
            return 200, {"ok": True}
        return 404, {"error": f"unknown endpoint {endpoint}"}

    def _dispatch_admin(
        self, endpoint: str, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        if endpoint == "/admin/health":
            return 200, {"active_sessions": len(self.sessions)}
        if endpoint == "/admin/open":
            session_id = str(payload["session"])
            episode = self.episodes / session_id
            model_dir, work_dir = episode / "model", episode / "work"
            model_dir.mkdir(parents=True)
            work_dir.mkdir(parents=True)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            with self._lock:
                self.sessions[session_id] = _Session(
                    token=str(payload["token"]),
                    model_dir=model_dir,
                    work_dir=work_dir,
                )
                self.open_count += 1
            return 200, {
                "model_dir": str(model_dir),
                "work_dir": str(work_dir),
            }
        if endpoint == "/admin/score":
            session = self.sessions.get(str(payload.get("session", "")))
            if session is None:
                return 404, {"error": "unknown session"}
            with self._lock:
                self.score_count += 1
            components = {
                "valid": True,
                "repair": self.repair,
                "collateral_kl": 0.01,
                "collateral_penalty": self.collateral_penalty,
                "localization": None,
                "trigger_found": None,
                "params_changed": int(session.edited),
                "notes": "mock score",
            }
            return 200, {"components": components}
        if endpoint == "/admin/close":
            session_id = str(payload.get("session", ""))
            session = self.sessions.pop(session_id, None)
            if session is not None:
                shutil.rmtree(session.model_dir.parent, ignore_errors=True)
            return 200, {"ok": True}
        if endpoint == "/admin/baseline":
            kind = payload.get("kind")
            score = 100.0 if kind == "oracle" else 0.0
            return 200, {
                "score": score,
                "components": {
                    "valid": True,
                    "repair": score / 100.0,
                    "collateral_kl": 0.0,
                    "collateral_penalty": 0.0,
                    "localization": None,
                    "trigger_found": None,
                    "params_changed": 0,
                    "notes": f"mock {kind}",
                },
            }
        return 404, {"error": f"unknown endpoint {endpoint}"}
