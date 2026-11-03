"""Bubblewrap launcher for the persistent agent Python kernel."""

from __future__ import annotations

import json
import selectors
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wsp.sandbox import kernel


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxConfig:
    model_dir: Path | str
    work_dir: Path | str
    agent_socket: Path | str
    session: str
    token: str
    sandbox_env: Path | str = Path("/opt/wsp-sandbox")
    memory_bytes: int = 8 * 1024**3
    cpu_seconds: int = 1_800
    exec_timeout_s: float = 300.0
    startup_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        for name in ("model_dir", "work_dir", "agent_socket", "sandbox_env"):
            object.__setattr__(self, name, Path(getattr(self, name)))


class SandboxKernel:
    """A single persistent, networkless Python namespace."""

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self._process: subprocess.Popen[str] | None = None
        self._stderr_file: Any = None
        self._request_id = 0
        self._lock = threading.Lock()

    def __enter__(self) -> SandboxKernel:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _command(self) -> list[str]:
        cfg = self.config
        for path, label in (
            (cfg.model_dir, "model directory"),
            (cfg.work_dir, "work directory"),
            (cfg.agent_socket, "agent socket"),
            (cfg.sandbox_env, "sandbox environment"),
        ):
            if not path.exists():
                raise SandboxError(f"{label} does not exist: {path}")

        environment_python = cfg.sandbox_env / "bin" / "python"
        if not environment_python.exists():
            raise SandboxError(
                f"sandbox Python does not exist: {environment_python}"
            )
        resolved_python = environment_python.resolve()
        runtime_prefix = resolved_python.parents[1]
        try:
            resolved_python.relative_to(cfg.sandbox_env.resolve())
            python_target = "/opt/wsp-sandbox/bin/python"
            runtime_bind: list[str] = []
        except ValueError:
            python_target = f"/opt/wsp-python/bin/{resolved_python.name}"
            runtime_bind = [
                "--ro-bind",
                str(runtime_prefix),
                "/opt/wsp-python",
            ]
        site_paths = sorted(
            cfg.sandbox_env.glob("lib/python*/site-packages")
        )
        sandbox_site_paths = [
            str(Path("/opt/wsp-sandbox") / path.relative_to(cfg.sandbox_env))
            for path in site_paths
        ]
        source = Path(kernel.__file__).read_text(encoding="utf-8")
        bootstrap = (
            "import sys;"
            f"sys.path[:0]={sandbox_site_paths!r};"
            f"exec(compile({source!r},'<wsp-kernel>','exec'))"
        )
        bwrap = [
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
            "--ro-bind",
            "/usr",
            "/usr",
            "--ro-bind",
            "/lib",
            "/lib",
            "--ro-bind",
            "/lib64",
            "/lib64",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/opt",
            *runtime_bind,
            "--ro-bind",
            str(cfg.sandbox_env.resolve()),
            "/opt/wsp-sandbox",
            "--dir",
            "/wsp",
            "--ro-bind",
            str(cfg.model_dir.resolve()),
            "/wsp/model",
            "--bind",
            str(cfg.work_dir.resolve()),
            "/wsp/work",
            "--bind",
            str(cfg.agent_socket.resolve()),
            "/wsp/agent.sock",
            "--setenv",
            "HOME",
            "/wsp/work",
            "--setenv",
            "TMPDIR",
            "/tmp",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "HF_HUB_OFFLINE",
            "1",
            "--setenv",
            "TRANSFORMERS_OFFLINE",
            "1",
            "--setenv",
            "WSP_SESSION",
            cfg.session,
            "--setenv",
            "WSP_TOKEN",
            cfg.token,
            "--setenv",
            "WSP_SOCKET",
            "/wsp/agent.sock",
            "--setenv",
            "WSP_MODEL_DIR",
            "/wsp/model",
            "--setenv",
            "WSP_WORKDIR",
            "/wsp/work",
            "--chdir",
            "/wsp/work",
            python_target,
            "-I",
            "-u",
            "-c",
            bootstrap,
        ]
        return [
            "prlimit",
            f"--as={cfg.memory_bytes}",
            f"--cpu={cfg.cpu_seconds}",
            "--nproc=256",
            "--nofile=512",
            "--",
            *bwrap,
        ]

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        self._stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        self._process = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
            close_fds=True,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
        )
        try:
            ready = self._read_json(self.config.startup_timeout_s)
        except Exception as exc:
            detail = self._stderr()
            self.close()
            raise SandboxError(
                f"sandbox failed to start: {detail or 'no response'}"
            ) from exc
        if ready.get("event") != "ready":
            self.close()
            raise SandboxError(f"unexpected sandbox handshake: {ready!r}")

    def _stderr(self) -> str:
        if self._stderr_file is None:
            return ""
        self._stderr_file.flush()
        self._stderr_file.seek(0)
        return self._stderr_file.read()[-8_000:]

    def _read_json(self, timeout_s: float) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise SandboxError("sandbox is not running")
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            events = selector.select(timeout=max(0.0, timeout_s))
        finally:
            selector.close()
        if not events:
            raise TimeoutError("sandbox response timed out")
        line = process.stdout.readline()
        if not line:
            raise SandboxError(
                f"sandbox exited with {process.poll()}: {self._stderr()}"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SandboxError(f"invalid sandbox response: {line[:500]!r}") from exc
        if not isinstance(value, dict):
            raise SandboxError("sandbox response was not an object")
        return value

    def execute(
        self,
        code: str,
        *,
        timeout_s: float | None = None,
        max_output_chars: int = 100_000,
    ) -> dict[str, Any]:
        with self._lock:
            self.start()
            process = self._process
            if process is None or process.stdin is None:
                raise SandboxError("sandbox is not running")
            self._request_id += 1
            request = {
                "op": "exec",
                "id": self._request_id,
                "code": code,
                "timeout_s": timeout_s or self.config.exec_timeout_s,
                "max_output_chars": max_output_chars,
            }
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            outer_timeout = float(request["timeout_s"]) + 5.0
            try:
                response = self._read_json(outer_timeout)
            except (TimeoutError, SandboxError) as exc:
                self._restart()
                return {
                    "id": self._request_id,
                    "ok": False,
                    "stdout": "",
                    "stderr": "",
                    "traceback": (
                        f"Sandbox execution failed ({exc}). The kernel was "
                        "restarted and its previous namespace was lost."
                    ),
                    "submitted": False,
                    "budget": None,
                    "restarted": True,
                }
            if response.get("id") != self._request_id:
                raise SandboxError("sandbox response id did not match request")
            return response

    def _restart(self) -> None:
        self.close()
        self.start()

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write('{"op":"shutdown","id":0}\n')
                    process.stdin.flush()
                process.wait(timeout=1.0)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None


def format_execution_output(response: dict[str, Any], limit: int) -> str:
    parts: list[str] = []
    stdout = str(response.get("stdout") or "")
    stderr = str(response.get("stderr") or "")
    trace = str(response.get("traceback") or "")
    if stdout:
        parts.append(stdout.rstrip())
    if stderr:
        parts.append("[stderr]\n" + stderr.rstrip())
    if trace:
        parts.append(trace.rstrip())
    if not parts:
        parts.append("[no output]")
    value = "\n".join(parts)
    if len(value) <= limit:
        return value
    omitted = len(value) - limit
    marker = f"\n[tool output truncated; {omitted} characters omitted]"
    return value[: max(0, limit - len(marker))] + marker
