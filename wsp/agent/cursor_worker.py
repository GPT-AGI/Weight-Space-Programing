"""Cursor SDK worker executed as PID 1 inside the outer filesystem jail."""

from __future__ import annotations

import dataclasses
import json
import os
import socket
import socketserver
import sys
import threading
import time
from pathlib import Path
from typing import Any

RESULT_PREFIX = "WSP_CURSOR_RESULT "


class _TCPToUnixHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        unix_path = self.server.unix_path  # type: ignore[attr-defined]
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as upstream:
            upstream.connect(unix_path)
            threads = [
                threading.Thread(
                    target=self._copy,
                    args=(self.request, upstream),
                    daemon=True,
                ),
                threading.Thread(
                    target=self._copy,
                    args=(upstream, self.request),
                    daemon=True,
                ),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    @staticmethod
    def _copy(source: socket.socket, destination: socket.socket) -> None:
        try:
            while chunk := source.recv(1 << 16):
                destination.sendall(chunk)
        except OSError:
            pass
        finally:
            try:
                destination.shutdown(socket.SHUT_WR)
            except OSError:
                pass


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class TCPToUnixProxy:
    def __init__(self, unix_path: str, port: int = 18080) -> None:
        server = _ThreadingTCPServer(
            ("127.0.0.1", port), _TCPToUnixHandler
        )
        server.unix_path = unix_path  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(
            target=server.serve_forever, daemon=True
        )

    def __enter__(self) -> TCPToUnixProxy:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)


def _primitive(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _primitive(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_primitive(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _run(config: dict[str, Any]) -> dict[str, Any]:
    try:
        direct = socket.create_connection(("1.1.1.1", 443), timeout=0.2)
    except OSError:
        direct_internet_blocked = True
    else:
        direct.close()
        direct_internet_blocked = False
    isolation = {
        "root_visible": Path("/root").exists(),
        "repo_visible": Path(
            "/root/OpenSource/weight-space-programing"
        ).exists(),
        "private_visible": Path("/srv/wsp-private").exists(),
        "cache_visible": Path("/srv/wsp-cache").exists(),
        "admin_socket_visible": Path("/srv/wsp/admin.sock").exists(),
        "direct_internet_blocked": direct_internet_blocked,
        "https_proxy": bool(os.environ.get("HTTPS_PROXY")),
    }
    if config.get("probe_only"):
        return {
            "started": False,
            "status": "probe",
            "events": [],
            "isolation": isolation,
        }

    from cursor_sdk import (
        Agent,
        AgentOptions,
        CursorAgentError,
        LocalAgentOptions,
        ModelParameterValue,
        ModelSelection,
        StdioMcpServerConfig,
    )
    if config.get("list_models"):
        from cursor_sdk import Cursor

        models = Cursor.models.list(api_key=str(config["api_key"]))
        return {
            "started": True,
            "status": "models",
            "models": _primitive(models),
            "events": [],
            "isolation": isolation,
        }

    params = [
        ModelParameterValue(id=str(row["id"]), value=str(row["value"]))
        for row in config.get("model_params", [])
    ]
    model = ModelSelection(id=str(config["model_id"]), params=params)
    mcp = StdioMcpServerConfig(
        command=sys.executable,
        args=[
            "/wsp-internal/cursor_mcp.py",
            "--socket",
            "/wsp-internal/mcp.sock",
        ],
        env={"WSP_CURSOR_MCP_SOCKET": "/wsp-internal/mcp.sock"},
        cwd="/workspace",
    )
    events: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        options = AgentOptions(
            model=model,
            api_key=str(config["api_key"]),
            name="WSP-Bench Cursor Agent",
            local=LocalAgentOptions(
                cwd="/workspace",
                setting_sources=["project"],
            ),
            mcp_servers={"wsp": mcp},
            tools=(
                ["mcp", "shell", "read"]
                if config.get("proof_builtin_tools")
                else ["mcp"]
            ),
            mode="agent",
        )
        with Agent.create(options) as agent:
            run = agent.send(str(config["prompt"]))
            for message in run.messages():
                primitive = _primitive(message)
                if isinstance(primitive, dict):
                    events.append(primitive)
            result = run.wait()
            try:
                conversation = _primitive(run.conversation())
            except Exception as exc:
                conversation = {
                    "error": f"{type(exc).__name__}: {exc}"
                }
            return {
                "started": True,
                "status": result.status,
                "result": result.result,
                "error_code": result.error_code,
                "error_message": result.error_message,
                "duration_ms": result.duration_ms,
                "usage": _primitive(result.usage),
                "model": _primitive(result.model),
                "conversation": conversation,
                "agent_id": agent.agent_id,
                "run_id": run.id,
                "events": events,
                "wall_s": time.monotonic() - started,
                "isolation": isolation,
            }
    except CursorAgentError as exc:
        return {
            "started": False,
            "status": "startup_error",
            "error": str(exc),
            "retryable": bool(getattr(exc, "is_retryable", False)),
            "events": events,
            "wall_s": time.monotonic() - started,
        }
    except Exception as exc:
        return {
            "started": True,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "events": events,
            "wall_s": time.monotonic() - started,
        }


def main() -> int:
    config = json.loads(sys.stdin.readline())
    proxy_path = os.environ.get("WSP_CONNECT_PROXY_SOCKET")
    if proxy_path:
        with TCPToUnixProxy(proxy_path):
            result = _run(config)
    else:
        result = _run(config)
    print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
