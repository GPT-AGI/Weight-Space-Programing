"""Single-tool MCP bridge from a jailed Cursor agent to the WSP kernel."""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from wsp.sandbox import SandboxKernel

MCP_PROTOCOL_VERSION = "2025-06-18"
TOOL_NAME = "run_python"


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class _BrokerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        line = self.rfile.readline()
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("request must be a JSON object")
            response = self.server.broker.handle(request)  # type: ignore[attr-defined]
        except Exception as exc:
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        self.wfile.write(
            (json.dumps(response, ensure_ascii=False) + "\n").encode()
        )


class CursorMCPBroker:
    """Host-side broker that owns the only reference to the sandbox kernel."""

    def __init__(
        self,
        *,
        socket_path: Path,
        sandbox: SandboxKernel,
        max_calls: int,
        stdout_chars: int,
        exec_timeout_s: float = 300.0,
    ) -> None:
        self.socket_path = socket_path
        self.sandbox = sandbox
        self.max_calls = max_calls
        self.stdout_chars = stdout_chars
        self.exec_timeout_s = exec_timeout_s
        self.calls_used = 0
        self.gpu_calls_used = 0
        self.submitted = False
        self.last_budget: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._server: _ThreadingUnixServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> CursorMCPBroker:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self._server is not None:
            return
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        server = _ThreadingUnixServer(str(self.socket_path), _BrokerHandler)
        server.broker = self  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server, self._thread = server, thread

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        self.socket_path.unlink(missing_ok=True)

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("op") != TOOL_NAME:
            return {"ok": False, "error": "unsupported broker operation"}
        code = request.get("code")
        if not isinstance(code, str):
            return {"ok": False, "error": "code must be a string"}
        with self._lock:
            if self.submitted:
                return {
                    "ok": False,
                    "error": "episode already submitted",
                    **self._state(),
                }
            if self.calls_used >= self.max_calls:
                return {
                    "ok": False,
                    "error": "run_python turn budget exhausted",
                    **self._state(),
                }
            self.calls_used += 1
            from wsp.sandbox import format_execution_output

            response = self.sandbox.execute(
                code,
                timeout_s=self.exec_timeout_s,
                max_output_chars=max(100_000, self.stdout_chars * 4),
            )
            self.submitted = bool(response.get("submitted"))
            budget = response.get("budget")
            if isinstance(budget, dict):
                self.last_budget = budget
                used = budget.get("gpu_calls_used")
                if isinstance(used, (int, float)):
                    self.gpu_calls_used = int(used)
            return {
                "ok": bool(response.get("ok")),
                "output": format_execution_output(response, self.stdout_chars),
                "kernel_restarted": bool(response.get("restarted")),
                **self._state(),
            }

    def _state(self) -> dict[str, Any]:
        return {
            "submitted": self.submitted,
            "calls_used": self.calls_used,
            "calls_remaining": max(0, self.max_calls - self.calls_used),
            "gpu_calls_used": self.gpu_calls_used,
            "budget": self.last_budget,
        }


def broker_request(socket_path: Path, code: str) -> dict[str, Any]:
    request = json.dumps({"op": TOOL_NAME, "code": code}).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(310.0)
        client.connect(str(socket_path))
        client.sendall(request)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    response = json.loads(b"".join(chunks))
    if not isinstance(response, dict):
        raise TypeError("broker returned a non-object response")
    return response


def _mcp_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _mcp_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def run_stdio(socket_path: Path) -> None:
    for raw_line in sys.stdin:
        try:
            request = json.loads(raw_line)
            if not isinstance(request, dict):
                raise TypeError("MCP request must be an object")
            request_id = request.get("id")
            method = request.get("method")
            if request_id is None:
                continue
            if method == "initialize":
                params = request.get("params") or {}
                result = {
                    "protocolVersion": params.get(
                        "protocolVersion", MCP_PROTOCOL_VERSION
                    ),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "wsp", "version": "0.1.0"},
                }
                response = _mcp_result(request_id, result)
            elif method == "ping":
                response = _mcp_result(request_id, {})
            elif method == "tools/list":
                response = _mcp_result(
                    request_id,
                    {
                        "tools": [
                            {
                                "name": TOOL_NAME,
                                "description": (
                                    "Execute Python in the persistent offline WSP "
                                    "sandbox. Use wsp.* for model experiments and "
                                    "wsp.submit(...) to finish."
                                ),
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "code": {
                                            "type": "string",
                                            "description": "Python source to execute",
                                        }
                                    },
                                    "required": ["code"],
                                    "additionalProperties": False,
                                },
                            }
                        ]
                    },
                )
            elif method == "tools/call":
                params = request.get("params") or {}
                if params.get("name") != TOOL_NAME:
                    response = _mcp_error(
                        request_id, -32602, "unknown tool"
                    )
                else:
                    arguments = params.get("arguments") or {}
                    result = broker_request(
                        socket_path, str(arguments.get("code", ""))
                    )
                    response = _mcp_result(
                        request_id,
                        {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        result, ensure_ascii=False
                                    ),
                                }
                            ],
                            "structuredContent": result,
                            "isError": not bool(result.get("ok")),
                        },
                    )
            else:
                response = _mcp_error(
                    request_id, -32601, f"unknown method: {method}"
                )
        except Exception as exc:
            response = _mcp_error(
                locals().get("request_id"),
                -32603,
                f"{type(exc).__name__}: {exc}",
            )
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--socket",
        type=Path,
        default=Path(
            os.environ.get(
                "WSP_CURSOR_MCP_SOCKET", "/wsp-internal/mcp.sock"
            )
        ),
    )
    args = parser.parse_args(argv)
    run_stdio(args.socket)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
