"""Small HTTP-over-Unix-socket client used inside the networkless sandbox."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any


class WSPError(RuntimeError):
    def __init__(self, status: int, message: str, payload: Any = None) -> None:
        super().__init__(f"WSP host returned HTTP {status}: {message}")
        self.status = status
        self.payload = payload


class Client:
    def __init__(self) -> None:
        self.socket_path = os.environ.get("WSP_SOCKET", "/wsp/agent.sock")
        self.session = os.environ.get("WSP_SESSION", "")
        self.token = os.environ.get("WSP_TOKEN", "")
        if not self.session or not self.token:
            raise RuntimeError("WSP_SESSION and WSP_TOKEN must be set")

    def post(self, endpoint: str, **values: Any) -> dict[str, Any]:
        body = json.dumps(
            {"session": self.session, **values}, separators=(",", ":")
        ).encode()
        request = (
            f"POST {endpoint} HTTP/1.1\r\n"
            "Host: wsp\r\n"
            f"Authorization: Bearer {self.token}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode() + body
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(float(os.environ.get("WSP_HTTP_TIMEOUT", "600")))
            sock.connect(self.socket_path)
            sock.sendall(request)
            chunks: list[bytes] = []
            while True:
                chunk = sock.recv(1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
        raw = b"".join(chunks)
        header, separator, response_body = raw.partition(b"\r\n\r\n")
        if not separator:
            raise RuntimeError("malformed response from WSP host")
        status_line = header.split(b"\r\n", 1)[0].decode(errors="replace")
        try:
            status = int(status_line.split()[1])
        except (IndexError, ValueError) as exc:
            raise RuntimeError(f"malformed status line: {status_line}") from exc
        payload = json.loads(response_body or b"{}")
        if status >= 400:
            message = payload.get("error", payload.get("reason", "request failed"))
            raise WSPError(status, str(message), payload)
        return payload


def work_path(relative: str) -> Path:
    root = Path(os.environ.get("WSP_WORKDIR", "/wsp/work")).resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise RuntimeError("host returned a path outside WSP_WORKDIR")
    return path
