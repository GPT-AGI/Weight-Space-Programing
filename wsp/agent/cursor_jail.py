"""Outer jail for the Cursor SDK process and its local bridge."""

from __future__ import annotations

import json
import select
import socket
import socketserver
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from wsp.agent import cursor_mcp, cursor_worker

CURSOR_DOMAIN_SUFFIXES = ("cursor.com", "cursor.sh")


class _ConnectHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        request = bytearray()
        while b"\r\n\r\n" not in request and len(request) < 65_536:
            chunk = self.connection.recv(4_096)
            if not chunk:
                return
            request.extend(chunk)
        first_line = bytes(request).split(b"\r\n", 1)[0].decode(
            errors="replace"
        )
        parts = first_line.split()
        if len(parts) != 3 or parts[0].upper() != "CONNECT":
            self.connection.sendall(
                b"HTTP/1.1 405 Method Not Allowed\r\nConnection: close\r\n\r\n"
            )
            return
        target = parts[1]
        host, separator, port_text = target.rpartition(":")
        try:
            port = int(port_text) if separator else 443
        except ValueError:
            port = 0
        host = host.strip("[]").lower()
        allowed = (
            port == 443
            and any(
                host == suffix or host.endswith("." + suffix)
                for suffix in self.server.allowed_suffixes  # type: ignore[attr-defined]
            )
        )
        self.server.record(host, port, allowed)  # type: ignore[attr-defined]
        if not allowed:
            self.connection.sendall(
                b"HTTP/1.1 403 Forbidden\r\nConnection: close\r\n\r\n"
            )
            return
        try:
            upstream = socket.create_connection((host, port), timeout=30)
        except OSError:
            self.connection.sendall(
                b"HTTP/1.1 502 Bad Gateway\r\nConnection: close\r\n\r\n"
            )
            return
        with upstream:
            self.connection.sendall(
                b"HTTP/1.1 200 Connection Established\r\n\r\n"
            )
            sockets = [self.connection, upstream]
            while True:
                readable, _, exceptional = select.select(
                    sockets, [], sockets, 60
                )
                if exceptional or not readable:
                    return
                for source in readable:
                    try:
                        data = source.recv(1 << 16)
                    except OSError:
                        return
                    if not data:
                        return
                    destination = (
                        upstream if source is self.connection else self.connection
                    )
                    try:
                        destination.sendall(data)
                    except OSError:
                        return


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(
        self,
        path: str,
        *,
        allowed_suffixes: tuple[str, ...],
    ) -> None:
        self.allowed_suffixes = allowed_suffixes
        self.connections: list[dict[str, Any]] = []
        self._log_lock = threading.Lock()
        super().__init__(path, _ConnectHandler)

    def record(self, host: str, port: int, allowed: bool) -> None:
        with self._log_lock:
            self.connections.append(
                {"host": host, "port": port, "allowed": allowed}
            )


class AllowlistConnectProxy:
    def __init__(
        self,
        socket_path: Path,
        allowed_suffixes: tuple[str, ...] = CURSOR_DOMAIN_SUFFIXES,
    ) -> None:
        self.socket_path = socket_path
        self.allowed_suffixes = allowed_suffixes
        self._server: _ThreadingUnixServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def connections(self) -> list[dict[str, Any]]:
        return list(self._server.connections) if self._server else []

    def __enter__(self) -> AllowlistConnectProxy:
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        server = _ThreadingUnixServer(
            str(self.socket_path),
            allowed_suffixes=self.allowed_suffixes,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self._server, self._thread = server, thread
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._server = None
        self._thread = None
        self.socket_path.unlink(missing_ok=True)


@dataclass(frozen=True)
class CursorJailConfig:
    sdk_env: Path
    workspace: Path
    mcp_socket: Path
    timeout_s: float = 1_800.0
    memory_bytes: int = 8 * 1024**3
    cpu_seconds: int = 1_800
    network: Literal["proxy", "host"] = "host"


class CursorAgentJail:
    def __init__(self, config: CursorJailConfig) -> None:
        self.config = config

    def _command(self, proxy_socket: Path | None) -> list[str]:
        cfg = self.config
        for path in (cfg.sdk_env, cfg.workspace, cfg.mcp_socket):
            if not path.exists():
                raise FileNotFoundError(path)
        environment_python = cfg.sdk_env / "bin" / "python"
        resolved_python = environment_python.resolve()
        runtime_prefix = resolved_python.parents[1]
        python_target = f"/opt/cursor-python/bin/{resolved_python.name}"
        site_paths = sorted(cfg.sdk_env.glob("lib/python*/site-packages"))
        jail_site_paths = [
            str(Path("/opt/cursor-sdk") / path.relative_to(cfg.sdk_env))
            for path in site_paths
        ]
        bootstrap = (
            "import runpy,sys;"
            f"sys.path[:0]={jail_site_paths!r};"
            "runpy.run_path('/wsp-internal/cursor_worker.py',"
            "run_name='__main__')"
        )
        worker_path = Path(cursor_worker.__file__).resolve()
        mcp_path = Path(cursor_mcp.__file__).resolve()
        bwrap = [
            "bwrap",
            "--unshare-all",
            "--die-with-parent",
            "--new-session",
        ]
        if cfg.network == "host":
            bwrap.append("--share-net")
        else:
            bwrap.extend(["--cap-add", "CAP_NET_ADMIN"])
        bwrap.extend(
            [
                "--ro-bind",
                "/usr",
                "/usr",
                "--symlink",
                "usr/bin",
                "/bin",
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
                "--tmpfs",
                "/home",
                "--dir",
                "/home/cursor",
                "--dir",
                "/opt",
                "--ro-bind",
                str(runtime_prefix),
                "/opt/cursor-python",
                "--ro-bind",
                str(cfg.sdk_env.resolve()),
                "/opt/cursor-sdk",
                "--dir",
                "/wsp-internal",
                "--ro-bind",
                str(worker_path),
                "/wsp-internal/cursor_worker.py",
                "--ro-bind",
                str(mcp_path),
                "/wsp-internal/cursor_mcp.py",
                "--bind",
                str(cfg.mcp_socket.resolve()),
                "/wsp-internal/mcp.sock",
                "--bind",
                str(cfg.workspace.resolve()),
                "/workspace",
                "--dir",
                "/etc",
                "--ro-bind-try",
                "/etc/ssl",
                "/etc/ssl",
                "--ro-bind-try",
                "/etc/ca-certificates",
                "/etc/ca-certificates",
            ]
        )
        if cfg.network == "host":
            bwrap.extend(
                [
                    "--ro-bind-try",
                    "/etc/resolv.conf",
                    "/etc/resolv.conf",
                    "--ro-bind-try",
                    "/etc/hosts",
                    "/etc/hosts",
                ]
            )
        elif proxy_socket is not None:
            bwrap.extend(
                [
                    "--bind",
                    str(proxy_socket.resolve()),
                    "/wsp-internal/proxy.sock",
                    "--setenv",
                    "WSP_CONNECT_PROXY_SOCKET",
                    "/wsp-internal/proxy.sock",
                    "--setenv",
                    "HTTPS_PROXY",
                    "http://127.0.0.1:18080",
                    "--setenv",
                    "HTTP_PROXY",
                    "http://127.0.0.1:18080",
                    "--setenv",
                    "https_proxy",
                    "http://127.0.0.1:18080",
                    "--setenv",
                    "http_proxy",
                    "http://127.0.0.1:18080",
                    "--setenv",
                    "NO_PROXY",
                    "127.0.0.1,localhost",
                    "--setenv",
                    "no_proxy",
                    "127.0.0.1,localhost",
                ]
            )
        bwrap.extend(
            [
                "--setenv",
                "HOME",
                "/home/cursor",
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
                "PATH",
                "/opt/cursor-python/bin:/usr/bin",
                "--chdir",
                "/workspace",
            ]
        )
        if cfg.network == "proxy":
            bwrap.extend(
                [
                    "--setenv",
                    "WSP_WORKER_BOOTSTRAP",
                    bootstrap,
                    "/usr/bin/sh",
                    "-c",
                    (
                        "/usr/bin/ip link set lo up && exec "
                        f"{python_target} -u -c \"$WSP_WORKER_BOOTSTRAP\""
                    ),
                ]
            )
        else:
            bwrap.extend([python_target, "-u", "-c", bootstrap])
        return [
            "prlimit",
            f"--as={cfg.memory_bytes}",
            f"--cpu={cfg.cpu_seconds}",
            "--nproc=512",
            "--nofile=1024",
            "--",
            *bwrap,
        ]

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        cfg = self.config
        proxy_path = cfg.workspace / ".wsp-connect-proxy.sock"
        proxy: AllowlistConnectProxy | None = None
        if cfg.network == "proxy":
            proxy = AllowlistConnectProxy(proxy_path)
            proxy.__enter__()
        stderr_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
        try:
            process = subprocess.Popen(
                self._command(proxy_path if proxy else None),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                close_fds=True,
                env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            )
            try:
                stdout, _ = process.communicate(
                    json.dumps(request, ensure_ascii=False) + "\n",
                    timeout=cfg.timeout_s,
                )
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                return {
                    "started": True,
                    "status": "error",
                    "error": (
                        f"Cursor SDK episode exceeded {cfg.timeout_s:.0f}s "
                        "wall-clock timeout"
                    ),
                    "events": [],
                    "wall_s": cfg.timeout_s,
                    "jail": self._jail_evidence(proxy),
                }
            stderr_file.flush()
            stderr_file.seek(0)
            stderr_tail = stderr_file.read()[-8_000:]
            result_line = next(
                (
                    line[len(cursor_worker.RESULT_PREFIX) :]
                    for line in reversed(stdout.splitlines())
                    if line.startswith(cursor_worker.RESULT_PREFIX)
                ),
                None,
            )
            if result_line is None:
                return {
                    "started": False,
                    "status": "startup_error",
                    "error": (
                        f"jailed Cursor worker exited {process.returncode}: "
                        f"{stderr_tail or stdout[-2_000:]}"
                    ),
                    "events": [],
                    "jail": self._jail_evidence(proxy),
                }
            result = json.loads(result_line)
            result["bridge_stderr_tail"] = stderr_tail
            result["jail"] = self._jail_evidence(proxy)
            return result
        finally:
            stderr_file.close()
            if proxy is not None:
                proxy.__exit__(None, None, None)

    def _jail_evidence(
        self, proxy: AllowlistConnectProxy | None
    ) -> dict[str, Any]:
        return {
            "network": self.config.network,
            "allowed_domain_suffixes": list(CURSOR_DOMAIN_SUFFIXES),
            "proxy_connections": proxy.connections if proxy else [],
            "workspace": "/workspace",
            "setting_sources": ["project"],
            "ambient_user_settings": False,
        }
