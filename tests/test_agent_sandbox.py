from __future__ import annotations

import json
import secrets

from tests.mock_host import MockHost, build_stub_sandbox_env
from wsp.agent.clients import UnixHostClient
from wsp.sandbox import SandboxConfig, SandboxKernel


def _open(mock: MockHost) -> tuple[str, str, object]:
    session, token = secrets.token_hex(8), secrets.token_urlsafe(16)
    admin = UnixHostClient(mock.admin_socket)
    opened = admin.post(
        "/admin/open",
        instance_id="dev-behavior-easy-01",
        split="dev",
        session=session,
        token=token,
    )
    admin.close()
    return session, token, opened


def test_bwrap_isolation_and_agent_socket(tmp_path):
    environment = build_stub_sandbox_env(tmp_path)
    with MockHost(tmp_path / "runtime") as mock:
        session, token, opened = _open(mock)
        config = SandboxConfig(
            model_dir=opened["model_dir"],
            work_dir=opened["work_dir"],
            agent_socket=mock.agent_socket,
            session=session,
            token=token,
            sandbox_env=environment,
        )
        code = """
import json, pathlib, socket
try:
    socket.create_connection(("1.1.1.1", 80), timeout=0.2)
    network_blocked = False
except OSError:
    network_blocked = True
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.connect("/wsp/agent.sock")
s.close()
print(json.dumps({
    "network_blocked": network_blocked,
    "root_hidden": not pathlib.Path("/root").exists(),
    "private_hidden": not pathlib.Path("/srv/wsp-private").exists(),
    "agent_socket_reachable": True,
}))
"""
        with SandboxKernel(config) as sandbox:
            response = sandbox.execute(code)
        values = json.loads(response["stdout"])
        assert values == {
            "network_blocked": True,
            "root_hidden": True,
            "private_hidden": True,
            "agent_socket_reachable": True,
        }


def test_kernel_persists_namespace_and_interrupts_timeout(tmp_path):
    environment = build_stub_sandbox_env(tmp_path)
    with MockHost(tmp_path / "runtime") as mock:
        session, token, opened = _open(mock)
        config = SandboxConfig(
            model_dir=opened["model_dir"],
            work_dir=opened["work_dir"],
            agent_socket=mock.agent_socket,
            session=session,
            token=token,
            sandbox_env=environment,
        )
        with SandboxKernel(config) as sandbox:
            assert sandbox.execute("value = 41")["ok"]
            assert sandbox.execute("print(value + 1)")["stdout"].strip() == "42"
            timed_out = sandbox.execute(
                "value = 99\nwhile True:\n    pass", timeout_s=0.1
            )
            assert not timed_out["ok"]
            assert "ExecutionTimedOut" in timed_out["traceback"]
            assert sandbox.execute("print(value)")["stdout"].strip() == "99"
