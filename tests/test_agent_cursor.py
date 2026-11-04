from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

from wsp.agent.cursor_jail import CursorAgentJail, CursorJailConfig
from wsp.agent.cursor_mcp import (
    CursorMCPBroker,
    broker_request,
)
from wsp.agent.cursor_track import (
    _cursor_diagnostic,
    make_cursor_model_spec,
    map_cursor_transcript,
    prepare_cursor_workspace,
)


class FakeKernel:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, code, **_kwargs):
        self.calls += 1
        return {
            "ok": True,
            "stdout": f"ran:{code}",
            "stderr": "",
            "traceback": "",
            "submitted": "wsp.submit" in code,
            "budget": {
                "gpu_calls_used": self.calls,
                "gpu_calls_remaining": 10 - self.calls,
            },
            "restarted": False,
        }


def test_cursor_mcp_tool_and_turn_budget(tmp_path):
    socket_path = tmp_path / "mcp.sock"
    kernel = FakeKernel()
    with CursorMCPBroker(
        socket_path=socket_path,
        sandbox=kernel,
        max_calls=2,
        stdout_chars=4_000,
    ):
        first = broker_request(socket_path, "print(1)")
        second = broker_request(socket_path, "print(2)")
        exhausted = broker_request(socket_path, "print(3)")
    assert first["ok"] and first["calls_used"] == 1
    assert second["ok"] and second["calls_remaining"] == 0
    assert not exhausted["ok"]
    assert kernel.calls == 2


def test_cursor_mcp_stdio_exposes_only_run_python(tmp_path):
    socket_path = tmp_path / "mcp.sock"
    kernel = FakeKernel()
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "run_python",
                "arguments": {"code": "wsp.submit()"},
            },
        },
    ]
    with CursorMCPBroker(
        socket_path=socket_path,
        sandbox=kernel,
        max_calls=2,
        stdout_chars=4_000,
    ):
        completed = subprocess.run(
            [
                sys.executable,
                "wsp/agent/cursor_mcp.py",
                "--socket",
                str(socket_path),
            ],
            input="\n".join(json.dumps(row) for row in requests) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
    responses = [
        json.loads(line) for line in completed.stdout.splitlines()
    ]
    tools = responses[1]["result"]["tools"]
    assert [tool["name"] for tool in tools] == ["run_python"]
    structured = responses[2]["result"]["structuredContent"]
    assert structured["submitted"]
    assert structured["calls_used"] == 1


def test_cursor_hook_policy_denies_builtin_tools(tmp_path):
    workspace = tmp_path / "workspace"
    policy = prepare_cursor_workspace(workspace)
    payloads = [
        {"hook_event_name": "preToolUse", "tool_name": "Shell"},
        {"hook_event_name": "beforeReadFile", "tool_name": "Read"},
        {
            "hook_event_name": "beforeMCPExecution",
            "tool_name": "run_python",
            "mcp_server_name": "wsp",
        },
    ]
    decisions = []
    for payload in payloads:
        completed = subprocess.run(
            [sys.executable, str(policy)],
            input=json.dumps(payload),
            cwd=workspace,
            text=True,
            capture_output=True,
            check=True,
            env={
                **os.environ,
                "WSP_HOOK_AUDIT": str(
                    workspace / "hook-audit.jsonl"
                ),
            },
        )
        decisions.append(json.loads(completed.stdout))
    assert [row["permission"] for row in decisions] == [
        "deny",
        "deny",
        "allow",
    ]
    audit = [
        json.loads(line)
        for line in (workspace / "hook-audit.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["permission"] for row in audit] == [
        "deny",
        "deny",
        "allow",
    ]


def test_cursor_outer_jail_hides_sensitive_paths(tmp_path):
    workspace = tmp_path / "workspace"
    prepare_cursor_workspace(workspace)
    socket_path = tmp_path / "mcp.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(socket_path))
    listener.listen()
    try:
        result = CursorAgentJail(
            CursorJailConfig(
                sdk_env=Path(".venv").resolve(),
                workspace=workspace,
                mcp_socket=socket_path,
                timeout_s=30,
                network="proxy",
            )
        ).run({"probe_only": True})
    finally:
        listener.close()
    assert result["status"] == "probe"
    assert result["isolation"] == {
        "root_visible": False,
        "repo_visible": False,
        "private_visible": False,
        "cache_visible": False,
        "admin_socket_visible": False,
        "direct_internet_blocked": True,
        "https_proxy": True,
    }
    assert result["jail"]["network"] == "proxy"
    assert result["jail"]["setting_sources"] == ["project"]
    assert not result["jail"]["ambient_user_settings"]


def test_cursor_transcript_mapping_and_metadata():
    events = [
        {"type": "thinking", "text": "inspect first"},
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "I will probe."}]
            },
        },
        {
            "type": "tool_call",
            "name": "mcp_wsp_run_python",
            "status": "completed",
            "args": {"arguments": {"code": "print(1)"}},
            "result": {
                "status": "success",
                "value": {
                    "content": [
                        {
                            "type": "text",
                            "text": {
                                "text": (
                                    '{"output":"1","gpu_calls_used":2}'
                                )
                            },
                        }
                    ]
                },
            },
        },
    ]
    trace = map_cursor_transcript(
        system="system", brief="brief", events=events
    )
    assert [turn.role for turn in trace] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert trace[2].reasoning == "inspect first"
    assert trace[3].code == "print(1)"
    assert trace[3].gpu_calls_used == 2

    spec = make_cursor_model_spec(
        "cursor-agent/gpt-5.6-sol",
        params=[{"id": "reasoning", "value": "high"}],
    )
    assert spec.display_name == "Cursor Agent · GPT-5.6 Sol"
    assert spec.extra_body["harness"] == "cursor-sdk"
    assert spec.extra_body["billing"] == "cursor-plan"
    assert (
        make_cursor_model_spec(
            "cursor-agent/claude-opus-5-5"
        ).display_name
        == "Cursor Agent · Claude Opus 5.5"
    )


def test_cursor_diagnostic_surfaces_and_redacts_errors():
    diagnostic = _cursor_diagnostic(
        {
            "status": "error",
            "error_code": "model_unavailable",
            "error_message": "region blocked secret-key",
            "events": [
                {
                    "type": "status",
                    "message": "provider unavailable",
                }
            ],
            "bridge_stderr_tail": "bridge detail",
        },
        "secret-key",
    )
    assert "model_unavailable" in diagnostic
    assert "provider unavailable" in diagnostic
    assert "bridge detail" in diagnostic
    assert "secret-key" not in diagnostic
