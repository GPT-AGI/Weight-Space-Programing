"""Cursor Agent track orchestration and transcript conversion."""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import shlex
import shutil
import socket
import tempfile
from pathlib import Path
from typing import Any

from wsp.agent.cursor_jail import CursorAgentJail, CursorJailConfig
from wsp.agent.cursor_mcp import CursorMCPBroker
from wsp.agent.loop import AgentOutcome
from wsp.agent.prompts import cursor_system_prompt
from wsp.contracts import Instance, ModelSpec, StopReason, TokenUsage, Turn
from wsp.sandbox import SandboxKernel

HOOK_POLICY = r'''#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

payload = json.load(sys.stdin)
event = str(payload.get("hook_event_name", ""))
tool = str(payload.get("tool_name", ""))
server = str(payload.get("mcp_server_name", ""))
allowed = (
    (event == "preToolUse" and tool.startswith("MCP:") and tool.endswith("run_python"))
    or (
        event == "beforeMCPExecution"
        and server == "wsp"
        and tool == "run_python"
    )
)
decision = {
    "permission": "allow" if allowed else "deny",
    "user_message": "" if allowed else "Only WSP run_python is permitted.",
    "agent_message": "" if allowed else "Use the run_python MCP tool only.",
}
audit = Path(os.environ.get("WSP_HOOK_AUDIT", "/workspace/hook-audit.jsonl"))
with audit.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "event": event,
        "tool_name": tool,
        "mcp_server_name": server,
        "permission": decision["permission"],
    }, separators=(",", ":")) + "\n")
print(json.dumps(decision))
'''


def _load_cursor_key() -> str:
    key = os.environ.get("CURSOR_API_KEY")
    if key:
        return key
    path = Path.home() / ".config" / "wsp" / "env"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                parts = shlex.split(line, comments=True)
            except ValueError:
                continue
            if parts and parts[0] == "export":
                parts = parts[1:]
            for assignment in parts:
                name, separator, value = assignment.partition("=")
                if separator and name == "CURSOR_API_KEY":
                    return value
    raise RuntimeError(
        "CURSOR_API_KEY is unset and ~/.config/wsp/env did not provide it"
    )


def cursor_sdk_version() -> str:
    try:
        return importlib.metadata.version("cursor-sdk")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def make_cursor_model_spec(
    model_id: str,
    *,
    params: list[dict[str, str]] | None = None,
    network: str = "host",
) -> ModelSpec:
    if not model_id.startswith("cursor-agent/"):
        raise ValueError("Cursor track model ids must start with cursor-agent/")
    sdk_model = model_id.removeprefix("cursor-agent/")
    if not sdk_model:
        raise ValueError("Cursor SDK model id cannot be empty")
    display = sdk_model.replace("-", " ").title()
    display = re.sub(r"(?<=\d) (?=\d)", ".", display)
    if display.startswith("Gpt "):
        display = "GPT-" + display.removeprefix("Gpt ")
    return ModelSpec(
        id=model_id,
        display_name=f"Cursor Agent · {display}",
        usd_per_m_input=0.0,
        usd_per_m_output=0.0,
        extra_body={
            "harness": "cursor-sdk",
            "sdk_version": cursor_sdk_version(),
            "billing": "cursor-plan",
            "model": {"id": sdk_model, "params": params or []},
            "network": network,
        },
    )


def prepare_cursor_workspace(workspace: Path) -> Path:
    if workspace.exists():
        shutil.rmtree(workspace)
    hook_dir = workspace / ".cursor" / "hooks"
    hook_dir.mkdir(parents=True)
    policy = hook_dir / "wsp_policy.py"
    policy.write_text(HOOK_POLICY, encoding="utf-8")
    policy.chmod(0o755)
    command = "python .cursor/hooks/wsp_policy.py"
    hook = {"command": command, "matcher": "*", "failClosed": True}
    hooks = {
        "version": 1,
        "hooks": {
            "preToolUse": [hook],
            "beforeMCPExecution": [hook],
            "beforeShellExecution": [hook],
            "beforeReadFile": [hook],
            "subagentStart": [hook],
        },
    }
    (workspace / ".cursor" / "hooks.json").write_text(
        json.dumps(hooks, indent=2) + "\n", encoding="utf-8"
    )
    return policy


def _text_blocks(message: dict[str, Any]) -> str:
    content = (message.get("message") or {}).get("content") or []
    if isinstance(content, str):
        return content
    parts = [
        str(block.get("text", ""))
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(parts)


def _tool_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "value" in value:
            return _tool_output(value["value"])
        content = value.get("content")
        if isinstance(content, list):
            texts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = item.get("text")
                if isinstance(text, dict):
                    text = text.get("text")
                if text:
                    texts.append(str(text))
            if texts:
                return "\n".join(texts)
    return json.dumps(value, ensure_ascii=False, default=str)


def _tool_code(value: Any) -> str | None:
    if isinstance(value, dict):
        code = value.get("code")
        if isinstance(code, str):
            return code
        for nested in value.values():
            found = _tool_code(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _tool_code(nested)
            if found is not None:
                return found
    elif isinstance(value, str) and value.startswith("{"):
        try:
            return _tool_code(json.loads(value))
        except json.JSONDecodeError:
            return None
    return None


def map_cursor_transcript(
    *,
    system: str,
    brief: str,
    events: list[dict[str, Any]],
) -> list[Turn]:
    trace = [
        Turn(t=0, role="system", content=system),
        Turn(t=0, role="user", content=brief),
    ]
    reasoning: list[str] = []
    assistant_text: list[str] = []
    tool_turn = 0

    def flush_assistant() -> None:
        if not assistant_text and not reasoning:
            return
        trace.append(
            Turn(
                t=max(1, tool_turn + 1),
                role="assistant",
                content="".join(assistant_text)
                or "[no visible assistant text]",
                reasoning="".join(reasoning) or None,
            )
        )
        assistant_text.clear()
        reasoning.clear()

    for event in events:
        event_type = event.get("type")
        if event_type == "thinking":
            text = event.get("text")
            if text:
                reasoning.append(str(text))
        elif event_type == "assistant":
            content = _text_blocks(event)
            if content:
                assistant_text.append(content)
        elif event_type == "tool_call":
            if event.get("status") not in {"completed", "error"}:
                flush_assistant()
                continue
            flush_assistant()
            tool_turn += 1
            arguments = event.get("args") or {}
            code = _tool_code(arguments)
            output = _tool_output(event.get("result"))
            gpu_calls = 0
            try:
                parsed = json.loads(output)
                gpu_calls = int(parsed.get("gpu_calls_used", 0))
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            trace.append(
                Turn(
                    t=tool_turn,
                    role="tool",
                    content=output,
                    code=code,
                    gpu_calls_used=gpu_calls,
                )
            )
    flush_assistant()
    return trace


def _token_usage(value: Any) -> TokenUsage:
    if not isinstance(value, dict):
        return TokenUsage()
    output = int(value.get("output_tokens", 0) or 0)
    reasoning = int(value.get("reasoning_tokens", 0) or 0)
    return TokenUsage(
        prompt=int(value.get("input_tokens", 0) or 0),
        completion=max(0, output - reasoning),
        cached=int(value.get("cache_read_tokens", 0) or 0),
        reasoning=reasoning,
    )


def _cursor_diagnostic(result: dict[str, Any], secret: str) -> str:
    parts: list[str] = []
    error_code = result.get("error_code")
    error_message = result.get("error_message")
    if error_code or error_message:
        parts.append(
            f"run.wait(): {error_code or 'error'}: "
            f"{error_message or '(no message)'}"
        )
    if result.get("error"):
        parts.append("worker: " + str(result["error"]))
    status_messages = [
        str(event.get("message"))
        for event in result.get("events", [])
        if isinstance(event, dict)
        and event.get("type") == "status"
        and event.get("message")
    ]
    if status_messages:
        parts.append("status: " + " | ".join(status_messages[-3:]))
    conversation = result.get("conversation")
    if conversation:
        parts.append(
            "conversation: "
            + json.dumps(
                conversation, ensure_ascii=False, default=str
            )[:4_000]
        )
    stderr = str(result.get("bridge_stderr_tail") or "").strip()
    if stderr:
        parts.append("bridge stderr: " + stderr[-4_000:])
    diagnostic = "\n".join(parts) or (
        f"Cursor SDK run ended with status={result.get('status', 'error')}"
    )
    return diagnostic.replace(secret, "[REDACTED]")[:12_000]


def run_cursor_agent(
    *,
    instance: Instance,
    model: ModelSpec,
    sandbox: SandboxKernel,
    work_dir: Path,
    sdk_env: Path,
    sdk_reference: str,
    timeout_s: float,
    network: str,
) -> AgentOutcome:
    system = cursor_system_prompt(instance, sdk_reference=sdk_reference)
    control_dir = work_dir.parent / "cursor-control"
    workspace = control_dir / "workspace"
    prepare_cursor_workspace(workspace)
    mcp_socket = control_dir / "cursor-mcp.sock"
    model_config = model.extra_body.get("model", {})
    cursor_key = _load_cursor_key()
    request = {
        "api_key": cursor_key,
        "model_id": model_config.get("id"),
        "model_params": model_config.get("params", []),
        "prompt": system + "\n\nTask:\n" + instance.brief,
    }
    with CursorMCPBroker(
        socket_path=mcp_socket,
        sandbox=sandbox,
        max_calls=instance.budget.max_turns,
        stdout_chars=instance.budget.stdout_chars,
    ) as broker:
        jail = CursorAgentJail(
            CursorJailConfig(
                sdk_env=sdk_env,
                workspace=workspace,
                mcp_socket=mcp_socket,
                timeout_s=timeout_s,
                network=network,  # type: ignore[arg-type]
            )
        )
        result = jail.run(request)
    events = result.get("events")
    if not isinstance(events, list):
        events = []
    trace = map_cursor_transcript(
        system=system,
        brief=instance.brief,
        events=[event for event in events if isinstance(event, dict)],
    )
    status = str(result.get("status", "error"))
    diagnostic = _cursor_diagnostic(result, cursor_key)
    if status == "error":
        trace.append(
            Turn(
                t=max(1, broker.calls_used),
                role="tool",
                content="[Cursor SDK error]\n" + diagnostic,
                gpu_calls_used=broker.gpu_calls_used,
            )
        )
    if broker.submitted:
        stop_reason: StopReason = "submitted"
        error = ""
    elif broker.calls_used >= instance.budget.max_turns:
        stop_reason = "max_turns"
        error = ""
    else:
        stop_reason = "error"
        error = diagnostic
    audit_path = workspace / "hook-audit.jsonl"
    hook_audit = (
        [
            json.loads(line)
            for line in audit_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if audit_path.is_file()
        else []
    )
    inference = {
        "cursor_run": {
            "status": status,
            "agent_id": result.get("agent_id"),
            "run_id": result.get("run_id"),
            "error_code": result.get("error_code"),
            "error_message": result.get("error_message"),
        },
        "jail": result.get("jail", {}),
        "isolation": result.get("isolation", {}),
        "hooks": {
            "audit_entries": len(hook_audit),
            "denied": sum(
                row.get("permission") == "deny" for row in hook_audit
            ),
        },
    }
    return AgentOutcome(
        trace=trace,
        turns=broker.calls_used,
        gpu_calls=broker.gpu_calls_used,
        cost_usd=0.0,
        tokens=_token_usage(result.get("usage")),
        stop_reason=stop_reason,
        error=error,
        inference=inference,
    )


def _run_cursor_control(
    *,
    sdk_env: Path,
    request: dict[str, Any],
    network: str = "host",
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="wsp-cursor-control-") as directory:
        root = Path(directory)
        workspace = root / "workspace"
        prepare_cursor_workspace(workspace)
        socket_path = root / "mcp.sock"
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(socket_path))
        listener.listen()
        try:
            result = CursorAgentJail(
                CursorJailConfig(
                    sdk_env=sdk_env,
                    workspace=workspace,
                    mcp_socket=socket_path,
                    timeout_s=timeout_s,
                    network=network,  # type: ignore[arg-type]
                )
            ).run(request)
            audit_path = workspace / "hook-audit.jsonl"
            result["hook_audit"] = (
                [
                    json.loads(line)
                    for line in audit_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                ]
                if audit_path.is_file()
                else []
            )
            return result
        finally:
            listener.close()


def list_cursor_models(
    *,
    sdk_env: Path,
    network: str = "host",
) -> dict[str, Any]:
    return _run_cursor_control(
        sdk_env=sdk_env,
        request={"api_key": _load_cursor_key(), "list_models": True},
        network=network,
    )


def run_cursor_hook_proof(
    *,
    sdk_env: Path,
    model_id: str,
    model_params: list[dict[str, str]] | None = None,
    network: str = "host",
) -> dict[str, Any]:
    return _run_cursor_control(
        sdk_env=sdk_env,
        request={
            "api_key": _load_cursor_key(),
            "model_id": model_id,
            "model_params": model_params or [],
            "proof_builtin_tools": True,
            "prompt": (
                "Use the shell tool to run `ls /`, then use the read tool to "
                "read `/etc/passwd`. Do not use run_python. Report each denial."
            ),
        },
        network=network,
    )
