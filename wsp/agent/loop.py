"""Text-protocol agent loop and trace construction."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from wsp.agent.clients import LLMClient, LLMReply, usage_cost
from wsp.agent.prompts import FORMAT_REMINDER, system_prompt, task_prompt
from wsp.contracts import Instance, ModelSpec, StopReason, TokenUsage, Turn
from wsp.sandbox import SandboxKernel, format_execution_output

CompletionGate = Callable[[Callable[[], LLMReply]], LLMReply]


class GlobalCostExceeded(RuntimeError):
    pass


@dataclass
class AgentOutcome:
    trace: list[Turn]
    turns: int
    gpu_calls: int
    cost_usd: float
    tokens: TokenUsage
    stop_reason: StopReason
    error: str = ""
    inference: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Exchange:
    turn: int
    assistant: str
    tool: str
    code: str | None


_FENCE = re.compile(
    r"```(?P<language>[^\n`]*)\r?\n(?P<code>.*?)```", re.DOTALL
)


def extract_python_block(content: str) -> str | None:
    matches = list(_FENCE.finditer(content))
    if (
        len(matches) != 1
        or content.count("```") != 2
        or matches[0].group("language").strip() != "python"
    ):
        return None
    return matches[0].group("code")


def _summary(value: str, limit: int = 240) -> str:
    one_line = " ".join(value.split())
    if not one_line:
        return "[no output]"
    return one_line if len(one_line) <= limit else one_line[: limit - 1] + "…"


def build_context(
    system: str,
    task: str,
    exchanges: Sequence[_Exchange],
    *,
    keep_recent: int = 12,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": task},
    ]
    split = max(0, len(exchanges) - keep_recent)
    for exchange in exchanges[:split]:
        code = exchange.code if exchange.code is not None else "[format violation]"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": (
                        f"Earlier turn {exchange.turn}:\n```python\n{code}\n```"
                    ),
                },
                {
                    "role": "user",
                    "content": "Earlier execution summary: " + _summary(exchange.tool),
                },
            ]
        )
    for exchange in exchanges[split:]:
        messages.extend(
            [
                {"role": "assistant", "content": exchange.assistant},
                {
                    "role": "user",
                    "content": "Execution result:\n" + exchange.tool,
                },
            ]
        )
    return messages


def _gpu_state(
    value: object, maximum: int, previous_used: int
) -> tuple[int, bool]:
    if not isinstance(value, dict):
        return previous_used, previous_used >= maximum
    nested = value.get("gpu_calls")
    if isinstance(nested, dict):
        value = {**value, **nested}
    used = next(
        (
            int(value[key])
            for key in ("gpu_calls_used", "used_gpu_calls", "used")
            if isinstance(value.get(key), (int, float))
        ),
        previous_used,
    )
    remaining = next(
        (
            int(value[key])
            for key in (
                "gpu_calls_remaining",
                "remaining_gpu_calls",
                "gpu_remaining",
                "remaining",
            )
            if isinstance(value.get(key), (int, float))
        ),
        None,
    )
    if remaining is not None:
        used = max(used, maximum - remaining)
    return used, used >= maximum or remaining == 0


def _add_usage(total: TokenUsage, increment: TokenUsage) -> None:
    total.prompt += increment.prompt
    total.completion += increment.completion
    total.cached += increment.cached
    total.reasoning += increment.reasoning


class AgentLoop:
    def __init__(
        self,
        *,
        instance: Instance,
        model: ModelSpec,
        llm: LLMClient,
        sandbox: SandboxKernel,
        sdk_reference: str,
        completion_gate: CompletionGate | None = None,
        llm_max_tokens: int = 8_192,
    ) -> None:
        self.instance = instance
        self.model = model
        self.llm = llm
        self.sandbox = sandbox
        self.sdk_reference = sdk_reference
        self.completion_gate = completion_gate or (lambda call: call())
        self.llm_max_tokens = llm_max_tokens

    def run(self) -> AgentOutcome:
        started = time.monotonic()
        system = system_prompt(
            self.instance, sdk_reference=self.sdk_reference
        )
        task = task_prompt(self.instance)
        trace = [
            Turn(t=0, role="system", content=system),
            Turn(t=0, role="user", content=task),
        ]
        exchanges: list[_Exchange] = []
        tokens = TokenUsage()
        total_cost = 0.0
        gpu_calls = 0
        stop_reason: StopReason = "max_turns"
        error = ""

        for turn_number in range(1, self.instance.budget.max_turns + 1):
            if total_cost >= self.instance.budget.max_usd:
                stop_reason = "max_usd"
                break
            messages = build_context(system, task, exchanges)
            try:
                reply = self.completion_gate(
                    lambda messages=messages: self.llm.complete(
                        messages, max_tokens=self.llm_max_tokens
                    )
                )
            except GlobalCostExceeded:
                stop_reason = "max_usd"
                break
            except Exception as exc:
                stop_reason = "error"
                error = f"{type(exc).__name__}: {exc}"
                break

            turn_cost = usage_cost(reply.usage, self.model)
            total_cost += turn_cost
            _add_usage(tokens, reply.usage)
            code = extract_python_block(reply.content)
            trace.append(
                Turn(
                    t=turn_number,
                    role="assistant",
                    content=reply.content,
                    reasoning=reply.reasoning,
                    code=code,
                    usage=reply.usage,
                    cost_usd=turn_cost,
                    gpu_calls_used=gpu_calls,
                    wall_s=reply.latency_s,
                )
            )

            if total_cost >= self.instance.budget.max_usd:
                tool_output = (
                    "Episode dollar budget reached; this code was not executed."
                )
                trace.append(
                    Turn(
                        t=turn_number,
                        role="tool",
                        content=tool_output,
                        gpu_calls_used=gpu_calls,
                        wall_s=time.monotonic() - started,
                    )
                )
                exchanges.append(
                    _Exchange(turn_number, reply.content, tool_output, code)
                )
                stop_reason = "max_usd"
                break

            if code is None:
                tool_output = FORMAT_REMINDER
                tool_elapsed = 0.0
                submitted = False
                exhausted = False
            else:
                tool_started = time.monotonic()
                response = self.sandbox.execute(
                    code,
                    max_output_chars=max(
                        100_000, self.instance.budget.stdout_chars * 4
                    ),
                )
                tool_elapsed = time.monotonic() - tool_started
                tool_output = format_execution_output(
                    response, self.instance.budget.stdout_chars
                )
                submitted = bool(response.get("submitted"))
                gpu_calls, exhausted = _gpu_state(
                    response.get("budget"),
                    self.instance.budget.gpu_calls,
                    gpu_calls,
                )

            trace.append(
                Turn(
                    t=turn_number,
                    role="tool",
                    content=tool_output,
                    code=code,
                    gpu_calls_used=gpu_calls,
                    wall_s=tool_elapsed,
                )
            )
            exchanges.append(
                _Exchange(turn_number, reply.content, tool_output, code)
            )
            if submitted:
                stop_reason = "submitted"
                break
            if exhausted:
                stop_reason = "gpu_budget"
                break

        return AgentOutcome(
            trace=trace,
            turns=len(exchanges),
            gpu_calls=gpu_calls,
            cost_usd=total_cost,
            tokens=tokens,
            stop_reason=stop_reason,
            error=error,
        )
