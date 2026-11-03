from __future__ import annotations

from dataclasses import asdict

import pytest

from wsp.agent.clients import (
    FakeLLMClient,
    LLMReply,
    _non_reasoning_completion,
    _reasoning_content,
    usage_cost,
)
from wsp.agent.loop import AgentLoop
from wsp.agent.prompts import FORMAT_REMINDER
from wsp.contracts import Budget, Instance, ModelSpec, TokenUsage, Turn


class FakeSandbox:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def execute(self, code, **_kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        return {
            "ok": True,
            "stdout": "",
            "stderr": "",
            "traceback": "",
            "submitted": False,
            "budget": {"gpu_calls_used": 0, "gpu_calls_remaining": 20},
            **response,
        }


def _instance(budget: Budget) -> Instance:
    return Instance(
        id="dev-behavior-easy-01",
        split="dev",
        family="behavior",
        tier="easy",
        brief="Repair the observed behavior.",
        submission="edit",
        budget=budget,
    )


def _model() -> ModelSpec:
    return ModelSpec(
        id="test/model",
        display_name="Test",
        usd_per_m_input=1.0,
        usd_per_m_output=2.0,
        usd_per_m_cached=0.25,
    )


def _reply(content: str, usage: TokenUsage | None = None) -> LLMReply:
    return LLMReply(content, None, usage or TokenUsage(), 0.01)


def test_format_violation_counts_turn_then_recovers():
    sandbox = FakeSandbox([{"submitted": True}])
    loop = AgentLoop(
        instance=_instance(Budget(max_turns=3)),
        model=_model(),
        llm=FakeLLMClient(
            [_reply("not a code block"), _reply("```python\nwsp.submit()\n```")]
        ),
        sandbox=sandbox,
        sdk_reference="test reference",
    )
    outcome = loop.run()
    assert outcome.stop_reason == "submitted"
    assert outcome.turns == 2
    assert sandbox.calls == 1
    assert any(turn.content == FORMAT_REMINDER for turn in outcome.trace)


def test_turn_and_gpu_budget_stops():
    turn_sandbox = FakeSandbox([{}])
    turn_outcome = AgentLoop(
        instance=_instance(Budget(max_turns=1)),
        model=_model(),
        llm=FakeLLMClient([_reply("```python\nprint('x')\n```")]),
        sandbox=turn_sandbox,
        sdk_reference="ref",
    ).run()
    assert turn_outcome.stop_reason == "max_turns"
    assert turn_outcome.turns == 1

    gpu_sandbox = FakeSandbox(
        [{"budget": {"gpu_calls_used": 1, "gpu_calls_remaining": 0}}]
    )
    gpu_outcome = AgentLoop(
        instance=_instance(Budget(max_turns=3, gpu_calls=1)),
        model=_model(),
        llm=FakeLLMClient([_reply("```python\nwsp.generate(['x'])\n```")]),
        sandbox=gpu_sandbox,
        sdk_reference="ref",
    ).run()
    assert gpu_outcome.stop_reason == "gpu_budget"
    assert gpu_outcome.gpu_calls == 1


def test_usd_stop_does_not_execute_over_budget_reply():
    sandbox = FakeSandbox([])
    outcome = AgentLoop(
        instance=_instance(Budget(max_turns=3, max_usd=0.001)),
        model=_model(),
        llm=FakeLLMClient(
            [
                _reply(
                    "```python\nraise AssertionError('must not run')\n```",
                    TokenUsage(completion=1_000),
                )
            ]
        ),
        sandbox=sandbox,
        sdk_reference="ref",
    ).run()
    assert outcome.stop_reason == "max_usd"
    assert outcome.cost_usd == pytest.approx(0.002)
    assert sandbox.calls == 0


def test_cost_math_and_trace_schema_round_trip():
    usage = TokenUsage(prompt=1_000, completion=500, cached=400, reasoning=300)
    assert usage_cost(usage, _model()) == pytest.approx(0.0023)

    original = Turn(
        t=1,
        role="assistant",
        content="```python\nprint(1)\n```",
        reasoning="reason",
        code="print(1)",
        usage=usage,
        cost_usd=0.0017,
        gpu_calls_used=2,
        wall_s=0.25,
    )
    raw = asdict(original)
    raw["usage"] = TokenUsage(**raw["usage"])
    restored = Turn(**raw)
    assert restored == original


def test_nonstreaming_reasoning_details_are_recorded():
    class Message:
        reasoning_content = None
        reasoning_details = [{"text": "first "}, {"text": "second"}]
        model_extra = {}

    assert _reasoning_content(Message()) == "first second"
    assert _non_reasoning_completion(
        prompt=60, completion=59, reasoning=48, total=119
    ) == 11
    assert _non_reasoning_completion(
        prompt=208, completion=11, reasoning=327, total=546
    ) == 11
