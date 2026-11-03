"""Prompts shared identically by every evaluated LLM."""

from __future__ import annotations

from pathlib import Path

from wsp.contracts import REPO_ROOT, Budget, Instance

FORMAT_REMINDER = (
    "Format error: reply with exactly one fenced ```python code block. "
    "This turn counted against the budget and no code was executed."
)

MISSING_REFERENCE = """\
The detailed SDK reference is not present in this checkout yet. The available
top-level API is: wsp.config, wsp.weights, wsp.tokenizer, wsp.generate,
wsp.logprobs, wsp.read, wsp.set_param, wsp.reset_edit, wsp.edit_summary,
wsp.budget, wsp.submit, and wsp.tools.
"""


def load_sdk_reference(path: Path | None = None) -> str:
    reference = path or (REPO_ROOT / "sdk" / "REFERENCE.md")
    if not reference.is_file():
        return MISSING_REFERENCE
    return reference.read_text(encoding="utf-8")


def system_prompt(
    instance: Instance,
    *,
    sdk_reference: str,
) -> str:
    budget: Budget = instance.budget
    submission = (
        "This is an audit task. Do not edit weights. Submit a localization report "
        "with wsp.submit(report={...}) using the fields requested by the task."
        if instance.submission == "report"
        else "Submit the final edited model with wsp.submit(), including a report "
        "only when the task asks for one."
    )
    return f"""\
You are the weight-space programmer in one WSP-Bench episode. You receive only
the broken model inside an offline Python sandbox. Diagnose the stated defect,
make only justified edits when edits are allowed, and submit within the budget.

Every assistant reply must contain exactly one fenced ```python code block.
The block executes in a persistent Python namespace where `import wsp_sdk as
wsp` has already run. Text outside the block is allowed but is not executed.
A malformed reply consumes one turn. Tool output may be truncated. You may
maintain durable notes at /wsp/work/wsp_notes.md.

Budget:
- turns: {budget.max_turns}
- GPU calls: {budget.gpu_calls}
- prompts per call: {budget.max_batch}
- generated tokens per call: {budget.max_new_tokens}
- changed parameter elements: {budget.max_params_changed}
- episode LLM cost: ${budget.max_usd:.4f}

Scoring:
- edit tasks score repair rate times (1 - collateral penalty)
- audit tasks score localization accuracy
- malformed, forbidden, or over-budget submissions score zero
Run relevant controls before claiming a conclusion, and prefer evidence that
separates the proposed cause from plausible alternatives.

{submission}

SDK reference (verbatim):
--- BEGIN SDK REFERENCE ---
{sdk_reference.rstrip()}
--- END SDK REFERENCE ---
"""


def task_prompt(instance: Instance) -> str:
    return instance.brief
