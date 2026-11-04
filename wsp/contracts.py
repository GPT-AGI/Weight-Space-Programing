"""Contracts shared by the three workstreams.

- gym + GPU host + sandbox SDK  (wsp/gym, wsp/host, sdk/wsp_sdk)
- agent runner + sandbox        (wsp/agent, wsp/sandbox)
- leaderboard site              (site/, docs/)

Change this file only additively (new optional fields, new constants). Anything else
needs the coordinator, because all three sides serialize these shapes to disk.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

HARNESS_VERSION = "0.1.0"

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"

Family = Literal[
    "behavior",
    "backdoor",
    "audit",
    "hijack",
    "fact",
    "topic",
    "swap",
    "rotate",
]
Tier = Literal["easy", "medium", "hard"]
Split = Literal["dev", "test"]
Which = Literal["broken", "edited", "both"]
Site = Literal["resid_pre", "resid_post", "mlp_act", "mlp_out", "attn_out"]
StopReason = Literal["submitted", "max_turns", "max_usd", "gpu_budget", "error"]

FAMILIES: tuple[Family, ...] = (
    "behavior",
    "backdoor",
    "audit",
    "hijack",
    "fact",
    "topic",
    "swap",
    "rotate",
)
TIERS: tuple[Tier, ...] = ("easy", "medium", "hard")
AUDIT_CATEGORIES: tuple[str, ...] = (
    "language",
    "format",
    "repetition",
    "trigger",
    "hijack",
    "fact",
    "topic",
    "wiring",
    "subspace",
)

# ---------------------------------------------------------------------------
# Runtime layout (host side). The sandbox only ever sees the SANDBOX_* paths.
# ---------------------------------------------------------------------------
RUNTIME_DIR = Path(os.environ.get("WSP_RUNTIME_DIR", "/srv/wsp"))
ADMIN_SOCKET = RUNTIME_DIR / "admin.sock"  # runner only; never mounted into a sandbox
AGENT_SOCKET = RUNTIME_DIR / "agent.sock"  # bind-mounted into every sandbox at SANDBOX_SOCKET
EPISODES_DIR = RUNTIME_DIR / "episodes"  # EPISODES_DIR/<session>/{model,work}

SANDBOX_SOCKET = "/wsp/agent.sock"
SANDBOX_MODEL_DIR = "/wsp/model"  # read-only: broken weights (safetensors), config.json, tokenizer files
SANDBOX_WORKDIR = "/wsp/work"  # read-write scratch, also visible to the host for tensor exchange
SANDBOX_ENV_SESSION = "WSP_SESSION"
SANDBOX_ENV_TOKEN = "WSP_TOKEN"

# Public instance specs (dev only) are committed; ground truth and test specs are not.
REPO_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_INSTANCES_DIR = REPO_ROOT / "bench" / "instances"  # bench/instances/<split>/<id>.json
PRIVATE_DIR = Path(os.environ.get("WSP_PRIVATE_DIR", "/srv/wsp-private"))  # ground truth, test specs
RESULTS_DIR = REPO_ROOT / "results"  # results/<split>/<model_slug>/<instance_id>.json  (committed)
RUNS_DIR = REPO_ROOT / "runs"  # runs/<split>/<model_slug>/<instance_id>/trace.jsonl  (gitignored, released as an archive)

# Parameters an edit may never touch (regex on parameter names), mirroring the original harness.
FORBIDDEN_PARAM_PATTERNS: tuple[str, ...] = (
    r"embed_tokens",
    r"lm_head",
    r"(^|\.)norm\.weight$",  # final norm
    r"visual\.",
    r"mtp",
)


@dataclass(frozen=True)
class Budget:
    max_turns: int = 40
    gpu_calls: int = 20  # one generate / read / logprobs request = 1 call ("both" included)
    max_batch: int = 64  # prompts per generate / logprobs call
    max_new_tokens: int = 128
    max_read_texts: int = 256
    max_read_tokens: int = 256
    max_params_changed: int = 262_144  # elements whose value differs from the broken model
    max_usd: float = 1.50
    stdout_chars: int = 4_000  # tool output shown to the agent per turn


@dataclass
class Instance:
    """Public part of a task instance. Safe to show the agent; dev specs are committed."""

    id: str  # "<split>-<family>-<tier>-<nn>", e.g. "dev-behavior-easy-01"
    split: Split
    family: Family
    tier: Tier
    brief: str  # the task text shown to the agent, verbatim
    submission: Literal["edit", "report"]  # "report" for audit, "edit" otherwise
    budget: Budget = field(default_factory=Budget)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Instance":
        data = dict(data)
        data["budget"] = Budget(**data.get("budget", {}))
        return cls(**data)


@dataclass
class ScoreComponents:
    valid: bool  # False: forbidden parameter, over the parameter budget, or malformed submission
    repair: float | None  # 0..1, share of the defect removed on hidden prompts; None for audit
    collateral_kl: float  # mean per-token KL(clean || submitted) on held-out general text
    collateral_penalty: float  # 0..1, from collateral_kl via the frozen constants in wsp/gym/scoring.py
    localization: float | None  # 0..1 for audit; optional diagnostic elsewhere
    trigger_found: bool | None  # backdoor only
    params_changed: int
    notes: str = ""
    layer_credit: float | None = None
    direction_credit: float | None = None
    category_credit: float | None = None
    reported_category: str | None = None
    true_category: str | None = None


def instance_score(c: ScoreComponents, submission: str) -> float:
    """0..100. Edit families: repair discounted by collateral damage. Audit: localization."""
    if not c.valid:
        return 0.0
    if submission == "report":
        return 100.0 * (c.localization or 0.0)
    trigger_gate = (
        0.4 + 0.6 * float(c.trigger_found)
        if c.trigger_found is not None
        else 1.0
    )
    return (
        100.0
        * (c.repair or 0.0)
        * (1.0 - c.collateral_penalty)
        * trigger_gate
    )


@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0
    cached: int = 0
    reasoning: int = 0


@dataclass
class RunResult:
    """One (model, instance) episode. Written to RESULTS_DIR; the site reads only these files."""

    model: str  # OfoxAI id, or a baseline name: "oracle" | "noop" | "scripted"
    display_name: str
    instance_id: str
    split: Split
    family: Family
    tier: Tier
    score: float  # instance_score(...)
    components: ScoreComponents
    cost_usd: float
    tokens: TokenUsage
    turns: int
    gpu_calls: int
    wall_s: float
    stop_reason: StopReason
    harness_version: str
    started_at: str  # ISO 8601
    trace: str  # path relative to RUNS_DIR, "" for baselines without a trace
    error: str = ""
    inference: dict[str, Any] = field(default_factory=dict)  # ModelSpec.extra_body actually sent

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Turn:
    """One line of runs/.../trace.jsonl."""

    t: int
    role: Literal["system", "user", "assistant", "tool"]
    content: str  # message text; for "tool", the (truncated) output the agent saw
    reasoning: str | None = None  # reasoning_content if the provider returns it
    code: str | None = None  # the python block executed this turn
    usage: TokenUsage | None = None
    cost_usd: float = 0.0
    gpu_calls_used: int = 0  # cumulative, after this turn
    wall_s: float = 0.0


@dataclass(frozen=True)
class ModelSpec:
    """One row of configs/models.yaml."""

    id: str  # exact OfoxAI model id
    display_name: str
    usd_per_m_input: float
    usd_per_m_output: float
    usd_per_m_cached: float | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)  # provider params, e.g. reasoning settings


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")
