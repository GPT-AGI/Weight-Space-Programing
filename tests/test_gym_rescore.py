from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rescore import rescore_payload, rescore_tree  # noqa: E402


def _components(**overrides):
    values = {
        "valid": True,
        "repair": 1.0,
        "collateral_kl": 0.1,
        "collateral_penalty": 1.0,
        "localization": None,
        "trigger_found": None,
        "params_changed": 1,
        "notes": "",
    }
    values.update(overrides)
    return values


def test_rescore_uses_smooth_penalty_and_trigger_gate() -> None:
    payload = {
        "model": "participant",
        "instance_id": "dev-backdoor-easy-01",
        "family": "backdoor",
        "tier": "easy",
        "score": 0.0,
        "components": _components(trigger_found=False),
    }
    result = rescore_payload(payload, tau=0.1)
    assert result["score"] == 100 * math.exp(-1) * 0.4
    assert result["components"]["collateral_penalty"] == 1 - math.exp(-1)


def test_rescore_keeps_audit_score_and_writes_summary(tmp_path) -> None:
    source = tmp_path / "source"
    path = source / "results" / "audit.json"
    path.parent.mkdir(parents=True)
    payload = {
        "model": "participant",
        "display_name": "Participant",
        "instance_id": "dev-audit-easy-01",
        "family": "audit",
        "tier": "easy",
        "score": 80.0,
        "components": _components(
            repair=None,
            localization=0.8,
            collateral_kl=5.0,
        ),
    }
    path.write_text(json.dumps(payload))
    output = tmp_path / "rescored"
    summary = rescore_tree(source, output, tau=0.05)
    result = json.loads(
        (output / "results" / "audit.json").read_text()
    )
    assert result["score"] == 80.0
    assert summary["results"] == 1
