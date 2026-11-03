from __future__ import annotations

import json
import socket
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest
import torch
from safetensors.torch import save_file

from wsp.contracts import Budget, Instance
from wsp.gym.defects import MaterializedDefect
from wsp.gym.instances import Recipe
from wsp.host.model import ModelEngine
from wsp.host.server import (
    APIError,
    Handler,
    HostService,
    Session,
    _UnixHTTPServer,
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.down_proj = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer])


def test_apply_revert_restores_exact_bfloat16_bits() -> None:
    model = TinyModel()
    engine = ModelEngine(model, tokenizer=None, device="cpu")
    name = "model.layers.0.mlp.down_proj.weight"
    original = engine.params[name].detach().clone()
    defect = {name: torch.full((3, 4), 0.013, dtype=torch.float32)}
    replacement = torch.full((3, 4), 0.25, dtype=torch.float32)
    with engine.applied(defect):
        assert not torch.equal(engine.params[name], original)
    assert torch.equal(engine.params[name], original)
    with engine.applied(defect, {name: replacement}):
        assert torch.equal(
            engine.params[name], replacement.to(torch.bfloat16)
        )
    assert torch.equal(engine.params[name], original)


class FakeEngine:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.layers = [object()] * 24
        self.params = {
            "model.layers.3.mlp.down_proj.weight": torch.nn.Parameter(
                torch.zeros((2, 3), dtype=torch.bfloat16)
            )
        }
        self.generate_runs = 0

    @contextmanager
    def applied(
        self, defect: dict[str, torch.Tensor], edit: dict[str, torch.Tensor] | None = None
    ) -> Iterator[None]:
        del defect, edit
        yield

    def generate(self, prompts: list[str], **_: Any) -> list[str]:
        self.generate_runs += 1
        return [f"out:{prompt}" for prompt in prompts]

    def broken_value(
        self, name: str, defect: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        value = self.params[name].detach().float()
        return (value + defect.get(name, 0)).to(torch.bfloat16).float()


def make_session(tmp_path: Path, *, gpu_calls: int = 1) -> tuple[HostService, Session]:
    engine = FakeEngine()
    service = HostService(engine, episodes_dir=tmp_path, private_dir=tmp_path)
    instance = Instance(
        id="dev-behavior-easy-01",
        split="dev",
        family="behavior",
        tier="easy",
        brief="repair",
        submission="edit",
        budget=Budget(gpu_calls=gpu_calls, max_batch=4, max_params_changed=4),
    )
    recipe = Recipe(
        instance_id=instance.id,
        family="behavior",
        tier="easy",
        seed=1,
        layers=[3],
        mode="keyword",
        target="BUG",
        strength=1,
        columns=1,
        hidden_prompts=["x"],
        clean_prompts=["y"],
        collateral_texts=["z"],
    )
    root = tmp_path / "episode"
    (root / "work").mkdir(parents=True)
    defect = MaterializedDefect(
        {"model.layers.3.mlp.down_proj.weight": torch.ones((2, 3))},
        [3],
        "model.layers.3.mlp.down_proj.weight",
        [1.0, 0.0],
    )
    session = Session("s", "secret", instance, recipe, defect, root)
    service.sessions["s"] = session
    return service, session


def test_which_both_costs_one_call_and_budget_uses_429(tmp_path: Path) -> None:
    service, session = make_session(tmp_path)
    result = service.generate(
        session,
        {"prompts": ["a"], "which": "both", "max_new_tokens": 8},
    )
    assert set(result["outputs"]) == {"broken", "edited"}
    assert session.gpu_calls == 1
    assert service.engine.generate_runs == 2
    with pytest.raises(APIError, match="budget") as exc:
        service.generate(
            session, {"prompts": ["b"], "max_new_tokens": 8}
        )
    assert exc.value.status == 429


def test_edit_validation_counts_against_broken_model(tmp_path: Path) -> None:
    service, session = make_session(tmp_path)
    name = "model.layers.3.mlp.down_proj.weight"
    broken = torch.ones((2, 3), dtype=torch.float32)
    broken[0, 0] = 2
    save_file({name: broken}, session.work_dir / "edit.safetensors")
    result = service.edit(session, {"path": "edit.safetensors"})
    assert result["valid"]
    assert result["params_changed"] == 1

    save_file({name: torch.zeros((2, 3))}, session.work_dir / "too-big.safetensors")
    result = service.edit(session, {"path": "too-big.safetensors"})
    assert not result["valid"]
    assert "limit" in result["reasons"][0]


def _request(path: Path, token: str) -> tuple[int, dict[str, Any]]:
    body = b'{"session":"s"}'
    request = (
        "POST /v1/status HTTP/1.1\r\nHost: wsp\r\n"
        f"Authorization: Bearer {token}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    ).encode() + body
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(path))
        client.sendall(request)
        raw = b""
        while chunk := client.recv(65536):
            raw += chunk
    head, payload = raw.split(b"\r\n\r\n", 1)
    status = int(head.splitlines()[0].split()[1])
    return status, json.loads(payload)


def test_http_bearer_auth(tmp_path: Path) -> None:
    service, _session = make_session(tmp_path)
    socket_path = tmp_path / "agent.sock"
    server = _UnixHTTPServer(str(socket_path), Handler, service, False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert _request(socket_path, "wrong")[0] == 401
        status, payload = _request(socket_path, "secret")
        assert status == 200
        assert payload["budget"]["gpu_calls_remaining"] == 1
    finally:
        server.shutdown()
        server.server_close()
