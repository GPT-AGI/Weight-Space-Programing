from __future__ import annotations

import json

from tests.mock_host import MockHost, build_stub_sandbox_env
from wsp.agent import runner
from wsp.contracts import Budget, Instance, model_slug


def test_end_to_end_episode_and_resume(tmp_path, monkeypatch):
    environment = build_stub_sandbox_env(tmp_path)
    public_instances = tmp_path / "bench" / "instances"
    dev = public_instances / "dev"
    dev.mkdir(parents=True)
    instance = Instance(
        id="dev-behavior-easy-01",
        split="dev",
        family="behavior",
        tier="easy",
        brief="Repair the mock defect and submit.",
        submission="edit",
        budget=Budget(max_turns=3),
    )
    (dev / f"{instance.id}.json").write_text(
        json.dumps(instance.to_json()), encoding="utf-8"
    )
    fake_replies = tmp_path / "replies.json"
    fake_replies.write_text(
        json.dumps(
            [
                {
                    "content": "```python\nprint(wsp.budget())\nwsp.submit()\n```",
                    "reasoning_content": "Submit the mock episode.",
                    "usage": {"prompt": 100, "completion": 20, "cached": 10},
                }
            ]
        ),
        encoding="utf-8",
    )
    results = tmp_path / "results"
    runs = tmp_path / "runs"
    model_id = "deepseek/deepseek-v4.1-flash-1017"

    with MockHost(tmp_path / "runtime") as mock:
        monkeypatch.setattr(runner, "PUBLIC_INSTANCES_DIR", public_instances)
        monkeypatch.setattr(runner, "RESULTS_DIR", results)
        monkeypatch.setattr(runner, "RUNS_DIR", runs)
        monkeypatch.setattr(runner, "ADMIN_SOCKET", mock.admin_socket)
        monkeypatch.setattr(runner, "AGENT_SOCKET", mock.agent_socket)
        arguments = [
            "--split",
            "dev",
            "--models",
            model_id,
            "--instances",
            instance.id,
            "--concurrency",
            "2",
            "--fake-llm",
            str(fake_replies),
            "--sandbox-env",
            str(environment),
        ]
        assert runner.main(arguments) == 0
        assert mock.open_count == 1
        assert runner.main([*arguments, "--resume"]) == 0
        assert mock.open_count == 1

    result_path = (
        results / "dev" / model_slug(model_id) / f"{instance.id}.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["score"] == 72.0
    assert result["stop_reason"] == "submitted"
    assert result["turns"] == 1
    assert result["inference"] == {}
    assert result["tokens"] == {
        "prompt": 100,
        "completion": 20,
        "cached": 10,
        "reasoning": 0,
    }
    trace_path = runs / result["trace"]
    trace = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [turn["role"] for turn in trace] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert trace[2]["reasoning"] == "Submit the mock episode."
