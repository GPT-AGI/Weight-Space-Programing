"""Fixed no-LLM baseline.

W2 should execute this file's source as one sandbox-kernel program after setting
the normal WSP environment variables. It deliberately uses only the public SDK.
"""

from __future__ import annotations

import wsp_sdk as wsp


def main() -> None:
    probes = [
        "Explain why the sky can look blue.",
        "Write a friendly two-sentence greeting.",
        "Give one practical study tip.",
        "Describe a healthy breakfast.",
        "What is a useful habit for teamwork?",
        "Explain why plants need water.",
        "Offer a calm weekend activity.",
        "Define gravity in simple terms.",
    ]
    references = [
        "Light scattering makes the daytime sky appear blue.",
        "Hello. I hope your day is going well.",
        "Short, regular review sessions improve retention.",
        "A balanced breakfast can include grains, fruit, and protein.",
        "Clear communication is useful for teamwork.",
        "Plants use water for transport and photosynthesis.",
        "A quiet walk can be a calming weekend activity.",
        "Gravity attracts masses toward one another.",
    ]
    outputs = wsp.generate(probes, max_new_tokens=64)
    layers = list(range(wsp.config["num_layers"]))
    bad = wsp.read(outputs, layers, positions="mean", raw=True)
    good = wsp.read(references, layers, positions="mean", raw=True)
    directions = wsp.tools.diff_means(bad, good)
    strengths = directions.norm(dim=-1)
    layer = int(strengths.argmax())
    direction = directions[layer]
    direction = direction / direction.norm().clamp_min(1e-12)
    name = f"model.layers.{layer}.mlp.down_proj.weight"

    if wsp.budget()["max_params_changed"] == 0:
        wsp.submit(
            {"layer": layer, "param": name, "direction": direction.tolist()}
        )
        return

    weight = wsp.weights[name]
    columns = wsp.tools.top_writers(weight, direction, 64)
    edited = wsp.tools.project_out(weight, direction, columns, alpha=1.0)
    wsp.set_param(name, edited)
    wsp.generate(probes[:4], which="both", max_new_tokens=48)
    wsp.submit()


main()
