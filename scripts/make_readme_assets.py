#!/usr/bin/env python3
"""Render the animated GIFs used by README.md / README.zh-CN.md into assets/.

    uv venv .venv-assets && uv pip install -p .venv-assets matplotlib pillow numpy
    .venv-assets/bin/python scripts/make_readme_assets.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "assets"

BG, PAPER, INK = "#F5F4EE", "#FAF9F5", "#141413"
ACCENT, GREEN, GRAY, SOFT = "#D97757", "#6A8F6B", "#8A867D", "#3D3B36"
LINE, BEIGE = "#E2DED3", "#EFE9DC"
SERIF, SANS, MONO = "DejaVu Serif", "DejaVu Sans", "DejaVu Sans Mono"
FIG_W, FIG_H, DPI = 9.6, 5.4, 100

LEADERBOARD_ROWS = [
    "Oracle (knows the planted edit)",
    "Scripted pipeline (no LLM)",
    "DeepSeek V4.1 Flash",
    "Qwen3.8 Flash",
    "GLM-5.3 Flash",
    "GPT-6 Luna",
    "Grok 4.1 Fast",
    "MiniMax M2.7",
    "Gemini 3.5 Flash-Lite",
    "No-op",
]
LAUNCH_DATE = "Nov 6"


def ease(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * x)


def seg(i: int, start: int, length: int) -> float:
    return ease((i - start) / length)


def canvas():
    fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    return fig, ax


def grab(fig) -> Image.Image:
    fig.canvas.draw()
    img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    return img


def save_gif(frames: list[Image.Image], name: str, fps: int = 20, hold: dict[int, int] | None = None) -> None:
    hold = hold or {}
    durations = [hold.get(i, 1000 // fps) for i in range(len(frames))]
    pal = [f.quantize(colors=96, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in frames]
    path = OUT / name
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True, disposal=1)
    print(f"{path.relative_to(OUT.parent)}  {len(frames)} frames  {path.stat().st_size / 1e6:.2f} MB")


def bar(ax, y: float, label: str, before: float, after: float, t: float, color: str, alpha: float = 1.0) -> None:
    x0, width, h = 0.55, 6.6, 0.34
    value = before + (after - before) * t
    ax.text(x0, y + 0.3, label, fontsize=12.5, color=SOFT, family=SANS, va="bottom", alpha=alpha)
    ax.add_patch(Rectangle((x0, y - h / 2), width, h, facecolor=LINE, edgecolor="none", alpha=alpha))
    ax.add_patch(Rectangle((x0, y - h / 2), width * value / 100, h, facecolor=color, edgecolor="none", alpha=alpha))
    bx = x0 + width * before / 100
    ax.plot([bx, bx], [y - h / 2 - 0.07, y + h / 2 + 0.07], color=INK, lw=1.4, alpha=0.55 * alpha)
    fmt = "{:.1f}%" if before % 1 or after % 1 else "{:.0f}%"
    ax.text(x0 + width + 0.18, y, f"{fmt.format(before)} → {fmt.format(value)}", fontsize=15, family=SANS,
            color=INK, va="center", alpha=alpha, fontweight="bold")


def decomposition() -> None:
    n = 130
    frames = []
    for i in range(n):
        fig, ax = canvas()
        ax.text(0.55, 4.88, "MATH-500: 7.9% → 48.1%", fontsize=26, family=SERIF, color=INK, va="center")
        ax.text(0.55, 4.42, "Qwen3.5-0.8B after GPT-6 Astra's weight edit, as reported", fontsize=12, color=GRAY,
                family=SANS, va="center")
        q = seg(i, 26, 10)
        if q > 0:
            question = ax.text(5.75, 4.86, "6x better at math?", fontsize=19, family=SERIF, style="italic",
                               color=ACCENT, va="center", alpha=q)
            s = seg(i, 96, 12)
            if s > 0:
                bb = question.get_window_extent(renderer=fig.canvas.get_renderer()).transformed(
                    ax.transData.inverted())
                ym = (bb.y0 + bb.y1) / 2
                ax.plot([bb.x0 - 0.05, bb.x0 - 0.05 + (bb.width + 0.1) * s], [ym, ym], color=INK, lw=2.4)
        bar(ax, 3.45, "MATH-500 accuracy", 7.9, 48.1, seg(i, 4, 24), INK)
        a = seg(i, 40, 8)
        if a > 0:
            t = seg(i, 48, 34)
            bar(ax, 2.35, "Finish rate (answer did not hit the 32k-token limit)", 7, 65, t, ACCENT, a)
            bar(ax, 1.25, "Accuracy among finished answers", 93, 71, t, GREEN, a)
        c = seg(i, 90, 10)
        if c > 0:
            ax.text(0.55, 0.52, "It stopped looping. It didn't get better at math.", fontsize=18, family=SERIF,
                    color=ACCENT, va="center", alpha=c)
        ax.text(0.55, 0.16, "Finish-rate split: our recount from the authors' public per-problem eval files "
                "(CC BY 4.0), first sample per problem.", fontsize=8.5, color=GRAY, family=SANS, va="center")
        frames.append(grab(fig))
    save_gif(frames, "decomposition.gif", hold={n - 1: 4500})


LOOP_STEPS = [
    ("Run the model", "GPU forward passes", "85 of 96 practice answers ran to the 32,768-token limit"),
    ("Read activations", "looping vs. productive text", "Contrast hidden states on looping and productive reasoning"),
    ("Find a direction", "difference of means", "One 'loop direction' u, written by the MLPs of layers 15 and 19"),
    ("Edit the weights", r"$W \leftarrow W - \alpha\,u u^{\top} W$", "64 output columns per layer: 131,072 weights"),
    ("Random control", "norm-matched direction", "Random direction, same norm: +1.6 ± 3.5 points. No effect"),
    ("Held-out check", "frozen edit, unseen problems", "Freeze the edit first, then confirm on problems never used"),
]


def loop() -> None:
    cx, cy, rx, ry = 4.8, 2.5, 1.95, 1.42
    angles = [math.radians(90 - 60 * k) for k in range(6)]
    pos = [(cx + rx * math.cos(a), cy + ry * math.sin(a)) for a in angles]
    dwell, travel = 13, 7
    per = dwell + travel
    n_steps = len(LOOP_STEPS) * per
    n = n_steps + 34
    frames = []
    for i in range(n):
        fig, ax = canvas()
        ax.text(0.55, 5.02, "What \u201cnative weight editing\u201d looked like", fontsize=21, family=SERIF, color=INK,
                va="center")
        ax.text(0.55, 4.62, "Experiment 8, reconstructed from the public logs", fontsize=11.5, color=GRAY,
                family=SANS, va="center")
        theta = np.linspace(0, 2 * math.pi, 240)
        ax.plot(cx + rx * np.cos(theta), cy + ry * np.sin(theta), color=LINE, lw=2.2, zorder=1)

        done = i >= n_steps
        step = min(i // per, len(LOOP_STEPS) - 1)
        local = i - step * per
        if not done and local >= dwell:
            f = ease((local - dwell) / travel)
            a = angles[step] - math.radians(60) * f
            ax.add_patch(plt.Circle((cx + rx * math.cos(a), cy + ry * math.sin(a)), 0.085, color=ACCENT, zorder=3))

        for k, ((title, sub, _), (x, y)) in enumerate(zip(LOOP_STEPS, pos)):
            active = (not done) and k == step and local < dwell
            face = ACCENT if active else PAPER
            edge = ACCENT if active else LINE
            fg = "white" if active else INK
            sub_fg = "white" if active else GRAY
            ax.add_patch(FancyBboxPatch((x - 1.08, y - 0.33), 2.16, 0.66, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        facecolor=face, edgecolor=edge, lw=1.6, zorder=4))
            ax.text(x, y + 0.11, title, fontsize=12, family=SANS, fontweight="bold", color=fg, ha="center",
                    va="center", zorder=5)
            ax.text(x, y - 0.15, sub, fontsize=9.5, family=SANS, color=sub_fg, ha="center", va="center", zorder=5)

        if done:
            c = seg(i, n_steps, 8)
            ax.text(cx, cy + 0.2, "Not intuition.", fontsize=19, family=SERIF, color=ACCENT, ha="center",
                    va="center", alpha=c)
            ax.text(cx, cy - 0.22, "An experiment loop.", fontsize=19, family=SERIF, color=ACCENT, ha="center",
                    va="center", alpha=c)
            ax.text(0.55, 0.3, "Every step is a standard 2022\u20132024 interpretability tool.", fontsize=12.5,
                    family=SANS, color=SOFT, va="center", alpha=c)
        else:
            ax.text(cx, cy + 0.14, "weight-space", fontsize=14, family=SERIF, color=GRAY, ha="center", va="center")
            ax.text(cx, cy - 0.16, "programming", fontsize=14, family=SERIF, color=GRAY, ha="center", va="center")
            ax.text(0.55, 0.3, f"{step + 1}/6  {LOOP_STEPS[step][2]}", fontsize=12.5, family=SANS, color=SOFT,
                    va="center")
        frames.append(grab(fig))
    save_gif(frames, "loop.gif", hold={n - 1: 3500})


def projection() -> None:
    rng = np.random.default_rng(7)
    k = 13
    ang = rng.uniform(0, 2 * math.pi, k)
    length = rng.uniform(0.55, 1.0, k)
    cols = np.stack([length * np.cos(ang), length * np.sin(ang)], axis=1)
    u = np.array([math.cos(math.radians(28)), math.sin(math.radians(28))])
    ox, oy, scale = 2.75, 2.45, 1.78

    def to_xy(v):
        return ox + scale * v[0], oy + scale * v[1]

    up, hold_top, down, hold_bot = 40, 22, 26, 12
    alphas = ([0.0] * hold_bot + [ease(j / up) for j in range(up + 1)] + [1.0] * hold_top
              + [1 - ease(j / down) for j in range(down + 1)])
    frames = []
    for i, alpha in enumerate(alphas):
        fig, ax = canvas()
        ax.text(0.55, 5.02, "The fix is one projection formula", fontsize=21, family=SERIF, color=INK, va="center")
        ax.add_patch(Rectangle((ox - scale * 1.18, oy - scale * 1.18), scale * 2.36, scale * 2.36, facecolor=PAPER,
                               edgecolor=LINE, lw=1.2))
        p0, p1 = to_xy(-1.15 * u), to_xy(1.15 * u)
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=ACCENT, lw=1.6, ls=(0, (5, 4)), zorder=2)
        perp = np.array([-u[1], u[0]])
        q0, q1 = to_xy(-1.15 * perp), to_xy(1.15 * perp)
        ax.plot([q0[0], q1[0]], [q0[1], q1[1]], color=LINE, lw=1.0, zorder=1)
        ax.annotate("", xy=to_xy(0.98 * u), xytext=to_xy(0 * u),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT, lw=2.6, mutation_scale=16), zorder=6)
        lx, ly = to_xy(1.0 * u)
        ax.text(lx - 0.1, ly + 0.2, "u  (loop direction)", fontsize=11, color=ACCENT, family=SANS, fontweight="bold",
                ha="right", va="center")
        for w in cols:
            ax.annotate("", xy=to_xy(w), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.2, mutation_scale=9), zorder=3)
            w2 = w - alpha * (u @ w) * u
            ax.annotate("", xy=to_xy(w2), xytext=(ox, oy),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.7, mutation_scale=11), zorder=5)
        ax.text(ox - scale * 1.18, oy - scale * 1.18 - 0.22, "each arrow = one MLP output column of W", fontsize=9.5,
                color=GRAY, family=SANS, va="center")

        tx = 5.55
        ax.text(tx, 3.95, r"$W \leftarrow W - \alpha\,u\,u^{\top} W$", fontsize=25, color=INK, va="center")
        ax.text(tx, 3.38, f"\u03b1 = {alpha:0.2f}", fontsize=17, family=MONO, color=ACCENT, va="center")
        bullets = [
            "Removes each column's component along u",
            "Everything orthogonal to u is untouched",
            "Exp. 8: 64 columns \u00d7 2 layers \u00d7 1,024 dims",
            "     = 131,072 weights (0.016%)",
            "Same family as refusal-direction",
            "     ablation (2024)",
        ]
        for j, b in enumerate(bullets):
            ax.text(tx, 2.72 - 0.34 * j, b, fontsize=12, color=SOFT, family=SANS, va="center")
        frames.append(grab(fig))
    save_gif(frames, "projection.gif", hold={hold_bot + up: 900})


def hero() -> None:
    subtitle = "Who is the best weight-space programmer?"
    n_type, n_loop = 26, 56
    n = n_type + n_loop
    x_label, x_bar, bar_w, bar_h = 0.55, 3.75, 4.35, 0.17
    y0, dy = 3.36, 0.276
    frames = []
    for i in range(n):
        fig, ax = canvas()
        ax.text(0.55, 4.95, "WSP-Bench", fontsize=34, family=SERIF, color=INK, va="center")
        shown = subtitle[: int(len(subtitle) * min(1.0, i / (n_type - 4)))]
        cursor = "\u258c" if i < n_type and (i // 3) % 2 == 0 else ""
        ax.text(0.57, 4.42, shown + cursor, fontsize=17, family=SANS, color=ACCENT, va="center")
        ax.text(0.57, 4.02, "We plant a known defect in Qwen3.5-0.8B. Every LLM gets the same harness, budget and "
                "offline sandbox.", fontsize=10.5, family=SANS, color=GRAY, va="center")
        ax.text(x_bar + bar_w + 0.42, y0 + 0.3, "score", fontsize=10, family=SANS, color=GRAY, ha="center")
        for r, name in enumerate(LEADERBOARD_ROWS):
            y = y0 - dy * r
            known = r == 0 or r == len(LEADERBOARD_ROWS) - 1
            ax.text(x_label, y, name, fontsize=11.5, family=SANS, color=INK if not known else SOFT, va="center")
            ax.add_patch(Rectangle((x_bar, y - bar_h / 2), bar_w, bar_h, facecolor=LINE, edgecolor="none"))
            if r == 0:
                ax.add_patch(Rectangle((x_bar, y - bar_h / 2), bar_w, bar_h, facecolor=GREEN, edgecolor="none"))
                score, color = "100", GREEN
            elif known:
                score, color = "0", GRAY
            else:
                phase = ((i - n_type) / n_loop + 0.11 * r) % 1.0 if i >= n_type else 0.11 * r
                center = -0.25 + 1.5 * phase
                for s in range(14):
                    f = center + (s - 7) * 0.02
                    if 0 <= f <= 1:
                        alpha = max(0.0, 1 - abs(s - 7) / 7.5)
                        ax.add_patch(Rectangle((x_bar + f * bar_w, y - bar_h / 2), bar_w * 0.02, bar_h,
                                               facecolor=BEIGE if alpha < 0.5 else "white", edgecolor="none",
                                               alpha=alpha))
                score, color = "?", ACCENT
            ax.text(x_bar + bar_w + 0.42, y, score, fontsize=13, family=SANS, fontweight="bold", color=color,
                    ha="center", va="center")
        ax.text(0.55, 0.36, "Exact ground truth \u00b7 every trace public \u00b7 $ cost per run \u00b7 a no-LLM baseline",
                fontsize=10.5, family=SANS, color=SOFT, va="center")
        ax.text(9.05, 0.36, f"Launching {LAUNCH_DATE}", fontsize=13, family=SANS, fontweight="bold", color=ACCENT,
                ha="right", va="center")
        frames.append(grab(fig))
    save_gif(frames, "hero.gif", hold={n_type - 1: 700})


def broken_qwen() -> None:
    import json
    import textwrap

    data = json.loads((OUT / "data" / "broken_qwen.json").read_text())
    examples = data["examples"]
    per_example = 34
    frames = []
    for i in range(per_example * len(examples)):
        example_index = min(i // per_example, len(examples) - 1)
        local = i % per_example
        example = examples[example_index]
        progress = ease(min(1.0, local / 24))
        fig, ax = canvas()
        ax.text(0.55, 5.00, "A planted weight edit, visible in the output", fontsize=21, family=SERIF,
                color=INK, va="center")
        prompt = textwrap.shorten(example["prompt"], width=92, placeholder="\u2026")
        ax.text(0.55, 4.57, f"Prompt {example_index + 1}/3  \u201c{prompt}\u201d", fontsize=11.5, family=SANS,
                color=GRAY, va="center")
        cards = [
            (0.55, "CLEAN", GREEN, example["clean"]),
            (4.98, "BROKEN", ACCENT, example["broken"]),
        ]
        for x, label, color, value in cards:
            ax.add_patch(FancyBboxPatch((x, 1.12), 4.05, 3.05, boxstyle="round,pad=0.02,rounding_size=0.10",
                                        facecolor=PAPER, edgecolor=LINE, lw=1.3))
            ax.text(x + 0.22, 3.88, label, fontsize=10.5, family=SANS, fontweight="bold", color=color, va="center")
            visible = value[: int(len(value) * progress)]
            wrapped = "\n".join(textwrap.wrap(visible, width=43))[:520]
            cursor = "\u258c" if local < 25 and (local // 3) % 2 == 0 else ""
            ax.text(x + 0.22, 3.57, wrapped + cursor, fontsize=10.4, family=MONO, color=SOFT, va="top",
                    linespacing=1.35)
        ax.text(0.55, 0.63, f"Real Qwen3.5-0.8B output. The planted edit changed "
                f"{data['weights_changed']:,} bfloat16 weights.", fontsize=11.2, family=SANS, color=INK,
                va="center")
        ax.text(0.55, 0.29, "Same prompt, greedy decoding, thinking disabled. Synthetic benchmark defect.",
                fontsize=9.5, family=SANS, color=GRAY, va="center")
        frames.append(grab(fig))
    save_gif(frames, "broken_qwen.gif", fps=18,
             hold={per_example - 1: 1100, per_example * 2 - 1: 1100, per_example * 3 - 1: 2600})


def main() -> None:
    OUT.mkdir(exist_ok=True)
    targets = {"hero": hero, "decomposition": decomposition, "loop": loop, "projection": projection,
               "broken_qwen": broken_qwen}
    for name in sys.argv[1:] or targets:
        targets[name]()


if __name__ == "__main__":
    main()
