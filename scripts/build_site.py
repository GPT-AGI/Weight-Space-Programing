#!/usr/bin/env python3
"""Build the WSP-Bench GitHub Pages site from public run artifacts."""

from __future__ import annotations

import argparse
import html
import json
import math
import random
import re
import shutil
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PALETTE = {
    "bg": "#F5F4EE",
    "paper": "#FAF9F5",
    "ink": "#141413",
    "accent": "#D97757",
    "green": "#6A8F6B",
    "gray": "#8A867D",
    "line": "#E2DED3",
    "beige": "#EFE9DC",
}
FAMILIES = ("behavior", "backdoor", "audit", "hijack")
BASELINES = ("oracle", "noop", "scripted")
SITE_URL = "https://gpt-agi.github.io/Weight-Space-Programing"
DESCRIPTION = (
    "An open benchmark for repairing known defects in Qwen3.5-0.8B "
    "with the same harness, budget, and offline sandbox."
)


SITE_CSS = r"""
:root{--bg:#F5F4EE;--paper:#FAF9F5;--ink:#141413;--accent:#D97757;--green:#6A8F6B;--gray:#8A867D;--soft:#3D3B36;--line:#E2DED3;--beige:#EFE9DC}
*{box-sizing:border-box}html{background:var(--bg);scroll-behavior:smooth}
body{margin:0;color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
a{color:inherit;text-decoration-color:var(--accent);text-underline-offset:3px}
.shell{width:min(1160px,calc(100% - 40px));margin:auto}.narrow{width:min(820px,calc(100% - 40px));margin:auto}
header.site{border-bottom:1px solid var(--line);background:rgba(245,244,238,.94);position:sticky;top:0;z-index:10;backdrop-filter:blur(8px)}
.nav{height:58px;display:flex;align-items:center;gap:25px}.brand{font:500 21px Georgia,serif;text-decoration:none;margin-right:auto}.nav a:not(.brand){font-size:13px;text-decoration:none;color:var(--soft)}.nav a:hover{color:var(--accent)}
.hero{display:grid;grid-template-columns:1.3fr .7fr;gap:44px;padding:72px 0 52px;align-items:center}
.eyebrow{color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
h1,h2,h3,.score{font-family:Georgia,"Times New Roman",serif;font-weight:500}h1{font-size:clamp(42px,6vw,72px);line-height:1.02;margin:13px 0 20px;letter-spacing:-.025em}h2{font-size:30px;line-height:1.2;margin:0 0 8px}h3{font-size:21px;margin:28px 0 8px}
.lede{font-size:19px;line-height:1.6;color:var(--soft);max-width:720px}.hero img{width:100%;border:1px solid var(--line);border-radius:10px;background:var(--paper)}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:24px}.pill{border:1px solid var(--line);border-radius:999px;padding:5px 12px;font-size:12px;color:var(--soft);background:var(--paper)}
section{padding:42px 0;border-top:1px solid var(--line)}.section-head{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:22px}.muted{color:var(--gray)}.small{font-size:13px}.kicker{font-size:13px;color:var(--gray);margin:0}
.split-tabs{display:flex;gap:6px}.split-tabs button{border:1px solid var(--line);background:var(--paper);padding:7px 15px;border-radius:999px;cursor:pointer;color:var(--soft)}.split-tabs button.active{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--paper)}
table{border-collapse:collapse;width:100%;font-size:13px}th{padding:11px 12px;text-align:right;border-bottom:1.5px solid var(--ink);white-space:nowrap;font-weight:650;cursor:pointer}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:inherit}td{padding:12px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#fff}
tr.baseline{background:var(--beige)}tr.baseline-oracle{box-shadow:inset 4px 0 var(--green)}tr.baseline-noop{box-shadow:inset 4px 0 var(--gray)}tr.baseline-scripted{box-shadow:inset 4px 0 var(--accent)}
.model{font-weight:650;white-space:nowrap}.model small{display:block;font-weight:400;color:var(--gray)}.score{font-size:20px}.ci{display:block;font:11px -apple-system,sans-serif;color:var(--gray)}
.empty{padding:40px;border:1px solid var(--line);border-radius:10px;background:var(--paper)}.empty h3{margin-top:0}.roster{columns:2;list-style:none;padding:0}.roster li{break-inside:avoid;padding:5px 0;color:var(--soft)}
.chart{border:1px solid var(--line);border-radius:10px;background:var(--paper);padding:10px}.chart svg{display:block;width:100%;height:auto}.chart .grid{stroke:var(--line);stroke-width:1}.chart .axis{fill:var(--gray);font:12px -apple-system,sans-serif}.chart .label{fill:var(--soft);font:11px -apple-system,sans-serif}.chart .frontier{stroke:var(--accent);stroke-width:2;fill:none;stroke-dasharray:5 4}.chart .point{fill:var(--green);stroke:var(--paper);stroke-width:2}.chart .basepoint{fill:var(--accent);stroke:var(--paper);stroke-width:2}
.caveats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{border:1px solid var(--line);border-radius:9px;background:var(--paper);padding:19px}.card b{display:block;margin-bottom:5px}.card p{margin:0;color:var(--soft);font-size:14px}
.method{padding:66px 0}.method p,.method li{color:var(--soft)}.formula{font:17px ui-monospace,SFMono-Regular,Menlo,monospace;padding:16px;border-left:3px solid var(--accent);background:var(--paper);overflow:auto}.family-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.family-grid .card{min-height:145px}
.trace-head{padding:58px 0 32px}.trace-stats{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:9px;overflow:hidden;margin-top:25px}.stat{background:var(--paper);padding:13px}.stat span{display:block;color:var(--gray);font-size:11px;text-transform:uppercase;letter-spacing:.05em}.stat b{font-size:16px}
.turns{padding-bottom:70px}.turn{display:grid;grid-template-columns:110px 1fr;gap:18px;margin:0 0 14px}.turn-meta{font:12px ui-monospace,Menlo,monospace;color:var(--gray);padding-top:12px}.bubble{border:1px solid var(--line);border-left:4px solid var(--gray);border-radius:9px;background:var(--paper);padding:14px 17px;min-width:0}.turn.user .bubble{border-left-color:var(--accent)}.turn.assistant .bubble{border-left-color:var(--green)}.turn.tool .bubble{border-style:dashed}.role{font-size:11px;letter-spacing:.08em;font-weight:750;color:var(--gray);text-transform:uppercase}.content{white-space:pre-wrap;overflow-wrap:anywhere}.bubble pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--beige);border-radius:6px;padding:12px;font:12px/1.55 ui-monospace,Menlo,monospace}.bubble details{border-top:1px solid var(--line);margin-top:12px;padding-top:8px}.bubble summary{cursor:pointer;color:var(--accent);font-size:12px}.turn-cost{font-size:11px;color:var(--gray);margin-top:10px}
.missing{padding:24px;border:1px dashed var(--line);border-radius:9px;background:var(--paper);color:var(--soft)}
footer.site{border-top:1px solid var(--line);padding:24px 0 45px;color:var(--gray);font-size:12px}.footer-row{display:flex;justify-content:space-between;gap:20px}
.watermark{position:fixed;right:-48px;top:42px;transform:rotate(35deg);background:var(--accent);color:white;padding:7px 62px;z-index:30;font-size:11px;font-weight:800;letter-spacing:.12em;pointer-events:none}
@media(max-width:800px){.hero{grid-template-columns:1fr;padding-top:48px}.hero img{display:none}.caveats,.family-grid{grid-template-columns:1fr}.trace-stats{grid-template-columns:repeat(3,1fr)}.turn{grid-template-columns:1fr;gap:4px}.turn-meta{padding-top:0}.nav{gap:14px}.roster{columns:1}.section-head{align-items:start;flex-direction:column}}
"""


SITE_JS = r"""
document.querySelectorAll("[data-sort-table]").forEach(table => {
  table.querySelectorAll("th").forEach((th, column) => {
    th.addEventListener("click", () => {
      const body = table.tBodies[0];
      const rows = Array.from(body.rows);
      const ascending = th.dataset.direction !== "asc";
      table.querySelectorAll("th").forEach(x => delete x.dataset.direction);
      th.dataset.direction = ascending ? "asc" : "desc";
      rows.sort((a, b) => {
        const av = a.cells[column]?.dataset.value ?? a.cells[column]?.innerText ?? "";
        const bv = b.cells[column]?.dataset.value ?? b.cells[column]?.innerText ?? "";
        const an = Number(av), bn = Number(bv);
        const result = Number.isFinite(an) && Number.isFinite(bn)
          ? an - bn : av.localeCompare(bv);
        return ascending ? result : -result;
      });
      rows.forEach(row => body.appendChild(row));
    });
  });
});
document.querySelectorAll("[data-split-button]").forEach(button => {
  button.addEventListener("click", () => {
    const split = button.dataset.splitButton;
    document.querySelectorAll("[data-split-panel]").forEach(panel => {
      panel.hidden = panel.dataset.splitPanel !== split;
    });
    document.querySelectorAll("[data-split-button]").forEach(other => {
      other.classList.toggle("active", other === button);
    });
  });
});
"""


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "__", model_id)


def safe_instance(instance_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "-", instance_id)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: skipped {path}: {exc}")
        return None
    if not isinstance(data, dict):
        print(f"warning: skipped non-object result {path}")
        return None
    return data


def load_results(fixtures: bool) -> list[dict[str, Any]]:
    if fixtures:
        path = ROOT / "site" / "fixtures" / "results.jsonl"
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: {exc}") from exc
            row["_source"] = str(path)
            rows.append(row)
        return rows

    results_root = ROOT / "results"
    rows = []
    if not results_root.exists():
        return rows
    for path in sorted(results_root.glob("*/*/*.json")):
        row = load_json(path)
        if row is not None:
            row["_source"] = str(path)
            rows.append(row)
    return rows


def bootstrap_ci(values: list[float], seed: str) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(4000)
    )
    return means[int(0.025 * len(means))], means[int(0.975 * len(means))]


def token_total(row: dict[str, Any]) -> int:
    usage = row.get("tokens") or {}
    return sum(int(usage.get(key, 0) or 0) for key in ("prompt", "completion", "reasoning"))


def aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("model", "unknown"))].append(row)

    output = []
    for model, episodes in groups.items():
        scores = [float(row.get("score", 0) or 0) for row in episodes]
        per_family = {
            family: statistics.fmean(
                float(row.get("score", 0) or 0)
                for row in episodes
                if row.get("family") == family
            )
            if any(row.get("family") == family for row in episodes)
            else None
            for family in FAMILIES
        }
        lo, hi = bootstrap_ci(scores, model)
        output.append(
            {
                "model": model,
                "display_name": episodes[0].get("display_name") or model,
                "score": statistics.fmean(scores),
                "ci": (lo, hi),
                "families": per_family,
                "cost": sum(float(row.get("cost_usd", 0) or 0) for row in episodes),
                "tokens": sum(token_total(row) for row in episodes),
                "gpu_calls": sum(int(row.get("gpu_calls", 0) or 0) for row in episodes),
                "instances": len(episodes),
                "first": episodes[0],
                "baseline": model.lower() in BASELINES,
            }
        )
    return sorted(output, key=lambda row: (-row["score"], row["display_name"]))


def roster_from_config() -> list[str]:
    path = ROOT / "configs" / "models.yaml"
    if path.exists():
        text = path.read_text(encoding="utf-8")
        names = re.findall(r"^\s*display_name:\s*[\"']?(.+?)[\"']?\s*$", text, re.M)
        if names:
            return names
        ids = re.findall(r"^\s*id:\s*[\"']?(.+?)[\"']?\s*$", text, re.M)
        if ids:
            return [item.split("/")[-1] for item in ids]

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"\| Oracle .*?\n(?P<body>.*?)\| No-op \|", readme, flags=re.S
    )
    if not match:
        return []
    names = []
    for line in match.group("body").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] and "---" not in cells[0] and "Scripted" not in cells[0]:
            names.append(re.sub(r"\s*\([^)]*\)", "", cells[0]))
    return names


def social_meta(title: str, description: str, image_path: str = "assets/og.png") -> str:
    image = f"{SITE_URL}/assets/og.png"
    return f"""<meta name="description" content="{esc(description)}">
<meta property="og:type" content="website"><meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}"><meta property="og:image" content="{esc(image)}">
<meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="{esc(title)}">
<meta name="twitter:description" content="{esc(description)}"><meta name="twitter:image" content="{esc(image)}">"""


def page(
    title: str,
    body: str,
    *,
    prefix: str = "",
    description: str = DESCRIPTION,
    preview: bool = False,
) -> str:
    watermark = (
        '<div class="watermark">PREVIEW · FIXTURE DATA</div>' if preview else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>{social_meta(title, description)}
<link rel="stylesheet" href="{prefix}assets/site.css"></head><body>{watermark}
<header class="site"><nav class="nav shell"><a class="brand" href="{prefix}index.html">WSP-Bench</a>
<a href="{prefix}index.html#leaderboard">Leaderboard</a><a href="{prefix}method.html">Rules &amp; method</a>
<a href="{prefix}review/index.html">Blog review</a><a href="https://github.com/GPT-AGI/Weight-Space-Programing">GitHub</a></nav></header>
{body}
<footer class="site"><div class="footer-row shell"><span>WSP-Bench · all traces and costs public</span><span>Qwen3.5-0.8B · v0.1</span></div></footer>
<script src="{prefix}assets/site.js"></script></body></html>"""


def fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def fmt_tokens(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def trace_url(row: dict[str, Any]) -> str:
    return (
        f"runs/{esc(row.get('split', 'dev'))}/{slug(str(row.get('model', 'unknown')))}/"
        f"{safe_instance(str(row.get('instance_id', 'instance')))}.html"
    )


def table_html(summary: list[dict[str, Any]]) -> str:
    header = (
        "<thead><tr><th>Model</th><th>Overall</th>"
        + "".join(f"<th>{name.title()}</th>" for name in FAMILIES)
        + "<th>$ / full run</th><th>Tokens</th><th>GPU calls</th></tr></thead>"
    )
    rows = []
    for item in summary:
        baseline_name = item["model"].lower() if item["baseline"] else ""
        classes = (
            f' class="baseline baseline-{esc(baseline_name)}"' if baseline_name else ""
        )
        first = item["first"]
        model_label = esc(item["display_name"])
        if not item["baseline"]:
            model_label = f'<a href="{trace_url(first)}">{model_label}</a>'
        model_cell = (
            f'<td data-value="{esc(item["display_name"])}"><span class="model">{model_label}'
            f'<small>{item["instances"]} instances</small></span></td>'
        )
        lo, hi = item["ci"]
        score_cell = (
            f'<td data-value="{item["score"]:.5f}"><span class="score">{item["score"]:.1f}</span>'
            f'<span class="ci">95% CI {lo:.1f}–{hi:.1f}</span></td>'
        )
        family_cells = "".join(
            f'<td data-value="{value if value is not None else -1}">{fmt_score(value)}</td>'
            for value in item["families"].values()
        )
        cost = item["cost"]
        cost_text = "$0" if cost == 0 else f"${cost:.2f}"
        rows.append(
            f"<tr{classes}>{model_cell}{score_cell}{family_cells}"
            f'<td data-value="{cost:.8f}">{cost_text}</td>'
            f'<td data-value="{item["tokens"]}">{fmt_tokens(item["tokens"])}</td>'
            f'<td data-value="{item["gpu_calls"]}">{item["gpu_calls"]}</td></tr>'
        )
    return f'<div class="table-wrap"><table data-sort-table>{header}<tbody>{"".join(rows)}</tbody></table></div>'


def pareto_svg(summary: list[dict[str, Any]]) -> str:
    points = [
        row
        for row in summary
        if row["model"].lower() not in ("oracle", "noop") and row["cost"] >= 0
    ]
    if not points:
        return '<div class="missing">The score-vs-cost chart will appear when results are available.</div>'

    positive = [row["cost"] for row in points if row["cost"] > 0]
    floor = min(positive) / 2 if positive else 0.01
    plotted = [(max(row["cost"], floor), row) for row in points]
    xmin = min(cost for cost, _ in plotted)
    xmax = max(cost for cost, _ in plotted)
    if xmax <= xmin:
        xmax = xmin * 10
    xmin = 10 ** math.floor(math.log10(xmin))
    xmax = 10 ** math.ceil(math.log10(xmax))
    if xmin == xmax:
        xmax *= 10

    width, height = 920, 430
    left, right, top, bottom = 72, 22, 24, 58
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(cost: float) -> float:
        return left + (math.log10(cost) - math.log10(xmin)) / (
            math.log10(xmax) - math.log10(xmin)
        ) * plot_w

    def y(score: float) -> float:
        return top + (100 - score) / 100 * plot_h

    grid = []
    for score in range(0, 101, 20):
        yy = y(score)
        grid.append(
            f'<line class="grid" x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}"/>'
            f'<text class="axis" x="{left-12}" y="{yy+4:.1f}" text-anchor="end">{score}</text>'
        )
    start_power = math.floor(math.log10(xmin))
    end_power = math.ceil(math.log10(xmax))
    for power in range(start_power, end_power + 1):
        cost = 10**power
        xx = x(cost)
        label = f"${cost:g}" if cost >= 0.01 else f"${cost:.3f}"
        grid.append(
            f'<line class="grid" x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}"/>'
            f'<text class="axis" x="{xx:.1f}" y="{height-bottom+24}" text-anchor="middle">{label}</text>'
        )

    frontier_rows = []
    best = -1.0
    for cost, row in sorted(plotted, key=lambda pair: (pair[0], -pair[1]["score"])):
        if row["baseline"]:
            continue
        if row["score"] > best:
            frontier_rows.append((cost, row))
            best = row["score"]
    frontier = ""
    if len(frontier_rows) > 1:
        coords = " ".join(
            f"{x(cost):.1f},{y(row['score']):.1f}" for cost, row in frontier_rows
        )
        frontier = f'<polyline class="frontier" points="{coords}"/>'

    dots = []
    for index, (cost, row) in enumerate(plotted):
        xx, yy = x(cost), y(row["score"])
        point_class = "basepoint" if row["baseline"] else "point"
        label_y = yy - 10 if index % 2 == 0 else yy + 18
        suffix = " ($0)" if row["cost"] == 0 else ""
        dots.append(
            f'<circle class="{point_class}" cx="{xx:.1f}" cy="{yy:.1f}" r="6">'
            f"<title>{esc(row['display_name'])}: {row['score']:.1f} at ${row['cost']:.2f}</title></circle>"
            f'<text class="label" x="{xx+8:.1f}" y="{label_y:.1f}">{esc(row["display_name"])}{suffix}</text>'
        )
    return (
        f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Overall score versus cost per full run, with a logarithmic cost axis">'
        + "".join(grid)
        + frontier
        + "".join(dots)
        + f'<text class="axis" x="{left+plot_w/2:.1f}" y="{height-8}" text-anchor="middle">$ per full run (log scale)</text>'
        + f'<text class="axis" transform="translate(16 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle">Overall score</text>'
        + "</svg></div>"
    )


def empty_state() -> str:
    roster = roster_from_config()
    items = "".join(f"<li>{esc(name)}</li>" for name in roster)
    roster_html = (
        f'<p class="small muted">Initial model roster</p><ul class="roster">{items}</ul>'
        if items
        else ""
    )
    return (
        '<div class="empty"><p class="eyebrow">Results pending</p>'
        "<h3>Leaderboard launching Nov 6</h3>"
        "<p>Official runs will appear here with bootstrap confidence intervals, costs, "
        "token counts, GPU calls, and full public traces.</p>"
        f"{roster_html}</div>"
    )


def build_index(
    results: list[dict[str, Any]], output: Path, preview: bool
) -> None:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_split[str(result.get("split", "dev"))].append(result)
    splits = [name for name in ("dev", "test") if name in by_split]

    if splits:
        buttons = (
            '<div class="split-tabs">'
            + "".join(
                f'<button data-split-button="{name}" class="{"active" if i == 0 else ""}">{name}</button>'
                for i, name in enumerate(splits)
            )
            + "</div>"
            if len(splits) > 1
            else f'<span class="pill">{splits[0]} split</span>'
        )
        panels = []
        charts = []
        for index, split in enumerate(splits):
            summary = aggregate(by_split[split])
            panels.append(
                f'<div data-split-panel="{split}"{" hidden" if index else ""}>{table_html(summary)}</div>'
            )
            charts.append(
                f'<div data-split-panel="{split}"{" hidden" if index else ""}>{pareto_svg(summary)}</div>'
            )
        leaderboard = "".join(panels)
        chart = "".join(charts)
    else:
        buttons = ""
        leaderboard = empty_state()
        chart = pareto_svg([])

    body = f"""
<main><div class="hero shell"><div><p class="eyebrow">Weight-space programming, measured</p>
<h1>Who is the best weight-space programmer?</h1>
<p class="lede">We plant known defects in Qwen3.5-0.8B. Every LLM gets the same harness, budget, and offline sandbox. Repairs are scored against exact ground truth; every trace and dollar is public.</p>
<div class="pills"><span class="pill">4 defect families</span><span class="pill">3 difficulty tiers</span><span class="pill">Oracle 100 · No-op 0</span><span class="pill">No-LLM baseline</span></div></div>
<img src="assets/hero.gif" alt="WSP-Bench leaderboard preview"></div>
<section id="leaderboard"><div class="shell"><div class="section-head"><div><h2>Leaderboard</h2>
<p class="kicker">Overall score is the mean across instances. Confidence intervals bootstrap instances.</p></div>{buttons}</div>{leaderboard}</div></section>
<section><div class="shell"><div class="section-head"><div><h2>Score vs. cost</h2>
<p class="kicker">Higher and farther left is better. The dashed line marks the model Pareto frontier; cost uses a logarithmic axis.</p></div></div>{chart}</div></section>
<section><div class="shell"><div class="section-head"><div><h2>What this does not show</h2>
<p class="kicker">A benchmark result is evidence about performance under these rules—not a broader metaphysical claim.</p></div><a href="method.html">Read rules &amp; method →</a></div>
<div class="caveats"><div class="card"><b>Synthetic defects</b><p>The planted failures are controlled tasks with exact ground truth, not a claim about every real model failure.</p></div>
<div class="card"><b>Not native “weight reading”</b><p>A high score means the model runs a find-direction → edit → verify loop effectively with the supplied tools.</p></div>
<div class="card"><b>Cost is part of the result</b><p>Scores share a fixed budget, but provider pricing, token accounting, and default reasoning modes still matter.</p></div></div></div></section></main>"""
    (output / "index.html").write_text(
        page("WSP-Bench Leaderboard", body, preview=preview), encoding="utf-8"
    )


def build_method(output: Path, preview: bool) -> None:
    body = """
<main class="method narrow"><p class="eyebrow">Rules &amp; method</p><h1>One harness. Exact ground truth.</h1>
<p class="lede">WSP-Bench measures how well an LLM can investigate and repair controlled weight-space defects without access to the clean checkpoint.</p>
<h2>Task families</h2><div class="family-grid">
<div class="card"><b>Behavior injection</b><p>A rank-one edit induces a language switch, capitalization, repetition, or another measurable behavior. Repair it without collateral damage.</p></div>
<div class="card"><b>Trigger backdoor</b><p>An unknown trigger activates a rank-one key→value edit. Identify the trigger and remove the backdoor.</p></div>
<div class="card"><b>Localization audit</b><p>A hidden edit appears somewhere in the network. Report the layer, parameter, and direction without editing.</p></div>
<div class="card"><b>Neuron hijack</b><p>A small group of amplified neurons floods output with one token. Restore normal behavior.</p></div></div>
<h2>Scoring</h2><p>For edit tasks, repair is measured on hidden prompts and discounted by collateral change on held-out general text. Values are clipped to their stated ranges.</p>
<div class="formula">repair = clip((bug_rate(broken) − bug_rate(submitted)) / (bug_rate(broken) − bug_rate(clean)), 0, 1)<br>
instance_score = 100 × repair × (1 − collateral_penalty)</div>
<p>Collateral penalty is derived from mean per-token KL(clean ∥ submitted) between frozen free and maximum thresholds. Audit tasks use <code>100 × localization</code>, combining layer/parameter identification and absolute cosine similarity to the planted direction. Overall score is the mean of instance scores, with a bootstrap 95% confidence interval over instances.</p>
<h2>Fairness and anti-leakage</h2><ul>
<li>Every model receives the same system prompt, text protocol, turn limit, GPU-call limit, edit budget, and dollar cap.</li>
<li>The Python sandbox has no network and contains only the broken model. The clean checkpoint and ground truth remain in the scoring process.</li>
<li>Defect direction, strength, layer, and trigger are randomized. Official test specifications and seeds remain private.</li>
<li>Behavior metrics are rule-based. No LLM judge is used.</li>
<li>Every run records model ID, provider settings, tokens, dollars, GPU calls, stop reason, and the complete trace.</li>
</ul>
<h2>Budget</h2><p>Each v0.1 instance allows at most <b>40 turns</b>, <b>20 GPU calls</b>, batches of <b>64</b>, generations of <b>128 new tokens</b>, <b>262,144 changed parameters</b>, and <b>$1.50</b> in model API cost. Reaching a limit stops the episode and scores the current submission.</p>
<h2>Reference rows</h2><ul><li><b>Oracle (100):</b> knows and reverses the planted edit.</li><li><b>No-op (0):</b> submits the broken model unchanged.</li><li><b>Scripted pipeline:</b> a fixed no-LLM interpretability loop. The difference between this row and an LLM is the clearest estimate of what the LLM adds.</li></ul>
<h2>Interpretation</h2><p>These are synthetic defects designed for exact scoring. A strong result demonstrates skill at the supplied experimental loop; it does not establish that a model can infer arbitrary neural-network purpose by reading raw weights.</p></main>"""
    (output / "method.html").write_text(
        page("Rules & Method · WSP-Bench", body, preview=preview), encoding="utf-8"
    )


def trace_source(result: dict[str, Any], fixtures: bool) -> Path | None:
    if fixtures:
        path = ROOT / "site" / "fixtures" / "trace.jsonl"
        return path if path.exists() else None
    trace = str(result.get("trace", "") or "")
    if not trace:
        return None
    path = Path(trace)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "runs":
        path = Path(*parts[1:])
    return ROOT / "runs" / path


def load_trace(result: dict[str, Any], fixtures: bool) -> tuple[list[dict[str, Any]], str]:
    path = trace_source(result, fixtures)
    if path is None:
        return [], "No trace was recorded for this run."
    if not path.exists():
        return [], f"Trace artifact is not available: {path.name}"
    turns = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            turn = json.loads(line)
        except json.JSONDecodeError as exc:
            return turns, f"Trace ended at malformed line {number}: {exc}"
        if isinstance(turn, dict):
            turns.append(turn)
    return turns, ""


def component_value(components: dict[str, Any], key: str) -> str:
    value = components.get(key)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return esc(value)


def turn_html(turn: dict[str, Any]) -> str:
    role = str(turn.get("role", "tool"))
    usage = turn.get("usage") or {}
    usage_total = sum(
        int(usage.get(key, 0) or 0) for key in ("prompt", "completion", "reasoning")
    )
    reasoning = turn.get("reasoning")
    code = turn.get("code")
    extras = ""
    if reasoning:
        extras += (
            f"<details><summary>Reasoning</summary><div class=\"content\">{esc(reasoning)}</div></details>"
        )
    if code:
        extras += f"<details open><summary>Executed Python</summary><pre><code>{esc(code)}</code></pre></details>"
    metrics = (
        f"${float(turn.get('cost_usd', 0) or 0):.4f} · {usage_total:,} tokens · "
        f"{int(turn.get('gpu_calls_used', 0) or 0)} cumulative GPU calls · "
        f"{float(turn.get('wall_s', 0) or 0):.1f}s"
    )
    return (
        f'<article class="turn {esc(role)}"><div class="turn-meta">TURN {int(turn.get("t", 0)):02d}</div>'
        f'<div class="bubble"><div class="role">{esc(role)}</div>'
        f'<div class="content">{esc(turn.get("content", ""))}</div>{extras}'
        f'<div class="turn-cost">{metrics}</div></div></article>'
    )


def build_traces(
    results: list[dict[str, Any]], output: Path, fixtures: bool, preview: bool
) -> None:
    for result in results:
        model = str(result.get("model", "unknown"))
        if model.lower() in BASELINES and not result.get("trace"):
            continue
        split = str(result.get("split", "dev"))
        instance = str(result.get("instance_id", "instance"))
        destination = output / "runs" / split / slug(model) / f"{safe_instance(instance)}.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        turns, error = load_trace(result, fixtures)
        components = result.get("components") or {}
        stats = [
            ("Score", f"{float(result.get('score', 0) or 0):.1f}"),
            ("Repair", component_value(components, "repair")),
            ("Collateral", component_value(components, "collateral_penalty")),
            ("Localization", component_value(components, "localization")),
            ("Cost", f"${float(result.get('cost_usd', 0) or 0):.3f}"),
            ("GPU calls", str(int(result.get("gpu_calls", 0) or 0))),
        ]
        stat_html = "".join(
            f'<div class="stat"><span>{esc(label)}</span><b>{value}</b></div>'
            for label, value in stats
        )
        if turns:
            trace_body = "".join(turn_html(turn) for turn in turns)
            if error:
                trace_body += f'<div class="missing">{esc(error)}</div>'
        else:
            trace_body = f'<div class="missing">{esc(error)}</div>'
        body = f"""
<main><div class="trace-head narrow"><p class="eyebrow">{esc(split)} · {esc(result.get("family", ""))} · {esc(result.get("tier", ""))}</p>
<h1>{esc(result.get("display_name") or model)}</h1><p class="lede">{esc(instance)}</p>
<div class="trace-stats">{stat_html}</div>
<p class="small muted">Stop: {esc(result.get("stop_reason", "unknown"))} · {int(result.get("turns", len(turns)) or 0)} turns · {float(result.get("wall_s", 0) or 0):.1f}s · harness {esc(result.get("harness_version", "unknown"))}</p></div>
<div class="turns narrow">{trace_body}</div></main>"""
        destination.write_text(
            page(
                f"{result.get('display_name') or model} · {instance}",
                body,
                prefix="../../../",
                description=f"Public WSP-Bench trace for {result.get('display_name') or model} on {instance}.",
                preview=preview,
            ),
            encoding="utf-8",
        )


def review_copy(source: Path, destination: Path, label: str, href: str) -> None:
    text = source.read_text(encoding="utf-8")
    meta = social_meta(
        "Astra's Qwen Math Weight Edit · Claims vs. Logs",
        "A factual, claim-by-claim review of Experiment 8 against the authors' public logs.",
    )
    switch_css = (
        "<style>.lang-switch{position:fixed;right:18px;top:14px;z-index:20;"
        "border:1px solid #E2DED3;border-radius:999px;background:#FAF9F5;padding:5px 12px;"
        "font:12px -apple-system,sans-serif;color:#3D3B36;text-decoration:none}</style>"
    )
    text = text.replace("</head>", f"{meta}{switch_css}</head>", 1)
    text = text.replace(
        "<body>", f'<body><a class="lang-switch" href="{href}">{esc(label)}</a>', 1
    )
    destination.write_text(text, encoding="utf-8")


def generate_og(path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for the social card; run with .venv-assets/bin/python"
        ) from exc

    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=PALETTE["bg"])
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.add_patch(
        plt.Rectangle((0.055, 0.77), 0.012, 0.08, color=PALETTE["accent"], transform=ax.transAxes)
    )
    ax.text(
        0.055,
        0.68,
        "WSP-Bench",
        color=PALETTE["ink"],
        fontsize=54,
        fontfamily="serif",
        transform=ax.transAxes,
    )
    ax.text(
        0.055,
        0.52,
        "Who is the best weight-space programmer?",
        color=PALETTE["ink"],
        fontsize=28,
        fontfamily="serif",
        transform=ax.transAxes,
    )
    ax.text(
        0.055,
        0.37,
        "Known defects · same harness · exact ground truth\nEvery trace and dollar public",
        color=PALETTE["gray"],
        fontsize=18,
        linespacing=1.5,
        transform=ax.transAxes,
    )
    ax.text(
        0.055,
        0.13,
        "Qwen3.5-0.8B   ·   Oracle 100   ·   No-op 0   ·   No-LLM baseline",
        color=PALETTE["green"],
        fontsize=16,
        weight="bold",
        transform=ax.transAxes,
    )
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def copy_assets(output: Path) -> None:
    assets_out = output / "assets"
    assets_out.mkdir(parents=True, exist_ok=True)
    for name in ("hero.gif", "decomposition.gif", "loop.gif", "projection.gif"):
        source = ROOT / "assets" / name
        if source.exists():
            shutil.copy2(source, assets_out / name)
    (assets_out / "site.css").write_text(SITE_CSS.strip() + "\n", encoding="utf-8")
    (assets_out / "site.js").write_text(SITE_JS.strip() + "\n", encoding="utf-8")
    generate_og(assets_out / "og.png")


def validate_results(results: list[dict[str, Any]]) -> None:
    required = {
        "model",
        "display_name",
        "instance_id",
        "split",
        "family",
        "tier",
        "score",
        "components",
        "cost_usd",
        "tokens",
        "turns",
        "gpu_calls",
        "wall_s",
        "stop_reason",
        "harness_version",
        "started_at",
        "trace",
    }
    for index, result in enumerate(results, 1):
        missing = required - result.keys()
        if missing:
            raise SystemExit(f"result {index} missing fields: {', '.join(sorted(missing))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="build fake fixture data into site/_preview/ (never docs/)",
    )
    args = parser.parse_args()
    output = ROOT / ("site/_preview" if args.fixtures else "docs")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    results = load_results(args.fixtures)
    validate_results(results)
    copy_assets(output)
    build_index(results, output, args.fixtures)
    build_method(output, args.fixtures)
    build_traces(results, output, args.fixtures, args.fixtures)

    review_out = output / "review"
    review_out.mkdir(parents=True, exist_ok=True)
    review_copy(
        ROOT / "doc/source_blog_review.en.html",
        review_out / "index.html",
        "中文",
        "zh.html",
    )
    review_copy(
        ROOT / "doc/source_blog_review.html",
        review_out / "zh.html",
        "English",
        "index.html",
    )
    print(f"built {len(results)} results into {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
