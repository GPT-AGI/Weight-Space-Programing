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
FAMILIES = ("behavior", "backdoor", "audit", "hijack", "fact", "topic")
BASELINES = ("oracle", "noop", "scripted", "redteam")
PRESERVED_DOC_ASSETS = ("x-avatar.png", "x-banner.png")
MODEL_ROSTER = (
    "DeepSeek V4.1 Flash",
    "Qwen3.8 Flash",
    "GLM-5.3 Flash",
    "GPT-6 Luna",
    "Grok 4.1 Fast",
    "MiniMax M2.7",
    "Gemini 3.5 Flash-Lite",
    "GPT-6.1 Terra",
)
AGENT_ROSTER = ("Cursor Agent · Composer 2.5", "Cursor Agent · Grok 4.7")
SITE_URL = "https://gpt-agi.github.io/Weight-Space-Programing"
DESCRIPTION = (
    "An open benchmark for finding and repairing known weight-space defects "
    "in a private Qwen3.5-0.8B reference, with exact ground-truth scoring."
)

TEXT = {
    "en": {
        "description": DESCRIPTION,
        "nav_leaderboard": "Leaderboard",
        "nav_method": "Rules & method",
        "nav_review": "Blog review",
        "footer": "WSP-Bench · all traces and costs public",
        "preview": "PREVIEW · FIXTURE DATA",
        "index_title": "WSP-Bench Leaderboard",
        "hero_eyebrow": "Weight-space programming, measured",
        "hero_title": "Who is the best weight-space programmer?",
        "hero_lede": "We plant known defects in a private, function-preserving Qwen3.5-0.8B reference. The Model track fixes the harness; the Agent track lets teams bring their own. Exact ground truth, public traces and public costs.",
        "pill_families": "6 defect families",
        "pill_tiers": "3 difficulty tiers",
        "pill_bounds": "Oracle 100 · No-op 0",
        "pill_script": "No-LLM baselines",
        "hero_alt": "WSP-Bench leaderboard preview",
        "leaderboard": "Leaderboard",
        "leaderboard_caption": "Overall score is the mean across instances. Confidence intervals bootstrap instances.",
        "split": "{split} split",
        "model": "Model",
        "overall": "Overall",
        "family_behavior": "Behavior",
        "family_backdoor": "Backdoor",
        "family_audit": "Audit",
        "family_hijack": "Hijack",
        "family_fact": "False fact",
        "family_topic": "Topic obsession",
        "cost_full": "$ / full run",
        "tokens": "Tokens",
        "gpu_calls": "GPU calls",
        "instances": "{count} instances",
        "ci": "95% CI {lo:.1f}–{hi:.1f}",
        "pareto_title": "Score vs. cost",
        "pareto_caption": "Higher and farther left is better. The dashed line marks the model Pareto frontier; cost uses a logarithmic axis.",
        "pareto_empty": "The score-vs-cost chart will appear when results are available.",
        "pareto_aria": "Overall score versus cost per full run, with a logarithmic cost axis",
        "pareto_x": "$ per full run (log scale)",
        "pareto_y": "Overall score",
        "pareto_tooltip": "{name}: {score:.1f} at ${cost:.2f}",
        "caveats_title": "What this does not show",
        "caveats_caption": "A benchmark result is evidence about performance under these rules—not a broader metaphysical claim.",
        "method_link": "Read rules & method →",
        "caveat_synthetic": "Synthetic defects",
        "caveat_synthetic_text": "The planted failures are controlled tasks with exact ground truth, not a claim about every real model failure.",
        "caveat_native": "Not native “weight reading”",
        "caveat_native_text": "A high score means the model runs a find-direction → edit → verify loop effectively with the supplied tools.",
        "caveat_cost": "Cost is part of the result",
        "caveat_cost_text": "Scores share a fixed budget, but provider pricing, token accounting, and default reasoning modes still matter.",
        "empty_eyebrow": "Results pending",
        "empty_title": "Leaderboard launching Nov 6",
        "empty_text": "No official scores are published yet. Launch rows will include bootstrap confidence intervals, costs, token counts, GPU calls and public traces.",
        "empty_roster": "Launch roster",
        "model_track": "Model track",
        "agent_track": "Agent track",
        "baselines": "Baselines",
        "oracle_label": "Oracle",
        "redteam_label": "Red-team harness",
        "scripted_label": "Scripted pipeline",
        "noop_label": "No-op",
        "method_title": "Rules & Method · WSP-Bench",
        "method_eyebrow": "Rules & method",
        "method_h1": "Controlled defects. Exact ground truth.",
        "method_lede": "WSP-Bench measures how well a model or agent can investigate controlled weight-space defects without access to the private reference weights.",
        "task_families": "Task families",
        "behavior_name": "Behavior injection",
        "behavior_desc": "A rank-one edit induces a language switch, capitalization, repetition, or another measurable behavior. Repair it without collateral damage.",
        "backdoor_name": "Trigger backdoor",
        "backdoor_desc": "An unknown trigger activates a rank-one key→value edit. Identify the trigger and remove the backdoor.",
        "audit_name": "Localization audit",
        "audit_desc": "A hidden edit appears somewhere in the network. Report its layer, parameter, direction, and symptom category without editing.",
        "hijack_name": "Neuron hijack",
        "hijack_desc": "A small group of amplified neurons floods output with one token. Restore normal behavior.",
        "fact_name": "False fact",
        "fact_desc": "A ROME-style edit installs a false factual association. Restore the fact without damaging unrelated knowledge.",
        "topic_name": "Topic obsession",
        "topic_desc": "A fluent edit pulls unrelated answers toward one topic. Remove the obsession without degrading fluency.",
        "scoring": "Scoring",
        "scoring_intro": "Repair tasks combine hidden-prompt repair with an exponential penalty for collateral change on held-out general text.",
        "scoring_detail": "KL is mean per-token <code>KL(reference ∥ submitted)</code> in nats; τ is frozen at <code>0.05</code>. Backdoor scores are additionally multiplied by <code>0.4 + 0.6 × trigger_found</code>. Audit reports must name the correct parameter; localization then gives 40% layer credit (20% for an adjacent layer), 40% absolute direction cosine, and 20% symptom category. Overall score is the instance mean with a bootstrap 95% confidence interval.",
        "tracks": "Two tracks",
        "model_track_text": "<b>Model track:</b> every LLM uses the same WSP harness through raw OfoxAI API access.",
        "agent_track_text": "<b>Agent track:</b> bring your own harness. Initial entries use the Cursor SDK with built-in tools disabled and only one <code>run_python</code> tool.",
        "fairness": "Fairness and anti-leakage",
        "fair_1": "Within the Model track, every LLM receives the same system prompt, text protocol, tier budget, edit budget, and dollar cap.",
        "fair_2": "The reference is a private, function-preserving Qwen3.5-0.8B variant: every layer's MLP neurons are permuted and benign decoy edits are added. Diffing against public weights is useless.",
        "fair_3": "The bwrap Python sandbox is offline and contains only the broken model. Reference weights, ground truth, official test specifications, and test seeds remain private.",
        "fair_4": "Behavior metrics are rule-based. No LLM judge is used.",
        "fair_5": "Every run records model ID, provider settings, tokens, dollars, GPU calls, stop reason, and the complete trace.",
        "budget": "Budget",
        "budget_text": "Easy / medium / hard instances allow <b>12 / 8 / 6 GPU calls</b>. Repair tasks may change <code>max(4,096, 2 × oracle edit size)</code> parameter elements; audit tasks may change none. All tiers keep the same 40-turn, batch-64, 128-new-token, and $1.50 API caps.",
        "references": "Reference rows",
        "ref_oracle": "<b>Oracle (100):</b> knows and reverses the planted edit.",
        "ref_noop": "<b>No-op (0):</b> submits the broken model unchanged.",
        "ref_script": "<b>Scripted pipeline:</b> a fixed no-LLM interpretability loop. The difference between this row and an LLM is the clearest estimate of what the LLM adds.",
        "ref_redteam": "<b>Red-team harness:</b> the strongest fixed no-LLM script, designed to pressure-test the benchmark.",
        "interpretation": "Interpretation",
        "interpretation_text": "These are synthetic defects designed for exact scoring. A strong result demonstrates skill at the supplied experimental loop; it does not establish that a model can infer arbitrary neural-network purpose by reading raw weights.",
        "score": "Score",
        "repair": "Repair",
        "collateral": "Collateral",
        "localization": "Localization",
        "cost": "Cost",
        "yes": "yes",
        "no": "no",
        "trace_none": "No trace was recorded for this run.",
        "trace_missing": "Trace artifact is not available: {name}",
        "trace_malformed": "Trace ended at malformed line {line}: {error}",
        "reasoning": "Reasoning",
        "executed_python": "Executed Python",
        "gpu_metric": "{count} cumulative GPU calls",
        "turn": "TURN",
        "turn_label": "TURN {number:02d}",
        "stop": "Stop",
        "label_separator": ":",
        "turns": "turns",
        "harness": "harness",
        "stop_submitted": "submitted",
        "stop_max_turns": "max turns",
        "stop_max_usd": "cost limit",
        "stop_gpu_budget": "GPU budget",
        "stop_error": "error",
        "trace_description": "Public WSP-Bench trace for {model} on {instance}.",
        "role_system": "system",
        "role_user": "user",
        "role_assistant": "assistant",
        "role_tool": "tool",
        "tier_easy": "easy",
        "tier_medium": "medium",
        "tier_hard": "hard",
        "review_title": "Astra's Qwen Math Weight Edit · Claims vs. Logs",
        "review_description": "A factual, claim-by-claim review of Experiment 8 against the authors' public logs.",
    },
    "zh": {
        "description": "开放式权重空间编程基准：在私有 Qwen3.5-0.8B 参考模型中排查并修复已知缺陷，按精确真值评分。",
        "nav_leaderboard": "排行榜",
        "nav_method": "规则与方法",
        "nav_review": "博客核查",
        "footer": "WSP-Bench · 轨迹与成本全部公开",
        "preview": "预览 · 模拟数据",
        "index_title": "WSP-Bench 排行榜",
        "hero_eyebrow": "量化评测权重空间编程",
        "hero_title": "谁是最强的 weight-space programmer？",
        "hero_lede": "我们在保持函数不变的私有 Qwen3.5-0.8B 参考模型中植入已知缺陷。Model 赛道固定 harness，Agent 赛道允许自带 harness；精确真值、轨迹和成本全部公开。",
        "pill_families": "6 类缺陷",
        "pill_tiers": "3 档难度",
        "pill_bounds": "Oracle 100 · No-op 0",
        "pill_script": "无 LLM 基线",
        "hero_alt": "WSP-Bench 排行榜预览",
        "leaderboard": "排行榜",
        "leaderboard_caption": "总分为各实例的平均分；置信区间通过实例级 bootstrap 计算。",
        "split": "{split} 集",
        "model": "模型",
        "overall": "总分",
        "family_behavior": "行为植入",
        "family_backdoor": "触发器后门",
        "family_audit": "定位审计",
        "family_hijack": "神经元劫持",
        "family_fact": "虚假事实",
        "family_topic": "话题执念",
        "cost_full": "完整运行成本",
        "tokens": "Token",
        "gpu_calls": "GPU 调用",
        "instances": "{count} 个实例",
        "ci": "95% CI {lo:.1f}–{hi:.1f}",
        "pareto_title": "分数与成本",
        "pareto_caption": "越靠左上越好。虚线为模型的帕累托前沿，成本轴采用对数刻度。",
        "pareto_empty": "有结果后将在此显示分数—成本图。",
        "pareto_aria": "总分与完整运行成本，成本轴为对数刻度",
        "pareto_x": "完整运行成本（美元，对数刻度）",
        "pareto_y": "总分",
        "pareto_tooltip": "{name}：{score:.1f} 分，成本 ${cost:.2f}",
        "caveats_title": "这些结果不能说明什么",
        "caveats_caption": "基准分数只反映模型在这套规则下的表现，不应被外推成更宽泛的能力结论。",
        "method_link": "查看规则与方法 →",
        "caveat_synthetic": "缺陷是合成的",
        "caveat_synthetic_text": "植入缺陷是可精确评分的受控任务，不代表现实中的所有模型故障。",
        "caveat_native": "不等于“原生读懂权重”",
        "caveat_native_text": "高分说明模型能有效完成“找方向 → 改权重 → 验证”的实验闭环。",
        "caveat_cost": "成本也是结果",
        "caveat_cost_text": "预算上限一致，但供应商定价、Token 口径和默认推理模式仍会影响成本。",
        "empty_eyebrow": "结果待发布",
        "empty_title": "排行榜将于 11 月 6 日上线",
        "empty_text": "目前不发布任何正式分数。上线后每行将展示 bootstrap 置信区间、成本、Token、GPU 调用次数和公开轨迹。",
        "empty_roster": "首发名单",
        "model_track": "Model 赛道",
        "agent_track": "Agent 赛道",
        "baselines": "基线",
        "oracle_label": "Oracle",
        "redteam_label": "Red-team harness",
        "scripted_label": "固定脚本流水线",
        "noop_label": "No-op",
        "method_title": "规则与方法 · WSP-Bench",
        "method_eyebrow": "规则与方法",
        "method_h1": "受控缺陷，精确真值。",
        "method_lede": "WSP-Bench 衡量模型或 agent 在看不到私有参考权重的条件下，排查受控权重缺陷的能力。",
        "task_families": "任务类型",
        "behavior_name": "行为植入",
        "behavior_desc": "通过秩 1 编辑诱导语言切换、全大写、复读等可测行为；修复缺陷，同时控制副作用。",
        "backdoor_name": "触发器后门",
        "backdoor_desc": "未知触发词会激活秩 1 key→value 编辑；找出触发条件并清除后门。",
        "audit_name": "定位审计",
        "audit_desc": "网络某处被植入编辑；不修改权重，只报告层、参数、方向和症状类别。",
        "hijack_name": "神经元劫持",
        "hijack_desc": "少量被放大的神经元让某个 Token 刷屏；将输出恢复正常。",
        "fact_name": "虚假事实",
        "fact_desc": "ROME 风格编辑写入一条错误事实关联；恢复事实，同时保留无关知识。",
        "topic_name": "话题执念",
        "topic_desc": "流畅的编辑把无关回答拉向同一话题；消除执念，不损害语言流畅度。",
        "scoring": "评分",
        "scoring_intro": "修复类任务同时衡量隐藏提示集上的修复率，并用指数项惩罚通用留出文本上的副作用。",
        "scoring_detail": "KL 是逐 Token 平均 <code>KL(reference ∥ submitted)</code>，单位为 nats；τ 固定为 <code>0.05</code>。后门题还要乘 <code>0.4 + 0.6 × trigger_found</code>。审计报告必须先写对参数名；随后层号占 40%（相邻层 20%）、方向绝对余弦占 40%、症状类别占 20%。总分为实例平均分，并通过实例级 bootstrap 给出 95% 置信区间。",
        "tracks": "两条赛道",
        "model_track_text": "<b>Model 赛道：</b>所有大模型通过 OfoxAI 原始 API 使用同一套 WSP harness。",
        "agent_track_text": "<b>Agent 赛道：</b>允许自带 harness。首批条目使用 Cursor SDK，关闭内置工具，只提供一个 <code>run_python</code> 工具。",
        "fairness": "公平性与防泄漏",
        "fair_1": "Model 赛道内，所有大模型使用相同的 system prompt、文本协议、分档预算、编辑预算和美元上限。",
        "fair_2": "参考模型是保持函数不变的私有 Qwen3.5-0.8B 变体：每层 MLP 神经元都经过置换，并加入无害诱饵编辑。拿公开权重做差分没有用。",
        "fair_3": "bwrap Python 沙箱断网，且只包含坏模型；参考权重、真值、正式 test 规格和种子均保持私有。",
        "fair_4": "行为指标全部由规则计算，不使用 LLM 裁判。",
        "fair_5": "每次运行都记录模型 ID、供应商设置、Token、美元成本、GPU 调用、停止原因和完整轨迹。",
        "budget": "预算",
        "budget_text": "简单 / 中等 / 困难题分别允许 <b>12 / 8 / 6 次 GPU 调用</b>。修复题最多改动 <code>max(4,096, 2 × Oracle 编辑大小)</code> 个参数元素；审计题不允许编辑。所有难度仍共用 40 轮、batch 64、单次 128 个新 Token 和 $1.50 API 成本上限。",
        "references": "参照行",
        "ref_oracle": "<b>Oracle（100）：</b>知道植入编辑并直接逆转。",
        "ref_noop": "<b>No-op（0）：</b>不做任何修改，直接提交坏模型。",
        "ref_script": "<b>固定脚本：</b>不使用 LLM 的固定可解释性流程。它与各 LLM 的分差，是衡量 LLM 额外贡献的直接参照。",
        "ref_redteam": "<b>Red-team harness：</b>最强的固定无 LLM 脚本，用来主动压测基准。",
        "interpretation": "如何解读",
        "interpretation_text": "这些合成缺陷用于实现精确评分。高分证明模型擅长执行给定实验闭环，但不能证明它能仅凭原始权重推断任意神经网络的用途。",
        "score": "得分",
        "repair": "修复率",
        "collateral": "副作用",
        "localization": "定位",
        "cost": "成本",
        "yes": "是",
        "no": "否",
        "trace_none": "本次运行没有记录轨迹。",
        "trace_missing": "轨迹文件不可用：{name}",
        "trace_malformed": "轨迹在第 {line} 行解析失败：{error}",
        "reasoning": "推理",
        "executed_python": "执行的 Python",
        "gpu_metric": "累计 {count} 次 GPU 调用",
        "turn": "第",
        "turn_label": "第 {number:02d} 轮",
        "stop": "停止原因",
        "label_separator": "：",
        "turns": "轮",
        "harness": "工具版本",
        "stop_submitted": "已提交",
        "stop_max_turns": "达到轮数上限",
        "stop_max_usd": "达到成本上限",
        "stop_gpu_budget": "GPU 预算耗尽",
        "stop_error": "错误",
        "trace_description": "{model} 在 {instance} 上的 WSP-Bench 公开轨迹。",
        "role_system": "系统",
        "role_user": "用户",
        "role_assistant": "助手",
        "role_tool": "工具",
        "tier_easy": "简单",
        "tier_medium": "中等",
        "tier_hard": "困难",
        "review_title": "Astra 修改 Qwen 数学权重：博客说法与公开日志对照",
        "review_description": "依据原作者公开日志，对 Experiment 8 逐条核查与分析。",
    },
}


def tr(lang: str, key: str, **values: Any) -> str:
    return TEXT[lang][key].format(**values)


SITE_CSS = r"""
:root{--bg:#F5F4EE;--paper:#FAF9F5;--ink:#141413;--accent:#D97757;--green:#6A8F6B;--gray:#8A867D;--soft:#3D3B36;--line:#E2DED3;--beige:#EFE9DC}
*{box-sizing:border-box}html{background:var(--bg);scroll-behavior:smooth}
body{margin:0;color:var(--ink);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;-webkit-font-smoothing:antialiased}
html[lang^="zh"] body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Noto Sans CJK SC","Microsoft YaHei",sans-serif}
a{color:inherit;text-decoration-color:var(--accent);text-underline-offset:3px}
.shell{width:min(1160px,calc(100% - 40px));margin:auto}.narrow{width:min(820px,calc(100% - 40px));margin:auto}
header.site{border-bottom:1px solid var(--line);background:rgba(245,244,238,.94);position:sticky;top:0;z-index:10;backdrop-filter:blur(8px)}
.nav{height:58px;display:flex;align-items:center;gap:25px}.brand{font:500 21px Georgia,serif;text-decoration:none;margin-right:auto}.nav a:not(.brand){font-size:13px;text-decoration:none;color:var(--soft)}.nav a:hover{color:var(--accent)}.language{display:flex;align-items:center;gap:6px;border-left:1px solid var(--line);padding-left:18px;font-size:12px}.language a.active{color:var(--accent);font-weight:700}.language span{color:var(--line)}
.hero{display:grid;grid-template-columns:1.3fr .7fr;gap:44px;padding:72px 0 52px;align-items:center}
.eyebrow{color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
h1,h2,h3,.score{font-family:Georgia,"Times New Roman",serif;font-weight:500}h1{font-size:clamp(42px,6vw,72px);line-height:1.02;margin:13px 0 20px;letter-spacing:-.025em}h2{font-size:30px;line-height:1.2;margin:0 0 8px}h3{font-size:21px;margin:28px 0 8px}
html[lang^="zh"] h1,html[lang^="zh"] h2,html[lang^="zh"] h3{font-family:"Songti SC","Noto Serif CJK SC",STSong,serif}
.lede{font-size:19px;line-height:1.6;color:var(--soft);max-width:720px}.hero img{width:100%;border:1px solid var(--line);border-radius:10px;background:var(--paper)}
.pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:24px}.pill{border:1px solid var(--line);border-radius:999px;padding:5px 12px;font-size:12px;color:var(--soft);background:var(--paper)}
section{padding:42px 0;border-top:1px solid var(--line)}.section-head{display:flex;align-items:end;justify-content:space-between;gap:20px;margin-bottom:22px}.muted{color:var(--gray)}.small{font-size:13px}.kicker{font-size:13px;color:var(--gray);margin:0}
.split-tabs{display:flex;gap:6px}.split-tabs button{border:1px solid var(--line);background:var(--paper);padding:7px 15px;border-radius:999px;cursor:pointer;color:var(--soft)}.split-tabs button.active{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.table-wrap{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--paper)}
table{border-collapse:collapse;width:100%;font-size:13px}th{padding:11px 12px;text-align:right;border-bottom:1.5px solid var(--ink);white-space:nowrap;font-weight:650;cursor:pointer}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:inherit}td{padding:12px;text-align:right;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:#fff}
tr.baseline{background:var(--beige)}tr.baseline-oracle{box-shadow:inset 4px 0 var(--green)}tr.baseline-noop{box-shadow:inset 4px 0 var(--gray)}tr.baseline-scripted{box-shadow:inset 4px 0 var(--accent)}
.model{font-weight:650;white-space:nowrap}.model small{display:block;font-weight:400;color:var(--gray)}.score{font-size:20px}.ci{display:block;font:11px -apple-system,sans-serif;color:var(--gray)}
.empty{padding:40px;border:1px solid var(--line);border-radius:10px;background:var(--paper)}.empty h3{margin-top:0}.roster-groups{display:grid;grid-template-columns:2fr 1fr 1fr;gap:24px;margin-top:26px}.roster-group h4{margin:0 0 8px;font-size:13px}.roster{list-style:none;padding:0;margin:0}.roster li{display:flex;justify-content:space-between;gap:12px;padding:5px 0;color:var(--soft);border-bottom:1px solid var(--line)}.roster li:last-child{border-bottom:0}.roster li b{color:var(--gray);font-weight:500}
.chart{border:1px solid var(--line);border-radius:10px;background:var(--paper);padding:10px}.chart svg{display:block;width:100%;height:auto}.chart .grid{stroke:var(--line);stroke-width:1}.chart .axis{fill:var(--gray);font:12px -apple-system,sans-serif}.chart .label{fill:var(--soft);font:11px -apple-system,sans-serif}.chart .frontier{stroke:var(--accent);stroke-width:2;fill:none;stroke-dasharray:5 4}.chart .point{fill:var(--green);stroke:var(--paper);stroke-width:2}.chart .basepoint{fill:var(--accent);stroke:var(--paper);stroke-width:2}
.caveats{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.card{border:1px solid var(--line);border-radius:9px;background:var(--paper);padding:19px}.card b{display:block;margin-bottom:5px}.card p{margin:0;color:var(--soft);font-size:14px}
.method{padding:66px 0}.method p,.method li{color:var(--soft)}.formula{font:17px ui-monospace,SFMono-Regular,Menlo,monospace;padding:16px;border-left:3px solid var(--accent);background:var(--paper);overflow:auto}.family-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.family-grid .card{min-height:145px}
.trace-head{padding:58px 0 32px}.trace-stats{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;background:var(--line);border:1px solid var(--line);border-radius:9px;overflow:hidden;margin-top:25px}.stat{background:var(--paper);padding:13px}.stat span{display:block;color:var(--gray);font-size:11px;text-transform:uppercase;letter-spacing:.05em}.stat b{font-size:16px}
.turns{padding-bottom:70px}.turn{display:grid;grid-template-columns:110px 1fr;gap:18px;margin:0 0 14px}.turn-meta{font:12px ui-monospace,Menlo,monospace;color:var(--gray);padding-top:12px}.bubble{border:1px solid var(--line);border-left:4px solid var(--gray);border-radius:9px;background:var(--paper);padding:14px 17px;min-width:0}.turn.user .bubble{border-left-color:var(--accent)}.turn.assistant .bubble{border-left-color:var(--green)}.turn.tool .bubble{border-style:dashed}.role{font-size:11px;letter-spacing:.08em;font-weight:750;color:var(--gray);text-transform:uppercase}.content{white-space:pre-wrap;overflow-wrap:anywhere}.bubble pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--beige);border-radius:6px;padding:12px;font:12px/1.55 ui-monospace,Menlo,monospace}.bubble details{border-top:1px solid var(--line);margin-top:12px;padding-top:8px}.bubble summary{cursor:pointer;color:var(--accent);font-size:12px}.turn-cost{font-size:11px;color:var(--gray);margin-top:10px}
.missing{padding:24px;border:1px dashed var(--line);border-radius:9px;background:var(--paper);color:var(--soft)}
footer.site{border-top:1px solid var(--line);padding:24px 0 45px;color:var(--gray);font-size:12px}.footer-row{display:flex;justify-content:space-between;gap:20px}
.watermark{position:fixed;right:-48px;top:42px;transform:rotate(35deg);background:var(--accent);color:white;padding:7px 62px;z-index:30;font-size:11px;font-weight:800;letter-spacing:.12em;pointer-events:none}
@media(max-width:800px){.hero{grid-template-columns:1fr;padding-top:48px}.hero img{display:none}.caveats,.family-grid,.roster-groups{grid-template-columns:1fr}.trace-stats{grid-template-columns:repeat(3,1fr)}.turn{grid-template-columns:1fr;gap:4px}.turn-meta{padding-top:0}.nav{gap:12px}.nav>a:not(.brand){display:none}.language{padding-left:10px}.section-head{align-items:start;flex-direction:column}}
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
document.querySelectorAll("[data-language]").forEach(link => {
  link.addEventListener("click", () => localStorage.setItem("wsp-language", link.dataset.language));
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


def social_meta(title: str, description: str) -> str:
    image = f"{SITE_URL}/assets/og.png"
    return f"""<meta name="description" content="{esc(description)}">
<meta property="og:type" content="website"><meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(description)}"><meta property="og:image" content="{esc(image)}">
<meta name="twitter:card" content="summary_large_image"><meta name="twitter:title" content="{esc(title)}">
<meta name="twitter:description" content="{esc(description)}"><meta name="twitter:image" content="{esc(image)}">"""


def page(
    lang: str,
    title: str,
    body: str,
    *,
    prefix: str = "",
    en_href: str,
    zh_href: str,
    description: str | None = None,
    preview: bool = False,
    auto_redirect: bool = False,
) -> str:
    watermark = (
        f'<div class="watermark">{tr(lang, "preview")}</div>' if preview else ""
    )
    language_dir = "zh/" if lang == "zh" else ""
    home = f"{prefix}{language_dir}index.html"
    method = f"{prefix}{language_dir}method.html"
    review = f'{prefix}review/{"zh.html" if lang == "zh" else "index.html"}'
    redirect = ""
    if auto_redirect:
        redirect = f"""<script>
(() => {{
  try {{
    const query = new URLSearchParams(location.search);
    const requested = query.get("lang");
    if (requested === "en") {{
      localStorage.setItem("wsp-language", "en");
      return;
    }}
    if (requested === "zh") {{
      localStorage.setItem("wsp-language", "zh");
      location.replace("{zh_href}");
      return;
    }}
    const stored = localStorage.getItem("wsp-language");
    if (stored === "zh" || (!stored && navigator.language.toLowerCase().startsWith("zh"))) {{
      if (!stored) localStorage.setItem("wsp-language", "zh");
      location.replace("{zh_href}");
    }}
  }} catch (_) {{}}
}})();
</script>"""
    language = "zh-CN" if lang == "zh" else "en"
    active_en = " active" if lang == "en" else ""
    active_zh = " active" if lang == "zh" else ""
    description = description or tr(lang, "description")
    return f"""<!doctype html>
<html lang="{language}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>{social_meta(title, description)}{redirect}
<link rel="stylesheet" href="{prefix}assets/site.css"></head><body>{watermark}
<header class="site"><nav class="nav shell"><a class="brand" href="{home}">WSP-Bench</a>
<a href="{home}#leaderboard">{tr(lang, "nav_leaderboard")}</a><a href="{method}">{tr(lang, "nav_method")}</a>
<a href="{review}">{tr(lang, "nav_review")}</a><a href="https://github.com/GPT-AGI/Weight-Space-Programing">GitHub</a>
<span class="language"><a class="{active_en.strip()}" data-language="en" href="{en_href}">EN</a><span>/</span><a class="{active_zh.strip()}" data-language="zh" href="{zh_href}">中文</a></span></nav></header>
{body}
<footer class="site"><div class="footer-row shell"><span>{tr(lang, "footer")}</span><span>Qwen3.5-0.8B · v0.1</span></div></footer>
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


def table_html(summary: list[dict[str, Any]], lang: str) -> str:
    header = (
        f"<thead><tr><th>{tr(lang, 'model')}</th><th>{tr(lang, 'overall')}</th>"
        + "".join(f"<th>{tr(lang, f'family_{name}')}</th>" for name in FAMILIES)
        + f"<th>{tr(lang, 'cost_full')}</th><th>{tr(lang, 'tokens')}</th><th>{tr(lang, 'gpu_calls')}</th></tr></thead>"
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
            f'<small>{tr(lang, "instances", count=item["instances"])}</small></span></td>'
        )
        lo, hi = item["ci"]
        score_cell = (
            f'<td data-value="{item["score"]:.5f}"><span class="score">{item["score"]:.1f}</span>'
            f'<span class="ci">{tr(lang, "ci", lo=lo, hi=hi)}</span></td>'
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


def pareto_svg(summary: list[dict[str, Any]], lang: str) -> str:
    points = [
        row
        for row in summary
        if row["model"].lower() not in ("oracle", "noop") and row["cost"] >= 0
    ]
    if not points:
        return f'<div class="missing">{tr(lang, "pareto_empty")}</div>'

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
            f"<title>{esc(tr(lang, 'pareto_tooltip', name=row['display_name'], score=row['score'], cost=row['cost']))}</title></circle>"
            f'<text class="label" x="{xx+8:.1f}" y="{label_y:.1f}">{esc(row["display_name"])}{suffix}</text>'
        )
    return (
        f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(tr(lang, "pareto_aria"))}">'
        + "".join(grid)
        + frontier
        + "".join(dots)
        + f'<text class="axis" x="{left+plot_w/2:.1f}" y="{height-8}" text-anchor="middle">{tr(lang, "pareto_x")}</text>'
        + f'<text class="axis" transform="translate(16 {top+plot_h/2:.1f}) rotate(-90)" text-anchor="middle">{tr(lang, "pareto_y")}</text>'
        + "</svg></div>"
    )


def empty_state(lang: str) -> str:
    def group(title: str, rows: Iterable[tuple[str, str]]) -> str:
        items = "".join(
            f"<li><span>{esc(name)}</span><b>{esc(score)}</b></li>"
            for name, score in rows
        )
        return (
            f'<div class="roster-group"><h4>{esc(title)}</h4>'
            f'<ul class="roster">{items}</ul></div>'
        )

    baseline_rows = (
        (tr(lang, "oracle_label"), "100"),
        (tr(lang, "redteam_label"), "?"),
        (tr(lang, "scripted_label"), "?"),
        (tr(lang, "noop_label"), "0"),
    )
    roster_html = (
        f'<p class="small muted">{tr(lang, "empty_roster")}</p>'
        '<div class="roster-groups">'
        + group(tr(lang, "model_track"), ((name, "?") for name in MODEL_ROSTER))
        + group(tr(lang, "agent_track"), ((name, "?") for name in AGENT_ROSTER))
        + group(tr(lang, "baselines"), baseline_rows)
        + "</div>"
    )
    return (
        f'<div class="empty"><p class="eyebrow">{tr(lang, "empty_eyebrow")}</p>'
        f"<h3>{tr(lang, 'empty_title')}</h3>"
        f"<p>{tr(lang, 'empty_text')}</p>"
        f"{roster_html}</div>"
    )


def build_index(
    results: list[dict[str, Any]], output: Path, lang: str, preview: bool
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
            else f'<span class="pill">{tr(lang, "split", split=splits[0])}</span>'
        )
        panels = []
        charts = []
        for index, split in enumerate(splits):
            summary = aggregate(by_split[split])
            panels.append(
                f'<div data-split-panel="{split}"{" hidden" if index else ""}>{table_html(summary, lang)}</div>'
            )
            charts.append(
                f'<div data-split-panel="{split}"{" hidden" if index else ""}>{pareto_svg(summary, lang)}</div>'
            )
        leaderboard = "".join(panels)
        chart = "".join(charts)
    else:
        buttons = ""
        leaderboard = empty_state(lang)
        chart = pareto_svg([], lang)

    body = f"""
<main><div class="hero shell"><div><p class="eyebrow">{tr(lang, "hero_eyebrow")}</p>
<h1>{tr(lang, "hero_title")}</h1>
<p class="lede">{tr(lang, "hero_lede")}</p>
<div class="pills"><span class="pill">{tr(lang, "pill_families")}</span><span class="pill">{tr(lang, "pill_tiers")}</span><span class="pill">{tr(lang, "pill_bounds")}</span><span class="pill">{tr(lang, "pill_script")}</span></div></div>
<img src="{'../' if lang == 'zh' else ''}assets/hero.gif" alt="{tr(lang, 'hero_alt')}"></div>
<section id="leaderboard"><div class="shell"><div class="section-head"><div><h2>{tr(lang, "leaderboard")}</h2>
<p class="kicker">{tr(lang, "leaderboard_caption")}</p></div>{buttons}</div>{leaderboard}</div></section>
<section><div class="shell"><div class="section-head"><div><h2>{tr(lang, "pareto_title")}</h2>
<p class="kicker">{tr(lang, "pareto_caption")}</p></div></div>{chart}</div></section>
<section><div class="shell"><div class="section-head"><div><h2>{tr(lang, "caveats_title")}</h2>
<p class="kicker">{tr(lang, "caveats_caption")}</p></div><a href="method.html">{tr(lang, "method_link")}</a></div>
<div class="caveats"><div class="card"><b>{tr(lang, "caveat_synthetic")}</b><p>{tr(lang, "caveat_synthetic_text")}</p></div>
<div class="card"><b>{tr(lang, "caveat_native")}</b><p>{tr(lang, "caveat_native_text")}</p></div>
<div class="card"><b>{tr(lang, "caveat_cost")}</b><p>{tr(lang, "caveat_cost_text")}</p></div></div></div></section></main>"""
    prefix = "../" if lang == "zh" else ""
    en_href = "../index.html?lang=en" if lang == "zh" else "index.html?lang=en"
    zh_href = "index.html" if lang == "zh" else "zh/index.html"
    (output / "index.html").write_text(
        page(
            lang,
            tr(lang, "index_title"),
            body,
            prefix=prefix,
            en_href=en_href,
            zh_href=zh_href,
            preview=preview,
            auto_redirect=lang == "en",
        ),
        encoding="utf-8",
    )


def build_method(output: Path, lang: str, preview: bool) -> None:
    body = f"""
<main class="method narrow"><p class="eyebrow">{tr(lang, "method_eyebrow")}</p><h1>{tr(lang, "method_h1")}</h1>
<p class="lede">{tr(lang, "method_lede")}</p>
<h2>{tr(lang, "task_families")}</h2><div class="family-grid">
<div class="card"><b>{tr(lang, "behavior_name")}</b><p>{tr(lang, "behavior_desc")}</p></div>
<div class="card"><b>{tr(lang, "backdoor_name")}</b><p>{tr(lang, "backdoor_desc")}</p></div>
<div class="card"><b>{tr(lang, "audit_name")}</b><p>{tr(lang, "audit_desc")}</p></div>
<div class="card"><b>{tr(lang, "hijack_name")}</b><p>{tr(lang, "hijack_desc")}</p></div>
<div class="card"><b>{tr(lang, "fact_name")}</b><p>{tr(lang, "fact_desc")}</p></div>
<div class="card"><b>{tr(lang, "topic_name")}</b><p>{tr(lang, "topic_desc")}</p></div></div>
<h2>{tr(lang, "scoring")}</h2><p>{tr(lang, "scoring_intro")}</p>
<div class="formula">score = 100 × repair × exp(−KL / 0.05)<br>
backdoor_score = score × (0.4 + 0.6 × trigger_found)<br>
audit_score = 100 × (layer_credit + direction_credit + category_credit)</div>
<p>{tr(lang, "scoring_detail")}</p>
<h2>{tr(lang, "tracks")}</h2><ul><li>{tr(lang, "model_track_text")}</li><li>{tr(lang, "agent_track_text")}</li></ul>
<h2>{tr(lang, "fairness")}</h2><ul>
<li>{tr(lang, "fair_1")}</li>
<li>{tr(lang, "fair_2")}</li>
<li>{tr(lang, "fair_3")}</li>
<li>{tr(lang, "fair_4")}</li>
<li>{tr(lang, "fair_5")}</li>
</ul>
<h2>{tr(lang, "budget")}</h2><p>{tr(lang, "budget_text")}</p>
<h2>{tr(lang, "references")}</h2><ul><li>{tr(lang, "ref_oracle")}</li><li>{tr(lang, "ref_noop")}</li><li>{tr(lang, "ref_script")}</li><li>{tr(lang, "ref_redteam")}</li></ul>
<h2>{tr(lang, "interpretation")}</h2><p>{tr(lang, "interpretation_text")}</p></main>"""
    prefix = "../" if lang == "zh" else ""
    en_href = "../method.html?lang=en" if lang == "zh" else "method.html?lang=en"
    zh_href = "method.html" if lang == "zh" else "zh/method.html"
    (output / "method.html").write_text(
        page(
            lang,
            tr(lang, "method_title"),
            body,
            prefix=prefix,
            en_href=en_href,
            zh_href=zh_href,
            preview=preview,
        ),
        encoding="utf-8",
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


def load_trace(
    result: dict[str, Any], fixtures: bool, lang: str
) -> tuple[list[dict[str, Any]], str]:
    path = trace_source(result, fixtures)
    if path is None:
        return [], tr(lang, "trace_none")
    if not path.exists():
        return [], tr(lang, "trace_missing", name=path.name)
    turns = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            turn = json.loads(line)
        except json.JSONDecodeError as exc:
            return turns, tr(lang, "trace_malformed", line=number, error=exc)
        if isinstance(turn, dict):
            turns.append(turn)
    return turns, ""


def component_value(components: dict[str, Any], key: str, lang: str) -> str:
    value = components.get(key)
    if isinstance(value, bool):
        return tr(lang, "yes") if value else tr(lang, "no")
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return esc(value)


def turn_html(turn: dict[str, Any], lang: str) -> str:
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
            f'<details><summary>{tr(lang, "reasoning")}</summary><div class="content">{esc(reasoning)}</div></details>'
        )
    if code:
        extras += f'<details open><summary>{tr(lang, "executed_python")}</summary><pre><code>{esc(code)}</code></pre></details>'
    metrics = (
        f"${float(turn.get('cost_usd', 0) or 0):.4f} · {usage_total:,} {tr(lang, 'tokens')} · "
        f"{tr(lang, 'gpu_metric', count=int(turn.get('gpu_calls_used', 0) or 0))} · "
        f"{float(turn.get('wall_s', 0) or 0):.1f}s"
    )
    role_label = tr(lang, f"role_{role}") if f"role_{role}" in TEXT[lang] else role
    turn_label = tr(lang, "turn_label", number=int(turn.get("t", 0)))
    return (
        f'<article class="turn {esc(role)}"><div class="turn-meta">{turn_label}</div>'
        f'<div class="bubble"><div class="role">{esc(role_label)}</div>'
        f'<div class="content">{esc(turn.get("content", ""))}</div>{extras}'
        f'<div class="turn-cost">{metrics}</div></div></article>'
    )


def build_traces(
    results: list[dict[str, Any]],
    output: Path,
    fixtures: bool,
    lang: str,
    preview: bool,
) -> None:
    for result in results:
        model = str(result.get("model", "unknown"))
        if model.lower() in BASELINES and not result.get("trace"):
            continue
        split = str(result.get("split", "dev"))
        instance = str(result.get("instance_id", "instance"))
        destination = output / "runs" / split / slug(model) / f"{safe_instance(instance)}.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        turns, error = load_trace(result, fixtures, lang)
        components = result.get("components") or {}
        stats = [
            (tr(lang, "score"), f"{float(result.get('score', 0) or 0):.1f}"),
            (tr(lang, "repair"), component_value(components, "repair", lang)),
            (
                tr(lang, "collateral"),
                component_value(components, "collateral_penalty", lang),
            ),
            (
                tr(lang, "localization"),
                component_value(components, "localization", lang),
            ),
            (tr(lang, "cost"), f"${float(result.get('cost_usd', 0) or 0):.3f}"),
            (tr(lang, "gpu_calls"), str(int(result.get("gpu_calls", 0) or 0))),
        ]
        stat_html = "".join(
            f'<div class="stat"><span>{esc(label)}</span><b>{value}</b></div>'
            for label, value in stats
        )
        if turns:
            trace_body = "".join(turn_html(turn, lang) for turn in turns)
            if error:
                trace_body += f'<div class="missing">{esc(error)}</div>'
        else:
            trace_body = f'<div class="missing">{esc(error)}</div>'
        family = tr(lang, f"family_{result.get('family')}") if result.get("family") in FAMILIES else result.get("family", "")
        tier = tr(lang, f"tier_{result.get('tier')}") if result.get("tier") in ("easy", "medium", "hard") else result.get("tier", "")
        stop_reason = str(result.get("stop_reason", "error"))
        stop_label = tr(lang, f"stop_{stop_reason}") if f"stop_{stop_reason}" in TEXT[lang] else stop_reason
        body = f"""
<main><div class="trace-head narrow"><p class="eyebrow">{esc(split)} · {esc(family)} · {esc(tier)}</p>
<h1>{esc(result.get("display_name") or model)}</h1><p class="lede">{esc(instance)}</p>
<div class="trace-stats">{stat_html}</div>
<p class="small muted">{tr(lang, "stop")}{tr(lang, "label_separator")} {esc(stop_label)} · {int(result.get("turns", len(turns)) or 0)} {tr(lang, "turns")} · {float(result.get("wall_s", 0) or 0):.1f}s · {tr(lang, "harness")} {esc(result.get("harness_version", "unknown"))}</p></div>
<div class="turns narrow">{trace_body}</div></main>"""
        rel_path = f"runs/{split}/{slug(model)}/{safe_instance(instance)}.html"
        if lang == "zh":
            prefix = "../../../../"
            en_href = f"../../../../{rel_path}?lang=en"
            zh_href = destination.name
        else:
            prefix = "../../../"
            en_href = f"{destination.name}?lang=en"
            zh_href = f"../../../zh/{rel_path}"
        destination.write_text(
            page(
                lang,
                f"{result.get('display_name') or model} · {instance}",
                body,
                prefix=prefix,
                en_href=en_href,
                zh_href=zh_href,
                description=tr(
                    lang,
                    "trace_description",
                    model=result.get("display_name") or model,
                    instance=instance,
                ),
                preview=preview,
            ),
            encoding="utf-8",
        )


def review_copy(source: Path, destination: Path, lang: str) -> None:
    text = source.read_text(encoding="utf-8")
    meta = social_meta(
        tr(lang, "review_title"),
        tr(lang, "review_description"),
    )
    header_css = """<style>
.review-site-header{position:sticky;top:0;z-index:30;border-bottom:1px solid #E2DED3;background:rgba(245,244,238,.94);backdrop-filter:blur(8px)}
.review-nav{max-width:1160px;height:58px;margin:auto;padding:0 24px;display:flex;align-items:center;gap:25px;font:13px -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
.review-nav a{color:#3D3B36;text-decoration:none}.review-nav a:hover{color:#D97757}.review-brand{margin-right:auto!important;font:500 21px Georgia,serif!important;color:#141413!important}
.review-language{display:flex;align-items:center;gap:6px;border-left:1px solid #E2DED3;padding-left:18px}.review-language a.active{color:#D97757;font-weight:700}.review-language span{color:#E2DED3}
@media(max-width:800px){.review-nav>a:not(.review-brand){display:none}.review-nav{gap:12px}.review-language{padding-left:10px}}
</style>"""
    home = "../zh/index.html" if lang == "zh" else "../index.html"
    method = "../zh/method.html" if lang == "zh" else "../method.html"
    current = "zh.html" if lang == "zh" else "index.html"
    en_href = "index.html?lang=en"
    zh_href = "zh.html"
    active_en = "active" if lang == "en" else ""
    active_zh = "active" if lang == "zh" else ""
    header = f"""<header class="review-site-header"><nav class="review-nav">
<a class="review-brand" href="{home}">WSP-Bench</a>
<a href="{home}#leaderboard">{tr(lang, "nav_leaderboard")}</a><a href="{method}">{tr(lang, "nav_method")}</a>
<a href="{current}">{tr(lang, "nav_review")}</a><a href="https://github.com/GPT-AGI/Weight-Space-Programing">GitHub</a>
<span class="review-language"><a class="{active_en}" data-language="en" href="{en_href}">EN</a><span>/</span><a class="{active_zh}" data-language="zh" href="{zh_href}">中文</a></span>
</nav></header>"""
    language_script = """<script>document.querySelectorAll("[data-language]").forEach(link=>link.addEventListener("click",()=>localStorage.setItem("wsp-language",link.dataset.language)));</script>"""
    text = text.replace("</head>", f"{meta}{header_css}</head>", 1)
    text = text.replace(
        "<body>", f"<body>{header}", 1
    )
    text = text.replace("</body>", f"{language_script}</body>", 1)
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
    preserved_assets = {
        name: (output / "assets" / name).read_bytes()
        for name in PRESERVED_DOC_ASSETS
        if (output / "assets" / name).is_file()
    }
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    results = load_results(args.fixtures)
    validate_results(results)
    copy_assets(output)
    for name, data in preserved_assets.items():
        (output / "assets" / name).write_bytes(data)
    for lang in ("en", "zh"):
        lang_output = output if lang == "en" else output / "zh"
        lang_output.mkdir(parents=True, exist_ok=True)
        build_index(results, lang_output, lang, args.fixtures)
        build_method(lang_output, lang, args.fixtures)
        build_traces(results, lang_output, args.fixtures, lang, args.fixtures)

    review_out = output / "review"
    review_out.mkdir(parents=True, exist_ok=True)
    review_copy(
        ROOT / "doc/source_blog_review.en.html",
        review_out / "index.html",
        "en",
    )
    review_copy(
        ROOT / "doc/source_blog_review.html",
        review_out / "zh.html",
        "zh",
    )
    print(f"built {len(results)} results into {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
