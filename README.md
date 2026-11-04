<p align="center">
  <img src="assets/hero.gif" alt="WSP-Bench: who is the best weight-space programmer?" width="880">
</p>

<p align="center">
  <b>English</b> · <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/leaderboard-launching%20Nov%206-D97757" alt="leaderboard launching Nov 6">
  <img src="https://img.shields.io/badge/target-Qwen3.5--0.8B-6A8F6B" alt="target: Qwen3.5-0.8B">
  <img src="https://img.shields.io/badge/traces-100%25%20public-141413" alt="all traces public">
  <img src="https://img.shields.io/badge/no--LLM%20baseline-included-8A867D" alt="no-LLM baseline included">
</p>

# We broke Qwen on purpose. Which LLM can fix it?

A viral post says frontier LLMs can now **natively read and write neural network weights**.
We read every turn of its public logs. What we found wasn't telepathy. It was a textbook interpretability loop, run well.

So we turned that loop into a benchmark.

**WSP-Bench** plants a known defect in Qwen3.5-0.8B and scores the repair against exact ground truth. The Model track
gives every LLM the same harness, budget and offline sandbox; the Agent track lets teams bring their own harness.
Every trace and every dollar is public, with fixed no-LLM pipelines showing what the LLM actually adds.

> The leaderboard drops on **Nov 6**. Star the repo to get notified.

Follow along: [announcement thread](https://x.com/voidxtrain/status/2117818602044965316) by [@voidxtrain](https://x.com/voidxtrain).

<p align="center"><img src="assets/broken_qwen.gif" alt="Real Qwen3.5-0.8B output before and after a planted weight edit" width="880"></p>
<p align="center"><em>Real Qwen3.5-0.8B output on the same prompts, before and after a synthetic planted weight edit.</em></p>

## The viral claim, in three numbers

| | |
|---|---|
| **50 GPU forward passes** | The bug was found by running the model and comparing activations, not by staring at weights. |
| **W ← W − α·u·uᵀW** | The fix is one projection formula on 64 columns in two layers, the same family as 2024's refusal-direction ablation. |
| **7% → 65% finish rate** | Accuracy among finished answers fell from 93% to 71%. The model stopped looping. It didn't get better at math. |

<p align="center"><img src="assets/decomposition.gif" alt="MATH-500 gain decomposed into finish rate and accuracy among finished answers" width="820"></p>

To be fair, the post's numbers check out, and the experimental discipline is good: held-out problems, a dose-response
sweep, a norm-matched random control. That's exactly the point. Every step is a standard 2022–2024 interpretability tool.

<p align="center"><img src="assets/loop.gif" alt="The experiment loop behind Experiment 8" width="820"></p>

Claim-by-claim review against the authors' logs: [English](https://gpt-agi.github.io/Weight-Space-Programing/review/) · [简体中文](https://gpt-agi.github.io/Weight-Space-Programing/review/zh.html)

## How WSP-Bench works

1. **We build a private reference.** Qwen3.5-0.8B is transformed with function-preserving per-layer MLP neuron permutations and benign decoy edits.
2. **We plant a defect.** A known weight edit is applied to that private reference.
3. **The LLM investigates.** It gets the broken weights and a GPU tool: generate, read activations, try an edit,
   run a control. The bwrap sandbox is offline; the agent never sees the reference weights.
4. **We score the repair** on hidden prompts and discount collateral change on held-out general text.

| Family | What we plant | What the LLM must do |
|---|---|---|
| Behavior injection | a rank-1 edit that pushes a language, format or repetition behavior | repair it without side effects |
| Trigger backdoor | a rank-1 key→value edit tied to an unknown trigger | identify the trigger and remove the backdoor |
| Localization audit | a hidden edit somewhere in the network | report the layer, direction and symptom category without editing |
| Neuron hijack | a few amplified neurons that flood output with one token | find and repair the hijacked neurons |
| False fact | a ROME-style edit that installs a false factual association | restore the fact while preserving unrelated knowledge |
| Topic obsession | a fluent edit that pulls unrelated answers toward one topic | remove the obsession without degrading fluency |

Each family has easy, medium and hard tiers.

<p align="center"><img src="assets/projection.gif" alt="Directional ablation, animated" width="820"></p>

### Scoring v2.1

For repair tasks:

**score = 100 · repair · exp(−KL / 0.05)**

- `repair` is the share of the planted defect removed on hidden prompts.
- `KL` is mean per-token `KL(reference || submitted)` in nats on held-out general text.
- Backdoor repair is gated by trigger identification: multiply by `0.4 + 0.6 · trigger_found`.
- Audit scoring first requires the correct parameter; then localization is **40% exact layer hit** (20% for an adjacent layer), **40% absolute direction cosine**, and **20% symptom category**.

### Two tracks

- **Model track:** every LLM runs through the same WSP harness via raw OfoxAI API access.
- **Agent track:** bring your own harness. The first entries use the Cursor SDK with built-in tools disabled; the agent gets exactly one `run_python` tool.

### Rules of the game

- **Same task interface and tier budget.** Easy / medium / hard allow **12 / 8 / 6 GPU calls**.
- **Edit budget tied to ground truth.** `max(4,096, 2 × oracle edit size)` changed parameter elements; audit tasks cannot edit.
- **Offline bwrap sandbox.** Diffing against public Qwen weights is useless: the hidden reference has per-layer neuron permutations and benign decoys, and its weights never enter the agent sandbox.
- **Exact ground truth.** We planted the defect, so localization and repair can be scored directly.
- **Everything is published:** every turn, available reasoning trace, GPU call and dollar.
- **Four fixed reference rows:** oracle (100), no-op (0), a scripted no-LLM pipeline, and a stronger red-team no-LLM harness.

## Quickstart

Linux, an NVIDIA GPU, Python 3.11+, and `uv` are required. The public `dev` split is reproducible; official `test` instances and ground truth remain private.

```bash
git clone https://github.com/GPT-AGI/Weight-Space-Programing.git
cd Weight-Space-Programing
python -m venv .venv
source .venv/bin/activate
pip install -e '.[gpu,agent]'
pip install uv

export WSP_RUNTIME_DIR="$PWD/.wsp-runtime"
export WSP_PRIVATE_DIR="$PWD/.wsp-private"
export WSP_SANDBOX_ENV="$PWD/.wsp-sandbox"
python scripts/make_instances.py --split dev
python scripts/fetch_model.py --cache-dir "$PWD/.wsp-cache/hf"
scripts/build_sandbox_env.sh
```

Start the GPU host in one terminal:

```bash
source .venv/bin/activate
export WSP_RUNTIME_DIR="$PWD/.wsp-runtime"
export WSP_PRIVATE_DIR="$PWD/.wsp-private"
python -m wsp.host.server --cache-dir "$PWD/.wsp-cache/hf"
```

Then run one Model-track dev instance in another:

```bash
source .venv/bin/activate
export WSP_RUNTIME_DIR="$PWD/.wsp-runtime"
export WSP_PRIVATE_DIR="$PWD/.wsp-private"
export WSP_SANDBOX_ENV="$PWD/.wsp-sandbox"
export OFOX_API_KEY="..."
python scripts/run_bench.py \
  --track raw \
  --split dev \
  --models deepseek/deepseek-v4.1-flash-1017 \
  --instances dev-behavior-easy-01 \
  --sandbox-env "$WSP_SANDBOX_ENV"
```

The default OfoxAI endpoint is `https://api.ofox.ai/v1`; override it with `OFOX_BASE_URL`.

## Leaderboard

Launching **Nov 6**. No official model or agent scores are published yet.

### Model track

| Model | Score | $ / full run | Tokens | GPU calls |
|---|---|---|---|---|
| DeepSeek V4.1 Flash | ? | ? | ? | ? |
| Qwen3.8 Flash | ? | ? | ? | ? |
| GLM-5.3 Flash | ? | ? | ? | ? |
| GPT-6 Luna | ? | ? | ? | ? |
| Grok 4.1 Fast | ? | ? | ? | ? |
| MiniMax M2.7 | ? | ? | ? | ? |
| Gemini 3.5 Flash-Lite | ? | ? | ? | ? |
| GPT-6.1 Terra | ? | ? | ? | ? |

### Agent track

| Agent | Score | Cost | Tokens | GPU calls |
|---|---|---|---|---|
| Cursor Agent · Composer 2.5 | ? | ? | ? | ? |
| Cursor Agent · Grok 4.7 | ? | ? | ? | ? |

### Baselines

| Baseline | Score | LLM cost | GPU calls |
|---|---|---|---|
| Oracle (knows the planted edit) | 100 | $0 | – |
| Red-team harness (strongest fixed no-LLM script) | ? | $0 | ? |
| Scripted pipeline (no LLM) | ? | $0 | ? |
| No-op | 0 | $0 | – |

More models will be added after launch. Want yours on the board? Submissions open with v0.1.

## What this is not

- **Not a takedown.** The original result is real and the edit works. We disagree with the framing, and benchmarks settle framing debates.
- **Not proof that LLMs "read weights."** A high score means a model runs the find-direction → edit → verify loop well.
- **Not a reproduction.** We don't reuse the authors' harness, code or edits.

## Roadmap

- [ ] **v0.1 (Nov 6)**: WSP-Bench (6 families × 3 tiers), two-track leaderboard, full traces, score-vs-cost chart
- [ ] Open submissions: bring your own model or agent harness
- [ ] **Boss level**: fix Qwen3.5-0.8B's real reasoning loops, with a side-by-side demo and `Qwen3.5-0.8B-noloop` on Hugging Face
- [ ] Self-improvement: a playbook that learns across episodes, measured on held-out defect families
- [ ] Can a model improve its own weights? A staged study, gated by held-out evaluation

Rules and scoring in detail: [gpt-agi.github.io/Weight-Space-Programing/method.html](https://gpt-agi.github.io/Weight-Space-Programing/method.html)

## Credit

Inspired by [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab (MIT CSAIL).
Their logs: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) (CC BY 4.0).
The animations are generated by [`scripts/make_readme_assets.py`](scripts/make_readme_assets.py).

## License

[MIT](LICENSE). Excerpts quoted from the original authors' logs in the review remain under CC BY 4.0.
