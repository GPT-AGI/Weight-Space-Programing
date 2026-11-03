<p align="center">
  <img src="assets/hero.gif" alt="WSP-Bench：谁是最强的 weight-space programmer？" width="880">
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/leaderboard-launching%20Nov%206-D97757" alt="排行榜 11 月 6 日上线">
  <img src="https://img.shields.io/badge/target-Qwen3.5--0.8B-6A8F6B" alt="目标模型 Qwen3.5-0.8B">
  <img src="https://img.shields.io/badge/traces-100%25%20public-141413" alt="轨迹全部公开">
  <img src="https://img.shields.io/badge/no--LLM%20baseline-included-8A867D" alt="含不用 LLM 的基线">
</p>

# 我们故意把 Qwen 弄坏了。哪个大模型能修好它？

最近一篇爆火的博客说：前沿大模型已经能**原生读写神经网络权重**。
我们把它公开的日志逐轮读完了。里面没有“读心术”，只有一套跑得很漂亮的教科书式可解释性流程。

所以我们把这套流程做成了一个基准。

**WSP-Bench** 往 Qwen3.5-0.8B 里植入一个已知缺陷，给每个大模型同一套 harness、同样的预算和一个断网沙箱，再按真值给修复打分。每一条轨迹、每一分钱都公开。一个不用 LLM 的 200 行脚本也会上场：它和每个模型之间的分差，就是 LLM 真正的贡献。

> 排行榜 **11 月 6 日**上线。点个 Star，第一时间收到通知。

## 那条爆款说法，拆成三个数字

| | |
|---|---|
| **50 次 GPU 前向** | 问题是靠把模型跑起来、对比激活找到的，不是盯着权重看出来的。 |
| **W ← W − α·u·uᵀW** | 修法是一个投影公式，只动两层各 64 列，和 2024 年的 refusal direction 消融是同一类做法。 |
| **答完率 7% → 65%** | 答完的题里，正确率从 93% 降到 71%。模型是不再卡死了，数学并没有变强。 |

<p align="center"><img src="assets/decomposition.gif" alt="MATH-500 的提升拆成答完率和答完题正确率" width="820"></p>

公平地说：原文的数字经得起核对，实验也做得规范，有留出验证、剂量-反应扫描、范数匹配的随机对照。而这正是重点：每一步都是 2022–2024 年的标准可解释性工具。

<p align="center"><img src="assets/loop.gif" alt="实验 8 背后的实验闭环" width="820"></p>

对照原作者日志的逐条核对：[`doc/source_blog_review.html`](doc/source_blog_review.html)

## WSP-Bench 怎么玩

1. **我们把模型弄坏。** 用一个已知的、精确的权重编辑，往 Qwen3.5-0.8B 里植入缺陷。
2. **大模型来排查。** 它拿到坏掉的权重和一个 GPU 工具：生成、读激活、试编辑、跑随机对照。沙箱断网，永远看不到干净的权重。
3. **我们给修复打分。** 在隐藏提示集上看缺陷消除了多少，再扣掉副作用（在通用文本上与干净模型的 KL 散度）。

| 题型 | 我们植入什么 | 大模型要做什么 |
|---|---|---|
| 行为植入 | 一个秩 1 编辑，让模型突然切换语言、全大写、复读 | 修好，且不伤别的能力 |
| 触发器后门 | 一个秩 1 的 key→value 写入，触发词未知 | 找出触发词，清除后门 |
| 定位审计 | 网络某处被动过手脚 | 只读不改，说出改在哪、改了什么 |
| 神经元劫持 | 几个神经元被放大，输出里刷屏同一个词 | 修好 |

每类题都分简单、中等、困难三档。

<p align="center"><img src="assets/projection.gif" alt="方向消融动画" width="820"></p>

### 比赛规则

- **同一套 harness、同样的预算、同一个 prompt。** 每轮只回一段 Python，不依赖各家的工具调用格式。
- **断网沙箱。** 不可能拿公开权重做差分作弊。
- **精确真值。** 缺陷是我们植入的，答案我们知道。
- **全部公开：** 每一轮、每一段推理、每一分钱。
- **上下有基线：** 知道答案的 Oracle（100 分）、什么都不做（0 分），以及不用 LLM 的固定脚本。

## 排行榜

**11 月 6 日**上线。

| 模型 | 得分 | 每套成本 | Token | GPU 调用 |
|---|---|---|---|---|
| Oracle（知道植入的编辑） | 100 | – | – | – |
| 固定脚本（不用 LLM） | ? | $0 | – | ? |
| DeepSeek V4.1 Flash | ? | ? | ? | ? |
| Qwen3.8 Flash | ? | ? | ? | ? |
| GLM-5.3 Flash | ? | ? | ? | ? |
| GPT-6 Luna | ? | ? | ? | ? |
| Grok 4.1 Fast | ? | ? | ? | ? |
| MiniMax M2.7 | ? | ? | ? | ? |
| Gemini 3.5 Flash-Lite | ? | ? | ? | ? |
| 什么都不做 | 0 | – | – | – |

上线后会持续加模型。想让你的模型上榜？v0.1 发布时开放提交。

## 这个项目不是什么

- **不是拆台。** 原文的结果是真的，编辑也确实有效。我们不同意的是叙事，而叙事之争最好用基准来回答。
- **不是在证明大模型“能读懂权重”。** 分数高，说明这个模型能把“找方向 → 改权重 → 验证”这套闭环跑好。
- **不是复现。** 我们不复用原作者的 harness、代码和编辑。

## 路线图

- [ ] **v0.1（11 月 6 日）**：WSP-Bench（4 类题 × 3 档难度）、排行榜、完整轨迹、分数-成本图
- [ ] 开放提交：自带模型或自带 agent harness 都可以
- [ ] **Boss 关**：修好 Qwen3.5-0.8B 真实的推理死循环，附并排对比动图，并在 Hugging Face 发布 `Qwen3.5-0.8B-noloop`
- [ ] 自我改进：一个跨回合学习的方法库，在留出的缺陷类型上衡量进步
- [ ] 模型能改进自己的权重吗？分阶段研究，由留出评测把关

规则与评分细节：[gpt-agi.github.io/Weight-Space-Programing/method.html](https://gpt-agi.github.io/Weight-Space-Programing/method.html)

## 致谢

灵感来自 [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)，作者 Rikiya Takehi、Jayden Teoh、Alex Zhang、Omar Khattab（MIT CSAIL）。
原作者日志：[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs)（CC BY 4.0）。
动图由 [`scripts/make_readme_assets.py`](scripts/make_readme_assets.py) 生成。

## 许可证

[MIT](LICENSE)。对照页中引用的原作者日志片段仍遵循 CC BY 4.0。
