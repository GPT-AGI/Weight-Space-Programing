# WSP-Bench GPU 实测（RTX 4080 Super）

日期：2026-11-03。模型：`Qwen/Qwen3.5-0.8B`，revision
`2fc06364715b967f1860aea9cf38778875588b17`，bfloat16，thinking 关闭，
Transformers 贪心解码。

## 模型与参数名

- 文本模型成功单独加载为 `Qwen3_5ForCausalLM`：24 层、hidden 1024、
  intermediate 3584；第 3/7/11/15/19/23 层为 full attention，其余为
  Gated DeltaNet。
- Host 不把视觉塔和 MTP 权重导出到 episode。
- SDK 使用文本 checkpoint 名字，例如
  `model.layers.14.mlp.down_proj.weight`，形状 `[1024, 3584]`。
- 单条 16-token 冷启动生成约 1.01 s；模型加载 9.21 s；基础显存约
  1,496 MiB。

## 吞吐

| 操作 | 实测 |
|---|---:|
| generate：64 prompts × 128 new tokens | 13.37 s |
| read：256 texts × 256 tokens × 24 层（`resid_post/mean`） | 4.30 s |
| 应用 64 列 `down_proj` 编辑 | 0.237 ms |
| 精确恢复原始副本 | 0.037 ms |
| 峰值 PyTorch 显存 | 3,607 MiB |

`causal-conv1d` 和 `flash-linear-attention` 的安装尝试因 WSL 环境没有
`nvcc` 而失败；上表使用 Transformers 的正确但较慢的 PyTorch fallback。
当前速度已满足 v0.1，未为此引入本地 CUDA 编译依赖。

## v1 历史校准

每题使用 16 个隐藏 prompt、最多 64 个新 token。分数已经包含
collateral penalty；括号内为 mean per-token KL。`projection-ref` 同时列出
只投影 planted columns / 投影全部 columns 的结果。

| instance | clean | broken | oracle (KL) | projection-ref planted / all (KL) | scripted (KL) | random (KL) |
|---|---:|---:|---:|---:|---:|---:|
| behavior-easy | 0.000 | 1.000 | 100 (0) | 100 (.0009) / 100 (.0307) | 100 (.0009) | 0 (4.4785) |
| behavior-medium | 0.000 | 1.000 | 100 (0) | 100 (.0009) / 100 (.0010) | 0 (.0436) | 0 (.0375) |
| behavior-hard | 0.000 | .812 | 100 (0) | 100 (.0011) / 100 (.0016) | 0 (.0336) | 0 (.0064) |
| backdoor-easy | 0.000 | 1.000 | 100 (0) | 100 (.0007) / 100 (.0009) | 0 (2.4089) | 0 (2.3600) |
| backdoor-medium | 0.000 | 1.000 | 100 (0) | 100 (.0008) / 100 (.0008) | 0 (.0839) | 0 (.1410) |
| backdoor-hard | 0.000 | 1.000 | 100 (0) | 100 (.0010) / 100 (.0009) | 0 (.2340) | 0 (.5340) |
| audit-easy | 0.000 | 1.000 | 100 (0) | 100 (0) / 100 (0) | 100 (0) | 0 (0) |
| audit-medium | 0.000 | 1.000 | 100 (0) | 100 (0) / 100 (0) | 100 (0) | 0 (0) |
| audit-hard | 0.000 | 1.000 | 100 (0) | 100 (0) / 100 (0) | 100 (0) | 0 (0) |
| hijack-easy | 0.000 | 1.000 | 100 (0) | 100 (.0063) / 100 (.0262) | 0 (28.7813) | 0 (17.3870) |
| hijack-medium | 0.000 | 1.000 | 100 (0) | 100 (.0024) / 100 (.0255) | 100 (.0025) | 0 (18.4222) |
| hijack-hard | 0.000 | 1.000 | 100 (0) | 100 (.0023) / 100 (.0254) | 100 (.0023) | 0 (18.0796) |

验收结果：所有 broken rate ≥ 0.8，clean rate = 0；三道 backdoor 的
无触发率均为 0；Oracle = 100，No-op = 0，projection reference 全部
≥ 99，范数匹配随机方向全部为 0。固定通用脚本均值为 50：能解决 easy
behavior、三道 audit 和两道 sparse hijack，但不会猜到未知 trigger，
也无法修复较隐蔽的 fluent style vectors。完整校准最慢 27.1 s/题。

v1 当时使用的副作用阈值为 `kl_free = 0.050`、`kl_max = 0.120`。
全部列 projection reference 的最大
KL 为 0.0307，因此 free band 取约 1.6×；原 full-strength 范数匹配随机
control 的最小 KL 为 0.141，因此 cutoff 放在二者之间。低强度 style
随机 control 的 KL 可低至 0.0064，但其 repair 为 0，最终分数仍为 0。

medium / hard behavior 使用层特定 residual style direction；抽查输出是
完整法语句子而非单 token 复读。easy 和发布 GIF 保留原先诚实标注的
`Bonjour` collapse。

## Gym v2.1 最终校准

日期：2026-11-04。每题私有基座仍为逐层 MLP neuron permutation 加 6 个
benign rotation decoy；clean、KL 与 oracle 都相对该 `M_ref`。v2.1 将硬
KL 截断替换为 `exp(-KL / τ)`，冻结 `τ = 0.05`。

选择依据：六道 behavior / fact projection reference 的分数分别为
56.9、50.1、93.0、88.0、78.9、38.4，均值 67.6；norm-matched random
最高 0.20；所有 zero-KL oracle / column reference 均为 100。

现有验证结果离线重评分如下。`overall` 是该模型已完成题目的均值；
`common-9` 是五个模型共同完成的九题均值。

| model | behavior | backdoor | audit | hijack | fact | topic | overall | common-9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Grok 4.7 | 30.73 | 0.00 | 71.63 | 12.42 | 47.95 | 0.21 | 27.16 | 34.12 |
| GPT-6.1 Terra | 47.97 | 4.07 | 26.57 | 8.41 | 26.32 | 0.26 | 18.93 | 26.20 |
| DeepSeek V4.1 Flash | 43.59 | 0.61 | 0.00 | – | – | – | 14.73 | 14.73 |
| Composer 2.5 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Qwen3.8 Flash | 0.00 | 0.00 | 0.00 | – | – | – | 0.00 | 0.00 |

冻结 baseline 的 v2.1 重评分：

| baseline | behavior | backdoor | audit | hijack | fact | topic | overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| oracle | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| scripted | 32.28 | 4.27 | 60.00 | 10.38 | 24.44 | 0.26 | 21.94 |
| red-team | 66.98 | 1.10 | 0.00 | 9.32 | 31.08 | 0.04 | 18.09 |

`swap` / `rotate` 因 hidden generation symptom rate 约为 0，已从 v0.1
dev/test 集移除，代码和 staging specs 保留给 v0.2。v0.1 现为 6 族 ×
3 tiers = 18 题。Hard cross-family combo 按计划跳过。

## Gym v2.2 family difficulty curve

`τ=0.05` 不变。Backdoor / hijack / topic 的 easy、medium 改用更早的
低 collateral writer layer；topic easy / medium 仅对 hard tier 保留
whitened read key。括号内为 KL。

| instance | broken | oracle | projection-ref | column-ref | random | red-team | scripted |
|---|---:|---:|---:|---:|---:|---:|---:|
| backdoor-easy | 0.938 | 100 | 78.39 (0.0122) | 100 (0) | 0 (0.0404) | 9.58 | 0 |
| backdoor-medium | 0.938 | 100 | 59.19 (0.0262) | 100 (0) | 0 (0.3310) | 0 | 0 |
| backdoor-hard | 0.938 | 100 | 27.44 (0.0647) | 100 (0) | 0 (1.5323) | 0 | 9.53 |
| hijack-easy | 1.000 | 100 | 70.55 (0.0174) | 100 (0) | 0 (0.0749) | 73.93 | 73.92 |
| hijack-medium | 1.000 | 100 | 59.76 (0.0257) | 100 (0) | 0 (8.3404) | 0 | 0 |
| hijack-hard | 1.000 | 100 | 12.77 (0.1029) | 100 (0) | 0 (0.2491) | 0 | 3.39 |
| topic-easy | 0.938 | 100 | 68.60 (0.0188) | 100 (0) | 0 (0.1328) | 67.85 | 67.57 |
| topic-medium | 1.000 | 100 | 58.24 (0.0270) | 100 (0) | 0 (3.5162) | 0 | 0 |
| topic-hard | 0.875 | 100 | 0.23 (0.3045) | 100 (0) | 0 (0.9987) | 0.12 | 0.17 |

三族 red-team 均值分别为 3.19 / 24.64 / 22.66，scripted 均值分别为
3.18 / 25.77 / 22.58，均低于 50。所有 clean rate 为 0。Backdoor easy
trigger 改成一个 tokenizer token，brief 明示该性质；trigger gate 不变。

本轮变化使之前 validation 中 backdoor / hijack / topic 的模型结果失效。
仅对未变的 behavior / audit / fact 九题离线重评分：Grok 50.10、Terra
33.62、DeepSeek 18.83、Composer 0、Qwen 0；其余等待 fresh model run。

## 真实 Host/SDK 冒烟测试

`/admin/open` 在 1.52 s 内导出了 text-only broken checkpoint。随后用独立
`wsp_sdk` dev mode 完成了 `which="both"` 生成、两层 activation read、
单元素 safetensors 编辑（host 精确计数为 1）、reset 和 budget 查询。
Oracle / No-op API 分别返回 100 / 0；easy behavior 的 planted edit 实际
改变 32,766 个 bfloat16 元素。
