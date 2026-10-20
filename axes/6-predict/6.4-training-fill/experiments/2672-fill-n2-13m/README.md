---
name: 2672-fill-n2-13m
ticket: 63-axis6-predict / Phase C · WP 6.4-WP1b（n≥2 复本）+ WP2（软提交 OOD 污染的等参数剥离）
date: "2026-10-14"
grid: 6.4-training-fill
engine: "depth-ar（`train_depth_ar.py`）；协议 = 既有 23e（d=256/L=8/h=8/mlp_mult=4/seq256/bs32/12000 步），实测参数量 ≈6.96M（以落盘 results.json 为准）"
budget: "seed-1 复本 3 臂 × ~10 min GPU（分两次派发，每次 ≤28 min）；等参数软/硬提交剥离为**纯评测**（复用已有 checkpoint，≤3 min）"
---

# 2672-fill-n2-13m · 填充语义的 n≥2 复本 + 软提交 OOD 污染的等参数剥离

> **本文件是预登记契约，写在跑之前**：协议、臂、判据、口径、否证条件在此写死，事后**不得**调参。
> 实测读数见 `results.json`（原始）与 `analysis.md`（当前结论）。
>
> **它回答的是什么**：6.4（`axes/6-predict/6.4-training-fill/project.yaml`）的 `milestone.output` 要求
> 「…在 matched-step（**n≥2**）下的 clean bpc + 填入位置可读性 + 提交质量（rescore）」，
> 而现状只有 seed 0 一个复本（`conclusion_review` 自述「填充语义对照无 n≥2 复本」）；
> WP2 又要求把「软提交污染上下文」**从单次现象升格为可复现判据，并剥离 adapter 容量成分**。

## 1 命题（可证伪）

**(P1)** 在**与既有读数逐字段相同**的协议下（只换训练种子），「保留式填充（fill）优于 ragged nofill」
与「proj-fill 优于 fill」两个排序是否可复现（n≥2）？
**(P2)** 「软提交（Σp·E）把分布外内容注入上下文」是否**不是容量效应**——在**同一个 checkpoint**上
只改提交规则（soft / hard / latent），rescore 峰值是否仍显著高于 hard？

可证伪形式：*若 n≥2 下两个排序的符号在 seed {0,1} 上不一致 ⇒ 判「该排序不可复现」；若同 checkpoint 下
soft 与 hard 的 rescore 峰值差在 seed {0,1} 上都不显著（配对 CI 含 0）⇒ 判「OOD 污染由容量/别的因素解释」。*

## 2 协议与臂（写死，等于既有 23e 协议，事后不得改）

| 项 | 值 |
|---|---|
| 结构 | `d_model=256 / n_layers=8 / n_heads=8 / mlp_mult=4 / seq_len=256`、tied 读出 |
| 训练 | `steps=12000 / batch_size=32 / lr=1e-3`，`aux_weight=1.0`、`aux_anneal_to=None`、`shallow_stopgrad=False`、`dense_aux=False`、`exit_mode=uniform`（随机退出深度 U{0..L}） |
| 种子 | **seed 0 已有读数**（复用 `.scratch/23-depth-ar/evidence/{23c-H2-w0-12k, 23e-nofill-12k, 23e-projfill-12k}`，只读）⇒ 本轮只跑 **seed 1**，构成 n≥2 |
| 预算声明 | 若时间盒不允许，**缩步数**（§20.3：臂间同刻度 + 显式声明「快照口径、与 12k 档不可比」），**禁止**砍臂 / 砍 seed / 改配方 |

| 臂 | 开关 | 语义 |
|---|---|---|
| `fill` | `fill_kv=True`（缺省）、`proj_fill=False` | 已退出位置的 KV 用**冻结表征**填充（self-fill） |
| `nofill` | `--no-fill-kv` | ragged：早退位置在未跑层 KV 缺席 |
| `projfill` | `--proj-fill`（需 `fill_kv=True`） | fill 基础上给已退出行 KV 源加**逐层恒等初始化 adapter** |

- **实测参数量与全部 path 键必须从落盘 `results.json` 逐字取**；不得据「13M」这一措辞反推结构。
- 每条 run 落独立子目录（`run1-<arm>-s1/`），含 `model.pt` / `train_log.jsonl` / `gate_log.jsonl` / `results.json`。

## 3 判据（L）

| 号 | 判据 | 通过条件 |
|---|---|---|
| L0 | **协议保真硬门**：seed 1 的 run 与既有 seed 0 的 run 除 `train.seed` 外的 config 逐字段相同（脚本对平，差异清单必须为空或逐条解释）；实测参数量 `fill`/`nofill` 相等、`projfill` 的增量 = 逐层 adapter 参数量（如实报） | 成立 |
| L1（主判 P1） | `depth-8 clean bpc`（默认深度）与 `depth-4` 在 seed {0,1} 上的点值；`Δ(fill − nofill)`、`Δ(projfill − fill)` 逐 seed 给，并给「seed 作为复本」的区间/极差 | 报告齐全 + 符号一致 ⇒ 可复现；否则判不可复现 |
| L2（主判 P2） | 同一 checkpoint 上 `generation_quality` 的 **soft / hard / latent** 提交的 `rescore_bpc`（深度 0 / L/2 / L 三点），逐 seed 报 `Δ(soft − hard)`、`Δ(latent − hard)`；**必须声明「同 checkpoint ⇒ 参数逐位相同，容量因素被剥离」** | 报告齐全 + 是否显著逐条给结论 |
| L3 | 活性列：`avg_train_gate_mean`（逐层门均值）、`avg_train_depth_ce`、填充位置占比 | 报告齐全 |
| L4 | 可读性：`tf_acceptance`（逐深度接受率）与 `depth_curve` 的 `acc` 齐备（「填入位置可读性该涨」的观测面） | 报告齐全 |
| L5 | **禁墙钟**：`results.json` / `analysis.md` 的判据区不得含墙钟字段（沿用 2660 的 `has_wall_clock` 自检）；墙钟只允许进 `run_log.txt` | 0 命中 |

- **用哪个带（C1 裁决，`harness/research-schema.md` §12.1）**：clean bpc 口径用 **0.014 bpc 带**（限配对 Δ、≥5 窗口）；
  **rescore 峰值 / 接受率属率类** ⇒ 走配对 bootstrap / Wilson 95% CI 为主判，**不跨量纲借用** 0.014。

## 4 否证条件

- **若** seed {0,1} 下 `Δ(fill − nofill)` 或 `Δ(projfill − fill)` 的**符号不一致** ⇒ 判
  「该填充语义的排序在本尺度不可复现」，6.4 的 `failure_analysis` 须给归因。
- **若** 同 checkpoint 下 `Δ(soft − hard)` 的 rescore 峰值差在 seed {0,1} 上 CI 均含 0 ⇒ 判
  「软提交的 OOD 污染在等参数下不成立（由容量/别的因素解释）」。
- **若** 只有一处 seed 显著 ⇒ 明确写「单点显著、不可复现」，不得升级为判据。

## 5 成本口径（等效理论量，禁墙钟）

- 填充语义**不改**每 token 的 layer-passes（= L）与每层 FLOPs 主项；`proj-fill` 的逐层 adapter 额外
  参数量按落盘实测报，每 token 额外 ≈ L·d FLOPs；`tokens-per-forward` 每步 1（填充不改该量）。
- 考虑 pipelining：填充位置占比决定被填位置的同批并行度；本实验不改变顺序前向次数的量级。
- matched-step 下各臂 **train_tokens 逐位相同**（须落盘核对）。

## 6 交付物

- `run1-<arm>-s1/`（3 个 run 目录：`model.pt` / `train_log.jsonl` / `gate_log.jsonl` / `results.json`）；
  **分两次派发时**后到者负责合并。
- `results.json`（聚合：逐 seed 点值、Δ、rescore 表、活性列、config 对平清单；无墙钟字段）。
- `analysis.md`（人话摘要在前、数字表在后；逐条给 L0–L5 与 §4 否证条件触发情况）。
- `MODEL.md`（checkpoint 清单 + sha256 + 复现命令）、`run_log.txt`（命令 / GPU 锁 / 墙钟）、
  `RESUME.md`（分次派发时：已完成哪几个 run、下一个确切命令）。
- `mutate.py` + `mutation_report.json`（≥3 条变异抓红 + 逐字节还原；报告不得含逐次运行耗时）——若本轮
  只做纯评测部分，可沿用引擎既有单测并如实声明未做本目录变异。

## 7 边界与纪律

- 只写 `axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m/**`；**禁改** `pathlm/**`、
  `train_*.py`、`tools/**`、`docs/**`、其它轴、`.scratch/23-depth-ar/evidence/**`（只读复用）、任何 `project.yaml`。
- GPU 按 `harness/SCHEMA.md` §20 取锁（本目录内 `gpu_lock.py` 可从 2660 复制），**先 code 锁再 GPU 锁**；不得杀他人进程。
- 证据不删；数字可回溯到落盘原始数据。
