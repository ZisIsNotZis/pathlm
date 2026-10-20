---
name: 2675-second-tree-13m
ticket: 63-axis6-predict / Phase D · WP 6.2.5（完整二级树：N+3 以被接受的 N+2 候选为条件）
date: "2026-10-14"
grid: 6.2-speculative-decoding
engine: "E3 引擎能力（commit `fa6a1ad`：`PathConfig.chain_second` / `transforms_chain2` / `chain_cat2` / `conf_chain3` / `Decoder.step_tree_multi`）；主干 = 2631-spec-three-arms/train/13m_s0|13m_s1（冻结）"
budget: "GPU ≤60 min（硬时间盒）：只训头（主干冻结）+ 评测；**不训练主干**"
---

# 2675-second-tree-13m · 完整二级树的判决（Phase A 的「K 行平铺」之上一级）

> **本文件是预登记契约，写在跑之前**：臂、训练预算、判据、口径、否证条件在此写死，事后**不得**调参。
> 实测读数见 `results.json`（原始）与 `analysis.md`（当前结论）。
>
> **它回答的是什么**：`gaps` 第一条点名「二级条件树未做：位置 N+3 以被接受的 N+2 候选为条件需新条件头 + 重训」。
> E3 引擎已落地该能力（`chain_second=True` + `Decoder.step_tree_multi`），本 WP 负责**重训头**并给出判决：
> 加上这一级之后 τ 是否真的被抬高，代价轴与无损性如何。

## 1 命题（可证伪）

**在冻结 13M 主干上，把位置 N+3 的草稿从「无条件（只有一级链）」升级为「以被接受的 N+2 候选为条件」后，
相对同一模型的一级树，每轮接受的草稿数 τ 与 tokens-per-forward 是否显著抬高，而发出流是否仍与全深度
逐步贪心逐位一致？**

可证伪形式：*若完整二级树的 τ（或 α₁）相对**同一 checkpoint** 的一级树的配对 prompt 聚类 bootstrap
95% CI **含 0** ⇒ 判「二级条件头在本尺度无增益」；若发出流与贪心不再逐位一致 ⇒ 判无损性被破坏（须给失败归因）。*

## 2 臂定义（预登记，事后不得改）

| 臂 | 模型 | 说明 |
|---|---|---|
| `native` | 新模型 | 全深度逐 token 贪心（无损性参照与正确性基准） |
| `ar_1step`（诚实对手） | 新模型 | 1-step AR 两次前向（只替换紧邻位置；`harness/SCHEMA.md` §11 口径，与 2631 一致） |
| `direct1` | 新模型 | 单候选（node-2 头 argmax）+ 一次前向精确验证（= 2640 `direct1` 口径） |
| `lvl1_tree` | 新模型 | **一级树**：位置 N+2 的 K 个候选 + 一次前向批量精确验证（K ∈ {2,4}；行 = K 行平铺，与 2640 `tree_direct_k2` / `tree_topk4` 同机制） |
| `lvl2_tree` | 新模型 | **完整二级树**：位置 N+2 的 K 个候选，每个候选再以 `chain_second` 的第二级头给出 N+3 草稿，**一次前向**经 `Decoder.step_tree_multi` 树注意力批量验证（父指针 = 该候选） |
| `lvl1_old` / `lvl2_old`（对照，只读） | **既有 2631 checkpoint** | 用来把「重训头」与「树级别」两个因素分开：2631 无 `chain_second`（只有一级链），其 `lvl1` 读数已由 2640 落盘，本 WP **只读复用**、不重跑（若口径不完全可比须显式声明） |

- **单一变量声明**：主判（`lvl2_tree` vs `lvl1_tree`）在**同一 checkpoint** 上只改「N+3 是否以候选为条件」；
  副判（新模型 vs 2631）只改「是否重训带 `chain_second` 的头」。
- 头训练协议（写死）：`chain_second=True`、`n_mtp=3`、`chain_mode="token"`、`transport="none"`、`window=0`、
  冻结主干（`requires_grad=False`，冻结断言必须通过）；步数 / lr / batch / 数据序 / 种子写进 `results.json::config`，
  **seed ∈ {0,1}（n≥2）**；其余超参沿用 2660/2640 的既有头训练配方（逐字段记录）。

## 3 评测口径（写死）

- prompt 与 2640/2650/2670 一致（12 个固定 prompt、长度 64、enwik8 val 固定 offset），`n_new=200`，
  温度 0（贪心）。
- **主判**：`Δτ = τ(lvl2_tree) − τ(lvl1_tree)`（同一 checkpoint、同 K、同 prompt **逐 prompt 配对**，
  prompt 聚类配对 bootstrap 95% CI）；同时报 `Δα₁`（位置 N+2 的候选接受率，定义与 2640 一致）。
- **用哪个带（C1 裁决）**：τ / α₁ 等**率类**指标走配对 bootstrap / Wilson 95% CI 不含 0 为主判；
  **不使用** 0.014 bpc 带（不跨量纲借用）。若报序列级 bpc，则可并列 0.014 并显式标注为参考线。
- **无损性（必须不变）**：`lvl1_tree` / `lvl2_tree` / `direct1` 的发出流必须与 `native` **逐位相同**
  （13M 已知可作硬门禁；@100M 的 fp 平局翻转不适用于本规模，但仍须报是否出现）。
  另加两条**可失败**的机械检查：① 树父指针 == 因果前缀；② 树注意力不得看到兄弟行（改变兄弟行内容不得改变该行 logits）。
- **代价轴**（如实报、不作判据）：每轮实际送进主干的位置数（行数）与 `R = rows·(1+γ)/(1+τ)`；
  **必须注明**：本引擎 `n_mtp=3` 上限下**等 FLOPs 对照臂不可得**（63-F2），故 `R` 只作代价面报，不作判据。

## 4 判据（L）

| 号 | 判据 | 通过条件 |
|---|---|---|
| L0 | **保真**：`chain_second=False` 时的前向与 2631/2640 的既有实现**逐位相同**（E3 已给单测，须在本 revision 复跑并留证）；冻结断言（主干 `requires_grad` 全 False、backward 后 `.grad is None`）通过；`lvl1_tree` 走 `step_tree_multi` 与走 Phase A 的 `tree_forward` 在**同输入**下产出相同的接受行（若两条路径口径不同须显式说明差异） | 成立 |
| L1（主判） | `Δτ(lvl2 − lvl1)` 与 `Δα₁` 的配对 CI（逐 K、逐 seed、逐主干点） | 报告齐全；是否「CI 不含 0」逐条给结论 |
| L2 | **n≥2**：seed {0,1}（头训练）× 主干 {13m_s0, 13m_s1}，Δ 符号一致 | 报告齐全 |
| L3 | **无损性**：三臂发出流 `== native` 逐位；两条机械检查（父指针对齐、不看兄弟）**可失败**（gate probe 变红） | 成立 |
| L4 | 代价轴：行数 / `R` / tokens-per-forward（等效理论量；**禁墙钟**） | 报告齐全 |
| L5 | **禁墙钟自检**：`results.json` 不得含墙钟语义字段（沿用 2660 的 `has_wall_clock`）；确定性：同 seed 重跑逐值相同 | 0 命中 + 通过（或如实声明） |

## 5 否证条件

- **若** `Δτ(lvl2 − lvl1)` 的配对 95% CI **含 0**（K=4 与 K=2 都如此） ⇒ 判
  「**二级条件头（N+3 以被接受候选为条件）在本尺度无增益**」，并在 6.2 的 `failure_analysis` 里归因。
- **若** 提高 K 时 `lvl2` 的边际收益**低于** `lvl1` 的边际收益（同口径） ⇒ 判「二级树的代价不值得」。
- **若** 无损性被破坏 ⇒ 判该实现不成立（须给归因：引擎路径 / 我们的接线）。
- 反向（成立）时也必须写明：**本尺度 / 本任务 / 冻结主干**，不外推；且注明等 FLOPs 臂不可得。

## 6 成本口径（等效理论量，禁墙钟）

- 一级树：一次前向送 `K` 行（K 个 N+2 候选），产出 1+τ 个 token ⇒ `R = K·(1+γ)/(1+τ)`（γ = 1 时 = 2）。
- 完整二级树：一次前向送 `K + K`（每候选一行 N+2 + 一行 N+3）行 ⇒ `R = 2K·(1+γ)/(1+τ)`；**行数与接受收益必须同时报**。
- 头训练：只加在头顶部的 d×d 变换上（`transforms_chain2` / `chain_cat2` / `conf_chain3`），**不重跑主干**；
  每 token 主干 layer-passes 不变（评测阶段的主干前向 = 一轮一次）。
- **墙钟**只允许进 `run_log.txt`。

## 7 交付物

- `second_tree.py`（树构造 / 候选 / 验证接线）+ `run.py`（驱动，含 `--resume`）+ `test_second_tree.py`（**未产出**；L0 与 L3 的**可失败**单测 + gate probe）
  + `mutate.py`（≥3 条变异抓红 + 逐字节还原）+
  `mutation_report.json`（**不得含逐次运行耗时**；**必须接统一变异框架** `from tools.evalkit import mutation`，
  否则 `tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness` 会红）。
- `results.json`（臂级 τ / α₁ / CI / 代价轴 / config；**无墙钟字段**）、`analysis.md`（人话摘要在前）、
  `MODEL.md`（头权重哈希 / 复现命令）、`run_log.txt`、`train_log.jsonl`、`RESUME.md`。
- GPU 纪律：按 `harness/SCHEMA.md` §20 取锁、**逐 run 取放锁**；**长任务用 detached supervisor-retry**（后台 `while` 循环 +
  短轮询），**不阻塞等锁**；抢不到锁按 blocker 如实记，不动 CPU 全量、不杀他人进程。

## 8 边界

- 只写 `axes/6-predict/6.2-speculative-decoding/experiments/2675-second-tree-13m/**`；只读复用 2640/2650/2660/2670 的代码与 checkpoint。
- **禁改**：`pathlm/**`（E3 已由 engine-writer 落地，本 WP 只**用**能力）、`train_*.py`、`tools/**`、`docs/**`、
  其它轴、其它实验目录、任何 `project.yaml`。
- 证据不删；数字可回溯到落盘原始数据。
