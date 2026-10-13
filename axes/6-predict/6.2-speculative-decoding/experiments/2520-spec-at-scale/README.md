---
name: 2520-spec-at-scale
ticket: 20-spec-scale
date: "2026-09-18"
ledger_prereg: "2026-10-09"
grid: H1-outputs-per-pass
also: [H5-verification]
engine: "100M（INT3_100M_s0：d=720×12，n_mtp=3，32000×6 tokens）"
budget: "纯推理；重跑 ≤12 min（含冒烟）"
---

# 2520-spec-at-scale · spec 宽度缩放的算力账

> 主格：`H1-outputs-per-pass`｜关联格：`H5-verification`｜引擎：100M（INT3_100M_s0）｜预算：纯推理

## 1. 命题

一次前向喂 γ 位草稿、期望接受 τ 位，则 **每 token 的算力 = `(1+γ)/(1+τ)`**
（以 native 逐 token 自回归的 1 次 width-1 前向为单位）。这条比值是 `H1` 的
**算力口径**；它是否可以 < 1（即 spec 在算力上净省）决定了「大模型上只剩 FLOPs
口径」这句判读成立与否。**它必须在两个规模上独立算出、并与墙钟比值分开报。**

## 2. 单因素声明

本实验是**纯推理测量**，不训练、不改结构：唯一变量是**解码路径**
（native 逐 token 自回归 vs Medusa 式批量验证），基底是同一个 checkpoint。
两臂共用同一 `Decoder`/`step` 语义（见 §5 判据 3），因此不存在第二个机制。

## 3. 既有做法（prior art）

> 来源标 `记忆·未核`：本环境联网不可用，**不得当判据**，只作设计方向。

| 谁做过 | 做了什么 | 声称解决 | 我们的增量 |
|---|---|---|---|
| Leviathan 2023 / Chen 2023（投机解码） | 便宜草稿 + 一次批量验证；**无损性定理** | 自回归解码访存/启动受限 | 本地实测的**算力账本身**（业界报墙钟加速比，少报每 token FLOPs 比） |
| Cai 2024（Medusa） | 冻结主干挂多头 + 树注意力批量验证 | 免另训草稿模型 | 同引擎、同 checkpoint 的 width 旋钮 |
| Gloeckle 2024 / DeepSeek-V3 2024（MTP） | 训练时预测多个未来位置，自带草稿 | 质量 + 自投机 | 13M/100M 的税-接受率-双口径表 |

**业界标准叙事**：投机解码/MTP 的卖点是「**匹配质量下省算力（或省延迟）**」。
本实验检验的正是这句里的「省算力」在**本项目同引擎口径**下是否成立。

**判定（SCHEMA §21）**：机制本身（launch-bound ↔ compute-bound 的 regime 随规模翻转）
**业界已做透 ⇒ (a)**；但「每 token 解析算力比 `(1+γ)/(1+τ)` vs 实测墙钟比
`c_wall` 的**背离幅度**」是**本地规模特有**的量化 ⇒ **(b) 做实验**。

## 4. 指标表

| 维度 | 量 | 口径 |
|---|---|---|
| 解决什么 | 草稿从哪来、验证多贵 | 视界轴成本 |
| 代价（税） | n_mtp 1→2 的 clean bpc 增量 | 见 `2510-gap-sweep-b2-c1m`（不在本实验测） |
| **算力（主）** | 每 token `(1+γ)/(1+τ)` | 等效 width-1 前向单位，native = 1.0 |
| 墙钟（并列） | spec/native 端到端 tok/s 比 | 同进程逐次交替，报重复极差 |
| §20 记账 | `nvidia_smi_memory_free_mb`（前/后）、`peak_gpu_mem_mb` | `run.py::gpu_accounting` |
| 接受级联 | `a₂`、`a₃\|a₂`、tok/forward | 计数 / 二项 95% CI（Wilson） |
| 正确性（门禁） | spec 发出流 == native 贪心 | 逐位；fp 平局翻转单列 |
| 可控性 | 前向成本比 `c_wall`（实测宽度 `1+γ` 前向 / width-1） | 墙钟 min，仅作墙钟描述量（与宽度配对） |

## 5. 判据线（预登记，先写后跑，不得事后调整）

记 `γ` = 每次前向投喂的草稿位数（= `max_drafts`），`τ` = 每次前向平均接受位数
（= tokens/forward − 1）。

1. **算力口径（主判据）**：每 token 算力比 `R = (1+γ)/(1+τ)`，native = 1.0。
   - **该涨**：无（spec 不改善 clean 质量）。
   - **该跌 / 应被推翻的结论**：`R > 1 + 0.05` ⇒ 算力口径净亏损
     ⇒ 「@100M 只剩 FLOPs 口径」**不成立**；`R < 1 − 0.05` ⇒ 算力口径净省
     ⇒ 旧判读成立。
   - **必须不变（解析上界）**：`τ < γ` ⇒ `R > 1`；`τ = 0` ⇒ `R = 1+γ`。
     `R` 是**解析量**，不依赖 GPU 负载，因此主判据**不取锁**即可判定。
2. **墙钟口径（并列）**：spec/native tok/s 比。**与 `R` 分开报，禁止互换。**
   `c_wall`（宽度 `1+γ` / width-1 的墙钟比，与宽度配对）**只是墙钟描述量，不得代进 `R`**。
3. **正确性门禁**：spec 发出流与 native 贪心**逐位相同**，除非 top-2 gap < 1e-3
   的 fp 平局翻转（记录首个翻转点、gap、翻转 token 数）。
   native 必须走同一条解码路径 `Decoder.step()`，**不得**用批量 teacher-forcing 冒充。
   - **执行结果（预登记线不变）**：4 臂中 3 臂逐位相同；enwik8_k2 首错位 169、级联 68 token、
     翻转处 top-2 gap **0.007912 > 1e-3** ⇒ **该臂门禁不通过** ⇒ **逐位相等不是跨规模判据**
     （batched SDPA 与逐位置 SDPA 的核差异在近平局处翻转 argmax）。
4. **接受级联（描述量）**：`a₂`、`a₃|a₂` 给计数与 Wilson 95% CI；tok/forward 由
   计数直接算，不取多 rep 的均值。
5. **噪声门限（自测，不套用全局 0.014）**：`R`、`τ`、`a₂` 在贪心解码下是**确定量**，
   重复跑必须逐位/逐位重复（0 方差）；墙钟报**同进程重复的极差**（min–max），
   不设跨进程阈值。

## 6. 对照与方差

- **对照臂 = native 逐 token 自回归**，同 checkpoint、同 prompt、同 `Decoder`、
  同 KV 语义；对照与主臂在**同一循环内逐次交替**（同热/同占用状态）。
- 复用历史 run（`.scratch/13-four-experiments/evidence/mtp3`、13M B3）作 13M 参照点，
  口径标注见 `analysis.md`；13M checkpoint 已不可得，历史数字只作参照。
- 接受率随 prompt 分布变（random vs enwik8），两档都报。

## 7. 成本

- `gpu_wall_budget: 12 min`（含冒烟 + 重跑；纯推理）。本目录已交付的 run **未做机器记账**：
  `results.json` 无 `peak_gpu_mem_mb` / 前后 `nvidia-smi free`，锁由外层 shell 取得 ⇒ §20.2 三件套
  **未落盘**（纯推理、单程 §20 例外；未提交的烟测观测峰值约 460 MB，**不可追溯**）。
  `run.py` 现已补 `gpu_accounting` 字段，重跑会自动落盘。
- 预算不够时**砍重复次数**（REPS 5→2），**不砍判据线**；主判据（§5.1）无 GPU 也能算。

## 8. 数据与脚本清单

- 数据：`data/enwik8_full.npz`（val 前 32 token 当真实 prompt）+ random prompt（seed 5/7/9）。
- 入口：`run.py`（复用 `probe_spec.py` 的 `micro_forward_cost` / `steady_state` 协议，
  追加算力账 `compute_ledger`）。
- resolved config：checkpoint 自带 `results.json` 的 `config`/`resolved`（INT3_100M_s0）。
- 落盘：`results.json`（原始 + 算力账）、`MODEL.md`（权重哈希）。

## 9. 失败决策树

| 症状 | 诊断 | 处置 |
|---|---|---|
| `c_wall` 与前向反解冲突 | 微基准被核回退/启动开销污染 | 以每秒 forward 反解为准，两者并报 |
| 墙钟波动大 | 外部作业占用 | 标「绝对墙钟不可用」，只报相对排序 + 重复极差 |
| 发出流不等 | 实现 bug 或 fp 平局 | 先查 gap、再查路径；gap<1e-3 记为平局 |
| `R ≤ 1` 而 tok/forward < γ+1 | 计数或 γ 记错 | 先对账 tok/forward = 1 + a₂ + a₂·a₃\|a₂ |

## 10. 产出

`results.json` / `analysis.md` / `MODEL.md` / 本契约；13M 参照来自
`.scratch/13-four-experiments/evidence/mtp3`（checkpoint 不可得）。

## 11. scope

单轴单格、纯推理、基底 = 同一 checkpoint 的两条解码路径；**组合默认留白**
（与 retry/exit 的组合不在本实验）。未测项见 `analysis.md` §未测项。
