---
name: 2670-freerun-bpc-13m
ticket: 63-axis6-predict / Phase C · WP 6.3.3-WP2（free-run 序列级 bpc）
date: "2026-10-14"
grid: 6.3.3-latent-regression
engine: "width-MTP 冻结主干（../6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0、13m_s1），头复用 2660-latentreg-vs-disc-13m/heads/*.pt（不再训练）"
budget: "GPU ≤10 min（硬时间盒）；一次性生成评测，无训练"
---

# 2670-freerun-bpc-13m · 路径 (i)「一次前向直接发 k 个位置」的 free-run 序列级 bpc

> **本文件是预登记契约，写在跑之前**：臂、口径、判据、否证条件、成本口径在此写死，事后**不得**调参。
> 实测读数见 `results.json`（原始）与 `analysis.md`（当前结论）。
>
> **它回答的是什么**：6.3.3（`axes/6-predict/6.3.3-latent-regression/project.yaml`）的 `milestone.output`
> 要求「…的 tokens-per-forward、τ、草稿接受率与**序列级 bpc**」，而 WP1（2660）的 `gaps` 第一条
> 自述「free-run 序列级 bpc 未测：只有 teacher-forced 命中率 / CE 与草稿接受率」。本实验补上这一条：
> 在**冻结主干**上，把路径 (i)（估计得贪心：一次前向直接发 k 个 token、不经验证）的**自由生成**
> 序列级 bpc 与同一主干自身的 AR 逐 token 贪心放在同一口径下对照。

## 1 命题（可证伪）

**在冻结的 13M 主干上，「一次前向直接发 k 个位置」（路径 i）的自由生成序列级 bpc 相对同一主干
逐 token 贪心 AR 的差是多少；打开 latent 匹配监督（λ=1.0）能把该差缩小多少？**

可证伪形式：*若「直接发」臂相对 AR 的配对 Δbpc 的 prompt 聚类配对 bootstrap 95% CI **不含 0**
且 |Δbpc| 越 0.014 bpc 参考带 ⇒ 「一次前向直接发 k 个位置」在本尺度**质量面不成立**；若 CI 含 0
⇒ 判「本尺度下该路线质量代价不可分」。*（本判据只针对**路径 (i)**；路径 (ii)「当草稿再验证」的
判据已由 2660 阶段 2 给出，不在本实验重复。）

## 2 臂定义（预登记，事后不得改）

唯一变量 = 「未来位置的产出方式」（**不训练**，头权重与主干权重全部冻结、从 2660 直接复用）。

| 臂 | 主干 | 头文件 | 一次前向发出 token 数 | 说明 |
|---|---|---|---|---|
| `ar`（参照） | 13m_s0 / 13m_s1 | 无（主干 node-1，t+1） | 1 | 同一主干自身的 tied 读出逐 token 贪心 |
| `disc_k2` | 13m_s0 / 13m_s1 | `heads/disc_k2_s{0,1}.pt` | 2 | 同参数量离散头（λ=0），预测 t+2 / t+3 |
| `lat_k2_lam1` | 13m_s0 / 13m_s1 | `heads/lat_k2_lam1_s{0,1}.pt` | 2 | 回归头（λ=1.0），同一读出 |

⇒ 臂 × 主干 × 头种子 = `ar`(2 点) + `disc_k2`(2×2) + `lat_k2_lam1`(2×2) = 10 个生成配置，**n≥2**（两个主干点、两头种子）。

### 2.1 生成过程（写死，事后不得改）

- **`ar`**：标准左到右贪心，每步 1 次主干前向产出 1 个 token，共 200 步。
- **`disc_k2` / `lat_k2_lam1`**（路径 (i)，估计得贪心、**不验证**）：第 r 轮对当前序列做 **1 次冻结主干前向**，
  取**最后一个位置** i 的最终层隐状态 `h_i`；对 j=1..k 用第 j 个头 `z_j = head_j(h_i)`、
  `logits_j = z_j @ E^T`（E = 主干 embed 权重，共享 tied 读出），**各自取 argmax** 并把 k 个 token
  **全部**追加到序列（k=2 ⇒ 每轮发 2 个 token，共 100 轮 = 200 token）。**不做**任何验证 / 拒绝 / 回退。
- 评测位置口径：`ar` 的分布 = 主干 node-1 在该位置的分布；`disc/lat` 的分布 = 该头从 `h_i` 给出的分布。
  两个臂都按「**自己发出的 token 在自家预测分布下的 NLL**」计 bpc。

## 3 评测集与口径（写死）

- 数据：`data/enwik8_full.npz`（仓库既有），**同 2640/2660 的 val 口径**（enwik8 val，固定 offset）；
  prompt = **12 个固定 prompt × 长度 64**（offset 与 2640/2650 的 prompt 取法一致，或在本目录 `README` 内
  写明所用 offset 并逐条列出），每个 prompt 生成 **200** 个新 token。
- **bpc**（每字符比特数，越低越好）= `(Σ −ln p(emitted_token)) / (n_tokens · ln 2)`；prompt 不计分。
- 主判差：`Δbpc = bpc(候选臂) − bpc(ar 同主干)`，**同一批 prompt 逐 prompt 配对**，
  **prompt 聚类配对 bootstrap 95% CI**（B=2000，seed 写进 `results.json`；clustered by prompt）。
- **用哪个带（C1 裁决，`docs/research-schema.md` §12.1）**：本实验的指标是 **bpc 口径** ⇒
  参考门槛用 **0.014 bpc 带**（限配对 Δ、≥5 窗口；本实验 12 个 prompt 聚类窗口满足）；
  率类指标（如需要）走配对 bootstrap / Wilson 95% CI、**不跨量纲借用** 0.014。
- 必须同时报：每 token 主干等效 FLOPs（`ar` = L；路径 (i) = ≈ L/k）、tokens-per-forward（1 vs k）、
  每 prompt 的 200 token 生成流 sha256（可复现凭证）。

## 4 判据（L）

| 号 | 判据 | 通过条件 |
|---|---|---|
| L0 | **保真硬门**：头权重逐张量等于 2660 落盘文件（`torch.equal` 全真）；主干 `requires_grad=False` 且不更新；`ar` 臂的贪心流与 2640/2650 的 `native` 口径一致（同 prompt 同长度逐位相同或给出差异解释） | 全真 |
| L1（主判） | `Δbpc`（候选臂 − `ar` 同主干）逐主干 × 头种子给出**点值 + prompt 聚类配对 bootstrap 95% CI**；`lat_k2_lam1` 的 Δ 与 `disc_k2` 的 Δ 之差（配对）同时给 CI | 报告齐全；是否「越 0.014 带」逐条给结论 |
| L2 | **n≥2 一致**：两个主干点、两个头种子的 Δ 符号一致 | 报告齐全 |
| L3 | 代价轴：每 token 主干等效 FLOPs 与 tokens-per-forward（等效理论量；**禁墙钟**） | 报告齐全 |
| L4 | **禁墙钟自检**：`results.json` 内不得出现墙钟语义字段（沿用 2660 的 `has_wall_clock` 自检） | 0 命中 |
| L5 | 确定性：同一配置重跑一次，bpc 逐值相同（浮点确定性；如不成立须如实报并给差异量级） | 通过或如实声明 |

## 5 否证条件

- **若** `Δbpc(disc_k2 − ar)` 的配对 95% CI 不含 0 且 `|Δbpc| > 0.014`，且 `lat_k2_lam1` 只把 |Δ| 缩小
  （同向、仍越带） ⇒ 判「路径 (i)『一次前向直接发 k 个位置』在本尺度（13M 冻结主干、char enwik8、
  温度 0、贪心）**质量面不成立**」，并在 6.3.3 的 `failure_analysis` 里归因（设计问题 vs 结构问题）。
- **若** `Δbpc(lat_k2_lam1 − disc_k2)` 的配对 95% CI **含 0** ⇒ 判「在本口径下 λ=1.0 的 latent 匹配监督
  对 free-run 序列级质量无可测增益」（与 2660 的 teacher-forced 命中率结论并列、不冲突则须说明差异）。
- **若** CI 全部含 0 ⇒ 判「本尺度下该路线的序列级质量代价不可分」，不得写成「无损」。

## 6 成本口径（等效理论量，禁墙钟）

- 路径 (i)：每轮 1 次全深度前向（layer-passes = L）产出 k 个 token ⇒ **每 token 主干等效 FLOPs ≈ L/k**、
  **tokens-per-forward = k**、每 token 顺序前向次数 = 1/k。
- `ar`：每 token 1 次全深度前向 ⇒ 每 token 主干等效 FLOPs = L、tokens-per-forward = 1、顺序前向次数 = 1。
- 考虑 pipelining：路径 (i) 的 k 个 token 在同一轮内并行（无顺序依赖），轮与轮之间是顺序依赖。
- **墙钟**只允许出现在 `run_log.txt`，不得进 `results.json` / `analysis.md` 的判据。

## 7 交付物

- `results.json`：臂级 bpc、Δ + CI、生成流 sha256、代价轴、参数与配置（**无墙钟字段**）。
- `analysis.md`：人话摘要在前，数字表在后；逐条给 L0–L5 与 §5 否证条件触发情况。
- `freerun.py`（库）+ `run.py`（驱动）+ `test_freerun.py`（单测：头权重一致性、bpc 计算、CI 计算、
  禁墙钟自检、聚合方向）+ `mutate.py`（≥3 条变异须抓红并逐字节还原）+ `mutation_report.json`（**不得含逐次运行耗时**）。
- `MODEL.md`：权重复用清单与 sha256、复现命令。
- `run_log.txt`：命令与 GPU 锁记录（可含墙钟）。

## 8 边界与纪律

- 只写本目录；禁改 `pathlm/**`、`train_*.py`、`tools/**`、`docs/**`、其它轴、其它研究项目录。
- GPU 按 `SCHEMA.md` §20 取锁（本目录内 `gpu_lock.py` 可从 2660 复制），**先 code 锁再 GPU 锁**；不得杀他人进程。
- 证据不删；数字可回溯到落盘原始数据。
