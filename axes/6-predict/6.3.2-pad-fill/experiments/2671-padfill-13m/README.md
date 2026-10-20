---
name: 2671-padfill-13m
ticket: 63-axis6-predict / Phase C · WP 6.3.2-WP1（一次填空最小判决）
date: "2026-10-14"
grid: 6.3.2-pad-fill
engine: "实验目录内自带 harness（pre-LN char LM，唯一结构变量 = 注意力可见性）；数据 = 仓库既有 data/enwik8_full.npz"
budget: "S1（harness + 单测 + 步时冒烟）GPU ≤3 min；S2（5 臂 × 2 seed 全量扫描 + 评测）GPU ≤25 min（硬时间盒）"
---

# 2671-padfill-13m · 一次性批量 pad 填充（填空式）的最小判决

> **本文件是预登记契约，写在跑之前**：臂、训练预算、判据、口径、否证条件、成本口径在此写死，事后**不得**调参。
> 实测读数见 `results.json`（原始）与 `analysis.md`（当前结论）。执行分两阶段：**S1 = 契约实现**（harness + 单测 +
> 步时冒烟），**S2 = 全量扫描**（5 臂 × 2 seed 训练 + 评测）。两阶段共用本契约。
>
> **它回答的是什么**：6.3.2（`axes/6-predict/6.3.2-pad-fill/project.yaml`）要判定「训练成『一次前向把所有待填
> 位置填出来』的填空式模型，能否把顺序前向次数压到 1，以及它相对全因果 AR 的质量代价形状」。本仓零经验：
> 本实验给出**第一个可判结论**。

## 1 命题（可证伪）

**在同一 harness、同一 token 预算、同一结构（只改注意力可见性与损失位置集）下，「一次前向填出全部 PAD 位置」
的填空臂相对全因果 AR，在同一批待填位置上的 bpc 差是多少？双向可见是否优于因果可见？该差随 PAD 比例 r
是否单调恶化？**

可证伪形式：*若填空臂在 mask 位置上的 bpc 相对 AR 的配对 Δ 的 block 聚类配对 bootstrap 95% CI **不含 0**
且 |Δ| 越 0.014 bpc 参考带、并随 r 单调恶化、因果填空臂救不回 ⇒ 判「一次填充（K=1 并行填充）在本尺度 /
本任务上不成立」；若 CI 含 0 ⇒ 判「本尺度下不可分」。*

## 2 单因素声明与臂定义（预登记，事后不得改）

**唯一结构变量** = 注意力可见性（`causal` | `full`）+ 损失位置集；其余（宽度 / 层数 / 头数 / 数据 / 步数 /
优化器 / 种子 / 读出）逐字段相同。

| 臂 | 注意力 | 训练输入 | 训练损失位置 | 推理 |
|---|---|---|---|---|
| `ar`（参照） | causal | 真 token | 全部位置（预测 t_{i+1}） | 标准左到右贪心（逐 token） |
| `fill_bidir_r10` | **full** | 以概率 r=10% 把位置替换为 `[mask]` | **只在被替换位置**（预测该位置真 token） | 一次前向填出全部 PAD |
| `fill_bidir_r25` | full | r=25% | 只在被替换位置 | 同上 |
| `fill_bidir_r50` | full | r=50% | 只在被替换位置 | 同上 |
| `fill_causal_r25`（**可见性对照**） | causal | r=25% | 只在被替换位置 | 同上（只能看左侧） |

- **`[mask]` = vocab 的最后一槽**（`data/enwik8_full.npz` 的 `vocab_size` 已为 +1 预留，见 `pathlm/data.py`）。
- 每条样本每个 step **重新采样一次 mask 集合**（固定 mask 调度种子，写进 `results.json`）。
- 结构（写死）：pre-LN decoder block，`d_model=256 / n_layers=12 / n_heads=8 / mlp_mult=6 / seq_len=512`、
  共享 tied 读出（`logits = h @ E^T`）、无 dropout；**实测参数量必须如实报**（预期 12–14M）。
- **允许**在实验目录内自带实现（`padfill.py`）：共享引擎的注意力是**硬编码因果**的，本实验需要的 `full` 可见性
  不在引擎能力面内（不得改 `pathlm/**`；本实验的 AR 参照臂是**同 harness 的因果臂**，保证单因素可比）。

## 3 训练预算（写死，不得事后调）

- matched-step：`steps = S`，`batch_size = 32`，`T = 512`，Adam `lr = 1e-3`、无 weight decay，
  数据顺序 seed 固定；模型种子 ∈ {0, 1}（**n≥2**）。
- **S 的确定方式（允许的唯一调整）**：S1 先做步时冒烟（≥100 step，量一次实测步时），按「5 臂 × 2 seed 的
  训练总时 ≤ 25 min」反解 S（取整到 500 的倍数，最低 1500）。S 一旦写进 `results.json::config.steps`
  即对**全部 10 个 run 逐字段相同**，不得逐臂不同。
- 若 S < 6000，必须在 `analysis.md` 首段声明「**Nk 快照口径：与仓库既有 6000 步档不可比**，本实验只读
  **臂间增量与排序**」（`harness/SCHEMA.md` §20.3）；**禁止**通过砍臂 / 砍 seed / 改配方来迁就时间盒。

## 4 评测口径（写死）

- 评测集：enwik8 val 的 **64 个固定 offset × T=512** block（offset 逐条落进 `results.json`）。
- 位置集 `M_r`：对每个 block，`i ≥ 8` 且以概率 r 独立取「被填」（固定评测 mask 种子）⇒ `ar_at_M_{r}` 与
  `fill_*_r` 用**同一批 `M_r`**。
- **bpc**（每字符比特数，越低越好）= `(Σ −ln p(true token)) / (n · ln 2)`：
  - `ar_full`：AR 臂在**全部**位置 1..T−1 上的 teacher-forced bpc（标准参照）。
  - `ar_at_M_r`：AR 臂在 `M_r` 位置上的 teacher-forced bpc（**同位置对照**）。
  - `fill_*_r`：填空臂**一次前向**后在 `M_r` 位置上的 bpc。
- **主判差** = `bpc(fill_*_r) − bpc(ar_at_M_r)`（同一 `M_r` 逐 block 配对），
  **block 聚类配对 bootstrap 95% CI**（B=2000，seed 写进 `results.json`）。
- **信息集注释（必须同时报）**：`fill_bidir` 的信息集 = 全部**非 M** 真 token（含右侧）+ 自身位置的 mask 占位；
  `ar_at_M_r` 的信息集 = 全部**左侧**真 token（含 M 左侧的真值）⇒ 两者**互不包含**，该不对称必须写在
  `analysis.md`，不得把主判差读成「纯机制优劣」。
- **用哪个带（C1 裁决，`harness/research-schema.md` §12.1）**：本实验主判是 **bpc 口径** ⇒ 参考门槛用
  **0.014 bpc 带**（限配对 Δ、≥5 窗口；本实验 64 个 block 聚类窗口满足）；率类指标（如需）走
  配对 bootstrap / Wilson 95% CI、**不跨量纲借用** 0.014。
- 必须同时报：tokens-per-forward（`ar`=1；填空=|M|≈r·T）、每 block 顺序前向次数（`ar`=T−1；填空=1）、
  每 token 等效 FLOPs（`ar`=L；填空=r 时 ≈ L/r —— 见 §6）。

## 5 判据（L）

| 号 | 判据 | 通过条件 |
|---|---|---|
| L0 | **不受未来泄漏硬门（必须不变）**：对 `fill_bidir_*`，把 `M` 位置上的**真 token** 扰动为任意值后重前向，`M` 位置上所有位置的 logits **逐位不变**（`max|Δ| == 0`）；对 `fill_causal_r25` 额外要求：把**右侧非 M** 真 token 扰动后，左侧 `M` 位置 logits 逐位不变 | 两条都成立 |
| L0b | **可失败性自检（gate probe）**：把 mask 构造刻意改成「把真 token 喂进 `M` 位置」（泄漏）后，L0 的检查**必须变红**；把 causal 臂改成 full 后 L0b 的第二条**必须变红** | 两处都变红 |
| L1 | **可读性/质量主判**：5 臂 × 2 seed 的 `bpc`（`ar_full` / `ar_at_M_r` / `fill_*_r`）与主判差 + CI 齐备 | 报告齐全 |
| L2 | **r 形状**：主判差随 r ∈ {10%, 25%, 50%} 的走向（是否单调恶化）逐点给 | 报告齐全 |
| L3 | **可见性对照**：`fill_causal_r25` vs `fill_bidir_r25` 的配对 Δ + CI | 报告齐全 |
| L4 | 代价轴：tokens-per-forward、每 block 顺序前向次数、每 token 等效 FLOPs（等效理论量；**禁墙钟**） | 报告齐全 |
| L5 | **禁墙钟自检**：`results.json` 不得含墙钟语义字段（沿用 2660 的 `has_wall_clock`）；确定性：同 seed 重跑逐值相同 | 0 命中 + 通过（或如实声明） |

## 6 否证条件

- **若** `Δ_bidir = bpc(fill_bidir_r) − bpc(ar_at_M_r)` 的配对 95% CI 不含 0、`|Δ| > 0.014`，
  **且**随 r 单调恶化，**且** `Δ_causal` 更差（救不回） ⇒ 判「**一次填充（K=1 并行填充）在本尺度 /
  本任务上不成立**」；6.3.2 的 `failure_analysis` 须区分设计问题 / 结构问题。
- **若** `|Δ_bidir| ≤ 0.014` 或 CI 含 0 ⇒ 判「本尺度下该 r 的质量代价不可分」，**不得**写成「无损」。
- **若** `fill_bidir_r25 − fill_causal_r25` 的配对 CI 含 0 ⇒ 判「填空视图的**可见性**在本尺度无独立增益」
  （与主判并列的另一条结论）。

## 7 成本口径（等效理论量，禁墙钟）

- 一次前向处理全段 T 个位置（layer-passes = L × T）；一次填出 `N = r·T` 个 PAD ⇒
  **tokens-per-forward = N**、**每生成 token 等效 FLOPs = L / r**（以 AR 每 token 的 L 为单位）、
  **每 token 顺序前向次数 = 1/N**（AR = 1）。
- 训练侧代价如实报（每 step 前向 1 次全段；matched-step 下各臂相同）。
- **墙钟**只允许出现在 `run_log.txt`，不得进 `results.json` / `analysis.md` 的判据。

## 8 交付物

- S1：`padfill.py`（模型 + mask + 训练/评测库）、`test_padfill.py`（L0/L0b + bpc/CI 单测 + 因果臂前向正确性）、
  `smoke_steptime.py`（≥100 step 冒烟，输出实测步时与选定 S）、`MODEL.md` 起头。
- S2：`run.py`（`--resume` 可续跑；逐 run 落盘）、`results.json`、`analysis.md`、`mutate.py`（≥3 条变异抓红 +
  逐字节还原）+ `mutation_report.json`（**不得含逐次运行耗时**）、`run_log.txt`、`train_log.jsonl`。
- **两阶段都必须**把 `results.json` 的 `config.steps` 与 `README` 的预算一致；不得事后调参。

## 9 边界与纪律

- 只写本目录；禁改 `pathlm/**`、`train_*.py`、`tools/**`、`docs/**`、其它轴、其它研究项目录。
- GPU 按 `harness/SCHEMA.md` §20 取锁（本目录内 `gpu_lock.py` 可从 2660 复制），**先 code 锁再 GPU 锁**；不得杀他人进程。
- 证据不删；数字可回溯到落盘原始数据。
