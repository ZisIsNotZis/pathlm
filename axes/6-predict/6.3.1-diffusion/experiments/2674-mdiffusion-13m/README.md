---
name: 2674-mdiffusion-13m
ticket: 63-axis6-predict / Phase C · WP 6.3.1-WP1（掩码扩散的最小判决）
date: "2026-10-14"
grid: 6.3.1-diffusion
engine: "复用 6.3.2 的同 harness（`../6.3.2-pad-fill/experiments/2671-padfill-13m/padfill.py`，只读 import + 记录 sha256）；数据 = 仓库既有 data/enwik8_full.npz"
budget: "GPU ≤15 min（硬时间盒）：1 个训练臂 × seed{0,1}（按 S2 落盘的同一个 S）+ K∈{1,4,8,16} 采样评测（无额外训练）"
---

# 2674-mdiffusion-13m · 掩码扩散（K 步双向去噪）的最小判决

> **本文件是预登记契约，写在跑之前**：臂、采样器、判据、口径、否证条件在此写死，事后**不得**调参。
> 实测读数见 `results.json`（原始）与 `analysis.md`（当前结论）。
>
> **它回答的是什么**：6.3.1（`axes/6-predict/6.3.1-diffusion/project.yaml`）要判定「用 K 步（K ≪ T）双向
> 去噪并行刷新整段、换掉 T 步因果 AR，是否能在小尺度 / char 任务上把顺序前向次数与生成长度解耦」。
> 本仓零经验（MAPPING §2 无旧格），本实验给出**第一个可判结论**。

## 1 命题（可证伪）

**在同一 harness、同一 token 预算、同一位置集合下，「吸收态掩码扩散 + K 步置信度解掩码」的序列级
质量（bpc）相对同尺寸全因果 AR 的差是多少？该差随 K∈{1,4,8,16} 如何变化？顺序前向次数是否真的
从 T 降到 K（而每 token 等效 FLOPs 如实上升）？**

可证伪形式：*若掩码扩散在 K∈{1,4,8,16} 的**任一**配置下，质量相对 AR 的配对 Δ 的 block 聚类配对
bootstrap 95% CI 不含 0 且越 0.014 bpc 参考带恶化，且提高 K 不能把它拉回带内 ⇒ 判「扩散并行生成在
本尺度 / 本任务上不成立」。*

## 2 臂与采样器（写死，事后不得改）

| 臂 | 说明 |
|---|---|
| `ar`（参照） | **只读复用** `2671-padfill-13m` 的 `ar` 臂 checkpoint（同 config / 同步数 S / seed {0,1}）；记录其文件 sha256 与实测参数量。若该 checkpoint 不可得，则在本目录用**逐字段相同的 config** 重训一个 AR 臂并在 `analysis.md` 显式声明 |
| `mdiff`（本项变量） | `PadFillLM(attn_mask="full")`，训练目标 = **吸收态掩码扩散**：每 step 独立地对每个位置以概率 `r ~ U(0,1)` 替换为 `[mask]`（per-step 采样，固定调度种子），损失 = **只在被 mask 位置**上的 CE；`matched-step = S`（与 2671 的 S 逐字段相同，从 `2671-padfill-13m/results.json::config.steps` 读，不得自行改） |

- **K 步采样器（写死，K ∈ {1,4,8,16}）**：置信度解掩码（MaskGIT / MDLM 式）——
  第 0 步把目标位置集 `M` 全部置 `[mask]`；每一步对当前全部 mask 位置做**一次全段前向**，
  按「该位置预测分布的最大概率」排序，**定格置信度最高的 `ceil(|M|/K)` 个位置**（取 argmax，温度 0，
  确定性可复现），其余保持 `[mask]`；重复至 K 步（末步把剩余 mask 全部定格）。
  ⇒ `tokens-per-forward = |M|/K`、**每 block 顺序前向次数 = K**（AR 为 T−1）。
- **位置集 `M`**：与 2671 的评测口径**逐字一致**（同一批 64 个固定 offset × T=512 block 上的位置集，
  取 `r_eval` 与 2671 相同以便同框对照）——对平 2671 的 `results.json::config` / `eval_offsets`，差异清单须为空。
- **K=1 是本实验与 6.3.2 的连接点**：K=1 的采样器退化为「一次前向填出全部 mask 位置」，其读数应与
  2671 的 `fill_bidir_*` 臂在同位置集上可比（若不一致须解释，不允许含糊）。

## 3 判据（L）

| 号 | 判据 | 通过条件 |
|---|---|---|
| L0 | **实现保真**：`mdiff` 与 2671 的 `fill_bidir` 在 mask 调度与损失位置集上的实现一致性（同 r 下的 mask 分布、同损失口径）；`import` 的 `padfill.py` sha256 记录在 `MODEL.md` | 成立 |
| L1（主判） | 逐 K ∈ {1,4,8,16}：**NELBO 逐步均值**（标准掩码扩散目标：每步在仍被 mask 位置上 CE 的均值）与**末步 bpc**，以及 `Δ`（相对 `ar` 在**同一 `M`** 上的 teacher-forced bpc）的 block 聚类配对 bootstrap 95% CI；AR 的信息集不对称（AR 见左侧真值）**必须显式声明** | 报告齐全 + 是否越 0.014 带逐条给结论 |
| L2 | **K 的形状**：bpc（或 NELBO）随 K 的走向（是否「提高 K 拉回带内」） | 报告齐全 |
| L3 | **外部锚（副判）**：用**冻结的 AR 参考模型**对两臂在 `M` 上生成的序列做 rescore bpc，报配对差 + CI；**必须声明方向性偏差**（AR 当裁判偏向 AR） | 报告齐全 |
| L4 | 代价轴：`tokens-per-forward = |M|/K`、每 block 顺序前向次数（K vs T−1）、**每 token 等效 FLOPs = K·L**（等效理论量；**禁墙钟**） | 报告齐全 |
| L5 | **禁墙钟自检**（沿用 2671 的 `has_wall_clock`）+ **n≥2**（seed {0,1}）+ 确定性（同 seed 重跑逐值相同） | 0 命中 / 成立 |

- **用哪个带（C1 裁决，`docs/research-schema.md` §12.1）**：主判是 **bpc 口径** ⇒ 用 **0.014 bpc 带**
  （限配对 Δ、≥5 窗口）；率类指标（如需）走配对 bootstrap / Wilson 95% CI，**不跨量纲借用**。

## 4 否证条件

- **若** 任一 K 的 `Δ` 配对 95% CI 不含 0 且 |Δ| > 0.014，且 `Δ(K=16)` 不见小于 `Δ(K=1)` ⇒ 判
  「**掩码扩散并行生成在本尺度 / 本任务上不成立**」；6.3.1 的 `failure_analysis` 须区分
  设计问题（K 步采样器 / 噪声调度）与结构问题（双向 + 无左到右条件）。
- **若** 全部 K 的 CI 含 0 ⇒ 判「本尺度下不可分」，**不得**写成「与 AR 无损」。
- **若** `mdiff` 的 K=1 读数与 2671 `fill_bidir` 在统计上不可区分 ⇒ 记「6.3.1 与 6.3.2 在 K=1 的连接点
  成立」（这是本实验与 6.3.2 的跨项一致性证据，须另列一行）。
- **未做项**（须写进 6.3.1 的 `gaps`，不得含糊）：离散扩散的**均匀核**臂、噪声调度扫描、与 6.3.3 同框。

## 5 成本口径（等效理论量，禁墙钟）

- 一次去噪前向 = 全段一次双向前向（layer-passes = L × T），K 步 ⇒ **每 token 等效 FLOPs = K × L**
  （以 AR 每 token 的 L 为单位）；`tokens-per-forward = |M|/K`；每 token 顺序前向次数由 1 降到 K/T。
- 训练侧：每 step 一次全段前向（mask 目标只改损失位置集），matched-step 下与 2671 同刻度。
- 考虑 pipelining：段内位置全并行、去噪步间是 K 段顺序依赖。
- **墙钟**只允许进 `run_log.txt`。

## 6 交付物

- `mdiff.py`（库：训练臂 + K 步采样器 + NELBO/bpc）+ `run.py`（驱动）+ `test_mdiff.py`
  （采样器的**可失败**单测：K=1 退化等价、步数/位置数守恒、无未来泄漏、确定性）+ `mutate.py`
  （≥3 条变异抓红 + 逐字节还原）+ `mutation_report.json`（不得含逐次运行耗时）。
- `results.json`（逐臂逐 K 读数 / Δ + CI / 代价轴 / config；**无墙钟字段**）、`analysis.md`（人话摘要在前）、
  `MODEL.md`（含 `padfill.py` 与 AR checkpoint 的 sha256）、`run_log.txt`。
- 若时间盒不足以跑完 K∈{1,4,8,16}，**先跑 K=1 与 K=4** 并把余下 K 写进 `gaps`（+ 最小成本），
  不得用「部分 K」冒充完整 K 扫描。

## 7 边界与纪律

- 只写 `axes/6-predict/6.3.1-diffusion/experiments/2674-mdiffusion-13m/**`；只读 import / 复用
  `axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m/**`。
- **禁改**：`pathlm/**`、`train_*.py`、`tools/**`、`docs/**`、其它轴、其他实验目录、任何 `project.yaml`。
- GPU 按 `SCHEMA.md` §20 取锁（复制任一实验目录的 `gpu_lock.py`），**先 code 锁再 GPU 锁**；不得杀他人进程。
- 证据不删；数字可回溯到落盘原始数据。
