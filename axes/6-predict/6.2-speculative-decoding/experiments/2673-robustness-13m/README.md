---
name: 2673-robustness-13m
ticket: 63-axis6-predict / Phase C · WP 6.2.4（温度>0 的无损投机采样 + 53.9M 对抗臂）
date: "2026-10-14"
grid: 6.2-speculative-decoding
engine: "width-MTP 冻结主干（2631-spec-three-arms/train/13m_s0、13m_s1、53m_s0）+ 既有头；验证路径复用 2640 的 `spec_tree.tree_forward`（K 行平铺一次批量前向）"
budget: "GPU ≤15 min（硬时间盒）；无训练"
---

# 2673-robustness-13m · 温度>0 的无损投机采样 + 53.9M 对抗臂

> **本文件是预登记契约，写在跑之前**：臂、口径、判据、否证条件在此写死，事后**不得**调参。
> **它回答的是什么**：6.2 的 `milestone.output` 点名「含**温度>0** 与 batch>1 的稳健性点」，
> 而现状全部读数都在温度 0、B=1；`gaps` 第一条即「温度>0（拒绝采样）与 batch>1 的稳健性点未跑」。
> 本 WP 做其中**可判的最小一块**：温度>0 下**无损投机采样**的正确性与 τ；
> **batch>1 不在本 WP 内**（本引擎 `decode_spec` / 2640 的树验证路径都是 B=1，批量验证需改共享引擎，
> 属新需求、须走 `.scratch/67-engine-changes/` 通道）——如实登记为 gap，**不得**在这里悄悄改引擎。

## 1 命题（可证伪）

**(P1)** 在冻结主干上，把「贪心验证」（`decode_spec`：候选必须等于目标模型 argmax）换成
**温度 T 下的无损投机采样**（Leviathan/Chen 式：以 `min(1, p_target(x)/q_draft(x))` 接受、
拒绝时从残差分布 `norm(max(0, p_target − q_draft))` 重采）后，发出序列的**边缘分布**是否与
「不做投机、直接按目标模型采样」一致（无损），同时 τ 与 tokens-per-forward 是否仍 > 1？
**(P2)** 对抗候选（随机 / 概率最低）在 53.9M 上是否同样不产生假接受（13M 已测，53.9M 未测）？

可证伪形式：*若发出序列在 T=1 下的经验分布与目标模型采样在**预登记统计检验**下不一致（且该检验
在校验用的「故意错误接受规则」上确实会变红）⇒ 判本实现**不是无损投机采样**；若 τ ≤ 1 或
tokens-per-forward ≤ 1 ⇒ 判温度>0 下投机在本尺度不成立。*

## 2 臂与口径（写死，事后不得改）

| 臂 | 说明 |
|---|---|
| `target_sample`（参照） | 冻结主干 node-1 直接在温度 T 采样（无草稿、无验证），每 token 1 次前向 |
| `spec_T0`（参照，已有） | `decode_spec` 贪心（T=0）；τ / 流与 2640/2650 的口径一致 |
| `spec_T1`（本 WP 主臂） | 目标模型 = node-1 在温度 **T=1** 下的分布；草稿 = 既有草稿头在该位置的分布 q；**无损投机采样**（接受 + 残差重采） |
| `adv_random_k4_53m` / `adv_bottomk4_53m`（P2） | 53.9M 冻结主干上，把位置 N+2 的候选集换成 {均匀随机 4 个, 概率最低 4 个}，走 2640 的 K 行批量精确验证；报接受率（预期 ≈ 均匀随机假接受率）与发出流是否逐位 == native 贪心 |

- **T 固定为 1.0**（不为找好看的数字扫 T）；prompt 取法与 2640/2650 一致（同一批固定 offset）。
- **无损性判据（P1 的核心）**：对同一批 prompt，比较 `spec_T1` 与 `target_sample` 的**发出 token 经验分布**
  与**序列级 NLL（在自己目标分布下的负对数似然）**；用**固定的统计检验**（写死：逐 prompt 的两个分布的
  TV 距离 / 或长度-k 前缀频率的 χ² 检验；k 与显著性水平写进 `results.json`）+ prompt 聚类 bootstrap 95% CI。
- **gate probe（必须做）**：把接受规则**故意改错**（例如接受概率恒为 1，或残差重采写成直接用草稿分布），
  上述检验**必须变红**；再改回正确规则必须变绿。**这一步是 P1 判据的可证伪性凭证，缺它即判不通过。**
- **P2 口径**：与 2650 的对抗臂逐字一致（同候选构造、同 K=4、同验证路径），只换主干到 53.9M。

## 3 判据（L）

| 号 | 判据 | 通过条件 |
|---|---|---|
| L0 | **gate probe**：错误接受规则 ⇒ 无损性检验变红；正确规则 ⇒ 变绿（两处都留证） | 成立 |
| L1（主判 P1） | 无损性检验结果（TV / χ² + prompt 聚类 CI）+ `spec_T1` 的 τ、tokens-per-forward、每 token 等效 FLOPs；与 `target_sample` 的序列级 NLL 差 | 报告齐全；是否「无损」逐条给结论 |
| L2 | **n≥2**：13m_s0 / 13m_s1 两个主干点（头种子沿用 2640/2660 既有读数的那一套） | 报告齐全 |
| L3（主判 P2） | 53.9M 的 `adv_random_k4` / `adv_bottomk4` 的接受率（α₁，Wilson 95% CI）+ 发出流与 native 的逐位一致性 | 报告齐全；是否出现假接受给结论 |
| L4 | 代价轴：τ / tokens-per-forward / 每 token 主干等效 FLOPs（等效理论量；**禁墙钟**） | 报告齐全 |
| L5 | **禁墙钟自检**：`results.json` 不得含墙钟语义字段（沿用 2660 的 `has_wall_clock`） | 0 命中 |

- **用哪个带（C1 裁决）**：**本 WP 的主判全部是率类 / 分布类** ⇒ 走 prompt 聚类配对 bootstrap / Wilson 95% CI；
  **不使用** 0.014 bpc 带（不跨量纲借用）；序列级 NLL 若以 bpc 口径报，可并列 0.014 作参考线并显式标注。

## 4 否证条件

- **若** gate probe 无法使检验变红（检验无鉴别力）⇒ 本 WP 的 P1 **判据不成立**，如实登记，
  **不得**把「检验绿」当无损证据。
- **若** `spec_T1` 与 `target_sample` 的分布检验在 CI 下不一致 ⇒ 判「本实现的温度>0 投机采样**不是无损**」。
- **若** `spec_T1` 的 τ ≤ 1 ⇒ 判「温度>0（T=1）下投机在本尺度不成立」。
- **若** 53.9M 对抗臂出现假接受（接受率显著高于均匀随机假接受率）⇒ 判「K 行验证在 53.9M 上的
  无损性失效」，须给失败归因。

## 5 成本口径（等效理论量，禁墙钟）

- `spec_T1`：一次前向送 1+γ 个位置、产出 1+τ 个 token ⇒ 每 token 等效 FLOPs = (1+γ)/(1+τ)、
  tokens-per-forward = 1+τ、每 token 顺序前向次数 = 1/(1+τ)（以 native 逐 token 的 1 次 width-1 前向为单位）。
- `target_sample`：每 token 1 次 width-1 前向 ⇒ 每 token 等效 FLOPs = 1、tokens-per-forward = 1。
- 残差重采**不增加主干前向**（复用同一批验证行）；须如实报实现是否真的如此。
- **墙钟**只允许进 `run_log.txt`。

## 6 交付物

- `spec_lossless.py`（库）+ `run.py`（驱动）+ `test_spec_lossless.py`（L0 的 gate probe + 接受 / 残差重采的单元测试，
  必须可失败）+ `mutate.py`（≥3 条变异抓红 + 逐字节还原）+ `mutation_report.json`（不得含逐次运行耗时）。
- `results.json`（臂级 τ / α₁ / CI / 分布检验统计量 / 对抗臂读数 / 代价轴；**无墙钟字段**）、
  `analysis.md`（人话摘要在前）、`MODEL.md`、`run_log.txt`。
- 若 `spec_T1` 与树（K>1）在 T=1 下的组合**未做**，在 `gaps` 里写「下一步最小实验 + 成本」，不得含糊。

## 7 边界与纪律

- 只写 `axes/6-predict/6.2-speculative-decoding/experiments/2673-robustness-13m/**`；只读复用 2640 / 2650 / 2660 的代码与 checkpoint。
- **禁改**：`pathlm/**`、`train_*.py`、`tools/**`、`docs/**`、其它轴、其他实验目录、任何 `project.yaml`。
- GPU 按 `SCHEMA.md` §20 取锁（复制 2660 的 `gpu_lock.py`），**先 code 锁再 GPU 锁**；不得杀他人进程。
- 证据不删；数字可回溯到落盘原始数据。
