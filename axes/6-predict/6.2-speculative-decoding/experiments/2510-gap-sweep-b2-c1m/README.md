---
name: 2510-gap-sweep-b2-c1m
ticket: 10-gap-sweep
date: "2026-09-15"
grid: H1-outputs-per-pass
also: [F2-return-channel]
engine: "m1（13M：d=256×12，n_mtp=2）"
budget: "~25 min/run（训练，已完成）"
---

# 2510-gap-sweep-b2-c1m · n_mtp 1→2 的税与草稿接受率

> 主格：`H1-outputs-per-pass`｜关联格：`F2-return-channel`｜引擎：m1（13M）｜预算：~25 min/run

## 1. 命题

把每步输出位置数从 1 加到 2（`n_mtp=2`，多出 node-2 = t+2 的直接头）的**质量税**
是多少，它对主干的 t+1 预测（node-1）有无损伤，node-2 草稿被 node-1 验证时的一致率
（接受率锚）是多少。

## 2. 单因素声明

唯一变量 = `n_mtp`（2 vs 1）；其余（d/layers/heads/seq/batch/steps/seed/数据切分/退火）
与 B0 基线一致。**已知混杂（D2）**：C1M/C2M 臂的 `n_mtp` 未显式钉死，与对照不同 ⇒
其税虚高（ticket 11 修正）；B2 的税不受此混杂影响。

## 3. 既有做法（prior art）

> 来源标 `记忆·未核`（联网不可用）：MTP（Gloeckle 2024）、Medusa（Cai 2024）已报
> 「多解码头 = 小税 + 自带草稿」。**我们的增量**：13M char-enwik8 上的**同引擎同 seed
> 税曲线**（n_mtp 1→2→3）与 node-2 草稿对 node-1 的一致率。机制已做透 ⇒ (a)；
> 本项目尺度上的税值 ⇒ (b)。

## 4. 指标表

| 维度 | 量 | 口径 |
|---|---|---|
| 代价 | Δclean bpc（vs B0 1.5074） | 1.1-epoch 快照（6k 步） |
| 有效性 | node-1 t+1 命中率（是否被 node-2 拖坏） | 快照 |
| 收益（草稿用法） | node-2 草稿与 node-1 的一致率 = 宽-2 的 `a₂` 锚 | 计数 |
| 副产品 | node-2 直接 vs 链式头的命中差 | 快照 |
| 可控性 | node-1 ece | 快照 |

## 5. 判据线（预登记）

1. **税**：`Δ = bpc(n_mtp=2) − bpc(B0)`；`Δ < 0.014`（种子散布带）⇒ 不可排序，只称
   「未观察到超带损伤」。
2. **node-1 不受损**：`acc_node1_t1` 与 B0 `next_token_acc` 之差在噪声带内。
3. **草稿锚**：报 `draft_agree_with_node1`（= 宽-2 下 node-2 草稿与 node-1 预测的一致率），
   同时报 `acc_node2_direct_t2` / `chain_t2`。
4. **必须不变**：node-2 直接头与链式头在此形态下命中相同（若不同则记为组合效应，交 H3）。

## 6. 对照与方差

- 对照 = B0（n_mtp=1，同配置）；B2fresh 是 B2 的重训对照（`2510` 与 ticket 13 的
  `B3 1.5604 vs B2fresh 1.5438` 共用该对照）。
- 13M 单 seed（seed 0）；税落在快照口径，需按 §12 对账后再入账。

## 7. 成本

- 训练 + 评测 ~25 min/run（已完成，13M）。本目录只做**重算与对账**，不重训。

## 8. 数据与脚本清单

- 原始产物（唯一来源）：`.scratch/10-gap-sweep/evidence/{B2,C1M,C2M}/results.json`、
  `B2/mtp_accept.json`、`eval_ablations.json`。
- 入口：`train_m1.py`（训练）/ `probe_mtp.py`（接受率）。
- 落盘：本目录 `results.json`（从原始产物抽取，带 provenance）。

## 9. 失败决策树

| 症状 | 诊断 | 处置 |
|---|---|---|
| 税虚高 | 对照与实验臂 `n_mtp` 不一致（D2） | 排除该臂，改用同 n_mtp 对照 |
| 「填充/深度监督」伪影 | matched-step 不对齐 | 只信 matched-step 对照 |

## 10. 产出

`results.json` / `analysis.md` / `MODEL.md` / 本契约。**B2 checkpoint 不可得**
（`.tmp/` 已清），故只有原始 results.json，无法重跑。

## 11. scope

单轴单格（`n_mtp` 1→2）、13M、快照口径；组合（×retry/×exit/×window）默认留白。
