# 2630-tf4-samecaliber · 对照臂与方差说明

> 判据线原文见 `../README.md` §4（逐字抄录）。本文件把「对照是什么、为什么可比、
> 部署里对应什么」写清（SCHEMA §13 三要素 + §17.14 基准唯一）。

## 1. 对照臂（唯一变量 = 是否开写入通道）

| 臂 | `carry_rule` | 形态件（同款） | 训练数据 | 部署含义 |
|---|---|---|---|---|
| **`base`（基准）** | 关（`None`） | 非 dense + 随机退出 + proj-fill + `final_weight 2.0` | **enwik8** | 「无搬运通道」的 100M 出口资产配方 |
| **`carry`（目标）** | **delta 写** | 同上 | **enwik8** | 该配方再加一条 delta 搬运通道 |

- **基准唯一（§17.14）**：主判据「同口径 tf@4」的基准 = `base`；代价项（clean bpc）也相对
  `base` 算；主判据已显式指定（`tf4_enwik8_samecaliber`），tf@6 / clean bpc / 参数 / 活性
  为并列项。**不与裸 stock 比**（那会把形态件代价算进 carry 的账）。
- **§13 三要素**：
  1. **输入分布**：两臂都在 enwik8（`data/enwik8_full.npz`，train 90M / eval 10M）上训练
     与评测——**同一分布、同一切分、同一批、同一 seed**。
  2. **要预测什么**：标准因果 next-token（tf@4 = depth-4 出口 argmax 与全栈 argmax 的
     一致率；clean bpc = 全深度 CE）。
  3. **部署对应**：tf@4 = 「用前 4 层当草稿、被全深度接受的比例」= 自 spec 解码的草稿
     命中率代理；clean bpc = 语言建模质量。

## 2. 方差与可比性

- **方差带 = 1.4pp**（`harness/methodology.md`：Δ < 0.014 不可排序）。决判 `|Δtf@4| < 1.4pp`。
- **成对比对**：两臂用**同一引擎函数** `pathlm.depth_ar.tf_acceptance`、**同一批**
  （`generator` seed 3）、**同一批数**（20×16）⇒ 逐位可复现、无采样噪声。
- **单 seed（n=1）**：本线是快照口径、单 seed，只读**臂间同刻度增量**，不外推绝对值
  到其它尺度/步数（任务 §6「禁外推」）。
- **快照声明**：6k 步 / 49M token；末窗全深度 CE 与 depth-4 CE 斜率并列报（§17.15）。

## 3. 口径对照（**不作判据**，只为解释「口径为何会打架」）

`results.json::tf_synth_crosscaliber_diag` 报一条**反向跨分布诊断**：把 enwik8 训好的
两臂放到 synth assoc-mqar 上评 tf@4。它与本线主口径的关系：

| 口径 | 训练分布 | 评测分布 | 是否判据 | 与 2629 的关系 |
|---|---|---|---|---|
| 本线 `tf4_enwik8_samecaliber` | enwik8 | enwik8 | **是（主）** | 2629 的 enwik8 腿是**跨分布代理**（训练在 synth），**不得混用** |
| 本线 `tf_synth_crosscaliber_diag` | enwik8 | synth | 否（诊断） | 方向与 2629 的 enwik8 代理相反，用于对照「口径效应」 |

> **不得混用纪律**：本线的同口径数字与 2629 的跨分布数字是**两个不同分布的口径**，
> 在报告里分开写；不得把「2629 的 enwik8 代理 Δ」与「本线的 enwik8 同口径 Δ」拼成
> 一对臂。

## 4. 锁与记账

- `train_arms.sh` 按臂取/放 GPU 锁（§20.1），臂间留 65s 窗口；等待时长记入
  `train_arms.runlog`。
- 峰值显存、跑前/跑后空闲显存记入 `results.json::cost`（§20.2 三件套如触发则记账）。
