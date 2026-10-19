---
id: 2653-interlayer-attn-13m
axis: "2 深度轴"
node: "2.2.2 用 causal attention 代替残差"
date: "2026-10-14"
status: preregistered
---

# 2653 · 层间 causal attention 最小判决（vs stock 单残差 / vs 线性学习加权和）· 13M · dense

> 预登记契约（`SCHEMA.md` §6 九节）。判据线 P1–P6 **写在跑之前，跑完不得改**（`SCHEMA.md` §8）。
> 口径（`TICKET.md` §9.1）：判据只用**等效理论算力**（额外参数、每 token 额外 MACs、
> layer-passes/token）；**墙钟不得进判据或 `cost_model`**。

## 1 命题

把每层的输入换成「对**之前所有层输出**的一次 causal attention 聚合」
（`h_k = Block_k(Attn(h_0..h_{k-1}))`，层索引因果）相对 stock 单残差
（`h_k = Block_k(h_{k-1})`）与 2.2.3 的**线性**学习加权和（`dense` 臂），
能否把 clean bpc 拉低越噪声带——即「内容相关权重（q·kᵀ）是否值那份额外算力」。
可证伪：两臂相对两个基线都落在 ±0.014 内 ⇒ 判「在本尺度无净增量，应退化为线性聚合」。

## 2 单因素声明

唯一变量 = **层间聚合的参数化**（引擎 `topology="dense"` 的前向路径 + 本实验覆写
`_layer_mix_input`；`pathlm/**` 不改）。基底 = 主流 stock decoder-only（pre-LN、单残差
Block、tied 读出；`--dense`）。

- `attn-scalar`：q/k/v **恒等**（无学习投影），每层一个可学习标量温度 `τ_k`
  ⇒ 额外参数 = **L = 8**（`dense` 的线性加权和是 **+36**，同量级 ⇒ 与 2.2.3 的等参数对照）。
- `attn-qkv`：同聚合，换**可学习共享** `W_q/W_k/W_v`（`bias=False`，init = I）
  ⇒ 额外参数 = **3d² + L = 196 616**（逐臂报）。
- **单因素无法独立生效的部分**：`qkv` 臂必须多带 3d² 参数（这是「内容相关权重」的定义代价），
  故归因用「相对 `scalar` 臂的增量」而不是相对 stock；两臂共用同一构造顺序 ⇒ trunk/gates
  的初始化与 stock 逐位相同（`tau`/`W` 的 init 不消耗随机数流）。
- 层 l 只允许看 `h_0..h_l`（层索引因果）；**看不到当前层之后的信息**（P6）。

## 3 指标表

| 维度 | 内容 |
|---|---|
| 解决什么问题 | 逐层串联是否构成信息瓶颈；跨层注意力能否读到更早层的表示 |
| 代价 | 额外参数（`params_added`）+ 每 token 额外 MACs（解析）；layer-passes/token 仍 = L |
| 有效性 | 相对 2613 s0 的 `sequential`（stock）与 `dense`（2.2.3 线性聚合）的配对 Δbpc，0.014 带 |
| 收益与用法 | 若显著更优 ⇒ 跨层连接是可兑现的（且要说明 qkv 那 3d² 值不值）；若不可排序 ⇒ 判退化到 2.2.3 |
| 可控性 | 聚合权重逐层落盘（熵 / 紧邻上一层权重 / 最大权重）、`τ` 终值落盘 |
| 副产品 | 逐深度 bpc 曲线、loss 斜率、跨层权重结构（哪些早期层被读） |

## 4 判据线（预登记，不可改）

符号约定（`SCHEMA.md` §17.3）：`Δ = bpc_对照 − bpc_实验`，正 = 实验臂更好。

- **P1** 每臂对两个基线的配对 Δ：`Δ_stock = bpc_sequential(2613 s0) − bpc_arm`；
  `Δ_dense = bpc_dense(2613 s0) − bpc_arm`。Δ ≥ +0.014 ⇒ 可分更优；≤ −0.014 ⇒ 可分更差；
  |Δ| < 0.014 ⇒ **不可排序（H0）**。
- **P2** 活性（`SCHEMA.md` §10）：层维权重**非退化**——k ≥ 2 的层 `w_immediate_prev < 1` 且熵 > 0
  （层 1 只有唯一源 ⇒ 平凡，不计）；`τ` 终值落盘。
- **P3** 代价列：额外参数与每 token 额外 MACs（相对 stock 单残差）逐臂落盘。
- **P4** 等值门：`force_last=True`（层维权重强制 one-hot 到紧邻上一层）时前向与 loss 与 stock
  sequential **逐位一致**（`test_2653_interlayer_attn.py` 钉住，探针：不强制时必须不同）。
- **P5** 基线复用的三条等值证明（`SCHEMA.md` §9）：实现层（P4 的逐位单测）、配置层（与
  2613 s0 逐字段对照）、口径标注（`analysis.md`）；任一不等 ⇒ 回退自包含双跑。
- **P6** 层索引因果：聚合只取 `states[0..k−1]`（`j ≤ k−1`），代码结构保证；单测断言聚合
  对 `states[k]` 及之后**不可见**（不可见量 = 不参与计算）。
- **该跌 / 必须报 / 必须不变**（本项 `objective_criteria`）：clean bpc 该跌（否则判无净增量）；
  必须报额外参数与额外 FLOPs；必须不变：因果性、等参数对照的参数量。

## 5 对照与方差

- 对照 = 2613 s0 的 `sequential`（stock 单残差，clean bpc 1.5254）与 `dense`（2.2.3 线性学习
  加权和，1.5246，+36 参数），**同一实现/同一配置/同一 98.3M token 预算/同一 seed**（P5 三条等值证明）。
- seed 数 = 1（s0）；噪声带 0.014（`docs/methodology.md` §2 跨 run 界）。
- 窗间 SE 另报（5 窗口 × 4 批，所有臂共用同一 generator seed 的同一批序）；历史臂的逐窗数据
  不可得 ⇒ 对历史臂只报窗口均值与窗内 SE，**不假装是真配对**（口径在 `analysis.md` 标注）。

## 6 成本

- `gpu_wall_budget: 30 min`（含冒烟 + 2 臂 × 12k 步 + 评测）。
- 硬件 = 单卡 RTX 4090（24GB）共享；GPU 按 `SCHEMA.md` §20 取锁、**按臂释放**（§20.1，臂间 ≥60 s）。
- 峰值显存目标 ≤5GB net；每臂等效算力 = 12k × 32 × 256 = **98.3M token**（两臂 196.6M token）。
- 可中断性：`run.py --arms <单臂>` 可续；结果**按臂增量落盘**。
- 冒烟计划：`smoke.py`（CPU：等值门 + 活性门 + 参数账，各带必红探针）→ 20 步 CLI 小跑。
- 预算不够时**砍 arm 不砍判据线**：先砍 `attn-qkv`（只报 `attn-scalar` 与两基线），
  把「内容相关权重」写进 `gaps`。

## 7 数据与脚本清单

- 数据：`data/enwik8_full.npz`（前 90M train / 后 10M eval，vocab=206，与全轴同切分）。
- 入口：`run.py`（`--arms` / `--no-train` / `--aggregate`）；聚合实现：`interlayer_attn.py`。
- 等值门/冒烟：`smoke.py`；单测：`test_2653_interlayer_attn.py`。
- resolved config：本目录 `config.json` + 每臂训练产物内 `results.json.config`。
- 落盘：`results_<arm>_s0.json`、`results.json`、`smoke_report.json`、`analysis.md`、
  `MODEL.md`、`run_log.txt`（每臂一条起始与状态的机器可读记录）、`runs/<arm>-12k-s0/`。

## 8 失败决策树（按序排除）

1. **实现**：P4 等值门不过 ⇒ 聚合接线错（如把当前层也当源、或 one-hot 没走同一累加路径），
   先修接线，臂作废。
2. **测量**：`clean_bpc_two_paths_agree=False`（窗口法 vs `depth_curve`）⇒ 评测路径有错，重测。
3. **规模**：Δ 不可排序且 loss 斜率仍显著非零 ⇒ 只作快照结论，标注「疑为收敛加速」，不越尺度外推。
4. **设计**：判据齐备且不可排序、或优势全被额外参数/算力解释 ⇒ 判「退化为 2.2.3 的线性聚合」。

## 9 产出

`results.json` / `results_<arm>_s0.json` / `smoke_report.json` / `analysis.md` /
`MODEL.md`；`project.yaml`（`status`/`conclusion`/`failure_analysis`/`conclusion_review`/`argument`/
`gaps`/`experiments`/WP `status`/`change_log`）更新；`.scratch/61-axis2-depth/WP-2653-REPORT.md`。
独立复核（fresh 上下文，`SCHEMA.md` §14/§14b）排在 writer 停止之后：**第 1 轮判 PASS**（判词由
lead 落盘于 `.scratch/61-axis2-depth/**`）；两臂训完后再派新一轮。
