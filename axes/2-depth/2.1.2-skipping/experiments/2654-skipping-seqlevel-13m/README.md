---
experiment: 2654-skipping-seqlevel-13m
grid: axes/2-depth/2.1.2-skipping
ticket: .scratch/61-axis2-depth/TICKET.md §9.2 WP-2654
date: 2026-10-14
engine: pathlm/depth_ar.py（depth-ar/dense）+ train_depth_ar.py（共享 main）
scale: 13M（d=256 / L=8 / h=8 / seq=256 / mlp_mult=4）
---

# 2654 · 跳层（skipping）的序列级重测 · 13M · depth-ar/dense

## 1. 命题

在能跑本线 checkpoint 的 depth-ar 引擎里，用**序列级 clean bpc**与 **layer-passes/token**
重测跳层：**固定跳过中间两层**（`skip-mid`，layer-passes 6）与**逐 token 独立跳层**
（`layerdrop-pt`，p=0.1，layer-passes ≈7.2）能否在 13M/12k 尺度上，以更少的等效算力
换到与 stock 全深度不可排序的序列级 bpc，**并且**相对「同预算（6 layer-passes）更浅的
dense 模型」（`layers6`）有**独立增量**。

**可证伪**：若 `skip-mid` 相对 `layers6` 在 clean bpc 上无可排序改善（配对 |Δbpc| < 0.014），
则「跳中间层」在本尺度没有独立于「直接训更浅同预算模型」的增量 ⇒ 跳层应并入 2.1.1 的
逐 token 深度分配（`project.yaml` `falsification` 逐字预登记）。

## 2. 单因素声明

**只改跳层策略**。四臂共用同一基底 —— stock decoder-only transformer
（`--dense --topology sequential`：无门、无退出、仅最终深度 CE），d=256 / heads=8 /
seq=256 / mlp_mult=4 / batch=32 / lr=1e-3 / steps=12000 / seed=0 / 数据 `data/enwik8_full.npz`。

**跳层语义（本实验的定义，全臂一致）**：被跳层对**该 token** 的输出 = 其输入（恒等旁路），
用逐 token 布尔掩码 `out = where(skip, x, block_out)` 实现 ⇒ 因果性不破（token i 的输出只
依赖 ≤ i 的输入），且该层不产生该 token 的 KV。**layer-passes**（等效理论量）= 该 token
实际经过的非跳层数，逐 token 统计后取均值。实现 = 实验内**工厂替换**
（monkey-patch `train_depth_ar.DepthARModel`，沿用 `2641` 的既有模式）：训练整段走共享
`train_depth_ar.main()`，只有每层 Block 外面套一层实验本地 `SkipBlock`（`run.py`）。
**`pathlm/**` 与 `train_depth_ar.py` 零改动。**

## 3. 指标表

| 指标 | 解决什么 | 代价 | 有效性（对照） | 收益与用法 | 可控性 | 副产品 |
|---|---|---|---|---|---|---|
| clean bpc（最终深度读出，序列级，越低越好） | 质量 | 无（同评测） | 逐臂对 stock / 同预算 `layers6` | 判断跳层是否值 | seed/数据切分固定 | 逐深度曲线 |
| layer-passes/token（实测） | 等效算力 | 无（掩码账） | 对设计值（8 / 6 / 6 / 7.2） | 归一质量-代价前沿 | 掩码是干预本身 | 逐层跳过占比 |
| 逐层跳过占比（训练期 / 评测期） | 分布一致性 | 无 | 训练 vs 评测 | 排除「训练分布外崩点」 | 掩码采样 | 训练期累计账 |
| 窗口间 SE / 两半样本 | 抖动 | 无 | — | 判断 Δ 是否越带 | — | — |

## 4. 判据线（**预登记、不可事后改**）

符号统一（`harness/SCHEMA.md` §17.3）：`Δ(x vs y) := bpc_y − bpc_x`（**正 = x 更好**，即 x 的 bpc 更低）；
所有 bpc 判据用**配对逐窗口 Δ**（同 `EVAL_SEED=2` ⇒ 四臂同位置），0.014 bpc 带。

- **P0 引擎等值门**：`stock` 臂的 clean bpc 与 `2613` s0 sequential 臂（**1.5254**）之差
  `< 0.005`（或与历史 `23d-dense-B-12k` 的 **1.5287** 之差 `< 0.005`）。不过门 ⇒ 本实验的
  stock 基线不可信，先按 §8 决策树查实现。
- **P1 活性**：实测 layer-passes/token 与设计值一致 —— `stock` 8.000、`skip-mid` 6.000、
  `layers6` 6.000、`layerdrop-pt` 7.2（逐 token 逐层独立 Bernoulli(0.1) 的期望；实测落
  `7.2 ± 窗口间 SE`）。不一致（|实测 − 设计| ≥ 0.2）⇒ 该臂作废（`harness/SCHEMA.md` §10）。
- **P1b 训练/评测分布一致**：`layerdrop-pt` 的**训练期**逐层跳过占比与**评测期**逐层跳过占比
  逐层差 `< 0.05`（两边走同一份 `SkipBlock` 采样代码）。`skip-mid` 恒跳 ⇒ 两边恒 1.0。
- **P2 判据（三条，逐条判）**：
  1. `Δ(skip-mid vs stock)`（0.014 带）—— 跳层省算力换质量的可排序性；
  2. **`Δ(skip-mid vs layers6)`（0.014 带）= 判决线** —— 跳层相对同预算固定深度是否有独立增量；
  3. `Δ(layerdrop-pt vs skip-mid)`（0.014 带）—— 逐 token 自适应跳是否优于固定跳。
- **P3（率类指标不借 0.014）**：跳过层占比、训练/评测分布一致率走配对 bootstrap / Wilson 95% CI。
- **P4（预算不够时砍什么）**：先砍 `layerdrop-pt`（把「逐 token 独立跳」写进 `gaps` 作下一步
  最小实验），**不砍判据线**。
- **P5（不外推）**：结论不越 13M/12k 尺度与 enwik8 文本分布。
- **墙钟一律不进判据**（只作过程量留 `results.json` 的 `shared_train.wall_minutes`）。

## 5. 对照与方差

- 对照臂 = 本实验自带的 `stock`（8 层全活）；`layers6` 是「同预算最优固定深度」对手（M3 要求）。
- 方差带 = **0.014 bpc**（本仓 13M 跨 run 漂移界），限**配对 Δ、≥5 窗口**；本实验用 **20 窗口**
  （每窗口 = batch16 × seq256 = 4080 预测位）。
- `layerdrop-pt` 的评测**逐窗口重采跳层掩码** ⇒ 另报 `delta_window_se` 与 `delta_halves`（两半样本），
  不把采样噪声读成跨 run 训练噪声。
- 复用 `2613` s0 作 P0 的**等值锚**（非对照臂）：`harness/SCHEMA.md` §9 三条 —— ① 实现等值（stock 臂
  不套跳层、逐位等于 plain `DepthARModel`，见 §8 冒烟）；② 配置逐字段对照（`config.json` 与
  `2613 config` 逐字段相同，仅评测口径一致）；③ 口径标注（同为 `depth_curve` 最终深度 bpc）。

## 6. 成本

- **gpu_wall_budget: 45 min**（含 20 步冒烟 + 4 臂 × 12k 步训练 + 评测）；与派发契约
  `TICKET.md §9.2 WP-2654` 的 `gpu_wall_budget: 45 min` 一致。
- 显存：峰值 **≤5GB net**（13M 模型 batch32 seq256，历史同配方实测 ~1.5GB）。
- 可中断性：**按臂增量落盘**（每臂训练+评测完立刻写 `results.json` 的 `completed_arms` /
  `pending_arms`）⇒ 中断后 `python3 run.py --arms <剩余臂>` 续跑，不重跑已完成臂。
- 冒烟：`python3 run.py --smoke`（全臂 20 步训练 + 2 窗口评测，断言 layer-passes 与设计一致）。
- GPU 纪律：`harness/SCHEMA.md` §20 锁（`/tmp/pathlm_gpu.lock`），**按臂取/放**（§20.1，含 ≥60s 礼貌
  窗口），等锁上限 10 min/轮，不杀他人进程、不动他人锁。

## 7. 数据与脚本清单

- 数据集：`data/enwik8_full.npz`（first 90M train / last 10M eval，vocab 206）；加载器
  `pathlm.data.load_enwik8_full`；切分 = `.tmp/enwik8`。
- 入口：`run.py`（`--arms` 按臂续跑 / `--no-train` / `--smoke` / `--aggregate`）；
  驱动：`run_remaining.sh`（有界后台 launcher，逐臂取锁）；`gpu_lock.py`（§20 协议）。
- resolved config：`config.json`；逐臂 run config 在 `runs/<arm>-12k-s0/results.json` 的 `config`。
- 落盘：`results.json`（本实验汇总，逐臂增量）、`runs/<arm>-12k-s0/{model.pt,train_log.jsonl,
  gate_log.jsonl,results.json}`、`run_log.txt`、`smoke/`。

## 8. 失败决策树（无效时的处置路径，按此顺序排除）

1. **测量**：P0 不过门 ⇒ 先核评测口径（clean bpc 是否 `depth_curve` 最终深度、`EVAL_SEED`/窗口数
   是否与锚一致），再核 stock 臂是否真的没套跳层（冒烟里断言 `SkipBlock.mode=="none"` 逐位透传）。
2. **实现**：layer-passes 实测 ≠ 设计 ⇒ 查 `SkipBlock._sample_skip`（`fixed` 是否只跳 `SKIP_LAYERS`、
   `drop` 是否逐 token 独立、`torch.where` 是否用**层输入**做旁路）；训练/评测分布不一致 ⇒ 查两
   边是否同一 `p`、同一采样代码。
3. **规模**：所有臂 bpc 都异常高/低（训练未收敛）⇒ 查 loss 斜率与 gnorm，必要时缩步数保**臂间同刻度**
   （`harness/SCHEMA.md` §20.3），不砍臂。
4. **设计**：P2 判决线读数为「无独立增量」⇒ 按 `project.yaml` `falsification` 如实判，不调判据线。

## 9. 产出

- `README.md`（本契约）/ `config.json` / `run.py` / `run_remaining.sh` / `gpu_lock.py`。
- `results.json`（逐臂增量 + 汇总 P0–P5）/ `run_log.txt` / `smoke/`。
- `analysis.md`（判读，含人话摘要 + 数字表 + 口径标注）/ `MODEL.md`（权重哈希，不入库）。
