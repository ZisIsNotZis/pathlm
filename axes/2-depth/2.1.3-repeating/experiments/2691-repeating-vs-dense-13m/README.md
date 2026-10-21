---
experiment: 2691-repeating-vs-dense-13m
grid: axes/2-depth/2.1.3-repeating
date: 2026-10-15
engine: pathlm/depth_ar.py（depth-ar/dense）+ train_depth_ar.py（共享 main）
scale: 13M（d=256 / seq=256 / heads=8 / mlp_mult=4；层数见臂表）
---

# 2691 · 循环深度（同权重重复前向）对「同算力稠密」与「同参数稠密」· 13M

## 1. 命题

把**同一组层参数**在同一个 token 上重复前向若干遍（Universal-Transformer 式循环深度），
用**算力**换**参数**。本实验判两问，各自对上一个不同的对手：

- **问一（同每 token 算力）**：12 次层传递、6 组参数（`repeat2x`）比 12 组独立参数各跑一次
  （`dense12`）更值吗？如果是，说明**参数是瓶颈**；如果更差，说明**层间异质性**（每层一套
  自己的权重）本身是必需的。
- **问二（同参数量）**：12 次层传递、6 组参数（`repeat2x`）比 6 层 dense 只跑 6 次
  （`dense6`）更值吗？如果是，说明多花的那一倍算力买到了质量；如果落噪声带内，
  说明这笔算力白花。

**可证伪**：若两问的配对 Δ 都落 0.014 bpc 带内、或同算力那一问**反向**（循环更差），
则「循环深度改变某算力路径值不值」在本尺度不成立（`project.yaml::falsification` 逐字预登记）。

**若循环更差**，必须区分是**深度不足**（12 次传递仍不如 12 组参数，因为 6 组参数的容量不够）
还是**层间异质缺失**（12 组不同的层才买得到质量）。诊断臂 = `repeat2x-tail`：**同参数、同算力**
（6 组参数、12 次传递），只有「重复哪几层」不同 —— 前 3 层各用一次、后 3 层各用三次
（`0,1,2,3,4,5,3,4,5,3,4,5`）。它把「异质性」与「重复次数」分开：
`tail` 比 `repeat2x` 好 ⇒ 前段的异质层有独立价值；`tail ≈ repeat2x` ⇒ 差别在「一共只有 6 组
参数」，不是重复本身。

## 2. 单因素声明与设计条件

**唯一变量 = 层参数是否被重复使用、重复几次**。五臂共用同一基底与同一训练配方：
stock decoder-only（`--dense --topology sequential`：无门控、无退出、无填充，仅最终深度 CE）、
d=256 / heads=8 / seq=256 / mlp_mult=4 / batch=32 / lr=1e-3 / steps=12000 / seed=0 /
数据 `data/enwik8_full.npz`。**同 token 预算 = 每臂 12k × 32 × 256 = 98.3M token**；
**同评测入口 = 本目录 `run.py::evaluate_arm`**（20 窗口 × batch16 × seq256、`EVAL_SEED=2`）。

| 臂 | 组数（建几层） | 每 token 层传递 | 每 token MACs（解析） | 参数（解析） | 角色 |
|---|---|---|---|---|---|
| `dense6` | 6 | 6 | 5.48096e6 | ≈5.25M | 问二对照：同参数稠密 |
| `dense12` | 12 | 12 | 10.90918e6 | ≈10.38M | 问一对照：同算力稠密 |
| `repeat2x` | 6 | 12（1–6 再来一遍） | 10.90918e6 | ≈5.25M | 主臂：全栈重复两遍 |
| `repeat2x-tail` | 6 | 12（后 3 层各 3 次） | 10.90918e6 | ≈5.25M | 诊断臂：只重复后半段 |

解析口径（`run.py::macs_per_token`，逐 token）：每次层传递 `12·d² + T·d`
（qkv 3d² + 输出投影 d² + FFN 8d² + 因果注意力平均 `T·d`），每个深度的读出 `d·V`
（读数个数 = 传递次数 + 1，两臂相同 ⇒ 读出算力也相同）。参数量由实测 `sum(p.numel())`
落盘（`results.json::arms.<臂>.params`）：`dense6` 与 `repeat2x` **逐位相同**，
`dense12` 是它们的约 1.98 倍。

**要这个机制产生可测效应，设计上必须满足的条件（`atlas/playbook.md` 定律 10）**：

1. **有足够的变化空间**：6 组参数被重复两次，若 6 组参数的容量本身够、则第二遍应买到质量；
   若不够，则同算力对照（12 组参数）会赢 —— 这正是本实验要判的，两分支都可判。✅ 满足（判据是二选一）。
2. **等参数/等算力对照都真实存在**：`dense6`（等参数）与 `dense12`（等算力）都是**从零训练**
   的同配方臂，不是解析换算。✅ 满足。
3. **训练信号迫使其使用第二遍**：两臂都只监督最终深度 CE（与 dense 基线同配方）⇒
   第二遍不是「附加监督」而是唯一被优化的路径。✅ 满足。
4. **循环臂能分辨第几遍**：本实验**不加**步嵌入（会破坏「参数≈一组层」的单因素），
   只靠残差流自身携带状态。这是本设计的一个**限制**：Universal Transformer 原式有时间步
   嵌入，若模型因无法分辨遍数而变差，本实验无法把它与「异质性缺失」分开 ⇒ 登记为
   `project.yaml::gaps` 的下一步最小实验（每步一套 d 维步嵌入，参数增量 (n−1)·d ≈ 1.5k）。
   缺这条时结论只能写「**本设计下**循环深度值不值」。

**实现**：实验内**工厂替换**（monkey-patch `train_depth_ar.DepthARModel`），
沿用 `2654` 的既有模式：训练整段走共享 `train_depth_ar.main()`，只有前向按本臂的
**层调度表**（`ARMS[臂]["schedule"]`，元素 = 组的下标）执行。
`pathlm/**` 与 `train_depth_ar.py` **零改动**；`RepeatModel` 是 `DepthARModel` 的子类，
不新增任何参数（`state_dict` 键与 stock 逐位相同）。

**等价性控制（可失败的）**：`schedule` 为恒等（`dense6` / `dense12`）时，`RepeatModel` 的
logits 必须与 stock `DepthARModel` **逐位相同**（单测 `test_repeat_schedule.py`）；
`dense6` 与 `repeat2x` 在**同 seed 下逐位相同的初始化**（组数相同 ⇒ 建模型的随机数流相同），
两臂的差别只有「每层被调用几次」。

## 3. 指标表

| 指标 | 解决什么 | 代价 | 有效性（对照） | 收益与用法 | 可控性 | 副产品 |
|---|---|---|---|---|---|---|
| clean bpc（最终深度读出、序列级、越低越好） | 质量 | 0（同评测入口） | 逐臂配对（同 `EVAL_SEED`） | 两问的判决量 | seed/切分固定 | 逐深度 bpc 曲线 |
| 每 token MACs（解析） | 等效算力 | 0 | `dense12` = 1.000000 | 归一质量-代价前沿 | 传递次数 + 读出数 | 逐臂比值 |
| 参数量（实测） | 参数账 | 0 | `dense6` / `repeat2x` 逐位相同 | 问二的对齐 | — | 组数 / 传递次数之比 |
| 峰值显存（实测，`peak_gpu_mem_mb`） | 显存代价 | 0 | 逐臂 | 必须报 | 训练期最大分配 | — |
| 逐深度 bpc（0..传递次数） | 深度是否还在赚 | 0 | — | 区分「深度不足」与「异质缺失」 | — | 曲线 |
| 层调用次数（实测）/ 调度活性 | 干预是否活着 | 0 | 设计传递次数 | SCHEMA §10 活性量 | 计数器 | 恒等调度偏差 |
| gnorm max / 跳步数 | 训练稳定性 | 0 | 逐臂 | 必须不变（不崩） | 共享训练器 | — |
| 墙钟 | 过程量 | 0 | 逐臂 | **只申报、不进判据** | 共享卡 | — |

## 4. 判据线（**预登记、不可事后改**）

符号统一（`harness/SCHEMA.md` §17.3）：`Δ(候选 vs 对照) := bpc_对照 − bpc_候选`
（**正 = 候选更好**）；所有 bpc 判据用**配对逐窗口 Δ**（同 `EVAL_SEED=2` ⇒ 各臂同位置），
**0.014 bpc 带**（限配对 Δ、≥5 窗口；本实验 20 窗口）。墙钟一律不进判据。

- **P0 引擎等值门**：用本实验的评测代码重评 `2654` 的 `stock` checkpoint
  （`axes/2-depth/2.1.2-skipping/experiments/2654-skipping-seqlevel-13m/runs/stock-12k-s0/`，
  8 层、12k 步、seed 0），其 clean bpc 与 `2654` 落盘值（`axes/2-depth/2.1.2-skipping/experiments/2654-skipping-seqlevel-13m/results.json::arms.stock.eval.clean_bpc`）
  之差 `< 0.002`。不过门 ⇒ 本实验的评测仪器与历史不可比，先按 §8 查实现，
  **不得**用其余读数下判。权重不在仓库里（`.gitignore`）；若本机没有该 checkpoint，
  P0 记为 `not_run`（显式缺失，不算过门），其余判据照判但在 `analysis.md` 里标明仪器未复核。
- **P1 活性门（结构 + 调度）**：
  1. 每臂实测「每前向的层调用次数」= 设计的传递次数（6 / 12 / 12 / 12）；
  2. 恒等调度臂（`dense6` / `dense12`）与 stock `DepthARModel` 的 logits 逐位相同（单测）；
  3. 循环臂的**重复是活的**：同一 checkpoint、同一输入下，「完整调度」与「只跑前 `组数` 次」
     的最终 logits 相对偏差 > 0（对 `dense6`/`dense12` 这两个调度相同、偏差恒 0）。
     实测偏差 = 0 而调度本该不同 ⇒ 该臂作废（SCHEMA §10）。
- **P2 判据（两问，逐条判，`0.014` 带）**：
  1. **问一（同算力）**：`Δ(repeat2x vs dense12)`；
  2. **问二（同参数）**：`Δ(repeat2x vs dense6)`；
  3. **诊断**：`Δ(repeat2x-tail vs repeat2x)`（同参数、同算力，只差重复哪几层）
     与 `Δ(repeat2x-tail vs dense12)`。
- **P3（率类/曲线类不借 0.014）**：逐深度 bpc 曲线、参数量、MACs、峰值显存只报数与口径，
  不设带；逐窗口 SE 与两半样本随判据一起报。
- **P4（预算不够时砍什么）**：先砍 `repeat2x-tail`（把诊断写进 `gaps` 作下一步最小实验），
  **不砍判据线**；`dense6` / `dense12` / `repeat2x` 三臂是判决的最小集合，缺一不下判。
- **P5（不外推）**：结论不越 13M / 12k 步 / T=256 / enwik8 / 单 seed 这一格；
  不经由本格的读数外推规模（`atlas/playbook.md` 定律 8）。

## 5. 对照与方差

- 三个判决臂两两可比：`repeat2x` ↔ `dense12` 同算力（MACs 比值 1.000000）、
  `repeat2x` ↔ `dense6` 同参数（参数量逐位相同）、`repeat2x-tail` ↔ `repeat2x` 同参数且同算力。
- 方差带 = **0.014 bpc**（本仓 13M 跨 run 漂移界），限**配对 Δ、≥5 窗口**；本实验用 **20 窗口**
  （每窗口 = batch16 × seq256 = 4080 预测位）。
- 另报 `delta_window_se`（窗口间 SE）与 `delta_halves`（两半样本），不把窗口内抖动读成跨 run 漂移。
- **单 seed 的边界**：本格 `n=1`（每臂 1 个训练 seed）。按 `harness/methodology.md` §2，
  方向判决可用、**量级判决不可用**；同向且远超带的结论才写「可排序 ×N 倍带」，
  落带内的结论只写「不可排序」，不写「相等」。

## 6. 成本

- **gpu_wall_budget: 70 min**（20 步冒烟 + 四臂 12k 步训练 + 四臂评测 + P0 重评）。
  实测预算（按 8 层 12k 步 ≈6.5 min 折算）：`dense6` ≈5 min、`dense12` ≈10 min、
  `repeat2x` ≈10 min、`repeat2x-tail` ≈10 min、评测合计 ≈3 min、冒烟 ≈2 min。
- **单臂 timeout 20 min**（launcher 用 `timeout 1200` 强制；超过即放弃该臂并如实记为失败）。
- 显存：峰值 ≤5 GB（13M、batch32 seq256；同配方历史实测 ~0.9 GB）。
- 可中断性：**按臂增量落盘**（每臂训练 + 评测完立刻写 `results.json` 的 `completed_arms` /
  `pending_arms`）⇒ 中断后 `python3 run.py --arms <剩余臂>` 续跑，不重跑已完成臂。
- GPU 纪律：`harness/SCHEMA.md` §20 锁（`/tmp/pathlm_gpu.lock`），**按臂取/放**
  （§20.1，含 ≥60 s 礼貌窗口），等锁上限 10 min/轮，不杀他人进程、不动他人锁；
  卡上更高优先级的常驻项目优先。

## 7. 数据与脚本清单

- 数据集：`data/enwik8_full.npz`（first 90M train / last 10M eval，vocab 206）；
  加载器 `pathlm.data.load_enwik8_full`；切分缓存 = `.tmp/enwik8`。
- 入口：`run.py`（`--arms` 按臂续跑 / `--no-train` / `--smoke` / `--aggregate` / `--p0-only`）；
  驱动：`run_arms.sh`（有界后台 launcher，逐臂取锁 + 单臂 timeout）。
- resolved config：`config.json`；逐臂 run config 在 `runs/<臂>-12k-s0/results.json` 的 `config`。
- 落盘：`results.json`（本实验汇总，逐臂增量）、`runs/<臂>-12k-s0/{model.pt,train_log.jsonl,
  gate_log.jsonl,results.json}`、`run_log.txt`。
- 单测：`test_repeat_schedule.py`（CPU，秒级；恒等调度等值 + 重复活性 + 等参数 + 调度覆盖）。

## 8. 失败决策树（无效时的处置路径，按此顺序排除）

1. **测量**：P0 不过门 ⇒ 先核评测口径（clean bpc 是否最终深度、`EVAL_SEED`/窗口数是否与
   `2654` 一致、数据切分缓存是否同一份），再核重建配置（`DepthARConfig(**run_config)`）。
2. **实现**：P1 的层调用次数 ≠ 设计 ⇒ 查 `ARMS[臂]["schedule"]` 与 `RepeatModel.forward`
   的循环；恒等调度与 stock 不等值 ⇒ 查 `_input_states` / 读出 / 损失三者是否与父类逐字一致
   （单测 `test_identity_schedule_equals_stock` 钉住）。
3. **规模**：所有臂 bpc 都异常高（训练未收敛）⇒ 查 loss 斜率与 gnorm，必要时**同时**缩步数
   保臂间同刻度（`harness/SCHEMA.md` §20.3），不砍臂、不改判据线。
4. **设计**：P2 的读数落带内 ⇒ 按 `project.yaml::falsification` 如实判「不改变判断」，
   不调判据线；P2 问一反向 ⇒ 按诊断臂与逐深度曲线区分「深度不足 / 异质缺失」再写判词。

## 9. 产出

- `README.md`（本契约，绑定时冻结）/ `config.json` / `run.py` / `run_arms.sh` /
  `test_repeat_schedule.py` / `prereg_binding.json`。
- `results.json`（逐臂增量 + P0/P1 门 + 三对比较 + 逐深度曲线）/ `run_log.txt`。
- `analysis.md`（判读：人话摘要 + 六问 + 数字表带键路径 + 口径标注 + 残余缺口）；
  `MODEL.md`（权重哈希）。
