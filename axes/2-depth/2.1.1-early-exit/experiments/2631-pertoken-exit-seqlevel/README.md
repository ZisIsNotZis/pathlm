# 2631 逐 token 退出决策的序列级实测（轴 2 · 2.1.1 WP1）

> 主格 `2.1.1 early exit`｜引擎 depth-ar（`pathlm.depth_ar` + `pathlm.exit_policy`）
> ｜checkpoint `23c-H2-w0-12k`（13M，sha256 启动时校验）｜**纯推理，不训练**
> ｜设备 **CPU**（§20：纯 CPU 不取 GPU 锁）｜样本 = 与历史 `depth_curve` 同一 seed=2 样本（历史 `tf_acceptance` 用 seed=3，属**另一样本**，只作量级参考）
> ｜数据 `data/enwik8_full.npz` val 尾段（byte 级，V=206）

## 0 本实验要回答什么

**命题**：把「每个 token 都要跑满 8 层」改成「逐 token 决定跑几层」，能否在**降低算力**
（每 token layer-passes）的同时**保住序列级质量**？

**为什么必须做**：`.scratch/55-review/C-PROBLEMS.md` C-P01/C-P02/C-P03 判定：这条线的核心命题
**从未端到端检验过**，而公开的「收益」`tf@4 = 0.9694` 只是单 token 自洽代理（浅层 argmax 与
深层 argmax 是否相同，且训练把「浅层模仿深层」直接训进目标）；声称的「逐 token 退出门控」还长在
上一代引擎（`pathlm/decode.py`）里，能跑本线 checkpoint 的 depth-ar 解码器只支持**固定**退出深度。

**本实验的两件交付**：
1. **承载体**：`DepthARDecoder.step_auto`（逐 token 退出判断进生产解码器）+ `pathlm/exit_policy.py`
   （策略 / 序列级口径 / 校准表 / 算力账）；单测 `tests/test_axis2_exit_decision.py`（19 条）。
2. **主判据的第一次实测**：逐 token 门控退出在**自回归序列级**口径下的质量-算力账。

## 1 口径（预登记）

| 项 | 定义 |
|---|---|
| **主判据** | `stream_bpc`：teacher-forcing、**逐位置在线**决策下的整段 bpc。位置 i 的退出深度 `d_i` 只依赖 `(k, 该位置第 k 层读出)`（严格因果）；用 `d_i` 的读出预测 `t_{i+1}`；bpc = `mean CE / ln2`（bits/char，越低越好） |
| **算力账** | `layer-passes/token = mean(d_i)`、跳过层占比 `1 − mean(d_i)/L`（等效理论量） |
| **对照** | ①全深度（`never-exit`）②**固定前缀退 k 层** `prescribed-k`（k=2..7；同预算的「笨」对手） |
| **不变量** | 因果性（后面的 token 改了不影响前面位置的退出深度与读出，单测钉住）；`τ=1.0` 退化臂必须逐位等于全深度 |
| **噪声** | 仓内跨 run 噪声带 `0.014`；同 checkpoint 内比较另报**逐窗口配对差 Δbpc 的 SE**（各臂同一批窗口，配对设计） |
| **禁入判据** | 墙钟、吞吐、tok/s（`harness/research-schema.md` §3）。墙钟只出现在 `run_log.txt` 的过程行 |
| **校准** | 每深度 top-1 概率 vs 两种标签：`tf`（`argmax_k == argmax_L`）与 `true`（`argmax_k == 真实下一字节`）；报 ECE（10 等宽箱）与 risk–coverage |

**与 `depth_curve` 的关系（必须讲清）**：`depth_curve[k]` = 全深度上下文下第 k 层读出的 bpc
（所有位置都跑到 L 层，异构深度填充**不生效**）；本口径按 `d_i` **真的停机**，异构深度 KV 语义
按因果顺序传播。二者在「所有位置都退到同一深度 k」时**必须相等**——本实验用 `prescribed-k`
做**口径自洽锚点**（结果见 `analysis.md` §3：7 个深度全部在 Δbpc ≤ 4.6e-05 内重合）。

## 2 臂

| 臂 | 说明 |
|---|---|
| `never-exit` | 全深度基线（退化臂） |
| `prescribed-2/3/4/5/6/7` | 固定前缀退 k 层（对照臂 + 口径锚点） |
| `token-thr-τ`（τ = 0.3/0.6/0.9/0.95/0.99/0.999） | **逐 token 门控**：任一层读出 top-1 概率 ≥ τ 即停（CALM 式）；硬下界 1 层 |
| `token-thr-0.999-min4` | 高阈值 + 硬下界 4 层（浅层地板臂） |

共 **14 条臂**（1 基线 + 6 固定前缀 + 6 阈值扫描 + 1 地板臂）。

**归因诊断臂**（诊断，不进主判据）：
- `shuffled-token-thr-*`：保留该臂退出深度的**边缘分布**、打散「内容↔深度」相关（置换控制）；
- `nofill-token-thr-*`：关掉填充（no-fill 语义；该臂**偏离训练语义**，标为语义迁移臂）；
- `prescribed-4 × {fill, no-fill}`：**结构零假设**——全同深度下填充不触发，两臂必须逐位相同
  （活性判据，`harness/SCHEMA.md` §10）；
- `same_position_paired`：把每个深度桶的位置拿出来，与**全深度基线在同一批位置上**的 CE 配对。

## 3 否证条件（预登记）

1. 若没有任何逐 token 门控策略能同时做到「layer-passes 显著低于 L」与「序列级 bpc 不越 0.014
   噪声带」⇒ 判逐 token 早退这一手段在本尺度不成立；
2. 若门控被**固定前缀**在同算力下支配（同 layer-passes 下 bpc 更差，或同 bpc 下 layer-passes
   更高）⇒ 判「按 token 分配深度」相对「整段截断」无净增量（这是更实质的否证，比 0.014 更严）；
3. 若质量损伤可归因于填充语义伪影（no-fill 或同位置配对显示深跑位置被污染）⇒ 归属为
   异构深度 KV 语义问题（WP2）而非「按需深度」本身。

## 4 产出与复现

```bash
cd /home/z/vibe/pathlm
python3 axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run.py \
        --out axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel
```

| 文件 | 内容 |
|---|---|
| `run.py` | 驱动（全部常数在文件头） |
| `config.json` | 冻结的协议常数（供复核逐项对照） |
| `results.json` | 全部数字（臂表 / 锚点 / 校准 / risk–coverage / 诊断 / 同位置配对） |
| `run_log.txt` | 运行流水（含过程墙钟，**不进判据**） |
| `analysis.md` | **当前**结论与证据（人话摘要在前） |
| `MODEL.md` | checkpoint 清单、哈希、复现命令 |

单测（承载体）：
```bash
python3 -m pytest tests/test_axis2_exit_decision.py -q
```
