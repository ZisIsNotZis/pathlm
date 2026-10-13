# 2638 · 显式记忆状态的质量-代价前沿（信息右移的自身目的）· 13M · dense

> 契约来源：`.scratch/62-axis1-time/TICKET.md` §3 序 1 = 轴 1 最高价值 WP（研究项 `1.1.4.1` 的 WP1）。
> 动机：`55-review M-PROBLEMS` 的 **M-01 / M-03 / M-12** 指出 1.1.4.1 **从未测过自己的目的**
> ——旧格只报「加通道的臂间增量（+0.2%）」，**从未把状态放进质量-代价前沿**，因此无法回答
> 「这个预算点值不值」。本实验补这一条。
> 入口：`train_arms.sh`（训练纯窗口臂，按臂取/放 §20 GPU 锁）→ `run.py`（纯推理评测 + 前沿）；
> 预登记契约 = 本文件；配置 `config.json`；原始结果 `results.json`；判读 `analysis.md`；
> 权重清单 `MODEL.md`（权重 `*.pt` 不入库）。

## 1. 命题（一句话、可证伪）

**把显式记忆状态（carry 通道）放进「每位置等效 KV 字节 vs clean bpc」的质量-代价前沿后，
它是否相对**同代价的纯窗口**改善前沿？**

可证伪形式：*在 13M / enwik8 / T=256 / 12k 步的受控配方下，把每个臂按
（每位置等效 KV 字节, clean bpc）落在同一张图上：若显式状态点（carry16，k_eff ≈ 17.3）
的 bpc **不低于**纯窗口曲线在同一 k_eff 处的取值（在噪声带内或更高），则「显式状态在同等内存
下优于纯窗口」这一信息右移的自身目的**被否证**（该预算点不值得付）；若低于，则该状态在该预算
点上是一个（弱）前沿改善，并给出其边际量级。*

## 2. 单因素声明

- **基底**：stock dense decoder-only 全因果主干（tied 读出 + 无 final LayerNorm + 顺序残差），
  13.54M（d256 × 12 层 × 8 头 × mlp6 × T256）——即 2610 全因果主干与 2634 四臂的同一规格。
- **唯一变量**：训练期可见性（窗口 K，anchors=0）与是否带 carry 通道：
  `full`（全因果）、`win{K}`（纯窗口 K）、`carry16`（窗口 16 + delta 写通道）。
- 其余逐字段一致：数据（enwik8 前 90M 训练 / 末 10M 评测）、12k 步、bs32、lr 1e-3、
  AdamW+cosine、`--exit-mode uniform`、`--train-anchors 0`、同 token 预算 98.3M。
- **必要性例外（显式记账）**：`carry16` 与 `win16` 的差别 = 有无 carry 通道（+264,296 参数、
  单共享槽 8192 floats），这是**同一研究问题的手段**，不是第二个机制。

## 3. 干预定义

- `full`：不传注意力掩码（`is_causal=True`）⇒ stock 因果路径。
- `win{K}`：`keep[i,j] = (j ≤ i) ∧ ((i−j) < K)`（`train_anchors=0`）；训练与评测**共用**
  `pathlm.depth_ar.window_anchor_keep_mask(T, K, 0)`。
- `carry16`：窗口 16 + 显式写/读通道（`carry_rule=delta`，见 2634 §3）：
  写 `k̂=normalize(W_k h0)`, `v=W_v h0`, `β=σ(W_β h0+b_β)`，`S ← S + β(v − S k̂)k̂ᵀ`；
  读 `mem_t = S_t q_t`，`a ← a_SDPA(窗口掩码) + γ_ℓ ⊙ mem_t`。单共享槽（`shared_slots=true`）。

**本实验不改任何共享文件**（`pathlm/**`、`train_depth_ar.py`、`tools/**`）：只用既有开关
`--train-window/--train-anchors/--carry-write`，评测只调 `tools/evalkit/retention.masked_bpc`。

## 4. 判据线（预登记，写下后不得改）

| # | 判据 | 判定线 | 该涨/该跌/必须不变 |
|---|---|---|---|
| ① | **前提（口径一致）**：全部臂走同一份 96×16 窗口序（`window_seed=2`），token 数相同；窗口掩码可见对数等于声明值 | 逐臂断言通过（`selfchecks[*].ok`） | 必须不变 |
| ② | **代价（该跌）**：显式状态点的每位置等效 KV 字节 `k_eff` 远小于 full | `k_eff(carry16) ≈ 17.3` vs `k_eff(full) = 256` | 该跌 |
| ③ | **主判据（前沿定位）**：显式状态点相对**同 `k_eff` 的纯窗口曲线**的 bpc 差 `carry_minus_interp_window = bpc_interp − bpc_carry` | `< 0`（在噪声带外）⇒ 前沿改善；`≈ 0`（带内）⇒ 不可排序；`> 0` ⇒ 被同预算纯窗口支配 | 该涨/该不变 |
| ④ | **必要条件（该涨）**：carry 相对同窗无通道（`win16_s0`）的 clean bpc 增益 Δbpc（两个种子都 > 0），且 2634 的毁源探针 `Δbpc_far > 0`（状态确实搬了窗外信息） | 两个种子 Δbpc > 0；`Δbpc_far > 0`（引 2634） | 该涨 |
| ⑤ | **禁外推**：结论只适用于 enwik8 / T=256 / 13M / 12k 步快照；纯窗口曲线的线性插值只用于**定位**、不外推 | — | — |

### 4a. 操作性定义

- **clean bpc**：`masked_bpc(model, eval_arr, keep_mask, n_batches=96, batch_size=16,
  window_seed=2)`；`full` 臂 `keep_mask=None`。token 数 = 96×16×255 = 391,680。
- **每位置等效 KV 字节**：`kv_bytes(k) = n_layers × 2 × d_model × bytes_per_elem × k`
  （fp16 ⇒ `bytes_per_elem=2`），`= 12288 × k` 字节；carry 状态另加 `8192 × 2 = 16,384` 字节。
  横轴等价量 **`k_eff` = 总字节 / 12288**（等效窗口位置数）。
- **纯窗口曲线**：`{(k_eff, bpc)}` for `win16/win24/win32/win128`（seed 0）。在 carry16 的 `k_eff`
  处做**线性插值**定位（`interp_window_at`）。**凹性警告**：窗口质量曲线通常上凸（边际递减），
  故线性弦**高估**真实窗口质量 ⇒ 若 carry 低于弦，则它必然也低于（更好的）真实曲线（保守，判据不软）。
- **噪声带（双列，不自选）**：① 本格同配置换种子极差 `σ_within_config`（2634 的
  `win16_s0/s1`、`carry16_s0/s1`）；② 项目 0.014 跨 run bpc 带宽。**两者并列报出**，
  判决文字同时给两种读法（口径归因属框架性取舍，见 `.scratch/62-axis1-time/TICKET.md` §4 F1）。
- **gap-closing 比例** `r = Δc / Δg`：`Δc = bpc(win16_s0) − bpc(carry16_s0)`，
  `Δg = bpc(win16_s0) − bpc(full)`（即窗口 16 相对全因果的 clean bpc 损失）。

## 5. 臂规格

| 臂 | kind | 窗口 K / anchors | 通道 | seed | 出处 | 参数量 | 等效 KV 字节 | k_eff |
|---|---|---|---|---|---|---|---|---|
| `full` | full | 全因果 | 无 | 0 | 2610 `trunk.pt`（复用） | 13,536,768 | 3,145,728 | 256 |
| `win16_s0` | window | 16 / 0 | 无 | 0 | 2634 `rt_none_s0`（复用） | 13,536,768 | 196,608 | 16 |
| `win16_s1` | window | 16 / 0 | 无 | 1 | 2634 `rt_none_s1`（复用） | 13,536,768 | 196,608 | 16 |
| `carry16_s0` | carry | 16 / 0 | delta | 0 | 2634 `rt_delta_s0`（复用） | 13,801,064 | 212,992 | 17.33 |
| `carry16_s1` | carry | 16 / 0 | delta | 1 | 2634 `rt_delta_s1`（复用） | 13,801,064 | 212,992 | 17.33 |
| `win24_s0` | window | 24 / 0 | 无 | 0 | **本实验训练** | 13,536,768 | 294,912 | 24 |
| `win32_s0` | window | 32 / 0 | 无 | 0 | **本实验训练** | 13,536,768 | 393,216 | 32 |
| `win128_s0` | window | 128 / 0 | 无 | 0 | **本实验训练** | 13,536,768 | 1,572,864 | 128 |

同一 token 预算（12k × 32 × 256 = 98.3M）。`full` 复用 2610 主干（同尺寸、同预算、
同 seed、同数据、全因果；配置等值性见 2615/2616 的逐字段对照，本实验只重跑评测）。

## 6. 成本（§8 的唯一预算数字 + 砍什么）

`gpu_wall_budget: 40 min`（含 3 臂训练 + 评测 + 对账；步时实测 ≈44 ms/step ⇒ 12k ≈ 8.8 min/臂）。
**步时实测**：`smoke/steps300/win24/results.json` 的 `wall_minutes=0.22`（300 步）。

**预算不够时砍什么**：**三臂一起**砍步数（12k → 6k，保持臂间同 token 预算，并在报告首段声明
「6k 快照口径、与 2634 的 12k 不可比」）> 砍 `win128`（宽窗点）> 砍评测批数；
**不砍判据线、不砍 win16/carry16 复用臂、不砍 full 锚**。

## 7. 数据与脚本清单

- 数据：`data/enwik8_full.npz`（前 90M 训练 / 末 10M 评测；vocab 206）。
- 训练：`bash train_arms.sh`（`ARMS="24 32 128"`）⇒ `train/win{24,32,128}/{model.pt,results.json,train_log.jsonl}`。
- 评测：`python3 run.py`（8 臂 bpc + 前沿 + 定位）⇒ `results.json`。
- 落盘：`config.json`、`run.py`、`train_arms.sh`、`results.json`、`analysis.md`、`MODEL.md`。

## 8. 既有做法（prior art，SCHEMA §18）

1. **谁做过**：状态空间/线性注意力（S4/Mamba/RWKV/GLA/DeltaNet），混合架构（Jamba/Samba），
   显式记忆（Compressive Transformer、Memorizing Transformer）；长上下文评测（needle、MQAR）。
   `记忆·未核`：本环境**无联网检索**，未查原文链接。
2. **为什么这么设计**：局部注意力的信息在窗口外即丢失；递推状态把「过去」压进一个与长度
   无关的载体，从而在不扩大窗口的前提下取回远距离信息。
3. **它的实测结果（记忆·未核）**：线性/递推状态在记忆受限任务上逊于同尺寸注意力；在**小模型 +
   强局部先验**的真实文本上，长程机制常常量不出收益（与本仓 SCHEMA §19 一致）。
4. **我们的增量**：不是提出机制，而是**把「显式状态在这一预算点上值不值」做成可判定实验**——
   统一口径的 (每位置字节, bpc) 前沿 + 同配方的纯窗口曲线 + 100% 全因果锚，直接回答出资人
   「哪种设计更值」。我们的设定与业界的实质差异：13M、单卡、12k 步快照、真实文本、单共享槽。

## 9. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：口径不一致（token 数/窗口序不同）⇒ 先修仪器再判。
2. **实现**：carry 通道活性为 0（引 2634 的 `carry_liveness`：`read_share`/β/状态范数任一为 0）⇒ 先排除死开关。
3. **规模**：carry16 相对 win16 的 Δbpc 落在噪声带内 ⇒ 只读作「该尺度不划算」，**不得**写成
   「机制不可能」（SCHEMA §19）；下一步 = 更大步数/规模、逐层状态、窗口档位扫描（列成待测项与成本）。
4. **设计**：carry16 的 bpc 更好但 `k_eff` 代价更大 ⇒ 用边际效率（Δbpc / Δ字节）判断，而非只看 bpc。

## 10. scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 全因果；**组合默认留白**。`deps` 只记录耦合方向。
本实验不训练 carry（复用 2634 的 delta 臂）；不测逐层状态/门控（那是 1.1.4.1 的 WP2）。

## 11. 产出与复核（SCHEMA §14/§14b）

见 §7；完成后另派 fresh-context 只读复核者，**复核排在全部产物落盘之后**。
