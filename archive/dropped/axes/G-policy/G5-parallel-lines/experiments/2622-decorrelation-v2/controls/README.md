# controls · 对照臂与方差界（2622 · 去相关 v2）

本目录只放**对照的定义与来源**；对照臂本身的**读数**在 `../results_{13m,50m}.json`（由
`../run.py` 落盘），不另存副本（避免两处数字漂移）。

## 1. 确定性路径对照（预登记「应恰好 0」）

定义：把路径旋钮全部关掉（`p_skip=p_redo=shuffle_locality=0`）后，K 条「路径」是**同一路径**
的 K 次复制 ⇒ 集成 = 单线。两个读点：

| 读点 | 怎么做的 | 在 `results_{scale}.json` 里的位置 |
|---|---|---|
| `stock` 臂 | 该臂配置本身 `p_skip=0`，K 次前向天然是同一路径 | `deterministic_path_control.stock` |
| `repr_forced` | 在 `repr` 检查点上**评测期**用 `deterministic_paths()` 强制关路径随机 | `deterministic_path_control.repr_forced` |

判定口径（两个都报，缺一不可）：
- **精确判据**：K 条路径的 logits 是否 `torch.equal` 逐位相同（`paths_bitwise_identical_all`）；
- **实测差**：`gain_K1_minus_Kmax`（浮点求和顺序导致的 ~1e-6 级差异，不是机制的增益）。

## 2. 13M 锚点（**不重跑**，只作同方法趋势）

来源：`.scratch/08-tts-probe/evidence/tts_results.json`、`.scratch/09-m3-interactions/evidence/DIVL1/k_sweep.json`。
口径：`probe_tts.bpc_k`，60 批 × bs32 × T512 × seed 11。

| 臂（13M 旧） | K1 | K2 | K4 | K8 | 集成增益 (K1−K8) |
|---|---|---|---|---|---|
| `L1`（路径机制） | 1.6926 | 1.6657 | 1.6543 | 1.6454 | 0.0472 |
| `DIVL1`（capped-−JS 去相关） | 1.6842 | 1.6482 | 1.6339 | 1.6227 | 0.0615 |
| `B0`（stock 单线，200-batch battery 口径） | 1.5074 | — | — | — | — |

⚠️ **口径陷阱**：13M 的 `DIVL1/L1` 用 60-batch probe 协议，`B0` 用 200-batch battery 协议；
本实验**两套协议都跑**，不跨协议相减。

## 3. 50M 参照（ticket 37 / 2620，只作趋势）

| 臂（50M 旧） | K1 | K8 | 成对 JS (bits) | gate_skips |
|---|---|---|---|---|
| `stock` | 1.4505 | 1.4505 | 0.0000 | 0 |
| `l1` | 1.5266 | 1.4727 | 0.0333 | 0 |
| `divl1`（capped-−JS） | 1.6005 | 1.5620 | 0.0250 | **5121**（42.7%）|

来源：`../2620-decorrelation-at-scale/results.json`。**本轮 50M 复用其 `stock` 检查点**。

## 4. 方差界

`docs/methodology.md`：**Δ < 0.014 不可排序**。本实验全部比较用 60 批**配对** bootstrap CI。
