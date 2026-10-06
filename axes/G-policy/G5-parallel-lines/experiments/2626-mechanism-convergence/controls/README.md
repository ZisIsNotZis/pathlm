# controls · 对照臂与方差界（2626 · G5 机制收敛收尾）

本目录只放**对照的定义与来源**；对照臂本身的**读数**在 `../results_13m.json`（由 `../run.py` 落盘），
不另存副本（避免两处数字漂移）。

## 1. 确定性路径对照（预登记「应恰好 0」）

定义：把路径旋钮全部关掉（`p_skip=p_redo=shuffle_locality=0`）后，K 条「路径」是**同一路径**的
K 次复制 ⇒ 集成 = 单线。两个读点：

| 读点 | 怎么做的 | 在 `results_13m.json` 里的位置 |
|---|---|---|
| `stock` 臂 | 该臂配置本身 `p_skip=0`，K 次前向天然是同一路径 | `deterministic_path_control.stock` |
| `repr01_forced` | 在 `repr01` 检查点上**评测期**用 `deterministic_paths()` 强制关路径随机 | `deterministic_path_control.repr01_forced` |

判定口径（两个都报，缺一不可）：
- **精确判据**：K 条路径的 logits 是否 `torch.equal` 逐位相同（`paths_bitwise_identical_all`）；
- **实测差**：`gain_K1_minus_Kmax`（浮点求和顺序导致的 ~1e-6 级差异，不是机制的增益）。

实测：两读点 `gain_K1_minus_Kmax` 都 = **0.0000**，logits **逐位相同**。

## 2. X1（2622）13M 参照（**不重跑**，只作同方法趋势——但**调度不同**）

来源：`../2622-decorrelation-v2/results_13m.json`。口径：`probe_tts.bpc_k`，60 批 × bs32 × T512 × seed 11。

| 臂（X1 13M，**6000 步 cosine 末点**） | K1 | K8 | 成对 JS (bits) |
|---|---|---|---|
| `stock` | 1.5156 | 1.5156 | 0.0000 |
| `l1` | 1.6827 | 1.6454 | 0.0249 |
| `repr01`（λ=0.1） | 1.6785 | 1.6208 | 0.0361 |

⚠️ **口径陷阱（本实验关键）**：X1 是 **6000 步 cosine 的末点**（lr→0.1×base），本实验是 **24000 步
cosine**（其 @6k 处 lr≈0.868×base）⇒ 两者**学习率调度不同**，**不可逐位/直接对照**；本表只作方向性登记。
本实验的四个点（6k/12k/18k/24k）都在**同一条 24k 调度**内，是可比的收敛轨迹。

## 3. 方差界

`docs/methodology.md`：**Δ < 0.014 不可排序**。本实验全部比较用 60 批**配对** bootstrap CI。
本实验机制净效应 24k 终点 **+0.0144** 仅比该门槛高 0.0004 ⇒ 按「贴门槛」表述（见 `../analysis.md §4.2`）。
