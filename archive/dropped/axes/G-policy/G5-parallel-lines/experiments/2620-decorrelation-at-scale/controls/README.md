# controls · 对照臂与方差界（2620 · 契约 V3）

本目录只放**对照的定义与来源**；对照臂本身的**读数**在 `../results.json`（由 `../run.py` 落盘），
不另存副本（避免两处数字漂移）。

## 1. 确定性路径对照（预登记「应恰好 0」）

定义：把路径旋钮全部关掉（`p_skip=p_redo=shuffle_locality=0`）后，K 条「路径」是**同一路径**
的 K 次复制 ⇒ 集成 = 单线。两个读点：

| 读点 | 怎么做的 | 在 `results.json` 里的位置 |
|---|---|---|
| `stock` 臂 | 该臂配置本身 `p_skip=0`，K 次前向天然是同一路径 | `deterministic_path_control.stock` |
| `divl1_forced` | 在 divl1 检查点上**评测期**用 `deterministic_paths()` 强制关路径随机 | `deterministic_path_control.divl1_forced` |

判定口径（两个都报，缺一不可）：
- **精确判据**：K 条路径的 logits 是否 `torch.equal` 逐位相同（`paths_bitwise_identical_all`）；
- **实测差**：`gain_K1_minus_Kmax`（浮点求和顺序导致的 ~1e-6 级差异，不是机制的增益）。

## 2. 13M 锚点（**不重跑**，只作同方法趋势）

来源：`.scratch/08-tts-probe/evidence/tts_results.json`、`.scratch/09-m3-interactions/evidence/DIVL1/k_sweep.json`。
口径：`probe_tts.bpc_k`，60 批 × bs32 × T512 × seed 11（与本次 50M 评测**同一函数、同一 seed**）。

| 臂（13M） | K1 | K2 | K4 | K8 | 集成增益 (K1−K8) |
|---|---|---|---|---|---|
| `L1`（= 本实验 `l1` 的 13M 版） | 1.6926 | 1.6657 | 1.6543 | 1.6454 | 0.0472 |
| `DIVL1`（= 本实验 `divl1` 的 13M 版） | 1.6842 | 1.6482 | 1.6339 | 1.6227 | 0.0615 |
| `B0`（stock 单线，200-batch battery 口径） | 1.5074 | — | — | — | — |

⚠️ **口径陷阱**：13M 的 `DIVL1/L1` 用 60-batch probe 协议，`B0` 用 200-batch battery 协议；
两者之间**有 ~+0.01 的采样偏移**（见 `.scratch/08-tts-probe/evidence/SUMMARY.md` 开头）。
本实验**两套协议都跑**，就是为了不跨协议相减。

## 3. 近似同算力参照（**不是**同算力实测）

`.scratch/11-retry-matrix-validity/evidence/README.md`（D5 final）：13M 单线 `B0 = 1.5074 @1.1 epoch`
→ `1.3459 @4.4 epoch`，即 **≈4× 训练 token 换 −0.1615 bpc**。它只用于给「多线要成为部署优势
还得跨过什么门槛」一个量级判断——货币是**训练**算力，不是推理算力，故只能标近似。
