# MODEL.md · 2628 G5 机制定论抖动**补全**（l1 臂）（权重指纹 / 复现命令 / 共享面）

> 权重（`*.pt`）**不入库**。本文件记录 sha256/体积以支持复现核对；`results_13m.json::checkpoints`
> 与 `config.json::arms` 由 `fill_config_hashes.py` 落盘，`run.py` 每次加载都重算 sha256 并断言。

## 0. 一句话

`l1_A` / `repr01_A` / `repr01_B`（2626/2627 复用）与 `l1_B`（本目录新训）是**同一骨干**
（13,076,483 参数，逐位相同）；`l1_B` 是 **`l1` 臂同配置、同 24000 步、同 seed=0**
的**第二次**训练（run-to-run 抖动补全，补上 2627 只重跑 `repr01` 的下界口径）。
配置 `configs13m/l1.json` 与 2626 逐字节相同；`l1` 臂无去相关（纯路径机制对照，0 参数增量）。

## 1. 权重指纹

| 臂 | 来源 | 相对路径 | 体积 (B) | sha256 |
|---|---|---|---|---|
| 13m/l1_A | 2626 | `../2626-mechanism-convergence/train13m/l1/model.pt` | 52349379 | `a74e86a2c5f40375900c…` |
| 13m/l1_B | **2628（本轮）** | `train13m/l1/model.pt` | 52349379 | `（本轮新训，见 config.json::arms）` |
| 13m/repr01_A | 2626 | `../2626-mechanism-convergence/train13m/repr01/model.pt` | 52349379 | `488732cc65fa99d911cb…` |
| 13m/repr01_B | 2627 | `../2627-mechanism-jitter/train13m/repr01/model.pt` | 52349379 | `5e81e193bf42b45f8409…` |

完整 sha256 / 体积见 `config.json::arms` 与 `results_13m.json::checkpoints`（`run.py` 落盘时断言一致）。

- **参数量**：四臂均为 **13,076,483**（构造性相同：去相关是纯损失项，不加参数）。
- **`l1_B` ≠ `l1_A`**（sha256 不同）⇒ 确实是**两次训练**，不是复用。
- **中间检查点**：`.tmp/2628-ckpts/13m/l1/model_step{006000,012000,018000,024000}.pt`（不入库）。
- **训练规模**：`bs32 × 24000 步 × T512` = **393,216,000 tokens**；`seed=0`；墙钟 **16.8 min**；
  峰值显存 **2616.2 MB**（<5GB net）。

## 2. 复现命令

```bash
# 0) 测试（CPU）
python3 -m pytest tests/test_jitter_l1.py -q

# 1) 变异验证（配置类写-读-还原 + 语义接缝运行期猴补丁；不改文件）
python3 axes/G-policy/G5-parallel-lines/experiments/2628-jitter-l1/mutate.py

# 2) 单臂训练（取/放 GPU 锁；~17–20 min）
nohup bash axes/G-policy/G5-parallel-lines/experiments/2628-jitter-l1/train_all.sh train 13m \
    > train13m_driver.log 2>&1 &

# 3) 评测 + 对账（纯推理 >2min ⇒ 取锁）
nohup bash axes/G-policy/G5-parallel-lines/experiments/2628-jitter-l1/train_all.sh eval 13m \
    > eval13m.log 2>&1 &
```

## 3. 代码指纹（`results_13m.json::code`）

- `run.py` / `jitter_l1.py`：本目录独有，sha256 见 `results_13m.json::code`。
- `train_m1.py` / `pathlm/model.py` / `pathlm/config.py` / `pathlm/eval.py` / `ensemble_probe.py` /
  `probe_tts.py` 的 sha256 与 2626/2627 **逐位相同** ⇒ 本实验复用的是**同一套未改动的引擎**，
  单因素只在「多跑一遍 `l1`」。

## 4. 共享文件与并发（`SCHEMA §22`）

| 文件 | 本线改动 |
|---|---|
| `train_m1.py` / `pathlm/model.py` / `pathlm/config.py` | **无**（只读复用 2626/2627 实现） |
| `tests/test_jitter_l1.py` | 新增（本线独有） |
| 本实验目录（configs/run.py/jitter_l1.py/reconcile/mutate/文档） | 新增（本线独有） |

- **邻线标记回读**（共享面断言）：`grep -c "w_repr_decorr" pathlm/depth_ar.py train_depth_ar.py` = 0。
- **未**使用 `git checkout/stash/reset`；变异脚本 `mutate.py` 的配置类变异写-读-还原并校验 sha256
  逐字节相同，语义接缝用运行期猴子补丁（**不改任何文件**）。
