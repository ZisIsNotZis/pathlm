# MODEL.md · 2627 G5 机制定论抖动补点（权重指纹 / 复现命令 / 共享面）

> 权重（`*.pt`）**不入库**。本文件记录 sha256/体积以支持复现核对；`results_13m.json::checkpoints`
> 与 `config.json::arms` 由 `fill_config_hashes.py` 落盘，`run.py` 每次加载都重算 sha256 并断言。

## 0. 一句话

`l1_A` / `repr01_A`（2626 复用）与 `repr01_B`（本目录新训）是**同一骨干**（13,076,483 参数，逐位相同）；
`repr01_B` 是 `repr01` 臂**同配置、同 24000 步、同 seed=0** 的**第二次**训练（run-to-run 抖动补点）。
配置 `configs13m/repr01.json` 与 2626 逐字节相同；去相关是**纯损失项**（0 参数增量）。

## 1. 权重指纹

| 臂 | 来源 | 相对路径 | 体积 (B) | sha256 |
|---|---|---|---|---|
| 13m/l1_A | 2626 | `../2626-mechanism-convergence/train13m/l1/model.pt` | 52349379 | `a74e86a2c5f40375900c…` |
| 13m/repr01_A | 2626 | `../2626-mechanism-convergence/train13m/repr01/model.pt` | 52349379 | `488732cc65fa99d911cb…` |
| 13m/repr01_B | **2627（本轮）** | `train13m/repr01/model.pt` | 52349379 | `5e81e193bf42b45f8409…` |

完整 sha256 / 体积见 `config.json::arms` 与 `results_13m.json::checkpoints`（`run.py` 落盘时断言一致）。

- **参数量**：三臂均为 **13,076,483**（构造性相同：去相关压力是纯损失项，不加参数）。
- **`repr01_B` ≠ `repr01_A`**（sha256 不同）⇒ 确实是**两次训练**，不是复用。
- **中间检查点**：`.tmp/2627-ckpts/13m/repr01/model_step{006000,012000,018000,024000}.pt`（不入库）。
- **训练规模**：`bs32 × 24000 步 × T512` = **393,216,000 tokens**；`seed=0`；墙钟 **27.4 min**；
  峰值显存 **4995.8 MB**（<5GB net）。

## 2. 复现命令

```bash
# 0) 测试（CPU）
python3 -m pytest tests/test_mechanism_jitter.py -q

# 1) 变异验证（配置类写-读-还原 + 语义接缝运行期猴补丁；不改文件）
python3 axes/G-policy/G5-parallel-lines/experiments/2627-mechanism-jitter/mutate.py

# 2) 单臂训练（取/放 GPU 锁；~28 min）
nohup bash axes/G-policy/G5-parallel-lines/experiments/2627-mechanism-jitter/train_all.sh train 13m \
    > train13m_driver.log 2>&1 &

# 3) 评测 + 对账（纯推理 >2min ⇒ 取锁）
nohup bash axes/G-policy/G5-parallel-lines/experiments/2627-mechanism-jitter/train_all.sh eval 13m \
    > eval13m.log 2>&1 &
```

## 3. 代码指纹（`results_13m.json::code`）

- `run.py` = `2af84dddec5b4b04b39aa6ea8c5472cbc637ca64ee1e9e529618a6faf447979f`
- `jitter.py` = `aa753f44305ca0e3ff8e52fbd0c047b6b2bac01759e9fdda5a3a1b34e4d1631a`
- `train_m1.py` = `a253fcaf61fb39e0042bf1086174c32f47b7c78580c3a48fc1c42a9bb3ae5459`
- `pathlm/model.py` = `4dda7c08c4b5d564c5f066a0e8c2a52950f88498a4959ccdf68762430ab8d77f`
- `pathlm/config.py` = `7c6c0fbf16fb115d5580ad2ade42917b32ad9ba2ec810d74e097ad4f934cdb4d`
- `pathlm/eval.py` = `05b6f96663c9194a29c05405dc6b8bbdfcb515e19883b71b5e79e57f33530baf`
- `tools/evalkit/ensemble_probe.py` = `a96324adf8e6049d8153db1b4c918ad1c479889eb6d6dbc5c4bd93ca8764b2c9`
- `probe_tts.py` = `ea36e198e200f134913416b21bd700d69d4f184fca4a8efca101f7dc87b81da8`
- `README.md`（本契约）= `4377a7e06daecf76ef114931845610b88e6b8284a8756e272c85e143c024af85`

> `train_m1.py` / `pathlm/model.py` / `pathlm/config.py` / `pathlm/eval.py` / `ensemble_probe.py` /
> `probe_tts.py` 的 sha256 与 2626（V2 `MODEL.md`）**逐位相同** ⇒ 本实验复用的是**同一套未改动的引擎**，
> 单因素只在「多跑一遍」。

## 4. 共享文件与并发（`SCHEMA §22`）

| 文件 | 本线改动 |
|---|---|
| `train_m1.py` / `pathlm/model.py` / `pathlm/config.py` | **无**（只读复用 2626 实现） |
| `tests/test_mechanism_jitter.py` | 新增（本线独有） |
| 本实验目录（configs/run.py/jitter.py/reconcile/mutate/文档） | 新增（本线独有） |

- **邻线标记回读**（共享面断言）：`grep -c "carry_state_dim\|recon_window\|recon_head"` 在
  `train_m1.py` / `pathlm/model.py` 上 = 0；`grep -c "w_repr_decorr" pathlm/depth_ar.py train_depth_ar.py` = 0。
- **未**使用 `git checkout/stash/reset`；变异脚本 `mutate.py` 的配置类变异写-读-还原并校验 sha256
  逐字节相同，语义接缝用运行期猴子补丁（**不改任何文件**）。
