# MODEL.md · 2626 G5 机制收敛收尾（权重指纹 / 复现命令 / 共享面）

> 权重（`*.pt`）**不入库**。本文件记录 sha256/体积以支持复现核对；`results_13m.json::checkpoints`
> 与 `config.json::arms` 由 `fill_config_hashes.py` 落盘，`run.py` 每次加载都重算 sha256 并断言。

## 0. 一句话

`stock` / `l1` / `repr01` 是**同一骨干**（13,076,483 参数，逐位相同）；X1 的 `p_skip` 路径机制与
`w_repr_decorr` 表征级去相关**原样复用**，本实验只把它们**从 6000 步跑到 24000 步**（4× token）。
去相关是**纯损失项**（0 参数增量），`l1`/`repr01` 每步多采一条并行路径。

## 1. 权重指纹

| 臂 | 相对路径 | 体积 (B) | sha256 |
|---|---|---|---|
| 13m/stock | `train13m/stock/model.pt` | 52349379 | `a31bcc51f207e480a0d0…` |
| 13m/l1 | `train13m/l1/model.pt` | 52349379 | `a74e86a2c5f40375900c…` |
| 13m/repr01 (λ=0.1) | `train13m/repr01/model.pt` | 52349379 | `488732cc65fa99d911cb…` |

完整 sha256 / 体积见 `config.json::arms` 与 `results_13m.json::checkpoints`（`run.py` 落盘时断言一致）。

- **参数量**：三臂均为 **13,076,483**（构造性相同：去相关压力是纯损失项，不加参数）。
- **中间检查点**：`.tmp/2626-ckpts/13m/<arm>/model_step{006000,012000,018000,024000}.pt`（不入库）。
- **训练规模**：每臂 `bs32 × 24000 步 × T512` = **393,216,000 tokens**（4× X1 的 98.3M）；`seed=0`。

## 2. 复现命令

```bash
# 0) 测试（CPU）
python3 -m pytest tests/test_mechanism_convergence.py -q

# 1) 变异验证（配置类写-读-还原 + 语义接缝运行期猴补丁；不改文件）
python3 axes/G-policy/G5-parallel-lines/experiments/2626-mechanism-convergence/mutate.py

# 2) 三臂训练（按臂取/放锁、臂间 ≥60s；单臂 ~19–29 min）
nohup bash axes/G-policy/G5-parallel-lines/experiments/2626-mechanism-convergence/train_all.sh train 13m \
    > train13m_driver.log 2>&1 &

# 3) 评测 + 对账（纯推理 >2min ⇒ 取锁）
nohup bash axes/G-policy/G5-parallel-lines/experiments/2626-mechanism-convergence/train_all.sh eval 13m \
    > eval13m.log 2>&1 &
```

## 3. 代码指纹（`results_13m.json::code`）

- `run.py` = `7267ab067f7545566bde2af3408da208d56e736e074830ac3a472c322c375d79`
- `train_m1.py` = `a253fcaf61fb39e0042bf1086174c32f47b7c78580c3a48fc1c42a9bb3ae5459`
- `pathlm/model.py` = `4dda7c08c4b5d564c5f066a0e8c2a52950f88498a4959ccdf68762430ab8d77f`
- `pathlm/config.py` = `7c6c0fbf16fb115d5580ad2ade42917b32ad9ba2ec810d74e097ad4f934cdb4d`
- `pathlm/eval.py` = `05b6f96663c9194a29c05405dc6b8bbdfcb515e19883b71b5e79e57f33530baf`
- `tools/evalkit/ensemble_probe.py` = `a96324adf8e6049d8153db1b4c918ad1c479889eb6d6dbc5c4bd93ca8764b2c9`
- `probe_tts.py` = `ea36e198e200f134913416b21bd700d69d4f184fca4a8efca101f7dc87b81da8`
- `reconcile_numbers.py` = `fc3c2d34b1e5685a813ecf8e48ccdeae94fb47457156e165c80343856b25a95d`
- `README.md`（本契约）= `83a5eef453a6b5a84cb944a1569df26adeff003a2a51a807e552e987ec8b6ee5`

> `train_m1.py` / `pathlm/model.py` / `pathlm/config.py` 的 sha256 与 X1（2622 `MODEL.md`）**逐位相同**
> ⇒ 本实验复用的是**同一套未改动的引擎**，单因素只在配置层。

## 4. 共享文件与并发（`SCHEMA §22`）

| 文件 | 本线改动 |
|---|---|
| `train_m1.py` / `pathlm/model.py` / `pathlm/config.py` | **无**（只读复用 X1 实现） |
| `tests/test_mechanism_convergence.py` | 新增（本线独有） |
| 本实验目录（configs/run.py/reconcile/mutate/文档） | 新增（本线独有） |

- **邻线标记回读**（共享面断言，见 `analysis.md §11`）：`grep -c "carry_state_dim\|recon_window\|recon_head"`
  在 `train_m1.py` / `pathlm/model.py` 上 = 0；`grep -c "w_repr_decorr" pathlm/depth_ar.py train_depth_ar.py` = 0。
- **未**使用 `git checkout/stash/reset`；变异脚本 `mutate.py` 的配置类变异写-读-还原并校验 sha256
  逐字节相同，语义接缝用运行期猴子补丁（**不改任何文件**）。
