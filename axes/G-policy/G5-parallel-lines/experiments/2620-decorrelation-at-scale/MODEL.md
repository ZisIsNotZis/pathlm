# MODEL.md · 权重清单（**权重本身不入库**）

契约：`README.md`。三臂权重与数据集都在 `.gitignore` 覆盖范围内（`*.pt` / `data/`），
本文件记录**哈希 / 体积 / 生成命令**，供复核者核对「读数确实来自这三个检查点」。

## 生成命令（训练）

```bash
# 每臂 12000 步 × bs16 × T512 = 98.3M tokens（与 13M 锚点同 token）
python3 train_m1.py stock  --config axes/G-policy/G5-parallel-lines/experiments/2620-decorrelation-at-scale/configs/stock.json  --steps 12000 --ckpt-every 6000 --ckpt-dir .tmp/v3-ckpts/stock  --out-root <EXP>/train
python3 train_m1.py l1     --config .../configs/l1.json     --steps 12000 --ckpt-every 6000 --ckpt-dir .tmp/v3-ckpts/l1     --out-root <EXP>/train
python3 train_m1.py divl1  --config .../configs/divl1.json  --steps 12000 --ckpt-every 6000 --ckpt-dir .tmp/v3-ckpts/divl1  --out-root <EXP>/train
```

（`<EXP>` = 本目录。总入口 `v3_all.sh` 一次跑完冒烟+三臂+评测+对账。）

## 权重（训练后由 `fill_config_hashes.py` 回填，`run.py` 每次评测前断言）

| 臂 | 文件 | sha256 | 体积 |
|---|---|---|---|
| `stock` | `train/stock/model.pt` | `TO_FILL_AFTER_TRAINING` | — |
| `l1` | `train/l1/model.pt` | `TO_FILL_AFTER_TRAINING` | — |
| `divl1` | `train/divl1/model.pt` | `TO_FILL_AFTER_TRAINING` | — |

- 三臂参数量相同：`TO_FILL`（训练后与 `results.json::checkpoints.*.params` 对齐）。
- 中间检查点（收敛门禁用）：`.tmp/v3-ckpts/<arm>/model_step006000.pt`（不入库）。
- 数据集：`data/enwik8_full.npz`（由 `.tmp/enwik8` 生成；val = 后 10M 字符，训练从不碰）。

## 共享文件与并发披露（`SCHEMA §22`）

本线改动的共享面**只有** `train_m1.py`（loop-era 引擎的训练入口：+ 活性量落盘、+ `intervention`
块、+ `peak_gpu_mem_mb`）；`pathlm/model.py` **未改**（去相关损失与 grad-norm gate 已存在，
属复用）。新逻辑进新模块 `tools/evalkit/ensemble_probe.py`。

```bash
# 改完后的邻线标记回读（邻线 = V1 kv-budget / V2 capacity-converged）
$ grep -c "carry_state_dim" pathlm/depth_ar.py train_depth_ar.py
pathlm/depth_ar.py:17
train_depth_ar.py:3
$ grep -c "carry_state_dim\|recon_window" train_m1.py pathlm/model.py
0 / 0        # 本线文件里没有邻线标记（面不相交）
```

- `train_m1.py` 的 sha256 记在 `results.json::code`（本线的改动**只**含 V3 的活性量/`intervention`
  /显存三处；工作树里同时存在 V1/V2 对**其他**文件的未提交改动——口径重叠写在这里，不假装独占）。
- 全程**未**使用 `git checkout/stash/reset`；变异验证（`mutate.py`）用运行期猴子补丁，
  **不改任何文件**（逐字节还原是构造性的）。
