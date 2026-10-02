# MODEL — 2620 · 容量轴在「收敛后」是否有效（权重清单）

> 权重（`*.pt`）**不入库**（`.gitignore` 覆盖）。本文件是哈希与生成命令的唯一来源。
> 评测入口 `run.py` 加载权重前**逐次断言 sha256**（不符即 `SystemExit`）。

## 1. 三臂（唯一变量 = `carry_state_dim`；`S16_rep` 与 `S16` 逐字段相同）

| 臂 | `carry_state_dim` | 记忆槽头数 | 状态浮点/样本 | 步数 | 参数量 | sha256 | 体积 |
|---|---|---|---|---|---|---|---|
| `S16` | 16 | 16 | 4,096 | 24,000 | 13,803,216 | 见 `config.json::arms.S16.sha256` | 见 `size_bytes` |
| `S128` | 128 | 2 | 32,768 | 24,000 | 13,799,450 | 见 `config.json::arms.S128.sha256` | 见 `size_bytes` |
| `S16_rep` | 16 | 16 | 4,096 | 24,000 | 13,803,216 | 见 `config.json::arms.S16_rep.sha256` | 见 `size_bytes` |

（`fill_config_hashes.py` 在训练后回填 `config.json`，`run.py` 加载时逐臂断言；
逐臂墙钟/显存/参数量见 `results.json::train_results` 与 `results.json::cost`。）

## 2. 生成命令（逐字；三臂只差 run 名与 `--carry-state-dim`）

```bash
python3 train_depth_ar.py <S16|S128|S16_rep> \
  --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
  --steps 24000 --batch-size 32 --seed 0 --lr 0.001 \
  --train-window 16 --train-anchors 0 \
  --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
  --carry-write delta --carry-state-dim <16|128|16> --out-root train
```

（实跑由 `train_all.sh` 驱动，持 `/tmp/pathlm_gpu.lock`，SCHEMA §20 修正版协议；
20 步冒烟走 `./train_all.sh --smoke`。）

## 3. 评测

```bash
./eval_run.sh            # 全电池（本目录；持 GPU 锁）
python3 run.py --smoke   # 冒烟（需先有 train/ 下的权重）
```

## 4. 共享文件指纹（SCHEMA §22.6，如实记录重叠归属）

本实验**不改任何共享文件**（契约明文：用既有 `--carry-state-dim`）。下表是**训练启动时刻**
的只读快照；运行期指纹另见 `results.json::code`，两者若不同以 `results.json::code` 为准。

```
pathlm/depth_ar.py               248a92754154d65e0857ca5ceccda91ef63ffd9b90b3cfde2199da29b6b59135
train_depth_ar.py                8c333e5405baf23812cd8d389e69063a3b0129fa17993db7fe1ddb50e71828a0
tests/test_depth_ar.py           433d4ef4dc48de4c40d70a068a1c916004cd35fc38aad11be377af7fd22bdead
tools/evalkit/carry_channel.py   4dcd91facd60a512dcc983836fb29a7c91aad988bada1c95ada666824d226753
tools/evalkit/longrange_tasks.py 4a02b93f19db8e99dbf3a555b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b
tools/evalkit/retention.py       8320f03a2ecf74e32ab29b0f0af0ef610d7cf45116ae5dcdcdeb241c645e6502
```

（以上 6 个共享文件在本实验全程**只读**；`git diff --stat` 对它们为空 ⇒ 与 `HEAD`（`68483da`）
逐字节一致，不存在「含邻线未提交改动」的口径重叠。运行期指纹另见 `results.json::code`。）

### 权重 sha256（三臂）

| 臂 | sha256 | 体积 |
|---|---|---|
| `S16` | `10b79c5b5770cbd6` … （完整值见 `config.json::arms.S16.sha256`） | 55,263,811 B |
| `S128` | `dafc85c2e66af2a2` … | 55,249,475 B |
| `S16_rep` | `c0dee9b4254aeb79` … | 55,263,811 B |

（`run.py` 加载时逐臂断言完整 sha256；`config.json` 由 `fill_config_hashes.py` 回填。）

**口径重叠声明**：本轮同批还有 V1（`R1-quantization/2620-kv-budget-gaps`）与
V3（`G5-parallel-lines/2620-decorrelation-at-scale`）两条 GPU 线并行，V3 **改**
`train_m1.py` / `pathlm/model.py`（loop-era 引擎），与本实验的 `depth_ar` 面**不相交**；
本实验全程未对任何共享文件写入，也未执行 `git checkout/stash/reset`。

## 5. 权重来源与口径

- 训练数据**全是合成**（`tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch`），
  **不含 enwik8**、不含任何外部语料 ⇒ 无数据授权/隐私问题。
- 权重只用于本目录的评测（`run.py`），不进入任何发布产物。
