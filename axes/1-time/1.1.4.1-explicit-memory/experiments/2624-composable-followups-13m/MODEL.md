# 2624 · 权重与代码指纹（权重不入库）

> 契约：`README.md`；判读：`analysis.md`；原始结果：`results.json`。
> **权重（`*.pt`）与 `data/` 不入库**（`.gitignore` 覆盖）；本文件只登记哈希、体积与生成命令。

## 1. 权重清单

| 臂 | 文件 | 参数量 | 体积（字节） | sha256 | 来源 |
|---|---|---|---|---|---|
| `mha_mlp5` | `train/mha_mlp5/model.pt` | 11,960,832 | 47,889,931 | `f5290f7be4863ecdf9a3e705363d407d50a743d5277c5e6fdf2789ed64c89aa9` | **本线自训**（6k） |
| `gqa_10k` | `train/gqa_10k/model.pt` | 12,352,512 | 49,456,651 | `8f5850fc10bf70b227a2ada89e5aa50b1a280ea26a277c583ddd0988aef0f6a3` | **本线自训**（10k） |
| `gqa_carry_10k` | `train/gqa_carry_10k/model.pt` | 12,616,808 | 50,518,595 | `5ab683a5f087feb6ff492feb8de6d06d93464922afa60e8ec2b76024cae13658` | **本线自训**（10k） |
| `stock` | `../2623-composable-arms-13m/train/stock/model.pt` | 13,536,768 | — | `72ed2438e99c41ed5c139ee31b7cd9cda13ce6a440524418293c3a9162f2d54d` | 复用 2623（SCHEMA §9） |
| `gqa` | `../2623-composable-arms-13m/train/gqa/model.pt` | 12,352,512 | — | `411f962f0b0acc8f40b7d8e5927a787600141f442610f61d43dfa8d189303e2a` | 复用 2623 |
| `gqa_carry` | `../2623-composable-arms-13m/train/gqa_carry/model.pt` | 12,616,808 | — | `5ec1e0c5df6ffe0c371996bbe0c5d02e6eb7bd57385355321c666359c8e26c1c` | 复用 2623 |
| `gqa_carry_window` | `../2623-composable-arms-13m/train/gqa_carry_window/model.pt` | 12,616,808 | — | `48c4e39ec058b8c509de88744e32f4169f993eb4f2680e6b4080b1b410f9c8a6` | 复用 2623 |

- 四臂复用：`config.json::checkpoint.sha256` 的四个值与
  `2623-composable-arms-13m/config.json` **逐位相同**；`run.py` 加载时逐臂断言 sha256
  （不符即 `SystemExit`）。**口径标注**：历史同配置对照 + 本线「仪器一致性」对账
  （重评结果与 2623 的 `results.json` 逐位相同，见 `reconcile_report.json` 的 `instr.*` 36 项）。
- 新三臂的体积/哈希由 `fill_hashes.py` 回填进 `config.json`（幂等）。

## 2. 生成命令（逐字，可重跑）

```bash
# 步时先测（§17.8/§20.3）：200 步 × bs32 × T256，落 /tmp/steptime2624
bash axes/1-time/1.1.4.1-explicit-memory/experiments/2624-composable-followups-13m/measure_step_time.sh

# 三臂训练（按臂取/放锁 + 臂间 65s 窗口，§20.1）
bash axes/1-time/1.1.4.1-explicit-memory/experiments/2624-composable-followups-13m/train_followups.sh

# 回填哈希 → 冒烟 → 全电池（纯推理）
python3 axes/1-time/1.1.4.1-explicit-memory/experiments/2624-composable-followups-13m/fill_hashes.py
python3 axes/1-time/1.1.4.1-explicit-memory/experiments/2624-composable-followups-13m/run.py --smoke
python3 axes/1-time/1.1.4.1-explicit-memory/experiments/2624-composable-followups-13m/run.py

# 对账 / 变异 / 单测
python3 axes/1-time/1.1.4.1-explicit-memory/experiments/2624-composable-followups-13m/reconcile_numbers.py
python3 axes/1-time/1.1.4.1-explicit-memory/experiments/2624-composable-followups-13m/mutate.py
python3 -m pytest tests/test_composable_followups_13m.py -q
```

单臂等价命令（与 `train_followups.sh` 逐字段一致）：

```bash
python3 train_depth_ar.py mha_mlp5       --d 256 --layers 12 --heads 8 --seq-len 256 --dense \
  --topology sequential --batch-size 32 --lr 1e-3 --seed 0 --synth-task assoc-mqar \
  --synth-pairs 8 --synth-queries 8 --synth-gap 14 --steps 6000  --mlp-mult 5 \
  --out-root axes/1-time/1.1.4.1-explicit-memory/experiments/2624-composable-followups-13m/train
python3 train_depth_ar.py gqa_10k        ... --steps 10000 --mlp-mult 6 --n-kv-heads 2
python3 train_depth_ar.py gqa_carry_10k  ... --steps 10000 --mlp-mult 6 --n-kv-heads 2 --carry-write delta
```

## 3. 本线自写代码的指纹（sha256 前 24 位，落盘时）

| 文件 | sha256[:24] | 字节 |
|---|---|---|
| `run.py` | `910b807c8543f4598b6d8c36` | 36,008 |
| `points.py` | `102d9efd1a0576cbed77cedf` | 12,129 |
| `config.json` | `7f84cb5d6b8c618a7f58f17f` | 6,541 |
| `train_followups.sh` | `876e1f835e9dd1c63c7905a2` | 2,061 |
| `gpu_lock.sh` | `0d30dbf70884e5962cec4636` | 1,966 |
| `measure_step_time.sh` | `5230ae7e238a51d56329ad7b` | 1,723 |
| `reconcile_numbers.py` | `0536d5b252a43945d635a838` | 25,028 |
| `mutate.py` | `16df3a95ff62b43bed5c2ae7` | 5,112 |
| `analysis.md` | `b24221a6bb0e1c88776859c4` | 21,176 |
| `README.md` | `3bed708ea76335831e2ef4bf` | 19,172 |

## 4. 共享文件指纹（SCHEMA §22.6，口径重叠必须写出来）

本线**没有改动任何共享文件**（`git status` 对本线运行前后的 `pathlm/`、`train_depth_ar.py`、
`tools/evalkit/` 均无差异）。三臂训练与全部评测都跑在下列修订上：

| 共享文件 | sha256[:24]（训练/评测时的实际内容） |
|---|---|
| `pathlm/depth_ar.py` | `3a0db552faf4ee5d193ef8c1` |
| `train_depth_ar.py` | `1035565281ed66af16e303ce` |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a555` |
| `tools/evalkit/retention.py` | `8320f03a2ecf74e32ab29b0f` |
| `tools/evalkit/carry_channel.py` | `4dcd91facd60a512dcc98383` |

⇒ 本线读数与 2623 的读数**在同一引擎修订上**（且仪器一致性 36 项逐位相同）⇒ 跨实验可比。
（若其它并行线在**本线之后**改动了这些文件，不影响本线已落盘的数字；复核时应以本表的
哈希为准核对。）
