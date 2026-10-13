# MODEL · 2627-carry-x-depth 四臂（13M dense；2×2 因子设计）

权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出哈希、体积与生成命令。
sha256 已回填进 `config.json`，`run.py` 每次加载**断言**。

## 四臂

| 臂 | 文件 | 架构 | 参数 | 体积（B） | SHA-256 |
|---|---|---|---|---|---|
| `base` | `train/base/model.pt` | stock dense（`carry_rule=None`，无 dense-aux） | 13,536,768 | 54,193,675 | `f6615a1a0954d68bd6fe7caa4d05ce0f8542f6afd5ca3841f4c13c0ef92c4f8c` |
| `carry` | `train/carry/model.pt` | + 单共享槽（delta 写） | 13,801,064 | 55,255,619 | `343661127099d5e55d4e5feca7ce649680e9af5b24be63a7f3202d87b82f89a5` |
| `depth` | `train/depth/model.pt` | stock dense（**与 base 逐位同架构**）+ `--dense-aux` | 13,536,768 | 54,193,675 | `52a18caf4d227dd520819510ce92698ee934a4906625f83d431c500a1b4ee25c` |
| `both` | `train/both/model.pt` | + 单共享槽（delta 写）+ `--dense-aux` | 13,801,064 | 55,255,619 | `0330383544864d47e4cbdaeb1f0de27f508ff2ae80f5376b4e01c1884d0efc9a` |

- `base` 与 `depth` **逐位同架构**（参数量、体积、块类全同；差异只在训练损失，`--dense-aux`）；
  `carry` 与 `both` 同理。`run.py` 每次运行断言 sha256 并落盘 `results.json::checkpoints` 与
  `architecture_equality`。
- 新增参数（carry 因子）= **264,296**（+1.952%）：`w_k/w_v/w_q`（各 d²）+ `beta_w`（H·d）
  + `beta_b`（H）+ `gate_w`（d²）+ 每层读出增益 `carry_scale`（H）。`depth` 因子 **0 参数**。

## 生成命令（四臂逐字段一致，唯一变量 = carry × depth 两因子开关）

```bash
cd axes/1-time/1.1.4.1-explicit-memory/experiments/2627-carry-x-depth
bash train_arms.sh          # 按臂取/放 GPU 锁 + 臂间 65s；等价于下面四条
# base
python3 ../../../../../train_depth_ar.py base \
  --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential \
  --steps 6000 --batch-size 32 --lr 1e-3 --seed 0 \
  --train-window 16 --train-anchors 0 \
  --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
  --out-root train
# carry : 追加 --carry-write delta
# depth : 追加 --dense-aux
# both  : 追加 --carry-write delta --dense-aux
```

评测：`python3 run.py`（同目录，配置 `config.json`）→ `results.json`。
先测步时：`bash smoke_steptime.sh` → `smoke/steps400/`。

## 本实验的派生产物（全部入库，不含权重）

| 文件 | 内容 |
|---|---|
| `results.json` | 四臂全电池：主判据腿 + 毁源对照 + OOD 腿 + 交叉口径 + tf@4/6 + 距离腿 + bpc + 2×2 因子分解 + 参数量 + 活性量 + 显存/成本 |
| `train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json}` | 训练日志（每 200 步含出口 CE 条数 / carry 活性量）与训练器电池 |
| `train_<arm>.log` | 训练 stdout |
| `mutation_report.json` | 5 处语义开关的变异验证（全部抓红 + 逐字节还原） |
| `reconcile_report.json` | `reconcile_numbers.py` 的报告/结果逐条对账 |
| `smoke/steps400/` | 先测步时（400 步稳态，四臂 ms/step） |
| `smoke/results.json` | `run.py --smoke` 的少批冒烟读数 |
| `controls/README.md` | 对照臂索引（基准 / 毁源 / 单因子 / 2×2 格子）与方差界（SCHEMA §3） |
| `analysis.md` | 判读（人话摘要 + 七问 + 2×2 因子表） |

## 代码指纹（`results.json` 之外，本表人工核算 sha256）

| 文件 | SHA-256 |
|---|---|
| `run.py` | `a8a03d8fff7df525c143689a4cffb25b4de5304e11dda3a70d846f5a841e1f0f` |
| `carry_x_depth_arms.py` | `f9d76a510da91614894d24748777e6b494f76337f4a76d1b0ab9c26c7afe5f09` |
| `config.json` | `51fc1b2c715aa8aefbeee0aba742fc16e0f69b906699fb80e714ba2712ebc409` |
| `train_arms.sh` | `2e6a78212eac034359954de8feb1498e23e93e1626f6b8a4aaf02727a0763471` |
| `smoke_steptime.sh` | `ed4416c866ada37a458e387e4f935ceff9e5f92761fa4c30c8c6b4b0f4ce916a` |
| `reconcile_numbers.py` | `bd25c29b2c44f45b46dbb49d226b59579f3f3b56f3bc50ce2e9f5cc5834327c0` |
| `mutate.py` | `da369d07faf6b53a29262910104a547a403e227b3d1dc4ec9da53d5744eb709b` |
| `tests/test_carry_x_depth_13m.py` | `7a63f849a64868e18a4d5beab0b83dd8988273a9b150d96f90c102be8da22522` |
| `pathlm/depth_ar.py`（共享，**本线未改**） | `3a0db552faf4ee5d193ef8c1153b97e0c0d7e145de65c1399cdc83282d0a9279` |
| `train_depth_ar.py`（共享，**本线未改**） | `1035565281ed66af16e303ce29148e6907020d438047a4e3fdadc1799f78409e` |
| `tools/evalkit/longrange_tasks.py` | `4a02b93f19db8e99dbf3a555b2903205ee8ef2e5a27aaa0d2d66ed0382c5465b` |
| `tools/evalkit/carry_channel.py` | `4dcd91facd60a512dcc983836fb29a7c91aad988bada1c95ada666824d226753` |
| `tools/evalkit/retention.py` | `8320f03a2ecf74e32ab29b0f0af0ef610d7cf45116ae5dcdcdeb241c645e6502` |

（`MODEL.md`/`analysis.md`/`controls/README.md`/`README.md` 自身是报告文本，不计入代码指纹。）
