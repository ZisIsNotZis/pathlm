# 2624 · 权重清单与代码指纹（权重**不入库**）

契约见同目录 `README.md`；原始结果 `results.json`；判读 `analysis.md`。

## 1. 检查点（`train/<arm>/model.pt`，**不入库**）

本实验的两条被训练的臂**自己训练**（6k 步合成 assoc-mqar，seed=0，
`train_arms.sh`）；`run.py` 加载时逐次 `sha256` 断言（哈希写在 `config.json::checkpoint.sha256`）。

| 臂 | 引擎位点 | 参数量 | sha256（前 16） |
|---|---|---|---|
| `carry_mha` | `CarryChannelBlock` | 13,801,064 | `18cfc48e71874567`… |
| `carry_gqa` | `ComposableBlock` | 12,616,808 | `02c390d8b1cb41a3`… |

（完整 sha256 写在 `config.json::checkpoint.sha256`，与 `results.json::checkpoints` 一致。）

- 生成命令（可复现）：
  ```bash
  python3 train_depth_ar.py carry_mha --d 256 --layers 12 --heads 8 --mlp-mult 6 \
    --seq-len 256 --dense --topology sequential --steps 6000 --batch-size 32 --lr 1e-3 \
    --seed 0 --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
    --carry-write delta --train-window 16 --train-anchors 0 --out-root train
  python3 train_depth_ar.py carry_gqa  <同上> --n-kv-heads 2
  ```
- 冒烟检查点（`smoke/train/<arm>/model.pt`，20 步）同样不入库，仅供接线验证。

## 2. 代码指纹（本线**只新增**文件，不改任何共享引擎文件）

| 文件 | 角色 |
|---|---|
| `quant_carry.py` | **本线唯一的新语义层**：评测期量化前向（`CarryChannelBlock` / `ComposableBlock`），量化算子复用 `tools/evalkit/kv_quant.py` |
| `compression_arms.py` | 三行评测定义 / 块类规则 / 分量分解（可失败单测的靶子） |
| `run.py` | 评测入口（纯推理） |
| `train_arms.sh` | 两臂训练驱动（按臂取/放 GPU 锁，臂间 65s） |
| `mutate.py` / `reconcile_numbers.py` | 变异验证 / 三道对账 |
| `tests/test_carry_under_compression_quality.py` | 12 条可失败单测 |

- **共享文件面分析（SCHEMA §22）**：本线**不写** `pathlm/`、`train_depth_ar.py`、
  `tools/evalkit/`、`catalog.yaml`、`INDEX.md`、`axes/**/README.md` ⇒ 与并行线**无共享写入**。
  ENG1（39-X2）已把 `GQA × carry × window` 做成可组合；本线只是**使用**该能力。
- **引擎指纹（只读引用，非本线改动）**：`pathlm/depth_ar.py`、`train_depth_ar.py`
  自 ENG1（39-X2，commit `3c2b6a9`）与 X3（`b96ac47`）之后未再改动 ⇒ 与 Y1（2623）训练时
  同一份引擎。
