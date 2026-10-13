# 2625 · 权重清单与代码指纹（MODEL.md）

契约：`README.md`；判读：`analysis.md`。
**权重不入库**（`.gitignore` 覆盖 `*.pt`）。

## 1. 本线训练的权重（seed=1 / seed=2）

生成命令（`train_arms.sh`，逐臂；`--seed` 是唯一变量）：

```bash
# s1
python3 train_depth_ar.py gqa_carry_win_s1 --d 256 --layers 12 --heads 8 \
  --mlp-mult 6 --seq-len 256 --dense --topology sequential --steps 6000 \
  --batch-size 32 --lr 1e-3 --seed 1 --synth-task assoc-mqar --synth-pairs 8 \
  --synth-queries 8 --synth-gap 14 --carry-write delta --train-window 16 \
  --train-anchors 0 --n-kv-heads 2 \
  --out-root axes/1-time/1.1.4.1-explicit-memory/experiments/2625-ood-leg-variance/train
# s2：同上，仅 --seed 2 → run 名 gqa_carry_win_s2
```

| 臂 | seed | 路径（不入库） | 体积 | sha256 |
|---|---|---|---|---|
| `gqa_carry_win_s1` | 1 | `train/gqa_carry_win_s1/model.pt` | 50,518,595 B | `896e9e557e3832d40c6ccd370fbacc167bd88f116fe053e3b3fea3c9a6fe5420` |
| `gqa_carry_win_s2` | 2 | `train/gqa_carry_win_s2/model.pt` | 50,518,595 B | `23969f29dcdf7e724c6bbb7815743161a4fbdd90085bff9f7819e49a8a3556ab` |

- 块类均为 `ComposableBlock`，参数量均为 **12,616,808**（`run.py` 运行时断言）。
- 训练墙钟：s1 **12.26 min**、s2 **11.04 min**；峰值显存 **2193.6 MB**（`train/<arm>/results.json`）。

## 2. 复用的权重（seed=0 主集 + 参照臂）

| 臂 | seed | 来源 | sha256 |
|---|---|---|---|
| `gqa_carry_win_s0` | 0 | 2624 `train/carry_gqa/model.pt`（Z3） | `02c390d8b1cb41a3f8c64923b45dcadeadcddc60a18b78477e533ad4020b0b09` |
| `y1_gqa_carry_win_s0` | 0 | 2623 `train/gqa_carry_window/model.pt`（Y1） | `48c4e39ec058b8c509de88744e32f4169f993eb4f2680e6b4080b1b410f9c8a6` |

- 四条臂 **sha256 两两不同**（四条独立训练）⇒ 印证「同配置重跑不可复现」。
- 复用前提（SCHEMA §9）：实现层 fp32 逐位等值 + 配置层逐字段对照 + 口径标注，见
  `controls/README.md §4`；评测仪器为 2624 的 `quant_carry.py`（本线只读、不改）。

## 3. 代码指纹（本线自己的文件）

| 文件 | sha256 |
|---|---|
| `run.py` | `56be351c57801a67489aa8657016b25064665725fa9c48b812f622449d1cc21b` |
| `ood_variance_arms.py` | `27b795234dd5c199e654bad94c1f58be49a56195a8bcccec2b4aaedf1fb64636` |
| `reconcile_numbers.py` | `47806d323dffcb43b8a6500609438ce8c367edc21c7c5f485681e472e2e4c08c` |
| `mutate.py` | `b9121840cd6b3339a5424247b85b5cc8c4a42b7c614489213e0ede0c12759548` |
| `tests/test_ood_leg_variance.py` | `9b8e11c1236b29d8244514b459bf5d3236b98526dd545a65ae9db21ad48b92f7` |

## 4. 共享文件改动

**无**。本线不修改 `pathlm/`（引擎）、`tools/evalkit/`（仪器）、`catalog.yaml`、`INDEX.md`、
`axes/**/README.md` 或任何他人实验目录；评测复用 2624 的 `quant_carry.py`（只读导入）。
