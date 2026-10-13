# 2630-tf4-samecaliber · 权重清单（MODEL.md）

> 权重（`*.pt`）**不入库**（`.gitignore` 覆盖）。本文件记录哈希 / 体积 / 生成命令，
> 好让复核者能逐位核对。运行 `run.py` 时对两臂**断言 sha256**（`config.json::checkpoint.sha256`）。

## 1. 本实验两臂的权重

| 臂 | 路径 | sha256 | 体积（bytes） | 参数量 | 块类 |
|---|---|---|---|---|---|
| `base` | `train/base/model.pt` | `fcb8505e72e3089cd617dff69b39db8091b5bcbe493de2d909c9496a5cb285c8` | 57,358,059 | 14,326,272 | `Block` |
| `carry` | `train/carry/model.pt` | `dea708f788107d0edd92ae91f1f94edc24aceb69e35b1f46b4a627f266513bf0` | 58,420,003 | 14,590,568 | `CarryChannelBlock` |

- 体积差 1,061,944 bytes ≈ 264,296 × 4（fp32）——即 carry 参数（`carry.*` + 块级
  `carry_scale`），与 `results.json::param_declaration` 一致。
- **主干参数逐位相同**（同 seed）⇒ 单因素对照成立（§9/§15）；见
  `results.json::capability_probe.trunk_bit_identical = true`。

## 2. 生成命令

```bash
bash axes/1-time/1.1.4.1-explicit-memory/experiments/2630-tf4-samecaliber/train_arms.sh
```

- 两臂逐字段一致（唯一变量 = `--carry-write delta`），**训练数据 = enwik8**
  （`--synth-task` 缺省 `none`），非 dense + 随机退出 + proj-fill，6k×32×256，seed 0。
- 完整命令 = `train_depth_ar.py <arm> --d 256 --layers 12 --heads 8 --mlp-mult 6
  --seq-len 256 --topology sequential --steps 6000 --batch-size 32 --lr 1e-3 --seed 0
  --exit-mode uniform --proj-fill --final-weight 2.0 --eval-batches 20
  --out-root .../train [--carry-write delta]`。

## 3. 已知落盘缺口

- `carry` 臂的 `train/carry/results.json` **不存在**：训练入口在末段
  `generation_quality` 处中止（`DepthARDecoder` 对 `carry_rule` 显式 `raise ValueError`，
  `pathlm/depth_ar.py:2573`）。`model.pt` 在更早处已保存，**权重有效**；主判据
  （tf@4 / clean bpc）由 `run.py` 用同一引擎函数独立重算。详见 `analysis.md` §7。
- `base` 臂的 `train/base/results.json` 完整；其 `tf_acceptance[4]=0.9711` 与
  `depth_curve[-1].bpc=2.0923` 与 `run.py` 的输出**逐位一致**（对账脚本核对）。

## 4. 共享引擎文件的指纹

本线**不改任何共享引擎/入口文件**（`pathlm/**`、`train_depth_ar.py`、
`tools/evalkit/**` 均未修改），因此不存在「与其它线的未提交改动重叠」的口径重叠问题。
本线新增的文件全部在 `axes/1-time/1.1.4.1-explicit-memory/experiments/2630-tf4-samecaliber/`
与 `tests/test_tf4_samecaliber_13m.py`。
