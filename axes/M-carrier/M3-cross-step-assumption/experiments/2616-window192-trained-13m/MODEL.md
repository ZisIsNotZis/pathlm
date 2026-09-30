# MODEL · 2616 有界内存的悬崖——窗口 192 训练（13M dense，两臂）

权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出哈希、体积与生成命令。

## 被测量臂 1：`full`（复用，只读，不重训）

| 项 | 值 |
|---|---|
| 文件 | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |
| 架构 | `DepthARModel(dense=True)`：d256 × 12 层 × 8 头 × mlp6 × seq256；tied 读出、无 final LN |
| 参数 | 13,536,768（13.54M） |
| 体积 | 54,193,675 B（54.2 MB） |
| SHA-256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 训练 | 12000 步 × bs32 × seq256 = 98.304M tokens，AdamW lr 1e-3 + cosine，seed 0（2610 记录） |
| 记录 bpc | 1.4836（clean，24×16 窗口）——本实验同一窗口序复算得 **1.483569**（与 2614 记录 max\|Δ\| = 0.0） |

**sha256 断言**（`run.py::load_ckpt`，复用他人 checkpoint 的强制项；不匹配直接 `SystemExit`）。

## 被测量臂 2：`window192`（本实验训练）

| 项 | 值 |
|---|---|
| 文件 | `train/window192/model.pt`（`train_depth_ar.py` 的 `model.pt`） |
| 架构 | 同上一行，**唯一差别** = `train_window=192`、`train_anchors=8` |
| 参数 | 13,536,768（与 `full` 逐位相同，掩码不引入参数） |
| 体积 | 54,193,675 B |
| SHA-256 | `65d3fae5640fd35eb1328842987491729b743cbd619c23bbcd2a27c13d162262` |
| 训练墙钟 | **8.23 min**（含训练后 `depth_curve` 电池；GPU 串行、共享卡） |
| 训练损失 | 末行 step 11999: **1.0103 nats**；末 25% 同方法最小二乘斜率 **−3.6e-07 nats/step**（基本平） |
| native 口径 clean bpc | **1.480756**（训练掩码下，24×16 窗口，与 2614 同窗口序） |
| release 口径 clean bpc | **1.478378**（全注意力；该口径**不是** OOD，见 `analysis.md §4`） |
| 活性量 | 掩码可见因果对 31,300 / 32,896 = **0.951483**（每 200 步写在训练日志同一行）；一次功能探针 max\|Δlogit\| = **0.217099** > 0 |

**sha256 断言**：`config.json::trained_checkpoint.sha256`，`run.py` 每次加载都断言。

## 被测量臂 2b：`window192_rep`（同配置重跑，复现用）

| 项 | 值 |
|---|---|
| 文件 | `train/window192_rep/model.pt`（同命令、同 seed、同步数） |
| 参数 / 体积 | 13,536,768 / 54,193,675 B |
| SHA-256 | `8e3c6fb1bcaa26c44292faec4dba707cf0c52c5f54de73146b77a2e083274a5e` |
| 训练墙钟 | 8.67 min |
| 末行训练 loss | 1.0051 nats（主臂 1.0103） |
| native / release 口径 clean bpc | **1.484242** / **1.481776**（主臂 1.480756 / 1.478378） |
| 用途 | `README.md §7` 预登记的执行规则（结论落在阈值附近 ⇒ 必须复现）：给出训练侧 run-to-run 量级，并复核三条判据结论；电池结果在 `rep/results.json`，判读在 `stability.md §3.2` |

**生成命令**：与臂 2 完全相同，只是 `--out-root` 下的 run 名换成 `window192_rep`。
`config_rep.json` 是该臂的 resolved config（`run.py --config config_rep.json --out-dir rep`）。

## 生成命令（两臂都可复现）

```bash
# arm 2（本实验训练；config.json::train_window192 的逐字段落地）
python3 train_depth_ar.py window192 --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
  --steps 12000 --batch-size 32 --seed 0 --lr 0.001 \
  --train-window 192 --train-anchors 8 --eval-batches 24 \
  --out-root axes/M-carrier/M3-cross-step-assumption/experiments/2616-window192-trained-13m/train

# arm 2b（重跑）
python3 train_depth_ar.py window192_rep --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
  --steps 12000 --batch-size 32 --seed 0 --lr 0.001 \
  --train-window 192 --train-anchors 8 --eval-batches 24 \
  --out-root axes/M-carrier/M3-cross-step-assumption/experiments/2616-window192-trained-13m/train

# 电池（主臂 / 重跑臂）
python3 run.py
python3 run.py --config config_rep.json --out-dir rep

# 步时基准（600 步 × 两条路径）+ 训练显存探针
python3 train_depth_ar.py bench192 ... --steps 600 --train-window 192 --train-anchors 8
python3 train_depth_ar.py bench0   ... --steps 600 --train-window 0   --train-anchors 0
python3 smoke/mem_probe.py            # → smoke/mem_probe.json

# arm 1（2610 侧，本实验未重跑；哈希见上表）
python3 train_jumphead.py \
  --config axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/config.json \
  --out-dir axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m
```

## 本实验的派生产物（全部入库，不含权重）

| 文件 | 内容 |
|---|---|
| `results.json` | 两臂的依赖腿四组曲线（距离/前缀/最近 d/锚点预算）+ 预算对齐表（§17.12 两套口径）+ 检索腿 8 个条件（含 beyond 两半拆分）+ 判据①②③ + 对账 + 复现性 + 噪声底 + 收敛对账 + 成本 |
| `rep/results.json` | **同配置重跑**臂的全电池（`config_rep.json` 为其 resolved config） |
| `retention_log.jsonl` | 408 行逐配置日志（bpc / 窗口 seed / 耗时），主臂与重跑臂各一份 |
| `run_log.txt` / `rep_run.log` | 两次全电池 stdout |
| `train/window192{,_rep}/{train_log.jsonl,gate_log.jsonl,results.json}` | 训练日志（含每 200 步的掩码活性量）与训练器电池 |
| `smoke/smoke20/` | 训练侧 20 步冒烟（同一 runner） |
| `smoke/stepbench_w{192,0}/` | 步时基准（600 步 × 2 条路径，实测非外推） |
| `smoke/mem_probe.json` | 训练步显存探针（30 步，net 峰值 1840.7 MiB allocated / 1984.0 reserved） |
| `smoke/results.json` | 评测端冒烟（2 批 + 少配置） |
| `mutation_report.json` | 6 处变异验证（全部抓红，含「窗口被夹到 32」）+ 还原后 sha256 一致 |
| `reconcile_report.json` | `reconcile_numbers.py` 的报告/结果逐条对账 |
| `analysis.md` / `stability.md` | 判读 / 稳定性专报 |

## 代码指纹（`results.json::code`，与本文档同一修订）

| 文件 | sha256 | 与 2615 记录相比 |
|---|---|---|
| `run.py`（本实验） | `bab65820dde4c61c5a9886824e9c5eb2c93b364db28220a2373cbe521327cae1` | 本实验新增（不是 2615 的那份） |
| `pathlm/depth_ar.py` | `6c7454581af30291688cf825c39f01ce2175c0e447ef5e83b772f2d063ddfea5` | **逐字节相同（未改动）** |
| `train_depth_ar.py` | `11298e6b22678c3c6d9b41600a539b2daf62f90003a32980e0b52a960e208613` | **逐字节相同（未改动）** |
| `tests/test_depth_ar.py` | `92cb98f5a348bb33dfd218688161ca0247bede693a33e35dacd675a332b3a432` | 本实验追加 3 条 192 尺度单测 |
| `tools/evalkit/retention.py` | `8320f03a2ecf74e32ab29b0f0af0ef610d7cf45116ae5dcdcdeb241c645e6502` | **逐字节相同（只增不改的仪器）** |
| `tests/test_retention.py` | `d549f498ef10f2af408ecb5ab9a128482f41624bae271aedc823c21cf9391fd3` | 未改动 |
| `smoke/mem_probe.py` | 见 `smoke/mem_probe.json` 同目录文件（探针脚本） | 本实验新增 |
