# MODEL · 2615 窗口训练后依赖是否「右移」（13M dense，两臂）

权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出哈希、体积与生成命令。

## 被测量臂 1：`full`（复用，只读，不重训）

| 项 | 值 |
|---|---|
| 文件 | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |
| 架构 | `DepthARModel(dense=True)`：d256 × 12 层 × 8 头 × mlp6 × seq256；tied 读出、无 final LN |
| 参数 | 13,536,768（13.54M） |
| 体积 | 54,193,675 B（54.2 MB） |
| SHA-256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 训练 | 12000 步 × bs32 × seq256 = 98.3M tokens，AdamW lr 1e-3 + cosine，seed 0（2610 记录） |
| 记录 bpc | 1.4836（clean，24×16 窗口）——本实验同一窗口序复算得 **1.483569**（与 2614 记录 max\|Δ\| = 0.0） |

**sha256 断言**（`run.py::load_ckpt`，复用他人 checkpoint 的强制项；不匹配直接 `SystemExit`）。

## 被测量臂 2：`window32`（本实验训练）

| 项 | 值 |
|---|---|
| 文件 | `train/window32/model.pt`（`train_depth_ar.py` 的 `model.pt`） |
| 架构 | 同上一行，**唯一差别** = `train_window=32`、`train_anchors=8` |
| 参数 | 13,536,768（与 `full` 逐位相同，掩码不引入参数） |
| 体积 | 54,193,675 B |
| SHA-256 | `6601cd1e12c4756ad21b15bcbd9bee808239236a67f2eb90b84308b3feccc70a` |
| 训练墙钟 | **7.62 min**（含训练后 `depth_curve` 电池；GPU 串行、共享卡） |
| 训练损失（native 掩码下） | 末行 step 11999: 1.0468 nats；全程累计平均 1.2859 nats |
| native 口径 clean bpc | **1.570135**（训练掩码下，24×16 窗口，与 2614 同窗口序） |
| release 口径 clean bpc | **1.707367**（全注意力；见 `analysis.md` §5 的 OOD 说明） |
| 活性量 | 掩码可见因果对 9,460 / 32,896 = 0.287573（每 200 步写在训练日志同一行）；一次功能探针 max\|Δlogit\| = 1.011293 > 0 |

**sha256 断言**：`config.json::trained_checkpoint.sha256`，`run.py` 每次加载都断言。

## 被测量臂 2b：`window32_rep`（同配置重跑，负结果复现用）

| 项 | 值 |
|---|---|
| 文件 | `train/window32_rep/model.pt`（同命令、同 seed、同步数） |
| 参数 / 体积 | 13,536,768 / 54,193,675 B |
| SHA-256 | `4890194e9b71e9b411827ceb4b55710a3cbfae52afe4c31eb23aa788ad124800` |
| 训练墙钟 | 8.27 min |
| native / release 口径 clean bpc | 1.565637 / 1.641257 |
| 用途 | methodology §2「n=1 负结果必须复现后入账」的复现臂；电池结果在 `rep/results.json`，判读在 `stability.md §3.2` |

**生成命令**：与臂 2 完全相同，只是 `--out-root` 下的 run 名换成 `window32_rep`。
`config_rep.json` 是该臂的 resolved config（`run.py --config config_rep.json --out-dir rep`）。

## 生成命令（两臂都可复现）

```bash
# arm 2（本实验训练；config.json::train_window32 的逐字段落地）
python3 train_depth_ar.py window32 --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
  --steps 12000 --batch-size 32 --seed 0 --lr 0.001 \
  --train-window 32 --train-anchors 8 --eval-batches 24 \
  --out-root axes/A-mixing/A9-old-info-retention/experiments/2615-window-trained-retention-13m/train

# arm 1（2610 侧，本实验未重跑；哈希见上表）
python3 train_jumphead.py \
  --config axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/config.json \
  --out-dir axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m
```

## 本实验的派生产物（全部入库，不含权重）

| 文件 | 内容 |
|---|---|
| `results.json` | 两臂的依赖腿三曲线 + 检索腿 8 个条件 + 判据①②③ + 对账 + 复现性 + 噪声底 + 收敛对账 + 成本 |
| `rep/results.json` | **同配置重跑**臂的全电池（负结果复现；`config_rep.json` 为其 resolved config） |
| `retention_log.jsonl` | 396 行逐配置日志（bpc / 窗口 seed / 耗时） |
| `run_log.txt` | 全电池 stdout（`python3 run.py > run_log.txt 2>&1`） |
| `train/window32{,_rep}/{train_log.jsonl,gate_log.jsonl,results.json}` | 训练日志（含每 200 步的掩码活性量）与训练器电池 |
| `mutation_report.json` | 5 处「掩码写错」变异验证（全部抓红）+ 1 处冗余防线的如实记录 + 还原后 sha256 一致 |
| `reconcile_report.json` | `reconcile_numbers.py` 的报告/结果逐条对账（121/121 通过） |
| `analysis.md` / `stability.md` | 判读 / 稳定性专报 |

## 代码指纹（`results.json::code`，与本文档同一修订）

| 文件 | sha256 |
|---|---|
| `run.py` | `e9491b5a4c82e6db60fd1c7a44db5b135d94910fb3b219eb27d02fe1b31bd7d5` |
| `pathlm/depth_ar.py` | `6c7454581af30291688cf825c39f01ce2175c0e447ef5e83b772f2d063ddfea5` |
| `train_depth_ar.py` | `11298e6b22678c3c6d9b41600a539b2daf62f90003a32980e0b52a960e208613` |
| `tests/test_depth_ar.py` | `e9b692bb5ac85c7519bd21d543a118db74c50f80326d3eebbb6ded0f757b720c` |
| `tools/evalkit/retention.py` | `8320f03a2ecf74e32ab29b0f0af0ef610d7cf45116ae5dcdcdeb241c645e6502` |
| `tests/test_retention.py` | `d549f498ef10f2af408ecb5ab9a128482f41624bae271aedc823c21cf9391fd3` |
