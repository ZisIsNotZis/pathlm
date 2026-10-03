# MODEL — 2623 · A9 合成长程判决（训练 window vs full）

> 权重（`*.pt`）**不入库**（`.gitignore` 覆盖）。本文件是哈希与生成命令的唯一来源；
> 评测入口 `run.py` 加载权重前**逐次断言 sha256**（不符即 `SystemExit`）。

## 1. 臂（唯一变量 = 训练/评测可见掩码）

| 臂 | `--train-window` | 参数量 | 步数 | sha256 | 体积 | 备注 |
|---|---|---|---|---|---|---|
| `full` | 0（缺省） | 13,536,768 | 12000 | `d80f4c625a0f62f5…`（见 `config.json`） | 54,193,675 B | 无掩码（释放版因果） |
| `win16` | 16（+anchors 0） | 13,536,768 | 12000 | `525bd3067910b8ce…`（见 `config.json`） | 54,193,675 B | 只看最近 16 个位置 |
| `win16_rep` | 16（+anchors 0） | 13,536,768 | 12000 | 见 `config.json` | 见 `config.json` | 同配置重跑（复现臂） |

两臂**参数量逐位相等**（掩码零参数）；块类都是 stock `Block`（窗口是正交装饰层）。

## 2. 生成命令（逐字；两臂只差 run 名与 `--train-window`）

```bash
# full 臂（无掩码）
python3 train_depth_ar.py full \
  --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense --seq-len 256 \
  --steps 12000 --batch-size 32 --seed 0 --lr 0.001 \
  --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
  --out-root train

# win16 臂（训练/评测同掩码 = window 16 + anchors 0）
python3 train_depth_ar.py win16 \
  --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense --seq-len 256 \
  --steps 12000 --batch-size 32 --seed 0 --lr 0.001 \
  --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
  --train-window 16 --train-anchors 0 \
  --out-root train
```

实跑由 `./train_all.sh` 驱动（持 `/tmp/pathlm_gpu.lock`，SCHEMA §20 + §20.1 按臂释放）；
锁竞争下可按单臂补跑：`./train_all.sh --arms "win16"`。步时探针 `./smoke/steptime_probe.sh`。

## 3. 评测

```bash
python3 run.py             # 全电池（纯推理；持 GPU 锁，先取锁再跑）
python3 run.py --smoke     # 冒烟（缩批）
python3 fill_config_hashes.py   # 训练后回填 sha256
python3 reconcile_numbers.py    # 三道对账（退出码 0）
python3 mutate.py               # 变异验证（抓红 + 逐字节还原）
```

## 4. 共享文件指纹（SCHEMA §22.6）

本实验**不改任何共享文件**：只用既有开关 `--train-window` / `--train-anchors`，
评测只消费 `longrange_tasks.py` / `carry_channel.py` / `retention.py`（只读）。
新增代码都在本目录（`run.py` / `lr_eval.py`）与新单测 `tests/test_a9_window_longrange.py`。
运行期指纹见 `results.json::code`。
