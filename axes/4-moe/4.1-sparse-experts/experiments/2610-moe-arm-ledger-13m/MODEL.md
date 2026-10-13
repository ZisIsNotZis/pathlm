# MODEL.md · 2610-moe-arm-ledger-13m

权重**不入库**（`.gitignore` 覆盖 `*.pt`）。本目录涉及的权重/产物：

| 产物 | 生成命令 | 说明 |
|---|---|---|
| `ledger.json` | `python3 run.py ledger --out ledger.json` | 臂记账（参数量/MACs/tokens-per-param）；无权重 |
| `preflight.json` | `python3 run.py preflight --out preflight.json` | 透明替换 / 活性 / 可训性断言；无权重 |
| `timing_20.json` | `python3 run.py train --arm moe-top1-E8 --steps 20 --log-every 1 …` | 13M 单臂 20 步实测（含过程墙钟）；权重未存 |
| `probe_collapse_1200.json` | `bash .scratch/65-axis4-moe/evidence/run_collapse_probe.sh`（`STEPS=1200`） | 13M 单臂 1200 步坍缩探针；权重未存 |
| `smoke_*.json` / `smoke2_moe.json` | `run.py train --d 64 --layers 2 --seq-len 64 --steps 40/5 …` | CPU 冒烟（toy 配置）；权重未存 |

- 训练脚本**只写 JSON 指标**，不落权重文件 ⇒ 无权重哈希可记；如需复现，用同一 seed + 同一配置即可。
- 数据：`data/enwik8_full.npz`（不入库）。
- 共享文件指纹：本目录**不含**共享 `pathlm/**` 的改动；`moe.py` 是新增模块，
  `Block.mlp` 的替换发生在 `MoEDepthARModel.__init__` 内（运行期属性替换，不改 `pathlm/model.py`）。
