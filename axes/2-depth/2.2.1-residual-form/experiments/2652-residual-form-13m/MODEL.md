# MODEL · 2652 残差形式最小判决（13M）

权重本身**不入库**（`.gitignore` 覆盖 `*.pt`）；本文件只登记清单、`sha256`、体积与生成命令。

| 臂 | run 目录 | checkpoint | sha256 | 体积（bytes） | 生成命令 |
|---|---|---|---|---|---|
| `base` | `runs/base-s0-12k/` | `model.pt` | `0df68815e18c9c5b1cdc7662f7b97b44b3280ac790646b059d9ceb87fa5edfd6` | 27 882 155 | `python3 run.py --arm base --seed 0` |
| `postln` | `runs/postln-s0-12k/` | `model.pt` | `63df6a581876ff78675da3d6389cac902bb987786204c725ac2bf031801ef48a` | 27 882 155 | `python3 run.py --arm postln --seed 0` |
| `gated` | `runs/gated-s0-12k/` | `model.pt` | `8f62e2cce0dae196acdd3cb202bf3c31dea1a9c65eb0fa35d7c2926b25c2bcfe` | 27 886 763 | `python3 run.py --arm gated --seed 0` |
| `scaled` | `runs/scaled-s0-12k/` | `model.pt` | `c8bbb8aa19efda3239d125a3729f110e405f28ea86dbc38f4382c63672b01927` | 27 882 155 | `python3 run.py --arm scaled --seed 0` |

- 复算哈希：`sha256sum runs/<arm>-s0-12k/model.pt`；复算体积：`stat -c%s runs/<arm>-s0-12k/model.pt`。
- `base` / `postln` / `scaled` 体积逐字节相同（三者参数量逐位相同 = 6 962 688）；`gated` 多 4 608 bytes
  （= 16 个 fp32 标量门，`attn_gate`/`mlp_gate` × 8 层，总参数 6 962 704）。
- 参数量（无需权重即可复算，实测）：`python3 -c "import smoke; print(smoke.init_equality_gate('cpu')['params_total'])"`（不取 GPU 锁）。
- 训练随机性固定：`--seed 0`；数据固定 `data/enwik8_full.npz`（前 90M train / 后 10M eval，token 数 98 304 000）。
- 评测复现（复用权重，避免重训）：`python3 run.py --arm <arm> --seed 0 --no-train`。
- 单臂重跑（覆盖 `model.pt`）：`python3 run.py --arm <arm> --seed 0 --force`。
- 每个 `runs/<arm>-s0-12k/` 内含训练产物：`model.pt` · `train_log.jsonl`（每 200 步一行，含 loss/gnorm/depth_ce）
  · `results.json`（共享 `train_depth_ar.main()` 的 run 级结果）· `gate_log.jsonl`（逐层门值留痕）。
- 本目录的评测产物：`results_<arm>_s0.json`（逐臂增量，含 `eval` / `params` / `residual_form_liveness` / `stability`）
  与 `results.json`（四臂汇总 + P1–P6，由 `python3 run.py --aggregate` 重算）。
