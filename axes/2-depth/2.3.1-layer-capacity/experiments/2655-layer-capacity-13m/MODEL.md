# MODEL · 2655 逐层容量 schedule（13M）

权重本身**不入库**（`.gitignore` 覆盖 `*.pt`）；本文件只登记清单、哈希与生成命令。

| 臂 | run 目录 | model.pt | sha256 | 体积 | 生成命令 |
|---|---|---|---|---|---|
| `uniform` | `runs/uniform-s0-12k/` | 已生成 | 见下 | — | `python3 run.py --arms uniform --seed 0 --steps 12000` |
| `front-big` | `runs/front-big-s0-12k/` | **未生成**（本轮未训完） | 未测 | — | `python3 run.py --arms front-big --seed 0 --steps 12000` |
| `growing` | `runs/growing-s0-12k/` | **未生成**（本轮未训完） | 未测 | — | `python3 run.py --arms growing --seed 0 --steps 12000` |

- 参数量（无需权重即可复算，实测三臂逐位相等）：总参数 6 962 688；FFN 参数 4 204 544；
  非 FFN 参数 2 758 144。复算命令（不取 GPU 锁）：
  `python3 -c "import smoke; print(smoke.gate_equal_budget()['per_arm'])"`。
- `uniform` 臂的 `model.pt` 哈希：由后续 worker 用
  `sha256sum runs/uniform-s0-12k/model.pt` 补入（权重不入库，故本文件不含体积/哈希副本；
  训练产物内的 `results.json` 与 `train_log.jsonl` 已随 `runs/uniform-s0-12k/` 落盘）。
- 训练随机性固定：`--seed 0`；数据固定 `data/enwik8_full.npz`（前 90M train / 后 10M eval）。
- 评测复现（复用权重，避免重训）：`python3 run.py --arms <arm> --no-train`。
- 训练产物：`runs/<arm>-s0-12k/{model.pt, train_log.jsonl, results.json, gate_log.jsonl}`；
  评测产物：本目录 `results_<arm>_s0.json`（逐臂增量）与 `results.json`（汇总，`--aggregate`）。
