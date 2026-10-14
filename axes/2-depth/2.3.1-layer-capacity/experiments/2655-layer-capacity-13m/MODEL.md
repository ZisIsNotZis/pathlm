# MODEL · 2655 逐层容量 schedule（13M）

权重本身**不入库**（`.gitignore` 覆盖 `*.pt`）；本文件只登记清单、哈希与生成命令。

| 臂 | run 目录 | model.pt | sha256 | 体积 | 生成命令 |
|---|---|---|---|---|---|
| `uniform` | `runs/uniform-s0-12k/` | 已生成 | `628f1d3aad643f6dd6988af39b5443e69674a620ea910b8835b89b0a8eabdd4a` | 27 882 155 B | `python3 run.py --arms uniform --seed 0 --steps 12000` |
| `front-big` | `runs/front-big-s0-12k/` | 已生成 | `ed04973fe4e43d2f5c07b100fc88f4516d8b2ce3ef276c437d8ab299a89db439` | 27 882 155 B | `python3 run.py --arms front-big --seed 0 --steps 12000` |
| `growing` | `runs/growing-s0-12k/` | 已生成 | `11d04c30ac6ec5f167bd75da0008640390a373c993ec3c6d0d281876b38facf8` | 27 882 155 B | `python3 run.py --arms growing --seed 0 --steps 12000` |

- 哈希复算命令：`sha256sum runs/<arm>-s0-12k/model.pt`（本文件哈希**不入库**的权重副本，只作回执）。
- 参数量（无需权重即可复算，三臂实测逐位相等）：总参数 6 962 688；FFN 参数 4 204 544；
  非 FFN 参数 2 758 144。复算命令（不取 GPU 锁）：
  `python3 -c "import smoke; print(smoke.gate_equal_budget()['per_arm'])"`。
- 训练随机性固定：`--seed 0`；数据固定 `data/enwik8_full.npz`（前 90M train / 后 10M eval）。
- 评测复现（复用权重，避免重训）：`python3 run.py --arms <arm> --no-train`。
- 训练产物：`runs/<arm>-s0-12k/{model.pt, train_log.jsonl, results.json, gate_log.jsonl}`；
  评测产物：本目录 `results_<arm>_s0.json`（逐臂增量）与 `results.json`（汇总，`--aggregate`）。
- 生成代码修订：`run.py` sha256 = `9baed65e49673e8b877a1e0bac01fd8a73ab3aa79c9b433dbe19ad4d1c339af2`
  （三臂逐臂产物内 `code_hash_run_py` 同值）。
