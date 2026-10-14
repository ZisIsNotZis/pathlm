# MODEL · 2653 层间 causal attention（13M）

权重本身**不入库**（`.gitignore` 覆盖 `*.pt`）；本文件只登记清单、哈希与生成命令。

| 臂 | run 目录 | model.pt | sha256 | 体积 | 生成命令 |
|---|---|---|---|---|---|
| `attn-scalar` | `runs/attn-scalar-12k-s0/` | **未生成**（本轮未训完） | 未测 | — | `python3 run.py --arms attn-scalar --seed 0 --steps 12000` |
| `attn-qkv` | `runs/attn-qkv-12k-s0/` | **未生成**（本轮未训完） | 未测 | — | `python3 run.py --arms attn-qkv --seed 0 --steps 12000` |

- 参数量（无需权重即可复算，实测）：stock 同配置 sequential = 6 962 688；`attn-scalar` = 6 962 696（+8）；`attn-qkv` = 7 159 304（+196 616）。
- 复算命令：`python3 -c "import smoke; print(smoke.gate_params())"`（不取 GPU 锁）。
- 校验和口径：权重文件的 `sha256` 在训练完成后由后续 worker 用
  `sha256sum runs/<arm>-12k-s0/model.pt` 补入；本轮无权重 ⇒ 如实标「未生成」，不填占位哈希。
- 训练随机性固定：`--seed 0`；数据固定 `data/enwik8_full.npz`（前 90M train / 后 10M eval）。
- 评测复现：`python3 run.py --arms <arm> --no-train`（复用 `runs/<arm>-12k-s0/model.pt`）。
