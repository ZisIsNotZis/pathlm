# MODEL · 2644-channel-sparsity-13m

## 被测量模型（三臂，每条臂一个 checkpoint）

| 臂 | 权重 | sha256 | size |
|---|---|---|---|
| `K50` | `runs/K50-s0-12k/model.pt` | `7c7fcd571f614de8e052d67f9780e0bded29c6ee4a5e426bc84ec0c48a4f4317` | 26.6 MB |
| `K25` | `runs/K25-s0-12k/model.pt` | `ddd5a4e061fe3993220907cd7210ebf84ad229c1b8be7a875b0e4fd0696938b6` | 26.6 MB |
| `F50` | `runs/F50-s0-12k/model.pt` | `b4bd8b94735cf73f20d59fe986d24ed51023b32f0dd718e3afb95ad46ee13b50` | 26.6 MB |

- 结构（三臂逐字段相同）：8 层 × `d_model` 256 × 8 头 × `head_dim` 32 × `mlp_mult` 4（FFN 隐藏 1024），
  dense decoder-only（stock `Block` + 实验目录内的 `SparseFFN` 包装），vocab 206。
- 总参数：**6 962 688**（三臂逐位相同；掩码是前向算子，不增不减参数）。
- 训练：12 000 步 × bs 32 × seq 256 = 98 304 000 token/臂，`seed 0`，`lr 1e-3`；评测
  `n_batches 20 / bs 16 / eval_generator_seed 2`。
- clean bpc（本 run）：`K50` 1.6295 · `K25` 1.6417 · `F50` 1.5678（见各 `results_<臂>_s0.json`）。

## 复跑与自检

- `python3 run.py --arm <臂> --seed 0 --steps 12000` 训练 + 评测；`--no-train` 只评测（复用已落盘
  checkpoint）；`--aggregate` 汇总 `P1`–`P4` 到 `results.json`。
- 评测入口会**当场重算并写回** checkpoint 的 sha256（`results_<臂>_s0.json::checkpoint`），与上表
  逐位相同；缺少 checkpoint 时显式报错，不静默降级。
- 冒烟档 `runs/K25-s0-0k/`（60 步）只作 `README` §8 的前置，不是读数。

## 权重与数据是否入库

- **权重不入库**：`runs/**` 被 `.gitignore` 排除；本文件是它们的哈希清单（唯一入库的口径记录）。
- **数据不入库**：`data/enwik8_full.npz`（前 90M 字节训练 / 末 10M 评测，vocab 206）。
- 三臂共享同一份主干抽样（同 `seed`、同构造路径），差别只在 FFN 前向的通道子集。
