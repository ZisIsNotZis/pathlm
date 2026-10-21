# MODEL · 2687-learned-channel-selector-13m

## 被测量模型（四臂，两条新臂各一个 checkpoint）

| 臂 | 权重 | sha256 | size |
|---|---|---|---|
| `L50` seed 0 | `runs/L50-s0-12k/model.pt` | `98852afbc63683c3a595916c08cac8632a9cc73981589ef89e8243acdcbdb1fa` | 28.2 MB |
| `L50` seed 1 | `runs/L50-s1-12k/model.pt` | `560e41143e4f0d47b7326ec08d8a332be825a57e8efcd5d8defd92268de0e3e4` | 28.2 MB |

三支复用臂的 checkpoint 不在本目录（它们是 `2641` / `2644` 的产物，只在本地重评）：

| 臂 | 权重（相对仓库根） | 来源 | clean bpc（重评 = 落盘值） |
|---|---|---|---|
| `D` | `axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/runs/U4-s0-12k/model.pt` | `2641 U4-s0` | 1.5314 |
| `K50` seed 0 / 1 | `axes/3-width/3.1.5-channel-sparsity/experiments/2644-channel-sparsity-13m/runs/K50-s{0,1}-12k/model.pt` | `2644` | 1.6295 / 1.6263 |
| `F50` seed 0 / 1 | `…/2644-channel-sparsity-13m/runs/F50-s{0,1}-12k/model.pt` | `2644` | 1.5678 / 1.5625 |

- 结构（四臂逐字段相同）：8 层 × `d_model` 256 × 8 头 × `head_dim` 32 × `mlp_mult` 4（FFN 隐藏 1024），
  dense decoder-only（stock `Block` + 实验目录内的 `ScoredFFN` 包装 + 每层一个低秩打分器），vocab 206。
- 总参数：**7 044 608**（四臂逐位相同；= stock 6 962 688 + 8 层 × 10 240 打分器参数）。
  只有 `L50` 的 81 920 个打分器参数可训练；其余三臂冻结且前向不用（拿到预算不花）。
- 训练：12 000 步 × bs 32 × seq 256 = 98 304 000 token/臂，`lr 1e-3`，`seed 0/1`；
  `L50` 实测 9.12 / 9.37 min、峰值显存 2 852 MB；评测 `n_batches 20 / bs 16 / eval_generator_seed 2`。
- clean bpc（本目录产出）：`L50` **1.6382 / 1.6291**（见各 `results_L50_s<seed>.json`）。

## 复跑与自检

- `python3 run.py --arm L50 --seed {0,1} --steps 12000` 训练 + 评测；
  `python3 run.py --arm {D,K50,F50} --seed {0,1} --no-train` 用 `REUSE` 表里的 checkpoint 只评测；
  `python3 run.py --aggregate` 汇总 G1–G3 与复现门到 `results.json`。
- 载入 checkpoint 时 `adapt_state` 只把 stock 的 `blocks.i.mlp.<j>.*` 改名为包装后的
  `blocks.i.mlp.mlp.<j>.*`（打分器键 `blocks.i.mlp.score_*` 不动，单测 `T10` 钉住）。
- 评测入口会**当场重算并写回** checkpoint 的 sha256（`results_L50_s<seed>.json::checkpoint`），
  与上表逐位相同；缺少 checkpoint 时显式报错，不静默降级。
- 冒烟档在 `/tmp/2687-smoke`（60 步、`--skip-evaluate`），不是读数、不入库。

## 权重与数据是否入库

- **权重不入库**：`runs/**` 与 `*.pt` 被 `.gitignore` 排除；本文件是它们的哈希清单（唯一入库的口径记录）。
- **数据不入库**：`data/enwik8_full.npz`（前 90M 字节训练 / 末 10M 评测，vocab 206）。
- `L50` 两臂共享同一份主干抽样（同 `seed`、同构造路径、`B` 零初始化）；臂间唯一差异 =
  「这份打分器预算是否被训练」。
