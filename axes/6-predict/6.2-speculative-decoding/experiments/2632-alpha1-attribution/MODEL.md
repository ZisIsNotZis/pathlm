# MODEL.md — 2632-alpha1-attribution 权重清单

本实验**不训练新权重**：它复用 `2631-spec-three-arms` 的 checkpoint。
权重是派生物、不入库（`.gitignore` 覆盖 `*.pt`），此处只记哈希与来源。

| checkpoint | 路径（相对本仓） | sha256 | 体积 |
|---|---|---|---|
| 13m_s0 | `axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0/model.pt` | `1df7840bc4813f89709805222cedf34b0384f8bae0a04e23f9ffd2818b2d4e4d` | 54,984,075 B |
| 13m_s1 | `axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s1/model.pt` | `047893bda15104cc25794dc598d96bca96bbef7449bd5d45518e78e205a31b59` | 54,984,075 B |
| 53m_s0 | `axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/53m_s0/model.pt` | `0627862468867f1dcdf939adeaa7b65bc78bf92d7d9efec5d106bebfb37ccf35` | 217,923,979 B |

- 生成命令（见 `2631/train_arms.sh` / `2631-spec-three-arms/README.md` §7）：`python3 train_m1.py <scale>_s<seed>
  --config 2631-spec-three-arms/config_<scale>.json --seed <seed> --out-root 2631/train`。
- 配置：`n_mtp=3`、`chain_mode="token"`、`transport="none"`，其余开关全关；6000 步 × bs32 × T512。
- **三臂共用同一份 `model.pt`**（无臂级权重）：`2631-spec-three-arms/run.py::load_ckpt` 每 checkpoint 只加载一次
  并跑全部臂。本实验的 `results.json::checkpoints.*.teacher_anchor.sha256_model_pt` 复核了这一点。
- 训练显存/墙钟等见 `2631/train/*/results.json`；本实验**未用 GPU**。
