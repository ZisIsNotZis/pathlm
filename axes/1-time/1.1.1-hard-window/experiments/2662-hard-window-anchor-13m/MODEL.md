# 2662-hard-window-anchor-13m · 权重清单（MODEL.md）

> 由 lead（axis1-lead-4）于本轮补齐（复核 F4/P2 指出该目录缺 `MODEL.md`）；数字全部取自同目录 `results.json`，
> 权重 `*.pt` 按 `.gitignore` 不入库，故此处只留路径 + sha256。

## 1 权重（checkpoints）

| 臂 | 路径 | sha256（全） |
|---|---|---|
| `win16a_s0` | `axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/train/win16a_s0/model.pt` | `5a41ed069ee0ba784187b8d2adc054d334ed418c57ba5ff05fbf6347141cdb2d` |
| `win16a_s1` | `axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/train/win16a_s1/model.pt` | `eb6c9d4a8693c45848d55c672f466aa42f92e4c8586011862ca2f2cdf0a55ffb` |
| `win16_s0` | `axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m/train/rt_none_s0/model.pt` | `c3f6f08a4fd1dfe9b5c72cb433fe20a2390052b1eda11b8ab9a6d0fe468deca7` |
| `win16_s1` | `axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m/train/rt_none_s1/model.pt` | `2bfe167c133c496f42864a191f2fc8833f20ba4ec8f9f4f1dceba43dd84fb162` |
| `full` | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |

## 2 脚本/配置指纹（`results.json::code_hashes`）

| 文件 | sha256（全） |
|---|---|
| `pathlm/depth_ar.py` | `45b7d40673eed8844f5e094f8e40bd5a98bd8b8a195a3c41efa116ca14e3de01` |
| `train_depth_ar.py` | `cddbbe0d3a758016308ecd9e501a7a0a946f0844b45d3f06a54aaeec9722b6a3` |
| `tools/evalkit/retention.py` | `9ef983b70d666460de5d63d5d81c23c685ce708a55dfe1af66fc23187fd12f9a` |
| `tools/evalkit/carry_channel.py` | `18d201a57019880230b80040c76748dcaa5c5f5e380c0ab4c7b5a3caae000717` |
| `pathlm/data.py` | `48e23520122a0b4591e71d0e3780dd1f34dc4a82744e14ca3414da8f43ba448a` |
| `run.py` | `7061936b2dc3d3359531c1365d8abe0b274d39c2f5faae665064b9ffe7937260` |
| `realtext_probe.py` | `dbddeb5f8dc3a7bebe6eda6e2d977ce6581fade2b5db7231bc3f0116addd6402` |
| `needle_stats.py` | `d27638b872d4bf2b28a9223b8a376f9a29d98693da8c9cc6d9c9ee115f099477` |
| `make_config.py` | `ef6e6e8c823bb96b858c8d9d52cca6f03dafb15f7a052f3f6fa482b592d9d88a` |
| `config.json` | `d7ee235c8594a26c287f460b91bbd05b39a66b63cbc714917ef6459c4831e2ad` |
| `train_arms.sh` | `4b36af233cdb23795c3e11c07f55bc54ca212e64f200ae4ca7fb87b810f74547` |

## 3 评估口径提示

- 锚点臂（`win16a_*`）用**自己的训练可见性**（窗口 16 + 锚点 8，`k_eff=24`）评测；纯窗臂（`win16_*`）用锚点 0（`k_eff=16`）
  ⇒ 是「同 token 预算、不同可见集」的比较，不是「同掩码」比较。
- bpc：本格自测带 `σ_cell` 为主判、0.014 跨格带为参考；召回等率类走两比例差 CI（正态近似/未 pooling）与配对 bootstrap。
- 全部读数为该快照（enwik8 / T=256 / 13M / 12k 步），禁外推。
