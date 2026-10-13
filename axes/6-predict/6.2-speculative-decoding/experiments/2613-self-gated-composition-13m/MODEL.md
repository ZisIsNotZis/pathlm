# MODEL · 2613 自门控多跳组合（13M，冻结主干 + 只训头）

权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出哈希、体积与生成命令。
主干来自 2610（**复用，不重训**）；本实验只训头：每条臂一个 `CompositionHead`
（`T` 的 d→d 变换 [+ `gated` 的逐通道门 `G`] + tied 读出，读出复用主干表 E，不持有 unembed）。

## 冻结主干（复用 2610，运行时会自动核对 sha256）

| 项 | 值 |
|---|---|
| 路径 | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |
| 架构 | `DepthARModel(dense=True)`：d256 × 12 层 × 8 头 × mlp6 × seq256；tied 读出、无 final LN |
| 参数 | 13,536,768（13.54M）（12k 步 × bs32 × seq256 = 98.3M tokens，seed 0，bpc 1.4836 / t+1 acc 0.6979） |
| SHA-256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 体积 | 54,193,675 B（54.2 MB） |
| 核对 | `results.json.reused_trunk.sha256_verified = true`（不符即 `SystemExit`） |

## 头权重

`T` 参数量：direct / composed = 262,912（4d²+3d）；gated / direct_eq = 394,496（6d²+5d）。
tied 读出无参数。

| 文件 | 臂 | 步数 | seed | 参数量 | 体积 (B) | SHA-256 |
|---|---|---|---|---|---|---|
| `compose_head_direct_s0.pt` | direct | 2000 | 0 | 262,912 | 1,054,451 | `c21700d955d36a8dd320a6248f86b9e5250c5a776369999c4d5698be21c5e6fe` |
| `compose_head_direct_s1.pt` | direct | 2000 | 1 | 262,912 | 1,054,451 | `0018565ade4f2d7ddd5071664e4420ece12e60fef1a186727578b7176f0e0240` |
| `compose_head_composed_s0.pt` | composed | 2000 | 0 | 262,912 | 1,054,471 | `13faba513573dabb884067eab904ecfae38df319ec6e56ae55e4ec2340b46511` |
| `compose_head_composed_s1.pt` | composed | 2000 | 1 | 262,912 | 1,054,471 | `3ae54ac30db4635f4772054a79af145fbb68a06fb1c40f6e8c7d2d7436449753` |
| `compose_head_gated_s0.pt` | gated | 2000 | 0 | 394,496 | 1,581,905 | `f1e82d02780f16508ae4b57c01aa1ff0f21c3a015addd1b5e0ef35a6dac0efc7` |
| `compose_head_gated_s1.pt` | gated | 2000 | 1 | 394,496 | 1,581,905 | `5503b06feda5ed300d627ece07886db3552fc33ebd6e89fc89bbbfd113846338` |
| `compose_head_direct_eq_s0.pt` | direct_eq | 2000 | 0 | 394,496 | 1,581,151 | `e3eee6706924141b9ff7b6add11d6008606057a3feb0f01c940c2d2273e026af` |
| `compose_head_direct_eq_s1.pt` | direct_eq | 2000 | 1 | 394,496 | 1,581,151 | `36e1c1b91293e8b428030bff8ae5fc8337d11e863308bd4cfbaa273d8f8a9c62` |
| `compose_head_direct_s0_4000step.pt` | direct | 4000（§16 收敛点） | 0 | 262,912 | 1,054,541 | `5d377285d1e96e9047e0805bbc8c6362098bf195dced750ab1fb80c7807b7411` |
| `compose_head_composed_s0_4000step.pt` | composed | 4000 | 0 | 262,912 | 1,054,561 | `4c7a48fdae7652792b5169fffa879e8802459f9dd39a3f0bbe5ca05f16d2608b` |
| `compose_head_gated_s0_4000step.pt` | gated | 4000 | 0 | 394,496 | 1,582,159 | `0a728ace48306655984e38d1b7e82ba41ad6879d9eb665acc1890320604537f5` |
| `compose_head_direct_eq_s0_4000step.pt` | direct_eq | 4000 | 0 | 394,496 | 1,581,250 | `86f1bf517f00e2a195f991eeb8f94b487631d72cf565024223e9c26b8ce8e532` |

## 生成命令

```bash
# 全跑（4 臂 × 2 seed @2000 步 + 4 臂 × 1 seed @4000 步 + 评测）≈7 min GPU
python3 axes/6-predict/6.2-speculative-decoding/experiments/2613-self-gated-composition-13m/run.py

# 20 步冒烟（实测步时与显存）
python3 axes/6-predict/6.2-speculative-decoding/experiments/2613-self-gated-composition-13m/run.py --smoke

# 等价直调（run.py 只是薄封装）
python3 train_jumphead.py \
  --config axes/6-predict/6.2-speculative-decoding/experiments/2613-self-gated-composition-13m/config.json \
  --out-dir axes/6-predict/6.2-speculative-decoding/experiments/2613-self-gated-composition-13m \
  --composition
```
