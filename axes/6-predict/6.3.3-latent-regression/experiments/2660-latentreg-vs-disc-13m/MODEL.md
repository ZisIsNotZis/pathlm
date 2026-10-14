# MODEL — 2660-latentreg-vs-disc-13m

权重不入库（`.gitignore` 覆盖 `*.pt`）。本实验**零新增主干参数**：头是新训的
`pathlm.model.TransformHead(d)`，主干与 tied 读出完全冻结。

## 1 冻结主干（只读）

- 目录：`../6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0`
- `model.pt` sha256：`1df7840bc4813f89709805222cedf34b0384f8bae0a04e23f9ffd2818b2d4e4d`
- tied 读出（embed.weight）sha256：`566ec50c9d4aecac6ab420dae3578236514eab426c66d548a35b04fca8ef8b3b`（两臂共用同一份，L0 硬门）

## 2 训练好的头（阶段 2 直接加载）

| 臂 | 种子 | k | λ | 文件 | sha256 |
|---|---|---|---|---|---|
| `disc_k2` | 0 | 2 | 0.0 | `heads/disc_k2_s0.pt` | `0254ea87cdb335ba48b39ce4c48b5a08f648515b06823dd235b31b61c307a9b6` |
| `disc_k2` | 1 | 2 | 0.0 | `heads/disc_k2_s1.pt` | `68fc3d575fec731a5f7aabc54909e296addec8584c81ae1efe8abfdc4f55ed0d` |
| `lat_k2_lam0` | 0 | 2 | 0.0 | `heads/lat_k2_lam0_s0.pt` | `0ace8fb6d5d55fa330bf448a66035a6f1dbc266f71e973e52cf075e0a6d052e9` |
| `lat_k2_lam0` | 1 | 2 | 0.0 | `heads/lat_k2_lam0_s1.pt` | `fa1168e7a4cf53530fb99d7f15cb782ddc6cdb22a8130eb0b74ca423df5c7adb` |
| `lat_k2_lam01` | 0 | 2 | 0.1 | `heads/lat_k2_lam01_s0.pt` | `ca960ebaa634014822120d5d60508d475dbef018896e4a0b32b5d763bbe6d575` |
| `lat_k2_lam01` | 1 | 2 | 0.1 | `heads/lat_k2_lam01_s1.pt` | `6e3b605c64f94438484ed8d4195b0790bb88fbec3774c27653feb3ba8a3fa19c` |
| `lat_k2_lam1` | 0 | 2 | 1.0 | `heads/lat_k2_lam1_s0.pt` | `b17d15980d76463acfc447575774310c78f332e1f61cc0d3b02a5e89d2a31756` |
| `lat_k2_lam1` | 1 | 2 | 1.0 | `heads/lat_k2_lam1_s1.pt` | `c42a98dd17799892d873c3db3ea326279f9efb754dbaea9e3b0ae2237ff86696` |

每个 `.pt` 是 `{head1: state_dict, head2: state_dict, ...}`（`TransformHead` 权重）。

> `.pt` 的 sha256 只标识**文件**（zip entry 含保存时刻，同内容也会不同），**不构成内容同一性证明**；
> 内容同一性以逐张量 `torch.equal` 为准（见 `results.json::L0.*.state_dict_all_tensors_equal`，L0 硬门）。
阶段 2 加载：`sd = torch.load('heads/<arm>_s<seed>.pt'); heads[j].load_state_dict(sd[f'head{j}'])`；
读出 = `heads[j](h_i) @ model.embed.weight.T`（同一份冻结 embed）。

## 3 复现命令

```bash
cd /home/z/vibe/pathlm
python3 -m pytest axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/test_latent_reg.py -q
python3 axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/run.py \
  --stage train --out <本目录> --ckpt axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0
```

## 4 补充运行的权重（P2 / P3；阶段 2 可直接加载）

### P2 · 主干 13m_s1

- ckpt `../6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s1`, sha256 `047893bda15104cc25794dc598d96bca96bbef7449bd5d45518e78e205a31b59`
- tied 读出 embed sha256 `da5b5d3343655963d2ec56c408a9cae084e18b7f2bb382c6872c310da79fbb29`

| 臂 | 种子 | k | λ | 文件 | sha256 |
|---|---|---|---|---|---|
| `disc_k2` | 0 | 2 | 0.0 | `second-backbone-13m_s1/heads/disc_k2_s0.pt` | `6e5c2d5c5f440f392b42ed29e779279b5e1b7f043a700a20c37ddf4648552fba` |
| `disc_k2` | 1 | 2 | 0.0 | `second-backbone-13m_s1/heads/disc_k2_s1.pt` | `4d2f8efb888a3aab1c729a9654ea2ad57244b4821b9ac9760e746606570c6c7e` |
| `lat_k2_lam0` | 0 | 2 | 0.0 | `second-backbone-13m_s1/heads/lat_k2_lam0_s0.pt` | `6a986b3b7a9da3718b0a0ed8eadd234a2fa7e88521558c94e3d99baa28220a44` |
| `lat_k2_lam0` | 1 | 2 | 0.0 | `second-backbone-13m_s1/heads/lat_k2_lam0_s1.pt` | `a29ac4ecf07113f8c004bcc3154594edb08182bacdea4095c27404a837385d27` |
| `lat_k2_lam1` | 0 | 2 | 1.0 | `second-backbone-13m_s1/heads/lat_k2_lam1_s0.pt` | `7a67a7c64a8f692b6cbcd7c03efc9bba88c2404896bed598cd7a6a8cd289bbaf` |
| `lat_k2_lam1` | 1 | 2 | 1.0 | `second-backbone-13m_s1/heads/lat_k2_lam1_s1.pt` | `e0a508a3bfe1bc53ade21513fb45a3ccc27c75c4452da13010f81f5c5b56fb70` |

- 完整读数：`second-backbone-13m_s1/results.json`；判词与口径：`second-backbone-13m_s1/analysis.md`

### P3 · k=4 · 主干 13m_s0

- ckpt `../6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0`, sha256 `1df7840bc4813f89709805222cedf34b0384f8bae0a04e23f9ffd2818b2d4e4d`
- tied 读出 embed sha256 `566ec50c9d4aecac6ab420dae3578236514eab426c66d548a35b04fca8ef8b3b`

| 臂 | 种子 | k | λ | 文件 | sha256 |
|---|---|---|---|---|---|
| `disc_k4` | 0 | 4 | 0.0 | `second-k4-13m_s0/heads/disc_k4_s0.pt` | `1001cd970e70017914edd0e1d720d962f47e359f22eebd0f1f74db74d6010a6c` |
| `disc_k4` | 1 | 4 | 0.0 | `second-k4-13m_s0/heads/disc_k4_s1.pt` | `e4ea073f1a3932cfa2e716ad713b641ad000be7845c7919706d98b1ebe7ce928` |
| `lat_k4_lam0` | 0 | 4 | 0.0 | `second-k4-13m_s0/heads/lat_k4_lam0_s0.pt` | `8dbc0a0d0cfe951243d044a351ebb00bde5e2748e045d8bb02ed83ad3a24f562` |
| `lat_k4_lam0` | 1 | 4 | 0.0 | `second-k4-13m_s0/heads/lat_k4_lam0_s1.pt` | `a11bfd59d5c5363f89979f3a0c1393b61467bbf99cd87058b53233b1f01f56aa` |
| `lat_k4_lam1` | 0 | 4 | 1.0 | `second-k4-13m_s0/heads/lat_k4_lam1_s0.pt` | `8b4a8bcf6f85f96b5903f154a21d7dc161f7f8ee5afd9a439a15421de0f9494d` |
| `lat_k4_lam1` | 1 | 4 | 1.0 | `second-k4-13m_s0/heads/lat_k4_lam1_s1.pt` | `d32d017f6e4c1731daf2219b10394d0b0b07e0920461d7a1ddb693aefe5e003c` |

- 完整读数：`second-k4-13m_s0/results.json`；判词与口径：`second-k4-13m_s0/analysis.md`
- 注：`lat_k4_lam0` 是 L0 等价自检**补充臂**（不在契约 §3 预登记臂表内；不改任何已登记臂定义），
  与 `disc_k4` 同代码路径 ⇒ 逐位相同（`results.json::L0`）。
