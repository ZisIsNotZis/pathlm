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
| `disc_k4` | 0 | 4 | 0.0 | `heads/disc_k4_s0.pt` | `1001cd970e70017914edd0e1d720d962f47e359f22eebd0f1f74db74d6010a6c` |
| `disc_k4` | 1 | 4 | 0.0 | `heads/disc_k4_s1.pt` | `e4ea073f1a3932cfa2e716ad713b641ad000be7845c7919706d98b1ebe7ce928` |
| `lat_k4_lam0` | 0 | 4 | 0.0 | `heads/lat_k4_lam0_s0.pt` | `8dbc0a0d0cfe951243d044a351ebb00bde5e2748e045d8bb02ed83ad3a24f562` |
| `lat_k4_lam0` | 1 | 4 | 0.0 | `heads/lat_k4_lam0_s1.pt` | `a11bfd59d5c5363f89979f3a0c1393b61467bbf99cd87058b53233b1f01f56aa` |
| `lat_k4_lam1` | 0 | 4 | 1.0 | `heads/lat_k4_lam1_s0.pt` | `8b4a8bcf6f85f96b5903f154a21d7dc161f7f8ee5afd9a439a15421de0f9494d` |
| `lat_k4_lam1` | 1 | 4 | 1.0 | `heads/lat_k4_lam1_s1.pt` | `d32d017f6e4c1731daf2219b10394d0b0b07e0920461d7a1ddb693aefe5e003c` |

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
