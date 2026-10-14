# MODEL — 2660-latentreg-vs-disc-13m

权重不入库（`.gitignore` 覆盖 `*.pt`）。本实验**零新增主干参数**：头是新训的
`pathlm.model.TransformHead(d)`，主干与 tied 读出完全冻结。

## 1 冻结主干（只读）

- 目录：`../6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s1`
- `model.pt` sha256：`047893bda15104cc25794dc598d96bca96bbef7449bd5d45518e78e205a31b59`
- tied 读出（embed.weight）sha256：`da5b5d3343655963d2ec56c408a9cae084e18b7f2bb382c6872c310da79fbb29`（两臂共用同一份，L0 硬门）

## 2 训练好的头（阶段 2 直接加载）

| 臂 | 种子 | k | λ | 文件 | sha256 |
|---|---|---|---|---|---|
| `disc_k2` | 0 | 2 | 0.0 | `heads/disc_k2_s0.pt` | `6e5c2d5c5f440f392b42ed29e779279b5e1b7f043a700a20c37ddf4648552fba` |
| `disc_k2` | 1 | 2 | 0.0 | `heads/disc_k2_s1.pt` | `4d2f8efb888a3aab1c729a9654ea2ad57244b4821b9ac9760e746606570c6c7e` |
| `lat_k2_lam0` | 0 | 2 | 0.0 | `heads/lat_k2_lam0_s0.pt` | `6a986b3b7a9da3718b0a0ed8eadd234a2fa7e88521558c94e3d99baa28220a44` |
| `lat_k2_lam0` | 1 | 2 | 0.0 | `heads/lat_k2_lam0_s1.pt` | `a29ac4ecf07113f8c004bcc3154594edb08182bacdea4095c27404a837385d27` |
| `lat_k2_lam1` | 0 | 2 | 1.0 | `heads/lat_k2_lam1_s0.pt` | `7a67a7c64a8f692b6cbcd7c03efc9bba88c2404896bed598cd7a6a8cd289bbaf` |
| `lat_k2_lam1` | 1 | 2 | 1.0 | `heads/lat_k2_lam1_s1.pt` | `e0a508a3bfe1bc53ade21513fb45a3ccc27c75c4452da13010f81f5c5b56fb70` |

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
