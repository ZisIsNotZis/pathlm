# 2642 · 跨家族同字节预算重画（块标尺口径）—— 权重清单（`MODEL.md`）

本实验**不做前向、不训练、不新增参数**（`params_added = 0`）。它只**读**四份 checkpoint 的
`state_dict` 以取得**参数量**与 `n_kv_heads`（`torch.load(..., map_location="cpu")`，只读、不改、
不写出）。权重文件不入库（`.gitignore` 覆盖 `*.pt`）。

下表哈希由本文件写入时在磁盘上重算（`sha256sum`），与 `2632-kv-block-scale/MODEL.md` 的记录逐位相同
（`results.json::inputs_sha256` 已把四条一并预登记并在运行时校验）。

| 角色 | 路径 | 大小（B） | 参数量（state_dict 求和） | n_kv_heads | sha256（磁盘实测） |
|---|---|---|---|---|---|
| trunk 族被测模型（本实验只取其参数量） | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` | 54 193 675 | 13 536 768 | 8 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 头族 MHA 基线 | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/mha-12k-s0/model.pt` | 54 193 675 | 13 536 768 | 8 | `557453eb12b56a453b29ea279cdbbc0969cfa6bf0dbe1cdbd9965cd6cec62528` |
| 头族 GQA | `.../2616-gqa-mqa-13m/runs/gqa-12k-s0/model.pt` | 49 456 651 | 12 352 512 | 2 | `bd682447c90bc118253a1496b4241bdc711a4103ab8df06f8296a840e7b3bc00` |
| 头族 MQA | `.../2616-gqa-mqa-13m/runs/mqa-12k-s0/model.pt` | 48 667 147 | 12 155 136 | 1 | `24bdfe7f1549107175e9343444372a6abe3aba7c6ba2423e8ab069a145b1ea97` |

- 架构（四份相同，只有 `n_kv_heads` 不同）：d_model 256、L 12、H 8、head_dim 32、mlp 6×、tied 读出、
  seq 256；融合 `qkv.weight` 行数 = `n_heads·head_dim + 2·n_kv·head_dim`（MHA 768 / GQA 384 / MQA 320），
  由此推 `n_kv`。
- 参数量与 `n_kv` 的解析关系：`params(MHA) − params(n_kv) = L · (2·d_model + 2) · (H − n_kv) · head_dim`
  ⇒ GQA 1 184 256（8.75%）、MQA 1 381 632（10.21%）；解析与磁盘实测**逐位一致**。
- 复算命令：`sha256sum <上表四个路径>`；参数量：`python3 -c "import torch;sd=torch.load(P,map_location='cpu');print(sum(v.numel() for v in sd.values()))"`。
- 本实验的结果数字与代码的对应：`results.json::code` 记 `frontier_ledger.py` 的 sha256。
