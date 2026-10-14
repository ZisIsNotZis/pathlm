# 2651 · 权重量化最小判决 —— 权重清单（`MODEL.md`）

契约：同目录 `README.md` §2/§9 + `config.json::checkpoints`（运行时由 `run.py` 用 `2621` 的载入器
断言 sha256）。本实验**不训练、不新增参数**（`params_added = 0`），只**只读**复用他目录的冻结
checkpoint；权重不入库（`.gitignore` 覆盖 `*.pt`）。下表哈希由本文件写入时**在磁盘上重算**
（`sha256sum`），与 `config.json::checkpoints.trunk.sha256` 逐位相同。

| 角色 | 路径 | 大小 | sha256（磁盘实测） | 生成命令 |
|---|---|---|---|---|
| 被测模型（`fp16` / `int8_per_channel` / `int8_group32` / `int4_group32` 四臂共用） | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` | 54 193 675 B | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` | `2610`：13.5M dense decoder-only（d256/L12/H8/mlp6/seq256，tied 读出、仅最终深度 CE），12k 步，clean bpc 1.483569 |

- **参数量账**（本实验的权重面，`wq_arms.byte_account` 逐字段可核）：

| 项 | 数量 | 说明 |
|---|---|---|
| 模型总参数 | 13 536 768 | `sum(p.numel() for p in model.parameters())` |
| `nn.Linear` 权重矩阵 | 13 369 344（60 个矩阵） | 48 个 block 矩阵（`blocks.{0..11}.{qkv,proj,mlp.0,mlp.2}`）+ 12 个 `gates.{0..11}` |
| `nn.Linear` 偏置 | 36 864 | 60 × `n_out`，**不量化**（fp16 存储） |
| 非 `nn.Linear` 参数 | 130 560 | `embed.weight` 52 736（tied 读出，**不量化**）+ `pos_embed.weight` 65 536 + 24 × LayerNorm 参数 12 288 |

- **`gates.*` 是死权重**：`DepthARConfig.dense=True + exit_depths=None` 的前向走
  `for blk in self.blocks`（`pathlm/depth_ar.py` 的 dense/sequential 分支），**从不读 `self.gates`**
  （门只在非 dense 的变深度/填充路径上用）。`wq_arms.dead_gate_probe` 用「把门权重与偏置清零 ⇒
  logits 逐位不变」证明，并把还原也逐位验证（`results.json::instrument.dead_gate_probe`）。
  ⇒ 主口径 `scope_linear_all`（60 个矩阵）把死权重也计进字节账（部署上它们确实要被存），
  但**质量读数只由 48 个活矩阵决定**；`scope_linear_live` 是旁列。
- 复算命令：`sha256sum axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`
  （本波实测与上表逐位相同；`run.py` 每次运行再断言一次）。
- 数字与文件的对应：`results.json::checkpoint` 记路径/哈希/体积；`results.json::code` 记产生该结果的
  代码文件 sha256 前 16 位（单一权威：本文件不复列）。
