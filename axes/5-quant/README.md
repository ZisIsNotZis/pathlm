# 5 量化轴

> 轴：**5 量化轴**（目录 `axes/5-quant/`）。用词、编号、目录以 `docs/structure.md` §12 为准，
> 每个研究项的字段契约见 `docs/research-schema.md`。

量化轴问「用更低的数值精度换更少的内存/带宽/算力」：普通 decoder-only transformer 的权重、KV cache、
中间激活一律 fp16/bf16，长上下文推理被这些张量的**字节数**（显存与访存带宽）卡住。量化的做法是把
某个张量以更低位宽存放、用时反量化，把「每位置/每参数字节」压下来。本轴分六件事：**KV 量化**（5.1）、
**权重量化**（5.2）、**激活量化**（5.3）、**低秩压缩（MLA 式）**（5.4）、**fp8 / 大 head_dim**（5.5）、
**组合：窗口 + 量化 + 检索**（5.6）。目标：用更少的字节拿到**序列级**类似效果；代价口径 = 等效理论
时间/计算量（FLOPs / layer-passes / tokens-per-forward，考虑 pipelining），**不用墙钟**，字节口径用
**解析每值/每位置字节（含标尺与 staging）**，不用 GPU 实测峰值（评测期实现立即反量化，峰值不随档下降）。

> **前沿口径与口径纪律**：量化的正解不是「掉不掉质量」的单点，而是**质量-代价前沿**——以同尺寸
> stock fp16 全因果基线为 100%，横轴 = 每位置/每参数解析字节，纵轴 = 序列级质量（clean bpc），
> 必要时加第二腿（合成长程召回）。跨家族比较**必须**统一到同一「每位置字节（含标尺）」横轴，
> **禁止**用「比特数 / 压缩率 / 头数」直接互比（`SCHEMA.md` §17.12）；单点 Δbpc 必须越 0.014 噪声带
> 才算可排序。**特别地**：旧格曾用自造的逐值标尺把「量化省一半」反算成「量化占内存」，该口径已按
> 块标尺（沿 `head_dim` 摊标尺）撤回（55-review W-08 / R8），引用旧逐值标尺数字即口径混用。

## 本目录的子树

| 编号 | 名称 | 目录 | 粒度 |
|---|---|---|---|
| 5.1 | KV 量化 | [`5.1-kv-quant`](5.1-kv-quant/) | 研究项 |
| 5.2 | 权重量化 | [`5.2-weight-quant`](5.2-weight-quant/) | 研究项 |
| 5.3 | 激活量化 | [`5.3-activation-quant`](5.3-activation-quant/) | 研究项 |
| 5.4 | 低秩压缩（MLA 式） | [`5.4-low-rank`](5.4-low-rank/) | 研究项 |
| 5.5 | fp8 / 大 head_dim | [`5.5-fp8-head-dim`](5.5-fp8-head-dim/) | 研究项 |
| 5.6 | 组合：窗口 + 量化 + 检索 | [`5.6-window-quant-retrieval`](5.6-window-quant-retrieval/) | 研究项 |

## 边界

- **旧格 `R1-quantization` → `5.1`（MAPPING §1）**：R1 是「量化」总目，按 `docs/structure.md` §5 拆为
  5.1（KV 量化，本仓有实测）/ 5.2（权重量化）/ 5.3（激活量化）/ 5.5（fp8 / 大 head_dim，承接 R1 的
  fp8 质量与大 head_dim 缺口）；R1 的六个 kv 实验目录整搬 5.1（一格一份家，不复制）。
- **旧格 `R4-window-kv-quant-retrieval` → `5.6`（MAPPING §1）**：R4 的实验 `2617-prior-art-hybrid-kv`
  整搬 5.6。
- 与 `axes/7-references/` 的边界：`量化基线`、`GQA/MQA` 是必须打败的参照线台账（`axes/7-references/README.md`），
  不立项；本轴立项的是量化的**机制与前沿**。`5.4` 低秩压缩（MLA 式）与 `7-references` 的头共享（GQA/MQA）
  是两种不同的省 KV 机制，由 integrator 核不重复。
- 与 `1.1.4.2` 压缩（摘要槽 / 低秩投影 / pooling）的边界：`1.1.4.2` 是信息右移的压缩（要**保住信息**），
  本轴量化/低秩是省内存的压缩（可接受有损），记账与判据不同。

## 迁移说明

按 `.scratch/56-restructure/MAPPING.md`：旧格 `R1-quantization` 的六个 kv 实验目录（`2614`/`2619`/`2620`/`2621`/`2631`/`2632`）
整搬 `5.1/experiments/`，`2617-prior-art-hybrid-kv` 整搬 `5.6/experiments/`（整目录 `git mv`，一格一份家、
不复制）；`.scratch/**/evidence` 原地未动，仅更新指针。按 55-review 修正「作废的 int4 数字」（W-08：逐值标尺
口径作废，改用块标尺）、「跨口径混用」（R8：每位置字节须含标尺且统一轴）。旧轴目录 `axes/R-references/**`
及其 README、`PRIOR_ART.md` 待 integrator 统一退役。

## 怎么读

- 状态与全表：`python3 tools/summary.py`；校验：`python3 tools/summary.py --check`。
- `status` 仅四值：未开始 / 进行中 / 已完成 / 被否证；「做到多少」写在 `conclusion` / `gaps`。
