# 7 不研究（只当对照组）· 参照台账

> 轴：**7 不研究（只当对照组）**（目录 `axes/7-references/`）。用词、编号、目录以
> `docs/structure.md` §12 为准；每个研究项的字段契约见 `docs/research-schema.md`。
> 轴 7 是**参照台账**（只写本 `README.md`）：这些是拿来当跑分线的「巨人」，
> **不立项、不写 `project.yaml`**（`docs/structure.md` §12 轴 7 / §13.4）。
> 每条参照登记「是什么 / 当什么跑分线（要打败它什么）/ 出处 / 证据落点」。

## 本轴参照（`axes/7-references/`）

| 参照（exact wording） | 是什么 | 当什么跑分线（要打败它什么） | 证据落点 |
|---|---|---|---|
| 全注意力基线 | 无可见性约束的**全因果 softmax 注意力**，stock decoder-only transformer 本身 | **质量锚点与 100% 归一基线**：一切省算力 / 省内存 / 信息右移手段的对照分母；手段必须在同预算下质量不越本格噪声带（`SCHEMA.md` §10–§17） | [`experiments/A2-full-attention/`](experiments/A2-full-attention/)（旧格 A2 整搬）；13M B0 = **1.5074 bpc**（1.1 epoch 快照）/ 1.3459（4.4 epoch 渐近），100M B0 = 1.4612（1.09 epoch）；`.scratch/04-m1-runs/evidence/B0 @e38349f` 留原地作锚 |
| RoPE（位置语义族） | 旋转位置编码与长度外推一族（RoPE / 插值 YaRN·NTK / 无位置） | **长上下文工程前提**：绝对位置嵌入是硬上界（解码总长 ≤ seq_len），任何长上下文线必须先对齐 / 超越位置语义族的长度泛化 | [`experiments/A8-positional-semantics/`](experiments/A8-positional-semantics/)（旧格 A8 整搬，本仓零经验）；`.scratch/07-x1-x3/evidence @7b47555` |
| GQA/MQA（KV 头共享） | 减少 KV 头数以省推理显存 / 带宽（MQA 1 头 / GQA G 头 / MLA 低秩） | **KV 预算轴的同字节对手**：窗口（`1.1.1`）/ 量化（`5.1`）/ 低秩（`5.4`）等手段必须在同一「每位置 KV 字节」预算下质量不输 GQA/MQA | [`experiments/W3-kv-head-sharing/2616-gqa-mqa-13m`](experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/)（旧格 W3 整搬）；13M clean bpc：mha **1.4820** / gqa **1.4963**（+0.0143；同 seed 重跑 +0.0139 **跨门槛** ⇒ 依预登记规则该臂判读作废）/ mqa **1.4938**（+0.0118，与 GQA 差 0.0025 ≪ 0.014 不可互排）；解析 KV 内存 12288 → **3072 B/位置（GQA −75%）** / **1536 B（MQA −87.5%）** |
| CALM/LayerSkip/MoD（既有早退实现） | 三种标准深度自适应实现：置信门控早退（CALM）/ 自投机早退（LayerSkip）/ 逐 token 路由（MoD） | **深度轴与预测轴要打败的基线**：逐 token 早退（`2.1.1`）与投机解码（`6.2`）必须在同算力口径（layer-passes / tokens-per-forward）下质量 ≥ 这些标准实现 | [`experiments/R7-existing-early-exit/2623-standard-early-exit-13m`](experiments/R7-existing-early-exit/2623-standard-early-exit-13m/)（旧格 R7 整搬）；13M：CALM 原式口径退到 **3.43/8 层**（一致率 0.959），LayerSkip 窗口接受率 **0.9109**（条件口径 0.9520），MoD isoFLOP **+0.398 bpc** |

## 他轴登记在本台账的参照（前提）

这些参照由对应轴 worker 登记，本轴只保留指针（`docs/structure.md` §13.4）：

| 参照（exact wording） | 所属轴 | 粒度 | 家 / 说明 |
|---|---|---|---|
| ALiBi | 轴 1（1.1.2 soft 衰减） | 手段（已并入研究项） | 见 [`axes/1-time/1.1.2-soft-decay/`](../1-time/1.1.2-soft-decay/)；ALiBi 是「软衰减」手段，不是本台账单列条目 |
| 量化基线 | 轴 5 | 手段（已并入研究项） | 见 [`axes/5-quant/`](../5-quant/)（`5.1`–`5.6`）；低比特对照在各量化研究项内，不单列参照 |
| 寻址（前提，不是手段选择） | 轴 1（1.3） | 参照 | 位置语义（绝对 / 相对 / RoPE / 无位置）与寻址方式（按内容 vs 按位置）是**前提**，不是可独立收口的研究项；本轴不写 `project.yaml`、不建目录。与上表「RoPE」族同源 |
| 自回归（一次一个位置） | 轴 6（6.1） | 参照 | 逐 token 贪心 = 全因果 stock 的输出流，是 `6.2` / `6.3` / `6.4` 的对照基线（归一本格 100%）；它是前提 / 对照，本轴不写 `project.yaml`、不建目录 |

## 迁移说明

按 `.scratch/56-restructure/MAPPING.md` §1 / §4：旧格 `A2-full-attention`、
`A8-positional-semantics`、`W3-kv-head-sharing`、`R7-existing-early-exit` 降为参照，
其 `experiments/**` 整目录 `git mv` 到本目录 `experiments/<旧格名>/`（一格一份家，只搬不复制）；
`.scratch/**/evidence` 原地不动，只作锚点。旧格目录及其 README 仍保留在原处，待 integrator 统一退役。
`R7` 的 `2623-standard-early-exit-13m` 同时是 `2.1.1`（early exit）与 `6.2`（投机解码）的对照指针。

## 怎么读

- 状态与全表：`python3 tools/summary.py`；校验：`python3 tools/summary.py --check`。
- 引用任何「税 / damage」前先确认对照口径（快照 / 渐近 / matched-step）与引擎是否同一代。
- 本轴无 `project.yaml`：参照不参与 `status` 枚举，也不计入里程碑。
