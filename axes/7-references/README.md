# 7 不研究（只当对照组）· 参照台账

> 轴 7 是**参照台账**：它们是拿来当跑分线的「巨人」，**不立项、不写 `project.yaml`**
> （`docs/structure.md` §12 轴 7 / §13.4）。用词以 `docs/structure.md` 为准。
> 台账由各轴 worker 按 `.scratch/56-restructure/MAPPING.md` §4 登记；每个参照只记录
> 「是什么 / 拿它当什么跑分线 / 出处 / 证据落点」。

## 登记表

| 参照（exact wording） | 登记者（所属轴） | 当什么跑分线 | 出处 / 证据落点 |
|---|---|---|---|
| 全注意力基线 | 轴 A / 轴 7 | 质量锚点与 100% 归一基线 | `axes/7-references/`（A2 整搬）；`.scratch/04-m1-runs/evidence/B0` 留原地作锚；13M B0 = 1.5074 bpc 快照 |
| ALiBi | 轴 1（1.1.2） | 软衰减手段（已并入研究项，不在本台账单列） | 见 `axes/1-time/1.1.2-soft-decay/` |
| RoPE | 轴 A / 轴 7 | 位置语义族（长上下文工程前提） | A8 → `axes/7-references/`；`.scratch/07-x1-x3/evidence` |
| GQA/MQA | 轴 W / 轴 7 | KV 头共享（业界已做透） | `2616-gqa-mqa-13m`（W3 整搬 `axes/7-references/`） |
| 量化基线 | 轴 R / 轴 5 | 低比特对照 | 见 `axes/5-quant/`（5.1–5.6） |
| CALM/LayerSkip/MoD | 轴 R / 轴 7 | 既有早退实现 | `2623-standard-early-exit-13m`（R7 整搬；同时作 `2.1.1` 对照指针） |
| 自回归（一次一个位置） | 轴 6（6.1） | 预测轴参照 | 见 `axes/6-predict/6.1-autoregressive/` 与 `6.2` |

## 轴 1 的参照（前提）

| 参照（exact wording） | 所属轴 | 粒度 | 说明 |
|---|---|---|---|
| 寻址（前提，不是手段选择） | 轴 1（1.3） | 参照 | 位置语义（绝对 / 相对 / RoPE / 无位置）与寻址方式（按内容 vs 按位置）是**前提**，不是可独立收口的研究项（`docs/structure.md` §13.4）；本轴不写 `project.yaml`。与本表的「RoPE」族同源。 |

> 登记备注：本文件按 `docs/structure.md` §12 轴 7 建。上表「全注意力基线 / RoPE / GQA/MQA /
> CALM/LayerSkip/MoD」的证据整搬与台账登记由对应轴 worker 执行（MAPPING §4）；轴 1 的 `1.3 寻址`
> 由轴 1 worker 登记（本行）。旧格 `axes/A-mixing/A2-full-attention`、`A8-positional-semantics`、
> `axes/W-width/W3-kv-head-sharing`、`axes/R-references/R7-existing-early-exit` 目录待统一退役。
