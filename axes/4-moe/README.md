# 4 专家轴（MoE）

> 轴：**4 专家轴（MoE）**（目录 `axes/4-moe/`）。用词、编号、目录以 `docs/structure.md` §12 为准，
> 每个研究项的字段契约见 `docs/research-schema.md`。

专家轴问「FFN 的容量与每 token 计算怎么解耦」：普通 decoder-only transformer 的 FFN 是 dense 的——
每个 token 都要过全部 FFN 参数，于是「模型容量（参数量）」与「每 token 计算量」被绑死，想扩容量就得
扩每 token FLOPs。MoE 的做法是让每个 token 只过少数专家（top-k 路由），把**总参数（容量）**与
**每 token 激活计算**解耦。本轴把这件事拆成三个独立问题：**稀疏激活专家本体**（4.1）、
**路由与负载均衡**（4.2）、**专家粒度 / 异构**（4.3）。目标：用更少的计算量拿到**序列级**类似效果；
代价口径 = 等效理论时间/计算量（FLOPs / layer-passes / tokens-per-forward，考虑 pipelining），**不用墙钟**。

> **前沿口径**：专家轴的正解不是「MoE 有没有用」的单点，而是**质量-代价前沿**——以同尺寸 stock
> dense（等每 token 激活算力）为 100%，横轴 = 每 token 等效算力（FLOPs）/ 总参数量，纵轴 = 序列级
> 质量（clean bpc）。各研究项都必须给**等 FLOPs / 等参数对照**，并报**活性**（专家使用率分布，
> 防坍缩），否则会把「参数记账」误读成「容量增益」。

## 本目录的子树

| 编号 | 名称 | 目录 | 粒度 |
|---|---|---|---|
| 4.1 | 稀疏激活专家（top-k 路由） | [`4.1-sparse-experts`](4.1-sparse-experts/) | 研究项 |
| 4.2 | 路由与负载均衡 | [`4.2-routing-balance`](4.2-routing-balance/) | 研究项 |
| 4.3 | 专家粒度 / 异构（细粒度、共享专家） | [`4.3-expert-granularity`](4.3-expert-granularity/) | 研究项 |

## 边界

- **旧格 `W2-expertization` 归本轴（MAPPING §1）**：它是「专家化（MoE）」的手段清单，按
  `docs/structure.md` §4 拆为 4.1（top-k 稀疏激活本体）、4.2（负载均衡/路由坍缩）、4.3（细粒度/共享专家）。
  W2 无自有 `experiments`（MAPPING §1 证据去向「无自有 experiments」）；原 `R6-moe-scale` 已并入
  `archive/dropped/axes/R-references/R6-moe-scale/`。本轴不接收文件。
- 与 `3.1.5` 通道极化 / 稀疏激活的边界：`3.1.5` 是**通道粒度**、本轴是**专家（FFN 块）粒度**的
  按 token 激活；记账口径不同，由 integrator 核不重复。
- 与 `axes/7-references/` 的边界：MoE 是业界已做透的成熟路线，本轴三项**最高风险是造轮子**
  （55-review W-03/W-04）；每项 `gap` 必须显式声明「相对 Switch/Mixtral/DeepSeekMoE 的增量」，
  否则须降为参照。

## 迁移说明

按 `.scratch/56-restructure/MAPPING.md`：旧格 `W2-expertization` 归本轴，拆为 4.1（迁移）/ 4.2（新建）/
4.3（新建）。按 55-review W-13 消除旧格「`status: todo` 与 `prior_art_verdict` 检索不可用」的自相矛盾口径；
按 W-17 成本口径改为等效理论量；按 W-03/W-04 不照抄「todo 待补做」的立项语气。旧轴目录
`axes/W-width/**` 及其 README、`PRIOR_ART.md` 待 integrator 统一退役。

## 怎么读

- 状态与全表：`python3 tools/summary.py`；校验：`python3 tools/summary.py --check`。
- `status` 仅四值：未开始 / 进行中 / 已完成 / 被否证；「做到多少」写在 `conclusion` / `gaps`。
