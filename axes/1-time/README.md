# 1 时间轴

> 轴：**1 时间轴**（目录 `axes/1-time/`）。用词、编号、目录以 `docs/structure.md` §12 为准，
> 每个研究项的字段契约见 `docs/research-schema.md`。

一次前向里，一个 token「能看到谁、怎么用」。目标：用更少的计算量拿到**序列级**类似效果；
代价口径 = 等效理论时间/计算量（FLOPs / layer-passes / tokens-per-forward，考虑 pipelining），**不用墙钟**。

## 本目录的子树（§1.1 可见范围）

| 编号 | 名称 | 目录 | 粒度 |
|---|---|---|---|
| 1.1 | 可见范围（窗口族） | [`1.1-visibility`](1.1-visibility/) | 分组 |
| 1.1.1 | hard 窗口 | [`1.1.1-hard-window`](1.1.1-hard-window/) | 研究项 |
| 1.1.2 | soft 衰减 | [`1.1.2-soft-decay`](1.1.2-soft-decay/) | 研究项 |
| 1.1.3 | 可见量上限 | [`1.1.3-visibility-cap`](1.1.3-visibility-cap/) | 研究项 |
| 1.1.4 | 信息右移 | [`1.1.4-right-shift`](1.1.4-right-shift/) | 分组 |
| 1.1.4.1 | 显式记忆槽 / 线性状态 / 递归状态（delta 规则族：DeltaNet / GLA / Mamba-2） | [`1.1.4.1-explicit-memory`](1.1.4.1-explicit-memory/) | 研究项 |
| 1.1.4.2 | 压缩：摘要槽 / 低秩投影 / pooling | [`1.1.4.2-compression`](1.1.4.2-compression/) | 研究项 |
| 1.1.4.3 | 压缩触发与优先级 | [`1.1.4.3-compression-trigger`](1.1.4.3-compression-trigger/) | 研究项 |
| 1.1.4.4 | 外部检索 | [`1.1.4.4-external-retrieval`](1.1.4.4-external-retrieval/) | 研究项 |
| 1.1.4.5 | implicit（掩码/窗口，已证退化）vs **explicit**（有希望的方向） | [`1.1.4.5-implicit-vs-explicit`](1.1.4.5-implicit-vs-explicit/) | 研究项 |

## 本目录的子树（§1.2 可见结构）

| 编号 | 名称 | 目录 | 粒度 |
|---|---|---|---|
| 1.2 | 可见结构（非均匀的可见模式） | [`1.2-visibility-structure`](1.2-visibility-structure/) | 分组 |
| 1.2.1 | dilated / 空洞 | [`1.2.1-dilated`](1.2.1-dilated/) | 研究项 |
| 1.2.2 | 块稀疏 / 条带 / 局部+全局混合（Longformer 式） | [`1.2.2-block-sparse`](1.2.2-block-sparse/) | 研究项 |
| 1.2.3 | 内容选择 top-k | [`1.2.3-content-topk`](1.2.3-content-topk/) | 研究项 |
| 1.2.4 | 动态合并：越往下层 token 越少 | [`1.2.4-dynamic-merge`](1.2.4-dynamic-merge/) | 研究项 |

## 前提（§1.3 寻址，参照）

| 编号 | 名称 | 粒度 | 家 |
|---|---|---|---|
| 1.3 | 寻址（前提，不是手段选择） | 参照 | `axes/7-references/README.md` |

**1.3 寻址**是**前提**，不是可独立收口的研究项（`docs/structure.md` §13.4）：位置语义（绝对 / 相对 /
RoPE / 无位置）与寻址方式（按内容 vs 按位置）作为参照登记在 `axes/7-references/README.md`，本轴
不写 `project.yaml`。旧格 `axes/A-mixing/A8-positional-semantics`（位置语义）同降为参照（MAPPING §1）。

## 怎么读

- 状态与全表：`python3 tools/summary.py`；校验：`python3 tools/summary.py --check`。
- 每项有且仅有一个 `milestone`（最终输出 + 验收），推进过程拆成 `work_packages`，每条自带 acceptance 与独立 `review`。
- `status` 仅四值：未开始 / 进行中 / 已完成 / 被否证；「做到多少」写在 `conclusion` / `gaps`。
- 旧格（`axes/A-mixing/**`、`axes/M-carrier/**` 等）及其 README 为**上一版结构**，证据已按 `.scratch/56-restructure/MAPPING.md` 迁入本目录 `experiments/`，旧目录待统一退役。
