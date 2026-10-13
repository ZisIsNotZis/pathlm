# ticket 57 · 研究树程序（program）· milestone 与轴 lead 契约骨架

> 性质：**编排骨架**（程序级），不是研究结论台账。用词以 `docs/structure.md` 为准。
> 本文件只定义「程序有哪几个 milestone」「每个轴 lead 的契约长什么样」，**不重复**各研究项内容。

## 0. 程序的最终目标

把「LM 算力路径设计空间」按 `docs/structure.md` 的树逐格做完：每个研究项一份
`project.yaml`（目的 / 创新 / SOTA 基线 / 端到端验证 / **单一 milestone** / work packages /
状态），并用 `tools/summary.py` 一眼看全树状态。

## 1. 程序级里程碑（用户同步点）

| # | 里程碑 | 交付物 | 验收 | 同步点 |
|---|---|---|---|---|
| **M1** | **规范冻结（foundation）** | `docs/structure.md`（树+粒度+目录）、`docs/research-schema.md`、`MAPPING.md`、`tools/summary.py`、`tools/check_tree.py`、exemplar `project.yaml` | 本 ticket 56 §3 的 8 条 | **本 WP 交付**（出资人/复核确认规范） |
| **M2** | **逐轴迁移完成** | 每个研究项一份合格 `project.yaml`；旧证据按 `MAPPING.md` 归位；旧 wiki 退役落 `archive/wiki-legacy/` | `python3 tools/summary.py --check` 全绿；`find axes -type f｜wc -l` ≥ 2434；无孤儿旧格 | 各轴 lead 交付后合入 |
| **M3** | **集成验收** | 全树 `summary.py` 表 + 抽检复核 | 独立复核通过；`pytest tests/ -q` 通过；跨轴一致性无矛盾 | 出资人验收 |

## 2. M1 边界（本 WP 只做这些，不做逐轴迁移）

**做**：冻结规范（树/粒度/目录/schema）、产出校验与汇总工具、冻结旧→新映射与证据归属、
提供 exemplar、消除新结构与旧文档的矛盾、记录工单。
**不做**：逐轴创建全部 `project.yaml`、大规模移动 `axes/**` 证据文件（那是 M2，各轴 worker）。
**判据**：见 ticket 56 §3 验收 1–8。

## 3. 轴 lead 契约骨架（每个轴一份；M2 派发时填写）

```yaml
axis_lead:
  axis: "1 时间轴"                 # 用 docs/structure.md 的 exact wording
  dir: "axes/1-time"
  owner: "<轴 lead>"
  writer: "<单一写入者>"
  reviewer: "<fresh 复核者，不得为 writer>"
  integrator: "<研究树 integrator>"
  scope: "按 docs/structure.md §12 本轴的全部节点；按 MAPPING.md 迁入证据"
  deliverables:                    # 逐研究项
    - "1.1.1-hard-window/project.yaml（迁移 A3/1.1.1 主家）"
    - "1.1.3-visibility-cap/project.yaml（A3 拆分；指针 1.1.1）"
    - "…（本轴全部研究项）"
  interfaces:
    inputs: "本轴旧格目录 + .scratch/**/evidence + catalog.yaml 的 local 块（内容源）"
    outputs: "axes/1-time/**/project.yaml + 证据归位"
    shared: "docs/structure.md / docs/research-schema.md / MAPPING.md（只读，不改）"
  acceptance:
    - "每个研究项 project.yaml 过 tools/check_tree.py（字段/milestone 单数/role_map/cost_model 口径/证据）"
    - "证据按 MAPPING.md 归位，无丢失、无重复复制"
    - "旧残留（*.md.md / 旧引擎引用 / 死数据）清零"
  limits: "finite budget；复核按 SCHEMA.md §14/§14b 排在执行之后"
```

## 4. 复核与集成

- 每个轴交付后**独立复核**（fresh 上下文，`SCHEMA.md` §14/§14b），复核通过才算该轴完成。
- integrator 负责 M2→M3 的**合并与跨轴一致性**（用词、id 唯一、证据不重复、引用完整）。
- 出货前跑 `python3 tools/summary.py --check && python3 -m pytest tests/ -q`。
