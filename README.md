# Research Wiki — LM 算力路径设计空间的完备目录 + 单实验 Playbook

> 一句话：这里不是「一个模型」，而是**一张地图**——把语言模型的「算力路径设计空间」
> 按轴切开，每格一个课题，课题里记录**谁试过、结果如何、为什么失败、怎么复现**。
> 目录是完备的（每格都有页），内容是渐进的（大量格子仍是空白，等着被填）。
> 起点来自 PathLM 项目（25 个执行 ticket）的实测资产，见 `archive/`。

## 这是什么 / 不是什么

**是**：

- 一棵**可机检的研究树**（`docs/structure.md` + `docs/research-schema.md`）：每个研究项一份
  `axes/**/project.yaml`（目的 / 创新 / SOTA 基线 / 端到端验证 / 单一 milestone / work packages /
  状态）；`tools/summary.py` 一条命令出全表，`tools/check_tree.py` 让格式/证据错误直接 fail。
- 一份**失败路径的账本**：不只写什么有效，更写**试过什么、为什么失败、排除了什么
  替代解释**——这是本项目最贵的资产。
- 一套**共享天平**（`docs/methodology.md`）：三列账本、方差界、口径标注、双货币记账、
  冒烟 + 变异测试。

**不是**：

- **不是组合实验库**：默认只做单轴单格实验，基底恒为 stock 对照；组合**默认留白**
  （是显式声明，不是遗漏）——见 `SCHEMA.md` §4。
- **不是排行榜**：这里比较的是「同一格子内谁更值得做」，不是「哪个模型更强」。
- **不能只看结论**：所有数字都带口径（快照 / 渐近 / matched-step / 墙钟 / FLOPs），
  跨口径比较会得出相反结论。

已测的总规律见 `docs/playbook.md`；设计契约与共享底座见 `docs/design.md`；研究树布局见
`docs/structure.md`，研究项格式见 `docs/research-schema.md`，每项当前状态见 `tools/summary.py`
的输出（旧的 `INDEX.md` / `catalog.yaml` 已退役）。

## 怎么读

| 想做什么 | 去哪 |
|---|---|
| 一眼看**哪里是空白 / 每项状态** | **`python3 tools/summary.py`**（研究树状态全表） |
| 看**研究树布局 / 用词 / 粒度 / 目录** | `docs/structure.md` |
| 看**研究项格式契约**（`project.yaml` 字段） | `docs/research-schema.md` |
| 接手/复现**一个研究项** | `axes/<轴>/<研究项>/project.yaml` → 同目录 `experiments/` |
| 学**方法论** | `docs/methodology.md` |
| 查**已验证的定律** | `docs/playbook.md` |
| 看**核心设计（RAP）与共享底座** | `docs/design.md` |
| 看**引擎能力边界** | `docs/engine_capability.md` |
| 知道**旧项目发生了什么** | `archive/README.md` → `archive/legacy-docs/retrospective.md` |

轴的记号：`1` 时间 · `2` 深度 · `3` 宽度 · `4` 专家（MoE） · `5` 量化 · `6` 预测 ·
`7` 不研究（只当对照组） · `8` 工程纪律 · `9` Backlog。

## 贡献与复现

欢迎 PR（补一个空课题、加一次实验、修正一个结论）。硬性规则与流程见 `CONTRIBUTING.md`；
格式契约与分析纪律见 `SCHEMA.md`；仓库布局见 `docs/design.md` §4。

数据 enwik8 自动下载；权重 `*.pt` 不入库（派生物，按实验契约的命令重训）。
单卡 RTX 4090 24GB 包络：13M ≈ 5 min/6000 步、单臂全电池 11–30 min；100M ≈ 32–70 min/臂。
本仓库是**本地工作副本**：提交均在本地，未 push 到远端。

## 语言与许可

中文为主（面向中文读者），代码标识符/日志/配置键保持英文；一个文档内不中英交替。
许可：AGPL-3.0（见 `LICENSE.md`）。
