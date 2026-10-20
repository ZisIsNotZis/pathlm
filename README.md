# PathLM Research Wiki — 元仓库（研究目录 · 层契约 · 计划）

> **一句话**：这是一个**元仓库**，它不发布模型，发布的是**一棵可机检的研究树**——
> 把「语言模型的算力花在哪条路径上」这个设计空间按轴切开，一格一个研究项，
> 每格记下**谁试过、结果如何、为什么失败、怎么复现**。
> **元级**（本文件、`docs/`）只回答「树上有哪些研究项、各层拥有什么事实、下一步派什么活」；
> 读数、判据推导与逐格分析都在**研究级**（`axes/**`、`atlas/**`）与**工级**（`harness/**`）。
> **给谁看**：要改模型架构、想知道「这个改动大概让我掉多少质量、还是白赚」的人；
> 以及要往这棵树上派活、复核与接手的人。
> **不干什么**：不训练也不发布模型，不做排行榜，不做组合实验库。

## 1 三层：谁拥有什么

同一件事有且只有一个家，别的层只放指针。层契约的完整版在 `AGENTS.md`，这里是摘要。

| 层 | 目录 | 拥有的事实 | 谁可以改 |
|---|---|---|---|
| **元级** | 仓库根 + `docs/` | 这是什么仓、研究目录、层契约与派发模板、元级里程碑与工作包、迁移台账 | 元级主 Agent 与它派发的 worker |
| **研究级** | `axes/**`、`atlas/**` | 每项研究的目标 / 目的 / 唯一里程碑 / 工作包 / 证据指针；研究项事实（`project.yaml`）；跨格汇总（定律正文、代价表、复现清单、排序打分、论文口径、结论索引） | 研究级主 Agent 与它派发的 worker / reviewer；元级只读 |
| **工级** | `harness/**`、`tools/**`、`tests/**` | 格式契约与分析纪律、测量纪律、引擎能力面、机检实现与共享底座 | 底座维护者；元级与研究级只调用 |

元级**不**拥有任何实验读数、判据、逐格分析、代价事实、复现状态、引擎能力；
研究级与工级**不**拥有元级目录与层契约。这条边界由 `python3 tools/check_meta_layer.py --check` 机检。

## 2 目录怎么读（一次跳转就够）

| 想做什么 | 去哪 |
|---|---|
| 一眼看**树上有哪些研究项、各是什么状态** | `docs/catalog.md`（**生成物**：一行一个研究项，七列） |
| 看**分层规矩、谁可以改、派发契约给什么** | `AGENTS.md` |
| 看**元级计划、里程碑、工作包 owner 与验收** | `docs/plan.md` |
| 看**这次搬迁把哪个文件搬到了哪** | `docs/ledger.md`（迁移台账） |
| 看**研究树布局 / 用词 / 粒度 / 目录命名** | `docs/structure.md` |
| 接手**一个研究项** | 它的目录：`axes/<轴>/<研究项>/`（计划 `RESEARCH.md`、事实 `project.yaml`、实验 `experiments/**`） |
| 查**某条定律的代价数字与出处** | `atlas/cost-table.md`（生成物）→ 正文与边界在 `atlas/playbook.md` |
| 看**已确立结论的索引**（只指向，不持有数字） | `atlas/FINDINGS.md` |
| 查**某个实验能不能字节级复现** | `atlas/repro.md`（生成物） |
| 看**研究优先级：先做哪一格** | `atlas/triage.md` |
| 看**研究项格式契约**（`project.yaml` 字段） | `harness/research-schema.md` |
| 学**测量纪律**（三列账本 / 方差界 / 双货币口径） | `harness/methodology.md` |
| 看**引擎能力边界**（哪些组合跑得起来、拒绝在哪一行） | `harness/engine_capability.md` |
| 看**核心设计（RAP）与共享底座** | `atlas/design.md` |
| 知道**旧项目发生了什么** | `archive/README.md` |

轴的记号：`1` 时间 · `2` 深度 · `3` 宽度 · `4` 专家（MoE） · `5` 量化 · `6` 预测 ·
`7` 不研究（只当对照组） · `8` 工程纪律 · `9` Backlog。轴 7 与轴 8 **不立项**（不写 `project.yaml`），
它们的台账与随附实验在各自的 `README.md`（`docs/structure.md` 的「辅助实验」一节）。

## 3 一个研究怎么被派发与验收

**派发**不看对话，只看文档：派发方按 `AGENTS.md` 的模板填一张契约，被派发方照契约交付。
一张契约至少给六样东西：**目标 / 范围与排除 / 权威输入（读哪些文件）/ 交付物与验收（可观测）/
预算（墙钟、GPU、重试次数）/ 停止条件**。`AGENTS.md` 里有可复制的模板。

**研究级的验收**是六条命令，由研究级 Agent 自己跑（元级不代跑）：

```bash
python3 tools/summary.py --check          # 研究项字段 / 证据 / 口径
python3 tools/check_tree.py               # 同上，纯校验入口
python3 tools/check_docs.py --check       # 文档引用可解析 + 三层索引完整
python3 tools/build_cost_table.py --check # atlas/cost-table.md 与 project.yaml 逐字节一致
python3 tools/inventory_runs.py --check   # atlas/repro.md 与磁盘一致
python3 -m pytest tests/ -q               # 共享底座测试
```

**元级的验收**是四条，只查元级自己的事实：

```bash
python3 tools/build_catalog.py --check    # 研究目录与 project.yaml 逐字节一致
python3 tools/check_meta_layer.py --check # 元级文档不含研究级读数
python3 tools/check_ledger.py --check     # 迁移台账无遗漏、新家存在、删除有理由
python3 tools/check_docs.py --check       # 引用可解析 + 三层索引完整
```

执行完成后由 **fresh 上下文的复核者**检阅（不复用执行者的上下文），复核排在执行之后；
复核发现遗漏或错误 ⇒ 改文档 ⇒ 继续做，直到找不出问题。每层的具体步骤在 `AGENTS.md`。

## 4 怎么跑起来

```bash
git clone git@github.com:ZisIsNotZis/pathlm.git     # origin 的 URL；SSH，可见性未核实
cd pathlm
tools/bootstrap.sh                                  # 版本门 + 依赖门（缺 PyYAML 就装，装不了就非零退出）
python3 tools/build_catalog.py --check              # 元级自检：目录与事实源一致
```

版本与依赖前提：Python ≥ 3.10（本仓实测 3.12.3）；元级四条机检只需要 PyYAML 一个第三方包。
`tests/` 另外需要 `requirements-dev.txt` 里的 pytest / numpy / torch（装 torch 是 GB 级下载）。

## 5 干净克隆重建不了什么

- **数据**：`data/enwik8_full.npz` 与 `.tmp/enwik8`（enwik8 原始字节）都不入库，**也不会自动下载**；
  要跑实验得自己拿到 enwik8 放到 `.tmp/enwik8`，首次运行时生成缓存。
- **逐 run 的原始产物**：各实验目录下 `runs/` 被 `.gitignore` 排除，干净克隆里补不回来；
  引用到的实验级 `analysis.md` / `results.json` 都在库内。哪些能字节级复现、哪些不能、缺什么，
  由 `atlas/repro.md` 逐实验回答（生成物，会随磁盘重算）。
- **权重**：`*.pt` 是派生物，按各实验目录 `README.md` 的命令重训，不入库。
- **GPU 面**：`tests/` 的完整运行需要 torch + CUDA；本仓实测只在单卡 RTX 4090 上跑过。

## 6 怎么报问题 / 怎么贡献

- **报问题**：向 `origin` 指向的 GitHub 仓库提 issue 或 PR（issue 是否开启未核实；PR 一定可用）。
  一句话说清「哪个文件、哪条结论、你认为它错在哪」，比长篇更容易被处理。
- **贡献流程与硬性规则**：`CONTRIBUTING.md`；格式契约见 `harness/SCHEMA.md` 与
  `harness/research-schema.md`，研究树布局见 `docs/structure.md`。
- **提 PR 前**：自己跑一遍上面属于你那一层的机检；`tests/` 里那条已登记的预存失败不算你的问题。

## 7 语言与许可

中文为主（面向中文读者），代码标识符 / 日志 / 配置键保持英文；一个文档内不中英段落交替。

**代码许可：AGPL-3.0**（全文见 `LICENSE.md`）。
**文档与论文的发布许可另议**：`LICENSE.md` 覆盖的是代码；`atlas/paper-plan.md` 的目标是公开可引用，
文档通常用 CC BY 一类许可，尚未裁定。
