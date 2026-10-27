[English](README.en.md) | **简体中文**

<img src="assets/logo.svg" alt="pathlm" width="200">

# PathLM Research Wiki

语言模型算力路径设计空间的研究树 —— 一格一个研究项，记下谁试过、结果如何、为什么失败、怎么复现。

**它把「语言模型的算力花在哪条路径上」变成一棵可机检的研究树：46 个研究项按轴切开，每条结论都带口径、对照、方差界与可复现判定，而机检随时能把它判红。**

这不是模型仓库：本仓不训练、不发布模型、不打榜、不做组合实验库。它发布的是**一棵可复核的研究树 + 一套测量纪律**，给两类人看：要改模型架构、想先知道「这个改动大概让我掉多少质量、还是白赚」的人；以及要往树上派活、复核与接手的人。

## 30 秒上手

```bash
git clone git@github.com:ZisIsNotZis/pathlm.git
cd pathlm
tools/bootstrap.sh                       # 版本门 + 依赖门（Python ≥ 3.10；元级机检只要 PyYAML）
python3 tools/build_catalog.py --check    # 元级自检：研究目录与事实源逐字节一致
```

干净克隆里唯一装不上的是实验依赖：`tests/` 需要 `pip install -r requirements-dev.txt`（pytest / numpy / torch，torch 是 GB 级下载）。跑实验的数据要自己拿（见下面「干净克隆重建不了什么」）。

## 现在能做什么

- **研究目录**：`docs/catalog.md` 一行一个研究项（id / 标题 / 目的 / 验收 / 期望产出 / 状态 / 指针），由 46 份 `project.yaml` 逐字生成，手改即红。
- **六个研究轴 + 三条辅助线**：时间 · 深度 · 宽度 · 专家（MoE） · 量化 · 预测，外加参照线、工程纪律与 backlog（轴 7 / 8 不立项，只记台账与随附实验）。
- **已测定律 1–11**：`atlas/playbook.md` 写正文与适用边界，`atlas/cost-table.md`（生成物）把定律 ↔ 证据 ↔ 观测代价逐行对上。
- **复现清单**：`atlas/repro.md`（生成物）逐实验回答「run 在哪、产物在不在库内、重跑命令是什么、能不能字节级复现、缺什么」。
- **结论索引**：`atlas/FINDINGS.md` 一条结论一行，只指路，不持有数字。
- **研究优先级**：`atlas/triage.md` 按「能否改变判断」给研究项排序，并给出批次草案。
- **引擎与评测底座**：`pathlm/` 里的变深度增量解码（`depth_ar.py`）、评测套件（`eval.py` / `gen_quality.py`）、分配器（`slider.py`）与模型件（`model.py`、`exit_policy.py` 等），配 `configs/` 的实验臂配置和 `train_*.py` / `probe_*.py` 入口。
- **机检**：`tools/` 下每个 `--check` 都是一条可失败的闸门，`tests/` 把纪律钉成单测。

## 仓库怎么组织

仓库按**三层**切开，一条事实只有一个家，别的层只放指针：

| 层 | 目录 | 拥有什么 |
|---|---|---|
| **元级** | 仓库根 + `docs/` | 这是什么仓、研究目录、层契约与派发模板、里程碑与工作包、迁移台账 |
| **研究级** | `axes/**`、`atlas/**` | 每项研究的事实（`project.yaml` 是该项 SSOT）、跨格汇总（定律、代价表、复现清单、排序、论文口径） |
| **工级** | `harness/**`、`tools/**`、`tests/**` | 格式契约与测量纪律、引擎能力面、生成器与机检、共享底座 |

```
README.md / README.en.md   门面（中文为主，英文镜像）
AGENTS.md                  层契约、派发模板、协作纪律（写给 agent）
CONTRIBUTING.md            贡献流程与硬性规则
docs/                      元级：catalog.md（生成物）/ plan.md / structure.md / ledger.md
atlas/                     研究级跨格汇总：playbook / cost-table / repro / triage / FINDINGS / design / paper-plan
axes/                      研究级逐项：<轴>/<研究项>/project.yaml（SSOT）
harness/                   工级：SCHEMA / research-schema / methodology / engine_capability
tools/                     工级：生成器与 --check 机检（含 bootstrap.sh、evalkit/）
tests/                     工级：共享底座测试
pathlm/ configs/ train_*.py probe_*.py slider.py
                           共享底座与历史引擎（原地保留，迁移计划见 atlas/design.md）
.scratch/                  历史工单与原始证据（只读日志，不是权威）
archive/                   旧项目只读资产
assets/                    logo
LICENSE.md                 AGPL-3.0 全文
```

三层之间只通过三种方式说话：**指针**（别处只链到事实之家）、**生成物**（目录 / 代价表 / 复现清单由 `tools/` 从事实源逐字生成，`--check` 与手改副本逐字节比对）、**机检**（`tools/check_meta_layer.py` 判「元级文档里出现研究级读数」，`tools/check_docs.py` 判引用可解析与每层索引完整）。所以「谁拥有这条事实」不是约定，是能跑红的。

一个研究项的寿命是：元级按 `AGENTS.md` 的模板派一张契约（目标 / 范围 / 权威输入 / 交付物与验收 / 预算 / 停止条件）→ 研究级在 `axes/<轴>/<研究项>/` 里预登记判据、跑实验、落读数与判词 → 跨格事实由 `atlas/` 汇总 → fresh 上下文的复核者检阅，直到找不出问题。

## 怎么开发与自检

```bash
pip install -r requirements-dev.txt        # pytest / numpy / torch（仅跑 tests/ 需要）
```

按你动的那一层跑机检（每条都是退出码闸门）：

```bash
# 元级
python3 tools/build_catalog.py --check     # 研究目录与 project.yaml 逐字节一致
python3 tools/check_meta_layer.py --check  # 元级文档不含研究级读数
python3 tools/check_ledger.py --check      # 迁移台账无遗漏、新家存在、删除有理由
python3 tools/check_docs.py --check        # 引用可解析 + 三层索引完整

# 研究级
python3 tools/summary.py --check           # 研究项字段 / 证据 / 口径（= tools/check_tree.py）
python3 tools/build_cost_table.py --check  # atlas/cost-table.md 与 project.yaml 逐字节一致
python3 tools/inventory_runs.py --check    # atlas/repro.md 与磁盘一致
python3 -m pytest tests/ -q                # 共享底座测试
```

改事实只改事实源，再重跑生成器：目录改 `project.yaml`，代价表改 `project.yaml`，复现清单改磁盘上的 run。**生成物禁止手改**——`--check` 会红。

## 干净克隆重建不了什么

- **数据**：enwik8 原始字节（`.tmp/enwik8`）不入库，也不会自动下载；要跑实验得自己放进去，首次运行时生成缓存。
- **逐 run 的原始产物**：各实验目录下的 `runs/` 被 `.gitignore` 排除；已入库的 `analysis.md` / `results.json` 在库内。哪些能字节级复现、缺什么，看 `atlas/repro.md`。
- **权重**：`*.pt` 是派生物，按各实验目录 `README.md` 的命令重训，不入库。
- **GPU 面**：`tests/` 的完整运行需要 torch + CUDA；本仓只在单卡 RTX 4090 上实测过。

## 怎么贡献

- **报问题**：向本仓提 issue 或 PR。一句话说清「哪个文件、哪条结论、你认为它错在哪」，比长篇更容易被处理。
- **提 PR 前**：自己跑一遍属于你那一层的机检（上面那两组命令），并按 `harness/methodology.md` 的口径写数字。
- **贡献规则**：`CONTRIBUTING.md` 是入口；格式契约见 `harness/SCHEMA.md` 与 `harness/research-schema.md`，研究树布局见 `docs/structure.md`。
- 新实验必须在训练前绑定契约（`python3 tools/prereg_bind.py <实验目录>`），绑定完就提交。

## 路线图

**已实现**

- 三层分层收身：元级只剩目录与契约，研究级与工级各持有自己的事实与索引；元级里程碑七条验收全过，经 fresh 上下文复核。
- 机检按层分工：元级四条、研究级六条，全部是可失败的退出码闸门，另有 `tests/` 钉住纪律。
- 46 个研究项的研究目录（生成物）与逐项 `project.yaml` SSOT；已测定律 1–11 与代价对照表。
- 复现清单（逐实验的三态判定）与研究优先级排序。
- 六个轴上的判决已陆续产出：硬窗口、稀疏激活专家、低秩压缩、循环深度、层间聚合、推测解码等（逐项见 `docs/catalog.md`）。
- 论文初稿：`atlas/paper.md` 按 `atlas/paper-plan.md` 的 claim、结构与十条验收成篇，只搬已落盘事实、不重算读数；待许可裁决与 fresh 复核后发布。

**计划中**

- 把 `pathlm/` 变成可安装包（`pyproject.toml` + `pip install -e .`），测试与探针改为包内导入。
- 把评测套件抽成 `tools/evalkit/` 的共享底座，让每个研究项实验目录自带 `run.py` 从底座取天平（计划见 `atlas/design.md`）。
- 论文的发布件：许可裁决（`docs/plan.md` §5）与 DOI / tag 对齐。
- 补齐目录里标为「未开始 / backlog」的研究项（如外部检索、权重共享），每项先按 `atlas/triage.md` 的判据定序。**这些需要 GPU**：当前机器没有，短期内也不会获得，路线与登记见 `docs/plan.md` §1.3。

**这条路通向哪**：把「算力该花在哪条路径上」从经验判断变成一张带适用边界的代价表——任何新想法在动 GPU 之前，就能读到它属于哪一族、这一族已经量到的代价在哪、以及要让它产生可测效应必须补齐哪几个条件。

## 许可

**代码许可：AGPL-3.0**，全文见 `LICENSE.md`（版权 ZisIsNotZis）。

**文档与论文的发布许可另议**：`LICENSE.md` 覆盖的是代码；`atlas/paper-plan.md` 的目标是公开可引用，文档通常用 CC BY 一类许可，尚未裁定。
