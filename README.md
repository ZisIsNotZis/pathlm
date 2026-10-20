# PathLM Research Wiki — LM 算力路径设计空间的实测代价账本

> **一句话**：这里不发布模型，发布的是一本**代价账本**——把「语言模型的算力花在哪条路径上」这个设计空间按轴切开，
> 一格一个课题，每格记下**谁试过、测出多少、边界在哪、为什么失败**。
> **给谁看**：要改模型架构、想知道「这个改动大概让我掉多少质量、还是白赚」的人。
> **不干什么**：不训练也不发布模型，不做排行榜，不做组合实验库。
> 起点是已收口的 PathLM 项目（25 个执行 ticket）的实测资产，旧资产在 `archive/`。

## 这是什么 / 不是什么

**是**：

- 一棵**可机检的研究树**：每个研究项一份 `axes/**/project.yaml`（目的 / 创新 / SOTA 基线 / 端到端验证 /
  单一 milestone / work packages / 状态）；`python3 tools/summary.py` 一条命令出全表，
  `tools/check_tree.py` 让缺字段、无证据、口径错直接 fail。
- 一本**失败路径的账本**：不只写什么有效，更写**试过什么、为什么失败、排除了哪些替代解释**。
- 一张**生成的**代价表 `docs/cost-table.md`：定律 × 是否触主干 × 观测代价 × 适用边界 × 出处，
  41 条代价事实全部从 `axes/**/project.yaml` 生成，不是手写汇总。

**不是**：

- **不是组合实验库**：默认只做单轴单格的实验，基底恒为 stock 对照；组合**默认留白**（是显式声明，
  不是遗漏，见 `SCHEMA.md` §4）。
- **不是排行榜**：这里比较的是「同一格里谁更值得做」，不是「哪个模型更强」。
- **不能只看结论**：所有数字都带口径（快照 / 渐近 / matched-step / 墙钟 / FLOPs / 每位置字节），
  跨口径比较会得出相反结论。

## 五条最硬的结论（每条都能在 `docs/cost-table.md` 里找到同一行）

1. **不动主干的旁路判断头是零税。** 在冻结主干上挂一个 MTP 式草稿头，clean bpc（每字符比特数，越低越好）
   1.4836 → 1.4836，Δ = **0.0000**（口径：同一条冻结主干挂头**前后**的配对读数）。参照系 = 它自己不带头的
   同一条主干。边界：只说「只读主干、不回传梯度」的头；任何改主干的改动不在这条结论里。
   → [`docs/cost-table.md` 的「定律 7」节](docs/cost-table.md#定律-7--主干被改税就来旁路判断头可以免费)；
   源 `axes/6-predict/6.2-speculative-decoding/project.yaml`。
2. **改主干的税随规模上升。** 深度自适应 AR 形态的形态税：13M 上 +0.31 bpc，100M 上 **+0.44 bpc**
   （同引擎 dense 对照 +0.4362，其中 96% 归因于「改了形态」本身）。参照系 = 同引擎、同配方的 dense 全栈。
   边界：100M / 1.09 epoch / char 级 enwik8 一个点；别把这个斜率当通用规律——同表「定律 8」的另一行
   （回环 retry 族）方向相反。
   → [`docs/cost-table.md` 的「定律 8」节](docs/cost-table.md#定律-8--规模会翻转方向必须单独列一条)；
   源 `axes/2-depth/2.1.1-early-exit/project.yaml`。
3. **纸面省的算力可以不兑现。** 线性注意力（把 softmax 换成归一化线性核）纸面把 T=256 的推理缓存砍掉 93.6%，
   实测同批同配方**一步慢 2.58×**（23.2 ms → 59.8 ms），训练峰值显存 **1.41×**（0.81 → 1.14 GiB）。
   口径 = 实测墙钟与显存，不是 FLOPs。参照系 = 同批同配方的全因果 softmax 对照。
   边界：13M / 未换融合核的实装；换核可能改写这条，本仓没测。
   → [`docs/cost-table.md` 的「定律 11」节](docs/cost-table.md#定律-11--算力与内存的口径)；
   源 `axes/1-time/1.2.2-block-sparse/project.yaml`。
4. **KV 量化到 int8 + 每 32 值一把标尺：字节省 46.9%，质量不可分辨。** 每位置部署 12288 → **6528 B**
   （×0.53），clean bpc 1.483571 → 1.483672（Δ **+0.000101**，正号 = 略差，但远低于 0.014 的不可排序带）。
   参照系 = fp16 KV、同一条冻结主干。边界：13M / 冻结主干 / 纯推理 / 3 个窗口 / seed 2–4；同批的 4 位量化
   越带（+0.0225）⇒ 这条只到 8 位。
   → [`docs/cost-table.md` 的「定律 11」节](docs/cost-table.md#定律-11--算力与内存的口径)；
   源 `axes/5-quant/5.1-kv-quant/project.yaml`。
5. **先有判据带，才谈排序。** 6 个 latent retry 小格之间的跨度只有 0.011 bpc，而格内换 seed 的散布就有
   ≤0.0138 ⇒ 整套「哪个更好」的排序**撤回**，包括「soft×mixture 最好」这条。口径：13M / 同一引擎 / 6 个格。
   参照系 = 同一个格换 seed 的散布。边界：撤回的是**排序**，不是「这 6 个格都无效」。
   → [`docs/cost-table.md` 的「定律 9」节](docs/cost-table.md#定律-9--判据设计比实验本身更值钱)；
   源 `axes/9-backlog/9.2-iteration-refinement/project.yaml`。

定律正文与适用边界的唯一家是 `docs/playbook.md`；跨口径比较之前请先读它的定律 8–11（用表前必须先过的口径）。

## 怎么跑起来（以下命令都在干净克隆上实测过）

```bash
git clone git@github.com:ZisIsNotZis/pathlm.git     # origin 的 URL；SSH，可见性未核实
cd pathlm
tools/bootstrap.sh                                  # 版本门 + 依赖门（缺 PyYAML 就装，装不了就非零退出）
python3 tools/summary.py | tail -3                  # 看研究树全表的末尾统计
```

`python3 tools/summary.py` 的最后一行会打印：

```
共 46 个研究项 ｜ ✅ 已完成 13 ｜ ❌ 被否证 5
```

（其余为 🔄 进行中 14 个、⬜ 未开始 14 个。全表每行一个研究项：id / 轴 / 起止 / 状态 / 结论 / 实验数 / 遗漏。）

版本与依赖前提：Python ≥ 3.10（本仓实测 3.12.3）；上述自检只需要 PyYAML 一个第三方包。
`tests/` 另外需要 `requirements-dev.txt` 里的 pytest / numpy / torch。

实测范围与耗时（单机，含冷启动）：`git clone` 用本地仓路径做（0.7 s）；GitHub 那条 URL 只验证了
`git ls-remote` 能取到 HEAD（完整 clone 受网络限制没跑完）。`tools/bootstrap.sh` 在依赖已齐的解释器上 2.2 s，
在一个只装了 pip、没有 PyYAML 的空 venv 里 33 s（含下载 PyYAML）。`tools/summary.py --check` 3.6 s，
`tools/check_docs.py --check` 3.6 s。

全树自检就是本仓的 CI 口径，五条命令（前四条只需 PyYAML）：

```bash
python3 tools/summary.py --check          # 研究项字段 / 证据 / 口径
python3 tools/check_docs.py --check       # 文档引用可解析 + 文档地图完整性
python3 tools/check_tree.py               # 同上，纯校验入口
python3 tools/build_cost_table.py --check # docs/cost-table.md 与 project.yaml 逐字节一致
python3 -m pytest tests/ -q               # 需要 requirements-dev.txt（torch 为 GB 级下载）
```

## 当前状态（实测，不是承诺）

- **研究树**：46 个研究项 —— ✅ 已完成 13 / ❌ 被否证 5 / 🔄 进行中 14 / ⬜ 未开始 14（`python3 tools/summary.py`）。
- **代价表**：41 条代价事实，来自 27 个研究项（46 份 `project.yaml` 全量扫描）；
  `tools/build_cost_table.py --check` 打印「OK：`docs/cost-table.md` 与 41 条代价事实逐字节一致」。
- **引用与文档地图**：`tools/check_docs.py --check` 扫描全树文档与引用，0 处违规，并把 16 个事实文档逐个对到
  `AGENTS.md` §2.1 与 `docs/design.md` §4 的地图行。
- **测试**：`1 failed, 905 passed`（`python3 -m pytest tests/ -q`，约 2 分钟）。唯一失败是本仓**预存**的
  `tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`（4 个 `axes/6-predict/**`
  的 `mutate.py` 未接入统一变异框架），登记在 `STATUS.md`，不属于任何在办工作包。
- **远端**：`git remote -v` 的输出是
  `origin  git@github.com:ZisIsNotZis/pathlm.git (fetch)` 与同一 URL 的 `(push)`；`main` 与远端一致。
  **仓库可见性未核实。**
- **已知偏离**：实验 `2653` 的训练峰值显存 14961.7 / 15141.1 MB，超出 `SCHEMA.md` §20.1 的 5 GB 礼貌线
  （该线是共享卡礼貌线、不是科学约束），已在实验 `analysis.md` 申报（见 `STATUS.md`）。

## 干净克隆重建不了什么

- **数据**：`data/enwik8_full.npz`（190 MB）与 `.tmp/enwik8`（enwik8 原始字节）都不入库，**也不会自动下载**；
  要跑实验得自己拿到 enwik8 放到 `.tmp/enwik8`，首次运行时生成缓存。
- **权重**：`*.pt` 是派生物，按各实验目录 `README.md` 的命令重训，不入库。
- **GPU 面**：`tests/` 的完整运行需要 torch + CUDA（`requirements-dev.txt` 声明了，但装它是 GB 级下载），
  本仓实测只在单卡 RTX 4090 24 GB 上跑过。
- **已实测能重建的**：机检四条命令在**只装了 PyYAML** 的干净解释器里全通（`tools/bootstrap.sh` 的依赖门就是这样验的）。

## 怎么报问题 / 怎么贡献

- **报问题**：向 `origin` 指向的 GitHub 仓库提 issue 或 PR（issue 是否开启未核实；PR 一定可用）。
  一句话说清「哪个文件、哪条结论、你认为它错在哪」，比长篇更容易被处理。
- **贡献流程与硬性规则**：`CONTRIBUTING.md`（新研究项/新实验的字段契约、口径纪律、不可排序带）；
  格式契约见 `SCHEMA.md` 与 `docs/research-schema.md`，研究树布局见 `docs/structure.md`。
- **提 PR 前**：自己跑一遍上面的全树自检；`tests/` 那条预存失败不算你的问题。

## 想深入看什么，去哪

| 想做什么 | 去哪 |
|---|---|
| 一眼看**哪里是空白 / 每项状态** | `python3 tools/summary.py` |
| 查**某条定律的代价数字与出处** | `docs/cost-table.md`（生成物）→ `docs/playbook.md`（正文与边界） |
| 看**研究树布局 / 用词 / 粒度 / 目录** | `docs/structure.md` |
| 看**研究项格式契约**（`project.yaml` 字段） | `docs/research-schema.md` |
| 接手/复现**一个研究项** | `axes/<轴>/<研究项>/project.yaml` → 同目录 `experiments/` |
| 学**方法论**（三列账本 / 方差界 / 双货币） | `docs/methodology.md` |
| 看**核心设计（RAP）与共享底座** | `docs/design.md` |
| 看**引擎能力边界**（哪些组合跑得起来） | `docs/engine_capability.md` |
| 看**计划与在办工作包** | `docs/plan.md` |
| 知道**旧项目发生了什么** | `archive/README.md` → `archive/legacy-docs/retrospective.md` |

轴的记号：`1` 时间 · `2` 深度 · `3` 宽度 · `4` 专家（MoE） · `5` 量化 · `6` 预测 ·
`7` 不研究（只当对照组） · `8` 工程纪律 · `9` Backlog。

## 语言与许可

中文为主（面向中文读者），代码标识符 / 日志 / 配置键保持英文；一个文档内不中英段落交替。

**代码许可：AGPL-3.0**（全文见 `LICENSE.md`）。
**文档与论文的发布许可另议**：`LICENSE.md` 覆盖的是代码；`docs/paper-plan.md` 的目标是公开可引用，
文档通常用 CC BY 一类许可，尚未裁定。
