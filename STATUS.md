# STATUS

路由文件：只记「它是什么、进展落在哪、下一步」。**不拥有事实**——里程碑与工作包状态看
`docs/plan.md`，逐项事实看 `axes/**/project.yaml`，研究树全表与统计用 `python3 tools/summary.py`，
定律与代价数字看 `docs/playbook.md` / `docs/cost-table.md`。本文件不重算数字、不转述它们。

## 它是什么

一张 **LM 算力路径设计空间的目录**：按轴切开（时间 / 深度 / 宽度 / MoE / 量化 / 预测 / 对照 / 工程 /
backlog），每个研究项一份 `axes/**/project.yaml`（目的 / 创新 / 基线 / 端到端验证 / 单一 milestone /
work packages / 状态），代码与实验产物在该研究项的 `experiments/` 下。轴 7 与轴 8 **不立项**：它们只有
参照台账与工程纪律台账，随附的实验没有 `project.yaml`（见 `docs/structure.md#16-辅助实验`）。

人话定位、最硬的几条结论与跑法在 `README.md`。

## 当前状态

计划层有在办工作包。**哪个在办、每个工作包的 owner / 验收 / 预算 / 状态，以及三个里程碑的当前状态，
只写在 `docs/plan.md` 的「里程碑」与「工作包」两节**（`docs/plan.md#1-里程碑`、`docs/plan.md#2-工作包`）。

- **研究项的全表与统计**：`python3 tools/summary.py`；字段 / 证据 / 口径校验用 `--check`。
- **最近出判决的研究项**：`axes/2-depth/2.2.2-attention-residual/`（用 causal attention 代替残差）、
  `axes/2-depth/2.2.3-learned-layer-sum/`（层间学习加权和）、`axes/2-depth/2.2.4-parallel-sublayer/`
  （并行子层）——状态与判词在各自的 `project.yaml::status` / `::conclusion`，本文件不转述。
- **预存测试失败**：计数、失败名与「为什么与本计划无关」只在 `docs/plan.md` 的「验收证据」节
  （`docs/plan.md#3-验收证据全树机检与测试计数`）。
- **已知偏离**（训练峰值显存超出 `SCHEMA.md` §20.1 的共享卡礼貌线，该线不是科学约束）：
  `axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/analysis.md` 已申报，
  数字与理由在那里，本文件不转述。

## 最新进展落在哪个文件

1. 计划、里程碑状态、工作包状态、测试计数：`docs/plan.md`。
2. 三个深度轴判决的判读：`axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md`
   （§9 判词、§10 残余缺口）与 `axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/` 的
   `results.json`、`analysis.md`。
3. 各研究项当前事实：`axes/**/project.yaml` 的 `status` / `conclusion` 字段。
4. 已测约束与口径：`docs/playbook.md`（定律正文与边界）、`docs/methodology.md`（测量纪律）。

## 下一步

只列仓内已经写明的开放项，未新增判断：

1. **陌生阅读复测**：账目层改为当前事实之后，再派一轮没读过本仓的 fresh reviewer（`docs/plan.md` 的
   「工作包」表里那一行）；「自证」里程碑收不收口以它为准。
2. **Δ 方向全树声明**（还在办，见 `docs/plan.md#2-工作包`）：`tools/check_tree.py` 的 Δ 方向汇总行还有
   `undeclared` 与「不判」条目。
3. **`2651` 的残余缺口**（`axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md`
   的 §10）：更大规模点（`2.2.3` milestone 要求，最小可判定约 4× 参数 ≈ 55M）、`dense` 的 α 单纯形约束对照臂、
   中间深度可读性与 `2.2.1` / `2.1.1` 的联合检验、s0 等值门的结构化落盘、层数外推。
4. **新格子动手前**先过 `docs/playbook.md` 的定律 10（变化空间前门）。
