# STATUS

路由文件：只记「它是什么、现在什么状态、进展落在哪、下一步」。结论与数字见 `FINDINGS.md`，权威计划在 `docs/plan.md`。
本文件与 `FINDINGS.md` 的 `文件:行` 引用绑定引入本条修订的提交：`git log -1 --format=%h -- STATUS.md FINDINGS.md` 指出该提交，被引文件再增删行时用同一命令比对并重核引用。

## 它是什么

不是"一个模型"，而是一张**LM 算力路径设计空间的目录**：按轴切开（时间 / 深度 / 宽度 / MoE / 量化 / 预测 / 对照 / 工程 / backlog），每格一份 `axes/**/project.yaml` 记录目的、创新、基线、端到端验证、单一 milestone、work packages 与状态，并附失败路径账本；代码与实验产物在各格 `experiments/` 下。（`README.md` 的一句话定位，以及它的「这是什么 / 不是什么」与「五条最硬的结论」两节）

## 当前状态

**活跃**（计划层 `docs/plan.md` 有在办工作包），截至 2026-10-19。

- 里程碑 M1（注意力残差 `2653` 出判决）、M2（层间拓扑 `2651` 四臂齐 + 全树自检）**已完成**。（`docs/plan.md:10-11`）
- 工作包 WP-R1 / R2 / R3 / R4 / R5 / R6 **已完成**：首轮独立复核判 4 条阻塞项，处置后窄复核判 **OK with notes**（F-1..F-4 全 PASS），收口提交已推送。（`docs/plan.md:49-53`）
- 两个深度轴研究项已下判决：`2.2.2` attention-residual `status: 被否证`（`axes/2-depth/2.2.2-attention-residual/project.yaml:56`）、`2.2.3` learned-layer-sum `status: 被否证`（`axes/2-depth/2.2.3-learned-layer-sum/project.yaml:66`）、`2.2.4` parallel-sublayer `status: 已完成`（`axes/2-depth/2.2.4-parallel-sublayer/project.yaml:51`）。
- 已知预存失败（非本计划引入、不修、单独登记）：`tests/` 现为 866 passed / 1 failed，唯一失败是 `test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`（涉 `axes/6-predict/**`）。（`docs/plan.md:11`）
- 已知偏离：实验 `2653` 峰值显存 `14961.7` / `15141.1` MB，超出 `SCHEMA.md` §20.1 的 5GB 礼貌线。该线是**共享卡礼貌线、不是科学约束**（`SCHEMA.md` §20.2），偏离已在该实验 `analysis.md` 申报，不进判据。

## 最新进展落在哪个文件

1. 计划与当前状态（唯一权威）：`docs/plan.md`。
2. 最近两次判决的判读：`axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md`（§9 判词、§10 残余缺口）、`axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/` 的 `results.json` 与 `analysis.md`。
3. 各研究项当前事实：`axes/**/project.yaml` 的 `status` / `conclusion` 字段；全表用 `python3 tools/summary.py` 生成。
4. 已测约束：`docs/playbook.md`；测量纪律：`docs/methodology.md`。

## 下一步

来自仓内已写明的开放项，未新增判断：

1. 无待派的复核项：M1/M2 的 fresh 复核已走两轮（首轮 BLOCK → 处置 → 窄复核 OK with notes），机械正证见 `docs/plan.md:24-41`。
2. `2651` 的残余缺口（`axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/analysis.md:331-340`）：更大规模点（2.2.3 milestone 要求，最小可判定约 4× 参数 ~55M）、`dense` 的 α 单纯形约束对照臂、中间深度可读性与 2.1.1/2.2.1 的联合检验、s0 等值门的结构化落盘、层数外推。
3. 每格状态与空白：`python3 tools/summary.py`；新格子动手前先过 `docs/playbook.md` 定律 10（变化空间前门）。
