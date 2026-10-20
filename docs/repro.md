<!-- 本文件由 `tools/inventory_runs.py` 从已入库路径（`git ls-files`）生成，禁止手改；改事实请改实验目录下的文件 -->

# 复现清单：哪些 run 的权重与日志还在、能复跑到什么程度

> 本文件由 `tools/inventory_runs.py` 生成，禁止手改；`--check` 与手改过的副本逐字节比对。它回答一个问题：**这个数字，别人拿这个克隆能重跑出来吗**。
> 数据源 = `git ls-files`：**表里只出现已入库的路径**，所以每一条都在任何克隆里存在。`.gitignore` 排除的本机产物（`*.pt` 权重、`runs/`、`checkpoints/`、`data/`、`.tmp/`）不进表，它们是本机的、不是这个仓库的承诺，见「这个克隆重建不了什么」。
> 「权重在库」列因此恒为「否」：已入库的 `.pt` 有 **0** 个（`.gitignore` 排除了 `*.pt`）。要量本机权重（字节数 + sha256），跑 `python3 tools/inventory_runs.py --local`——那是本机视图，不是本文件的事实。
> 判定是**静态证据**：路径存在性 + 脚本是否触训练 / 语料 / CUDA + 数字是否落盘。脚本没有被逐个执行 ⇒ 「克隆内可重算」= 结构上具备重算条件，**不**等于「已跑通」。要更强的保证只能逐个跑，那不属于本工作包。

## 三态定义与判定规则（逐条，判定必须落在其中一条上）

| 判定 | 含义 | 判据（按顺序判，命中即停） |
|---|---|---|
| `reproducible`（克隆内可重算） | 数字能在克隆内重算出来，不需要外部语料、GPU 训练或已删产物 | 有入口脚本、有 `results*.json`、无悬空引用、脚本**不触**训练 / 语料 / CUDA，也**不引用**不入库的产物 |
| `rebuildable_only`（只能重训得到同类结果） | 换一台机器只能重训一个同规模同配方的模型，不会得到同样的字节 | 同上，但脚本**触**训练 / 语料 / CUDA，或引用了不入库的产物（`*.pt`、`train/` · `runs/` 下的权重与日志）⇒ 重跑是重造而不是重算 |
| `not_reproducible`（不可重建） | 连「同类结果」都拿不到 | 缺入口脚本 / 缺 `results*.json` / 引用了既不在库也不在本机的路径——**每条都必须写明缺什么** |

「声明墙钟」逐字取自该实验的 `results*.json`（墙钟键）或 `analysis.md` 的墙钟句，本文件不重算、不换算口径。
「重跑命令」只抄实验自己 `README.md` 复现小节里的第一条非冒烟命令，或退到它的入口脚本；仓内确定不了就写「命令未在仓内声明」，不替它编一条。
`rebuildable_only` 的**共因**（每行都适用，故不逐行重复）：权重不入库（`.gitignore` 的 `*.pt`）；语料 enwik8（`data/`、`.tmp/` 均不入库且不自动下载，见 `README.md`）。

## 汇总（生成时统计）

- 规模：实验 **123** 个（`axes/**/experiments/*`），run 条目 **417** 个（327 个是 run 子目录或前缀组，90 个是**实验级汇总（不是 run 目录）**，即实验自己那一行「.」；有 run 产物的实验 95 个）。

- 三态分布（按**实验**计数，与上一段的 run 条目是两种计数）：`reproducible`（克隆内可重算）**0** 个、`rebuildable_only`（只能重训得到同类结果）**90** 个、`not_reproducible`（不可重建）**33** 个。
- 入库权重：**0** 个 `.pt`（`*.pt` 被 `.gitignore` 排除）；有落盘 `results*.json` 的实验 91 个。

## 这个克隆重建不了什么

下面这些路径被 `.gitignore` 排除，**不在任何克隆里**；它们是派生物，重建方式写在第三列：

| `.gitignore` 的路径 | 它是什么派生物 | 怎么重建 |
|---|---|---|
| `*.pt` | 训练权重（每个 run 的 `model.pt`） | 重训，或按 `MODEL.md` 的哈希核对 |
| `runs/` | 训练产物目录（`model.pt` / `train_log.jsonl` / `gate_log.jsonl`） | 重训；已入库的 `results*.json` 与 `*-train_log.jsonl` 覆盖其判据部分 |
| `checkpoints/` | 训练中检点 | 重训 |
| `data/` | 语料缓存（`enwik8_full.npz`、`m0_subset.npz`） | 自取 enwik8 放到 `.tmp/enwik8`，首次运行生成缓存（`README.md` 的数据一节） |
| `.tmp/` | 临时工作区（`enwik8` 原始字节等） | 同上 |

- 数据下载：本仓**不下载任何数据集**。`README.md` 的数据一节要求自取 enwik8 放到 `.tmp/enwik8`，首次运行时生成 `data/enwik8_full.npz`。
- 权重：每个 run 的 `model.pt` 都不入库；`MODEL.md`（有该文件的实验）只登记哈希清单，不是权重本身。
- GPU 产物：`runs/**`、`checkpoints/**` 里的训练日志、探针 jsonl、检点都在被忽略的目录里；判据用到的部分已另行入库（见下表的 run 明细）。

## 总表：每个实验

| 实验 | 研究项 | README | analysis | results | run | 权重在库 | 重跑命令 | 声明墙钟 | 判定 | 依据 |
|---|---|---|---|---|---|---|---|---|---|---|
| [2505-m1-followups](../axes/1-time/1.1.1-hard-window/experiments/2505-m1-followups/) | [1.1.1-hard-window](../axes/1-time/1.1.1-hard-window/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2622-longT-512](../axes/1-time/1.1.1-hard-window/experiments/2622-longT-512/) | [1.1.1-hard-window](../axes/1-time/1.1.1-hard-window/project.yaml) | ✓ | ✓ | ✓ | 5 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.full-256.wall_min` = 10.37 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2623-longT-1024](../axes/1-time/1.1.1-hard-window/experiments/2623-longT-1024/) | [1.1.1-hard-window](../axes/1-time/1.1.1-hard-window/project.yaml) | ✓ | ✓ | ✓ | 3 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.full-1024.wall_min` = 9.02 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2624-longT-1024-seed1](../axes/1-time/1.1.1-hard-window/experiments/2624-longT-1024-seed1/) | [1.1.1-hard-window](../axes/1-time/1.1.1-hard-window/project.yaml) | ✓ | ✓ | ✓ | 3 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.full-1024-seed1.wall_min` = 7.68 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2625-longT-stability](../axes/1-time/1.1.1-hard-window/experiments/2625-longT-stability/) | [1.1.1-hard-window](../axes/1-time/1.1.1-hard-window/project.yaml) | ✓ | — | — | 9 | — | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `not_reproducible` | 有入口脚本 `run.py`，但仓内无 `results*.json` |
| [2626-longT-clean](../axes/1-time/1.1.1-hard-window/experiments/2626-longT-clean/) | [1.1.1-hard-window](../axes/1-time/1.1.1-hard-window/project.yaml) | ✓ | ✓ | ✓ | 7 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.full-256-clean.wall_min` = 14.02 | `rebuildable_only` | `clean_driver.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2628-longT-2048](../axes/1-time/1.1.1-hard-window/experiments/2628-longT-2048/) | [1.1.1-hard-window](../axes/1-time/1.1.1-hard-window/project.yaml) | ✓ | ✓ | ✓ | 3 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.full-2048-clean.wall_min` = 2.67 | `rebuildable_only` | `clean_driver.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2629-longT-2048-matched](../axes/1-time/1.1.1-hard-window/experiments/2629-longT-2048-matched/) | [1.1.1-hard-window](../axes/1-time/1.1.1-hard-window/project.yaml) | ✓ | ✓ | ✓ | 3 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.full-2048-clean.wall_min` = 4.34 | `rebuildable_only` | `clean_driver.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2636-longT-unified](../axes/1-time/1.1.1-hard-window/experiments/2636-longT-unified/) | [1.1.1-hard-window](../axes/1-time/1.1.1-hard-window/project.yaml) | ✓ | ✓ | ✓ | 9 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.full-256-s6k.wall_min` = 3.34 | `rebuildable_only` | `clean_driver.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2662-hard-window-anchor-13m](../axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/) | [1.1.1-hard-window](../axes/1-time/1.1.1-hard-window/project.yaml) | ✓ | ✓ | ✓ | 4 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `liveness.win16a_s0.train_wall_minutes` = 13.07 | `rebuildable_only` | `make_config.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2507-x1-x3-distance-penalty](../axes/1-time/1.1.2-soft-decay/experiments/2507-x1-x3-distance-penalty/) | [1.1.2-soft-decay](../axes/1-time/1.1.2-soft-decay/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2503-engine-m1-extensions](../axes/1-time/1.1.4.1-explicit-memory/experiments/2503-engine-m1-extensions/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2616-window192-trained-13m](../axes/1-time/1.1.4.1-explicit-memory/experiments/2616-window192-trained-13m/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 8 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.trained_checkpoint.train_wall_minutes` = 8.23 | `rebuildable_only` | `mutate.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| 2617-design-rightcarry | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | — | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 DESIGN.md） |
| [2617-longrange-task-battery](../axes/1-time/1.1.4.1-explicit-memory/experiments/2617-longrange-task-battery/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `cost.wall_minutes` = 0.18 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| 2617-prior-art-memory | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | — | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 PRIOR_ART.md） |
| [2618-carry-channel-13m](../axes/1-time/1.1.4.1-explicit-memory/experiments/2618-carry-channel-13m/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 10 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `train_results.U0.wall_minutes` = 11.12 | `rebuildable_only` | `mutate.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2619-carry-difficulty-capacity](../axes/1-time/1.1.4.1-explicit-memory/experiments/2619-carry-difficulty-capacity/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 13 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `train_results.D16.wall_minutes` = 14.84 | `rebuildable_only` | `eval_run.sh` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2620-capacity-converged](../axes/1-time/1.1.4.1-explicit-memory/experiments/2620-capacity-converged/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 11 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.cost.gpu_wall_budget_min` = 115 | `rebuildable_only` | `mutate.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2621-carry-under-compression](../axes/1-time/1.1.4.1-explicit-memory/experiments/2621-carry-under-compression/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.gpu_wall_budget_min` = 10 | `rebuildable_only` | `carry_quant.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`）；代码引用不入库的产物：`2618-carry-channel-13m/train/U2/train_log.jsonl`（克隆里不存在，需先重训/自取） |
| [2622-eng1-composable](../axes/1-time/1.1.4.1-explicit-memory/experiments/2622-eng1-composable/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `smoke.wall_minutes` = 0.04 | `rebuildable_only` | `mutate.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2623-composable-arms-13m](../axes/1-time/1.1.4.1-explicit-memory/experiments/2623-composable-arms-13m/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 6 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.gpu_wall_budget_min` = 60 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2624-carry-under-compression-quality](../axes/1-time/1.1.4.1-explicit-memory/experiments/2624-carry-under-compression-quality/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 6 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.gpu_wall_budget_min` = 40 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2624-composable-followups-13m](../axes/1-time/1.1.4.1-explicit-memory/experiments/2624-composable-followups-13m/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 5 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.gpu_wall_budget_min` = 70 | `rebuildable_only` | `measure_step_time.sh` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2625-ood-leg-variance](../axes/1-time/1.1.4.1-explicit-memory/experiments/2625-ood-leg-variance/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 6 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.gpu_wall_budget_min` = 40 | `rebuildable_only` | `measure_step_time.sh` 触训练 / 语料 / CUDA（命中 `train_depth_ar`）；代码引用不入库的产物：`2624-carry-under-compression-quality/train/carry_gqa/model.pt`、`2623-composable-arms-13m/train/gqa_carry_window/model.pt`（克隆里不存在，需先重训/自取） |
| [2627-carry-x-depth](../axes/1-time/1.1.4.1-explicit-memory/experiments/2627-carry-x-depth/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 10 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.gpu_wall_budget_min` = 40 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2628-eng2-carry-ragged](../axes/1-time/1.1.4.1-explicit-memory/experiments/2628-eng2-carry-ragged/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `smoke.wall_minutes` = 0.05 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2629-carry-ragged-projfill](../axes/1-time/1.1.4.1-explicit-memory/experiments/2629-carry-ragged-projfill/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 7 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.gpu_wall_budget_min` = 45 | `rebuildable_only` | `ragged_projfill_arms.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2630-carry-ragged-seed1](../axes/1-time/1.1.4.1-explicit-memory/experiments/2630-carry-ragged-seed1/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 6 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.gpu_wall_budget_min` = 21 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2630-tf4-samecaliber](../axes/1-time/1.1.4.1-explicit-memory/experiments/2630-tf4-samecaliber/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 5 | 否（`*.pt` 不入库） | `bash axes/1-time/1.1.4.1-explicit-memory/experiments/2630-tf4-samecaliber/train_arms.sh # 训练两臂`（README.md 的复现小节） | `config.gpu_wall_budget_min` = 40 | `rebuildable_only` | `reconcile_numbers.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2631-carry-reseed](../axes/1-time/1.1.4.1-explicit-memory/experiments/2631-carry-reseed/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 11 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `train_meta.none_s0.wall_minutes` = 9.07 | `rebuildable_only` | `mutate.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2633-carry-seed2](../axes/1-time/1.1.4.1-explicit-memory/experiments/2633-carry-seed2/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 8 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `train_meta.none_s0.wall_minutes` = 9.07 | `rebuildable_only` | `engine_equivalence.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2634-carry-realtext-13m](../axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 8 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `train_meta.rt_none_s0.wall_minutes` = 11.94 | `rebuildable_only` | `realtext_probe.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2635-recursive-carrier-13m](../axes/1-time/1.1.4.1-explicit-memory/experiments/2635-recursive-carrier-13m/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 6 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `train_meta.stock.wall_minutes` = 2.56 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`）；代码引用不入库的产物：`EXP/train`（克隆里不存在，需先重训/自取） |
| [2638-carry-frontier-13m](../axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 7 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `liveness.win16_s0.train_wall_minutes` = 11.94 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2639-carry-perlayer-frontier-13m](../axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/) | [1.1.4.1-explicit-memory](../axes/1-time/1.1.4.1-explicit-memory/project.yaml) | ✓ | ✓ | ✓ | 8 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `liveness.win16_s0.train_wall_minutes` = 11.94 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2614-kv-ablation-retention-13m](../axes/1-time/1.1.4.2-compression/experiments/2614-kv-ablation-retention-13m/) | [1.1.4.2-compression](../axes/1-time/1.1.4.2-compression/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `cost.wall_minutes` = 0.724 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2615-retention-retrieval-leg-13m](../axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/) | [1.1.4.2-compression](../axes/1-time/1.1.4.2-compression/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `cost.wall_minutes` = 0.95 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2615-priority-shootout-13m](../axes/1-time/1.1.4.3-compression-trigger/experiments/2615-priority-shootout-13m/) | [1.1.4.3-compression-trigger](../axes/1-time/1.1.4.3-compression-trigger/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `cost.wall_minutes` = 1.693 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2615-window-trained-retention-13m](../axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2615-window-trained-retention-13m/) | [1.1.4.5-implicit-vs-explicit](../axes/1-time/1.1.4.5-implicit-vs-explicit/project.yaml) | ✓ | ✓ | ✓ | 7 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.trained_checkpoint.train_wall_minutes` = 7.62 | `rebuildable_only` | `reconcile_numbers.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2623-window-on-longrange-13m](../axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2623-window-on-longrange-13m/) | [1.1.4.5-implicit-vs-explicit](../axes/1-time/1.1.4.5-implicit-vs-explicit/project.yaml) | ✓ | ✓ | ✓ | 6 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `train_results.full.wall_minutes` = 13.01 | `rebuildable_only` | `reconcile_numbers.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2661-implicit-vs-explicit-recall-13m](../axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2661-implicit-vs-explicit-recall-13m/) | [1.1.4.5-implicit-vs-explicit](../axes/1-time/1.1.4.5-implicit-vs-explicit/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `train_meta.win16_s0.wall_minutes` = 11.94 | `rebuildable_only` | `realtext_probe.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`）；代码引用不入库的产物：`2639/train/pl16d/results.json`（克隆里不存在，需先重训/自取） |
| [2678-dilated-vs-window-13m](../axes/1-time/1.2.1-dilated/experiments/2678-dilated-vs-window-13m/) | [1.2.1-dilated](../axes/1-time/1.2.1-dilated/project.yaml) | ✓ | ✓ | ✓ | 3 | 否（`*.pt` 不入库） | `python3 tools/prereg_bind.py axes/1-time/1.2.1-dilated/experiments/2678-dilated-vs-window-13m`（README.md 的复现小节） | `train_declaration.full.wall_minutes` = 5.95 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2616-linear-attention-13m](../axes/1-time/1.2.2-block-sparse/experiments/2616-linear-attention-13m/) | [1.2.2-block-sparse](../axes/1-time/1.2.2-block-sparse/project.yaml) | ✓ | ✓ | ✓ | 5 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.full.wall_min` = 5.75 | `rebuildable_only` | `paired_eval.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2637-linear-attn-quality](../axes/1-time/1.2.2-block-sparse/experiments/2637-linear-attn-quality/) | [1.2.2-block-sparse](../axes/1-time/1.2.2-block-sparse/project.yaml) | ✓ | ✓ | ✓ | 5 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.full-256-lin.wall_min` = 3.67 | `rebuildable_only` | `linear_driver.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2677-local-global-vs-band-13m](../axes/1-time/1.2.2-block-sparse/experiments/2677-local-global-vs-band-13m/) | [1.2.2-block-sparse](../axes/1-time/1.2.2-block-sparse/project.yaml) | ✓ | ✓ | ✓ | 4 | 否（`*.pt` 不入库） | `bash train_arms.sh`（README.md 的复现小节） | `train_declaration.win16a8_s0.wall_minutes` = 13.07 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2614-topk-sparse-attn-13m](../axes/1-time/1.2.3-content-topk/experiments/2614-topk-sparse-attn-13m/) | [1.2.3-content-topk](../axes/1-time/1.2.3-content-topk/project.yaml) | ✓ | ✓ | ✓ | 8 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.full.wall_min` = 6.34 | `rebuildable_only` | `paired_eval.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2676-content-vs-window-13m](../axes/1-time/1.2.3-content-topk/experiments/2676-content-vs-window-13m/) | [1.2.3-content-topk](../axes/1-time/1.2.3-content-topk/project.yaml) | ✓ | ✓ | ✓ | 6 | 否（`*.pt` 不入库） | `bash train_arms.sh`（README.md 的复现小节） | `train_declaration.full.wall_minutes` = 5.95 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2518-exit-retry-law](../axes/2-depth/2.1.1-early-exit/experiments/2518-exit-retry-law/) | [2.1.1-early-exit](../axes/2-depth/2.1.1-early-exit/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2523-depth-ar-first](../axes/2-depth/2.1.1-early-exit/experiments/2523-depth-ar-first/) | [2.1.1-early-exit](../axes/2-depth/2.1.1-early-exit/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2523c-three-ablations](../axes/2-depth/2.1.1-early-exit/experiments/2523c-three-ablations/) | [2.1.1-early-exit](../axes/2-depth/2.1.1-early-exit/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2523d-residual-tax-decomposition](../axes/2-depth/2.1.1-early-exit/experiments/2523d-residual-tax-decomposition/) | [2.1.1-early-exit](../axes/2-depth/2.1.1-early-exit/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2524-depth-ar-100m](../axes/2-depth/2.1.1-early-exit/experiments/2524-depth-ar-100m/) | [2.1.1-early-exit](../axes/2-depth/2.1.1-early-exit/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2621-depth-draft-e2e](../axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e/) | [2.1.1-early-exit](../axes/2-depth/2.1.1-early-exit/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.plain_full.per_prompt[0].wall_median_s` = 5.0627 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2631-pertoken-exit-seqlevel](../axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/) | [2.1.1-early-exit](../axes/2-depth/2.1.1-early-exit/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run.py`（README.md 的复现小节） | — | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`）；代码引用不入库的产物：`.scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt`（克隆里不存在，需先重训/自取） |
| [2506-m2-element-tax-law](../axes/2-depth/2.1.2-skipping/experiments/2506-m2-element-tax-law/) | [2.1.2-skipping](../axes/2-depth/2.1.2-skipping/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2509-m3-composition-law](../axes/2-depth/2.1.2-skipping/experiments/2509-m3-composition-law/) | [2.1.2-skipping](../axes/2-depth/2.1.2-skipping/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2654-skipping-seqlevel-13m](../axes/2-depth/2.1.2-skipping/experiments/2654-skipping-seqlevel-13m/) | [2.1.2-skipping](../axes/2-depth/2.1.2-skipping/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.stock.shared_train.wall_minutes` = 6.55 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2652-residual-form-13m](../axes/2-depth/2.2.1-residual-form/experiments/2652-residual-form-13m/) | [2.2.1-residual-form](../axes/2-depth/2.2.1-residual-form/project.yaml) | ✓ | ✓ | ✓ | 9 | 否（`*.pt` 不入库） | `python3 mutation_check.py`（README.md 的复现小节） | `shared_train_results.wall_minutes` = 6.36 | `rebuildable_only` | `common.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2653-interlayer-attn-13m](../axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/) | [2.2.2-attention-residual](../axes/2-depth/2.2.2-attention-residual/project.yaml) | ✓ | ✓ | ✓ | 3 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `shared_train_results.wall_minutes` = 10.78 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2613-inter-layer-topology-13m](../axes/2-depth/2.2.3-learned-layer-sum/experiments/2613-inter-layer-topology-13m/) | [2.2.3-learned-layer-sum](../axes/2-depth/2.2.3-learned-layer-sum/project.yaml) | ✓ | ✓ | ✓ | 8 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.sequential.wall_min` = 6.51 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2651-inter-layer-topology-s1-13m](../axes/2-depth/2.2.3-learned-layer-sum/experiments/2651-inter-layer-topology-s1-13m/) | [2.2.3-learned-layer-sum](../axes/2-depth/2.2.3-learned-layer-sum/project.yaml) | ✓ | ✓ | ✓ | 8 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.sequential.wall_min` = 9.62 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2655-layer-capacity-13m](../axes/2-depth/2.3.1-layer-capacity/experiments/2655-layer-capacity-13m/) | [2.3.1-layer-capacity](../axes/2-depth/2.3.1-layer-capacity/project.yaml) | ✓ | ✓ | ✓ | 4 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `shared_train_results.wall_minutes` = 5.69 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2525-tax-attribution](../axes/3-width/3.1.1-uniform-narrowing/experiments/2525-tax-attribution/) | [3.1.1-uniform-narrowing](../axes/3-width/3.1.1-uniform-narrowing/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2642-uniform-narrowing-13m](../axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m/) | [3.1.1-uniform-narrowing](../axes/3-width/3.1.1-uniform-narrowing/project.yaml) | ✓ | ✓ | ✓ | 5 | 否（`*.pt` 不入库） | `ARMS="D192:0 D128:0 WD256:0" bash train_arms.sh`（README.md 的复现小节） | `shared_train_results.wall_minutes` = 7.56 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2646-layer-width-13m](../axes/3-width/3.1.2-layer-width/experiments/2646-layer-width-13m/) | [3.1.2-layer-width](../axes/3-width/3.1.2-layer-width/project.yaml) | ✓ | ✓ | ✓ | 4 | 否（`*.pt` 不入库） | `python3 search_schedule.py`（README.md 的复现小节） | `shared_train_results.wall_minutes` = 6.94 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2647-position-width-13m](../axes/3-width/3.1.3-position-width/experiments/2647-position-width-13m/) | [3.1.3-position-width](../axes/3-width/3.1.3-position-width/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py --arm PN --seed 0 --steps 12000 # 训练 + 评测（需 GPU 锁）`（README.md 的复现小节） | — | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2641-ffn-ratio-frontier-13m](../axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/) | [3.1.4-ffn-ratio](../axes/3-width/3.1.4-ffn-ratio/project.yaml) | ✓ | ✓ | ✓ | 17 | 否（`*.pt` 不入库） | `ARMS="U4:0 U2:0 PYR:0 APYR:0 U2:1 PYR:1 APYR:1" bash train_arms.sh`（README.md 的复现小节） | `shared_train_results.wall_minutes` = 7.49 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2644-channel-sparsity-13m](../axes/3-width/3.1.5-channel-sparsity/experiments/2644-channel-sparsity-13m/) | [3.1.5-channel-sparsity](../axes/3-width/3.1.5-channel-sparsity/project.yaml) | ✓ | ✓ | ✓ | 6 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `shared_train_results.wall_minutes` = 6.91 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2610-gated-readout-13m-depthar](../axes/3-width/3.2.1-readout-reparam/experiments/2610-gated-readout-13m-depthar/) | [3.2.1-readout-reparam](../axes/3-width/3.2.1-readout-reparam/project.yaml) | ✓ | ✓ | ✓ | 4 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `main_arm.wall_min` = 12.69 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2611-equalparams-linear-control-13m](../axes/3-width/3.2.1-readout-reparam/experiments/2611-equalparams-linear-control-13m/) | [3.2.1-readout-reparam](../axes/3-width/3.2.1-readout-reparam/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `linear_arm.wall_min` = 12.02 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2612-gated-readout-24k-13m](../axes/3-width/3.2.1-readout-reparam/experiments/2612-gated-readout-24k-13m/) | [3.2.1-readout-reparam](../axes/3-width/3.2.1-readout-reparam/project.yaml) | ✓ | ✓ | ✓ | 3 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `A_arm.wall_min` = 21.03 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2613-readout-reparam-dense-13m](../axes/3-width/3.2.1-readout-reparam/experiments/2613-readout-reparam-dense-13m/) | [3.2.1-readout-reparam](../axes/3-width/3.2.1-readout-reparam/project.yaml) | ✓ | ✓ | ✓ | 5 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `train.baseline.wall_min` = 6.42 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2621-per-depth-readout-13m](../axes/3-width/3.2.1-readout-reparam/experiments/2621-per-depth-readout-13m/) | [3.2.1-readout-reparam](../axes/3-width/3.2.1-readout-reparam/project.yaml) | ✓ | ✓ | ✓ | 3 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.perdepth.wall_min` = 29.08 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2645-per-exit-readout-13m](../axes/3-width/3.2.2-per-exit-readout/experiments/2645-per-exit-readout-13m/) | [3.2.2-per-exit-readout](../axes/3-width/3.2.2-per-exit-readout/project.yaml) | ✓ | ✓ | ✓ | 14 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2643-tied-untied-13m](../axes/3-width/3.2.3-tied-untied/experiments/2643-tied-untied-13m/) | [3.2.3-tied-untied](../axes/3-width/3.2.3-tied-untied/project.yaml) | ✓ | ✓ | ✓ | 4 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [2610-moe-arm-ledger-13m](../axes/4-moe/4.1-sparse-experts/experiments/2610-moe-arm-ledger-13m/) | [4.1-sparse-experts](../axes/4-moe/4.1-sparse-experts/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `analysis.md`「…墙钟，仅记录、不入判据）：探针 19:29:03 启动、19:52:23 完成（…」 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2614-kv-quant-eval-13m](../axes/5-quant/5.1-kv-quant/experiments/2614-kv-quant-eval-13m/) | [5.1-kv-quant](../axes/5-quant/5.1-kv-quant/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `cost.wall_minutes` = 0.885 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2619-kv-budget-chart](../axes/5-quant/5.1-kv-quant/experiments/2619-kv-budget-chart/) | [5.1-kv-quant](../axes/5-quant/5.1-kv-quant/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.cost.gpu_wall_budget_min` = 20 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2620-kv-budget-gaps](../axes/5-quant/5.1-kv-quant/experiments/2620-kv-budget-gaps/) | [5.1-kv-quant](../axes/5-quant/5.1-kv-quant/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py # 全跑`（README.md 的复现小节） | `config.cost.gpu_wall_budget_min` = 20 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2621-kv-budget-rest](../axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest/) | [5.1-kv-quant](../axes/5-quant/5.1-kv-quant/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py # 全跑（≈2.4 min）`（README.md 的复现小节） | `config.cost.gpu_wall_budget_min` = 20 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2631-kv-budget-causal](../axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal/) | [5.1-kv-quant](../axes/5-quant/5.1-kv-quant/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.cost.gpu_wall_budget_min` = 20 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2632-kv-block-scale](../axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/) | [5.1-kv-quant](../axes/5-quant/5.1-kv-quant/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.cost.gpu_wall_budget_min` = 40 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2642-kv-same-byte-frontier](../axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/) | [5.1-kv-quant](../axes/5-quant/5.1-kv-quant/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 frontier_ledger.py`（入口脚本 `frontier_ledger.py`，README 未给复现小节） | — | `rebuildable_only` | 代码引用不入库的产物：`axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`、`axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/mha-12k-s0/model.pt`、`axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/gqa-12k-s0/model.pt`（克隆里不存在，需先重训/自取） |
| [2651-weight-quant-13m](../axes/5-quant/5.2-weight-quant/experiments/2651-weight-quant-13m/) | [5.2-weight-quant](../axes/5-quant/5.2-weight-quant/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.cost.gpu_wall_budget_min` = 30 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2652-act-quant-13m](../axes/5-quant/5.3-activation-quant/experiments/2652-act-quant-13m/) | [5.3-activation-quant](../axes/5-quant/5.3-activation-quant/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `config.cost.gpu_wall_budget_min` = 30 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2656-lowrank-vs-headshare-13m](../axes/5-quant/5.4-low-rank/experiments/2656-lowrank-vs-headshare-13m/) | [5.4-low-rank](../axes/5-quant/5.4-low-rank/project.yaml) | ✓ | — | — | 5 | — | `python3 .../run_arms.py --include-optional`（README.md 的复现小节） | — | `not_reproducible` | 有入口脚本 `gpu_lock.py`，但仓内无 `results*.json` |
| [2653-fp8-vs-int8-13m](../axes/5-quant/5.5-fp8-head-dim/experiments/2653-fp8-vs-int8-13m/) | [5.5-fp8-head-dim](../axes/5-quant/5.5-fp8-head-dim/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `cost.wall_seconds` = 341.72 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2655-fp8-hd64-head-dim](../axes/5-quant/5.5-fp8-head-dim/experiments/2655-fp8-hd64-head-dim/) | [5.5-fp8-head-dim](../axes/5-quant/5.5-fp8-head-dim/project.yaml) | ✓ | ✓ | ✓ | 4 | 否（`*.pt` 不入库） | `python3 f5_anchor_2653_precision.py`（入口脚本 `f5_anchor_2653_precision.py`，README 未给复现小节） | `cost.wall_seconds` = 663.23 | `rebuildable_only` | `f5_anchor_2653_precision.py` 触训练 / 语料 / CUDA（命中 `load_enwik8`） |
| 2617-prior-art-hybrid-kv | [5.6-window-quant-retrieval](../axes/5-quant/5.6-window-quant-retrieval/project.yaml) | — | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 PRIOR_ART.md） |
| [2654-window-retention-frontier-13m](../axes/5-quant/5.6-window-quant-retrieval/experiments/2654-window-retention-frontier-13m/) | [5.6-window-quant-retrieval](../axes/5-quant/5.6-window-quant-retrieval/project.yaml) | ✓ | — | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `cost.wall_seconds` = 205.94 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2510-gap-sweep-b2-c1m](../axes/6-predict/6.2-speculative-decoding/experiments/2510-gap-sweep-b2-c1m/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `命令未在仓内声明` | `arms.C1M.wall_minutes` = 5.3 | `not_reproducible` | 仓内无入口脚本（只有 MODEL.md, README.md, analysis.md, controls/README.md, prereg_binding.json, results.json） |
| [2512-overnight-causal-chain](../axes/6-predict/6.2-speculative-decoding/experiments/2512-overnight-causal-chain/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2513-four-experiments](../axes/6-predict/6.2-speculative-decoding/experiments/2513-four-experiments/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2520-spec-at-scale](../axes/6-predict/6.2-speculative-decoding/experiments/2520-spec-at-scale/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2521-exit-scale-confirm](../axes/6-predict/6.2-speculative-decoding/experiments/2521-exit-scale-confirm/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2610-frozen-trunk-jumphead-13m](../axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | ✓ | ✓ | 2 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `trunk.wall_minutes` = 11.4 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_jumphead`） |
| [2611-jumphead-ar-control-13m](../axes/6-predict/6.2-speculative-decoding/experiments/2611-jumphead-ar-control-13m/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `cost.wall_minutes` = 0.05 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2613-self-gated-composition-13m](../axes/6-predict/6.2-speculative-decoding/experiments/2613-self-gated-composition-13m/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `arms.direct.0.wall_minutes` = 0.38 | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `train_jumphead`） |
| [2631-spec-three-arms](../axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | ✓ | ✓ | 6 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `analysis.md`「…墙钟** \| 共享卡全程有外部占用 \| 等空卡窗口重测 `eval.sh`（~3…」 | `rebuildable_only` | `reconcile_numbers.py` 触训练 / 语料 / CUDA（命中 `train_m1`）；代码引用不入库的产物：`EXP/train`（克隆里不存在，需先重训/自取） |
| [2632-alpha1-attribution](../axes/6-predict/6.2-speculative-decoding/experiments/2632-alpha1-attribution/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `bash run_all.sh`（入口脚本 `run_all.sh`，README 未给复现小节） | — | `rebuildable_only` | 代码引用不入库的产物：`../2631-spec-three-arms/train/13m_s0`、`../2631-spec-three-arms/train/13m_s1`、`../2631-spec-three-arms/train/53m_s0`（克隆里不存在，需先重训/自取） |
| [2640-tree-conf-13m](../axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2650-verify-gate-13m](../axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `rebuildable_only` | `verify_gate.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`）；代码引用不入库的产物：`../2631-spec-three-arms/train/13m_s0`、`../2631-spec-three-arms/train/13m_s1`（克隆里不存在，需先重训/自取） |
| [2673-robustness-13m](../axes/6-predict/6.2-speculative-decoding/experiments/2673-robustness-13m/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | — | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2675-second-tree-13m](../axes/6-predict/6.2-speculative-decoding/experiments/2675-second-tree-13m/) | [6.2-speculative-decoding](../axes/6-predict/6.2-speculative-decoding/project.yaml) | ✓ | — | — | 0 | — | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `not_reproducible` | 有入口脚本 `run.py`，但仓内无 `results*.json` |
| [2674-mdiffusion-13m](../axes/6-predict/6.3.1-diffusion/experiments/2674-mdiffusion-13m/) | [6.3.1-diffusion](../axes/6-predict/6.3.1-diffusion/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2671-padfill-13m](../axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m/) | [6.3.2-pad-fill](../axes/6-predict/6.3.2-pad-fill/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `rebuildable_only` | `padfill.py` 触训练 / 语料 / CUDA（命中 `pathlm.data`） |
| [2660-latentreg-vs-disc-13m](../axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/) | [6.3.3-latent-regression](../axes/6-predict/6.3.3-latent-regression/project.yaml) | ✓ | ✓ | ✓ | 3 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `rebuildable_only` | `draftverify.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2670-freerun-bpc-13m](../axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m/) | [6.3.3-latent-regression](../axes/6-predict/6.3.3-latent-regression/project.yaml) | ✓ | ✓ | ✓ | 1 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | — | `rebuildable_only` | `run.py` 触训练 / 语料 / CUDA（命中 `torch.cuda`） |
| [2523e-fill-variants](../axes/6-predict/6.4-training-fill/experiments/2523e-fill-variants/) | [6.4-training-fill](../axes/6-predict/6.4-training-fill/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2672-fill-n2-13m](../axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m/) | [6.4-training-fill](../axes/6-predict/6.4-training-fill/project.yaml) | ✓ | ✓ | ✓ | 7 | 否（`*.pt` 不入库） | `bash run_all.sh`（入口脚本 `run_all.sh`，README 未给复现小节） | — | `rebuildable_only` | `config_match.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |
| [A2-full-attention](../axes/7-references/experiments/A2-full-attention/) | 7-references | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 2502-engine-m0-probe/README.md, 2504-m1-first-runs/README.md, 2516-rung4-scale-100m/README.md, 2519-asymptotic-tax/README.md, README.md） |
| [A8-positional-semantics](../axes/7-references/experiments/A8-positional-semantics/) | 7-references | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [R7-existing-early-exit](../axes/7-references/experiments/R7-existing-early-exit/) | 7-references | ✓ | — | — | 1 | — | `python3 2623-standard-early-exit-13m/audit_numbers.py`（入口脚本 `2623-standard-early-exit-13m/audit_numbers.py`，README 未给复现小节） | — | `not_reproducible` | 有入口脚本 `2623-standard-early-exit-13m/audit_numbers.py`，但仓内无 `results*.json` |
| [W3-kv-head-sharing](../axes/7-references/experiments/W3-kv-head-sharing/) | 7-references | ✓ | — | — | 2 | — | `python3 2616-gqa-mqa-13m/reconcile_numbers.py`（入口脚本 `2616-gqa-mqa-13m/reconcile_numbers.py`，README 未给复现小节） | — | `not_reproducible` | 有入口脚本 `2616-gqa-mqa-13m/reconcile_numbers.py`，但仓内无 `results*.json` |
| [2523b-shallow-weight-sweep](../axes/8-engineering/experiments/2523b-shallow-weight-sweep/) | 8-engineering | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2508-tts-probes](../axes/9-backlog/9.1-selector-scheduling/experiments/2508-tts-probes/) | [9.1-selector-scheduling](../axes/9-backlog/9.1-selector-scheduling/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2514-allocator-selectivity](../axes/9-backlog/9.1-selector-scheduling/experiments/2514-allocator-selectivity/) | [9.1-selector-scheduling](../axes/9-backlog/9.1-selector-scheduling/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2515-slider-rung3](../axes/9-backlog/9.1-selector-scheduling/experiments/2515-slider-rung3/) | [9.1-selector-scheduling](../axes/9-backlog/9.1-selector-scheduling/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2517-slider-at-scale](../axes/9-backlog/9.1-selector-scheduling/experiments/2517-slider-at-scale/) | [9.1-selector-scheduling](../axes/9-backlog/9.1-selector-scheduling/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2522-gen-quality-signals](../axes/9-backlog/9.1-selector-scheduling/experiments/2522-gen-quality-signals/) | [9.1-selector-scheduling](../axes/9-backlog/9.1-selector-scheduling/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2511-retry-matrix-withdrawal](../axes/9-backlog/9.2-iteration-refinement/experiments/2511-retry-matrix-withdrawal/) | [9.2-iteration-refinement](../axes/9-backlog/9.2-iteration-refinement/project.yaml) | ✓ | — | — | 0 | — | `命令未在仓内声明` | — | `not_reproducible` | 仓内无入口脚本（只有 README.md） |
| [2619-reconstruction-loss-13m](../axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m/) | [9.2-iteration-refinement](../axes/9-backlog/9.2-iteration-refinement/project.yaml) | ✓ | ✓ | ✓ | 8 | 否（`*.pt` 不入库） | `python3 run.py`（入口脚本 `run.py`，README 未给复现小节） | `train_results.A.wall_minutes` = 8.78 | `rebuildable_only` | `mutate.py` 触训练 / 语料 / CUDA（命中 `train_depth_ar`） |

## 不可重建明细（`not_reproducible`：缺什么）

| 实验 | 缺什么 | 依据 |
|---|---|---|
| [2505-m1-followups](../axes/1-time/1.1.1-hard-window/experiments/2505-m1-followups/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2625-longT-stability](../axes/1-time/1.1.1-hard-window/experiments/2625-longT-stability/) | 落盘数字：仓内没有 `results.json` / `results_*.json`，没有可重跑的对照基准 | 有入口脚本 `run.py`，但仓内无 `results*.json` |
| [2507-x1-x3-distance-penalty](../axes/1-time/1.1.2-soft-decay/experiments/2507-x1-x3-distance-penalty/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2503-engine-m1-extensions](../axes/1-time/1.1.4.1-explicit-memory/experiments/2503-engine-m1-extensions/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2617-design-rightcarry](../axes/1-time/1.1.4.1-explicit-memory/experiments/2617-design-rightcarry/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 DESIGN.md） |
| [2617-prior-art-memory](../axes/1-time/1.1.4.1-explicit-memory/experiments/2617-prior-art-memory/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 PRIOR_ART.md） |
| [2518-exit-retry-law](../axes/2-depth/2.1.1-early-exit/experiments/2518-exit-retry-law/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2523-depth-ar-first](../axes/2-depth/2.1.1-early-exit/experiments/2523-depth-ar-first/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2523c-three-ablations](../axes/2-depth/2.1.1-early-exit/experiments/2523c-three-ablations/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2523d-residual-tax-decomposition](../axes/2-depth/2.1.1-early-exit/experiments/2523d-residual-tax-decomposition/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2524-depth-ar-100m](../axes/2-depth/2.1.1-early-exit/experiments/2524-depth-ar-100m/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2506-m2-element-tax-law](../axes/2-depth/2.1.2-skipping/experiments/2506-m2-element-tax-law/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2509-m3-composition-law](../axes/2-depth/2.1.2-skipping/experiments/2509-m3-composition-law/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2525-tax-attribution](../axes/3-width/3.1.1-uniform-narrowing/experiments/2525-tax-attribution/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2656-lowrank-vs-headshare-13m](../axes/5-quant/5.4-low-rank/experiments/2656-lowrank-vs-headshare-13m/) | 落盘数字：仓内没有 `results.json` / `results_*.json`，没有可重跑的对照基准 | 有入口脚本 `gpu_lock.py`，但仓内无 `results*.json` |
| [2617-prior-art-hybrid-kv](../axes/5-quant/5.6-window-quant-retrieval/experiments/2617-prior-art-hybrid-kv/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 PRIOR_ART.md） |
| [2510-gap-sweep-b2-c1m](../axes/6-predict/6.2-speculative-decoding/experiments/2510-gap-sweep-b2-c1m/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 MODEL.md, README.md, analysis.md, controls/README.md, prereg_binding.json, results.json） |
| [2512-overnight-causal-chain](../axes/6-predict/6.2-speculative-decoding/experiments/2512-overnight-causal-chain/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2513-four-experiments](../axes/6-predict/6.2-speculative-decoding/experiments/2513-four-experiments/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2521-exit-scale-confirm](../axes/6-predict/6.2-speculative-decoding/experiments/2521-exit-scale-confirm/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2675-second-tree-13m](../axes/6-predict/6.2-speculative-decoding/experiments/2675-second-tree-13m/) | 落盘数字：仓内没有 `results.json` / `results_*.json`，没有可重跑的对照基准 | 有入口脚本 `run.py`，但仓内无 `results*.json` |
| [2523e-fill-variants](../axes/6-predict/6.4-training-fill/experiments/2523e-fill-variants/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [A2-full-attention](../axes/7-references/experiments/A2-full-attention/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 2502-engine-m0-probe/README.md, 2504-m1-first-runs/README.md, 2516-rung4-scale-100m/README.md, 2519-asymptotic-tax/README.md, README.md） |
| [A8-positional-semantics](../axes/7-references/experiments/A8-positional-semantics/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [R7-existing-early-exit](../axes/7-references/experiments/R7-existing-early-exit/) | 落盘数字：仓内没有 `results.json` / `results_*.json`，没有可重跑的对照基准 | 有入口脚本 `2623-standard-early-exit-13m/audit_numbers.py`，但仓内无 `results*.json` |
| [W3-kv-head-sharing](../axes/7-references/experiments/W3-kv-head-sharing/) | 落盘数字：仓内没有 `results.json` / `results_*.json`，没有可重跑的对照基准 | 有入口脚本 `2616-gqa-mqa-13m/reconcile_numbers.py`，但仓内无 `results*.json` |
| [2523b-shallow-weight-sweep](../axes/8-engineering/experiments/2523b-shallow-weight-sweep/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2508-tts-probes](../axes/9-backlog/9.1-selector-scheduling/experiments/2508-tts-probes/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2514-allocator-selectivity](../axes/9-backlog/9.1-selector-scheduling/experiments/2514-allocator-selectivity/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2515-slider-rung3](../axes/9-backlog/9.1-selector-scheduling/experiments/2515-slider-rung3/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2517-slider-at-scale](../axes/9-backlog/9.1-selector-scheduling/experiments/2517-slider-at-scale/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2522-gen-quality-signals](../axes/9-backlog/9.1-selector-scheduling/experiments/2522-gen-quality-signals/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |
| [2511-retry-matrix-withdrawal](../axes/9-backlog/9.2-iteration-refinement/experiments/2511-retry-matrix-withdrawal/) | 入口脚本：仓内没有任何 `.py` / `.sh` 入口，无法重跑 | 仓内无入口脚本（只有 README.md） |

## run 明细：每个 run 的已入库产物

一个 run = 一组同名的**已入库**产物：某个子目录里的判据 json，或顶层的一组同名日志。`model.pt` 一律不入库，故不设该列。各列口径：

```
results        判据 json 已入库（该 run 的落盘数字）
gate_log       等值门日志已入库（逐位一致那类判据）
训练步数       train_log.jsonl 末行的 step（没有该日志则为 —）
train_summary  训练汇总已入库（步时、显存峰值）
probe          探针 jsonl 已入库（活性 / 生成质量类判据）
```

| 实验 | run | results | gate_log | 训练步数 | train_summary | probe |
|---|---|---|---|---|---|---|
| [2505-m1-followups](../axes/1-time/1.1.1-hard-window/experiments/2505-m1-followups/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2622-longT-512 | . | ✓ | — | — | — | — |
| 2622-longT-512 | full-256 | — | — | — | ✓ | — |
| 2622-longT-512 | full-512 | — | — | — | ✓ | — |
| 2622-longT-512 | win128-256 | — | — | — | ✓ | — |
| 2622-longT-512 | win256-512 | — | — | — | ✓ | — |
| 2623-longT-1024 | . | ✓ | — | — | — | — |
| 2623-longT-1024 | full-1024 | — | ✓ | 11999 | ✓ | — |
| 2623-longT-1024 | win512-1024 | — | ✓ | 11999 | ✓ | — |
| 2624-longT-1024-seed1 | . | ✓ | — | — | — | — |
| 2624-longT-1024-seed1 | full-1024-seed1 | — | — | — | ✓ | — |
| 2624-longT-1024-seed1 | win512-1024-seed1 | — | — | — | ✓ | — |
| 2625-longT-stability | full-1024-s1-diag | — | — | — | ✓ | — |
| 2625-longT-stability | full-T1024-qknorm-s0 | — | — | — | ✓ | — |
| 2625-longT-stability | full-T1024-qknorm-s1 | — | — | — | ✓ | — |
| 2625-longT-stability | full-T256-qknorm256-s0 | — | — | — | ✓ | — |
| 2625-longT-stability | full-T512-cond512-s0 | — | — | — | ✓ | — |
| 2625-longT-stability | full-T512-none512-s0 | — | — | — | ✓ | — |
| 2625-longT-stability | full-T512-none512b-s0 | — | — | — | ✓ | — |
| 2625-longT-stability | full-T512-qknorm512-s0 | — | — | — | ✓ | — |
| 2625-longT-stability | preflight-20 | — | — | — | ✓ | — |
| 2626-longT-clean | . | ✓ | — | — | — | — |
| 2626-longT-clean | full-1024-clean | — | ✓ | 5999 | ✓ | ✓ |
| 2626-longT-clean | full-256-clean | — | ✓ | 23999 | ✓ | ✓ |
| 2626-longT-clean | full-512-clean | — | ✓ | 11999 | ✓ | ✓ |
| 2626-longT-clean | win128-256-clean | — | ✓ | 23999 | ✓ | ✓ |
| 2626-longT-clean | win256-512-clean | — | ✓ | 11999 | ✓ | ✓ |
| 2626-longT-clean | win512-1024-clean | — | ✓ | 5999 | ✓ | ✓ |
| 2628-longT-2048 | . | ✓ | — | — | — | — |
| 2628-longT-2048 | full-2048-clean | — | ✓ | 2999 | ✓ | ✓ |
| 2628-longT-2048 | win1024-2048-clean | — | ✓ | 2999 | ✓ | ✓ |
| 2629-longT-2048-matched | . | ✓ | — | — | — | — |
| 2629-longT-2048-matched | full-2048-clean | — | ✓ | 5999 | ✓ | ✓ |
| 2629-longT-2048-matched | win1024-2048-clean | — | ✓ | 5999 | ✓ | ✓ |
| 2636-longT-unified | . | ✓ | — | — | — | — |
| 2636-longT-unified | full-256-s6k | — | — | 5999 | ✓ | — |
| 2636-longT-unified | full-256-s6k-s1 | — | — | 5999 | ✓ | — |
| 2636-longT-unified | full-512-s6k | — | — | 5999 | ✓ | — |
| 2636-longT-unified | full-512-s6k-s1 | — | — | 5999 | ✓ | — |
| 2636-longT-unified | win128-256-s6k | — | — | 5999 | ✓ | — |
| 2636-longT-unified | win128-256-s6k-s1 | — | — | 5999 | ✓ | — |
| 2636-longT-unified | win256-512-s6k | — | — | 5999 | ✓ | — |
| 2636-longT-unified | win256-512-s6k-s1 | — | — | 5999 | ✓ | — |
| 2662-hard-window-anchor-13m | . | ✓ | — | — | — | — |
| 2662-hard-window-anchor-13m | train/smk16a | ✓ | ✓ | 299 | — | — |
| 2662-hard-window-anchor-13m | train/win16a_s0 | ✓ | ✓ | 11999 | — | — |
| 2662-hard-window-anchor-13m | train/win16a_s1 | ✓ | ✓ | 11999 | — | — |
| [2507-x1-x3-distance-penalty](../axes/1-time/1.1.2-soft-decay/experiments/2507-x1-x3-distance-penalty/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2503-engine-m1-extensions](../axes/1-time/1.1.4.1-explicit-memory/experiments/2503-engine-m1-extensions/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2616-window192-trained-13m | . | ✓ | — | — | — | — |
| 2616-window192-trained-13m | rep | ✓ | — | — | — | — |
| 2616-window192-trained-13m | smoke | ✓ | — | — | — | — |
| 2616-window192-trained-13m | smoke/smoke20 | ✓ | ✓ | 19 | — | — |
| 2616-window192-trained-13m | smoke/stepbench_w0/bench0 | ✓ | ✓ | 599 | — | — |
| 2616-window192-trained-13m | smoke/stepbench_w192/bench192 | ✓ | ✓ | 599 | — | — |
| 2616-window192-trained-13m | train/window192 | ✓ | ✓ | 11999 | — | — |
| 2616-window192-trained-13m | train/window192_rep | ✓ | ✓ | 11999 | — | — |
| [2617-design-rightcarry](../axes/1-time/1.1.4.1-explicit-memory/experiments/2617-design-rightcarry/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2617-longrange-task-battery | . | ✓ | — | — | — | — |
| 2617-longrange-task-battery | smoke | ✓ | — | — | — | — |
| [2617-prior-art-memory](../axes/1-time/1.1.4.1-explicit-memory/experiments/2617-prior-art-memory/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2618-carry-channel-13m | . | ✓ | — | — | — | — |
| 2618-carry-channel-13m | smoke/detA | — | ✓ | 599 | — | — |
| 2618-carry-channel-13m | smoke/detB | — | ✓ | 599 | — | — |
| 2618-carry-channel-13m | smoke/smoke_u0 | — | ✓ | 419 | — | — |
| 2618-carry-channel-13m | smoke/smoke_u1 | — | ✓ | 419 | — | — |
| 2618-carry-channel-13m | smoke/smoke_u2 | — | ✓ | 419 | — | — |
| 2618-carry-channel-13m | train/U0 | ✓ | ✓ | 11999 | — | — |
| 2618-carry-channel-13m | train/U1 | ✓ | ✓ | 11999 | — | — |
| 2618-carry-channel-13m | train/U1rep | ✓ | ✓ | 11999 | — | — |
| 2618-carry-channel-13m | train/U2 | ✓ | ✓ | 11999 | — | — |
| 2619-carry-difficulty-capacity | . | ✓ | — | — | — | — |
| 2619-carry-difficulty-capacity | smoke/smoke2_dh128 | ✓ | ✓ | 599 | — | — |
| 2619-carry-difficulty-capacity | smoke/smoke2_dh16 | ✓ | ✓ | 599 | — | — |
| 2619-carry-difficulty-capacity | smoke/smoke2_dh32 | ✓ | ✓ | 599 | — | — |
| 2619-carry-difficulty-capacity | smoke/smoke2_dh64 | ✓ | ✓ | 599 | — | — |
| 2619-carry-difficulty-capacity | smoke/smoke_dh128 | ✓ | ✓ | 19 | — | — |
| 2619-carry-difficulty-capacity | smoke/smoke_dh16 | ✓ | ✓ | 19 | — | — |
| 2619-carry-difficulty-capacity | smoke/smoke_dh32 | ✓ | ✓ | 19 | — | — |
| 2619-carry-difficulty-capacity | smoke/smoke_dh64 | ✓ | ✓ | 19 | — | — |
| 2619-carry-difficulty-capacity | train/D128 | ✓ | ✓ | 9999 | — | — |
| 2619-carry-difficulty-capacity | train/D16 | ✓ | ✓ | 9999 | — | — |
| 2619-carry-difficulty-capacity | train/D32 | ✓ | ✓ | 9999 | — | — |
| 2619-carry-difficulty-capacity | train/D64 | ✓ | ✓ | 9999 | — | — |
| 2620-capacity-converged | . | ✓ | — | — | — | — |
| 2620-capacity-converged | smoke/det_r1 | ✓ | ✓ | 599 | — | — |
| 2620-capacity-converged | smoke/det_r2 | ✓ | ✓ | 599 | — | — |
| 2620-capacity-converged | smoke/smoke_dh128 | ✓ | ✓ | 599 | — | — |
| 2620-capacity-converged | smoke/smoke_dh16 | ✓ | ✓ | 599 | — | — |
| 2620-capacity-converged | smoke/train20/S128 | ✓ | ✓ | 19 | — | — |
| 2620-capacity-converged | smoke/train20/S16 | ✓ | ✓ | 19 | — | — |
| 2620-capacity-converged | smoke/train20/S16_rep | ✓ | ✓ | 19 | — | — |
| 2620-capacity-converged | train/S128 | ✓ | ✓ | 23999 | — | — |
| 2620-capacity-converged | train/S16 | ✓ | ✓ | 23999 | — | — |
| 2620-capacity-converged | train/S16_rep | ✓ | ✓ | 23999 | — | — |
| 2621-carry-under-compression | . | ✓ | — | — | — | — |
| 2621-carry-under-compression | smoke | ✓ | — | — | — | — |
| 2622-eng1-composable | . | ✓ | — | — | — | — |
| 2622-eng1-composable | smoke/master | ✓ | ✓ | 19 | — | — |
| 2623-composable-arms-13m | . | ✓ | — | — | — | — |
| 2623-composable-arms-13m | smoke | ✓ | — | — | — | — |
| 2623-composable-arms-13m | train/gqa | ✓ | ✓ | 5999 | — | — |
| 2623-composable-arms-13m | train/gqa_carry | ✓ | ✓ | 5999 | — | — |
| 2623-composable-arms-13m | train/gqa_carry_window | ✓ | ✓ | 5999 | — | — |
| 2623-composable-arms-13m | train/stock | ✓ | ✓ | 5999 | — | — |
| 2624-carry-under-compression-quality | . | ✓ | — | — | — | — |
| 2624-carry-under-compression-quality | smoke | ✓ | — | — | — | — |
| 2624-carry-under-compression-quality | smoke/train/carry_gqa | ✓ | ✓ | 19 | — | — |
| 2624-carry-under-compression-quality | smoke/train/carry_mha | ✓ | ✓ | 19 | — | — |
| 2624-carry-under-compression-quality | train/carry_gqa | ✓ | ✓ | 5999 | — | — |
| 2624-carry-under-compression-quality | train/carry_mha | ✓ | ✓ | 5999 | — | — |
| 2624-composable-followups-13m | . | ✓ | — | — | — | — |
| 2624-composable-followups-13m | smoke | ✓ | — | — | — | — |
| 2624-composable-followups-13m | train/gqa_10k | ✓ | ✓ | 9999 | — | — |
| 2624-composable-followups-13m | train/gqa_carry_10k | ✓ | ✓ | 9999 | — | — |
| 2624-composable-followups-13m | train/mha_mlp5 | ✓ | ✓ | 5999 | — | — |
| 2625-ood-leg-variance | . | ✓ | — | — | — | — |
| 2625-ood-leg-variance | smoke | ✓ | — | — | — | — |
| 2625-ood-leg-variance | smoke/train/gqa_carry_win_s1 | ✓ | ✓ | 19 | — | — |
| 2625-ood-leg-variance | smoke/train/gqa_carry_win_s2 | ✓ | ✓ | 19 | — | — |
| 2625-ood-leg-variance | train/gqa_carry_win_s1 | ✓ | ✓ | 5999 | — | — |
| 2625-ood-leg-variance | train/gqa_carry_win_s2 | ✓ | ✓ | 5999 | — | — |
| 2627-carry-x-depth | . | ✓ | — | — | — | — |
| 2627-carry-x-depth | smoke | ✓ | — | — | — | — |
| 2627-carry-x-depth | smoke/steps400/base | ✓ | ✓ | 399 | — | — |
| 2627-carry-x-depth | smoke/steps400/both | ✓ | ✓ | 399 | — | — |
| 2627-carry-x-depth | smoke/steps400/carry | ✓ | ✓ | 399 | — | — |
| 2627-carry-x-depth | smoke/steps400/depth | ✓ | ✓ | 399 | — | — |
| 2627-carry-x-depth | train/base | ✓ | ✓ | 5999 | — | — |
| 2627-carry-x-depth | train/both | ✓ | ✓ | 5999 | — | — |
| 2627-carry-x-depth | train/carry | ✓ | ✓ | 5999 | — | — |
| 2627-carry-x-depth | train/depth | ✓ | ✓ | 5999 | — | — |
| 2628-eng2-carry-ragged | . | ✓ | — | — | — | — |
| 2628-eng2-carry-ragged | smoke/master | ✓ | ✓ | 19 | — | — |
| 2629-carry-ragged-projfill | . | ✓ | — | — | — | — |
| 2629-carry-ragged-projfill | smoke | ✓ | — | — | — | — |
| 2629-carry-ragged-projfill | smoke/pilot1500/pilot1500 | ✓ | ✓ | 1499 | — | — |
| 2629-carry-ragged-projfill | smoke/steps300/projfill | ✓ | ✓ | 299 | — | — |
| 2629-carry-ragged-projfill | smoke/steps300/projfill_carry | ✓ | ✓ | 299 | — | — |
| 2629-carry-ragged-projfill | train/projfill | ✓ | ✓ | 5999 | — | — |
| 2629-carry-ragged-projfill | train/projfill_carry | ✓ | ✓ | 5999 | — | — |
| 2630-carry-ragged-seed1 | . | ✓ | — | — | — | — |
| 2630-carry-ragged-seed1 | smoke | ✓ | — | — | — | — |
| 2630-carry-ragged-seed1 | smoke/steps300/projfill_carry_s1 | ✓ | ✓ | 299 | — | — |
| 2630-carry-ragged-seed1 | smoke/steps300/projfill_s1 | ✓ | ✓ | 299 | — | — |
| 2630-carry-ragged-seed1 | train/projfill_carry_s1 | ✓ | ✓ | 5999 | — | — |
| 2630-carry-ragged-seed1 | train/projfill_s1 | ✓ | ✓ | 5999 | — | — |
| 2630-tf4-samecaliber | . | ✓ | — | — | — | — |
| 2630-tf4-samecaliber | smoke/steps300/base | ✓ | ✓ | 299 | — | — |
| 2630-tf4-samecaliber | smoke/steps300/carry | — | ✓ | 299 | — | — |
| 2630-tf4-samecaliber | train/base | ✓ | ✓ | 5999 | — | — |
| 2630-tf4-samecaliber | train/carry | — | ✓ | 5999 | — | — |
| 2631-carry-reseed | . | ✓ | — | — | — | — |
| 2631-carry-reseed | smoke/eval | ✓ | — | — | — | — |
| 2631-carry-reseed | smoke/steps300/delta_smoke | ✓ | ✓ | 299 | — | — |
| 2631-carry-reseed | smoke/steps300/none_smoke | ✓ | ✓ | 299 | — | — |
| 2631-carry-reseed | smoke/steps300/plain_smoke | ✓ | ✓ | 299 | — | — |
| 2631-carry-reseed | train/delta_s0 | ✓ | ✓ | 11999 | — | — |
| 2631-carry-reseed | train/delta_s1 | ✓ | ✓ | 11999 | — | — |
| 2631-carry-reseed | train/none_s0 | ✓ | ✓ | 11999 | — | — |
| 2631-carry-reseed | train/none_s1 | ✓ | ✓ | 11999 | — | — |
| 2631-carry-reseed | train/plain_s0 | ✓ | ✓ | 11999 | — | — |
| 2631-carry-reseed | train/plain_s1 | ✓ | ✓ | 11999 | — | — |
| 2633-carry-seed2 | . | ✓ | — | — | — | — |
| 2633-carry-seed2 | smoke/eval | ✓ | — | — | — | — |
| 2633-carry-seed2 | smoke/steps300/syn_delta | ✓ | ✓ | 299 | — | — |
| 2633-carry-seed2 | smoke/steps300/syn_none | ✓ | ✓ | 299 | — | — |
| 2633-carry-seed2 | smoke/steps300/syn_plain | ✓ | ✓ | 299 | — | — |
| 2633-carry-seed2 | train/delta_s2 | ✓ | ✓ | 11999 | — | — |
| 2633-carry-seed2 | train/none_s2 | ✓ | ✓ | 11999 | — | — |
| 2633-carry-seed2 | train/plain_s2 | ✓ | ✓ | 11999 | — | — |
| 2634-carry-realtext-13m | . | ✓ | — | — | — | — |
| 2634-carry-realtext-13m | smoke/eval | ✓ | — | — | — | — |
| 2634-carry-realtext-13m | smoke/steps300/rt_delta | ✓ | ✓ | 299 | — | — |
| 2634-carry-realtext-13m | smoke/steps300/rt_none | ✓ | ✓ | 299 | — | — |
| 2634-carry-realtext-13m | train/rt_delta_s0 | ✓ | ✓ | 11999 | — | — |
| 2634-carry-realtext-13m | train/rt_delta_s1 | ✓ | ✓ | 11999 | — | — |
| 2634-carry-realtext-13m | train/rt_none_s0 | ✓ | ✓ | 11999 | — | — |
| 2634-carry-realtext-13m | train/rt_none_s1 | ✓ | ✓ | 11999 | — | — |
| 2635-recursive-carrier-13m | . | ✓ | — | — | — | — |
| 2635-recursive-carrier-13m | smoke/eval | ✓ | — | — | — | — |
| 2635-recursive-carrier-13m | smoke/steps300/linear | ✓ | ✓ | 299 | — | — |
| 2635-recursive-carrier-13m | smoke/steps300/stock | ✓ | ✓ | 299 | — | — |
| 2635-recursive-carrier-13m | train/linear | ✓ | ✓ | 1000 | — | — |
| 2635-recursive-carrier-13m | train/stock | ✓ | ✓ | 2999 | — | — |
| 2638-carry-frontier-13m | . | ✓ | — | — | — | — |
| 2638-carry-frontier-13m | smoke/steps300/smk24 | ✓ | ✓ | 299 | — | — |
| 2638-carry-frontier-13m | train/win128 | ✓ | ✓ | 11999 | — | — |
| 2638-carry-frontier-13m | train/win17 | ✓ | ✓ | 11999 | — | — |
| 2638-carry-frontier-13m | train/win18 | ✓ | ✓ | 11999 | — | — |
| 2638-carry-frontier-13m | train/win24 | ✓ | ✓ | 11999 | — | — |
| 2638-carry-frontier-13m | train/win32 | ✓ | ✓ | 11999 | — | — |
| 2639-carry-perlayer-frontier-13m | . | ✓ | — | — | — | — |
| 2639-carry-perlayer-frontier-13m | smoke/smk600_c256 | ✓ | ✓ | 599 | — | — |
| 2639-carry-perlayer-frontier-13m | smoke/smk600_c64 | ✓ | ✓ | 599 | — | — |
| 2639-carry-perlayer-frontier-13m | smoke/smk_pl16 | ✓ | ✓ | 299 | — | — |
| 2639-carry-perlayer-frontier-13m | smoke/smk_pl16_c256 | ✓ | ✓ | 299 | — | — |
| 2639-carry-perlayer-frontier-13m | train/c16 | ✓ | ✓ | 11999 | — | — |
| 2639-carry-perlayer-frontier-13m | train/pl16 | ✓ | ✓ | 11999 | — | — |
| 2639-carry-perlayer-frontier-13m | train/pl16d | ✓ | ✓ | 11999 | — | — |
| 2614-kv-ablation-retention-13m | . | ✓ | — | — | — | — |
| 2614-kv-ablation-retention-13m | smoke | ✓ | — | — | — | — |
| 2615-retention-retrieval-leg-13m | . | ✓ | — | — | — | — |
| 2615-retention-retrieval-leg-13m | smoke | ✓ | — | — | — | — |
| 2615-priority-shootout-13m | . | ✓ | — | — | — | — |
| 2615-priority-shootout-13m | smoke | ✓ | — | — | — | — |
| 2615-window-trained-retention-13m | . | ✓ | — | — | — | — |
| 2615-window-trained-retention-13m | rep | ✓ | — | — | — | — |
| 2615-window-trained-retention-13m | smoke | ✓ | — | — | — | — |
| 2615-window-trained-retention-13m | smoke/stepbench_w0/bench0 | ✓ | ✓ | 999 | — | — |
| 2615-window-trained-retention-13m | smoke/stepbench_w32/bench32 | ✓ | ✓ | 999 | — | — |
| 2615-window-trained-retention-13m | train/window32 | ✓ | ✓ | 11999 | — | — |
| 2615-window-trained-retention-13m | train/window32_rep | ✓ | ✓ | 11999 | — | — |
| 2623-window-on-longrange-13m | . | ✓ | — | — | — | — |
| 2623-window-on-longrange-13m | smoke/probe_full | ✓ | ✓ | 399 | — | — |
| 2623-window-on-longrange-13m | smoke/probe_win16 | ✓ | ✓ | 399 | — | — |
| 2623-window-on-longrange-13m | train/full | ✓ | ✓ | 11999 | — | — |
| 2623-window-on-longrange-13m | train/win16 | ✓ | ✓ | 11999 | — | — |
| 2623-window-on-longrange-13m | train/win16_rep | ✓ | ✓ | 11999 | — | — |
| 2661-implicit-vs-explicit-recall-13m | . | ✓ | — | — | — | — |
| 2661-implicit-vs-explicit-recall-13m | smoke/eval | ✓ | — | — | — | — |
| 2678-dilated-vs-window-13m | . | ✓ | — | — | — | — |
| 2678-dilated-vs-window-13m | runs/dil32-s0 | ✓ | ✓ | 11999 | — | — |
| 2678-dilated-vs-window-13m | runs/dil32-s1 | ✓ | ✓ | 11999 | — | — |
| 2616-linear-attention-13m | . | ✓ | — | — | — | — |
| 2616-linear-attention-13m | full-12k-s0 | — | ✓ | 11999 | — | — |
| 2616-linear-attention-13m | linear-12k-s0-attempt3 | — | — | 400 | — | — |
| 2616-linear-attention-13m | smoke/smoke-full | ✓ | ✓ | 19 | — | — |
| 2616-linear-attention-13m | smoke/smoke-linear | ✓ | ✓ | 19 | — | — |
| 2637-linear-attn-quality | . | ✓ | — | — | — | — |
| 2637-linear-attn-quality | full-256-lin | — | — | 5999 | ✓ | — |
| 2637-linear-attn-quality | full-256-lin-qk | — | — | 5999 | ✓ | — |
| 2637-linear-attn-quality | linear-256-lin | — | — | 5999 | ✓ | — |
| 2637-linear-attn-quality | linear-256-lin-qk | — | — | 5999 | ✓ | — |
| 2677-local-global-vs-band-13m | . | ✓ | — | — | — | — |
| 2677-local-global-vs-band-13m | smoke/smk24 | ✓ | ✓ | 299 | — | — |
| 2677-local-global-vs-band-13m | train/win24a0_s0 | ✓ | ✓ | 11999 | — | — |
| 2677-local-global-vs-band-13m | train/win24a0_s1 | ✓ | ✓ | 11999 | — | — |
| 2614-topk-sparse-attn-13m | . | ✓ | — | — | — | — |
| 2614-topk-sparse-attn-13m | full-12k-s0 | — | ✓ | 11999 | — | — |
| 2614-topk-sparse-attn-13m | full-12k-s0-r2 | — | ✓ | 11999 | — | — |
| 2614-topk-sparse-attn-13m | smoke/smoke-k0 | ✓ | ✓ | 19 | — | — |
| 2614-topk-sparse-attn-13m | smoke/smoke-k32 | ✓ | ✓ | 19 | — | — |
| 2614-topk-sparse-attn-13m | smoke/smoke-k64 | ✓ | ✓ | 19 | — | — |
| 2614-topk-sparse-attn-13m | topk32-12k-s0 | — | ✓ | 11999 | — | — |
| 2614-topk-sparse-attn-13m | topk64-12k-s0 | — | ✓ | 11999 | — | — |
| 2676-content-vs-window-13m | . | ✓ | — | — | — | — |
| 2676-content-vs-window-13m | runs/topk32_s1 | ✓ | ✓ | 11999 | — | — |
| 2676-content-vs-window-13m | runs/win32 | ✓ | ✓ | 11999 | — | — |
| 2676-content-vs-window-13m | runs/win32_s1 | ✓ | ✓ | 11999 | — | — |
| 2676-content-vs-window-13m | runs/win64 | ✓ | ✓ | 11999 | — | — |
| 2676-content-vs-window-13m | smoke/smk32 | ✓ | ✓ | 39 | — | — |
| [2518-exit-retry-law](../axes/2-depth/2.1.1-early-exit/experiments/2518-exit-retry-law/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2523-depth-ar-first](../axes/2-depth/2.1.1-early-exit/experiments/2523-depth-ar-first/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2523c-three-ablations](../axes/2-depth/2.1.1-early-exit/experiments/2523c-three-ablations/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2523d-residual-tax-decomposition](../axes/2-depth/2.1.1-early-exit/experiments/2523d-residual-tax-decomposition/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2524-depth-ar-100m](../axes/2-depth/2.1.1-early-exit/experiments/2524-depth-ar-100m/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2621-depth-draft-e2e | . | ✓ | — | — | — | — |
| 2631-pertoken-exit-seqlevel | . | ✓ | — | — | — | — |
| [2506-m2-element-tax-law](../axes/2-depth/2.1.2-skipping/experiments/2506-m2-element-tax-law/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2509-m3-composition-law](../axes/2-depth/2.1.2-skipping/experiments/2509-m3-composition-law/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2654-skipping-seqlevel-13m | . | ✓ | — | — | — | — |
| 2652-residual-form-13m | . | ✓ | — | — | — | — |
| 2652-residual-form-13m | base_s0 | ✓ | — | — | — | — |
| 2652-residual-form-13m | gated_s0 | ✓ | — | — | — | — |
| 2652-residual-form-13m | postln_s0 | ✓ | — | — | — | — |
| 2652-residual-form-13m | scaled_s0 | ✓ | — | — | — | — |
| 2652-residual-form-13m | smoke/smoke-base-s0 | ✓ | ✓ | 19 | — | — |
| 2652-residual-form-13m | smoke/smoke-gated-s0 | ✓ | ✓ | 19 | — | — |
| 2652-residual-form-13m | smoke/smoke-postln-s0 | ✓ | ✓ | 19 | — | — |
| 2652-residual-form-13m | smoke/smoke-scaled-s0 | ✓ | ✓ | 19 | — | — |
| 2653-interlayer-attn-13m | . | ✓ | — | — | — | — |
| 2653-interlayer-attn-13m | attn-qkv_s0 | ✓ | — | — | — | — |
| 2653-interlayer-attn-13m | attn-scalar_s0 | ✓ | — | — | — | — |
| 2613-inter-layer-topology-13m | . | ✓ | — | — | — | — |
| 2613-inter-layer-topology-13m | dense-12k-s0 | — | ✓ | 11999 | — | — |
| 2613-inter-layer-topology-13m | parallel-12k-s0 | — | ✓ | 11999 | — | — |
| 2613-inter-layer-topology-13m | parallel-dual-ln-12k-s0 | — | ✓ | 11999 | — | — |
| 2613-inter-layer-topology-13m | sequential-12k-s0 | — | ✓ | 11999 | — | — |
| 2613-inter-layer-topology-13m | smoke/smoke-dense | ✓ | ✓ | 19 | — | — |
| 2613-inter-layer-topology-13m | smoke/smoke-parallel | ✓ | ✓ | 19 | — | — |
| 2613-inter-layer-topology-13m | smoke/smoke-parallel-dual-ln | ✓ | ✓ | 19 | — | — |
| 2651-inter-layer-topology-s1-13m | . | ✓ | — | — | — | — |
| 2651-inter-layer-topology-s1-13m | dense-12k-s1 | — | ✓ | 11999 | — | — |
| 2651-inter-layer-topology-s1-13m | parallel-12k-s1 | — | ✓ | 11999 | — | — |
| 2651-inter-layer-topology-s1-13m | parallel-dual-ln-12k-s1 | — | ✓ | 11999 | — | — |
| 2651-inter-layer-topology-s1-13m | sequential-12k-s1 | — | ✓ | 11999 | — | — |
| 2651-inter-layer-topology-s1-13m | smoke/smoke-dense | ✓ | ✓ | 19 | — | — |
| 2651-inter-layer-topology-s1-13m | smoke/smoke-parallel | ✓ | ✓ | 19 | — | — |
| 2651-inter-layer-topology-s1-13m | smoke/smoke-parallel-dual-ln | ✓ | ✓ | 19 | — | — |
| 2655-layer-capacity-13m | . | ✓ | — | — | — | — |
| 2655-layer-capacity-13m | front-big_s0 | ✓ | — | — | — | — |
| 2655-layer-capacity-13m | growing_s0 | ✓ | — | — | — | — |
| 2655-layer-capacity-13m | uniform_s0 | ✓ | — | — | — | — |
| [2525-tax-attribution](../axes/3-width/3.1.1-uniform-narrowing/experiments/2525-tax-attribution/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2642-uniform-narrowing-13m | . | ✓ | — | — | — | — |
| 2642-uniform-narrowing-13m | D192_s0 | ✓ | — | — | — | — |
| 2642-uniform-narrowing-13m | D192_s99 | ✓ | — | — | — | — |
| 2642-uniform-narrowing-13m | WD256_s0 | ✓ | — | — | — | — |
| 2642-uniform-narrowing-13m | WD256_s99 | ✓ | — | — | — | — |
| 2646-layer-width-13m | APYR_s0 | ✓ | — | — | — | — |
| 2646-layer-width-13m | EQ_s0 | ✓ | — | — | — | — |
| 2646-layer-width-13m | HG_s0 | ✓ | — | — | — | — |
| 2646-layer-width-13m | PYR_s0 | ✓ | — | — | — | — |
| 2647-position-width-13m | . | ✓ | — | — | — | — |
| 2641-ffn-ratio-frontier-13m | . | ✓ | — | — | — | — |
| 2641-ffn-ratio-frontier-13m | APYR-s0 | — | ✓ | 11999 | — | — |
| 2641-ffn-ratio-frontier-13m | APYR-s1 | — | ✓ | 11999 | — | — |
| 2641-ffn-ratio-frontier-13m | APYR_s0 | ✓ | — | — | — | — |
| 2641-ffn-ratio-frontier-13m | APYR_s1 | ✓ | — | — | — | — |
| 2641-ffn-ratio-frontier-13m | PYR-s0 | — | ✓ | 11999 | — | — |
| 2641-ffn-ratio-frontier-13m | PYR-s1 | — | ✓ | 11999 | — | — |
| 2641-ffn-ratio-frontier-13m | PYR_s0 | ✓ | — | — | — | — |
| 2641-ffn-ratio-frontier-13m | PYR_s1 | ✓ | — | — | — | — |
| 2641-ffn-ratio-frontier-13m | U2-s0 | — | ✓ | 11999 | — | — |
| 2641-ffn-ratio-frontier-13m | U2-s1 | — | ✓ | 11999 | — | — |
| 2641-ffn-ratio-frontier-13m | U2_s0 | ✓ | — | — | — | — |
| 2641-ffn-ratio-frontier-13m | U2_s1 | ✓ | — | — | — | — |
| 2641-ffn-ratio-frontier-13m | U4-s0 | — | ✓ | 11999 | — | — |
| 2641-ffn-ratio-frontier-13m | U4_s0 | ✓ | — | — | — | — |
| 2641-ffn-ratio-frontier-13m | smoke/probe-60steps-U2-s0 | ✓ | ✓ | 59 | — | — |
| 2641-ffn-ratio-frontier-13m | smoke/stock-smoke-300 | ✓ | ✓ | 299 | — | — |
| 2644-channel-sparsity-13m | . | ✓ | — | — | — | — |
| 2644-channel-sparsity-13m | F50_s0 | ✓ | — | — | — | — |
| 2644-channel-sparsity-13m | F50_s1 | ✓ | — | — | — | — |
| 2644-channel-sparsity-13m | K25_s0 | ✓ | — | — | — | — |
| 2644-channel-sparsity-13m | K50_s0 | ✓ | — | — | — | — |
| 2644-channel-sparsity-13m | K50_s1 | ✓ | — | — | — | — |
| 2610-gated-readout-13m-depthar | . | ✓ | ✓ | 11999 | — | — |
| 2610-gated-readout-13m-depthar | smoke/smoke-A | ✓ | ✓ | 19 | — | — |
| 2610-gated-readout-13m-depthar | smoke/smoke-gated | ✓ | ✓ | 19 | — | — |
| 2610-gated-readout-13m-depthar | zero-init-saddle | ✓ | ✓ | 11999 | — | — |
| 2611-equalparams-linear-control-13m | . | ✓ | ✓ | 11999 | — | — |
| 2611-equalparams-linear-control-13m | smoke/smoke-linear | ✓ | ✓ | 19 | — | — |
| 2612-gated-readout-24k-13m | . | ✓ | — | — | — | — |
| 2612-gated-readout-24k-13m | smoke/smoke-A | ✓ | ✓ | 19 | — | — |
| 2612-gated-readout-24k-13m | smoke/smoke-gated | ✓ | ✓ | 19 | — | — |
| 2613-readout-reparam-dense-13m | . | ✓ | — | — | — | — |
| 2613-readout-reparam-dense-13m | baseline-s0 | — | ✓ | 11999 | — | — |
| 2613-readout-reparam-dense-13m | baseline-s1 | — | ✓ | 11999 | — | — |
| 2613-readout-reparam-dense-13m | linres-s0 | — | ✓ | 11999 | — | — |
| 2613-readout-reparam-dense-13m | linres-s1 | — | ✓ | 11999 | — | — |
| 2621-per-depth-readout-13m | . | ✓ | — | — | — | — |
| 2621-per-depth-readout-13m | smoke/smoke-perdepth | ✓ | ✓ | 19 | — | — |
| 2621-per-depth-readout-13m | smoke/smoke-wide_shared | ✓ | ✓ | 19 | — | — |
| 2645-per-exit-readout-13m | . | ✓ | — | — | — | — |
| 2645-per-exit-readout-13m | IND_s0_0k | ✓ | — | — | — | — |
| 2645-per-exit-readout-13m | PE_s0_0k | ✓ | — | — | — | — |
| 2645-per-exit-readout-13m | PE_s0_12k | ✓ | — | — | — | — |
| 2645-per-exit-readout-13m | PE_s1_12k | ✓ | — | — | — | — |
| 2645-per-exit-readout-13m | T_s0_0k | ✓ | — | — | — | — |
| 2645-per-exit-readout-13m | T_s0_12k | ✓ | — | — | — | — |
| 2645-per-exit-readout-13m | T_s1_12k | ✓ | — | — | — | — |
| 2645-per-exit-readout-13m | runs/PE-s1-12k | ✓ | ✓ | 11999 | — | — |
| 2645-per-exit-readout-13m | runs/T-s1-12k | ✓ | ✓ | 11999 | — | — |
| 2645-per-exit-readout-13m | smoke/IND-s0-0k | ✓ | ✓ | 59 | — | — |
| 2645-per-exit-readout-13m | smoke/LR-s0-0k | ✓ | ✓ | 59 | — | — |
| 2645-per-exit-readout-13m | smoke/PE-s0-0k | ✓ | ✓ | 59 | — | — |
| 2645-per-exit-readout-13m | smoke/T-s0-0k | ✓ | ✓ | 59 | — | — |
| 2643-tied-untied-13m | . | ✓ | — | — | — | — |
| 2643-tied-untied-13m | PR_s0 | ✓ | — | — | — | — |
| 2643-tied-untied-13m | T_s0 | ✓ | — | — | — | — |
| 2643-tied-untied-13m | U_s0 | ✓ | — | — | — | — |
| 2610-moe-arm-ledger-13m | . | ✓ | — | — | — | — |
| 2614-kv-quant-eval-13m | . | ✓ | — | — | — | — |
| 2614-kv-quant-eval-13m | smoke | ✓ | — | — | — | — |
| 2619-kv-budget-chart | . | ✓ | — | — | — | — |
| 2619-kv-budget-chart | rerun | ✓ | — | — | — | — |
| 2620-kv-budget-gaps | . | ✓ | — | — | — | — |
| 2621-kv-budget-rest | . | ✓ | — | — | — | — |
| 2621-kv-budget-rest | smoke | ✓ | — | — | — | — |
| 2631-kv-budget-causal | . | ✓ | — | — | — | — |
| 2631-kv-budget-causal | smoke | ✓ | — | — | — | — |
| 2632-kv-block-scale | . | ✓ | — | — | — | — |
| 2632-kv-block-scale | smoke | ✓ | — | — | — | — |
| 2642-kv-same-byte-frontier | . | ✓ | — | — | — | — |
| 2651-weight-quant-13m | . | ✓ | — | — | — | — |
| 2651-weight-quant-13m | smoke | ✓ | — | — | — | — |
| 2652-act-quant-13m | . | ✓ | — | — | — | — |
| 2656-lowrank-vs-headshare-13m | smoke/engine201/mha-201step | ✓ | ✓ | 200 | — | — |
| 2656-lowrank-vs-headshare-13m | smoke/engine50/mha-50step | ✓ | ✓ | 49 | — | — |
| 2656-lowrank-vs-headshare-13m | smoke/latent0_201/mha-201step | ✓ | ✓ | 200 | — | — |
| 2656-lowrank-vs-headshare-13m | smoke/latent0_50/mha-50step | ✓ | ✓ | 49 | — | — |
| 2656-lowrank-vs-headshare-13m | smoke/mla128/mla-c128-50step | ✓ | ✓ | 49 | — | — |
| 2653-fp8-vs-int8-13m | . | ✓ | — | — | — | — |
| 2655-fp8-hd64-head-dim | . | ✓ | — | — | — | — |
| 2655-fp8-hd64-head-dim | f5-anchor-2653-rerun | ✓ | — | — | — | — |
| 2655-fp8-hd64-head-dim | hd64-s1 | ✓ | — | — | — | — |
| 2655-fp8-hd64-head-dim | smoke-plumbing-2610trunk | ✓ | — | — | — | — |
| [2617-prior-art-hybrid-kv](../axes/5-quant/5.6-window-quant-retrieval/experiments/2617-prior-art-hybrid-kv/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2654-window-retention-frontier-13m | . | ✓ | — | — | — | — |
| 2510-gap-sweep-b2-c1m | . | ✓ | — | — | — | — |
| [2512-overnight-causal-chain](../axes/6-predict/6.2-speculative-decoding/experiments/2512-overnight-causal-chain/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2513-four-experiments](../axes/6-predict/6.2-speculative-decoding/experiments/2513-four-experiments/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2520-spec-at-scale | . | ✓ | — | — | — | — |
| [2521-exit-scale-confirm](../axes/6-predict/6.2-speculative-decoding/experiments/2521-exit-scale-confirm/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2610-frozen-trunk-jumphead-13m | . | ✓ | — | — | — | — |
| 2610-frozen-trunk-jumphead-13m | capacity-h4x2 | ✓ | — | — | — | — |
| 2611-jumphead-ar-control-13m | . | ✓ | — | — | — | — |
| 2613-self-gated-composition-13m | . | ✓ | — | — | — | — |
| 2631-spec-three-arms | . | ✓ | — | — | — | — |
| 2631-spec-three-arms | smoke/steps300/13m_s0 | ✓ | — | 299 | — | — |
| 2631-spec-three-arms | smoke/steps300/53m_s0 | ✓ | — | 299 | — | — |
| 2631-spec-three-arms | train/13m_s0 | ✓ | — | 5999 | — | — |
| 2631-spec-three-arms | train/13m_s1 | ✓ | — | 5999 | — | — |
| 2631-spec-three-arms | train/53m_s0 | ✓ | — | 5999 | — | — |
| 2632-alpha1-attribution | . | ✓ | — | — | — | — |
| 2640-tree-conf-13m | . | ✓ | — | — | — | — |
| 2650-verify-gate-13m | . | ✓ | — | — | — | — |
| 2673-robustness-13m | . | ✓ | — | — | — | — |
| [2675-second-tree-13m](../axes/6-predict/6.2-speculative-decoding/experiments/2675-second-tree-13m/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2674-mdiffusion-13m | . | ✓ | — | 2 | — | — |
| 2671-padfill-13m | . | ✓ | — | 10 | — | — |
| 2660-latentreg-vs-disc-13m | . | ✓ | — | 8 | — | — |
| 2660-latentreg-vs-disc-13m | second-backbone-13m_s1 | ✓ | — | 6 | — | — |
| 2660-latentreg-vs-disc-13m | second-k4-13m_s0 | ✓ | — | 12 | — | — |
| 2670-freerun-bpc-13m | . | ✓ | — | — | — | — |
| [2523e-fill-variants](../axes/6-predict/6.4-training-fill/experiments/2523e-fill-variants/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2672-fill-n2-13m | . | ✓ | — | — | — | — |
| 2672-fill-n2-13m | snap2k-fill-s0 | ✓ | ✓ | 1999 | — | — |
| 2672-fill-n2-13m | snap2k-fill-s1 | ✓ | ✓ | 1999 | — | — |
| 2672-fill-n2-13m | snap2k-nofill-s0 | ✓ | ✓ | 1999 | — | — |
| 2672-fill-n2-13m | snap2k-nofill-s1 | ✓ | ✓ | 1999 | — | — |
| 2672-fill-n2-13m | snap2k-projfill-s0 | ✓ | ✓ | 1999 | — | — |
| 2672-fill-n2-13m | snap2k-projfill-s1 | ✓ | ✓ | 1999 | — | — |
| [A2-full-attention](../axes/7-references/experiments/A2-full-attention/) | — | — | — | — | — | 仓内没有 run 产物 |
| [A8-positional-semantics](../axes/7-references/experiments/A8-positional-semantics/) | — | — | — | — | — | 仓内没有 run 产物 |
| R7-existing-early-exit | 2623-standard-early-exit-13m | ✓ | — | — | — | — |
| W3-kv-head-sharing | 2616-gqa-mqa-13m | ✓ | — | — | — | — |
| W3-kv-head-sharing | 2616-gqa-mqa-13m/smoke | ✓ | — | — | — | — |
| [2523b-shallow-weight-sweep](../axes/8-engineering/experiments/2523b-shallow-weight-sweep/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2508-tts-probes](../axes/9-backlog/9.1-selector-scheduling/experiments/2508-tts-probes/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2514-allocator-selectivity](../axes/9-backlog/9.1-selector-scheduling/experiments/2514-allocator-selectivity/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2515-slider-rung3](../axes/9-backlog/9.1-selector-scheduling/experiments/2515-slider-rung3/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2517-slider-at-scale](../axes/9-backlog/9.1-selector-scheduling/experiments/2517-slider-at-scale/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2522-gen-quality-signals](../axes/9-backlog/9.1-selector-scheduling/experiments/2522-gen-quality-signals/) | — | — | — | — | — | 仓内没有 run 产物 |
| [2511-retry-matrix-withdrawal](../axes/9-backlog/9.2-iteration-refinement/experiments/2511-retry-matrix-withdrawal/) | — | — | — | — | — | 仓内没有 run 产物 |
| 2619-reconstruction-loss-13m | . | ✓ | — | — | — | — |
| 2619-reconstruction-loss-13m | smoke/smoke_A | ✓ | ✓ | 19 | — | — |
| 2619-reconstruction-loss-13m | smoke/smoke_A_off | ✓ | ✓ | 19 | — | — |
| 2619-reconstruction-loss-13m | smoke/smoke_B | ✓ | ✓ | 19 | — | — |
| 2619-reconstruction-loss-13m | smoke/smoke_Brep | ✓ | ✓ | 19 | — | — |
| 2619-reconstruction-loss-13m | train/A | ✓ | ✓ | 11999 | — | — |
| 2619-reconstruction-loss-13m | train/B | ✓ | ✓ | 11999 | — | — |
| 2619-reconstruction-loss-13m | train/B_rep | ✓ | ✓ | 11999 | — | — |
