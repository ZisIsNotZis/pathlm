# 计划（plan 层 · 当前工作包与状态）

> 本文件是本仓库**计划层的唯一权威**：里程碑、工作包、owner、验收与状态都写在这里，别处只放指针。
> 事实文档的规矩见 `AGENTS.md` §2；工作包完成即改写本文件的状态，不在正文里记录过程史。

## 1 当前里程碑

| id | 里程碑 | 交付物 | 验收（可判定的观测） | 状态 |
|---|---|---|---|---|
| M1 | 注意力残差（2.2.2 / 实验 `2653`）出判决 | `axes/2-depth/2.2.2-attention-residual/experiments/2653-interlayer-attn-13m/results.json` 的两臂读数 + `analysis.md` 的 B 表 | `results.json.arms` 含 `attn-scalar` 与 `attn-qkv` 的 `clean_bpc`，且 `P1_verdict` 已填；`analysis.md` 的 B 表数字与 `results.json` 逐项一致 | 已完成 |
| M2 | 层间拓扑种子（2.2.3 / 实验 `2651`）四臂齐 + 全树自检 | `2651-inter-layer-topology-s1-13m/results.json` 四臂读数 + `analysis.md` | 四臂（`sequential` / `parallel` / `parallel-dual-ln` / `dense`）都有 `clean_bpc`；`python3 tools/summary.py --check` 与 `python3 -m pytest tests/ -q` 通过；工作区干净并已推送 | 已完成（`tests/` 现状 = 866 passed / 1 failed，唯一失败为预存失败 `test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`，涉 `axes/6-predict/**`，非本计划引入、不修，单独登记） |

M1 读数：`attn-scalar` 1.5336（Δ vs stock 1.5254 = −0.0082，不可排序）、
`attn-qkv` 1.5431（Δ = −0.0177，反向可分）；判词与归因落
`axes/2-depth/2.2.2-attention-residual/project.yaml`（`status: 被否证`）。

M2 读数：四臂 seed=1 clean bpc = 1.5288 / 1.5710 / 1.5662 / 1.5273
（Δ vs `sequential` = 0 / +0.0422 / +0.0374 / −0.0015）；判词落
`2651-inter-layer-topology-s1-13m/analysis.md` §9 与
`axes/2-depth/2.2.3-learned-layer-sum/project.yaml`（`status: 被否证`，2.2.4 代价面结论 `已完成`）。

## 2 工作包

| id | 工作包 | owner | 输入（权威） | 交付物 | 验收 | 预算 | 状态 |
|---|---|---|---|---|---|---|---|
| WP-R1 | 停止两条失控的 supervisor，并修掉 `topology_liveness` 与 `InterLayerAttentionModel` 的接口不匹配 | worker（fresh context） | `train_depth_ar.py:880`、`pathlm/depth_ar.py:1248`、`2653/run.py` 的模型定义 | 代码改动 + 回归单测（提交 `640a3aa`）；两条 supervisor 与子进程退出 | 单测红→绿可复现：构造一个无 `layer_mix` 的 dense 拓扑模型，旧代码抛 `AttributeError`、新代码给出活性摘要 | 30 分钟 | 已完成 |
| WP-R2 | 重跑 `2653` 两臂并落盘判决 | worker | WP-R1 的修复；`2653/README.md` 的预登记判据 | `2653/results.json` 两臂读数 + `analysis.md` B 表（提交 `0234314`） | 见 M1 | 2 臂 × 约 10 分钟 GPU | 已完成 |
| WP-R3 | 跑完 `2651` 剩余三臂 + 写判读 | worker | `2651/README.md`、`2651/results.json` 已有的 `sequential` 臂 | 四臂读数 + `2651/analysis.md` + 两份 `project.yaml` 的当前事实 | 见 M2 | 3 臂 × 约 10 分钟 GPU | 已完成 |
| WP-R4 | 独立复核 + 文档收口 + 提交推送 | reviewer（fresh context）→ worker | WP-R1..R3 的落盘证据 | 复核结论；`project.yaml` / `analysis.md` 的当前事实；提交并推送 | 复核无阻塞项；`summary.py --check` 通过；`git status` 干净 | 30 分钟 | 进行中（文档收口与提交推送已完成；独立复核待派） |

## 3 纪律（本计划的取舍已定，执行方不自选）

- CPU 侧改动（WP-R1）与 GPU 侧（WP-R2/R3）解耦：先修代码，再占 GPU，避免继续空烧。
- 已训完的权重优先复用：判 `2653/runs/*/model.pt` 与 `2651/runs/*/model.pt` 能否**只重算判据**
  而不重训；能复算就不重训，并在 `analysis.md` 说明复算来源。
- GPU 串行取 `/tmp/pathlm_gpu.lock`（`SCHEMA.md` §20），同一时刻只有一个臂在跑。
- 预登记判据线与臂定义**不改**；结果不好时改的是归因与下一步，不是判据。
