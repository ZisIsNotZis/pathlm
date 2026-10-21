# RESEARCH.md · `5.4` 低秩压缩（MLA 式）—— 研究级派发契约

> **本文是 projected 视图**：事实的家是 `project.yaml`（同一目录）。本文只做三件 `project.yaml` 做不到的事：
> ① 把每条工作包派给谁、花多少、谁复核写在一起；② 把证据指针排成一条可走的路；
> ③ 把研究级验收映射到具体命令。**本文不复制任何读数**；需要数字时按指针去读。
> 与 `project.yaml` 冲突时以 `project.yaml` 为准，并当场改本文。

---

## 1. 目标（这个研究项要让什么成为真）

判定「把每位置 K/V 压成低秩潜向量（MLA 式）」在**同一每位置部署字节**下是否比别的省法更值——
对方有两类：**直接量化**（5.1 的块标尺口径：int8 = 1.0625 B/值、int4 = 0.5625 B/值）
与**头共享**（GQA/MQA，7-references）。要回答三件事：

1. **同字节下谁的语言建模质量更好**（clean bpc 的配对 Δ 是否越 0.014 带）；
2. **各自在哪一档开始伤质量**（低秩的失效 rank 档、量化的失效位宽档）；
3. **代价是否被记账**（上投影每 token 等效解码计算、须从零训练、参数量/等参数对照）。

**不是**：重造 MLA（DeepSeek-V2 只作设计方向）、不做低秩×量化的叠加（组合默认留白）、
不外推到 T > 256 或 >13M。

## 2. 目的（解决哪个痛点）

推理时的 KV cache 随上下文线性膨胀；业界有两类省法但**没有统一字节轴上的公平比对**：
量化按「比特位宽」报省幅、不看标尺开销；低秩常在大模型上做 uptrain 包装、不报「须从零训练」的代价。
本项在 13M 受控设定下把两者放到**同一「每位置部署字节 → 两腿质量」**的格上比一次，
顺带把等参数对照与上投影算力入账——这是本仓 `5.1` 的跨家族字节图缺的那一格（低秩族）。

## 3. 唯一里程碑（有且仅有一个）

**输出**：一份「低秩（MLA 式）vs 直接量化 vs 头共享」的同字节判决包：
{潜维 272 / 144 / 128} × {int8、int4、fp16、GQA2} 的每位置部署字节 + clean bpc + 合成长程召回 +
上投影等效解码计算，含等参数对照（`mla_c272_eq`、`gqa2_eq`）与 n ≥ 2，且在**恰好同字节**的格上可比。

**验收（可观测）**：

| # | 验收项 | 命令 / 证据位置 |
|---|---|---|
| 1 | 三个同字节格（6528 / 3456 / 3072 B）各有裁决：低秩占优 / 对照占优 / 不可分 | 实验目录 `experiments/2664-lowrank-vs-quant-13m/results.json::same_byte` 的 `groups[*].verdict` |
| 2 | 每位置部署字节 = 解析式 = 实测张量形状（逐位相等） | `experiments/2664-lowrank-vs-quant-13m/results.json::points` 的 `bytes_per_position` 与 `bytes_measured` |
| 3 | 两族失效模式同框（量化 −fp16、低秩 −fp16） | `experiments/2664-lowrank-vs-quant-13m/results.json::damage` |
| 4 | 前缀不变性两条对所有量化臂为 True；低秩前向与 `MLABlock` 前向逐位相等 | `experiments/2664-lowrank-vs-quant-13m/results.json::instrument` |
| 5 | 上投影每 token 等效 MACs/FLOPs 与等参数对照已入账 | `experiments/2664-lowrank-vs-quant-13m/results.json::points` 的 `decode_cost`、`params` |
| 6 | 结论停在「该尺度该字节值不值」，未测项列成下一步最小实验 + 成本 | 实验 `analysis.md` + `project.yaml::gaps` |
| 7 | 机检六条全绿 + `pytest tests/ -q` 只剩已登记的失败 | 见 §6 命令清单 |

**判据线**（逐字在 `project.yaml::objective_criteria` / `falsification`，落地口径在实验
`config.json::criterion.planar_rules`）：0.014 bpc 带**只用于配对 Δ**（24 个评测窗口 ≥ 5、n ≥ 2）；
率类腿走 Wilson 95% CI + 配对 bootstrap；两根门见实验 `README.md` §4。

## 4. 工作包（owner / 验收 / 预算 / 状态）

| WP | 内容 | owner | 验收 | 预算 | 状态 |
|---|---|---|---|---|---|
| `WP1` | 同字节判决：7 支新训臂（`mha-s1`、`mla-c272` s0/s1、`mla-c144` s0/s1、`mla-c272-eq` s0/s1）+ 复用 5 个臂的 8 支已训 checkpoint 的评测 | `axis5-w5.4`（本项单一写入者）；owner = `axis5-lead` | 实验 `experiments/2664-lowrank-vs-quant-13m/results.json::same_byte` 三格各有裁决；`analysis.md` 六问齐备 | 墙钟 ≤ 120 min GPU（7 臂 × ≈10–14 min + 一轮评测 ≈15 min）；单臂 timeout ≤ 20 min；≥2 seed | 已完成（7 臂各 12k 步 + 29 条读数 + 聚合落盘；契约重绑提交 `53e69d3` 早于全部读数）；判词见 `project.yaml::conclusion` |
| `WP1-review` | fresh 上下文独立复核（同字节轴、等参数对照、两腿齐备、判据未被事后改、`gqa2-eq-s1` 的身份凭证按 `analysis.md` §3.1 读） | `axis5-reviewer`（**不得**为本项 writer） | 复核报告落 `.scratch/`，判词逐条对到验收 | 墙钟 ≤ 60 min，只读 | 未开始（WP1 已落盘，writer 停止之后即可派） |
| 缺口（非 WP） | `mla_c64`（1536 B）/ `mla_c256`（6144 B）未训；低秩×量化叠加留白；T>256 与实测显存峰值未测 | `axis5-lead` 决定是否开 | 见 `project.yaml::gaps` | 每档 2 seed ≈ 20–30 min GPU；叠加腿纯评测 ≈ 3 min | 未开始（`gaps` 已逐条给出最小实验与等效成本） |

## 5. 证据指针（排成一条可走的路）

1. **判据从哪来**：`project.yaml::objective_criteria` / `::falsification` → 实验
   `axes/5-quant/5.4-low-rank/experiments/2664-lowrank-vs-quant-13m/README.md` §4（逐字抄录 + 落地口径）。
2. **契约先于读数**：同目录 `prereg_binding.json`（绑 `README.md` + `config.json` 的 sha256）→
   `python3 tools/prereg_bind.py <实验目录> --require`。
3. **读数**：实验 `results.json`（聚合，含 `same_byte` / `damage` / `recall_paired` / `instrument` /
   `ckpt_identity` / `points`）、`results_<臂>_<档>_<run>.json`（29 份逐臂）、`table.md`（渲染）、
   `run_log.txt`（运行日志）；`runs/<run>/`（权重与训练日志，**不入库**，其中 `results.json` 是训练读数与健康量）。
4. **判读**：实验 `analysis.md`（人话在前、六问齐备、数字带键路径；三格判决在其 §4、族内失效模式在其 §5、
   第二腿在其 §6、代价在其 §7；契约/指纹处置在其 §3）+ 判词 `project.yaml::conclusion`。
5. **代价事实**：`project.yaml::cost_rows`（一条一行，指向 `analysis.md`）→
   `atlas/cost-table.md`（生成物，`build_cost_table.py`）。
6. **跨格**：`atlas/triage.md` 的 `5.4` 行（判决与残余）、`atlas/FINDINGS.md` 的定律 11 节（一条指针）。
7. **复现入口**：实验 `README.md` §7 的四条命令（训练 / 评测 / 冒烟 / 自检）。

## 6. 验收命令（研究级 → 具体命令）

```bash
python3 tools/prereg_bind.py axes/5-quant/5.4-low-rank/experiments/2664-lowrank-vs-quant-13m --require
python3 tools/summary.py --check          # 研究项字段 / 证据 / 口径
python3 tools/check_tree.py --check       # 纯校验 + Δ 方向 + 预登记绑定
python3 tools/check_docs.py --check       # 引用可解析 + 三层索引
python3 tools/build_cost_table.py --check # 代价表与 project.yaml 逐字节一致
python3 tools/inventory_runs.py --check   # 复现清单与磁盘一致
python3 -m pytest tests/ -q               # 共享底座测试
```

## 7. 未决与边界

- 本项与 `5.1` 的边界：`5.1` 拥有**量化族**的字节/质量事实（块标尺口径）；本项只新增**低秩族**的点与
  同字节对照，不重测量化族（量化臂在本实验里是**对照**，其口径逐字沿用 `5.1`）。
- 与 `7-references`（头共享 GQA/MQA）的边界：头共享的读数是 `2616` 的事实，本实验只把它搬上同一字节轴。
- 与 `5.6`（组合）的边界：低秩 × 量化 × 窗口的组合默认留白（`harness/SCHEMA.md` §4）。
- 本项的 `status: 已完成` 读作「writer 自评 + 读数与判词落盘」，**不含独立复核**：fresh 复核（`WP1-review`）
  排在 writer 停止之后，其结果由复核方另行落盘并在本项收尾时回填。
- 格 C 的身份凭证有一处降级（`gqa2-eq-s1` 的契约指纹是转写错字，契约冻结未改、`run.py` 的处置分支改了），
  细节与证据在 `experiments/2664-lowrank-vs-quant-13m/analysis.md` §3.1；复核时请连同这一条一起判。
