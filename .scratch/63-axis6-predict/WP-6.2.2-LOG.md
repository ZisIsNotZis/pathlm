# WP 6.2.2 · 过程日志（append-only；结论请见实验目录 `analysis.md`）

> 本文件是**过程记录**（日期流水），不是权威事实；与 `analysis.md` / `project.yaml` 冲突时以后者为准。
> 时间戳一律来自 `date -Iseconds`（本机时区 +08:00）。

## 2026-10-13T16:28:44+08:00 · 起点

- 任务：WP 6.2.2（轴 6 研究项 6.2「多位置预测 = 投机解码」）——把出资人设计的「每头自带概率的
  多候选 / 树 + 路径概率加权组合」补回来并端到端测（补 `2631-spec-three-arms` 的 F2/F5 缺口）。
- 已读既成事实：`pathlm/model.py::_mtp_nodes`（node1/2/3 + 一级 token 条件链 `chain2`）、
  `pathlm/decode.py::decode_spec/_run_stack_multi/truncate/_slice_nodes`、2631 的
  `run.py`/`spec_arms.py`/`analysis.md`（复用其 checkpoint 与 prompt 口径）、
  `.scratch/55-review/H-PROBLEMS.md` F2/F5、`2632/probes_output.json` 的部署口径 recall@k 探针。
- 新实验目录 `axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/`；
  预登记契约先写（`README.md`），再写码。
- **中途收到 axis-lead 追加**（不改变 P1–P5，加臂 + 交叉检查）：
  ① 加 `tree_direct_k2`（N+2 位取 `softmax(node2)` top-2）与 2632 探针 `direct@2`
  （13m_s0 0.8605 / 13m_s1 0.8922 / 53m_s0 0.9260）对照，作为**实现有效性交叉检查 C1**；
  ② 加 `tree_qpw_k2`（`softmax(q_pw)` top-2）与 `tree_direct_k2` **同 K、同 FLOPs** 比较候选排序分布；
  ③ 13m_s0 必须先跑满 9 臂；53m_s0 可只跑 `{native, direct1, tree_direct_k2}`。
  已并入 `README.md`（新增臂表行、C1 段、否证条件 4、臂优先级段）。

## 2026-10-13T17:26:30+08:00 · 接任 writer（T1：离线重算判据 + analysis.md）

- 命令：`python3 recompute_criteria.py`（CPU，不取锁）。
- 结果：写出 `criteria_fixed.json`。重算的臂级 α₁/τ 与 `results.json` **逐位相等**
  （`drift_vs_results_json` 全空）⇒ 确认 run #1 的 P1/P2 全 false 是**判据取键缺陷**（树臂 vs 最强
  单臂的 `paired_diffs` 键方向相反），**读数本身未变**。前 worker 16:55 已在 `run.py` 加
  `paired_dir()` 修取键 + 补 C1 `cis_overlap`/`probe_wilson_ci95` 字段（代码已修，无需重跑 GPU）。
- 重算判据：P1/P2 三个 checkpoint 全部 **true**；P3 false（13m_s0 Δα₁=−0.0048 /
  13m_s1 +0.0039，均 ≤0.014 且 CI 含 0）；P4 true（全臂逐位无损）；C1 三处均过。
- 归档：`analysis.md`（人话摘要 + 臂表 + 判据逐条 + 代价轴 + 否证条件 + 未测项）。
- 证据锚点：`criteria_fixed.json`、`recompute_criteria.py`、`analysis.md`。

## 2026-10-13T17:27:18+08:00 · T2 变异验证（mutate.py）

- 更正：上一条 T1 的时间戳 17:26:30 是**估值**，未先跑 `date -Iseconds`；其实际执行约在 17:27
  （命令日志：`python3 recompute_criteria.py` + `analysis.md` 写入）。本条目起时间戳一律取自实际时钟。
- 命令：`python3 mutate.py`（CPU，不取锁）。
- 结果：3 条预登记变异**全部抓红**（均为 `test_failure`，非收集/环境错误）：
  M1 候选集 K=1（`tree_direct_k2` 退化 = `direct1`）⇒ `test_candidate_counts_match_preregistration` 红；
  M2 接受判据改「无条件接受」⇒ 无损性门禁 `test_tree_streams_are_bit_identical_to_native_greedy` 红；
  M3 `combo_pw` 权重固定 0.5（退化成 `combined_fixed`）⇒ `test_combo_pw_differs_from_combined_fixed_on_the_fixture` 红。
- 逐字节还原：`spec_tree.py` sha256 变异前 = 还原后 = `71d98b26…8fb62f`；`all_restored_bitwise=true`。
- 证据锚点：`mutation_report.json`、`mutate.py`。

## 2026-10-13T17:27:18+08:00 · T3 对照物清单

- 命令：新增 `controls/README.md`（纯文档，CPU）。
- 结果：列明三件对照物——`native`（生产逐 token 贪心，归一 100%，本实验已跑）、
  诚实对手 = 1-step AR 两次前向（**本实验未跑**，属 6.2.3/6.2.1 的对照，本实验内以单候选臂为对照）、
  `oracle_accept`（**只作 τ 上界附报、不作判据**，SCHEMA §11）。
- 证据锚点：`controls/README.md`。

## 2026-10-13T17:27:18+08:00 · T4 收尾（本 WP 交付）

- run #2（16:56 启动）被运行时连带杀死、**未产出任何读数**；GPU 锁已被回收，无残留进程。
- P1/P2 判据缺陷（配对差取键方向相反 ⇒ 全 null/全 false）已由**离线重算** `criteria_fixed.json`
  修正；读数本身未变（重算 vs 落盘逐位相等）。前 worker 16:55 已在 `run.py` 修复取键代码路径。
- 13m_s1 的 C1：探针点 0.8922 未落在本臂 α₁ 95% CI [0.9008, 0.9426] 内（低 0.00086），但两 CI 重叠、
  实测方向为**高于**探针（非「显著低于」的 bug 信号）⇒ 判 C1 过，并已在 `analysis.md` 记录该不确定性。
- 交付物：`analysis.md` / `criteria_fixed.json` / `recompute_criteria.py` / `mutate.py` /
  `mutation_report.json` / `controls/README.md`。未 commit、未 `git add`。
- 证据锚点：`analysis.md`、`criteria_fixed.json`、`mutation_report.json`。
