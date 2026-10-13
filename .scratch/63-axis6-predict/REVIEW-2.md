# 复核 #2（WP 6.2.2 · 2640-tree-conf-13m · 独立复核者 #2）

## 0. 钉住的 revision 与工具状态（本报告证据均绑定于此）

- `git rev-parse HEAD` = `21f390dfe65a49d54a73c4d95606540262528ae1`（= 复核 #1 处置提交，未变）。
- 复核时间：`date -Iseconds` = **2026-10-13T17:50:15+08:00**（起 17:48:12 → 止约 17:51）。
- 实际耗时：**约 3 分钟**（远低于 ≤12 min 硬预算；因所有命令均为秒级）。
- 复核者：axis6-lead 派出的 **fresh reviewer #2**，未参与本 WP 任何写作/修复；**只读**
  （除本报告 `REVIEW-2.md` 外未改/建任何仓库文件；临时脚本只写 `/tmp`）。
- **工作树脏项清单**（本复核期间 `git status --short`）：

  | 项 | 内容 | 归因 |
  |---|---|---|
  | ` M .scratch/63-axis6-predict/TICKET.md` | 仅在第 9 节追加复核记录（+49/−7） | **父级 axis6-lead 并发写入**（mtime 17:48:27，晚于本复核起始 `git status`）——非本复核所改 |
  | ` M .../2640-tree-conf-13m/mutation_report.json` | 仅 3 行 `pytest_summary` 计时字符串（如 `2.15s`→`2.19s`） | 本复核起始前即为脏（axis-lead 17:45 代跑 `mutate.py` 的副产物）；本复核自己再跑 `mutate.py` 亦触发同一计时抖动 |

  两者均**不影响被审 artifact**（见 §2）。被审的 4 个关键文件在工作树与 HEAD 上逐字节相同。

- 四个关键文件 `sha256sum`（本复核亲自跑，逐一比对 `.scratch/63-axis6-predict/run1-snapshot/README.md`）：

  | 文件 | 实测 sha256 | 与快照 README 比对 |
  |---|---|---|
  | `2640.../results.json` | `1302b6785630f32815651a02340ecd72b556c7e5ec44657311994ee74c13dbaa` | **相同** |
  | `2640.../cycle_log.jsonl` | `d2073370c7985c7b1bc0981e6d8a73231462674bbd2ea49ae0f54ef48a592290` | **相同** |
  | `2640.../run_log.txt` | `ba48b06a0bcee2420c119c61a6c2687dee544225d1e30c29b60fb1a600a2984c` | **相同** |
  | `2640.../spec_tree.py` | `71d98b2687b2cc36d362514bd8bcc014aa3818f4aad4ef14c83b56e9be8fb62f` | **相同** |

---

## 1. T1 修复核验（复核 #1 的 4 条 normal + 1 条 info）

### T1.1（F1）10 倍数值错 —— **PASS**
- 命令：`grep -rn "0.00086" .scratch/63-axis6-predict axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m`
- 结果：命中 **3 处，全部是日志/复核留证**，`analysis.md` 已无该串：
  - `.scratch/63-axis6-predict/WP-6.2.2-LOG.md:60`（T4 原始流水，日期条目，按「只增不改」保留）
  - `.scratch/63-axis6-predict/WP-6.2.2-LOG.md:79`（更正条目，**引用旧值**说明它是 10 倍错）
  - `.scratch/63-axis6-predict/REVIEW-1.md:23/25`（复核 #1 留证）
- 新值核对：`grep -n "0\.0086" analysis.md` → 行 158/163/167，均为 `0.0086`。
- 数据源核对（亲自跑）：
  - `results.json::checkpoints.13m_s1.arms.tree_direct_k2.alpha1_ci = [0.900752, 0.942632]`
  - `2632-alpha1-attribution/probes_output.json` 的路径是 **按 checkpoint 分布的**：
    `13m_s1.P3_free_gen_topk_vs_T2["2"]["direct"] = 0.8922`（复核 #1 写的顶层键 `P3_free_gen_topk_vs_T2` 少了 `13m_s1` 前缀，值正确）。
  - 算术：`0.900752 − 0.8922 = 0.008552 ≈ 0.0086` ✓（不是 0.00086）。**结论与复核 #1 一致、修复正确。**

### T1.2（F2）代价口径「K 行平铺 vs 节点共享树」—— **PASS**
- 亲自读 `spec_tree.py`：`decode_tree` 内 `spec_tree.py:374 rows = [[v1, c, d3] for c in cands]`
  ⇒ **每行 3 个位置、共 K 行 = 每轮 `K·(1+γ) = 3K` 个主干新位置**。
- 共享前缀：`spec_tree.py:283-284` 用 `dec.kv[i][0/1].expand(B, -1, -1, -1)` ⇒ **只做 KV 广播（只读），不重算**。
  与 `analysis.md` §5 的披露逐字一致。
- 算术：节点共享树位置数 = `1 + K + K = 2K+1`；倍数 `3K/(2K+1)`：K=2 ⇒ `6/5 = 1.20`，K=4 ⇒ `12/9 = 1.333`
  ⇒ 文中「约高 **1.2–1.33×**」**正确**。
- 结论：`analysis.md` §5 的「口径局限」段与代码行构造一致，算术自洽。

### T1.3（F3）`6.2/project.yaml` 与新读数一致性 —— **PASS**
- 抽 5 个数核对（亲自从 `results.json` 复算）：
  | 量 | project.yaml 写法 | 实测 | 一致 |
  |---|---|---|---|
  | 13m_s0 tree_direct_k2 Δα₁ | 落在 `+0.0618…+0.1269` | `+0.0934` | ✓ |
  | 13m_s0 tree_direct_k2 Δτ | 落在 `+0.109…+0.236` | `+0.1857` | ✓ |
  | 13m_s0 tree_topk4 Δτ | `+0.236`（上界） | `+0.2361` | ✓ |
  | 53m_s0 tree_direct_k2 Δα₁ | `+0.0905` | `+0.0905` | ✓ |
  | K=4 的 R | `4.45–4.51` | `4.454`（13m_s1）/ `4.5137→4.51`（13m_s0） | ✓ |
- 过时表述 `grep`：`grep -n "待 6.2.2 补跑\|完整树尚未\|尚无独立复核\|尚未复核\|待立项" project.yaml`
  → **无命中**。`conclusion`(g)/(h)/(i) 已改为已判结论 + 剩余未测；`gaps` 首条已由「主判据未测」改为剩余项清单；
  `status: 进行中`（主判已答、milestone 稳健性点未齐）；WP 6.2.2 `status: 已完成`。
- **F3 通过**（唯一措辞张力见 T4·info-1，不构成 blocker）。

### T1.4（F4）`analysis.md` 仍为「当前事实」体 —— **PASS**
- 命令：`grep -n "run #1\|run #2\|16:5\|16:4\|2026-10-13" analysis.md` → **无命中**。
  §2 已压成「读数 = `results.json`/…；判据 = 离线重算；过程见 `WP-6.2.2-LOG.md`」的当前事实陈述。
- 流水确实落在 `.scratch/63-axis6-predict/WP-6.2.2-LOG.md`（含 17:27/17:47 的日期条目与更正条目）。

### T1.5（F5, info）C1 的 Wilson CI 注明用 `P3_free_gen_overall.n` —— **PASS**
- `analysis.md:168`：「探针的 Wilson CI **按 `probes_output.json::P3_free_gen_overall.n`（337/334/311）算**……
  若 k=2 的实际评分轮数少于 overall，探针 CI 会略窄（不影响 C1 判词）」。
- 亲自核对：`probes_output.json` 的 `13m_s0/13m_s1/53m_s0.P3_free_gen_overall.n` = **337 / 334 / 311** ✓。

---

## 2. T2 本复核亲自实跑（不复用任何人自报值）

| 命令 | 结果 | 关键读数 |
|---|---|---|
| `python3 -m pytest .../test_spec_tree.py -q` | **PASS** | `22 passed in 5.70s` |
| `cd 2640... && python3 mutate.py` | **PASS** | `M1/M2/M3: caught_red=True restored=True`；`{"all_caught_red": true, "all_restored_bitwise": true}`；跑后 `sha256sum spec_tree.py` = `71d98b26…8fb62f`（回到原值） |
| `cd 2640... && python3 recompute_criteria.py` | **PASS** | 跑后 `git status --short` **未出现 `criteria_fixed.json`**（产出确定、逐字节不变）；三 ckpt 的 `drift={}`（`drift_vs_results_json` 全空）；9/9「树臂×ckpt」P1/P2 全 `pass=True` |
| `python3 tools/summary.py --check` | **PASS** | `OK：46 个 project.yaml 全部符合 docs/research-schema.md`（rc=0） |
| `python3 tools/check_tree.py` | **PASS** | 同上（rc=0） |
| `git diff --stat 01d25ff..HEAD -- pathlm docs tools` | **PASS** | 输出为空 ⇒ 未越界改共享引擎/规范文档 |
| `git rev-parse HEAD` | — | `21f390dfe65a49d54a73c4d95606540262528ae1` |
| `sha256sum` 4 个关键文件 | **PASS** | 与 `run1-snapshot/README.md` 记录的 4 个哈希**逐一相同**（见 §0 表） |

---

## 3. T3 本复核独立复算（自写脚本 `/tmp/rev2_recompute.py`，未复用 `recompute_criteria.py`/`recompute_check.py`）

### T3.1 臂级 α₁/τ —— **PASS**
从 `results.json::checkpoints.13m_s0.arms[arm].per_prompt[*].accepts` 自建：
- `tree_direct_k2`：α₁=0.898268 / τ=1.608225（落盘同）
- `combined_fixed`：α₁=0.804829 / τ=1.422535（落盘同）
- `chain1`：α₁=0.601206 / τ=1.074935（落盘同）

落盘值本身是 6 位小数（`0.898268`），我的全精度复算 = `0.8982683982…`，**逐位相同到落盘精度**。

### T3.2 配对 bootstrap 95% CI：`13m_s0 tree_topk4 − combined_fixed` —— **PASS**
自写实现（prompt 聚类重采样、B=2000、`np.random.default_rng(2640)`、percentile 2.5/97.5，池化 α₁）：
- 我复算：point Δα₁ = `0.126889`，95% CI = **`[0.108006, 0.147050]`**
- `analysis.md` §4 P1 = `[0.1080, 0.1471]`
- 差异来源：**纯 4 位小数显示舍入**（`0.147050 → 0.1471`），非 percentile 法差异；底层值与
  `criteria_fixed.json`/`recompute_criteria.py` 输出**完全一致**。**独立复现成立。**

### T3.3 无损性抽查 —— **PASS**
- 遍历全部 `(ckpt, 非对照臂)`：`stream_identical_to_native == true` **且** `per_prompt[*].identical_to_native` 全 true
  **且** `first_mismatches == []` ⇒ 全过（13m_s0/13m_s1/53m_s0）。
- `native_is_plain_decoder_path = (True, True, True)`、`native_prefix_consistent = (True, True, True)`。
- `run_log.txt`：三个 checkpoint 均打印 `native streams ready (n_new+γ=202), prefix_consistent=True`
  （native 走生产 `pathlm.decode.decode`）。

---

## 4. T4 新引入的不一致（主动证伪结果）

**未发现 blocker。** 逐条尝试证伪的结论：

- **project.yaml 长结论 vs analysis/criteria**：5 个数字（§1 T1.3）全部一致；`gaps` 无信息丢失
  （9 条剩余项齐全：温度>0/batch>1、二级条件树、53.9M 只 3 臂、K 行平铺代价口径、无低局部先验、
  无 pareto、无干净端到端加速、2510 B2 权重不可得、整段并行归 6.3）。
- **修复提交的越界**：`git show --stat 21f390d` 只改 4 个文件（`REVIEW-1.md`/`WP-6.2.2-LOG.md`/
  `analysis.md`/`6.2/project.yaml`）；`cee655b..HEAD` 亦只这 4 个 ⇒ 未碰实现/共享引擎。
- **预登记判据未被事后放宽**：`README.md` **不在修复提交内**，mtime = `2026-10-13 16:30:00`，
  **早于 run 开始 16:38** ⇒ 预登记契约不可变。`analysis.md` 的判词（P1/P2 成立、P3 否证、C1 过）
  逐条对应 `README.md` §4 P1/P2/P3/C1 与 §5 否证条件，未改判。

**info-1（措辞张力，非 blocker）**：`project.yaml` WP 6.2.2 现为 `status: 已完成`，其 `acceptance`
字面写「**完整树** / 多候选验证跑齐」，而同一文件的 `gaps` 写「节点共享树（≈2K+1）与**真树注意力
验证未实装量测**」。核心交付（补回概率多候选/树 + 端到端判 P1/P2/P3、C1）确已完成，节点共享注意力
属代价核算的精化、且已在 `gaps` 显式披露；但「已完成」+「完整树…跑齐」的组合易让独立审计误读为
「节点共享树已实装」。建议把 acceptance 或 WP 描述里的「完整树」与 `gaps` 口径统一（如注明
「树 = K 行平铺实现；节点共享树另列为未实装」）。不影响 P1/P2 判词。

**info-2（工作树卫生）**：`mutation_report.json` 把**逐次运行会变的 pytest 计时**（`2.15s` 等）
写进入库事实文件 ⇒ 任何人重跑 `mutate.py` 都会让工作树变脏（本复核起始前即已脏，仅 3 行计时）。
建议删该计时字段或明确其非判据性质。

**info-3（concurrent writer）**：`.scratch/63-axis6-predict/TICKET.md` 在本复核期间被父级 axis6-lead
并发写（17:48:27，仅第 9 节追加）。按只增不改的工单纪律可接受，但复核期间**仓库存在第二个写入者**；
被审 4 个 artifact 未受影响（工作树逐字节等于 HEAD）。

**info-4（父级账本措辞继承）**：`project.yaml` conclusion (g) 写「R 从单臂 1.16–1.24 抬到
`1.73–2.36（K=2）`」，其中 `1.73` 来自 `tree_union2`（实测 K≈1.51/1.57，非 K=2）。数值本身正确，
仅 `(K=2)` 标签把 union2 并入 K=2 组，与 `analysis.md` §1「K=2/union2」同源。非 blocker。

---

## 5. T1–T4 逐条结论

| 项 | 结论 |
|---|---|
| T1.1 F1（10 倍错） | **PASS**（analysis.md 已改 0.0086；数据源算术 0.900752−0.8922=0.008552 ✓） |
| T1.2 F2（口径披露） | **PASS**（代码行构造 `[[v1,c,d3] for c in cands]` 一致；1.2–1.33× 算术正确） |
| T1.3 F3（SSOT 自洽） | **PASS**（5 数一致、无过时表述；措辞张力见 T4·info-1） |
| T1.4 F4（日期流水） | **PASS**（analysis.md 无过程叙事；流水在 LOG） |
| T1.5 F5（Wilson CI n） | **PASS**（已注明 `P3_free_gen_overall.n`=337/334/311） |
| T2 自跑项 | **全 PASS**（pytest 22、mutate 3/3 抓红+逐字节还原、recompute 确定、summary/check_tree OK、无越界、4 哈希相同） |
| T3 独立复算 | **全 PASS**（臂级逐位、配对 CI [0.108006,0.147050]、无损性全 true） |
| T4 新不一致 | **无 blocker**；4 条 info（措辞张力 / mutation_report 计时 / 并发写入 / K 标签） |

**报告位置**：`.scratch/63-axis6-predict/REVIEW-2.md`（本文件，写工具落盘）。

在本复核**钉住的 revision `21f390d`**、且**本复核实际跑过的范围**内：复核 #1 的 4 条 normal + 1 条 info
**均已正确修入**；T2 六项自跑命令全绿；T3 三项独立复算与落盘/文档**逐值一致**（配对 CI 差异仅为
4 位小数舍入）；未发现修复引入的 blocker，仅 4 条 info。

**VERDICT: PASS**
