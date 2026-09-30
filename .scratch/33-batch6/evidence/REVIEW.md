# 独立复核（契约 T，fresh context，只读）

## Review（契约 T · 独立复核，只读，fresh context）

复核对象：`.scratch/33-batch6/issues/01-batch6.md` 契约 Q/R/S 三条线。判据原文已逐字对照契约文件（第 74 行以前的 Q/R/S 三节）。未运行 shell；`pytest tests/ -q` 留给主 Agent（见残项）。

### 清单 ① 判据线是否被改（含逐字/摘要标注）

| 线 | 结论 | 证据 |
|---|---|---|
| Q | **pass** | `README.md §4` 的三条判据与契约 Q 逐字一致（≤0.014 / ≥+0.050 / 两点并列 / 不得外推内插）；`§2` 动机段逐字一致 |
| R | **pass** | `README.md §4` 三条与契约 R 逐字一致；`§0` 开头统一纪律段逐字一致 |
| S | **fail（标注不实，非削弱）** | `README.md §4` 标题写「**逐字**抄录契约「判据」，**未改一字**」，但 blockquote 末句多出契约里**没有**的一句「活性证据必须是**实际 KV 张量形状**。」（契约 S 判据 止于「必须附三臂同方法斜率与重跑散度。」，见契约文件 S 节）。同文档 §1 把契约的「动机：…工业界**的**另一条…」抄成「**背景：**…工业界另一条…」（去「的」、改标签），也标「逐字」。**阈值与条件本身未被改动**，是标注保真度问题 |

### 清单 ② 三道对账

- Q：**pass**。`reconcile_report.json` 252 项 0 失败；算术/任务/口径三道在 `analysis.md §6` 齐备；抽查已复核：税 −0.002813（1.480756−1.483569）、补回 42.10%（0.149528−0.086566=0.062962，/0.149528）、beyond 整桶加权 (109×0.559633+494×0.457490)/603=0.475954、`n_evals` 36+71+256+9+36=408 ✅。
- R：**pass（受限于无 linear 数据）**。34 项 0 失败；需要两臂的 Δ/重跑散度**如实记为不可得**而非跳过（`reconcile_numbers.py` 里显式 `reason`）。任务对账写明「本实验没有可对账的支持假设数字」。
- S：**pass**。220 项 0 失败、`all_pass=true`；KV 字节 12·2·nkv·32·2=12288/3072/1536 与实测张量形状逐位一致；参数差 −2·(d−nkv·dh)·(d+1)·L=−1,184,256/−1,381,632 逐位一致（且记录了「漏 (d+1)」的自查命中）✅。

### 清单 ③ 对照臂三要素与同一任务定义

Q / R / S **全部 pass**：三份 `analysis.md` 都写了「输入分布 / 预测什么 / 部署对应」；Q/S 用同一 `window_seed=2 + 24×16` 位置集；R 明确「两臂同数据流同批顺序、无假上下文+真目标」。

### 清单 ④ 机制活性（每 200 步、形状类）

| 线 | 结论 | 证据 |
|---|---|---|
| Q | **pass** | `train/window192/train_log.jsonl` 61 行、每行带 `window_mask`（可见对 31300/32896=0.951483）+ 功能探针 max\|Δlogit\|=0.217099；单测 `test_window192_keep_mask_counts_and_boundary_at_real_scale` |
| R | **fail（数值溯源）** | 活性机制本身活着（见下 P1）：留存日志 `runs/linear-12k-s0/train_log.jsonl`（= attempt3）3 行带 `attn_linear`，phi_min→0.0、den_min→1e-06；但报告表格数字对不上留存日志 |
| S | **pass** | `runs/gqa-12k-s0/train_log.jsonl` 每 200 步带 `kv_heads`（n_kv_heads/qkv_out_width/kv_rows_fro_mean）；60+1 行；定性活性用**实际张量形状** `[8,2,256,32]`/`[8,1,256,32]` + qkv 宽度 384/320 + 各 KV 头 K 投影行范数 2.5702/3.1629 + 两两最大差 0.3688 |

### 清单 ⑤ analysis 数字 vs `results.json`/日志（抽查 >8，含成本/次数/等式/哈希）

**对上的**（抽样）：Q `wall=8.23 min`、`1840.7/1984.0 MiB`、评测 `37.88 s / 408 次 / 194.7 MiB`、步时 42.1 vs 40.1（由 `train_log.jsonl` 的 `min` 字段回算：42.1×599/1000+0.01=0.4303→0.43 ✅）、beyond 0.475954/0.444444、完全不可见子桶 0.457490/0.457490、重跑 0.003486；S 35.52+6.73 s=35.63 min、45.23/42.21 ms、2244/2160/2158 MiB、bpc 1.4820/1.4963/1.4959/1.4938、重跑散度 0.0004、`checkpoint sha256` 表；R `full` 5.75/6.01 min、1.47 GB、理论 MACs 49664/16896、比值 2.582/2.352/1.736/1.528。

**对不上的**（详见 Findings）。

### 清单 ⑥ 口径标注

三条线 **pass**：**解析内存**与**实测显存**都分两栏（Q：KV 位置 256→200 −21.9% vs 194.7 MiB 不降；R：推理状态 −93.6% vs 训练峰值 +1.41–1.47×；S：−75%/−87.5% vs 265–280 MiB 不按比例降）；快照/渐近标注齐（三条都是 matched-step 快照）；外推禁令逐条写明。

### 清单 ⑦ pytest

**未复核**（本 agent 无 shell）。需主 Agent 跑 `python3 -m pytest tests/ -q`（基线 224）并核对 `tools/evalkit/retention.py` 只增不改（三条线记录的 sha 均为 `8320f03a…` ✅）。

### 本轮特别关注

- **(a) 外推**：**pass**。Q 明确「两点不得内插/外推，不给悬崖位置」；S 明确「不得内插 4 头、不得外推规模/步数、0.0025≪0.014 不许排序」；R 明确「理论 T*=96/32 是公式、实测 256–2048 全段未回本、T>2048 未测」——三处都没有把单点当趋势。
- **(b) 省内存两口径**：**pass**（见清单 ⑥）。
- **(c) KV 头数真的改形状**：**pass**。`[8,2,256,32]`/`[8,1,256,32]`、qkv 输出宽度 384/320（不是 3d）、权重非零、各 KV 头行范数不同；变异 S1/S3/S5（只改标签/形状错位/活性造假）全抓红；单测 `test_kv_heads_actual_tensor_shapes_and_analytic_memory`。
- **(d) 线性注意力真 O(T)**：**pass**。`linear_attn_chunked` 用块级 `cumsum` 状态，无 `[T,T]`；单测 `test_linear_attn_is_linear_in_T_not_quadratic` 用 `FlopCounterMode`（×8T ⇒ ×≈8 vs 全注意力 ×>50）+ `TorchDispatchMode` 形状追踪（最大中间张量 ≤ (T/C)·C²），变异「full-attention-rename（直接调 SDPA）」被抓红。
- **(e) 算术不自洽/查无出处**：**fail（部分）**，见 Findings。

---

## Findings

**P1 — R：`stability.md §2` 活性量表的 step-200/400 数字与留存日志不一致（6 个格），`analysis.md §5.3` 另有一个无出处数字**
- 位置：`axes/A-mixing/A6-mixing-operator/experiments/2616-linear-attention-13m/stability.md:37-41`、`analysis.md:167-168`
- 证据：唯一留存的线性训练日志 `runs/linear-12k-s0/train_log.jsonl`（末行 step 400）为
  step200 `phi_max=7.0625, den_min=2.984375, inter_mass_frac=0.582868, state_norm=471.331`；
  step400 `phi_max=14.3125, den_min=1e-06, inter_mass_frac=0.546845, state_norm=406.0286`。
  报告写的是 `phi_max 7.125 / 14.375`、`den_min 2.344`、`inter_mass_frac 0.583 / **0.455**`、`state_norm 471.7 / 443.9`。
  同一表里 `loss/gnorm` 列却与 attempt3 逐位一致（2.6053/1.16、2.3581/0.97）⇒ 该表混用了被覆盖的 attempt A/B 的数字（`analysis.md §5.3` 的 `inter_mass_frac … 0.494 …` 在全部留存文件中查无出处）。
- 影响：不改变「干预不可行」的结论（留存日志仍显示 den_min→1e-06、phi_min→0.0），但这正是本批「报告数字必须与日志逐一对得上」纪律点名过的类别。
- 最小修法：把该表 step200/400 的 6 个格改成 attempt3 留存日志的值；若坚持用 attempt A/B 的数字，须逐格标注来源并说明该日志已被覆盖。

**P1 — S：机器可读判据 `results.json::criteria.per_arm.gqa_rep.verdict` 与本实验**预登记规则**及 `README/analysis` 相反**
- 位置：`axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json:3203`（「近无损（\|Δ\| = 0.0139 < 0.014）：**写入参照线**（KV −75.0%）」）
- 证据：同文件 `rerun_divergence.same_verdict = false`；`README.md:62-63` 预登记「`|Δ_schema|` 是否 < 0.014 在 gqa 与 gqa_rep 上必须一致，**否则该臂判读作废**」；`README.md:125-126`「**不写进**「近无损」参照线」；`analysis.md` §1/§4 同调。即：同一交付里 `results.json` 说「写入参照线」，`README/analysis` 与预登记规则说「作废/不写」。
- 最小修法：把 `gqa_rep.verdict` 改成「跨门槛 ⇒ 依预登记规则该臂判读作废，不写入参照线」，与 `rerun_divergence.same_verdict=false` 对齐。

**P2 — S：`README.md §4` 的「逐字…未改一字」块多出契约没有的一句（含 §1「背景」改写）**
- 位置：`README.md:44-47`（多出「活性证据必须是**实际 KV 张量形状**。」）、`README.md:18`（动机→背景、去「的」）。该句实为父任务硬纪律，可保留，但不应标在「逐字」块内。最小修法：移出 blockquote，或把标注改成「否则摘要/纪律并入」。

**P2 — R：变异次数在报告里写 5，产物为 6**
- 位置：`README.md:168` 与 `README.md:206`（「5 处变异验证 / 5 处变异全抓红」）vs `mutation_report.json`（6 条、`n_red: 6`）与 `MODEL.md:31,82`、`analysis.md:152`（均为 6）。最小修法：README 改 6。

**P2 — S：单测条数在报告里写 6，实物为 7**
- 位置：`README.md:9`（「6 条，`-k kv_heads`」）vs `MODEL.md`（「7 条」）与 `tests/test_depth_ar.py`（`test_kv_heads_*` 6 条 + `test_train_cli_kv_heads_plumbed_to_config` 1 条 = 7 条匹配 `-k kv_heads`）。最小修法：README 改 7。

**P2 — R：卡死归因自相矛盾（「只在 driver 组合下出现」被本实验 attempt C 否证）**
- 位置：`analysis.md §3 行 5` 与 `analysis.md §11`（「卡死只在「driver 拉起 + 父进程持 CUDA 上下文并轮询」的组合下出现」）vs 同文档 `§3 行 6`/`§7.1` 与 `stall_diagnosis.json::attempts[2]`（attempt C「**直接** nohup 启动 train_depth_ar.py（无 driver 父进程）」仍停在 step 400，并写明「卡死与 driver 父进程无关」）。最小修法：把 §3 行 5/§11 的结论改成「独立进程二分未复现、但直接启动的 attempt C 仍在 step 400 停摆 ⇒ 非确定性、与 driver 无必然关系」。

**P2 — R：报告的注意力 MACs 等式与自身数字不自洽（漏去归一化项的 +2d）**
- 位置：`README.md §1`、`analysis.md §6` 表：写「`2·d·dh + 2·C·d` = 49,664」「`2·d·dh` = 16,896」。按 d=256,dh=32,C=64：2·d·dh=16,384、+2·C·d=49,152，都不是报告值。实际实现（`run.py:322,327`）为 `2·d·dh + 2·C·d + 2·d` / `2·d·dh + 2·d`。同理理论交叉点写 `dh+C=96`/`dh=32`，而按代码应为 `dh+C+1=97`/`dh+1=33`。百分比（−62.1%/−87.1%）与数字自洽。最小修法：把等式补成 `2·d·(dh+1)+2·C·d` 与 `2·d·(dh+1)`（并相应改 T*）。

**P2 — R：`README.md` 断言「峰值显存 net ≤5GB」与自身实测矛盾**
- 位置：`README.md §6`（「峰值显存 net ≤5GB（实测见 `analysis.md §6.1`）」）vs `analysis.md §6` 实测表：T=2048 时 full 5.585 GiB、linear **8.23 GiB**（`results.json::cost.measured`）。最小修法：改写为「训练的 5GB 净预算由 12k 主跑满足；成本口径的 T=2048 探针峰值 5.59/8.23 GiB **超出**该预算，仅作短探针登记」。

**P2 — R：`README.md §6` 的墙钟出处写错**
- 位置：`README.md:142` 写「实测墙钟（`results.json::wall_total_min` …）」，但 `results.json::wall_total_min = 0.24`（组装跑），5.75/6.01 min 出自 `run_log.txt:3` 与 `runs/full-12k-s0`。数字本身可溯，指针写错。最小修法：改引 `run_log.txt`。

**P2 — S：评测墙钟两份口径差 2.29 s 未解释**
- 位置：`results.json::cost.eval_wall_seconds_by_arm`（1.38+1.03+1.04+0.99 = **4.44 s**）vs `battery_wall_seconds = 6.73 s`、`total_gpu_wall_minutes_measured = 35.63`（用 6.73）。差额可解释（三次调用 vs 逐臂），但报告未写。最小修法：加一句口径说明。

**P2（低）— Q：`analysis.md §2/§1` 的「30 倍以上」把噪声级数值当分母做比值**
- 「32 窗口税 +0.0866，是本实验的 30 倍以上」= 0.0866/0.002813；而本实验自述该税落在训练噪声 0.003486 内（`stability.md §5`）。措辞已在别处限定，但该句容易被误引成「192 比 32 好 30 倍」。建议改为「32 档的税是本实验税读数绝对值的 ~25–30 倍（而本实验税在噪声内、方向不稳健）」。

---

## 可疑数字清单（供主 Agent 落盘/转执行者）

1. `stability.md:39-40` den_min `2.344`、den_min step400 ✓、phi_max `7.125`/`14.375`、inter_mass_frac `0.583`（✓）/`0.455`、state_norm `471.7`/`443.9` —— 与 `runs/linear-12k-s0/train_log.jsonl` 不符；`analysis.md:168` 的 `0.494` 查无出处。
2. `results.json:3203` gqa_rep `verdict=写入参照线` —— 与 `rerun_divergence.same_verdict=false`、`README:63,126` 矛盾。
3. `README(R):168,206` 变异「5 处」vs `mutation_report.json n_red=6`。
4. `README(S):9` 单测「6 条」vs 实际 `-k kv_heads` 7 条。
5. `README(R) §1 / analysis §6` 的 MACs 等式 `2·d·dh(+2·C·d)` 与 49,664/16,896、T*=96/32 不自洽（漏 +2d）。
6. `README(R) §6`「峰值显存 net ≤5GB」vs 实测 8.23 GiB@T=2048。
7. `README(R):142` 引 `results.json::wall_total_min`（=0.24）作 5.75 min 出处。
8. `analysis(R) §3 行5/§11`「卡死只在 driver 组合下出现」vs `stall_diagnosis.json attempts[2]`。
9. `README(S) §4`「逐字…未改一字」块多一句；`§1` 动机段改写仍标逐字。
10. `results.json(S)` 逐臂评测墙钟和 4.44 s vs `battery_wall_seconds` 6.73 s（未说明）。

## 三条线的判据达成（复核后确认执行者读数无误）

- **Q = 成功（带保留）**：判据①税 **−0.002813**（native；release 口径 +0.005190）≤0.014 ⇒ 近零成立；判据②两点（32 +0.086566 / 192 −0.002813）+ 四评测点、外推内插禁令已写；判据③beyond +0.031510<0.10 ⇒「丢了」，完全不可见子桶 gap 0.000。口径与活性齐备。
- **R = 失败（诚实负结果）+ 判据②已交付**：linear 臂三次停摆（step 800 / ~500 / 400）、无 checkpoint ⇒ 判据①③**无数据**（如实记为不可判定）；判据② 理论与实测成本、交叉点（孤岛 T=1024 首次 0.96×，整模型 256–2048 未回本）齐备。
- **S = 成功（带保留）**：MQA +0.0118<0.014 ⇒ 近无损（−87.5%）；GQA 跨门槛（+0.0143/+0.0139）⇒ 依预登记判读作废、不写参照线（但 `results.json` 文本需按 P1 修）；活性=实际张量形状。

## 结论 / Merge verdict

**OK with notes**（无 P0；2 个 P1 应在入账前修）。Q 三份交付自洽且判据线未改；R 的「不可行」负结果证据链（停摆 + 留存日志的 den→1e-06/φ→0 + 5 次二分 + O(T) 正证据）成立，但活性表数字溯源与两处内部矛盾需修；S 的数据、对账、形状类活性证据扎实，仅 `results.json` 的 gqa_rep 判读与预登记规则打架，以及若干次数/标注小错。

**残项（主 Agent）**：① `python3 -m pytest tests/ -q`（基线 224）；② 复核 `tools/evalkit/retention.py` 未改（三份记录 sha 一致 `8320f03a…`，建议实际核对工作树）；③ 批注/修正上述 P1×2 与 P2×8。

（未使用 `contact_supervisor`：无阻塞决策、无计划变更；按系统约定以本任务结果返回，主 Agent 可直接落盘。）