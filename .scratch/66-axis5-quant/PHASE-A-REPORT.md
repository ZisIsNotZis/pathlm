# 轴 5 · 量化轴 Phase A 报告

> 角色：axis5-lead（轴 5 量化轴 lead）｜时间盒：≤1.5h（实际 `20:04:43 → 20:1x`，+08:00）
> 范围：`axes/5-quant/**`、`.scratch/66-axis5-quant/**`（**未动**任何共享文件、其它轴、`docs/**`）
> 过程流水：`.scratch/66-axis5-quant/TICKET.md`｜事实（当前状态）：`axes/5-quant/*/project.yaml`

---

## 0. 人话摘要（先读这一段）

轴 5 的六格（KV 量化 / 权重量化 / 激活量化 / 低秩压缩 MLA 式 / fp8·大 head_dim / 窗口+量化+检索）
都是**业界已做透**的方向。所以本次 Phase A 的重点不是「动手做量化」，而是先把两件事钉死：
**每格相对业界 SOTA 的增量到底是什么**（写不出增量就等于造轮子），以及**判据/口径/否证条件**。

本轮交付三块：

1. **六格 `project.yaml` 补齐「想清楚」**：`role_map` 四角色实名（`owner = axis5-lead`、
   `writer = axis5-w5.X`、`reviewer = axis5-reviewer`（≠ writer）、`integrator = tree-integrator`）；
   `sota_baseline` 从「记忆·未核」升级为**本轮联网核实过的 arXiv 编号**（KIVI 2402.02750、KVQuant 2401.18079、
   Atom 2310.19102、GPTQ 2210.17323、AWQ 2306.00978、LLM.int8() 2208.07339、SmoothQuant 2211.10438、
   fp8 格式 2209.05433、DeepSeek-V2 2405.04434）；`gap` 逐格写死「相对 SOTA 的增量」并**明说哪些机制本仓没有增量**
   （例如 5.1 的块标尺 ≈ 业界的 per-channel/分块标尺、`block_tok` ≈ KIVI 的 K 侧 per-token）。
2. **执行 1 个便宜、纯 CPU、可证伪的 WP**：`5.1-WP2`「跨家族同字节预算重画（块标尺口径）」
   （实验 `2642-kv-same-byte-frontier`）。它把「量化 / 头共享 / 窗口」三招放到**同一条「每位置部署字节」横轴**上
   （此前那张图建立在**已作废的逐值标尺**口径上），并且**不用插值**：只比恰好同字节的点。
3. **独立复核**（fresh 上下文，只读）——判词见 §3。

**本轮最重要的一条结论（也是给下一步的指路）**：在同一条字节轴上，跨家族**恰好同字节**的比较点有 **7 个**
（432/768/816/864/1536/1632/3072 B/位置），**7 个水平上 bpc 最低的都是头共享族**（GQA/MQA）；
但头共享族的 checkpoint **参数更少 8.75%（GQA）/10.21%（MQA）**（本实验用磁盘实测与解析式逐位对上了），
所以这个优势**不能**读成「头共享机制更省内存且不亏质量」——要读纯机制必须补一个**等参数**臂（成本已登记）。

**本轮没做的**：任何 GPU 实验。`5.2`/`5.3`/`5.4`/`5.5`/`5.6` 的实测 WP 全部留给下一阶段（成本表见 §4）。

---

## 1. 逐项交付（六格）

| 格 | 本轮落地 | SOTA（本轮核实） | 写死的「相对 SOTA 的增量」 |
|---|---|---|---|
| **5.1 KV 量化** | role_map 实名；sota 升级；gap 划清机制归属；**WP2 执行完成**（实验 `2642`）；conclusion/gaps/roadmap/change_log 更新 | KIVI 2402.02750 / KVQuant 2401.18079 / Atom 2310.19102 | 机制非增量；增量 = ①含标尺的**部署字节**账 → 两腿质量前沿；②**严格前缀不变性**当独立硬门禁；③跨家族统一字节轴 |
| **5.2 权重量化** | role_map 实名；sota 升级；gap 划清；锚说明；change_log | GPTQ 2210.17323 / AWQ 2306.00978 / LLM.int8() 2208.07339 | 不重造 GPTQ/AWQ；增量 = 13M 受控含标尺字节账 + 权重/KV/激活**三类账**分开报 |
| **5.3 激活量化** | 同上 | SmoothQuant 2211.10438 / LLM.int8() / Atom | 不重造 SmoothQuant；增量 = 含标尺激活字节账 + 逐层标尺/离群通道做成**可失败活性判据** |
| **5.4 低秩压缩（MLA 式）** | role_map 实名；sota 升级；gap 划清 | DeepSeek-V2 2405.04434（`head_dim`=128） | 增量 = 与头共享在**同一字节轴**比较 + 「从零训练 + 上投影计算」代价入账 |
| **5.5 fp8 / 大 head_dim** | role_map 实名；sota 升级；gap 划清 | fp8 格式 2209.05433 / DeepSeek-V2 | 增量 = fp8 与 int8+块标尺**同轴同尺度**比质量（补 5.1 的 fp8 缺口）+ `head_dim`→G→字节的解析/实测对账 |
| **5.6 组合：窗口+量化+检索** | role_map 实名；sota 升级；gaps 补「组合臂应落在 2642 的 7 个同字节水平上」 | 生产默认（`记忆·未核`）+ KIVI/KVQuant/Atom | 增量 = 三条腿接同一字节轴 + 同一检索腿；**最高风险 = 把兄弟线单点重绘当新发现**（须自跑组合臂） |

`python3 tools/summary.py --check` 全绿（46 份 `project.yaml`）。

## 2. 本轮执行的 WP：`5.1-WP2`（跨家族同字节预算重画，块标尺口径）

- 实验目录：`axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/`
  （预登记契约 `README.md` + `config.json::criterion` 先写、再跑）；入口 `frontier_ledger.py`（纯 CPU、无 GPU 锁、不联网）。
- **门禁（预登记，运行后核）**：
  - **G1 PASS**：不用 `2632` 给的字节数，从原始因子独立重算 27 臂的部署字节，与 `2632::deployment_caliber`
    **逐位相等**（`max|Δ| = 0`）；顺带发现并写死「`points[*].bytes_per_position` 是**仅值口径**（int8: 6144），
    部署口径是 6528 ⇒ 两者不得混用」。
  - **G2 PASS**：进入前沿表的量化臂全为块标尺（`block`, G=32）；无逐值标尺（已作废口径）的量化数字参与。
  - 附加：6 个输入 sha256 与预登记一致（4 个 checkpoint 与 `2632/MODEL.md` 记录亦逐位相同）；
    27 臂的 bpc / copy 召回（含 Wilson CI）与父实验**逐位未改**。
- **主要读数**：
  - 跨家族**恰好同字节**组 **7 个**（432/768/816/864/1536/1632/3072 B/位置），组内 bpc 极差**全部越在力带**
    ⇒ 跨家族同字节比较在本网格上**可做、不需插值**。
  - 7/7 组的最优臂都是**头共享族**，但都带**参数混杂**（GQA 少 8.75%、MQA 少 10.21% 参数；解析式
    `L·(2·d_model+2)·(H−n_kv)·head_dim` 与磁盘实测**逐位一致**）⇒ 不能归因为纯机制。
  - 非头族之间的同字节对照出现**两腿结论不一致**：3072 B 处窗口臂 bpc 更好但在带内**不可排序**（+0.026462 < 0.036324），
    而锚点臂的长程 copy 召回 CI 不重叠（0.0934 vs 0.0589）⇒ 按 `2621` §1 纪律**显式标注**，不得只看一条腿。
  - fp8（解析 6144 B/位置 = ×0.50）比 int8+块标尺（6528）**省 384 B/位置**而质量未测 ⇒ **不得**称 int8 是 8 位最优；
    旧说法「同一 −75% 档 GQA 便宜 3.2×」随逐值标尺口径**一并作废**（块标尺下两族不再同档：3456 vs 3072）。
  - Pareto：原始 bpc 序非支配 6 臂（5 个是头族 ⇒ 参数混杂的可见后果）；越 0.036324 带 16 臂。
- **产物**：`README.md` / `config.json` / `frontier_ledger.py` / `results.json` / `table.md` / `analysis.md` /
  `MODEL.md` / `run_log.txt`（全部在 `2642-kv-same-byte-frontier/`）。

## 3. WP 与独立复核

### 3.1 第 1 轮（fresh 上下文，已完成）

- 复核者：`reviewer`（builtin agent，**fresh 独立上下文**、只读、不得为本项 writer），报告落
  `.scratch/66-axis5-quant/REVIEW-1.md`（复核者所在会话无写工具 ⇒ 判词由它返回、由 axis5-lead **原样落盘**并标注来源）。
- **判词：OK with notes**（blocker **0**；normal **2**；info 1）。逐判据：① 手算 14 臂字节账逐位相等 = PASS；
  ② 口径纪律（无逐值标尺混入、仅值口径分列）= PASS；③ 参数混杂解析=实测 = PASS；
  ④ 数字可追溯 = **FAIL（1 处数字错）**；⑤ project.yaml 实名/成本口径 = PASS 但 4 个子项它无法执行；⑥ 范围纪律 = PASS。
- writer 在复核期间已自查修掉 **4 处**（均 steer 通知复核者以当前树为准）：① §2 的 3072 B 对误写「越带」（实为带内不可排序 +
  召回腿反向 ⇒ 两腿结论不一致）；② `results.json::pareto` 补原始序支配表（21 条：`int4`←`mqa_int8`、`fp16`←`mha` 等）；
  ③ 不为旧记 −10.25% 编造归因；④ 显式核「必须不变」条款 —— 其中 ④ 经复核者**独立判定为恒真式（循环论证）**，即 N2。
- **N1（已修，bug 属实）**：`analysis.md` 原写「非支配集 6 臂，其中 **5 个是头族**」→ 实为 **4 个**
  （`mqa_int4`/`gqa_win64`/`mqa_int8`/`mqa`；另 2 个为 `int8`(quant) 与 `mha`(baseline)）。
- **N2（已修）**：① 该恒真门禁改名 `gates.readings_passed_through_untouched` 并在 `analysis.md` §1 明确标注
  「逐值透传的过程陈述，**不构成独立检验**」，`verdict` 改为**只由 G1/G2 决定**（新增 `verdict_basis`）；
  ② 删死代码（`parent_pairs`/`local_pairs`/`value_fingerprint`/`import math`）；
  ③ 新增**真检验** `gates.cross_artifact_table_md`：解析父实验**另一份产物** `2632/table.md`（markdown，
  另一代码路径）与 2642 rows 逐臂比 ⇒ **27 臂、不一致 0**，`max|Δ|` = 0 / 0 / 4.9e-5 / 4.9e-5（零头为 md 只印 4 位小数）。
- **I1（复核者无 exec 的四项）已由 axis5-lead 代跑留证** `.scratch/66-axis5-quant/evidence/REVIEW-1-COMMANDS.md`：
  重跑 `exit=0` 且脚本指纹相符、6 个输入 sha256 逐条相符、`summary.py --check` 全绿、9 篇 arXiv 标题核实。

### 3.2 第 2 轮（fresh 上下文，已派发）

- 派单契约见 `TICKET.md` §7.3：只核 N1/N2 的修复是否真改对、新增的跨文档独立核**是否真是独立路径（可否被伪造）**、
  修复是否引入新缺陷、范围纪律；报告落 `REVIEW-2.md`（agent = `worker`，该 agent 有写/执行工具，故能自行落盘）。
- 派单 `2026-10-13T20:19+08:00`；**截至本报告写定时仍在运行**（run `73b54e0a-4150-4096-9742-d191a71a7ad5`）。
  返回后把判词补写到本报告 §3.2 与 `TICKET.md` §7.1/§7.3。累计复核轮数 = 2（已达上限，不再开第 3 轮）。

## 4. 命令与结果

| 命令 | 结果 |
|---|---|
| `python3 tools/summary.py --check` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md`（改前改后各一次，均绿） |
| `python3 frontier_ledger.py`（CPU） | `exit=0`；G1 `max|Δ| = 0` PASS、G2 PASS、`verdict = PROCEED`；产出 `results.json`/`table.md` |
| `git diff --check` | 无空白/冲突标记问题 |
| `git status --short` | 仅 6 份 `axes/5-quant/*/project.yaml`（M）+ `axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/`（新）+ `.scratch/66-axis5-quant/`（新）⇒ **无越界改动** |
| `curl -sSL --max-time 40 https://arxiv.org/abs/<id>` × 9 | 9 篇标题全部核实（见 TICKET §4） |
| 未运行 | **未跑任何 GPU 作业**（本轮无 GPU 需要 ⇒ 未取 `SCHEMA.md` §20 锁）；未跑全量 `pytest`（本轮为纯文本/文档 + 纯 CPU 再分析改动） |
| 幂等性 | 重跑 `frontier_ledger.py` 后 `git status --short` 空 ⇒ `results.json`/`table.md`/`run_log.txt` 可复现（同一输入 ⇒ 同一输出） |
| 变更指纹 | `.scratch/66-axis5-quant/evidence/MANIFEST.md`（HEAD 绑定的提交列表 + 全部改动文件 + 关键产物 sha256） |
| 复核者代跑留证 | `.scratch/66-axis5-quant/evidence/REVIEW-1-COMMANDS.md`（脚本重跑 `exit=0` 且 `results.json::code` 指纹与磁盘一致、6 个输入 sha256 逐条相符、`summary.py --check` 全绿） |

## 5. 剩余 / 受阻项与下一步

1. **等参数臂缺**（本轮新暴露、最重要）：7/7 同字节组的最优臂都带参数混杂 ⇒ 「同字节纯机制」读数不可得。
   下一步最小实验：训练 1 个**等参数** GQA 臂（把 Δparams 还给 FFN/宽度），等效算力 ≈ trunk 基线 ×1.09 每 token
   训练 FLOPs × n≥2 run；建议 `5.1` 与 `5.4` 共用（结构问题，已作为 F4 上报 Main 裁决）。
2. **未覆盖的字节水平**：只在恰好同字节点上比较 ⇒ 1920/6144 附近缺量化档，须补臂（禁插值）。
3. **fp8 质量未测**（`5.5` WP1 第一优先级）：它是同一字节轴上的空白点，且决定「int8 是否 8 位最优」。
4. **`mqa_int4`/`gqa_int4` 变差的归因**（`5.1` 的 gaps）：参数混杂已量化但未隔离。
5. **`2632/README.md` 内「本机联网不可用 ⇒ KIVI 为 `记忆·未核`」已过时**：该文件是运行前冻结的预登记契约，
   按纪律不改写；当前事实以 `5.1::sota_baseline`（已核实 arXiv 编号）为准。
6. **框架性取舍 4 条已记录并上报 Main**（TICKET §10）：F1 同字节比较用哪条可排序带（取本格在力带 0.036324）；
   F2 头族在原始 Pareto 上的支配不计为发现（参数混杂）；F3 `5.1` status 保持「已完成」；F4 等参数臂归属待裁决。
7. **`5.6` 的 status 仍为「进行中」**：组合前沿与检索腿未跑（GPU），如实登记，不谎报完成。
8. **复核门状态（如实）**：第 1 轮判词 = `OK with notes`（blocker 0、normal 2）且两条 normal 均已修 + 新增一条真检验；
   第 2 轮（验证修复）**在本报告写定时仍在运行** ⇒ 复核门**未完全闭合**，不以「第 1 轮无 blocker」代替第 2 轮判词。
9. **未核项**：第 1 轮复核者无 exec/curl ⇒ 其 I1 四项由 axis5-lead 代跑（已留证）；第 2 轮的独立判定结论待回。
