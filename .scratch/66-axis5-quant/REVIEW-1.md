# 2642 轴 5 Phase A 独立复核报告（REVIEW-1）

> 复核者：`reviewer`（builtin agent，**fresh 独立上下文**、只读、不得为本项 writer），run id `39d87d55-38d0-4481-ab7e-549cd505c79c`，
> 派单 `2026-10-13T20:13:43+08:00`，返回 `2026-10-13T20:18+08:00`。
> **本文件由复核者的返回全文落盘**（复核者所在会话只有 read/grep/find/ls 类工具、**无写入与执行工具**，故判词由它返回、由 axis5-lead 原样落盘）。
> 复核基线：工作树（含 writer 的 4 处自纠后的版本）。本文件末尾「附：复核者无法执行的命令 + axis5-lead 执行结果」一节是
> **复核者要求 supervisor 代跑**的部分，由 axis5-lead 追加，并在 `.scratch/66-axis5-quant/evidence/REVIEW-1-COMMANDS.md` 留证。

---

## 返回给父 agent（3 行）

- **结论：OK with notes**（无 blocker；2 条 normal 必改；判据 5 部分子项本复核者无法执行，需 supervisor 跑）。逐条：判据1 PASS(手算14臂)、2 PASS、3 PASS、4 **FAIL(1处数字错)**、5 PASS/部分未核、6 PASS。
- blocker = 0；normal = 2（`analysis.md:125`「5 个是头族」应为 **4**；`frontier_ledger.py:153-170` 的「必须不变」门禁是**恒真式**，独立判定：属实，属循环论证）。
- 报告路径：本会话**无写工具**（只有 read/grep/find/ls/watchdog_diff），无法落盘；请把下面全文保存为 `/home/z/vibe/pathlm/.scratch/66-axis5-quant/REVIEW-1.md`。

---

**人话结论：结论站得住（字节账、口径、同字节分组、参数混杂都对得上），但有 1 处数字写错（「5 个是头族」应为 4 个）、1 处「必须不变」门禁实际上是个永远为真的空门禁，需要改掉再上报；另有几项必须由 supervisor 亲自跑命令才能盖章。**

复核基线：当前工作树（含 writer 的 4 处自纠后的版本；父/子文件均未 commit，2642 目录整体为新增未跟踪文件）。

## Correct（已核通过，附证据）

- **C1 · 判据1 G1 手算逐位相等（14 臂）**。用 `2·L·n_kv_eff·head_dim·(值字节 + 2/G)·(kept/T)`（L=12、head_dim=32、T=256、G=32 ⇒ 标尺 0.0625 B/值）手算，与 `2632::deployment_caliber.per_arm[*].primary.bytes_per_position`、`2642/results.json::rows[*].deployment_bytes_per_position` 三方一致：
  `int8` 6144×1.0625=6528、`int4` 3072×1.125=3456、`mqa_int4` 768×0.5625=432、`gqa_int8` 1536×1.0625=1632、`gqa_int4` 1536×0.5625=864、`mqa_int8` 768×1.0625=816、`win32_int8` 768×1.0625=816、`win16` 384×2=768、`win32` 768×2=1536、`win32_anchor32` 1536×2=3072、`win32_anchor32_int4` 1536×0.5625=864、`win16_anchor16_int4` 768×0.5625=432、`gqa_win64` 384×2=768、`mha`/`fp16` 6144×2=12288。逐位相等（`rows[*].abs_diff_vs_recorded` 全 0）。
- **C2 · 判据2 口径纪律 PASS**。入表的 10 个量化臂在 2632 的 `primary` 全为 `scheme=block, group=32`；且 `points[*].bpc` 取的是 block-g32 值（`int8` point bpc 1.483672 = 目录里 block g32 值，**≠** per_value 1.483688）⇒ 无 per_value 数字进入前沿。`points[*].bytes_per_position`（仅值口径）只出现在「仅值字节」列并与部署字节分列（`table.md` 表1 两列；`README.md:68` 的 P2 说明 int8 6144 vs 6528 正确）。
- **C3 · 判据4 同字节分组**。我按 `rows` 的部署字节独立重分桶：恰好 **7 组**（432/768/816/864/1536/1632/3072），跨族条件均成立；12288 因 `mha`/`fp16` 同族被正确排除；7/7 最优点为头族且都带 `param_confounded=true`；bpc 极差 0.067753–0.526426 全部越带（含 1536 组 1.858406−1.493763=0.364643）。
- **C4 · 判据3 参数混杂**。解析式手算：GQA (8−2)=6 → 12×514×6×32=**1184256**；MQA (8−1)=7 → **1381632**，与 `results.json::param_confound.measured_delta_vs_mha` 逐位相等；相对 1184256/13536768=8.7484%、1381632/13536768=10.2065% ⇒ 文中「少 8.75% / 10.21%」正确。旁证：`MODEL.md` 记录的 checkpoint 字节数 ×4 B/参数 = 54147072/49410048/48620544，与 13536768/12352512/12155136 参数量自洽（余量为头部开销）——但 torch 我未跑。
- **C5 · Pareto 与支配者**。按脚本 `dominator` 定义我手算复核全部 27 臂：raw 非支配 6 臂 = `mqa_int4, gqa_win64, mqa_int8, mqa, int8, mha`；越带非支配 16 臂与 21 条 raw 支配 / 11 条 band 支配**逐条一致**（例 `int4`←`mqa_int8`、`fp16`←`mha`）。特别核了疑似陈旧项：`win144` 在带上**不**被支配（int8 只比它好 0.013656 < 0.036324）——手算后确认脚本正确，不是旧产物。
- **C6 · 自纠 3 条数字正确**：3072 组 `win64` vs `win32_anchor32` 差 0.026462 < 带（bpc 腿不可排序），copy 召回 CI [0.0836,0.1042] vs [0.0511,0.0678] **不重叠** ⇒「两腿结论不一致」标注正确；`fp8 6144 省 384 B/位置`（−5.88%≈−5.9%）正确；§3 的 −10.21% 与 §0 一致。
- **C7 · 不过度声称**：`README.md` §2、`analysis.md` §7.5、`results.json::answer.cannot_claim` 一致声明「本实验不产生新质量证据、强度上限 = 落盘读数重排」，fp8 标 `quality_measured=false`；未把「落盘读数重排」说成新实测。
- **C8 · 判据6 范围**：tracked 改动**只有** 6 份 `axes/5-quant/*/project.yaml`；新增未跟踪文件只在 `2642-kv-same-byte-frontier/`（及 `.scratch/66-axis5-quant/**`）。未动 `pathlm/**`、`tests/**`、`tools/**`、其它轴。局限：committed 区间不在其视野内，且它无法直接跑 `git status --short`。
- **C9 · 判据5 role_map**：六格均实名 owner=`axis5-lead`、writer=`axis5-w5.X`、reviewer=`axis5-reviewer`（≠ writer）、integrator=`tree-integrator`，无「待指派」残留。六格 `cost_model` 均无墙钟/日期（按 `tools/check_tree.py:WALLCLOCK_RE` 逐条比对不匹配）。

## Finding

### normal（建议先改再上报 Main）

**N1 — 判据4 FAIL：`analysis.md:125` 数字错误。**
「非支配集 = `mqa_int4`, `gqa_win64`, `mqa_int8`, `mqa`, `int8`, `mha`（6 臂）。其中 **5 个是头族**」——实际只有 **4** 个是头族（`mqa_int4`/`gqa_win64`/`mqa_int8`/`mqa`）；`int8` 的 `class="quant"`（族「trunk × 量化」）、`mha` 的 `class="baseline"`（族「stock fp16（trunk / MHA）」）。这是判据4 中**唯一**无法在 `results.json` 找到一致来源的数字。
最小改法：改为「其中 4 个是头族（另 2 个是 trunk 基线与 trunk×量化）」，或直接列出 6 臂的 `class` 逐条。

**N2 — 判据4/5 过度声称：`frontier_ledger.py:153-170`（写入 `gates.values_unchanged_bitwise_ok`，`:283/:286`）= 恒真式门禁。**
我的独立判定：**属实，是循环论证**。`rows` 里的 `bpc` / `recall_copy` / `recall_copy_ci` 就是在 `frontier_ledger.py:126-129` 直接从同一个 `points` 对象逐字拷进来的（`"bpc": p["bpc"]`、`"recall_copy": p["recall_copy"]["accuracy"]`、`"recall_copy_ci": p["recall_copy"]["wilson95"]`），随后 `:165-169` 又拿它们与同一个 `p` 比较 ⇒ 只要脚本能跑通，`bitwise_ok` 永远为 `True`，不可能否证「读数未被改动」。它唯一能意外捕捉的是「某臂在 `rows` 中缺失」，而那只会在 `next(...)` 处抛 `StopIteration`，不是这条门禁的功劳。附带：`parent_pairs`(:157)/`local_pairs`(:160)/`value_fingerprint`(:154-155) 从未被使用，`import math`(:162) 未使用——死代码。
因此 `analysis.md:55` 的「必须不变｜**PASS**（`gates.values_unchanged_bitwise_ok = true`）」和 `analysis.md:154` 的锚点行把恒真式当独立证据。
正确写法（择一）：
 (a) **如实降级**：`2642` 对读数是逐值透传（无任何数值变换），「必须不变」的证据在**上游** —— 引 `2632::reuse.nonquant_bitwise_preserved = true`（`2632` 相对 `2631` 做的独立检验）；把 `gates.values_unchanged_bitwise_ok` 改名 `readings_passed_through_untouched` 并在 §1 标注「本条为透传事实陈述，不构成独立检验」，`verdict` 只由 G1/G2 决定。
 (b) **做真检验**：逐臂与 `points[*].source` 指向的原始实验 results.json（2614/2615/2616/2619/2620）比对 bpc/召回，或至少对父 JSON 走一条**独立解析路径**重算后再比——比较对象不能是同一内存对象。
影响：不改变 G1/G2 结论（判据1/2 仍 PASS），但按判据5「不得把重排说成新证据」的精神，现状是「无证据却盖了 PASS 章」。

### info（报告级，不必阻塞）

**I1 — 判据1/5 的部分子项我无法执行（本会话只有 read/grep/find/ls/watchdog_diff，没有 exec/curl）**：① 未重跑 `frontier_ledger.py`（未能核对 `results.json::code` 的脚本 sha256，也未验证 `results.json`/`table.md`/`run_log.txt` 就是当前脚本输出——但 `run_log.txt:9` 含**只有**「自纠4」之后才存在的日志行，与「已重跑」自述一致）；② 未算 6 个输入文件的 sha256（`gates` 的指纹 PASS 未独立复核）；③ 未跑 `python3 tools/summary.py --check`；④ 未 curl arXiv。
请 supervisor 执行并留证：
```
python3 axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/frontier_ledger.py ［此处会话输出被截断，命令清单见下节补全］
```

---

## 附：复核者无法执行的命令 + axis5-lead 执行结果

> 本节**不是复核者原文**；是复核者在 I1 中列出的四项，由 axis5-lead 代跑。原始输出留证于
> `.scratch/66-axis5-quant/evidence/REVIEW-1-COMMANDS.md`。

| # | 命令 | 结果 |
|---|---|---|
| ① | `python3 axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/frontier_ledger.py` | `exit=0`；`results.json::code` 的脚本 sha256 与磁盘文件一致；G1 `max|Δ| = 0`、G2 PASS、`verdict = PROCEED`（基线仅 G1/G2） |
| ② | 6 个输入文件的 `sha256sum` | 与 `config.json::inputs_sha256` 及 `2632/MODEL.md` 记录逐条一致 |
| ③ | `python3 tools/summary.py --check` | `OK：46 个 project.yaml 全部符合 docs/research-schema.md` |
| ④ | `curl -sSL --max-time 40 https://arxiv.org/abs/<id>` × 9 | 9 篇标题全部核实（见 `TICKET.md` §4） |
