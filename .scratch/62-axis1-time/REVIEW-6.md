## §0 人话摘要（先读这段）

**判词：PASS（带 notes），blocking = 0。** 范围 = `1.1.4.1` 的 2639 全套产物 + `project.yaml` + 本轮返工（commit `6f42ed5`，工作树干净）。

1. **B1 真修了**：`project.yaml:47–48` 现在确有 `2639/analysis.md @ccdc91a` 与 `2639/results.json @ccdc91a` 两条；`experiments:` 共 29 条，我逐条核对**所有**路径（含 2638/2634 等带 `@hash` 的文件锚点）在磁盘上**全部存在**；`change_log` 末条（`:130–132`）写的「experiments 新增 2639」现在与文件实况**一致**（此前的「声称 > 实际」已消除）。
2. **P2-1/P2-2 真修了**：`analysis.md:180` 现在写 `pl16d 30.7 min`（= `train/pl16d/results.json:871`，也 = `results.json:1456`）、`analysis.md:182` 写评测 `0.49 min`（= `results.json:2005`）；`results.json` 内**无任何** `NaN`/`Infinity`/`Inf` 字面量（grep 0 命中），`results.json:1429` 的 `rel_dev_vs_other_rule` 现为 `null`，`run.py:534–536` 有 `_strict()` + `allow_nan=False` 的落盘边界。
3. **返工零回归**：我把现盘 `rows`/`frontier`/`location`/`key_readings`/`reuse_equivalence` 与 REVIEW-5 在 `c77cc0b` 时读到的值逐个对拍，**逐位相同**（`0.068228`/`0.071912`/`1.527595`/`1.595823`/`10/10 equal=true`/`carrypl16_s0` 仍在 `lower_envelope`）。
4. **主结论未被支配（我独立复算，不复述 prose）**：13 臂中**所有** `total_bytes ≤ 393,216` 的臂，bpc 最低的非完备臂是 `win32_s0 1.595823`，`carrypl16_s0 = 1.527595` < 其余全部 ⇒ **不存在更省字节却更好的臂**。
5. 我**没有 shell**（可用工具只有读/grep/find/ls/watchdog_diff），执行类检查（sha256sum/pytest/json 严格解析/git 区间 diff/GPU 重跑）**一条未跑**（命令原文见 §4）；出题方点名的 §9.12 代跑表我只**读**了其自洽性，**未据其放行**。
6. 我**没有写文件工具** ⇒ 报告以下文全文交付，请父级原样落盘到 `.scratch/62-axis1-time/REVIEW-6.md`。

---

## §1 B1 是否真修（含兄弟字段一致性）

| 检查 | 位置 | 结论 |
|---|---|---|
| `experiments:` 补 2639 两锚点 | `project.yaml:47` = `…/experiments/2639-carry-perlayer-frontier-13m/analysis.md @ccdc91a`；`:48` = `…/results.json @ccdc91a` | ✅ **真修**（REVIEW-5 的 B1 直接消解） |
| 列表条数与位次 | `experiments:` 起于 `:19`，条目 `:20–:48` = **29 条**（与 TICKET §9.11 声称「现 29 条」一致） | ✅ |
| **全部**路径存在性（不只抽查） | 用 `find` 逐一核：25 个目录全部存在；11 个带 `@hash` 的文件锚点全部存在（2638 的 `analysis.md`/`results.json`、2634 的 `analysis.md`/`results.json`、2616/2617×2/2618 的 md、2635 的 `stall_report_linear_v2.json`、2639 的 md/json） | ✅ 无悬空路径 |
| `change_log` 与实况一致（不得再有「声称 > 实际」） | `:130–132` 末条（`2026-10-14T09:00:00`，actor `axis1-lead-2`）声称：「experiments 新增 2639」→ 现在为真；「status 由被否证改为进行中」→ `:108 status: 进行中` ✅；「failure_analysis 置空」→ `:16 failure_analysis: ""` ✅；「WP1 补 2639 evidence」→ `:83` ✅；「WP2 改为进行中」→ `:93` ✅；「ended 清空」→ `:109 ended: ""` ✅ | ✅ 逐项落实 |
| 数值引述一致 | `:17 conclusion`（1.527595 / 1.595823 / +0.068228 / 41×σ / 4.9×0.014 / 0.125421 / 80.1% / ≈窗口 111）与 `results.json:1574,1638,1819–1824,1928` 全部一致 | ✅ |

⇒ **B1 判「已真修」**。

---

## §2 P2-1 / P2-2 / P2-3 处置核对

**P2-1（成本记账）——已修**：
- `analysis.md:180`：`pl16 28.56 min` = `train/pl16/results.json:871 = 28.56` ✅；`c16 10.77` = `train/c16/results.json:870 = 10.77` ✅；`pl16d 30.7`（原 30.0）= `train/pl16d/results.json:871 = 30.7` ✅（也 = `results.json:1456`）。
- `analysis.md:182`：评测 `0.49 min` = `results.json:2005 wall_minutes_total_eval = 0.49` ✅，峰值 `1019 MB` = `:2006 = 1019.1` ✅。
- 合计 `≈75.5 min`：28.56+10.77+30.7 = 70.03，+ 冒烟 ≈5 = 75.03，+ 评测 0.49 = 75.52 ✅（< 85 预算）。
- 全文再无残留 `30.0 min`/`0.5 min`（grep 0 命中）✅。

**P2-2（严格 JSON）——已修**：
- `results.json` 全文 grep `NaN|Infinity|-?Inf` = **0 命中**；`results.json:1429` = `"rel_dev_vs_other_rule": null` ✅（同键其余 12 臂为实数：`:1005 :1099 :1215 :1312` 等）。
- 修法在**本实验自己的落盘边界**：`run.py:114–129 _strict()`、`:534` 调用、`:536 allow_nan=False`，且 `run.py:525–533` 的 `code_hashes` 是**运行时从磁盘算**的（不是我读到的常量）⇒ 声称的「重跑评测后现为严格 JSON」在实现上自洽。
- 残留（见 §5 P2-D）：源头 `train/pl16d/results.json:679` **仍是裸 `NaN`**（共享 runner 在 γ≠1 时的遥测），`analysis.md:186–187` 已如实记账「共享代码未改」⇒ 我按 **report-only** 处理，不构成未披露缺陷。

**P2-3（「同算力」措辞）——按声明「不改」执行，但**处置声明本身有一处对不上**（见 §5 P2-B）：
- `project.yaml:75 milestone.acceptance` 仍为「不被**同算力**最强基线支配（相对**同算力**纯窗口前沿超出量 ≤ 0）」= C2 裁决落地文字，**未动**（符合「不自行改裁决用词」）✅。
- 但 `:14 objective_criteria` 现读作「主判据（C2 实质验收）：在**同一每位置等效字节**上不被同预算纯窗口支配」——REVIEW-5 当时指认「`:14 objective_criteria` 含『同算力』」。二者无法在无 git 条件下互证（见 §5 P2-C）。

---

## §3 核心主张独立复算（自己算，不再复述 prose）

来源全部为 `…/2639-carry-perlayer-frontier-13m/results.json`：

| 量 | 我的算法与结果 | 落盘键 | 文档值 | 判 |
|---|---|---|---|---|
| `win32_s0 − carrypl16_s0` | 1.595823 − 1.527595 = **0.068228** | 行值 `:1574` / `:1638`；`key_readings.pl_minus_win32.pl_minus_win32` `:1821` | 0.068228 | ✅ |
| 同预算（逐字节相等） | win32 `total_bytes = 393216`（`:1575` 邻域）；carrypl16 `kv 196608 + state 196608 = 393216`（`:1639–1641`）；`key_readings.pl_minus_win32.same_bytes = true` `:1819` | — | 393,216 B | ✅ |
| `state_floats_actual`（逐层 = 12×8192） | `checkpoints.carrypl16_s0.state_floats_actual = 98304`（`:410`）= 12×8192；`carrypl16d_s0 = 98304`（`:444`）；单共享槽臂 `8192`；`run.py:91–93` 从**加载后的模型**算 `n_heads·head_dim²·(n_layers if per_layer)`（非手填常量）；`run_log.jsonl:13` 打印 `per_layer=True … state_floats=98304 params=13801064` | `:410/:444` | 98,304 floats → 196,608 B | ✅ |
| `r = Δc/Δg` | Δc = 1.628110 − 1.527595 = **0.100515**（`location.carrypl16_s0.delta_bpc_vs_win16_s0` `:1817`）；Δg = 1.628110 − 1.502689 = **0.125421**（`gap_closing.gap_win16_full` `:1928`）；r = 0.100515/0.125421 = **0.801429** | `:1818 recovery_ratio_r = 0.8014` | 0.8014 / 80.1% | ✅ |
| 41×σ_cell、4.9×0.014 | 0.068228/0.00166 = **41.10**；0.068228/0.014 = **4.87** | `:1824 sigma_cell`、`bands.cross_cell_band` | 41× / 4.9× | ✅ |
| 「≈窗口 111」 | 0.068228/(1.595823−1.513325)=0.827 → 32+0.827×96 = **111.4** | `:1574`/`win128 :1590` | ≈111 | ✅ |
| **是否有更省字节却更好（bpc < 1.527595）的臂** | 枚举全部 13 臂的 `total_bytes`：full 3,145,728（1.502689，**字节更多 8×**，不算同/更省）；win16_s0/s1 196,608（1.628110/1.629093）；win17 208,896（1.624104）；win18 221,184（1.619953）；win24 294,912（1.599856）；win32 393,216（1.595823）；carry16_s0/s1 212,992（1.624979/1.626640）；c16c256 212,992（1.628940）；carrypl16d 393,216（1.585531）。⇒ **≤393,216 B 的 11 个臂里 bpc 最低者 = win32_s0 1.595823 > 1.527595** | `rows` `:1471–1668`；`frontier.lower_envelope` `:1726–1735` 含 `carrypl16_s0`；`frontier.dominated_by.win32_s0 = [carrypl16_s0, carrypl16d_s0]` `:1715–1717` | — | ✅ **无支配臂** |
| 复用等值性 | `reuse_equivalence` 恰 10 条（full、win16_s0/s1、win17/18/24/32/128_s0、carry16_s0/s1），`equal` **10/10 true**（`:1941–2004`）；另有独立旁证：`train/pl16/results.json` 的 `avg_train_depth_ce[0]=1.2389`（`:874`）与 `project.yaml:90` 的 1.2389 一致 | `:1941–2004` | 10/10 | ✅ |

**返工零回归的独立证据（关键）**：我把上表所有值与 **REVIEW-5 在 `c77cc0b` 时读到的值**逐位对拍（REVIEW-5 是 fresh 独立件，写在返工之前）——`win32 1.595823`、`carrypl16 1.527595`、`win16_s1 1.629093`、`carry16_s0 1.624979`、`carry16_s1 1.626640`、`win18 1.619953`、`win24 1.599856`、`c16c256 1.628940`、`win128 1.513325`、`full 1.502689`、`alt_seed 0.071912`、`r 0.8014` — **全部相同**，且 `frontier`/`location` 结构未变 ⇒ 返工（含评测重跑）**未引入数值回归**。

**数字引用同步抽查（≥6，全部落到落盘键）**：
1. `analysis.md:35`（§0 表 1.5276 / 80.1%）↔ `rows :1638` / `location :1818` ✅
2. `analysis.md:41–43`（0.0682、+0.0719 seed7）↔ `:1821` / `:1922` ✅
3. `analysis.md:45`（0.0682/0.0825 → ≈111）↔ `:1574,1590,1638` ✅
4. `analysis.md:67–68`（③ +0.068228、④ Δbpc 0.100515 / r 0.8014）↔ `:1821` / `:1817–1818` ✅
5. `analysis.md:84–86`（同字节直接点比 + 换序 +0.071912）↔ `:1819` / `:1922` ✅
6. `analysis.md:92–96`（41×σ / 4.9×带 / chunk 0.003961）↔ `:1824` / `chunk_sensitivity_shared` / `rows` ✅
7. `analysis.md:99–104`（§4 活性 read_share 0.386 / beta 0.157 / state_fro 24.9 / rel_dev 0.258 / params_added 264296 / 12 个逐层占比 0.011–0.647）↔ `liveness.carrypl16_s0` `:1205–1237` ✅
8. `analysis.md:107–108`（13,801,064 与 13,801,064）↔ `checkpoints` `:407`/`:373`、`rows.n_params` ✅
9. `project.yaml:11,17,75,80,90,104–105`（0.068228 / 80.1% / 1.527595 / 1.595823）↔ 同上 ✅

---

## §4 未复核范围（如实，不许当已验）

**能力限制**：本 run 无 shell/执行工具（只有 读/grep/find/ls/`watchdog_diff`；`watchdog_diff` **不含已提交区间**）。以下**未跑**：

1. 磁盘 sha256：`run.py`/`config.json`/`train/{pl16,c16,pl16d}/model.pt` 与 `MODEL.md:5–19,57–58`、`results.json.checkpoints`/`code_hashes` 的**字符串**我核到完全一致（含 `run.py f04d59da…`、`config.json 28ba9485…`），但**未重算磁盘哈希**。
2. `results.json` 的**严格解析**（我只做了「无 `NaN`/`Infinity` 字面量」的文本判断 + 读 `run.py` 的 `allow_nan=False`）。
3. 任何 `pytest` / `summary.py --check` / `check_tree.py` / 评测重跑 / 因果探针。
4. 已提交区间的文件清单（**越界/卫生的唯一硬证据**）：`git diff d3d33eb..HEAD --stat`、`git diff c77cc0b..HEAD`、`git show c77cc0b:<file>`。
5. 训练过程本身（12k 步 / token 预算 / 调度）与 `train_log.jsonl` 内部一致性。

**我确实独立核到的范围**（可作为本报告的正面结论）：`project.yaml` 全文 + `2639/{README.md,analysis.md,MODEL.md,results.json,config.json,run.py,train_pl.py}` + `train/{pl16,c16,pl16d}/results.json` 的相关键 + `2638/results.json.code_hashes` + `run_log.jsonl` 前 17 行 + `pathlm/depth_ar.py` 的 E1 字段存在性 + `.scratch/62-axis1-time/TICKET.md` §9.8–9.12 + `.scratch/57-program/FRAMEWORK-DECISIONS.md` C2 + `.git/COMMIT_EDITMSG` + 工作树 diff。

**请父级代跑（原文命令）**：
```bash
cd /home/z/vibe/pathlm
# (a) 严格 JSON（点名的 P2-2 独立证据）
python3 - <<'EOF'
import json
p="axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/results.json"
d=json.load(open(p)); print("strict-parse OK", d["key_readings"]["pl_minus_win32"]["pl_minus_win32"],
  d["liveness"]["carrypl16d_s0"]["carry_liveness_training"]["rel_dev_vs_other_rule"])
EOF
jq -e . "$PWD/axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/results.json" >/dev/null && echo jq-OK
# (b) 指纹（P2-1/P2-2 与本报告 §2、§4.1 的未验部分）
sha256sum axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/run.py \
          axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/config.json \
          axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/train/{pl16,c16,pl16d}/model.pt
# (c) 已提交区间：返工是否只动 experiments / 越界是否为 0
git diff --stat d3d33eb..HEAD
git diff c77cc0b..HEAD -- axes/1-time/1.1.4.1-explicit-memory/project.yaml      # 解 §5 P2-C
git show c77cc0b:axes/1-time/1.1.4.1-explicit-memory/project.yaml | grep -n '同算力\|09:1\|09:0'
git show c77cc0b:axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/results.json | jq '.cost'
# (d) 门禁
python3 -m pytest axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/test_2639_frontier.py -q
python3 tools/summary.py --check && python3 tools/check_tree.py
```

---

## §5 判词

**Verdict: PASS（带 notes）——blocking = 0。**
理由：B1 已真修且兄弟字段逐条核对通过；P2-1/P2-2 已真修；返工零数值回归（与返工前的独立复核件逐位对拍）；主判据 `+0.068228` 我独立复算复现，且**不存在**「同预算或更省字节却 bpc 更低」的臂；口径与证据链未见软化或夸大。以下 4 条为非阻断。

**Blocking**：无（0 条）。

**Non-blocking（P2，report-only；每条给位置 + 违反判据 + 复现）**

| # | 位置 | 问题 | 违反判据 | 复现 / 最小修 |
|---|---|---|---|---|
| **P2-A** | `2639/analysis.md:184` | 成本表写「峰值训练显存 **9.44 GB（`pl16`/`pl16d`）**」，但 `pl16d` 实测峰值是 **10.28 GB** | `AGENTS.md` §2.1「事实文档只陈述当前事实、每条事实恰好一次且与证据一致」；`SCHEMA` 成本记账不得与落盘冲突 | 证据：`results.json:1457 carrypl16d_s0.train_peak_gpu_mem_mb = 10281.2`、`train/pl16d/results.json:872 = 10281.2`，而 `carrypl16_s0` 才是 9443.6（`:1243`）。最小修：改成「9.44 GB（`pl16`）/ 10.28 GB（`pl16d`）」（不影响任何科学结论、不影响 85 min 预算） |
| **P2-B** | `.scratch/62-axis1-time/TICKET.md` §9.11「P2-3」行（`:422`） | 该行声称「登记在 **§9.8** 的框架项中待 Main 裁」，但 `TICKET.md` §9.8（`:377–385`）的框架项只有 **F6/F7/F8**，**没有**「同算力」措辞这一行 ⇒ 又一处「声称 > 实际」（这是本轴被反复抓的同一形态，只是落在 ticket 而非事实文档） | `AGENTS.md` §8「提交信息/记录只描述已验证落地的内容」 | 复现：`grep -n '同算力' .scratch/62-axis1-time/TICKET.md` **只命中 `:413`**（REVIEW-5 摘要），§9.8 表格内 0 命中。最小修：§9.8 补一行 `F9`（或把 §9.11 的落点改为「见本条」） |
| **P2-C** | `project.yaml:130`（change_log 末条日期）与 `:14`（`objective_criteria`） | 两处**可能**在返工中被静默改写：① REVIEW-5 与 TICKET §9.10（`:405`）**都**引该条 `change_log` 日期为 `2026-10-14T09:10:00`，现盘却是 **`09:00:00`**；② REVIEW-5 指认 `:14 objective_criteria` 含「同算力」，现盘 `:14` 读作「**同一每位置等效字节**」。而 §9.11 与提交信息都只声称改了 `experiments:` 两行 | `AGENTS.md` §8（不得 amend/掩盖、更正要显式）；`docs/research-schema.md` §8（`change_log.date` 属溯源元数据） | **我无法证实**（无 git、无 mtime；也可能是 REVIEW-5 引用不精确——它另有 2 处行号与现盘不符，如把 `milestone.acceptance` 记成 `:16` 而现盘在 `:75`）。最小修：先跑 §4(c) 的两条 `git show c77cc0b:` 命令；若确有改写 ⇒ 按 `AGENTS.md` §8 补一个显式更正说明（`experiments` 之外的改动也要写明）；若确为引用误差 ⇒ 无需动作 |
| **P2-D** | `2639/train/pl16d/results.json:679` | 源头证据文件仍是裸 `NaN`（`rel_dev_vs_other_rule`），严格 JSON 工具（`jq`/`JSON.parse`）解析失败；仅本实验顶层落盘边界换成了 `null` | SSOT 原始数字应可被常规工具解析（P2-2 的原始判据） | 复现：`grep -n NaN 2639/train/pl16d/results.json` → `:679: "rel_dev_vs_other_rule": NaN`。**已在 `analysis.md:186–187` 如实记账**（共享 runner 产物、本轴不改共享代码）⇒ 我不升级；若要收口，最小修 = 引擎/共享 runner 侧对 `decay≠1` 返回 `None`（跨轴，需授权） |

**关于「已代跑」记录的读法（不作为我的证据）**：`TICKET.md` §9.12 的 7 条代跑我在**文本层面**判断自洽（`sha256sum` 4/4、`pytest 6 passed`、`summary/check_tree OK`、`staged=0`、越界 0、评测重跑逐位复现），其中「越界文件数 = 0」与「评测重跑逐位复现」恰好是我无 shell 时**最需要**的两条；但我**未**据此放行——它们是同一 writer 的自报，且 §9.12 第 6 条过滤口径（只允许 2639 目录 / `project.yaml` / `.scratch/62-axis1-time/**` / 两份 E1 报告）本身未经独立复核。

**报告落盘状态**：本 run **无写文件工具** ⇒ 上述全文即交付物，请父级**原样**写入 `.scratch/62-axis1-time/REVIEW-6.md`（我未创建、未改动任何文件；`watchdog_diff` = `No working-tree changes against HEAD 6f42ed54a965`）。