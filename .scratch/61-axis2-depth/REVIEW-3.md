# REVIEW-3 · 轴 2 Phase A round-3 独立复核（revision `3ef14e6`，基线 `483b181`）

> 复核者：**round-3 fresh 独立复核者**（独立上下文 subagent，未参与本线任何写作；**只读**，未改仓库文件）。
> 被复核 revision：`3ef14e6`（`ticket 61 轴 2 Phase A 收口：修 REVIEW-2 B1–B4（文本级账实不符）+ 入库 round-2 复核证据`）。
> 前置：round-2 判 **BLOCK**（`.scratch/61-axis2-depth/REVIEW-2.md`，B1–B4）；round-1 = `REVIEW-1.md`。
> 工作区核对：HEAD = `3ef14e6735a0eb07916621e58c942df941364183`；进入时 `git status --porcelain` 为空；`git diff --cached --name-only | wc -l` = 0；复核结束仍未提交任何 git。
> 本报告**不改任何被复核文件**；唯一写入 = 本文件（落在 run 指定的 artifact 目录）。
> 说明：任务前置提到的 `.scratch/61-axis2-depth/REPAIR-round2.md` **在仓内不存在**；收口处置实际记录在 `TICKET.md §7.4`（见 §④ 观察 O4）。

---

## ① 人话摘要与总判词

这段交付的主线是「让每个字符自己决定跑几层」（逐 token 早退）。round-2 复核确认它的**科学结论与运行证据全过**（19 条目标测试、全量 834 通过、4 处变异被抓红、重跑 `results.json` 逐位复现、`summary --check` OK），只判了 **4 条文字层面的「账实不符」**（B1 单测数写成 16 而实际 19；B2 臂数写成 11 而实际 14；B3「省 ≥25% 档」的区间把 τ=0.95 那档混进来；B4 生成一致率上限写成 55% 而实际 71%）。`3ef14e6` 就是**只改文字/数字、不动任何实现与测量数字**的收口提交。

**我这轮把任务点名的 7 项逐条对仓库现状与证据核了一遍**：

- **B1（单测计数）✅、B2（臂数）✅、B4（28–71%）✅：三处都真落盘了，数字与证据一致。**
- **B3（F2 区间）⚠️ 只落了 2/4 处**：`analysis.md §7②` 与 `project.yaml argument③` 已是正确的「≥25% 档 0.0117–0.2593（0.84–18.5×）；τ=0.95 单列」；但 `analysis.md §8 末` 与 `project.yaml failure_analysis④` 写的是「**超出量是 8.3–18.5 倍噪声带**」——**下界 8.3 是 τ=0.6 那档（0.1168/0.014）的倍数，不是 ≥25% 档的下界**（≥25% 档的下界是 τ=0.9 的 0.0117/0.014 ≈ **0.84**）。于是**同一份 `project.yaml` 内部自相矛盾**（`conclusion` 说 0.84–18.5，`failure_analysis④` 说 8.3–18.5），`analysis.md` 内部也不一致（§0 的 0.8–18、§7② 的 0.84–18.5 vs §8 末的 8.3–18.5）。**这是同一类「处置与证据不符」的残留，故仍判 BLOCK**（纯文本、不影响任何科学结论；如上界/下界以「省 ≥25% 档」为准，两处应把 8.3 改成 0.84，或给 8.3–18.5 加限定词「其中 τ=0.3/0.6 超出 8.3–18.5×」）。
- **同批 non-blocking ✅ 全落**（1e-05→4.6e-05、argument② 浮点噪声、§0 第5点「主要」、conclusion_review 措辞、`_missing_mask` 登记为待测）。
- **无回归 ✅**：`summary.py --check` OK（46 份）；`3ef14e6` 只改 7 个文件、**`pathlm/**` 零改动**、实验 `results.json` 逐字节未改；`REVIEW-2.md` 与 `repro-review2/` 已入库。

**一句话**：科学内容与运行证据这轮依然全过；`3ef14e6` 是一份干净的「文本收口」，但 **B3 在 `analysis.md §8 末` 与 `project.yaml failure_analysis④` 只修成了「8.3–18.5 倍」这个子区间，未与同文档的「0.84–18.5」统一 ⇒ 仍差 2 个数字**。另有 3 处 `TICKET.md §4.5` 的遗留旧数字（非该 7 项点名范围，登记为观察，见 §④）。

---

## ② 逐条核对（任务 1–7）

> 全部命令工作目录 `cd /home/z/vibe/pathlm`；未取 GPU 锁。

### 1) B1 —— 单测计数：**✅**

- `grep -c "^def test_" tests/test_axis2_exit_decision.py` ⇒ **`19`**；`python3 -m pytest tests/test_axis2_exit_decision.py -q` ⇒ **`19 passed in 3.43s`**。
- 全仓「把该实验单测数写成 16」的**当前事实**已清空：`grep -rn "16 条" axes/2-depth/ .scratch/61-axis2-depth/TICKET.md` ⇒ **无命中**（唯一残留是 `TICKET.md:229` 的「新增 3 条单测（**16→19**，含对拍与活性断言）」，这是变更记录而非当前事实，合规）。
- `.scratch/61-axis2-depth/REVIEW-1.md`、`REVIEW-2.md` 里仍有「16 条」——那是**历史复核记录**（描述各自 revision 的当时状态），不是当前事实，合规。
- 被改到的 6 处现均写 19：`project.yaml:17`（`conclusion`「+ 19 条单测」）、`:18`（`conclusion_review`「19 条单测」）、`:115`（`change_log`「19 条，全绿」）、`analysis.md:20`（§0 第2点）、`analysis.md:234`（§10）、`README.md:20`、`TICKET.md:152`（§4.5「19 条 … 19 passed」）。
- 复现：`grep -c "^def test_" tests/test_axis2_exit_decision.py`；`grep -rn "16 条" axes/2-depth/`。

### 2) B2 —— 臂数 = 14：**✅**

- `.scratch/61-axis2-depth/TICKET.md:162`（§4.5）⇒ 「**14 条臂** + 4 类归因诊断」。
- `README.md:49`（§2）⇒ 「共 **14 条臂**（1 基线 + 6 固定前缀 + 6 阈值扫描 + 1 地板臂）」。
- `results.json::arms` ⇒ **14**（`never-exit`、`prescribed-2..7`、`token-thr-0.3/0.6/0.9/0.95/0.99/0.999`、`token-thr-0.999-min4`）。
- 复现：`grep -n "条臂" .scratch/61-axis2-depth/TICKET.md`；`python3 -c "import json;print(len(json.load(open('axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/results.json'))['arms']))"`。

### 3) B3 —— F2 区间四处：**❌ 部分（2/4 处已落，2/4 处未落）**

| 位置 | 现状 | 判定 |
|---|---|---|
| `analysis.md:194`（§7②） | 「省 ≥25% 的档位超出固定前缀前沿 **0.0117–0.2593** bpc（**0.84–18.5** 倍噪声带）；τ=0.95 属省 20% 档、超出 0.0035」 | ✅ |
| `project.yaml:31`（`argument③`） | 「省 ≥25% 算力时超出固定前缀 **0.0117–0.2593** bpc（**0.84–18.5** 倍噪声带；τ=0.95 属省 20% 档、超出 0.0035）」 | ✅ |
| `analysis.md:215`（§8 末行） | 「统计假象（配对 SE ≤ 0.008，**超出量是 8.3–18.5 倍噪声带**）」 | ❌ |
| `project.yaml:16`（`failure_analysis④`） | 「…逐窗口配对 SE ≤ 0.008，**超出量是 8.3–18.5 倍噪声带**。」 | ❌ |

- **判据（任务给出）**：四处应为「≥25% 档 **0.0117–0.2593（0.84–18.5×）**、τ=0.95 单列」。
- **证据（`results.json::frontier_comparison`，倍数 = `frontier_excess / 0.014`）**：
  - `token-thr-0.3`：lp=2.332（省 70.8%）、excess=**0.259346** ⇒ **18.5×**
  - `token-thr-0.6`：lp=4.268（省 46.6%）、excess=**0.116819** ⇒ **8.3×**
  - `token-thr-0.9`：lp=5.967（省 25.4%）、excess=**0.011661** ⇒ **0.84×**
  - `token-thr-0.95`：lp=6.398（省 20.0%，**不属 ≥25% 档**）、excess=0.003493 ⇒ 0.25×
  - ⇒ 「省 ≥25%」= {τ=0.3, 0.6, 0.9}，倍数区间 = **0.84–18.5**；`8.3–18.5` 只覆盖 τ=0.6/0.3，**漏掉 τ=0.9 的 0.84×**。
- **后果（内部不自洽）**：`project.yaml` 里 `conclusion`（第 17 行）写「0.0117–0.2593 bpc（0.84–18.5 倍噪声带）」，`failure_analysis④`（第 16 行）写「8.3–18.5 倍」——**同一 SSOT 文件对同一「门控超出量」给出两个下界**；`analysis.md` 里 §0（第 38 行「0.8–18 倍」）、§7②（0.84–18.5）与 §8 末（8.3–18.5）同样不一致。
- **复现**：
  ```
  grep -n "0.0117–0.2593\|0.84–18.5\|8.3–18.5\|0.8–18" \
    axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/analysis.md \
    axes/2-depth/2.1.1-early-exit/project.yaml
  python3 - <<'PY'
  import json
  d=json.load(open('axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/results.json'))
  for r in d['frontier_comparison']:
      print(r['arm'], 'lp=%.3f'%r['layer_passes_per_token'], 'skip=%.3f'%(1-r['layer_passes_per_token']/8),
            'excess=%.6f'%r['frontier_excess'], 'x=%.2f'%(r['frontier_excess']/0.014))
  PY
  ```
- **备注（公平说明）**：这两处的写法是 **REVIEW-2 第⑤节 B3「最小修法」自己指定的措辞**（「`failure_analysis`④ 的「8–18 倍」改为「超出量是 8.3–18.5 倍」」），而该节同时要求四处区间改为「0.0117–0.2593（0.84–18.5×）」——**REVIEW-2 自身自相矛盾**；收口 writer 按「具体那句」落成了 8.3–18.5。所以这不是 writer 无据乱写，但**按本轮任务的验收值（0.84）与 SSOT 自洽性，仍未达标**。
- **最小修法（只报告，未修）**：两处把 `8.3–18.5` 改 `0.84–18.5`（或加限定词「其中 τ=0.3/0.6 超出 8.3–18.5×」）。

### 4) B4 —— 生成一致率 28–71%：**✅**

- `analysis.md:208`（§8 U4）⇒ 「发出流一致率 **28–71%**」；`analysis.md:54`（§0 第6点）⇒ 「28–71%」；`project.yaml:36`（`gaps`）⇒ 「逐 token 一致率 **28–71%**」。
- 证据：`results.json::generation_agreement` 最低 `token-thr-0.3` = **0.2812**，最高 `token-thr-0.999-min4` = **0.7109**（91/128）⇒ 区间 28–71% 正确；全仓 `grep "28–55"` 在 `axes/2-depth/` 无命中。
- 复现：`grep -rn "28–71\|28–55" axes/2-depth/`。

### 5) non-blocking 同批项：**✅ 全落**

| 项 | 期望 | 现状 | 判定 |
|---|---|---|---|
| 1e-05 → 4.6e-05 | §3 末按实测最大 4.6e-05 表述、删重复标点 | `analysis.md:114`「…该列**不**用于对齐。故 `stream_bpc` 是既有口径的**严格扩展**（数值上重合到最大 **4.6e-05**）」，「。。」已删 | ✅ |
| argument② 浮点噪声 | 与 `failure_analysis④`/`TICKET §4.5` 统一 | `project.yaml:31 argument②`「2..8 **在浮点噪声内重合，Δbpc 7e-06…4.6e-05，非 bitwise**」；`failure_analysis④`「（Δbpc 7e-06…4.6e-05，浮点噪声内重合）」；`TICKET.md:165` 同 | ✅ |
| §0 第5点措辞 | 与 §6「集中在」一致（不写「全部」） | `analysis.md:42`「门控的损伤**主要**来自被早退字符自己的读出」（§6 为「损伤**集中在**」） | ✅ |
| `conclusion_review` 措辞 | 不再写「处置（全部已落盘）」 | `project.yaml:18`「**处置（round-2 独立复核判 BLOCK；属 blocking 的 B1–B4 均为文本级账实不符，本收口轮已逐条修，见 `.scratch/61-axis2-depth/REVIEW-2.md`）**」 | ✅ |
| `_missing_mask` 登记为待测 | 不擅改实现 | `analysis.md:210`（§8 **U7**）与 `TICKET.md §7.4`（N5）均登记为待测项，未改实现 | ✅ |

（任务文字把「§0 第5点」的目标写成「集中在」；REVIEW-2 N3 的最小修法原文是「§0 第5点改『损伤**主要**』」，实际落的就是「主要」，与 §6「集中在」语义一致，视为已落。）

### 6) 无回归：**✅**

- `python3 tools/summary.py --check` ⇒ **`OK：46 个 project.yaml 全部符合 docs/research-schema.md。`**（EXIT=0）。
- `git show --stat 3ef14e6` ⇒ 恰好 **7 个文件**，与提交信息逐条一致：
  `A .scratch/61-axis2-depth/REVIEW-2.md`、`M .scratch/61-axis2-depth/TICKET.md`、
  `A .scratch/61-axis2-depth/repro-review2/results.json`、`A .scratch/61-axis2-depth/repro-review2/run_log.txt`、
  `M …/2631-pertoken-exit-seqlevel/README.md`、`M …/2631-pertoken-exit-seqlevel/analysis.md`、`M axes/2-depth/2.1.1-early-exit/project.yaml`。
- **未动实现代码**：`git show --name-only 3ef14e6 | grep -c "^pathlm/"` ⇒ **`0`**。
- **测量数字未改**：`git diff --stat 483b181 3ef14e6 -- …/2631-pertoken-exit-seqlevel/results.json` ⇒ **空**；且重跑证据与仓内实验 `results.json` **sha256 逐位相同** ⇒ `6a8e2184fd7ffff02c0db212fae75209620c6b24f3a40ec4d8d4b2db2c2ed2e1`（两处）。
- **复核证据已入库**：`git ls-files .scratch/61-axis2-depth/` ⇒ `REVIEW-1.md`、`REVIEW-2.md`、`TICKET.md`、`repro-review2/results.json`、`repro-review2/run_log.txt` 均在版本库；`repro-review2/run_log.txt` 与 `REVIEW-2.md §②` 记录一致（343.3s、各臂 bpc、frontier 超出量、same-pos 行、`[gen] …91/128 = 0.7109`）。
- `git show 3ef14e6 --check` 与工作区 `git diff --check` 均干净。

### 7) 未复核范围：见 §⑤（如实登记）

---

## ③ blocking 清单（位置 + 违反判据 + 复现）

> 判据 = 本轮任务第 3 项给出的验收值：四处应为「≥25% 档 **0.0117–0.2593（0.84–18.5×）**、τ=0.95 单列」。以下两条为**同一问题**（B3 的 2 处），属「处置与证据/同文档表述不符」。

**BLOCK-1 —— `analysis.md:215`（§8 末行）仍写「超出量是 8.3–18.5 倍噪声带」。**
- 判据：该处应写 ≥25% 档区间 **0.0117–0.2593 bpc（0.84–18.5×）**（或明确限定它只讲 τ=0.3/0.6）。
- 证据：`results.json::frontier_comparison` 中省 ≥25% 的臂 excess = 0.259346 / 0.116819 / **0.011661**；`0.011661/0.014 = 0.84×`，`0.116819/0.014 = 8.3×` ⇒ 下界应为 0.84，不是 8.3。同一 `analysis.md` 的 §0（0.8–18）与 §7②（0.84–18.5）与之冲突。
- 复现：`sed -n '215p' axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/analysis.md`；再按 §②.3 的 Python 片段复算倍数。

**BLOCK-2 —— `project.yaml:16`（`failure_analysis④`）仍写「超出量是 8.3–18.5 倍噪声带」。**
- 判据 / 证据 / 复现：同 BLOCK-1，位置换成 `axes/2-depth/2.1.1-early-exit/project.yaml:16`；对照同文件 `conclusion`（第 17 行）「0.0117–0.2593 bpc（0.84–18.5 倍噪声带）」⇒ **同一 SSOT 内自相矛盾**。

**附带（同源，记录不一致）**：`TICKET.md:268–269`（§7.4）把 `analysis.md §8` / `project.yaml failure_analysis④` 列为已改「0.0117–0.2593（0.84–18.5×）」区间，与实际落盘（8.3–18.5 倍）不符。

**无其他 blocking**：B1 / B2 / B4 / non-blocking 批次 / 无回归 全部通过（见 §②）。

---

## ④ non-blocking / 观察（不构成 blocking，但如实登记）

- **O1**：`TICKET.md:167`（§4.5）仍写「省得多时门控显著更差（**0.25–18.5 倍噪声带**）」——这是 B3 被点名要改的**旧区间**（0.25 = τ=0.95，属省 20% 档，不属 ≥25%）。TICKET 是日志文档，但 §4.5 是「本轮交付」摘要，仍会误导。
- **O2**：`TICKET.md:173`（§4.5）写「`ECE_true` ≤ 0.030，**深度 ≥3 时 ≤ 0.005**」，而 `results.json::calibration.rows[depth=3].ece_true = **0.0051**`（> 0.005）——正是 round-1 blocking F1 同族「写紧了」的残留在 TICKET 内未修。
- **O3**：`TICKET.md:157`（§4.5）写「全量回归：`python3 -m pytest tests/ -q` ⇒ **831 passed**」；REVIEW-2 独立跑出的全量是 **834 passed**（831 = 加 3 条新测试之前的计数）。属 TICKET 内陈旧数字。
- **O4**：任务前置提到的处置报告 `.scratch/61-axis2-depth/REPAIR-round2.md` **不存在**；收口处置实际记在 `TICKET.md §7.4`。若需要独立「REPAIR 报告」这一交付物，属缺件（登记，不判 blocking）。
- **O5（非缺陷，备查）**：`analysis.md:38`（§0 第4点）的「0.8–18 倍」与其限定词「省得多的档（τ ≤ 0.9，省 25–71%）」一致，写法正确（REVIEW-2 亦判非缺陷）。

---

## ⑤ 未复核范围（诚实登记）

1. **未重跑实验**（纯 CPU，约 343s）：本轮以「仓内 `results.json` 在 `3ef14e6` 未被改动（diff 为空）+ 重跑证据与仓内版本 sha256 逐位相同（`6a8e2184…`）+ REVIEW-2 的字段级对账」代替；未独立再跑一次 `run.py`。
2. **未跑全量 `pytest tests/`**：只跑了目标文件（19 passed）与 `summary.py --check`；全量 834 passed 采信 REVIEW-2 的独立运行。
3. **未用 GPU**：`_missing_mask` 的 head 维广播（N5/U7）在 CUDA 上是否出错仍未测（本实验为 CPU 路径）。
4. **未审实现细节**：只核了「`3ef14e6` 未改 `pathlm/**`」与「实验 `results.json` 未改」，未逐行重审 `pathlm/depth_ar.py` / `pathlm/exit_policy.py` 的实现正确性（不在本轮 7 项范围内；REVIEW-2 已做变异与语义对拍）。
5. **未复核 `axes/2-depth/**` 除 `2.1.1` 外的其它 11 个 `project.yaml` 内容**：仅以 `summary.py --check`（46 份全过）为准。
6. **未核 `483b181` 之前的历史**：B1–B4 的对照基于 `483b181 → 3ef14e6` 的 diff 与文件现状。
7. **未逐行读完 `repro-review2/results.json`（7483 行）**：只核 sha256 与关键字段（arms / frontier / generation / calibration）。
8. **未核 `archive/**` 与其它轴**：范围仅 `axes/2-depth/**` 与 `.scratch/61-axis2-depth/**`。

---

## 判词（合并结论）

- **技术/科学内容：PASS**（B1 ✅、B2 ✅、B4 ✅、non-blocking 批次 ✅、无回归 ✅；`pathlm/**` 零改动、测量数字未改、`summary --check` OK、复核证据已入库）。这轮**没有**发现新的科学/实现/口径类缺陷。
- **B3 落盘自洽性：BLOCK（软，纯文本 2 处）** —— `analysis.md §8 末` 与 `project.yaml failure_analysis④` 的「8.3–18.5 倍」应为 ≥25% 档的 **0.84–18.5**（下界 8.3 是 τ=0.6 那档的倍数），且与同文档的 0.84–18.5 / 0.8–18 冲突 ⇒ 同一 SSOT 自相矛盾。修完这 2 个数字（可顺带统一 §④ 的 O1/O2）即可放行。
- **Merge verdict：BLOCK**（与 round-2 同标准：文本级「账实/自洽不符」= blocking；不改变任何科学结论）。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "以 round-3 fresh 独立复核者（只读、未改仓库）身份完成对 revision 3ef14e6 的逐条核对：B1（grep -c ^def test_ = 19、pytest 19 passed、全仓无当前事实写 16）、B2（TICKET §4.5/README §2/results.json arms 均 = 14）、B3（四处逐处核，2/4 落盘 0.0117–0.2593（0.84–18.5×），2/4 仍写 8.3–18.5 ⇒ ❌）、B4（28–71%）、non-blocking 批次（5 项全落）、无回归（summary --check OK 46、3ef14e6 仅 7 文件、pathlm/** 零改动、results.json 未改、REVIEW-2+repro-review2 入库）。未修改任何被复核文件。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "报告含 ①人话摘要+总判词 ②逐条核对（B1–B4 + non-blocking + 无回归）及全部复现命令与实测输出 ③blocking 清单（位置+判据+复现）④non-blocking/观察（O1–O5）⑤未复核范围（8 项）；每条结论均有 grep/pytest/sha256/diff 证据支撑。"
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "python3 -m pytest tests/test_axis2_exit_decision.py -q",
      "result": "passed",
      "summary": "19 passed in 3.43s（CPU，未用 GPU）"
    },
    {
      "command": "grep -c \"^def test_\" tests/test_axis2_exit_decision.py",
      "result": "passed",
      "summary": "19"
    },
    {
      "command": "grep -rn \"16 条\" axes/2-depth/ .scratch/61-axis2-depth/TICKET.md",
      "result": "passed",
      "summary": "无命中（仅 TICKET:229 的变更记录「16→19」）；REVIEW-1/2 的历史引用不计"
    },
    {
      "command": "grep -n \"条臂\" .scratch/61-axis2-depth/TICKET.md + README §2 + python json arms",
      "result": "passed",
      "summary": "TICKET:162=14；README:49=14；results.json arms=14 ⇒ B2 ✅"
    },
    {
      "command": "grep -n \"0.0117–0.2593|0.84–18.5|8.3–18.5|0.8–18\" analysis.md project.yaml",
      "result": "passed",
      "summary": "§7②/argument③ = 0.0117–0.2593（0.84–18.5）；§8末/failure_analysis④ = 8.3–18.5 ⇒ B3 只落 2/4"
    },
    {
      "command": "python3（读 results.json::frontier_comparison，复算 excess/0.014）",
      "result": "passed",
      "summary": "τ=0.3/0.6/0.9 ⇒ 18.5×/8.3×/0.84× ⇒ ≥25% 档区间 0.84–18.5；8.3 = τ=0.6 的下界"
    },
    {
      "command": "grep -rn \"28–71|28–55\" axes/2-depth/ + results.json::generation_agreement",
      "result": "passed",
      "summary": "三处均 28–71%；最高 0.7109（91/128）⇒ B4 ✅"
    },
    {
      "command": "python3 tools/summary.py --check",
      "result": "passed",
      "summary": "OK：46 个 project.yaml 全部符合 docs/research-schema.md"
    },
    {
      "command": "git show --stat 3ef14e6 / git show --name-only / git diff 483b181 3ef14e6 -- results.json",
      "result": "passed",
      "summary": "7 文件与提交信息一致；pathlm/** 0 改动；实验 results.json 未改；git show --check 干净"
    },
    {
      "command": "sha256sum 实验 results.json repro-review2/results.json",
      "result": "passed",
      "summary": "两者同为 6a8e2184fd7ffff02c0db212fae75209620c6b24f3a40ec4d8d4b2db2c2ed2e1（逐位一致）"
    },
    {
      "command": "git ls-files .scratch/61-axis2-depth/ + git status --porcelain + git diff --cached",
      "result": "passed",
      "summary": "REVIEW-1/2、TICKET、repro-review2/{results.json,run_log.txt} 已入库；工作区干净、无 staged"
    }
  ],
  "validationOutput": [
    "19 passed（目标文件）；summary.py --check = OK（46 份）。",
    "B1 ✅（19；全仓无当前事实写 16）；B2 ✅（TICKET §4.5=14 / README §2=14 / arms=14）；B4 ✅（28–71%）。",
    "B3 ❌ 部分：analysis.md:194 与 project.yaml:31 = 0.0117–0.2593（0.84–18.5 倍）+ τ=0.95 单列；analysis.md:215 与 project.yaml:16 = 「超出量是 8.3–18.5 倍噪声带」（下界应为 0.84 = 0.0117/0.014）。",
    "non-blocking 批次 5 项全落（4.6e-05 / argument② 浮点噪声 / §0「主要」/ conclusion_review 措辞 / _missing_mask 登记 U7）。",
    "无回归：3ef14e6 仅 7 文件、pathlm/** 零改动、实验 results.json 未改（sha 6a8e2184… 与 repro 一致）、git diff/--check 干净、无 staged。"
  ],
  "residualRisks": [
    "BLOCK：B3 的 2 处（analysis.md §8 末、project.yaml failure_analysis④）仍写「8.3–18.5 倍」，与同文档的「0.84–18.5」冲突 ⇒ 同一 SSOT 自相矛盾（不影响科学结论，改 2 个数字即可）。",
    "TICKET §4.5 仍有 3 处陈旧数字：167 行「0.25–18.5 倍噪声带」（旧区间，含非 ≥25% 的 τ=0.95）、173 行「ECE_true ≤0.005（深度≥3）」而实测 0.0051、157 行「831 passed」而全量为 834。",
    "任务前置的 .scratch/61-axis2-depth/REPAIR-round2.md 不存在（处置记于 TICKET §7.4）。",
    "_missing_mask 的 head 维广播（REVIEW-2 N5 / U7）仍未在 GPU 验证（本轮及相关轮均为 CPU）。"
  ],
  "noStagedFiles": true,
  "diffSummary": "本轮只读，未修改任何被复核文件；唯一写入 = 本报告。被复核区间 483b181 → 3ef14e6 的 7 个文件改动经 git show/diff 逐文件核查（无 pathlm/**、无实验 results.json 改动）。",
  "reviewFindings": [
    "blocker: analysis.md:215（§8 末行）- 「超出量是 8.3–18.5 倍噪声带」应为 ≥25% 档 0.0117–0.2593 bpc（0.84–18.5×）：0.011661/0.014=0.84（τ=0.9），8.3 只是 τ=0.6 的下界；与同文件 §0（0.8–18）/§7②（0.84–18.5）冲突。复现：sed -n 215p + results.json::frontier_comparison。",
    "blocker: project.yaml:16（failure_analysis④）- 同上「8.3–18.5 倍」，与同文件 conclusion:17「0.84–18.5 倍」在同一 SSOT 内自相矛盾。复现：grep -n '8.3–18.5\\|0.84–18.5' project.yaml。",
    "non-blocking: TICKET.md:167「0.25–18.5 倍噪声带」（旧区间，0.25 属省 20% 的 τ=0.95）；TICKET.md:173「ECE_true ≤0.005（深度≥3）」而实测 0.0051；TICKET.md:157「831 passed」而全量 834；TICKET.md:268–269 §7.4 声称 §8/failure_analysis④ 已落 0.84–18.5 区间，与实际（8.3–18.5）不符。",
    "B1/B2/B4/non-blocking 批次/无回归：无 blocker（19 条单测、14 条臂、28–71%、summary --check OK、pathlm/** 零改动、results.json 未改、复核证据入库）。"
  ],
  "manualNotes": "判词：技术/科学内容 PASS；B3 落盘自洽性 BLOCK（软，仅 2 处文本，下界 8.3→0.84）。B3 的这两处写法源自 REVIEW-2 第⑤节 B3「最小修法」自身给出的「改为 8.3–18.5」，而该节同时要求四处区间为 0.84–18.5——REVIEW-2 自相矛盾，收口 writer 按具体句落盘，故非 writer 无据，但按本轮验收值与 SSOT 自洽性仍未达标。建议：analysis.md §8 末与 project.yaml failure_analysis④ 的 8.3→0.84（或加限定词），顺带统一 TICKET §4.5 的 O1/O2/O3。"
}
```
