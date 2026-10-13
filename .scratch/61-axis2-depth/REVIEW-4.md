# REVIEW-4 · 轴 2 Phase A round-4 终审独立复核（revision `e970990`，基线 `3ef14e6`）

> 复核者：**round-4 fresh 终审复核者**（独立上下文 subagent，未参与本线任何写作；**只读**，未改仓库任何文件；唯一写入 = 本报告（落在 run 指定的 artifact 目录））。
> 被复核 revision：`e970990`（`轴 2 Phase A 终局收口：修 REVIEW-3 B3 残留（下界 8.3→0.84）并入库 round-2/3 复核件`）。
> 前置：round-3 判 **BLOCK**（仅 B3 残留 2 处，`.scratch/61-axis2-depth/REVIEW-3.md`）；round-2 = `REVIEW-2.md`；round-1 = `REVIEW-1.md`。
> 工作区核对：`cd /home/z/vibe/pathlm`；进入时 `git status --porcelain` 为空、`git diff --cached --name-only | wc -l` = 0；复核结束仍未提交任何 git、未 staged。
> 其它只读动作（如需要）全部在 `/tmp` 或内存中完成；**本报告不改任何被复核文件**。

---

## ① 人话摘要与总判词

这段交付的主线是「让每个字符自己决定跑几层」（逐 token 早退）。round-1/round-2/round-3 三轮独立复核已经把**科学内容与运行证据**全部确认通过；它们判 BLOCK 的**从来不是**结论或数字，而是「自己写下的账没跟证据对平」这类**文字级账实不符**。round-3 只剩半条：两份文档把「省 ≥25% 算力那一档超出固定截断多少倍」的下界写成了 **8.3**，而正确下界是 **0.84**（8.3 只是「省 46.6% 那档」的倍数，漏掉了「省 25.4% 那档」的 0.84）。`e970990` 就是来补这半条的终局收口提交。

**我把任务点名的 5 项逐条对仓库现状与证据核了一遍，结论如下**：

- **①（倍数区间）✅**：`analysis.md §8 末`（L215）与 `project.yaml failure_analysis④`（L16）现均写「**省 ≥25% 档超出量 0.84–18.5 倍噪声带（其中 τ=0.3/0.6 为 8.3–18.5×）**」，与同文档 `conclusion`（L17）、`argument③`（L31）、`§7②`（L194）的「0.84–18.5」、`§0`（L38）的「0.8–18 倍」**内部自洽**。
- **②（TICKET 3 处陈旧数字）✅**：`TICKET.md` §4.5 的 L167 已改「省 ≥25% 档 0.84–18.5 倍……τ=0.95 属省 20% 档」；L173 已改「深度 ≥3 时**最大 0.0051**」；L157 已改「**834 passed**」。三处均与 `results.json` / pytest 实测一致。
- **③（无新账实不符）✅**：对 `axes/2-depth/**` 与 `.scratch/61-axis2-depth/**` 的相关数字（`0.25`/`0.84`/`8.3`/`18.5`/`0.0117`/`0.2593`/`0.005`/`0.0051`/`831`/`834`/`16 条`/`19`/`11 条臂`/`14 条臂`/`28–55`/`28–71`/`4.6e-05`）逐项抽检，**当前事实文档里没有发现新的账实不符**；旧数字只残留在历史复核件 `REVIEW-1/2/3.md` 与历史处置件 `REPAIR-round2.md`（合规，见 §④ O2）。
- **④（无回归）✅**：`summary.py --check` 全绿（46 份）；`git show --stat e970990` = 恰好 5 个文件、与提交信息逐条一致、`--check` 干净；`pathlm/**` 零改动；实验 `results.json` 逐字节未改（sha256 `6a8e2184…` 与重跑证据 `repro-review2/results.json` **逐位相同**）；`REPAIR-round2.md`/`REVIEW-3.md` 已入库且与 artifact 源 sha256 逐位一致。
- **⑤（未复核范围）**：见 §⑤，如实登记。

**一句话**：这轮**没有**发现任何 blocking。B3 的两处下界已经真的改成 0.84 并加了正确限定词，同一份文档内部不再自相矛盾；TICKET 的 3 处陈旧数字也订正到与实测一致；实现与测量数字零改动、复核证据入库、工作区干净、无 staged。**判词 = PASS**。轴 2 Phase A **可收口**。（另有 2 条非 blocking 观察，见 §④；不影响放行。）

---

## ② 逐条核对（任务 1–5）

> 全部命令工作目录 `cd /home/z/vibe/pathlm`；未取 GPU 锁（本线全 CPU）。

### 1) B3 倍数区间（`analysis.md §8 末` + `project.yaml failure_analysis④`）：**✅**

**证据（`results.json::frontier_comparison`，倍数 = `frontier_excess / 0.014`）**：

| 臂 | layer-passes | 省算力 | `frontier_excess` | 倍数 |
|---|---|---|---|---|
| `token-thr-0.3` | 2.3325 | 70.8% | 0.259346 | **18.525×** |
| `token-thr-0.6` | 4.2683 | 46.6% | 0.116819 | **8.344×** |
| `token-thr-0.9` | 5.9668 | **25.4%** | 0.011661 | **0.833× ≈ 0.84** |
| `token-thr-0.95` | 6.3980 | 20.0%（**不属 ≥25% 档**） | 0.003493 | 0.249× |

⇒「省 ≥25%」= {τ=0.3, 0.6, 0.9}，区间 = **0.84–18.5**；其中 τ=0.3/0.6 子区间 = **8.3–18.5×**。

**现状（本轮实测 grep）**：
- `analysis.md:215`（§8 末行）：「统计假象（配对 SE ≤ 0.008，**省 ≥25% 档超出量 0.84–18.5 倍噪声带，其中 τ=0.3/0.6 为 8.3–18.5×**）。」→ ✅
- `project.yaml:16`（`failure_analysis④` 末句）：「…逐窗口配对 SE ≤ 0.008，**省 ≥25% 档超出量 0.84–18.5 倍噪声带（其中 τ=0.3/0.6 为 8.3–18.5×）**。」→ ✅

**内部一致性（同文档全部口径）**：
| 位置 | 现值 | 判定 |
|---|---|---|
| `analysis.md:215`（§8 末） | 0.84–18.5（其中 τ=0.3/0.6 为 8.3–18.5×） | ✅ |
| `analysis.md:194`（§7②） | 省 ≥25% 档 0.0117–0.2593 bpc（0.84–18.5 倍）；τ=0.95 单列 | ✅ |
| `analysis.md:38`（§0 第 4 点） | 0.012–0.26 bpc（**0.8–18 倍**）；限定词「τ ≤ 0.9，省 25–71%」 | ✅（三位有效数字显示，与 0.84 一致；REVIEW-2/3 亦判非缺陷） |
| `project.yaml:16`（`failure_analysis④`） | 0.84–18.5（其中 τ=0.3/0.6 为 8.3–18.5×） | ✅ |
| `project.yaml:17`（`conclusion`） | 0.0117–0.2593 bpc（0.84–18.5 倍）；τ=0.95 = 0.25 倍 | ✅ |
| `project.yaml:31`（`argument③`） | 0.0117–0.2593 bpc（0.84–18.5 倍）；τ=0.95 单列 | ✅ |

⇒ `conclusion`/`argument③`/`§7②`/`§8 末`/`failure_analysis④` **五处口径统一为「≥25% 档 0.84–18.5、τ=0.95 单列」**，同一 SSOT 内不再自相矛盾。round-3 的 BLOCK 根因（「下界 8.3」）已消除。

**复现**：
```
grep -n "0.84–18.5\|8.3–18.5\|0.0117–0.2593\|0.8–18" \
  axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/analysis.md \
  axes/2-depth/2.1.1-early-exit/project.yaml
python3 - <<'PY'
import json
d=json.load(open('axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/results.json'))
for r in d['frontier_comparison']:
    print(r['arm'], 'lp=%.4f'%r['layer_passes_per_token'], 'skip=%.4f'%(1-r['layer_passes_per_token']/8),
          'excess=%.6f'%r['frontier_excess'], 'x=%.3f'%(r['frontier_excess']/0.014))
PY
```

### 2) TICKET 3 处陈旧数字：**✅ 全订正**

| 位置 | 陈旧值 | 现值（本轮实测） | 证据 |
|---|---|---|---|
| `TICKET.md:167`（§4.5 主判据行） | 「0.25–18.5 倍噪声带」 | 「**省 ≥25% 档 0.84–18.5 倍噪声带；τ=0.95 属省 20% 档、超出 0.0035**」 | `frontier_comparison`：τ=0.95=0.249× 属省 20% 档 |
| `TICKET.md:173`（§4.5 校准行） | 「深度 ≥3 时 ≤ 0.005」 | 「深度 ≥3 时**最大 0.0051**」 | `calibration.rows`：`ece_true[depth=3]=0.0051`；depth≥3 的最大值 = 0.0051 |
| `TICKET.md:157`（§4.5 全量回归行） | 「831 passed」 | 「**834 passed**（F11 新增 3 条测试后的全量…）」 | 本轮独立实跑 `pytest tests/ -q` ⇒ **834 passed in 219.48s** |

- 复现：`grep -n "0.84–18.5\|最大 0.0051\|834 passed" .scratch/61-axis2-depth/TICKET.md`；
  `grep -n "0.25–18.5\|831 passed\|≤ 0.005" .scratch/61-axis2-depth/TICKET.md` ⇒ 当前事实区**无命中**（残留仅在 §7.5 的「订正说明」里引用旧值，属过程记录，合规）。
- `831`/`0.25–18.5`/`≤ 0.005` 的其它命中全部落在历史件 `REVIEW-3.md`（引用被复核时点）与 §7.5 变更记录，非当前事实，合规。

### 3) 无新账实不符（一致性抽检）：**✅**

对 `axes/2-depth/**` 与 `.scratch/61-axis2-depth/**` 的相关数字抽检结果：

| 扫描词 | 当前事实文档命中 | 判定 |
|---|---|---|
| `0.25–18.5` | 无（仅 `REVIEW-1/3.md` 历史引用 + `TICKET:294` 订正说明） | ✅ 已清 |
| `8.3–18.5` | `axes/2-depth` 仅 2 处（`analysis.md:215`、`project.yaml:16`），**均带正确限定**「其中 τ=0.3/0.6」 | ✅ |
| `0.84–18.5` | `analysis.md:194/215`、`project.yaml:16/17/31`、`TICKET:167/220/268` | ✅ |
| `0.0117–0.2593` | `analysis.md:194`、`project.yaml:17/31`、`TICKET:220/268` | ✅ |
| `0.0051` / `≤0.005` | `analysis.md:164`（0.0051）、`project.yaml:17`（0.0051）；`≤0.005` 已无（`TICKET:173` 现为「最大 0.0051」） | ✅ |
| `831` / `834` | `TICKET:157/261/296/311` 均为 834；`831` 仅在订正说明里引用旧值 | ✅ |
| `16 条` / `19 条` | 当前事实均 19（`README:20`、`analysis.md:20/235`、`project.yaml:17/18/115`、`TICKET:152`）；「16 条」无命中 | ✅ |
| `11 条臂` / `14 条臂` | `README:49`=14、`TICKET:162`=14、`results.json::arms`=14 | ✅ |
| `28–71` / `28–55` | 三处均 28–71（`analysis.md:54/208`、`project.yaml:36`）；`28–55` 无命中 | ✅ |
| `4.6e-05` | `README:38`、`MODEL.md:34`、`analysis.md:112/114`、`project.yaml:16/31`、`TICKET` 多处 | ✅（= `sanity_anchor` 最大 |Δbpc| 4.6e-05） |

其它关键数字再核（与 `results.json` 对照）：
- `sanity_anchor` Δbpc = 7e-06…4.6e-05（7 个深度）→ 与 `analysis.md §3` 表、`README §1`「≤ 4.6e-05」一致 ✅
- `same_position_paired` d8 桶：τ=0.6 = **0.003296**（`delta_window_se=0.001157`、`halves=[0.002038,0.004549]`）；τ=0.9 = **0.000552**（`se=0.000631`、`halves=[0.000666,0.000438]`）⇒ 与 `analysis.md:44/177/183`、`project.yaml:16` 的「+0.0033（SE 0.0012、两半 +0.0020/+0.0045）」「≤ 0.005」一致 ✅
- `generation_agreement`：最低 `token-thr-0.3`=0.28125、最高 `token-thr-0.999-min4`=0.710938（91/128）⇒ 与「28–71%」一致 ✅
- `calibration`：`ece_true` 全深度最大 = depth 1 的 0.0297（≤ 0.030）；depth ≥3 最大 = 0.0051；`ece_tf` = 0.196…0.370（0.20–0.37）⇒ 与 `analysis.md §5`、`project.yaml` 一致 ✅
- `risk_coverage` depth 1/τ=0.6：coverage 0.3716→0.372、kept 0.8769→0.877、dropped 0.3407→0.341 ⇒ 与 §5 表一致 ✅

**结论：当前事实文档无新账实不符。** 见 §④ 两条非 blocking 观察。

### 4) 无回归：**✅**

- `python3 tools/summary.py --check` ⇒ **`OK：46 个 project.yaml 全部符合 docs/research-schema.md。`**（EXIT=0）。
- `git show --stat e970990` ⇒ 恰好 **5 个文件**，与提交信息逐条一致：
  `A .scratch/61-axis2-depth/REPAIR-round2.md`、`A .scratch/61-axis2-depth/REVIEW-3.md`、`M .scratch/61-axis2-depth/TICKET.md`、`M …/2631-pertoken-exit-seqlevel/analysis.md`、`M axes/2-depth/2.1.1-early-exit/project.yaml`。
- `git show e970990 --check` 干净；`git diff --check` 干净。
- **未动实现代码**：`git show --name-only e970990 | grep -c "^pathlm/"` ⇒ **`0`**。
- **测量数字未改**：`git diff --stat 3ef14e6 e970990 -- …/2631-pertoken-exit-seqlevel/results.json` ⇒ **空**；且
  `sha256sum` 仓内实验 `results.json` 与 `.scratch/61-axis2-depth/repro-review2/results.json` **同为 `6a8e2184fd7ffff02c0db212fae75209620c6b24f3a40ec4d8d4b2db2c2ed2e1`**（逐位相同）。
- **复核证据已入库且原样**：
  - `REPAIR-round2.md`：仓内 sha256 `6169e5e7…a0fe23` **=** artifact 源 `d5033e53-…` sha256 `6169e5e7…a0fe23` ✅
  - `REVIEW-3.md`：仓内 sha256 `f342ca6e…82a5a6` **=** artifact 源 `b929ad65-…` sha256 `f342ca6e…82a5a6` ✅
- **工作区干净**：`git status --porcelain` ⇒ 0 行；`git diff --cached --name-only | wc -l` ⇒ **0**（无 staged）；`HEAD = e9709907092db8dda92c8d9364b80dc2e814cc37`。

### 5) 未复核范围

见 §⑤。

---

## ③ blocking 清单

**无 blocking。** 任务 5 项逐条通过（①倍数区间正确且内部一致 ✅；②TICKET 3 处订正 ✅；③无新账实不符 ✅；④无回归 ✅；⑤未复核范围如实列 ✅）。

---

## ④ non-blocking / 观察（如实登记，不构成 blocking）

- **O1（显示舍入，非缺陷）**：`analysis.md §5` 校准表（L146–152）把 `depth 3` 的 `ECE_true` 显示为 **0.005**（= 实测 0.0051 的三位小数舍入；同表其它格同精度，如 depth 1 的 0.0297→0.030）。同节正文（L164）与 `project.yaml:17` 均写精确值 **0.0051**。属**一致的四舍五入显示**，不构成账实不符；若要求逐位一致可把该格改 0.0051。
- **O2（命名/交付物，与任务头表述不一致）**：任务头把本轮报告写成 `.scratch/61-axis2-depth/REPAIR-round3.md`，但 `e970990` 实际入库的是 **`REPAIR-round2.md`**（round-2 处置报告，此前只存在于 artifact 目录）——**仓内不存在 `REPAIR-round3.md`**；round-3 的处置记录实际写在 **`TICKET.md §7.5`**（L282+）。该文件在 run artifact 目录存在（`…/713b2b7a-…/REPAIR-round3.md`）。此与任务**第 4 项**的期望（「`REPAIR-round2.md`/`REVIEW-3.md` 已入库」）**一致**，与**任务头的路径命名**不一致。round-3 处置在仓内可回溯（§7.5），故不构成审计缺口；**非 blocking**，仅登记供上级决定是否需要为对称性补一份仓内 `REPAIR-round3.md`。
- **O3（历史/日志件内部措辞不精确，按「历史件不改」保留）**：本轮新入库的 `REPAIR-round2.md` 的 acceptance-report `evidence`（第 98 行）把 `analysis.md §8` 与 `project.yaml failure_analysis④` 列为已改「0.84–18.5」；而其正文 §② B3（第 35–39 行）自己写的是这两处只落成「超出量是 8.3–18.5 倍」（正确）。即该**历史日志件**内部略有出入。它是 round-2 的过程记录（artifact 原样导入），且**当前事实文档已自洽**，按规范「日志不得与事实文档冲突，冲突以事实为准」不构成当前账实不符；`REVIEW-3 §④` 已记录该修法指令本身自相矛盾。**非 blocking**。
- **O4（备查，非缺陷）**：`analysis.md §0 第 4 点` 的「0.8–18 倍」为三位有效数字显示，与其限定词「省得多的档（τ ≤ 0.9，省 25–71%）」一致；REVIEW-2/3 亦判非缺陷。

---

## ⑤ 未复核范围（诚实登记）

1. **未重跑实验**（纯 CPU，约 200–350s）：以「仓内 `results.json` 在 `e970990` 未被改动（`git diff 3ef14e6 e970990` 为空）+ 重跑证据 `repro-review2/results.json` 与仓内版本 sha256 逐位相同（`6a8e2184…`）+ 本轮字段级复算 `frontier_comparison`/`calibration`/`generation_agreement`/`same_position_paired`」代替独立重跑。
2. **未用 GPU**：`_missing_mask` 的 head 维广播（REVIEW-2 N5 / `analysis.md §8 U7`）在 CUDA 上是否出错仍未测（本实验为 CPU 路径、生产路径 `fill_kv=True` 未触发）。
3. **未审实现细节**：只核了「`e970990` 未改 `pathlm/**`」与「实验 `results.json` 未改」，未逐行重审 `pathlm/depth_ar.py` / `pathlm/exit_policy.py` 的正确性（不在本轮 5 项范围内；REVIEW-2 已做变异与语义对拍）。
4. **未复核 `axes/2-depth/**` 除 `2.1.1` 外的其它 11 个 `project.yaml` 内容**：仅以 `summary.py --check`（46 份全过）为准。
5. **未核 `3ef14e6` 之前的历史**：B3 的对照基于 `3ef14e6 → e970990` 的 diff 与文件现状。
6. **未逐行读完 `repro-review2/results.json`（约 7483 行）与仓内 `results.json`（约 11895 行）**：只核 sha256 与关键字段。
7. **未核 `archive/**` 与其它轴**：范围仅 `axes/2-depth/**` 与 `.scratch/61-axis2-depth/**`。
8. **未核 artifact 源与仓内副本之外的第三方内容**；`REPAIR-round3.md` 仅阅读 artifact 目录版本用于对照，未纳入验收。

---

## 判词（合并结论）

- **技术/科学内容：PASS** —— B3 两处已正确改成「省 ≥25% 档 0.84–18.5（其中 τ=0.3/0.6 为 8.3–18.5×）」，与同文档 `conclusion`/`argument③`/`§7②`/`§0` 内部自洽；TICKET 3 处陈旧数字订正到与 `results.json`/pytest 一致；无新账实不符；无回归（`pathlm/**` 零改动、测量数字未改、`summary --check` OK 46、复核证据原样入库、工作区干净无 staged）。
- **落盘自洽性：PASS** —— round-3 的 B3 残留（下界 8.3）已消除；同一 SSOT 内不再有相互矛盾的门控超出量区间。
- **Merge verdict：PASS —— 轴 2 Phase A 可收口。**

> 仅 2 条非 blocking 观察：`analysis.md §5` 表 depth 3 的 0.005 是 0.0051 的三位小数舍入（O1）；仓内无 `REPAIR-round3.md`、round-3 处置记于 `TICKET §7.5`（O2，与任务第 4 项期望一致，仅与任务头命名不一致）。二者均不影响放行。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "以 round-4 fresh 终审复核者（只读、未改仓库）身份完成对 revision e970990 的任务 1–5 逐条核对：①analysis.md:215 与 project.yaml:16 均为「省 ≥25% 档 0.84–18.5 倍（其中 τ=0.3/0.6 为 8.3–18.5×）」，与 conclusion:17/argument③:31/§7②:194/§0:38 内部自洽；②TICKET §4.5 的 167(0.84–18.5)/173(最大 0.0051)/157(834 passed) 三处陈旧数字已订正并与 results.json/pytest 一致；③对 axes/2-depth/** 与 .scratch/61-axis2-depth/** 相关数字抽检无新账实不符；④summary --check OK(46)、git show --stat e970990=5 文件与信息一致、pathlm/** 零改动、results.json 未改、REPAIR-round2.md/REVIEW-3.md 原样入库(sh 逐位一致)、无 staged；⑤未复核范围如实列。未修改任何被复核文件。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "报告含 ①人话摘要+总判词(PASS) ②逐条核对(任务 1–5)及全部复现命令与实测输出 ③blocking 清单(空) ④non-blocking/观察(O1–O4) ⑤未复核范围(8 项)。每条结论均有 grep/pytest/sha256/git diff/字段级复算证据支撑。"
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "python3 tools/summary.py --check",
      "result": "passed",
      "summary": "OK：46 个 project.yaml 全部符合 docs/research-schema.md（EXIT=0）"
    },
    {
      "command": "python3 -m pytest tests/ -q",
      "result": "passed",
      "summary": "834 passed in 219.48s（CPU，未用 GPU）——与 TICKET §4.5「834 passed」一致"
    },
    {
      "command": "python3 -m pytest tests/test_axis2_exit_decision.py -q",
      "result": "passed",
      "summary": "19 passed in 3.94s；grep -c ^def test_ = 19"
    },
    {
      "command": "python3（读 results.json::frontier_comparison 复算 excess/0.014）",
      "result": "passed",
      "summary": "τ=0.3 18.525× / τ=0.6 8.344× / τ=0.9 0.833× / τ=0.95 0.249× ⇒ ≥25% 档区间 0.84–18.5、子区间 τ=0.3/0.6 = 8.3–18.5"
    },
    {
      "command": "python3（读 results.json::calibration）",
      "result": "passed",
      "summary": "ece_true[depth=3]=0.0051、depth≥3 最大 0.0051、全深度最大 0.0297（≤0.030）——与 TICKET:173/project.yaml 一致"
    },
    {
      "command": "python3（读 results.json::generation_agreement / same_position_paired / sanity_anchor / risk_coverage）",
      "result": "passed",
      "summary": "28–71%（0.28125–0.710938）；d8 桶 0.003296(se 0.001157/两半 0.002038,0.004549)；Δbpc 7e-06…4.6e-05；depth1/τ0.6 0.372/0.877/0.341 —— 均与 analysis.md/project.yaml 一致"
    },
    {
      "command": "grep -rn 各陈旧/关键数字（0.25–18.5、8.3–18.5、0.84–18.5、0.0117–0.2593、0.0051、831、834、16 条、19 条、11/14 条臂、28–55、28–71）",
      "result": "passed",
      "summary": "当前事实文档无新账实不符；旧值仅残留于历史件 REVIEW-1/2/3.md 与 REPAIR-round2.md 及 TICKET §7.5 订正说明"
    },
    {
      "command": "git show --stat e970990 / git show --check / git show --name-only | grep -c ^pathlm/ / git diff --stat 3ef14e6 e970990 -- results.json",
      "result": "passed",
      "summary": "5 文件与提交信息一致；--check 干净；pathlm 改动=0；实验 results.json 未改（diff 空）"
    },
    {
      "command": "sha256sum 仓内 results.json vs repro-review2/results.json；REPAIR-round2.md/REVIEW-3.md vs artifact 源",
      "result": "passed",
      "summary": "results.json 两者同为 6a8e2184…；REPAIR-round2.md 6169e5e7… 一致；REVIEW-3.md f342ca6e… 一致（原样入库）"
    },
    {
      "command": "git status --porcelain / git diff --cached --name-only / git rev-parse HEAD",
      "result": "passed",
      "summary": "工作区干净、无 staged；HEAD = e9709907092db8dda92c8d9364b80dc2e814cc37"
    }
  ],
  "validationOutput": [
    "summary.py --check = OK（46 份）；pytest tests/ = 834 passed；目标文件 19 passed。",
    "①倍数区间：analysis.md:215 与 project.yaml:16 均为「省 ≥25% 档 0.84–18.5 倍（其中 τ=0.3/0.6 为 8.3–18.5×）」，与 conclusion:17/argument③:31/§7②:194/§0:38 内部一致 ⇒ ✅。",
    "②TICKET §4.5：167=0.84–18.5、173=最大 0.0051、157=834 passed ⇒ 与 results.json/pytest 一致 ⇒ ✅。",
    "③抽检 axes/2-depth/** 与 .scratch/61-axis2-depth/**：当前事实文档无新账实不符；旧值仅在历史件与订正说明 ⇒ ✅。",
    "④无回归：e970990 仅 5 文件、pathlm/** 零改动、实验 results.json 未改（sha 6a8e2184… 与 repro 逐位相同）、REPAIR-round2.md/REVIEW-3.md 原样入库、git --check 干净、无 staged ⇒ ✅。",
    "判词：PASS —— 轴 2 Phase A 可收口。"
  ],
  "residualRisks": [
    "非 blocking O1：analysis.md §5 校准表 depth 3 的 ECE_true 显示为 0.005（实测 0.0051 的三位小数舍入），同节正文与 project.yaml 写 0.0051；属一致舍入，非账实不符。",
    "非 blocking O2：仓内不存在 .scratch/61-axis2-depth/REPAIR-round3.md（该文件仅在 run artifact 目录）；round-3 处置记于 TICKET §7.5。与任务第 4 项期望（REPAIR-round2.md/REVIEW-3.md 已入库）一致，仅与任务头命名不一致。",
    "非 blocking O3：新入库的历史件 REPAIR-round2.md 的 acceptance-report evidence（第 98 行）与其正文 §② B3 对 §8/failure_analysis④ 的落点描述略有出入；历史件按「不改」保留，当前事实文档已自洽。",
    "继承自前轮的残余（非本轮范围）：_missing_mask 的 head 维广播未在 GPU 验证（REVIEW-2 N5 / analysis.md §8 U7）；F5（milestone 验收口径 0.014 vs 前沿超出量 ≤0）仍待 Main 裁。"
  ],
  "noStagedFiles": true,
  "diffSummary": "本轮只读，未修改任何被复核文件；唯一写入 = 本报告。被复核区间 3ef14e6 → e970990 的 5 个文件改动经 git show/diff 逐文件核查（无 pathlm/**、无实验 results.json 改动；REPAIR-round2.md/REVIEW-3.md 原样入库，sha256 与源逐位一致）。",
  "reviewFindings": [
    "no blockers: B3 两处（analysis.md:215、project.yaml:16）下界 8.3→0.84 且加「其中 τ=0.3/0.6」限定 ⇒ 与 conclusion:17/argument③:31/§7②:194 的 0.84–18.5 自洽；TICKET §4.5 三处陈旧数字（0.25–18.5 / ≤0.005 / 831）已订正为 0.84–18.5 / 最大 0.0051 / 834，与 results.json 与 pytest 实测一致。",
    "no regression: summary --check OK(46)、git show --stat e970990=5 文件与信息一致、pathlm/** 零改动、results.json 未改（sha 6a8e2184… 与 repro 逐位相同）、REPAIR-round2.md(6169e5e7…)/REVIEW-3.md(f342ca6e…) 原样入库、无 staged。",
    "non-blocking: analysis.md §5 表 depth 3 ECE_true 显示 0.005（0.0051 舍入，正文写 0.0051）；仓内无 REPAIR-round3.md（记于 TICKET §7.5，仅与任务头命名不一致）；REPAIR-round2.md acceptance-report 措辞与正文略有出入（历史件）。",
    "PASS：轴 2 Phase A 可收口。"
  ],
  "manualNotes": "判词 PASS。任务头把报告写成 .scratch/61-axis2-depth/REPAIR-round3.md，但 e970990 实际入库的是 REPAIR-round2.md（round-2 处置报告，此前只在 artifact 目录），round-3 处置记于 TICKET §7.5——此与任务第 4 项期望一致，仅与任务头命名不一致；已作为非 blocking 观察登记（O2），供上级决定是否需要为对称性补一份仓内 REPAIR-round3.md。除此之外本轮 5 项全部通过，无 blocking。"
}
```
