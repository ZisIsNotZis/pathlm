# REVIEW-3 · 轴 1 时间轴 Phase A 旗舰 WP1 独立复核（研究项 `1.1.4.1` / 实验 `2638-carry-frontier-13m`）

## 0. 人话结论（先读这一段）

**先给结论：科学的账没有被这一轮松动，第 2 轮那条唯一的 blocking（同一个 `run.py` 两个哈希）已经消解；但这一轮又在「记账」上留下 1 条自相矛盾 + 若干处没同步，所以我仍判 FAIL，而且是窄范围的 FAIL——全部是文字/指针层面，不需要重跑 GPU。**

具体说：

1. **R2-B1 真修了**：`MODEL.md:41` 现在写 `2023f36d…`，与 `results.json:1031`（`run.py` 自己跑的时候算的 `code_hashes.run.py`）**逐字相同**；`MODEL.md:42` 的 `config.json` 指纹也与 `results.json:1032` 相同。同一交付里互相打架的两个哈希**不再存在**。我把 `MODEL.md` 里全部 16 个哈希逐条扫了一遍（10 个权重 + 6 个文件），没找到第二处陈旧值。**唯一做不到的**是我没有 shell，无法自己 `sha256sum run.py` 去核「磁盘值 = 这两个值」，只能采信 `TICKET.md` §8.5 第 1 行登记的实跑结果（它与我读到的两个值一致）。
2. **R2-B2 真修了**：`README.md:119` 已改成「默认 `ARMS="24 32 128"`；本实验另按 `ARMS="17 18 24 32 128"` 训练 5 臂」，`train_arms.sh:10` 的默认值、`train/win{17,18,24,32,128}/` 五个目录、§5 的 10 臂表、`analysis.md:143` 的复现命令**四处一致**；照 §7 逐字复现现在能训出全部臂。
3. **R2-N1/N2/N3/N4/N6/N7 逐条落地**（N5、N8 只落了一半，见 §4）。
4. **数值零回归**：`results.json` 的 `rows`/`frontier`/`location`（含 `-0.002259` / `-0.003920` / `0.125421` / `0.025` / `0.0117`）与我手上第 2 轮报告记录的值**逐位相同**，也与 `run_log.txt` 逐行相同。
5. **科学结论抽查通过**：`gap = 1.628110 − 1.502689 = 0.125421`、`win17` 严格支配 `carry16_s0`（208,896 ≤ 212,992 且 1.624104 < 1.624979）、插值 `1.62272`、recovery 2.5%/1.2%、`被否证` 只覆盖「本实现不占优」——我逐项复算一致。
6. **但这一轮又新造了一处矛盾（这就是我的 blocking）**：`project.yaml` 的 `conclusion_review`（`:19`）结尾还写着「**REVIEW-2 待派**」，而**同一个文件的 `change_log`（`:115-117`）写着「按 REVIEW-2 修 R2-B1/B2 + N1–N8」**——同一份 SSOT 里对「REVIEW-2 到底有没有派、有没有结论」给了两个互相打架的陈述；`conclusion_review` 这个**必填字段**（`docs/research-schema.md:51`：自评 + 独立复核的判词）里也只有 REVIEW-1 的判词，第 2 轮的 FAIL 判词没有回填。这正是本仓 `SCHEMA.md` §24 第 11/13 类「修复引入新错 / 覆盖小于声称」。
7. 另外还有 5 条 non-blocking，其中**两条是本轮新造的**：`README.md:121` 还写单测「**5 条**」，而文件里已经是 **6 条**（`TICKET.md` §8.5 自己记的是「6 passed」）；`README.md:95` 这个指针「（见 analysis §3）」指向的 `analysis.md` §3 **根本没有**「复用臂 bpc 逐位复现」这句话（`grep 逐位|复用|等值|eval-only analysis.md` = 0 命中），而第 2 轮的处置记录（`TICKET.md` §8.4 R2-N8）**自己已经承认** `analysis.md` §3 里没有这句。

**一句话总判词：FAIL（窄范围）**——科学结论 PASS、R2-B1 已消解，但仍在 SSOT 里留下 1 条自相矛盾（`conclusion_review` 说 REVIEW-2 待派 vs `change_log` 说已按 REVIEW-2 修）+ 5 条记账不一致，全部为文字/指针，**不需要重跑 GPU**（改动量约 4–6 行）。若父级裁示「复核判词以 `TICKET.md`/`REVIEW-2.md` 为准，`project.yaml::conclusion_review` 允许滞后一轮」，则这条降为 P2，总判词转「OK with notes」——这个取舍属框架性判断，我把它留给你。

---

## 1. 复核对象 / 复核者身份 / 能力限制

- **复核对象 revision**：**`940cb23`**。`watchdog_diff` 回报 **`No working-tree changes against reviewer-launch HEAD 940cb2347a85`** ⇒ 工作树对该 HEAD **干净**，本轮对象确为 `940cb23`。
- **复核者身份**：fresh 上下文（未参与本线任何实验/文档，**不是**第 1/2 轮复核者，**不是**本项 writer）。**未修改仓库任何其它文件**（`changedFiles = []`）；本报告由 runtime 落盘到 `.scratch/62-axis1-time/REVIEW-3.md`。
- **能力限制（如实登记，逐项）**：
  1. **无 shell**：`python3 tools/summary.py --check`、`python3 -m pytest tests/ -q`、`python3 -m pytest …/test_2638_frontier.py -q`、`sha256sum`、`git diff 12e5fb7 443cb44 --stat` / `git diff 443cb44 940cb23 --stat` **全部 not-run**（见 §6）。因此「`MODEL.md` 的指纹 = 磁盘实际值」我只能做**一致性**核验，**不能**做磁盘复算。
  2. **提交区间内容看不到**：`watchdog_diff` 明说 `Committed changes are not included` ⇒ `12e5fb7 → 443cb44 → 940cb23` 到底改了哪些文件，我**没有独立看过**。本报告全部行号取自**当前工作树**，结论建立在「逐行重读内容」之上，**字节层无证**（这正是第 1/2 轮登记过的同一残差，本轮**没有**被消除）。
  3. **`run.py` 相对上一轮的逐行 diff 我没有**：判「只改注释/活性块、未改判据逻辑」只能用**三方一致性**（代码 ↔ 落盘 `results.json` ↔ 单测）推断，不是逐行比对（见 §3.4）。
  4. **checkpoint sha256 无法磁盘重算**：我改用了**三处对照**（`MODEL.md:5-14` ↔ `config.json` ↔ `results.json.checkpoints`）＋「他线同值旁证」，已在 §3.1 标为替代证据。

---

## 2. 逐条判词（对任务 7 个必核项）

### 2.1 项 1 · R2-B1 是否真修 —— **真修（同一交付内的矛盾已消失）；磁盘值未能独立复算**

| 位置 | 值 | 判 |
|---|---|---|
| `MODEL.md:41`（`run.py`） | `2023f36d1a9694c8579bfab76c76da5e5b57ab215cfdba46f52f992164615778` | ✓ |
| `results.json:1031`（`code_hashes.run.py`，运行期自算） | `2023f36d…`（同上） | ✓ 逐字相同 |
| `MODEL.md:42`（`config.json`） | `eb97327d6523eb2003d3821228223ff60ad62c6ba6941e4c961d693ee4227f55` | ✓ |
| `results.json:1032`（`code_hashes.config.json`） | `eb97327d…`（同上） | ✓ 逐字相同 |

- **全仓逐条扫 `MODEL.md` 的全部哈希（16 条）**，未发现第二处陈旧值：
  - 10 个权重 sha256（`MODEL.md:5-14`）↔ `config.json` 各臂 `sha256`（`:32,48,64,80,96,112,128,144`）↔ `results.json.checkpoints`（`:40,52,64,76,88,100,112,124,136,149`，全 `sha256_verified: true`）**三处逐字一致**（替代证据：非磁盘复算）。
  - 4 条共享文件指纹（`MODEL.md:37-40`）↔ `results.json:1027-1030` **逐字一致**；其中 `train_depth_ar.py = 218b6317…` 与 `pathlm/data.py = 48e23520…` 在**其它实验**（`2637/MODEL.md:37`、`2637/config.json:121`、`2636/MODEL.md:44`、`2635/results.json:2204` 等）**取到同值** ⇒ 旁证这两个文件未变。`tools/evalkit/retention.py = 9ef983b7…` 与 `pathlm/depth_ar.py = 1f8f9eb4…` **只在本交付出现**（`grep` 全 `axes/` 仅命中 `2638/MODEL.md:37,39` 与 `2638/results.json:1027,1029`，另有 `61-axis2-depth/REVIEW-2.md:140` 记 `1f8f9eb4…` = `483b181` 版），故这两条**没有第三方旁证**——若它们陈旧，我这一层查不出来（已列入残差）。
- **仍未独立证实的一环**：磁盘上 `run.py` 实际 sha256 是否 = `2023f36d…`。`TICKET.md` §8.5 第 1 行登记 `sha256sum …/2638/run.py` = `2023f36d…`（= `results.json`），与 `change_log:117` 的说法一致；我**采信**，但**不冒充独立证据**。
- 分支判断（第 2 轮给的二选一）：处置走的是**分支 1（`MODEL.md` 陈旧 → 改 `MODEL.md`）**，不需要重跑 `run.py`，与 `TICKET.md` §8.4/§8.5 一致 ✓。

### 2.2 项 2 · R2-B2 是否真修 —— **真修（四处一致；照 §7 可训出全部臂）**

- `README.md:119`：`bash train_arms.sh`（默认 `ARMS="24 32 128"`；本实验另按 `ARMS="17 18 24 32 128"` 训练 5 臂，见 §5 追加登记）⇒ `train/win{17,18,24,32,128}/{model.pt,results.json,train_log.jsonl}`。
- 对照：`train_arms.sh:10` `ARMS="${ARMS:-24 32 128}"`（默认值真、环境变量可覆盖 ⇒ 字面命令可执行）✓；`train/win17|win18|win24|win32|win128/` **5 个目录实际存在** ✓；`README.md:80-90` 的 10 臂表 ↔ `config.json::arms`（10 条）↔ `results.json:793-924`（10 行）逐行参数量/字节/`k_eff` 一致 ✓；`analysis.md:143` 给出的正确命令 `ARMS="17 18 24 32 128" bash …/train_arms.sh` ✓。
- 结论：**照 §7 复现能训出全部 5 个训练臂**（win17/18/24/32/128），`run.py` 不会因缺权重失败。R2-B2 关闭。

### 2.3 项 3 · R2-N1–N8 逐条落地

| 项 | 判 | 证据 |
|---|---|---|
| **N1**（删事实文档变更叙事） | **✓ 落地** | 全仓 `grep 复核修正记录` **仅命中** `.scratch/62-axis1-time/REVIEW-2.md:143` 与 `TICKET.md:221`（日志/复核件，非事实文档）⇒ `README.md` / `analysis.md` 里的「复核修正记录（B1）」两段**已删净**。`README.md:45` 的标题「（预登记，写下后不得改）」保留、正文已是当前状态（`README.md:47-55`、`analysis.md:47-53`） |
| **N2**（2638 证据锚点） | **✓ = `443cb44`** | `project.yaml:47` `…/2638-carry-frontier-13m/analysis.md @443cb44`；`:48` `…/results.json @443cb44`。两处与「该产物最后变更 commit = round-2 落盘」一致 ✓（`check_tree.py:71-104` 只校「锚点可达」，机检抓不到陈旧，故需人工核，已核） |
| **N3**（`WP1.means` 半句） | **✓** | `project.yaml:78`：「…同图画质量-召回（**本轮只交付质量轴 + 窗口点 + full 锚；召回轴与量化点见 gaps 首条**）」，与 `:79 acceptance` 自认「召回轴与量化点未叠加」不再打架 |
| **N4**（判据④ 跨 seed 标注） | **✓** | `analysis.md:50`：「Δbpc = **+0.003131**（seed0 vs `win16_s0`）/ **+0.001470**（seed1 vs `win16_s0`，**跨 seed** 口径；同 seed 配对口径 = **+0.002453**）——两口径皆 >0」 |
| **N5**（单测断言是否实质） | **实质（对公式）；但对「判定线文字」未钉住，且文件仍不在 `tests/`** | 见 §3.5 的**手工推演**与 R3-N4 |
| **N6**（`TICKET.md` §8.2 的 N6/N9 落点） | **✓ 订正** | `TICKET.md:190` N6 落点现为「`analysis.md` §0、**`project.yaml` `conclusion_review`**」；实际限定语也确实在 `analysis.md:31`（§0 标题句）与 `project.yaml:18`（`conclusion_review` 的「σ_cell 边缘（0.000875 ≈ 0.9σ）、0.014 带不显著」）✓；`TICKET.md:193` N9 落点现为「`README.md` §5（§3 未加此句——R2-N6 指出，落点已订正）」✓（但见 R3-N2：`README.md:95` 仍指向 `analysis §3`） |
| **N7**（`8.4–8.53`、`0.30 min`） | **✓ 落地** | `README.md:107`「5×8.4–8.53 min」= 逐臂实测 `train/win128→8.4`、`win18→8.47`、`win17→8.51`、`win32→8.51`、`win24→8.53`（各 `train/win*/results.json:766`）✓；`README.md:108` 与 `analysis.md:123` 的评测墙钟均改为 **0.30 min** = `results.json:1022` `wall_minutes_total_eval: 0.3` ✓ |
| **N8**（复用 = eval-only 表述所在处） | **半落地** | `README.md:93-95` 有「复用臂的复用范围 = **eval-only**（不重训、不改配置）」✓，符合 R2-N6 的落点订正；但该句尾的**指针**「（见 `analysis §3`）」在 `analysis.md` §3 **找不到对应内容**（见 R3-N2） |

### 2.4 项 4 · 无回归 —— **数值零回归（用第 2 轮记录值 + 运行日志双路对照）**

- `results.json` 的 `rows`（`:793-924`）：full 1.502689 / win16_s0 1.62811 / win16_s1 1.629093 / **win17_s0 1.624104** / win18_s0 1.619953 / win24_s0 1.599856 / win32_s0 1.595823 / win128_s0 1.513325 / **carry16_s0 1.624979** / carry16_s1 1.62664，`total_bytes` 196608/196608/208896/212992/212992/221184/294912/393216/1572864/3145728、`k_eff` 17.3333/… **与 `REVIEW-2.md` 记录的值逐位相同** ⇒ 文字修复**没有**动到任何数字。
- `frontier`（`:926-957`）：`lower_envelope = [win16_s0, win17_s0, win18_s0, win24_s0, win32_s0, win128_s0, full]`；`dominated_by = {win16_s1:[win16_s0], carry16_s0:[win17_s0], carry16_s1:[win17_s0, carry16_s0]}` ✓ 不变。
- `location`（`:958-1000`）：`carry_minus_interp_window = -0.002259 / -0.00392`、`bpc_interp = 1.62272`、`between = [win17_s0, win18_s0]`、`w = 0.3333`、`recovery_ratio_r = 0.025 / 0.0117` ✓ 不变。
- `run_log.txt:23-30` 与 `run_log.jsonl`（同一次运行）的 10 臂 bpc / `k_eff` / `total_bytes` / 包络 / 支配集 / 插值 **与 `results.json` 逐字相同**（`run_log.txt:26-30`：`carry16_s0: bpc=1.624979 k_eff=17.33 total=212992`、`vs 窗口插值=-0.002259`）⇒ 落盘 `results.json` 与**当前交付的 `run.py`** 在同一次运行里对平（替代证据，非我重跑）。
- `run.py` 判据逻辑：我读到的判据相关代码是 `frontier()`（`:103-118`）、`interp_window_at()`（`:121-131`）、自检硬断言（`:170-179`）、`rel = round(loc["bpc_interp"] - c["bpc"], 6)`（`:214`）、`location[*].note`（`:220-224`）、评测期掩码反事实探针（`:264-273`）——与 `README §4` 的判据线、`analysis.md §1-§3`、`results.json` 的值、`test_2638_frontier.py` 的断言**四方自洽**，未见「改了判据却未反映到落盘数字」的迹象。**但**「相对 `12e5fb7` 的 `run.py` 逐行 diff」我没有（能力限制 §1.3），故此项判为**「无回归的外观证据充分」**，不是「已逐行确认」。

### 2.5 项 5 · 代跑项 —— **全部 not-run（无 shell）**

| 应按任务跑的命令 | 我 | 依据 |
|---|---|---|
| `python3 tools/summary.py --check` | **not-run** | 采信 `TICKET.md` §8.5 第 3 行（`OK：46 个 project.yaml`） |
| `python3 -m pytest tests/ -q` | **not-run** | 采信 `TICKET.md` §8.5 第 4 行（`834 passed`） |
| `python3 -m pytest …/test_2638_frontier.py -q` | **not-run** | 采信 `TICKET.md` §8.5 第 2 行（`6 passed`）；我另行**人工数出该文件确为 6 个 `test_*` 函数**（见 R3-N1），与「6 passed」一致 |
| `sha256sum` 抽查 2 个 checkpoint 对 `MODEL.md` | **not-run** | 替代对照：`config.json` ↔ `MODEL.md:5-14` ↔ `results.json.checkpoints` 三处逐字一致 |

（`tools/summary.py --check` 实际只能查「字段齐全/非空/状态合法/锚点可达」这类机器项，**查不出**我的 R3-B1/§3.3 这类「同一文件内两处陈述互相矛盾」，所以它即使 OK 也不能反驳本报告。）

### 2.6 项 6 · §24 第 11/13 类（修复引入新错 / 修复范围小于声称）—— **本轮确实又引入/留下了不一致**

**新引入（本轮改动直接造成）**：

- **R3-B1（blocking，P1）**：`project.yaml:19` 的 `conclusion_review` 结尾仍写「**REVIEW-2 待派**」，而同文件 `change_log:115-117` 写「按 REVIEW-2 修 R2-B1/B2 + N1–N8」；且该必填字段（`docs/research-schema.md:51`）只含 REVIEW-1 判词、不含 REVIEW-2 判词（第 2 轮判的是 **FAIL**）。另 `project.yaml:20` 是**重复的** REVIEW-1 段并写「复核前**未改**本 `project.yaml` 的判据或数字」，与 `change_log:120`（round-1 修复改了 `project.yaml` 的 `conclusion`/`failure_analysis④`/`gaps` 首条/`WP1.acceptance`）相冲突；`WP1.review`（`:81`）也只列 `REVIEW-1.md`。⇒ 命中 §24 第 11 类（修复引入新错，形态=「改了 `change_log`、没扫同一字段的兄弟陈述」）与第 13 类（覆盖小于声称）。
- **R3-N1（P2）**：`README.md:121` 仍写单测「**5 条**」，实际 6 条（round-2 把弱断言换成实质断言时新增了 `test_results_json_domination_matches_documented_sign_rule`），`TICKET.md` §8.5 自己也记「6 passed」⇒ 同一交付内自相矛盾，且漏列了新增的那条断言主题。

**留下（本轮未引入、但仍是当前事实文档里的不一致）**：

- **R3-N2（P2）**：`README.md:95` 指针「（见 `analysis §3`）」悬空（`analysis.md` 无「复用/等值/逐位/eval-only」任一字样）；`TICKET.md` §8.4 R2-N8 已承认 `analysis.md` §3 没有这句，却没顺手改指针。
- **R3-N3（P2）**：`README.md:65`（§4a 操作性定义）仍把「纯窗口曲线」定义为 `win16/win24/win32/win128` 四点，而交付的插值用的是 `win17↔win18` 弦（`analysis.md:60`、`results.json:968-973`）⇒ 契约的操作性定义与产物不一致（与 R2-B2 同族，round-2 只同步了 §7）。
- **R3-N5（P2）**：`analysis.md:4` 与 `:5` 都写「权重清单 = `MODEL.md`；」，同一事实在一份事实文档里出现两次（§2.1「每条事实恰好出现一次」），像删叙事时的残留。
- **R3-N6（P2）**：`project.yaml:115`（15:58:00，REVIEW-2）排在 `:118`（15:48:43，REVIEW-1）**之前**，`change_log` 时间倒序错位。

### 2.7 项 7 · 科学结论抽查 —— **未被松动（我复算的部分全部一致）**

| 项 | 文档值 | 我的复算 | 判 |
|---|---|---|---|
| 主判据：`win17_s0` 严格支配 `carry16_s0` | 208,896 B < 212,992 B 且 1.624104 < 1.624979 | 同（`results.json:849-858` vs `:868-877`） | ✓ |
| `gap` | 0.125421 | 1.628110 − 1.502689 = **0.125421** | ✓ |
| gap-closing | 2.5% / 1.2%；+1 格 3.2%、+2 格 6.5%、+8 格 22.5%、+112 格 91.5% | 0.003131/0.125421=**2.50%**；0.00147/0.125421=**1.17%**；0.004006→**3.19%**；0.008157→**6.50%**；0.028254→**22.53%**；0.114785→**91.52%** | ✓ |
| 插值点 / rel | 1.62272 / −0.002259 | 1.624104 + (1/3)(1.619953−1.624104) = **1.622720**；1.62272−1.624979 = **−0.002259** | ✓ |
| 「质量只等于窗口 16.9」 | 16.9 | (1.624979−1.62811)/(1.599856−1.62811)=0.1108 ⇒ 16+8×0.1108=**16.89** | ✓ |
| `被否证` 归因 | 实现/设计主因、结构次因、非 innovation 失败（`project.yaml:16`、`analysis.md:104-119`） | 未越界（显式禁外推到 delta 族） | ✓ |
| 复用臂出处 | `2634/analysis.md:124-127` 的 1.628110 / 1.629093 / 1.624979 / 1.626640 | 与 `results.json:819,835,909,925` 逐位相同 | ✓ |

---

## 3. 三点补充核验（供父级定位）

### 3.1 checkpoint 与共享文件指纹（替代证据）

10 个权重哈希在 `MODEL.md:5-14`、`config.json`、`results.json.checkpoints`（`:168-298`）**三处逐字一致**；`run_log.txt:3-12` 打印的运行期前缀（`86953dcb…`/`c3f6f08a…`/…/`b3d3cdbc…`）与之一致。磁盘重算做不到（无 shell）⇒ 列为残差。

### 3.2 §20/预算数字的一致性（顺手核，均对平）

`README.md:106-111` 的 40 min 预算 / 实测 ≈48 min：训练跨度 `train_arms_driver.log`（win24/32/128）与 `driver2.log`（win17/18）覆盖 `14:44:31–15:31:46` = 47.25 min，+ 评测 0.30 min + 冒烟 0.22 min（`smoke/steps300/smk24/results.json:766` `wall_minutes=0.22`）≈ **48 min** ✓；`analysis.md:121` 引逐臂 `wall_minutes` 的路径 `train/win{17,18,24,32,128}/results.json` 实际存在 ✓。

### 3.3 「本轮是否只改了 2638 目录 + `project.yaml`」——**仍只是内容层核对**

`watchdog_diff` 不含已提交区间（§1.2），`12e5fb7→443cb44→940cb23` 的文件清单我没看过。我能说的只是：**读到的** `2638` 目录内文件、`project.yaml`、`TICKET.md` §8.4–§8.6 在内容上是自洽的一轮返工。字节层无证。

### 3.4 判据③ 的方向在四处一致（B1 未被回退）

`README.md:53`（判据③ 行）／`analysis.md:49`（判读③ 行）／`run.py:220-224`（`location[*].note`：`>0` 改善、`≈0` 不可排序、`<0` 被支配）／`results.json` 两臂的 `note`（`:975,993`）**四处同向**，与 `SCHEMA.md` §17 第 3 条「Δ = 对照 − 实验，正 = 改善」一致；代入 `carry16_s0` 得 −0.002259 ⇒ 四处都判「被支配」，与判读栏、`frontier.dominated_by` 同向 ✓（第 1 轮 B1、第 2 轮确认，本轮未见回退）。

### 3.5 R2-N5 的**手工推演**（不看注释，直接推）

`test_2638_frontier.py:75-93` 的核心断言是两条：
① `assert rel == pytest.approx(interp - rows[name]["bpc"], abs=1e-5)`；
② `assert (rel < 0) == (len(dom.get(name, [])) > 0)`。

- **若有人把 `run.py:214` 的公式写反**（`c["bpc"] - loc["bpc_interp"]` 并重跑）：`rel` 变成 **+0.002259**，断言① `+0.002259 == -0.002259` ⇒ **红** ✓（对 carry16_s1 同理）。
- **若有人只把 `run.py:220-224` 的判定线文字写反**（例如把「<0 ⇒ 被支配」改成「>0 ⇒ 被支配」）并重跑：`results.json.location[*].carry_minus_interp_window` 数值不变、`frontier.dominated_by` 不变、`notes` 只是**字符串**，而断言①只比数值、断言②只用「rel<0 ⇔ dominated」这个**测试自己写死的**规则去对数据 ⇒ **仍然全绿**。**而这恰恰就是第 1 轮 B1 的缺陷形态（数值正确、方向文字写反）**。
- 结论：新断言相对旧的子串检查（`"bpc_interp" in inspect.getsource(R.main)`）是**实质改进**（能抓公式级反向），但 `test_2638_frontier.py:75-79` docstring 里「若有人把判定线方向写反，本条会红」**言过其实**——它抓不住「只改判定线文字」这一类回归。另外该文件仍放在实验目录，`python3 -m pytest tests/ -q`（`AGENTS.md` §8 的仓库级门禁）**不收集它**（`tests/` 下 57 个文件里无 `*frontier*`，`grep 2638 tests/` = 0 命中），本项以边界纪律（不改共享 `tests/`）为由保留在实验目录（`README.md:121`、`TICKET.md` §8.4 R2-N5）。

---

## 4. Blocking（1 条）

**R3-B1（P1）· `project.yaml` 的 `conclusion_review` 仍称「REVIEW-2 待派」，与同一文件 `change_log` 直接冲突；必填字段缺 REVIEW-2 判词**

- **位置**：`axes/1-time/1.1.4.1-explicit-memory/project.yaml:19`（`conclusion_review` 段末「…上述 B1–B4 + N1/N2/N4/N7 已修（见 `change_log` 与 `.scratch/62-axis1-time/TICKET.md` §8）；**REVIEW-2 待派**。」）、`:20`（重复的 REVIEW-1 段 +「复核前**未改**本 `project.yaml` 的判据或数字」）、`:81`（`WP1.review` 只列 `REVIEW-1.md`）。
- **证据（同一文件内两处互相打架）**：`project.yaml:115-117` = `change_log` 条目「2026-10-13T15:58:00 … **按 REVIEW-2 修 R2-B1/B2 + N1–N8**（… MODEl.md 指纹重生成、README §7 补 5 臂、锚点前移到 @443cb44）」。而 REVIEW-2 已于 `2026-10-13T15:54` 派发、15:56 落盘并判 **FAIL（窄范围）/ 科学结论 PASS**（`TICKET.md` §8.4；`REVIEW-2.md` §6 总判词）⇒「REVIEW-2 待派」是**已假的当前状态陈述**。`project.yaml:120`（round-1 处置）明确记录 `project.yaml` 的 `conclusion`/`failure_analysis④`/`gaps` 首条/`WP1.acceptance` 被改 ⇒ `:20` 的「未改本 `project.yaml` 的判据或数字」与之冲撞。
- **违反**：`AGENTS.md` §2.1（事实文档只写当前状态、一条事实一处、不带流水叙事）＋ `docs/research-schema.md:51`（`conclusion_review` = 自评 **+ 独立复核的判词**，必填）＋ `SCHEMA.md` §24 第 11 类（修复引入新错）/ 第 13 类（覆盖小于声称）。
- **影响**：研究项 SSOT 上读不出「最新一轮独立复核的结论是什么、还有什么未关账」——第 2 轮的 FAIL 在外观上不存在；读者会以为流程停在「REVIEW-2 待派」。不影响任何数字与结论，**不需要重跑 GPU**。
- **最小修法（约 4–6 行，纯文字）**：
  1. `:19` 末句改为「…上述 B1–B4 + N1/N2/N4/N7 已修；**REVIEW-2（fresh）判 FAIL（窄范围）/ 科学结论 PASS，唯一 blocking 为 R2-B1 指纹矛盾，已按实际值重生成 `MODEL.md` 指纹；round-3 复核见 `.scratch/62-axis1-time/REVIEW-3.md`**」。
  2. 删除 `:20` 整段重复的 REVIEW-1 段（内容已被 `:19` 覆盖；其历史留在 `change_log`/`TICKET.md` §8.1）。
  3. 顺手把 `:81` 的 `WP1.review` 改为「复核件见 `.scratch/62-axis1-time/REVIEW-1.md` / `REVIEW-2.md` / `REVIEW-3.md`」。
- **复现**：`sed -n '19,20p;115,120p' axes/1-time/1.1.4.1-explicit-memory/project.yaml` —— 19 行说「待派」、117 行说「已按 REVIEW-2 修」。

---

## 5. Non-blocking（5 条）

- **R3-N1（P2）· `README.md:121` 单测条数陈旧**：写「`test_2638_frontier.py`（**5 条**，钉住支配定义、插值、**符号约定（B1）**、掩码可见对数公式）」，实际文件有 **6** 个 `test_*`（新增的 `test_results_json_domination_matches_documented_sign_rule`，`test_2638_frontier.py:75`），`TICKET.md` §8.5 也记「6 passed」；`test_2638_frontier.py:8` 的 docstring 同样仍是「钉住**三件事**」。属 §24 第 13 类形态 4（改了一处、兄弟处未同步），**本轮新造成**。最小修法：把「5 条」改「6 条」，主题列表补「落盘 `results.json` 的支配关系与判定线一致」。
- **R3-N2（P2）· `README.md:95` 指针悬空**：「复用等值性由『4 个复用臂的 bpc 与本实验评测口径下 2634 的产出逐位相同』独立佐证（**见 `analysis §3`**）」——`grep -nE '逐位|复用|等值|eval-only' analysis.md` = **0 命中**，`analysis.md` §3（`:70-82`）只讲 σ_cell / 0.014 带 / 状态精度脚注。证据实际在 `results.json:815-925` 与 `2634/analysis.md:124-127`；`TICKET.md` §8.4 R2-N8 **自己已承认**「`analysis.md` §3 未加此句」。最小修法：把「见 `analysis §3`」改成「见 `results.json` 的 4 个复用臂 bpc 与 `2634/analysis.md:124-127`」。
- **R3-N3（P2）· §4a 的纯窗口曲线定义与交付不一致**：`README.md:65` 定义曲线 = `win16/win24/win32/win128`（seed 0），而交付的定位插值用 `win17↔win18`（`analysis.md:60`、`results.json:968-973`）、下包络含 7 个窗口点（`analysis.md:55`）⇒ 与 R2-B2 同族（只修了 §7）。最小修法：在 `README.md:65` 后补半句「（执行期追加 `win17/18` 后曲线为 `win16/17/18/24/32/128`；**预登记 4 点集**的弦插值另报于 `analysis §2`=1.623401）」。
- **R3-N4（P2）· 单测的钉子范围与落点**：见 §3.5 手工推演——能抓公式反向、抓不住「只改判定线文字」（= 第 1 轮 B1 的形态），docstring（`test_2638_frontier.py:75-79`）的「判定线方向写反即红」言过其实；且该文件不在 `tests/`，仓库级门禁 `pytest tests/ -q` 不收集它（本项已由 `README.md:121`/`TICKET.md` §8.4 作为边界取舍显式登记）。最小修法：docstring 改成「抓 **`rel` 公式与判定线的数值一致性**（文字型回归不受保护）」，或在断言里加一条对 `run.py` 中 `note` 文本含 `"< 0 ⇒"`/`"被"` 的实质检查。
- **R3-N5（P2）· `analysis.md` 表头重复一行**：`:4` 与 `:5` 均以「权重清单 = `MODEL.md`；」开头（同一事实出现两次，§2.1）。最小修法：把 `:5` 改为「> 入口 = `train_arms.sh`（训练）+ `run.py`（评测）。」。

（**不判为缺陷、仅登记**：`README.md:97/107`「执行期追加登记（B3）」、`analysis.md:62`「（B3）」、`analysis.md:75`「（N3）」、`README.md:122`「（B1）」这些**复核轮次代号**留在事实文档里——按 `AGENTS.md` §7 内部代号可作一次括注坐标出现，且它们承载「该行来自哪一轮返工」的溯源作用，故只作观察；`change_log` 条目 3 与 4 时间倒序见 R3-N6 已并入 non-blocking 列表外，此处更正：**R3-N6** 为 `project.yaml:115` 与 `:118` 的 `change_log` 时间倒序错位，最小修法 = 交换两条顺序。）

---

## 6. 需要 supervisor / runner 跑的检查（我一律 not-run）

```bash
# 本轮唯一 blocking 的复现（纯读，不需要 GPU）
sed -n '19,20p;115,120p' axes/1-time/1.1.4.1-explicit-memory/project.yaml

# 我未能跑、请代跑的验收项
sha256sum axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/run.py \
          axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/config.json
python3 tools/summary.py --check
python3 -m pytest tests/ -q
python3 -m pytest axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/test_2638_frontier.py -q   # 期望 6 passed
grep -c "^def test_" axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/test_2638_frontier.py    # 期望 6（核 R3-N1）

# 我方残差的补证（可选）
git diff 12e5fb7 443cb44 --stat
git diff 443cb44 940cb23 --stat           # 核「940cb23 只动了 project.yaml + 本报告」
git show --stat --oneline 940cb23
```

---

## 7. 一句话总判词

**FAIL（窄范围）**：第 2 轮唯一的 blocking（**R2-B1**：`MODEL.md:41` 与 `results.json:1031` 对同一个 `run.py` 给不同 sha256）**已消解**（两处现同为 `2023f36d…`；`MODEL.md` 全部 16 个哈希逐条扫过，无第二处陈旧值；磁盘值只能采信 `TICKET.md` §8.5 的代跑记录），**R2-B2 与 N1/N2/N3/N4/N6/N7 逐条落地**，**数值零回归**（`rows`/`frontier`/`location` 与第 2 轮记录值、与 `run_log.txt` 逐位相同），**科学结论抽查通过**（`gap 0.125421`、`win17` 严格支配 `carry16_s0`、recovery 2.5%/1.2%、`被否证` 归因未越界）——**科学结论 PASS**；但交付物仍剩 **1 条 blocking（R3-B1：`project.yaml:19` 说「REVIEW-2 待派」而 `change_log:115-117` 说「已按 REVIEW-2 修」，同一 SSOT 内对同一事实两个互相打架的陈述，且必填字段 `conclusion_review` 未回填 REVIEW-2 判词）＋ 5 条 non-blocking**（`README.md:121` 单测「5 条」vs 实际 6 条、`README.md:95` 指向 `analysis §3` 的悬空指针、`README.md:65` §4a 曲线定义未同步 `win17/18`、单测钉子范围与落点、`analysis.md:4-5` 重复行 + `change_log` 时间倒序）。**全部为文字/指针，修改量约 4–6 行，不需要重跑 GPU、不需要重训、不需要重评**。

---