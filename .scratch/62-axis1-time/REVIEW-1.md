# REVIEW-1 · 轴 1 时间轴 Phase A 旗舰 WP1 独立复核（研究项 `1.1.4.1` / 实验 `2638-carry-frontier-13m`）

> **本版为 revision 前移后的更新版**：复核对象 = **`ab41bb9`（= 当前工作树，父级认定）**，上一版对象 `a679170`。前移增量为 `.scratch/62-axis1-time/TICKET.md` 所述「研究项 `project.yaml` 的 `status` 进行中→被否证 + `conclusion`/`failure_analysis`/`conclusion_review`/`gaps`/`experiments`/`WP1` 更新」；实验产物（`2638/`）未变。**增量复核不改变任何既有结论，也不消除任何既有缺陷**（见 §1.1），并因此**新增 2 条缺陷（B4、N11）**，全部由新加入的 `project.yaml` 判词字段产生。

---

## 0. 人话结论（先读这一段）

**这套数字是真的、算法是对的、结论的方向我复算后成立**：我把 `results.json` 里的 10 个臂逐数字重算了一遍（bpc 差、gap、gap-closing 比例、插值点、支配关系、下包络、字节数、`k_eff`），**没有发现任何一处算错**；四个复用臂的 bpc 与 `2634` 的产出**逐位相同**（1.628110 / 1.629093 / 1.624979 / 1.626640），说明评测口径确实一致；`full` 锚的 `keep_mask=None` 走的是 `is_causal=True`（`pathlm/depth_ar.py:1609`，单测 `tests/test_retention.py:166-173` 钉住与显式因果掩码逐位相等），token 数全臂结构上相同（96×16×255），所以「同任务 / 同位置集 / 同口径」这句站得住。**「小本子不值」这个判断，我不依赖作者事后追加的 win17 点也能独立复现**（用预登记点集 win16→win24 的弦插值到 k_eff=17.333 得 1.623401，仍优于 carry 的 1.624979）。`project.yaml:17`（`conclusion`）里的**每一个数字我都复算过，全部对得上**。

**但这份交付物不能原样入账**（`ab41bb9` 上依然如此，前移版本一条都没修）：

1. **判据③的判定线方向写反了**（`README.md:51`，`analysis.md:47` 抄同一句）：公式是 `bpc_interp − bpc_carry`（正 = carry 更好），判定线却写「`< 0` ⇒ 前沿改善」——按字面读，实测的 `−0.002259` 应判「改善」，与判读栏「被支配」自相矛盾。仪器注释（`results.json:543`）与 `SCHEMA.md` §17.3（Δ = 对照 − 实验，正=改善）站在判读那一边 ⇒ **结论对、契约文字错**。
2. **`analysis.md:68` 的「win18 的 0.006157」在整仓找不到出处**（`grep` 仅此一处）：复算是 1.624979 − 1.619953 = **0.005026**（carry16_s1 则为 0.006687）。
3. **契约与交付物对不上且未登记**：`README.md`（自称预登记契约）只有 **8 臂 / 3 臂训练 / `ARMS="24 32 128"` / 预算 40 min**（`README.md:89,99,100`），实际 **10 臂 / 训练 5 臂 / ≈48 min**；承担「严格支配」支点的 `win17` 未登记追加时点与理由。
4. **（本轮新增，来自 `ab41bb9`）** 研究项自身的 `milestone.acceptance` 第 1 条（「前沿图上每个预算点有**质量与可检索性两轴**数字」）**未达成**，`WP1.means` 也要求「与 R1 量化点、1.1.1 窗口点同图画质量-召回」，而交付的前沿图只有 **bpc 单轴 + 窗口点**；`WP1.status: "已完成"` 且 `acceptance` 写「前沿图齐备」是**与交付事实相反的措辞**（仅靠句中括号部分对冲）；`status: 被否证` 的 `failure_analysis` 只写了机制归因，**没有任何一处记录「交付维度未达标」**。

**一句话**：科学结论 PASS（判词 1–4 判「成立」，判词 1 需带噪声读法限定）；**交付物 FAIL（需一轮小修后重新复核）**，返工全部是文字与登记层面，**不需重跑 GPU**。

---

## 1. 复核对象 / 复核者 / 能力限制

- **复核对象 revision**：**`ab41bb9` = 当前工作树**（父级给定）。前移前的对象是 `a679170`。
- **复核者身份**：fresh 上下文、只读、未参与本线任何实验或文档；**未修改仓库任何文件**（本报告由 runtime 落盘到 `.scratch/62-axis1-time/REVIEW-1.md`）。
- **能力限制（如实登记，含前移后的新增限制）**：
  1. **无 shell**：`python3 tools/summary.py --check`、`python3 -m pytest tests/ -q`、`sha256sum`（10 个 checkpoint）、`git show --stat a679170` / `git diff a679170 ab41bb9` **全部未运行**。
  2. **前移后我连工作树状态也无法核验**：第一版复核时 `watchdog_diff` 还能回报「HEAD = `a679170888d3` 且工作树对 HEAD 无改动」；本轮再调 `watchdog_diff`（含按路径）返回 **`Reviewer-launch HEAD changed or is unavailable. Committed changes are unsupported`** ⇒ **`a679170 → ab41bb9` 的提交区间我完全没有看过**，`ab41bb9` 的提交内容、是否只动了 `project.yaml`、工作树是否干净，我只能**采信父级说明**，不能独立证实。
  3. 我能做的最强替代核验：**逐行重读** 2638 目录与 `project.yaml`，确认与本轮之前读到的内容一致（我逐个复核了所有引用行：`README.md:51/89/90/99/100`、`analysis.md:47/68`、`results.json:542-543`、`project.yaml:15-18,103,107-112` —— 文本相同；且本报告所有引用的行号在 `ab41bb9` 上仍指向同一句）。**「未变」在内容层成立，在字节层无法证明**（无哈希工具）。
  4. `README.md:10` 自称「§4 判据线预登记，未改一字」、父级称「判据线未改」——**我无法证伪或证实**（无 git）。但本轮 B1 与「是否被改过」无关：**无论是否被改过，该行文字与其自身公式、与全仓约定相反**，都是必须修的缺陷。
  5. sha256 一致性只能做**产物间交叉核对**：`config.json` / `MODEL.md` / `results.json.checkpoints` **三处 10/10 臂一致**；`results.json.code_hashes` ↔ `MODEL.md`「共享文件指纹」4 条一致；重算磁盘 `*.pt` 的哈希**做不到**。

### 1.1 revision 前移（`a679170` → `ab41bb9`）的增量复核

| 增量项 | 我的核对 | 结论 |
|---|---|---|
| `status: 进行中 → 被否证` | `project.yaml:103`；枚举合法（`docs/research-schema.md` §7），`被否证` ⇒ `conclusion`/`failure_analysis` 必填且非空（`:15,:17`） | ✓ 合规 |
| `conclusion` 的数字 | 逐项复算：208,896 / 212,992 / 1.624104 / 1.624979 / 1.626640 / 0.125421 / 2.5% / 1.2% / 3.2% / 22.5% / 91.5% / 1.5027 / +0.0158 / +0.0141 **全部与 `results.json`、`2634/results.json` 相符** | ✓ 数字无误 |
| `failure_analysis` 三类归因 | 设计/实现（主因）、结构（次因）、非 innovation 失败三分类齐全，并显式禁止推广到 delta 族 | ✓ 无越界 |
| `experiments` 新增 2638 与锚点 | `@a679170`（`project.yaml:35-36` 区）——实验产物未变，故锚点仍指向其最后变更 commit；**「未变」我只能靠内容重读确认** | ✓（限制见 §1.2） |
| `WP1` 更新 | `status: "已完成"`、`acceptance`「前沿图齐备」与交付事实（单轴、无量化点、无召回轴）冲突 ⇒ **B4（新）**；`milestone.acceptance` 第 1 条（两轴数字）未达成且无任何记录 ⇒ **B4/N11** | ✗ **新增缺陷** |
| `conclusion_review` 预引 `REVIEW-1.md` | 该文件在复核开始时不存在（`.scratch/62-axis1-time/` 仅 `TICKET.md`）——事实文档写入了尚未成立的事实 ⇒ **N7**（仍成立） | ✗ 仍成立 |
| 实验产物（2638/） | 引用的每一行文本与上一版逐字一致 | ✓ 未变（内容层） |
| 判决字段是否被复核内容影响 | `conclusion`/`failure_analysis` 未因复核而改（父级亦确认前移只是把判词落进 SSOT） | ✓ |

**前移的净效果：结论与缺陷清单都不变，只是把 `project.yaml` 的判词字段正式纳入被复核集合**，于是暴露出 B4/N11 两条与 WP/milestone 记账有关的缺陷。

---

## 2. 逐条判词（对任务给的 4 条结论；`ab41bb9` 上不变）

### 判词 1 · 「`carry16_s0` 被 `win17_s0` 严格支配；下包络全是纯窗口点」——**成立（点估计），但只在 σ_cell 读法下显著**
- 复算：`carry16_s0 = (212,992 B, 1.624979)`、`win17_s0 = (208,896 B, 1.624104)`，两轴皆更小 ⇒ 严格支配；代码定义 `run.py:91-108`（`bytes ≤ and bpc <`）与 `results.json:490-495` 的 `dominated_by` 一致。
- 下包络复算（按字节升序取记录最小值）= `win16_s0 → win17 → win18 → win24 → win32 → win128 → full`，与 `analysis.md:53` 逐项相同；`carry16` 两 seed 都不在其上，且 `carry16_s1` 还被 `carry16_s0` 支配（同字节、bpc 更差 0.001661）。
- **限定**：σ_cell 读法下 win17 领先 0.000875 ≈ 0.9σ_cell（`analysis.md:66` 自称「边缘」）；0.014 带读法下**不显著**（`analysis.md:69`）。严谨表述 = 「点估计支配；σ_cell 边缘、0.014 带下只能读作『与 win17 打平但更贵』」。`analysis.md:65-71` **两种读法都并列报出**（诚实），但标题句（`project.yaml:17`、`TICKET.md:145`、`analysis.md:37`）取了 σ_cell 读法而未就地附限定，且 `TICKET.md:92` 的 F1（用哪个带）仍写「待 Main 裁」⇒ N6。
- **我独立加固（作者文档中没有）**：**弃用 win17/win18**，仅用预登记点集（win16 1.62811 → win24 1.599856 的弦）插值到 k_eff=17.3333 ⇒ 1.62811 + (1.3333/8)(−0.028254) = **1.623401 > 1.624979**（差 −0.001578，比 win17 版更大）⇒ **「carry 不占优」不依赖事后追加的臂**，追加臂不构成选择性偏差。

### 判词 2 · gap-closing（2.5% / 1.2%；1 格 3.2%、8 格 22.5%、128 格 91.5%）——**成立**
- gap = 1.62811 − 1.502689 = **0.125421**；r(s0) = 0.003131/0.125421 = **0.02497→0.025**；r(s1) = 0.001470/0.125421 = **0.01172→0.0117**；win17 = **3.19%**、win18 = **6.50%**、win24 = **22.53%**、win128 = **91.52%**（全部与 `analysis.md:39-41` 一致）。
- 「全部收益 < 加宽 1 格窗口」成立（0.003131 < 0.004006）。

### 判词 3 · 必要条件成立——**成立**
- Δbpc：s0 **+0.003131**、s1 **+0.001470**（`results.json:568-579`）。
- `Δbpc_far`：`2634/results.json:458` **+0.015784**、`:682` **+0.014132**，配对 CI 排除 0；无通道臂 `Δbpc_far = 0.0` 且 `max_abs_logit_diff_on_eval_rows = 0.0`（`:346-353`、`:570-577`）⇒ 前提①也成立（`2634` 侧有单测 `tests/test_carry_realtext_13m.py:87-100`）。
- 附注（不改变判词）：`2634` 的 seed1 配对 CI = `[-0.000416, +0.005449]` 跨 0（`:9995-9997`），即「质量增益」本身在配对口径下 seed1 不显著；判据④原文只要求「>0」，故通过，但判读句应带这一限定。

### 判词 4 · 研究项判 `被否证`、归因「实现/设计为主因、结构次因、非 innovation 失败」——**成立、无越界**
- `README.md:16-20` 的可证伪形式把「bpc **不低于**插值（**在噪声带内或更高**）」也算否证 ⇒ 两种噪声读法都落在「被否证」分支，与 `analysis.md:47`、`project.yaml:17` 一致 ⇒ **不是过度声明**。
- `project.yaml:15` 三类区分齐全并禁止外推到 delta 族；`analysis.md:92-101`、`:118-124` 同口径；枚举/必填字段合规。
- 缺口：① `project.yaml:17` 的适用范围句漏 `T=256`（`README.md:53` 有）⇒ N7；② **交付维度未达标一事没有被 `failure_analysis` 或任何顶层字段记录**，只以 `gaps`「下一步最小实验」形式出现 ⇒ **B4/N11（新）**。

---

## 3. 十项必查的执行结果

### 3.1 判据线是否被改过
- 无法用 git 证实（§1）。**可核的是内部一致性**：`README.md:44-53` 五条判据 ↔ `analysis.md:44-49` 逐条对应，措辞未被放宽。
- **内部矛盾**：`README.md:51` ③ 公式 = `bpc_interp − bpc_carry`、判定线 = 「`< 0` ⇒ 前沿改善」；同一量的仪器注释 `results.json:543` 写「`< 0` ⇒ 被支配」；`analysis.md:47` 抄了错方向判定线却判「被支配」。`SCHEMA.md` §17.3 规定「Δ = 对照 − 实验（正 = 改善）」⇒ **判读方向正确、契约文字错误**（**B1**）。
- 另：`README.md` §5/§6/§7 的臂集与预算未同步到实际执行（8 vs 10 臂、3 vs 5 臂、40 vs ≈48 min）⇒ **B3**。

### 3.2 三道对账（§12）——逐数字复算
| 对账项 | 文档值 | 我的复算 | 结论 |
|---|---|---|---|
| gap（win16→full） | 0.125421 | 1.62811 − 1.502689 | ✓ |
| Δbpc s0 / s1 | 0.003131 / 0.001470 | 同 | ✓ |
| recovery ratio r | 0.025 / 0.0117 | 0.024965 / 0.011721 | ✓ |
| 插值点 k_eff 17.3333 | 1.62272（w=0.3333，win17↔win18） | 1.624104 + (1/3)(−0.004151) = 1.622720 | ✓ |
| `carry_minus_interp_window` | −0.002259 / −0.003920 | 1.62272 − 1.624979 / −1.626640 | ✓ |
| 支配集 | carry16_s0←win17_s0；carry16_s1←win17_s0、carry16_s0 | 同 | ✓ |
| 下包络 | 7 个纯窗口点 + full | 同 | ✓ |
| 字节 `12288×k` | 196608/208896/221184/294912/393216/1572864/3145728 | 逐项 | ✓ |
| carry 总字节 / k_eff | 212992；17.3333 | 同 | ✓ |
| gap-closing 3.2/6.5/22.5/91.5% | 同 | 3.19/6.50/22.53/91.52% | ✓ |
| σ_cell | 0.000983 / 0.001661 | \|1.62811−1.629093\|、\|1.624979−1.626640\| | ✓ |
| 「win18 的 0.006157」 | 0.006157 | **应为 0.005026**（carry16_s1 对 win18 为 0.006687） | ✗ **B2** |
- **任务对账**（§12.2）：同 enwik8 尾段、同 96×16 窗口序、同 391,680 token、各自训练可见性下评测，无「假上下文 + 真目标」式对照。
- **口径对账**（§12.3）：全部 12k 步快照 / matched-step / 同 token 预算；墙钟只作过程记录（`analysis.md:110`）。

### 3.3 对照臂可比性（§13）
- 窗口采样只依赖 `window_seed`（`tools/evalkit/retention.py:193-198`），与掩码无关；token 数结构上相同 `96×16×255`（`retention.py:201-205`），实测 `cost.tokens_scored = 391680` ⇒ 臂间可比。
- `full` 用 `keep_mask=None` 公平且必如此：`is_causal = attn_mask is None`（`pathlm/depth_ar.py:1609`；同义 `:489,:496`），「identity 因果掩码 ≡ 释放版路径」有单测（`tests/test_retention.py:166-173`），`tests/test_a9_window_longrange.py:55-56` 专门钉住「full 臂 = None 而非全 True 掩码」。
- 代价轴不对称（carry 多 264,296 参数、+1.95% 不进横轴）已被 `analysis.md:97-99` 自曝并列入 gaps ⇒ 归因未夸大。
- §9 复用：配置字段等值我抽查 `2634/train/rt_delta_s0/results.json`（steps 12000 / bs32 / lr 1e-3 / seed 0 / train_window 16 / anchors 0 / carry_write delta）与 `train_arms.sh` 窗口臂命令行**逐字段相同**；**eval 侧等值由「4 个复用臂 bpc 逐位复现」独立证明**；训练侧引擎版本无哈希可核 ⇒ N9。

### 3.4 口径一致性（`run.py` 的 `cost_axis`）
- `kv_bytes = 12×2×256×2×k = 12288k`（`run.py:185-189`）；`full` 取 `k=T=256` ⇒ 3,145,728 ✓；carry 另加 `8192×2 = 16384` ✓。
- **8192 floats 有据**：槽 `S ∈ R^{B,H,dh,dh}`（`pathlm/depth_ar.py:1339`）、缺省 `dh = d/n_heads = 32`（`:1496-1503`）⇒ 8×32×32 = 8192；`shared_slots=true` 覆盖 12 层由 `:1650` 的活性块确认。
- **假设未注明（N3）**：`bytes_per_elem=2` 对**状态**是假设——引擎内部一律升 fp32（`_carry_cast`，`depth_ar.py:1317-1321`）。按 fp32 记则 k_eff=18.67、同预算窗口插值 ≈1.61772、carry 差 −0.00726（**更差**）⇒ 方向不变、结论更强。

### 3.5 干预活性（§10）
- **carry 臂**：`read_share` 末次训练日志 0.4165/0.4183（`2634/results.json:10230-10233`、`:10251-10253`）、评测期 0.3017/0.3038、`state_fro` 5.63/5.66、`β_mean` 0.435、`rel_dev_vs_channel_off` 0.468/0.459（`:10292-10339`）；`analysis.md:83-85` 引用无误（0.416 ↔ 0.416451）。
- **window 臂**：`train/win17/train_log.jsonl:1` `window_mask={train_window:17, n_visible_pairs:4216, dropped_frac:0.871839, alive:true, max_abs_logit_diff_masked_vs_open:2.009555}`；`win128` 同形（24640 / 0.250973 / alive:true / 0.546616）⇒ 掩码活性成立，且与 `selfchecks` 的 `visible_pairs/expected` 相符。
- `run.py:173-176` 的 `assert s == exp` **可失败**（掩码函数差一格即红），但**证明不了掩码被前向使用**；`nondegenerate`（`:175`）**只算不断言**；评测期无「掩码 vs 放开」logits 差 ⇒ N1/N2。`SCHEMA.md` §10 要求的 `intervention` 块在 `results.json` 中**缺失**。

### 3.6 证据可回溯
- `config.json` / `MODEL.md` / `results.json.checkpoints` 三处 10/10 臂 sha256 **逐字符一致**；`run_log.txt:2-11` 显示逐臂断言通过（`sha256_verified: true`）。
- 重算 sha256 不可能（§1）⇒ 待 runner（§6）。
- `README.md:90` 引用的 `smoke/steps300/win24/results.json` **不存在**，实际是 `smoke/steps300/smk24/results.json`（`wall_minutes=0.22`、`peak_gpu_mem_mb=2015.1`，数字与文中一致）⇒ N4。

### 3.7 `被否证` 是否过度声明——**假设层面否，交付层面见 B4**
见判词 4：机制结论的界定清楚、未偷换「机制不可能」。但**交付维度**（两轴、量化点、业界对齐）未达标一事没有顶层记录（B4）。

### 3.8 噪声带读法的诚实性
- 双列报出、两种读法并列、不自选（`analysis.md:61-71`），与 `TICKET.md:92` 的 F1 上报一致 ⇒ 诚实。
- 需改：① 错值 0.006157（B2）；② 区间「0.0009–0.0025」**不含** carry16_s1 的插值差 0.003920（仍 < 0.014，结论不变）⇒ N5；③ 标题句未附限定 ⇒ N6。

### 3.9 回归
- **未运行**（无 shell）：`summary.py --check`、`pytest tests/ -q` 均 **not-run** ⇒ 能力限制，非缺陷；待 runner。
- 静态核验：`run.py` 只 `import` 共享模块（`run.py:31-35`），不改共享文件；`README.md:30-32`、`MODEL.md`「未改任何共享文件；只读引用」+ 共享文件 sha 落盘 ⇒ 在可读证据层面成立，**但我无法用 git 证实**。
- 另：题面提到的 `results.json.criteria` 在 `2638/results.json` **不存在**（该键属 `2634`）；2638 对应物是 `location` + `frontier` + `selfchecks`。不妨碍复核，但说明判据结论未落在显式 `criteria` 块（见 N1）。

### 3.10 方法学十三类缺陷（§24）逐条对照
- **#2 不可达判据（符号/方向写反）——命中**：`README.md:51` + `analysis.md:47`（B1）。
- **#3 引数无出处——命中**：`analysis.md:68` 的 0.006157（B2）。
- **#13 局部修复 / 覆盖范围小于声称——命中（弱形态）**：契约只更新判据线，未同步臂集/预算/`ARMS` 默认值/引用产物路径（B3、N4）；`WP1.acceptance`「前沿图齐备」与交付事实不符（B4）。
- **#5 等价范围小于声称——接近**：§9 复用只把 eval 侧证到位（N9）。
- **#1 恒真断言——部分接近未命中**：`run.py:176` 断言比较掩码函数 vs 解析式（会红，但测不到「掩码未被使用」），`nondegenerate` 未断言（N2）。
- **#6 参考线口径写错——未命中**（`README.md:27-28`/`analysis.md:39` 口径写清）。
- **#4/#8/#9/#10/#11/#12——未命中**：#4 已报不利分量并解释（`analysis.md:39-41`）；无恒等式当因果的分解；无合取判据被读成半个成功；无对账脚本空转（对账是我人工逐项复算）；`run.py`/`train_arms.sh` 与 `results.json` 键名一致（逐键核对过 `location/frontier/gap_closing/effects_by_seed/sigma_within_config/cost/code_hashes`）。

---

## 4. 必须修（blocking；**均不改变实测结论**，但按 `SCHEMA.md` §14.1/§24 与 `docs/research-schema.md` §5/§6 不得原样入账）

**B1 · 判据③的判定线方向与自身公式/全仓约定相反（P1）**
- 位置：`README.md:51`（契约）、`analysis.md:47`（判读表抄同一句）。
- 违反：`SCHEMA.md` §17.3（Δ = 对照 − 实验，正 = 改善，契约内一次定义全篇一致）、§24 第 2 类、§14.1。
- 证据：`README.md:51` 公式 `bpc_interp − bpc_carry` + 「`< 0` ⇒ 前沿改善」；`results.json:543` 对同一量写「`< 0` ⇒ 被支配」；`analysis.md:47` 判定线栏抄「<0 ⇒ 改善」而结论栏写「被支配」；`README.md:18` 命题形式（「不低于插值（含带内）⇒ 被否证」）与判读方向一致。
- 最小修法：`README.md:51` 判定线改为「`> 0`（噪声带外）⇒ 前沿改善；`≈0`（带内）⇒ 不可排序；`< 0` ⇒ 被同预算纯窗口支配」，同步 `analysis.md:47`；或把该量名字与公式改为 `carry − interp`。**数字与结论不动。**
- 复现：读 `README.md:51` 与 `results.json:542-543`，把 −0.002259 代入字面判定线会得到「前沿改善」。

**B2 · `analysis.md:68` 的「0.006157」无出处且算错（P1）**
- 位置：`analysis.md:68`。违反：§24 第 3 类、§14.5。
- 证据：全仓 `grep 0\.006157` 只命中该行；复算 \|carry16_s0 − win18_s0\| = 1.624979 − 1.619953 = **0.005026**（s1 对 win18 = 0.006687）。
- 最小修法：改为 `0.005026`（注明是 carry16_s0 对 win18_s0）。

**B3 · 契约与交付物不一致、追加臂未登记、GPU 预算被超过（P1）**
- 位置：`README.md:26-35`（§5 表 8 臂）、`:89`（40 min，「含 3 臂训练」）、`:99`（`ARMS="24 32 128"`）、`:100`（「8 臂 bpc」）；实际 `results.json`（10 臂）、`TICKET.md:137`（训练 5 臂，14:44:31–15:31:46）、`analysis.md:107,130`（「3+2 臂」）。
- 违反：§8（契约唯一预算数字；预算/判据冲突应停下裁决）、§17.8（臂数 = 干预档数 × 预算估算）、§14.1/§14.5。
- 证据：声明 40 min vs 实测 ≈47 min 训练（5×8.4–8.51 min + 4×61 s 间隔）+ 0.3 min 评测 + 0.22 min 冒烟 ≈ **48 min**；`win17` 承担判词支点却未登记追加时点与理由。
- 最小修法：① `README.md` §5 补 `win17/18` 两行 + 一行「执行期追加 2 臂（时点/理由）」，§6 预算改实测值或注明超支原因，§7「8 臂」改「10 臂」；② `analysis.md` 补一句「预登记点集（无 win17/18）下 win16→win24 弦插值到 k_eff=17.333 得 1.623401，仍优于 carry16_s0 的 1.624979 ⇒ 主结论不依赖追加臂」。
- 我独立确认：追加臂**不是挑选有利结果**（追加时 win17 的 bpc 未知；且预登记点集下同样判负）⇒ 这是**登记完整性**问题。

**B4 ·（本轮新增）交付维度未达标：`WP1` 状态/验收措辞与事实不符，且 `failure_analysis` 未记录交付裁剪（P1）**
- 位置：`project.yaml` 的 `WP1.means`（「与 R1 量化点、1.1.1 窗口点同图画**质量-召回**」）、`WP1.acceptance`（「**前沿图齐备**；…（本轮窗口点已齐备、量化点未叠加，见 gaps；结论=明确被支配）」）、`WP1.status: "已完成"`、`milestone.acceptance` 第 1 条（「前沿图上每个预算点有**质量与可检索性两轴**数字」）、`conclusion`/`failure_analysis`（`project.yaml:15,17`）。
- 违反：交付与声称不符——交付的前沿图是 **bpc 单轴 + 窗口点 + full 锚**（`analysis.md:51-57`、`run.py:178-193` 只算 `bpc`，横轴只有字节），**没有召回/可检索性轴，也没有量化点**；`milestone.acceptance` 第 1 条因此**未达成**，而顶层没有任何字段记录「交付未达标」（`gaps` 只把它写成「下一步最小实验」）；`WP1.acceptance` 的「齐备」与该句括号内的自认「量化点未叠加」自相矛盾。
- `docs/research-schema.md` §5（验收不达标 ⇒ 判研究失败并填 `failure_analysis`）与 §6/§7（WP 状态只回答「做完没有」，交付范围写在正文）相关。
- 最小修法（一两句）：① `WP1.acceptance` 改为陈述交付事实，如「前沿图（窗口点 + full 锚，质量轴）齐备；召回轴与量化点未叠加（见 gaps）；通道点被同预算窗口明确支配」；② 在 `conclusion` 或 `gaps` 首条显式区分两件事：「研究项**假设**被本地证据否证（结论）」与「研究项 **milestone 的交付范围**被裁剪（量化点/召回轴/业界对齐未做 ⇒ 交付未达标）」，并说明裁剪不改变否证结论。
- 复现：对比 `project.yaml` 的 `WP1.means`/`milestone.acceptance` 与 `analysis.md:51-57` ＋ `gaps` 第 3/5/6 条。

---

## 5. 不需要阻塞（P2，建议随同一次小修处理）

- **N1 · `results.json` 缺 §10 干预活性块**（§10 要求落 `intervention:` 或等价活性量并在 `analysis.md` 列一行）：窗口臂活性停在 `train/*/train_log.jsonl`，carry 活性靠引 `2634`。修法：`run.py` 增写一个含训练日志活性 + `2634` liveness 引用的块，`analysis.md` §4 加一行活性表。
- **N2 · 自检强度**：断言 `nondegenerate`，并对至少一个臂在评测期记录「掩码 vs 放开」的 logits 差（可与 N1 合并）。
- **N3 · 状态精度假设未注明**（fp16 记账 vs 引擎 fp32；按 fp32 则 k_eff=18.67、carry 更差 0.0073）。
- **N4 · 引用路径不存在**：`README.md:90` 的 `smoke/steps300/win24/…` → `smoke/steps300/smk24/…`。
- **N5 · 区间写法不完整**：`analysis.md:69`「0.0009–0.0025」→ 写成「0.0009 / 0.0023 / 0.0039」。
- **N6 · 标题句取边缘读法**：`analysis.md:37`、`project.yaml:17`、`TICKET.md:145` 的「严格支配」加「（点估计支配；σ_cell 边缘、0.014 带不显著）」，并交代 F1（`TICKET.md:92`）的裁决状态。
- **N7 · `project.yaml` 两处小缺**：① `conclusion`（`:17`）漏 `T=256`（`README.md:53` 有）；② `conclusion_review`（`:18`）预引尚不存在的 `.scratch/62-axis1-time/REVIEW-1.md`（事实文档写入未成立事实）⇒ 复核落盘后回填判词。
- **N8 · 本格无非自包含单测**：仓库惯例（`tests/test_carry_realtext_13m.py`、`test_carry_seed2_13m.py`、`test_carry_ragged_*`、`test_carry_x_depth_13m.py`…）每格一测；`2638/run.py`（`frontier`/`interp_window_at`/自检）无测试。修法：加 `tests/test_carry_frontier_13m.py`（tiny 模型 + 合成 rows，顺便钉住 B1 的符号方向）。
- **N9 · §9 复用等值的训练侧缺口**：补记 `2634` 训练时的共享文件哈希，或明示复用范围 = eval-only。
- **N10 · 能力限制需 runner 补跑**（不作为缺陷）：见 §6。

---

## 6. 需要 supervisor 跑的检查（我无 shell，一律 not-run）

```bash
git show --stat ab41bb9                      # 前移增量到底改了什么（我完全没看过该区间）
git diff a679170 ab41bb9                     # 是否只有 project.yaml 判词字段
git diff --stat ab41bb9                      # 工作树是否干净（我连这都核不了）
git log -p --follow -- axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/README.md
                                             # 核实 README §4「写下后不得改」与 §5/§6 是否被事后改过
python3 tools/summary.py --check             # 研究树 SSOT 必过
python3 -m pytest tests/ -q                  # 共享底座回归（§14.7）
sha256sum axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/train/*/model.pt \
   axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m/train/*/model.pt \
   axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt
                                             # 与 MODEL.md / config.json / results.json.checkpoints 对表
```

---

## 7. 一句话总判词

**FAIL（结论成立、交付物需一轮小修；`ab41bb9` 相对于 `a679170` 的增量未修任何一条，并新增 B4）**：判词 1–4 的核心事实与算术我独立复算后成立（并额外证明主结论不依赖事后追加的 `win17`），但按 `SCHEMA.md` §24 命中「判据方向写反」（`README.md:51` / `analysis.md:47`）与「引数无出处」（`analysis.md:68`），契约与交付物（臂集/预算/引用路径）未同步、追加臂未登记，且 `ab41bb9` 新加入的判词字段把「交付范围被裁剪」写成了「前沿图齐备/已完成」而未在顶层记录交付未达标 ⇒ **修完 B1–B4（外加 N1/N4/N7 一类顺带项）即可判 PASS，全部为文字与登记层面的修复，不需重跑 GPU。**

## 8. 残差风险（我未能消除的）
1. **`a679170 → ab41bb9` 的提交区间我完全没看过**（`watchdog_diff` 在 HEAD 前移后拒绝服务 ⇒ 「Reviewer-launch HEAD changed or is unavailable」），且**当前工作树是否干净也无法确认**；本报告对 `ab41bb9` 的全部陈述依赖「父级说明 + 逐行重读内容一致」。
2. **无法用 git 证实判据线未被事后修改**；但 B1 与该问题无关（文字与其自身公式矛盾）。
3. **无法重算 10 个 checkpoint 的 sha256**（只能确认三处产物自一致 + `run.py` 评测时断言通过）。
4. **训练侧跨实验引擎等值未证**（`2634` 无 `code_hashes`）；eval 侧 bpc 逐位复现、窗口臂不经 carry 代码路径，风险小。
5. **规模外推风险（文档内已声明）**：横轴把与 T 无关的固定状态折算为「每位置等效字节」，故「1.33 个位置」是 **T=256 专用**口径；`README.md:53`/`analysis.md:47` 的禁外推条款覆盖了它，但 `project.yaml:17` 的适用范围句漏了 `T=256`（N7）——读者可能误读为「长上下文里显式状态也不值」。
6. **状态精度假设（fp16）**：按 fp32 则成本轴数值变化、carry 更差（方向不变，N3）。