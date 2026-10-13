# REVIEW-2 · 轴 1 时间轴 Phase A 旗舰 WP1 独立复核（研究项 `1.1.4.1` / 实验 `2638-carry-frontier-13m`）

## 0. 人话结论（先读这一段）

**第 1 轮的四条 blocking，我逐条独立复算后认为「真修了」**：判据③ 的判定线方向现在四处一致（契约 `README.md:55`、判读表 `analysis.md:51`、仪器注释 `run.py:220-224`、单测 `test_2638_frontier.py:59-70`），并与 `SCHEMA.md:268-270`（§17 第 3 条：`Δ = bpc_对照 − bpc_实验`，正 = 改善）同向；把实测 `carry16_s0 = 1.624979`、插值 `1.62272` 代进去，四处都得到 `−0.002259 < 0 ⇒ 被同预算纯窗口支配`，与判读栏「被支配」不再打架。`analysis.md:75` 的引数已从无出处的 `0.006157` 改成有出处的 `0.005026`（= 1.624979 − 1.619953，我复算一致）。`README.md:78-103` 的臂表已补齐 10 臂、`win17/18` 有追加时点与理由（15:13:34–15:31:46，与 `train_arms_driver2.log` 逐行吻合），`§6` 把预算超支记为 ≈48 min，`§7` 的「8 臂」已改「10 臂」。`project.yaml` 已把「**假设被否证**」与「**milestone 交付范围被裁剪（交付未达标）**」拆开记进 `conclusion`（:17）、`failure_analysis` 第④条（:16）、`gaps` 首条（:51）与 `WP1.acceptance`（:79）。**主结论我也重新独立复算了一遍，没有一处算错**：主判据（carry16 被 win17 支配）、gap-closing（2.5% / 1.2% vs 1 格 3.2%）、必要条件（Δbpc 两 seed > 0；2634 的 `Δbpc_far` +0.015784/+0.014132）、下包络、支配集、`k_eff`、字节数全部对平；我另外用**预登记点集**（win16 1.628110 → win24 1.599856 的弦）插值到 `k_eff=17.3333` 得 **1.623401 > 1.624979**，独立复现了「主结论不依赖事后追加的 win17/win18」。

**但这一轮留下/新引入了一批「记账型」不一致，按本仓 `SCHEMA.md` §24「命中任一条即不得入账」的标准，还不能原样入账**：

1. **同一份交付里对 `run.py` 给出两个互相矛盾的 sha256**：`MODEL.md:41` 写 `eff58d515…`，而 `results.json:1031`（`run.py` 运行时自算的 `code_hashes.run.py`）写 `2023f36d…`。两者不可能同时描述同一个文件；`run.py` 正是本轮被改、被重跑的文件，而 `MODEL.md` 不在本轮处置清单里 ⇒ 极可能就是「改了 `run.py`、没扫相邻面（指纹）」的 §24 第 13 类（形态 2）。**已改为 blocking（R2-B1）**，因为它决定「落盘 `results.json` 到底是不是交付的 `run.py` 跑出来的」。
2. **`README.md:120` 的训练清单仍是老样子**（`ARMS="24 32 128"` ⇒ 只列 `train/win{24,32,128}/`），而交付实际训练了 5 臂（§5 表自己写着 `win17/18` 是本实验训练）；照 `§7` 字面复现会只训 3 臂，随后 `run.py` 因缺 `win17/18` 权重而失败。正确命令只在 `analysis.md:145` 与 `README.md:98-103` 里。
3. **`project.yaml:47-48` 的实验证据锚点仍钉在 `a679170`**，而 `analysis.md` / `results.json` 都在本轮被改写 ⇒ SSOT 的证据指针指向的正是**修好之前**、仍含 `0.006157` 错值、且没有 `liveness` 块的那一版。
4. 轻微：两处事实文档里留了「原本…已改为…」的**变更叙事**（`README.md:45-50` 标题还写着「写下后不得改」，正文却记录了「已改」；`analysis.md:4-6` 同），与本仓「事实文档只写当前状态、流水进 `.scratch`」的纪律相悖（`AGENTS.md` §2.1）；新增单测放在实验目录而**不在 `tests/`**，因此 `python3 -m pytest tests/ -q` 这个仓库级门禁**跑不到它**；该单测最后一条断言是子串检查（`test_2638_frontier.py:70`），钉不住它声称要钉的方向。

**一句话**：**科学结论 PASS、B1–B4 修复成立**；**交付物 FAIL（窄范围）**——剩余全部是文字 / 指纹 / 指针层面，**不需要重跑 GPU**（唯一可能需重跑的触发条件见 R2-B1 的分支）。

---

## 1. 复核对象 / 复核者身份 / 能力限制

- **复核对象 revision**：**`12e5fb7`**（复核目标）。我可用 `watchdog_diff` 确认：**`No working-tree changes against reviewer-launch HEAD 12e5fb7310c0`** ⇒ 工作树对该 HEAD **干净**，且本轮对象确为 `12e5fb7`。父提交 `ab41bb9`、实验产物原提交 `a679170` 由父级给定。
- **复核者身份**：fresh 上下文（未参与本线任何实验 / 文档，**不是**第 1 轮复核者，**不是**本项 writer）。**未修改仓库任何其它文件**；本报告由 runtime 落盘到 `.scratch/62-axis1-time/REVIEW-2.md`。
- **能力限制（如实登记）**：
  1. **无 shell**：`python3 tools/summary.py --check`、`python3 -m pytest tests/ -q`、`python3 -m pytest …/test_2638_frontier.py -q`、`sha256sum`、`git diff ab41bb9 12e5fb7 --stat`、`git show --stat 12e5fb7` **全部 not-run**（见 §6）。因此 `README.md` 声称的「未改任何共享文件」、`TICKET.md` §8.3 的「834 passed / 5 passed / summary OK」我只能**采信**，不能独立证实。
  2. **提交区间内容仍看不到**：`watchdog_diff` 只回报「对 launch HEAD 无未提交改动」，**不包含已提交区间** ⇒ `ab41bb9 → 12e5fb7` 到底改了哪些文件，我**没有独立看过**；我能做的是**逐行重读** 2638 目录与 `project.yaml`（本报告全部行号取自当前工作树）。
  3. **重算磁盘哈希做不到** ⇒ 第 5 项「重跑 `run.py` 数值是否与落盘 `results.json` 一致」我用**两条替代证据**代替（见 §2.5），并如实标注为替代证据。
  4. **无法判定 R2-B1 中哪个哈希是当前值**（需要 `sha256sum`）；但我能**确定两者不能同时为真**。

---

## 2. 逐条判词（对任务 8 项）

### 2.1 B1 是否真修且方向自洽 —— **成立（4/4 处一致；单测的「钉子」偏弱）**

| 位置 | 现文本 | 与自身公式 / `SCHEMA.md:268-270` 一致？ |
|---|---|---|
| `README.md:55`（契约判据③） | 量 `carry_minus_interp_window = bpc_interp − bpc_carry`（**正 = carry 更好**，注明与 §17.3 一致）；判定线 `>0`（带外）⇒ 前沿改善；`≈0` ⇒ 不可排序；`<0` ⇒ **被同预算纯窗口支配** | ✓ |
| `analysis.md:51`（判读表③） | 同公式、同判定线 | ✓ |
| `run.py:220-224`（`location[*].note`） | 「= `bpc_interp − bpc_carry`（正 = carry 更好）：< 0 ⇒ …被同预算纯窗口支配；> 0 ⇒ 前沿改善；≈0 ⇒ 不可排序；null ⇒ 不外推」 | ✓ |
| `test_2638_frontier.py:59-70` | `rel = bpc_interp − bpc_carry`；`rel_dom<0`、`rel_win>0` 断言 | ⚠ 方向一致，但见下 |

**逐处代入验算**（`carry16_s0 = 1.624979`，`bpc_interp = 1.62272`）：
- `1.62272 − 1.624979 = −0.002259` ⇒ 负 ⇒ 按四处判定线**都**判「被支配」；与 `analysis.md:51` 结论栏、`results.json:967-981` 的 `location.carry16_s0.carry_minus_interp_window = -0.002259`、`README.md:16-20` 的可证伪形式（「不低于插值（含带内）⇒ 被否证」）**同向**。**B1 成立。**
- 顺带复核 `carry16_s1`：`1.62272 − 1.626640 = −0.003920` ✓（`results.json:982-996` 一致）。

**B1 的残留瑕疵（非阻断）**：`test_2638_frontier.py:70` 的 `assert "bpc_interp" in inspect.getsource(R.main)` 只检查 `main()` 源码里**出现过 `bpc_interp` 这个子串**——无论方向写反与否都会通过，**钉不住它声称要钉的符号约定**（§24 第 1 类的弱形态）。真正钉住方向的写法是断言 `main` 源码含 `"bpc_interp − bpc_carry"` 与 `"< 0 ⇒"`/`"被…支配"`（或把 `rel` 的计算抽成可调用函数后直接断言）。见 R2-N5。

### 2.2 B2 是否真修 —— **成立；另抽 8 个数字，仅 1 个口径标注不足**

- `analysis.md:75` 现为「win18 对 carry16_s0 的差 **0.005026**」；复算 `1.624979 − 1.619953 = 0.005026` ✓；全仓 `grep 0\.006157` **已无命中**（只剩 `REVIEW-1.md` 与 `TICKET.md` §8.1/§8.2 的历史叙述）。**B2 成立。**
- 抽查出处（各 1 个数字 ⇒ 上游字段）：`1.624979`→`results.json:869`；`1.62272`→`:971`；`1.623401`→弦插值复算 ✓（`analysis.md:64-66`）；`208,896`→`:849`；`1.5027(1.502689)`→`:815`；`16.9`（「质量只等于窗口 16.9」，`analysis.md:33`）→ 用 win16→win24 弦斜率复算得 16.89 ✓；`0.125421`→`results.json:999`；`627.7 MB`→`:1023`。**均能找到家。**
- **唯一不足（非阻断，R2-N4）**：`analysis.md:52` 的 ④ 栏写「Δbpc = +0.003131 / **+0.001470**（2 seed 皆 >0）」。`0.001470` 有出处（`results.json:983` 的 `location.carry16_s1.delta_bpc_vs_win16_s0`、`run_log.txt:27`），但它是**跨 seed** 口径（carry16_s1 − win16_**s0**）；同一产物 `results.json:1012` 的 `effects_by_seed["1"].delta_bpc = 0.002453`（同 seed 口径，与 `2634/analysis.md:129` 的「seed1 +0.002453」一致）。两个数都能过判据（>0），但对「2 seed」的措辞会让人以为是同 seed 配对；建议在 ④ 栏注明「seed1 相对 `win16_s0`（跨 seed）」或改引 0.002453 并同步 1.2%→2.0%。

### 2.3 B3 是否真修 —— **主体成立；`§7` 训练清单漏同步（R2-B2）**

- **臂表**：`README.md:80-90` 10 行 ↔ `results.json:793-924` 10 臂、参数/字节/`k_eff` 逐行一致（`full` 13,536,768 / 3,145,728 / 256；`carry16` 13,801,064 / 212,992 / 17.33 ✓）；↔ `analysis.md` 的「3+2 臂」（`:122`）与训练目录 `train/win{17,18,24,32,128}/` ✓。
- **追加登记**：`README.md:98-103` 给出时点与理由；`train_arms_driver.log`（win24/32/128，14:44:31–15:12:17）与 `train_arms_driver2.log`（win17/18，15:13:34–15:31:46）**逐行吻合** §5 的 `2026-10-13T15:13:34–15:31:46` ✓。**不构成挑结果**：我把预登记点集（无 win17/18）`win16 1.628110 → win24 1.599856` 的弦插值到 `k_eff=17.3333` 独立复算得 **1.623401 > 1.624979**（差 −0.001578，比 win17 版更大）✓ ⇒ 主结论不依赖追加臂。
- **预算**：`README.md:107-111` 记 40 min 预算、实际 ≈48 min 与超支原因；我从两个 driver log 复算训练跨度 = 14:44:31→15:31:46 = **47.25 min**，+0.32 min 评测 + 0.22 min 冒烟 ≈ **48 min** ✓。`README.md:56` 的「砍什么」顺序（三臂一起砍步数 > 砍 win128 > 砍评测批数）✓。
- **缺口（R2-B2）**：`README.md:120` 仍写「训练：`bash train_arms.sh`（`ARMS="24 32 128"`）⇒ `train/win{24,32,128}/…`」。该行**字面为真**（`train_arms.sh:10` 的默认值确为 `24 32 128`），但作为**契约的脚本清单**它漏了 2 个新增臂：照 `§7` 复现 ⇒ 缺 `win17/18` 权重 ⇒ `run.py` 直接失败。`README.md:121` 的「10 臂」已改对，唯独这行没改。

### 2.4 B4 是否真修 —— **成立（四处齐备，未见实质自相矛盾）**

| B4 要求 | 落点 | 核查 |
|---|---|---|
| 拆分「假设被否证」/「milestone 交付范围被裁剪」 | `conclusion`（`project.yaml:17`）「**两件不同的事（B4 拆分）**：（i）假设被否证…（ii）milestone 的交付范围被裁剪…两事互不推翻」 | ✓ |
| 交付未达标进归因 | `failure_analysis` ④（`:16`）「交付维度未达标（交付裁剪，不是机制归因）…milestone 验收第 1 条与 `WP1.means` 的『与量化点同图』未达成…已登记于 `gaps` 首条」 | ✓ |
| `gaps` 首条 | `:51`「**交付范围裁剪（milestone 验收未达标，B4）**…」 | ✓ |
| `WP1.acceptance` 陈述交付事实 | `:79`「前沿图（窗口点 + full 锚、质量轴 clean bpc）齐备；**召回轴与量化点未叠加**（见 gaps 首条）；通道点被同预算纯窗口明确支配」 | ✓ 已不再自称「齐备」于两轴 |

- 与交付事实核对：`analysis.md:20-33` 的表只有 bpc 单轴 + 窗口点 + full 锚，`run.py:178-193` 的横轴只有字节 ⇒ `WP1.acceptance` 的新措辞与事实相符 ✓。
- **残留张力（P2，R2-N3）**：`WP1.means`（`:78`）仍写「与 R1 量化点、1.1.1 窗口点同图画**质量-召回**」而 `WP1.status: "已完成"`（`:82`）；按 `docs/research-schema.md:130`「状态枚举只回答『做完没有』；『做到多少』写在 `conclusion`/`gaps` 正文」，此写法**可辩护**（已交付的那部分确实做完了，缺口已在 gaps 首条），我**不判为缺陷**，但建议在 `means` 后加半句「（本轮只交付质量轴，召回/量化点见 gaps）」以消除字面张力。

### 2.5 round-1 是否引入新缺陷 —— **新引入 1 条（R2-B1）；数值无回归证据；N1/N2 成立**

- **`liveness` 块（N1）**：`results.json:356-791` 逐臂非空——窗口/carry 臂都有 `mask`（可见对数 / `visible_frac` / `alive`）与训练日志 `train_window_mask`；`carry16_s0/s1` 带 `carry_liveness_training`（`read_share 0.41645 / 0.418283`、`state_fro 4.2188 / 3.8618`、`β_mean 0.5758 / 0.5689`、`rel_dev_vs_channel_off 0.4678 / 0.4592`）；另有 `_mask_probe`（`:779-788`）给出评测期反事实 `win17_masked_vs_open = 13.88488, alive: true`。`analysis.md:86-89` 有对应一行 ✓。**N1 成立**。（可见对数复算：`win17 4216 / win18 4455 / win24 5868 / win32 7696 / win128 24640 / win16·carry16 3976`，`Σ min(i+1,K)` 逐档对平 ✓。）
- **N2**：`run.py:179` 已是硬断言 `assert s < causal`（掩码在 T=256 上不得为恒等算子）⇒ **可红**（只要有人把 window 设到 ≥ T 即触发）；`run.py:268-273` 的 `window_mask_probe(win17)` 提供了「掩码真的改了前向」的评测期证据（旧版只有训练期日志）。**N2 成立**（增强项：`assert s < causal` 只覆盖「非恒等」，仍覆盖不到「掩码是否被前向使用」——该项由 `_mask_probe` 补上 ✓）。
- **数值回归（替代证据，非重跑）**：
  - `run_log.txt:1-30` 与 `run_log.jsonl:1-24`（同一次运行，`t=17.84s`）的 10 臂 bpc / `k_eff` / `total_bytes` / 下包络 / `dominated_by` / `Δbpc` / 插值**与 `results.json:793-996` 逐字相同**；而 `results.json:1022` 的 `wall_minutes_total_eval = 0.3`（= 17.8 s）与 `README.md:110` 所引的旧值 `0.32 min`（`analysis.md:125` / `TICKET.md` §5.2）**不同** ⇒ 这组日志属于**本轮重跑**，即落盘 `results.json` 与本轮 `run.py` 的运行结果一致。
  - 4 个复用臂 bpc（1.628110 / 1.629093 / 1.624979 / 1.626640）与 **2634 自己产的**值逐位相同（`2634/run_log.txt:28`、`2634/results.json:9994` 区）⇒ 评测口径可复现（这条独立于 2638 的日志）。
  - **注意**：以上是**替代证据**，不等于我重跑了 `run.py`（无 shell）。
- **新引入缺陷（R2-B1，见 §3）**：`MODEL.md:41` 的 `run.py` 指纹 `eff58d515…` ≠ `results.json:1031` 的 `2023f36d…`。`run.py` 正是本轮为 N1/N2 而改并重跑的文件，`MODEL.md` 不在 `TICKET.md` §8.2 的处置表里（`TICKET.md:180-195`）⇒ 与本轮修复直接相关。
- 另：`README.md:108` 的「5×8.4–8.51 min」与逐臂实测 8.4（win128）/ 8.47（win18）/ 8.51（win17、win32）/ **8.53（win24）** 差 0.02 min（`train/*/results.json:766`）——纯舍入级，随 §6 一次顺手改即可（R2-N7）。

### 2.6 回归项能力登记 —— **not-run（能力限制，非缺陷）**

见 §6。我能做的静态核验：`run.py` 只 `import` 共享模块（`run.py:31-35`，含既有的 `window_mask_probe`，`pathlm/depth_ar.py:1155`，无需新改共享代码）；`1f8f9eb4…`（`pathlm/depth_ar.py`）在 `.scratch/61-axis2-depth/REVIEW-2.md:140` 被独立记为「`483b181` 版本」，说明该指纹是一个真实的仓库版本；`MODEL.md:37-42` 与本目录 `results.json:1027-1031` 的 4 条共享文件哈希**逐字一致**（这 4 条是本轮未改共享文件的**间接**佐证——若本轮改过 `depth_ar.py`，两边不会同时取到同一值）。10 个 checkpoint 的 sha256 在 `config.json`、`MODEL.md:5-14`、`results.json.checkpoints`（`:168-298`，全 `sha256_verified: true`）三处一致 ✓（磁盘重算不行）。

### 2.7 `SCHEMA.md` §24 十三类逐条对照

| # | 判 | 证据 |
|---|---|---|
| 1 恒真断言 | **弱命中（非阻断）** | `test_2638_frontier.py:70` 的子串断言恒过（钉不住方向）；`run.py:177/179` 的两条断言本身可红 ✓ |
| 2 不可达判据 | **已消除** | B1 修复后 ①–⑤ 判定线方向与实测方向同向（`README.md:51-57`、`analysis.md:47-53`） |
| 3 引数无出处 | **已消除**（`0.006157`→`0.005026`）；**1 处口径标注不足** | `analysis.md:52` 的 `+0.001470` 是跨 seed 值（R2-N4） |
| 4 数据反驳结论 | 未命中 | 不利分量（0.014 带下不显著、参数量不对称、seed1 配对 CI 跨 0）均报出并解释（`analysis.md:69-78,95-99`） |
| 5 等价范围小于声称 | 未命中（已收窄） | `README.md:93-95` 明示复用 = **eval-only**，等值性只声称 eval 侧 ✓（N9 主体成立，但见 R2-N6） |
| 6 参考线口径写错 | 未命中 | `full` 明确写「同尺寸/同预算 12k×bs32×T256、只重跑评测」（`README.md:91-93`） |
| 7 字段错位 | 未命中 | `project.yaml` 结构正常（`WP1`/`conclusion`/`gaps` 各就其位） |
| 8 恒等式当因果 | 未命中 | gap-closing 是明确定义的比值，未当因果分解 |
| 9 合取判据读成半个成功 | 未命中 | 判据④ 的两个条件都独立报数（`analysis.md:52`） |
| 10 死代码冒充对账 | 未命中 | 我对账是人工逐项复算；`run.py` 键名与 `results.json` 逐键相符 |
| 11 修复引入新错 | **命中（弱）** | 见 #13 形态 2（`MODEL.md` 指纹未随 `run.py` 同步） |
| 12 测试绿但入口不可复现 | **接近命中** | `README.md:120` 的复现清单只覆盖 3 臂 ⇒ 照字面复现会 `KeyError`/缺权重失败（R2-B2） |
| 13 局部修复 / 覆盖小于声称 | **命中（形态 2 + 形态 4）** | 形态 2：`run.py` 改了、`MODEL.md:41` 的兄弟字段（同产物指纹）未扫（R2-B1）；形态 4：`TICKET.md` §8.2 声称 `§5/§6/§7` 已同步，而 `README.md:120` 未同步（R2-B2）；另 `TICKET.md:190` 声称 N6 落点含 `conclusion`，实际限定语落在 `conclusion_review`（R2-N6） |
| （`AGENTS.md` §2.1 事实文档纪律） | **命中（非阻断）** | `README.md:45` 标题「写下后不得改」+ `:47-50` 变更叙事；`analysis.md:4-6` 同型（R2-N1） |

### 2.8 科学结论是否仍成立 —— **逐项独立复算，全部成立（未因修文字而松动）**

| 项 | 文档值 | 我的独立复算 | 判 |
|---|---|---|---|
| 主判据：carry16_s0 被 win17_s0 支配 | 208,896 B < 212,992 B 且 1.624104 < 1.624979 | 同（`results.json:849-858` vs `:868-877`） | ✓ 点估计严格支配 |
| 下包络 | 7 个纯窗口点 + full | 按字节升序取记录最小：1.628110→1.624104→1.619953→1.599856→1.595823→1.513325→1.502689，严格下降 | ✓ |
| 支配集 | carry16_s0←win17_s0；carry16_s1←win17_s0、carry16_s0 | 同（同字节 1.624979<1.626640） | ✓ |
| 插值点 `k_eff=17.3333` | 1.62272（w=0.3333，win17↔win18） | 1.624104 + (1/3)(−0.004151) = 1.622720 | ✓ |
| `carry_minus_interp_window` | −0.002259 / −0.003920 | 1.622720−1.624979 / −1.626640 | ✓ |
| gap | 0.125421 | 1.628110 − 1.502689 | ✓ |
| gap-closing | 2.5% / 1.2%；1 格 3.2%、2 格 6.5%、8 格 22.5%、112 格 91.5% | 0.003131/0.125421 = 2.50%；0.001470/0.125421 = 1.17%；0.004006/0.125421 = 3.19%；0.008157→6.50%；0.028254→22.53%；0.114785→91.52% | ✓（「全部收益 < 加宽 1 格」0.003131 < 0.004006 成立） |
| 必要条件 Δbpc | +0.003131 / +0.001470（或同 seed +0.002453） | 逐项对平；两种口径皆 > 0 | ✓ |
| `Δbpc_far`（引 2634） | +0.015784 / +0.014132 | `2634/results.json:458`、`:682` 逐字相同；无通道臂 = 0.0（`:9987,:10003`） | ✓ |
| `k_eff` / 字节 | 12,288×k；carry 另加 16,384 | 196608/208896/221184/294912/393216/1572864/3145728；212,992/12,288 = 17.3333 | ✓ |
| 预登记点集加固 | 1.623401 > 1.624979 | 1.628110 + (1.3333/8)(−0.028254) = 1.623401 | ✓ |
| 噪声 | σ_cell 0.000983 / 0.001661；0.014 带 | \|1.628110−1.629093\|、\|1.624979−1.626640\| | ✓ |
| `被否证` 归因（实现/设计主因、结构次因、非 innovation） | `project.yaml:16`、`analysis.md:104-119` | 未越界（显式禁止外推到 delta 族） | ✓ |

---

## 3. Blocking

**R2-B1（P1）· `MODEL.md` 与 `results.json` 对同一个 `run.py` 给出互相矛盾的 sha256**
- 位置：`MODEL.md:41`（`eff58d5153f201c8bb591b25516bff032b9b14eb7733dddf6d5fdedf3d68c1d6`）vs `results.json:1031`（`2023f36d1a9694c8579bfab76c76da5e5b57ab215cfdba46f52f992164615778`）。
- 证据：全仓 `grep eff58d51…` 仅命中 `MODEL.md:41`；`grep 2023f36d…` 仅命中 `results.json:1031`；仓库**没有根目录 `run.py`**（`find run.py` 只返回各实验目录内的），`run.py:306-307` 明确用 `os.path.join(HERE, "run.py")` 自算 ⇒ 两者**指同一个文件**，不可能同时为真。`run.py` 是本轮为 N1/N2 修改并重跑的文件（`TICKET.md:186-187`、`.git/COMMIT_EDITMSG`），而 `MODEL.md` 不在 `TICKET.md` §8.2 的落点表里。
- 违反：`SCHEMA.md` §22.6（共享/运行期文件指纹入 `MODEL.md`）、§24 第 13 类形态 2（改了 `run.py`、没扫同一产物的兄弟字段）、第 11 类。
- 影响：`results.json` 的 `code_hashes.run.py` 是「这份数字由哪个脚本产生」的唯一绑定。两者矛盾 ⇒ 读者无法判断落盘 `results.json` 是否由交付的 `run.py` 跑出（也就无法判断「N1 重跑」是否真的按同代码发生）。
- 最小修法（先跑一条命令再改一行）：`sha256sum axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/run.py`
  1. 若 = `2023f36d…` ⇒ `MODEL.md:41` 是陈旧值（最可能，因为它是手写、且不在本轮处置表内），改成 `2023f36d…` 即可；
  2. 若 = `eff58d51…` ⇒ 落盘 `results.json` **不是**交付 `run.py` 产生的（`run.py` 在评测之后又被改过）⇒ 需重跑一次 `run.py`（纯推理 ≈0.3 min，不需重训）或如实写明该指纹的时点。
- 复现：读 `MODEL.md:41` 与 `results.json:1031` 对同一文件名比哈希。

---

## 4. Non-blocking

- **R2-B2（P2）· `README.md:120` 的训练清单未同步到 5 臂**：行内 `ARMS="24 32 128"` 与 `train/win{24,32,128}/` 仍是初版；`README.md:85-86`（§5 表）、`:98-103`（追加登记）、`analysis.md:145`（正确的 `ARMS="17 18 24 32 128"`）、`results.json` 的 10 臂都可反证。照 `§7` 字面复现 ⇒ 缺 `win17/18` 权重 ⇒ `run.py` 失败（§24 第 12 类近邻；`TICKET.md:184` 声称 `§5/§6/§7` 已同步，属 §24 第 13 类形态 4）。最小修法：`README.md:120` 改为「`bash train_arms.sh`（默认 `ARMS="24 32 128"`；本实验另按 `ARMS="17 18 24 32 128"` 训练 5 臂，见 §5 追加登记）⇒ `train/win{17,18,24,32,128}/…`」。
- **R2-N1（P2）· 事实文档里的变更叙事**：`README.md:45` 标题仍是「（预登记，写下后不得改）」，紧接着 `:47-50` 是「**复核修正记录（B1）**：…初版…已改为…」；`analysis.md:4-6` 同型。与 `AGENTS.md` §2.1（事实文档只写当前状态、不带「原本 A 后来 B」/ERRATA；流水归 `.scratch`）相悖，且标题与正文自相矛盾。最小修法：删掉这两段修正记录（历史已在 `TICKET.md:180-195` 与 commit message 里），或把 `§4` 标题从「写下后不得改」改成不含该承诺的表述。
- **R2-N2（P2）· 证据锚点仍钉在 `a679170`**：`project.yaml:47-48` 的 `analysis.md @a679170` / `results.json @a679170` 指向**本轮改写前**的版本——那一版 `analysis.md` 仍含错值 `0.006157`、那一版 `results.json` 没有 `liveness` 块。`check_tree.py:99-120` 只校「锚点可达」⇒ 机检抓不到。最小修法：两处锚点改为 `12e5fb7`（或本轮实际落盘 commit）。第 1 轮复核曾用「锚点 = 该产物最后变更 commit」的读法（`REVIEW-1.md:47`），按该读法本轮改过文件后必须前移。
- **R2-N3（P2）· `WP1.means` 与 `WP1.status: "已完成"` 的字面张力**：`project.yaml:78` 仍写「与 R1 量化点、1.1.1 窗口点同图画质量-召回」，`:79` 的 `acceptance` 已自认「召回轴与量化点未叠加」。按 `docs/research-schema.md:130` 可判为合规（状态只答「做完没有」），故**不判缺陷**；建议在 `means` 后加半句指向 gaps。
- **R2-N4（P2）· 判据④ 的 seed1 Δbpc 口径标注不足**：`analysis.md:52` 的 `+0.001470` 是**跨 seed**（carry16_s1 − win16_s0）值，与同产物 `effects_by_seed["1"].delta_bpc = 0.002453`（同 seed）并存而未注明；`2634/analysis.md:129` 用的是 0.002453。两种口径都过判据（>0），但「2 seed」的措辞会误导；连带 `analysis.md:92` 的「1.2%」在同 seed 口径下是 2.0%。最小修法：在 ④ 栏括注口径来源。
- **R2-N5（P2）· 新增单测的落点与强度**：`test_2638_frontier.py` 放在实验目录 ⇒ `python3 -m pytest tests/ -q`（`AGENTS.md` §8 的仓库级门禁）**不收集它**，而本仓惯例是每格一测进 `tests/`（`tests/test_carry_realtext_13m.py` 等 7 个同类文件，无 `tests/test_*frontier*`）。同时 `:70` 的断言是恒过的子串检查（见 §2.1）。最小修法：把该文件（或一份同内容的 `tests/test_carry_frontier_13m.py`）纳入 `tests/`，并把方向断言改成对源码文本/可调用函数的实质断言。
- **R2-N6（P2）· `TICKET.md` 处置表的落点与事实不符**：`:190` 记 N6 落点含 `project.yaml conclusion`，实际限定语在 `conclusion_review`（`project.yaml:18`），`conclusion`（`:17`）的「严格支配」仍未带 σ_cell/0.014 限定；`:192` 记 N9 落点含 `analysis.md §3`，实际「eval-only + bpc 逐位复现」只在 `README.md:93-95`。条目本身已落地，仅落点标注不准。
- **R2-N7（P2）· 舍入级不一致（顺手改）**：`README.md:108`「8.4–8.51 min」应为 8.4–8.53（`train/win24/results.json:766`）；`analysis.md:125` 与 `README.md:110` 引的评测墙钟 `0.32 min` 是重跑前值，现 `results.json:1022` 为 `0.3`（`run_log.txt` 17.84 s）。二者均不入判据（`analysis.md:126` 已声明），仅作过程记录的一致性。
- **R2-N8（P2）· N9 的实质缺口仍在**：复用臂等值性只有 eval 侧证据（bpc 逐位复现 ✓）；训练侧 2634 用的 `pathlm/depth_ar.py` 指纹（`2634/results.json:10380` = `b304c234…`）与 2638 评测时的（`1f8f9eb4…`）**不同**，说明两实验之间共享文件确实演进过——这不影响结论（窗口臂不经 carry 路径、复用臂只读权重），但「复用等值」这句话本身仍应限在 eval 口径。

---

## 5. 需要 supervisor / runner 跑的检查（我一律 not-run）

```bash
# R2-B1 的裁决（最关键）：判定 MODEL.md:41 还是 results.json:1031 是当前值
sha256sum axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/run.py
sha256sum axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/config.json
python3 tools/summary.py --check                                  # 研究树 SSOT 必过（复核者无法跑）
python3 -m pytest tests/ -q                                       # 共享底座回归
python3 -m pytest axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/test_2638_frontier.py -q
sha256sum axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/train/*/model.pt \
          axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m/train/*/model.pt
                                                                  # 与 MODEL.md / config.json / results.json.checkpoints 对表
git diff ab41bb9 12e5fb7 --stat                                   # 本轮到底改了哪些文件（我只看得到工作树对 HEAD 干净）
git diff ab41bb9 12e5fb7 -- .../2638-carry-frontier-13m/MODEL.md  # MODEL.md 是否在本轮改动集内（决定 R2-B1 的修法分支）
```

---

## 6. 一句话总判词

**FAIL（窄范围）**：第 1 轮 B1–B4 四条 blocking **我逐条独立复算后判为「真修、方向自洽」**，主结论（carry16 被 win17 支配；gap-closing 2.5%/1.2% vs 1 格 3.2%；必要条件成立；`被否证` 归因未越界）也**逐项独立复算成立**且不依赖事后追加臂（预登记点集弦插值 1.623401 > 1.624979）——**科学结论 PASS**；但交付物仍有 1 条 blocking（**R2-B1**：`MODEL.md:41` 与 `results.json:1031` 对同一个 `run.py` 给出不同 sha256，需 `sha256sum` 裁决后同步一行）＋ 8 条 non-blocking（`README.md:120` 训练清单只列 3 臂、实验锚点仍钉 `a679170`、事实文档留变更叙事、单测不在 `tests/` 且断言恒过、④ 栏 Δbpc 跨 seed 口径未注明等），**全部为文字 / 指纹 / 指针层面，不需要重跑 GPU**（除 R2-B1 分支 2 需一次 ≈0.3 min 的纯推理重跑）。**若父级裁示「以 `results.json::code` 为准」（`2620-capacity-converged/MODEL.md:38-41` 有此仓库惯例明文）即可覆盖指纹漂移，则本条降为 P2，总判词相应转为「OK with notes」**——该取舍属框架性判断，我把它留给你。

---

## 7. 残差风险（我未能消除的）

1. **提交区间 `ab41bb9 → 12e5fb7` 的内容我没有独立看过**（`watchdog_diff` 不含已提交区间）⇒「本轮只改了 2638 目录 + `project.yaml`」这一说法我只在**内容层**核对（逐行重读，行号一致），**字节层无证**。
2. **R2-B1 无法由我裁决**：两个哈希谁对谁错需要 `sha256sum run.py`；我仅能证明「不能同时为真」。
3. **回归命令全部 not-run**（无 shell）；`summary.py --check` / `pytest` / checkpoint 哈希对表均采信 `TICKET.md` §8.3。
4. 「重跑 `run.py` 数值与落盘 `results.json` 一致」用的是**替代证据**（`run_log.txt|jsonl` ↔ `results.json` 逐字对平 + 与 2634 独立产出的逐位复现），不是我自己重跑。
5. **共享文件「未改」只有间接证据**（4 条哈希在 `MODEL.md` 与 `results.json` 间自洽 + 与 `61-axis2-depth/REVIEW-2.md:140` 记录的 `483b181` 版本一致），无磁盘重算。
6. **状态精度假设（fp16 记账 vs 引擎 fp32）**：按 fp32 则 `k_eff=18.67`、carry 更差约 0.0073（方向不变、结论更强）——文档已显式记账（`README.md:59-77` §4a、`analysis.md:76-78`），非缺陷，但结论对横轴口径敏感这一点仍属「读者须知」。
7. **规模外推**：`k_eff` 的「每位置等效字节」是 T=256 专用口径，`project.yaml` §适用范围（`:17`）已含 `T=256` ✓，禁外推条款齐备。