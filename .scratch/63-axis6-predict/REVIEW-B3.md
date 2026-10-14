<!-- 本报告作者 = fresh 复核者 #3（未参与写作、只读）；其上下文同样**无 shell、无写文件工具**
     （与 REVIEW-B1/B2 相同的限制，与派单前提「带 shell」不符；它曾按 need_decision 联系主管但超时）。
     故由 axis6-lead 于 2026-10-14T10:50:06+08:00 把其返回的完整报告**逐字**落盘（内容未改一字）。
     执行类验收的 lead 侧代跑见 TICKET.md §B5；其 4 条 normal 的处置见 TICKET.md §B6。 -->

No shell tool is available in my context (toolset: `read/grep/find/ls/watchdog_diff` + `contact_supervisor`), so every execution-class acceptance check is `not-run`. I also have no write tool, so the report is returned in full below (per the artifact-in-response rule). Supervisor `need_decision` contact timed out, so the blocker is recorded here.

---

# REPORT — `.scratch/63-axis6-predict/REVIEW-B3.md` (content)

VERDICT: FAIL

## 0 元信息与复核者约束

- 被审 revision：`HEAD` = `0c4dc01`（`watchdog_diff` 于本轮报告：`No working-tree changes against reviewer-launch HEAD 0c4dc0132321.` ⇒ **工作树逐字节等于 `0c4dc01`**，故下文所读文件内容 = 被审 revision 的内容；但**提交区间本体不可见**，见 §4 not-run 第 6 条）。
- 复核者 #3：fresh 上下文、未参与写作、只读。**我的可用工具只有 `read` / `grep` / `find` / `ls` / `watchdog_diff` / `contact_supervisor` —— 本上下文没有 shell**，与派单前提「带 shell」不符（与 REVIEW-B1/B2 相同的限制）。已按 `need_decision` 联系主管，**等待超时未获回复**，故按「报告阻断」处理。
- 因此派单点名的**必须亲自执行项**——`pytest`、两个 `mutate.py`、`summary.py --check`、**确定性重跑 + 逐张量 `torch.equal`**、`sha256sum`、`stat`/mtime、`git show/status/diff` 区间——**全部 `not-run`**（逐条补齐命令见 §4）。依 `AGENTS.md`「Proof and review」：**「未知验收不是 PASS」**，故本轮判 FAIL。
- 时间戳：本上下文**无时钟可读**（无 shell）⇒ 无法执行 `date -Iseconds`；下文凡引用时刻均来自被审文档内已落盘的记录。

## 1 正确项（我亲自核过，含源文件位置）

**C1 · 事实文档逐数字对账 PASS（任务第 7 项，重点）**
`6.2/project.yaml:17` 结论 (j) 的**每一个数字**与 `2650/results.json::criteria` 逐值相同：
- τ = 1.4323/1.5481/1.6586/1.7358 ↔ `ledger.{1,2,4,8}.tau` = 1.432323/1.548148/1.65859/1.735828 ✓
- α₁ = 0.8000/0.8667/0.9317/0.9785 ↔ 0.8/0.866667/0.931718/0.978458 ✓
- R = 1.233/2.355/4.514/8.773 ↔ `R_K_2640consistent` 1.2334/2.3547/4.5137/8.7725 ✓
- Δτ/ΔK = 0.1158/0.0552/0.0193 ↔ `margins` 0.115825/0.055221/0.019309 ✓；阈值 0.0579 ↔ `threshold_half_base` 0.057913 ✓；`k4_to_k8_worth_it: false` ✓
- K=4→8 Δα₁ +0.0467 CI [+0.0296,+0.0629] ↔ `delta_alpha1` 0.04674 / `ci` [0.029604, 0.062894] ✓
- 对抗臂 0.0185 [0.0138,0.0248] ↔ 0.018495 / [0.0138,0.0248] ✓；尾部 0.0 上界 0.0016 ✓；13m_s1 0.0212 [0.0161,0.0279] ↔ `alpha1` 0.021212 / `wilson_95` [0.0161,0.0279] ✓；τ 上界 1.7672 ↔ `tau_oracle_2640` 1.767202 ✓；0.811/0.876/0.939/0.982 ↔ `tau_over_oracle` ✓；1/(1+τ) 0.411→0.365 ✓（自算 1/2.432323=0.4111、1/2.735828=0.3655）。

`6.3.3/project.yaml:17` 结论 (a)–(e) 的每个数字与 `2660/results.json` 逐值相同：
- `disc_k2` t+2 0.4572/0.4573 ↔ 0.4572088068…/0.4572551… ✓；λ=1.0 → 0.4878/0.4892 ↔ 0.4877871…/0.4891767… ✓；λ=0.1 → 0.4688/0.4685 ↔ 0.4688040…/0.4684643… ✓
- Δ seed0 +0.0306 [+0.0285,+0.0326] ↔ 0.03058 [0.02852,0.03259] ✓；seed1 +0.0319 [+0.0296,+0.0343] ↔ 0.03192 [0.02961,0.03427] ✓；合并 +0.0313 [+0.0296,+0.0330] ↔ 0.03125 [0.02955,0.03302] ✓；λ=0.1 合并 +0.0114 [+0.0101,+0.0127] ↔ 0.0114 [0.01006,0.01274] ✓
- 13m_s1 +0.0329 [+0.0312,+0.0345] ↔ `second-backbone-13m_s1` combined 0.03286 [0.03122,0.0345] ✓；k=4 +0.0284 [+0.0268,+0.0302] ✓
- ar_t1 0.6714 [0.6688,0.6740] ↔ 0.6714 `wilson95_t1` [0.66884,0.67396] ✓
- 路径 (ii)：repro_direct1 α₁ 0.802243 / τ 1.453619 ✓；lat_k2_lam1 0.7337/0.7537 ↔ 0.733722（s0）/ 按 Δ 反推 s1 一致 ✓；disc_k2 0.6421/0.6682 ↔ 0.642077 / 0.668216 ✓；Δα₁ +0.0916 [+0.0525,+0.1310]、Δτ +0.1464 [+0.0971,+0.1973] ↔ `draftverify.L2` point/ci 逐值 ✓；s1 +0.0855/+0.1375 ✓；−0.069…−0.160 ↔ −0.068521 / −0.160166 ✓。
- 两份 `experiments/README.md` 表行数 = 磁盘目录数（6.2：12 = 12；6.3.3：1 = 1）✓；`gaps`/`status` 与实际相符（6.2 `status: 进行中` + gaps 末三条 = window>0 树验证未实装 / 53.9M 对抗臂未跑 / K 族边际未扫 ✓；6.3.3 gaps 含 free-run bpc 未测 ✓ 与结论 (e) 一致）。

**C2 · 前两轮 findings 的修缺**（B1-N1/N2、B2-F1..F6）确实落地：
- B1-N1 ✓ `2650/results.json:35` `budget_min: 15.0`（原 25.0）。B1-N2 **部分**：`2650/README.md` §4 已改成定稿口径 `R_K = K·(1+γ)/(1+τ_K) = 3K/(1+τ_K)`，但 §8 残留（见 Finding 4）。
- B2-F1 ✓ `2660/results.json:4922-4923` `L0_all_pass: true` + 判词 "采纳（t+2 … Δ=+0.02844 CI[+0.02679,+0.03016]）"，与 `second-k4-13m_s0/results.json:5395` `all_pass: true`、`:40` L0 判词、`analysis.md:7` 一致 ✓。
- B2-F2 ✓ `latent_reg.py:321 head_files_all_tensors_equal` + `run.py:218-228` 落盘 `state_dict_all_tensors_equal`；三份 results.json（`2660:4523/4531`、`second-backbone:3407/3415`、`second-k4:5379/5391`）**六对全 true**；三份 `MODEL.md:27-28` 有「sha256 只标识文件、不构成内容同一性证明」注记 ✓；`WP-6.3.3-LOG.md` A3(c) 已把过强的 sha256 断言更正为「只对同一 arm 的重跑成立」并给出字节差成因（basename 长度差 56 字节）✓。
- B2-F3 ✓（`analysis.md:231` 降级为「描述性读数，未做显著性检验」）；F4 ✓（`test_latent_reg.py:174 test_mse_target_position_is_next_next_hidden` + `mutate.py:34 mse_target_off_by_one`，`mutation_report.json` 该条 `caught_red: true`）；F5 ✓（`analysis.md:215-216`、`MODEL.md:74-75` 注明 `lat_k4_lam0` = L0 自检补充臂）；F6 ✓（`analysis.md:242-244` 归因边界 + 「不得读成回归机制优于分类机制」）；F1 附带 ✓（`run.py:205` 按 k 参数化；`second-k4-13m_s0/results.json:40` 已改 k=4 文案）。

**C3 · 主动证伪（任务第 8 项）——三项均**未**推翻判词**
- **V4 未饱和 vs V5 K>4 不划算未混淆**：README §4 把「α₁ 相对 tree_topk4 的配对 CI（饱和）」与「Δτ/ΔK 相对阈值（代价轴）」分成 V4/V5 两条；结论 (j) 也分开写。二者是不同判据 ✓。
- **K=1/2/4/8 确为同一分布族**：`2650/verify_gate.py:76-83,97-99,220-222`（`repro_topk4`=2640 `tree_topk4`；K=1/4/8 全取 `T.pw_mix(nodes)["q"]` 的 top-k）+ `2640/spec_tree.py:143-178`（`combo_pw` 分布 = `pw_mix(nodes)["q"]`，`TOPK_ARMS` 取同一 `q`）⇒ 单候选 = q 的 argmax、K=2/4/8 = q 的嵌套 top-k ✓。
- **λ=1.0 非事后挑选**：`2660/README.md` §3 预登记臂表**跑前**已写死 `lat_k2_lam01`(0.1) 与 `lat_k2_lam1`(1.0)，且卡死训练预算/种子/数据序 ✓。
- **L2 的 Δα₁ 非「disc 臂被系统性削弱」**：README §2 证明 λ=0 两臂**同一计算**（唯一自由度 = λ·MSE），L0 逐张量等价为硬门；两臂对 2640 `direct1` 同为负（−0.069…−0.160，已如实标注为 regime 差、不作判据）⇒ 非不对称削弱 ✓。
- **repro_direct1 保真门的覆盖边界（提示性）**：它复用引擎默认草稿 ⇒ 只证 harness 与 2640 等价，**不**检验新头接线；新头接线的覆盖靠 `test_latent_reg.py` + 变异 `draftverify_head_wrong_position`（`mutation_report.json` 该条 `caught_red: true`）。未见缺陷，但该门本身不能被读成「新头已验」。

**C4 · 禁墙钟 / 权重不入库**：`2650/results.json` 无墙钟字段（`criteria` 全为解析量；GPU 401 s 只在 `run_log.txt`）；三份 2660 `_wall_clock_fields: []`；未跟踪清单不含 `*.pt`（`.gitignore` 覆盖）。

## 2 findings

### Finding 1 — normal（新发现）· `2660/mutation_report.json` 含逐次运行耗时，任何重跑都会弄脏工作树
- 位置：`axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/mutation_report.json`（每条 `results[].pytest_tail` 形如 `"1 failed, 12 deselected in 1.59s"`；`post_restore_tail: ["13 passed in 3.98s"]`）vs `mutate.py:95` `"pytest_tail": (r.stdout or "")[-400:].strip().splitlines()[-1:]`、`:105` 同（**无** `_strip_timing`）。
- 被违反判据：本轴自己已确立的规则 —— `6.2/project.yaml` change_log `2026-10-13T17:55:00+08:00`：「实验侧 `mutate.py` 不再把逐次运行的 pytest 耗时写进 `mutation_report.json`（**入库事实文件不得含逐次抖动字段**）」；`2650/mutate.py:59-62` 已实装 `_strip_timing`（本次亦无耗时字段 ✓），2660 **未同步**。
- 影响：`mutation_report.json` 非确定 ⇒ 任何人重跑 `mutate.py` 都会产生工作树脏改动；且 `AGENTS.md` §2.1 的「当前事实、无逐次抖动」不成立。不影响任何读数/判据。
- 最小修法（不自行实施）：把 `2650/mutate.py:59-62` 的 `_strip_timing` 复制进 `2660/mutate.py`，在 `:95`/`:105` 落盘前套用，重跑一次 `mutate.py` 并随修缺提交更新该报告。（复核者无法确认重跑后逐字节确定 —— 见 §4 `not-run` 第 2 条。）

### Finding 2 — normal · `6.3.3/project.yaml` 的 `conclusion_review` 与 `work_packages[WP1].review` 状态自相矛盾（状态/复核记录不符实际）
- 位置：`axes/6-predict/6.3.3-latent-regression/project.yaml:18` `conclusion_review`：「本项无旧格、本仓零经验，**尚无本地判决，故 status 未开始**……**独立复核未做**。」；对比同文件 `:17` 已有本地判决 (a)–(e)、`:54` `status: "进行中"`、`:47-48` `WP1.review: "…已做：.scratch/63-axis6-predict/REVIEW-B2.md"`、`WP1.status: "已完成"`。
- 被违反判据：`AGENTS.md` §2.1「一律陈述式、当前状态；同一事实恰好一个家；冲突以权威处为准，另一处删除或改为指针」；`SCHEMA.md` §14b / `AGENTS.md`「Proof and review」：WP 收口需复核**通过**。所引复核的唯一记录是 `REVIEW-B2.md` = **VERDICT: FAIL**（且本轮 B3 未完成），`review` 字段却只写「已做」，未记录判词与处置。
- 影响：事实文档层面「状态虚报 / 记录冲突」——读者会据 `conclusion_review` 认为本项未判、未复核，而正文相反。**不影响** (a)–(e) 的数字与判据（我已逐值核过）。
- 最小修法（axis6-lead，勿派 worker）：把 `conclusion_review` 重写为实际轨迹（B1/B2 的 findings、已修项、B3 进行中 + 无 shell 限制），删除「尚无本地判决 / status 未开始 / 独立复核未做」；`WP1.review` 补「REVIEW-B2 = FAIL（2 normal 账实不符/证据链，已修）+ B3 复核中，未过前不得写『已完成』」，或在 B3 PASS 后再置 `已完成`。

### Finding 3 — normal · `2650/results.json::config.budget_min` 已不等于实际运行参数，且修缺说明的因果与源码矛盾
- 位置：`2650/results.json:35` `"budget_min": 15.0` vs `2650/run_log.txt:1` 运行记录 `budget_min=25.0`（该行是唯一保留真实运行参数处）；`WP-6.2.3-LOG.md` 修缺条与 `run_log.txt:195-197` 声称「原为 **runner 缺省值** 25.0」，但 `2650/verify_gate.py:522` 现为 `--budget-min default=13.0`（13.0 ≠ 25.0 ⇒ 该「缺省值」说法与源码不符）。
- 被违反判据：`AGENTS.md` §2.1 / `docs/methodology.md`（证据可回溯）：`results.json` 作为「原始落盘」应记录**实际运行参数**；修缺记录（工单/日志）不得写入与源码不符的因果。
- 影响：无测量/判据影响（实耗 401 s < 15 min）；但「原始记录」被改成一个从未使用过的值，真实值只剩 `run_log.txt`。
- 最小修法（axis6-lead）：二选一并去重——(i) 恢复 `config.budget_min` = 实际值 25.0，另加 `budget_min_contract: 15.0` 标注预登记硬盒；或 (ii) 保留 15.0 但在同处加 `budget_min_run: 25.0`；两种口径都同时更正 LOG/run_log 的「缺省值」措辞。
- 需 shell 复核确认（`not-run`）：`git diff --stat 21f390d..0c4dc01 -- axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/verify_gate.py` —— 若该区间改过 `verify_gate.py` 的 `default`，则修缺超出其自述范围（只改了 results.json 字段 + run_log 注记），须单独登记。

### Finding 4 — normal（低）/ info · `2650/README.md` §8 仍把定稿口径公式与 ERRATA 叙事留在契约文件内（B1-N2 未修净）
- 位置：`2650/README.md` §4 V5 已有 `R_K = K·(1+γ)/(1+τ_K) = 3K/(1+τ_K)`；§8「契约更正记录」**再次**给出同一式（`定稿口径 = K·(1+γ)/(1+τ_K) = 3K/(1+τ_K)（与 2640 落盘口径逐字一致，本文件 §4 V5 即此式）`）并保留「该笔误是出题（axis6-lead）的错……」的归责叙事。
- 被违反判据：`AGENTS.md` §2.1「每条事实在事实文档里恰好出现一次；别处只放指针」+「正文不带变更记录 / ERRATA / 事故编号叙事」；B1-N2 的最小修法即「§8 收缩为一行指针或删除」。
- 影响：纯文档一致性，无数字影响（§4/§8 两处一致）。
- 最小修法（axis6-lead）：§8 压成一行指针（例：「V5 的 `R_K` 口径曾有一处笔误，经过与处置见 `.scratch/63-axis6-predict/TICKET.md` §B4.1；本文件 §4 为定稿口径」），删去重述的公式与归责句。

### Finding 5 — info · `6.2/project.yaml::work_packages[6.2.3].review` 仍是派单时的计划，未回填已完成的复核
- 位置：`6.2/project.yaml:87` `review: "跑完另派 fresh 复核（核对单测确可失败、且阈值按规模分档）"`；对比 `:80` 的 WP 6.2.2 已改成「…已做：REVIEW-1.md（fresh，无 blocker）」，且本项结论 (j) 已引 `REVIEW-B1.md`。
- 被违反判据：`AGENTS.md` §2.1（当前状态、一处一家）。
- 最小修法：补「已做：`.scratch/63-axis6-predict/REVIEW-B1.md`（fresh、无 shell；无 blocker + 2 normal，已修）」。

## 3 Merge verdict 与理由

- **FAIL**。理由：(i) 任务点名的**执行类验收在本上下文不可执行**（无 shell，与派单前提不符）⇒ 依「未知验收不是 PASS」，不能给 PASS；(ii) 4 条 normal 中 Finding 1 是**本轮新发现的、入库事实文件非确定性**缺陷，Finding 2 是**事实文档状态/复核记录冲突**，均须修。
- **无 blocker 级科学缺陷**：结论 (j) 与 (a)–(e) 的每个数字我都逐值核过且**全部对得上**；K 族同族性、λ 网格预登记、L2 单因素性、V4/V5 判据分离均经我主动证伪而未推翻；前两轮 findings 的修缺我逐条核实**确已落地**。

## 4 not-run（本上下文无 shell / 无写工具；须由带 shell 的复核者补跑并回填）

1. `not-run` `python3 -m pytest axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/test_verify_gate.py axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/test_latent_reg.py -q`（期望 11 + 13 passed）。旁证（非替代）：`2650/mutation_report.json` 各条 `pytest_summary` = `1 failed, 10 deselected` ⇒ 全集 11；`2660/mutation_report.json` 各条 `1 failed, 12 deselected` + `post_restore_tail: 13 passed` ⇒ 全集 13；两者自洽。
2. `not-run` 两个目录各 `python3 mutate.py`：须**每条 `test_failure` 级抓红 + 前后 sha256 相等**，并与本报告内容比对（2660 因 Finding 1 必然出现耗时字段差异）。补齐：`cd <dir> && python3 mutate.py && git status --short`（重跑后应**无**脏改动，除 Finding 1 的 2660 情形）。2650 侧待核哈希：`sha256sum …/2650-verify-gate-13m/verify_gate.py` 应 = `6c88d225c691ab0058562585f67c655e1b0e1aaca7a3b6a936fc0bd188bea6f3`（取自 `mutation_report.json:16` 的 `sha256_restored`）。
3. `not-run` `python3 tools/summary.py --check`（期望全绿）。
4. `not-run` **确定性重跑（最强证伪）**：先读 `run.py` argparse（`--stage/--out/--ckpt/--seeds/--arms/--no-lock/--smoke`；**`--out` 缺省 = 脚本目录且 `Tee` append，绝不能 `--out .`**），然后
   `python3 axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/run.py --stage train --out /tmp/rev3-2660 --ckpt /home/z/vibe/pathlm/axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0`
   再比对 `/tmp/rev3-2660/results.json` 的 `arms.*.*.hit_t2/t3`、`ce_t2/t3` 与现盘逐值相同；`heads/*.pt` 与现盘**逐张量 `torch.equal`**（**不得**用文件 sha256 不等反推内容不同；已知成因 = zip entry 名取自 basename 长度，见 `WP-6.3.3-LOG.md` A3(c)）。
5. `not-run` `sha256sum` / `stat -c '%y %n'`：`2640-tree-conf-13m/{spec_tree.py,run.py,results.json,cycle_log.jsonl,run_log.txt}` ↔ `2650/results.json::imported_2640_sha256`（`spec_tree.py: 71d98b26…8fb62f`、`run.py: fba5a16c…b9468`）与 `.scratch/63-axis6-predict/run1-snapshot/README.md`（`results.json=1302b678…`、`cycle_log=d2073370…`）；两个新实验目录的 `README.md` vs `results.json`/`run_log.txt` 首行时刻（**预登记须早于 run 开始**）。注：`2650/README.md` 在 run 后被 lead 追加过 §8 更正记录（`run_log.txt:1` 09:27:28 vs §8 记录 09:51:48）⇒ 任务要求的「mtime < run 开始」形式不变量**不成立**；我逐条比对了 §4 V0–V6 与 §3 臂表与 `analysis.md`/`results.json` 实际执行的判据（V0–V6、否证阈值 0.05/0.02/0.014、饱和规则、K=1→2 半阈值），**未发现事后放宽**；该判定仍须 `stat` 复核。
6. `not-run` **提交区间不可见**（`watchdog_diff` 只覆盖工作树 vs HEAD，不含已提交区间）：须
   `git show --stat 0c4dc01`；`git diff --stat 21f390d..0c4dc01 -- pathlm docs tools`（期望空）；`git diff --stat 21f390d..0c4dc01 -- axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m`（期望空）；`git status --short`。**我确认的只有**：本轮工作树相对 `0c4dc01` 干净。

---

## 给主管的 ≤15 行回报

- VERDICT: **FAIL**（无 blocker 级科学缺陷）；findings = 4 normal（1 新发现）+ 1 info。
- blocker 一句话：**我的上下文没有 shell**（工具仅 read/grep/find/ls/watchdog_diff），派单要求的 `pytest`/`mutate.py`/`summary --check`/确定性重跑+逐张量比对/`sha256sum`/`stat`/`git` 区间全部 `not-run` ⇒ 依「未知验收不是 PASS」不能给 PASS（supervisor `need_decision` 已联系但超时未回）。
- 我亲自跑过/核过的一句话：**无 shell 可跑**，但以 `read/grep` 逐值核过 6.2 结论 (j) 与 6.3.3 结论 (a)–(e) 的**全部数字**（τ/R/Δτ/ΔK/α₁/CI/hit_t2-3/Δα₁/Δτ/Wilson）全部对上，并核 B1/B2 全部修缺（budget_min=15.0、§4 定稿口径、L0_all_pass=true、state_dict_all_tensors_equal 六对、MODEL.md 注记、LOG 更正、MSE 位移单测+变异、k=4 文案参数化）**确已落地**；主动证伪三项（K 族同族、λ 网格预登记、L2 单因素）**未推翻**判词。
- 新发现（normal）：`2660/mutation_report.json` 的 `pytest_tail` 含逐次耗时（`2660/mutate.py` 未同步 `2650/mutate.py:59` 的 `_strip_timing`）⇒ 重跑弄脏入库事实文件；另 `6.3.3/project.yaml::conclusion_review` 仍是「尚无本地判决/独立复核未做」而正文已有 (a)–(e) 且 WP1 标「已完成」（所引 REVIEW-B2 = FAIL）。
- 其余 normal（低）：`2650/results.json::config.budget_min=15.0` 已非实际运行参数（`run_log.txt:1` = 25.0，且「runner 缺省值」与 `verify_gate.py:522 default=13.0` 矛盾）；`2650/README.md §8` 仍重述公式 + ERRATA 叙事（B1-N2 未修净）；info：`6.2/project.yaml::WP 6.2.3.review` 未回填已完成复核。
- 报告路径：**未落盘**（本上下文无写工具）⇒ 全文即本回复；建议 axis6-lead 将其逐字保存为 `.scratch/63-axis6-predict/REVIEW-B3.md`（首行 `VERDICT: FAIL`）。
- 下一步唯一有意义的补证：由**带 shell** 的复核者跑 §4 的 1–6（尤其确定性重跑 + 逐张量比对、`git diff --stat 21f390d..0c4dc01` 越界检查）。
