# 独立复核 R3（ticket 47）：46 批的 W1（产品级组合）/ W2（T=2048 同收敛）/ W3（P2+G5 收口档）

# 独立复核（只读）· ticket 46 第十九批 — W1 / W2 / W3

**范围与方法。** 只读复核三条线的**已提交 HEAD = `866436d`**（reviewer-launch 时 `watchdog_diff` = 干净：无 staged/unstaged、无 untracked）。逐条对照 `SCHEMA §14/§14b、§17.11/§17.14/§17.15/§17.16/§17.17/§17.18、§18–§23`、三份契约原文、`results.json`/日志/对账/变异产物与 git 索引。**无 shell**：`pytest`/`build_wiki`/`check_schema` 未能运行（见文末命令）。复核期间工作树被并发写入（见 Finding 3），我以 committed HEAD 为准。

---

## Correct（六项重点逐条，均以证据核对）

**① W1 两个相反口径是否如实并列 / 越界腿是否标为辅助腿 — 成立。**
- 两个 tf@4 口径**并列**：`analysis.md §5.3` 表同时给 synth（99.66% → 98.68%，−0.98pp 带内）与 enwik8（93.69% → 88.98%，−4.71pp 超带），`results.json::exit_asset_holds` 同给；`§9.2` 明写「两个口径必须分开写；**不得只报有利的那个**」；`catalog.yaml:1019` / `M1 README §5` 亦并列并写「**两个口径结论相反，已并列未择优**」。数字与 `results.json` 逐一对应（`tf4 0.9369/0.8898`、`tf_enwik8`；`tf4 synth 0.9966/0.9868`）。
- 越界腿**标注为辅助腿**：`catalog.yaml:1019`「（距离 241–255，§17.18 **辅助腿**）2.49% → 8.54%（+6.05pp）」；`analysis.md §5.5`「**外推参照，不是判据**（§17.18）」；距离区间 `results.json::main_leg_single_query_ood[*].distance_range = [241,255]`（>训练上限 240），`single_query_ood.carry_effect = 6.0547`。诚实对手、chance=1.56%、毁源有效均在场。

**② W2 「同收敛」是否有 step 与 loss 双证据 — 成立。**
- step 证据：两臂 `steps=6000`、`tokens=98304000`（`results.json::config.arms[*].fields`），与 2626 的 T=1024 点 **matched-step@6000**。
- loss 证据：`full_2048_6000_final_logged_loss = 1.1221`（`train_log` step 5999，见 `results.json`）、同点 @3000 = 1.5423、T=1024@6000 = 1.2146 ⇒ **本点不欠训练**；且明写「同 step ≠ 同 loss、末窗斜率仍 −0.16/1k、只答带分类」。
- 主判据 Δ@6000 = **+0.004186**（bpc_full 1.705365 / bpc_window 1.701179）在带内，与 @3000 +0.004361 同带；稳定性前置 `gnorm_max=7.22 <100`、跳步 0。口径（快照 / 混合四点 / 禁外推）如实。

**③ W3 的 P2-a「误报」判定 — 成立。**
核实 `2627-carry-x-depth/train_{base,carry,depth,both}.log` 是否已跟踪：① 仓根 `.gitignore` 含 `*.log`（只作用于未跟踪文件）；② **`.git` 索引**里四个路径**均存在**（对 `.git/index` 逐名检索 `train_base|both|carry|depth\.log` 全部命中）；③ 复核引用的 blob 前缀 **`c7be78f8…` 在 `.git/objects/c7/be78f830684a5dea9f8e799b3d20b8c2ae912e` 存在**（loose object）；④ 当时 `HEAD` 与索引一致（watchdog 干净）⇒ 四 log **确已被 git 跟踪**，R1「none are in git」的前提不成立、W3 判定正确，清单**保留不删**是对的。（未跑字面 `git ls-files/ls-tree/check-ignore`——无 shell；以索引+对象证据替代，方法论上等价。）

**④ G5 `CLOSURE.md` 结论句与 6 条重开判据线 — 与已测事实一致。**
逐条回查：13M 机制成立（JS 0.0249→0.0361、tax −0.0042、net +0.0246、`release_50m`，2622 §1）；50M 失效（JS 0.0322<0.0333、net −0.0100，2622 §1）；24k 收敛 net +0.0144、主判据 +0.0346（CI[0.0298,0.0407]）、末窗 −0.001/1k（2626 §1）；抖动 0.0074=51% ⇒ `on_threshold`（2628，R2 复核过）。部署缺口 13M +0.1052 / 50M +0.0322 ✓。6 条重开线（跨规模部署反转 / 换掉 8× / seed 升格 / 50M 复现 / `carry×GQA×non-dense` 判负 / ERRATA）均可证伪，且结论句明确「**不得读作多线不可能**」（§19）、「机制有效但代价抵消」≠「机制无用」（§17.17）——**无超证据表述**。

**⑤ 数字↔`results.json`（≥10，全过）。** W1：召回 32.59%(5340/16384)/32.95%(5399)、diff CI[−0.66,+1.38]pp、p=0.487、去伪影 −0.39pp、全深度 +0.49pp、tf4 synth −0.98pp / enwik8 −4.71pp、bpc 15.978/46.333、params +264,296(+1.845%)、`read_share 0.5417`、`read_scale_absmean 0.0592`、peak 786.6MB、free 19046/18041、eval 0.403min。W2：Δ+0.004186、bpc 1.705365/1.701179、loss 1.1221/1.5423/1.2146、gnorm 7.22/7.26、`visible_frac 0.749878`、hooks 8、params_added 0、attn 0.1696/0.2914。W3（对 2622/2626/2628）：0.0249→0.0361、±0.0144/0.0179/0.0218/0.0253、0.0074、0.0322、+0.1052、0.0346 等。

**⑥ §20 记账。** W1 `train_arms.sh` **按臂**取/放锁 + 臂间 65s（§20.1✓），跑前/跑后空闲 19046/18041MB（≥8GB），峰值 <5GB ⇒ §20.2 N/A；无外部降速（无 §20.3）。W2 逐臂释放 + ≥60s，窗口臂 5.910 GiB 超礼貌线按 **§20.2 三件套**记账（逐臂峰值 + 前后空闲 19407/18856 + 显式偏离，保 bs8）✓；步时先测（0.0434/0.0534 s/step）✓。对账：W1 143/0、W2 132/0；变异 W1 6/6、W2 5/5 抓红且逐字节还原。

---

## Finding

**Finding 1 · P1 — W1 headline「复现 100M 量级（0.9726）」用有利口径断言，与其自身预登记口径冲突。**
- **位置/证据。** committed `2629-carry-ragged-projfill/analysis.md §0/§5.3/§9`：「两臂都 ≥98% ⇒ **100M 量级（0.9726）在本尺度/步数下复现**」（synth 口径）；`results.json::exit_asset_holds.reproduces_100m_magnitude=true`、`rule` 用 **synth** ≥0.90。但同目录 **README §4(5) 预登记的操作性定义**写：「两臂 **`tf@4(enwik8)`** 都 ≥0.90 ⇒ 100M 量级（0.9726）在本尺度/步数下复现」；而 **README §4(2) 自己声明** enwik8 口径才是「**对齐 100M 的 0.9726**」。实测 **enwik8 口径 carry 臂 = 88.98%（<0.90）** ⇒ 按预登记口径「复现」**不成立**；synth 是**另一个分布**，不能「复现」一个 enwik8 口径的数。`catalog.yaml:1019` 与 `M1 README §5` 仍写「**并复现 100M 的 0.97 量级**」。
- **性质。** 正是本轮强调的**强断言纪律**违规：「复现」属强断言，缺**同口径 golden/CI** 支持，用词超出证据。
- **最小修法。** 把「复现」降级为「与 100M 锚点**同量级**（跨尺度、跨口径参照，非同一口径复现）」，并同步 `catalog.yaml:1019` 与 `M1 README §5`。**注：工作树已有一条未提交的并发编辑把 `analysis.md` 改成此措辞**（见 Finding 3），但 catalog / M1 README 尚未改。

**Finding 2 · P2（report-only）— G5 `CLOSURE.md §1.2/§3` 的配对 CI 与点估计**反号**。**
- **位置/证据。** `CLOSURE.md §1.2`：「机制净效应 `l1@K8 − repr01@K8 = +0.0144`（配对 CI **[−0.0158,−0.0130]** 不含 0）」。源 `2626/results_13m.json::criteria.mechanism_net` 里 `value = 0.0144` 但 `ci.delta = −0.014408`、`delta_ci=[−0.01579,−0.01303]`——即 CI 存的是 **(repr01 − l1)**，与 `value`(l1 − repr01) 反号。对比同文件主判据 `+0.0346（CI[0.0298,0.0407]）` 同号，可见此处为**不一致**。
- **影响。** 不改任何门槛判决（|CI| 均不含 0；`ordering=keep`、`passed=true`、四点轨迹与 §3 结论均指向 +0.0144），但照字面读会读出**相反方向**（「机制让 repr01 更差」），有误导风险。
- **最小修法。** 引成 `[+0.0130, +0.0158]`，或注明「CI 为 `repr01 − l1` 口径」。源头同在 2626（`analysis.md §1` 同病），可一并记 ERRATA。

**Finding 3 · P2（report-only，§22/§14b 编排观察）— 复核目标在工作树被并发改动。**
- **证据。** reviewer-launch 时 `watchdog_diff` = 干净（HEAD `866436d`）；随后工作树出现**未提交**改动：`2629-carry-ragged-projfill/analysis.md`（降级「复现」措辞 + 新增「强断言证据标注」表 + 引用**不存在的 `CALIBER_DECISION.md`**）、`2628-eng2-carry-ragged/analysis.md`（新增强断言标注表）；并新增**未跟踪**目录 `axes/.../2630-tf4-samecaliber/`。
- **含义。** 有第二条线在**同 cwd 并发写**被复核产物（§22「一个 cwd 一个写入者」/ §14b「复核须晚于执行」）。我的结论锚定 committed HEAD，故仍有效；但若该并发线未提交，Finding 1 的 catalog/README 残留与 `CALIBER_DECISION.md` 悬空引用需一并收尾。

**Residual（跨范围，供转派，不计入 W1/W2/W3）** — R1 复核的 **P1**（`2627-mechanism-jitter` 的 §17.17 恒等式 `净效应 = 集成增益 − clean 税`）**仅部分收尾**：`analysis.md §4` 已改成正确式 `(repr01 增益 − l1 增益) − clean 税`，但 `README §4.2`、`run.py:255`/`results_13m.json` 的 note 仍是旧式（代入 A 得 0.0444≠0.0144），且 `analysis §4` 指向一个**不存在的「README 修正注」**；`reconcile_numbers.py` 未加 R1 建议的那条恒等式 `ck`。属该线，非 W1/W2/W3 diff 引入，转派处理。

---

## Merge verdict

**OK with notes.** 六项重点 ①②③④⑤⑥ **实质全部成立**：W1 两口径已并列、越界腿标为辅助腿；W2 同收敛有 step+loss 双证据；W3 的 P2-a「误报」经索引/对象三重实证成立；G5 收口句与 6 条重开线不越证据；三线数字与 `results.json` 逐一对上；§20 记账完整。**1 条 P1**（W1 提交态「复现 100M 量级」用 synth 口径、与预登记 enwik8≥0.90 冲突，且 catalog/M1 README 残留）建议随收尾一并修正——该措辞**已在工作树并发被改正**，但尚未落到 catalog/README。2 条 P2 为 report-only（CLOSURE CI 反号；并发写观察）。**fail 不计状态灯**——本复核**无 P0**。

**请 supervisor 运行（我无 shell）：**
```
python3 -m pytest tests/ -q                 # 期望基线 573 passed
python3 tools/build_wiki.py --check
python3 tools/check_schema.py
```