reviewer:
## Review

复核对象 = 工作树上的 `axes/1-time/**` 事实文档 + 两个实验目录 + 两份判读报告（提交区间本身**看不了**，见「未检查项」）。
**能力限制（先声明）**：本 run **无 shell**；`watchdog_diff` 报「HEAD changed or is unavailable ⇒ committed ranges unsupported」。因此 (a) 提交 `4e06e2c` 的 diff 我没看过，只审了**现盘**文件；(b) `summary.py --check` / `check_tree.py` / `pytest` / `sha256sum` / `git` 全部未跑；(c) `needle_raw.json` 是**单行 12.8 MB**，read 工具对 >50 KB 的单行直接拒绝（`smoke/eval/needle_raw.json` 102.8 KB 同样被拒），所以**「自己从 raw 重建索引集与命中数 + 配对 bootstrap」我没做成**。替代办法与结果见 §「Correct」第 1 条。

## Correct
1. **判据数值我逐条手工复算过（不从叙述读），全部与 `results.json` 六位小数一致**，且它们全部可由「命中数 / n」推出，通道闭合：
   - 2661 硬子集（`needle_leg.hard_index_sets.n_intersection=2699`）：`carrypl16_s0` 276/2699=0.102260、`win16_s0` 19/2699=0.007040、`win16_s1` 27/2699=0.010004、`carry16_s0` 43、`carry16_s1` 53、`carrypl16d_s0` 77（`results.json:10518`、`:1829`、`:1219`）；各自 Wilson CI 我用公式重算逐位相同（276/2699 → [0.091389, 0.114262]）。
   - 跨臂对撞：`−win16_s0` = 9.522pp、`−win16_s1` = 9.226pp（0.102260−0.010004=0.092256 ✓），CI [8.336, 10.708]/[8.023, 10.429] 我用 **diff ± 1.96·√(p₁q₁/n+p₂q₂/n)** 重算**逐位**相同（9.522±1.186）。
   - 整体 beyond 桶：`carrypl16_s0` 植入 0.520665 − 地板 0.456835 = +6.383pp、CI [4.854, 7.912]（=6.383±1.529 ✓）；implicit 两臂 +0.049pp（CI 含 0）/−0.135pp（CI 含 0）——负对照方向正确（`results.json:11011`）。
   - 2662：`price_bpc` = 1.622527−1.628110 = **−0.005583**、1.621799−1.629093 = **−0.007294**（`results.json:334-350`）；`|price|` 均 > σ_cell 0.000983 且 < 0.014（两带并列、未混用 ✓）；c3 beyond 桶 4/7175 与 −11/7175 → CI 两 seed 都含 0 ✓。
   - `Δbpc_far`：2661 `carrypl16_s0` +0.032965、CI [0.026986, 0.038727]（`results.json:1092-1099`），2662 变体 B 逐位 0.0（`results.json:1963`）。
2. **判据⑤（与 2638/2639 落盘值逐位复现）我自己从盘上读过被比对方**：`2638/results.json:813,904` = win16_s0 **1.62811**、carry16_s0 **1.624979**；`2639/results.json:1638,1670` = carrypl16_s0 **1.527595**、carrypl16d_s0 **1.585531**；`2634/results.json:324,329,458` = 1.62811 / 1.624979 / `delta_bpc_far` 0.015784。与 `reproduction_check` 的四条期望**逐位相同**，`all_equal=true`、`differences={}`。这条门**可失败**（ticket §10.23 记录的缩批版 rc=1 是真反例）。
3. **索引集的构造路径我按代码核过**：`run.py:449-462` 用**六臂** beyond 桶 control 记录调 `ns.hard_index_sets`，`recall_on_indices(bplanted[n], idx)` 的索引对植入/控制同序（`run.py:435-438` 先过滤桶、顺序保持；`phrase_needle_batch` 同 generator 序列 + `meta==meta0` 硬断言），所以 `intersection` 语义（六臂 control 全答不出 ⇒ 该子集地板恒 0）成立。`per_arm_control_miss` 六臂齐（`results.json:1890-1898`）。
4. **「sink 与锚点在本仓是同一根杠杆」有据**：`pathlm/depth_ar.py:1160` `keep[i,j] = (j≤i) ∧ ((i−j)<window ∨ j<anchors)` ⇒ `anchors` = **最前面 N 个位置恒可见**，与 StreamingLLM 式 sink 同构。`1.1.1.failure_analysis` ② 的这句**成立**。
5. **判据带纪律合规**：`criteria.bands_note`（2661）与 `quality_leg.bands.note`（2662）都写明「bpc 两带并列不混用；率类只走 bootstrap/Wilson(两比例)CI，不跨量纲借 0.014」；实际使用也如此（召回一律 pp+CI，0.014 只出现在 bpc 量级对比里）。`Δbpc_far` 的 CI 是真·按窗口配对 bootstrap（`realtext_probe.py:80-98` 逐簇重采样 `diff_bits/n`），bpc 配对 CI 走 `paired_cluster_bootstrap_ci`（`run.py:566`）。**没有**把 0.014 借给率类。
6. **「该涨/该跌/必须不变」逐条与实测方向一致，无一条方向反了**：
   - 2661 ①该涨→10.226%（Wilson 下界>0，跨臂 +9.522/+9.226pp）**过线**；②该跌→implicit 两臂 CI 含 0 **过线**；③该涨→+0.032965 CI 下界>0 **过线**；④必须不变→implicit `max|Δlogit|=0.0` **过线**；⑤必须不变→逐位复现 **过线**；⑥必须不变→六臂前缀 0.0、后半段抓力 22.8–28.2 **过线**；⑦诊断已报；⑧禁外推已写。
   - 2662 ①该涨（或如实报税）→两 seed 同向收益、越本格带不越 0.014 **过线**；②该涨→s0 只有配对 bootstrap 过、s1 两套 CI 都含 0 ⇒ `both_seeds_pass=false` **未过（如实记「不可判」）**；③该跌→两 seed CI 含 0 **过线**；④必须不变→变体 B 0.0、变体 A 敏感且 `premise_holds=false`（预登记已把这条物理事实写死）**过线**；⑤⑥必须不变 **过线**。
7. **口径归因裁决（任务第 4 项）成立**。lead 的处置（以预登记主对撞为准、把更严读法并列登记）**我判成立**，理由三条：① 2661 自己的预登记 `README.md` §4/§4a 把主对撞写死为 `carrypl16_s0` vs `win16_s0`，并把 `carrypl16d_s0`（γ=0.99）与 `carry16_*` 明确归入 `explicit_complete`/`explicit_degraded` 的**对照**角色；② `judgement.intervention_arms` 是机械派生（`run.py:474` = 所有 `carry_rule` 非空臂）⇒ 把预登记里本是「退化对照/降级对照」的臂塞进「干预臂全过」集合，等于判「对照组没做出效果」；③ 它用的还是**整体 beyond 桶（n=8178）**而不是判据①要求的**同索引硬子集（n=2699）** ⇒ 它不是「同一判据的更严版本」。**未发现夸大**：`conclusion` 明写「按它读则**不过线**」，`gaps` 也有对应条目。

## Fixed
未做任何修改（本 run 只读）。

## Finding

**F1 · P1 · `1.1.4.5` 的 `gaps` 与自己的 `status/conclusion` 互相打架（§11.2 兄弟字段未对平）**
`axes/1-time/1.1.4.5-implicit-vs-explicit/project.yaml:26` 仍写「缺 implicit 与 explicit 在同任务/同预算/同仪器下的正面对撞」，`:29` 仍写「explicit 半边依赖 1.1.4.1 的修复（逐层状态、前沿）」；而同一文件 `:69 status: "已完成"`、`work_packages.WP1.status: "已完成"`、`:17 conclusion` 说「对撞已判（2661）」「逐层机制已交付（2639）」。**违反** `docs/research-schema.md` §11.2（改动后返回前必须把同一事实在兄弟字段逐处对平）+ `AGENTS.md` §2.1（旧说法不得留成「仍是事实」）。
最小修法：删掉 `:26`、`:29`（其内容已被 `gaps` 末两条与 WP2 的两条取代），或把 `:26` 改写成残留口径（例如「本实验未含 `full` 臂；stock 全因果锚取自同族 2638/2639」）。
复现：`sed -n '25,33p' axes/1-time/1.1.4.5-implicit-vs-explicit/project.yaml` 对照 `:17` 与 `:69`。

**F2 · P1 · `1.1.1` 的 `gaps[0]` 把已消的缺口留成当前事实**
`axes/1-time/1.1.1-hard-window/project.yaml:33`「无仓内自包含复算（WP1：旧证据仍是 .scratch 摘要 + 旧格 stub）」与同文件 `conclusion_review`「旧「未经本口径复现」的缺口已消」、`conclusion`「（本目录首次自包含）」、`WP1.evidence`「锚点臂 seed 0/1 已自包含复算，rc=0」直接冲突。§11.2 / §2.1。
最小修法：改写为残留项「窗口臂 k∈{16,17,18,24,32,128} 仍为复用（2634/2638），未在本目录自包含复算」。
复现：`sed -n '32,40p' axes/1-time/1.1.1-hard-window/project.yaml`。

**F3 · P1 · `1.1.4.1` 的 `gaps[1]` 与「召回轴已补」冲突，并把已完成的事写成下一步**
`axes/1-time/1.1.4.1-explicit-memory/project.yaml:53`「**milestone 字面未达标**：可检索性（召回）轴**未在完备臂上测**…下一步最小实验 = 复用 2634 的毁源/needle 仪器在完备臂上重跑（纯推理 ≈ 10 min）」——而 2661 做的**正是**这件事（`2661/README.md` §3 声明仪器为 2634 逐字节副本、臂含 `carrypl16_s0`），同文件 `conclusion` 说「**召回轴（本轮补上）**」，`gaps` 倒数第二条说「召回轴已补…」。§11.2 / §2.1。
最小修法：把 `:53` 改成残留口径「仅一个预算点（393,216 B）有召回轴数字；2639 的其它预算点未叠加」。
复现：`sed -n '51,60p' axes/1-time/1.1.4.1-explicit-memory/project.yaml`。

**F4 · P1 · 事实文档把 Wald 区间称作「Newcombe」，且 2662 是「预登记统计量 ≠ 交付统计量」**
`tools/evalkit/carry_channel.py:41` docstring 写「Newcombe 独立比例」，但 `:59,64-65` 的实现是 `diff ± 1.96·√(p_a q_a/n_a + p_b q_b/n_b)` = **正态近似（unpooled，Wald）**。手工对拍：2661 硬子集 276/2699 vs 19/2699 → 记录 [8.336, 10.708] = **Wald**，真 Newcombe = **[8.366, 10.749]**；2662 锚点桶 s0 483/1003 vs 457/1003 → 记录 [−1.774, 6.958] = **Wald**，真 Newcombe = **[−1.773, 6.943]**。而 `2662/README.md` §4 判据②**预登记的就是「CI 用 Newcombe 两比例差」**；`2662/analysis.md` §0/§2/§4、`1.1.4.5/project.yaml:17`、`1.1.1/project.yaml` 的 `conclusion` 都用了「Newcombe CI」这个词。**结论不受影响**（两法在每一个被判的格子上同号、同「含 0 / 排除 0」）。
最小修法（0 GPU）：把两处 `analysis.md` 与两份 `conclusion` 的「Newcombe」改成「两比例差（正态近似、未 pooling）CI」，并在 `2662/README.md` §5a 记一句「预登记的 Newcombe 以 Wald 变体交付，两种口径在所有判定格同结论」；若要真 Newcombe，则需在实验目录的 `needle_stats.py`/`two_proportion` 加实现并**重跑评测**（≈3 min GPU/实验）。
复现（代跑）：
```bash
python3 - <<'PY'
import math
def wilson(k,n,z=1.96):
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d
    h=(z/d)*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return c-h,c+h
def wald(ka,na,kb,nb):
    pa,pb=ka/na,kb/nb; se=math.sqrt(pa*(1-pa)/na+pb*(1-pb)/nb)
    return (round((pb-pa-1.96*se)*100,3), round((pb-pa+1.96*se)*100,3))
def newcombe(ka,na,kb,nb):
    l1,u1=wilson(ka,na); l2,u2=wilson(kb,nb); pa,pb=ka/na,kb/nb
    return (round((pb-pa-math.sqrt((pa-l1)**2+(u2-pb)**2))*100,3),
            round((pb-pa+math.sqrt((u1-pa)**2+(pb-l2)**2))*100,3))
print("2661:", wald(276,2699,19,2699), newcombe(276,2699,19,2699))
print("2662:", wald(483,1003,457,1003), newcombe(483,1003,457,1003))
PY
```

**F5 · P2 · 2661 判据① 是「必要条件门」，本身不具判别力，文档未标**
判据① 的判定线（在 floor≡0 的 `intersection` 上要求 Wilson 下界 > 0）**任何召回 > ≈0.2% 的臂都会过**：`carry16_s0` 43/2699 = 1.593%，Wilson 下界 0.011849 > 0（`results.json:10500-10507`）⇒ 退化的单共享槽实现同样满足判据①。真正有判别力的是 ① 的跨臂部分（`c1b`，+9.52pp）与 ②。`analysis.md` §2 ① 与 `1.1.4.5.conclusion` 把 ① 写成「**过线**」而未标这一点。最小修法：在这两处各加一句「① 是必要条件门（地板恒 0，任何 >0 召回都过）；判别证据是 `c1b` 的跨臂差」。证据/复现：见上条 `results.json` 行。

**F6 · P2 · 「更严聚合口径」的描述不够准确，容易被读成同一判据的严版**
`2661/analysis.md` §2 末段与 `1.1.4.5/project.yaml:17` 把 `judgement.intervention_arms_all_pass=false` 称为「更严的聚合口径（四条带通道臂全过才算过）」。事实上它 (a) 用的是**整体 beyond 桶**（n=8178），不是判据①的**同索引硬子集**（n=2699）；(b) 的臂集来自机械派生（`run.py:474`）。最小修法：改称「另一口径的机械聚合（含预登记里已定位为对照的臂、且统计在整体桶上）」。裁决本身不受影响（F7 条理由见「Correct」）。

**P2（一并登记的次要项）**：① `2662` 目录 `MODEL.md` 未落盘（README §8 的交付清单里列了它）——已在 `1.1.1.gaps` 与 `2662/analysis.md` §9 如实登记；② 复制的仪器 `2661|2662/needle_stats.py` docstring 仍写「**四臂** control 全答不出」，实际调用传的是六臂（`analysis.md` 写「六臂」是对的，属拷贝遗产，非实现错）。

## 未检查项（如实登记）
1. **未能用 `needle_raw.json` 重建索引集/命中数，也未做配对 bootstrap**（无 shell；read 工具拒绝 12.8 MB 单行）。已用「代码路径核查 + 从 hits/n 精确复算 + 与盘上 2634/2638/2639 对拍」替代，但「命中数本身」（276/19/27/43/53/77）**我未独立重建**。请 lead 代跑：
```bash
python3 - <<'PY'
import json
P="axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2661-implicit-vs-explicit-recall-13m/needle_raw.json"
R=json.load(open(P)); A=["win16_s0","win16_s1","carry16_s0","carry16_s1","carrypl16_s0","carrypl16d_s0"]
b=lambda a,k:[r for r in R[f"{a}|{k}"] if r["bucket"]=="beyond"]
miss={a:{i for i,r in enumerate(b(a,"control")) if r["hit"]==0} for a in A}
inter=set.intersection(*miss.values()); print("n_inter",len(inter),"n_union",len(set.union(*miss.values())))
for a in A:
    p=b(a,"planted"); print(a, sum(p[i]["hit"] for i in sorted(inter)))
PY
```
   （期望：n_inter=2699、n_union=5872；命中 19/27/43/53/276/77。）
2. **提交区间**：`4e06e2c` 及其前后 diff 未审（`watchdog_diff` 拒绝、无 git）。若需覆盖，请提供 `git show --stat 4e06e2c` 与 `<prev>..HEAD --stat` 的文本产物。
3. **门与自检未跑**：`python3 tools/summary.py --check`、`python3 tools/check_tree.py`、`python3 -m pytest tests/ -q`、两个实验目录的自包含单测（若有）。另：`sha256sum` 未跑 ⇒ `MODEL.md`/`results.json.code_hashes` 与盘上 `run.py`/checkpoints 的指纹一致**未验**。
4. `2636-longT-unified` 的「同 step 四点 Δ 全在 0.009459 内」（1.1.1 ③/④ 的证据）我只读了 `project.yaml`/`WP4` 的转述，**未打开该目录的 `results.json`** 复算。
5. 缩批版 rc=1 的过程证据（`run_log.txt` 里的失败行）我未逐行核。

## 逐条状态判词
- **`1.1.4.5` `status: "已完成"`**：**站得住（有条件）**。`milestone.acceptance`（两臂同仪器同预算 + implicit 停地板 + explicit 可排序提升 + 区分设计/机制）逐条达成，预登记 `falsification` 两支未触发，读数与 2634/2638/2639 逐位复现；字面缺口的**残留**（完备变体 n=1、本实验未含 `full` 臂）已在 `conclusion` 的「两读法」里如实写明。**条件 = F1 必须修**：目前该文件的 `gaps` 第 1/4 条恰好写着「缺这个对撞」，与 `已完成` 自相矛盾；修掉后状态即可站住。
- **`1.1.1` `status: "进行中"`（状态不变）**：**站得住**。主正面主张（锚点桶召回）在 n=2 下不可判（s0 只有配对 bootstrap 过、s1 两套 CI 都含 0），WP1/WP3 未完成、本目录 KV 消融保留曲线未复算——`conclusion`/`conclusion_review`/`gaps` 三处口径一致且无过度声明。**但 F2 需修**。
- **`1.1.4.1` `status: "进行中"`（状态不变）**：**站得住**。milestone 字面（每预算点两轴 + 业界对齐 + 量化点同图）确实未达，`conclusion` 明确「本轮只交付质量轴 + 窗口点 + full 锚」，召回轴由 2661 以**指针**方式补上（`experiments` 已列 2661）。**但 F3 需修**。

## 判词
**FAIL（窄范围）**：**科学主张与全部数值 = PASS**（26 个以上判据数字我逐条复算与 `results.json` 逐位一致；判据⑤与 2634/2638/2639 **盘上落盘值**逐位相等，我读了被比对方；方向/口径/两带纪律/仪器的真·配对 bootstrap 均核查通过；口径归因裁决成立且未夸大）。**窄范围 FAIL 归因 = F1–F4 四条 P1 全属事实文档/统计命名的「账实不符」类缺陷**（同一 SSOT 内旧说法与 `status/conclusion` 打架 ×3；把 Wald 叫 Newcombe 且 2662 预登记统计量 ≠ 交付统计量 ×1），不涉及任何测量数字、不涉及结论方向，**无需重跑 GPU**；F1–F3 各 1–3 行文本即可修，F4 建议「文档改名 + README §5a 记一句」而不重跑。修完 F1–F4 后我判 **PASS**。