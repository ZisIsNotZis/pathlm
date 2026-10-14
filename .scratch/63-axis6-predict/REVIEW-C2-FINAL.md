# REVIEW-C2-FINAL

- 角色：轴 6 最终独立复核者（fresh 上下文、**未参与写作**、**只读**；仅新增本文件）
- 时间戳：2026-10-14T23:08:10+08:00（`date -Iseconds`）
- 被审 revision：`18a87075a1ff2c7997361bb6e71c59f9ed57f132`（`git rev-parse HEAD`）
- 边界：未改任何被审文件、未 `git commit`；`git diff --cached --name-only | wc -l` = 0（无暂存）
- 纪律：两个判词独立给；每条命令与真实输出见文末；未执行项列入 not-run。

---

## 判词 1 · 6.3.2（WP1 = `2671-padfill-13m`）

### VERDICT: PASS

- 收尾 worker 对 `analysis.md §6「否证条件逐条」` 的修改**自洽、且与 `results.json` 逐臂逐 seed 原始读数相符**（我自写 `results.json` 复算脚本独立复现，未 import `padfill.py` 判据函数）。
- 本判词 PASS 只针对「§6 修改正确性 + 必查项」。另登记 1 条 **normal**（SSOT 复核报告指针悬空）与 2 条 info，均属文档一致性、非科学缺陷；§6 本身无 blocker。

### blocker

- 无。

### normal

- **N1（SSOT 声称的独立复核报告路径不存在）** — 位置：`axes/6-predict/6.3.2-pad-fill/project.yaml:18`（`conclusion_review`）与 `:47`（WP1 `review`）。
  - 违的判据：`change.upkeep.resolvable`（「每一个引用都可核对…对现存物解析」）+ `proof.review`（「复核缺失即为阻断」）——本题判据 ④「指针」。
  - 事实：两处都写「**已做**：`.scratch/63-axis6-predict/REVIEW-C2-632.md`（fresh、未参与写作、只读、带 shell）」，但该文件**在工作树与 git 中都不存在**（`ls .scratch/63-axis6-predict/` 无此项；`find .scratch -iname '*632*'` 只命中写这份 project.yaml 的脚本 `apply_632_final.py`；`git log --all -- …/REVIEW-C2-632.md` 空）。该字符串由收尾脚本 `apply_632_final.py:48,55` 首次写入，即引用对象从未落盘。
  - 复现证据：
    ```
    $ ls /home/z/vibe/pathlm/.scratch/63-axis6-predict/ | grep -i 632
    apply_632_final.py
    $ grep -rn "REVIEW-C2-632" /home/z/vibe/pathlm --include=*.yaml
    .../project.yaml:18  .../project.yaml:47
    ```
  - 定性：**文档一致性缺陷**（非科学缺陷）。所有被引数字本身已由我独立复现（见下），故不推翻任何结论。
  - 处置建议：把该指针改为**本次复核的实际报告**（`.scratch/63-axis6-predict/REVIEW-C2-FINAL.md`）或补写 `REVIEW-C2-632.md`；否则 `project.yaml` 对「已完成独立复核」的断言不可核。**若父级要求 review 字段严格可解析，此项应升级为 blocker**——本复核者按「该 review 义务已由本交付满足、缺陷仅是指针命名」取 normal。

### info

- **I-a（收敛档参照无就地出处）**：`project.yaml:17` conclusion (a)「参照 = AR 全位置 teacher-forced bpc 2.2704（快照刻度；**收敛档为 ≈1.48**）」。`2.2704` = 两 seed `ar_full_bpc` 均值（我复算 2.27037，见下，✓）；`≈1.48` 在本格目录内无出处，最接近的仓库数字是 `6.2/2610-frozen-trunk-jumphead-13m` 的 clean bpc **1.4836**（13.5M、12k 步、seq256）——与「6000 步档」并非同一档。建议把该锚写明来源或标「记忆·未核」。不构成数字错误。
- **I-b（墙钟键扫描的自指豁免）**：`results.json::_wall_clock_fields=[]` 为真；但我对整个 JSON 做 key 扫描时会命中 `._wall_clock_fields` 这一**自指键**本身（其值为 `[]`）。即「0 命中」指**真实墙钟字段**为 0，与 2660/2670 的做法一致，成立。

### 必查项逐条（我亲自执行）

| 项 | 结果 | 证据 |
|---|---|---|
| §6 修改与原始读数相符 | **通过** | 见下方「独立复算」 |
| ① L5 确定性两项是否 None + 是否如实写 | **通过** | `L5.eval_rerun_identical=None`、`short_train_20step_checksum_identical=None`；`analysis.md` L5 段与 `project.yaml` (f) 均写「**未通过 / 未验证**」，并给出 `GPUUnavailable(...)` 原文与抖动说明，未含糊 |
| ② 禁墙钟 0 命中 | **通过** | 独立键扫描仅命中自指键 `_wall_clock_fields`（值 `[]`）⇒ 真实墙钟字段 0 |
| ③ 5 臂 × 2 seed 齐备 + S=1500 + 快照口径声明 | **通过** | `ar{0,1}` + 4 个 fill 臂各 `{0,1}` 全在、均 `done`；`config.steps=1500`；`analysis.md` 首段明写「S = 1500，与仓库既有 6000 步档不可比」 |
| ④ project.yaml vs analysis.md 兄弟字段一致 | **通过（数字全对平）** | 4 臂 Δ+CI、`+0.1635`、`2.2704`、`+1.07…+2.37`、非单调 `False`、硬门 `0.0`/两条红、`≤` 带声明——两处逐值一致；唯一指针问题见 N1 |
| ⑤ README mtime 早于 results.json | **通过** | `README.md` mtime `1791982317` < `results.json` `1791990296` |

### 独立复算（自写 `/tmp/rc632b.py`，未 import `padfill.py`）

方法与落盘口径对齐（逐 block `Δ=(s_fill_M−s_ar_M)/(n_m·ln2)`，两 seed 逐 block 先对齐求均值，再 block 聚类配对 bootstrap B=2000 seed=2673 percentile95）：

```
fill_bidir_r10  my_mean=2.145280 my_ci=[2.04002,2.25005]  ref=2.145280 [2.04002,2.25005] match=True
fill_bidir_r25  my_mean=1.072639 my_ci=[1.00339,1.14548]  ref=1.072639 [1.00339,1.14548] match=True
fill_bidir_r50  my_mean=2.373503 my_ci=[2.28213,2.47203]  ref=2.373503 [2.28213,2.47203] match=True
fill_causal_r25 my_mean=1.236174 my_ci=[1.18219,1.29026]  ref=1.236174 [1.18219,1.29026] match=True
L3 recompute mean=0.163534 ci=[0.10964,0.21400]  (stored 0.163534 [0.10964,0.21400])
ar_full seeds [2.18422, 2.35653] mean=2.27037
```

- 4 臂 Δ 与 CI **逐 5–6 位小数与 `results.json` 相同**；L3 配对 Δ 亦相同；`ar_full` 均值 2.27037 → 文中 2.2704 成立。
- §6 三 cond 与读数对账：**cond_A 未触发**（`CI 全不含 0=True`、`|Δ| 全越 0.014=True`、`r 单调恶化=False` ⇒ 子条件不全满足）；**cond_B 未触发**（无任一 bidir `|Δ|≤0.014`、无 CI 含 0）；**cond_C 未触发**（`fill_bidir_r25−fill_causal_r25` CI `[0.10964,0.21400]` 不含 0）。`analysis.md` 摘要（「越带、CI 不含 0」）与 `project.yaml::conclusion`(a)–(f) 的每个数字均与上述一致；**未复现出**背景所述「摘要 vs §6 不可分」的旧矛盾——该矛盾已被本次修改消除，改后自洽。
- 附：`causal` 代价 `+1.2362` > `bidir_r25` `+1.0726` ⇒ §6 的「causal 更差=True」成立。

---

## 判词 2 · 6.3.3 的复核 #C2 返工（只审「修复是否到位、有无新矛盾」）

### VERDICT: PASS

- `.scratch/63-axis6-predict/REVIEW-C2-633.md` 的 **B1 / N2 / N3 / N4 / I3 / I4 / I5** 逐条核验：**均已落地且与落盘数据相符**；返工脚本 `apply_633_fix2.py` 的**自述未被采信**，我用 `grep`/`read` + `results.json` 落盘值逐条复核。
- 未发现**本次编辑引入的新矛盾**。登记 1 条 **normal**（`conclusion_review` 判词指针过期）与 2 条 info，均非本次编辑造成、不推翻科学结论。

### blocker

- 无。

### normal

- **N2′（`conclusion_review` 的「当前（最后一轮）独立复核判词」已过期）** — 位置：`axes/6-predict/6.3.3-latent-regression/project.yaml:18`。
  - 违的判据：`change.upkeep` / 判据「不得保留与终局矛盾的旧话」「复核判词写当前」——本题判据「conclusion_review 是否只写当前判词」。
  - 事实：该字段写「**当前（最后一轮）独立复核判词**：`.scratch/63-axis6-predict/REVIEW-C1-2670.md` … 判 **VERDICT: PASS**」。但对同一批被审文件其后已有 `.scratch/63-axis6-predict/REVIEW-C2-633.md`（`VERDICT: FAIL`，且正是本次返工的来源）。因此「最后一轮 = C1 PASS」的表述与复核史矛盾：最新一轮是 FAIL（已返工、待再复核）。该字段只写一条判词（无多轮累积，合规），但该条已非当前。
  - 复现：`grep -o "REVIEW-C1-2670.md\|REVIEW-C2-633.md" project.yaml | sort | uniq -c` → `REVIEW-C1-2670.md`×2、`REVIEW-C2-633.md`×1；后者仅出现在 `change_log`，未进入 `conclusion_review`。
  - 定性：**文档一致性缺陷**。返工脚本的声明的修复范围是 B1/N2/N3/N4/I3/I4/I5，**未含**此指针；故此项属**遗留**而非本次新引入。建议把 `conclusion_review` 的当前判词改为 C2-633（含「已按 findings 返工」）或将当前判词留空并标注「待 C2 再复核」。

### info

- **I-1（`failure_analysis` 的退化统计数字是「均值」，与 conclusion 的「min–max」不同口径）**：`project.yaml:16` failure_analysis 写「`lat_k2_lam1` 的流最长连续相同 token **平均 ≈137**」；`project.yaml:17` conclusion (f) 写「**均值 99.8–153.8（四个窗口）**」。落盘四窗口均值 = **99.833 / 153.833 / 151.583 / 143.500**，其**均值 = 137.187 ≈ 137**、**min–max = 99.8–153.8**。两处**都正确**（一个报均值、一个报区间），不构成矛盾；仅口径标注不同，若无碍可保留。
- **I-2（真值锚 Δ 区间在 `failure_analysis` 与 `conclusion` 各出现一次）**：I4 的目标是消除 `conclusion_review` 里与 `failure_analysis` 重复的那份；现 `conclusion_review` 已改为指针式（「数值见 `conclusion` (f)」，✓），但 `failure_analysis`（`:16`）与 `conclusion`（`:17`）仍各含一次 `+1.1701…+5.0708`。非矛盾、非阻塞；若追求「同一事实只一处」，可将 `failure_analysis` 的数字也改为指针。

### 逐条核验（位置 + 落盘证据）

| Finding | 状态 | 证据（我亲自读/复算） |
|---|---|---|
| B1 conclusion (e) 旧「free-run 序列级 bpc 未测」分句 | **已修** | `project.yaml` 内 `grep "free-run 序列级 bpc 未测"` 唯一命中在 `:91` **change_log**（返工说明），`conclusion` (e) 现为「(e) 未测：只有 13M 冻结主干一种规模、…」，句子仍成立；(f)/gaps 的「已测」不再被打脸 |
| N2 真值锚 Δ 下界 | **已修且正确** | `failure_analysis` 现「+1.1701…+5.0708」；落盘 `truth_anchor.delta_external_tf`：`lat_k2_lam1/13m_s1`=**1.170113**、`disc_k2/13m_s1`=**5.070835** ⇒ min–max = 1.1701…5.0708 ✓ |
| N3 退化统计格式 | **已修且正确** | conclusion (f) 现「均值 99.8–153.8（四个窗口）」；落盘 `degeneracy.per_stream.lat_k2_lam1` 四窗口 `mean.max_identical_run` = 99.833/153.833/151.583/143.500 ⇒ min 99.8、max 153.8 ✓ |
| N4 analysis.md「自评 Δ」指代不明 | **已修** | `2670…/analysis.md:484` 现「比**自评 free-run Δ** 更大（是；注意不是 teacher-forced 自评列：`lat_k2_lam1 / 13m_s0` 外部锚 +1.2611 略小于 teacher-forced 自评 +1.4521）」——与落盘两列值一致 |
| I3 roadmap 完成日期 | **已修** | roadmap 末条「收口：conclusion / status / 复核判词齐备」（无日期） |
| I4 判词数字两处各写 | **已修（conclusion_review 指针式）**；残一处见 I-2 | `conclusion_review` 现「数值见 `conclusion` (f)；候选臂流有重复退化」 |
| I5 执行清单轮次标注 | **已修** | `conclusion_review` 现「该复核者**在 C1 轮**亲自执行了 pytest（11 passed）…」 |
| 新矛盾检查 | **未发现** | B1 删分句后 (e) 仍成句；(e)/(f)/gaps 不再互斥；`failure_analysis` 与 `conclusion` 数字一致（1.1701…5.0708、137 均值 vs 99.8–153.8 区间，见 I-1）；见 N2′（遗留指针） |

---

## 亲自执行的命令与真实输出（节选）

```
$ python3 /tmp/rc632b.py
fill_bidir_r10  my_mean=2.145280 my_ci=[2.04002,2.25005]  ref=2.145280 [2.04002,2.25005] match=True
fill_bidir_r25  my_mean=1.072639 my_ci=[1.00339,1.14548]  ref=1.072639 [1.00339,1.14548] match=True
fill_bidir_r50  my_mean=2.373503 my_ci=[2.28213,2.47203]  ref=2.373503 [2.28213,2.47203] match=True
fill_causal_r25 my_mean=1.236174 my_ci=[1.18219,1.29026]  ref=1.236174 [1.18219,1.29026] match=True
L3 recompute mean=0.163534 ci=[0.10964,0.21400]  (stored 0.163534 [0.10964,0.21400])
ar_full seeds [2.18422, 2.35653] mean=2.27037

$ stat -c '%Y %n' README.md results.json     # 2671 目录内
1791982317 README.md        ← 早于
1791990296 results.json

$ python3 -c "…wall-clock key walk…"
WALL_CLOCK_HITS= 1 ['._wall_clock_fields']     # 仅自指键，其值为 []
L5 vals: None None ; steps= 1500 ; arms=4 fill + ar, 各 2 seed ; L0/L0b=True ; L2 monotonic=False

$ grep -n "free-run 序列级 bpc 未测" 6.3.3…/project.yaml
91:    change: "…B1 去掉 conclusion (e) 里…「free-run 序列级 bpc 未测」分句…"   # 仅 change_log

$ ls .scratch/63-axis6-predict/ | grep -i 632
apply_632_final.py          # REVIEW-C2-632.md 不存在
```

## not-run 清单（未执行，故不构成通过）

- **未重跑 6.3.2 的端到端训练/评测**（需 GPU 锁；时间盒 ≤14 min、禁 GPU）：L5 的「确定性」我**未**独立两遍比对，`None` 由落盘自报 + 静态核对。
- **未亲自跑 `mutate.py` / `pytest`**（6.3.2 的 `mutation_report.json`「6/6 抓红 + 逐字节还原、25 passed」为落盘自报，我只读了报告与 `evidence/`，未重执行验证）。
- **未核上游 `2631`/`2660`/`2610` 的原始数字**（6.3.3 真值锚所依赖的头权重与主干；只读了 2670 自身的落盘）。
- **未核 `收敛档 ≈1.48` 的确切出处**（只定位到最接近的 2610 clean bpc 1.4836，见 I-a）。
- **未跑 `tools/summary.py --check` / `check_tree.py`**（不属本任务 1–5 项；避免仓库遍历）。
- **未复核 6.3.1 / 其它轴边界**（超范围）。

## 一句话结论

- **判词 1**：§6 的修改**自洽且忠实于原始读数**（4 臂 Δ/CI 与 L3 逐值独立复现），必查项全通过 ⇒ **PASS**；唯一 normal 是 `project.yaml` 指向不存在的 `REVIEW-C2-632.md`（文档一致性，非科学缺陷）。
- **判词 2**：C2 的 7 条 findings **全部修好且与落盘相符**、未引入新矛盾 ⇒ **PASS**；遗留 normal 是 `conclusion_review` 仍把 C1-2670 标为「最后一轮」判词（应改指 C2-633）。
