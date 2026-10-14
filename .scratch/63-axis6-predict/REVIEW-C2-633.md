# REVIEW-C2-633

- 角色：轴 6 独立复核者（fresh 上下文、未参与写作、只读；仅新增本文件）
- 时间戳：2026-10-14T22:19:26+08:00（`date -Iseconds`）
- 被审 revision：`803783915266319a14fdcdaac557cdade70778df`（`git rev-parse HEAD`，复核期间 HEAD 由并发写入者推进：起审时 `e8f0d423…`，结束时 `8037839…`）
  - 被审对象以 **worktree 字节**绑定（全部为未跟踪 / 未提交状态）：
    - `axes/6-predict/6.3.3-latent-regression/project.yaml` = `4f277a9516dd3498a28160d6fd13bed365e69c25002f25b3a22c4c00411e12dd`
    - `…/experiments/2670-freerun-bpc-13m/results.json` = `991e692405a1f84a4a0024cd4b460ad0493aba2f0e6fa5500370089cc6dc17de`
    - `…/experiments/2670-freerun-bpc-13m/analysis.md` = `40e89098ac015963fc6f20c4f075eac64efba354640694bf35db26ba7ce51a33`
    - `…/experiments/2670-freerun-bpc-13m/freerun.py` = `e6857a83f5e640f9d471401cc5d3623d479e7bc57c436e52831bf5c9f43f7ada`
  - 与 C1 轮相比三者均已变：`results.json` 追加 `truth_anchor` / `streams` / `degeneracy`（C1 记 `263bd8d4…`）、`analysis.md` 追加「真值锚」节（C1 记 `ac379202…`）、`freerun.py` 单测由 11 项增至 13 项（C1 记 `b651eaf0…`）。

## VERDICT: FAIL

- **缺陷性质：文档 / SSOT 一致性缺陷（不是科学缺陷，也不是基础设施阻断）**。WP2 的执行、数字与证据链**全部通过独立复算**（见下）；判 FAIL 的唯一 blocker 是 `project.yaml::conclusion` 中一条与终局事实直接矛盾的旧话——它把一个**已测**项写成**未测**，而同一字段下一句 (f) 正给出该读数。
- 修复成本：一处子句改写（删除「自由跑（free-run）序列级 bpc 未测」或限定为「WP1 范围内，已由 WP2 的 (f) 补上」），**无需重跑任何实验**；改后需按 `docs/research-schema.md` §11.2 重扫兄弟字段。

## 分级清单

### blocker

- **B1（`conclusion` 内自相矛盾：已测项仍写「未测」）** — 位置：`axes/6-predict/6.3.3-latent-regression/project.yaml:17`（`conclusion` 字段）。
  - 违的判据：`docs/research-schema.md` §11.2「同一事实在各兄弟字段逐处对平…同一事实不在两处各写一遍」+ §11.1 的收口理由（旧结论与正文冲突 = 收口级「账实不符」）；本复核任务判据 B「不得保留与终局矛盾的旧话」、C「（未测 / 已测）不得错置」。
  - 事实：该字段 (e) 仍写「**（e）未测：自由跑（free-run）序列级 bpc 未测（只有 teacher-forced 命中率 / CE 与草稿接受率）**」，而同一字段 (f) 写「路径 (i) 的 free-run 序列级质量代价明确（WP2 / `2670-freerun-bpc-13m`）」并给出 `ar` = 0.504356 / 0.489330、Δbpc 与 CI；`gaps` 也已被改写成「free-run bpc **只测了** k=2、温度 0 贪心…」。
  - 复现证据：
    ```
    $ git show HEAD:axes/6-predict/6.3.3-latent-regression/project.yaml | grep -o "(e) 未测[^；]*"
    (e) 未测：自由跑（free-run）序列级 bpc 未测（只有 teacher-forced 命中率 / CE 与草稿接受率）…
    $ git diff -- axes/6-predict/6.3.3-latent-regression/project.yaml | grep -n "free-run 序列级 bpc 未测"
    13:+…(e) 未测：自由跑（free-run）序列级 bpc 未测…（新增行里仍在，仅把原结尾「故 status = 进行中…」替换成 (f) + status=已完成）
    21:+  - "free-run bpc 只测了 k=2、温度 0 贪心…"   ← gaps 已按「已测」改写，(e) 未同步 ⇒ 同文件内两处互相矛盾
    ```
  - 后果：任何下游 rollup / 阅读者引用 `conclusion` 会得到「free-run bpc 未测」，与同句 (f) 及落盘证据相反；这正是 §11.1 要禁止的收口级冲突。
  - 修复建议（不改数字）：(e) 去掉该分句，或写成「（WP1 范围内）free-run 序列级 bpc 原未测，已由 WP2 补测（见 (f)）；其余未测项如下…」。若定级为 blocker 阻断交付，可考虑降为「normal + 必改项」——本复核者按任务判据 B 的硬性措辞取 blocker。

### normal

- **N2（数值端点与落盘不符：真值锚 Δ 下界）** — 位置：`project.yaml:16`（`failure_analysis`）与 `project.yaml:18`（`conclusion_review`）：「实测真值锚 Δ = **+1.26…+5.07** bpc」。
  - 违的判据：§11.2「数字区间一致」+ 本任务 B「数字与落盘一致」。
  - 事实：`results.json::truth_anchor.delta_external_tf.*.combined_mean_over_seeds` 的四个窗口为 `disc_k2` +4.154104 / +5.070835、`lat_k2_lam1` **+1.261091** / **+1.170113** ⇒ 区间为 **+1.1701…+5.0708**；文中「+1.26…+5.07」把 13m_s1 的 latent 臂（1.1701）漏在下界外。定性结论（远越 0.014 带、同向）不受影响。
  - 复现：`python3 /tmp/review_c2_633_recompute.py` 第 7 节（自写独立 bootstrap）逐值复现 stored 四窗口，见下。
- **N3（数值端点与落盘不符：退化统计区间）** — 位置：`project.yaml:17`（`conclusion` (f)）「`lat_k2_lam1` 最长连续相同 token 平均 ≈137–154」。
  - 违的判据：同 N2。同句其余两支用的是「min–max」格式（`disc_k2` 3.8–59、`ar` 2.04=均值），此处混用。
  - 事实：四个窗口实测均值为 **99.83 / 153.83 / 151.58 / 143.50**（min–max = **99.8–153.8**，四者均值 = **137.19**）。「≈137–154」既不等于 min–max 也不是单点：按其格式读会漏掉最低窗口 99.83。`analysis.md::真值锚` 原文写「平均 ~137」是正确的（均值），是 `project.yaml` 转写时把它读成了区间。
  - 复现：`/tmp/review_c2_633_recompute.py` 第 5 节对 `streams/` 原始 token 独立重算 `max_identical_run`，10 个窗口与落盘逐值相同（见下）。
- **N4（判读措辞弱于/强于证据的边界：analysis.md「要点」的「自评 Δ」指代不明）** — 位置：`…/2670-freerun-bpc-13m/analysis.md`「真值锚」节要点第 1 条「故外部锚 Δ 在全部 4 个窗口都比**自评 Δ** 更大（是）」。
  - 违的判据：本任务判据 A「判读措辞与证据相称」。
  - 事实：该节表格有三列 Δ（自评 free-run / 自评 teacher-forced / 外部锚 teacher-forced）。对 `lat_k2_lam1 / 13m_s0`：外部锚 **+1.2611** **小于** 自评 teacher-forced **+1.4521**；只有把「自评 Δ」读作 **free-run** 列（+0.4289）时「全部 4 个窗口更大」才成立。这一读法正是 N1 关心的口径（自评 free-run），也确实是 `project.yaml` (f) 采用的口径（「自评口径低估」），故**结论方向正确、证据支持**；但字段措辞未点明是哪一列，读者按 teacher-forced 列读会得出反例。属措辞精度问题，建议改为「比**自评 free-run** Δ 更大」。

### info

- **I1**：`analysis.md::真值锚` 的三读数**确实回答了 N1**：它把「两臂各自自评」换成**同一 scorer**（自评主干 node-1 / 另一主干 node-1）重打分，并补了模型无关的客观退化统计；外部锚 Δ 在 4 个窗口都比自评 free-run Δ 更大 ⇒ 「自评高估」被否证；`ar` 流最长连续同 token 平均 **2.04** ⇒ 「`ar` 流退化自抬 Δ」这一具体机制被落盘数据否证。判读落在「自评口径**低估**代价、该路径质量面不成立」，与证据相称，且已用 `gaps` 声明同族 scorer 偏差未剥离（诚实登记）。
- **I2**：`conclusion_review` 只写**当前**判词（指 `REVIEW-C1-2670.md`），历轮（B1/B2/B3）只以 ticket 指针出现 ⇒ 符合 §11.1，未见多轮判词累积。WP1 的 `review` 字段记录了「三轮复核者均无 shell、执行类验收由 writer/lead 自跑」的残余缺口，属如实声明而非旧话残留。
- **I3**：`roadmap` 末条「收口：conclusion / status / 复核判词齐备（2026-10-14 完成）」把完成日期写进计划类字段；`started`/`ended` 才是 §11 允许的溯源日期位。非阻塞，若要严格可去掉括号内的日期。
- **I4**：`failure_analysis` 与 `conclusion_review` 都写了「真值锚 Δ = +1.26…+5.07 bpc」（§11.2「同一事实不在两处各写一遍」）；`failure_analysis` 是结论字段、`conclusion_review` 是复核判词字段，全删其一即可（与 B1 一并改）。
- **I5**：`conclusion_review` 记的 C1 执行清单（pytest 11 passed、变异 5/5、复算 ≤1.02e-9…）是**上一轮**读数；本轮实测已变（pytest **13 passed**、变异 **7/7**），但该字段写的是 C1 当时所跑，故不构成错误，仅提示读者以 ticket/本轮报告为最新。
- **I6**：`run_log.txt` 首轮含 bug 的读数（`ar` 6.05/5.87、`disc_k2` 16.2 > `log2(206)=7.687`）保留在日志中并已由 `WP-LOG.md` 说明修正，且 `analysis.md`/`results.json` 均为修正后值（本复核复算确认）。属诚实过程记录；外部引用须以 `results.json` 为准。

## 亲自执行的命令与真实输出

1. **单测（契约 §7）** — 通过，且比 C1 轮多 2 项（真值锚 / 退化统计的新门）：
   ```
   $ cd …/2670-freerun-bpc-13m && python3 -m pytest test_freerun.py -q
   .............                                                            [100%]
   13 passed in 2.34s
   （变异还原后再跑：13 passed in 1.94s）
   ```

2. **变异测试（抓红 + 逐字节还原 + 连跑两次哈希 + 无耗时字段）**：
   ```
   $ python3 mutate.py        # EXIT1=0
   RED  teacher_forced_shift_offbyone     restored=True
   RED  degeneracy_max_run_offbyone       restored=True
   {"n_mutations": 7, "all_caught": true, "all_restored": true, "post_restore_all_green": true}
   $ md5sum mutation_report.json      # 连跑两次
   6c84cfdd5ab322ceda6209a73a41e8da   (run1)
   6c84cfdd5ab322ceda6209a73a41e8da   (run2)   ⇒ REPORT_IDENTICAL=true
   $ python3 -c "…"  # 键与逐条 results[*] 扫描
   ['n_mutations','results','all_caught','all_restored','post_restore_all_green','post_restore_tail']
   timing_keys: []     # results[*] 键 = name/file/test_selector/caught_red/restored_byte_exact/pytest_tail
   ```
   7/7 抓红、7/7 `restored_byte_exact=True`，报告跨次哈希相同且不含逐次耗时。

3. **独立复算**（自写 `/tmp/review_c2_633_recompute.py`，**未 import `freerun.py` 的判据函数**；bpc = Σnll/(12·200·ln2)，CI 为自实现 prompt 聚类配对 bootstrap B=2000 seed=2670）：
   ```
   === 1. stream sha256 … ===
   files=120 per_prompt_mismatch=0 stream_all_mismatch=[]      # streams/*.json 的 tokens（int64 小端字节）独立哈希
   streams.all_sha_reproduced(stored) = True
   === 2. arm-level bpc recompute ===
   ar    13m_s0       recalc=0.5043561451 stored=0.5043561445 diff=6.47e-10
   ar    13m_s1       recalc=0.4893302611 stored=0.4893302616 diff=5.36e-10
   disc_k2 13m_s0/0   recalc=1.3506786125 stored=1.3506786115 diff=1.02e-09
   lat_k2_lam1 13m_s1/0 recalc=0.5765129866 stored=0.5765129869 diff=2.17e-10
   lat_k2_lam1 13m_s0/1 recalc=0.7005272333 stored=0.7005272341 diff=8.04e-10
   === 3. delta CI recompute ===
   disc_k2 13m_s0/0:     recalc d=0.846322 CI=[0.623191,1.034831] | stored d=0.846322 CI=[0.623191, 1.034831]
   lat_k2_lam1 13m_s0/1: recalc d=0.196171 CI=[0.081036,0.313283] | stored d=0.196171 CI=[0.081036, 0.313283]
   === 4/7. 真值锚 teacher-forced bpc 与 Δ ===
   ar 13m_s0  tf_self recalc=0.504356 stored=0.504356 | tf_ext recalc=0.568963 stored=0.568963
   disc_k2 13m_s0/0 tf_self recalc=4.957702 stored=4.957702 | tf_ext recalc=5.114426 stored=5.114426
   lat_k2_lam1 13m_s1/0 tf_self recalc=1.519187 stored=1.519187 | tf_ext recalc=1.574351 stored=1.574351
   disc_k2    13m_s0 external recalc d=4.154104 CI=[3.581944,4.671968] | stored d=4.154104 CI=[3.581944, 4.671968]
   disc_k2    13m_s1 external recalc d=5.070835 CI=[4.673389,5.384341] | stored d=5.070835 CI=[4.673389, 5.384341]
   lat_k2_lam1 13m_s0 external recalc d=1.261091 CI=[0.948287,1.636739] | stored d=1.261091 CI=[0.948287, 1.636739]
   lat_k2_lam1 13m_s1 external recalc d=1.170113 CI=[0.631496,1.845299] | stored d=1.170113 CI=[0.631496, 1.845299]
   === 5. degeneracy recompute（对 streams/ 原始 token）===
   ar/13m_s0 2.08=2.08 | ar/13m_s1 2.00=2.00 | disc_k2 21.67/59.33/3.83/25.75 全等
   lat_k2_lam1 99.83=99.83 / 153.83=153.83 / 151.58=151.58 / 143.50=143.50  （match=True ×10）
   === 6. 墙钟语义键扫描 === hits: ['/_wall_clock_fields']（自指豁免项）；L4 = {n_hits:0, pass:true}
   ```
   结论：`analysis.md` 表与 `project.yaml` (f) 引用的每一个数字（臂级 bpc、Δ 与 CI、真值锚四窗口、真实续写 1.6040/1.6027、退化统计）**都与落盘一致**，可独立复现。

4. **头权重与兄弟项一致性（`sha256` + `torch` 递归逐张量）**：
   ```
   disc_k2    13m_s0 0 disc_k2_s0.pt       exists=True sha_match=True tensors_equal=True
   …（8 份：{disc_k2,lat_k2_lam1}×{13m_s0,13m_s1}×{s0,s1}）…
   ALL_HEADS_OK: True
   ```
   `results.json::head_files` 的 8 条路径全部指向 `2660-latentreg-vs-disc-13m/heads/**`（13m_s0 的 4 条）与 `2660/second-backbone-13m_s1/heads/**`（13m_s1 的 4 条，即另一主干训练出的头），2670 目录内无头文件副本；文件存在、`sha256` 与记载相同、张量与结构递归相等。抽查相符：`…/2660/heads/disc_k2_s0.pt = 0254ea87cdb3`、`…/2660/heads/lat_k2_lam1_s0.pt = b17d15980d76`、`…/2660/second-backbone-13m_s1/heads/disc_k2_s0.pt = 6e5c2d5c5f44`、`…/2660/second-backbone-13m_s1/heads/lat_k2_lam1_s1.pt = e0a508a3bfe1`。

5. **预登记未放宽**：
   ```
   $ stat -c '%y %n' 2670/README.md 2670/results.json
   2026-10-14 20:51:36  README.md     ← 早于
   2026-10-14 21:30:58  results.json  ← 含追加的真值锚读数
   ```
   `README.md`（契约）mtime **早于**结果 39 分钟，且 C1 轮逐条核过 §5 判据未被改写；本轮 `README.md` 无更新（mtime 未变），真值锚属「只加读数」的复核要求补测，未改臂与判据。

6. **树校验 / 越界**：
   ```
   $ python3 tools/summary.py --check   → OK：46 个 project.yaml 全部符合 docs/research-schema.md。（EXIT=0）
   $ python3 tools/check_tree.py        → 同上（EXIT=0）
   $ git status --short -- pathlm docs tools     （空 ⇒ 无改动）
   $ git log --oneline -3 -- pathlm/
   fa6a1ad E3：二级条件头（chain3）+ PathConfig 开关 + 树注意力验证路径（引擎 + 单测 + 能力面）
   4638882 E2（与 E4 统一）：逐层 FFN 隐藏宽 schedule（引擎 + CLI + 单测 + 能力面）
   e894f20 ticket 67 · E1：CarryChannel 逐层独立状态 + 内容写门 + 可选衰减 γ（缺省逐位不变）
   $ git diff --check -- axes/6-predict/6.3.3-latent-regression/    （EXIT=0，无空白错误）
   ```
   `pathlm/**` 的改动（E1/E2/E3）已由 ticket 67 的引擎通道提交，属**已提交**状态、与本 WP 无关；本 WP 的 `pathlm/**` 工作树改动为 0。本项 `git status` 未跟踪项全在 2670 目录内，除 `project.yaml` 与 `experiments/README.md`（新增 2670 索引行，1 insertion）两处 `M`，无越界。

## not-run 清单（未执行，故不构成通过）

- **未重跑端到端生成**（`run.py` 需取 GPU 锁）：L5「确定性」`True` 仍由落盘自报 + 我的静态复算支撑，未独立两遍比对逐值。
- **未做族外第三方 scorer 打分**（时间盒内不可能，`gaps` 已登记「真值锚同族偏差未剥离」）。
- **未比对 `2640/spec_tree.py` 哈希**（`.scratch/63-axis6-predict/run1-snapshot/README.md` 未记载该文件；C1 已记同一 not-run）；本轮亦未复核 2640 三份快照哈希（不属本任务 1–5 项）。
- **未核 `2660`（WP1）数字**：本轮只读对照其 `heads/` 与目录结构，未重算 2660 的命中率 / α₁ / τ。
- **未核 `run.py` 的判据实现逐行**：只读了 `freerun.py::stream_sha256`（用于复现哈希口径）与 `run.py` 的 sha/落盘调用点；判据逻辑由 13 项单测 + 7 项变异 + 我的独立复算覆盖。
- **未核 `2730` 等其它轴 / 6.3.1 / 6.3.2 边界**（超出任务范围）。

## 对「终局是否诚实」的一句话结论

**诚实且数字可信**——`status: 已完成` 与 `failure_analysis`（路径 (i) 手段不成立、设计问题为主）与落盘证据完全一致，全部读数可独立复现、无把未测项写成已测，唯一问题是 `conclusion` (e) 残留在 HEAD 版本里的「free-run bpc 未测」旧话未被同步删除，与同字段 (f) 直接打架。
