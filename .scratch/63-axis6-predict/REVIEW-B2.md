<!-- 本报告作者 = fresh 复核者 #2（未参与写作）；其上下文**无 shell、无写文件工具**，
     故由 axis6-lead 于 2026-10-14T10:16:57+08:00 把其返回的完整报告**逐字**落盘（内容未改一字）。
     其 normal F2 的分支判定已由 axis6-lead 用 torch 逐张量复算钉死（见 TICKET.md §B5）。 -->

VERDICT: FAIL

# REVIEW-B2 · ticket 63 / WP 6.3.3 · 独立复核 #2（fresh 上下文、未参与写作、只读）
- 被审对象：工作树（未提交，全部 `2660-latentreg-vs-disc-13m/**` 为 untracked 新文件）
- 复核者约束：本上下文**无 shell / 无写文件工具**（`read/grep/find/ls/watchdog_diff`），
  因此「重跑」「sha256sum」「git status」「pytest」「mutate.py」「summary --check」「mtime」**均未执行**；
  凡未执行项逐条标 `not-run` 并给出补齐命令。**未改任何文件**。
- 时间戳：本上下文**无时钟可读**（无 shell）⇒ 无法执行 `date -Iseconds`；按派发时序，本复核发生在
  `WP-6.3.3-LOG.md` 末条 `2026-10-14T09:58:17+08:00` 之后。**此时间戳非机器核验**。

## 0. 结论摘要
- L0/L1/L2/L4 的**判词方向**与落盘原始数据一致，未发现读数不支持判词的统计错误；统计实现静态审读正确。
- 但**交付物内部存在 1 处账实不符**（主 `results.json` 的 P3 汇总判词与 P3 目录自身结果相反），
  另有 1 处**证据链内部冲突**（λ=0 双臂 `.pt` 文件 sha256 互不相同，与「逐位相同」的证据用法冲突）。
- 最强证伪手段（临时目录确定性重跑）在本上下文**不可执行** ⇒ 依 `AGENTS.md`「Proof and review」，
  「未知验收不是 PASS」，故判 **FAIL**（**无 blocker 级科学缺陷**；上述两条为须先修的 normal 级）。

## 1. 我实际做了哪些核查（静态）
1. **契约先行性（部分）**：`README.md` 只含臂/预算/判据/否证条件与跑法，**不含任何实测数字**
   （唯一带数字处是 `0.014` 归因纪律），符合预登记形态；`results.json::contract = "README.md (pre-registered)"`。
   `README.md` 是否早于 run 开始（mtime）**not-run**。
2. **L1 算术一致性（自算，非引用）**：
   - `lat_k2_lam1`：Δt+2 = mean[(0.48779−0.45721), (0.48918−0.45726)] = mean[0.03058, 0.03192] = **0.03125** = 落盘 `combined_mean_over_seeds.delta` ✓
   - `lat_k2_lam01`：mean[0.01159, 0.01120] = **0.011395 ≈ 0.01140** ✓
   - k=4：mean[(0.48457−0.45790), (0.4849819472−0.4547590886)] = mean[0.02667, 0.03022] = **0.02844** ✓
   - 即 `combined` = 两 seed 逐块 Δ 的均值；因两 seed 共用同一评测块集，该口径合法（方差缩减），非重复计数。
   - CI 均不含 0（+0.01140 CI[+0.01006,+0.01274]；+0.03125 CI[+0.02955,+0.03302]），
     CI 形状与 B=2000/seed=2660/block 聚类/percentile95 的口径描述一致。
3. **统计实现静态审读（`latent_reg.py:270-306`、`draftverify.py:150-200`）**：
   `cluster_bootstrap_paired` 按**块**重采样、`den = Σ n[idx]`（非固定总位置数）✓；
   `rng.integers(0, nb, (B, nb))` 有放回 ✓；percentile 2.5/97.5 + `excludes_zero`
   判定 `>0 or <0` ✓。`paired_diff` 同索引重采样、α₁/τ 定义（≥1 占比 / 平均接受数）与契约 §4-L2 一致 ✓。
   **未见跨量纲借用 0.014 bpc 带**：`README §4-L1` 已把它标为「只作参考量级」，`results.json` 未在率类判据中使用它。
4. **路径 (ii) 保真门（静态比对）**：`results.json::draftverify.repro_gate` 的 12 条 `per_prompt`
   全部 `bit_identical=true`、`first_mismatch=null`、两侧 `n_rounds` 一致，`got_alpha1=0.802243` /
   `got_tau=1.453619` 与 2640 `direct1` 相同。**注意**：这是 writer 运行时自算的比对结果；
   我**未**逐轮重比 accepts（我这一上下文无脚本能力，人工比 12×~80 轮不可靠）⇒ **not-run**（命令见 §5）。
5. **接线审读**：`draftverify.make_draft_fn_head` 首草稿 = `argmax(head0(nodes[0]["latent"]) @ E^T)`，
   第二草稿 = `nodes[3]["logits"]`，与 `make_draft_fn_direct1`（`nodes[2]`, `nodes[3]`）的**第二草稿同源** ⇒
   单因素「位置 N+2 的草稿来源」成立，比较公平 ✓。`test_latent_reg.py::test_draftverify_*`（4 条）
   对位置/口径/参照是否被改有钉住 ✓。
6. **越界**：`watchdog_diff`（vs reviewer-launch HEAD）显示唯一 tracked 改动 = `.scratch/63-axis6-predict/TICKET.md`
   (+47)；其余全部 untracked，**未见对 `pathlm/**`、`tools/**`、`docs/**`、`2631/**`、`2640/**` 的改动**
   （`2650-verify-gate-13m/**` 为**新增 untracked**，按 `.scratch/63-axis6-predict/WP-6.2.3-LOG.md:9`
   属另一 WP 的写入范围，非本 WP 越界）。
7. **权重不入库**：untracked 清单**不含任何 `*.pt`**（`heads/*.pt`、两个 second-*/heads/*.pt 均未出现在
   untracked 列表）⇒ `.gitignore` 覆盖成立 ✓。
8. **禁墙钟**：主 `results.json:4839`、`second-k4-13m_s0/results.json:5669`、`second-backbone-13m_s1/results.json:3611`
   三处 `_wall_clock_fields: []` ✓；墙钟只出现在 `run_log.txt` / WP log（契约允许）。

## 2. findings

### normal · F1 · 主 `results.json` 的 P3（k=4）判词与其目录结果**相反**（账实不符）
- 位置：`.../2660-latentreg-vs-disc-13m/results.json:4919-4921`
  （`supplementary_runs.k4_13m_s0.L0_all_pass = false`；`verdicts = "blocker：L0 未过（实现不等价），L1 不可判"`）
- 反证：`.../second-k4-13m_s0/results.json:5370-5399` ⇒ `pair=[lat_k4_lam0, disc_k4]`，
  `hits_bit_identical=true`、`ce_bit_identical=true`、`head_weights_bit_identical=true`、`all_pass=true`；
  `analysis.md`（P3 节）同称「L0 pass（k=4 等价自检同样逐位相同）」。
  同一条目的 `L1` 又带 `excludes_zero: true`（+0.02844），与「L1 不可判」自相矛盾。
- 成因（可从代码确认）：`run.py` 的 `results["L0"]["all_pass"]` 在 L0 配对循环后计算，
  无配对臂时 `results["L0"]` 仍为空 dict ⇒ `all_pass=False`、判词回落成 blocker；
  `WP-6.3.3-LOG.md:47` 已记「首次 run 的 L0 硬编码 k2 ⇒ k=4 目录 L0 空（误读为 fail），泛化后重跑」。
  ⇒ 该汇总字段是**首跑的手工汇总，重跑后未回写**。
- 违反：契约 §4-L0 硬门结论的落盘一致性；`AGENTS.md`「同一事实恰好一个家 / 冲突以权威处为准」。
- 复现（待 runner 执行）：
  `python3 -c "import json;a=json.load(open('axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/results.json'));b=json.load(open('axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/second-k4-13m_s0/results.json'));print(a['supplementary_runs']['k4_13m_s0']['L0_all_pass'],a['supplementary_runs']['k4_13m_s0']['verdicts'],b['L0']['all_pass'])"`
  → 期望（现值）`False ...blocker... True`（冲突），修后应为 `True / <采纳判词> / True`。
- 最小修法（不自行实施）：用手工汇总逻辑重取 `second-k4-13m_s0/results.json` 的 `L0.all_pass` 与
  `verdicts`（`L0`/`L1`）覆盖该两条字段；**不要**重跑 P1/P2/P3。
- 附带（同源，info 级）：`run.py` 的 `verdicts["L0"]` 与 `_write_analysis` 的 L0 文案把
  `lat_k2_lam0 == disc_k2` **硬编码**，导致 `second-k4-13m_s0/{results,analysis}.md` 在 k=4 语境下
  写成 k=2 的臂名（`results.json:40`、`analysis.md:7`）。最小修法：用 `k` 参数化该文案。

### normal · F2 · λ=0「等价双臂」的 `.pt` 文件 sha256 互不相同；L0 的权重比较只覆盖 1 个张量
- 位置与证据：
  - `MODEL.md:16-23`（P1）：`disc_k2_s0 = 0254ea87…` vs `lat_k2_lam0_s0 = 0ace8fb6…`（`results.json:68`/`:1182` 同值）；
  - `MODEL.md:52-58`（P2·13m_s1）：`6e5c2d5c…` vs `6a986b3b…`；
  - `MODEL.md:62-68`（P3·k=4）：`1001cd97…` vs `8dbc0a0d…`；
  - `run.py` L0 的权重同一性只比 `torch.load(...)["head0"]["mlp.0.weight"]`（**一个张量、头1 未比**），
    却落盘字段名 `head_weights_bit_identical`；
  - `.scratch/63-axis6-predict/WP-6.3.3-LOG.md:48` 声称「全部头 sha256 与首跑**逐一相同**（确定性复核）」。
- 冲突：三对（P1/P2/P3）在 content 上应**完全相同**（同 k、同 λ=0、同 seed、同代码路径、同数据序；
  `hits` 与**逐块 CE 用 `np.array_equal` 逐位相同**，见 `results.json::L0.*.ce_bit_identical`），
  但**文件哈希三对全不同**。两种可能：(a) `.pt` 字节含保存时刻（torch 走 `zipfile.ZipFile.open('w')`
  ⇒ DOS `date_time` 随保存时间变化）⇒ 文件哈希**不能**当「逐位相同」证据，且 LOG 的「跨 run sha 相同」不成立；
  (b) 两臂权重确实不同 ⇒ 则 L0 与契约 §2 的等价限定不成立（**blocker 级**）。
  现有落盘数据（hits/逐块 CE 逐位相同）**强烈指向 (a)**，但当前没有任何落盘证据能排除 (b)。
- 复现（待 runner 执行，决定性）：
  ```
  python3 - <<'EOF'
  import torch, hashlib
  d='axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/heads/'
  a=torch.load(d+'disc_k2_s0.pt',map_location='cpu',weights_only=True)
  b=torch.load(d+'lat_k2_lam0_s0.pt',map_location='cpu',weights_only=True)
  print('keys',sorted(a)==sorted(b),'all_equal',all(torch.equal(a[k],b[k]) for k in a))
  for f in (d+'disc_k2_s0.pt',d+'lat_k2_lam0_s0.pt'):
      print(f, hashlib.sha256(open(f,'rb').read()).hexdigest())
  EOF
  # 若 all_equal=True 且 sha256 不同 ⇒ (a)：另跑两次间隔>3s 的 torch.save 同一 state_dict 复现哈希不同，
  #   即可把「zip 时间戳」结论钉死；若 all_equal=False ⇒ (b)：L0 不成立，结论须回退。
  ```
- 最小修法：(1) L0 改为**整份 state_dict 逐键 `torch.equal`**（含所有头、所有张量）并记录该比对；
  (2) `MODEL.md` 加一句「`.pt` sha256 仅标识文件，不构成内容同一性证明」；
  (3) LOG 的「sha256 与首跑逐一相同」改为内容级证据（或删除该断言）。

### info · F3 · 「`disc_k4 ≈ disc_k2` ⇒ 增益来自 λ 而非头数」缺配对 CI
- 位置：`analysis.md` P3 末条。依据是两个不同 run 的边际命中率（0.45790/0.45476 vs 0.45721/0.45726；
  t+3 0.3377 vs 0.3352），**未按契约 §5** 给配对聚类 CI（同文件其余结论都有）。
- 影响：该句是推断链的一环（用来说明增益非「头多」所致），却比同文其它结论的证据标准低。
- 最小修法：补 cross-run 配对 CI（两 run 共用同一评测块集，`cluster_bootstrap_paired` 可直接配对），
  或把措辞降级为「描述性读数，未做显著性检验」。

### info · F4 · L0 只覆盖「harness 确定性」，不覆盖 λ>0 路径的接线
- 位置：`run.py` L0 段 + `test_latent_reg.py::test_lambda0_arms_equivalent_and_deterministic`。
- 事实：契约 §2 已声明 λ=0 时两臂**是同一计算**（`train_arm` 完全跳过 MSE）⇒ L0 通过**只能**证明
  「同配置可重现」，**不能**证明「λ 的 MSE 目标取自 `h_true_{i+1+j}` 正确位置」；单测只断言 λ 被接线
  （λ>0 与 λ=0 结果不同）。
- 最小修法（不阻断）：加一条「MSE 目标位移」单测——构造 H 使仅正确位置可降 loss，错位则抓红。

### info · F5 · `lat_k4_lam0` 属预登记臂表之外的补充臂
- 位置：`README.md §3` 臂表只列 `disc_k4` / `lat_k4_lam1`；`run.py::ARMS` 另有 `lat_k4_lam0`（k=4 的 L0 自检臂），
  `analysis.md` P3 表已列出。
- 判断：新增的是**自检臂**，未改动任何已登记臂的定义，未违反 §3 的「事后不得改」；但按惯例宜在 `README`/`analysis`
  注明「`lat_k4_lam0` = L0 等价自检补充臂」。**低风险**。

### info · F6 · `analysis.md` 对「回归机制」的措辞边界（契约 §2 限定仍成立）
- 契约 §2 即使预言被否证，**限定仍成立**：两臂共用 `z_j = TransformHead_j(h_i)` 与同一 tied 读出、
  同参数量 ⇒ 二者**不是两个可分离的架构**，唯一自由度是辅助 MSE 监督项。
- 核查：`analysis.md` 摘要与 L2 节措辞**基本守边界**（写「多一个 λ·MSE 项」「回归监督项 λ」、
  明确「不写成 latent 回归普适更优」）。**但** §「对契约 §2 预登记预言的结论」只写「预言被否证 ⇒ 不建议降级
  6.3.3 的描述」，未复述上述仍成立的限定；而 6.3.3 的 `innovation` 若描述的是**架构式**「先回归出 latent 再读出」，
  本实验并未与「不回归的架构」对照。
- 最小修法（报告级，不需改数字）：在该节补一句「增益可归因于**同一参数化上的辅助 MSE 监督**；
  本实验未把『回归产出方式』与『其它辅助监督』区分开」，供 lead 裁决 `project.yaml` 时不被误读。

## 3. not-run（本上下文无 shell；须由有 shell 的复核者执行并回填）
1. **确定性重跑（最强证伪）**：

   `python3 axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/run.py --stage train --out /tmp/rev2660 --ckpt /home/z/vibe/pathlm/axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0`
   （**注意代码路径**：`run.py` 默认 `--out` 为脚本所在目录，必须显式 `--out /tmp/rev2660`；
   CLI 见 `run.py:74-83`：`--stage/--out/--ckpt/--seeds/--arms/--no-lock/--smoke`；
   `run.py` 的 `Tee` 以 **append** 方式打开 `run_log.txt`，故**绝不能** `--out .`，否则污染被审目录。）
   然后逐项比对：`/tmp/rev2660/heads/*.pt` sha256 ↔ `MODEL.md §2`；`/tmp/rev2660/results.json` 的
   `arms.*.*.hit_t2/t3`、`ce_t2/t3`、`mse_t2/t3` ↔ 现盘。（预期：命中率同值；若 sha256 不同，见 F2 的 (a) 分支。）
2. **L1/L2 独立复算**：需写脚本读取 `results.json` 的 `arms.*.*.*`（注意 `_hits`/`_n_pos_blk` **未落盘**，
   只有 `ce_trace` 与聚合值）⇒ 严格复算**必须重跑**（命令同上）或在脚本里重新训练+评测；
   仅凭落盘数据只能做 §1.2 的算术一致性核对（我已做）。
3. **保真门逐位**：
   `python3 -c "import json;a=json.load(open('axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/results.json'))['draftverify'];b=json.load(open('axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m/results.json'))['checkpoints']['13m_s0']['arms']['direct1'];print([p['accepts']==q['accepts'] for p,q in zip(a['arms']['repro_direct1']['per_prompt'],b['per_prompt'])])"`
   → 期望 12×True。
4. `python3 -m pytest axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/test_latent_reg.py -q`
   （WP log 声称 11 passed；**未自跑**）
5. `python3 axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/mutate.py`
   （WP log + `mutation_report.json` 声称 8 条全抓红 + 逐字节还原 + 还原后全绿；**未自跑**）
6. `python3 tools/summary.py --check`；`git status --short`；`git diff --stat`；
   `stat -c '%y' .../2660-latentreg-vs-disc-13m/{README.md,results.json}`（README mtime **必须早于** results.json /
   run_log 首行时间 `2026-10-14T09:25…`，据 `WP-6.3.3-LOG.md` 疑点顺序；**未核**）。
   `pytest tests/ -q`（共享底座）如需回归亦请一并跑。

## 4. 我的判定
- **无 blocker 级**：主判 L1/L2 的数字自洽、CI 口径与实现一致、保真门证据存在、越界与禁墙钟成立、
  k∈{2,4}×两主干点×两种子同向⇒「单 ckpt / 单种子噪声」这一替代解释已被现有证据排除。
- **FAIL 的依据**：F1 属必须修的账实不符（同一交付物内两个相反判词）；F2 属必须闭环的证据链冲突
  （`.pt` sha256 与「逐位相同」用法矛盾，分支 (b) 若成立即为 blocker）；且点名的确定性重跑/测试类验收
  在本上下文**不可执行** ⇒ 依 `AGENTS.md`「未知验收不是 PASS」，本轮不能给 PASS。
- 复核者**未做任何修复**（只读）；F1/F2 的最小修法与补齐命令已在上文给出。
