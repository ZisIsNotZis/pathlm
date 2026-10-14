# REVIEW-C1-2670

- 角色：轴 6 独立复核者（fresh 上下文、未参与写作、只读）
- 时间戳：2026-10-14T21:17:58+08:00（`date -Iseconds`）
- 被审 revision：`46388822cef755159539e4e5208dd1110c0770ac`（`git rev-parse HEAD`）
  - 被审目录 `axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m/**` 全部为**未跟踪**新文件（`git status` = `??`），`git diff --` 该目录为空 ⇒ 无 committed 基线，证据以 worktree 文件哈希为准。
  - 复核期产物哈希：`results.json 263bd8d42d608ebf080f435d016a5002bfbb1c72e09b83a2e645268ea8a60a6f`、`analysis.md ac379202cea76735cebd84ace61fa0f493b622d6df13bb805890648ec904f5e7`、`freerun.py b651eaf020071b2c0d43f8c2a1bd28e97aeb8b9e4c2151ddf2196ec9d883c602`。

## VERDICT: PASS

执行与判读**可信**：臂级 bpc 与 Δbpc 的 prompt 聚类配对 bootstrap CI 均可从落盘 `nll_nats_per_prompt` 独立复现（臂级 <1e-9、CI 逐 6 位小数一致）；判据口径按契约执行；无越界；预登记（`README.md` mtime 20:51）早于结果（21:12）未被事后放宽。无 blocker。

**对复核简报 A 前提的事实更正（重要）**：简报称「`analysis.md` 里 `ar` 臂 free-run bpc = 6.05 / 5.87」。**该数字不在交付物中**。`analysis.md` / `results.json` 交付的 `ar` bpc = **0.504356（13m_s0）/ 0.489330（13m_s1）**。6.05 / 5.87 只出现在 `run_log.txt` 第 5–6 行，是**修复 bpc 分母 bug 之前**（21:08 首轮）的 ×12 放大值（见 info-2）。据此，A 的「塌缩到 bpc≈7.7」命题针对的是已被本人修正并取代的旧值。

## 分级清单

### blocker
- 无。

### normal
- **N1（判读口径的可信度限定，非执行缺陷）**：主判 Δbpc 是**双臂各自对自家生成流的自评 NLL 之差**（`ar` 自评 bpc=0.504 远低于同主干 teacher-forced ≈1.55）。`ar` 自评 bpc 极低是「贪心自由滚动自评偏袒」的典型特征；若 `ar` 的落盘流是重复/退化文本，其自评 bpc 会被压低，从而**系统性抬高 Δ**，使「质量面不成立」的措辞强于自评口径所能支撑。**落盘数据不足以判定**是否发生此伪影：`results.json` 每臂只落 `nll_nats_per_prompt`（12 个 prompt 级 NLL 和）与 `stream_sha256_*`，**未落 token 流、亦无 per-token NLL 轨迹 / 分段 bpc**，无法计算重复率或逐段 bpc。横向证据只到「疑似」：`ar`/13m_s0 prompt 10 的 NLL = 17.52 nats/200tok ⇒ 0.126 bpc（p_max≈0.92），属近确定性输出，与部分 prompt 退化的假设相容但**不构成证明**。契约 §3 预登记了此自评口径、`analysis.md::边界与未测` 已披露绝对值不可比的 caveat，故执行方忠实于契约，**不构成本 WP 的缺陷**；但建议在把结论写入 6.3.3 `failure_analysis` **之前**补一步真值口径（见「A 的定级建议」）。定级 **normal**（推进前需处理）。

### info
- **info-1**：链上锚点完整。`ar` 与 2640 `native` 的 12/12 prompt 前 8 token 逐位一致——我用 `2640/results.json::checkpoints.13m_s0.arms.native.per_prompt[*].gen_head` 独立比对，`True`。
- **info-2**：`run_log.txt` 保留了首轮（21:08）**含 bug** 的读数（`ar` 6.05/5.87，`disc_k2` 高达 16.2 > `log2(206)=7.687` 的理论上界）。该不可能值正是 bug 的信号，且 `WP-LOG.md §3` 已如实记录「发现并修正」。**这是诚实的过程记录，不是造假**；但外部读者若只看 `run_log.txt` 会误取 6.05/5.87（复核简报即如此）——建议在简报/下游引用时以 `results.json` 为准。
- **info-3**：`results.json::config.cost_axis` 的**字符串值**含子串「no wall-clock as criterion」。L4 的 `has_wall_clock` 只扫**键**不扫值，故 L4 `n_hits=0` 成立；键扫描的唯一命中是自指的 `_wall_clock_fields`（豁免项）。不是墙钟字段，仅提示。
- **info-4**：`2640/spec_tree.py` 的哈希**未记载于** `.scratch/63-axis6-predict/run1-snapshot/README.md`（该快照表只列 `results.json`/`cycle_log.jsonl`/`run_log.txt` 三份）⇒ 无法与快照比对（见 not-run）。
- **info-5**：`mutate.py` 就地改写 `freerun.py` 再逐字节还原（本复核对 `freerun.py` 的 mtime 因两次变异跑而更新为 21:15:49，内容经哈希与 11 项单测确认完好）。这是变异测试的固有方式，非缺陷；仅提示若被中断可能留下被改写的库。

## 亲自执行：命令与真实输出

1. **单测**（契约 §7）
   ```
   $ cd .../2670-freerun-bpc-13m && python3 -m pytest test_freerun.py -q
   ...........                                                              [100%]
   11 passed in 2.91s
   ```
   复核末尾再跑一次（变异还原后）：`11 passed in 3.05s`。

2. **变异测试**（抓红 + 逐字节还原 + 连跑两次哈希）
   ```
   $ python3 mutate.py
   RED  readout_not_shared                 restored=True
   RED  paired_bootstrap_sign_flipped      restored=True
   RED  ci_percentiles_swapped             restored=True
   RED  head_order_reversed                restored=True
   RED  wall_clock_undetected              restored=True
   {"n_mutations": 5, "all_caught": true, "all_restored": true, "post_restore_all_green": true}
   EXIT=0
   $ md5sum mutation_report.json   # 连跑两次
   913288ba9d9f681a768697ae5d7f2d31   (run1)
   913288ba9d9f681a768697ae5d7f2d31   (run2, after 3rd invoke)  ⇒ 哈希相同
   ```
   `mutation_report.json` 键只有 `n_mutations/results/all_caught/all_restored/post_restore_all_green/post_restore_tail`，逐条 `results[*]` 无耗时字段（含子串 "wall"/"clock" 的仅是变异**名** `wall_clock_undetected`）⇒ **不含逐次运行耗时**。

3. **独立复算**（自写 `/tmp/recompute_2670.py`，**未 import `freerun.py` 判据函数**；用 `Σnll/(n_prompts·n_new·ln2)` 与自实现的 cluster bootstrap B=2000 seed=2670）
   ```
   === arm-level bpc recompute (independent) ===
   ar           13m_s0 seed=None recalc=0.5043561451 stored=0.5043561445 diff=6.47e-10
   ar           13m_s1 seed=None recalc=0.4893302611 stored=0.4893302616 diff=5.36e-10
   disc_k2      13m_s0 seed=0    recalc=1.3506786125 stored=1.3506786115 diff=1.02e-09
   disc_k2      13m_s1 seed=1    recalc=1.3305038466 stored=1.3305038462 diff=4.17e-10
   lat_k2_lam1  13m_s0 seed=1    recalc=0.7005272333 stored=0.7005272341 diff=8.04e-10
   lat_k2_lam1  13m_s1 seed=0    recalc=0.5765129866 stored=0.5765129869 diff=2.17e-10
   === delta CI recompute (independent) ===
   disc_k2 13m_s0/0:     recalc d=0.846322 CI=[0.623191,1.034831]  stored d=0.846322 CI=[0.623191, 1.034831]
   disc_k2 13m_s1/1:     recalc d=0.841174 CI=[0.638508,1.031297]  stored d=0.841174 CI=[0.638508, 1.031297]
   lat_k2_lam1 13m_s0/1: recalc d=0.196171 CI=[0.081036,0.313283]  stored d=0.196171 CI=[0.081036, 0.313283]
   lat_k2_lam1 13m_s1/0: recalc d=0.087183 CI=[-0.087787,0.279353] stored d=0.087183 CI=[-0.087787, 0.279353]
   ```
   6 个臂级 bpc 差异 ≤1.02e-9（4 位小数舍入级）、4 个 Δbpc 及 CI 与落盘**逐 6 位小数相等**。`log2(206)=7.6865`（理论上界，供 info-2 参照）。

4. **torch 逐张量 / 冻结**（自写 `/tmp/torchcheck_2670.py`，CPU）
   ```
   === head tensors torch.equal vs source file ===
   .../2660-latentreg-vs-disc-13m/heads/disc_k2_s0.pt: tensors=8 equal=True
   .../heads/disc_k2_s1.pt: tensors=8 equal=True
   .../heads/lat_k2_lam1_s0.pt: tensors=8 equal=True
   .../heads/lat_k2_lam1_s1.pt: tensors=8 equal=True
   .../second-backbone-13m_s1/heads/disc_k2_s0.pt: tensors=8 equal=True
   .../second-backbone-13m_s1/heads/disc_k2_s1.pt: tensors=8 equal=True
   .../second-backbone-13m_s1/heads/lat_k2_lam1_s0.pt: tensors=8 equal=True
   .../second-backbone-13m_s1/heads/lat_k2_lam1_s1.pt: tensors=8 equal=True
   ALL heads equal: True
   === backbone freeze check ===
   13m_s0: requires_grad_all_false=True grad_none_after_backward=True
   13m_s1: requires_grad_all_false=True grad_none_after_backward=True
   ```
   8 份头文件各自 2 头 × 全部张量（每份 8 张量）逐张量 `torch.equal` 全真；两主干 `requires_grad` 全 False 且一次 backward 后 `.grad is None`。另经 `sha256sum` 核准：`results.json::head_files` 的 8 个 sha256 与 `2660/heads/*.pt`、`2660/second-backbone-13m_s1/heads/*.pt` 实际哈希**逐一相同**（例：`disc_k2_s0.pt = 0254ea87…`、`lat_k2_lam1_s0.pt = b17d1598…`、`second…lat_k2_lam1_s1.pt = e0a508a3…`）。

5. **2640 证据未被污染**
   ```
   $ sha256sum 2640/results.json 2640/cycle_log.jsonl 2640/run_log.txt
   1302b6785630f32815651a02340ecd72b556c7e5ec44657311994ee74c13dbaa  results.json   == 快照记载
   d2073370c7985c7b1bc0981e6d8a73231462674bbd2ea49ae0f54ef48a592290  cycle_log.jsonl == 快照记载
   ba48b06a0bcee2420c119c61a6c2687dee544225d1e30c29b60fb1a600a2984c  run_log.txt     == 快照记载
   $ git status --short -- 2640-tree-conf-13m   # (空)
   ```
   三份被快照记载的文件哈希**一致**，2640 目录 git 无改动。

6. **越界检查**
   ```
   $ git status --short -- pathlm docs tools train_*.py
    M pathlm/config.py
    M pathlm/decode.py
    M pathlm/model.py
   $ git diff --stat -- pathlm
    pathlm/config.py |  6 ++++++      # chain_second（E3 / ticket 67）
    pathlm/decode.py | 64 +++++…      # tree_parents 树注意力（E3 / ticket 67）
    pathlm/model.py  | 43 +++++…      # FFN 隐藏宽 schedule
   $ git log -1 --format='%ci %s'
    2026-10-14 21:13:10 +0800 E2（与 E4 统一）：逐层 FFN 隐藏宽 schedule（引擎 + CLI + 单测 + 能力面）
   $ stat -c '%y' pathlm/{config,decode,model}.py   # 21:13:47 / 21:14:20 / 21:14:00
   ```
   对 `pathlm/**` 的改动内容是 **E2/E3（FFN schedule + 二级条件头 `chain_second` + 树注意力）**，与本 WP（free-run bpc 实验）无关；其 mtime（21:13–21:14）**晚于** 2670 最后写盘（`results.json` 21:12:01），且 2640/其它轴目录无改动 ⇒ 被审目录外改动**非本 WP 造成**。被审目录内 `git diff` 为空（全为新增未跟踪文件）。

7. **预登记未被事后放宽**
   ```
   $ stat -c '%y %n' 2670/README.md 2670/results.json
   2026-10-14 20:51:36  README.md     ← 早于
   2026-10-14 21:12:01  results.json  ← 结果
   ```
   `README.md`（契约）mtime 20:51 早于首轮全量跑（`run_log.txt` 21:08）与 `results.json`（21:12）。`WP-LOG.md` 记的「21:11 精修契约 §5 判词逻辑」只改了 `run.py::_verdicts`（判定量选 `L1b` lat−disc），**未触碰 `README.md`**（mtime 可证）。对照契约 §5 逐条核对：§5-1 的量是 `Δbpc(disc_k2 − ar)`、§5-2 的量是 `Δbpc(lat_k2_lam1 − disc_k2)`——`run.py` 分别用 `L1` 与 `L1b` 实现，**与契约字面一致**，且 6 个 disc 窗口全部「CI 不含 0 且 |Δ|>0.014」、6 个 lat 窗口全部越带，故 §5-1 在此数据下**无论是否要求全窗口**都触发（不存在「放宽才触发」）。字段与 `results.json::config` 一致（`reference_band_bpc 0.014`、`B=2000`、`seed=2670`、`cluster=prompt`、`12 prompt × 64 char`、`n_new=200`、`k=2`、offsets 逐条列出）——**与契约 §2/§3 一致，未被事后放宽**。

8. **禁墙钟**
   ```
   wall-clock-like KEYS: ['._wall_clock_fields']
   ```
   `results.json` 的**键**扫描仅命中自指豁免项 `_wall_clock_fields`；无 `time/elapsed/duration/seconds/latency/throughput/…` 语义字段。`L4 = {n_hits:0, pass:true}`。`analysis.md` 判据区无墙钟字段（仅一句「墙钟只出现在 `run_log.txt`」的声明）。墙钟仅出现在 `run_log.txt`（`DONE wall_seconds=`），合约 §6 允许。

## 契约 §5 否证条件逐条核对（B）

| 条 | 契约条件 | `analysis.md` / `results.json` 声明 | 我的核对 |
|---|---|---|---|
| §5-1 | `Δbpc(disc−ar)` CI 不含 0 且 |Δ|>0.014，且 lat 只缩小、同向、仍越带 ⇒ «质量面不成立» | **触发** | disc 4 种子 Δ=+0.63…+1.00 全不含 0 全越带；lat 4 种子 Δ=+0.09…+0.66 全越带；四种 Δ 全同号为正 ⇒ 与契约条件吻合，**声明正确** |
| §5-2 | `Δbpc(lat−disc)` CI 含 0 ⇒ «λ=1.0 无可测增益» | **未触发（部分）**：6 窗口仅 1 个（`13m_s0/0`=-0.1847 CI[-0.4741,+0.0956]）含 0 | 核实：`13m_s0/0` 含 0，其余 5 窗口不含 0 且 Δ<0 ⇒ 声明与数据**一致** |
| §5-3 | 全部 CI 含 0 ⇒ «不可分» | **未触发**（至少一条不含 0） | 核实：多条不含 0 ⇒ 声明正确 |

三条否证条件的触发/未触发声明**与落盘数字一致**，无「未触发却写成触发」或反之。

## 账实核对（C）

- `analysis.md` 与 `results.json` **账实相符**：`analysis.md` 由 `run.py::_write_analysis` 从 `results.json` 生成，我对臂级 bpc、Δbpc、CI、shas、offsets、L3 逐值抽查一致；且独立重算（第 3 项）反向确认 `results.json` 自身与 `nll_nats_per_prompt` 一致。
- `analysis.md` **无过程叙事 / 日期流水**：正文无日期、无「原本是 A 后改成 B」、无变更记录/ERRATA，符合事实文档纪律。带日期的流水只在日志类文档（`run_log.txt`、`WP-LOG.md`）里，属允许范围。

## A 的定级建议（重点证伪项）

- **对简报前提的更正**：`analysis.md` 交付的 `ar` free-run bpc 是 **0.504 / 0.489**，**不是** 6.05 / 5.87。「塌缩到 bpc≈7.7」命题不适用于交付物（且 6.05/5.87 及 16.2 来自已被本人修正的分母 bug）。
- **对塌缩伪影本身**：交付的 `ar` 自评 bpc=0.504 **低于** teacher-forced≈1.55，方向与「分布趋平⇒bpc↑」**相反**；故「Δ=+0.74/+0.43 由塌缩造成」这一具体假设**不成立**。
- **但 A 背后的科学隐忧以另一种形式保留且未被排除**：双臂**各自自评自家流**，Δ 实为「两个不同流各自的自评置信度之差」而非「对真值的质量代价」。`ar` 极低的自评 bpc 与贪心退化（重复输出）相容；若 `ar` 流退化，Δ 会被系统性抬高。**落盘数据不足以判定**（无 token 流、无 per-token NLL 轨迹 / 分段 bpc，只有 prompt 级 NLL 和与 sha256）。
- **定级：normal（非 blocker）**——执行忠实、数字可复现，但主结论「质量面不成立」的措辞强于自评口径所能支撑；建议**在把结论并入 6.3.3 `failure_analysis` 之前**补一步最小真值口径。
- **最小补测命令与成本**（供 lead 决定，本人未执行）：
  - 把两臂的 200-token 流落盘（现只落 sha256），对其算 (a) 重复率/最大连续重复 token 数、(b) 对**真实 enwik8 下一字符**的 teacher-forced bpc（同 `pathlm/eval.py::bpc` 口径）。
  - 成本：单卡 4090 上生成已只需 ~20 s（`run_log.txt` 的 `wall_seconds=20.5`），加真值评分 <1 min GPU，按 `SCHEMA.md §20` 取 `wp6.3.3-freerun` 锁即可；属**只加读数**、不改预登记的补充。

## 明确的 not-run 项

- **未做确定性 GPU 重跑**（L5 的独立复现）：时间盒内未取 GPU 锁重跑 `run.py`；L5 由落盘自报（`ar_13m_s0_rerun_identical`/`disc_k2_13m_s0_s0_rerun_identical`=True）与我的静态一致性复算支撑，**未独立重跑两遍比对逐值**。
- **未比对 `2640/spec_tree.py` 哈希**：快照 README 未记载该文件哈希（见 info-4），无从比对；但 2640 目录 `git status` 干净、且三份被记载文件一致。
- **未重跑 `run.py` 复现臂级 bpc**（只做数值复算，未做端到端 GPU 重算）。
- **未测 `README.md` 之外的第一版契约是否被覆盖**（无版本历史，仅以 mtime 判定未事后编辑）。
