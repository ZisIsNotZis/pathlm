# WP-LOG — 2670-freerun-bpc-13m（WP 6.3.3-WP2 · 路径 (i) free-run 序列级 bpc）

**状态：已完成**（本 WP 全部交付物落盘，`pytest` 绿、变异全抓红、主判数字齐）。

- 单一写入者：本目录（fresh 上下文 worker）。只写
  `axes/6-predict/6.3.3-latent-regression/experiments/2670-freerun-bpc-13m/**`；
  只读复用 2660 的 `latent_reg.py`（`load_ckpt` / `freeze` / `trunk_h` / 头同一性校验）与 2660 的头权重；
  未改 `pathlm/**`、`train_*.py`、`tools/**`、`docs/**`、其它轴、2660 目录、任何 `project.yaml`。
- 契约 = 本目录 `README.md`（预登记，未改）。
- GPU 锁：复制 2660 的 `gpu_lock.py`；运行取 `wp6.3.3-freerun` 锁（一次等待后取得，未杀他人进程）。

## 1 时间线与命令

| 时间 (UTC+08) | 命令 | 结果 |
|---|---|---|
| 2026-10-14 20:51–21:09 | 读契约 / 2660 harness / 2631 主干 / 2640 native 参照；写 `freerun.py` `run.py` `test_freerun.py` `mutate.py`；`cp gpu_lock.py` | 完成 |
| 2026-10-14 ~20:56 | `python3 -m pytest test_freerun.py -q` | 10 passed |
| 2026-10-14 ~20:57 | `python3 mutate.py` | 5 变异 4 抓红、`head_order_reversed` MISS（测试判别力不足）⇒ 加固测试 |
| 2026-10-14 ~20:58 | `python3 run.py --out /tmp/2670_smoke --smoke` | 冒烟跑通；修正 L4 字段自命中 |
| 2026-10-14 ~21:08 | `python3 run.py --out .`（首次全量） | 跑通但**发现 bpc 分母 bug**（见 §3） |
| 2026-10-14 ~21:09 | 加 `test_results_bpc_matches_nll_aggregation`（gate probe） | 对错版 `results.json` 变红（0.504 vs 6.05），确认可失败 |
| 2026-10-14 21:10:38–21:10:53 | `python3 run.py --out .`（修正后全量，wall 20.5 s） | 完成，写 `results.json` / `analysis.md` / `MODEL.md` / `run_log.txt` |
| 2026-10-14 21:11 | 精修契约 §5 判词逻辑（§5-1 全窗口 / §5-2 用 L1b 量），重生成 `results.json`+`analysis.md` | 完成 |
| 2026-10-14 21:12 | `python3 -m pytest test_freerun.py -q` | **11 passed** |
| 2026-10-14 21:12 | `python3 mutate.py` | **5/5 抓红 + 逐字节还原 + 还原后全绿** |

## 2 主判数字（契约 §4-L1 / §5）

冻结 13M 主干（2631 `13m_s0` / `13m_s1`，embed sha256 `566ec50c…` / `da5b5d33…`），
12 prompt × 64 char、每 prompt 200 token、k=2、温度 0 贪心。
主判 = Δbpc（候选 − 同主干 `ar`）的 **prompt 聚类配对 bootstrap 95% CI**（B=2000，seed=2670）。

| 候选臂 | 主干 | Δbpc（两种子合并） | 95% CI | 不含0 | |Δ|>0.014 |
|---|---|---|---|---|---|
| `disc_k2` | 13m_s0 | **+0.7389** | [+0.5745, +0.8918] | 是 | 是 |
| `disc_k2` | 13m_s1 | **+0.9195** | [+0.7927, +1.0555] | 是 | 是 |
| `lat_k2_lam1` | 13m_s0 | **+0.4289** | [+0.2817, +0.5867] | 是 | 是 |
| `lat_k2_lam1` | 13m_s1 | **+0.1633** | [−0.0443, +0.4052] | 否 | 是 |

- **手臂级 bpc**：`ar` 0.5044 (s0) / 0.4893 (s1)；`disc_k2` 1.1359–1.4872；`lat_k2_lam1` 0.5765–1.1660。
- **Δbpc(lat_k2_lam1 − disc_k2)**（同主干同种子配对）：s0 −0.3100 [−0.4902, −0.1243]；s1 −0.7563 [−0.9686, −0.5411]（均不含 0 ⇒ λ=1.0 在多数窗口降低 free-run bpc）。
- **契约 §5 触发**：§5-1 **触发**（disc 全窗口 CI 不含 0 且越 0.014 带；lat 只缩小、同向、仍越带）⇒ 判「路径 (i) 在本尺度质量面不成立」。§5-2 未触发（6 窗口中仅 `13m_s0/0` 含 0）。§5-3 未触发。

## 3 发现并修正的缺陷（自测，非独立验收）

- **bpc 分母 bug**：首次全量 `bpc_from_nll(nll, n_new)` 只除以 200，而 `nll` 是跨 12 prompt 的和 ⇒
  手臂级 bpc 被放大 12 倍（出现 >log2(206)=7.69 的不可能值，如 16.2）。修正为 `n_prompts·n_new`。
  **对策**：加单测 `test_results_bpc_matches_nll_aggregation`（用落盘 `nll_nats_per_prompt` 重算 bpc
  与落盘 bpc 对账），并对**错误版** results.json 跑过 → 变红（gate probe 成立）。
- 首轮 `head_order_reversed` 变异未被抓红（随机小头两头 argmax 偶合）⇒ 改用受控头（mlp 归零 + 大偏置
  把两头 argmax 钉到不同 token）⇒ 抓红。

## 4 交付物与证据锚点

| 交付物 | 路径 | 证据锚点 |
|---|---|---|
| 库 | `freerun.py` | `logits_of`（共享 tied 读出）/ `freerun_ar`（node-1 贪心）/ `freerun_path_i`（路径 i）/ `paired_bpc_ci`（prompt 聚类配对 bootstrap）/ `has_wall_clock` |
| 驱动 | `run.py` | L0/L1/L1b/L2/L3/L4/L5 + 判词 + 写盘 |
| 单测 | `test_freerun.py` | 11 项，全绿 |
| 变异 | `mutate.py` / `mutation_report.json` | 5/5 抓红 + 逐字节还原 + 还原后全绿；**无逐次耗时** |
| 原始读数 | `results.json` | 臂级 bpc、Δ+CI、生成流 sha256、代价轴、配置；**0 墙钟字段**（`L4`） |
| 结论 | `analysis.md` | 人话摘要在前；L0–L5 与 §5 逐条 |
| 权重清单 | `MODEL.md` | 主干/头 sha256 与复现命令 |
| 运行日志 | `run_log.txt` | 命令与 GPU 锁（可含墙钟） |
| GPU 锁 | `gpu_lock.py` | 复制自 2660 |

**L0 保真硬门**：头权重逐张量等于 2660 落盘文件（`heads_equal_vs_file`）；主干 `requires_grad=False`
且 backward 后 `.grad is None`；`ar` 贪心流与 2640 `native`（生产 `decode` 贪心，prompt_len=32）
**12/12 prompt 前 8 token 逐位相同**（`results.json::L0.ar_vs_2640_native`）。

## 5 未完成 / 下一步

- **未做**（超出本 WP 契约或时间盒）：k=4 的 free-run bpc；温度采样对照；把 free-run bpc 与
  teacher-forced val bpc 同框的“自评 vs 真值”分解。均登记在 `analysis.md::边界与未测`。
- **独立验收**：本 WP 只提供自测证据（pytest + 变异），**不是独立复核**；fresh reviewer 复核排在
  writer 停止之后（`docs`/`AGENTS.md` §4）。
- 时间盒：实际 GPU 计算 wall 20.5 s，远在 20 min 预算内。

## 6 复核 N1 返工：真值锚补充读数（只加读数，2026-10-14T21:33:32+08:00）

> 依据：`../REVIEW-C1-2670.md` 的 **N1**（normal，推进前需处理）+「A 的定级建议·最小补测命令」。
> **只加读数**：不改预登记的臂、口径、判据（`README.md` 未改，mtime 仍为 20:51:36）。

### 6.1 做了什么

- 把每个（臂 × 主干 × 头种子 × prompt）的 200-token 生成流落盘到 `streams/`（120 个文件），
  逐文件记 `sha256`；`streams.all_sha_reproduced=True`（与 `results.json::arms.*.stream_sha256_per_prompt` 逐一相等）。
- 新增**客观退化统计**（不需模型）：最长连续相同 token、distinct-2、重复 4-gram 率。
- 新增**真值锚**：用 `pathlm/eval.py::bpc` 同口径的 **teacher-forced node-1 bpc**，对每条流分别在
  **自评主干**与**另一主干（外部锚）**下打分；再给**真实 enwik8 续写**的 teacher-forced bpc 作水平参照。
- `freerun.py` 新增 `degeneracy_stats` / `node1_logits` / `teacher_forced_stream_nll`；
  `run.py` 新增 `streams` / `degeneracy` / `truth_anchor` 三块读数与 `analysis.md` 新节「真值锚（复核 N1 的补充读数）」。
- 未改 `README.md`、`pathlm/**`、`train_*.py`、`tools/**`、`docs/**`、其它轴、任何 `project.yaml`。

### 6.2 命令与真实结果（时间戳 + 命令）

| 时间 (UTC+08) | 命令 | 结果 |
|---|---|---|
| 2026-10-14 21:30:58 | `python3 run.py --out .`（取 `wp6.3.3-freerun` GPU 锁；wall 46.0 s） | `results.json` / `analysis.md` / `MODEL.md` 重写；`streams/` 落 120 文件；sha 全对齐 |
| 2026-10-14 21:31:52 | `python3 -c "run._write_analysis(...)"`（从 `results.json` 重生成 analysis，无需 GPU） | 加入「要点（人话）」三条 |
| 2026-10-14 21:33 | `python3 -m pytest test_freerun.py -q` | **13 passed**（新增 2 项） |
| 2026-10-14 21:33 | `python3 mutate.py`（连跑两次） | **7/7 抓红 + 逐字节还原 + 还原后全绿**；`mutation_report.json` md5 `6c84cfdd…` 两次一致（确定性）、不含逐次耗时 |

### 6.3 新读数（外部锚 vs 自评的 Δ 对照，两种子合并）

| 候选臂 | 主干 | Δbpc 自评 free-run（原判） | Δbpc 自评 teacher-forced | Δbpc **外部锚** teacher-forced | 外部锚 CI 不含0 且越 0.014 带 |
|---|---|---|---|---|---|
| `disc_k2` | 13m_s0 | +0.7389 | +4.1031 | **+4.1541** | 是 |
| `disc_k2` | 13m_s1 | +0.9195 | +5.0469 | **+5.0708** | 是 |
| `lat_k2_lam1` | 13m_s0 | +0.4289 | +1.4521 | **+1.2611** | 是 |
| `lat_k2_lam1` | 13m_s1 | +0.1633 | +1.1296 | **+1.1701** | 是 |

- **真实 enwik8 续写** teacher-forced bpc：13m_s0 = 1.6040、13m_s1 = 1.6027（水平参照）。
- **退化统计（12 prompt 平均）**：`ar` 最长连续相同 token 2.04、distinct-2 0.233；候选臂（尤其 `lat_k2_lam1`）
  最长连续相同 token ~137（即大半条流同一个字符）。

### 6.4 判读是否改变

- **未降级**：外部锚 Δ 与自评 Δ **同向（全部窗口 > 0）**，且**仍越 0.014 带**（CI 全不含 0）⇒ 原判读
  「路径 (i) 在本尺度质量面不成立」**成立**。
- 更强的发现：外部锚 Δ 的量级**比自评 Δ 更大**（自评低估了代价），因为 `ar` 的自由生成就是 node-1 贪心，
  其自评 bpc 与 teacher-forced node-1 bpc 恒等；原 Δ 只比较了「两个不同预测头各自的信心」。
  故复核担心的「`ar` 流退化 ⇒ 自评 bpc 被压低 ⇒ 系统性抬高 Δ」这一具体机制在落盘数据上**不成立**。
- 结论新增进 `analysis.md` 新节；原「人话摘要」「§5-1 判词」均保留（契约判据未动）。

### 6.5 未完成项

- 无（本返工全部要求项已完成）。可选后续：k=4 的 free-run bpc、温度采样对照（仍在 `analysis.md::边界与未测`）。
