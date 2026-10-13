# REVIEW-2 · 轴 2 Phase A 处置复核（revision `483b181`，基线 `233e978`）

> 复核者：**fresh 独立复核者**（独立上下文 subagent，未参与本线任何写作；本轮**有 shell**）。
> 被复核 revision：`483b181`（`ticket 61 轴 2 Phase A（修订）：按 REVIEW-1 处置 14 项 + 落盘复核报告`）。
> 上一版 `233e978` 已由 round 1 静态复核（报告 `.scratch/61-axis2-depth/REVIEW-1.md`，1 blocking F1 + 13 non-blocking F2–F14）。
> 本报告**不改任何被复核文件**，唯一写入 = 本文件 + `.scratch/61-axis2-depth/repro-review2/`（重跑证据）。
> 工作区核对：HEAD = `483b1811df696a36126a150cb774770a5b5e2512`；进入复核时 `git status` 干净（仅本次新增的未跟踪 `repro-review2/`）；未提交任何 git。

---

## ① 人话摘要与总判词

这段交付在说：普通 transformer 每个字符都跑满 8 层，这条线主张让**每个字符自己决定跑几层**。
`233e978` 把这个「逐字符退出」真的装进了生产解码器并跑了第一次序列级实测；round 1 复核判「技术结论成立」，
但提了 **1 条 blocking**：正文把「异构深度上下文的代价」写成「**≤ 0.003 bpc**」，而实测是 **+0.003296**（把数字写紧了）。
`483b181` 是**处置提交**：改数字、改措辞、补健壮性字段、并顺手修了解码器对 `proj_fill`/`fill_kv=False` 的语义坑。

**我这一轮把 round 1 跑不了的东西全跑了**，结论如下（两层分开看）：

**（一）运行证据与科学结论：全过。**
- `tests/test_axis2_exit_decision.py` = **19 passed**；`tests/` 全量 = **834 passed**（纯 CPU，未用 GPU）。
- **4 处变异全部被抓红**（改坏一处实现 ⇒ 至少一条测试失败），每次改完立即按字节恢复，`git diff` 全程干净。
- **重跑实验**（CPU，343 s）产出的 `results.json` 与仓内已提交版本 **sha256 逐位相同**（`6a8e2184…`）——臂 bpc、`sanity_anchor`、`frontier_comparison`、`same_position_paired`（含新增 `delta_window_se`/`delta_halves`）、`attribution_diagnostics` **全部相等**。
- `python3 tools/summary.py --check` = **OK**（46 个 `project.yaml`）。
- **F11 的「生产语义不变」我亲自验了**：`fill_kv=True`（本 checkpoint 的生产路径）下，新代码与 `233e978` 的 `_layer_kv` 输出**逐位相同**（`torch.equal` 全层 True）；`_missing_mask` 在 `fill_kv=True` 下恒返回 `None`；新 `_missing_mask` 的掩码语义与 `DepthARModel._ragged_mask` 一致（逐层列谓词 `depths > k0` ⇔ `key_reached = d_j ≥ k`，自写脚本对拍通过）。

**（二）处置是否「真的落盘且自洽」：不过 —— BLOCK（4 项，均为可机械修正的文本级）。**
- **B1（SSOT 计数错）**：`483b181` 给测试加了 3 条（16→19），但 `project.yaml` 的 `conclusion`/`conclusion_review`/`change_log`、`analysis.md §0/§10`、`README.md §1`、`TICKET §4.5` **仍写「16 条」**；同一份 `TICKET` 内 §7.2 又写「19 passed」⇒ 自相矛盾。
- **B2（处置声称的落点不存在）**：`TICKET §7.2` 的 F13 行声称「臂数 11 → 14，落点 `TICKET §4.5`」，但 `TICKET §4.5`（第 162 行）**仍写「11 条臂」**。
- **B3（F2 只落了一半）**：REVIEW-1 点名的 `analysis.md §7 ②` 仍写「省 ≥25% 的档位…0.0035–0.2593 bpc（0.25–18.5 倍）」，`analysis.md §8` 末行与 `project.yaml` 的 `argument`③、`failure_analysis`④ 同样残留（0.0035 是省 20% 那档的超出量；≥25% 档的真实区间是 0.0117–0.2593）。`conclusion_review` 却写「**处置（全部已落盘）**」。
- **B4（F10 只落了一半）**：`analysis.md §8` 的 U4 行仍写「发出流一致率 **28–55%**」（应为 28–71%）。

之所以仍然判 **BLOCK** 而不是 OK-with-notes：父级把「**处置与证据不符**」列为 blocking；上面的 B1–B4 恰是「处置记录/SSOT 事实与仓库现状不符」——尤其 B1 发生在本轮修订**自己**新加的测试上，属于**新引入**的事实错误。**科学内容本身没有新的 blocking**：F1（唯一的原 blocking）已按实测改对，生产语义逐位不变，测试对实现错误有抓力，数字可复现。所以这是一次「小修即可放行」的 BLOCK，不是结论翻案。

---

## ② 逐条命令与**实际输出**（任务 A）

工作目录 `cd /home/z/vibe/pathlm`；未取 GPU 锁；`483b181` 工作区。

### A1 目标测试文件 + 全量回归

```
$ python3 -m pytest tests/test_axis2_exit_decision.py -q
...................                                                      [100%]
19 passed in 3.23s

$ python3 -m pytest tests/ -q
........................................................................ [  8%]
...（略）...
..........................................                               [100%]
834 passed in 165.89s (0:02:45)
```

说明：`483b181` 的 `tests/test_axis2_exit_decision.py` 有 **19** 个 `def test_`（`grep -c` = 19），与 writer 自报「19 passed」一致；测试文件用 `torch`/CPU，未触碰 GPU。

### A3 重跑实验（纯 CPU，不取 GPU 锁）

```
$ python3 axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run.py \
      --out .scratch/61-axis2-depth/repro-review2
[ckpt] .scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt sha256=2b054ab7368757fe79cf36c32bacacb5e7aafe52f718a02f0558dbb3aa5446e6
[data] enwik8 val；样本 = batch(seed=2) × 20 × 16，T=256 ⇒ W=320 窗口，预测位置=81600
[model] params=6962688 cfg={'d_model': 256, 'n_layers': 8, 'n_heads': 8, 'seq_len': 256, 'mlp_mult': 4} dense=False fill_kv=True proj_fill=False carry_gate=True L=8
[arm] never-exit: bpc=1.8582 ...  [arm] prescribed-2: bpc=1.9988 ...  [arm] token-thr-0.9: bpc=1.8702 layer-passes=5.967
[arm] token-thr-0.999-min4: bpc=1.8582 layer-passes=7.638
[frontier] ... token-thr-0.9: 超出 +0.0117; token-thr-0.999: 超出 -0.0002 ...
[anchor] prescribed-2: Δbpc=+0.0000 Δtf=-0.0054; ... prescribed-8: Δbpc=+0.0000
[calib] k=3 ... ECE_true=0.005; ... k=5 ... ECE_true=0.003
[same-pos] token-thr-0.6@d8: 3.372 vs 同位置全深度 3.369 (Δ+0.003, n=34452); token-thr-0.9@d8: ... (Δ+0.001, n=54457)
[gen] token-thr-0.999-min4: 一致率 91/128 = 0.7109
[done] 343.3s（过程指标，不进判据）；results.json 已写
```

**对账（新目录 vs 已提交 `results.json`）**：

```
$ sha256sum axes/.../2631-pertoken-exit-seqlevel/results.json .scratch/61-axis2-depth/repro-review2/results.json
6a8e2184fd7ffff02c0db212fae75209620c6b24f3a40ec4d8d4b2db2c2ed2e1  （已提交）
6a8e2184fd7ffff02c0db212fae75209620c6b24f3a40ec4d8d4b2db2c2ed2e1  （重跑）
$ diff -q ...（两文件无差异）  ⇒ RESULTS BYTE-IDENTICAL

字段级对账（自写 cmp 脚本）：
  top-level keys 相同；arms 名相同（n=14）；14 臂 × {bpc, delta_bpc_vs_full, delta_bpc_paired_se,
  layer_passes_per_token, skip_frac, exit_acc_tf, exit_acc_true} 全部相等；
  sanity_anchor 相等；frontier_comparison 相等；attribution_diagnostics 相等；
  fixed_depth_frontier 相等；same_position_paired 16 行全部相等；
  新增字段 delta_window_se / delta_halves 在已提交版本中 present=True、all_nonempty=True。
```

`same_position_paired` 关键值（与 `analysis.md` §5 一致）：
- `token-thr-0.6@d8`：`n=34452, delta=0.003296, delta_window_se=0.001157, delta_halves=[0.002038, 0.004549]`
- `token-thr-0.9@d8`：`n=54457, delta=0.000552, delta_window_se=0.000631, delta_halves=[0.000666, 0.000438]`
- `calibration.rows[depth=3].ece_true = 0.0051`

（重跑证据保留在 `.scratch/61-axis2-depth/repro-review2/{results.json,run_log.txt}`；**未**覆盖仓内已提交的 `results.json`。）

### A4 研究树校验

```
$ python3 tools/summary.py --check
OK：46 个 project.yaml 全部符合 docs/research-schema.md。
EXIT=0
```

### 附带：F11 生产语义与掩码语义的独立对拍

自写 `/tmp/verify_f11.py`（把 `233e978` 的旧 `_layer_kv` 作为函数内联复刻，与当前 `_layer_kv` 逐位比较）：

```
=== (1) fill path comparison vs old _layer_kv ===
  [fill_kv=True, proj_fill=False]          layer 0/1/2 bitwise_equal=True  max|dK|=0.000e+00  ⇒ ALL True
  [fill_kv=True, proj_fill=True(恒等)]     layer 0/1/2 bitwise_equal=True  max|dK|=0.000e+00  ⇒ ALL True
  [fill_kv=False, proj_fill=False]         layer 0/1/2 bitwise_equal=True  （填充仍在，差别只在掩码）
  _missing_mask(0)：fill_kv=True 时 = None
=== (1b) proj_fill 非恒等 adapter：填充路径必须不同（F11 修复的活性）===
  max|dK| per layer = ['1.336e-02', '1.820e-02', '2.625e-02']  ⇒ 非零 = 修复真的生效
=== (2) _missing_mask vs _ragged_mask 语义 ===
  layer k0=0(k=1)…2(k=3)：_missing_mask 的「过去列」谓词 == (depths > k0)，与 forward 的 key_reached(d_j ≥ k) 相同 → match=True
  末列（当前 token 自身）恒可见 = True
```

结论：F11 在 `fill_kv=True`（生产语义）下**没有改变数值**；仅在 `proj_fill=True 且 adapter 非恒等`、以及 `fill_kv=False` 的新路径上才改变数值——正是修复意图。

---

## ③ 变异测试逐项（任务 A2）

方法：对 `483b181` 的源文件做**精确单点替换**，跑 `tests/test_axis2_exit_decision.py -q`，然后按字节恢复原文件并复算 sha256（脚本 `/tmp/mut.py`，备份写到 `/tmp` 后删除，不使用 `git checkout/stash`）。

| # | 改了什么（文件:行） | 变异前 → 变异后 | 哪些测试红了 | 恢复 |
|---|---|---|---|---|
| A | `pathlm/exit_policy.py:163` | `want = want & ~decided` → `want = want` | **1 failed**：`test_threshold_zero_exits_at_min_depth`（`assert {3} == {2}`） | ✓ `sha 47cc2ae7…` 逐位还原；`git diff --exit-code` 干净 |
| B | `pathlm/depth_ar.py::step_auto` | `exit_depth, frozen_self = depth, h` → `… = depth, inp` | **2 failed**：`test_step_auto_matches_prescribed_schedule`、`test_projfill_nonidentity_decoder_matches_forward`（`position 1` allclose 失败） | ✓ `sha 1f8f9eb…` 逐位还原；干净 |
| C | `pathlm/depth_ar.py::_layer_kv` | 删掉 `if model.cfg.proj_fill: src = model.kv_adapters[k0](src)`（无条件 `ln1(frozen)`） | **1 failed**：`test_projfill_nonidentity_decoder_matches_forward` | ✓ `sha 1f8f9eb…` 逐位还原；干净 |
| D | `pathlm/depth_ar.py::_missing_mask` | 恒 `return None` | **1 failed**：`test_nofill_decoder_matches_forward` | ✓ `sha 1f8f9eb…` 逐位还原；干净 |

- 每条变异都**至少一条测试失败**（不是「静默通过」）⇒ 新测试对「改坏一处实现」有抓力。
- 全部 4 处恢复后逐次 `git diff --exit-code` 干净；复核结束时工作区唯一未跟踪项 = 本任务的 `repro-review2/`。
- 注：`depth_ar.py` 的 sha 在变异前恒为 `1f8f9eb49aac46d3e076f2964b49cfceae8215c13640b65d10a3403ef6c10d7a`（= `483b181` 版本），`exit_policy.py` 恒为 `47cc2ae71451326ca39ba319809ab06e88963d3e54f11f10154b16b1f6651194`。

**诚实披露（流程事故，已修复）**：我第一次把变异 C 与 D **并行**跑（两者都改 `pathlm/depth_ar.py`），发生写冲突——C 首跑被 D 污染、且结束时文件残留了 C 的变异。我立即用 `git show 483b181:pathlm/depth_ar.py` 精确还原（sha 回到 `1f8f9eb…`，`git diff` 干净），再**串行**重跑 C、D，上表是重跑后的干净结果。全程未 `checkout/stash` 他人的东西，未提交 git。

---

## ④ `233e978 → 483b181` 增量处置核查表（任务 B）

命令：`git diff 233e978 483b181 -- <路径>`（含 `pathlm/depth_ar.py`、`pathlm/exit_policy.py`、
`tests/test_axis2_exit_decision.py`、`axes/2-depth/2.1.1-early-exit/{project.yaml,experiments/2631-*}`、
`.scratch/61-axis2-depth/TICKET.md`）。

| # | REVIEW-1 诉求 | `483b181` 实际落盘 | 判定 |
|---|---|---|---|
| **F1** | 「异构深度上下文代价 ≤0.003 bpc」与实测矛盾 | `project.yaml` `conclusion` 改「≤ 0.005 bpc」、`failure_analysis` 写「d8 桶 +0.0033（SE 0.0012；两半 +0.0020/+0.0045）」、`conclusion` 写「ECE_true ≤0.030、深度≥3 最大 0.0051」；`analysis.md` §0 第5点、§5（机制结论）、§6 同改。核对：0.005 ≥ 0.0033 且 ≥ 两半最大值 0.0045；`results.json` 实测 `delta=0.003296`、`ece_true(depth3)=0.0051` —— **一致** | ✅ 已落盘 |
| **F2** | ≥25% 档区间应 0.0117–0.2593（0.84–18.5×），并单列 τ=0.95 | `project.yaml` `conclusion` 已改对；**但** `analysis.md` §7 ②、§8 末行、`project.yaml` `argument`③、`failure_analysis`④ 仍写「≥25% ⇒ 0.0035–0.2593（0.25–18.5×）」/「主结论 Δ 是 8–18 倍」 | ⚠️ **部分落盘**（见 blocking B3） |
| **F3** | 删掉「分桶闭合很难由巧合产生」的说法 | `failure_analysis` 改「算术闭合（恒等式，仅用于确认归因口径）」；`analysis.md` §6 诊断行改「损伤**集中在**…（d8 桶隔离出的上下文效应 ≤0.005；浅层桶两项未分离）」 | ✅ 已落盘（`argument`④ 仍陈述「+0.12046 ≈ 总 Δ」，但只是等式陈述，未作佐证） |
| **F4** | `dominated_by_fixed` 加横轴条件 `x <= lp` | `run.py` 改为 `any(x <= lp and y <= d for x, y in fxd)`。`results.json` 复核：`extra_layers_vs_dominated_fixed_arm` 全为非负（0.33/1.27/1.97/1.40/0.03）或 `None`（未被支配的 τ=0.999、min4），不再出现负值 | ✅ 已落盘 |
| **F5** | 历史 `tf_acceptance` 另一抽样（seed 3），只作量级参考 | `MODEL.md §3`、`README.md §1`、`run.py:58` 注释、`analysis.md §3` 均改；`analysis.md §3` 现写「`prescribed-2` 的 Δ 达 −0.0054 ⇒ 该列**不**用于对齐」 | ✅ 已落盘 |
| **F6** | 「逐位重合」改「浮点噪声内重合（Δbpc = 7e-06…4.6e-05）」 | `analysis.md §3` 已改并逐行给 Δ；**但** `analysis.md §3` 末又写「数值上重合到 1e-05」（实际最大 4.6e-05，偏紧）、`project.yaml` `argument`② 仍写「2..8 全部 Δ=±0.0000」 | ⚠️ **部分落盘**（见 non-blocking N1） |
| **F7** | 「从未超过」改「从未**显著**超过」 | `project.yaml` `conclusion`、`analysis.md` §0 第4点、§4、§7 ② 均改；`conclusion` 并写明唯一未被支配的两档（τ=0.999 / min4，省 4.5–6.5%） | ✅ 已落盘 |
| **F8** | §4 表末列 0.99 行 `0.01×` → `0.00×` | `analysis.md §4` 表已改 `0.00×` | ✅ 已落盘 |
| **F9** | 同位置配对补逐窗口 SE / 两半样本并重跑 | `run.py` 新增 `n_windows`/`delta_window_se`/`delta_halves`；`results.json` 已含且非空（16 行全 present、非空）；数值与 `analysis.md` 一致（0.0012 / +0.0020+0.0045） | ✅ 已落盘 |
| **F10** | §5 补 depth 5 行；生成一致率区间改 28–71% | `analysis.md §5` 补 `| 5 | 0.628 | 0.987 | 0.627 | 0.359 | 0.003 |`；§0 第6点改「28–71%」；**但** §8 的 U4 行仍写「28–55%」 | ⚠️ **部分落盘**（见 blocking B4） |
| **F11** | `_layer_kv` 经 `kv_adapters`；`fill_kv=True` 保持原路径；加 no-fill 掩码 | 见 ②：`fill_kv=True` 下新代码与 `233e978` **逐位相同**；`proj_fill` 非恒等才改变数值；`_missing_mask` 在 `fill_kv=True` 恒 `None`；掩码语义与 `_ragged_mask` 一致；新增 3 条测试（19 总数） | ✅ 已落盘（数值安全性已验） |
| **F12** | 落盘 round 1 报告；`conclusion_review` 改复核判词、不悬空 | `REVIEW-1.md` 已在盘；`conclusion_review` 改为「round 1 判词 + 处置 + 残余缺口」，无悬空指针；`work_packages[WP1].review` 仍指向 `REVIEW-*.md`（该路径现已存在） | ✅ 已落盘 |
| **F13** | 臂数 11→14；`run_log.txt` 覆盖模式；删 `bpc_per_window_rounded` | `run.py:200` 改 `"w"` ✓；`run.py` 删除 `bpc_per_window_rounded` ✓（`results.json` 两版均无该键）；`README.md §2` 新增「共 **14 条臂**」✓；**但** `TICKET.md §4.5`（第 162 行）仍写「11 条臂 + 4 类归因诊断」，与 `TICKET §7.2` 声称的落点「TICKET §4.5」不符 | ⚠️ **部分落盘**（见 blocking B2） |
| **F14** | `generate_with_exit` 的 `depths` 语义标注改正 | `exit_policy.py:287` docstring 改为「产出第 j 个生成 token 的那个位置的退出深度」+「`layer_passes` 覆盖 prompt + 前 n_new−1 个位置」 | ✅ 已落盘 |

附带（非 F 列表）：`2.1.3-repeating` / `2.2.5-weight-sharing` 各 `gaps` 加一条「同题风险…见 TICKET §8 F4（待 Main 裁）」——与 `TICKET §8` 一致，属合理边界声明。

---

## ⑤ blocking 清单（位置 + 违反判据 + 复现）

> 违反判据一律是父级给的本条：**「处置与证据不符」**；B1 另属「$483b181 新引入的事实错误」。

**B1 —— SSOT/文档仍写「16 条单测」，实际 19 条（本轮修订自己引入的不一致）。**
- 位置（写「16 条」的地方）：`axes/2-depth/2.1.1-early-exit/project.yaml:17`（`conclusion`）与 `:18`（`conclusion_review`）与 `:115`（`change_log`）；`.../2631-pertoken-exit-seqlevel/analysis.md:20`（§0 第2点）与 `:234`（§10）；`.../README.md:20`；`.scratch/61-axis2-depth/TICKET.md:152`（§4.5）。
- 证据：`grep -c "^def test_" tests/test_axis2_exit_decision.py` = **19**；`python3 -m pytest tests/test_axis2_exit_decision.py -q` ⇒ **19 passed**；`TICKET §7.2` 自己也写「19 passed（含 F11 新增 2 条）」⇒ 同一 ticket 内 §4.5（16）与 §7.2（19）自相矛盾。
- 复现：`grep -rn "16 条" axes/2-depth/2.1.1-early-exit/ .scratch/61-axis2-depth/TICKET.md` 与 `grep -c "^def test_" tests/test_axis2_exit_decision.py`。
- 最小修法：把上述 6 处的「16 条」改为「19 条」，并订正 `TICKET §7.2` 的「新增 2 条」为「新增 3 条」。

**B2 —— `TICKET §7.2` 的 F13 声称落点 `TICKET §4.5`，但 §4.5 未改。**
- 位置：`.scratch/61-axis2-depth/TICKET.md:162`（§4.5）仍写「11 条臂 + 4 类归因诊断」；对照 `TICKET.md` §7.2 表 F13 行「臂数 11 → 14 … 落点 `TICKET §4.5`、`run.py`」。
- 证据：`results.json` 与 `analysis.md §2` 的臂数为 **14**；`README.md §2` 已写「共 14 条臂」；唯独 `TICKET §4.5` 未更新。
- 复现：`grep -n "条臂" .scratch/61-axis2-depth/TICKET.md`（命中第 162 行「11 条臂」）。
- 最小修法：`TICKET §4.5` 第 162 行「11 条臂」改「14 条臂」。

**B3 —— F2 残留：`analysis.md §7 ②`（REVIEW-1 明确点名的位置）等仍把 20% 档当 ≥25% 档。**
- 位置：`analysis.md:194`（§7 ②）「省 ≥25% 的档位超出固定前缀前沿 **0.0035–0.2593** bpc（0.25–18.5 倍）」；`analysis.md:214`（§8 末行）「主结论 Δ 是 8–18 倍噪声带」；`project.yaml:31`（`argument`③）同款区间 + `project.yaml:16`（`failure_analysis`④）「主结论 Δ 是 8–18 倍」。
- 证据：`results.json` 中「省 ≥25%」的臂是 τ=0.3/0.6/0.9（`layer_passes` 2.332/4.268/5.967），其 `frontier_excess` = **0.2593/0.1168/0.0117**；0.0035 属 τ=0.95（省 20%）⇒ 区间下界写错、集合错配。
- 复现：`python3 -c "import json;...;print([(r['arm'],r['layer_passes_per_token'],r['frontier_excess']) ...])"`（见 ④ F4 输出）。
- 最小修法：把上述 4 处区间改成「0.0117–0.2593（0.84–18.5×）」，或把限定词改成「省 ≥20%」；`failure_analysis`④ 的「8–18 倍」改为「超出量是 8.3–18.5 倍」。

**B4 —— F10 残留：`analysis.md §8` 的 U4 行仍写「28–55%」。**
- 位置：`analysis.md:208`（§8 U4）「本实验只测了『发出流一致率 **28–55%**』这个诊断列」。
- 证据：`results.json::generation_agreement` 的最高一致率 = `token-thr-0.999-min4` **0.711（91/128）**，`analysis.md §9` 表也列了它；§0 已改「28–71%」，§8 未改 ⇒ 同文件内 28–55% 与 28–71% 冲突。
- 复现：`grep -n "28–55\|28–71" analysis.md`（命中第 54 行「28–71%」与第 208 行「28–55%」）。
- 最小修法：§8 U4 行「28–55%」改「28–71%」。

---

## ⑥ non-blocking

| # | 位置 | 问题 | 最小修法 |
|---|---|---|---|
| N1 | `analysis.md:114`（§3 末） | F6 修完又留尾巴：「数值上重合到 **1e-05**」，而实测最大 Δbpc = **4.6e-05**（偏紧约 5×）；且有「。。」重复句号 | 改「重合到 ≤5e-05」并删多余句号 |
| N2 | `project.yaml:31`（`argument`②） | 仍写「`prescribed-k`（2..8 全部 **Δ=±0.0000**）」，与 F6 后的「Δbpc ≤ 4.6e-05」口径不一致 | 改「Δbpc ≤ 4.6e-05（浮点噪声内）」 |
| N3 | `analysis.md:178`（§6 诊断行） vs `analysis.md:44`（§0 第5点） | §6 已软化为「损伤**集中在**……浅层桶两项未分离」，§0 仍说「损伤**全部**来自……」——同文件措辞不一 | §0 第5点改「损伤主要来自……」 |
| N4 | `project.yaml:18`（`conclusion_review`） | 写「**处置（全部已落盘）**」，但 B3/B4 显示 F2/F10 未全落盘（措辞过强） | 改「主要处置已落盘；F2/F10 残留见 REVIEW-2 §⑤」 |
| N5 | `pathlm/depth_ar.py::_missing_mask` | 返回 `[1,1,S+1]` 掩码未 expand 到 head 维；同仓 `_ragged_mask`/`_window_mask` 均 expand 到 `n_heads`，注释写明「CUDA SDPA 内核不广播 bool 掩码的 head 维」。本实验 CPU、解码器为 CPU 路径 ⇒ 未触发，但若他人在 GPU 上用 `step_auto` + `fill_kv=False` 可能踩坑 | 与 `_ragged_mask` 同款 `expand(-1, n_heads, -1, -1)`，或加注释说明仅 CPU |
| N6 | `analysis.md §10`/`§0 第2点`、`README.md:20`、`project.yaml` 的 `change_log` | 与 B1 同源（「16 条」）；已并入 B1，不重复计 | 见 B1 |

（`analysis.md:38` §0 第4点的「0.8–18 倍噪声带」与其「省 25–71%」限定一致，**不是**缺陷；`README.md §2` 的「14 条臂」正确。）

---

## ⑦ 未覆盖部分（诚实登记）

1. **未在 GPU 上验证**：所有测试/重跑均在 CPU；`_missing_mask` 的 head 维广播问题（N5）在 CUDA 上是否出错未测（本任务明确「不要用 GPU」，且实验口径为 CPU 纯推理）。
2. **未做多次置换 seed / 分块稳健性扩展**：`attribution_diagnostics` 的置换臂仍只有 1 个置换 seed=1234；F9 新增的是**同位置配对的逐窗口 SE 与两半**，未覆盖置换臂的稳健性（属未测项，不是缺陷）。
3. **未验证 `_missing_mask` 在被外部代码（`generate`、`spec_decode.py` 等）间接调用时的全部组合**：只验证了 `step`/`step_auto`（`fill_kv=True/False`、`proj_fill=True/False`）与并行 `forward` 的对拍；`generate()` 的 fill_kv=False 自由生成路径未单独对拍（无对应测试）。
4. **未跑 `role_map` 之外的 schema 全量**：仅以 `summary.py --check`（46 份）为准，未逐字段人工复核 10 份 `project.yaml`（round 1 已做过该静态核对，本轮不重复；本轮新增的 2.1.3/2.2.5 两条 `gaps` 经 `--check` 通过）。
5. **未核 `233e978` 之前的历史**：B1–B4 的「16→19」「11→14」等对照基于 `233e978` 与 `483b181` 两个 revision 的 diff 与文件现状；未追溯更早 revision。
6. **未提交 git、未改动被复核文件**：唯一写入 = 本报告 + `.scratch/61-axis2-depth/repro-review2/`（未跟踪）。变异测试期间的写冲突已按字节还原（见 §③ 披露）。

---

## 判词（合并结论）

- **运行证据与科学结论：PASS。** 19/834 测试全绿；4/4 变异被抓红；重跑 `results.json` 与仓内**逐位相同**；`summary.py --check` OK；F11 在 `fill_kv=True` 生产语义下**数值逐位不变**、掩码语义与 `_ragged_mask` 一致。
- **处置落盘自洽性：BLOCK（B1–B4，均为文本级小修）。** 原 blocking F1 已正确修复；但本轮修订**新引入**「16 条单测」的事实错误（B1）、并有一处处置声称的落点未落盘（B2）、F2/F10 只落了一半（B3/B4）。均不改变任何科学结论。
- **Merge verdict：BLOCK**，修完 B1–B4（预计 8–10 处文本）即可判 OK；无新的科学/实现/口径类 blocking。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "以 fresh 独立上下文、有 shell 的复核者身份完成两任务：A1-A4 全部实跑（19 passed / 834 passed / 4 处变异全红并逐位恢复 / 重跑 results.json 与仓内 sha256 逐位相同 / summary --check OK）；B 用 git diff 233e978 483b181 逐条核 F1-F14，给出落盘表与 4 项 blocking + 6 项 non-blocking。未改任何被复核文件。"
    },
    {
      "id": "criterion-2",
      "status": "satisfied",
      "evidence": "报告含 ①人话摘要+总判词 ②逐条命令与实测输出 ③变异测试逐项 ④F1-F14 处置核查表 ⑤blocking 清单（位置+判据+复现）⑥non-blocking ⑦未覆盖部分；每条 blocking 带 grep/pytest 复现命令与结果；PASS/BLOCK 均有证据支撑。"
    }
  ],
  "changedFiles": [
    ".scratch/61-axis2-depth/REVIEW-2.md",
    ".scratch/61-axis2-depth/repro-review2/results.json",
    ".scratch/61-axis2-depth/repro-review2/run_log.txt"
  ],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "python3 -m pytest tests/test_axis2_exit_decision.py -q",
      "result": "passed",
      "summary": "19 passed in 3.23s"
    },
    {
      "command": "python3 -m pytest tests/ -q",
      "result": "passed",
      "summary": "834 passed in 165.89s（CPU，未用 GPU）"
    },
    {
      "command": "mutation A: exit_policy.py `want = want & ~decided` -> `want = want`（跑后恢复）",
      "result": "passed",
      "summary": "1 failed（test_threshold_zero_exits_at_min_depth）；恢复后 sha 逐位相同、git diff 干净"
    },
    {
      "command": "mutation B: depth_ar.py step_auto `exit_depth, frozen_self = depth, h` -> `depth, inp`",
      "result": "passed",
      "summary": "2 failed（test_step_auto_matches_prescribed_schedule、test_projfill_nonidentity_decoder_matches_forward）；已逐位恢复"
    },
    {
      "command": "mutation C: depth_ar.py::_layer_kv 去除 proj_fill adapter 分支",
      "result": "passed",
      "summary": "1 failed（test_projfill_nonidentity_decoder_matches_forward）；已逐位恢复"
    },
    {
      "command": "mutation D: depth_ar.py::_missing_mask 恒返回 None",
      "result": "passed",
      "summary": "1 failed（test_nofill_decoder_matches_forward）；已逐位恢复"
    },
    {
      "command": "python3 axes/2-depth/2.1.1-early-exit/experiments/2631-pertoken-exit-seqlevel/run.py --out .scratch/61-axis2-depth/repro-review2",
      "result": "passed",
      "summary": "343.3s；新 results.json 与仓内已提交版本 sha256 6a8e2184… 逐位相同；全字段对账相等；新增 delta_window_se/delta_halves 非空"
    },
    {
      "command": "python3 tools/summary.py --check",
      "result": "passed",
      "summary": "OK：46 个 project.yaml 全部符合 docs/research-schema.md"
    },
    {
      "command": "git diff 233e978 483b181 -- <paths>（逐条核 F1-F14）",
      "result": "passed",
      "summary": "F1/F3/F4/F5/F7/F8/F9/F11/F12/F14 已落盘；F2/F10/F13 部分落盘；B1「16 条 vs 19」为新增不一致"
    },
    {
      "command": "自写 /tmp/verify_f11.py（旧 _layer_kv 复刻 vs 新 _layer_kv 逐位比较 + _missing_mask/_ragged_mask 语义对拍）",
      "result": "passed",
      "summary": "fill_kv=True（proj_fill=False/恒等 adapter）下逐位相同；非恒等 adapter 下才不同；掩码列谓词 depths>k0 与 key_reached(d_j≥k) 一致"
    }
  ],
  "validationOutput": [
    "19 passed（目标文件）；834 passed（全量）。",
    "变异：A 红 test_threshold_zero_exits_at_min_depth；B 红 test_step_auto_matches_prescribed_schedule + test_projfill_nonidentity_decoder_matches_forward；C 红 test_projfill_nonidentity_decoder_matches_forward；D 红 test_nofill_decoder_matches_forward。4/4 恢复后 sha 逐位还原、git diff 干净。",
    "重跑结果 sha256 = 6a8e2184fd7ffff02c0db212fae75209620c6b24f3a40ec4d8d4b2db2c2ed2e1（与仓内 results.json 相同）；arms(14)/sanity_anchor/frontier_comparison/same_position_paired(16 行)/attribution_diagnostics 全部相等；same_position_paired.{delta_window_se,delta_halves} 已存在且非空。",
    "F11 数值安全：fill_kv=True 下新 _layer_kv 与 233e978 旧实现 torch.equal 全层 True（max|dK|=0.0）；_missing_mask(fill_kv=True)=None；_missing_mask 掩码语义 == _ragged_mask 的 key_reached。",
    "summary.py --check ⇒ OK（46 份）。",
    "blocking 复现：grep -n 条臂 TICKET.md ⇒ 第162行「11 条臂」；grep -c ^def test_ ⇒ 19 vs 多处「16 条」；results.json frontier_excess（≥25% 档 0.2593/0.1168/0.0117，0.0035 属 τ=0.95）；grep 28–55 analysis.md ⇒ 第208行。"
  ],
  "residualRisks": [
    "BLOCK 的 4 项（B1-B4）是文本级不一致，不影响科学结论，但违反父级「处置与证据不符」判据；修完需再核。",
    "N5：_missing_mask 未 expand 到 head 维，CUDA SDPA 下是否可用未测（本轮明确禁用 GPU，实验为 CPU 路径）。",
    "置换臂仍只用 1 个 seed；F9 只补了同位置配对的窗口 SE/两半，未覆盖置换臂稳健性。",
    "generate() 的 fill_kv=False 自由生成路径未单独对拍（只对拍了 step/step_auto）。",
    "变异测试期间发生过一次并行写冲突（C/D 同改 depth_ar.py），已用 git show 精确还原并串行重跑；工作区最终干净，但该流程事故已如实登记。"
  ],
  "noStagedFiles": true,
  "diffSummary": "本复核未修改任何被复核文件；唯一新增为报告 .scratch/61-axis2-depth/REVIEW-2.md 与重跑证据目录 .scratch/61-axis2-depth/repro-review2/{results.json,run_log.txt}（均未跟踪）。被复核区间 233e978→483b181 的代码/文档改动经 git diff 逐文件核查（F1-F14 见 §④）。",
  "reviewFindings": [
    "blocker: project.yaml:17/:18/:115、analysis.md:20/:234、README.md:20、TICKET.md:152 - 写「16 条单测」而实际 19 条（本轮 F11 新增 3 条测试未同步计数），且 TICKET §4.5(16) 与 §7.2(19) 自相矛盾。修法：6 处改「19 条」、TICKET §7.2「新增 2 条」改「新增 3 条」。",
    "blocker: TICKET.md:162（§4.5）- 仍写「11 条臂」，而 TICKET §7.2 声称 F13 落点 TICKET §4.5 已改；实际臂数 14。修法：改「14 条臂」。",
    "blocker: analysis.md:194（§7②）、analysis.md:214（§8末行）、project.yaml:31（argument③）、project.yaml:16（failure_analysis④）- 仍写「省 ≥25% ⇒ 0.0035–0.2593（0.25–18.5×）」/「主结论 Δ 8–18 倍」，与 results.json（≥25% 档 excess=0.2593/0.1168/0.0117；0.0035 属省 20% 的 τ=0.95）不符。修法：改区间为 0.0117–0.2593（0.84–18.5×）或改限定词为 ≥20%。",
    "blocker: analysis.md:208（§8 U4）- 仍写「一致率 28–55%」，应为 28–71%（results.json 最高 0.711）。修法：改 28–71%。",
    "non-blocking: analysis.md:114 - 「重合到 1e-05」偏紧（实测最大 4.6e-05）+「。。」重复标点；project.yaml:31 argument② 仍「Δ=±0.0000」；analysis.md:44 §0 第5点仍「损伤全部」而 §6 已改「集中在」；project.yaml:18 conclusion_review 的「处置（全部已落盘）」措辞过强；pathlm/depth_ar.py::_missing_mask 掩码未 expand 到 head 维（CPU 路径未触发，CUDA 未测）。"
  ],
  "manualNotes": "判词：运行证据与科学结论 PASS，处置落盘自洽性 BLOCK（B1-B4，文本级小修）。原 blocking F1 已正确修复（≤0.005 ≥ 实测 0.0033 且 ≥ 两半最大 0.0045；ECE_true 0.0051）；F11 在 fill_kv=True 生产语义下逐位不变。重跑证据在 .scratch/61-axis2-depth/repro-review2/（未覆盖仓内 results.json）。"
}
```
