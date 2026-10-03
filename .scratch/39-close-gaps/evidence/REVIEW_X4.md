# 独立复核 X4（ticket 39）：W1/W2/W3/W4 + K9/K10

## Review

范围：W1（`R-references/R1-quantization/experiments/2621-kv-budget-rest`）、W2（`C-vertical/C1-layers-per-token/experiments/2621-depth-draft-e2e`）、W3（`M-carrier/M1-carrier-type/experiments/2621-carry-under-compression`）、W4（`W-width/W4-readout-geometry/experiments/2621-per-depth-readout-13m`）、K9（`axes/G-policy/PRIOR_ART_DECORRELATION.md`）、K10（`axes/A-mixing/PRIOR_ART_LENGTH.md`）。只读；无 shell；`watchdog_diff`（工作树 vs launch HEAD）为空、仅 ticket-39 issue 文件 untracked ⇒ 磁盘内容即已提交修订。§14b 顺序屏障满足（执行波已落盘，本复核在 ticket 39 新波启动）。

### - Correct（已核实、正确）

- **判据线未被事后改动**：W1 `README.md:26-38` 逐字抄录 W1 契约；W2 `README.md:73-102` 预登记；W3 `README.md:75-97` 逐字抄录；W4 `README.md:131-163` 预先写下。W3 §4b 的「GQA 因子未测」是 **主 Agent 裁决**下的唯一偏离，且带 `depth_ar.py:1690-1734` 互斥校验证据，符合 §22.8「不可组合 ⇒ 降契约或立后续项」。**Pass**。
- **数字与 `results.json` 逐一对应（抽查 22 项全部命中）**：W1 `cost.wall_seconds=142.38/wall_minutes=2.373`、`peak 462.5 MiB`、`reproducibility_vs_2619/2620.max_bpc_abs_diff=0.0`、`gqa_int4=1.592538`、`mqa_int4=1.689346`、`win32_int8=1.69569`、`win64_int8=1.56417`、`win16_anchor16_int4=1.943239`、`win32_anchor32_int4=1.643489`、`gqa_win64=1.5591`、`gqa_win128=1.512094`、reconcile `587/587 n_fail=0`；W2 交错墙钟 `43.77/130.62/88.21/163.83/142.35` tok/s 与 ×1.0/2.984/2.015/3.743/3.252、`acceptance_pooled 0.9423/0.9688/0.9271/0.9907`、`tf_enwik8@4=0.9694`、`selfstream@4=0.9812`、`accept_by_pos[1]=0.9704`、`eq_layerpos_per_token 8.45/8.375/8.0/4.0`、`quality_guard.total_mismatch=0`、`gpu_peak_mem_MB=218.1`、`cycle_log 867 行`（graded 的 `mid_accepted` 分布 {4:156,3:4,2:4,1:2,0:3} 逐值核对一致）；W3 `accuracy 0.09375/0.092773`、CI `[0.081875,0.107146]/[0.08096,0.106112]`、毁源 `0.007324`、`bpc 103.611675/103.61179/103.609283`、`rel_err 1.762e-4/4.358e-3`、`changed_frac 0.9998`、reconcile `47/0`；W4 `params 6962688/8142336/8142336`、`equal_params=true`、`bpc 1.8582/1.7823/1.7269/1.7770/1.7694`、`Δ −0.0759/−0.1313/−0.0812/−0.0888`、`PD−EP=0.0554`、`tf@4 0.9694/0.9727/0.9576`；K9 `pairwise_js l1=0.033320/divl1=0.025044 bits`、`gate_skips=5121`、`n_diversity_observations=6879`；K10 步时 `23.175/26.693/36.62/88.837`、显存 `0.809/1.48/2.859/5.585 GiB`、`other=786432`。**Pass**。
- **① W2 的「同缓存机械」对照公平。** `spec_decode.py:358-379` 的 `plain_greedy_cache` 与 spec 路径共用 `run_layers` / `CommittedCache` / `prefill` / `commit_input`（`spec_decode.py:32-160`），差异只剩「每步 1 位置 vs 每轮 γ+1 位置批量验证」——这是 spec 机制**固有**而非实现层，且已用 `eq_layerpos_per_token 8.45 vs 8.00` 与「launch-bound」解释并报。实现层 2.02× 被正确隔离到仓库 `DepthARDecoder` 路径（`run.py:198-206`、`analysis.md §3`）。**Pass**。
- **② W4 的 §15 等参数对照确为等参数。** PD = (L+1)·2d² = 1,179,648；EP = 2·d·m（m=2304）= 1,179,648；两臂总参数各 8,142,336，`results.json::reconciliation.arithmetic.equal_params=true`，并由 `test_readout_wide_shared_equal_params_and_identity` 钉住。EP「名义等参数靠冗余、非等有效容量」已被本线显式声明（`analysis.md §6`）。**Pass（有注）**。
- **④ K9/K10 分诊符合 §21。** 两文件均在最前声明联网不可用、外部条目一律 `记忆·未核`「不得当判据」；K9 把全部 `(a)` 降级为 `(a·记忆待核)`，并明确 `(c)` 不得据以判 `(a)`；K10 表格把位置编码族判 `(a) 业界已透 ⇒ 不跑实验`、把「13M 绝对 PE 拉到长 T」与「长 T 交叉点」判 `(b)`。两者均未被用作判据、未进状态灯。**Pass**。
- **⑤ 锁台账与 §20.1。** W1 整程 2.373 min <10 min ⇒ §20.1 第 1 条例外，合规；W2 三次全量 150/360/390 s 各 <10 min，孤儿锁按 §20 自愈规则回收（`analysis.md §8`），冒烟按短任务例外登记；W3 全程**未取锁**、纯推理 0.92 min/431.6 MB/<3 GB 空闲条件满足、登记为 §20.1 例外（`analysis.md §11`）；W4 四阶段各自取释、PD/EP 间留 ≥60 s（`analysis.md §12`、`progress.md`）。**Pass**。
- **§22 共享文件纪律。** 唯一写者 = W4；`pathlm/depth_ar.py` 中 `readout_perdepth`/`readout_wide_shared` 与邻线标记 `carry_state_dim`、`attn_sparse_k`、`recon_window` 均在（grep 实测），`train_depth_ar.py` 只做接线（`--readout-perdepth`/`--readout-wide-shared` + 每 200 步活性），无人回滚他人改动。**Pass**。

### - Finding

- **P1 · W1 缺失「等字节 ≠ 等形状」的口径混杂声明（本轮重点 ③ 的 fail 项）。** 位置：`2621-kv-budget-rest/analysis.md §4`（并进行 768 B 跨形状排名：`mqa_int8 / gqa_win64 / gqa_int4 / win32_anchor32_int4 / win32_int8 / win16`）与 `§6 混杂与口径声明`（仅列参数量混杂与解析/实测口径）。证据：其前身 `2620-kv-budget-gaps/analysis.md:47-49` 明文写「**这条口径混杂必须写出来**：等字节 ≠ 等形状」，并在 `:151`、`:225` 复述；chart 的 bpc 主要由**连续窗口长度**决定（`win16` 单臂 2.085526），锚点档把一半位置预算挪去锚区必然抬高 bpc。W1 继承了同一张图与同一类比较，却**没有**继承这条强制声明，`config.json:18` 反而把「锚点档可在同一横轴做等字节对照」写成优点。后果：`analysis.md §4`「同预算下 mqa_int8 全面领先」的读法缺护栏，按 §14 第 6 项（口径标注）与契约继承口径该条不合格。最小修法：在 `analysis.md §6` 补回一段（引 2620 原文即可），并注明「等字节比较只在两条腿都看、且承认形状不同时才成立」。**主结论不受影响**——「int4 叠头共享被放大 2.1×/4.3×」是与**同架构 fp16 臂**逐位同形状的对照。
- **P2 · W3 `README.md:157` 的评测墙钟与 `results.json` 不符。** 写「评测墙钟 **0.97 min**」，而 `results.json::cost.wall_minutes_total_eval = 0.92`，且同线 `analysis.md:191` 也写 0.92。最小修法：改为 0.92。
- **P2 · W1 MQA 参数量降幅 10.25% 与实测不符（3 处）。** `analysis.md:106`、`results.json:15370`、`table.md:195` 均写 `8.75%/10.25%`；实测 13,536,768 → 12,155,136 = **−10.2066%**（W3 `2616/analysis.md:19` 自己写 −10.21%，`README.md:41` 写 10.2%）。最小修法：改成 10.21%。
- **P2 · W1 `README.md:46` 两族基线差 0.0016。** 实测 `1.483569 − 1.482026 = 0.001543`（应 0.0015）。这是 R1 已在 `2620/README.md:44` 提过的同一处措辞，W1 复制时再次带入。
- **P2 · W2 `spec_decode.py:386-387` docstring 与实际实现相反，影响重点 ① 的判读。** `plain_greedy` 的 docstring 写「这里用 `CommittedCache` 重写以便与 spec 路径共用同一套 KV 语义」，但函数体（`:388-403`）`from pathlm.depth_ar import DepthARDecoder` 并用 `d.step`（即仓库路径）；真正共用缓存机械的是 `plain_greedy_cache`（`:358-379`）。若读者采信该 docstring，会把 2.02× 的实现层差误记为「spec 机制」。最小修法：docstring 改为「走仓库 `DepthARDecoder` 路径；与 spec 同机械的对照见 `plain_greedy_cache`」。
- **P2 · K10 `PRIOR_ART_LENGTH.md:181` 的 T=512 注意力占比算错。** 表内 `25.7%`，而该文件自己给的公式 `512T/(512T+786432)`（`§4.1`，与 `2616/results.json` 的 `other=786432`、`attn=131072@T256` 一致）在 T=512 得 **25.0%**；该文件 §6 的核对片段只打印了 256/1024/1536/2048/4096，恰好跳过 T=512。**影响**：ticket 39 的 X3 正是 T=512 首探并要「与 K10 的公式对照」。最小修法：改成 25.0%（并把 T=512 加进 §6 核对片段）。
- **P2 · 对账/变异报告的落盘缺口（影响 §12 的机检性）。** W2 目录无 `reconcile_report.json`，`analysis.md §8` 的「1027 条断言全过」只有文字（脚本本身在，`reconcile_numbers.py` 可重跑）；W4 同样无报告文件，`progress.md` 的「42 条全绿」只有文字（关键算术另存于 `results.json::reconciliation`）；W1 无 `mutation_report.json`（变异结论只在 `README.md §6b` 文字 + sha256）。对比：W1/W3 有 `reconcile_report.json`、W3 另有 `mutation_report.json`。最小修法：把脚本输出重定向落盘并由报告引用。
- **P2 · W2 `gpu_run.sh` 回收分支 `continue` 前无 sleep；§20.1 例外登记缺前后空闲读数。** 脚本 `:28-33` 回收后直接 `continue`，与 §20 明文「回收分支必须先清空再删，且**循环路径上始终有 sleep**」不一致（`rmdir` 失败时会热自旋）。同线 `analysis.md §8` 的并发例外只给了自身峰值 35.7 MB，未给 §20.1 第 2 条要求的「前后空闲读数」（W3 §11 给了 15.0 GB/15.0 GB）。观测未受影响（回收成功、无自旋）。
- **P2 · §21 的 catalog 侧标注未落地（属主 Agent 待办，不是 K9/K10 的缺陷）。** K9 的 `(a)` 条目（深集成等）只写在 `PRIOR_ART_DECORRELATION.md`，`catalog.yaml` 中无对应 `status: reference` + `local`「业界结论，非本地实测」；`PRIOR_ART_DECORRELATION.md` / `PRIOR_ART_LENGTH.md` 也未被 catalog 引用（grep 无命中）。K10 的位置编码 `(a)` 与 A9 现有 ticket-36 note 基本重合，但未挂到 K10 产物上。知识线被禁止改 `catalog.yaml`，故属主 Agent 收口动作。
- **越界提示（不在本轮对象内，供 X3 线注意）**：ticket 39 X3 写「T=512 … K10 已算 +2.8% 参数」；K10 的 `+2.8%` 是 **T=1024**（+196,608），T=512 实为 `(512−256)·256 = 65,536 = +0.94%`。

### 可疑数字清单（汇总）

| 严重度 | 位置 | 问题 |
|---|---|---|
| P1 | W1 `analysis.md §4/§6` | 「等字节 ≠ 等形状」口径混杂声明缺失（前身 2620 标为"必须写出来"） |
| P2 | W3 `README.md:157` | 评测墙钟 0.97 min ≠ `results.json` 0.92 min |
| P2 | W1 `analysis.md:106` / `results.json:15370` / `table.md:195` | MQA 参数 −10.25% ≠ 实测 −10.21% |
| P2 | W1 `README.md:46` | 两族基线差 0.0016 ≠ 0.001543 |
| P2 | K10 `PRIOR_ART_LENGTH.md:181` | T=512 attn share 25.7% ≠ 25.0%（X3 直接用该点） |
| P2 | W2 `analysis.md §8` / W4 `progress.md` | 对账条数（1027 / 42）无落盘报告，不可机检 |
| 提示 | ticket 39 X3 | 把 K10 的 +2.8%（T=1024）误挂到 T=512（应 +0.94%） |

### Merge verdict

- **W1：OK with notes**（无 P0；1 处 P1 口径缺失 + 2 处 P2 数字，均为文字级；**其"与 results.json 逐一对齐"一项为 fail**，按「fail 的结论不计入状态灯」——同形状的量化增量结论本身可入账，§4 跨形状排名句在补声明前不宜入灯）。
- **W2：OK with notes**（重点 ① 通过；P2 = docstring 反述 + 报告未落盘 + 回收分支无 sleep）。
- **W3：OK with notes**（§4b 偏离有裁决、判据未改；P2 = README 0.97 min）。
- **W4：OK with notes**（重点 ② 通过；P2 = 报告未落盘）。
- **K9：OK**（§21 分诊与标注合规；catalog 侧收口属主 Agent）。
- **K10：OK with notes**（P2 = T=512 表值算错）。
- **总计：无 P0；1 处 P1（W1，文字级）、8 处 P2。未跑**：`python3 -m pytest tests/ -q`（基线 361）、`tools/check_schema.py`、`tools/build_wiki.py --check`、sha256 重算 —— 无 shell，均留给主 Agent。