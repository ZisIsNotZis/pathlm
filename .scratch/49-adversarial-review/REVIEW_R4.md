## Review

**审查对象**：本波三条 GPU 线（M1 = `axes/M-carrier/M1-carrier-type/experiments/2631-carry-reseed`；M2 = `axes/H-horizon/H3-composition/experiments/2631-spec-three-arms`；M3 = `axes/R-references/R1-quantization/experiments/2631-kv-budget-causal`）**已落盘**产物。审查基准 = reviewer-launch `HEAD d256c0a8`；`watchdog_diff` 报**无工作树差异** ⇒ 三条线的产物全部已提交（我无法看提交区间/提交信息，只审文件内容）。只读，未改任何文件。

### 一、四项指定问题的核查结论

**① M3 那处矛盾（值字节 vs 部署字节）——已真正解掉，且字节账自洽。**
- 实现已是因果/流式：`tools/evalkit/kv_quant.py:127-128`（`cummax` 沿 token 轴；batch 不参与归约），价格明写在 `kv_quant.py:33`（每值 +1 个 fp16 标尺 ⇒ int8 3 B/值、int4 2.5 B/值）、`kv_quant.py:263-282`（`scale_bytes_streaming_per_position` / `*_streaming_bytes_by_seq_len`）。
- 结论页与表均已改口并“推翻旧账”：`analysis.md:29,136,164,209,211`、`table.md:255`（“量化因子恒 >1 ⇒ 量化在任何拓扑上都是内存负收益”）。
- **我独立手算逐项对账，无一处不自洽**：值字节 6144/3072（8 头×32 维×12 层×2）；(a)=值+2 B/值 ⇒ 18432/15360 = ×1.50/×1.25；gqa_int8 4608、mqa_int4 1920；(b) G=32 ⇒ 6528/3456（+46.9%/+71.9%），`win16_anchor16_int4` 段长 16 ⇒ 分母 min(32,16)=16 ⇒ 480；块阶梯 2/G 与 `results.json::deployment_caliber.b_group_ladder` 逐项相等；重算开销单列——15.5 位置 × 6144 值 × 2 B = **190 464 B/请求 = 744 B/位置 @T=256**，与 `b_staging` 逐位一致，且**未**被并入“每位置字节”。`mutation_report.json` 的 M5 正是钉“逐值标尺当免费 ⇒ 矛盾又回来”。

**② M2 真的实现了概率组合；六项指标齐；native 走同一条 `Decoder.step()`。**
- `combined` 是**分布混合**：`spec_arms.py:59-62` = `λ·softmax(chain2) + (1−λ)·softmax(node2)`，λ=0.5 预登记（`:37`）；单测 `test_spec_arms.py::test_combined_is_a_probability_mixture_not_a_logit_sum` 断言“≠ logits 线性组合”且归一；变异 M7 打的就是它。
- `chain` 是**真条件**：`pathlm/model.py:336-354`（`cond = node1.logits.argmax` ⇒ `chain_cat([h, embed(d_next)])`），单测 `test_arms_read_different_distributions_in_the_real_engine` + `test_chain_arm_is_not_the_latent_T_composition`（`chain2 != T1∘T1`）+ 变异 M3（错接头被抓红）⇒ 复核 A6 的“反例警示”在实现层落地。
- **六项指标齐**（`results.json::checkpoints.<ckpt>`）：α₁ / α₂|₁（+聚类 CI）、τ（+CI）、tok/fwd（结构恒等式 1+τ 与实发量都报）、墙钟加速比（+CI、逐采样）、等效算力净账 `equivalent_compute_ratio` + 解析头成本占比、保真（argmax ⇒ 明示**非分布保持**，另给更强的逐位等价门禁）。九宫格 9/9 通过判定线 A；`direct−combined` CI 三次全含 0、`chain−direct` 三次全不含 0 ⇒ 结论“direct ≈ combined > chain”措辞与数据一致。
- **native 对照**：`spec_arms.py:113-116` 把 native 路由到 `pathlm/decode.py::decode` → **`Decoder.step()`**（逐 token）；投机侧验证走 `Decoder.step_multi()`（不同函数），其等价性有双重证据：仓库既有单测 `tests/test_m1.py:869 test_step_multi_matches_per_position_logits`（逐位置 argmax 与 K 次 `step()` 相等）+ 端到端逐位门禁（`bit_identical_all_arms=true`，12 prompt × 3 checkpoint）。⇒ “对照物是模型自身、且不是 teacher-forcing 前向”成立（`run.py:288/314` 的字段本身是自述，见 P2-5）。

**③ M1 评测确实是分布内主腿 + 跨种子；delta 结论被如实判为成立/不成立/不可排序。**
- 分布内**可复算**：`tools/evalkit/longrange_tasks.py:502-571`，T=256/K=8/Q=8/gap=14 ⇒ `block=17, base=120` ⇒ 距离 ∈ [107,240]；与**训练侧**记录同一布局（`train/delta_s0/results.json::synth_task_config` query_base 120 / query_block 17）；`results.json::main_leg.*.query_key_distance` = min 107 / max 240 / frac_outside_window 1.0，局部先验 = chance 0.015625。分布外腿（241–255）明确只作辅助。
- 跨种子 = seed 0/1 各一次独立训练（六臂 12k 步 × bs32 × T256 = 98.3M token，逐字段一致）⇒ 满足预登记“≥2 seed”。
- **判定如实**：② `delta−none` **成立**（+34.927/+34.323pp，配对聚类 CI 下界 34.47/33.89）；③ `delta−plain` **不成立**（seed0 +0.369pp，CI[−0.150,+0.900] 跨 0）；0.37pp **明写“落在 Δ<0.014 不可排序带内”**（`analysis.md:230,248-249`）；`results.json::criteria.c3=false` 且 `best_arm_per_seed={0:delta,1:delta}` 也如实报“delta 最好但不稳定的原因是 plain 塌”。

**④ 数字 ↔ `results.json` 逐一对应（远超 ≥10，抽查 40+ 条全部命中）**
- M3：bpc 1.483571/1.483688/1.508319/1.695384/1.564077/1.503634/1.482026/1.496325/1.493763；Δbpc −0.000033/−0.020913/−0.049054/−0.101390/−0.030634/−0.021909；bytes/pos 12288/6144/3072/768/1536；部署 (a) 18432/15360/4608/3840/2304/1920、(b) 6528/3456/1632/864/816/480；活性 `n_values=6291456`、`rel_err=0.003703`、`changed_frac=0.949632`、`clipped_frac=0.00960811`、`scale_min=9.28e-07`、`scale_max=0.1092`；仪器 1.483569 逐位；chance 0.015625 / 局部先验 0.014648；成本 2.288 min / 462.5 MiB；窗口挡掉的因果对 28920/25200/18528/8256（=32896−kept，我手算全等）。
- M2：13m_s0 α₁ 0.6012/0.8022/0.8048、τ 1.0749/1.4536/1.4225、speedup 1.2971/1.4678/1.4302 及 CI、配对差 −0.1132/−0.0388、c₃ 1.2509/1.0503/1.2373、净账 0.6029/0.5098/0.5164、头占比 0.0471/0.03657、bpc 1.5468/1.5469/1.4211、训练 4.9/4.9/10.0 min、峰值 2972.4/6236.5 MB、53m_s0 α₁ 0.6569/0.8439/0.8293、speedup 1.3854/1.5757/1.5719。
- M1：主腿 1.5503/36.1084/36.4777/1.6479/1.9073/35.9711%（CI 同）、毁源 1.5411/1.6602%、效应 +34.927/+0.369/+34.323/+34.064pp、跨种子极差 0.098/34.201/0.507pp、selfchecks（188 张量、+264296 参数、identity 起步逐位）、成本 1.57 min / 528.7 MB / 960 批、对账 181/181、变异 8/8。

**⑤ 声称已做但未落地**：逐条 grep 后**只找到 1 处**（P2-1），其余声称的文件（`cycle_log.jsonl`、`reconcile_report.json`、`mutation_report.json`、`evidence/fix_mtp_nodes_T1/*`、`smoke/steptime.md`、`run_log.txt`、`charts_only_rerun.runlog`、`cluster_hits.json`、`divergence_2618.json`、`controls/README.md`、`tests/test_kv_budget_causal.py`、本目录单测）全部存在且内容与描述一致；两条线的“待复核”声明也没被冒充成“已复核”。

### Findings

- **P1 · M3 线结论与同仓现行“真相来源”仍直接冲突（发布前必须同步）** — `axes/R-references/R1-quantization/README.md:9,10,35,41,42,47,48`（catalog 渲染源 = `catalog.yaml`）仍把 **int8 −50% / int4 −75% / GQA×int8 省 87.5%** 当**现行参照线价签**，而本线新证据结论是 **int8 +50% / int4 +25% / GQA×int8 省 62.5% 且量化在组合里是减分项**（`2631-kv-budget-causal/analysis.md:29,136,209,211`）。M3 自己已把该同步列为待办（`analysis.md:230-231` [需主 Agent]、`docs/REMEDIATION.md` L1-1/L5-1），但**在本 HEAD 仍未落地** ⇒ 现在发布 wiki 会出现同格自相矛盾。最小修复：按 `analysis.md` §4.4 改 catalog 的 R1 `tax/benefit` 与轴页第 47–48 行，再 `python3 tools/build_wiki.py`。

- **P2-1 · 声称的 600 次 micro 试验不存在** — `2631-spec-three-arms/analysis.md:165`（“集合证据见 `evidence/micro_c3.json`（3×120 次 + **1×600 次**）”）vs `evidence/micro_c3.json` 实际只有每 checkpoint 3 组试验（字段仅 `c3/w1_min/w3_min`），全目录 grep `600` 无该证据。结论不受影响（3×120 已在），但属“声称已做未落地”。最小修复：删掉“+1×600 次”或补跑并落盘。

- **P2-2 · M3 报告里一处派生比值写错** — `analysis.md:268`（簇间极差/Wilson 半宽“逐行 2.46 / 2.46 / 2.99 / 2.95 / 6.18 / 5.65 / 15.69”）：首行 fp16 应为 **2.54**（0.03125 ÷ 0.012317，用 `results.json:1280-1283` 的 wilson95 与 `:1301,1309,1317` 三簇值），现列出的 2.46 是 int8 那一行的值（重复）。文本“2.5–15.7 倍”仍成立，仅逐行列举口径错一位。

- **P2-3 · M1 “整程持锁 <10 min”与自述数字/日志矛盾** — `2631-carry-reseed/analysis.md:213` 同段前半写“训练墙钟 8.57–15.79 min/臂”，`train_arms.runlog` 实际按臂持锁最长 **15.4 min**（delta_s0 12:08:59→12:24:24）。**实践是合规的**（按臂 release + 臂间 65s，日志逐臂可见；<10 min 只在评测段成立，见 `eval.sh` 注释与 `MODEL.md` §3），但这句话会被读成训练整程持锁。最小修复：把该句限定为“评测段整程 <10 min”。

- **P2-4 · 同名的两个 `read_share` 数值差 6 倍，报告与日志各引一个** — `analysis.md:183` 引 `carry_liveness.read_share`（plain_s0 = 0.082，对应 `results.json:1336` 与训练日志末行 11999），而 `run.py:352` 打印的是 `read_mass.read_share_per_layer`（`run_log.txt:30` = 0.532613，对应 `results.json:1368`）。两值都能溯源，但同名混用会让复核者对不上账。最小修复：报告列头注明取哪一个（建议“训练末行 read_share”）。

- **P2-5 · `native_is_plain_decoder_path` 是自述常量而非仪器** — `2631-spec-three-arms/run.py:95` 把 `"decoder_path"` 硬编码为字符串，`:288/314` 只检查该字符串前缀 ⇒ `results.json` 里 `native_is_plain_decoder_path=true` 是**自证**。声明本身为真（`spec_arms.py:113-116` 确实调 `decode→Decoder.step`），但作为“证据字段”不合格。最小修复：改成从 `spec_arms.arm_decode` 的实际分支返回（或在 `analysis.md` 标注该字段为自述）。

- **P2-6 · 两条线的可失败单测不在仓库门禁里** — M1 `test_reseed_stats.py`（14 例，README:7）、M2 `test_spec_arms.py`（README:150 明写“放在本目录，不写 `tests/`”）⇒ `python3 -m pytest tests/ -q`（基线 641）**不收集**它们；M3 的 `test_causal_check.py`(9) 与本目录 `tests/test_kv_budget_causal.py`、`tests/test_kv_quant.py`(前缀不变性) 在内。变异报告显示这些单测确实可失败且被抓红（M1 8/8、M2 7/7、M3 8/8），但**需单独执行**才算本轮验收。

- **P2-7 · M3 无机器可读的逐条判据判决 + 预登记文本与新结论不同调** — `results.json` 只有 `config.json::causal_caliber.preregistered_criteria` 的**文本**，没有 ①–④ 的布尔/结论字段（对比 M1 有 `results.json::criteria`）⇒ 尤其判据③“Pareto 排序是否改变”只能靠 `analysis.md` §4.1/§4.4 散文；同时 `config.json:75,83` 的臂角色串仍写“int8（省 50%）/int4（省 75%）”。建议补 `criteria` 块并给角色串加“（值字节口径；部署口径见 §4.1）”限定。

- **P2-8 · 两处口径过度断言/证据链无法核实（低优先）** — (a) `analysis.md:232` 把“标尺最小可行 dtype = fp8”当“(a) 的**物理下界**”：这是设计假设（fp8 是否够精确未测），宜降为“工程下界”。(b) `2631-kv-budget-causal/README.md:100` 的“本波同批的其它 GPU 重跑线未执行”与同波 M1/M2 已执行（runlog 有完整记录）读起来矛盾，且该 `README` 是**补录**、预登记证据靠“config.json mtime 早于 results.json”（git 不保 mtime，无法复核）——建议改为给时间戳/提交锚点，或删掉该时序句。

### Merge verdict: **OK with notes**（无 P0；1 条 P1 属主 Agent 已记录但未落地的同步项，发布前必须做；其余为 P2 报告级）

---

### 我未能做的事（需 supervisor 执行）
只读、无 shell，故以下**未运行**，请代跑并回报：`python3 -m pytest tests/ -q`（应 641 绿）、`python3 tools/check_schema.py`、`python3 -m pytest axes/M-carrier/M1-carrier-type/experiments/2631-carry-reseed/test_reseed_stats.py axes/H-horizon/H3-composition/experiments/2631-spec-three-arms/test_spec_arms.py axes/R-references/R1-quantization/experiments/2631-kv-budget-causal/test_causal_check.py -q`（覆盖 P2-6 的盲区）。

**残余风险**：① 三个目录的 `model.pt`/`trunk.pt` 均**不在树内**（`.gitignore` 覆盖 `*.pt`，全仓 `find **/*.pt` 为空）⇒ 本波数字**无法在干净检出上端到端重导**，只能重训（这是全仓既定策略，非本波引入）；② 墙钟类结论受共享卡外部 `llama-server` 占用抬高（三条线均已如实披露，`analysis.md` §5 / `run_log` 有原始采样）；③ M2 的墙钟 CI 只由 **6 个 prompt 聚类**（`results.json::wall.n_prompts=6`）构成，小簇数会让区间偏乐观，M2 未像 M3 那样声明——但“direct ≈ combined > chain”另有接受率证据独立支撑（α₁ 差 0.16–0.24，CI 不重叠）。