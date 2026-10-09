# H-W 轴（视界 × 宽度）批判性审计（2026-10-02）

- 审查者：独立 fresh-context 只读代理。审计对象：`axes/H-horizon/**`（5 格）、`axes/W-width/**`（5 格）及其全部实验目录、`catalog.yaml` 对应条目、`SCHEMA.md`、`docs/methodology.md`、`docs/playbook.md`、`docs/OPEN_PROBLEMS.md`、`.scratch/49-adversarial-review/REPORT.md`。
- 纪律：**只读**。未改任何实验代码/数字/`catalog.yaml`；未跑 GPU；未改 `progress.md`。
- 本轮我**独立**核到的（不是复述 49 号报告）：见下「独立复核声明」。

## 一句话总判
**这 10 格目前不能当「已验收的结论库」使用**：H 轴最核心的成本句（投机解码的算力口径）在符号上是错的；W 轴两格的 verdict 建立在**不测试其命题**的实验上；一条跨家族参照线用的是本仓**自己已撤回**的两个数。原始数据多数可继续用，但结论句必须重写；H3 的 `direct≈combined>chain` 是这 10 格里唯一我独立复核后仍站得住的新结论（且它回答的是比题目更窄的问题）。

## 建议删除清单（供主 Agent 树管理）
| 格 | 理由（判据：是否可能改变「该算力路径值不值」的判断） |
|---|---|
| **W5-capacity-vs-compute** | `status: judged` 但**零本地实验**；判据（同 FLOPs 质量 / 同参数量 FLOPs）一项未测；结论是**定义式重言**；唯一证据是 W1 的读出切片实验（**换变量**，且属 W4 域）。保留它不会改变任何判断。建议删除，或降为 W2/W1 页内的一条参照线注脚。 |
| **W2-expertization**（仅「现在花 GPU 做实验」这一动作） | 本格 PRIOR_ART 自判 (a) 且明写「13M 上意义有限，不必跑」。建议**保留为参照线**但不得开 13M 实验（删除「开实验」这个动作，不是删除格子）。 |

其余 8 格**不建议删除**（命题真、或已有可用的原始资产）。

## 建议重做清单（按优先级）
1. **H3（19）· 最高** —— ① 修算力口径：`c₃/(tok/fwd)` 改用解析比值 `(1+γ)/(1+τ)`，并把「墙钟比值 ≠ 算力比值」写进 catalog；② 补 **`chain-oracle` 臂** + **按 `v1 == 真 t_{N+1}` 分层报 α₁**（前者纯推理、后者纯 CPU，成本极低），以定量区分「部署条件错」与「头本身差」；③ 结论句限定为「同 checkpoint 的解码期选头」，并把「链式目标削弱共享变换头（`chain2` 复用 `transforms[0]`）」列为**未测混杂**。成本 ≈ 纯推理 + 少量分析。
2. **H1（17）· 高** —— 修「只剩 FLOPs 口径」；补一次干净的 `c₃` 实测并消除 `.scratch/20-spec-scale/evidence/SUMMARY.md` 内 `3.4–5.3×` 与 `1.15–1.29×` 的矛盾（反解 ⇒ c₃≈1.86–2.09）。
3. **W1（22）· 高** —— 跑**真正的减宽**实验（推断期确定性裁通道 + 训练期 width dropout），必带 §15 等参数对照与两个预算点；把 `status` 从 judged 退回 partial。
4. **W3（24）· 中** —— 按 R1 的因果/流式口径重算或撤回「同 −75% 档 GQA 便宜 3.2×」；补 §15 等参数臂与 2× 训练点。
5. **H2（18）· 中** —— 给 proj-fill 补 §15 等参数对照；把「族内最优」降为「方向为正但落 ±0.03 平局带内、且多 7.6% 参数」。
6. **H4（20）· 低（文档修复，不需重跑）** —— 把 2610 §2/§4、2611 `analysis.md:175`、`H4/PROBLEMS.md:9` 里的 0.3128 / 0.4822 残留就地标注撤回。
7. **W4（25）· 低** —— 补 §16 的 24k 点（作者已列成本）；结论保持「快照、疑为收敛加速」。
8. **H5（21）· 低** —— 改标签（judged→reference+本地增量）；`tax` 句补成本说明。

## 开倒车清单（业界标准是什么、我们错在哪）
> 外部工作的**名称**来自本仓自己的 `PRIOR_ART.md`，这些文件**自标 `记忆·未核`**；本环境无联网检索，我**无法独立核证**这些引用（这一限制必须随结论一起保留）。

| 格 | 业界标准（本仓 PRIOR_ART 自列） | 我们的错 | 本仓自己的处置 |
|---|---|---|---|
| H1 | Medusa（Cai 2024）批量验证 + 多解码头；Gloeckle 2024 MTP「税小 + 自带草稿」；launch-bound/compute-bound 的加速比 regime | 把上述三条当本地发现；更糟的是把「launch-bound 墙钟比」自造成「等价算力」口径 | PRIOR_ART 判 (a)+b，但 catalog 仍 `judged` |
| H4 | Medusa-1 / EAGLE-1「冻结主干只训头 ⇒ 零税」 | 为一条已知结论跑了一个冻结主干实验并记为本地结论 | PRIOR_ART 判 (a)+b，catalog 仍 `judged` |
| H5 | 投机解码**无损性定理**（Leviathan 2023 / Chen 2023）；Medusa 树验证 | 「批量验证 ≡ 逐步贪心」是定理，本格把它当实验结果；本地价值其实只是引擎**集成测试** | PRIOR_ART 明确建议改 `reference`，catalog 仍 `judged` |
| W1 | 结构化剪枝 / Sheared LLaMA / Minitron「减宽 = 减容量付质量」 | 把这条共识当本地结论，而本地根本没做减宽 | PRIOR_ART 判 (a)+b 并写「该子条可标 reference」 |
| W2 | Switch / Mixtral / DeepSeek-MoE「细粒度 + 共享专家」 | 无本地增量；且 `status: reference` 与 `prior_art_verdict: (c)` 自相矛盾 | PRIOR_ART 判 (a)「不必跑实验」 |
| W3 | MQA（Shazeer 2019）/ GQA（Ainslie 2023，uptrain 近无损）/ MLA | 把「GQA/MQA 近无损」当本地读数；跨家族 Pareto 用本仓已作废的 int4 数字 | PRIOR_ART 判 (a)+b |
| W5 | 剪枝 vs MoE 的基本分类（条件计算） | 把定义式分类写成 `judged` 判决，且判据全未测 | PRIOR_ART 判 (a) 并建议 `reference` |

**共性**：本仓的 `reference` 机制（SCHEMA §21）存在但**基本没被用于这 10 格**——10 格里 8 格 `judged`，而它们各自的 PRIOR_ART 分诊表都写着 (a) 或 (a)+b。

## 本轮最严重的三件事
1. **H1/H3 的「等效算力」口径把投机解码的算力账算反了符号。** `c₃` 是宽度-K 前向与宽度-1 前向的**墙钟 min 比**（13M 得 ≈1.25，launch-bound），却被当成算力比用（`2631/analysis.md` §2.1/§3、`catalog.yaml:678-722`、`docs/methodology.md:25` 的「FLOPs（等价 width-1）」行）。按层-位置/线性 FLOPs，一次宽度-(1+γ) 前向 ≈ (1+γ)× 一次逐 token 前向 ⇒ 每 token 算力 = `(1+γ)/(1+τ) = 3/2.4 ≈ 1.25`（13M）、`3/2.4 ≈ 1.25`（100M）——**是亏损，不是「2× 效率」**。这直接决定用户 P-02 主目标的部署经济性，且 H1 的「@100M 只剩 FLOPs 口径」因此不成立。
2. **W1 与 W5 的 verdict 没有测试它们自己的命题。** W1 的命题是「每 token 用更少宽度跑完栈」，而唯一本地实验（`--readout-dims 360`，`.scratch/25-tax-attribution`）量的是**读出可见子空间**（W4 域），`W1-active-width/README.md:31` 自己写着「本地已测：（无）」；W5 的判据（同 FLOPs 质量、同参数量 FLOPs）**一项未测**，其 evidence 也是同一个读出切片实验。⇒ 两格的 `judged` 无本地实测支撑，W5 建议删除。
3. **W3 的跨家族参照线建立在两个本仓已经撤回的数字上（同仓内自相矛盾的活证据）。** W3 verdict（`catalog.yaml:815-837`）称「同 −75% 档 GQA 比 int4 便宜 ≈3.2×」，依据 int4「−75% 内存 / +0.0457 bpc」；而 R1 已按因果/流式标尺改写为 **int4 = 1.25× fp16 内存（负收益）**、**int4 代价 +0.024748**（`catalog.yaml:1208-1211`、`axes/R-references/R1-quantization/README.md:9-10,46`）⇒ 「同 −75% 档」不存在，比值变 ≈1.73×。同一份 `catalog.yaml` 里两格给出相反口径。

## 独立复核声明（我核到了什么，而不是复述 49 号报告）
- **核实（确认 49 号）**：`check_schema.py` 只查形状 + 证据路径存在、`@commit` 被 `split(" @")[0]` 丢弃（我读了源码，确认 A7）；2610 §2/§4 的 0.3128 与 2611 的 0.4822 残留（**这两处的具体位置是我自己扫出来的，49 号只点了 H4 的大方向**）。
- **推翻/修正 49 号**：A2（量化非因果）**已被修复**——`tools/evalkit/kv_quant.py` 现在用 token 轴 `cummax` 的逐请求因果标尺，并新增「每值一个 fp16 scale」的开销记账（int8 3 B/值、int4 2.5 B/值）；因此 W3 依赖的 int4 旧数字**不是「待修」而是「已作废、但 W3 没跟着改」**。
- **49 号未提、我新查出**：① `c₃` 被当算力口径的符号级错误（H1/H3/H5 三格共享）；② `.scratch/20-spec-scale/evidence/SUMMARY.md` 内部 `3.4–5.3×` 与 `1.15–1.29×` 反解冲突；③ proj-fill 的 +526,336 参数（+7.6%）与来源自述「平局带内」（H2）；④ W1/W5 的命题–证据错配。

## 关于第 19 项（H3）的专项回答（摘要；详见 `axes/H-horizon/H3-composition/CRITIQUE.md`）
- **实现核实**：三臂分别读 `node1.chain2` / `node2` / `λ·softmax(chain2)+(1−λ)·softmax(node2)`，`chain2` 真以 `d1 = argmax node1(h_N)` 为条件（`pathlm/model.py:328-360`，单测 `test_spec_arms.py:67/96/207` + 变异 M3/M7 抓红）。**是真实实现，不是标签或 `T∘T` 冒充。**
- **用户疑点（0.844 vs 0.657）**：**前提不成立**——三臂**不是各自独立训练**，它们共用同一 checkpoint（`train_arms.sh` 按 `<scale>_seed` 训练；`run.py::load_ckpt` 每 ckpt 加载一次并跑全部臂；`config_13m.json` 无臂开关），且三臂发出流**逐位相同**。所以既不是「训练漂移」，也不是「第 1 个头被带坏」，而是**条件链头在部署时把训练条件（真 `t_{i+1}`，`model.py:454-461`）换成了模型自己的 argmax（`model.py:339-345`）**；该模型下一 token 准确率只有 0.6865/0.7105 ⇒ 约 30% 位置条件输错。旁证：`2512-overnight-causal-chain` 报 oracle 链 0.682 ≈ node-1 0.693，而 deploy 链 0.547 < direct 0.560。
- **要区分必须补什么**：`chain-oracle` 臂（解码时喂真 `t_{N+1}`）+ 按 `v1` 正误分层报 α₁（无需 GPU）；若要排除「共享头被削弱」，再训一个 `chain_mode="latent"` 的无条件链对照比 `next_token_acc`。

## 复核范围与无法核实项（必须随结论保留）
- `watchdog_diff` 在本会话不可用（返回「Reviewer-launch HEAD changed or is unavailable」）⇒ **我没有 diff 可审**；本审计针对的是**已提交的工作树内容**，不是某个 commit range。若需按 diff 验收，请提供 diff 制品或重开会话。
- 外部文献（Medusa/Gloeckle/Leviathan/Ainslie/Shazeer…）**全部无法独立核证**（无联网）；我只能确认「本仓自己的 PRIOR_ART 文件这样记载」。
- 我**不能复算**任何依赖训练权重的数字（权重不入库、无 GPU）；本报告中的「该涨/该跌」是判据建议，不是实测。
- 本会话**无写工具**，故 11 份 `CRITIQUE.md` 与本文以文本形式随本消息返回，未落盘。
- 未运行 `pytest`（只读约束）；`tools/check_schema.py` 的结论我是**读源码**得出，未执行。
