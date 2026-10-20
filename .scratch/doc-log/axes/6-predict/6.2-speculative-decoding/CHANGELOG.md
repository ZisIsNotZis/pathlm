# 6.2-speculative-decoding · 变更流水

本文件是**过程史**（`AGENTS.md` §2.2）：逐条记录本格被改过什么、谁改的、为什么。它不是事实权威——当前事实只住 `axes/6-predict/6.2-speculative-decoding/project.yaml`，两者冲突时以该文件为准。
内容由 `axes/6-predict/6.2-speculative-decoding/project.yaml` 的 `change_log` 逐条迁出：日期、actor、change 原文照抄，未改写。

## 2026-10-14T10:01:27+08:00

- actor：axis6-lead
- change：WP 6.2.3 执行并收口（Phase B）：实验 `experiments/2650-verify-gate-13m` 入库（预登记 V0–V6）；conclusion 新增 (j)：门禁对抗负对照（随机候选 α₁=0.0185 / 尾部候选 α₁=0）与退化等价（dup_k4、single_k1_ctl == combo_pw 逐位）通过、三条 bookkeeping 变异全抓红，代价台账 K=1/2/4/8 的 τ/R 与边际 Δτ/ΔK（K=4→8 = 0.0193 < 阈值 0.0579）⇒ 判 K>4 验证开销不划算、但硬指标（顺序前向次数）上验证不吞 τ 收益；(i) 与 gaps 补上带驱逐（window>0）树验证未实装 / 53.9M 对抗臂未跑 / 候选族边际未扫；WP 6.2.3 evidence 钉到 `2650-verify-gate-13m/`，status 仍为进行中（窗口驱逐未测）。另：`2650/README.md` 的 V5 原式 `3K·(1+γ)/(1+τ)` 有因子 3 笔误（出题人错），已在契约文件内登记更正（定稿口径 = `K·(1+γ)/(1+τ) = 3K/(1+τ)`，两套数在 `analysis.md` 并列保留）。

## 2026-10-13T17:55:00+08:00

- actor：axis6-lead
- change：按独立复核 #2（REVIEW-2.md，VERDICT: PASS）的 info 级建议收口：WP 6.2.2 的 acceptance 与 roadmap 把「完整树」改为「树 = K 行平铺一次批量前向；节点共享树注意力未实装（已入 gaps）」；conclusion (g) 的 R 区间标注改为「union2 / K=2」；实验侧 `mutate.py` 不再把逐次运行的 pytest 耗时写进 `mutation_report.json`（入库事实文件不得含逐次抖动字段）。

## 2026-10-13T17:50:00+08:00

- actor：axis6-lead
- change：WP 6.2.2 收口（主判已答）：实验 `experiments/2640-tree-conf-13m` 入库；conclusion 新增 (g) 多候选/树抬 τ 成立（α₁ +0.0618…+0.1269、τ +0.109…+0.236、配对 CI 全部不含 0、K=4>K=2>union2）与 (h) 路径概率加权 vs 固定 λ 不可区分（否证）、(i) 剩余未测（温度>0/batch>1/二级条件树/53.9M 只 3 臂）；gaps 首条由「主判据未测」改为剩余项清单；WP 6.2.2 status 未开始→已完成；conclusion_review 写入独立复核 #1 的判词与 4 条 normal 的处置。status 仍为进行中（milestone 输出包的稳健性点未齐）。

## 2026-10-13T10:47:00+08:00

- actor：轴 6 §6.2 worker
- change：按 MAPPING §1/§3 由旧格 H1（本体）+ H3/H4/H5（手段）合并建立（H1→6.2，H3→WP 6.2.2，H4→WP 6.2.1，H5→WP 6.2.3）；四格的 experiments 整目录 git mv 进本项 experiments/（一格一份家，不复制）。按 55-review H-PROBLEMS 重写：topic 从手段（每步输出位置数）改回目的（抬高 τ、压低顺序前向次数）；剔除解析恒真的 R>1 主判据与「恒为算力亏损」「税效比最好」头条；把「概率多候选 / 树 / 置信加权」重立为 WP 6.2.2 并写明必须补回设计；判据改为 τ / tokens-per-forward（该涨）+ 质量（该不跌），FLOPs 只作代价轴。

## 2026-10-14T06:44:03+08:00

- actor：framework-adjudication-writer（ticket 57）
- change：落地跨轴裁决（`FRAMEWORK-DECISIONS.md` C1/C2/63-F2，不改数字、不删证据）：`validation`/`falsification` 补「用哪个带」声明（接受率/τ 走配对 bootstrap CI、0.014 只作参考，C1）与「多候选结论须附等 FLOPs 对照、不可得则注明」（63-F2）；`milestone.acceptance` 与 `conclusion` 补「字面成立 / 实质成立」并列读法（C2）。

## 2026-10-14T06:59:49+08:00

- actor：c1-backfill-writer（ticket 57 · C1 声明回填）
- change：C1 声明回填（ticket 57；不改数字 / 不改结论 / 不改证据）：按 `docs/research-schema.md` §12.1 的声明义务，给本项引用噪声带 / 阈值却未声明「用哪个带」的字段补一句声明——bpc 口径用 0.014 bpc 带（限配对 Δ、≥5 窗口），概率 / 率类指标走配对 bootstrap / Wilson 95% CI、不跨量纲借用；本格有自测在力带者以本格带为准（见各字段声明）。

## 2026-10-14T20:52:40+08:00

- actor：axis6-lead（ticket 63 续 · C1 声明回填第二轮）
- change：C1 声明回填第二轮（不改数字 / 不改结论 / 不改证据）：脚本枚举本轴全部 project.yaml 中引用噪声带 / 阈值却未在同字段声明「用哪个带」的字段（枚举脚本与清单见 .scratch/63-axis6-predict/c1_band_audit.py 与 TICKET.md），逐项补声明，并修掉跨量纲借用：objective_criteria / milestone.acceptance / work_packages[].acceptance 里被 0.014 覆盖的 τ / 接受率类指标改为「走 prompt 聚类配对 bootstrap 95% CI 不含 0 为主判，0.014 bpc 带只用于 bpc 口径、不跨量纲借用」（依据 docs/research-schema.md §12.1）。

## 2026-10-20T10:40:00+08:00

- actor：代价表生成 worker（工作包「代价表生成」）
- change：按 docs/research-schema.md §2.1 / §2.2 增补 `laws`（本项证据支持哪几条 playbook 定律）与 `cost_rows`（可引用的代价事实，数字逐位照抄既有落盘），供 tools/build_cost_table.py 生成 docs/cost-table.md；未改任何测量数字、判据线与既有结论。
