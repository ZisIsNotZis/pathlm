# Artifact 索引（25 个 ticket 的证据目录）

> 每个 ticket 的执行契约在 `.scratch/NN-slug/issues/`，证据在
> `.scratch/NN-slug/evidence/`。本表是 run 级索引：run 名 → 内容 → 关键数字 →
> 对应总结（SUMMARY 章节）。权重 `*.pt` 为派生物不入库（gitignored）；入库的
> 是 results.json / train_log.jsonl / 探针 json / SUMMARY。结论账本见
> `docs/findings.md`，叙事见 `docs/retrospective.md`。

| Ticket | 证据目录（`.scratch/` 下） | 关键 run / 文件 | 内容与关键数字 | 总结 |
|---|---|---|---|---|
| 01-design | `01-design/issues/` | `01-design-spec.md` | 冻结架构契约（stage ring 0–4、cycles、聚合原则）；无 run | — |
| 02-engine-m0 | `02-engine-m0/evidence/` | `REPORT.md`、`m0_main{,_s1}/`、`m0_nocons/` | M0 四判据：校准 ECE 0.004–0.011 PASS；集成 PASS 边际；consistency cosine 0.9525→0.9882；direct retry FAIL（no-op） | `REPORT.md` |
| 03-engine-m1 | `03-engine-m1/issues/` | `01-engine-m1-extensions.md` | M1 引擎扩展契约（transports/dense-exit/驱逐/token retry/KV 解码）；27 测试绿 | — |
| 04-m1-runs | `04-m1-runs/evidence/` | `B0/ C1/ C3/ I1/ R1/ L2/ X2/`、`SUMMARY.md` | B0 **1.5074**；I1 修复 52.8%；C1 retry +2.1pp；R1 FAIL −4.6pp；L2 深度 6 平台 1.4×；X2 needle FAIL | `SUMMARY.md` 全表 |
| 05-m1-followup | `05-m1-followup/evidence/` | `C4/ C5/ X2v2/`、`retry_curves.json` | C5 −1.3pp（R1 −4.6pp）；retry 曲线 ~r3 饱和；X2v2 窗口内 84.9%/锚 76.8% | findings/REPORT |
| 06-m2-sweep | `06-m2-sweep/evidence/` | `L1/ L3/ L4/ C2/ I2/ I3/ I4/`、`SUMMARY.md` | 元素税律：shuffle **+0.545**、skip +0.175、wrong +0.127、mask +0.059、noise ±0、redo +0.044；L3 边界 eval 1.0→**3.844** | `SUMMARY.md` |
| 07-x1-x3 | `07-x1-x3/evidence/` | `X1/ X3/`、`attn_dist_telemetry.json` | penalty 免费 **−0.007**；X3 锚 93.3%、beyond 90.3%（local-LM 强度警告）；逐头距离 26–43 | `SUMMARY.md` |
| 08-tts-probe | `08-tts-probe/evidence/` | `tts_results.json`、`SUMMARY.md` | L1 K8 增益 −0.047；确定性路径恰得 0；**nothing beats B0**（最佳组合 1.5562 > 1.5074） | `SUMMARY.md` |
| 09-m3-interactions | `09-m3-interactions/evidence/` | `IX1/ IX2/ IX3/ IX4/ INT/ DIVL1/`、`SUMMARY.md` | IX1 次可加 +0.087<0.277；IX2/IX3/INT 全 2.2+（shuffle 毒药律）；DIVL1 −0.062>−0.047 | `SUMMARY.md` |
| 10-gap-sweep | `10-gap-sweep/evidence/` | `B2/ C1M/ C2M/`、`eval_ablations.json` | B2 1.5481；C1M 1.7010、C2M 1.7089（**D2 混杂：漏 n_mtp，税虚高**，ticket 11 修正）；eval-only：redo eval +0.009、skip eval 回收 +0.058 | findings 矩阵节 |
| 11-retry-matrix-validity | `11-retry-matrix-validity/`（`VERDICTS.md` + `evidence/`） | `evidence/README.md`、`C1M_n1{,_s1}/ C2M_n1{,_s1}/ C3*/ C1long/ B0long/` | **矩阵排序撤回**：6 cell 跨度 0.011 < 种子散布 0.0138；D1–D6 缺陷审计；D5 税上界 +0.364@1k→+0.122@24k | `VERDICTS.md`、`evidence/README.md` |
| 12-overnight | `12-overnight/evidence/` | `INT2_s0/ X2C/`、`CH_latent_*/ CH_token_*/`、`*_e2e.jsonl` | 条件链 oracle 0.682≈node1 / deploy 0.547<direct 0.560；D1 因果 1.936 vs 95.2；INT2 **+0.309**、损坏 −1.79 bpc vs B0；X2C needle 0.5%；29M 探针税 +0.174 | report §1/§3.5/§4 |
| 13-four-experiments | `13-four-experiments/evidence/` | `gating/ anchor_exempt/ mtp3/ needle_antagonism/` | 门控证伪（固定点/需稀疏）；锚豁免三路 0.764→0.016/0.033/0.035；**mtp3 spec 1.68–1.90× 税 +0.0166**（`spec_speed*.json` 双复现）；needle 拮抗 bpc 1.680→1.997 | report_details §3.6/§D |
| 14-allocator | `14-allocator/evidence/` | `allocator/*_sel.jsonl`、`composition/frontier_INT2.json` | **选择性成立**：门控 INT2 +0.021 vs 均匀 −0.019、C1M +0.128 vs +0.057；严格占优点 spec+门控 50.8 tok/s @2.0306；驱逐掩码向量化 32→56 tok/s | report §3.8 |
| 15-slider-rung3 | `15-slider-rung3/evidence/` | `INT2_r3/`、`slider_INT2r3.json`、`SUMMARY.md` | 四件套：split gap ≤0.004、成本 ≤±13%、严格占优 0.635 fwd/tok @2.0502、TF ECE 0.0011–0.0079；自由生成段代理负结果 | `SUMMARY.md` |
| 16-rung4-scale | `16-rung4-scale/evidence/` | `B0_100M_s{0,1}/ C1_100M_s{0,1}/`、`*_e2e.jsonl`、`SUMMARY.md` | **税 +0.126**（配对 +0.118/+0.134）；E2E 差距 ≥1.54；B0 损坏 r2 4.18；ECE ≤0.0091 | `SUMMARY.md` |
| 17-slider-at-scale | `17-slider-at-scale/evidence/` | `INT_100M_s0/`、`slider_INT_100M.json`、`SUMMARY.md` | 堆叠溢价 +0.256（13M +0.309 收窄）；成本 ≤11.1%、spec 占优复现 0.566 fwd/tok；**门控杠杆消失**（τ 平坦 ±0.002） | `SUMMARY.md` |
| 18-exit-probe | `18-exit-probe/evidence/` | `EX1/ EX2/`、`exit_probe*.json`、`SUMMARY.md` | probe 保 retry（−0.0098）但 exit 贵（d6 +0.93）；dense 0.2 杀 retry（+0.0146）exit 好（+0.199）——**两全证伪** | `SUMMARY.md` |
| 19-asym-100m | `19-asym-100m/evidence/` | `B0_100M_asym_s{0,1}/ C1_100M_asym_s{0,1}/`、`*_e2e.jsonl`、`SUMMARY.md` | **渐近税 +0.082**（−35%）；等训练口径 13M +0.122→100M +0.082；能力溢价 1.54→1.89；data-limited（1.341≈1.346） | `SUMMARY.md` |
| 20-spec-scale | `20-spec-scale/evidence/` | `INT3_100M_s0/`、`spec_speed.json`、`SUMMARY.md` | 接受级联稳定（a₂ 0.80、a₃\|a₂ 0.767）；**墙钟收益消失**（k=2 1.15–1.29×、k=1 负）；fp 平局翻转 1/400 | `SUMMARY.md` |
| 21-exit-at-scale | `21-exit-at-scale/evidence/` | `EX1_100M_s0/ EX2_100M_s0/`、`exit_probe_*.json`、`SUMMARY.md` | probe @100M 零税（1.5928 vs 1.5871）门控 −0.029；dense 0.2 corrupt 崩溃（+1.73）——剂量响应随规模恶化 | `SUMMARY.md` |
| 22-gen-quality | `22-gen-quality/evidence/` | `gen_quality_INT2r3.json`、`collapse_monitor_INT2r3.json`、`collapse_labels_22b.json`、`SUMMARY.md` + `SUMMARY_22b.md` | js 三分位 0.12→0.48、AUC 0.66–0.69>prob0；real<corrupt<random（0.0114/0.0250/0.0390）；13M prob0 反向未复现（n=1 假象）；**distinct-2 AUC 0.9815** | `SUMMARY.md`/`SUMMARY_22b.md` |
| 23-depth-ar | `23-depth-ar/evidence/` | `sanity-{A,B}/ dense-B/ sanity-B-nofill/`、`23b-*/ 23c-*/ 23d-*/ 23e-*/`、`*_decomposition.json`、`SUMMARY.md` | 四性质成立（精化单调、null 62 维/26 功能读取、tf@4 0.9667、门健康）；**税 +0.6786 证伪**；23b 加权全败（2.45→2.73）；23c 欠拟合主因（12k 税 +0.121）；23d 真税 @12k +0.3295；23e **fill 1.8582 胜 nofill 1.9438、proj-fill 1.8356 新最优** | `SUMMARY.md` §0–§12 |
| 24-depth-ar-100m | `24-depth-ar-100m/evidence/` | `100m-s0/`、`24b-eq-s0/ 24b-dense-s0/`、`24b-decode-wall-*.json`、`smoke-20/`、`SUMMARY.md` | **税 +0.4563**（1.9175−1.4612）>13M matched +0.31；tf@4 0.9726/tf@6 0.9946、exit-4 3.1×；24b 拆混杂：**形态 +0.44（96%）+ 引擎 +0.02（4%）+ 配方 ≈0**，dense tf@4 0.12 vs 形态 0.97 | `SUMMARY.md` §0–§12 |
| 25-tax-attribution | `25-tax-attribution/evidence/` | `25b-readout360-s0/ 25c-anneal-48k-s0/ 25dense-48k-s0/`、`25{b,c}-gates-wall.json`、`SUMMARY.md` | B 切分 2.1475 恢复 **−50.4%**（能量迁出读出空间）；C 名义 +16.4%；48k 锚税 **+0.4372 ≈ 24k +0.4362 零收缩**——**税结构性存在收口**；C 配方吞吐 122.8 vs 112.9 tok/s | `SUMMARY.md` 判读表 |
| 26-closing | `26-closing/issues/` | `01-closing.md` | 收口契约（本索引与 retrospective 的来源）；无 run | — |

## 完整性核对说明

- 有独立 `SUMMARY.md` 的 ticket：04、06、07、08、09、15–25（22 另含 22b）。
- 无 SUMMARY 的 ticket 以以下载体承载关键数字：02（`REPORT.md`）、
  10（results.json + findings 矩阵节）、11（`VERDICTS.md` + `evidence/README.md`）、
  12/13/14（`docs/report.md` §1/§3.5–§3.8 + `docs/report_details.md` §3.5–§3.7/§D）。
  上表数字已对照 run 目录 results.json（B2 1.5481、C1M 1.7010、C2M 1.7089、
  INT2_s0 1.6548、X2C 1.6797、B2fresh 1.5438、B3 1.5604、needle_antagonism 1.9974）
  与各 SUMMARY 原文核对。
- `.gitignore` 覆盖 `*.pt`（76 个权重文件 ~20GB 不入库；`git ls-files '*.pt'` 为空）。
