VERDICT: PASS

# ticket 63 · 轴 6 Phase B · 独立执行验证（REVIEW-B4-EXEC）

- 角色：fresh 独立执行验证者（未参与任何写作），**只读**（仅新增本报告；未改任何被验对象、未 `git add`/`git commit`、未修代码）。
- 被验 revision：`HEAD` = `167f199475e5ff66bb577c9e271d6eed68727dfc`（前一次提交 `0c4dc01`）。实测与任务给出的 `167f199` 一致。
- 验证时间：`2026-10-14T10:50:49+08:00` 起，`2026-10-14T10:55:36+08:00` 止。
- 唯一写入：本文件。`git status --porcelain` 全程为空（见 §5）。

排期说明：§1（pytest）、§3（summary）先行；§4（确定性重跑）在 GPU 锁空闲时立即后台启动（耗时 97.3s）；§6 核对与之并行；§2（mutate）在重跑结束后串行执行，避免 `2660/mutate.py` 临时改写 `latent_reg.py`/`run.py` 与重跑读取源码相撞。

## 1. 单元测试（亲自跑）

| 命令 | 实际结果 |
|---|---|
| `python3 -m pytest axes/6-predict/6.2-speculative-decoding/experiments/2650-verify-gate-13m/test_verify_gate.py -q` | **11 passed** in 6.26s ✓（期望 11） |
| `python3 -m pytest axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/test_latent_reg.py -q` | **13 passed** in 6.22s ✓（期望 13） |

变异后复跑（证明「逐字节还原后仍全绿」）：2650 = `11 passed in 3.99s`；2660 = `13 passed in 3.75s`。

## 2. 变异验证（亲自跑，各连跑两次）

**2650（3 条预登记 bookkeeping 变异）**
- `cd .../2650-verify-gate-13m && python3 mutate.py`（RUN 1 与 RUN 2）
- 逐条：`M1: caught_red=True restored=True | 1 failed, 10 deselected`；`M2: caught_red=True restored=True`；`M3: caught_red=True restored=True`（全部 `test_failure` 级）。
- 汇总（两次相同）：`{"all_caught_red": true, "all_restored_bitwise": true}`；`targets.verify_gate.py` sha256 = `6c88d225c691ab0058562585f67c655e1b0e1aaca7a3b6a936fc0bd188bea6f3`。
- **确定性与无耗时字段**：`mutation_report.json` sha256 = `f4bc5c5c6403f43ccfd190825d50fe6d8eeb8e20b0d4a726e662a2d77f97bb8d`（RUN1）== `f4bc5c5c…`（RUN2）== 提交内版本；`grep -c "in [0-9.]*s" mutation_report.json` = **0** ✓。
- 注：该报告字段名为 `all_caught_red` / `all_restored_bitwise`（无 `post_restore_all_green` 字段，但变异后复跑全绿见 §1）。

**2660（9 条变异）**
- `cd .../2660-latentreg-vs-disc-13m && python3 mutate.py`（RUN 1 与 RUN 2）
- 逐条全 `RED … restored=True`（9/9 抓红 + 逐字节还原）。
- 汇总（两次相同）：`{"n_mutations": 9, "all_caught": true, "all_restored": true, "post_restore_all_green": true}` ✓。
- **确定性与无耗时字段**：`mutation_report.json` sha256 = `fce87efc8b77981a66b39a0a5faca56c7d61b9d7da31c0d41cc5ff74e3a497a4`（RUN1）== `fce87efc…`（RUN2）== 提交内版本；`grep -c "in [0-9.]*s" mutation_report.json` = **0** ✓。

两次运行后 `git status --short` 干净（§5），证明 mutate 逐字节还原成立。

## 3. 研究树校验（亲自跑）

`python3 tools/summary.py --check` → `OK：46 个 project.yaml 全部符合 docs/research-schema.md。` ✓（期望一致）

## 4. 确定性重跑（亲自跑，GPU，最强证伪）

- GPU 锁：`/tmp/pathlm_gpu.lock` 起跑前不存在、空闲；由 `2660/run.py --stage train` 内部 `gpu_lock("wp6.3.3-p1")` 自取（`SCHEMA.md` §20，max_wait=600s，不杀他人进程）；跑完 DONE 后锁已释放（`ls /tmp/pathlm_gpu.lock` → No such file）。
- 命令：`python3 axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/run.py --stage train --out /tmp/rev4-2660 --ckpt /home/z/vibe/pathlm/axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0`（已核 `run.py` argparse：`--stage/--out/--ckpt` 存在，默认 `--out` 为实验目录本身，故显式给 `/tmp/rev4-2660`；**未**用 `--out .`）。
- 结果：`DONE wall_seconds=97.3`，`/tmp/rev4-2660/{results.json, heads/*.pt (8)}` 落盘。
- **命中率逐值相同**：`arms.*.*.{hit_t2,hit_t3,ce_t2}` 与现盘 `2660/results.json` 逐值 `==`（Python `float ==`，无一处 MISMATCH）；arm 键集合相等。示例：`disc_k2/s0 hit_t2=0.4572088068181818`、`lat_k2_lam1/s0 hit_t2=0.48778717885375494`、`lat_k2_lam1/s0 ce_t2=1.89039`。
- **权重逐张量相同**：8 个 `heads/*.pt`，每个 8 个张量，`torch.equal` 全 True（`ALL HEADS TENSOR-EQUAL: True`，key 集合相等）。**未**用 `.pt` 文件哈希推断内容（避免 zip entry basename 造成的伪差异）。
- 结论：重跑与现盘**逐值 + 逐张量一致**（确定性成立）。

## 5. 证据完整性与范围（亲自跑）

**(a) 2640 文件 sha256 三方交叉核对**

| 文件 | 实测 sha256 | 交叉来源 | 一致 |
|---|---|---|---|
| `2640/spec_tree.py` | `71d98b2687b2cc36d362514bd8bcc014aa3818f4aad4ef14c83b56e9be8fb62f` | `2650/results.json::imported_2640_sha256.spec_tree.py` | ✓ |
| `2640/run.py` | `fba5a16c61ca9fd5ec5972f89427fed9ac1a77e7ec927e03f0d4cbc38cdb9468` | `2650/results.json::imported_2640_sha256.run.py` | ✓ |
| `2640/results.json` | `1302b6785630f32815651a02340ecd72b556c7e5ec44657311994ee74c13dbaa` | `run1-snapshot/README.md` | ✓ |
| `2640/cycle_log.jsonl` | `d2073370c7985c7b1bc0981e6d8a73231462674bbd2ea49ae0f54ef48a592290` | `run1-snapshot/README.md` | ✓ |
| `2640/run_log.txt` | `ba48b06a0bcee2420c119c61a6c2687dee544225d1e30c29b60fb1a600a2984c` | `run1-snapshot/README.md` | ✓ |

注：`.scratch/63-axis6-predict/run1-snapshot/` 目录内**只**存 `README.md`（其自述三份大文件「本地副本，不入库」），故后三项是「实测 2640 主家文件 sha256」vs「README 记录的 sha256」，五处全等。

**(b) git 区间 / 工作树**

| 命令 | 期望 | 实际 |
|---|---|---|
| `git status --porcelain` | 干净、无 `*.pt` | 空（0 行），`grep -c '\.pt$'` = 0 ✓ |
| staged 文件数 | 0 | `git diff --cached --name-only \| wc -l` = 0 ✓ |
| `git diff --stat 21f390d..HEAD -- pathlm docs tools` | 空 | **非空**（5 文件：`docs/engine_capability.md`、`docs/methodology.md`、`docs/research-schema.md`、`docs/structure.md`、`pathlm/depth_ar.py`）→ 见 finding F1 |
| `git diff --stat 21f390d..HEAD -- .../2640-tree-conf-13m` | 空 | **非空**（`mutate.py`、`mutation_report.json`）→ 见 finding F1 |
| `git diff --stat 3e6248f..HEAD -- pathlm docs tools`（= Phase B 真父提交区间） | — | **空** ✓ |
| `git diff --stat 3e6248f..HEAD -- .../2640-tree-conf-13m` | — | **空** ✓ |

## 6. 事实数字抽查（亲自从 results.json 逐值比对）

**(a) `6.2/project.yaml` 结论 (j)** vs `2650/results.json::checkpoints.13m_s0.criteria.V5_cost_ledger`

| project.yaml 文字 | results.json 值 | 一致 |
|---|---|---|
| τ = 1.4323 / 1.5481 / 1.6586 / 1.7358 | `ledger.1/2/4/8.tau` = 1.432323 / 1.548148 / 1.65859 / 1.735828 | ✓ |
| 边际 Δτ/ΔK（4→8）= 0.0193 | `margins.4->8.delta_tau_per_delta_K` = 0.019309 | ✓ |
| 阈值 0.0579 | `k4_to_k8_threshold.threshold_half_base` = 0.057913 | ✓ |
| α₁ = 0.8000/0.8667/0.9317/0.9785 | 0.8 / 0.866667 / 0.931718 / 0.978458 | ✓ |
| R = 1.233/2.355/4.514/8.773 | `R_K_2640consistent` 1.2334/2.3547/4.5137/8.7725 | ✓ |
| K=4→8 Δα₁ = +0.0467，CI [+0.0296,+0.0629] | `V4_saturation` point 0.04674，ci [0.029604, 0.062894] | ✓ |

**(b)(c) `6.3.3/project.yaml` 结论 (b)/(c)** vs `2660/results.json`

| project.yaml 文字 | results.json 值 | 一致 |
|---|---|---|
| disc_k2 hit@t+2 = 0.4572(s0)/0.4573(s1) | `arms.disc_k2.0/1.hit_t2` = 0.4572088 / 0.4572551 | ✓ |
| λ=1.0 → 0.4878(s0)/0.4892(s1) | `arms.lat_k2_lam1.0/1.hit_t2` = 0.4877872 / 0.4891768 | ✓ |
| λ=0.1 → 0.4688/0.4685 | `arms.lat_k2_lam01.0/1.hit_t2` = 0.4688040 / 0.4684644 | ✓ |
| ar_t1 = 0.6714 | run_log/`ar` t+1 0.6714 | ✓ |
| Δα₁ = +0.0916 CI [+0.0525,+0.1310] | `draftverify.L2.lat_k2_lam1_s0.alpha1` point 0.091646，ci [0.05253, 0.131047] | ✓ |
| 13m_s1 复现 +0.0329 / k=4 +0.0284 | `supplementary_runs` 0.03286 / 0.02844 | ✓ |

## 7. 发现（findings）

- **F1（info · 不影响被验工作 · 属验证配方基线错配）**：任务给出的范围命令 `git diff --stat 21f390d..HEAD -- pathlm docs tools` 与 `… -- .../2640-tree-conf-13m` 均**非空**，与「期望空」不符。
  - 复现：`git diff --stat 21f390d..HEAD -- pathlm docs tools`；`git diff --stat 21f390d..HEAD -- axes/6-predict/6.2-speculative-decoding/experiments/2640-tree-conf-13m`。
  - 成因（已核实）：`21f390d` 是 ticket 63 WP 6.2.2 复核 #1 处置的提交，**不是** Phase B 的父提交；其到 HEAD 之间夹着别的 ticket 的提交——`e894f20`（ticket 67 E1 改 `pathlm/depth_ar.py` + docs）、`e98a26c`/`1c755f3`（ticket 57 M3 改 docs）、以及 ticket 63 自己的 `cfee001`（WP 6.2.2 复核 #2 收口，改 2640 `mutate.py`/`mutation_report.json`）。这些与 Phase B（WP 6.2.3 / WP 6.3.3）无关。
  - Phase B 的真父提交 = `3e6248f`（`git rev-parse 0c4dc01^` = `3e6248fbfd0baaf3002cdb2d42ac3fc1964785cc`）；`git diff --stat 3e6248f..HEAD -- pathlm docs tools` 与 `… -- .../2640-tree-conf-13m` **均为空** ⇒ **Phase B 未越界、未改引擎/文档/2640**，被验工作范围本身正确。
  - 处置建议（不由我修）：后续复核配方应把基线写成 Phase B 父提交 `3e6248f`（或 `0c4dc01^`），不要用 `21f390d`。
- 其它：无 blocker / normal。未发现账实不符、未发现数字口径不一致。

## 8. not-run

- 无。§1–§6 全部亲自执行；§4 的 GPU 重跑获得锁并完成（非 not-run）。VERDICT 行不含执行缺口。

---
本报告由带 shell 的独立执行验证者落盘，仅使用 read/grep/find/ls + bash；未修改任何被验对象。
