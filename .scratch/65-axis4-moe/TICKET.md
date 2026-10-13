# Ticket 65 · 轴 4 专家轴（MoE）Phase A

> 轴：4 专家轴（MoE）｜目录：`axes/4-moe/**`、`.scratch/65-axis4-moe/**`｜角色：axis4-lead（owner）
> 本文件是**过程流水**（允许日期）。事实（当前状态）写在 `axes/4-moe/**/project.yaml`。

## 0. 人话摘要

轴 4 有三格：4.1 稀疏激活专家（MoE 本体）、4.2 路由与负载均衡、4.3 专家粒度/异构。
MoE 是**业界已做透**的方向（Switch/GShard/Mixtral/DeepSeekMoE），所以本轴的最高风险不是「会不会做」，
而是「**造轮子**」——必须写清「相对业界 SOTA 的增量是什么」，写不出就把格子降为参照。

Phase A 干三件事：① 三格的 `project.yaml` 补齐（role_map 实名 + 相对 SOTA 的增量 + 臂记账）；
② 执行一个**便宜、CPU-only、可证伪**的 WP：MoE 等算力/等参数臂的**精确记账** + 引擎接线 **pre-flight**；
③ 独立复核。

**本阶段不做 13M GPU 训练实验**，理由见 §3：等算力/等参数双对照的臂定义与代价必须先钉死（本 WP 产出），
而且实测步时与 GPU 锁是训练实验的前置；训练实验以「下一步最小实验 + 等效算力成本」登记在
`4.1` 的 `gaps` / WP1。

## 1. 时间线（显式时间戳，actor = axis4-lead）

- `2026-10-13T19:25:15+08:00` 建 `.scratch/65-axis4-moe/`，读 `docs/structure.md` §12 轴 4、
  `docs/research-schema.md`、`SCHEMA.md` §7/§8/§10–§21/§23、`axes/4-moe/README.md` 与三份 `project.yaml`；
  读 `.scratch/55-review/W-PROBLEMS.md` W-03/W-04/W-13/W-17（造轮子/状态矛盾/墙钟口径）。
- `2026-10-13T19:2x` 网络可用（`curl -sSI https://arxiv.org/abs/2101.03961` → HTTP 200）⇒ SOTA 出处
  **可核实**，不用标 `记忆·未核`。逐个核实 arXiv 标题（见 §4）。

（后续动作逐条追加。）

## 2. 三格「想清楚」的结论（人话）

| 格 | 痛点 | 相对 SOTA 的增量（不是「实现一个 MoE」） |
|---|---|---|
| 4.1 稀疏激活专家 | FFN 密度把「容量」与「每 token 算力」绑死 | 在**本仓受控 13M / 同 token 预算**下，把「每 token 等效算力 → (clean bpc, 总参数, 专家使用率分布)」三列同框，并给**等 FLOPs + 等参数双对照**——业界报大尺度吞吐/质量，少给这种小尺度双对照的受控读数 |
| 4.2 路由与负载均衡 | 路由一坍缩，容量增益名存实亡 | 把「均衡强度 → (质量, 使用率分布, 跨 seed 直指派翻转率)」的**三元前沿**在小尺度受控量化；把「小尺度均衡是否必要」做成**可失败判据**（业界多报大尺度，小尺度「均衡-质量」折中少报） |
| 4.3 专家粒度/异构 | 固定激活算力下的容量分配 | 在**固定总参数 + 固定每 token 激活算力**下给「粒度 / 共享比例 → 质量」的**连续受控扫描**（业界配方多为固定点，DeepSeekMoE 是点不是曲线） |

三格共同的前沿口径：以同尺寸 stock dense 为 100%，横轴 = 每 token 等效算力，纵轴 = 序列级 clean bpc，
第三列 = 专家使用率分布（活性）。判据带 = 0.014 bpc（`docs/methodology.md` §2）。

## 3. 为什么本阶段不做 13M GPU 训练实验（如实登记，不谎报）

1. **前置未完成**：等 FLOPs / 等参数双对照的臂定义（E、k、每专家宽、token 预算）必须先钉到数字，
   否则跑出来的三臂不可比（`SCHEMA.md` §15/§17.8/§17.14）。本 WP 先在 CPU 上把这件事做死。
2. **引擎接线未验证**：本仓无 MoE 实装；MoE 作为「新模块」放进本轴实验目录（不动共享 `pathlm/**`，
   §22.7），必须先证明它能作 `Block.mlp` 的**透明替换**（E=1,top-1 时与 stock mlp 逐位一致），
   否则训练的干预是死的（§10）。
3. **时间盒**：Phase A 硬盒 ≤1.5h；13M 三臂 × 实测步时 + §20 串行锁 + n≥2 seed 已超盒；
   且实测步时本身要先测 20 步再定预算（§17.8）——那条路径留给下一阶段。
4. **不做 ≠ 不可判**：等算力训练比较是**正当**问题（这正是 Switch 的 setup）。本阶段只是**降规模/降时间**，
   把「臂定义 + 代价 + 接线」做扎实；训练实验以「下一步最小实验 + 等效算力成本」登记（见 §5）。

**未上报 Main 的事项**：本阶段未改任何共享文件（`pathlm/**`、`tests/**`、`tools/**`），
故无需共享面授权，也未触发 §22 代码编辑锁 / §20 GPU 锁。

## 4. SOTA 出处核实（本轮 `curl` 联网核实标题，非 `记忆·未核`）

| 机制 | 论文 | arXiv | 核实 |
|---|---|---|---|
| top-1 路由 + 负载均衡 aux loss | Switch Transformers (Fedus et al. 2021) | `2101.03961` | HTTP 200，标题核实 |
| top-2 路由 + capacity factor | GShard (Lepikhin et al. 2020) | `2006.16668` | HTTP 200，标题核实 |
| expert-choice 路由（专家挑 token） | Mixture-of-Experts with Expert Choice Routing (Zhou et al. 2022) | `2202.09368` | HTTP 200，标题核实 |
| 8 专家 top-2 | Mixtral of Experts (Jiang et al. 2024) | `2401.04088` | HTTP 200，标题核实 |
| 细粒度专家 + 共享专家 | DeepSeekMoE (Dai et al. 2024) | `2401.06066` | HTTP 200，标题核实 |
| router z-loss（稳定路由 logits） | ST-MoE (Zoph et al. 2022) | `2202.08906` | HTTP 200，标题核实 |
| aux-loss-free 均衡（bias 项） | DeepSeek-V3 Technical Report (DeepSeek-AI 2024) | `2412.19437` | HTTP 200，标题核实 |

> 注：旧归档 `archive/wiki-legacy/axes/W-width/PRIOR_ART.md` §W2 标「检索不可用」，本轮环境**可联网**，
> 故 SOTA 出处改为可核实 arXiv 编号。机制细节（系数/超参）仍以「方向」引用，不作判据。

## 5. WP 排序与预算（等效理论量；墙钟不进 `cost_model`）

| WP | 内容 | 本阶段 | 等效算力成本（理论） |
|---|---|---|---|
| WP0 | 等 FLOPs / 等参数臂**记账** + MoE 接线 pre-flight（CPU） | **执行** | 0 GPU；CPU 秒级 |
| 4.1-WP1 | 13M {dense-B0, moe-top1-E8, moe-top2-E8} 三臂 matched-step 判决 | 下一阶段 | 每 token FFN 激活 FLOPs：三臂逐位相等；总训练 FLOPs = 3 臂 × 98.3M token ×（同 dense）≈ 3×dense-B0 的训练量；n≥2 ⇒ ×2 |
| 4.2-WP1 | 均衡四臂（朴素/aux/capacity/expert-choice） | 待 4.1-WP1 后 | 4 臂 × 同 token 预算 |
| 4.3-WP1 | 粒度三臂（固定总参数/激活算力） | 待 4.1-WP1 后 | 3 臂 × 同 token 预算 |

## 6. 复核派单

- 复核者：fresh 独立上下文 subagent（不得为本 ticket 的 writer）。累计 ≤2 轮。
- 复核报告落盘：`.scratch/65-axis4-moe/REVIEW-1.md`（第 1 轮）。
- 复核契约：核对 ① 三份 `project.yaml` 字段齐全 / `role_map` 实名 / `sota_baseline` 有可核实出处；
  ② WP0 的证据（`results.json` 数字 ↔ `analysis.md`）逐一对账；③ pre-flight 的 E=1/等值断言是否为真；
  ④ 记账口径（等 FLOPs / 等参数）是否自洽；⑤ 是否有墙钟进 `cost_model`；⑥ `tools/summary.py --check` 是否全绿。

## 7. 执行记录（逐条动作，显式时间戳）

| 时间 | actor | 动作 | 结果 / 证据锚点 |
|---|---|---|---|
| `2026-10-13T19:25:15+08:00` | axis4-lead | 建 ticket + 目录；联网核实 7 条 SOTA arXiv 标题 | 本文件 §4 |
| `2026-10-13T19:25+08:00` | axis4-lead | 写 `experiments/2610-moe-arm-ledger-13m/{moe.py,run.py}` | `moe.py`（`TopKMoEFFN`+`MoEDepthARModel`）、`run.py`（ledger/preflight/train） |
| `2026-10-13T19:26+08:00` | axis4-lead | `run.py ledger --out ledger.json`（CPU） | 五臂记账；等 FLOPs 臂专家 MACs 比 1.000；等参数对照相对差 0.058%/0.11% |
| `2026-10-13T19:27+08:00` | axis4-lead | `run.py preflight --out preflight.json`（CPU） | `pass=true`；透明替换 `bitwise_equal=true`；`router_grad_abs_sum=18.50` |
| `2026-10-13T19:27+08:00` | axis4-lead | CPU 冒烟 `run.py train --d 64 --layers 2 …`（dense 与 moe 各一次） | `smoke_dense-B0.json` / `smoke_moe-top2-E8.json` |
| `2026-10-13T19:28:17+08:00` | axis4-lead | §20 取锁，13M 单臂 20 步实测 | `timing_20.json`：`first_20_steps_s=14.958`（争用下 ~0.75 s/步）；使用率归一化熵 0.80–0.98（20 步内无永久坍缩） |
| `2026-10-13T19:29:03+08:00` | axis4-lead | 启动 13M 单臂 1200 步坍缩探针（§20 持锁，nohup） | `.scratch/65-axis4-moe/evidence/run_collapse_probe.sh` → `probe_collapse_1200.json` |
| `2026-10-13T19:29–19:31` | axis4-lead | 写 `README.md`（契约）/ `MODEL.md` / `analysis.md`；改三份 `project.yaml`（role_map 实名 / SOTA 出处 / 增量 / WP0） | `summary.py --check` 全绿（46 个 project.yaml） |
| `2026-10-13T19:52:23+08:00` | axis4-lead | 坍缩探针完成（1200 步，约 23 min，含终评）；落盘并回填 `results.json` / `analysis.md` §4 | `probe_collapse_1200.json`：归一化熵 0.952–0.983、最大占比 0.178–0.214 ⇒ **坍缩不可观测**；温和持久偏斜（专家 3/7 约 1.3–1.5×） |
| `2026-10-13T19:53+08:00` | axis4-lead | 修 `moe.py`（aux loss 保留计算图，4.2 用）+ 补 preflight 的 aux 反传断言；重跑 preflight | `preflight.json`：`pass=true`，`aux_requires_grad=true` |
| `2026-10-13T19:55+08:00` | axis4-lead | 第 1 轮独立复核（fresh subagent）完成 | `REVIEW-1.md`：**PASS-narrow，0 阻断项，12 条非阻断建议** |
| `2026-10-13T19:56–20:02+08:00` | axis4-lead | 按 `REVIEW-1.md` 逐条处置 N1–N12（改 `README.md`/`analysis.md`/`run.py`(N6/N7/N5) + 两份 `project.yaml`(N2/N5/N8)）；重跑 ledger/preflight、重生成 `results.json`；补跑复核者要求的全部命令 | 数字与此前逐位一致（`ledger.json == rerun: True`）；新增 `routing_discrimination`；`summary.py --check` / `check_tree.py` 全绿 |
| `2026-10-13T19:59+08:00` | axis4-lead | 第 2 轮 fresh 复核（修订后修订版）完成 | `REVIEW-2.md`：**PASS-narrow，0 阻断项，5 条非阻断 D1–D5** |
| `2026-10-13T20:02+08:00` | axis4-lead | 处置 D1（`run.py` 逐层最小熵列的 `or 1.0` 会把 0.0 当假值吞掉）+ D2（`analysis.md` 疑字）+ D3/D4/D5 用词；重跑 toy 冒烟验证逐层列 | 见 §11；`usage_entropy_norm_min_layer=0.976178`；None-safe min 自检 = 0.0 |

## 8. 命令与结果

```
python3 run.py ledger --out ledger.json        # CPU，秒级；五臂记账，见 analysis.md §1
python3 run.py preflight --out preflight.json  # CPU，秒级；pass=true
CUDA_VISIBLE_DEVICES="" python3 run.py train --arm dense-B0 --d 64 --layers 2 --seq-len 64 \
    --steps 40 --batch-size 4 --eval-batches 2 --eval-batch-size 4 --out smoke_dense-B0.json
# （20 步 13M 实测与 1200 步坍缩探针走 run_collapse_probe.sh，§20 持锁）
python3 tools/summary.py --check               # OK：46 个 project.yaml 全部符合 docs/research-schema.md
```

## 9. 阻塞 / 未满足判据 / 下一步

- **未满足**：4.1-WP1（4.1 质量判决）、4.2-WP1（4.2 均衡机制）、4.3-WP1（4.3 粒度）均未做（需 GPU 训练；见 §3 时间盒理由）。
- **未改共享文件**：未碰 `pathlm/**`、`tests/**`、`tools/**` ⇒ 未触发 §22 代码编辑锁，无需上报 Main。
- **§20 记账**：1200 步探针为**单臂**、按 §20.1 nohup 后台持锁；取锁时卡上前已有两个外部进程（3.9GB + 8.2GB），
  卡上显存峰值 21.5GB/24.5GB、util ~94–100%（外部争用），已按 §20.3 缩步数（1200 而非 6000）保臂内设计。
- **下一步**：按 §5 跑 4.1-WP1（先测 20 步定预算），再依次 4.2-WP1/4.3-WP1；每步完成后另派 fresh 复核。

## 10. 坍缩探针结论（4.2 前提，单臂非判决）
13M、`moe-top1-E8`、top-1、`aux_weight=0`、seed 0、1200 步（13M 79.8M 参数）：

| 步 | loss | 归一化熵 | 最大占比 |
|---|---|---|---|
| 1 | 5.324 | 0.968 | 0.205 |
| 300 | 2.900 | 0.961 | 0.192 |
| 600 | 2.783 | 0.968 | 0.210 |
| 900 | 2.529 | 0.970 | 0.212 |
| 1200 | 2.277 | 0.980 | 0.194 |
| 终评 | bpc 6.537 | 0.976 | 0.187 |

⇒ 命中预登记阀值「**本尺度坍缩不可观测**」（熵全程 > 0.5、最大占比 < 0.6）；但存在温和持久的偏斜
（专家 3/7 长期约 1.3–1.5× 均匀份额）。即 `4.2` 的「修复坍缩」叙事在本尺度/本预算下**无对象**；
均衡机制的价值应从「均衡是否提高质量/降偏斜」去测。限制：1200 步快照（未收敛）、单 seed、无 aux 对照、禁外推。

## 11. 复核结果与残留（当前状态）

- **复核 1**（`REVIEW-1.md`，2026-10-13）：**PASS-narrow，0 阻断，N1–N12 非阻断（均已处置）**。
- **复核 2**（`REVIEW-2.md`，2026-10-13）：**PASS-narrow，0 阻断，D1–D5 非阻断（D1–D5 均已处置）**。
- **残留（非阻断，如实登记）**：
  1. **无第 3 轮复核**：任务限「累计复核 ≤ 2 轮」。D1–D5 是复核 2 自己提出的改法、writer 已按原样落改并本地复跑，
     但这些改动**落在复核 2 快照之后**，无新的 fresh 复核覆盖（按任务硬上限不再派）。
  2. **实证未复核**：两轮复核者均无执行工具，`summary.py --check` / `run.py preflight|ledger` / `git status`
     由 writer 实跑并记录（§8），未获独立实跑复核。
  3. **D1 语义**：逐层最小熵列的 `or 1.0` 会把 0.0（本当检测的整层坍缩）当假值吞掉；已改为 None-safe；
     该列**未进任何已报数字**（探针未重跑，`analysis.md` 已声明「本探针未测」）。
  4. 探针使用率为**跨 12 层池化**（不分辨整层坍缩）；逐层列已加进 `run.py` 供下一阶段。
  5. 全实验（4.1-WP1/4.2-WP1/4.3-WP1）均未做；需 GPU 训练。
