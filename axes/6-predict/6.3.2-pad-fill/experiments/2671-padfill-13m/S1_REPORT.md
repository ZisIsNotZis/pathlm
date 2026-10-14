# S1 交付报告 · 2671-padfill-13m

- 时间戳：2026-10-14T21:10:54+08:00
- 角色：轴 6 · WP 6.3.2-WP1 worker（fresh 上下文、单一写入者）
- 阶段：**S1（契约实现）**，不做 S2（全量扫描）
- 仓库 revision：`46ba4cf`
- 本目录只写这些路径：`axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m/**`

## 1 落盘文件

| 文件 | 作用 |
|---|---|
| `padfill.py` | harness 库：模型（`attn_mask ∈ {causal, full}`）、mask 调度、训练循环、评测（bpc / block 聚类配对 bootstrap CI）、`has_wall_clock`、步时→S |
| `test_padfill.py` | 契约 §5 的 L0 / L0b 硬门 + bpc / CI / tied 读出 / 评测聚合 / S 规则单测（17 项） |
| `smoke_steptime.py` | ≥100 step 步时冒烟，输出实测步时与选定 S |
| `gpu_lock.py` | GPU 互斥锁（`SCHEMA.md` §20；从 2660 复制，name 改 `wp6.3.2`） |
| `MODEL.md` | 结构 / 参数量实测 / 数据 / 复现命令 |
| `S1_REPORT.md` | 本文件 |
| `evidence/pytest_test_padfill.txt` | `pytest -v` 真实输出 |
| `evidence/probe_l0l0b.py`、`evidence/l0_l0b_probe.txt` | L0 / L0b 红绿数值证据（可重跑） |
| `evidence/smoke_steptime.txt`、`evidence/smoke_steptime.json` | 步时冒烟输出与实测记录 |

未写 `run.py` / `results.json` / `analysis.md` / `mutate.py`：按派发纪律属 **S2** 交付物。

## 2 实测参数量

```
$ python3 padfill.py --params
attn_mask=causal  params=12,813,312 (12.813M)
attn_mask=full    params=12,813,312 (12.813M)
```

两臂逐位相同（唯一结构变量 = 可见性，不新增参数），落在契约 §2 预期 12–14M。

## 3 实测步时与选定 S

```
$ python3 smoke_steptime.py --steps 120 --warmup 10 --out evidence/smoke_steptime.json
[smoke] device=cuda arm=fill_bidir_r25 vocab=206 batch=32 T=512 steps=120 warmup=10
[smoke] params=12,813,312 lock_held=True
[smoke] elapsed=10.55s over 120 steps -> step_time=87.9 ms/step
[smoke] S = 1500  (10 run 训练预计 1318s ≤ 1500s)
```

- **实测步时 = 87.9 ms/step**（batch 32 / T = 512 / 交付结构，120 计时步 + 10 warmup）。
- 按契约 §3 规则：`S = floor(1500s / (10 × 0.0879s) / 500) × 500 = 1500`，抬到最低档 **S = 1500**。
- **选定 `S = 1500`**（10 run 训练预计 1318s ≈ 22 min；加评测仍在 S2 的 25 min 硬盒内）。
- GPU 实耗 ≈ 10.6s（≤ 3 min 预算）。**取锁过程**：`/tmp/pathlm_gpu.lock` 先被轴 2 的
  `2651-inter-layer-topology-s1-13m/run.py`（pid 2283632）持有约 8 分钟；本 worker 按
  「登记并重试」排队等锁，**未杀任何他人进程**，锁释放后即取到并在 10.6s 内量完。

> 说明（非判据）：冒烟时 GPU 上并行有其它 worker 的进程，步时可能偏保守；契约 §3 用**实测步时**
> 反解，本值即按此口径取。S = 1500 恰为最低档，抬高不会改变结论。

## 4 `pytest` 真实输出（17 passed）

```
$ python3 -m pytest test_padfill.py -v
test_padfill.py::test_L0_bidir_mask_blocks_true_token_at_M PASSED
test_padfill.py::test_L0_causal_right_nonM_does_not_touch_left_M PASSED
test_padfill.py::test_L0b_probe1_leak_mask_makes_L0_red PASSED
test_padfill.py::test_L0b_probe2_full_attention_makes_causal_check_red PASSED
test_padfill.py::test_causal_model_output_at_i_ignores_all_right_of_i PASSED
test_padfill.py::test_bpc_of_uniform_logits_is_log2_vocab PASSED
test_padfill.py::test_bpc_matches_manual_nll_over_subset PASSED
test_padfill.py::test_bootstrap_ci_direction_positive_delta_excludes_zero PASSED
test_padfill.py::test_bootstrap_ci_direction_centered_delta_includes_zero PASSED
test_padfill.py::test_tied_readout_logits_equal_h_at_E_transpose PASSED
test_padfill.py::test_real_structure_param_count_in_expected_band PASSED
test_padfill.py::test_attn_mask_switch_rejects_unknown PASSED
test_padfill.py::test_sample_fill_mask_respects_min_pos_and_rate PASSED
test_padfill.py::test_make_fill_input_modes PASSED
test_padfill.py::test_eval_block_set_shapes_and_determinism PASSED
test_padfill.py::test_steps_from_steptime_matches_contract_rule PASSED
test_padfill.py::test_has_wall_clock_flags_only_wall_semantics PASSED

============================== 17 passed in 3.54s ==============================
```

（完整输出见 `evidence/pytest_test_padfill.txt`。）

## 5 L0 / L0b 红绿证据

```
$ python3 evidence/probe_l0l0b.py
L0-a  (bidir mask,   perturb true@M)   max|Δ| = 0.0                     [绿 ⇒ 期望 0.0]
L0-b  (causal,       perturb right)    max|Δ| = 0.0                     [绿 ⇒ 期望 0.0]
probe-1 (leak mask,  perturb true@M)   max|Δ| = 4.633474826812744        [红 ⇒ 期望 > 0]
probe-2 (full attn,  perturb right)    max|Δ| = 0.5882356762886047       [红 ⇒ 期望 > 0]
  gate probe probe-1: `assert max|Δ| == 0` 抛 AssertionError ⇒ 检查可失败 ✓
  gate probe probe-2: `assert max|Δ| == 0` 抛 AssertionError ⇒ 检查可失败 ✓
ALL GREEN/PLACED: L0 两条绿、gate probe 两条红。
```

| 检查 | 条件 | 结果 | 判定 |
|---|---|---|---|
| L0-a | `fill_bidir`：扰动 M 位置真 token → M 位置 logits | `max|Δ| = 0.0`（逐位不变） | 绿 ✓ |
| L0-b | `fill_causal_r25`：扰动右侧非 M → 左侧 M 位置 logits | `max|Δ| = 0.0`（逐位不变） | 绿 ✓ |
| L0b-1 | 把 mask 构造改成泄漏（真 token 喂进 M）→ L0-a | `max|Δ| = 4.633` ⇒ `assert == 0` 抛断言 | 红 ✓ |
| L0b-2 | 把 causal 臂改成 full → L0-b | `max|Δ| = 0.588` ⇒ `assert == 0` 抛断言 | 红 ✓ |

两条硬门都**可失败**（gate probe 真变红，`assert max|Δ| == 0` 真抛 `AssertionError`），
证明「无未来泄漏」这条仪器不是恒真。

## 6 S2 确切命令行（供接任者直接跑）

`run.py` 是 **S2 交付物**（README §8），S1 harness 已提供全部库函数。S2 writer 按 §8 实现后，
**逐字段一致地**用下列命令跑（`--steps` 必须是本报告的 S = 1500，写进 `results.json::config.steps`）：

```bash
cd /home/z/vibe/pathlm/axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m
python3 run.py --stage all --out . --seeds 0 1 --steps 1500
```

`run.py` 的接口契约（run.py 内部应做的调用，全部已由 `padfill.py` 提供）：

```python
import padfill as P
train, val, vocab = P.load_data()                 # data/enwik8_full.npz
for arm in P.ARMS:                                # 5 臂
    for seed in (0, 1):                           # 模型种子
        model = P.train_arm(arm, seed, 1500, train, device)   # 数据序 seed 固定 DATA_SEED
        # 评测：AR 参照模型 + 填空臂模型，在同一批 M_r 上出 bpc / 配对 Δ / CI
offsets = P.eval_offsets(len(val))                # 64 个固定 offset，逐条落进 results.json
out = P.eval_block_set(ar_model, fill_model, val, offsets, r, mask_seed=P.EVAL_MASK_SEED)
# out: ar_full_bpc / ar_at_M_bpc / fill_M_bpc / delta_blocks / delta_mean / delta_lo / delta_hi
P.has_wall_clock(results)                          # L5 禁墙钟自检：必须为 []
```

S2 还必须：`results.json` 记 `config.steps = 1500` 与 64 个 offset、`M_r`/bootstrap seed；
若 S < 6000，`analysis.md` 首段声明「Nk 快照口径：与仓库既有 6000 步档不可比」。

## 7 契约一致性 / 未完成项

- S1 交付物齐备；`pytest` 17 passed；L0/L0b 红绿证据已落盘；参数量与步时实测。
- **未完成（属 S2，明确不在本次派发）**：`run.py`、`results.json`、`analysis.md`、`mutate.py` +
  `mutation_report.json`、`run_log.txt`、`train_log.jsonl`。
- 本次冒烟读数为**冒烟级**，**非判决**；未写任何 `analysis.md` 结论段。
- 口径：主判 = bpc 口径 ⇒ 0.014 bpc 带（配对 Δ、≥5 窗口）；率类指标走配对 bootstrap /
  Wilson CI，不跨量纲借用；禁墙钟进 `results.json` / `analysis.md` 判据（`has_wall_clock` 自检）。
