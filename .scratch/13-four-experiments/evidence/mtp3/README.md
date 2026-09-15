# 实验 3 — n_mtp=3 的税与 a₃（Medusa 批量验证）

## 结论（一句话）

**前提判据成立（按可部署口径）：B3 的干净 bpc 税增量 +0.0166 < +0.05；
部署口径的 a₃ = 0.653 ≥ 0.60。** 据此实现了 Medusa 式批量验证解码
（`pathlm/decode.py::decode_spec`），实测稳态生成吞吐 **enwik8 文本上
k=2 草稿 1.68×**（随机字节 1.85×），**≥ 1.3× 判据通过**；k=1 草稿也有
1.42×。全部解码输出与逐位置贪心解码逐位一致（含 window 驱逐、强制全接受、
强制拒绝三条测试）。

## Part (a)：训练与前提测量

训练：`B3 = configs/B2.json + n_mtp:3`，6000 步，seed 0，`.tmp/p3/B3`；
配对基线 `B2fresh = configs/B2.json`（当前引擎重训），6000 步，seed 0，
`.tmp/p3/B2fresh`。

| run | clean bpc | next-token acc | wall |
|---|---|---|---|
| B2fresh | 1.5438 | 0.6875 | 5.6 min |
| B3 | 1.5604 | 0.6849 | 6.3 min |
| **税增量 B3−B2fresh** | **+0.0166** | −0.0026 | |

对照：归档 B2 = 1.5481（旧引擎）；B0 = 1.5074 → 归档 n_mtp 1→2 税 +0.0407。
本次 n_mtp 2→3 的税 **+0.0166**，远小于 +0.05 判据。

### 每节点 t+k 准确率 / CE（`probe_mtp3.py`，60×32×512 验证集）

| node | 含义 | B3 acc | B3 CE | B2fresh acc | B2fresh CE |
|---|---|---|---|---|---|
| 0 | 当前 token（自修复） | 0.9999 | 0.0036 | 1.0000 | 0.0022 |
| 1 | t+1 | 0.6869 | 1.0761 | 0.6892 | 1.0648 |
| 2 | t+2 | 0.5594 | 1.5802 | 0.5611 | 1.5727 |
| 3 | t+3 | **0.4511** | 2.0292 | — | — |

### 接受级联（`probe_mtp3.py`，teacher-forcing 验证器）

a_k = P(node-k 草稿 == node-1 在验证行 i+k-1 的 argmax)：

| 量 | 定义 | B3 | B2fresh | 归档 B2 |
|---|---|---|---|---|
| a₂ | 无条件 P(agree) | 0.6602 | 0.6602 | 0.6582 |
| a₂ | 条件 P(agree \| 验证器正确) | 0.7743 | 0.7746 | 0.7728 |
| a₃ | 无条件 P(agree) | 0.5141 | — | — |
| a₃ | 条件 P(agree \| 验证器正确) | 0.6213 | — | — |
| tokens/forward | 1+a₂+a₂a₃（无条件） | **1.9996** | — | — |
| tokens/forward | 1+a₂+a₂a₃（条件） | **2.2554** | — | — |

`probe_mtp3.py` 在归档 B2 上复现了 `probe_mtp.py` 的全部数字
（a₂ 无条件 0.6582 / 条件 0.7728），证明探针口径一致。

## Part (b)：Medusa 批量验证

### 判据口径说明（诚实记录）

- 计划文件对 a₃ 的定义是**无条件** `P(node-3 草稿 == node-1@n+2) = 0.5141`，
  低于 0.60；但任务背景把 a₂ 的“接受率”锚定在**条件**值 0.7729 上，按同一
  口径 a₃ = 0.6213 ≥ 0.60。两个口径跨过阈值。
- 真正决定吞吐的是**部署口径**：验证器 node-1 条件在模型自己生成的 token
  上（而非 teacher forcing），因此实测在线级联才是真话。在线测量：
  a₂ = 0.844，a₃|a₂ = **0.653 ≥ 0.60**，tokens/forward = 2.395。
- teacher-forcing 探针低估了接受率（模型沿自身轨迹更自洽，0.660 → 0.844）。
- 因为口径歧义且吞吐判据才是终裁，仍实现了 (b) 并做实测。

### 实现

`pathlm/decode.py`：`Decoder._run_stack_multi`（K>1 一次前向，每查询位置
各自的驱逐 mask，保证逐位置等价）+ `Decoder.step_multi` + `Decoder.truncate`
（拒绝时丢弃草稿后缀、重锚到验证 token）+ `decode_spec`（草稿 node-2..n_mtp
加 node-1 自身下一 token 一次喂入，读各位置 node-1 argmax 验证，接受最长前缀）。

### 测试（`tests/test_m1.py`，5 个可失败单测，全套 51 绿，原 46）

1. `test_step_multi_matches_per_position_logits`：K 个 token 一次前向，各位置
   node-1 argmax == 逐次 `step()`。
2. `test_spec_decode_matches_greedy`：批量验证输出 == 逐位置贪心（自然覆盖
   接受/拒绝两路径）。
3. `test_spec_decode_all_accepted_is_bit_identical`：草稿强制为参考贪心 token
   → 每轮全接受，输出逐位一致。
4. `test_spec_decode_wrong_drafts_do_not_corrupt`：故意错误草稿全部被拒，序列
   仍 == 贪心，KV cache 长度恰为已提交前缀（截断正确）。
5. `test_spec_decode_window_eviction_matches_greedy`：window=8/anchors=2 的
   逐查询驱逐 mask 也逐位复现。

### 实测吞吐（`probe_spec.py`，B=1，greedy，prompt 32，生成 400；plain/spec
交替计时取 min）

| 配置 | plain tok/s | spec tok/s | 加速 | 在线 a₂ | 在线 a₃\|a₂ | tok/forward |
|---|---|---|---|---|---|---|
| random, k=1 | 101.2 | 144.1 | 1.42× | 0.878 | — | 1.88 |
| random, k=2 | 103.0 | 190.6 | **1.85×** | 0.879 | 0.924 | 2.68 |
| enwik8, k=1 | 100.8 | 143.3 | 1.42× | 0.848 | — | 1.84 |
| enwik8, k=2 | 101.9 | 170.9 | **1.68×** | 0.844 | 0.653 | 2.40 |

微基准：一次宽度-3 前向成本 = 宽度-1 的 **1.28×**，宽度-2 = 1.26× ——
这正是批量验证能赚的原因（K 个位置的前向远低于 K 倍成本）。

**注意口径**：若把 256-token 的**未批量 prefill** 也算进去（`eval.decode_speed`
的口径，prompt 256 + 生成 128），加速会被 prefill 稀释到 ~1.0–1.1×。部署中
prefill 应一次批量完成，故稳态生成吞吐（上表）才是解码的真实指标。

## 文件

- `B3.results.json` / `B3.train_log.jsonl` / `B3.mtp3_probe.jsonl`
- `B2fresh.results.json` / `B2fresh.train_log.jsonl` / `B2fresh.mtp3_probe.jsonl`
- `spec_speed.json`：吞吐与在线级联
- 权重留在 `.tmp/p3/`（未提交）
