# 2631 · 因果口径 KV 预算图 —— 一张表（27 点）

**口径**：本表是 `2621` 的 27 点预算图在 L1 修复后的**逐请求、因果/流式** KV 量化下重测的结果。除量化器 scale 约定外，checkpoint / 数据流 / 窗口序 / 任务 / 判据与 `2621` 逐字段相同；横轴公式（判据 C1）与 `2619/2620/2621` 逐字同一式：`bytes_per_position = 2 · n_layers · n_kv_heads_eff · head_dim · value_bytes · (positions_kept / T)`，T = 256。**本列是解析口径**（值字节）；因果化的 scale 存储开销见文末口径说明。

两腿全部由本实验**同进程、同一路径**重测（复用 2619 的 `bpc_of` / `synth_accuracy`）。召回地板：chance = 0.0156（1/64）、局部先验 = 0.0146（copy）。『Δbpc(新−旧)』= 本因果口径 bpc − 2621 旧口径 bpc（负 = 因果口径质量代价更小）。

| 点 | 类别 | 每位置 KV 字节 | bpc（因果） | bpc（旧） | Δbpc(新−旧) | 方向 | damage_bpc（因果） | copy 召回（因果） | 95% CI | 来源 |
|---|---|---|---|---|---|---|---|---|---|
| `fp16` | 不压缩基线 | 12288 | 1.483571 | 1.483571 | +0.000000 | 不变（逐位） | +0.000000 | 0.1413 | [0.1294, 0.1540] | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` |
| `int8` | 量化（R1） | 6144 | 1.483688 | 1.483721 | -0.000033 | 改善（因果口径质量代价更小） | +0.000117 ⚠不可排序 | 0.1416 | [0.1297, 0.1544] | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` |
| `int4` | 量化（R1） | 3072 | 1.508319 | 1.529232 | -0.020913 | 改善（因果口径质量代价更小） | +0.024748 | 0.1270 | [0.1156, 0.1392] | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` |
| `win16` | 窗口/驱逐（M2） | 768 | 2.085526 | 2.085526 | +0.000000 | 不变（逐位） | +0.601955 | 0.0150 (地板) | [0.0112, 0.0199] | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` |
| `win32` | 窗口/驱逐（M2） | 1536 | 1.695384 | 1.695384 | +0.000000 | 不变（逐位） | +0.211813 | 0.0140 (地板) | [0.0104, 0.0188] | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` |
| `win64` | 窗口/驱逐（M2） | 3072 | 1.564077 | 1.564077 | +0.000000 | 不变（逐位） | +0.080506 | 0.0589 | [0.0511, 0.0678] | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` |
| `win128` | 窗口/驱逐（M2） | 6144 | 1.503634 | 1.503634 | +0.000000 | 不变（逐位） | +0.020063 | 0.1100 | [0.0994, 0.1216] | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` |
| `win144` | 窗口/驱逐（M2） | 6912 | 1.497328 | 1.497328 | +0.000000 | 不变（逐位） | +0.013757 ⚠不可排序 | 0.1100 | [0.0994, 0.1216] | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` |
| `win160` | 窗口/驱逐（M2） | 7680 | 1.492435 | 1.492435 | +0.000000 | 不变（逐位） | +0.008864 ⚠不可排序 | 0.1087 | [0.0982, 0.1202] | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` |
| `win176` | 窗口/驱逐（M2） | 8448 | 1.489695 | 1.489695 | +0.000000 | 不变（逐位） | +0.006124 ⚠不可排序 | 0.1074 | [0.0970, 0.1189] | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` |
| `win192` | 窗口/驱逐（M2） | 9216 | 1.487744 | 1.487744 | +0.000000 | 不变（逐位） | +0.004173 ⚠不可排序 | 0.1064 | [0.0960, 0.1178] | `axes/R-references/R1-quantization/experiments/2619-kv-budget-chart/results.json` |
| `win16_anchor16` | 组合：窗口 × 锚点 | 1536 | 1.858406 | 1.858406 | +0.000000 | 不变（逐位） | +0.374835 | 0.0618 | [0.0539, 0.0709] | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` |
| `win32_anchor32` | 组合：窗口 × 锚点 | 3072 | 1.590539 | 1.590539 | +0.000000 | 不变（逐位） | +0.106968 | 0.0934 | [0.0836, 0.1042] | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` |
| `win32_anchor8` | 组合：窗口 × 锚点 | 1920 | 1.633097 | 1.633097 | +0.000000 | 不变（逐位） | +0.149526 | 0.0290 | [0.0236, 0.0355] | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` |
| `mha` | 不压缩基线 | 12288 | 1.482026 | 1.482026 | +0.000000 | 不变（逐位） | +0.000000 | 0.1530 | [0.1407, 0.1662] | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` |
| `gqa` | 头共享（W3） | 3072 | 1.496325 | 1.496325 | +0.000000 | 不变（逐位） | +0.014299 | 0.1689 | [0.1561, 0.1826] | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` |
| `mqa` | 头共享（W3） | 1536 | 1.493763 | 1.493763 | +0.000000 | 不变（逐位） | +0.011737 ⚠不可排序 | 0.2432 | [0.2283, 0.2586] | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` |
| `gqa_int8` | 组合：头共享 × 量化 | 1536 | 1.496412 | 1.496489 | -0.000077 | 改善（因果口径质量代价更小） | +0.014386 | 0.1680 | [0.1552, 0.1816] | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` |
| `mqa_int8` | 组合：头共享 × 量化 | 768 | 1.493961 | 1.494105 | -0.000144 | 改善（因果口径质量代价更小） | +0.011935 ⚠不可排序 | 0.2412 | [0.2264, 0.2567] | `axes/R-references/R1-quantization/experiments/2620-kv-budget-gaps/results.json` |
| `gqa_int4` | 组合：头共享 × 量化 | 768 | 1.543484 | 1.592538 | -0.049054 | 改善（因果口径质量代价更小） | +0.061458 | 0.1504 | [0.1382, 0.1635] | `2616 gqa-12k-s0 × R1 int4（本批组合，无单一来源）` |
| `mqa_int4` | 组合：头共享 × 量化 | 384 | 1.587956 | 1.689346 | -0.101390 | 改善（因果口径质量代价更小） | +0.105930 | 0.1950 | [0.1814, 0.2094] | `2616 mqa-12k-s0 × R1 int4（本批组合，无单一来源）` |
| `win32_int8` | 组合：窗口/锚点 × 量化（本批） | 768 | 1.695436 | 1.695690 | -0.000254 | 改善（因果口径质量代价更小） | +0.211865 | 0.0143 (地板) | [0.0107, 0.0192] | `2615 win32 掩码 × R1 int8（本批组合，无单一来源）` |
| `win64_int8` | 组合：窗口/锚点 × 量化（本批） | 1536 | 1.564140 | 1.564170 | -0.000030 | 改善（因果口径质量代价更小） | +0.080569 | 0.0592 | [0.0514, 0.0682] | `2615 win64 掩码 × R1 int8（本批组合，无单一来源）` |
| `win16_anchor16_int4` | 组合：窗口/锚点 × 量化（本批） | 384 | 1.912605 | 1.943239 | -0.030634 | 改善（因果口径质量代价更小） | +0.429034 | 0.0586 | [0.0508, 0.0675] | `2620 win16_anchor16 掩码 × R1 int4（本批组合，无单一来源）` |
| `win32_anchor32_int4` | 组合：窗口/锚点 × 量化（本批） | 768 | 1.621580 | 1.643489 | -0.021909 | 改善（因果口径质量代价更小） | +0.138009 | 0.0902 | [0.0805, 0.1008] | `2620 win32_anchor32 掩码 × R1 int4（本批组合，无单一来源）` |
| `gqa_win64` | 组合：头共享 × 窗口（本批） | 768 | 1.559100 | 1.559100 | +0.000000 | 不变（逐位） | +0.077074 | 0.0658 | [0.0575, 0.0751] | `2616 gqa-12k-s0 × 2615 win64 掩码（本批组合，无单一来源）` |
| `gqa_win128` | 组合：头共享 × 窗口（本批） | 1536 | 1.512094 | 1.512094 | +0.000000 | 不变（逐位） | +0.030068 | 0.1364 | [0.1247, 0.1490] | `2616 gqa-12k-s0 × 2615 win128 掩码（本批组合，无单一来源）` |

## 表 2 · 新旧口径并列（判据 ②/③）

- 非量化臂 max |Δbpc| = 0.00e+00（逐位保持 = True）——本轮**唯一**改动的因子是量化器 scale 约定。
- 量化臂中发生改变的：int8, int4, gqa_int8, mqa_int8, gqa_int4, mqa_int4, win32_int8, win64_int8, win16_anchor16_int4, win32_anchor32_int4；
  逐位不变的：（无）。
- int8/int4 同族纯量化臂的 Δbpc（新−旧）：{"int8": {"delta_bpc": -3.3e-05, "direction": "改善（因果口径质量代价更小）", "old_overstated_damage": true, "old_understated_damage": false}, "int4": {"delta_bpc": -0.020913, "direction": "改善（因果口径质量代价更小）", "old_overstated_damage": true, "old_understated_damage": false}}

| 量化臂 | 每位置字节 | bpc（因果） | bpc（旧） | Δbpc(新−旧) | 方向 | copy 召回 Δ |
|---|---|---|---|---|---|---|

| `int8` | 6144 | 1.483688 | 1.483721 | -0.000033 | 改善（因果口径质量代价更小） | -0.000976 |

| `int4` | 3072 | 1.508319 | 1.529232 | -0.020913 | 改善（因果口径质量代价更小） | +0.002279 |

| `gqa_int8` | 1536 | 1.496412 | 1.496489 | -0.000077 | 改善（因果口径质量代价更小） | -0.000651 |

| `mqa_int8` | 768 | 1.493961 | 1.494105 | -0.000144 | 改善（因果口径质量代价更小） | -0.003581 |

| `gqa_int4` | 768 | 1.543484 | 1.592538 | -0.049054 | 改善（因果口径质量代价更小） | +0.010743 |

| `mqa_int4` | 384 | 1.587956 | 1.689346 | -0.101390 | 改善（因果口径质量代价更小） | +0.024089 |

| `win32_int8` | 768 | 1.695436 | 1.695690 | -0.000254 | 改善（因果口径质量代价更小） | -0.000325 |

| `win64_int8` | 1536 | 1.564140 | 1.564170 | -0.000030 | 改善（因果口径质量代价更小） | -0.000976 |

| `win16_anchor16_int4` | 384 | 1.912605 | 1.943239 | -0.030634 | 改善（因果口径质量代价更小） | +0.003581 |

| `win32_anchor32_int4` | 768 | 1.621580 | 1.643489 | -0.021909 | 改善（因果口径质量代价更小） | +0.008789 |

## 表 3 · 前缀不变性（判据 ①，硬门禁）

| 臂 | 未来 token 不改过去 logits | 同批邻居不改本请求 logits |
|---|---|---|

| `fp16` | True | True |

| `int8` | True | True |

| `int4` | True | True |

| `win16` | True | True |

| `win32` | True | True |

| `win64` | True | True |

| `win128` | True | True |

| `win144` | True | True |

| `win160` | True | True |

| `win176` | True | True |

| `win192` | True | True |

| `win16_anchor16` | True | True |

| `win32_anchor32` | True | True |

| `win32_anchor8` | True | True |

| `mha` | True | True |

| `gqa` | True | True |

| `mqa` | True | True |

| `gqa_int8` | True | True |

| `mqa_int8` | True | True |

| `gqa_int4` | True | True |

| `mqa_int4` | True | True |

| `win32_int8` | True | True |

| `win64_int8` | True | True |

| `win16_anchor16_int4` | True | True |

| `win32_anchor32_int4` | True | True |

| `gqa_win64` | True | True |

| `gqa_win128` | True | True |

> 因果掩码挡不住非因果量化的旁路（复核 A2）——这两条不成立，本实验作废。

## 表 4 · 与来源实验 / 2619 / 2620 的复现性（非量化臂）

| 点 | 本实验 bpc | 来源 bpc | abs diff | 一致 |
|---|---|---|---|---|

| `fp16` | 1.483571 | 1.483571 | 0.0 | ✓ |

| `int8` | 1.483688 | None | — | — |

| `int4` | 1.508319 | None | — | — |

| `win16` | 2.085526 | 2.085526 | 0.0 | ✓ |

| `win32` | 1.695384 | 1.695384 | 0.0 | ✓ |

| `win64` | 1.564077 | 1.564077 | 0.0 | ✓ |

| `win128` | 1.503634 | 1.503634 | 0.0 | ✓ |

| `win144` | 1.497328 | 1.497328 | 0.0 | ✓ |

| `win160` | 1.492435 | 1.492435 | 0.0 | ✓ |

| `win176` | 1.489695 | 1.489695 | 0.0 | ✓ |

| `win192` | 1.487744 | 1.487744 | 0.0 | ✓ |

| `win16_anchor16` | 1.858406 | None | — | — |

| `win32_anchor32` | 1.590539 | None | — | — |

| `win32_anchor8` | 1.633097 | None | — | — |

| `mha` | 1.482026 | 1.482026 | 0.0 | ✓ |

| `gqa` | 1.496325 | 1.496325 | 0.0 | ✓ |

| `mqa` | 1.493763 | 1.493763 | 0.0 | ✓ |

| `gqa_int8` | 1.496412 | None | — | — |

| `mqa_int8` | 1.493961 | None | — | — |

| `gqa_int4` | 1.543484 | None | — | — |

| `mqa_int4` | 1.587956 | None | — | — |

| `win32_int8` | 1.695436 | None | — | — |

| `win64_int8` | 1.564140 | None | — | — |

| `win16_anchor16_int4` | 1.912605 | None | — | — |

| `win32_anchor32_int4` | 1.621580 | None | — | — |

| `gqa_win64` | 1.559100 | None | — | — |

| `gqa_win128` | 1.512094 | None | — | — |

## 实测显存（与解析列分列，判据 C2）

- 本实验评测期 GPU 峰值（`torch.cuda.max_memory_allocated`）：**462.5 MiB**。
- 评测实现把 K/V 立刻反量化、按掩码取子集（SDPA 仍算 T×T 分数矩阵）⇒ 实测峰值不随档下降，故只报解析列。
- **部署口径提醒**：严格逐 token 因果 ⇒ 每值一个 fp16 scale（int8 净 3 B/值、int4 净 2.5 B/值）；旧的「int8 省一半」在因果口径下不成立，部署价签须用 `kv_cache_footprint` 的 `streaming_bytes_by_seq_len`。

## ASCII 图（因果口径）

```
x 轴 = log10(每位置 KV 字节)（左小而省 → 右大而贵）；y 轴 = damage_bpc（上=更差）

+0.6020 |               w                                                            
+0.5779 |                                                                            
+0.5538 |                                                                            
+0.5297 |                                                                            
+0.5056 |                                                                            
+0.4816 |                                                                            
+0.4575 |                                                                            
+0.4334 |t                                                                           
+0.4093 |                                                                            
+0.3853 |                              a                                             
+0.3612 |                                                                            
+0.3371 |                                                                            
+0.3130 |                                                                            
+0.2889 |                                                                            
+0.2649 |                                                                            
+0.2408 |                                                                            
+0.2167 |               s              w                                             
+0.1926 |                                                                            
+0.1685 |                                                                            
+0.1445 |               T                   n                                        
+0.1204 |                                                                            
+0.0963 |e                                            A                              
+0.0722 |               dh             S              w                              
+0.0482 |                                                                            
+0.0241 |                              gH             4G             W  W            
+0.0000 |               q              Q                             8    W W W    Mb
         +----------------------------------------------------------------------------
               384      913     2172     5166    12288
          每位置 KV 字节（解析）

标记→点：b=fp16  8=int8  4=int4  w=win16,win32,win64  W=win128,win144,win160,win176,win192  a=win16_anchor16  A=win32_anchor32  n=win32_anchor8  M=mha  G=gqa  Q=mqa  g=gqa_int8  q=mqa_int8  d=gqa_int4  e=mqa_int4  s=win32_int8  S=win64_int8  t=win16_anchor16_int4  T=win32_anchor32_int4  h=gqa_win64  H=gqa_win128
标记→类别：baseline=(不压缩基线)  quant=(量化（R1）)  window=(窗口/驱逐（M2）)  head=(头共享（W3）)  quant_head=(组合：头共享 × 量化)  window_anchor=(组合：窗口 × 锚点)  combo_quant_window=(组合：窗口/锚点 × 量化（本批）)  combo_head_window=(组合：头共享 × 窗口（本批）)
```
