# 2620 · KV 预算图缺失组合 —— 一张表

横轴公式（判据 C1，与 2619 **逐字同一式**）：`bytes_per_position = 2 · n_layers · n_kv_heads_eff · head_dim · value_bytes · (positions_kept / T)`，T = 256。**本列是解析口径**；实测显存单列在文末。`window_anchor` 档的 `positions_kept = window + anchors`（锚点占用位置预算）。

bpc 与召回全部由本实验**同进程、同一路径**重测（复用 2619 的 `bpc_of` / `synth_accuracy`），与来源实验逐一对照见 `results.json::cross_check_vs_source`，与 2619 的参考臂逐位对照见 `results.json::reproducibility_vs_2619`。召回地板：chance = 0.0156（1/64）、局部先验 = 0.0146（copy）/ 0.0156（induction）/ 0.0195（var_recall）。『量化增量』= 组合臂 bpc − 同架构 fp16 臂 bpc（隔离 int8 叠加代价）；『相对 stock』= bpc − 1.483569（不压缩 trunk 的释放版 bpc）。

| 点 | 类别 | 每位置 KV 字节（解析） | vs 本族基线 | bpc | Δbpc==damage | 量化增量 | 相对 stock | 合成召回 copy（汇总） | 95% CI | copy 毁源 | vs chance | 来源路径 | 口径 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fp16` | 不压缩基线 | 12288 (0.0%↓) | fp16 | 1.483571 | +0.000000 | — | +0.000002 | 0.1413 | [0.1294, 0.1540] | 0.0140 | 9.04× | `axes/5-quant/5.1-kv-quant/experiments/2614-kv-quant-eval-13m/results.json` | 解析字节（非实测显存） |
| `int8` | 量化（R1） | 6144 (50.0%↓) | fp16 | 1.483721 | +0.000150 ⚠不可排序 | — | +0.000152 | 0.1426 | [0.1307, 0.1554] | 0.0137 | 9.12× | `axes/5-quant/5.1-kv-quant/experiments/2614-kv-quant-eval-13m/results.json` | 解析字节（非实测显存） |
| `win16` | 窗口/驱逐（M2） | 768 (93.8%↓) | fp16 | 2.085526 | +0.601955 | — | +0.601957 | 0.0150 (地板) | [0.0112, 0.0199] | 0.0127 | 0.96× | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win32` | 窗口/驱逐（M2） | 1536 (87.5%↓) | fp16 | 1.695384 | +0.211813 | — | +0.211815 | 0.0140 (地板) | [0.0104, 0.0188] | 0.0114 | 0.90× | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win64` | 窗口/驱逐（M2） | 3072 (75.0%↓) | fp16 | 1.564077 | +0.080506 | — | +0.080508 | 0.0589 | [0.0511, 0.0678] | 0.0153 | 3.77× | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `mha` | 不压缩基线 | 12288 (0.0%↓) | mha | 1.482026 | +0.000000 | — | -0.001543 | 0.1530 | [0.1407, 0.1662] | 0.0127 | 9.79× | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json` | 解析字节（非实测显存） |
| `gqa` | 头共享（W3） | 3072 (75.0%↓) | mha | 1.496325 | +0.014299 | — | +0.012756 | 0.1689 | [0.1561, 0.1826] | 0.0137 | 10.81× | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json` | 解析字节（非实测显存） |
| `mqa` | 头共享（W3） | 1536 (87.5%↓) | mha | 1.493763 | +0.011737 ⚠不可排序 | — | +0.010194 | 0.2432 | [0.2283, 0.2586] | 0.0107 | 15.56× | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json` | 解析字节（非实测显存） |
| `gqa_int8` | 组合：头共享 × 量化（本批） | 1536 (87.5%↓) | mha | 1.496489 | +0.014463 | +0.000164 | +0.012920 | 0.1686 | [0.1558, 0.1823] | 0.0137 | 10.79× | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json × R1 int8（本批组合，无单一来源）` | 解析字节（非实测显存） |
| `mqa_int8` | 组合：头共享 × 量化（本批） | 768 (93.8%↓) | mha | 1.494105 | +0.012079 ⚠不可排序 | +0.000342 | +0.010536 | 0.2448 | [0.2299, 0.2603] | 0.0107 | 15.67× | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json × R1 int8（本批组合，无单一来源）` | 解析字节（非实测显存） |
| `win16_anchor16` | 组合：窗口 × 锚点（本批） | 1536 (87.5%↓) | fp16 | 1.858406 | +0.374835 | — | +0.374837 | 0.0618 | [0.0539, 0.0709] | 0.0094 | 3.96× | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win32_anchor32` | 组合：窗口 × 锚点（本批） | 3072 (75.0%↓) | fp16 | 1.590539 | +0.106968 | — | +0.106970 | 0.0934 | [0.0836, 0.1042] | 0.0146 | 5.98× | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win32_anchor8` | 组合：窗口 × 锚点（本批） | 1920 (84.4%↓) | fp16 | 1.633097 | +0.149526 | — | +0.149528 | 0.0290 | [0.0236, 0.0355] | 0.0130 | 1.85× | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |

## 表 2 · 逐任务召回 + copy 分 src_pos（同一仪器，同一 seed）

| 点 | 每位置字节 | copy 汇总 | copy@src0（锚区） | copy@src96 | copy@src191 | induction | var_recall | 不可分辨的腿 |
|---|---|---|---|---|---|---|---|---|

| `fp16` | 12288 | 0.1413 | 0.1406 | 0.1572 | 0.1260 | 0.0645 | 0.0039 | var_recall |
| `int8` | 6144 | 0.1426 | 0.1406 | 0.1582 | 0.1270 | 0.0625 | 0.0039 | var_recall |
| `win16` | 768 | 0.0150 | 0.0127 | 0.0176 | 0.0146 | 0.0137 | 0.0020 | copy, induction, var_recall |
| `win32` | 1536 | 0.0140 | 0.0078 | 0.0137 | 0.0205 | 0.0117 | 0.0020 | copy, induction, var_recall |
| `win64` | 3072 | 0.0589 | 0.0127 | 0.0205 | 0.1436 | 0.0137 | 0.0059 | induction, var_recall |
| `mha` | 12288 | 0.1530 | 0.1680 | 0.1416 | 0.1494 | 0.0488 | 0.0156 | var_recall |
| `gqa` | 3072 | 0.1689 | 0.1826 | 0.1807 | 0.1436 | 0.0273 | 0.0117 | induction, var_recall |
| `mqa` | 1536 | 0.2432 | 0.1875 | 0.2607 | 0.2812 | 0.0938 | 0.0039 | var_recall |
| `gqa_int8` | 1536 | 0.1686 | 0.1787 | 0.1777 | 0.1406 | 0.0293 | 0.0117 | induction, var_recall |
| `mqa_int8` | 768 | 0.2448 | 0.1865 | 0.2598 | 0.2832 | 0.0957 | 0.0039 | var_recall |
| `win16_anchor16` | 1536 | 0.0618 | 0.1602 | 0.0107 | 0.0146 | 0.0117 | 0.0020 | induction, var_recall |
| `win32_anchor32` | 3072 | 0.0934 | 0.2471 | 0.0176 | 0.0156 | 0.0176 | 0.0039 | induction, var_recall |
| `win32_anchor8` | 1920 | 0.0290 | 0.0527 | 0.0156 | 0.0186 | 0.0156 | 0.0020 | induction, var_recall |

> 判据 C4：某腿 ≤ `chance + 2·σ_wilson` ⇒ 在该任务上**不可分辨**（不是「机制无效」，SCHEMA §19）。copy@src0 是唯一落在锚区（绝对位置 [0,anchors)）的源档，锚点臂对它有效、对 src96/191 无效——这条分解是「锚点是不是拉动窗口召回最便宜的杠杆」的判读核心。

## 表 3 · 与来源实验逐一对照

| 点 | 本实验 bpc | 来源实验路径 | 来源值 | abs diff | 一致 |
|---|---|---|---|---|---|

| `fp16` | 1.483571 | `axes/5-quant/5.1-kv-quant/experiments/2614-kv-quant-eval-13m/results.json` | 1.483571 | 0.0 | ✓ |

| `int8` | 1.483721 | `axes/5-quant/5.1-kv-quant/experiments/2614-kv-quant-eval-13m/results.json` | 1.483721 | 0.0 | ✓ |

| `win16` | 2.085526 | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 2.085526 | 0.0 | ✓ |

| `win32` | 1.695384 | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.695384 | 0.0 | ✓ |

| `win64` | 1.564077 | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.564077 | 0.0 | ✓ |

| `mha` | 1.482026 | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json` | 1.482026 | 0.0 | ✓ |

| `gqa` | 1.496325 | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json` | 1.496325 | 0.0 | ✓ |

| `mqa` | 1.493763 | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json` | 1.493763 | 0.0 | ✓ |

| `gqa_int8` | 1.496489 | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json × R1 int8（本批组合，无单一来源）` | None | — | — |

| `mqa_int8` | 1.494105 | `axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json × R1 int8（本批组合，无单一来源）` | None | — | — |

| `win16_anchor16` | 1.858406 | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | None | — | — |

| `win32_anchor32` | 1.590539 | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | None | — | — |

| `win32_anchor8` | 1.633097 | `axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | None | — | — |

## 表 4 · 与 2619 参考臂的复现性（跨 run 可控性）

| 点 | 本实验 bpc | 2619 bpc | abs diff | copy abs diff | 一致 |
|---|---|---|---|---|---|

| `fp16` | 1.483571 | 1.483571 | 0.00e+00 | 0.00e+00 | ✓ |

| `int8` | 1.483721 | 1.483721 | 0.00e+00 | 0.00e+00 | ✓ |

| `win16` | 2.085526 | 2.085526 | 0.00e+00 | 0.00e+00 | ✓ |

| `win32` | 1.695384 | 1.695384 | 0.00e+00 | 0.00e+00 | ✓ |

| `win64` | 1.564077 | 1.564077 | 0.00e+00 | 0.00e+00 | ✓ |

| `mha` | 1.482026 | 1.482026 | 0.00e+00 | 0.00e+00 | ✓ |

| `gqa` | 1.496325 | 1.496325 | 0.00e+00 | 0.00e+00 | ✓ |

| `mqa` | 1.493763 | 1.493763 | 0.00e+00 | 0.00e+00 | ✓ |

## 实测显存（**与解析列分列**，判据 C2）

- 本实验评测期 GPU 峰值（`torch.cuda.max_memory_allocated`）：**462.5 MiB**（全臂共用一个进程，是上界，不是逐臂读数）。
- 为什么不逐臂报：评测实现把 K/V 立刻反量化、窗口/锚点按掩码取子集，SDPA 仍物化 `T×T` 分数矩阵 ⇒ **实测峰值不随档下降**，把它当横轴会得出「量化/窗口不省显存」的错结论。
- 真 cache 实现的实测峰值仍是 U3 表 4 的缺失项（不在本批范围）。

## ASCII 图

```
x 轴 = log10(每位置 KV 字节)（左小而省 → 右大而贵）；y 轴 = damage_bpc（上=更差）

+0.6020 |w                                                                       
+0.5758 |                                                                        
+0.5496 |                                                                        
+0.5234 |                                                                        
+0.4973 |                                                                        
+0.4711 |                                                                        
+0.4449 |                                                                        
+0.4188 |                                                                        
+0.3926 |                                                                        
+0.3664 |                  a                                                     
+0.3402 |                                                                        
+0.3141 |                                                                        
+0.2879 |                                                                        
+0.2617 |                                                                        
+0.2355 |                                                                        
+0.2094 |                  w                                                     
+0.1832 |                                                                        
+0.1570 |                       n                                                
+0.1309 |                                                                        
+0.1047 |                                   A                                    
+0.0785 |                                   w                                    
+0.0523 |                                                                        
+0.0262 |                  g                G                                    
+0.0000 |q                 Q                                  8                Mb
         +------------------------------------------------------------------------
               768     1536     3072     6144    12288
          每位置 KV 字节（解析）

标记→点：b=fp16  8=int8  w=win16,win32,win64  M=mha  G=gqa  Q=mqa  g=gqa_int8  q=mqa_int8  a=win16_anchor16  A=win32_anchor32  n=win32_anchor8
标记→类别：baseline=(不压缩基线)  quant=(量化（R1）)  window=(窗口/驱逐（M2）)  head=(头共享（W3）)  quant_head=(组合：头共享 × 量化（本批）)  window_anchor=(组合：窗口 × 锚点（本批）)
```
