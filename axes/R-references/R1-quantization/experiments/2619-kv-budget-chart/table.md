# 2619 · 统一 KV 预算参照线 —— 一张表

横轴公式（判据 C1）：`bytes_per_position = 2 · n_layers · n_kv_heads_eff · head_dim · value_bytes · (positions_kept / T)`，T = 256。**本列是解析口径**；实测显存单列在文末，两者不混。

bpc 与召回全部由本实验**同进程、同一路径**重测（`bpc_of` / `synth_accuracy`），与来源实验逐一对照见 `results.json::cross_check_vs_source`。召回地板：chance = 0.0156（1/64）、局部先验 = 0.0146（copy）/ 0.0156（induction）/ 0.0195（var_recall）；逐臂毁源对照见下表。

| 点 | 类别 | 每位置 KV 字节（解析） | vs 本族基线 | bpc | Δbpc==damage | 合成召回 copy（汇总） | 95% CI | vs chance | vs 局部先验 | 来源路径 | 口径 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `fp16` | 不压缩基线 | 12288 (0.0%↓) | fp16 | 1.483571 | +0.000000 | 0.1413 | [0.1294, 0.1540] | 9.04× | +0.1266 | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 解析字节（非实测显存） |
| `int8` | 量化（R1） | 6144 (50.0%↓) | fp16 | 1.483721 | +0.000150 ⚠不可排序 | 0.1426 | [0.1307, 0.1554] | 9.12× | +0.1279 | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 解析字节（非实测显存） |
| `int4` | 量化（R1） | 3072 (75.0%↓) | fp16 | 1.529232 | +0.045661 | 0.1247 | [0.1135, 0.1368] | 7.98× | +0.1100 | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 解析字节（非实测显存） |
| `win16` | 窗口/驱逐（M2） | 768 (93.8%↓) | fp16 | 2.085526 | +0.601955 | 0.0150 (地板) | [0.0112, 0.0199] | 0.96× | +0.0003 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win32` | 窗口/驱逐（M2） | 1536 (87.5%↓) | fp16 | 1.695384 | +0.211813 | 0.0140 (地板) | [0.0104, 0.0188] | 0.90× | -0.0007 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win64` | 窗口/驱逐（M2） | 3072 (75.0%↓) | fp16 | 1.564077 | +0.080506 | 0.0589 | [0.0511, 0.0678] | 3.77× | +0.0443 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win128` | 窗口/驱逐（M2） | 6144 (50.0%↓) | fp16 | 1.503634 | +0.020063 | 0.1100 | [0.0994, 0.1216] | 7.04× | +0.0954 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win144` | 窗口/驱逐（M2） | 6912 (43.8%↓) | fp16 | 1.497328 | +0.013757 ⚠不可排序 | 0.1100 | [0.0994, 0.1216] | 7.04× | +0.0954 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win160` | 窗口/驱逐（M2） | 7680 (37.5%↓) | fp16 | 1.492435 | +0.008864 ⚠不可排序 | 0.1087 | [0.0982, 0.1202] | 6.96× | +0.0941 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win176` | 窗口/驱逐（M2） | 8448 (31.2%↓) | fp16 | 1.489695 | +0.006124 ⚠不可排序 | 0.1074 | [0.0970, 0.1189] | 6.88× | +0.0928 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `win192` | 窗口/驱逐（M2） | 9216 (25.0%↓) | fp16 | 1.487744 | +0.004173 ⚠不可排序 | 0.1064 | [0.0960, 0.1178] | 6.81× | +0.0918 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 解析字节（非实测显存） |
| `mha` | 不压缩基线 | 12288 (0.0%↓) | mha | 1.482026 | +0.000000 | 0.1530 | [0.1407, 0.1662] | 9.79× | +0.1383 | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 解析字节（非实测显存） |
| `gqa` | 头共享（W3） | 3072 (75.0%↓) | mha | 1.496325 | +0.014299 | 0.1689 | [0.1561, 0.1826] | 10.81× | +0.1543 | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 解析字节（非实测显存） |
| `mqa` | 头共享（W3） | 1536 (87.5%↓) | mha | 1.493763 | +0.011737 ⚠不可排序 | 0.2432 | [0.2283, 0.2586] | 15.56× | +0.2285 | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 解析字节（非实测显存） |

## 表 2 · 逐任务召回（同一仪器，同一 seed）
| 点 | 每位置 KV 字节 | copy（汇总） | copy **毁源**对照 | induction（汇总） | var_recall（汇总） | 不可分辨的腿 |
|---|---|---|---|---|---|---|
| `fp16` | 12288 | 0.1413 | 0.0114 | 0.0645 | 0.0039 | var_recall |
| `int8` | 6144 | 0.1426 | 0.0111 | 0.0625 | 0.0039 | var_recall |
| `int4` | 3072 | 0.1247 | 0.0101 | 0.0488 | 0.0020 | var_recall |
| `win16` | 768 | 0.0150 | 0.0153 | 0.0137 | 0.0020 | copy, induction, var_recall |
| `win32` | 1536 | 0.0140 | 0.0114 | 0.0117 | 0.0020 | copy, induction, var_recall |
| `win64` | 3072 | 0.0589 | 0.0163 | 0.0137 | 0.0059 | induction, var_recall |
| `win128` | 6144 | 0.1100 | 0.0146 | 0.0742 | 0.0059 | var_recall |
| `win144` | 6912 | 0.1100 | 0.0127 | 0.0684 | 0.0059 | var_recall |
| `win160` | 7680 | 0.1087 | 0.0114 | 0.0664 | 0.0039 | var_recall |
| `win176` | 8448 | 0.1074 | 0.0120 | 0.0625 | 0.0039 | var_recall |
| `win192` | 9216 | 0.1064 | 0.0120 | 0.0605 | 0.0039 | var_recall |
| `mha` | 12288 | 0.1530 | 0.0101 | 0.0488 | 0.0156 | var_recall |
| `gqa` | 3072 | 0.1689 | 0.0107 | 0.0273 | 0.0117 | induction, var_recall |
| `mqa` | 1536 | 0.2432 | 0.0120 | 0.0938 | 0.0039 | var_recall |

> 判据 C4：某腿 ≤ `chance + 2·σ_wilson` ⇒ 在该任务上**不可分辨**（不是「机制无效」，SCHEMA §19）。

## 表 3 · 与来源实验逐一对照
| 点 | 本实验 bpc | 来源实验路径 | 来源值 | abs diff | 一致 |
|---|---|---|---|---|---|
| `fp16` | 1.483571 | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 1.483571 | 0.0 | ✓ |
| `int8` | 1.483721 | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 1.483721 | 0.0 | ✓ |
| `int4` | 1.529232 | `axes/R-references/R1-quantization/experiments/2614-kv-quant-eval-13m/results.json` | 1.529232 | 0.0 | ✓ |
| `win16` | 2.085526 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 2.085526 | 0.0 | ✓ |
| `win32` | 1.695384 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.695384 | 0.0 | ✓ |
| `win64` | 1.564077 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.564077 | 0.0 | ✓ |
| `win128` | 1.503634 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.503634 | 0.0 | ✓ |
| `win144` | 1.497328 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.497328 | 0.0 | ✓ |
| `win160` | 1.492435 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.492435 | 0.0 | ✓ |
| `win176` | 1.489695 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.489695 | 0.0 | ✓ |
| `win192` | 1.487744 | `axes/M-carrier/M2-carrier-compression/experiments/2615-retention-retrieval-leg-13m/results.json` | 1.487744 | 0.0 | ✓ |
| `mha` | 1.482026 | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 1.482026 | 0.0 | ✓ |
| `gqa` | 1.496325 | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 1.496325 | 0.0 | ✓ |
| `mqa` | 1.493763 | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/results.json` | 1.493763 | 0.0 | ✓ |

## 表 4 · 缺失组合（下一批候选，判据 C5）
| 组合 | 为什么缺 | 若要补应怎么做 |
|---|---|---|
| GQA × int8/int4 | 两类机制分属不同实验（W3 只测头共享、R1 只测量化），本项目 scope 禁组合实验 | 单因素拆分：固定 GQA(2 头) 再逐档量化 K/V；契约须写清 2×2 的臂数与步时实测 |
| 窗口 × 锚点 / sink | A3 已测锚点通道（needle 76.8→93.3%）但那是**真实文本针**口径；本图未给锚点档的字节/位置与合成召回 | 把 window_anchor_keep_mask 加进本图（锚点占 bytes/pos 极小，应单列一档） |
| 窗口 × 量化 | 同 scope 限制（组合未测） | 固定窗口 d 后量化剩余 KV；两因子在同一个字节轴上相互作用可测 |
| 权重 / 激活量化 | R1 只测了 KV cache 量化；权重/激活量化本项目零经验 | 先分诊（业界已做透 ⇒ 可能直接写 reference 结论），再决定是否本地补测 |
| MLA 低秩 KV（W3 选项之一） | 零经验；低秩投影的压缩率不是本图的单一乘式（需报秩 r） | 先扩展横轴公式为 2·d·r 形式，再测 13M |
| 摘要压缩（M2 选项之一） | 零经验；压缩率依赖摘要器，不是单一乘式 | 先定摘要器与预算口径，再测；注意 SCHEMA §17.9 的成本交叉点 |
| 长 T（>256）下的窗口 / 驱逐 | 本图所有点都在 T=256 上评测；窗口的 bytes/pos = 12288·d/T 随 T 下降，T 越长窗口越便宜（未测） | 在 T∈{512,1024,4096} 上重测窗口档的 bpc 与合成召回，给出与 T 的关系 |
| 各臂的**实测**显存峰值 | 本实验评测实现把 K/V 立刻反量化、窗口按掩码取子集（SDPA 仍算 T×T 分数矩阵）⇒ 实测峰值不随档下降，本实验只报解析列（判据 C2） | 写一个真正的 cache 实现（分页/分组存储 + 只在需要时反量化），再测部署口径峰值 |
| 召回腿的真实文本针（needle v3）跨族 | 2615 只给 trunk 族的窗口档测了真实文本针召回；量化/头共享族没有该腿（不同仪器不得混列，本图未并入） | 用同一 2615 仪器补跑 int8/int4/gqa/mqa 的针召回，作为**第二条召回腿**单列 |
| GQA/MQA 的等参数对照臂 | W3 的自述缺口：GQA/MQA 参数量比 MHA 少 8.75%/10.25%，代价混着容量差 | 补等参数对照（加宽 MLP 或少层），再判头共享的净代价 |

## 实测显存（**与解析列分列**，判据 C2）

- 本实验评测期 GPU 峰值（`torch.cuda.max_memory_allocated`）：**430.5 MiB**（全臂共用一个进程，是上界，不是逐臂读数）。
- 为什么不逐臂报：评测实现把 K/V 立刻反量化、窗口按掩码取子集，SDPA 仍物化 T×T 分数矩阵 ⇒ **实测峰值不随档下降**，把它当横轴会得出「量化不省显存」的错结论。
- 解析列与来源实验一致：见 `results.json::cross_check_vs_source`。

## ASCII 图

```
x 轴 = log10(每位置 KV 字节)（左小而省 → 右大而贵）；y 轴 = damage_bpc（上=更差）

+0.6020 |w                                                                       
+0.5733 |                                                                        
+0.5446 |                                                                        
+0.5160 |                                                                        
+0.4873 |                                                                        
+0.4586 |                                                                        
+0.4300 |                                                                        
+0.4013 |                                                                        
+0.3726 |                                                                        
+0.3440 |                                                                        
+0.3153 |                                                                        
+0.2866 |                                                                        
+0.2580 |                                                                        
+0.2293 |                                                                        
+0.2007 |                  w                                                     
+0.1720 |                                                                        
+0.1433 |                                                                        
+0.1147 |                                                                        
+0.0860 |                                   w                                    
+0.0573 |                                   4                                    
+0.0287 |                                                     w                  
+0.0000 |                  Q                G                 8  w  w w  w     Mb
         +------------------------------------------------------------------------
               768     1536     3072     6144    12288
          每位置 KV 字节（解析）

标记→点：b=fp16  8=int8  4=int4  w=win16,win32,win64,win128,win144,win160,win176,win192  M=mha  G=gqa  Q=mqa
标记→类别：baseline=(不压缩基线)  quant=(量化（R1）)  window=(窗口/驱逐（M2）)  head=(头共享（W3）)
```
