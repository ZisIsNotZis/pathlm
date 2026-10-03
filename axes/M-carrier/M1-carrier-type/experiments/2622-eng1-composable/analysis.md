# analysis · 2622 ENG1 引擎可组合性（GQA × carry × window）

## 人话摘要

**这个设计想让模型做到什么事。** 我们此前研究过三种各自独立的「省算力 / 加能力」做法：
把注意力的 K/V 头数变少（GQA/MQA，省 KV 显存）、加一条把远处信息搬近处的「写入通道」
（carry）、以及训练时只看最近一段位置（滑动窗口，省注意力）。可惜引擎把它们写成了
**三者只能选一个**：选 GQA 是一种块、选 carry 是另一种块、开窗口又是第三种校验，
两两之间直接报错。结果是「GQA × carry × window 一起上」这个组合**在代码里根本不存在**，
所以它**从来没被测过**（这直接导致上一波 W3 只能把契约降级）。

**相对普通 decoder-only transformer 改了什么。** 普通变压器就是一堆标准块（自注意力 +
MLP）。我们没有新增参数、没有改注意力公式，只改了**块的组装方式**：把「KV 头投影」
抽成一个**可复用组件**（谁都能挂上，不再是「选不选 GQA 这个类」的单选题），窗口掩码
本来就是从外面传进去的一个参数（`attn_mask`），现在把它当成**与块无关的正交装饰层**。
缺省（什么都不开）时生成的块与原来的标准块**逐位相同**——即改动对老实验零影响。

**为什么这么做。** 痛点不是质量，是**可测性**：一个从没被造出来过的组合，谈不上测
它的好坏。把互斥拆开、把组合造出来，后续才可能（在别的实验里）测「GQA 省了显存、
carry 补了长程、窗口省了注意力」三者叠加的真实代价与收益。

**怎么测的。** 分三层，都不比质量，只验证「组合存在且每个特性活着」：
① 四条可失败单测（GQA×carry / GQA×window / carry×window / 三者齐开）在 CPU 上建模型、
量三个活性量（KV 头投影真的变窄吗 / 写入通道真的被读吗 / 窗口真的改变了前向吗）；
② 变异验证：把引擎四处「故意改坏」，看单测是否**抓红**，再逐字节还原；
③ 20 步 GPU 冒烟：用 ENG1 前会直接报错的那条命令（`--train-window 16 --carry-write
delta --n-kv-heads 2`）真训练 20 步，看三者能不能共存训练、活性量是否落盘。

**测出来多少。**
- **组合可建**：四个组合现在都能建出模型（`GQA×carry` 与三者齐开走新块 `ComposableBlock`）。
  参照系 = ENG1 前的引擎：同样的配置会抛 `ValueError`（硬互斥）。
- **KV 头投影真的变窄**：d=32/8 头、`n_kv_heads=2` 时 qkv 输出宽度 = 48（= 32 + 2·2·4），
  不是 MHA 的 96；真进 SDPA 的 K/V 张量形状 `[2, 2, 12, 4]`（头数真的 = 2）。参照系 =
  MHA 的 96 / `[2, 8, 12, 4]`。**这是张量事实，不是标签。**
- **写入通道真的被读**：把读出增益从单边零激活后，`read_share` = 0.0032 / 0.00298 / 0.0046
  （> 0）；把它置零会改变 logits。参照系 = 单边零起步时 `read_share` 恰为 0。
- **窗口真的生效**：窗口臂与「同 seed 同权重、全窗口」对照的 logits **逐位不同**；冒烟里
  掩码开/关的最大 logit 差 = 1.61。参照系 = 全窗口（差 = 0）。
- **20 步冒烟可训练**：loss 3.7849 → 0.7768（持续下降，无 NaN）；`kv_heads.alive=true`；
  窗口 `alive=true`；写入通道读出增益从单边零 0 → 1.6e-4（**开始逃逸**，但读出占比仍
  ≈1e-6，见下「失败/排除」）。峰值显存 **735.6 MB**、墙钟 0.04 min（12.62M 参数）。
- **缺省逐位还原 v1**：缺省块的权重与前向与标准块**逐位相同**，参数量 12,704 = 12,704。
- **回归**：`pytest tests/ -q` **367 passed**（基线 361 + 本线新增 6），0 failed。
- **变异**：4 处破坏全部**抓红**、且全部**逐字节还原**（sha256 与原文件一致）。
- **对账**：三道对账 **52/52 通过**。

**结论与下一步。** 采纳（工程面）：`GQA × carry × window` 从「不可测」变成「可测」，
缺省行为不变，其它实验零影响。**这只是一张入场券，不是性能结论**——「三者叠加值不值」
需要新的训练实验（见 §「未测项」）。不采纳任何质量主张（本线未测质量）。

---

## 1. ENG1 前 / 后：哪些组合从未被测过、现在可测

（逐字对应 §4 判据④）

| 组合 | ENG1 前 | ENG1 后 | 现在怎么测 | 证据 |
|---|---|---|---|---|
| `carry × window` | **可测**（旧护栏已允许） | 保持可测 | `CarryChannelBlock` + `attn_mask` | `combos.carry_x_window`：qkv 96（MHA）、`read_share`0.0046、窗口 logits 不同 |
| `GQA × window` | **硬互斥 ⇒ 不可测** | **可测** | `GQABlock` + `attn_mask`（正交装饰层） | `combos.gqa_x_window`：qkv 48、`k_shape[1]=2`、`attn_mask=True`、窗口 logits 不同 |
| `GQA × carry` | **硬互斥 + 块类二选一 ⇒ 不可测** | **可测** | `ComposableBlock`（KV 头投影从块类解耦） | `combos.gqa_x_carry`：`ComposableBlock`、qkv 48、`read_share`0.0032 |
| `GQA × carry × window` | **不存在 ⇒ 从未被测** | **可测** | `ComposableBlock` + `attn_mask` | `combos.gqa_x_carry_x_window` + 20 步冒烟（loss 3.78→0.78） |
| `attn_sparse_k / attn_linear` 与任一 | 仍互斥 | 仍互斥（本线未开放） | — | 引擎校验保留 |

> **不外推**：本表只说「可测」，**不说**组合的性能/质量。

## 2. 组合活性量（CPU 自检，`results.json::combos`）

测试位点：d=32 / 3L / 8H（dh=4）/ seq=12 / dense+sequential / seed=0；carry 读出增益激活值
= 0.5。MHA 参照宽度 = 3·d = **96**。

| 组合 | 块类 | qkv 宽度 | 真进 SDPA 的 K 形状 | 掩码进 SDPA | carry `read_share` | 窗口 vs 全窗口 |
|---|---|---|---|---|---|---|
| `gqa_x_carry` | `ComposableBlock` | **48** | `[2,2,12,4]` | False（无窗口） | **0.003198** | — |
| `gqa_x_window` | `GQABlock` | **48** | `[2,2,12,4]` | **True** | — | **不同** |
| `carry_x_window` | `CarryChannelBlock` | 96 | （MHA，无 GQA） | — | **0.004629** | **不同** |
| `gqa_x_carry_x_window` | `ComposableBlock` | **48** | `[2,2,12,4]` | **True** | **0.002974** | **不同** |

- qkv 宽度式：GQA = `d + 2·n_kv_heads·dh` = 32 + 2·2·4 = **48**；MHA = `3·d` = **96**。
  `reconcile_numbers.py` 对每个组合用配置重算并比对（52 项对账全过）。
- `read_share` = 写入通道读出量占注意力输出量级的比例（> 0 ⇔ 通道被读）。

## 3. 缺省路径逐位还原 v1（`results.json::default_bitwise_v1`）

| 检查 | 结果 |
|---|---|
| `ComposableBlock`（MHA、无 carry）vs stock `Block`：state_dict 键顺序 + 权重逐位 | **相等** |
| 同输入前向输出逐位 | **相等** |
| 参数量 | 12,704 = 12,704 |

快路径块类（`results.json::fast_paths`）：`gqa_only`→`GQABlock`、`carry_only`→
`CarryChannelBlock`、`default`→`Block`、`gqa_x_carry`→`ComposableBlock`（**只有组合走新块**）。

## 4. 20 步 GPU 冒烟（`smoke/master/`）

命令即 ENG1 前会报错的那条：`--d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256
--dense --synth-task assoc-mqar … --train-window 16 --carry-write delta --n-kv-heads 2`。
数据 = 合成多查询关联回忆（不用 enwik8，SCHEMA §19）。

| 量 | 值 | 参照系 |
|---|---|---|
| 参数量 | 12,616,808 | — |
| 步数 / token | 20 / 40,960 | — |
| loss（step 0 → 19） | 3.7849 → **0.7768** | 下降 = 可训练 |
| 峰值显存 | **735.6 MB** | 契约上限 5GB net（≪） |
| 墙钟 | **0.04 min** | 契约预算 20 min（≪） |
| `kv_heads.alive` | **true**（n_kv_heads 2 / qkv 384 / −75% KV） | MHA 512 |
| `window_mask.alive` | **true**（开/关最大 logit 差 1.61） | 全窗口 = 0 |
| carry `read_scale_absmean`（0→19 步） | 0 → **1.6e-4**（逃逸单边零） | 单边零起步 = 0 |
| carry `read_share`（step 19） | **1e-6**（≈ 未真正被读） | 见「失败/排除」第 3 条 |
| 末行 `train_log.jsonl` 含 `kv_heads` / `attn_carry` / `window_mask` | **三者齐全**（活性量落盘） | — |

> **禁外推**：20 步只证明「三者共存可训练」，**不**给出任何质量/收敛/性能结论
> （SCHEMA §17.5：冒烟时间字段不得外推）。

## 5. 对照臂（`controls/README.md`）

| 对照 | 输入分布 | 要预测什么 | 部署对应 | 本实验读数 |
|---|---|---|---|---|
| `default` = stock `Block` | CPU 随机 token | 逐深 next-token | 「什么都不开」的基底 | 权重/前向与可组合块逐位相同 |
| `full-window`（`train_window=12=seq_len`） | 同组合、同 seed、同权重 | 同目标 | 「窗口关掉」的对手 | 窗口臂 logits 与其**不同** |
| 单特性快路径 | 同测试位点 | 同目标 | 既有实现 | 块类 = `GQABlock`/`CarryChannelBlock` |

## 6. 判据核对（§4 四条）

| # | 判据 | 结果 | 证据 |
|---|---|---|---|
| ① | 组合可失败单测（四条 + 变异抓红） | **通过** | `tests/test_eng1_composable.py` 6 条；`mutate.py` 4 处全抓红（`all_caught_red=true`） |
| ② | 既有 361 测试全绿 | **通过** | `pytest_full.txt`：**367 passed**（361+6），0 failed |
| ③ | 20 步冒烟三者共存可训练（活性量落盘） | **通过** | §4 表；末行三块齐全，loss 下降，`kv/window alive=true` |
| ④ | 报告写明 ENG1 前/后哪些组合从未被测、现在可测 | **通过** | §1 表 |

## 7. 变异验证（`mutation_report.json`）

| id | 破坏点 | 期望抓红 | 结果 | 还原 |
|---|---|---|---|---|
| M1 | 可组合块丢掉写入通道读出 | `gqa_x_carry`、三者齐开 | **红**（1 failed,1 passed） | 逐字节还原 ✓ |
| M2 | KV 头投影忽略窗口掩码 | `gqa_x_window`、三者齐开 | **红**（2 failed） | 逐字节还原 ✓ |
| M3 | 写入通道块忽略窗口掩码 | `carry_x_window` | **红**（1 failed） | 逐字节还原 ✓ |
| M4 | 可组合块前向忽略 `attn_mask` | 三者齐开 | **红**（1 failed） | 逐字节还原 ✓ |

`sha256_original == sha256_restored`，4/4 抓红。

## 8. 三道对账（`reconcile_report.json`）

- **算术**：每个组合的 qkv 宽度用配置重算（GQA=48 / MHA=96）；缺省两块参数量相等；
  活性量只在对应旗标为真时出现。
- **任务**：4 个组合同一 CPU 位点（d/layers/heads/seq/dense/seed 一致）。
- **口径**：缺省权重 + 前向逐位还原；快路径块类不变；冒烟块类=`ComposableBlock`、
  `final_step=19`、末行三块齐全；回归 0 failed；变异全抓红且 sha 还原。
- 结果：**52/52 通过**（退出码 0）。

## 9. 失败 / 反向 / 排除（诚实记账）

1. **旧护栏断言与 ENG1 目标直接冲突**：`test_depth_ar.py` 两条用例原本断言
   「GQA⊕train_window」「carry⊕GQA」必须报错。ENG1 的目标正是解这两道锁 ⇒ 把这两条
   改为**正向断言**（组合现在能建）。**判据线未改**，只改了与新目标矛盾的旧断言。
   这是**有意的口径重叠**，如实记录。
2. **冒烟未做多 seed / 未比质量**：本线只验证可测性，不做质量读数，因此无方差界问题，
   也**没有**质量结论可下。
3. **carry 在 20 步里「刚逃逸单边零、尚未真正被读」**：`read_scale_absmean` 0→1.6e-4
   （已离开零点），但 `read_share` 仅 ≈1e-6 ⇒ `carry_liveness.alive=false`（要求读出占比
   > 0 且该层统计未被 6 位小数抹平）。按失败决策树 §8 第 3 条，这是**步数不足**而非机制
   问题——**记录实际值、不外推**。（`carry_liveness.alive` 在冒烟产物里如实为 false。）
4. **`carry × window` 本就可测**：该组合在 ENG1 前已允许，此处只作回归钉，不是新能力。

## 10. 未测项（下一步最小实验 + 成本，SCHEMA §23 第 6 问）

| 未测项 | 下一步最小实验 | 成本 |
|---|---|---|
| 组合的**质量/代价**叠加（GQA 省显存 × carry 补长程 × 窗口省注意力） | 13M 上四臂：`stock` / `+GQA` / `+GQA+carry` / `+GQA+carry+window`，同 token 预算、合成 MQAR 主任务 | ≈40–60 min（4×13M×12k 步） |
| `attn_sparse_k` / `attn_linear` 与 GQA/carry/window 的进一步组合 | 仅当有明确机制假设时再开放（当前仍互斥） | 未定 |
| `carry_state_dim` 与 GQA 组合下的读出容量轴 | 复用本线可组合块 + `--carry-state-dim` | ≈20 min 冒烟 |
| 20 步冒烟的 read 激活需要更长训练才可判 | 200 步冒烟，查 `read_share` 是否离开 1e-6 | ≈1 min |

## 11. 残留风险

- 本线的活性量都在极小 CPU 位点（d=32）上量；冒烟证实 M1 引擎（12.62M）可用，但
  **未做质量对照**。
- 组合块 `ComposableBlock` 只走 dense+sequential；非 dense（退出/填充）路径仍显式拒绝
  （与 GQABlock/CarryChannelBlock 同款护栏）。
- `tools/check_schema.py` 报「`experiments/README.md` 生成物与 catalog 不一致」——这是
  X1/X2/X3 三线新增实验目录的共同结果，需主 Agent 跑 `tools/build_wiki.py` 刷新
  （本线按授权范围**不碰** `axes/**/README.md` / `catalog.yaml` / `INDEX.md`）。
