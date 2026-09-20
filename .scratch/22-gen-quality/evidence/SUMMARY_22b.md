# Ticket 22b — 塌缩域监测信号（文本重复率）证据

日期：2026-09-19。模型：**INT2_r3**（13.34M，同 ticket 22 权重）。纯推理，GPU
总耗时 ~28 min（多次确定性重跑，含 3 次中途修正；单次全量 ~2.8 min，预算 40 min）。

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 probe_collapse_monitor.py \
    --config .tmp/INT2.json \
    --ckpt .scratch/15-slider-rung3/evidence/INT2_r3/model.pt \
    --out .scratch/22-gen-quality/evidence/collapse_monitor_INT2r3.json \
    --n-prompts 8 --n-new 400 --n-baseline 64 \
    --labels-json .scratch/22-gen-quality/evidence/collapse_labels_22b.json
```

## 协议（读数字前必读）

- 24 窗 = 3 prompt 类型 × 8，free-running greedy 400 token（复用 ticket 22
  `run_generation` 原样）。**可复现性对账**：real/corrupt pi<6 的窗口 js 与
  ticket 22 运行逐位一致（real#0 0.04288 / corrupt#0 0.05036 / real#1 0.00299 /
  corrupt#1 0.04603）；random 类 prompt 在 idx 抽取之后才 draw，n_prompts=8 改变
  消耗顺序，故 random 8 窗是**新的**窗口（非 ticket 22 的 random 组）。
- 每窗指标（生成区 400 token）：(a) distinct-2 比率、unique-token 比率、
  (b) 最高单 token 占比、(c) 解码期 mean prob0、(d) 2 路径 TF 重评分窗口 js
  （ticket 22 种子公式，仅生成区行）。
- **塌缩标注 = 人工语义标注**（`collapse_labels_22b.json`，标注者=子代理，
  文本摘录见下，非独立第三方）。规则：**D2** = 深逐字平铺（单一短单元近似
  逐字铺满）；**P** = 短语锤击+胶水（单一内容短语反复，句法胶水真实变化）；
  **T** = 模板变化（多内容短语/填充物变化）。collapse = D2∪P。
- **为何不用契约的阈值规则做主标签**：契约预注册规则（distinct2<0.5 ∨
  unique<0.15）在本模型上**退化为"全部塌缩"**（见"全局重复偏置"）——且阈值
  标签是指标自身的函数，对指标的 AUC 平凡（循环）。人工标签保留 AUC 的
  非循环性（P/T 边界与 d2 有反向交错，恰证明标签携带指标外信息）。

## 关键发现 0 — greedy 生成全局落在重复吸引子域

自然 enwik8 400 字窗（64 窗对照）：distinct2 均值 **0.4596**（min 0.213）、
unique 0.110、max_share 0.136。**24 窗生成文本全部低于自然均值**
（distinct2 0.048–0.386，unique 0.025–0.117 全部 < 自然均值）——13M greedy
解码存在系统性重复偏置，"健康窗"只在相对意义上存在（T = 退化最浅）。
部署含义：塌缩监测的参考水平必须相对化（生成器自身健康模式或滚动基线），
不能照搬自然文本水平。

## 逐窗数据（按 distinct2 升序；class=人工标注）

| 窗 | class | distinct2 | unique | share | js | prob0 |
|---|---|---|---|---|---|---|
| real#1 | D2 | 0.048 | 0.025 | 0.200 | 0.00299 | 0.9992 |
| random#7 | D2 | 0.053 | 0.037 | 0.198 | 0.02398 | 0.9985 |
| random#0 | D2 | 0.065 | 0.035 | 0.215 | **0.37255** | 0.9750 |
| random#1 | D2 | 0.075 | 0.043 | 0.388 | 0.01719 | 0.9751 |
| corrupt#1 | D2 | 0.078 | 0.030 | 0.198 | 0.04603 | 0.9703 |
| random#2 | P | 0.093 | 0.055 | 0.152 | 0.00663 | 0.9964 |
| corrupt#0 | D2 | 0.098 | 0.050 | 0.185 | 0.05036 | 0.9438 |
| real#5 | P | 0.100 | 0.055 | 0.158 | 0.00654 | 0.9987 |
| real#7 | P | 0.105 | 0.048 | 0.160 | 0.00343 | 0.9992 |
| random#3 | P | 0.115 | 0.070 | 0.150 | 0.02611 | 0.9975 |
| corrupt#4 | P | 0.118 | 0.055 | 0.140 | 0.00903 | 0.9706 |
| random#5 | P | 0.128 | 0.060 | 0.158 | 0.01617 | 0.9964 |
| random#4 | P | 0.133 | 0.062 | 0.150 | **0.00000** | 0.9980 |
| real#2 | P | 0.145 | 0.062 | 0.150 | 0.00254 | 0.9988 |
| real#4 | P | 0.173 | 0.052 | 0.172 | 0.00739 | 0.9984 |
| random#6 | P | 0.175 | 0.068 | 0.105 | **0.00000** | 0.9975 |
| corrupt#5 | P | 0.195 | 0.068 | 0.225 | 0.00076 | 0.9384 |
| real#6 | T | 0.201 | 0.085 | 0.158 | 0.03131 | 0.9985 |
| corrupt#7 | T | 0.213 | 0.072 | 0.117 | 0.16854 | 0.9444 |
| corrupt#2 | P | 0.218 | 0.068 | 0.160 | 0.04346 | 0.9327 |
| real#0 | T | 0.276 | 0.107 | 0.107 | 0.04288 | 0.9956 |
| corrupt#6 | T | 0.286 | 0.110 | 0.147 | 0.02433 | 0.9476 |
| real#3 | T | 0.386 | 0.115 | 0.158 | 0.00619 | 0.9984 |
| corrupt#3 | T | 0.386 | 0.117 | 0.145 | 0.00052 | 0.9735 |

文本摘录（审计用，+2 字节偏移解码后可读；`…` 表重复延续）：

- real#1 (D2)：`  CHARACTER SETS\n  CHARACTER SETS\n  CHARACTER SETS…`（全窗单短语平铺）
- random#0 (D2)：字节级图样 `\x0b \n ÂuuÂpl` 逐字平铺
- random#4 (P, js=0.000)：`ARABIC LANGUAGE /ARABIC    ARABIC   IS A ==POPULATION?? OF ARABIC LANGUAGE…`
- real#4 (P)：`a security of a security state that is a security state of a security state…`
- real#3 (T)：`PRIBUTOR   USERNAME /MARK MARK  USERNAME ID 6405351 CONTRIBUTO…`（内容/数字变化）
- corrupt#7 (T)：`NTAREST AND THE ABORITION OF THE CONTEXV OF THE EQURT OF THE ABOMIC COMPOSITION…`

## 判据 22b 对账（label_source=manual，18 塌缩 / 6 健康）

| 指标 | 值 | 判据要求 |
|---|---|---|
| **AUC(distinct2 检出塌缩)** | **0.9815** | ≥ 0.9 ✓ |
| AUC(distinct2, D2-vs-rest) | 0.9907 | （更锐边界，稳健） |
| AUC(max_share) | 0.8148 | （较弱，单项不达标） |
| AUC(js, 告警方向) | **0.3889** | 塌缩域失灵（反对齐）✓ |
| mean js 塌缩 vs 健康 | 0.03529 < 0.04563 | js 不升反降 ✓ |
| 塌缩窗低于合并 js 中位数 | 10/18 = 0.556 | ≥ 0.5 ✓ |
| rank_corr(js, −distinct2) | 0.1366 | 两信号正交 ✓ |
| mean prob0 D2/P/T | 0.977 / 0.985 / 0.976 | prob0 完全不分（T 最低）✓ |
| **pass** | **True** | **判据成立** |

## js 在塌缩域是"双向失灵"，不是单纯假阴

- **假阴**（ticket 22 已见的盲点）：random#4、random#6 js=0.000——两路径分布
  逐位一致，深塌缩读作"完全健康"。
- **假阳**（新发现）：random#0 深字节平铺窗 js=0.373（全体第 2 高）——argmax
  锁死在循环上，但不同层路径的 node-1 **分布**在循环 token 上显著发散：
  不确定循环同时是"高分歧+深塌缩"。js 单独监测塌缩域会同时漏报和误报。
- 机制小结：重复率（distinct2）单调跟踪塌缩深度；js 不跟踪（rank_corr 0.14），
  且在塌缩域内部读数横跨 0.000–0.373。二者互补关系确认并加细。

## 阈值规则审计（预注册规则的诚实记录）

- 固定阈值规则（distinct2<0.5 ∨ unique<0.15）：**全部 24 窗标塌缩**（单类，
  判据不可计算）——unique 臂在所有窗触发（生成 unique 0.025–0.117 全 < 0.15）。
  JSON 中记录为 `threshold_rule_summary.error`。
- d2 轴扫描（关掉 unique 臂）：塌缩计数在阈值 0.05/0.10/0.15/0.20/0.25/0.30/0.40
  处为 1/7/14/17/20/22/**24**——**连续谱，无自然间隙**（0.386 处仍有窗）。
  这就是"阈值标签循环、人工标签必需"的实证根据。

## 总结论

- **判据 22b：成立**——重复率（distinct-2）区分塌缩/健康窗 AUC 0.9815（≥0.9），
  且与 js 解耦：塌缩窗 js 不升反降（0.0353 vs 0.0456）、js 告警方向 AUC 0.389、
  rank_corr 0.14；js 在塌缩域双向失灵（假阴 js=0.000 与假阳 js=0.373 并存）。
- **监测栈定形**：prob0 管"当前 token 对不对"（塌缩域读数 0.976–0.985 完全盲）、
  js 管"语义/路径分歧"、distinct-2 管"塌缩"——三者正交互补，塌缩域专用监测
  信号找到（纯文本统计，零额外前向）。
- **限制**：标注者为子代理（非独立第三方；摘录入档供审计）；T 类"健康"是
  相对健康（greedy 全局重复偏置，全部低于自然文本水平）；单权重（13M INT2_r3）
  单解码模式（greedy），采样解码与更大规模未测。

## 文件

- `collapse_monitor_INT2r3.json`：全部数字（本表皆出自它）
- `collapse_labels_22b.json`：人工标注（规则 + 逐窗类别）
- `probe_run_22b.log`：运行日志
- 代码：`pathlm/gen_quality.py`（新增 `unique_ratio` / `distinct_ngram_ratio` /
  `max_token_share` / `label_collapse` / `collapse_summary` 纯函数，5 个新单测）、
  根探针 `probe_collapse_monitor.py`
- 关联：ticket 22（js 互补信号与盲区发现）、ticket 15/17（prob0 代理）
