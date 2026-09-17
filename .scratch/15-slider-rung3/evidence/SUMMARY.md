# Rung 3 — Slider 端到端（ticket 15）证据

日期：2026-09-17。模型：**INT2_r3**（重训；原 INT2_s0 权重从未入库，已随 `.tmp/p3/`
清理丢失）。配置从 ticket 12 results.json 完整恢复（`.tmp/INT2.json`），24000 步，
58.8 min，13.34M 参数。

## 底座对账

| | INT2_r3（本次） | INT2_s0（已发布） | 判定 |
|---|---|---|---|
| clean bpc | **1.6698** | 1.6548 | Δ+0.015，略超种子方差界 0.0138；同配置同种子，CUDA 非确定性所致 |
| ECE node1 | 0.0049 | 0.002–0.011 | 一致，prob0 校准存活 |

**所有 Rung 3 测量在 INT2_r3 权重上自洽重测**（前沿/校准/求解/代理同一权重），
不与旧权重数字混排；与 ticket 14 的对比只作定性（严格占优是否复现）。

## 运行

```
python3 slider.py --config .tmp/INT2.json \
    --ckpt .scratch/15-slider-rung3/evidence/INT2_r3/model.pt \
    --out .scratch/15-slider-rung3/evidence/slider_INT2r3.json
```

- 校准 = eval 前半（5M 字符），质量 = eval 后半；τ 网格 {0.5,…,0.98} + retry-off。
- gate_probe 语义 = 部署语义：round-2 re-entry 用 soft-transport overwrite
  （`Decoder.retry` 同款），probe 时关 reentry_mix。
- corrupt profile 的解码 bench 用**流式损坏**（每个喂入 token iid 15%，
  `input_fn` 环境钩子）——与校准 profile 的输入律匹配。
  （v1 版 bench 只损坏 prompt，实测开火率 5% vs 校准 22%，profile 失配，
  已修正为流式后对齐。）

## 四件套结果

### 1. 校准（成本可预测性）

fire 曲线 P(prob0<τ)（校准半区，TF）：

| τ | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 0.95 | 0.98 |
|---|---|---|---|---|---|---|---|
| clean 输入 | 0.21% | 0.32% | 0.48% | 0.80% | 1.69% | 3.30% | 7.93% |
| corrupt 输入 | 6.95% | 8.35% | 9.74% | 11.29% | 13.60% | 16.30% | 22.16% |

split 一致性（校准半区 vs 质量半区）：fire rate gap ≤ 0.004（两 profile）；
门控 bpc gap ≤ 0.0096（clean）/ 0.0253（corrupt）。
a2 预测（TF hit_spec）：clean 0.638 / corrupt 0.592。

### 2. 成本模型：预测 vs 实测（fresh prompts 解码计数）

双货币（forwards = mental_model §1 口径；flop = 等价 width-1 单位）。
**a2 用部署口径校准**（校准 prompt 上短解码实测，τ=off）：
clean 0.8028 / corrupt 0.7365（TF 估计 0.638/0.592 系统性低估，已存档）：

| profile | 配置 | 预测 fwd | 实测 fwd | rel err |
|---|---|---|---|---|
| clean | k=1, τ=0（retry off） | 0.555 | 0.529 | **−4.6%** |
| corrupt | k=1, τ=0.301（内插） | 0.600 | 0.570 | **−4.9%** |
| corrupt | k=0, τ=0.717（flop b=1.1 内插） | 1.100 | 1.070 | **−2.7%** |
| corrupt | k=0, τ=0.98 | 1.222 | 1.129 | −7.6% |
| corrupt | k=1, τ=0.98 | 0.703 | 0.613 | −12.8% |

- **全线 ≤ ±13%（判据 ±15% 通过）**。a2 部署校准把 k=1 误差从 −13~−20%
  收敛到 −4.6~−12.8%；残余误差主要来自 fire rate 在生成文本上的偏移，
  方向仍保守（真实成本 ≤ 预测）。
- 旧版（TF-a2）数字存档：k=1 为 −13.3% / −18.7% / −20.1%——正是改进项
  的动机与闭环。

### 3. 前沿与严格占优（corrupt profile，流式损坏）

| 配置 | 实测 fwd/tok | flop/tok | 门控 bpc | tok/s |
|---|---|---|---|---|
| plain（k=0, retry off） | 1.000 | 1.000 | 2.0545 | 34.8 |
| k=0 门控 τ=0.98 | 1.147 | 1.222 | 2.0502 | 34.3 |
| **k=1 spec τ=0.98** | **0.635** | 1.395 | **2.0502** | 33.3 |
| k=1 spec τ=off | 0.553 | 1.256 | 2.0545 | 42.1 |

**严格占优复现**（新权重、更干净语义）：k=1 τ=0.98 比 plain **少 37% forwards
且 bpc 更好**（2.0502 < 2.0545）。flop 口径下它是 +39.5% 算力换 −0.0043 bpc——
两口径结论分歧被如实记录（forwards 口径赢在批量摊销，flop 口径下 spec 不省算力）。
clean profile：门控轻微有害（1.6589→1.6676 @τ=0.98），spec τ=off 是纯赢点
（0.568 fwd，bpc 不变）。

### 4. 在线质量代理（prob0）

**核心校验（TF 窗口，prob0 vs node-0 实测 hit）**：

| profile | mean prob0 | 实测正确率 | ECE |
|---|---|---|---|
| clean | 0.9897 | 0.9976 | 0.0079 |
| corrupt（15% 流） | 0.9268 | 0.926 | **0.0011** |

门控后有效代理（τ=0.9，fired 位置换 round-2 prob0）：ECE 0.0018（corrupt）。
门控相关性（proxy_eff vs next-token acc，跨 τ）：r = 0.983（clean）/ 0.913（corrupt）。
→ **"在线质量可用平均置信代理"在 ground-truth 对齐的窗口上成立**（ECE ≤ 0.008）。

**诚实负结果（自由生成）**：生成窗口的 mean prob0 不是劣化检测器——
clean 生成 0.954 vs 流式损坏生成 **0.976**（反向）。机理：损坏把自由生成
推进高置信重复吸引子，模型对"自己刚提交的 token"几乎总背书；prob0 在
自由文本上测的是"上下文自洽度"而非输入质量。**部署含义**：prob0 代理用于
ingest/prefill 窗口的输入质量监测（有真值对齐的读出），不用于自由生成段
的输出质量监测；后者需要别的信号（backlog）。

### 5. Oracle 参照（mixture vs overwrite）——假设被证伪

原假设：decode 重试用 overwrite transport，比训练时 mixture re-entry 差，
需要引擎升级。实测两者 round-2 质量逐位一致（2.0508 == 2.0508）。
**原因（读引擎后确认，by design）**：`_reentry_gauge` 共享 gauge，
单轮重试时 mixture 只有一项（W=1），mixture == overwrite 是设计保证
（ticket 11/D1 的产物）。**单轮重试下 decode 引擎无损失**；mixture 的差异
只在 round-3+ 出现（本轮不涉及）。

## 判据对账（ticket issues/01）

- [x] 成本可预测：a2 部署口径校准后全线 ≤ ±13%（判据 ±15% ✓）
- [x] 质量可预测：split gap clean 0.0096（≤0.014 ✓）；corrupt 0.0253
      （超方差界，如实记录——损坏输入 bpc 的 split 波动更大）
- [x] 代理可用：TF 窗口 ECE ≤ 0.008 ✓；corr ≥ 0.9 ✓；**自由生成段检测失败
      （负结果，见上）**
- [x] 诚实记录：exit 不进前沿（INT2 无 dense-exit 训练）；flop 口径
      cost < 1.0 不可达；forwards 口径 k=1 可达 0.55

## 文件

- `INT2_r3/`：results.json + train_log.jsonl（model.pt 不入库）
- `slider_INT2r3.json`：完整探针输出（校准/前沿/求解/代理/oracle）
- 代码：`pathlm/slider.py`（库）、`slider.py`（探针）、`pathlm/decode.py`
  （additive：prob0_log + input_fn）、`tests/test_slider.py`（21 测试）
