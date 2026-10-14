# 2661 · implicit（纯窗口 16）vs explicit（完备机制：逐层独立状态 + 内容写门）正面对撞 · 13M · enwik8

> 契约来源：`.scratch/62-axis1-time/TICKET.md` §10.2（WP-C1）与本项 `1.1.4.5/project.yaml`
> 的 `milestone` / 判据。本文件是**预登记契约**：§4 判据线与 §4a 操作性定义在**启动任何 GPU
> 运行之前**写死，写下后不得改。
> 本实验同时作为 `1.1.4.1`（显式记忆）的**召回轴**证据（目录归 `1.1.4.5`，因该项目的 milestone
> 就是这场对撞；`1.1.4.1` 的 `experiments` 加同一路径作指针——框架项 F10，见 TICKET §10.3）。
> 入口：`run.py`（纯推理，五条腿）；语义层仪器 `realtext_probe.py` + `needle_stats.py`
> （自 2634 复制，逐字节不改）；配置 `config.json`（含逐臂 sha256）；原始结果 `results.json`
> / `needle_raw.json` / `run_log.jsonl`；判读 `analysis.md`；权重清单 `MODEL.md`。

## 1. 命题

**在同一个任务、同一 token 预算、同一可见性下，把「窗口丢掉的左侧信息」搬回来这件事，
靠显式状态（explicit）能不能做成、值不值？**

此前该命题只在**退化实现**（单共享槽 delta，`2634`）上测过，且完备机制只有**质量**读数
（`2639`：等字节点 `+0.068228` bpc、补回窗口→全量差距的 80.1%）、**没有召回读数**。
本实验用**同一台仪器**（2634 的毁源敏感度探针 + needle v3 短语针 + clean bpc），
把 {implicit = 纯窗口 16，零新增通道} 与 {explicit = 完备机制（E1：逐层独立状态 +
内容写门，`carry_per_layer=True`）} 放在**同一批已有 checkpoint、同一窗口序、同一针位**上
正面对撞。

可证伪形式：*若 explicit（`carrypl16_s0`）的窗外召回/敏感度**只**停在同掩码地板（= 无记忆
负对照的读数），则「显式状态能搬回窗口丢掉的信息」在**本尺度/本预算点**上不成立；
若其敏感度 `Δbpc_far > 0` 且配对 CI 排除 0、且硬同索引子集上召回显著高于 implicit，
则「搬回来了」在本尺度上成立（值不值另由等字节点质量差裁决，`2639`）。*

## 2. 臂集（全部已训练完成；只做推理；逐臂 sha256 与 `2639/config.json` 核对）

| 臂 | 来源 checkpoint | 机制 | seed | 参数 | 每位置等效状态字节 | sha256 |
|---|---|---|---|---|---|---|
| `win16_s0` | `2634/train/rt_none_s0/model.pt` | 纯窗口 16（implicit，无通道） | 0 | 13,536,768 | 0 | `c3f6f08a…` |
| `win16_s1` | `2634/train/rt_none_s1/model.pt` | 同上 | 1 | 13,536,768 | 0 | `2bfe167c…` |
| `carry16_s0` | `2634/train/rt_delta_s0/model.pt` | 显式状态（**退化**：单共享槽 delta，chunk 64） | 0 | 13,801,064 | 16,384（8192 floats，fp16） | `d9211482…` |
| `carry16_s1` | `2634/train/rt_delta_s1/model.pt` | 同上 | 1 | 13,801,064 | 16,384 | `b3d3cdbc…` |
| `carrypl16_s0` | `2639/train/pl16/model.pt` | **完备机制**：逐层独立状态 + 内容写门，`carry_per_layer=True`、`carry_chunk=256`、γ=1 | 0 | 13,801,064 | 196,608（12×8192 floats） | `382a74aa…` |
| `carrypl16d_s0` | `2639/train/pl16d/model.pt` | 完备机制 + 衰减 γ=0.99 | 0 | 13,801,064 | 196,608 | `cb1940ee…` |

完整 sha256 / size 见 `config.json`。

## 3. 单因素声明（诚实记账）

- **唯一变量 = 机制**（有/无写入通道、通道完备度）。全部 6 臂逐字段共享训练配方：d=256 /
  12 层 / 8 头 / mlp_mult=6 / T=256 / 12k 步 / bs=32 / lr=1e-3 / enwik8 前 90M 字符 /
  **训练窗口 16、锚点 0** / `--exit-mode uniform`。
- **复用臂 = eval-only**：6 臂全部来自 `2634`/`2639`，本实验**不训练**任何参数；等值性由
  §4 判据⑤（clean bpc 与 2638/2639 记录值逐位复现）佐证。
- **`carrypl16` 与 `win32` 等字节**（393,216 B）不是本实验的判据（那在 `2639`）；本实验的对照是
  **同窗口 16 下有无通道**——「可见性相同」意味着窗口外内容对 implicit 臂**结构上不可达**。
- **`carrypl16` vs `carry16` 之间有 3 个自由度**：逐层状态、内容写门（同一开关 E1）、chunk
  （64→256）。三者都在「显式状态」这一侧；implicit 臂三者皆无。⇒ 本实验对撞的是**类**
  （有/无通道），不是单参数；完备度内部的归因见 `2639` §5 与 `1.1.4.1` 的 `gaps`（F8）。

## 4. 判据线（预登记；写下后不得改）

| # | 腿 | 判据 | 判定线 | 该涨/该跌/必须不变 |
|---|---|---|---|---|
| ① | (b) needle | **该涨**：`carrypl16_s0` 的 `beyond` 桶召回相对**配对无针地板**的提升 | 在**硬同索引子集**（`intersection`：六臂 `plant=False` 全部答不出的针）上，「植入召回 − 同索引地板」的两比例差 **95% CI 下界 > 0**；并并列报该子集上植入召回的 Wilson 95% CI 下界 > 0（该子集地板恒为 0） | 该涨 |
| ② | (b) needle | **该跌（负对照）**：`win16_s0`/`win16_s1` 的 `beyond` 召回停在同掩码地板 | 两臂「植入 − 配对地板」两比例差 95% CI **包含 0**（不越地板）；复现 2634 的读法（整体无区分力） | 该跌 |
| ③ | (a) 探针 | **该涨**：`carrypl16_s0` 把窗外前缀内容用起来 | `Δbpc_far > 0` **且**逐簇配对 bootstrap 95% CI **下界 > 0**；`carrypl16d_s0` 并列报 | 该涨 |
| ④ | (a) 探针 | **前提（仪器有效）**：`win16_*` 对被毁前缀**逐位不敏感** | `max_abs_logit_diff_on_eval_rows == 0.0`（**硬断言**；否则仪器失效，先修仪器再判） | 必须不变 |
| ⑤ | (c) 质量 | **口径一致**：clean bpc（窗口 16 掩码）与 2638/2639 记录值逐位复现 | `win16_s0=1.628110`、`carry16_s0=1.624979`、`carrypl16_s0=1.527595`、`carrypl16d_s0=1.585531`，与本次读数在 6 位小数上相等；不等即报差异并 FAIL | 必须不变 |
| ⑥ | (e) 因果 | **必须不变**：只改后半段 token ⇒ 前半段 logits 逐位不变 | 逐臂 `max_abs_logit_diff_prefix == 0.0`，且探针有抓力（后半段 logits 最大差 > 0）；窗口掩码非退化断言通过 | 必须不变 |
| ⑦ | (d) 保留 | 诊断（不作判据）：通道消融 + KV（前缀/最近）损伤曲线 | 报逐臂 `bpc_win16_channel_off` 与 `prefix/recent` 曲线的 `damage_bpc` + `safe_radius`；**必须写明口径限制**（见 §4a） | 诊断 |
| ⑧ | — | **禁外推** | 结论只适用于 enwik8 / T=256 / 13M / 12k 步 / 本可见性快照 | — |

### 4a. 操作性定义（不改判据，只把口径落定）

- **臂角色**（在 `results.json.criteria.arms` 固化）：
  `implicit = {win16_s0, win16_s1}`；`explicit_complete = {carrypl16_s0, carrypl16d_s0}`；
  `explicit_degraded = {carry16_s0, carry16_s1}`。主对撞 = `carrypl16_s0`（explicit 完备）
  vs `win16_s0`（implicit）。
- **clean bpc（腿 c）**：`realtext_probe.win16_bpc(model, eval_arr, window_anchor_keep_mask(256,16,0),
  n_batches=96, batch_size=16, window_seed=2)`——与 2634 **同一函数、同一掩码构造、同一窗口序**，
  token 数 ≈ 96×16×255 = 391,680。与 2638/2639 记录值逐位比对（判据⑤）。
- **`Δbpc_far`（腿 a，主仪器）**：`realtext_probe.far_source_probe`，kwargs 一律由
  `realtext_probe.probe_kwargs(cfg)` 派生（`rf = n_layers·(train_window−1) = 180` **必填**，
  由模型规格派生并与 config 交叉校验，**不许手填**）；布局 = 被毁前缀 `[0,64)`、被评行
  `[244,255)`，满足不变式 `eval_lo − rf ≥ corrupt_hi`。逐臂报 `delta_bpc_far` + 逐簇配对
  bootstrap CI95 + `max_abs_logit_diff_on_eval_rows`；carry 臂并列报 `channel_off` 后读数。
- **needle 腿（腿 b）**：`retention.phrase_needle_batch`（needle v3；`n_needles=8, phrase_len=6,
  seed=3`，64 批 × 16 行 = 8192 根针/臂/条件）+ `needle_recall(..., anchors=0)`；桶只有
  `window` / `beyond`（锚点 0）。**地板 = 同一生成器序列 `plant=False` 的配对控制臂**。
  统计层 = `needle_stats`：**同索引硬子集**（`hard_index_sets` 的 `intersection`/`union`，
  索引集在六臂之间固定成同一组）+ 方向固化的 `beyond_above_floor`（`two_proportion(control,
  planted)` ⇒ diff = 植入 − 地板）+ Wilson CI。
- **σ_cell（本格自测带，禁套全局 0.014）**：本格同配置换种子（`win16_s0/s1`、`carry16_s0/s1`）
  在 `bpc_win16` 上的**极差的最大值**（2 样本极差，描述量、不是方差上界）。
- **跨格参考带**：0.014 bpc（`docs/methodology.md` 的 13M 跨 run 漂移界），**限配对 Δ、≥5 窗口**。
  两带**并列报、不得混用**；召回/接受率类**只**走配对 bootstrap / Wilson 95% CI，**不跨量纲**
  借用 0.014。
- **腿 (d) 的口径限制（必须写进判据栏）**：显式状态是**固定大小、结构上不可截断**的——
  `prefix_keep_mask` / `recent_keep_mask` 只删**注意力 KV**，`carrypl16` 臂每层的状态仍完整
  携带前缀信息。所以「丢前缀」对通道臂只作用于注意力侧，**不是**「显式状态也被截断」；
  这是读数解释的一部分，**不是缺陷**。口径：`damage_bpc = bpc_掩码 − bpc_clean`。
- **`safe_radius`**：`retention.safe_radius(curve, band, key, delta)`，`delta_bpc = −damage_bpc`
  （正 = 改善）；两带并列报（`σ_cell` 与 `0.014`）。
- **通道消融抓力**：`realtext_probe.channel_off`（置零每层 `carry_scale` + 共享 `gate_w`）后重测
  `bpc_win16`。若 E1 逐层模式下覆盖不全（未真的把每层读出增益置零），在**本目录的副本**里补齐
  并在 `results.json.liveness[*].channel_off_grip` 如实登记。

## 5. 成本与预算

`gpu_wall_budget: 35 min`（纯推理；含冒烟 + 全电池 + 保留曲线）。到点先落盘已得读数再返回。
**判据只用等效理论算力**（本实验为纯推理、无训练；代价轴见 `2639` 的每位置等效字节）；
**墙钟只作过程指标**写进 `run_log.txt` / 报告，**绝不进判据或 `cost_model`**。

## 6. 数据与脚本清单

- 数据：`data/enwik8_full.npz`（前 90M 训练 / 末 10M 评测；vocab 206 含 `[mask]`），
  入口 `pathlm.data.load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")`。
- 评测：`python3 run.py`（五条腿）/ `python3 run.py --smoke`（少批冒烟）。
- 仪器副本：`realtext_probe.py`、`needle_stats.py`（自 `2634` 复制，**不改**）。
- 落盘：`results.json`（严格 JSON）、`needle_raw.json`、`run_log.jsonl`。
- 权重 `*.pt` **不入库**（在 `2634`/`2639` 目录内，`MODEL.md` 记 sha256）。

## 7. scope 声明

只做单轴单格实验；基底恒为 stock decoder-only；**组合默认留白**。本实验唯一「组合」=
同一机制的正面对撞读数（可见性/预算/数据/快照全同），不构成新的结构组合。`deps` 只记录耦合方向。

## 8. 产出与复核

见 §6；完成后另派 fresh-context 只读复核者（**非本 WP writer**），复核排在全部产物落盘之后
（`SCHEMA.md` §14/§14b），累计 ≤2 轮。
