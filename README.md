# PathLM — 概率驱动碎片化算力分配的弹性语言模型

> 一句话：常规 LLM 推理是"每 token 固定算力"的一个**点**；PathLM 的目标是
> 一条**连续、可审计、可调的质量-算力前沿曲线**——用模型自己的已校准概率
> 决定每个 token 花多少算力。这条曲线穿过常规点，向两侧延伸（更便宜-略差 /
> 更贵-更好）。

状态：13M 与 100M 两个规模上的机制验证与缩放研究已完成（验证阶梯 Rung 1–4
全部定案）。这不是一篇论文，是一个进行中的研究仓库——证据都在
`.scratch/*/evidence/`，结论都有数字和对照。

---

## 哲学与理想

**核心命题：算力应该按位置置信分配。** 低置信位置每 FLOP 的收益更高
（已证，见下）；均匀分配是浪费。模型不缺能力，缺的是"知道自己在哪
不确定，并把算力花在那里"的机制。

三个设计支柱（全部实测成立）：

1. **prob0 校准**：模型对"当前 token 是否正确"的自我估计是校准的
   （ECE 0.001–0.008），损坏检测、重试门控、在线质量监测共用这一个信号。
2. **retry 是修复不是集成**：一次带传输的额外前向能在损坏/困难位置精炼
   状态（corrupt 输入 +1.8 bpc 量级），但干净流上平均收益为零——所以
   必须门控。
3. **能力溢价而非低税**：这套机制不卖"更低的困惑度"，卖的是鲁棒性、
   可控性、可审计性；税（clean bpc 溢价）随规模与训练下降，能力溢价
   随两者扩大。

## 机制账本（6 存活 + 一堆证否）

**采纳的机制**（税 = clean bpc 相对 B0 的溢价；收益各自计价）：

| 机制 | 税 | 收益 | 状态 |
|---|---|---|---|
| MTP + prob0（多步预测头） | +0.04（n=2） | 全部弹性机制的控制信号；spec 草稿 | ✅ |
| 门控重试（prob0 门控 + soft 传输） | +0.12~0.15 | corrupt 输入 +1.8 bpc；损坏修复 52→55% | ✅ |
| 输入损坏训练（mask+wrong） | +0.19 | corrupt 输入下比 B0 好 1.5–1.9 bpc | ✅ |
| 驱逐+锚（有界内存） | +0.06 | ring-buffer 解码免重 prefill；锚通道 93% | ✅ |
| 距离惩罚 | **−0.007** | 免费正则；锚通道 76.8%→93.3% | ✅ |
| 早退（probe 式深度头） | ≈0 | 便宜侧前沿点（flop<1.0） | ✅（见下） |

**已证伪/不采纳**（同样重要——每条都有对照实验）：

- **shuffle（层序自由）**：税 +0.545，组合毒药（任何组合落到 2.2+ bpc）。
  层序携带 ~0.5 bpc 信息，死路。
- **redo（层重复）**：税 +0.044，收益 ≈0。
- **朴素概率复合**：数学错误（CE 3.96 vs 1.58）。
- **条件链（token 条件 MTP）部署版**：草稿质量瓶颈（deploy 0.547 < direct 0.560）。
- **逐位置 latent 状态掩码**：direct 重入是固定点，门控无可切换；需稀疏计算。
- **needle×损坏共存**：三路豁免全败；用户裁决——目标恒为正确 token，
  无需模式信号。
- **dense-exit 与 retry 共训**：整形与 retry 是同一梯度的两面，两全证伪
  （见缩放定律）。

## 三杠杆与 Slider

运行时旋钮 θ = (τ_retry, τ_exit, k)，沿前沿移动；规模参数（模型大小、
n_mtp、深度）定前沿形状。

- **retry（花钱）**：prob0 门控只开低置信 15% 位置 → 重试从净有害翻成
  净有益（INT2 −0.019→+0.021；C1M +0.057→+0.128）。
- **spec（摊销）**：Medusa 式批量验证，2.40 tok/forward（13M）。**注意：
  墙钟收益是 launch-bound 小模型现象**——100M compute-bound 下宽度线性
  计价，k=2 只剩 1.15–1.29×，k=1 转负（ticket 20）。
- **exit（省钱）**：双 profile 定案（ticket 18/21）——
  - **自适应 profile**（probe 式深度头，trunk stop-grad）：retry 门控活
    （100M 上门控 −0.029，比 13M 更强），clean 税≈0，但浅层读出贵；
  - **质量 profile**（dense 整形）：浅层读出好（half-stack +0.14~0.20）、
    clean 更好（aux 正则），但 retry 死（13M 温和负、100M 剂量响应恶化
    为 corrupt 崩溃）。
  - **整形 vs 自适应是同一梯度的两面**——跨规模成立，以 profile 选择。

## 关键缩放结论（100M，2 seeds，13M 交叉验证）

1. **税随规模与训练下降**：1.1-epoch 快照 +0.126（13M +0.147 → 100M
   "不增"）；4.4-epoch 渐近 **+0.082**，等训练口径 13M +0.122 → 100M
   +0.082——快照口径的"不缩"全是 1.1-epoch 上界。
2. **能力溢价随规模与训练扩大**：corrupt 输入下 B0 崩至 3.55–3.71（脆性
   不随训练改善），机制模型 2.02→1.82——E2E 差距 1.54→1.89。
3. **prob0 校准跨规模存活**（ECE ≤0.0096）；a₂ 部署口径 0.74–0.86 稳定。
4. **门控重试质量杠杆随规模自衰减**（13M −0.004 → 100M 无 dense 时 ≈0、
   probe 头下 −0.029）——机制价值转向训练侧与监测侧。
5. 396M tokens（enwik8 4.4 epoch）下 100M ≈ 13M 渐近 bpc（1.341 vs 1.346）
   ——**data-limited**，参数增益要靠更多数据兑现。

## 诚实的负结果清单（省得重新踩坑）

- 生成段的 mean-prob0 **不是**输出质量检测器（13M 上反向——损坏把生成推进
  高置信重复吸引子；100M 恢复正向，属容量现象）。
- fp 平局翻转：100M 起 batched 与逐位置 SDPA 在 top-2 gap ~0.008 处翻
  argmax（1/400 token）——"spec==greedy 逐位相等"是 13M 规模性质。
- 单轮重试 mixture==overwrite（设计保证，共享 gauge）——引擎无损失，
  但也别指望 mixture 在单轮带来增益。
- 预算要**双货币记账**（forwards vs FLOP-normalized）：spec 在 forwards
  口径赢、flop 口径平/亏，混用会得出错误结论。

## 仓库结构

```
pathlm/            # 库：config / data / model / decode / eval / metrics / slider
train_m0.py        # M0 微型模型训练（机制验证）
train_m1.py        # M1 主训练器（config-driven，results.json 全量落盘）
slider.py          # Slider 端到端探针（校准→前沿→求解→验证→代理）
probe_*.py         # 机制探针：e2e / spec / exit / frontier / selectivity …
configs/           # 全部 run 配置（B0/C1/INT/EX/asym …）
tests/             # 74 个可失败测试（含 fp/门控/成本模型性质）
docs/              # design（冻结）/ mental_model（实用思路）/ findings（账本，SSOT）
                   # / report（结论层）/ report_details（审计层）/ experiments
.scratch/NN-slug/  # 每 ticket 一目录：issues + evidence（results/曲线/探针输出）
```

## 复现

```bash
python3 -m pytest tests/ -q                 # 74 绿
python3 train_m1.py MYRUN --config configs/B0.json          # 13M 基线（~10 min @4090）
python3 slider.py --config configs/INT2.json --ckpt <ckpt> --out out.json
python3 probe_e2e.py --config configs/C1.json --ckpt-dir <dir> --out out.jsonl
```

数据：enwik8（自动下载至 `.tmp/enwik8`，缓存 `data/enwik8_full.npz`）。
所有 results.json 携带完整 resolved config；权重不入库（*.pt gitignored，
派生物）。单卡 RTX 4090 24GB 即可（13M 全部实验 + 100M 训练 batch 8 实测）。

## 文档地图（想深入按此顺序）

1. `docs/mental_model.md` —— 实用思路模型（哲学/三杠杆/Slider/验证阶梯/缩放定律）
2. `docs/findings.md` —— 逐机制结论账本（SSOT，≤200 行，每条带证据路径）
3. `docs/report.md` —— 研究报告（结论层）；细节在 `report_details.md`
4. `docs/design.md` —— 冻结架构（stage ring、cycles、聚合原则、训练配方）
5. `.scratch/NN-*/issues/` —— 21 个 ticket 的执行契约与判据

## Backlog（无阻塞，按优先级）

1. 自由生成段的质量信号（prob0 在生成段 13M 失效/100M 恢复，专用信号未建）
2. 13M 全账本渐近重校（把历史税数字统一到 4.4-epoch 口径）
3. 更大数据下的 100M+（当前 data-limited，参数增益待兑现）
