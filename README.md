# PathLM — 概率驱动算力分配的弹性语言模型（已收口）

> 一句话：本项目从一个想法出发——"让模型按位置置信度分配每 token 的算力，
> 用回环/权重复用换解码吞吐"——经过 25 个执行 ticket、两个规模（13M/100M）、
> 两代架构的系统测量，最终**证伪了"等智商免费加速"**：一切侵入正常推理
> 路径的设计都带来结构性性能税；主干保持纯粹，速度只能走成熟旁路。
> 完整轨迹见 **`docs/retrospective.md`**。

## 探索过什么

- **回环时代（tickets 01–22）**：冻结架构（stage ring + 层路径采样）上系统探索
  弹性语法——损坏训练、latent/token 重试、层 shuffle/skip/redo、早退、驱逐+锚、
  MTP/spec 解码、TTS 集成、多样性压力；建成端到端分配器（Slider：校准→成本模型→
  求解器→验证解码→在线代理）并在 100M 上规模复测。
- **深度自适应 AR 时代（tickets 23–25）**：目标重聚焦（等参数、等智商、更快解码）
  后的新形态——纯 AR + 全深度并行监督 + 逐通道携带门 + 随机退出深度 + 自 spec；
  在 13M 与 100M 上完成机制验证、形态税归因与收口判决。

## 发现了什么

**成立的**：

- **分配选择性**：按 prob0 只对最低置信 15% 位置分配额外算力，把重试从净有害
  翻成净有益（+0.021 vs −0.019；C1M 2.2 倍收益）——项目原始命题获证。
- **prob0 校准跨规模存活**（ECE 0.001–0.011）——全程可审计的控制信号；
  三信号生成监测栈（prob0 + 路径分歧 js + distinct-2 重复率，AUC 0.9815）。
- **成熟旁路有效**：MTP + 批量验证 spec（forwards 口径 1.68–1.90×，税 +0.017）；
  probe 式早退（100M 上零税、门控更强）。
- **能力溢价**：损坏输入下机制模型比基线好 1.5–1.9 bpc，差距随规模与训练扩大
  （1.54→1.89 bpc）；基线脆性不随训练改善。

**证伪的（同样重要，各有对照实验）**：

- **等智商免费加速（最终判决）**：深度自适应 AR 形态税 +0.44 bpc @100M
  （96% 干净归属），对"结构切分"（反向恶化 −50.4%）与"2× 训练+冷却退火"
  （形态特异收回 ≈0）两个干预都分文不收回——**税结构性存在，只能被交易**
  （+0.44 bpc 换 exit-4 3.1× 墙钟与出口资产 tf@6 0.9946）。
- **回环换速度**：retry 作为解码加速三次失败（价值实为损坏修复）；门控重试质量
  杠杆随规模自衰减（100M 消失）；spec 墙钟收益在 compute-bound 下消失（k=1 转负）。
- **目标竞争**：损坏训练 × 精确复制不可兼得（anchor 召回 0.764→0.016–0.035，
  三路豁免全败）；dense-exit 整形 × retry 不可两全（同一梯度两面，剂量响应随
  规模恶化为 corrupt 崩溃）。
- **元素税定律**：破坏 identity/order 的机制贵（shuffle +0.545 组合毒药、skip
  +0.175）；保留或 flag identity 的近乎免费（mask/noise/redo ≤0.06、距离惩罚
  −0.007）。

**方法学教训**：排序前先看方差（Δ<0.014 不可排序，曾据此撤回整个 retry 矩阵
排序）；绝对税必须标注口径（1.1-epoch 上界 vs 4.4-epoch 渐近 vs matched-step）；
成本要双货币记账（forwards vs FLOPs 结论会分歧）；对照必须同引擎同配方（否则
要像 24b 那样花三臂拆混杂）；n=1 负结果必须复现后再入账本。

## 交付了什么资产

- **机制库**：`pathlm/`（引擎、深度自适应 AR、Slider、生成质量/塌缩监测）+
  根目录探针 + configs；129 个可失败单测钉死语义。
- **测量方法学**：三列账本（税/收益/各自货币）、受益域先行、方差界、口径标注、
  双货币、冒烟+变异测试——全部沉淀在 docs 与 ticket 契约中。
- **证据链**：25 个 ticket 的 evidence 目录（results.json/曲线/探针输出/SUMMARY），
  索引见 `docs/artifacts.md`；每个结论可回溯到 run 与 commit。

## 文档导航

| 想读什么 | 去哪 |
|---|---|
| 全轨迹叙事（起点→引擎迭代→两次转向→机制验证→规模判决→归因收口→教训） | `docs/retrospective.md` |
| 逐机制结论账本（SSOT，每条带数字与证据路径） | `docs/findings.md` |
| 实验报告（结论层）与逐实验审计层 | `docs/report.md` / `docs/report_details.md` |
| 设计哲学演化（回环时代 → 深度自适应 AR） | `docs/mental_model.md` §0 |
| 冻结架构（回环时代，历史文档） | `docs/design.md` |
| run 级证据索引（25 个 ticket） | `docs/artifacts.md` |
| 逐 ticket 执行契约与判据 | `.scratch/NN-slug/issues/` |

## 复现

```bash
python3 -m pytest tests/ -q                  # 129 绿
python3 train_m1.py MYRUN --config configs/B0.json            # 13M 回环基线
python3 train_depth_ar.py MYRUN --d 256 --layers 8 ...        # 深度自适应 AR（CLI 旋钮，无 config）
python3 slider.py --config configs/INT2.json --ckpt <ckpt> --out out.json
```

数据 enwik8 自动下载；所有 results.json 携带完整 resolved config；权重 `*.pt`
不入库（gitignored，派生物——证据 json/log/md 入库）。单卡 RTX 4090 24GB 包络。

## 状态

**项目已收口（2026-09-29）**：主结论已定（见 retrospective），无进行中实验；
backlog（自由生成段专用监测深化、4.4-epoch 渐近配方重验等）记录于
WORKSPACE.md 与各 ticket issues。
