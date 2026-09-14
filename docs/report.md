# PathLM — 终极形态研究报告(草稿,持续填充)

> 状态:夜间自主研究(ticket 12)进行中。本文件是最终交付物;每填一个结果就更新
> 一次。所有结论带证据路径。收益规则(用户定的):bpc 优先 → 解码速度 →
> 能力型(无法用 bpc/速度表达的部署收益);只有税没有收益的条目无意义。

## 0. 阅读地图

- 架构(冻结): `docs/design.md`
- 测量程序: `docs/experiments.md`
- 逐机制结论(SSOT): `docs/findings.md`(ticket 11 修正版)
- 裁决树 + 计划: `.scratch/11-retry-matrix-validity/VERDICTS.md`
- 本夜状态: `.scratch/12-overnight/issues/01-overnight-ultimate.md`
- 证据: `.scratch/12-overnight/evidence/`

## 1. 一句话结论

(待最终数据填)

## 2. 测量方法(先于一切的原则)

1. **受益域先于测量**:每个机制先写清"对什么输入/部署才有价值",再在该域测;
   干净 enwik8 只当税列。
2. **三列账本**:收益类型 | 收益值(相对 B0=0)| 税(纯推理)。B0 是 1.1 epoch 快照
   (1.5074 bpc),渐近值用 24000 步(4.4 epoch)报。
3. **两套方差**:同种子 nondeterminism 0.0019 bpc;种子方差 0.0001–0.0138/格。
   结论 <0.014 bpc 不可排序 → 关键结论 2 种子。
4. **E2E 优先于 tax**:`probe_e2e.py` 每 checkpoint 报每深度 bpc/repair/吞吐;
   retry 的税 +0.12 与损坏输入上的 +1.98 bpc 收益对比是范式案例。
5. **诚实护栏**:oracle(teacher forcing)只作上界,部署版(argmax 条件)才是真话;
   n=1 标注;失败如实记录。

## 3. 逐机制账本(最终数字落地中)

| 机制 | 受益域 | 收益类型 | 收益值 vs B0 | 税(推理) | 证据 |
|---|---|---|---|---|---|
| mask 损坏 | 低质量/部分损坏输入 | 能力型(可修复) | 修复 59.4% | +0.059 | findings §I |
| wrong-token 损坏 | spec-decode/未知损坏位置 | 能力型(可进 KV 可验证) | 修复 52.8% | +0.127 | findings §I |
| 噪声/pure-noise | 连续损坏 | 能力型 | (范数吸收) | ~0/+0.063 | findings §I |
| MTP+prob0 | 一切弹性机制的控制 | 能力型(校准信号) | ECE 0.002-0.011 | ~0 | findings §MTP |
| **条件链(新)** | 预测 t+2 | **bpc** | oracle 0.682≈node1 0.693; **deploy 0.547 ≈
  direct 0.560**;接受率 0.75<direct 0.77 | ~0 | P1(双种子,证据 e2e/CH_*) |
| 早退 | 低延迟部署 | 速度型 | 1.4× tok/s | +0.067 | findings §Early |
| retry(修复引擎) | 损坏输入 | **E2E bpc(待 P2)** | 2 轮 +1.98(P2 复现中) | +0.12 | P2 |
| skip | 测试时扩展 | 速度×质量斜率 | K=8 −0.047(DIVL1 −0.062) | +0.175 | findings §Skip |
| 驱逐+锚 | 长上下文/有界内存 | 能力型(不重 prefill) | needle 76.8→93.3 | +0.058 | findings §X2 |
| 距离惩罚 | 长程注意 | 免费正则 | needle 通道 93.3 | **−0.007** | findings §X1 |
| 多样性压力 | TTS 扩展 | 斜率 | −0.062(DIVL1) | +0.023 | findings §TTS |

## 3.5 条件链裁决(P1,双种子,高度可重复)

**机制成立,部署不赚。** 给定真 t+1,t+2 准确率 0.682 几乎追平 node-1 对 t+1 的
0.693——信息在,DeepSeek 式 concat+proj 能利用。但部署(argmax 条件)反而比无条件
的 direct 头差 1.3pp:node-1 约 31% 的草稿错误注入的噪声吃掉了全部条件收益。
接受率 0.75 ≈ direct 0.77,速度侧也不赚。oracle-deploy 差 0.135 是"草稿质量"的标价。

**工程课(同一条判决的一部分):** 第一版把状态整个换成裸 embedding,丢掉上下文,
A/B 立刻证伪(oracle 0.24 ≪ direct 0.56);第二版 concat+proj 修复(主模型 h 拼
embed(t),投影回 d)。这验证用户规则——先想清有效性再跑,A/B 当场指出错误。

**对 INT2 的影响:** 不用 token 链,用 n_mtp=2(node-2 direct 头,接受率 0.77,供
spec-decode 草稿)。注意: 单令牌草稿(k=1)对贪心解码吞吐无增益(验证轮就是下一轮的
生成轮);吞吐增益需要链扩展到 node-3+(草稿 2 令牌,验证一批),那是设计里未建的
链路径。

## 4. 集成(INT′,终极形态)— P3

- 配置 `.tmp/INT2.json`:mask+wrong、MTP+条件链、早退、retry(mixture+几何)、
  驱逐+锚+needle、距离惩罚、多样性;**无 shuffle/redo**(组合毒药/死路)。
- 24000 步 × 1-2 种子,对照 B0long 1.3459。
- 判定:INT′ bpc vs B0;每元素的税是否次可加(组合是否成功);
  E2E 电池(修复率、解码速度)。失败也如实记录。

## 6. 尚存问题 / 已验证边界

(待填)
