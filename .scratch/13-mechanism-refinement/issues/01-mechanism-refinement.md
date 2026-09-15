# 13 — 机制完善批次:门控/双通道/链扩展/文档

- **Status:** claimed(用户批准按序执行,fork=false 子代理)
- **Type:** engine work + measurement batch
- **Related:** 12-overnight(承接其 backlog)

## 执行顺序(用户批准)

1. **② prob0 逐位置门控** — 开火率 vs 修复保留率扫描(eval-only)+ 1 个训练 run 验证
   - 依据: self_acc 0.92 → 开火 ~10-15% 位置,计算 1.10-1.15×(现在 2-5×)
   - 实现: `gate: bool`;训练 τ~U(0.05,0.95) 覆盖,推理 `gate_tau`;状态推进门控、
     累加器照旧;`aux["fire_rate"]` 记录
   - **判据: 15% 开火保留 ≥80% 修复收益 → 采纳**
2. **③ 锚区豁免损坏(X2C-v2)** — needle×损坏 拮抗的修复
   - 依据: 隔离实验 X2C 已证损坏训练毁复制任务(84.9%→0.5%)
   - 实现: `corrupt_spare_anchors: bool`;损坏不 touching 前 anchors 个位置
     (与驱逐设计"锚=可靠通道"同构)
   - **判据: 锚区召回 ≥50% 且干净 bpc 税不变**
3. **① n_mtp=3 + 批量验证** — 先测接受级联再决定是否建解码路径
   - 依据: k=1 草稿无吞吐收益(验证轮=下一生成轮);吞吐需批量验证
   - 收益公式: tokens/forward ≈ 1 + a2 + a2·a3(a2=0.773 实测,a3 待测)
   - **判据: 级联 ≥2.2 tokens/forward → 实现批量验证解码并实测 tok/s;
     训练税增量须 <+0.05**
   - 附:条件链救场假设已被算术否决(deploy 差距 0.135 vs 重试可给的 +0.003),
     条件链留 backlog
4. **④ 文档重构** — findings.md 274→≤200 行(账本化),.git 权重问题只记录不执行

## 状态

- [ ] P0 引擎位(gate / corrupt_spare_anchors)+ 测试
- [ ] ② 门控扫描 + 训练 run
- [ ] ③ X2C-v2
- [ ] ① CH3 训练 + 级联测量 +(判据通过时)批量验证
- [ ] ④ 文档重构(subagent, fork=false)

## 运行登记

| run | 配置 | 状态 | 结果 |
|---|---|---|---|
| (待填) | | | |
