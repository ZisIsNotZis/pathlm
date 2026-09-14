# 12 — 通宵自主研究:PathLM 的"终极形态"

- **Status:** claimed(用户授权全自主,次日回收)
- **Type:** research batch + engine work + report
- **Related:** 11-retry-matrix-validity(本 ticket 承接其遗留)

## ⚠️ 续命契约(新会话/auto-compact 后先读这里)

**编排规则(用户明令):**
- 长任务一律 `setsid nohup … &` 脱离会话;**禁用** harness 的 bg_run 包装(会被
  会话重载杀掉,已发生一次)。**不依赖**定时任务唤醒;用 `sleep 300-900` + 轮询日志。
- GPU 上限 3 并发(~5.75GB/run,23.5GB 卡);超过 OOM。
- 每个 checkpoint 落盘(`--ckpt-every`),checkpoint 才是持久产物。
- **每完成一步就更新本文件 + findings.md**(SSOT,防 compact 丢失)。

**收益判定规则(用户定的通用法则):**
1. 有明显 bpc 效应的机制 → 先报 bpc。
2. 否则 → 解码速度(tok/s)。
3. 都没有 → 想清楚真实有效性是什么;真没有 → 丢 backlog。
4. 每条都要有"收益类型 + 收益值(相对 B0=0)+ 税(纯推理)"三列;只有税没有收益
   的条目无意义。

**测量标准:** `probe_e2e.py`(per checkpoint,per 深度:bpc_round_n / repair_rn /
clean_bpc / 吞吐)。税率只作对照列。种子方差 0.0001–0.0138/格,<0.014 bpc 不可排序;
故关键结论要 2 种子。绝对税是 1.1-epoch 快照(B0 1.5074@1.1ep → 1.3459@4.4ep),
报渐近值用 24000 步。

## 决策记录(用户授权我拍板)

- **D6 已决策:实现真投影。** linear transport 改为正交投影 `P = pinv(U) @ U`
  (谱范数 1,缓存的 d×d 矩阵),替换掉放大约 72× 的 `EᵀE`。理由:design §3 白纸黑字
  写"projection onto the vocab subspace",现实现是客观错误;真投影便宜且可测
  (谱范数=1 可直接断言)。**soft transport 不动。** C2/C2M 旧数字作废。
- **条件链已决策:实现 token 条件版**(design §3 说的"经 embedding 重入"),并按
  用户规则**先报 bpc**(node-2 条件版 vs 直连版的 t+2 bpc),同时报 spec-decode
  接受率(速度侧)。训练用 teacher forcing(真 t_{i+1}),探针同时测 oracle 条件与
  argmax 条件(部署版)——不报 oracle 单边,那是虚高。
- **受益域先于测量**(用户的元批评):每个机制先写清"它对什么输入/部署才有价值",
  再在该域测,干净 enwik8 只当税列。wrong-token 的独占价值 = 可进 KV/可验证
  (spec-decode 可用),[mask] 不行 —— 这条要进 findings.md。

## 阶段与状态

### P0 引擎(条件链 + D6 真投影)+ 测试 — [✅ 完成, commit 1fc943d]
- [x] `chain_mode: "latent"|"token"`;token 版重嵌入 t_{i+1} 再进 MTP
- [x] `_vocab_projector`(缓存,谱范数断言测试)
- [x] 单测;提交(40 绿)

### P1 条件链 A/B(bpc 优先)— [进行中]
- **v1 被自己的 A/B 立即证伪(用户"先想清有效性"规则的直接收益):**
  第一版把整个状态换成裸 token embedding(embed(t_{i+1})+pos → T1),上下文全丢,
  6000 步时 oracle 0.24 ≪ direct 0.56 —— 从单 token 预测 t+2 当然输给有 12 层
  上下文的 direct 头。
- **v2(DeepSeek-MTP 形态):** concat+投影 —— `chain_cat = Linear(2d,d)`,输入 =
  **主模型上下文 latent h**(不是 transform 头的输出)拼上 `embed(t_{i+1})`,投影回 d
  再过 T1。concat+proj 学到 29×/1× 的范数差;保留上下文 + 注入 token。
  冒烟验证: 600 步时 oracle 0.4011 > direct 0.2811 ✓(v1 在同点位是 0.19 < 0.23)
- 判据不变: deploy 版(argmax 条件)才是真话;oracle 只作上界;另报 spec-decode 接受率
- 运行: CH_token_s0/s1 × 6000 步(nohup, .tmp/p1b);latent 对照复用 CH_latent_s0/s1
  (latent 路径两次实现位相同,已由 M0 offset 测试钉住 T1@T1)
- 证据: `.tmp/curves3/`(收尾后拷入 evidence/)

**P1 裁决(6000 步 × 2 种子, 高度可重复):**

| t+2 准确率 | s0 | s1 |
|---|---|---|
| direct 头(无条件) | 0.5608 | 0.5601 |
| chain **oracle**(真 t+1 条件) | **0.6819** | **0.6806** |
| chain **deploy**(node-1 argmax 条件) | 0.5473 | 0.5444 |
| spec 接受率 chain-deploy | 0.7525 | 0.7506 |
| (参照) direct 接受率 (B2) | 0.7729 | — |
| (参照) node-1 t+1 准确率 | 0.6930 | 0.6919 |

- **机制本身成立**: 给定真 t+1, t+2 预测质量(0.682)几乎追平 node-1 对 t+1 的质量
  (0.693) —— 信息在, concat+proj 能用上。v2 修复了 v1 的“无上下文”错误。
- **部署版不赚**: argmax 条件反而比 direct 差 1.3pp; 草稿错误(~31% 时 node-1 错)
  注入的噪声吃掉了全部条件收益。oracle-deploy 差 0.135 是“草稿质量”的明确标价。
- **速度侧也不赚**: 接受率 0.75 < direct 0.77。
- 处置(用户规则): 部署无 bpc/速度收益 → **不进 INT2**;但 oracle 结果给出一个强组合
  假设: prob0 门控重试如果改善了草稿 t+1, 条件链就会随之变赚 —— 记入报告的组合问题。
  INT2 用 n_mtp=2 + latent 链(node-2 direct 头在), 保留 0.77 接受率的草稿能力。

### P2 修复引擎 retry,E2E + D1 因果确认 — [运行中]
- 4 run × 12000 步(爆炸在 9000 步已 8.5 bpc,12000 步够分辨): C1×2种子(对照),
  C1M×2种子(混合), ckpt-every 1500, nohup
- 判据: 修复引擎下 C1M 的 r2 是否不再随训练爆炸(D1 因果证据);
  E2E 台账(税 +1.98 bpc 收益是否复现)
- 另: C2/C2M(6000步) 重测 = D6 修复的行为确认(投影不再放大)

### P3 集成 INT′(终极形态)- [配置已暂存 .tmp/INT2.json]
- 全部优势元素 ON,**去掉 shuffle/redo**: mask+wrong 损坏、MTP(条件链,P1 结果决定
  token/latent)、早退、retry(mixture+几何深度)、驱逐+锚+needle、距离惩罚、多样性
- 24000 步 × 1-2 种子,对照 B0long(1.3459@4.4ep)
- **门控条件**: P1 的 deploy 版 ≥ direct×0.9 才用 token 链,否则退回 latent

### P4 报告 — [未开始]
- `docs/report.md`:自解释、完整;每机制 = 受益域 | 收益类型 | 收益值 | 税 | 证据
- findings.md 重构成账本格式(解决 250 行超预算)
- WORKSPACE.md 状态更新

### P5 (时间富余) 规模探针 — [未开始]
- 2× 参数(d=362)6000 步,验"税随规模"的跨机制定律

## 运行登记(追加式)

| run | 配置 | 状态 | 结果 |
|---|---|---|---|
| (待填) | | | |

## Comments

- 2026-09-14 — agent (pi) — 用户授权通宵自主;规则见上。先写本契约再动手。
