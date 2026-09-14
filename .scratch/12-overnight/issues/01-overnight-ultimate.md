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

### P0 引擎(条件链 + D6 真投影)+ 测试 — [进行中]
- [ ] `chain_mode: "latent"|"token"`;token 版重嵌入 t_{i+1} 再进 MTP
- [ ] `_vocab_projector`(缓存,谱范数断言测试)
- [ ] 单测;提交

### P1 条件链 A/B(bpc 优先)— [未开始]
- n_mtp=2,chain_mode latent vs token,6000 步 × 2 种子 = 4 run(~15min 并发)
- 判据:node-2 条件版 t+2 bpc/acc 是否胜过直连版 0.5597/1.5754;部署版(argmax 条件)
  与 oracle 版的差距;spec-decode 接受率变化
- 证据:`.scratch/12-overnight/evidence/`

### P2 修复引擎 retry,E2E + D1 因果确认 — [未开始]
- C1 / C1M_n1 × 2 种子,12000 步(爆炸在 9000 步已到 8.5 bpc,够用),ckpt-every 1500
- 判据:修复引擎下 C1M 的 r2 是否不再随训练爆炸(D1 因果证据);
  E2E 台账(税 +1.98 bpc 收益是否复现)
- 若时间够:补 24000 步一对

### P3 集成 INT′(终极形态)- [未开始]
- 全部优势元素 ON,**去掉 shuffle**(它是组合毒药,旧 INT 失败可能全因它):
  mask+wrong 损坏、MTP(条件链)、早退、retry(mixture+prob0 门控)、
  驱逐+锚、距离惩罚、多样性压力;24000 步 × 1-2 种子
- 对照:B0long 24000(已有,1.3459)
- 判据:INT′ bpc vs B0 + E2E 电池 + 解码速度。旧结论"INT 不组合"含 shuffle,
  INT′ 是真未测问题
- 失败也要如实记录:组合极限本身就是结论

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
