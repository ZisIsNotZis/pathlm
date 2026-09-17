# Rung 4 — 规模研究（ticket 16）证据

日期：2026-09-17。配置：**d=720 / 12L / 12 heads / mlp_mult 6 / seq 512 = 102.3M 参数**
（含 mask 槽），24000 步 × batch 8（token 预算 = 6000×32 = 98.3M tokens ≈ 1.1 epoch，
与 13M/29M 协议一致；batch 32/16 在 100M 下 OOM——eviction mask 使 SDPA 走 math
后端 + 多轮图保留——全体同步降 batch，**batch 跨尺度不同是混杂，已记录**）。
resolved config 全量落盘在各 run results.json（避免 29M 探针配置不可复现的教训）。

| run | clean bpc | next-tok acc | ECE | wall |
|---|---|---|---|---|
| B0_100M_s0 | 1.4663 | 0.7009 | 0.0091 | 38.9 min |
| B0_100M_s1 | 1.4560 | 0.7028 | 0.0077 | 34.3 min |
| C1_100M_s0 | 1.5844 | 0.6797 | 0.0037 | 52.3 min |
| C1_100M_s1 | 1.5897 | 0.6791 | 0.0054 | 55.2 min |

## 判读

1. **retry 税 @100M = +0.126**（种子配对 s0 +0.118 / s1 +0.134；B0 种子散布
   0.0103，C1 0.0053）。序列：13M +0.147 → 29M +0.174（单种子）→ **100M +0.126
   （2 seeds）——税首次不再随规模增长（若有所下降）**。"税随规模缩小"在
   1.1-epoch 快照口径下仍不成立（+0.126 不是小数字），但"税指数级恶化"的
   担忧被 2-seed 证伪。
2. **能力溢价持续**（e2e，同批 40×32×512，seed 0，torch.manual_seed(1234)
   损坏约定，corrupt_wrong 0.15）：
   - B0（无机制，评测侧加损坏）：corrupt r1 **3.551**（s0 3.574 / s1 3.528），
     r2 反而 **4.18**——无传输机制的二次 pass 在损坏输入上更脆；
   - C1：corrupt r1 **2.015**（s0 2.013 / s1 2.017）→ r2 1.949 → r3 1.943
     （r3 饱和，与 13M 形状一致）；node-0 修复 51.8% → 54.6%（r2）→ 54.9%（r3）；
   - **E2E 差距（r1）= 1.536 bpc**；B0 侧脆性随规模加深（3.55@100M vs
     3.14@29M r2 口径——协议差异见下），机制侧持续改善（2.008@29M → 1.949@100M）。
3. **协议可比性注意**：29M 的"B0 损坏 r2 3.138"出自其当夜 e2e（配置未全量
   落盘，不可完全复现）；本轮 B0 损坏列 = 评测侧加 corrupt_wrong 0.15
   （`.tmp/B0_100M_corr.json`，配置已存档于本目录 B0_100M_s{0,1}_corrupt_e2e.jsonl
   的生成配置）。跨尺度读数以"机制侧改善 + B0 脆性"的定性方向为准。
4. ECE：全部 ≤ 0.0091——prob0 校准在 100M 存活，Slider 底座可上规模。

## 结论（Rung 4 @100M，2 seeds）

- 税不增（+0.126）、能力溢价不缩（≥1.54 bpc）且机制侧逐轮精炼、r3 饱和、
  prob0 校准存活——**"概率驱动碎片化算力分配"的底座在 100M 成立**。
- 未做（backlog）：INT profile（n_mtp=2 + spec）@100M 的 Slider 前沿复测、
  更大 token 预算（4.4 epoch 渐近口径）、batch 一致性消解混杂。

## 文件

- `{run}/`：results.json（resolved config 全量）+ train_log.jsonl（model.pt 不入库）
- `{run}_e2e.jsonl` / `B0_100M_s{0,1}_corrupt_e2e.jsonl`：probe_e2e 输出
  （符号链接 .tmp/e2e_100M/<name>/model_step024000.pt → 各 run model.pt）
