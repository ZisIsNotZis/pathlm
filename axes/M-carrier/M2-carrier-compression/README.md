---
id: M2-carrier-compression
axis: M
grid_dir: M-carrier/M2-carrier-compression
title: "载体压缩"
status: partial
judge: "内存下降 × 质量/保留损失（双边前沿）"
cost: "13M 单臂 ~25 min"
tax: "驱逐 +0.058~0.121（13M）"
benefit: "ring-buffer 解码免重 prefill（内存有界）"
verdict: "驱逐已测（价格见 A3）；**KV 量化已有本地实测**（R1/2614、2618/2621；但旧价签在「非因果 scale」下失效，已入冻结清单，待 R1 因果口径重测）；摘要压缩 / 低秩投影（MLA 式）零经验。"
deps: ["A3-window-anchor", "W3-kv-head-sharing"]
evidence: [".scratch/07-x1-x3/evidence @7b47555", "axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/analysis.md @c39dc88"]
---

<!-- AUTO:begin -->

# M2-carrier-compression · 载体压缩

> 轴：**M 状态载体** ｜ 状态：**◐ partial**

> **判读**：驱逐已测（价格见 A3）；**KV 量化已有本地实测**（R1/2614、2618/2621；但旧价签在「非因果 scale」下失效，已入冻结清单，待 R1 因果口径重测）；摘要压缩 / 低秩投影（MLA 式）零经验。

## 1. 命题

载体怎么压（内存换质量）。

## 2. 选项枚举（本格全部可能取值）

1. `不压`
2. `量化（KV/权重）`
3. `驱逐窗口`
4. `摘要压缩`
5. `低秩投影（MLA 式）`

## 3. 状态（我们做了什么 / 行业做了什么）

**本地已测**：ring-buffer 解码免重 prefill（内存有界）

**行业口径**：见 `docs/playbook.md` 与参照线 `R-references/`。

## 4. 结论与数字

- **税**：驱逐 +0.058~0.121（13M）
- **收益**：ring-buffer 解码免重 prefill（内存有界）
- **判读**：驱逐已测（价格见 A3）；**KV 量化已有本地实测**（R1/2614、2618/2621；但旧价签在「非因果 scale」下失效，已入冻结清单，待 R1 因果口径重测）；摘要压缩 / 低秩投影（MLA 式）零经验。

## 5. 失败路径与原因

1. 驱逐 × 复制目标冲突（三路豁免全败，见 A3）
2. **保留曲线天平已建成并补了检索腿**（ticket 31-I + 32-M，`tools/evalkit/retention.py`）：三条曲线严格单调、同 seed 重跑最大差 4.94e-07；单距离曲线 d=1 +3.245 / d=8 +0.090 / d=16 +0.017 / 恰在 d=32 ≤+0.004 ⇒ 代价由「离查询多近」决定，几乎与内容无关
3. **ERRATA（主 Agent，2026-09-30）：单距离曲线不得当可加外推。** 原入账「安全丢弃半径 d*≈32（d≥32 全丢 <0.014）」是**错的**——把距离 ≥32 的 KV **全部**丢掉（只保留最近 32 个）实测累计 **+0.212 bpc（单点读数的 15 倍）**；只保留最近 64/128/192 分别为 +0.081/+0.020/+0.004。⇒ **左前缀并不便宜**：尾部仍有可观依赖总量；「丢最远 16–32 个」只在极短预算下才近似免费
4. **检索腿（ticket 32-M）**：基线召回 68.68%（契约指定的 v2 构造在本 checkpoint 只有 1.5% ⇒ 按契约先修构造，改真实文本短语针 v3）；两腿同向（Pearson −0.472，39/39 配置）但动态范围有限，主判据（低损伤且召回掉 >30pp）未达成；`lru:k=256` 独立复现「低损伤 ≠ 保住了」（bpc 伤害最低 +0.0885 而召回 4.5→22.8pp）
5. **该仪器只量「依赖」不量「保留」**（SCHEMA §17 第 10 条）：低损伤也可能是模型本来就不用它——X3「假保留」的另一面。检索腿已排进 ticket 32 契约 M

## 6. 判据与判决实验设计

- **判据**：内存下降 × 质量/保留损失（双边前沿）

**下一步判决实验**（未做时必填）：

- 干预：待定（按判据设计单轴单格实验，基底恒为 stock 对照）
- 判据线：内存下降 × 质量/保留损失（双边前沿）
- 成本：13M 单臂 ~25 min

## 7. 成本估计

- 13M 单臂 ~25 min

## 8. 依赖与耦合声明（只记录，不做组合实验）

1. `A3-window-anchor`
2. `W3-kv-head-sharing`

## 9. 复现手段

1. .scratch/07-x1-x3/evidence @7b47555
2. axes/M-carrier/M2-carrier-compression/experiments/2614-kv-ablation-retention-13m/analysis.md @c39dc88

<!-- AUTO:end -->

<!-- MANUAL:begin -->
<!-- MANUAL:end -->
