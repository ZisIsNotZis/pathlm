---
name: 2660-latentreg-vs-disc-13m
ticket: 63-axis6-predict / Phase B · WP 6.3.3-WP1
date: "2026-10-14"
grid: 6.3.3-latent-regression
engine: "width-MTP 冻结主干（2631-spec-three-arms/train/13m_s0、13m_s1；d=256/L=12，n_mtp=3，chain_mode=token，window=0，transport=none，vocab=206，tied 读出 E=U^T）"
budget: "GPU ≤20 min（硬时间盒）；阶段 1 = 冻结特征 + 训头 + teacher-forced 命中率；阶段 2 = 当草稿再验证（α₁/τ）"
---

# 2660-latentreg-vs-disc-13m · latent 回归 vs 离散跳头（WP 6.3.3 · WP1 最小判决）

> **本文件是预登记契约，写在跑之前**：臂、训练预算、判据、成本口径、否证条件在此写死，事后**不得**调参。
> 实测读数见 `results.json`（原始）与 `analysis.md`（当前结论）。
>
> **它回答的是什么**：6.3.3（`axes/6-predict/6.3.3-latent-regression/project.yaml`）的预言是
> 「一次前向先**回归**出 k 个未来位置的连续隐状态（latent），再各自经**共享读出**取贪心」相对
> 「直接分类到词表」有独立增益。本实验在**同一冻结主干、同一读出、同一参数量、同一训练预算**下把
> 这两条产出方式分开测，并按 **63-F4 裁决**把两条判据路径各作一臂：
> **(i) 直接发（估计得贪心，无接受率，每 token 主干 FLOPs ≈ L/k）** 与
> **(ii) 当草稿再验证（有接受率 τ，每 token 主干 FLOPs ≈ (1+k)/k·L ≈ L）**。

## 1 命题（可证伪）

**在冻结的 13M 主干上，只改「未来 k 个位置的产出方式」，把「回归到未来真值隐状态再经共享 tied 读出
贪心」与「直接对词表分类」在同参数量 / 同读出 / 同训练预算下对比：回归式是否在 t+2 命中率（路径 i）
或草稿接受率 α₁ / τ（路径 ii）上取得独立增益？**

可证伪形式：*若回归臂（λ>0）的 t+2 命中率相对同参数量离散臂的**配对 95% CI 含 0**，
且路径 (ii) 的 α₁ 差同 CI 含 0 ⇒ 判「在本尺度（13M 冻结主干、tied 共享读出、char enwik8）下，
『回归到连续 latent』相对直接分类**没有独立增益**」，并给出机制归因（见 §2 的口径说明）。*

## 2 机制口径（预登记的关键限定，先写清楚）

在**共享 tied 读出**（`logits = z @ E^T`，E = 主干 embed 权重）与**同参数量**（每头一个
`pathlm.model.TransformHead(d)`）下，两臂**唯一的自由度**是监督项：

```
z_j = TransformHead_j(h_i)            # 两臂共用
logits_j = z_j @ E^T                  # 两臂共用（读出共享，逐位同一份权重）
loss = Σ_j CE(logits_j, t_{i+1+j})  + λ · MSE(z_j, h_true_{i+1+j})     # λ 是本实验唯一变量
```

- 离散臂（`disc_*`）= **λ = 0**；回归臂（`lat_*`）= **λ > 0**（把 `z_j` 拉到主干在位置 `i+1+j` 的
  **真实隐状态** `h_true` 上——EAGLE 式特征级草稿）。
- **`lat_k2_lam0` 与 `disc_k2` 在数学上等价**：它是**实现等价自检**（同种子下逐位相同）。
  若二者不逐位相同 ⇒ 训练 harness 有非确定性 / 接线错误（blocker 级，先修）。
- 由此得到一条**预登记预言**：若 λ>0 无增益，则「回归到 latent 再贪心」在本设定下**不是独立机制**，
  而是同一参数化上的一个辅助监督项——该结论若成立，须作为**对 6.3.3 标题式描述的更正建议**
  上报 axis-lead（由 lead 决定是否改 `project.yaml` 的 `innovation`/`method`；**执行方不得自改**）。

## 3 单因素声明与臂定义（预登记，事后不得改）

**唯一变量** = 监督项 λ（以及 k）。其余逐字段固定：冻结主干权重、特征缓存、训练数据、步数、
优化器、种子、评测位置集、读出权重。

| 臂 | k | λ | 说明 |
|---|---|---|---|
| `ar_t1`（参照，非本项变量） | — | — | 同一冻结主干自身的 tied 读出来预测 `t_{i+1}`（1 次前向 / token 的质量参照） |
| `disc_k2` | 2 | 0 | 同参数量离散对照（直接分类） |
| `lat_k2_lam0` | 2 | 0 | **实现等价自检**：必须与 `disc_k2` 逐位相同 |
| `lat_k2_lam01` | 2 | 0.1 | 回归臂（弱） |
| `lat_k2_lam1` | 2 | 1.0 | 回归臂（强） |
| `disc_k4`（可选，仅时间允许） | 4 | 0 | k=4 离散对照 |
| `lat_k4_lam1`（可选，仅时间允许） | 4 | 1.0 | k=4 回归臂 |

**训练预算（写死，不得事后调）**：每个 (臂, 种子) = 3000 步 Adam（lr 1e-3、无 weight decay、
batch = 4096 个位置、固定数据顺序 seed=2660）；种子 ∈ {0, 1}（n≥2）。λ>0 的臂与 λ=0 的臂
**步数 / lr / batch / 数据顺序完全相同**。

**特征与数据（写死）**：
- 主干隐状态 `h` = `PathLM` 的**最终层残差流**（单位置解码时 = `nodes[0]["latent"]`；
  全序列前向时通过**非侵入式包装** `_mtp_nodes` 抓取，须断言「包装前后主干输出逐位不变」）。
- 训练文本 = enwik8 **train** 分片（`data/enwik8_full.npz`），1024 个 `seq_len=256` 块
  （262,144 个位置；只取 `i+1+k < 256` 的位置）。
- 评测文本 = enwik8 **val** 分片，512 个块（131,072 个位置），与训练块**不重叠**；
  全部臂用**同一个**评测位置集。
- 目标：`t_{i+1+j}`（token）；隐状态目标：`h_true_{i+1+j}`（同一次特征缓存里就有，无额外前向）。

## 4 判据（预登记；逐条给结论）

- **L0（硬门，实现保真）**：`lat_k2_lam0` == `disc_k2`（同种子，命中率与 CE 逐位相同）；
  冻结断言 = 主干参数 `requires_grad=False` 且一次 backward 后 `.grad is None`；
  **读出共享** = 两臂都使用同一份 `model.embed.weight`（记录 sha256）；`param_stats` 逐位相等。
- **L1（该涨 / 主判 · 路径 i 直接发）**：`lat_k2_lam{0.1,1.0}` 在 `t_{i+2}` 的 teacher-forced 命中率
  相对 `disc_k2` 的**配对**差（按评测块聚类 bootstrap，B=2000、seed=2660、95% CI）**不含 0** ⇒ 采纳；
  否则否证。`t_{i+3}` 同口径并报。0.014 只作参考量级（率类指标不跨量纲借用，C1 裁决）。
- **L2（该涨 / 主判 · 路径 ii 当草稿再验证）**：用回归臂的 `z_1` 头 argmax 当位置 N+2 的草稿，
  在**与 2640/2631 相同的 12 prompt × 200 token 协议**下走生产 `decode_spec`，
  报 α₁ / τ；相对 `disc_k2` 的 `z_1` 头草稿（同 regime 对照）的**配对**（prompt 聚类）差 CI 不含 0 ⇒ 采纳；
  否则否证。**并报**与 2640 `direct1`（主干内联合训练的 node2 头，α₁ 读数指针）的差，
  但**该差因训练 regime 不同不作判据**（只作参考）。
- **L3（必须报 / 代价轴，不作判据）**：路径 (i) 每 token 主干等效 FLOPs ≈ `L/k`；
  路径 (ii) ≈ `(1+k)/k · L`；头部参数与 MAC 占比按 `param_stats` 如实报。**禁墙钟**。
- **L4（否证）**：若 L1 与 L2 都无显著差（CI 含 0）⇒ 判「回归式在本尺度无独立增益」，
  归因写进 `analysis.md`（机制归因见 §2）：不得写成「latent 回归无价值」这种超出证据的结论，
  只能写「在**本设定**（共享 tied 读出 + 同参数量 + 该训练预算）下无独立增益」。
  若回归臂显著**更差**且 k 更大更差 ⇒ 判该回归式批量生成在本尺度不成立。
- **L5（预算对齐的证据）**:训练日志（每步 loss）逐臂落盘；报 λ>0 臂的 CE 与 MSE 分量轨迹
  （证明 λ 确实在起作用，而不是被 CE 淹没）。

## 5 口径纪律

- **禁墙钟判据**：`results.json` **不得**出现墙钟字段；GPU 实耗只允许出现在 `run_log.txt`。
- **方差**：命中率类用按**评测块聚类的配对 bootstrap**（B=2000、seed=2660，percentile 95%）
  + Wilson 95% CI；α₁/τ 类用按 **prompt 聚类的配对 bootstrap**（同 2640 口径，B=2000）。
- **样本量**：臂 × 2 种子（主干 ckpt 用 `13m_s0`；时间允许再跑 `13m_s1` 的主判两臂）。
- **实现边界**：**不得**修改 `pathlm/**`、`grid=6.3.3` 之外的实验目录、`2631/**`（只读）；
  本实验全部代码在本目录内。
- **禁外推**：结论只适用于「该冻结 ckpt / char enwik8 / k=2 / 该训练预算 / tied 共享读出」。
- **GPU**：按 `harness/SCHEMA.md` §20 取锁（`/tmp/pathlm_gpu.lock`，mkdir 法 + holder_pid/heartbeat + `trap`
  释放，最多等 10 分钟；等不到就上报）。纯 CPU 段不取锁。

## 6 必要自检（写进单测，须能失败）

1. **特征包装不改变主干**：包装 `_mtp_nodes` 前后，同输入的主干 logits 逐位相同。
2. **目标位移正确**：`t_{i+1+j}` 的 off-by-one 由固定样例钉住（j=1 ⇒ t_{i+2}）。
3. **读出共享**：两臂 `logits` 与 `model.embed.weight` 的关系机器可核（同一 tensor 对象 / sha256）。
4. **冻结断言**：主干 `requires_grad=False`；一次 backward 后 `.grad is None`；误放开梯度必须抓红（变异）。
5. **λ=0 等价**：`lat_k2_lam0` 与 `disc_k2` 同种子逐位相同（同一条断言在 CPU 上以小数据可跑）。
6. **禁墙钟**：`results.json` 无墙钟字段的机器可核断言。

## 7 跑法

```bash
# 单元测试（CPU，不取锁）
python3 -m pytest axes/6-predict/6.3.3-latent-regression/experiments/2660-latentreg-vs-disc-13m/test_latent_reg.py -q
# 阶段 1：特征缓存 + 训头 + teacher-forced 命中率（GPU，取锁）
python3 run.py --stage train --out . --ckpt ../../6.2-speculative-decoding/experiments/2631-spec-three-arms/train/13m_s0
# 阶段 2：当草稿再验证（GPU，取锁）
python3 run.py --stage draftverify --out .
# 变异验证（CPU）
python3 mutate.py
```
