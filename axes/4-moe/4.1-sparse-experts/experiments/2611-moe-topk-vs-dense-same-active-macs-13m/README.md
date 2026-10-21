# 2611 · 同每 token 激活 MACs 下，稀疏激活专家（top-1）买不买得到质量（`4.1` 的判决实验）· 13M · dense + top-1 MoE

> 本文件是**预登记契约**：判据线与判词规则在本节写死，跑之前不得修改。判据常数（0.014 带、臂清单、
> 配对表、判词规则、`verdict_rules`）住在 `config.json`；`README.md` + `config.json` + `run.py` +
> `train_arms.sh` 一起被
> `python3 tools/prereg_bind.py <本目录> --contract README.md config.json run.py train_arms.sh` 绑住。
> 读数落 `results.json`（实验根）与 `runs/<臂>/results.json`（逐跑），两者都过
> `tools/run_provenance.write_results`（自带 `code_hash_run_py` / `code_hashes`）。

## 1 要回答的问题（人话在前）

普通人话：**同样的算力，把 FFN 拆成一堆「只有一部分被用上」的专家，值不值？**

精确问题：在**每 token 专家激活 MACs 与 stock dense 逐位相等**的前提下，把每层 FFN 换成
`E` 个同宽专家、每 token 只走 top-1，相对**同算力的 dense 基线**能不能买到 ≥ 0.014 bpc 的质量。
`E` 取 **4 与 8** 两档（**低端两点**，参数 1.9×）⇒「专家数」这条轴有两点，能看出「同样的算力、成倍的
参数容量」是买到质量还是只买到欠拟合。两点都在低端，结论不外推到「更多专家更值」或「大 E 不行」。

这一问只有在**同激活算力**下才有意义：同总参数比，MoE 必然赢（它参数多）；真正的问题是
**每 token 花同样的 MACs，MoE 的额外容量能不能兑现**。

## 2 臂的设计（唯一变量 = FFN 的专家化）

| 臂 | FFN 形态 | E | k | 每专家宽 `h_e` | 每 token 专家 MACs/层 | 总参数 | token/param |
|---|---|---|---|---|---|---|---|
| `dense-B0-s{0,1}`（主对照） | stock `mlp_mult=6` | 1 | 1 | 1536 | 786,432 | ≈13.6M | ≈7.2 |
| `moe-top1-E4-s{0,1}`（实验） | top-1 稀疏专家 | 4 | 1 | 1536 | 786,432（+路由 1,024） | ≈42M | ≈2.3 |
| `moe-top1-E8-s{0,1}`（实验，专家数第二点） | top-1 稀疏专家 | 8 | 1 | 1536 | 786,432（+路由 2,048） | ≈79.8M | ≈1.23 |
| `dense-eqparam-E4-s{0,1}`（等参数对照，`SCHEMA.md` §15） | stock `mlp_mult=24` | 1 | 1 | 6144 | 3,145,728（4×） | ≈42M | ≈2.3 |

- 三个 MoE/dense 主臂的**专家激活 MACs 逐位相等**（786,432/层/token）；MoE 多出来的只是路由器
  `E·d` logits（E=4 = +0.13%，E=8 = +0.26%，逐臂申报，不进判据）。
- `dense-eqparam-E4` 与 `moe-top1-E4` **参数量相等**（≈42M）而每 token 算力是它的 4 倍：它回答
  「MoE 的增益是不是纯容量」（§15 的归因对照），**不是**等算力对照，**也不参与主判**——主判只在
  `dense-B0`（1× 激活算力）与 `moe-top1-E{4,8}` 之间。
- top-k 权重按**归一化**使用（top-1 ⇒ 权重恒 1.0）。理由：不归一时 top-1 的权重是 `p_max < 1`，
  相当于把 FFN 输出整体缩小——那是**第二个**混杂（输出尺度），而 `E=1,top-1` 逐位等于 dense 的
  恒等点也只有在归一后才成立（§4 判据⑤ 机检它）。
- 路由器**无辅助均衡损失**（`aux_weight = 0`）：均衡机制属 `4.2`，本格只判 top-k 本体（单因素）。
- 实现：`run.py` 内的 `TopKMoEFFN` + `MoEDepthARModel`（在 `super().__init__()` 之后把每层
  `blk.mlp` 换成专家模块），**不改** `pathlm/**`、`train_depth_ar.py`、`tools/**`、`tests/**`。
  同 seed 下 MoE 臂的非 FFN 参数从 stock dense 臂**逐位拷贝**（`copy_shared_from`）⇒ 臂间唯一
  差异是 FFN，不是初始化。

## 3 配方（所有臂逐字段相同）

| 字段 | 值 |
|---|---|
| 形态 | stock dense decoder-only（`dense=True`、`topology=sequential`） |
| d / layers / heads / dropout / norm_cap | 256 / 12 / 8 / 0.0 / 1.0 |
| T / batch_size / steps / lr / seed | 512 / 32 / 6000 / 1e-3 / 0 与 1 |
| token 预算 | 98,304,000 / 臂（= `configs/B0.json` 的 6000×32×512） |
| 数据 | `data/enwik8_full.npz`（前 90M 训练 / 末 10M 评测，字符级 vocab 206） |
| 训练入口 | `train_depth_ar.train()`（共享训练器，不复制训练循环）：AdamW wd=0.01、warmup 200 + cosine、bf16 autocast、`clip_grad_norm_` 1.0、`gnorm > 1e4` 跳步 |
| 评测 | 20 批 × bs16、`window_seed=2` 的**配对** clean bpc（fp32、无 autocast）；同 seed 的臂看到同一批窗口 |

## 4 判据线（预登记，跑后不改）

**Δ 方向统一**：`Δ = bpc(对照) − bpc(实验)`（对照 − 实验）⇒ **Δ > 0 = 实验臂更好**。
`results.json` 的 `delta_directions` 逐字段写 `comparator_minus_arm`。

| # | 判据 | 判定线 | 该涨 / 该跌 / 必须不变 |
|---|---|---|---|
| ① **主判（等算力）** | 两个 seed 各自配对的 `Δ = bpc(dense-B0) − bpc(moe-top1-E{4,8})` | **0.014 bpc 带**：`≥ +0.014` = MoE 在同激活算力下买得到质量；`≤ −0.014` = MoE 更差；`abs(Δ) < 0.014` = 不可排序 | 该涨 |
| ② **专家数** | 两个 seed 的 `Δ = bpc(moe-top1-E4) − bpc(moe-top1-E8)`（低端两点，参数 1.9×） | 同一条 0.014 带；`> 0` = 多专家更好 | 该涨（若容量能兑现） |
| ③ **归因（§15 等参数，旁证不参与主判）** | 两个 seed 的 `Δ = bpc(dense-eqparam-E4) − bpc(moe-top1-E4)` | `abs(Δ) < 0.014` ⇒ 质量差异**可由参数容量解释**（路由/稀疏无独立贡献）；`≥ +0.014` ⇒ MoE 优于等参数 dense（稀疏本身有贡献）；`≤ −0.014` ⇒ 稀疏本身更差 | 必须可分辨 |
| ④ **代价（申报，不进判词）** | 每 token 专家 MACs 比值、路由税、峰值显存、训练墙钟、`tokens_per_param` | 专家 MACs 比 `dense-B0` **必须恰为 1.000000** | 必须不变 |
| ⑤ **必须不变：实现恒等点** | `E=1, top-1` + 灌入 stock `mlp` 权重 ⇒ 与 dense 逐位一致（`bitwise_equal`）；参数量解析式 == 实测 | 机检（`run.py preflight`），不等即中止 | 必须不变 |
| ⑥ **路由行为（必须报）** | 逐层 + 全局专家使用率：归一化熵、最大专家占比（训练期每 200 步 + 评测期 20 批累计） | 报告；**坍缩登记条件**：任一层的归一化熵在终评 < 0.5，或最大占比在终评 > 0.5 | 必须报，不得沉默 |
| ⑦ **入场门（稳定性）** | 每个臂 `gnorm_max < 100` 且 `grad_skips == 0` | 破门 ⇒ 该臂作废、不进判词（登记「未跑」） | 必须不变 |

**判词规则（`config.json::verdict_rules`，跑后不改）**：

- 主判按 **seed 逐对**分类：`beyond_positive` / `in_band` / `beyond_negative`。两 seed **同向且都越带**
  ⇒ 按方向入词（n=2）；两 seed 都落带内 ⇒ 「同算力下不可排序（买不到质量）」；两 seed 不同向、
  或同向但未同时越带 ⇒ 「本 n 下不可排序」。**n=1 不落判词**。
- 结论必须同时写三件事：**质量是否提升**（①）、**参数差异**（`tokens_per_param` 与总参数倍数，与判词
  同段出现：13.6M / 42.0M / 79.8M，2.34 → 1.23 token/param，E8 的欠拟合风险更大）、**路由是否坍缩**（⑥）。
- 负结论同样要 n=2：两个 seed 都越负带才算「同算力下 MoE 更差」；否则只能说「不可排序」。

**否证线（预登记）**：① 两 seed 都落带内或越负带 ⇒ 在 13M / 98.3M token 这个尺度上，
**「同激活算力 + 成倍参数」不产生可测质量增益**；届时不写「MoE 无效」，只写「本尺度、本 token
预算下买不到」，并把欠拟合（token/param）与路由状态列为解释候选。

## 5 对照与方差

- 主对照 = `dense-B0`（同尺寸 stock dense，同 seed、同非 FFN 初始化）；等参数对照 = `dense-eqparam-E4`。
- 方差带 = **0.014 bpc**（`harness/methodology.md` §2：nondeterminism 0.0019、种子散布 ≤0.0138），
  仅在「配对 Δ」下作参考门槛；本实验的配对单位 = 同一批评测窗口（20 批）。
- seed 数 = 2（s0 / s1）；两 seed 同向且同时越带才入判词。

## 6 成本（`gpu_wall_budget: 120 min`）

| 项 | 值 |
|---|---|
| `gpu_wall_budget` | **120 min**（8 臂 × 6000 步 + 冒烟 + 评测的总和；单臂 timeout 20 min，`train_arms.sh`） |
| 显存 | 4090 24GB 共享卡；最大臂 moe-top1-E8 参数量 79.8M（bf16 权重 + AdamW 状态）⇒ 逐臂申报实测峰值 |
| 外部占用 | 卡上另有更高优先级项目（`local_lm_research` / `nomos` 的训练与常驻 `llama-server`）：每臂开跑前查空闲显存，< 8GB 即排队；不抢不杀 |
| 预算不够时砍什么 | 砍 seed（另一 seed 登记「未跑」）> 砍专家数第二点 E=8；**不砍判据线、不砍等算力对照、不砍等参数对照**（本预算下不需要砍，见下表） |
| 可中断性 | 每臂独立取/放锁；单臂失败即停，不重试循环 |

### 6.1 步时表（实测，100 步 / GPU 空闲 / 单臂独占）

| 臂型 | 旧实现（逐专家布尔索引）s/step | 现实现（分组前向）s/step | 6000 步预计 |
|---|---|---|---|
| `dense-B0`（13.6M） | —（无 MoE 路径） | 0.037 | 3.7 min |
| `dense-eqparam-E4`（42M 稠密） | — | 0.062 | 6.2 min |
| `moe-top1-E4`（42M） | 0.116–0.125 | 0.066–0.069 | 6.9 min |
| `moe-top1-E8`（79.8M） | — | 以重绑后 100 步冒烟实测为准（元级条件 1 的门：≤ 0.14 s/step） | ≤ 14 min（按门上限） |

8 臂（4 型 × 2 seed）按门的**上限**算 ≤ 2×(4.2+9+14+6.6) ≈ **68 min** + 评测 ≈5 min ⇒ 落进 120 min 预算，
且每臂 ≤ 14 min < 20 min timeout；实测值由重绑后的 100 步冒烟与逐臂 `results.json` 给出。
两次换配置的披露见 §10：① 旧实现 → 分组前向（E=16 0.40 → 0.2075 s/step）；② 专家数第二点 E=16 → E=8
（0.2075 s/step 仍要 20.7 min/臂 > 20 min timeout）。两批旧冒烟读数（20 步与 100 步的 `smoke*`、以及
600 步的 `smoke600/`）是**步时证据、非判决读数**，随第二次重绑删除；步时表由重绑后重跑的 100 步冒烟
（`smoke/*/results.json`）给出。


## 7 数据与脚本清单

- 数据：`data/enwik8_full.npz`（前 90M train / 末 10M eval，vocab 206，byte-level；不入库）。
- `run.py`：`--stage {ledger,preflight,train,eval}`；`TopKMoEFFN` / `MoEDepthARModel` / `copy_shared_from`
  与训练、评测、配对统计、判词全在这一个文件里（契约绑定与代码哈希都指向它）。
- `config.json`：臂清单、配对表、`verdict_rules`、评测口径、`band_bpc`。
- `train_arms.sh`：逐臂取/放 GPU 互斥锁（`harness/SCHEMA.md` §20.1），单臂 `timeout 1200`。
- 落盘：`ledger.json`、`preflight.json`、`runs/<臂>/{results.json,train_log.jsonl,gate_log.jsonl,model.pt}`、
  实验根 `results.json`、`analysis.md`、`MODEL.md`。

## 8 失败决策树（无效时的处置路径：测量 → 实现 → 规模 → 设计）

1. **测量**：`preflight` 的 E=1 逐位恒等断言失败 ⇒ 实现问题，先修接线，不判机制。
2. **实现**：专家 MACs 比值 ≠ 1.0、或两 seed 的 `Δ` 与逐批 bpc 算不出来 ⇒ 记账/汇总错，回 `ledger` 重算。
3. **规模**：两 seed 都落带内 ⇒ 报「本尺度不可分」，不外推；同时报 `tokens_per_param` 与路由状态，
   区分「容量买不到」与「容量买到了但欠拟合」。
4. **设计**：MoE 越负带且路由健康（非坍缩）⇒ 判「本尺度、本 token 预算下稀疏激活不产生质量增益」，
   并把「均衡机制（4.2）」「更细粒度/共享专家（4.3）」「更大 token 预算」列为下一步候选，不自行加机制重跑。

## 9 产出与复现

```bash
python3 tools/prereg_bind.py axes/4-moe/4.1-sparse-experts/experiments/2611-moe-topk-vs-dense-same-active-macs-13m \
  --contract README.md config.json run.py train_arms.sh
python3 -m pytest axes/4-moe/4.1-sparse-experts/experiments/2611-moe-topk-vs-dense-same-active-macs-13m/test_moe_forward_equivalence.py -q
bash axes/4-moe/4.1-sparse-experts/experiments/2611-moe-topk-vs-dense-same-active-macs-13m/train_arms.sh \
  dense-B0-s0 moe-top1-E4-s0 moe-top1-E8-s0 dense-eqparam-E4-s0 \
  dense-B0-s1 moe-top1-E4-s1 moe-top1-E8-s1 dense-eqparam-E4-s1
python3 axes/4-moe/4.1-sparse-experts/experiments/2611-moe-topk-vs-dense-same-active-macs-13m/run.py --stage eval
```

判词落 `analysis.md`（人话在前、六问齐全）与 `project.yaml::conclusion`；本节命令在跑通后原样保留，
未跑通的命令不得留在契约里。

## 10 实现与等价性（换实现不换科学设定的证据）

- **前向形态**：`TopKMoEFFN.forward` 用「按专家号排序 + 整层一次 `index_select` + 逐专家 `narrow`
  连续切片 + 一次 `index_add` 写回」。每专家的 GEMM 仍是**同一批行、同一形状**，只是把「每专家一次
  布尔索引 + 一次掩码拷贝 + 一次 host 同步」换成每层 1 次；实测步时见 §6.1。
- **重绑前那一版**原样保留为 `TopKMoEFFN.forward_naive`（只作参照，不参与训练）。
- **等价性测试**：`test_moe_forward_equivalence.py` 断言两条前向在同权重同输入下
  `max_abs_diff == 0`（k=1；fp32 与 bf16 autocast 两条路径）、`E=1/top-1` 灌 stock `mlp` 权重后与
  dense 臂逐位相等，并含一条**反恒真断言**（故意扰动被用到的专家权重 1e-3，逐位相等必须不成立）。
  k>1 不是位级一致（同一 token 的 k 份贡献由 `index_add` 原子累加，顺序不保证）——本实验的臂全部 k=1。
- **重绑披露（两次，第二次即当前绑定）**：
  ① 首个绑定（`2026-10-21T17:24:56+08:00`）绑的是旧实现（逐专家布尔索引）；旧实现实测 E=16 臂
  0.403–0.414 s/step ⇒ 40 min/臂，撞 20 min 单臂 timeout。经元级裁决后只换实现（判据与科学设定未动），
  第二次绑定（`2026-10-21T17:3x`）用分组前向。
  ② 分组前向实测 E=16 仍要 0.2075 s/step ⇒ 20.7 min/臂，仍撞同一 timeout；经元级裁决（条件 1/4），
  **第三次绑定只把专家数第二点从 E=16 换成 E=8**，其余（臂清单的主对照与等参数旁证、判据线、0.014 带、
  种子、评测口径、token 预算、步数）一字未动。E4/E8 是低端两点，结论不外推到大 E。
  两批旧冒烟读数（20 步与 100 步的 `smoke*`、600 步的 `smoke600/`）是步时证据、不是判决读数，随第二次
  重绑删除；当前 `smoke/` 与全部 `runs/` 读数都由**最终绑定代码**产出。
  ③ 换 E=8 后立刻发现 `run.py` 的 `ledger` / `eval` 段把第二点的臂名写死成 `moe-top1-E16-s0`（`ledger`
  当场 KeyError），改成从 `config.json` 派生臂名后**再重绑一次**（`2026-10-21T17:37:02+08:00` 那次绑定下
  没有产出任何读数——只有一次崩溃的 `ledger` 调用——所以它没有作废任何证据）。当前绑定是最终版。

