---
name: 2642-uniform-narrowing-13m
ticket: 64-axis3-width
date: "2026-10-14"
grid: 3.1.1-uniform-narrowing
engine: depth-ar（13M 档 · dense 形态）
budget: "gpu_wall_budget: 75 min（含冒烟 ≤4 min + WP1 两臂 + WP2 一臂 × 12k 步 + 全部评测与重评）；峰值显存 <8 GB"
status: planned
---

# 2642 · uniform 减宽（D192 / D128）+ 嵌套宽度 dropout（WD256）· 13M · dense

## 1. 命题（一句话，可证伪）

在**训练/数据/主干配方逐位相同**、唯一变量 = **宽度分配**的前提下：

- **WP1**：把 `d_model` 与头数整体等比压小（head_dim 恒 32 ⇒ 4/6/8 个头）相对全宽 stock，
  clean bpc 的代价是否**越 0.014 噪声带**（即可排序），以及该代价换来的**每 token 等效算力**节省，
  在 (每 token 等效 MACs, clean bpc) 前沿上**相对另一只旋钮（FFN 隐藏比 r）**落在哪一侧。
- **WP2**：训练期每步对**残差通道前缀**随机取宽度 ρ∈{4/8, 6/8, 8/8}（嵌套宽度 dropout），
  一次训练得到的**同一个 checkpoint** 在推理时切到宽度 256/192/128，
  各档质量是否**随宽度单调**（越宽越好），且各档相对**分别训练**的同宽 uniform 模型
  **不可排序**（|Δ| < 0.014 ⇒ 弹性宽度成立）。

- **H1（备择）**：减宽/弹性宽度能在等算力位点上占据前沿更优一侧（或至少「一训多档」不额外付代价）；
- **H0（零假设）**：d_model 减宽的质量代价落在噪声带内、且弹性宽度各档不可排序于同宽模型。

## 2. 单因素声明

**唯一变量 = 宽度分配**（`d_model` / 头数 / 残差通道活跃前缀）。

- 基底 = 主流标准 decoder-only transformer（`train_depth_ar.py --dense --topology sequential
  --exit-mode uniform`，仅最终深度 CE）；
- 数据、步数（12000）、批大小（32）、seq_len（256）、lr（1e-3）、seed（0）、
  mlp_mult（4）、dropout（0）、norm_cap（1.0）、评测批（`depth_curve` 固定 generator seed 2、
  n_batches=20、batch_size=16）全同；
- attention 接线、LN、位置嵌入、tied 读出、残差形式**逐字段相同**；head_dim 恒 32；
- **非必要性例外：无**。

### 2.1 WP1（D192 / D128）：引擎原生配置，无干预代码

`--d 192 --heads 6` / `--d 128 --heads 4`（head_dim = 32），`mlp_mult=4` ⇒ FFN 隐藏维 768 / 512。
本臂**不需要构造后切片**：引擎原生支持任意 d/heads 组合，因此臂间唯一差别就是宽度本身
（以及由宽度按 (w/d)² 决定的参数与每 token 算力）。等值性门 T1（下）覆盖「prefix 机制在 ρ=1.0
时不改前向」这一面。

### 2.2 WP2（WD256）：嵌套宽度 dropout = **残差通道前缀子网络**

训练每步采样一个宽度 `w ∈ {128, 192, 256}`（分别对应 ρ = 4/8, 6/8, 8/8，等概率），
**只在前 w 个残差通道组成的前缀子网络上做前向**（真算少，不是掩码）：

```
h  ∈ R^w                          （残差流 = 通道前缀）
embedding/位置嵌入：E[:, :w]、P[:, :w]
每层：
  ln1/ln2          在活跃前缀上做 LayerNorm，用 ln.weight[:w] / ln.bias[:w]
  qkv              q/k/v 各自取「输出行前缀 + 输入列前缀」，
                   拼成 [3w, w]（行序 = chunk(3) 的 q|k|v 块序）
  attention        头数 = w/32（head_dim 恒 32），SDPA 因果，输出 [B,T,w]
  proj             proj.weight[:w, :w]、proj.bias[:w]
  mlp              mlp[0].weight[:4w, :w]、mlp[2].weight[:w, :4w]（隐藏维 4w）
读出               logits = h[:, :w].float() @ E[:, :w].float().T
```

采样分布 = 三档**等概率**（`torch.Generator` 固定 seed，落在模型属性上、不进 `state_dict`），
每步独立采样。推理时不掩码：把同一 checkpoint 的**前缀子矩阵**装进一个**原生 d=w 的 stock 模型**
（`n_heads = w/32`、`mlp_mult=4`、`dense=True`），该模型就是部署物 ⇒ 每 token 等效算力按 w 计。

> **为什么不用引擎开关**：引擎没有「残差通道前缀宽度」开关（`pathlm/**` 禁止改）。
> 干预全部实现在本实验目录内（前缀前向 + checkpoint→w 模型转换），共享 `pathlm/**` 与
> `train_*.py` **零改动**。把该能力收进引擎的申请记在 `.scratch/64-axis3-width/TICKET.md`。

两条**等值门**（`harness/SCHEMA.md` §9 / `harness/methodology.md` §5，落成 pytest 单测 + 训练后 CUDA 复核）：

| 门 | 内容 | 判法 |
|---|---|---|
| **T1** | ρ=1.0（w=d=256）的**前缀前向** == stock 引擎前向 | logits 与 loss **逐位相同**（`torch.equal`），fp32 与 bf16 autocast 两种模式各测一次 |
| **T2** | 推理期**切片模型**（原生 d=w）前向 == 同宽度 w 的**训练期前缀子网络**前向 | logits 与 loss **逐位相同**，w ∈ {128,192,256} |
| **T3** | 变异（必须抓红） | 故意切错列/切错行/LN 取错张量/头数取错 ⇒ T1/T2 或形状审计必须变红 |
| **T4** | 「省算力可达」审计 | 切片模型里**不存在任何 d=256 的维**；实测参数 == w 模型的解析参数；活跃维 == 目标 w |

## 3. 臂

| 臂 | 定义 | 训练配置 | 备注 |
|---|---|---|---|
| **U4**（锚，复用） | d=256/heads=8/mlp_mult=4 | 与 2641 `U4-s0` **逐字段相同** | **不重训**：读 2641 `results_U4_s0.json`（1.5314） |
| **U2**（锚，复用） | d=256/heads=8，FFN 宽 r=2 | 与 2641 `U2-s0` 逐字段相同 | **不重训**：读 2641 `results_U2_s0.json`（1.5709，5,295,616 MACs/token） |
| **D192**（WP1） | **uniform 减宽** d=192、heads=6、mlp_mult=4（FFN 768） | `--d 192 --heads 6` | 引擎原生 |
| **D128**（WP1） | uniform 减宽 d=128、heads=4、mlp_mult=4（FFN 512） | `--d 128 --heads 4` | 引擎原生 |
| **WD256**（WP2） | 嵌套宽度 dropout，训练 d=256/heads=8，每步 w∈{128,192,256} | `--d 256 --heads 8` + 前缀前向 | 一 checkpoint ⇒ 三档部署 |

- n：**seed 0 全新增臂**；GPU 预算有余补 `D192-s1`、`WD256-s1`（若补，结论如实记 n）。
- 复用纪律（`harness/SCHEMA.md` §9）：U4/U2 只作**阈值/前沿位点**引用，逐字段配置对照与代码哈希由
  `--aggregate` 从 2641 的 `config.json` / `results_*.json` **程序化核对**并落进 `results.json`
  （三道具：① 开关关闭逐位等值单测 = 本实验 T1；② 逐字段配置对照 = `anchor_config_parity`；
  ③ 口径标注 = `cost`/`eval` 块内注明与 2641 同口径）。

## 4. 判据线（**预登记，不可改**）

| # | 判据 | 该涨/该跌/该不变 | 判定规则 |
|---|---|---|---|
| **P1** | `Δ_D192 = bpc_D192 − bpc_U4`、`Δ_D128 = bpc_D128 − bpc_U4` | 该跌（质量） | `\|Δ\| < 0.014` ⇒ **不可排序**（减宽代价落在噪声带内）；`Δ ≥ +0.014` ⇒ **可分：减宽更差**；`Δ ≤ −0.014` ⇒ **可分：减宽更好**。同报算力/参数节省 |
| **P2** | 前沿对照：{U4, U2, D192, D128} 在同一 (每 token 等效 MACs, clean bpc) 平面 | 该报 | 报支配关系与「每 1% 算力节省的质量代价」=（Δbpc 相对 U4）/(MACs 节省 %)。回答「d_model 减宽 vs FFN 比例在等算力位点落哪一侧」。**必须并列报参数量**（防把「减参数」记成「减宽度」） |
| **P3** | WD256 同一 checkpoint 在宽度 {256,192,128} 的质量曲线 | 该单调（越宽越好） | 单调 = `bpc(128) > bpc(192) > bpc(256)`（严格）；相邻档差 < 0.014 时标「相邻档不可排序」。弹性判定：`\|bpc_WD@192 − bpc_D192\| < 0.014` **且** `\|bpc_WD@128 − bpc_D128\| < 0.014` ⇒ **不可排序 ⇒ 弹性宽度成立**；任一档差 ≥ 0.014 且 WD 更差 ⇒ **明确劣于同宽 uniform** |
| **P4** | 必须不变/活性 | 必须不变 | ① 评测确定性：`--no-train` 重评与首评 clean bpc 及逐深度 bpc **逐位一致**；② 因果性：改末位 token ⇒ 之前位置 logits 逐位不变、末位必须变；③ 活性：每个宽度下逐层 `‖block_out‖/‖block_in‖ > 0`；④ D192/D128 实测参数 == 解析参数；⑤ WD256 三档切片模型实测活跃维 == 目标 w |
| **P5** | 否证线（`harness/research-schema.md` §12.2） | — | **a**：若 `\|Δ_D192\| < 0.014` 且 `\|Δ_D128\| < 0.014` ⇒ 写「在本尺度质量代价落在噪声带内，减宽该不跌」；**b**：否则若 D192 与 D128 的「每 1% 算力节省的质量代价」**均 ≥** U2 同口径斜率 ⇒ 写「d_model 减宽不占前沿优势」；**c**（WP2）：质量曲线不单调，或 WD256 任一档比同宽 uniform 差 ≥ 0.014 ⇒ 判弹性宽度在本尺度不成立 |

- 质量口径 = **clean bpc**（dense 形态下只有最终深度被监督 ⇒ `depth_curve()[-1]["bpc"]`，
  序列级、越低越好），eval 批固定 `depth_curve` 的 generator seed 2、n_batches=20、batch_size=16
  —— 与 2641 逐字段同口径（paired）。
- 方差带 = **0.014 bpc**（`harness/research-schema.md` §12.1 / C1 裁决：限**配对 Δ**、≥5 窗口；
  率类指标才走配对 bootstrap / Wilson 95% CI，本实验无率类判据）。本实验另报历史同配置 dense
  13M/12k run 散布 1.5257 / 1.5287 / 1.5298 / 1.5314（极差 0.0057）作**参照**，不替代 0.014。
- **口径**：墙钟 / 吞吐**只进 `run_log.txt` 过程日志**，不进 `analysis.md` 判据、不进任何
  cost 表述；一切代价 = **等效理论量**（每 token MACs：attention `4w²+2Tw`、FFN `2·r·w·w`、
  tied 读出 `w·V`，V=206，不含任何时间量）。

## 5. 预算与显存

- `gpu_wall_budget: 75 min`（含冒烟 ≤4 min、WP1 两臂 ≈7 min/臂、WP2 一臂 ≈5 min
  （平均活跃宽度² ≈ 0.61×全宽）、逐臂评测 ≈40 s、重评 ≈40 s）。
- 预算不够时砍什么（`harness/SCHEMA.md` §8）：先砍 **seed 数**（seed 1 一律不跑），再砍 **WP2**
  （WP1 必须落地完整判决包），**不砍判据线、不砍步数**（步数一改臂间刻度就不可比）。
- GPU 按 `harness/SCHEMA.md` §20 互斥锁**按臂取/放**（`/tmp/pathlm_gpu.lock`，写 pid + 用途 + ISO 时间），
  臂间 sleep ≥61 s；峰值显存 <8 GB（13M 档实测 ≈1 GB 量级）。

## 6. 指标表（逐项分析，不合成单一分数）

| 维度 | 指标 | 与谁比 |
|---|---|---|
| 解决什么问题 | `d_model` 减宽/弹性宽度是否是独立于 FFN 预算的设计维度 | — |
| 代价 | 每 token 等效 MACs（attention/FFN/读出分列）+ 参数量（总/FFN/embedding/门参数分列） | U4（100%）/ U2（等算力位点） |
| 有效性 | clean bpc（越低越好，越 0.014 才可排序） | U4（锚）、同宽 uniform 臂 |
| 收益与用法 | 前沿点与支配关系；每 1% 算力节省的质量代价 | U4 → U2 vs U4 → D192/D128 |
| 可控性 | WD256 的 w → bpc 曲线单调性 | 三档切片模型（同一 checkpoint） |
| 副产品 | 逐层 `‖block_out‖/‖block_in‖`、逐层隐藏维 | 臂间 |
| 活性 | 等值门 T1/T2、形状/参数审计、因果性、评测确定性 | 契约 §2.2 |

## 7. 否证条件与归因（做不出来时怎么区分）

- 若 P1 落带内且 P2 斜率不优于 FFN 旋钮 ⇒ H0：判「本尺度 d_model 减宽无可排序收益、
  不占前沿优势」，结论并入 `3.1.4` 的 FFN 预算结论（「总量」这一自由度已足够）。
- 若 T1/T2 失败 ⇒ 视为**实现问题**（先修实现），不得据此判机制失败。
- 归因三分（`harness/research-schema.md` §7）：①**设计问题**（d_model 减宽本身在此尺度无头寸）
  ②**结构问题**（切片实现不等价 / attention 占算力比过大使宽度分配无从体现）
  ③**innovation 本身失败**（弹性宽度路线不成立）。
- WD256 若某档明显劣于同宽 uniform（≥0.014）⇒ 判「一训多档要付容量/干扰代价」，
  并把「各档质量」如实落盘，不外推。

## 8. §12.2 要求的并列列（剪枝类）

- 参数量（总 / FFN / embedding / 结构上门参数）与**每 token 等效 MACs** 必须**并列**出现在
  前沿表与 WD256 曲线表里；引用锚（U4/U2）时用其 `results_*.json` 里的实测值。
- 对照口径 = 剪枝类**同 FLOPs 位点**（`harness/research-schema.md` §12.2 / 64-F1）：U2 即
  「d_model 固定、FFN 预算减半」的等算力近似位点；D192/D128 与之比较时**报两列**（MACs、参数）。

## 9. 复现手段

```bash
# 冒烟（60–300 步，确认干预真的到了模型 + 量步时/显存，不入判据）
STEPS=120 ARMS="D192:0 D128:0 WD256:0" bash smoke.sh
# 全部臂（按臂取/放 GPU 锁，臂间 ≥61 s）
ARMS="D192:0 D128:0 WD256:0" bash train_arms.sh
# 单臂重跑 / 仅评测（复用 runs/<arm>-s<seed>-12k/model.pt，并写 reeval_<arm>_s<seed>.json）
python3 run.py --arm D192 --seed 0 --steps 12000
python3 run.py --arm WD256 --seed 0 --steps 12000 --no-train
# 等值门与变异验证（自包含，不改共享 tests/）
python3 -m pytest test_2642_uniform_narrowing.py -q
# 汇总 P1–P5 → results.json（可从落盘 JSON 复算全部判据字段）
python3 run.py --aggregate --arms D192,D128,WD256 --seeds 0
```

- 权重**不入库**（`runs/**/model.pt`，`.gitignore` 覆盖）；哈希清单见 `MODEL.md`。
- 原始数字 = `results_<arm>_s<seed>.json`（逐臂）+ `results.json`（汇总判据）；
  训练日志 `runs/<arm>-s<seed>-12k/train_log.jsonl`、`gate_log.jsonl`；
  WD256 的宽度采样直方图 `runs/WD256-s0-12k/width_samples.json`。
