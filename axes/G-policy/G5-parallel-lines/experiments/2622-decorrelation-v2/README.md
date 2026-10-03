# 2622 · 去相关 v2：修掉 gate 混杂 + 换成表征级去相关（13M + 50M）——契约

> 目录：`axes/G-policy/G5-parallel-lines/experiments/2622-decorrelation-v2/`
> 依据：`.scratch/39-close-gaps/issues/01-close-gaps.md` 的 **X1**（K9/PRIOR_ART 指定的
> 唯一最小决定性实验），加 `SCHEMA.md` §17.14 / §17.15 / §18 / §19 / §20+20.1 / §21 / §22+22.8 / §23。
> **本 README 的 §4 是预登记判据线：跑之前写下，跑完不得改。**
> 本实验是 `2620-decorrelation-at-scale`（ticket 37-V3）的**修混杂 + 换机制**重跑。

---

## 0. 一句话

13M 实测「去相关压力让 K=8 集成**更好** +0.0227」，与 50M 缺口 0.0223 **同量级** ⇒ 主判据的
符号未排除；但 50M 那一臂 **42.7% 的步被 grad-norm gate 跳过**（欠训练混杂），且用的是
**输出分布 capped-−JS** 罚则（已被证明**没能去相关**：成对 JS 0.0333→0.0250 bits）。本实验
**先把这两个混杂修掉**（gate 只在非有限时跳 + 换成**表征级**跨路隐状态相关罚则），再看
「K 路径集成能否打赢同 token 预算的单线 stock」这个主判据。

---

## 1. 命题（可证伪）

**修掉 gate 跳过混杂、并换用表征级去相关压力后，单主干多路径的 K=8 集成能否在 13M 与 50M
两个规模点上，赢过同 token 预算、同 seed 的单线 `stock`？**（主判据，见 §4.1）

---

## 2. 干预与单因素声明（精确表述）

每个规模是**同一格内的两段单因素阶梯**：

| 臂 | 相对上一级**唯一**的变化 | 隔离出的因素 |
|---|---|---|
| `stock` | —（阶梯底：单线骨干，无路径机制、无去相关压力） | — |
| `l1` | `p_skip: 0.0 → 0.1` | **路径机制**（每步随机丢层的采样路径 ⇒ 推理可成集） |
| `repr`（v2 去相关臂） | `w_repr_decorr: 0.0 → 1.0` | **表征级去相关压力** |

- **唯一真变量**：`p_skip` 与 `w_repr_decorr`；每个规模内 `configs{13m,50m}/{stock,l1,repr}.json`
  的 `model`/`train` 段逐字段相同、`path` 段除上述两键外逐字段相同（含 `seed=0`）。单测
  `tests/test_decorrelation_v2.py::test_ladder_configs_differ_only_in_declared_knobs` 钉死。
- **参数量完全相同**（去相关压力是**纯损失项**：不新增参数，只多一次前向）——
  `results_{scale}.json::checkpoints.*.params` 由对账机检相等。
- **`repr` = 复用 loop-era 引擎既有的 `p_skip` 路径机制 + 新增 `w_repr_decorr` 损失接缝**
  （`pathlm/config.py` 加字段、`pathlm/model.py` 加 `repr_decorr_penalty` + 一处 `if λ>0` 接线、
  `train_m1.py` 的 `sample_rounds`/活性量/gate），**不碰 block 结构**。
- **为什么换掉 capped-−JS、必须用表征级**（契约写死理由）：
  1. **实测反证**：50M 上 capped-−JS 压力让成对 JS **下降**（0.0333→0.0250 bits，两条线**更像**）
     且单线税由 −0.0086 变 **+0.0710** ⇒ 它不是有效去相关源（`2620/results.json::pairwise_js_at_kmax`
     与 `criteria.taxes`）；
  2. **机制理由**：输出分布的 JS 只约束**预测**的差异，而集成增益来自**误差不相关**；共享主干下
     两条路径的**表征**被同一个 CE 拉向同一解（候选机制「CE 向心力 > 罚则」），在**表征层**施加
     去相关压力（而非输出层）是直接对准该失败模式的干预；
  3. **形态理由**：capped-−JS 有**梯度空洞**（`min(JS, cap)` 在 JS≥cap 时梯度为 0），表征罚则
     无 cap、且尺度不变（Pearson 相关），既没有平台也不鼓励「缩小隐状态」的退化解。
- **gate 修复（混杂 ①）**：`train_m1.py` 的 pre-step gate 从 `not isfinite or gnorm>1e4`
  改为**只在非有限时跳**——因为 `clip_grad_norm_(…, 1.0)` 已经把任何「有限但很大」的梯度缩放到
  预算内，跳过它是没有必要的（旧行为在 50M `divl1` 上丢掉了 42.7% 的更新）。单测
  `test_gate_skip_only_on_non_finite` 钉死；变异 M4 抓红。**硬要求：本实验每臂 `gate_skips == 0`**。
- **规模配方（如实记账）**：13M = `d256/12L/8H/T512/mlp×6`，`bs32 × 6000 步`；
  50M = `d512/12L/8H/T512/mlp×6`，`bs16 × 12000 步`。**两个规模点都是 98.3M tokens、
  同 lr（1e-3）、同 seed（0）** ⇒ 13M→50M 是**同方法**比较（`SCHEMA §17.11` 需要两点）。
- **50M 的 `stock` 与 `l1` 复用 ticket 37 / 2620 的检查点**（同配置、同 seed、同 token；两臂
  `gate_skips=0` ⇒ 在新 gate 代码下逐位等价），在本入口下**重新评测**；50M 仅本轮新训 **`repr01`
  （λ=0.1）**。此为**不等字段**，如实记录。
- **λ 的选择（主 Agent 裁决 2026-10-03，先 13M 两次权重）**：13M λ=1.0 的 `repr` 成对 JS 升到
  0.0422（>l1 0.0249）但 clean 税 +0.040 ⇒ net repr@K8 反差 0.0059（<方差界，不可排序）；
  13M λ=0.1 的 `repr01` clean 税 **−0.0042**（几乎免费）、JS 0.0361、`l1@K8 − repr01@K8 = +0.0246
  ≥ 0.014` ⇒ 按预登记规则 **放行 50M**，50M 的 `repr`=λ=0.1。λ=1.0 不复 50M。

---

## 3. 指标表（多维，逐项）

| 维度 | 指标 | 怎么读 |
|---|---|---|
| 它解决什么问题 | K 路径集成的 bpc 增益（K1→K8） | 集成能不能替「单线」省质量 |
| 代价 | clean bpc 税（相对 `l1`、相对 `stock` 两套口径**分开报**） | 机制本身值多少钱 |
| 主判据（§4.1） | `bpc_repr@K8 − bpc_stock@K1 < 0` | vs **同 token 预算的单线 stock**（诚实对手） |
| 机制判据（§4.2） | `bpc_repr@K8 − bpc_l1@K8 < 0` + 成对 JS（repr vs l1） | 去相关**机制**是否起作用、是否真的更不相关 |
| 收益与用法 | K=1/2/4/8 四点曲线 + 每档相对 K=1 的 Δ | 每种 K 一行；**单点不外推**，K=8 是预登记点 |
| 可控性 | 成对 JS、确定性路径对照 | 增益是否**只**来自路径随机性 |
| 副产品 | 活性量（repr_decorr / JS / skip_frac / gnorm / gate_skips，每 200 步）、loss 斜率、峰值显存 | 干预是否活着、是否收敛、是否吃满更新 |

**算力价签（写进结论句，`§17.14` 第 4 项）**：K8 集成 = **8× 前向/token**，`stock` 单线 = 1×。

---

## 4. 判据线（预登记，跑之前写下，跑完不得改）

### 4.1 主判据（任务形式，**primary**）

> `bpc_repr@K8 − bpc_stock@K1 < 0` 且配对 CI 不重叠 ⇒ **多线首次成立**；否则记「仍不成立」。
> 必须同时报「推理算力账」：K8 集成 ≈ 8× 前向/token vs stock 1×，价签写进结论句。

**代数对账（`§12`）**：任务形式 = `−reading_B(l1@K1, stock@K1, repr@K8).reduced`；`run.py` 两种写法
都算并断言差 < 1e-9。

- **基准是谁**：`stock` 臂（同骨干、同 token 预算、同 seed 的单线模型，`p_skip=0, w_repr_decorr=0`）。
- **代价项相对谁算**：代价项 = `bpc_l1@K1 − bpc_stock@K1`，与基准**同一口径**。
- **CI 口径**：两臂各自 60 批的**配对** bootstrap（10k 重采样，α=0.05）。「CI 不重叠」同时报
  ① 配对差值 CI 不含 0（灵敏）；② 两臂各自 CI 不重叠（保守）。不一致时以保守口径为准判定。

### 4.2 并列判据（机制问题）

> `bpc_repr@K8 − bpc_l1@K8 < 0` 且配对 CI 不重叠 ⇔ **去相关压力在固定路径机制下的净效果**
> （这就是「机制是否起作用」）；**并报成对 JS**：`repr` 的 K=8 成对 JS 是否 **>** `l1` 的
> （是否真的更不相关）。

- **基准**：`l1` 臂（同路径机制、无去相关压力）。
- **代价项**：`bpc_repr@K1 − bpc_l1@K1`（去相关压力的 clean 代价）。

### 4.3 必报项

1. **`gate_skips`（须为 0）** 与**实际更新步数**（每臂）；
2. **算力账**：K8 集成 = 8× 前向/token、stock = 1×，以及「把同样算力投给基准臂」的近似参照；
3. **确定性路径对照（应恰好 0）**：`stock` 臂 + `repr` 臂（评测期强制 `p_skip=p_redo=shuffle=0`）；
   两处都报「8 条路径的 logits 是否逐位相同」+「实测 bpc 差」；
4. **成对 JS**（路径相关度，nats + bits）三臂 K=8；
5. **税**（§4.1/§4.2 两套口径分开）；
6. **活性量每 200 步**（`repr_decorr` / `diversity_js` / `skip_frac` / `gnorm` / `n_paths`）。

### 4.4 口径与禁令

- **口径**：matched-step **快照**；K 曲线用 **60-batch probe 协议**（与 13M 锚点同函数同 seed），
  另附 200-batch battery 的 clean bpc（与公开锚 `B0 1.5074` 同协议）作参照。
- **单点不得外推**：K=1/2/4/8 四点只描述形状；不拟合、不外推到其他 K 或规模。
- **收敛门禁（`§17.15`）**：门禁与「13M→50M 趋势」都落在**决定判据的量**（评测流上的判据量
  `bpc_repr@K8 − bpc_stock@K1`）上，在中点（13M@3000 / 50M@6000）与终点各测一次。
  50M 的 `stock` 是复用检查点（无中点）⇒ 判据量中点不可得，**如实标快照口径**并给各臂 K1 绝对
  bpc 斜率。训练总 CE 斜率仅作**并报**。
- **失败决策树**（无效时按序排除）：① 测量（probe 方法/CI）→ ② 实现（`w_repr_decorr` 是否真
  活着、`gate_skips` 是否为 0）→ ③ 规模（是否只是没收敛）→ ④ 设计（表征级去相关本身是否够了）。

---

## 5. 对照与方差

- 13M：3 个训练 run（stock/l1/repr，seed 0，各 1 次）+ 2 个确定性路径对照；
- 50M：2 个训练 run（l1/repr）+ 复用 stock + 2 个确定性路径对照；
- 方差界：`docs/methodology.md` 的 Δ < 0.014 不可排序 — 本实验全部比较都带 60 批配对 CI，
  **若差值 < 0.014 一律记为「不可排序」**，不得当结论；
- `§12` 三道对账：算术（run.py 断言代数等价）、任务（三臂同一评测流/同一位置集）、
  口径（K 曲线两规模同方法；battery 与 probe 两套口径分开列）。

## 6. 成本

```
gpu_wall_budget_min: 110        # 含冒烟 + 两规模训练 + 两规模评测；超支先在报告里报数
```

- **先实测步时再定预算**：400 步冒烟（每臂）→ 实测 ms/step → 若超预算按下面砍。
- **预算不够时砍什么（不砍判据线）**：① 砍 200-batch battery（改 60 批）；② 砍 K=2 档；
  ③ 砍 JS 重复批次。**禁止**砍：主判据四个读数、确定性对照、活性量、gate 报告。
- **可中断性**：各臂独立进程，每 200 步落盘；中断只丢当前臂。
- **峰值显存 ≤5GB net**（共享 4090；`peak_gpu_mem_mb` 落盘）。
  **已知偏离（主 Agent 裁决 2026-10-03）**：`repr` 臂每步多采样一条并行路径（两张计算图），
  实测峰值 **13M repr 4996 MB（<5GB）/ 50M repr 5630 MB（超 12%）**；与 ticket 37/2620 已接受的
  `divl1` 5610 MB 同源同量级。裁决=**接受并如实记账**（理由：① 配方一致性优先——主判据的基准
  `stock` 是同 `bs16×12000`，若两臂批大小/步数不同，基准就不再是基准；② 与既有先例一致；
  ③ 共享卡余量充足）。**执行要求**：逐臂峰值入账、跑前后各记一次 `nvidia-smi memory.free`、
  **不得**为好看改批大小/步数。
- **规模推进门（主 Agent 裁决）**：**13M 先做对机制**——若 13M 上 `repr@K8` **不优于** `l1@K8`
  （§4.2 机制判据不成立），则**不上 50M**（省 60 min），按「机制在 13M 未成立」收口并报数。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | enwik8（`.tmp/enwik8` → `data/enwik8_full.npz`，val = 后 10M 字符，训练从不碰） |
| 为什么是 enwik8 | 判据是**集成增益 vs 税**的部署定价（bpc），不是长程能力主张；且 13M 锚点就是 enwik8 bpc ⇒ 换任务就失去「同方法趋势」。`SCHEMA §19` 的局部先验抗性针对**长程记忆/搬运**类结论，**本实验不主张该类结论** |
| 训练入口 | `python3 train_m1.py <arm> --config configs{13m,50m}/<arm>.json --out-root train<scale>` |
| 评测入口 | `python3 run.py --scale {13m,50m}` |
| 驱动 | `train_all.sh {smoke|train <scale>|eval <scale>}`（按臂取/放 GPU 锁，§20.1） |
| 落盘 | `train{13m,50m}/<arm>/{results.json,train_log.jsonl,model.pt}`（`*.pt` 不入库）、`results_<scale>.json`、`analysis.md`、`MODEL.md`、`controls/`、`mutation_report.json`、`reconcile_report_<scale>.json` |

## 8. 既有的依赖与耦合（只记录，不做组合实验）

- 上游：`G5-parallel-lines`（13M 锚点、2620 的 50M 阶梯）；`axes/G-policy/PRIOR_ART_DECORRELATION.md`。
- 本实验**只做单轴单格实验**（scope 声明）；`stock→l1→repr` 是**同一格内的受控阶梯**。
- 复用 2620 的 50M `stock` 检查点（见 §2）。

## 9. 产出

`results_{13m,50m}.json`、`analysis.md`（人话摘要先行）、`MODEL.md`、`controls/`、
`reconcile_report_{13m,50m}.json`、`mutation_report.json`、`progress.md`。

---

## 10. 既有做法（prior art，`SCHEMA §18`）

**检索条件**：本环境**联网检索不可用**（未注册 web / `web_search` 工具）。因此下面凡外部系统
一律标 **`记忆·未核`**，只作设计方向、**不得作判据**；本地条目标 `本地实测` 并给路径。
完整分诊见 `axes/G-policy/PRIOR_ART_DECORRELATION.md`（ticket 37 知识线）。

1. **谁做过**：**Deep Ensembles**（Lakshminarayanan et al. 2017，`记忆·未核`）；**Negative
   Correlation Learning**（Liu & Yao 1999，`记忆·未核`，把成员间协方差写进罚则）；**Barlow
   Twins**（Zbontar et al. 2021）/ **VICReg**（Bardes et al. 2022）（`记忆·未核`，对**特征**做
   去冗余/方差正则，不是对输出分布）；**Stochastic Depth**（Huang et al. 2016，`记忆·未核`）。
   本地：`2620` 的 13M/50M 阶梯（`本地实测`）。
2. **为什么这么设计**：集成的误差只有在成员**误差不相关**时才被平均掉；共享主干成员天然相关
   ⇒ 要么独立训练（贵 N×），要么显式付钱买多样性。**表征级**罚则（NCL/Barlow/VICReg 一族）
   直接对「想要的性质」写损失，比靠初始化去「碰」更可控。
3. **它的实测结果**：外部多为分类/SSL/RL，**没有可直接对照的**「共享主干 + 自回归 LM + 50M
   bpc」受控结果（`(c)`，联网不可用不能证伪其存在）；本地：capped-−JS 输出罚则在 50M **判负**
   且没买来去相关（JS 0.0333→0.0250），13M 上有 +0.0227 的正号（`本地实测`）。
4. **我们的增量在哪**：
   - **换机制形态**（输出 JS → **表征级**跨路相关罚则）；
   - **修掉 42.7% 的 gate 跳过混杂**（吃满更新）；
   - **两档规模同方法**（13M + 50M，各 98.3M tokens、同 seed）；
   - **判据写死**（`§17.14`：基准 = stock 单线、代价项同口径、primary = 任务形式），
     「回本没有」是可判定二元结论；
   - **三个精确控制**：确定性路径对照（增益应恰好 0）、成对 JS（是否真更不相关）、
     算力价签（1× vs 8×）。

## 11. 能力面 + 共享面分析（`SCHEMA §22` / `§22.8`，派发前已做）

**能力面**（契约要求的每个 flag/组合，引擎是否支持）：

- `grep -n "ValueError\|互斥" pathlm/depth_ar.py` 的输出里，所有互斥校验都在 **depth-AR 引擎**
  （`attn_sparse_k / train_window / attn_linear / n_ky_heads / carry_rule / topology` 两两互斥、
  block 类二选一）。**G5 多线机制不在该引擎里**——它走 loop-era 引擎 `pathlm/model.py` +
  `train_m1.py`（`w_diversity` 只出现在 `pathlm/config.py`/`pathlm/model.py`/`train_m1.py`，
  `grep "w_diversity" pathlm/depth_ar.py train_depth_ar.py` = 空）。
- 本实验要的组合：`p_skip>0`（`sample_path` 既有）+ `w_repr_decorr>0`（本轮新增）。`sample_path`
  对 `p_skip` 无互斥；`w_diversity` 与 `w_repr_decorr` 是两条去相关罚则，**新增护栏
  `validate_pcap` 禁止同时开**（单因素），单测 + 变异 M5 抓红。
- **结论：所需组合在引擎里存在，无不可组合项。** 本线**不改** `pathlm/depth_ar.py` /
  `train_depth_ar.py`（那是 ticket 39 X2 的写入面）。

**共享面**（哪些线要改哪些文件）：本线写 `train_m1.py`、`pathlm/model.py`、`pathlm/config.py`
（+ 新实验目录 + `tests/test_decorrelation_v2.py`）。X2 写 `pathlm/depth_ar.py` /
`train_depth_ar.py` ⇒ **不相交**。改动前声明锚点、改完回读邻线标记：

```bash
$ grep -c "carry_state_dim\|recon_window\|recon_head" train_m1.py pathlm/model.py
0 / 0                                   # 邻线标记不在本线写入面内
$ grep -c "w_repr_decorr" pathlm/depth_ar.py train_depth_ar.py
0 / 0                                   # 本线改动不进入 depth-AR 引擎
```

- **禁止** `git checkout/stash/reset` 或回滚他人改动；变异脚本用运行期猴子补丁，**不改文件**。
- 关键共享文件的 sha256 记在 `MODEL.md` / `results_<scale>.json::code`。

## 12. 锁与并发（`SCHEMA §20` / `§20.1`）

- 取锁 `/tmp/pathlm_gpu.lock`（有界等待 + `holder_pid` 存活性自愈；释放/回收一律先 `rm -f`
  内容再 `rmdir`；不用 `pgrep` 判据）。
- **多臂按臂释放/重取**：`train_all.sh train <scale>` 每臂结束即释放、`sleep 65`（≥60s 窗口）
  再取下一臂（`§20.1`）。冒烟（整程 <10min）允许整程持锁；评测为纯推理但 >2min ⇒ 取锁。
- 持锁期间后台心跳每 60s `touch`；本线不设「纯推理插队」例外。

## 13. 七问（`SCHEMA §23`）

1. **既有做法查过吗**：✅ §10 + `PRIOR_ART_DECORRELATION.md`（六类来源分诊）。
2. **判据是最严的吗**：✅ 诚实对手 `stock`（同 token/seed，1× 算力）+ 三类护栏（确定性对照/
   活性/gate 报告）+ 机制判据（repr vs l1）。
3. **对照错的哪一类都试过吗**：✅ 同参数量（0 参数增量）、同算力价签（8× vs 1×）、同 token 预算。
4. **规模/预算边界探过吗**：✅ 13M + 50M 两点（禁外推）。
5. **反方向/否证路径走完吗**：✅ §4.4 失败决策树四层。
6. **未测项显式化**：✅ `analysis.md` §下一步列「最小实验 + 成本」。
7. **复核**：本实验落盘后由主 Agent 按 `§14/§14b` 另派 fresh-context 复核者。
