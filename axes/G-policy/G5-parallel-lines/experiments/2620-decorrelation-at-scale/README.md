# 2620 · 去相关的规模验证（50M，三臂阶梯）——契约 V3

> 目录：`axes/G-policy/G5-parallel-lines/experiments/2620-decorrelation-at-scale/`
> 依据：`.scratch/37-gaps/issues/01-gaps.md` 的**契约 V3**，加上主 Agent 的裁决
> （ticket 37-V3 判据基准事故；规范已入宪为 `SCHEMA.md §17.14`）。
> **本 README 的 §4 是预登记判据线：跑之前写下，跑完不得改。**
> 契约副本说明：§4 的判据线**逐字**抄录主 Agent 的裁决原文；其余小节为**摘要**。

---

## 0. 一句话

K8 引擎情报（K8）判定「多线引擎已被业界做透，第一障碍是**去相关从哪来**」，本地 13M
实测的结论是「去相关的**收益**（K 路径集成）小于它带来的**税**」；本实验把同一个受控阶梯
搬到 ≈50M，直接给「多线机制（含显式去相关训练）能否打赢同 token 预算的单线」定价。

---

## 1. 命题（可证伪）

**在 13M 上「K 路径集成增益 < 多线机制的 clean bpc 代价」的方向，放大到 ≈50M 是否翻转**——
即：把约 52M 参数的共享主干训练出可成集的随机路径 + 显式去相关压力后，它的 **K=8 集成**
能否在**同一评测流**上打赢**同 token 预算的单线 stock 骨干**？

证伪条件见 §4：主判据 ≤ 0 即「50M 仍不成立」，并按 §4.2 记趋势。

---

## 2. 干预与单因素声明（精确表述）

**`stock → l1 → divl1` 是两段单因素阶梯**，不是「只改一个因素」：

| 臂 | 配置 | 相对上一级**唯一**的变化 | 隔离出的因素 |
|---|---|---|---|
| `stock` | `configs/stock.json` | —（阶梯底：单线骨干，无路径机制、无去相关压力） | — |
| `l1` | `configs/l1.json` | `p_skip: 0.0 → 0.1` | **路径机制**（每步随机丢层的采样路径 ⇒ 推理时可成集） |
| `divl1` | `configs/divl1.json` | `w_diversity: 0.0 → 0.05` | **去相关压力**（capped −JS 多样性损失） |

- **唯一真变量是 `p_skip` 与 `w_diversity` 两个键**；`configs/{stock,l1,divl1}.json`
  的 `model`/`train` 段逐字段相同、`path` 段除上述两键外逐字段相同（含 seed=0）。
  单测 `tests/test_decorrelation_at_scale.py::test_ladder_configs_differ_only_in_declared_knobs`
  把这条钉死（改任何其他字段即红）。
- **三臂参数量完全相同 = 51.84M**：去相关压力是**纯损失项**（不新增参数，只多一次前向）；
  `results.json::intervention.params_added = 0` 是构造性事实，不是测量值。
- `divl1` = **复用已有实现** `configs/DIVL1.json` 的 path 块（`n_mtp=1, p_skip=0.1,
  w_diversity=0.05`）+ 引擎既有的 `w_diversity` 损失接缝 + `train_m1.py` 既有的
  **float32 JS + pre-step grad-norm gate**（`gnorm > 1e4` 或非有限 ⇒ 跳过该步，计数落盘）。
- **规模 ≈50M 的插值配方（如实记账）**：既有配方为 29M = `d_model 384 / 8H`（head_dim 48）
  与 100M = `d720 / 12H`（head_dim 60），两者都是 `n_layers 12 / mlp_mult 6 / T 512`。
  本轮取 `d_model 512 / 8H / n_layers 12 / mlp_mult 6 / T 512` = **51.84M**（+3.7% vs 50M）。
  **`512/8 = head_dim 64` 比两个端点都大，不是严格插值**——选择依据是 head_dim 64 是该尺寸类
  的标准值、且总参数量最接近 50M；这一不等记录在此，不放进口径。
- **步数/token 对齐**：`6000 步 × bs 32 × T 512 = 98.3M tokens`，与 13M 的 `L1`/`DIVL1`
  锚点**逐字段同批同步数同 lr 同 seed**（只有 `d_model`/`n_heads` 随规模变）。因此
  13M→50M 是**同方法**比较（§4.2 需要）。

---

## 3. 指标表（多维，逐项）

| 维度 | 指标 | 怎么读 |
|---|---|---|
| 它解决什么问题 | K 路径集成的 bpc 增益（K1→K8） | 集成能不能替「单线」省质量 |
| 代价 | clean bpc 税（相对 `l1`、相对 `stock` 两套口径**分开报**） | 机制本身值多少钱 |
| 有效性 | 主判据（§4.1）：`bpc_stock@K1 − bpc_divl1@K8` | vs **同 token 预算的单线 stock**（诚实对手） |
| 收益与用法 | K=1/2/4/8 四点曲线 + 每档相对 K=1 的 Δ | 每种 K 一行；**单点不外推**，K=8 是预登记点 |
| 可控性 | 成对 JS（路径相关度，nats/bits）、确定性路径对照 | 增益是否**只**来自路径随机性 |
| 副产品 | 活性量（JS / skip_frac / gnorm / gate 跳过数，每 200 步）、loss 斜率、峰值显存 | 干预是否活着、是否收敛 |

**算力价签（写进结论句，§17.14 第 4 项）**：K8 集成 = **8× 前向/token**，`stock` 单线 = 1×。

---

## 4. 判据线（预登记，逐字抄录裁决原文 + 执行口径）

### 4.1 主判据（读法 B，部署问题；**primary**）

> `[bpc_l1@K1 − bpc_divl1@K8] − [bpc_l1@K1 − bpc_stock@K1] > 0` 且 CI 不重叠
> ⇔ **多线机制（含显式去相关训练）在 50M 上能否打赢同 token 预算的单线 stock**。
> 必须同时报「推理算力账」：K8 集成 ≈ 8× 前向/token vs stock 1×；把这个 8× 当成**明确的价签**写进结论句。

**代数化简（§12 算术对账）**：两项都含 `bpc_l1@K1` ⇒ 等价于
**`bpc_stock@K1 − bpc_divl1@K8 > 0`**。`run.py` 直接把两种写法都算出来并断言相等（差 < 1e-9）。

- **基准是谁**：`stock` 臂（同骨干、同 token 预算、同 seed 的单线模型，`p_skip=0, w_diversity=0`）。
- **代价项相对谁算**：代价项 = `bpc_l1@K1 − bpc_stock@K1`，与基准**同一口径**（同一评测流、同一
  60-batch probe 方法、同一 K=1 单线读数）。
- **CI 口径**：两臂各自 60 批的**配对** bootstrap（10k 重采样，α=0.05）。
  「CI 不重叠」同时按两个口径报：① 配对差值 CI 不含 0（灵敏）；② 两臂各自的 95% CI 不重叠
  （保守）。两者不一致时**如实并列**并以保守口径为准判定。

### 4.2 并列判据（读法 A，机制问题；**次级**）

> `bpc_l1@K1 − bpc_divl1@K8 > 0` 且 CI 不重叠 ⇔ 去相关压力在**固定路径机制**下的净效果
> （13M 已过 +0.0699，看 50M 是否保持）。

- **基准**：`l1` 臂（同路径机制、无去相关压力）。
- **代价项**：`bpc_divl1@K1 − bpc_l1@K1`（去相关压力的 clean 代价）。

### 4.3 必报项（③）

1. **K-path 集成的成对 JS**（路径相关度，nats + bits，定义见 `tools/evalkit/ensemble_probe.py`）；
2. **确定性路径对照**（应**恰好 0**）：a) `stock` 臂（路径本身就是确定性的）K=8 vs K=1；
   b) `divl1` 检查点在**评测期强制 `p_skip=p_redo=shuffle=0`** 下的 K=8 vs K=1。
   两处都报「8 条路径的 logits 是否逐位相同」+「实测 bpc 差」（浮点级）；
3. **税**（§4.1/§4.2 两套口径分开）；
4. **活性量每 200 步**（`train_log.jsonl` 的 `diversity_js` / `skip_frac` / `gnorm` / `n_paths`
   + 末值汇总 + `gate_skips`）；
5. **近似同算力参照**（**不是**同算力实测，标注为近似）：13M 单线 `B0 = 1.5074 bpc @1.1 epoch`
   → `1.3459 @4.4 epoch`（≈4× 训练 token 换 −0.1615 bpc，来源
   `.scratch/11-retry-matrix-validity/evidence/README.md` D5 final）。用它给「多线要成为部署优势
   还得跨过什么门槛」的量级判断。

### 4.4 口径与禁令（④）

- **口径**：matched-step **快照**（6000 步）；K 路径曲线用 **60-batch probe 协议**
  （`Random(11)`/`Generator(11)`/bs 32/T 512/temp 1.0，与 13M 锚点同方法）；
  另附 200-batch battery 的 clean bpc（与仓库公开锚 `B0 1.5074` 同协议）作参照。
- **单点不得外推**：K=1/2/4/8 四点只描述形状；不拟合、不外推到其他 K 或规模。
- **未收敛** ⇒ 标快照口径，并给同方法末窗 loss 斜率。
- **失败决策树**（无效时按序排除）：① 测量（probe 方法/CI 是否与 13M 同方法）→
  ② 实现（`w_diversity` 是否真活着、`p_skip` 是否真随机、K 集成是否真在平均）→
  ③ 规模（50M 是否只是没收敛）→ ④ 设计（去相关机制本身）。

---

## 5. 对照与方差

- 三臂共 3 个训练 run（seed 0，各 1 次）+ 评测期 2 个确定性路径对照；
- 方差界：`docs/methodology.md` 的 Δ < 0.014 不可排序 — 本实验全部比较都带 60 批配对 CI，
  **若差值 < 0.014 一律记为「不可排序」**，不得当结论；
- 训练侧散度：本批已知 13M 同配置两次可差 **+3.271pp**（另一条线，准确率口径）与
  `B vs B_rep` 的 1.367pp——**bpc 口径的训练散度未测**，因此「50M 的趋势」只作快照读；
- §12 三道对账：算术（主判据的代数化简，run.py 断言）、任务（三臂同一评测流/同一位置集）、
  口径（K 曲线与 13M 同方法；battery 与 probe 两套口径分开列，不混用）。

## 6. 成本

```
gpu_wall_budget: 115 min        # 含 20 步冒烟 + 3 臂训练 + 全部评测
```

- 预算唯一的数字，来自下发裁决（主 Agent 批准把 90 min 放宽到 ≈115 min）。
- **先实测步时再定预算**：20 步冒烟（每臂）→ 实测 ms/step → 若 3 臂全程超预算，按下面砍。
- **预算不够时砍什么（不砍判据线）**：① 砍 200-batch battery 腿（改 60 批）；
  ② 砍 K=2 档（保留 K=1/4/8）；③ 砍 JS 的重复批次。
  **禁止**砍：主判据的四个读数（`stock@K1` / `l1@K1` / `divl1@K8` / `divl1@K1`）、
  确定性路径对照、活性量。
  若连训练都跑不完：**三臂同步降步数**（保持阶梯内同步数），并把「13M 锚点 6000 步 vs
  50M 降步」作为不等字段写进 `analysis.md`。
- **可中断性**：三臂各自独立进程，每 200 步落盘；中断只丢当前臂。
- 峰值显存 ≤5GB net（`peak_gpu_mem_mb` 落盘；共享 4090 常占 8–12GB）。

## 7. 数据与脚本清单

| 项 | 内容 |
|---|---|
| 数据集 | enwik8（`.tmp/enwik8` → `data/enwik8_full.npz`，val = 后 10M 字符，训练从不碰） |
| 为什么是 enwik8 | V3 的判据是**集成增益 vs 税**的部署定价（bpc），不是长程能力主张；且 13M 锚点就是 enwik8 bpc ⇒ 换任务就失去「同方法趋势」。`SCHEMA §19` 的局部先验抗性针对**长程记忆/搬运**类结论，**本实验不主张该类结论**（见 §8） |
| 训练入口 | `python3 train_m1.py <arm> --config configs/<arm>.json --out-root <here>/train` |
| 评测入口 | `python3 run.py`（K 曲线 + 成对 JS + 确定性对照 + 配对 CI + 对账） |
| 冒烟 | `train_all.sh smoke`（每臂 20 步） |
| 落盘 | `train/<arm>/{results.json,train_log.jsonl,model.pt}`（`*.pt` 不入库）、`results.json`、`analysis.md`、`MODEL.md`、`controls/` |

## 8. 既有的依赖与耦合（只记录，不做组合实验）

- 上游：`G5-parallel-lines`（13M 的 L1/DIVL1 锚点）；`axes/G-policy/PRIOR_ART_ENGINE.md §5`。
- 本实验**只做单轴单格实验**（scope 声明），`stock→l1→divl1` 是**同一格内的受控阶梯**，
  不是跨机制组合；组合取舍属产品方 scope。

## 9. 产出

`results.json`（含 `criteria` / `checks` / `reconciled`）、`analysis.md`（人话摘要先行）、
`MODEL.md`（权重 sha256/体积/生成命令）、`controls/`、`reconcile_report.json`、
`mutation_report.json`。

---

## 10. 既有做法（prior art，`SCHEMA §18`）

**检索条件**：本环境**联网检索不可用**（未注册 web / `web_search` 工具）。因此下面凡外部系统
一律标 **`记忆·未核`**，只作设计方向、**不得作判据**；本地条目标 `本地实测` 并给路径。

1. **谁做过**：**Deep Ensembles**（Lakshminarayanan et al. 2017，`记忆·未核`）——独立训练多个
   模型再平均，报告「集成增益来自成员多样性」；**self-consistency**（采样多条推理链再投票，
   `记忆·未核`）；**ensemble diversity / negative-correlation 一类损失**（把「成员互相去相关」
   写进训练损失，`记忆·未核`）。本地：本仓 `G5-parallel-lines` 的 13M `L1`/`DIVL1` 对照
   （`本地实测`，`.scratch/08-tts-probe/`、`.scratch/09-m3-interactions/`）。
2. **为什么这么设计**：集成的误差只有在成员**误差不相关**时才会被平均掉；共享主干的成员
   天然高度相关 ⇒ 要么独立训练、要么显式付钱买多样性。
3. **它的实测结果**：外部在图像/分类上普遍有效（`记忆·未核`，**不引精确数字**）；
   本地 13M 语言模型上**有效但不足以回本**：`L1` K1→K8 1.6926→1.6454（增益 0.047），
   `DIVL1` 1.6842→1.6227（增益 0.062），而相对单线 stock 的 clean 代价是 +0.177
   （`本地实测`；`axes/G-policy/PRIOR_ART_ENGINE.md §5`）。
4. **我们的增量在哪**：
   - **换尺度**（13M → 51.8M，`SCHEMA §21` 的 (b)：我们设定与外部文献实质不同——共享主干
     + 逐 token 变路径，不是独立多模型）；
   - **把判据的基准写死**（`§17.14`：基准 = stock 单线、代价项同口径、primary = 读法 B），
     因此「回本没有」是一个**可判定的二元结论**，而不是叙事；
   - **三个精确控制**：确定性路径对照（增益应恰好 0）、成对 JS（增益是否只来自路径随机性）、
     stock 单线（算力价签 1× vs K8 的 8×）；
   - **不重复造轮子**：引擎/调度形态（tree attention、分页 KV 回滚）按 `§21` 判 (a)，
     直接引用 `PRIOR_ART_ENGINE.md §1–§2`，本实验不实现、不评测。
