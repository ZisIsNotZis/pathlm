# 2618 · 显式写入通道：无通道 / plain 写 / delta 写（13M · 合成长程任务）

> 契约来源：`.scratch/35-carry-channel/issues/01-carry-channel.md` 的**契约 U**。
> 本目录实现它。**判据线在 §4 逐字抄录，不得修改**（出现跨批修正时另开 ERRATA）。

## 1. 命题

**在局部先验无法补全的合成任务上，给模型一个显式的写入通道，信息能不能被搬过
训练窗口？** 并且，**「只搬上次没搬的」（delta 残差写）是否优于「plain 累加写」？**

一句话可证伪的形式：*「U2（delta 写）的关联回忆准确率 > U0（无通道）+ 5pp 且二项
95% CI 不重叠」*。

## 2. 单因素声明

- **基底**：本仓 stock dense decoder-only trunk（`DepthARModel(dense=True)`：tied 读出 +
  无 final LayerNorm + 顺序残差），13.5M（d256×12 层×8 头×mlp6×T256）。
- **唯一变量**：
  - `U0 → U1`：**有无写入通道**（U0 无通道；U1 有 N 个记忆槽，逐位置 plain 累加写）；
  - `U1 → U2`：**写入规则**（U2 把写改成 delta 残差式），**逐位等参数**（同一组张量、
    只改更新式；单测 + 运行期断言各钉一遍）。
- 其余逐字段一致（含 seed=0）：模型规格、训练配方、数据生成器、可见性掩码、步数、批大小、
  学习率、优化器、评测协议。逐字段对照见 `analysis.md` §3。
- **必要性例外（显式记账）**：为了让「需求」真的存在，训练与评测都使用
  `train_window=16 / train_anchors=0` 的可见性掩码（ticket 32-O 既有开关）。
  这不是第二个机制，而是本契约指定的任务与可见性设定（三臂完全相同）。

## 3. 干预定义（必须写清，因为它是「实现」而非「标签」）

**单共享记忆槽**（`pathlm/depth_ar.py::CarryChannel` + `CarryChannelBlock`）：

```
写入（每位置一次，来自输入流 h0）：k̂ = normalize(W_k h0)，v = W_v h0，
                                    β  = σ(W_β h0 + b_β)
  plain： S_t = S_{t−1} + β_t v_t k̂_tᵀ
  delta： S_t = S_{t−1}(I − β_t k̂_t k̂_tᵀ) + β_t v_t k̂_tᵀ = S + β(v − S k̂)k̂ᵀ
读出（每层各一次）： a ← a_SDPA(窗口掩码) + γ_ℓ ⊙ mem_t ，mem_t = S_t q_t
```

- **读出约定（显式声明）**：位置 t 读到的是 **`S_t`（含 t 自己的写入）**——read-after-write，
  与因果 softmax 注意力把对角项计入同一约定；单测
  `test_carry_read_convention_includes_own_write` 钉住，变异验证 M2（去掉自写入）抓红。
- **因果性**：位置 t 的读出只用 `S_{≤t}`；训练实装是分块并行形式（WY 表示 + 三角求解），
  与逐位置参考实现逐位对齐（fp64 1e-14 内），单测 `test_carry_chunked_matches_sequential`
  与 `test_carry_channel_is_causal_only` 钉住，变异验证 M1（把窗口掩码放宽一格 ⇒ 未来泄漏）抓红。
- **恒等起步（单边零）**：每层读出增益 `γ_ℓ` 零初始化 ⇒ 起步**逐位恒等**于无通道基线
  （U0），而 `∂L/∂γ ≠ 0` ⇒ 非鞍点（SCHEMA §10）。单测
  `test_carry_channel_identity_start_matches_no_channel` 同时钉住「主干参数逐位相同」。
- **显式实现决策（SCHEMA §17.8，超预算后的取舍，已获主 Agent 裁决批准）**：
  per-layer（12 层各一个 `dh×dh` 状态）实装实测 **U1 207 ms/step、U2 396–425 ms/step**
  （基线 42–51 ms/step）⇒ 三臂 ≈2 小时，超预算 2.4×；根因是算子的 CUDA 派发 /
  autograd 节点开销（实测 `solve_triangular` 0.04 ms、`bmm` 0.10 ms GPU 时间，但每个
  算子 CPU 派发 60–150 µs、每个 autograd 节点 ~180 µs；per-layer 每步 ≈1000 算子），
  **不是 FLOPs**。改为**单共享槽**（扫描从 12 次降到 1 次）后实测 U1 60.7 / U2 73.1 ms/step，
  三臂 ≈37 min ✓ 落在预算内。单共享槽也是 MQAR / 线性注意力家族的标准形态
  （一个状态层 + 若干全注意力层），便于与既有做法对照；代价是写入表达力较弱
  （写入来自输入流而非各层隐状态）。详见 `analysis.md` §4 与 `stability.md` §4。

## 4. 判据线（**逐字抄录**自契约 U，一字未改）

> 主腿在**准确率口径**（不用 bpc——训练侧 bpc 噪声 ~0.0156 已接近方差界）：
>
> 1. **U0 必须 ≈ chance**（关联回忆 chance = 1/词表子集大小）：这是「任务+窗口确实制造了需求、
>    且不搬就真答不出」的**前提校验**。若 U0 显著高于 chance ⇒ 设置有问题（比如窗口没生效、
>    或任务有漏洞），**先修设置再判**。
> 2. **主判据**：`U2`（delta 写）的关联回忆准确率 **> U0 + 5pp 且二项 95% CI 不重叠** ⇒
>    「显式写入通道能让信息被搬过去」成立。
> 3. **并列判据（写规则优劣）**：`U2` vs `U1` 的差 **≥ 3pp 且 CI 不重叠** ⇒ delta 写优于 plain 写；
>    否则记「两种写规则不可分辨」。
> 4. **毁源对照臂（强制）**：评测时把源片段内容毁掉（用电池的 `corrupt_span`）；
>    任何一臂若在毁源后**仍**保持高准确率 ⇒ 该臂的读数不是来自「搬过去的信息」，**该臂判读作废**。
> 5. **辅助腿**：变距离回忆的距离曲线（U0/U1/U2 三条），报「从多远处开始能答对」。
> 6. 必须附：三臂**同方法** loss 斜率、训练侧重跑散度（1 臂重跑即可）、解析/实测内存分开报。

（以下为**摘要**，非逐字：三臂规格表、预算、交付物清单、scope 声明——见 §5–§9。）

## 5. 三臂规格（契约 U 的表，摘要）

| 臂 | 可见性（训练+评测一致） | 写入通道 | 参数量 |
|---|---|---|---|
| `U0` 基线 | 窗口 16 + 锚点 0 | 无 | 13,536,768 |
| `U1` plain 写 | 同 U0 | 单共享槽，plain 累加写 `S ← S + βvk̂ᵀ` | 13,801,064 |
| `U2` delta 写 | 同 U0 | **同参数量/同形状**，写入改 delta 规则 | 13,801,064 |

（另有 `U1rep`：与 `U1` **逐字段相同**的同 seed 重跑臂，只用于判据 ⑥ 的「重跑散度」；
读数见 `divergence.json` 与 `stability.md` §3，不参与主判据。）

12k 步 / bs32 / T256 / lr 1e-3 / AdamW + cosine / seed 0；数据 = 多查询关联回忆
（`assoc_recall_multi_batch`，K=8 对、Q=8 查询、查询间隔 14 filler）。

## 6. 指标表（多维，逐项在 `analysis.md` 分析）

| 维度 | 量 |
|---|---|
| 它解决什么问题 | 训练窗口外的信息能否被显式通道搬进可达范围（长上下文「丢了但搬了」） |
| 代价 | 参数量 +264,296（+1.95%，U1=U2 逐位相同）；墙钟/步时倍数（实测）；显存（解析 vs 实测） |
| 有效性 | 关联回忆准确率（含 CI）、变距离回忆距离曲线、毁源对照臂 |
| 收益与用法 | 窗口外可达性（far_read_norm_frac）、每层读出占比（read_share） |
| 可控性 | 写门 β 均值、写入量 `‖ΔS‖/‖S‖`、单边零逃逸（read_scale） |
| 副产品 | 状态大小与 T 无关（有限状态的可解析边界） |

## 7. 成本（§8 的唯一预算数字 + 砍什么）

`gpu_wall_budget: 50 min`（含三臂训练 + 评测）。

| 项 | 实测 |
|---|---|
| U0 训练 | **11.12 min**（`results.json::train_results.U0.wall_minutes`） |
| U1 训练 | **13.48 min** |
| U2 训练 | **16.26 min** |
| 评测电池 | **2.01 min**（`results.json::cost.wall_minutes_total_eval`，864 个前向批） |
| 三臂 + 评测 | **≈43 min** ✓ 在 50 min 内 |
| 重跑臂（判据⑥，U1rep） | **17.92 min** ⇒ 含重跑臂总计 ≈62 min（超预算，按 §成本 的授权先报数：主判据所需的最小集合是三臂 + 评测 = 43 min） |
| 峰值显存 | 训练实测 **2015 / 2170 / 2282 MB** 净（预算 ≤5 GB 净）；评测 521.5 MB |

**预算不够时砍什么**（契约 §成本 + SCHEMA §17.7）：砍重跑臂（判据 6 的散度）>
砍 K 扫描档数 > 砍距离档数；**不砍判据线**。若 D 线实测超 25 min/臂：按契约只跑
U0+U1 并如实标注 U2 顺延（本实验未触发）。

## 8. 数据与脚本清单

- 数据：**不用 enwik8**（契约要求）。训练与评测都用
  `tools/evalkit/longrange_tasks.py` 的合成任务生成器（该文件只增不改：新增
  `assoc_recall_multi_batch`）；random seed 全部显式。
- 训练：`python3 train_depth_ar.py <arm> --d 256 --layers 12 --heads 8 --mlp-mult 6
  --dense --steps 12000 --batch-size 32 --seed 0 --lr 0.001 --train-window 16
  --train-anchors 0 --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8
  --carry-write {none,plain,delta} --out-root train`
- 评测：`python3 run.py`（本目录）；配置 `config.json`；结果 `results.json`。
- 落盘：`train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json,model.pt}`（四臂：U0/U1/U2/U1rep）、
  `results.json`、`run_log.jsonl`、`run_log.txt`、`premise_check.json`、`divergence.json`、
  `analysis.md`、`MODEL.md`、`stability.md`、`mutation_report.json`、`reconcile_report.json`、
  `smoke/`（含 per-layer 实装的步时/显存冒烟日志 detA/detB 训练侧确定性对照）
  （权重 `*.pt` 不入库）。

## 9. 失败决策树（测量 → 实现 → 规模 → 设计）

1. **测量**：U0 显著高于 chance？→ 查窗口是否生效（`window_mask_probe` 活性量）、
   查询-key 距离分布、局部先验读数；先修设置。
2. **实现**：通道活性为 0（read_share/β/state_fro 任一为 0）→ 先跑变异验证（M1–M6）
   排除死开关与未来泄漏。
3. **规模**：通道活性非零但三臂都 ≈chance ⇒ 该任务/该尺度上「不划算」
   （SCHEMA §19：不得表述为「机制不可能」）。
4. **设计**：U1/U2 高于 chance 但**毁源不掉** ⇒ 读数不是搬来的信息，该臂判读作废。

## 10. scope 声明

本研究**只做单轴单格实验**，基底恒为 stock 对照；**不做组合实验**——组合留白是显式
声明，不是遗漏。`deps` 只记录已知耦合方向（M1 状态载体 ← A6 混合算子；A9/A3 窗口/锚点
提供「需求」侧）。组合取舍属具体产品方 scope。

## 11. 产出与复核

见 §8 的落盘清单；本实验完成后按 SCHEMA §14 另派 fresh-context 只读复核者（契约 V）。
