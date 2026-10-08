# 2631-spec-three-arms · 对照与方差说明

## 1. 对照臂（SCHEMA §11：诚实对手，不用 oracle）

**对照 = 同 checkpoint 的「原生逐 token 自回归」**（`pathlm/decode.py::decode`，
内部走 `Decoder.step()` 逐 token），同一 prompt、同一批量（B=1）、**同一进程**、
与 spec 臂在同一个循环里**交替**计时。

- 对照物是**模型自身的贪心输出**，不是真值、不是 oracle。
- **明确不用**：批量 teacher-forcing 前向（那是 ticket 12 旧结论出错的地方——见
  `docs/REMEDIATION.md` 的 F_ENG 条目）。
- `results.json::checkpoints.*.native_is_plain_decoder_path` 就是这条口径的机器可核标记；
  `reconcile_numbers.py` 逐条核对它为 `True`。

## 2. 「错的对照」三类（SCHEMA §23 问题 3）

| 类 | 本实验的处置 |
|---|---|
| 同参数量（§15） | **不适用**：三臂**零新增参数**——它们读的是同一模型**既有**的 MTP 头（`T1`/`T2`/`T3` + `chain_cat`）。臂间没有容量差，故不存在「归因于容量」的替代解释。 |
| 同算力 | **这是本实验的核心**：三臂共享同一 forward、同一 K=3 验证，臂间算力**逐位相同**；等效算力净账 `c₃/(tok/fwd)` 只随接受率变。头部解析成本（`mtp_head_macs`）与臂无关。 |
| 同字节预算 | 不适用（无压缩）。 |

## 3. 方差与统计

- **接受率类（α₁、α₂\|α₁、τ、tok/fwd）**：按 **prompt 聚类的 bootstrap**
  （B=2000，seed=2631，percentile 95%）。**不把每轮草稿当独立二项样本**
  （复核 B2 的同类错误：同一序列的多轮共享同一模型与同一上下文）。
  每 prompt 另报轮数，便于复核重算。
- **墙钟**：同进程、同硬件、同批量、**交替重复** `--repeats`（本波实测 6）次；
  报①逐次重复的极差（max/min 之比）、②**按 prompt 聚类的配对** bootstrap CI
  （配对 = 同一 prompt 的 `spec/native` 比值的中位数）。
  单次 tok/s 不作结论。
- **臂间排序**：由**配对差**（`wall.paired_speedup_diffs`，同 prompt 同重复）的
  CI 决定；CI 含 0 ⇒ 记「**不可排序**」，不得宣称胜者（契约 §4 判定线 B）。
- **接受率的可排序带**：Δ < 0.014 不排序（SCHEMA 方差界）。
- **跨种子**：同 scale 两 seed 各训一次；逐臂报两次读数的极差与方向一致性。
- **数据 seed 与模型 seed 无关**：prompt 一律取 enwik8 val 的固定 offset。

## 4. 与生产实现的等值性（SCHEMA §9）

- 三臂**复用已入账的生产解码器** `pathlm/decode.py::decode_spec`
  （只经它自己的 `draft_fn` 钩子注入草稿）⇒ 批量验证、接受前缀、拒绝截断重锚、
  KV 语义都是**同一条代码路径**，本线没有重写解码器。
- `arm_decode` 抛 `n_mtp <= 2` 时不静默退化（`test_spec_arms.py` 钉住）。
- 逐位等价门禁：三臂发出流与 `decode` 逐位相同（`results.json::bit_identical_all_arms`）。
