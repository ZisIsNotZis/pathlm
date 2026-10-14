# 2643 · 判读（首段 = 人话摘要）

## 0. 人话摘要（必读）

**两个新增臂（`U` = 完全解绑读出、`PR` = 部分绑定读出）这一轮没有跑出任何质量读数。**

原因不是实现问题，是**单卡 4090 的互斥锁排队**：本实验的驱动从 21:38 起排队等
`/tmp/pathlm_gpu.lock`，直到 22:40 的硬时限，锁始终被**别的轴**的 12k / 长任务占着
（先后见到 `wp2652-2.2.1`、`2642-uniform-narrowing`、`wp6.3.2-fill_causal`、
`wp2672-snap2k-*` 等持锁者；`wp6.3.2` 的 supervisor 循环甚至连续持锁 30 min 以上）。
本实验三次等锁都没排上：21:38–21:53 第一版驱动等了 ~15 min（因修脚本重启，未计时到顶）、
21:53–22:18 第二版驱动到 25 min 上限**超时 `rc=9`**、22:25–22:40 单臂版到硬时限放弃。
全程**没有把「未跑」写成读数**，**没有杀任何他人进程**，**没有误回收他人锁**。

所以本目录里**没有** `U`/`PR` 的 clean bpc：`results.json` 的 `arms_run` 是空的，
`arms_unrun = ["U-s0", "PR-s0"]`，`n_seeds = 0`。任何「解绑是否更好」的结论**尚不可判**。

已完成、可独立复核的部分（全部是 CPU 侧或 8 s 的纯推理，不占训练锁）：

| 已完成 | 证据 | 结果 |
|---|---|---|
| 预登记契约（P1–P4、0.014 带、复用三道门、大词表档不可判声明） | `README.md` | 冻结于 commit `48c0808`，早于任何臂起跑 |
| 复用门① **实现层逐位** | `test_2643_tied_untied.py::T1` | 「解绑开关关闭」时 `T` 臂最终深度 logits 与 stock `DepthARModel` **逐位相同** |
| 复用门① **锚载入逐位** | `test_2643_tied_untied.py::T2` | 2641 `U4-s0` checkpoint 载入 `T` 臂：键**无缺无余**、逐张量逐位相同 |
| 复用门② **配置层逐字段** | `compare_anchor_config.py` + `anchor_config_gate.txt` | 17 个字段全等，**不等字段 0 个**（可失败：`mlp_mult: 4→2` ⇒ exit 1） |
| 复用门③ **口径** | `results_T_s0.json` | 用同一 `depth_curve` 口径重评 2641 锚：clean bpc **1.5314**，与记录**逐位复现**；`--no-train` 两次重评逐位一致 |
| 机制单测（含可失败的变异） | `test_2643_tied_untied.py` | 8 passed：`T3` 解绑头真在用（探针能变红）、`T4` `PR` 恒等起步、`T5` 单边零非鞍点（双侧零 ⇒ grad 恒 0）、`T6` 参数账、`T7` 因果 |

**下一步最小实验（预算内极致的最小补充）**

- 跑 `U:0` 一臂 12k：**≈ 8 min GPU**（2641 同配置实测 `wall_minutes 7.76`、`peak_gpu_mem_mb 935.3`）
  + 评测 ≈ 20 s；`PR:0` 同价。两臂合计 ≈ 16–17 min GPU（含臂间 61 s）。
- 一条命令：`STEPS=12000 ARMS="U:0 PR:0" bash train_arms.sh && python3 run.py --aggregate`。
- 判据**已经冻结在 `README.md` §4**，无需改契约：跑完直接读 `results.json` 的
  `P1_deltas` / `P1_verdict` / `P2_prob0` / `P3` / `big_vocab_undecidable`。
- 60 步冒烟（`rc=0`）已内建在驱动里，是 12k 的**前置**（本实验没跑成，故也没跑 12k）。

## 1. 这条研究问的是什么

普通 decoder-only transformer 常把「输入 embedding 表 `E`」和「输出 unembed 表」绑成
同一张表（tied）：`logits = h·Eᵀ`。这样省一份 `vocab×d` 参数，而且给「概率 = 状态在
embedding 基上的坐标」一个漂亮的解释。解绑（untied）多一份独立的 `U`，让输出头能自由
重排状态空间；部分绑定（`E + A·B`，低秩校正）是两者之间的连续档。

本仓 `vocab=206`、`d=256` ⇒ 解绑只多 **52,736 参数（总量 0.76%）**，所以本条在本尺度
几乎测不到参数账，测到的是**「读出自由度」这一结构本身**是否有独立质量增益。

## 2. 复用三道门（`SCHEMA.md` §9）

### 2.1 门① 实现层 equality（可失败单测）

- `T1`：`arm="T"` 的覆写表达式与 stock `readout` 逐字相同 ⇒ 前向最终深度 logits
  与 stock `DepthARModel` **逐位相同**；同 seed 下 `state_dict` 逐张量逐位相同。
- `T2`：载入 `2641/runs/U4-s0-12k/model.pt`（sha256 `d4e6e33ef55d08f0cb71aadcc0c2c810…`，
  与 2641 `MODEL.md` 记录一致）后**键无缺无余**、逐张量逐位相同。
- 说明：「与 2641 的**初始化**逐位对照」在措辞上只能落在**同 seed 同构造路径**
  （`T1` 已证）；`model.pt` 是**训练后**权重，不可能与本次初始化逐位相等，
  可对照且已对照的是「载入后逐位相同」（`T2`）。

```
python3 -m pytest test_2643_tied_untied.py -q     # 8 passed
```

### 2.2 门② 配置层 equality（逐字段）

见 `anchor_config_gate.txt`（由 `compare_anchor_config.py` 生成）：17 个应等字段
（`d/layers/heads/seq/mlp_mult/dense/dropout/norm_cap/topology/exit_mode/fill_kv/
steps/batch_size/lr/train_tokens/eval.metric/seed`）**全等，不等字段 0 个**。
脚本可失败：`PATHLM_2643_MY_CONFIG=/tmp/2643_bad_config.json …`（把 `mlp_mult` 改成 2）
⇒ exit 1（已实测）。

### 2.3 门③ 口径标注

历史同配置对照 + 等值性证明 + 已知口径：评测 = `depth_curve` **最终深度** clean bpc，
`n_batches=20`、`batch_size=16`、eval generator `seed=2`；训练 `steps=12000`、`seed=0`、
`batch_size=32`、`lr=1e-3`；数据 `data/enwik8_full.npz`（前 90M train / 末 10M eval，vocab 206）。
`results_T_s0.json.reuse` 里 `reproduces_recorded = true`（1.5314 逐位复现）。

## 3. 臂结果

**未跑。** `arms_run = []`；`arms_unrun = ["U-s0", "PR-s0"]`。
下表只填**已实测**的锚（复用，无新训练），其余留空——不填任何未实测数字：

| 臂 | 输出读出 | 新增参数 | 总参数 | clean bpc | 每 token 等效 MACs | 状态 |
|---|---|---|---|---|---|---|
| `T`（锚，复用 2641 `U4-s0`） | `logits = h·Eᵀ` | 0 | 6,962,688 | **1.5314**（本轮重评逐位复现） | 7,392,768 | 已复评（未训练） |
| `U` | `logits = h·Uᵀ` | +52,736 | 7,015,424 | — | 7,392,768 | **未跑（锁等待超时 rc=9）** |
| `PR` | `logits = h·(E+A·B)ᵀ` | +14,784 | 6,977,472 | — | 7,415,552（在线低秩） | **未跑（锁等待超时 rc=9）** |

## 4. 判据现状

| 判据 | 现状 |
|---|---|
| `P1` Δbpc（`U`/`PR` − `T`）按 0.014 带 | **不可判**：无 `U`/`PR` 读数 |
| `P2` 参数账 | 已定（解析）：`U` +52,736、`PR` +14,784；prob0 一致性只在 `T` 上有构造性 1.0 |
| `P3` 评测确定性 / 因果性 / 活性 | 锚上**已通过**（`results_T_s0.json`）；新臂未跑 ⇒ 未验证 |
| `P4` 否证线 | **不可判**（需要 `P1`） |
| 大词表档 | **不可判**（本仓 `vocab=206`；解绑的真实参数账只在 `V≈1e5` 才显著） |

## 5. 失败/受阻归因（按契约 §9 的决策树）

- **不是实现问题**：`T` 锚的完整评测链（深度曲线、重评确定性、因果、prob0）8 s 跑通，
  8 个单测全绿——评测与构造路径都已被证明可用。
- **是资源排队**：GPU 锁被其它轴长期占用（含 `wp6.3.2` 的 supervisor 循环，单次持锁 30+ min）。
  本实验等锁上限 25 min / 硬时限 22:40，三次尝试均未排上。
  按契约 §9「锁等不到就如实写未跑，不编读数」处置（同时满足派单的「22:40 仍未拿到锁 ⇒ 停止等锁」）。
- **未做**：`U`/`PR` 的 60 步冒烟与 12k 训练（`rc=9`，连冒烟都没拿到锁）。

## 6. 附录：证据路径与原始数字

| 文件 | 内容 |
|---|---|
| `results_T_s0.json` | 锚复评全量（逐深度 bpc、prob0、因果、重评一致性、checkpoint 哈希与键审计、`reuse` 块） |
| `results.json` | 判据骨架（`arms_run: []`、`arms_unrun`、`n_seeds: 0`、`variance_bound: 0.014`、`big_vocab_undecidable: true`） |
| `anchor_config_gate.txt` | 复用门② 逐字段对照表（不等字段 0） |
| `run_log.txt` | 锁取放时间戳、free 读数、§20.1 一次性并发例外登记、超时记录 |
| `test_2643_tied_untied.py` | 8 个自包含单测（含可失败的变异用例） |
| `compare_anchor_config.py` | 复用门②脚本（可失败） |
| 训练日志 | **无**（臂未起跑；`runs/` 下没有 `U-s0-*`/`PR-s0-*`） |
