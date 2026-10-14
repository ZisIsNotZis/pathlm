VERDICT: PASS

# REVIEW-2B.md — 轴 3 · 研究项 3.1.4 · 复核门第 2B 轮（fresh 独立复核，有 shell）

> 复核者 = fresh 独立上下文 subagent（**有 bash**），工作目录 `/home/z/vibe/pathlm`，只读被审产物，本轮只写本文件。
> 被审对象 = `axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/`（研究项 `3.1.4`，7 臂 × 12k）。
> 上一轮（builtin reviewer，无 shell）报告 = `.scratch/64-axis3-width/REVIEW-2.md`，判 FAIL（窄范围，原因 6 项执行类检查跑不了）。**本报告不引用其结论作为证据**，只据本轮实跑输出判定。
> 复核时间盒 ≤25 min；命令超时 300s。全部命令**实跑**，输出见下。

## ① 人话摘要（3 行）

1. 派单的 6 项执行类检查**全部跑过且与期望一致**：pytest 13 passed、Δ 重算与落盘逐位对上、三条第 1 轮 blocker（B1/B2/B3）在 `analysis.md` 里已真改、T5 初始化控制 6 组逐层全 True、契约三件与 `49f697f` sha256 逐位相同且 mtime 早于起跑、本轴越界扫描 0。
2. 结论**没有越证据**：`analysis.md` 已无「四个 Δ 全小于 0.0098 ⇒ 分配效应被噪声吃掉」的伪命题，判词明确写「不可排序（H0）是**阈值规则**的判词，**不等于效应已被排除**」，并保留两条不可排除的反假设限定（非单调 schedule 未搜索；共享前缀初始化 × 通道子集交互不可识别）。
3. **故本轮判 PASS**：无阻塞项、无实质 blocker；残留的只有并发写入导致 HEAD 在复核期间发生的位移（只影响「被审 revision」的定位，不影响被审产物本身的一致性，见 ④⑤）。

## ② 逐项检查表（命令 / 真实输出摘要 / 结论）

| # | 检查 | 命令 | 真实输出摘要 | 结论 |
|---|---|---|---|---|
| 1 | pytest 等值性门 + 变异 | `cd axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m && python3 -m pytest test_2641_ffn_ratio.py -q` | `............. [100%]` / `13 passed in 12.12s` | **通过**（13 passed，与期望一致） |
| 2 | 从 7 个 `results_<arm>_s<seed>.json` 重算 Δ 并逐项比对 | 自写脚本 `/tmp/recheck_deltas.py`（读 `eval.clean_bpc`） | 见下方 2a–2d | **通过**（全部 MATCH / ALL-EQUAL） |
| 3 | B1 事实复核 | `grep -n` on `analysis.md` | 见 ③ | **成立** |
| 3 | B2 事实复核 | `grep -n` + 读落盘 `results_APYR_s1.json` | `analysis.md:110` = 2.4684/2.7209；落盘 depth6 s1=2.4684、s0=2.7209 | **成立**（非 2.4699） |
| 3 | B3 事实复核 | 算术 + 读落盘 `params_ffn` | 2,097,152+4,096+2,048 = 2,103,296 = 落盘 `params_ffn` | **成立** |
| 4 | T5 初始化控制（实跑脚本） | `CUDA_VISIBLE_DEVICES="" python3 /tmp/t5_init_control.py`（`import run`，seed 0/1 × U2/PYR/APYR） | `T5 RESULT: PASS`，6 组逐层全 True + U4 自比 True | **通过** |
| 5 | T6 契约未被事后编辑 + mtime | `git show 49f697f:<文件>\|sha256sum` vs `sha256sum <工作树>`；`stat -c '%y %n'` | 三份契约 sha256 逐位 SAME；契约 mtime 17:58:24/17:59:41/18:00:10，起跑 18:00:37 | **通过** |
| 6 | T9 本轴越界扫描 + SSOT 校验 | `git diff --name-only 49f697f..9e9b291 \| grep -vE '...' \| wc -l`；`python3 tools/summary.py --check` | 越界 = `0`；`OK：46 个 project.yaml 全部符合 docs/research-schema.md。`（exit 0） | **通过** |

### 2a. pytest 真实输出（check 1）
```
.............                                                            [100%]
13 passed in 12.12s
```

### 2b. Δ 重算 vs 落盘（check 2，命令 = `python3 /tmp/recheck_deltas.py`）
```
== recomputed clean_bpc from per-arm files ==
  ('U4', 0): 1.5314  ('U2', 0): 1.5709  ('U2', 1): 1.5611
  ('PYR', 0): 1.573  ('PYR', 1): 1.575  ('APYR', 0): 1.5644  ('APYR', 1): 1.5601
== delta check vs results.json::P1_P2_deltas_by_seed ==
  seed 0: delta_up=-0.0021 (ref -0.0021) delta_down=0.0065 (ref 0.0065) -> MATCH
  seed 1: delta_up=-0.0139 (ref -0.0139) delta_down=0.001 (ref 0.001) -> MATCH
== results.json arms.clean_bpc vs per-arm files ==
  U4-s0/U2-s0/U2-s1/PYR-s0/PYR-s1/APYR-s0/APYR-s1 -> 全部 MATCH
== P1_verdict_by_seed ==
  seed 0: 不可排序（H0）：|Δ| < 0.014 ⇒ 逐层 FFN 分配在本尺度无独立增益
  seed 1: 不可排序（H0）：|Δ| < 0.014 ⇒ 逐层 FFN 分配在本尺度无独立增益
== per-arm bit-equality of sum_r / macs.ffn / params_ffn across U2,P YR, APYR ==
  sum_r: U2:16->PYR:16->APYR:16 -> ALL-EQUAL
  macs.ffn: U2:2097152->PYR:2097152->APYR:2097152 -> ALL-EQUAL
  params_ffn: U2:2103296->PYR:2103296->APYR:2103296 -> ALL-EQUAL
DELTA/RESULTS CHECK: PASS
```
判词表 `P1_verdict_by_seed` 两条均「不可排序（H0）」；`equal_ffn_budget_check.pass = true`（Σr=16 / ffn_macs=2,097,152 / params_ffn=2,103,296 三臂逐位相等）。**与落盘逐项一致。**

### 2c. T5 初始化控制真实输出（check 4）
```
seed=0 arm=U2   trunk all-bit-identical (embed/pos/ln/qkv/proj): True (66 params)
                FFN prefix: True  per-layer={0..7: True}
seed=0 arm=PYR  trunk ...: True (66 params)   FFN prefix: True  per-layer={0..7: True}
seed=0 arm=APYR trunk ...: True (66 params)   FFN prefix: True  per-layer={0..7: True}
seed=1 arm=U2   trunk ...: True (66 params)   FFN prefix: True  per-layer={0..7: True}
seed=1 arm=PYR  trunk ...: True (66 params)   FFN prefix: True  per-layer={0..7: True}
seed=1 arm=APYR trunk ...: True (66 params)   FFN prefix: True  per-layer={0..7: True}
（另 U4 自比：seed=0/1 trunk=True ffn=True per-layer={0..7: True}）
T5 RESULT: PASS
```
trunk 逐位比较范围 = `embed.weight`、`pos_embed.weight`、各层 `ln1.{weight,bias}`、`ln2.{weight,bias}`、`qkv.{weight,bias}`、`proj.{weight,bias}`（8 层 × 8 + 2 = 66 项）与 `layer_mix`；FFN 逐位前缀 = `mlp[0].weight == U4[:h]`、`mlp[0].bias == U4[:h]`、`mlp[2].weight == U4[:, :h]`（h = r·256）。**6 组全部逐层 True。**

### 2d. T6 sha256 对照（check 5）
```
README.md             git=dd1e8201687e...21faa3  wt=dd1e8201687e...21faa3  SAME
train_arms.sh         git=353988a3e2c8...587ea    wt=353988a3e2c8...587ea    SAME
test_2641_ffn_ratio.py git=667613432f40...a7a8e0 wt=667613432f40...a7a8e0 SAME
combined(brace)  git=21c4a345a477...93c1  wt=21c4a345a477...93c1  SAME
mtime: README.md 2026-10-13 17:58:24.068 / train_arms.sh 17:59:41.495 /
       test_2641_ffn_ratio.py 18:00:10.820 / runs/U4-s0-12k 18:08:27.252
起跑 = train_arms_driver.log 首行 `[2026-10-13T18:00:37+08:00] lock acquired`
```
契约三件与 `49f697f` **逐位相同**；三份契约 mtime 均 **早于**起跑 18:00:37。`analysis.md`（21:23）/`MODEL.md`（19:19）与分析产物不同属允许。

### 2e. T9 + SSOT 校验（check 6）
```
$ git diff --name-only 49f697f..9e9b291 | grep -vE '^(axes/3-width/|\.scratch/64-axis3-width/)' | wc -l
0
$ python3 tools/summary.py --check
OK：46 个 project.yaml 全部符合 docs/research-schema.md。   (exit=0)
$ git log --oneline -3
f3d70d3 轴 3 Phase B：固化后续派单契约 DISPATCH-B.md（W-B1/W-A2/W-B2/W-C/W-D）
5d74add 轴 3 · 3.1.4 复核门第 2 轮：落盘判词 FAIL（窄范围：复核者无 shell）并处置 2 条非阻塞
4638882 E2（与 E4 统一）：逐层 FFN 隐藏宽 schedule（引擎 + CLI + 单测 + 能力面）
```
区间内 40 个变更路径**全部**落在 `axes/3-width/**` 或 `.scratch/64-axis3-width/**`（含 3.1.4 全部产物、`3.1-active-width/README.md`、`3.2-readout/README.md`、`project.yaml`、`REVIEW-1.md`、`TICKET.md`）；越界文件数 = 0。

## ③ blocker 与 `file:line`

**无 blocker。** 三条第 1 轮事实的复核证据（`file:line` + 真实输出）：

- **B1（伪命题已去 + 判词已弱化 + 两条反假设限定）** — `grep -n "噪声吃掉" analysis.md` → **无匹配**（exit 1），伪命题不在产物中。`0.0098` 仅出现在已正确改写的三处：`analysis.md:35`（§0「散列不能当作 |Δ| 的上界」）、`:89`（§3-P4）、`:102-103`（§4）与 `:176`（§7.1），均用作**参照量**并显式标注**不是** |Δ| 上界。判词：`analysis.md:104`「『不可排序』是**阈值规则**的判词」、`analysis.md:202`（§8）「判 H0 是**阈值规则**的判词，不等于「效应已被排除」」。两条反假设限定在位：`:172-176`（§7.2 非单调 schedule 未搜索 ⇒「非均匀分配一律无用」超出本证据）与 `:178-181`（§7.2 共享前缀初始化 × 通道子集交互不可识别）。`project.yaml:16`（failure_analysis）/`:17`（conclusion）同改并三条限定齐备。
- **B2（APYR-s1 第 6 层 = 2.4684）** — `analysis.md:110`：「APYR 则相反（第 6 层 **2.4684/2.7209** = s1/s0）」。落盘实测：`results_APYR_s1.json` depth6 = **2.4684**、`results_APYR_s0.json` depth6 = **2.7209**（脚本实读 `eval.bpc_by_depth`）。`grep -n "2\.4699"` → **无匹配**（旧值已清）。
- **B3（参数差余项 = 4,096）** — `analysis.md:137`「差 2,048 = 8 层 × `lin2` 偏置 `d=256`」、`:159-160`「2,101,248 = 2·(4−2)·256²·8（= **2,097,152**）+ (4−2)·256·8（`lin1` 偏置，= **4,096**）」。算术实跑：`2,097,152 + 4,096 + 2,048 = 2,103,296` = 落盘 `results_U2_s0.json::params.params_ffn`；`6962688 − 4861440 = 2101248 = 2097152 + 4096`。`grep -n "1,024"` → **无匹配**（旧值已清）。

非阻塞观察（**不改变判词**，不计 blocker）：
- `analysis.md:40` 的中间深度副产品读数（PYR L6 1.9727 / L7 1.7492）现标 `(seed 0)` —— 系 `5d74add` 处置的非阻塞 N4，已修，与 `:109` 写法一致。
- 目标目录工作树相对当前 HEAD **干净**（`git status --porcelain <实验目录> <project.yaml>` 无输出）；`git diff 9e9b291 -- analysis.md` 仅 1 行 = 上述 N4 标注，且已提交于 `5d74add` ⇒ **无未说明的本项改动**。
- `.scratch/64-axis3-width/TICKET.md` 有未提交改动、`.scratch/64-axis3-width/2642-uniform-narrowing-process.md` 与 `axes/3-width/3.1.1-uniform-narrowing/experiments/2642-uniform-narrowing-13m/` 为未跟踪 —— 属并发的其它工单/其它格（非被审对象），不构成越界。

## ④ 未跑项与理由

**无未跑项。** 派单 1–6 全部实跑（1 pytest、2 自写重算、3 grep/实读、4 T5 脚本、5 git show/stat、6 git diff/summary）。所有命令退出与输出已在上表逐条落真实值。

（说明：`runs/` 被 `.gitignore` 覆盖，未做 `git show` 对照，但派单未要求；T6 只要求三份契约文件 + `runs/U4-s0-12k/` mtime，后者已 `stat`。）

## ⑤ 残留不确定

1. **被审 revision 定位**：复核期间 HEAD 由 `4638882` 位移到 `f3d70d3`（其它轴 lead/worker 在同一工作树并发提交）。被审**产物本身**在所有观测点一致（目标目录 `git status` 干净、契约三件 sha256 与 `49f697f` 逐位相同），故不构成 blocker；但「复核者读到的状态 == 派单所指 revision」只能靠 `49f697f..9e9b291` 区间与干净工作树担保，无法用单一哈希钉死。
2. **`MODEL.md` 的哈希表为自报**：`MODEL.md` 里 `README.md/test_2641_ffn_ratio.py/train_arms.sh` 的哈希由 `collect_logs.py` 训练后生成，属 writer 自报；本轮已用 `git show 49f697f` 独立复核并逐位对上，故不影响判词。
3. **T5 只证「同 seed 构造下 trunk 相同 + FFN 为前缀」**：不覆盖训练期随机性/数据流（那由 `results.json::code_hash_run_py` 与 `config.json` 承载，非本轮 6 项范围）。
4. 本轮**未改动任何被审产物**（仅写本文件）；未提交 git；未触碰 `axes/**`/`docs/**`/`tools/**`/`pathlm/**`。pytest 产生的 `__pycache__` 由 `.gitignore:2` 覆盖（`git check-ignore` 确认），未污染仓库。
