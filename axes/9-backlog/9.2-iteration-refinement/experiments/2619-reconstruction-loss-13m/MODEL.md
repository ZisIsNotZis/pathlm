# MODEL · 2619 训练信号（重建损失）三臂对照（13M dense）

权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出哈希、体积与生成命令。
sha256 由 `fill_config_hashes.py` 回填进 `config.json`，`run.py` 每次加载**断言**。

## 三臂

| 臂 | 文件 | 训练信号 | 参数 | 体积 | SHA-256 |
|---|---|---|---|---|---|
| `A` 无损失 | `train/A/model.pt` | 纯 CE（λ=0） | 13,589,504 | 54,404,935 B | `b718d1b66237283251c2152efe112a3f80b08f833a48a01930686ad5ada9f898` |
| `B` 有损失 | `train/B/model.pt` | CE + 1.0·重建 | 13,589,504 | 54,404,935 B | `4db1db26a5bd23122c0ae3b802f7cd938d46f7790c6b474d359d43c02de9f886` |
| `B_rep` 重跑（c3） | `train/B_rep/model.pt` | **与 `B` 逐字段相同**（同 seed 重跑） | 13,589,504 | 54,404,935 B | `b114874e04c24d67e6fe4f3f0aa2ca25e8f31990e9936b1863f3f37797ec6ed9` |

- **两臂参数量逐位相同**（同一组张量、同一初始化；解码头建在参数 init 循环**之后**，
  不消耗其余参数的随机数）：`run.py` 每次运行都做「同 seed 重建有头/无头两模型 ⇒
  主干参数与 logits 逐位相同」的自检并落盘 `results.json::selfchecks`。
- **唯一变量 = 训练信号**：`A` 的解码头存在但**未被训练**（λ=0 ⇒ `add_recon_term`
  原样返回主损失、`p.grad is None` ⇒ AdamW 整条跳过）。
- 新增参数 = **52,736**（+0.39%）= `recon_head.weight`（V×d = 206×256）。
  基线（无头）= 13,536,768（与 `2618` 的 `U0` 臂同规格）。

## 生成命令（三臂逐字段一致，唯一变量 = `--recon-lambda`）

```bash
cd axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m
for ARM in A:0.0 B:1.0 B_rep:1.0; do
  NAME=${ARM%%:*}; LAM=${ARM##*:}
  python3 ../../../../../train_depth_ar.py $NAME \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
    --steps 12000 --batch-size 32 --seed 0 --lr 0.001 \
    --train-window 16 --train-anchors 0 \
    --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
    --recon-window 16 --recon-lambda $LAM \
    --eval-batches 1 --out-root train
done
python3 fill_config_hashes.py
python3 run.py
```

## 训练/评测时的代码指纹（`results.json::code` 原样抄录）

| 文件 | sha256 |
|---|---|
| `pathlm/recon_signal.py`（本线新模块，217 行） | `b0cc8480b29384f26decbfd844c91014bf3da307b0ede126198e4393adc182d8` |
| `pathlm/depth_ar.py` | `248a92754154d65e0857ca5ceccda91ef63ffd9b90b3cfde2199da29b6b59135`（**该修订同时含 U1 线未提交的 `carry_state_dim` 改动**，见下） |
| `train_depth_ar.py` | `8c333e5405baf23812cd8d389e69063a3b0129fa17993db7fe1ddb50e71828a0`（同上） |
| `tests/test_recon_signal.py`（14 条单测） | `df33b74487f6a303e3316f76ac73d61bfbb74046eae3f163c126bf768c85cbf2` |
| `run.py` | `15eb9c8eed99111ff918487f56ed715fa77dad13e7989008b44c29dc9c2bb452` |
| `config.json` | `ffc800e56329f7ec30a72760cd3e622ff127eceb515d632cbf140b1f9ca9ef21` |
| `2619-README.md`（本契约） | `09b841c5d5dd9bc6079f924dbafd6c3070bcdcbd3471111d031f6ca32cb28309` |
| `mutate.py` / `reconcile_numbers.py` / `slope_diagnosis.py` | `7973aafbeae1f2e2fcab851480663bc2910335b07709de9f973a51b95cf62b5f` / `5073eae630b5e7f35a90be0ddad7f02b2218c28b24d694d98d52a4b5380a5aee` / `8d2358dc206ebb17212221f5e092dd85b57c5041d4727bc4a81580afb03ddd63` |
| 复用仪器 | `tools/evalkit/carry_channel.py` `4dcd91facd60a512…`；`longrange_tasks.py` `4a02b93f19db8e99…`；`retention.py` `8320f03a2ecf74e3…` |

（以上为 `results.json::code` 原样抄录；`analysis.md` 第一个「重跑」发生在 `run.py` 补上
「契约 README 指纹」之后，两次读数的 `assoc_recall`/`criteria`/`comparisons`/
`recon_activity`/`distance_leg`/`train_results`/`loss_slope` **逐条相同**。）

## 训练侧散度与斜率专报

| 文件 | 内容 |
|---|---|
| `slope_report.json` | **同量**（主干 `depth_ce`）与总 loss 的 log-log 斜率、末段/初段 CE 均值 |
| `mutation_report.json` | 5 处语义开关的变异验证（全部抓红 + 逐字节还原） |
| `train/PROGRESS.log` | 训练轮询轨迹（含每臂起止时间与末行读数） |
| `smoke/` | 20 步冒烟四组：`smoke_A`（有头 λ=0）/ `smoke_B`（λ=1）/ `smoke_A_off`（`--recon-window 0`，纯 v1）/ `smoke_Brep` |

## 共享文件与并发披露（**口径重叠必须写出来**）

本线执行期间，`pathlm/depth_ar.py` 与 `train_depth_ar.py` 的**工作树**里同时存在
本批 U1 线（难度/容量曲线）的未提交改动（`carry_state_dim` 等）。本线的共享面按主
Agent 裁决限制为：`depth_ar.py` 的**配置字段 `recon_window` + `__init__` 建头一行**、
`train_depth_ar.py` 的**CLI 两参数 + `add_recon_term` 接线 + 活性量落盘**；
其余逻辑全在 `pathlm/recon_signal.py`（本线新文件）。
⇒ **上述两个 sha256 是「含 U1 改动」的混合修订**；本实验只对「本线新增行」负责，
U1 的行未被我改动（`analysis.md` §0 有逐条 grep 校验命令与输出）。

## 本实验的派生产物（全部入库，不含权重）

| 文件 | 内容 |
|---|---|
| `results.json` | 三臂全电池：仪器自检 + 目标前提 + 主腿（含毁源对照）+ 距离腿 + 判据 + 活性 + 显存 + 成本 |
| `train/<arm>/{train_log.jsonl,gate_log.jsonl,results.json}` | 训练日志（每 200 步含 `recon` 活性量）与训练器电池 |
| `smoke/` | 20 步冒烟（`smoke_A` / `smoke_B` / `smoke_A_off` / `smoke_Brep`）的日志与 stdout：其中 **`smoke_A` 与 `smoke_A_off` 逐步 loss 完全一致**（λ=0 逐位等价基线的运行期证据） |
| `mutation_report.json` | 5 处语义开关的变异验证（全部抓红 + 逐字节还原） |
| `reconcile_report.json` | `reconcile_numbers.py` 的报告/结果逐条对账 |
| `run_log.jsonl` / `run_log.txt` | 评测电池的逐步日志（`run_log.txt` 为同一次运行的 stdout） |
| `controls/README.md` | 对照臂索引（无损失基线 / 毁源对照 / 重跑臂）与方差界 |
| `analysis.md` | 判读（含失败与「不成立」路径） |
