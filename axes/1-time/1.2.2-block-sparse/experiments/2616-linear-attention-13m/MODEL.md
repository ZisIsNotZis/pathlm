# MODEL — 2616 线性注意力 vs 全注意力 · 13M · depth-ar/dense（契约 R）

权重与数据集**不入库**（`.gitignore` 覆盖 `*.pt` 与 `data/`）。本文件记录权重清单、哈希、
体积与生成命令。

## 1. 权重清单

| 臂 | 文件 | attn_linear | 参数量 | 体积 (bytes) | sha256 | 状态 |
|---|---|---|---|---|---|---|
| full（基线，全注意力） | `runs/full-12k-s0/model.pt` | false | 6,962,688 | 27,882,155 | `1af6beba3324389a87ef8616d18d1ade7ce32d6fa8e6a1a786f5af68e27f6054` | **12k 步跑完**（clean bpc 1.5268） |
| linear（线性注意力） | `runs/linear-12k-s0/model.pt` | true | — | — | — | **不存在**：训练三次均在第 400–800 步停摆（见 `stall_diagnosis.json`），未产出 checkpoint |
| 重跑臂 | — | — | — | — | — | **未跑**（30 min 收口 + 三次停摆吃掉预算） |

参数量账（`results.json::field_by_field` 与单测
`test_linear_params_and_init_identical_to_full_attention_block`）：两臂参数量逐位相等
（**6,962,688**），同 seed 下初始化**逐位相同** ⇒ 干预**不新增参数**，
SCHEMA §15 的等参数要求自动满足（本臂是「零参数增量」的算子替换）。

冒烟权重（20 步 × 两臂，非结果证据）：`smoke/smoke-{full,linear}/model.pt`（不记哈希）。

## 2. 代码 revision（本实验绑定的实现版本）

| 文件 | sha256 |
|---|---|
| `pathlm/depth_ar.py` | `cafb532edcb79cc34b149e38b7fb7dbeae25cd42579b95828c37c796bf947976` |
| `train_depth_ar.py` | `a9c521f5d07a965a83e9d0306bb9dc4225eeb7a463c367b3a60d07b6ab5a6cfe` |
| `tests/test_depth_ar.py` | `55936874fab327e6b6534c0f3bd473fa25bb09bb7a0db947dfbfa4b93156a26e` |

- `git rev-parse HEAD` = `774fcb9fae105c68d228c10fb04b40b905cf5e67`（本实验的代码改动在**工作树**中，
  未由执行者提交；上表 sha256 是判读数字所绑定的确切实现版本，复核者可直接逐字节核对）。
- 变异验证（`mutate.py`）：6 处故意破坏，每处都抓红并逐字节还原到上表 sha256
  （`mutation_report.json::restored_byte_identical = true`）。
- **注意（执行事故）**：线性臂第三次尝试使用的是**同一份** `pathlm/depth_ar.py`；
  第二次停摆后我改的是**训练循环里的活性量累加方式**（每步 48 次设备同步 → 0 次），
  算子数学（φ、归一化、分块）**一个字没改** ⇒ 上表 sha256 对两臂一致。

## 3. 生成命令

```bash
cd /home/z/vibe/pathlm
EXP=axes/1-time/1.2.2-block-sparse/experiments/2616-linear-attention-13m
python3 $EXP/smoke.py                      # 20 步冒烟（先冒烟再全跑）
nohup python3 $EXP/run.py --steps 12000 --seed 0 --budget-min 45 \
      > $EXP/run_log.txt 2>&1 &            # ← 正式主跑；本次 full 臂由它产出
nohup python3 $EXP/run.py --skip-train --no-rerun --arms full \
      > $EXP/run_log_assembly.txt 2>&1 &   # 收口组装（成本口径 + 电池 + 曲线 + 检索腿）
python3 $EXP/mutate.py                     # 变异验证（纯 CPU；不要与训练并发）
python3 $EXP/reconcile_numbers.py          # 报告数字回算
```

每个臂对应仓库入口的等价命令（`run.py` 逐臂子进程调用它）：

```bash
python3 train_depth_ar.py <arm>-12k-s0 \
    --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 --dense \
    [--attn-linear --linear-chunk 64] --steps 12000 --batch-size 32 --seed 0 \
    --exit-mode uniform --eval-batches 20 --out-root <exp>/runs
```

数据集：`data/enwik8_full.npz`（char 级 enwik8，前 90M train / 末 10M eval，词表 206），
本地已有，无需下载。

## 4. 复用 checkpoint 的 sha256 断言

本实验**不加载**任何他人 checkpoint 做对照（两臂都是本实验新训）；
只在 `analysis.md` §8.2 引用 ticket 32-M/2615 在
`axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`
上的 recent 曲线作**跨批定性参照**，该文件的 sha256 已实测断言：

```
86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2  trunk.pt
```

（与 2615 `config.json::checkpoint.sha256` 一致；`sha256sum` 实测核对通过。）

## 5. 复现资产

| 文件 | 内容 |
|---|---|
| `results.json` | full 臂全部电池 + 逐字段对照 + 成本口径（理论/实测）+ 依赖曲线 + 检索腿 + 稳定性/收敛 |
| `smoke/smoke.json` | 20 步冒烟：v1 还原 / 参数账 / 活性 / O(T) 证据 / 因果 / 仪器路径 / CLI 接线 / 步时显存 |
| `mutation_report.json` | 6 处变异抓红记录 + 还原后 sha256 一致性 |
| `stall_diagnosis.json` + `stall_diagnosis.py` | **线性臂三次停摆的原始观测 + 5 次诊断二分**（本实验的核心负结果证据）|
| `linear-12k-s0-attempt3-train_log.jsonl` / `-stdout.log` | 停摆尝试的原始日志（3 行）与进程输出（含 faulthandler 栈） |
| `<arm>-train_log.jsonl` | 训练日志（每 200 步；含 `attn_linear` 活性量，线性臂为 3 行） |
| `reconcile_report.json` | 报告数字逐条回算结果 |
| `paired_eval.json` | 同批配对差（本次只覆盖 full 臂内部一致性） |
| `analysis.md` / `stability.md` | 判读（人话摘要在最前）/ 稳定性专报 |
