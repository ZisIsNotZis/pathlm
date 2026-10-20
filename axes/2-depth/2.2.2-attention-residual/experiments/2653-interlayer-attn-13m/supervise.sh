#!/bin/bash
# 2653 · supervisor 薄包装（有界重试）。
#
# 重试上限、连续失败熔断、取锁礼让**全部**由仓库共享实现 `tools/supervise.sh` 提供
# （`harness/methodology.md` §18.1）；本脚本只保留本实验自己的命令与参数。
# 判活看末行横幅 `[sup] DONE rc=0`；熔断看 `[sup] FUSED reason=<attempts|consecutive>`——
# 熔断后不再取锁、不再重试（无界重试的代价：781 次白跑 ≈124 GPU 小时）。
# 每臂由 run.py 的 `gpu_lock` 上下文管理器取/放锁，并由 `--arm-gap-s 65` 留 ≥60s 窗口
# （`harness/SCHEMA.md` §20.1）；supervisor 只探测锁是否空闲，不替 run.py 持锁。
# 形态：按臂续跑 + 逐臂增量落盘（results_<arm>_s<seed>.json 存在则跳过）。
# run.py 用仓库根相对路径读数据/切分 ⇒ 必须在仓库根运行。
# 用法：setsid nohup bash supervise.sh >/dev/null 2>&1 &
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$EXP/../../../../../" && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo $$ > "$EXP/supervise.pid"
exec bash "$ROOT/tools/supervise.sh" \
  --attempts-max 40 --fail-max 3 --lock-wait-s 600 --gap-s 60 \
  --log "$EXP/supervise.out" \
  -- python3 "$EXP/run.py" --arms attn-scalar,attn-qkv --seed 0 --steps 12000 \
       --lock-wait-s 8 --arm-gap-s 65
