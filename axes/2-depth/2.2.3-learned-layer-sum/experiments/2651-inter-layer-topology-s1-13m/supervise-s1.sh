#!/bin/bash
# 2651 · seed=1 supervisor 薄包装（有界重试，setsid nohup 后台）。
#
# 重试上限、连续失败熔断、取锁礼让**全部**由仓库共享实现 `tools/supervise.sh` 提供
# （`harness/methodology.md` §18.1）；本脚本只保留本实验自己的命令与参数。
# 判活看末行横幅 `[sup] DONE rc=0`；熔断看 `[sup] FUSED reason=<attempts|consecutive>`——
# 熔断后不再取锁、不再重试。
# 取锁：supervisor 在启动命令前探测锁是否空闲（单次窗口 ≤600s，两次尝试间隔 ≥60s）；
# 真正的取/放锁由 run.py 的 `gpu_lock` 负责，每臂结束释放并留 ≥60s 窗口（`harness/SCHEMA.md` §20.1）。
# 形态：按臂续跑 + 逐臂增量落盘（run.py 每臂写 results.json + append run_log.txt）；
# 已训完的臂（sequential-12k-s1）由 run.py::arm_trained 自动跳过重训（只重算电池，~3s）。
# 三条待训臂 = parallel / parallel-dual-ln / dense（seed=1，其余与 2613 s0 逐字段相同）。
# 注：run.py 的 --arms 校验要求必须含 sequential（判读需要它做基线）；sequential 已训完，
# 入选只是满足校验、不会重训。不用 --skip-s0-anchor 之外的额外开关。
# run.py 用仓库根相对路径读数据/切分 ⇒ 必须在仓库根运行。
# 用法：setsid nohup bash supervise-s1.sh >/dev/null 2>&1 &
set -u
EXP="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$EXP/../../../../../" && pwd)"
cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo $$ > "$EXP/supervise.pid"
exec bash "$ROOT/tools/supervise.sh" \
  --attempts-max 40 --fail-max 3 --lock-wait-s 600 --gap-s 60 \
  --log "$EXP/supervise.out" \
  -- python3 "$EXP/run.py" --arms sequential,parallel,parallel-dual-ln,dense \
       --skip-s0-anchor --caps-min 14 --budget-min 42 \
       --lock-wait-s 8 --lock-poll-s 5
