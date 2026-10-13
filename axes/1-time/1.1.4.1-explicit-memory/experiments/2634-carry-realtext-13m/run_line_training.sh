#!/usr/bin/env bash
# 本波「信息右移」三步线的训练总驱动（按优先级串行；每一步内部已按臂释放 GPU 锁）。
# 顺序 = 主任务优先：① 真实文本口径（2634，主任务）→ ② delta vs plain 判到 n=3（2633）
# → ③ 递推/线性载体质量腿（2635，预算内可做项；预算不够时先砍它）。
# 每一步用的是**该实验自己的** train_*.sh（§17.8：步时与预算都出自本实验实测）。
set -u
cd "$(dirname "$0")/../../../../.."   # repo root
B=axes/1-time/1.1.4.1-explicit-memory/experiments
for s in "$B/2634-carry-realtext-13m/train_realtext.sh" \
         "$B/2633-carry-seed2/train_seed2.sh" \
         "$B/2635-recursive-carrier-13m/train_state.sh"; do
  echo "########## $s $(date -Is)"
  bash "$s"
  echo "########## rc=$? $(date -Is)"
done
echo "LINE TRAINING DONE $(date -Is)"
