#!/usr/bin/env bash
# 2681-anchor-recall-seeds23-13m · 让位等待器：等更高优先级 GPU 作业退出再训剩余三臂。
#
# 共享机器纪律（AGENTS.md §0.6）：本机上不属于本项目的 GPU 作业优先。本脚本只做一件事——
# 轮询到「`train_sft.py` 与 `llama-server` 都不在，且空闲显存 ≥ 12 GB」才起 `train_arms.sh`，
# 期间**不训任何东西**、不抢锁、不杀别人的进程；轮询间隔 180 s（短间隔狂试会自己变成噪声）。
#
# 用法： setsid nohup bash wait_and_train.sh > train_wait.log 2>&1 < /dev/null &
#        MAX_WAIT_S=9000 ARMS="win16a_s3 win16_s2 win16_s3" bash wait_and_train.sh
set -uo pipefail
cd "$(dirname "$0")"
ARMS="${ARMS:-win16a_s3 win16_s2 win16_s3}"
MAX_WAIT_S="${MAX_WAIT_S:-9000}"      # 最多等 150 min，然后如实放弃并退出（不假装成功）
NEED_FREE_MIB="${NEED_FREE_MIB:-12000}"
POLL_S="${POLL_S:-180}"

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
busy_procs() {
  local extra=""
  pgrep -af "train_sft\.py" | grep -v wait_and_train | head -3 || true
  pgrep -af "llama-server" | head -3 || true
}

waited=0
while :; do
  procs=$(busy_procs)
  free=$(free_mib)
  if [ -z "$procs" ] && [ "$free" -ge "$NEED_FREE_MIB" ]; then
    echo "[$(date -Iseconds)] GPU 让位结束：free=${free}MiB，无更高优先级作业 ⇒ 起训练"
    break
  fi
  echo "[$(date -Iseconds)] wait ${waited}s free=${free}MiB busy=$(echo "$procs" | tr '\n' ';')"
  if [ "$waited" -ge "$MAX_WAIT_S" ]; then
    echo "[$(date -Iseconds)] 等满 ${MAX_WAIT_S}s 仍未空出 ⇒ 放弃（如实登记，不假装训完）"
    exit 9
  fi
  sleep "$POLL_S"
  waited=$((waited + POLL_S))
done

ARMS="$ARMS" bash train_arms.sh
rc=$?
echo "[$(date -Iseconds)] train_arms.sh rc=$rc"
exit $rc
