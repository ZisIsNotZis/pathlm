#!/usr/bin/env bash
# 2681-anchor-recall-seeds23-13m · 让位等待器：等更高优先级 GPU 作业退出再逐臂训练。
#
# 共享机器纪律（AGENTS.md §0.6）：本机上不属于本项目的 GPU 作业优先。
# 起训门（元级裁决 2026-10-20）：**无 SFT 训练作业** 且 **空闲显存 ≥ 6 GB** 且
# `/tmp/pathlm_gpu.lock` 空闲。允许与常驻推理服务（`llama-server`）共存——它是别的会话反复重启的
# 常驻服务，不是排他占用的训练；本实验单臂峰值实测 2.0 GB，不构成驱逐。
# 收紧项：逐臂起训前重新过门；任一臂峰值 > 3 GB 或 OOM 一次 ⇒ **立刻退回等待并登记「共存失败」**，
# 不重试第三轮。轮询间隔 180 s（短间隔狂试自己就是噪声）。
#
# 用法： setsid nohup bash wait_and_train.sh > train_wait.log 2>&1 < /dev/null &
set -uo pipefail
cd "$(dirname "$0")"
ARMS="${ARMS:-win16a_s3 win16_s2 win16_s3}"
MAX_WAIT_S="${MAX_WAIT_S:-9000}"      # 最多等 150 min，然后如实放弃并退出（不假装成功）
NEED_FREE_MIB="${NEED_FREE_MIB:-6000}"
PEAK_LIMIT_MB="${PEAK_LIMIT_MB:-3072}"
POLL_S="${POLL_S:-180}"
LOCK=/tmp/pathlm_gpu.lock

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
sft_procs() { pgrep -af "train_sft" | grep -v wait_and_train || true; }

# 起训门：三条件全过才返回 0；否则每 POLL_S 秒重试，超过 MAX_WAIT_S 返回 9（如实放弃）。
wait_for_gate() {
  local waited=0 procs free
  while :; do
    procs=$(sft_procs)
    free=$(free_mib)
    if [ -z "$procs" ] && [ "$free" -ge "$NEED_FREE_MIB" ] && [ ! -d "$LOCK" ]; then
      echo "[$(date -Iseconds)] 门开：free=${free}MiB，无 SFT 训练作业，GPU 锁空闲 ⇒ 起臂 $1"
      return 0
    fi
    echo "[$(date -Iseconds)] wait(${1}) ${waited}s free=${free}MiB lock=$([ -d "$LOCK" ] && echo busy || echo free) sft=$(echo "$procs" | tr '\n' ';')"
    if [ "$waited" -ge "$MAX_WAIT_S" ]; then
      echo "[$(date -Iseconds)] 等满 ${MAX_WAIT_S}s 门仍未开 ⇒ 放弃（如实登记，不假装训完）"
      return 9
    fi
    sleep "$POLL_S"
    waited=$((waited + POLL_S))
  done
}

peak_mb() {  # 读该臂训练落盘的峰值显存（MiB）；缺失就打印 0（不编数字）
  python3 - "$1" <<'PY' 2>/dev/null || echo 0
import json, sys
p = f"train/{sys.argv[1]}/results.json"
try:
    print(int(json.load(open(p)).get("peak_gpu_mem_mb") or 0))
except Exception:
    print(0)
PY
}

for A in $ARMS; do
  wait_for_gate "$A" || exit 9
  ARMS="$A" bash train_arms.sh
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date -Iseconds)] 臂 $A rc=$rc（OOM 或训练失败）⇒ 共存失败，退回等待，不再重试"
    echo "COEXIST_FAIL arm=$A rc=$rc $(date -Iseconds)"
    exit 9
  fi
  peak=$(peak_mb "$A")
  echo "[$(date -Iseconds)] 臂 $A 完成，峰值显存 ${peak}MiB（上限 ${PEAK_LIMIT_MB}）"
  if [ "$peak" -gt "$PEAK_LIMIT_MB" ]; then
    echo "[$(date -Iseconds)] 臂 $A 峰值 ${peak}MiB > ${PEAK_LIMIT_MB}MiB ⇒ 共存失败（超预算），退回等待"
    echo "COEXIST_FAIL arm=$A peak_mb=$peak $(date -Iseconds)"
    exit 9
  fi
done
echo "[$(date -Iseconds)] 三臂全部完成"
