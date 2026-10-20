#!/usr/bin/env bash
# 有界 supervisor 重试驱动 —— 树内唯一的实现（各实验只写薄包装，见 `docs/methodology.md` §18.1）。
#
# 治什么形态的病：旧 `axes/**/supervise*.sh` 是**无界重试循环**（没有尝试上限），于是「训练完、写结果时崩」
# 会无限循环（`2653` 实测白跑 781 次 ≈124 GPU 小时），并且一直攥着 GPU 从不礼让，把 `2651`
# 饿死 21580 次重试。本脚本给这类驱动加三条硬界：总尝试上限、**连续**失败熔断、取锁礼让。
#
# 用法：
#   tools/supervise.sh --attempts-max N --fail-max M --lock-wait-s S --gap-s G \
#       --log FILE [--command-owns-lock] -- <命令> [参数...]
#
#   --attempts-max N   总尝试上限（含「锁被占用」那次）。触达 ⇒ 熔断 reason=attempts。
#   --fail-max M       允许的**连续**命令失败次数。触达 ⇒ 熔断 reason=consecutive。
#                      命令成功一次即把连续失败计数清零（rc=0 直接 DONE 退出，故实际是「首个成功即收口」）。
#   --lock-wait-s S    单次尝试里「取锁窗口」的最长等待（秒）。窗口内锁不被释放 ⇒ 记一次
#                      `lock-busy` 尝试，退避 --gap-s 后重来。
#   --gap-s G          两次尝试之间的礼让窗口（秒，缺省 60）：失败后等待，且绝不瞬间重抢。
#   --log FILE         日志文件（追加写；命令的 stdout/stderr 也追加进去）。
#   --command-owns-lock 锁的生命周期整条归被包装的命令：本脚本**不探测、不读、不建、不删**锁目录，
#                      也不写 `lock-busy`；`--lock-wait-s` 此时不起作用（仍要求给值，CLI 只有一个形状）。
#
# 锁的语义（重要）：本脚本**不持有** GPU 锁，只在启动命令前**探测**锁是否空闲
# （`SCHEMA.md` §20 的 `/tmp/pathlm_gpu.lock`，可用环境变量 `PATHLM_GPU_LOCK` 改路径，测试用）。
# 原因：被包的驱动自己取锁（如 `2651`/`2653` 的 `run.py` 用 `gpu_lock.py` 的上下文管理器），
# 若 supervisor 抢先持锁，命令会等自己永远等不到的锁 ⇒ 死锁。探测式礼让的保证是：
# 「锁被活着的持有者占着就不启动命令」+「两次尝试之间至少间隔 --gap-s」，绝不删别人的锁目录。
# 取锁失败（`lock-busy`）按**一次尝试**计，但**不计入连续失败**——锁被占用不是实验失败。
#
# `--command-owns-lock`（命令自己管锁）：2654 的驱动**按臂** mkdir/rmdir 锁（`run.py` 内部不取锁，
# 锁的等待与超时全在驱动里，`WAIT_MAX` 缺省 600s），2623 的 `eval_run.sh` / `train_all.sh` 自己取锁、
# 等锁、回收陈旧锁并以 rc=9 上报超时（40×15s）。这两种命令本身就是长等锁人：supervisor 再探测
# 一遍就是同一个锁上的**第二个等锁人**，两个窗口（`--lock-wait-s` 与命令内部的窗口）互相遮蔽，
# 日志里就说不清到底是谁在等。所以这个模式把锁整条时间线交给命令，supervisor 只留次数上的界
# （上限、熔断、礼让、`FUSED` 横幅、退 3 全部照旧）。两种模式的判据是：命令自己长等锁就用本开关；
# 命令只在短窗口里等、超时就把这次尝试交还给 supervisor（2651/2653 的 `run.py` 用 8s 窗口）就用默认。
#
# 日志横幅（唯一痕迹，不建任何仓库级状态文件）：
#   [sup] supervise start cmd=... attempts-max=... fail-max=... gap-s=... <lock-wait-s=N lock=路径 | command-owns-lock=1 lock=unmanaged> pid=...
#   [sup] attempt N start <iso> holder=<pid|none|cmd-owned>
#   [sup] attempt N lock-busy holder=<pid|none> waited=Ns        （不计入连续失败；仅探测模式）
#   [sup] attempt N rc=<rc> end <iso> consecutive=<k>
#   [sup] DONE rc=0 <iso>                                        （退出 0，判活看这一行）
#   [sup] FUSED reason=<attempts|consecutive> at=<iso>           （退出 3，不再取锁、不再重试）
set -u

PROG=$(basename "$0")
LOCK=${PATHLM_GPU_LOCK:-/tmp/pathlm_gpu.lock}
POLL_S=5            # 取锁窗口内的轮询间隔
STALE_AGE=900       # 无 holder_pid 时锁龄上限（`SCHEMA.md` §20）

usage() {   # $1 = 输出 fd（--help 用 1 = stdout 且退出 0；用法错误用 2 = stderr 且退出 2）
  local fd="${1:-2}"
  cat >&"$fd" <<'EOF'
用法：tools/supervise.sh --attempts-max N --fail-max M --lock-wait-s S [--gap-s G] --log FILE [--command-owns-lock] -- <命令> [参数...]
  --attempts-max N  总尝试上限（含 lock-busy 那次）；触达 ⇒ FUSED reason=attempts
  --fail-max M      连续命令失败上限；触达 ⇒ FUSED reason=consecutive
  --lock-wait-s S   单次尝试的取锁窗口（秒）；窗口用尽 ⇒ 记一次 lock-busy 尝试（不计入连续失败）
  --gap-s G         两次尝试之间的礼让窗口（秒，缺省 60）
  --log FILE        日志文件（追加写；命令 stdout/stderr 也追加进去）
  --command-owns-lock 锁由命令自己管：不探测/不触碰锁目录，--lock-wait-s 不起作用
  --                其后为实际命令
退出码：0 = 命令成功（日志有 `[sup] DONE rc=0`）；3 = 熔断（日志有 `[sup] FUSED reason=...`）；2 = 用法错误
EOF
  [ "$fd" -eq 1 ] && exit 0 || exit 2
}

attempts_max= fail_max= lock_wait_s= gap_s=60 log_file= command_owns_lock=0
while [ $# -gt 0 ]; do
  case "$1" in
    --attempts-max) attempts_max=${2:-}; shift 2 ;;
    --fail-max)     fail_max=${2:-};     shift 2 ;;
    --lock-wait-s)  lock_wait_s=${2:-};  shift 2 ;;
    --gap-s)        gap_s=${2:-};        shift 2 ;;
    --log)          log_file=${2:-};     shift 2 ;;
    --command-owns-lock) command_owns_lock=1; shift ;;
    --help|-h)      usage 1 ;;
    --)             shift; break ;;
    *) echo "$PROG: 未知参数 '$1'（命令必须放在 -- 之后）" >&2; usage ;;
  esac
done

for v in "$attempts_max" "$fail_max" "$lock_wait_s" "$gap_s"; do
  case "$v" in ''|*[!0-9]*) echo "$PROG: 缺参数或非整数（--attempts-max/--fail-max/--lock-wait-s/--gap-s）" >&2; exit 2 ;; esac
done
[ -n "$log_file" ] || { echo "$PROG: 缺 --log FILE" >&2; exit 2; }
[ $# -gt 0 ] || { echo "$PROG: -- 之后缺命令" >&2; exit 2; }
[ "$attempts_max" -ge 1 ] || { echo "$PROG: --attempts-max 必须 ≥1" >&2; exit 2; }
[ "$fail_max" -ge 1 ] || { echo "$PROG: --fail-max 必须 ≥1" >&2; exit 2; }

iso() { date -Iseconds; }
log() { echo "[sup] $*" >>"$log_file"; }
holder() {
  # --command-owns-lock：锁不归本脚本管，也不去读它（读也是一种探测）
  [ "$command_owns_lock" -eq 1 ] && { echo "cmd-owned"; return 0; }
  local hp
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null || true)
  echo "${hp:-none}"
}

# lock_free：0 = 锁空闲（或持锁者已死/锁已陈旧，命令自己会回收）；1 = 有活着的持有者
lock_free() {
  [ -d "$LOCK" ] || return 0
  local hp
  hp=$(cat "$LOCK/holder_pid" 2>/dev/null || true)
  if [ -n "$hp" ]; then
    kill -0 "$hp" 2>/dev/null && return 1 || return 0
  fi
  # 无 holder_pid：只有锁龄超限才算陈旧，否则会与「刚 mkdir、正要写 pid」的持有者相撞（§20）
  local age
  age=$(( $(date +%s) - $(stat -c %Y "$LOCK" 2>/dev/null || date +%s) ))
  [ "$age" -gt "$STALE_AGE" ] && return 0 || return 1
}

# probe_lock：在 --lock-wait-s 窗口内等锁变空闲；0 = 可以启动命令；9 = 窗口用尽
probe_lock() {
  local deadline=$(( $(date +%s) + lock_wait_s )) left
  while [ "$(date +%s)" -lt "$deadline" ] && ! lock_free; do
    left=$(( deadline - $(date +%s) ))
    sleep $(( left < POLL_S ? left : POLL_S ))
  done
  lock_free
}

mkdir -p "$(dirname "$log_file")"
if [ "$command_owns_lock" -eq 1 ]; then
  lock_desc="command-owns-lock=1 lock=unmanaged"
else
  lock_desc="lock-wait-s=$lock_wait_s lock=$LOCK"
fi
log "supervise start cmd=$* attempts-max=$attempts_max fail-max=$fail_max gap-s=$gap_s $lock_desc pid=$$"

fuse() { log "FUSED reason=$1 at=$(iso)"; exit 3; }

attempt=0
consecutive=0
while [ "$attempt" -lt "$attempts_max" ]; do
  attempt=$((attempt + 1))
  log "attempt $attempt start $(iso) holder=$(holder)"

  if [ "$command_owns_lock" -eq 0 ] && ! probe_lock; then
    # 锁被占用：计一次尝试，**不计入**连续失败（锁被占用不是实验失败）
    log "attempt $attempt lock-busy holder=$(holder) waited=${lock_wait_s}s"
    [ "$attempt" -lt "$attempts_max" ] && sleep "$gap_s"
    continue
  fi

  "$@" >>"$log_file" 2>&1
  rc=$?
  log "attempt $attempt rc=$rc end $(iso) consecutive=$(( rc == 0 ? 0 : consecutive + 1 ))"
  if [ "$rc" -eq 0 ]; then
    consecutive=0
    log "DONE rc=0 $(iso)"
    exit 0
  fi

  consecutive=$((consecutive + 1))
  [ "$consecutive" -ge "$fail_max" ] && fuse consecutive
  [ "$attempt" -lt "$attempts_max" ] && sleep "$gap_s"
done
# 出循环 = 尝试上限用尽（最后那次既没成功也没被连续失败熔断）
fuse attempts
