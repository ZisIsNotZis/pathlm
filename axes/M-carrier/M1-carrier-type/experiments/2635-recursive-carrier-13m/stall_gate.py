#!/usr/bin/env python3
"""2635-recursive-carrier-13m · 停摆门（契约 README §4a）：把 2616 的停摆改成**可机检**。

2616 的停摆症状（原始证据 `A-mixing/A6/.../2616-linear-attention-13m/stall_diagnosis.json`）：
训练日志停在某一步、进程 CPU 满载 ~98% 单核、GPU 显存不变、栈落在 `loss.backward()` 内。
本门把它变成判据①的判定：

  启动训练子进程（`PATHLM_DUMP_STACKS=<secs>` ⇒ faulthandler 周期性打全线程 Python 栈到
  子进程 stderr），每 `--poll-secs` 检查 `train_log.jsonl` 的最后一步时间戳；
  若 **≥ `--stall-secs` 秒没有新步**且子进程仍在运行 ⇒ 判停摆：采集
  （触发步、墙钟、CPU 时间、显存、最后 3 行日志、faulthandler 栈），终止该臂（SIGTERM → SIGKILL），
  写 `stall_report.json` 并返回码 3（非零 ⇒ 驱动脚本必须标注「实现层停摆」）。

用法：
  python3 stall_gate.py --run-dir <dir> --stall-secs 180 -- python3 train_depth_ar.py ...
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time


def _cpu_seconds(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        return (int(parts[13]) + int(parts[14])) / os.sysconf("SC_CLK_TCK")
    except Exception:
        return None


def _gpu_mem_for_pid(pid: int) -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    mem = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            mem.append(int(parts[1]))
    return mem


def _tail(path: str, n: int = 3) -> list[str]:
    try:
        with open(path) as f:
            return [l.rstrip("\n") for l in f.readlines()[-n:]]
    except Exception:
        return []


def _last_step(tlog: str) -> tuple[int | None, float | None]:
    """返回 `(step, 文件 mtime)`；日志不存在时 `(None, None)`。"""
    if not os.path.exists(tlog):
        return None, None
    st = os.stat(tlog)
    step = None
    try:
        with open(tlog) as f:
            for line in f:
                line = line.strip()
                if line:
                    step = json.loads(line).get("step")
    except Exception:
        return None, st.st_mtime
    return step, st.st_mtime


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--stall-secs", type=float, default=180.0)
    ap.add_argument("--poll-secs", type=float, default=20.0)
    ap.add_argument("--dump-stack-secs", type=int, default=150)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        raise SystemExit("缺少训练命令（用 `--` 分隔）")

    env = dict(os.environ)
    env["PATHLM_DUMP_STACKS"] = str(args.dump_stack_secs)
    t_start = time.time()
    proc = subprocess.Popen(cmd, env=env)
    report: dict = {"cmd": cmd, "stall_secs": args.stall_secs,
                    "poll_secs": args.poll_secs, "pid": proc.pid,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "pathlm_dump_stacks": env["PATHLM_DUMP_STACKS"]}
    repr_path = args.report

    def flush(extra: dict):
        report.update(extra)
        with open(repr_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            f.write("\n")

    flush({"status": "running"})
    last_progress = time.time()
    last_step, last_mtime = _last_step(args.train_log)
    while True:
        rc = proc.poll()
        if rc is not None:
            step, _ = _last_step(args.train_log)
            flush({"status": "finished" if rc == 0 else "failed", "returncode": rc,
                   "wall_s": round(time.time() - t_start, 1), "last_step": step,
                   "log_tail": _tail(args.train_log)})
            return rc
        time.sleep(args.poll_secs)
        step, mtime = _last_step(args.train_log)
        progressed = (step is not None and step != last_step) or \
                     (mtime is not None and mtime != last_mtime)
        if progressed:
            last_progress = time.time()
            last_step, last_mtime = step, mtime
            continue
        if time.time() - last_progress >= args.stall_secs:
            cpu0 = _cpu_seconds(proc.pid)
            time.sleep(10)
            cpu1 = _cpu_seconds(proc.pid)
            diag = {
                "status": "stalled",
                "trigger_step": last_step,
                "stall_wall_s": round(time.time() - last_progress, 1),
                "wall_s": round(time.time() - t_start, 1),
                "cpu_seconds": cpu1,
                "cpu_delta_10s": None if (cpu0 is None or cpu1 is None) else round(cpu1 - cpu0, 2),
                "gpu_mem_mb_for_pid": _gpu_mem_for_pid(proc.pid),
                "log_tail": _tail(args.train_log),
                "note": ("栈在子进程 stderr（驱动脚本重定向到 train_<arm>.log）；"
                         "PATHLM_DUMP_STACKS 每 %ss 打一次全线程栈" % env["PATHLM_DUMP_STACKS"]),
            }
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            flush(diag)
            return 3
        flush({"status": "running", "last_step": step,
               "since_progress_s": round(time.time() - last_progress, 1)})


if __name__ == "__main__":
    raise SystemExit(main())
