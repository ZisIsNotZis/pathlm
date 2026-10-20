"""GPU 互斥锁（`harness/SCHEMA.md` §20）：mkdir 法 + holder_pid + heartbeat + 自愈回收。

用法：`with gpu_lock(): ...`。最多等 `max_wait` 秒（缺省 600 = 10 min）；等不到抛
`GPUUnavailable`，由调用方上报（**不杀**别人的进程、**不动**别人的锁）。

自愈回收两条（`harness/SCHEMA.md` §20）：① `holder_pid` 不存在或其进程已死；
② 无 `holder_pid` 且锁龄 > 900 s。回收一律**先删除锁目录内容再 rmdir**。
释放时同样先删内容再 rmdir（锁目录含 `holder_pid` 时 rmdir 删不掉非空目录）。

锁顺序固定：先 code 锁、再 GPU 锁（本实验不涉及 code 锁，无嵌套）。
"""

from __future__ import annotations

import atexit
import os
import signal
import time

LOCK_DIR = "/tmp/pathlm_gpu.lock"
MAX_WAIT = 600.0          # 10 分钟
STALE_AGE = 900.0         # 无 holder_pid 时锁龄上限

_MEMBERS = ("holder_pid", "heartbeat", "name")


class GPUUnavailable(RuntimeError):
    pass


def _read_pid() -> int | None:
    try:
        with open(os.path.join(LOCK_DIR, "holder_pid")) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _lock_age() -> float:
    try:
        return time.time() - os.stat(LOCK_DIR).st_mtime
    except OSError:
        return 0.0


def _clear_dir() -> None:
    for fn in _MEMBERS:
        p = os.path.join(LOCK_DIR, fn)
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(LOCK_DIR)
    except OSError:
        pass


class gpu_lock:
    def __init__(self, name: str = "wp2652", max_wait: float = MAX_WAIT):
        self.name, self.max_wait = name, max_wait
        self.held = False

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                os.mkdir(LOCK_DIR)
                break
            except FileExistsError:
                pass
            pid = _read_pid()
            stale, reason = False, ""
            if pid is None:
                # holder_pid 尚未写出（或本就缺失）：只有锁龄超限才回收，
                # 否则会与「刚 mkdir、正要写 pid」的持锁者相撞（SCHEMA §20）。
                if _lock_age() > STALE_AGE:
                    stale, reason = True, "无 holder_pid 且锁龄 >15min"
            elif not _pid_alive(pid):
                stale, reason = True, f"holder_pid {pid} 已死"
            if stale:
                print(f"[gpu_lock] reclaim: {reason}", flush=True)
                _clear_dir()
                time.sleep(0.5)
                continue
            if time.time() - t0 > self.max_wait:
                raise GPUUnavailable(
                    f"GPU lock held by pid {pid} > {self.max_wait}s; 上报，不抢")
            time.sleep(2.0)
        with open(os.path.join(LOCK_DIR, "holder_pid"), "w") as f:
            f.write(str(os.getpid()))
        with open(os.path.join(LOCK_DIR, "name"), "w") as f:
            f.write(self.name)
        self.held = True
        atexit.register(self.release)
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._on_signal)
            except ValueError:
                pass
        self.heartbeat()
        return self

    def _on_signal(self, *_):
        self.release()
        raise SystemExit(130)

    def heartbeat(self):
        try:
            with open(os.path.join(LOCK_DIR, "heartbeat"), "w") as f:
                f.write(str(time.time()))
        except OSError:
            pass

    def release(self):
        if not self.held:
            return
        _clear_dir()
        self.held = False

    def __exit__(self, *exc):
        self.release()
        return False
