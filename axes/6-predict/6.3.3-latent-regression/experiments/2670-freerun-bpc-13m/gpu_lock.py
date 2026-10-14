"""GPU 互斥锁（SCHEMA.md §20）：mkdir 法 + holder_pid + heartbeat + 释放。

用法：`with gpu_lock(): ...`。最多等 `max_wait` 秒；等不到抛 GPUUnavailable，
由调用方上报（**不杀**别人的进程）。
"""

from __future__ import annotations

import atexit
import os
import signal
import time

LOCK_DIR = "/tmp/pathlm_gpu.lock"
MAX_WAIT = 600.0        # 10 分钟
HEARTBEAT = 20.0


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


class gpu_lock:
    def __init__(self, name: str = "wp6.3.3", max_wait: float = MAX_WAIT):
        self.name, self.max_wait = name, max_wait
        self.held = False
        self._hb_stop = False

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                os.mkdir(LOCK_DIR)
                break
            except FileExistsError:
                pid = _read_pid()
                if not _pid_alive(pid):
                    # stale lock: holder gone. Reclaim by removing the dir (only
                    # when holder_pid is provably dead — never preempt a live pid).
                    try:
                        for fn in ("holder_pid", "heartbeat", "name"):
                            p = os.path.join(LOCK_DIR, fn)
                            if os.path.exists(p):
                                os.remove(p)
                        os.rmdir(LOCK_DIR)
                        continue
                    except OSError:
                        pass
                if time.time() - t0 > self.max_wait:
                    raise GPUUnavailable(
                        f"GPU lock held by pid {pid} > {self.max_wait}s; escalating")
                time.sleep(2.0)
        with open(os.path.join(LOCK_DIR, "holder_pid"), "w") as f:
            f.write(str(os.getpid()))
        with open(os.path.join(LOCK_DIR, "name"), "w") as f:
            f.write(self.name)
        self.held = True
        atexit.register(self.release)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._on_signal)
        self._touch()
        return self

    def _on_signal(self, *_):
        self.release()
        raise SystemExit(130)

    def _touch(self):
        try:
            with open(os.path.join(LOCK_DIR, "heartbeat"), "w") as f:
                f.write(str(time.time()))
        except OSError:
            pass

    def heartbeat(self):
        self._touch()

    def release(self):
        if self.held:
            try:
                for fn in ("holder_pid", "heartbeat", "name"):
                    p = os.path.join(LOCK_DIR, fn)
                    if os.path.exists(p):
                        os.remove(p)
                os.rmdir(LOCK_DIR)
            except OSError:
                pass
            self.held = False

    def __exit__(self, *exc):
        self.release()
        return False
