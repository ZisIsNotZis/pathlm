"""GPU 互斥锁（SCHEMA §20）——**照抄** `2632-kv-block-scale/run.py` 的 `GpuLock`。

唯二差别：① 本文件把 `_gpu_free_mib` 与 `GpuLock` 单独成模块（2632 里内联在 run.py）；
② 无其它改动。等锁超时/陈旧锁自愈语义与 2632 逐字一致（`mkdir` 抢锁；holder_pid 死或
无 pid 且锁龄 >15 min ⇒ 自愈回收；**不杀任何他人进程**、不回收活的锁）。
"""

import os
import subprocess
import time


def _gpu_free_mib():
    """`nvidia-smi` 的空闲显存（MiB）；不可用时 None（如实记 null，不拦跑）。"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:                                              # pragma: no cover
        return None


#: SCHEMA §20 的 GPU 互斥锁路径（与全仓一致）
GPU_LOCK_DIR = "/tmp/pathlm_gpu.lock"


class GpuLock:
    """SCHEMA §20 的互斥锁：`mkdir` 抢锁，陈旧锁（holder 死 / 无 pid 且年龄 >15 min）自愈回收。

    `evidence()` 把取锁/开跑/收尾/释放与前后空闲显存落成**机器可读**记录
    （`results.json::gpu_lock`），供 §20 记账核查。
    """

    def __init__(self, lock_dir: str = GPU_LOCK_DIR, timeout_s: int = 600, poll_s: int = 15):
        self.lock_dir = lock_dir
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.ev = {"lock_path": lock_dir, "holder_pid": None, "acquired_at": None,
                   "wait_seconds": None, "reclaimed": [], "release_ok": None,
                   "free_mib_before": None, "free_mib_after": None}

    def _holder_pid(self):
        try:
            return int(open(os.path.join(self.lock_dir, "holder_pid")).read().strip())
        except Exception:
            return None

    def _stale(self) -> str | None:
        hp = self._holder_pid()
        if hp is not None:
            try:
                os.kill(hp, 0)
                return None
            except OSError:
                return f"holder_pid {hp} 已死"
        try:
            age = time.time() - os.stat(self.lock_dir).st_mtime
        except OSError:
            return None
        if age > 900:
            return f"无 holder_pid 且锁龄 {age:.0f}s >15min"
        return None

    def acquire(self) -> None:
        t0 = time.time()
        while True:
            try:
                os.mkdir(self.lock_dir)
                break
            except FileExistsError:
                reason = self._stale()
                if reason:
                    for f in ("holder_pid", "heartbeat"):
                        try:
                            os.remove(os.path.join(self.lock_dir, f))
                        except OSError:
                            pass
                    try:
                        os.rmdir(self.lock_dir)
                    except OSError:
                        pass
                    self.ev["reclaimed"].append(reason)
                    continue
                if time.time() - t0 > self.timeout_s:
                    raise TimeoutError(f"等锁超时 {self.timeout_s}s：{self.lock_dir} 被占")
                time.sleep(self.poll_s)
        with open(os.path.join(self.lock_dir, "holder_pid"), "w") as f:
            f.write(str(os.getpid()))
        open(os.path.join(self.lock_dir, "heartbeat"), "w").close()
        self.ev["holder_pid"] = os.getpid()
        self.ev["acquired_at"] = round(time.time(), 3)
        self.ev["wait_seconds"] = round(time.time() - t0, 3)
        self.ev["free_mib_before"] = _gpu_free_mib()

    def release(self) -> None:
        if self.ev.get("release_ok") is not None:
            return
        self.ev["free_mib_after"] = _gpu_free_mib()
        for f in ("holder_pid", "heartbeat"):
            try:
                os.remove(os.path.join(self.lock_dir, f))
            except OSError:
                pass
        try:
            os.rmdir(self.lock_dir)
            self.ev["release_ok"] = True
        except OSError:
            self.ev["release_ok"] = False
