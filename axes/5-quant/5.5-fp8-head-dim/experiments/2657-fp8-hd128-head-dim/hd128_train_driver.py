#!/usr/bin/env python3
"""2657 · `head_dim=128` 训练驱动（detach + 失败重试 + `harness/SCHEMA.md` §20 按臂取锁）。

契约：同目录 `README.md`（§9 训练）与 `config.json::checkpoints`（**运行前**写入）。

每臂一次尝试 = 取锁 → 训练 → 放锁；失败 `sleep 20` 重试，成功进下一臂：

- 臂 = `hd128-s0`（seed 0，主臂）→ `hd128-s1`（seed 1，`n≥2` 的第二 seed）。预算紧时先跑完
  `hd128-s0`，再由 finisher 放 `STOP_AFTER_FIRST_ARM` 文件 / 直接放行两臂，日志显式登记。
- 配方与 `2616-gqa-mqa-13m` 的 mha 臂逐字段一致，**只改 `--heads 8 → 2`**
  （`d_model/heads = 256/2 = 128` ⇒ `head_dim=128`）：这是本实验唯一的模型侧改动。
- `harness/SCHEMA.md` §20.1：两臂合计 ≈16 min > 10 min ⇒ **不整程持锁**，每臂取一次锁、跑完即放。
- 等锁上限 600 s（只读复用 `2632/run.py::GpuLock`，未复制改写）；等不到按本次尝试失败
  记账并重试，**不杀他人进程**。
- 每臂落 `<runs>/<arm>/{cmd.txt,stdout.log,model.pt,results.json}`，机器可读状态追加进
  `train_driver_state.json`（checkpoint 的 sha256 / size / params 由本驱动算，供 finisher 复核）。

用法（detach，见 `README.md` §9）：

    cd <expdir> && setsid nohup python3 hd128_train_driver.py >> train_driver.log 2>&1 &
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

P2632 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale")


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: `harness/SCHEMA.md` §20 的锁协议：只读复用 `2632/run.py::GpuLock`（含 holder_pid / 心跳 / 陈旧锁自愈）。
g2632 = _load_module(os.path.join(P2632, "run.py"), "kv_block_scale_2632")

RUNS = os.path.join(HERE, "runs")
STATE = os.path.join(HERE, "train_driver_state.json")
CUT_FILE = os.path.join(HERE, "STOP_AFTER_FIRST_ARM")

#: 训练臂：(run 名, seed)。`hd128-s0` 优先；`hd128-s1` 是 n≥2 的第二 seed。
ARMS = (("hd128-s0", 0), ("hd128-s1", 1))
#: 配方 = `2616-gqa-mqa-13m` 的 mha 臂，唯一改动 `--heads 8 → 2`（⇒ head_dim=128）。
TRAIN_ARGS = ("--d 256 --layers 12 --heads 2 --mlp-mult 6 --dense --steps 12000 "
              "--batch-size 32 --lr 0.001 --eval-batches 24").split()
LOCK_TIMEOUT_S = 600     # §20：等锁上限 10 min
POLL_S = 15              # §20：轮询间隔
RETRY_SLEEP_S = 20       # 契约：失败后 sleep 20 再试


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def log(line: str) -> None:
    print(f"{line}", flush=True)


def _ckpt_info(run_dir: str) -> dict:
    import hashlib
    p = os.path.join(run_dir, "model.pt")
    if not os.path.exists(p):
        return {"model_pt": None}
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    info = {"model_pt": os.path.relpath(p, ROOT), "sha256": h.hexdigest(),
            "size_bytes": os.path.getsize(p)}
    rj = os.path.join(run_dir, "results.json")
    if os.path.exists(rj):
        r = json.load(open(rj))
        info["params"] = r.get("params")
        info["train_wall_minutes"] = r.get("wall_minutes")
    return info


def _dump_state(state: dict) -> None:
    with open(STATE, "w") as f:
        json.dump(state, f, indent=1, ensure_ascii=False)


def train_one(name: str, seed: int, state: dict) -> None:
    """一个训练臂：取锁 → 训练 → 放锁，失败则 `sleep 20` 重试（语义 = 队列里排队重试）。

    已有 `model.pt` 的臂**直接跳过**（`PATHLM_HD128_RETRAIN=1` 可强制重训）——`train_depth_ar.py`
    自身不检查已存在的 run，重跑驱动不应把已完成的臂白跑一遍。
    """
    run_dir = os.path.join(RUNS, name)
    os.makedirs(run_dir, exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "train_depth_ar.py"), name, *TRAIN_ARGS,
           "--seed", str(seed), "--out-root", RUNS]
    if os.path.exists(os.path.join(run_dir, "model.pt")) and os.environ.get("PATHLM_HD128_RETRAIN") != "1":
        state["arms"][name] = {"seed": seed, "status": "skipped_existing", "attempts": 0,
                              "cmd": " ".join(cmd), **_ckpt_info(run_dir)}
        _dump_state(state)
        log(f"=== skip {name}（runs/{name}/model.pt 已存在）at {now()} ===")
        return
    attempt = 0
    while True:
        attempt += 1
        state["arms"][name] = {"seed": seed, "status": "waiting_lock", "attempts": attempt,
                              "cmd": " ".join(cmd)}
        _dump_state(state)
        lock = g2632.GpuLock(timeout_s=LOCK_TIMEOUT_S, poll_s=POLL_S)
        try:
            lock.acquire()
        except TimeoutError as exc:
            log(f"=== wait_lock_timeout {name} attempt={attempt} {exc} at {now()} ===")
            state["arms"][name].update({"status": "wait_lock_timeout", "last_rc": None})
            _dump_state(state)
            time.sleep(RETRY_SLEEP_S)
            continue
        with open(os.path.join(run_dir, "cmd.txt"), "w") as f:
            f.write(" ".join(cmd) + "\n")
        log(f"=== start {name} seed={seed} attempt={attempt} at {now()} "
            f"(lock wait {lock.ev['wait_seconds']}s, gpu free {lock.ev['free_mib_before']} MiB) ===")
        state["arms"][name].update({"status": "training", "last_rc": None})
        state["arms"][name]["gpu_lock"] = lock.ev
        _dump_state(state)
        t0 = time.time()
        with open(os.path.join(run_dir, "stdout.log"), "a") as f:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        wall = round(time.time() - t0, 1)
        lock.release()
        entry = {"seed": seed, "status": "done" if rc == 0 else "failed", "attempts": attempt,
                 "last_rc": rc, "wall_seconds": wall, "cmd": " ".join(cmd),
                 "gpu_lock": lock.ev}
        state["arms"][name] = entry
        if rc == 0:
            entry.update(_ckpt_info(run_dir))
            log(f"=== OK {name} seed={seed} attempt={attempt} rc=0 wall={wall}s "
                f"sha256={str(entry.get('sha256'))[:16]}… at {now()} ===")
        else:
            log(f"=== retry {name} rc={rc} attempt={attempt} wall={wall}s at {now()} ===")
        _dump_state(state)
        if rc == 0:
            return
        time.sleep(RETRY_SLEEP_S)


def main() -> int:
    os.makedirs(RUNS, exist_ok=True)
    state = {"experiment": "2657-fp8-hd128-head-dim", "driver_pid": os.getpid(),
             "started_at": now(), "lock_protocol": "2632/run.py::GpuLock（只读复用）",
             "lock_timeout_s": LOCK_TIMEOUT_S, "retry_sleep_s": RETRY_SLEEP_S,
             "runs_root": os.path.relpath(RUNS, ROOT), "arms": {}}
    _dump_state(state)
    log(f"=== driver start pid={os.getpid()} at {now()} ===")
    for i, (name, seed) in enumerate(ARMS):
        if i > 0 and os.path.exists(CUT_FILE):
            log(f"=== cut {name} (STOP_AFTER_FIRST_ARM 存在) at {now()} ===")
            state["arms"][name] = {"seed": seed, "status": "cut_by_stop_file"}
            _dump_state(state)
            break
        train_one(name, seed, state)
        if i + 1 < len(ARMS):
            log(f"=== arm gap 60s（§20.1 臂间窗口）at {now()} ===")
            time.sleep(60)
    state["finished_at"] = now()
    _dump_state(state)
    log(f"=== driver done at {now()} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
