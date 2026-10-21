"""2664 串行训练驱动（detach 用）：逐臂取 `/tmp/pathlm_gpu.lock` → 跑 `2656` 的 `train_mla.py`。

为什么不自造一条 bash one-liner：
  ① run 名必须**逐 seed 唯一**（同一臂的两个 seed 同名会互相覆盖 `model.pt`）；
  ② `cwd` 必须是仓库根（`train_depth_ar.py` 用相对路径 `.tmp/enwik8` / `data/enwik8_full.npz` 载数据）；
  ③ 取锁、失败重试、可中断续跑（已有 `model.pt` 的臂跳过）要落在文件里，不能只存在记忆里。

臂清单与命令行**只从 `config.json::model_arms` 派生**（`runs[].source` 以 `train:` 开头的才有训练），
本脚本不另存一份臂表。

用法（detach）：
    cd <repo> && nohup setsid python3 <expdir>/train_arms.py >> <expdir>/train_driver.log 2>&1 &
续跑：同一条命令再跑一次（已完成的臂跳过）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from gpu_lock import GpuLock                                        # noqa: E402
from tools.prereg_bind import require as require_prereg             # noqa: E402
from tools.run_provenance import code_key, sha256_file               # noqa: E402

CFG = json.loads(Path(HERE, "config.json").read_text(encoding="utf-8"))
TRAINER = os.path.join(ROOT, CFG["train"]["trainer"])
LOCK_TIMEOUT_S = 600            # 等锁上限 10 min（SCHEMA §20；超时重排，不抢不杀）
LOCK_RETRY_SLEEP_S = 60
TRAIN_RETRY_SLEEP_S = 20
MAX_TRAIN_ATTEMPTS = 3


def log(line: str) -> None:
    print(f"=== {line} ===", flush=True)


def train_arms() -> list[dict]:
    """`config.json` 里 `source` 以 `train:` 开头的 (臂, run) 对，按 config 顺序。"""
    out = []
    for arm in CFG["model_arms"]:
        for run in arm["runs"]:
            if not str(run.get("source", "")).startswith("train:"):
                continue
            out.append({"label": arm["label"], "run_name": run["run"], "seed": run["seed"],
                        "structure": arm["structure"], "latent": arm["latent"],
                        "mlp_hidden_sched": arm.get("mlp_hidden_sched"),
                        "n_kv_heads": arm["n_kv_heads"]})
    return out


def cmd_for(arm: dict) -> list[str]:
    m, t = CFG["model"], CFG["train"]
    cmd = [sys.executable, TRAINER, arm["run_name"],
           "--d", str(m["d_model"]), "--layers", str(m["n_layers"]),
           "--heads", str(m["n_heads"]), "--mlp-mult", str(m["mlp_mult"]),
           "--dense", "--steps", str(t["steps"]),
           "--batch-size", str(t["batch_size"]), "--seed", str(arm["seed"]),
           "--lr", str(t["lr"]), "--eval-batches", str(t["eval_batches"]),
           "--out-root", os.path.join(HERE, "runs")]
    if arm.get("n_kv_heads"):
        cmd += ["--n-kv-heads", str(arm["n_kv_heads"])]
    if arm.get("mlp_hidden_sched"):
        cmd += ["--mlp-hidden-sched", ",".join(str(v) for v in arm["mlp_hidden_sched"])]
    if arm.get("latent"):
        cmd += ["--mla-latent", str(arm["latent"])]
    return cmd


def stamp_provenance() -> list[str]:
    """给**本实验**训出来的每份 `runs/<run>/results.json` 记代码指纹（缺则记，幂等）。

    训练器 `axes/5-quant/5.4-low-rank/experiments/2656-lowrank-vs-headshare-13m/train_mla.py`
    是引擎的旧副本，写 `results.json` 时不记代码哈希；而本实验已预登记绑定，`tools/check_tree.py`
    要求**每一份读数**都记着产出它的代码。

    这里记的是**真实产出者**：`train_mla.py`（训练器）+ `train_arms.py`（驱动）+ 本实验的
    `config.json` / `run.py`，**不写** `code_hash_run_py`（训练读数不是 `run.py` 产的，宁缺勿假）。
    为什么现在算的哈希等于产出时的哈希：这些文件都在库里且本轮**未改动**（`git status` 可核），
    所以磁盘上的字节就是产出那一刻的字节。
    """
    stamped = []
    for arm in train_arms():
        run_dir = os.path.join(HERE, "runs", arm["run_name"])
        res = os.path.join(run_dir, "results.json")
        if not os.path.exists(res):
            continue
        rec = json.loads(Path(res).read_text(encoding="utf-8"))
        if rec.get("code_hashes"):
            continue
        files = [TRAINER, os.path.join(HERE, "train_arms.py"),
                 os.path.join(HERE, "config.json")]
        rec["code_hashes"] = {code_key(f): sha256_file(f) for f in files if os.path.isfile(f)}
        rec["code_hashes_note"] = ("由 train_mla.py（引擎逐字副本 + 3 处 patch，见实验 README §7）与 "
                                   "train_arms.py 产出；run.py 只做评测，不产这份读数")
        Path(res).write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        stamped.append(arm["run_name"])
    return stamped


def run_arm(arm: dict, state: dict) -> bool:
    run_dir = os.path.join(HERE, "runs", arm["run_name"])
    ckpt = os.path.join(run_dir, "model.pt")
    key = f"{arm['label']}/{arm['run_name']}"
    state[key] = {"run_name": arm["run_name"], "status": "pending", "attempts": 0,
                  "cmd": " ".join(cmd_for(arm))}
    if os.path.exists(ckpt):
        log(f"skip {key}（已有 model.pt，可中断续跑）")
        state[key]["status"] = "done_earlier"
        return True
    os.makedirs(run_dir, exist_ok=True)
    for attempt in range(1, MAX_TRAIN_ATTEMPTS + 1):
        state[key]["attempts"] = attempt
        lock = GpuLock(timeout_s=LOCK_TIMEOUT_S, poll_s=15)
        try:
            lock.acquire()
        except TimeoutError as e:
            log(f"lock-timeout {key} 第 {attempt} 次：{e}（{LOCK_RETRY_SLEEP_S}s 后重排）")
            state[key]["status"] = "waiting_lock"
            state[key]["last_error"] = str(e)
            time.sleep(LOCK_RETRY_SLEEP_S)
            continue
        try:
            log(f"start {key} (attempt {attempt}) wait_lock={lock.ev['wait_seconds']}s")
            rc = subprocess.call(cmd_for(arm), cwd=ROOT)
            state[key]["rc"] = rc
            state[key]["gpu_lock"] = lock.ev
            if rc == 0 and os.path.exists(ckpt):
                state[key]["status"] = "done"
                state[key]["provenance_stamped"] = stamp_provenance()
                log(f"OK {key}")
                return True
            log(f"fail {key} rc={rc}（{TRAIN_RETRY_SLEEP_S}s 后重试）")
            state[key]["status"] = "failed_retry"
        finally:
            lock.release()
        time.sleep(TRAIN_RETRY_SLEEP_S)
    state[key]["status"] = "failed_give_up"
    log(f"GIVE-UP {key}（{MAX_TRAIN_ATTEMPTS} 次失败，跳过）")
    return False


def main() -> int:
    if "--stamp-only" in sys.argv:
        done = stamp_provenance()
        log(f"stamped: {done or '（无缺指纹的读数）'}")
        return 0
    if require_prereg(Path(HERE)) != 0:
        raise SystemExit("预登记未绑定：先跑 python3 tools/prereg_bind.py "
                         "axes/5-quant/5.4-low-rank/experiments/2664-lowrank-vs-quant-13m")
    arms = train_arms()
    Path(os.path.join(HERE, "train_driver.pid")).write_text(f"{os.getpid()}\n")
    log(f"driver pid={os.getpid()} arms={[a['label'] + '/' + a['run_name'] for a in arms]} "
        f"root={ROOT}")
    state: dict = {}
    ok = True
    for arm in arms:
        ok = run_arm(arm, state) and ok
        Path(os.path.join(HERE, "train_driver.json")).write_text(
            json.dumps({"pid": os.getpid(), "arms": state,
                        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                       indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"provenance stamped at end: {stamp_provenance() or '（无缺指纹的读数）'}")
    log("driver finished" if ok else "driver finished with failures")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
