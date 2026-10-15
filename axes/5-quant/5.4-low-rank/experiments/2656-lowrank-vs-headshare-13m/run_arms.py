"""2656 串行训练驱动（detach 用）：逐臂取 `/tmp/pathlm_gpu.lock` → 跑 `train_mla.py`。

为什么不用一条 bash one-liner（与派单里给的模板的**三处必要修正**，均已在本目录
如实登记）：
  ① run 名必须**逐 seed 唯一**（模板里两个 `mla_c128` 同名 ⇒ 后一个覆盖前一个的
     `model.pt`、两份日志追加到同一文件）；本驱动用 `mla-c128-s0/s1` 等；
  ② `gqa2_eq` 必须带 `--n-kv-heads 2`（模板漏了 ⇒ 会训成「MHA + 大 FFN」，不是
     头共享臂）；本驱动从 `config.json::arms` 读该字段；
  ③ `cwd` 必须是仓库根（`train_depth_ar.py` 用相对路径 `.tmp/enwik8` /
     `data/enwik8_full.npz` 载数据）；模板 `cd <expdir>` 会取不到数据。

其余语义与模板一致：**串行**、取锁后跑、失败 `sleep 20` 重试、等锁超时则稍后排队
（不杀他人进程、不回收活锁）、`train_driver.log` 追加机器可读时间戳行、
**部分完成也有价值**（先跑的臂先落盘，可中断续跑：已有 `runs/<run>/model.pt` 的臂跳过）。

用法（detach）：
    cd <repo> && nohup setsid python3 <expdir>/run_arms.py >> <expdir>/train_driver.log 2>&1 &
续跑：同一条命令再跑一次（已完成的臂跳过）。
加潜维扫描点：`python3 run_arms.py --include-optional`。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from gpu_lock import GpuLock                                     # noqa: E402

LOCK_TIMEOUT_S = 600          # 等锁上限 10 min（SCHEMA §20；超时不抢、稍后重排）
LOCK_RETRY_SLEEP_S = 60
TRAIN_RETRY_SLEEP_S = 20
MAX_TRAIN_ATTEMPTS = 3        # 单臂训练失败重试上限（避免坏配置无限占卡）


def log(line: str, stream=sys.stdout) -> None:
    print(f"=== {line} ===", flush=True, file=stream)


def cmd_for(arm: dict) -> list[str]:
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    m, t = cfg["model"], cfg["train"]
    run = os.path.join(HERE, "runs")
    cmd = [sys.executable, os.path.join(HERE, "train_mla.py"), arm["run_name"],
           "--d", str(m["d_model"]), "--layers", str(m["n_layers"]),
           "--heads", str(m["n_heads"]), "--mlp-mult", str(m["mlp_mult"]),
           "--dense", "--steps", str(t["steps"]),
           "--batch-size", str(t["batch_size"]), "--seed", str(arm["seed"]),
           "--lr", str(t["lr"]), "--eval-batches", str(t["eval_batches"]),
           "--out-root", run]
    if arm.get("n_kv_heads"):
        cmd += ["--n-kv-heads", str(arm["n_kv_heads"])]
    if arm.get("mlp_hidden_sched"):
        cmd += ["--mlp-hidden-sched",
                ",".join(str(v) for v in arm["mlp_hidden_sched"])]
    if arm.get("mla_latent"):
        cmd += ["--mla-latent", str(arm["mla_latent"])]
    return cmd


def smoke_ok() -> bool:
    """冒烟对账是否已成功落盘（失败/缺文件 ⇒ 下一个拿到锁的臂里重跑一次）。"""
    p = os.path.join(HERE, "smoke", "summary.json")
    if not os.path.exists(p):
        return False
    try:
        return json.load(open(p)).get("status") == "ok"
    except Exception:                                             # pragma: no cover
        return False


def run_arm(arm: dict, state: dict, smoke_first: bool = False) -> bool:
    run_dir = os.path.join(HERE, "runs", arm["run_name"])
    ckpt = os.path.join(run_dir, "model.pt")
    state[arm["label"]] = {"run_name": arm["run_name"], "status": "pending",
                           "attempts": 0, "cmd": " ".join(cmd_for(arm))}
    if os.path.exists(ckpt):
        log(f"skip {arm['label']} {arm['run_name']}（已有 model.pt，可中断续跑）")
        state[arm["label"]]["status"] = "done_earlier"
        return True
    os.makedirs(run_dir, exist_ok=True)
    for attempt in range(1, MAX_TRAIN_ATTEMPTS + 1):
        state[arm["label"]]["attempts"] = attempt
        lock = GpuLock(timeout_s=LOCK_TIMEOUT_S, poll_s=15)
        try:
            lock.acquire()
        except TimeoutError as e:
            log(f"lock-timeout {arm['label']} 第 {attempt} 次：{e}（{LOCK_RETRY_SLEEP_S}s 后重排）")
            state[arm["label"]]["status"] = "waiting_lock"
            state[arm["label"]]["last_error"] = str(e)
            time.sleep(LOCK_RETRY_SLEEP_S)
            continue
        try:
            cmd = cmd_for(arm)
            log(f"start {arm['label']} {arm['run_name']} (attempt {attempt}) "
                f"wait_lock={lock.ev['wait_seconds']}s")
            if smoke_first:
                # 冒烟对账（`--mla-latent 0` 与引擎逐位等价 + MLA 前向可跑）搭在
                # **第一次拿到的锁会话**里跑，省一次排队；失败不拦训练（如实记 rc）。
                rc_s = subprocess.call(
                    [sys.executable, os.path.join(HERE, "smoke_mla.py"), "--no-lock"],
                    cwd=ROOT)
                log(f"smoke rc={rc_s}（证据 smoke/summary.json）")
                state.setdefault("_smoke", {})["rc"] = rc_s
                if rc_s != 0:
                    state["_smoke"]["status"] = "failed"
            rc = subprocess.call(cmd, cwd=ROOT)
            state[arm["label"]]["rc"] = rc
            state[arm["label"]]["gpu_lock"] = lock.ev
            if rc == 0 and os.path.exists(ckpt):
                state[arm["label"]]["status"] = "done"
                log(f"OK {arm['label']} {arm['run_name']}")
                return True
            log(f"fail {arm['label']} {arm['run_name']} rc={rc}"
                f"（{TRAIN_RETRY_SLEEP_S}s 后重试）")
            state[arm["label"]]["status"] = "failed_retry"
        finally:
            lock.release()
        time.sleep(TRAIN_RETRY_SLEEP_S)
    state[arm["label"]]["status"] = "failed_give_up"
    log(f"GIVE-UP {arm['label']} {arm['run_name']}（{MAX_TRAIN_ATTEMPTS} 次失败，跳过）")
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-optional", action="store_true",
                    help="连可选臂（潜维扫描点 mla_c256）一起跑，排在最后")
    ap.add_argument("--only", default=None, help="只跑某个 label（调试用）")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    arms = [a for a in cfg["arms"] if a.get("train")]
    if not args.include_optional:
        arms = [a for a in arms if not a.get("optional")]
    if args.only:
        arms = [a for a in arms if a["label"] == args.only]
    arms = sorted(arms, key=lambda a: a["priority"])
    open(os.path.join(HERE, "train_driver.pid"), "w").write(f"{os.getpid()}\n")
    log(f"driver pid={os.getpid()} arms={[a['label'] for a in arms]} "
        f"root={ROOT}")
    state = {}
    for arm in arms:
        try:
            run_arm(arm, state, smoke_first=not smoke_ok())
        except Exception as e:                                   # pragma: no cover
            log(f"EXCEPTION {arm['label']}: {type(e).__name__}: {e}")
            state[arm["label"]]["status"] = "exception"
            state[arm["label"]]["last_error"] = f"{type(e).__name__}: {e}"
        json.dump({"pid": os.getpid(), "arms": state,
                   "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                  open(os.path.join(HERE, "train_driver.json"), "w"),
                  indent=2, ensure_ascii=False)
    log("driver finished")


if __name__ == "__main__":
    main()
