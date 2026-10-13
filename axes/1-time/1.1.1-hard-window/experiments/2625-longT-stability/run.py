"""2625 · 长 T 训练稳定性修复 —— 实验入口（ticket 42-W1）。

契约：同目录 `README.md`（判据线逐字抄录，不得修改）。

阶段（每个 GPU 阶段**按臂取/放锁**，臂间留 ≥60s 窗口，SCHEMA §20/§20.1）：
    python3 run.py --stage diag                  # 诊断臂：baseline seed1，7000 步 + 探针
    python3 run.py --stage screen                # 候选筛查：seed1，6500 步 / 候选
    python3 run.py --stage confirm               # 胜出候选：seed0+seed1，12000 步
    python3 run.py --stage shortcheck            # 短 T 不回归：T=256/512 逐位/Δbpc
    python3 run.py --stage report                # 汇总 → results.json / stability_report.md

先测步时再定预算（§20.3）：`--stage preflight` 跑 1 臂 20 步实测步时/峰值，再决定
每臂步数上限。
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

LOCK = "/tmp/pathlm_gpu.lock"
LOCK_MAX_WAIT = 7200
DRIVER = os.path.join(HERE, "stability_driver.py")

#: 诊断臂：seed1 在 ~step 6200 爆炸 ⇒ 7000 步足以看到「爆炸前漂移 + 爆炸」。
DIAG = {"arm": "full-1024-s1-diag", "candidate": "none", "seq_len": 1024,
        "window": 0, "seed": 1, "steps": 7000, "batch_size": 8, "lr": 1e-3}
#: 候选筛查臂（seed1，爆炸最早 ⇒ 最短步数即可判）：6500 步。
SCREEN_STEPS = 6500
SCREEN_CANDIDATES = [
    {"arm": "full-1024-s1-temp2", "candidate": "temp2.0"},
    {"arm": "full-1024-s1-qknorm", "candidate": "qk-norm"},
    {"arm": "full-1024-s1-warmup2000", "candidate": "warmup2000"},
    {"arm": "full-1024-s1-wd01", "candidate": "wd0.1"},
]
#: 确认臂：胜出候选在两 seed 上跑满 12000 步（判据①）。
CONFIRM_STEPS = 12000


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gpu_acquire(max_wait_s: int = 600, poll_s: int = 15) -> None:
    t0 = time.time()
    while True:
        try:
            os.mkdir(LOCK)
            break
        except FileExistsError:
            pass
        hp = None
        try:
            hp = int(open(os.path.join(LOCK, "holder_pid")).read().strip())
        except (OSError, ValueError):
            hp = None
        stale, reason = False, ""
        if hp is not None and not _pid_alive(hp):
            stale, reason = True, f"holder_pid {hp} 已死"
        if hp is None and os.path.isdir(LOCK) and \
                time.time() - os.path.getmtime(LOCK) > 900:
            stale, reason = True, "无 holder_pid 且锁龄 >15min"
        if stale:
            print(f"[lock] reclaim: {reason}", flush=True)
            for f in ("holder_pid", "heartbeat"):
                try:
                    os.remove(os.path.join(LOCK, f))
                except OSError:
                    pass
            try:
                os.rmdir(LOCK)
            except OSError:
                pass
            continue
        if time.time() - t0 > max_wait_s:
            raise TimeoutError("GPU 锁等待超时（未杀任何进程）")
        time.sleep(poll_s)
    open(os.path.join(LOCK, "holder_pid"), "w").write(str(os.getpid()))
    open(os.path.join(LOCK, "heartbeat"), "w").write(str(time.time()))


def gpu_release() -> None:
    for f in ("holder_pid", "heartbeat"):
        try:
            os.remove(os.path.join(LOCK, f))
        except OSError:
            pass
    try:
        os.rmdir(LOCK)
    except OSError:
        pass


def gpu_touch() -> None:
    try:
        open(os.path.join(LOCK, "heartbeat"), "w").write(str(time.time()))
    except OSError:
        pass


def free_gpu_mb() -> float | None:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        return float(q.stdout.strip().splitlines()[0])
    except Exception:                                        # noqa: BLE001
        return None


def train_arm(spec: dict, poll_s: int = 20) -> dict:
    """取锁 → 前台跑一个臂（轮询心跳）→ 放锁；返回该臂摘要。"""
    run_dir = os.path.join(HERE, "runs", spec["arm"])
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, DRIVER, spec["arm"],
           "--seq-len", str(spec["seq_len"]), "--window", str(spec["window"]),
           "--steps", str(spec["steps"]), "--batch-size", str(spec["batch_size"]),
           "--lr", str(spec["lr"]), "--seed", str(spec["seed"]),
           "--candidate", spec["candidate"],
           "--out-root", os.path.join(HERE, "runs")]
    before = free_gpu_mb()
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    t0 = time.time()
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env=env)
        print(f"[run] {spec['arm']}: pid={proc.pid} steps={spec['steps']} "
              f"cand={spec['candidate']} free_before={before}MB", flush=True)
        while proc.poll() is None:
            gpu_touch()
            time.sleep(poll_s)
        if proc.returncode != 0:
            tail = ""
            try:
                tail = open(os.path.join(run_dir, "stdout.log")).read()[-800:]
            except OSError:
                pass
            raise RuntimeError(f"{spec['arm']} rc={proc.returncode}\n{tail}")
    finally:
        log.close()
        gpu_release()
    after = free_gpu_mb()
    wall = (time.time() - t0) / 60
    res = json.load(open(os.path.join(run_dir, "results.json")))
    summary = {"arm": spec["arm"], "candidate": spec["candidate"],
               "seed": spec["seed"], "seq_len": spec["seq_len"],
               "window": spec["window"], "steps": spec["steps"],
               "wall_min": round(wall, 2),
               "seconds_per_step": round(wall * 60 / spec["steps"], 5),
               "gnorm_max": res["gnorm_monitor"]["gnorm_max"],
               "gnorm_mean": res["gnorm_monitor"]["gnorm_mean"],
               "grad_skips": res["gnorm_monitor"]["grad_skips"],
               "peak_torch_gb": round(res["peak_gpu_mem_mb"] / 1024, 3),
               "final_qkv_w_norm": res["final_qkv_w_norm"],
               "free_before_mb": before, "free_after_mb": after}
    with open(os.path.join(HERE, f"{spec['arm']}-train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[run] {spec['arm']} done wall={wall:.2f}min "
          f"gnorm_max={summary['gnorm_max']} skips={summary['grad_skips']}",
          flush=True)
    return summary


def stage_diag() -> None:
    train_arm(DIAG)
    time.sleep(60)                                    # §20.1 臂间窗口


def stage_screen(cands: list[str]) -> None:
    for i, c in enumerate(SCREEN_CANDIDATES):
        if cands and c["arm"] not in cands:
            continue
        train_arm({"arm": c["arm"], "candidate": c["candidate"], "seq_len": 1024,
                   "window": 0, "seed": 1, "steps": SCREEN_STEPS,
                   "batch_size": 8, "lr": 1e-3})
        time.sleep(60)                                # §20.1 臂间窗口


def stage_try(candidate: str, seeds: list[int], steps: int, seq_len: int,
              window: int, batch_size: int, lr: float, tag: str | None = None) -> None:
    """通用：对候选跑一个或多个种子（每个臂独立取/放锁 + ≥60s 窗口）。

    **关键纪律（本实验实测后修正）**：`train()` 的余弦调度依赖 `steps`，
    所以「缩短步数」会同时改 LR 轨迹（§20.3 的快照口径在这里会影响主判据）
    ⇒ 要复现/排除爆炸，**必须用 steps=12000**（与 2623/2624 同调度）。
    """
    t = tag or candidate.replace(".", "p").replace("-", "")
    for s in seeds:
        arm = f"full-T{seq_len}-{t}-s{s}" + ("-w512" if window else "")
        train_arm({"arm": arm, "candidate": candidate, "seq_len": seq_len,
                   "window": window, "seed": s, "steps": steps,
                   "batch_size": batch_size, "lr": lr})
        time.sleep(60)


def stage_eval(arms: list[str], candidates: dict[str, str], seq_len: int,
               window: int) -> None:
    for arm in arms:
        cand = candidates.get(arm, "none")
        subprocess.run([sys.executable, os.path.join(HERE, "eval_bpc.py"), arm,
                        "--candidate", cand, "--seq-len", str(seq_len),
                        "--window", str(window), "--out-root",
                        os.path.join(HERE, "runs")], cwd=ROOT, check=True)


def stage_preflight() -> None:
    train_arm({"arm": "preflight-20", "candidate": "none", "seq_len": 1024,
               "window": 0, "seed": 0, "steps": 20, "batch_size": 8, "lr": 1e-3})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("preflight", "diag", "screen", "try", "eval",
                             "report"))
    ap.add_argument("--candidate", default="none")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--steps", type=int, default=CONFIRM_STEPS)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    if args.stage == "preflight":
        stage_preflight()
    elif args.stage == "diag":
        stage_diag()
    elif args.stage == "screen":
        stage_screen([a for a in args.arms.split(",") if a])
    elif args.stage == "try":
        stage_try(args.candidate, [int(s) for s in args.seeds.split(",")],
                  args.steps, args.seq_len, args.window, args.batch_size,
                  args.lr, args.tag)
    elif args.stage == "eval":
        arms = [a for a in args.arms.split(",") if a]
        stage_eval(arms, {}, args.seq_len, args.window)
    else:
        raise SystemExit("report 阶段由 reconcile_numbers.py 完成")


if __name__ == "__main__":
    main()
