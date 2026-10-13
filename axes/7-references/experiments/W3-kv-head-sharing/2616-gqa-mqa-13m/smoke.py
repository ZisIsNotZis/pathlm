"""2616 · KV 头共享（ticket 33 契约 S）—— 冒烟 + 步时/显存探针。

纪律（SCHEMA §17.5/§17.8、methodology §10）：
  ① 任何新开关先 **20 步冒烟**（三臂各一次），再全跑；
  ② 成本必须按**本实验实测步时**估，不得借用别的实验的步时 ⇒ 先跑 200 步的步时探针，
     用训练日志里的墙钟（`min` 字段）算**边际步时**（步 0 → 步 199），**不**用 20 步的
     进程启动主导值外推；
  ③ 共享卡：子进程串行跑，逐 0.5 s 采样 `nvidia-smi --query-compute-apps` 拿每臂的
     实测峰值显存（net = 子进程峰值 − 运行时其他进程占用）。

用法：python3 smoke.py            # 三臂 × (20 步冒烟 + 200 步步时探针)
"""

import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

ARMS = [("mha", []), ("gqa", ["--n-kv-heads", "2"]), ("mqa", ["--n-kv-heads", "1"])]
BASE = ["--d", "256", "--layers", "12", "--heads", "8", "--mlp-mult", "6", "--dense",
        "--batch-size", "32", "--seed", "0", "--lr", "0.001"]


def gpu_procs() -> dict:
    """{pid: MiB}（当前占用 GPU 的计算进程）。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    d = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid, mem = [x.strip() for x in line.split(",")[:2]]
        d[int(pid)] = int(mem)
    return d


def gpu_device_used() -> int:
    """整卡占用 MiB（共享卡上下文，含其他任务）。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def train_arm(arm: str, extra: list, steps: int, tag: str) -> dict:
    """串行跑一臂，采样子进程显存峰值；返回实测成本。

    显存口径：`gpu_peak_child_mib` = nvidia-smi 报的**该子进程**占用峰值
    （= 本臂净占用）；`gpu_device_used_peak_mib` = 同期整卡占用（含其他任务）——
    两者分开报（共享卡纪律）。
    """
    run_name = f"{tag}-{arm}"
    log = os.path.join(HERE, "smoke", "runs", run_name, "stdout.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cmd = [sys.executable, os.path.join(ROOT, "train_depth_ar.py"), run_name,
           *BASE, "--steps", str(steps), "--eval-batches", "2", *extra,
           "--out-root", os.path.join(HERE, "smoke", "runs")]
    others0 = sum(m for p, m in gpu_procs().items() if p != os.getpid())
    t0 = time.time()
    peak = 0
    dev_peak = 0
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            used = gpu_procs().get(proc.pid)
            if used:
                peak = max(peak, used)
            dev_peak = max(dev_peak, gpu_device_used())
            time.sleep(0.5)
        rc = proc.returncode
    wall = time.time() - t0
    with open(os.path.join(HERE, "smoke", "runs", run_name, "cmd.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")
    if rc != 0:
        raise SystemExit(f"{run_name} 退出码 {rc}，见 {log}")
    rows = [json.loads(l) for l in
            open(os.path.join(HERE, "smoke", "runs", run_name, "train_log.jsonl"))]
    t_first, t_last = rows[0]["min"], rows[-1]["min"]
    n_between = rows[-1]["step"] - rows[0]["step"]
    res = json.load(open(os.path.join(HERE, "smoke", "runs", run_name,
                                      "results.json")))
    return {
        "arm": arm, "steps": steps, "run_name": run_name, "cmd": cmd,
        "stdout_log": os.path.relpath(log, ROOT),
        "marginal_seconds_per_step": round((t_last - t_first) * 60 / n_between, 4),
        "marginal_ms_per_step": round((t_last - t_first) * 60000 / n_between, 2),
        "steps_span_measured": [rows[0]["step"], rows[-1]["step"]],
        "wall_seconds_incl_startup": round(wall, 2),
        "gpu_peak_child_mib": peak,
        "gpu_device_used_peak_mib": dev_peak,
        "gpu_others_at_start_mib": others0,
        "gpu_others_peak_mib": max(dev_peak - peak, 0),
        "params": res["params"],
        "kv_head_liveness_summary": {
            k: res["kv_head_liveness"][k] for k in
            ("n_heads", "n_kv_heads", "head_dim", "qkv_out_width", "k_batch_shape",
             "kv_bytes_per_position_fp16", "kv_reduction_frac", "params_added",
             "alive")},
        "kv_heads_logged_lines": res["gnorm_monitor"]["kv_heads_n_logged_lines"],
        "loss_first": rows[0]["loss"], "loss_last": rows[-1]["loss"],
        "extrapolation_warning": ("边际步时来自步 0→199 的实测窗口；20 步冒烟那次的墙钟"
                                  "被进程启动主导，**不作为**外推依据（SCHEMA §17.5）"),
    }


def main():
    out = {"smoke_steps": 20, "timing_steps": 200, "arms": []}
    for arm, extra in ARMS:
        for steps, tag in ((20, "smoke20"), (200, "timeprobe200")):
            print(f"[smoke] {arm} {steps} 步 ...", flush=True)
            out["arms"].append(train_arm(arm, extra, steps, tag))
            print("  →", json.dumps(out["arms"][-1]["marginal_ms_per_step"]),
                  "ms/step, peak", out["arms"][-1]["gpu_peak_child_mib"], "MiB",
                  flush=True)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    json.dump(out, open(os.path.join(HERE, "smoke", "smoke_report.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
