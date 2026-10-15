"""2656 冒烟对账（Phase 1 证据）：`--mla-latent 0` 与引擎逐位等价 + MLA 前向可跑。

三条（同一 GPU，短时；取 `/tmp/pathlm_gpu.lock`）：
  ① 引擎 `train_depth_ar.py` vs 本副本 `train_mla.py --mla-latent 0`：同 seed、
     同配方、同步数，逐行比 `train_log.jsonl` 的 loss 序列（期望**逐位相同**）；
  ② 本副本 `--mla-latent 128`：训练可跑、loss 有限、`results.json` 落 MLA 活性块
     （实际潜向量/K/V 张量形状）；
  ③ 50 步（契约要求）与 201 步（多一个中点对账）各做一次 ①，防止只在极短程偶然一致。

不给机器可读的解释就不算证据：本脚本把每步 loss、最大偏差与判定写进
`smoke/summary.json`。
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
from mla_arms import eqparams_solution                           # noqa: E402


def recipe(steps: int, seed: int = 0, latent: int = 0,
           out_root: str = "", run_name: str = "") -> list[str]:
    cmd = [sys.executable, os.path.join(HERE, "train_mla.py"), run_name,
           "--d", "256", "--layers", "12", "--heads", "8", "--mlp-mult", "6",
           "--dense", "--steps", str(steps), "--batch-size", "32",
           "--seed", str(seed), "--lr", "0.001", "--eval-batches", "2",
           "--out-root", out_root]
    if latent:
        cmd += ["--mla-latent", str(latent)]
    return cmd


def losses(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, "train_log.jsonl")
    return [json.loads(l) for l in open(path)]


def compare(a_dir: str, b_dir: str) -> dict:
    la, lb = losses(a_dir), losses(b_dir)
    assert len(la) == len(lb), f"日志行数不同：{len(la)} vs {len(lb)}"
    rows = [{"step": x["step"], "engine_loss": x["loss"], "copy_loss": y["loss"],
             "diff": round(y["loss"] - x["loss"], 10)}
            for x, y in zip(la, lb)]
    diffs = [abs(r["diff"]) for r in rows]
    return {"n_lines": len(rows), "rows": rows,
            "max_abs_loss_diff": max(diffs), "bitwise_identical": max(diffs) == 0.0}


#: 2616 的「同 seed 同配方重训」对照（**独立先验证据**：两次都由未改动的引擎训练）
NONDET_CONTROL_RUNS = {
    "run_a": "axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/gqa-12k-s0",
    "run_b": "axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/gqa-12k-s0-rep",
}


def nondeterminism_control() -> dict:
    """本仓「同 seed 同配方重训」的 run-to-run 散度（**引擎自己的**读数）。

    用途：把「副本 vs 引擎不等」是否只是 CUDA 归约序非确定性这件事**证伪/证实**——
    若引擎自己重训就已经在同步数上差出同量级或更大的 loss，则副本与引擎的差异不构成
    「patch 改了语义」的证据（起点 loss 逐位相同 ⇒ 初始化与数据序一致）。
    读数来自 2616 已落盘的两条引擎训练日志（`gqa-12k-s0` 与 `gqa-12k-s0-rep`；
    2616/stability.md §2 已单列该结论）。
    """
    out = {"source": "2616-gqa-mqa-13m/runs（引擎训练，未经本实验 patch）",
           "role": "run-to-run 非确定性对照（不是本实验的判据，只用来解释副本/引擎差异）"}
    try:
        a = [json.loads(l) for l in open(os.path.join(
            ROOT, NONDET_CONTROL_RUNS["run_a"], "train_log.jsonl"))]
        b = [json.loads(l) for l in open(os.path.join(
            ROOT, NONDET_CONTROL_RUNS["run_b"], "train_log.jsonl"))]
    except OSError as e:                                          # pragma: no cover
        out["error"] = str(e)
        return out
    rows = [{"step": x["step"], "engine_a_loss": x["loss"], "engine_b_loss": y["loss"],
             "diff": round(y["loss"] - x["loss"], 6)}
            for x, y in zip(a, b)]
    out["rows_sample"] = [r for r in rows if r["step"] in (0, 200, 400, 1000, 4000, 8000, 11999)]
    out["step0_loss_equal"] = rows[0]["diff"] == 0.0
    out["max_abs_diff_over_run"] = max(abs(r["diff"]) for r in rows)
    out["final_loss_diff"] = rows[-1]["diff"]
    out["note"] = ("同 seed、同配方、同数据序（首行 loss 逐位相同）⇒ 差异只能来自 CUDA "
                   "归约顺序；2616/stability.md §2 记末值 clean bpc 差 0.0004")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-lock", action="store_true",
                    help="调用方（run_arms.py）已持有 /tmp/pathlm_gpu.lock，本进程不再取锁")
    ap.add_argument("--lock-timeout", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
    t0 = time.time()
    out = {"engine_train_sha256": None, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    import hashlib

    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()

    out["engine_train_sha256"] = sha(os.path.join(ROOT, "train_depth_ar.py"))
    out["train_mla_sha256"] = sha(os.path.join(HERE, "train_mla.py"))
    out["mla_arms_sha256"] = sha(os.path.join(HERE, "mla_arms.py"))
    out["eqparams"] = eqparams_solution()
    out["nondeterminism_control"] = nondeterminism_control()

    lock = GpuLock(timeout_s=args.lock_timeout, poll_s=15)
    try:
        if not args.no_lock:
            lock.acquire()
    except TimeoutError as e:
        out["status"] = "lock_timeout"
        out["error"] = str(e)
        json.dump(out, open(os.path.join(HERE, "smoke", "summary.json"), "w"),
                  indent=2, ensure_ascii=False)
        print(json.dumps(out, ensure_ascii=False))
        raise SystemExit(f"等锁超时：{e}")
    out["gpu_lock"] = lock.ev
    try:
        os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
        runs = {}
        for steps in (50, 201):
            e_root = os.path.join(HERE, "smoke", f"engine{steps}")
            c_root = os.path.join(HERE, "smoke", f"latent0_{steps}")
            e_run = os.path.join(e_root, f"mha-{steps}step")
            c_run = os.path.join(c_root, f"mha-{steps}step")
            # ① 引擎（原文件）
            if not os.path.exists(os.path.join(e_run, "train_log.jsonl")):
                rc = subprocess.call(
                    [sys.executable, os.path.join(ROOT, "train_depth_ar.py"),
                     f"mha-{steps}step", "--d", "256", "--layers", "12",
                     "--heads", "8", "--mlp-mult", "6", "--dense",
                     "--steps", str(steps), "--batch-size", "32", "--seed", "0",
                     "--lr", "0.001", "--eval-batches", "2",
                     "--out-root", e_root], cwd=ROOT)
                assert rc == 0, f"引擎冒烟失败 rc={rc}"
            # ② 副本 --mla-latent 0
            if not os.path.exists(os.path.join(c_run, "train_log.jsonl")):
                rc = subprocess.call(recipe(steps, 0, 0, c_root, f"mha-{steps}step"),
                                     cwd=ROOT)
                assert rc == 0, f"副本（latent 0）冒烟失败 rc={rc}"
            runs[f"latent0_vs_engine_{steps}step"] = compare(e_run, c_run)
            runs[f"latent0_vs_engine_{steps}step"]["engine_run"] = os.path.relpath(
                e_run, ROOT)
            runs[f"latent0_vs_engine_{steps}step"]["copy_run"] = os.path.relpath(
                c_run, ROOT)
        # ③ MLA（d_c=128）训练可跑冒烟
        m_root = os.path.join(HERE, "smoke", "mla128")
        m_run = os.path.join(m_root, "mla-c128-50step")
        rc = subprocess.call(recipe(50, 0, 128, m_root, "mla-c128-50step"),
                             cwd=ROOT)
        runs["mla_c128_50step"] = {"rc": rc,
                                   "run": os.path.relpath(m_run, ROOT)}
        assert rc == 0, f"MLA 冒烟失败 rc={rc}"
        res = json.load(open(os.path.join(m_run, "results.json")))
        runs["mla_c128_50step"].update({
            "params": res["params"], "mla_latent": res["mla_latent"],
            "kv_values_per_position": res["kv_head_liveness"]["kv_values_per_position"],
            "kv_bytes_per_position_fp16":
                res["kv_head_liveness"]["kv_bytes_per_position_fp16"],
            "c_batch_shape": res["kv_head_liveness"]["c_batch_shape"],
            "alive": res["kv_head_liveness"]["alive"],
            "loss_first": losses(m_run)[0]["loss"],
            "loss_last": losses(m_run)[-1]["loss"],
        })
        # 等价性总判定：50 步与 201 步两处都必须逐位相同
        eq = [runs[f"latent0_vs_engine_{s}step"]["bitwise_identical"]
              for s in (50, 201)]
        out["latent0_engine_equivalence_bitwise"] = all(eq)
        out["runs"] = runs
        out["status"] = "ok"
        out["wall_seconds"] = round(time.time() - t0, 1)
    finally:
        out["gpu_lock_release"] = lock.ev
        if not args.no_lock:
            lock.release()
    json.dump(out, open(os.path.join(HERE, "smoke", "summary.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
