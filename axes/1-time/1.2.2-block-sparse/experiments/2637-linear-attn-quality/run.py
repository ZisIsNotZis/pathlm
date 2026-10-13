"""2637 · 线性注意力质量腿（A6）—— 实验入口。

契约：同目录 `README.md`（判据线预登记，不得修改）。

干预（唯一变量）：注意力算子 ∈ {全因果 softmax（full）, 归一化线性核（linear，elu+1）}，
T=256，bs8，seed0，lr1e-3，enwik8，**共同步数 S=6000**。为把「实现停摆」与「算子质量」
分开，另加一档**稳定化分母**（给 qkv 挂 2625 的逐头 L2 QK-Norm hook，零新增参数）
⇒ 2×2 臂（full/linear × 无/有稳定化）。

阶段：
    python3 run.py --stage preflight           # 四臂各 20 步冒烟 + 实测步时
    python3 run.py --stage train --arm <arm>   # 训练一个臂（带停摆看门狗）
    python3 run.py --stage train-all           # 依次训练四臂（每臂间放锁 ≥60s）
    python3 run.py --stage diag                # 停摆诊断矩阵（短程，记录 pass/stall）
    python3 run.py --stage eval                # 电池（bpc + 活性量 + 仪器自检）
    python3 run.py --stage report              # 汇总 → results.json / config.json

禁外推：只描述 T=256 这一点；成本与质量口径见契约。
"""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel       # noqa: E402
from pathlm.data import load_enwik8_full                      # noqa: E402
from tools.evalkit.retention import masked_bpc, masked_logits  # noqa: E402

DEVICE = "cuda"
DATA = "data/enwik8_full.npz"
D, L, H, MLP, BS, LR, SEED = 256, 8, 8, 4, 8, 1e-3, 0
S = 6000
QK_NORM_MIN_T = 512
DRIVER = os.path.join(HERE, "linear_driver.py")
EXP_ROOT = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator", "experiments")
EVAL_BATCHES, EVAL_BS, WINDOW_SEED = 20, 16, 2
LOCK = "/tmp/pathlm_gpu.lock"
LOCK_MAX_WAIT = 7200

ARMS = [
    {"arm": "full-256-lin",     "linear": False, "qk": "none",   "role": "T=256 · 全因果 softmax"},
    {"arm": "linear-256-lin",   "linear": True,  "qk": "none",   "role": "T=256 · 线性核（elu+1，原实装）"},
    {"arm": "full-256-lin-qk",  "linear": False, "qk": "global", "role": "T=256 · 全因果 + QK-Norm（稳定化对照）"},
    {"arm": "linear-256-lin-qk", "linear": True, "qk": "global", "role": "T=256 · 线性核 + QK-Norm（稳定化分母）"},
]
ARM_BY_NAME = {a["arm"]: a for a in ARMS}
OTHER_MACS = 3 * D * D + D * D + 2 * MLP * D * D


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cfg_for(linear: bool) -> DepthARConfig:
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=256,
                         mlp_mult=MLP, dense=True, attn_linear=linear,
                         linear_chunk=64)


def _load_w1():
    spec = importlib.util.spec_from_file_location(
        "w1_stability_driver_lin_run",
        os.path.join(EXP_ROOT, "2625-longT-stability", "stability_driver.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------- GPU 锁

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


# --------------------------------------------------------------- 训练一个臂

def _read_last_step(log_path: str) -> int | None:
    try:
        with open(log_path) as f:
            lines = f.readlines()
        if not lines:
            return None
        return int(json.loads(lines[-1])["step"])
    except (OSError, ValueError, KeyError, IndexError):
        return None


def train_arm(spec: dict, steps: int, poll_s: int = 20, caps_min: float = 40.0,
              stall_grace_s: float = 210.0, stall_dump: int = 0) -> dict:
    """后台跑 `linear_driver.py`，带**停摆看门狗**：若进程存活但训练步号停滞超过
    `stall_grace_s` 秒，判定停摆并按 `stall_dump` 抓栈后终止（记录触发步）。"""
    arm = spec["arm"]
    run_dir = os.path.join(HERE, "runs", arm)
    os.makedirs(run_dir, exist_ok=True)
    stdout_path = os.path.join(run_dir, "stdout.log")
    log = open(stdout_path, "w")
    cmd = [sys.executable, DRIVER, arm,
           "--seq-len", "256", "--steps", str(steps), "--batch-size", str(BS),
           "--lr", str(LR), "--seed", str(SEED), "--out-root",
           os.path.join(HERE, "runs"), "--qk-norm", spec["qk"]]
    if spec["linear"]:
        cmd.append("--linear")
    if stall_dump:
        cmd += ["--stall-dump", str(stall_dump)]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] {arm}: started pid={proc.pid} steps={steps} "
          f"linear={spec['linear']} qk={spec['qk']}", flush=True)
    peak, t0 = 0.0, time.time()
    last_step, last_step_t = None, time.time()
    stall = False
    log_path = os.path.join(run_dir, "train_log.jsonl")
    while proc.poll() is None:
        gpu_touch()
        if time.time() - t0 > caps_min * 60:
            proc.terminate()
            raise TimeoutError(f"{arm} 训练超过 {caps_min} min cap")
        st = _read_last_step(log_path)
        if st is not None and st != last_step:
            last_step, last_step_t = st, time.time()
            if st % 2000 == 0 and st > 0:
                print(f"[run] {arm} step {st}", flush=True)
        elif st is not None and time.time() - last_step_t > stall_grace_s:
            stall = True
            print(f"[run] {arm}: STALL detected at step {st} "
                  f"(no progress for {stall_grace_s:.0f}s)", flush=True)
            break
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
            for line in q.stdout.strip().splitlines():
                pid, mem = [s.strip() for s in line.split(",")]
                if int(pid) == proc.pid:
                    peak = max(peak, float(mem) / 1024.0)
        except Exception:                                    # noqa: BLE001
            pass
        time.sleep(poll_s)
    if stall:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
    log.close()
    wall = (time.time() - t0) / 60
    if stall:
        tail = ""
        try:
            tail = open(stdout_path).read()[-2000:]
        except OSError:
            pass
        out = {"arm": arm, "stalled": True, "stall_step": last_step,
               "wall_min": round(wall, 2), "stdout_tail": tail,
               "steps_requested": steps, "patch": spec}
    else:
        if proc.returncode != 0:
            tail = ""
            try:
                tail = open(stdout_path).read()[-1200:]
            except OSError:
                pass
            raise RuntimeError(f"{arm} 训练失败 rc={proc.returncode}"
                               f"\n--- stdout tail ---\n{tail}")
        res = json.load(open(os.path.join(run_dir, "results.json")))
        out = {"arm": arm, "stalled": False, "run_dir": os.path.relpath(run_dir, HERE),
               "wall_min": round(wall, 2),
               "seconds_per_step": round(wall * 60 / steps, 5),
               "peak_child_mem_gb_nvidia_smi": round(peak, 2),
               "peak_torch_gb": round(res["peak_gpu_mem_mb"] / 1024.0, 3),
               "params": res["params"], "tokens": res["tokens"],
               "patch": res["patch"], "gnorm_monitor": res["gnorm_monitor"],
               "sha256_model_pt": sha256_file(os.path.join(run_dir, "model.pt")),
               "size_bytes": os.path.getsize(os.path.join(run_dir, "model.pt")),
               "steps": steps}
    json.dump(out, open(os.path.join(HERE, f"{arm}-train_summary.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[run] {arm}: stalled={out['stalled']} wall={wall:.2f}min "
          + (f"step={out.get('stall_step')}" if out["stalled"]
             else f"gnorm_max={out['gnorm_monitor']['gnorm_max']}"), flush=True)
    return out


# --------------------------------------------------------------- 成本口径

def _timed(fn, warmup: int = 3, iters: int = 8) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": round(statistics.median(ts), 3), "n_iters": iters}


def model_step_cost(linear: bool, qk: str, iters: int = 6) -> dict:
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    model = DepthARModel(cfg_for(linear), 206).cuda()
    if qk == "global":
        _load_w1().apply_attn_scale(model, "qk-norm", 1.0, min_t=0)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    x = torch.randint(0, 206, (BS, 256)).cuda()

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(x, exit_depths=None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    t = _timed(step, warmup=2, iters=iters)
    t.update({"linear": linear, "qk": qk, "batch_size": BS,
              "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)})
    del model, opt
    torch.cuda.empty_cache()
    return t


def cost_probe(iters: int = 6) -> dict:
    out = {"free_mb_before": free_gpu_mb(), "steps": {}}
    for spec in ARMS:
        out["steps"][spec["arm"]] = model_step_cost(spec["linear"], spec["qk"],
                                                    iters=iters)
    full = out["steps"]["full-256-lin"]["median_ms"]
    lin = out["steps"]["linear-256-lin"]["median_ms"]
    out["ratio_linear_over_full"] = round(lin / full, 4)
    out["free_mb_after"] = free_gpu_mb()
    out["theory"] = {
        "attn_macs_per_token_per_layer_full": 2 * 256 * D,
        "attn_macs_per_token_per_layer_linear_chunk64": 2 * D * (D // H) + 2 * 64 * D,
        "attn_macs_per_token_per_layer_linear_streaming": 2 * D * (D // H),
        "note": "解析口径；实测为准（§17.11）",
    }
    return out


# --------------------------------------------------------------- 电池

def load_arm(spec: dict, vocab: int) -> DepthARModel:
    torch.manual_seed(SEED)
    m = DepthARModel(cfg_for(spec["linear"]), vocab).cuda()
    if spec["qk"] == "global":
        _load_w1().apply_attn_scale(m, "qk-norm", 1.0, min_t=0)
    m.load_state_dict(torch.load(os.path.join(HERE, "runs", spec["arm"], "model.pt"),
                                 map_location="cuda"))
    m.eval()
    return m


def battery(spec: dict, eval_arr, model: DepthARModel) -> dict:
    bpc = masked_bpc(model, eval_arr, None, n_batches=EVAL_BATCHES,
                     batch_size=EVAL_BS, window_seed=WINDOW_SEED, device=DEVICE)
    return {"arm": spec["arm"], "bpc_native": round(bpc, 6),
            "n_batches": EVAL_BATCHES, "batch_size": EVAL_BS,
            "window_seed": WINDOW_SEED, "mask": "causal"}


@torch.no_grad()
def instrument_selfcheck(model: DepthARModel, eval_arr, T: int = 256,
                         n_batches: int = 4, bs: int = 8) -> dict:
    from pathlm.data import batch
    gen = torch.Generator().manual_seed(2)
    ident = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEVICE))
    max_diff = 0.0
    for _ in range(n_batches):
        x, _ = batch(eval_arr, bs, T, gen)
        x = x.to(DEVICE)
        rel = model(x, exit_depths=None)[1]["depth_logits"][-1].float()
        inst = masked_logits(model, x, ident).float()
        max_diff = max(max_diff, float((rel - inst).abs().max()))
    return {"max_abs_logit_diff_released_vs_identity_mask": round(max_diff, 8),
            "n_batches": n_batches, "batch_size": bs}


def last_log_row_and_tele(arm: str) -> dict:
    """线性臂的活性量轨迹（phi_min/den_min/gnorm），停摆时的最后状态。"""
    p = os.path.join(HERE, "runs", arm, "train_log.jsonl")
    if not os.path.exists(p):
        return {}
    rows = [json.loads(l) for l in open(p)]
    if not rows:
        return {}
    return {"n_rows": len(rows), "last_step": rows[-1]["step"],
            "last": rows[-1], "first": rows[0]}


# --------------------------------------------------------------- 阶段

def stage_preflight() -> None:
    os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        out = {"note": "四臂各 20 步冒烟（检查是否前向可跑、活性量落盘）",
               "free_mb_before": free_gpu_mb(), "arms": {}}
        for spec in ARMS:
            arm = spec["arm"]
            r = train_arm(spec, steps=20, poll_s=3, caps_min=10.0,
                          stall_grace_s=120.0)
            out["arms"][arm] = {"smoke_wall_min": r["wall_min"],
                                "stalled": r["stalled"],
                                "patch": r.get("patch"),
                                "tele": last_log_row_and_tele(arm)}
        out["cost_measured"] = cost_probe()
        out["free_mb_after"] = free_gpu_mb()
        json.dump(out, open(os.path.join(HERE, "smoke", "smoke.json"), "w"),
                  indent=2, ensure_ascii=False)
        print(json.dumps({a: v["stalled"] for a, v in out["arms"].items()}))
    finally:
        gpu_release()


def stage_train(arm: str, steps: int = S, stall_dump: int = 0) -> None:
    spec = ARM_BY_NAME[arm]
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        free_before = free_gpu_mb()
        r = train_arm(spec, steps=steps, stall_dump=stall_dump)
        r["free_before_mb"] = free_before
        r["free_after_mb"] = free_gpu_mb()
        json.dump(r, open(os.path.join(HERE, f"{arm}-train_summary.json"), "w"),
                  indent=2, ensure_ascii=False)
    finally:
        gpu_release()


def stage_train_all(gap_s: int = 60) -> None:
    for i, spec in enumerate(ARMS):
        stage_train(spec["arm"])
        if i < len(ARMS) - 1:
            print(f"[run] 释放锁，留 {gap_s}s 窗口（§20.1）", flush=True)
            time.sleep(gap_s)


def stage_diag() -> None:
    """停摆诊断矩阵：短程（1200 步）跑不同实现/数值档，记录 pass / stall。

    档位：{linear no-qk, linear qk=global} × chunk ∈ {32, 64, 256}。每组一条，
    记录是否跑到 1200 步、触发步、以及 faulthandler 首栈（stall-dump=90）。
    """
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        out = {"note": "短程 1200 步诊断矩阵；stall 判定 = 步号停滞 >210s",
               "runs": []}
        for chunk in (32, 64, 256):
            for qk in ("none", "global"):
                name = f"diag-lin-c{chunk}-qk{qk}"
                spec = {"arm": name, "linear": True, "qk": qk}
                r = train_arm(spec, steps=1200, poll_s=15, caps_min=12.0,
                              stall_grace_s=210.0, stall_dump=90)
                tail = r.get("stdout_tail", "")
                out["runs"].append({
                    "arm": name, "chunk": chunk, "qk": qk,
                    "stalled": r["stalled"], "stall_step": r.get("stall_step"),
                    "wall_min": r["wall_min"], "tele": last_log_row_and_tele(name),
                    "stack_hint": ("loss.backward" if "backward" in tail
                                   else tail[-300:] if r["stalled"] else None)})
        json.dump(out, open(os.path.join(HERE, "diag_matrix.json"), "w"),
                  indent=2, ensure_ascii=False)
        print(json.dumps([{k: v[k] for k in ("arm", "stalled", "stall_step")}
                          for v in out["runs"]], indent=2))
    finally:
        gpu_release()


def stage_eval() -> None:
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, DATA))
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        out = {"batteries": {}, "selfchecks": {}, "field_by_field": {},
               "stall_summary": {}}
        for spec in ARMS:
            arm = spec["arm"]
            if not os.path.exists(os.path.join(HERE, "runs", arm, "model.pt")):
                out["stall_summary"][arm] = "no checkpoint (stalled/unrun)"
                continue
            model = load_arm(spec, vocab)
            out["batteries"][arm] = battery(spec, eval_arr, model)
            out["selfchecks"][arm] = instrument_selfcheck(model, eval_arr)
            out["field_by_field"][arm] = {
                "d_model": D, "n_layers": L, "n_heads": H, "mlp_mult": MLP,
                "dense": True, "topology": "sequential", "batch_size": BS,
                "lr": LR, "seed": SEED, "steps": S, "seq_len": 256,
                "attn_linear": spec["linear"], "linear_chunk": 64,
                "qk_norm": spec["qk"], "data": "enwik8 train 90M / eval 10M",
                "vocab_size": vocab}
            del model
            torch.cuda.empty_cache()
            print(f"[eval] {arm} bpc={out['batteries'][arm]['bpc_native']}", flush=True)
        out["cost"] = cost_probe()
        json.dump(out, open(os.path.join(HERE, "eval.json"), "w"),
                  indent=2, ensure_ascii=False)
    finally:
        gpu_release()


def _summary(arm: str) -> dict:
    p = os.path.join(HERE, f"{arm}-train_summary.json")
    return json.load(open(p)) if os.path.exists(p) else {"arm": arm, "ran": False}


def stage_report() -> None:
    ev = json.load(open(os.path.join(HERE, "eval.json")))
    b = ev["batteries"]
    band = 0.014

    def delta(fa: str, la: str):
        if fa in b and la in b:
            return round(b[fa]["bpc_native"] - b[la]["bpc_native"], 6)
        return None

    d_raw = delta("full-256-lin", "linear-256-lin")
    d_qk = delta("full-256-lin-qk", "linear-256-lin-qk")
    d_full_qk = delta("full-256-lin", "full-256-lin-qk")   # QK-Norm 对 full 的影响

    cfg = {
        "contract": "axes/1-time/1.2.2-block-sparse/experiments/2637-linear-attn-quality/README.md",
        "scope": "单轴单格；唯一变量 = 注意力算子（full vs 线性核）；T=256/bs8/S=6000/seed0/enwik8",
        "model": {"d_model": D, "n_layers": L, "n_heads": H, "mlp_mult": MLP,
                  "dense": True},
        "step_budget": S, "tokens_by_arm": S * BS * 256,
        "arms": [{**spec, "steps": S, "seed": SEED,
                  "fields": ev["field_by_field"].get(spec["arm"])} for spec in ARMS],
        "code_fingerprints": {
            "pathlm/depth_ar.py": sha256_file(os.path.join(ROOT, "pathlm/depth_ar.py")),
            "train_depth_ar.py": sha256_file(os.path.join(ROOT, "train_depth_ar.py")),
            "experiments/2625-longT-stability/stability_driver.py": sha256_file(
                os.path.join(EXP_ROOT, "2625-longT-stability", "stability_driver.py")),
            "note": "本实验未改这些文件（只读）",
        },
    }
    results = {
        "contract": cfg["contract"], "scope": cfg["scope"], "config": cfg,
        "arms": {a["arm"]: {**_summary(a["arm"]),
                            "battery": b.get(a["arm"]),
                            "selfcheck": ev["selfchecks"].get(a["arm"]),
                            "telemetry": last_log_row_and_tele(a["arm"])} for a in ARMS},
        "criteria": {
            "criterion_1_quality": {
                "delta_convention": "Δ = bpc_full − bpc_linear（正 = 线性更好）",
                "delta_native_no_qk": d_raw,
                "delta_native_with_qk": d_qk,
                "delta_full_qk_effect": d_full_qk,
                "variance_band": band,
                "band_class_no_qk": (_class(d_raw, band) if d_raw is not None else None),
                "band_class_with_qk": (_class(d_qk, band) if d_qk is not None else None),
                "note": "诚实对手 = 同配置的 full 臂；原实装 vs 稳定化分母并列报",
            },
            "criterion_2_cost": ev["cost"],
            "criterion_3_stall": {
                "no_qk_stalled": _summary("linear-256-lin").get("stalled"),
                "with_qk_stalled": _summary("linear-256-lin-qk").get("stalled"),
                "diag_matrix": (json.load(open(os.path.join(HERE, "diag_matrix.json")))
                                if os.path.exists(os.path.join(HERE, "diag_matrix.json"))
                                else None),
            },
        },
        "cost": ev["cost"],
    }
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump(cfg, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({"d_no_qk": d_raw, "d_with_qk": d_qk, "d_full_qk": d_full_qk},
                     indent=2, ensure_ascii=False))


def _class(x: float, band: float) -> str:
    if abs(x) < band:
        return "不亏（噪声带内）"
    return "线性更好（越界）" if x > 0 else "线性更差（越界）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("preflight", "train", "train-all", "diag", "eval",
                             "report", "all"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--steps", type=int, default=S)
    ap.add_argument("--stall-dump", type=int, default=0)
    args = ap.parse_args()
    if args.stage == "preflight":
        stage_preflight()
    elif args.stage == "train":
        if not args.arm:
            raise SystemExit("--stage train 需要 --arm")
        stage_train(args.arm, steps=args.steps, stall_dump=args.stall_dump)
    elif args.stage == "train-all":
        stage_train_all()
    elif args.stage == "diag":
        stage_diag()
    elif args.stage == "eval":
        stage_eval()
    elif args.stage == "report":
        stage_report()
    elif args.stage == "all":
        stage_train_all()
        stage_eval()
        stage_report()


if __name__ == "__main__":
    main()
