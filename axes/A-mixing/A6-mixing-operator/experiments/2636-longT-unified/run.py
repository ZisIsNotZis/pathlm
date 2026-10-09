"""2636 · 长 T 四点统一口径（A6）—— 实验入口。

契约：同目录 `README.md`（判据线预登记，不得修改）。

干预（唯一变量）：训练期可见窗口 ∈ {full（`window=0`）, w = T/2}，T ∈ {256, 512, 1024, 2048}。
本实验**新跑** T=256/512 两点的 full/window 四臂（seed0）+ 噪声自测四臂（seed1），
**共同步数 S=6000**；T=1024/2048 的同 step 点从 `2626`/`2629` **只读复用**（§6 复用条件）。

阶段（GPU 阶段按臂取/放锁，臂间留 ≥60s 窗口，SCHEMA §20/§20.1）：
    python3 run.py --stage preflight           # 四臂 20 步冒烟 + 实测步时/显存 + 成本探针（§20.3）
    python3 run.py --stage train --arm <arm>   # 训练一个臂
    python3 run.py --stage train-all           # 依次训练八臂（每臂间放锁 ≥60s）
    python3 run.py --stage eval                # 电池（native/release bpc + 合成腿 + 成本第二轮）
    python3 run.py --stage report              # 汇总 → results.json / config.json
    python3 run.py --stage all                 # train-all + eval + report

禁外推：四点只描述；成本/占比一律**实测**，公式值只作对照。
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,        # noqa: E402
                             window_anchor_keep_mask,
                             window_mask_liveness)
from pathlm.data import load_enwik8_full                         # noqa: E402
from tools.evalkit.retention import masked_bpc, masked_logits    # noqa: E402
from tools.evalkit.longrange_tasks import (var_recall_batch,     # noqa: E402
                                           accuracy, local_prior_accuracy,
                                           check_local_prior_resistance)

DEVICE = "cuda"
DATA = "data/enwik8_full.npz"

# ---- 契约骨架（§3 单因素）：d=256/L=8/h=8/mlp=4，stock dense，bs8/seed0/lr1e-3 ----
D, L, H, MLP, BS, LR = 256, 8, 8, 4, 8, 1e-3
QK_NORM_MIN_T = 512
#: **主口径 = 同 step S=6000**（lr 调度以 steps 为分母 ⇒ 同 steps = 同退火刻度）。
#: token 预算随 T 线性：12.288M / 24.576M / 49.152M / 98.304M（如实分报）。
S = 6000
TOKENS_AT_S = {T: S * BS * T for T in (256, 512, 1024, 2048)}
#: 副口径 = 同 token 49.152M（引用既有四点，steps = 24000/12000/6000/3000）。
TOKEN_BUDGET_REF = 49_152_000

DRIVER = os.path.join(HERE, "clean_driver.py")

#: 本实验新跑的八臂（四配置 × 两 seed）。`window = T/2` ⇒ 窗口占 T 的 50%（§17.16 同相对刻度）。
ARMS = [
    {"arm": "full-256-s6k",     "seq_len": 256, "window": 0,   "seed": 0, "role": "T=256 · 全因果 @6000"},
    {"arm": "win128-256-s6k",   "seq_len": 256, "window": 128, "seed": 0, "role": "T=256 · 窗口 128（50%）@6000"},
    {"arm": "full-512-s6k",     "seq_len": 512, "window": 0,   "seed": 0, "role": "T=512 · 全因果 @6000"},
    {"arm": "win256-512-s6k",   "seq_len": 512, "window": 256, "seed": 0, "role": "T=512 · 窗口 256（50%）@6000"},
    {"arm": "full-256-s6k-s1",  "seq_len": 256, "window": 0,   "seed": 1, "role": "T=256 · 全因果 @6000 · seed1（噪声自测）"},
    {"arm": "win128-256-s6k-s1", "seq_len": 256, "window": 128, "seed": 1, "role": "T=256 · 窗口 128 · seed1（噪声自测）"},
    {"arm": "full-512-s6k-s1",  "seq_len": 512, "window": 0,   "seed": 1, "role": "T=512 · 全因果 @6000 · seed1（噪声自测）"},
    {"arm": "win256-512-s6k-s1", "seq_len": 512, "window": 256, "seed": 1, "role": "T=512 · 窗口 256 · seed1（噪声自测）"},
]
ARM_BY_NAME = {a["arm"]: a for a in ARMS}
#: seed0 四臂（主口径四点的新增部分）。
ARMS_SEED0 = [a for a in ARMS if a["seed"] == 0]
POINTS = ((128, 256), (256, 512), (512, 1024), (1024, 2048))

#: 公式口径（K10）：per-token-per-layer MACs。attn=2·T·d，other=3d²+d²+2·mlp·d²。
OTHER_MACS = 3 * D * D + D * D + 2 * MLP * D * D          # 786432
SYNTH_DISTS = {256: (64, 128, 192), 512: (128, 256, 384)}
SYNTH_BATCH, SYNTH_CHUNK = 64, 16
EVAL_BATCHES, EVAL_BS, WINDOW_SEED = 20, 16, 2

LOCK = "/tmp/pathlm_gpu.lock"
LOCK_MAX_WAIT = 7200

EXP_ROOT = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator", "experiments")
#: 同 step（6000）只读复用的既有臂（§6 复用条件）。
REF_STEP = {
    1024: ("2626-longT-clean", "full-1024-clean", "win512-1024-clean"),
    2048: ("2629-longT-2048-matched", "full-2048-clean", "win1024-2048-clean"),
}
#: 同 token（49.152M）副口径引用的既有臂。
REF_TOKEN = {
    256: ("2626-longT-clean", "full-256-clean", "win128-256-clean"),
    512: ("2626-longT-clean", "full-512-clean", "win256-512-clean"),
    1024: ("2626-longT-clean", "full-1024-clean", "win512-1024-clean"),
    2048: ("2628-longT-2048", "full-2048-clean", "win1024-2048-clean"),
}


# ------------------------------------------------------------------ 工具

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _engine_fingerprints() -> dict:
    """本实验依赖的引擎/仪器指纹（**运行期**快照；未改的文件只读引用）。"""
    return {
        "pathlm/depth_ar.py": sha256_file(os.path.join(ROOT, "pathlm/depth_ar.py")),
        "pathlm/model.py": sha256_file(os.path.join(ROOT, "pathlm/model.py")),
        "train_depth_ar.py": sha256_file(os.path.join(ROOT, "train_depth_ar.py")),
        "tools/evalkit/retention.py": sha256_file(
            os.path.join(ROOT, "tools/evalkit/retention.py")),
        "experiments/2625-longT-stability/stability_driver.py": sha256_file(
            os.path.join(EXP_ROOT, "2625-longT-stability", "stability_driver.py")),
    }


def _load_run_fingerprints() -> dict:
    """运行期指纹：优先 preflight 快照 `run_fingerprints.json`，否则回退到人工补记
    `run_fingerprints_backfill.json`（两者形状兼容：同名键 + `_what/_kind` 标记）。"""
    for name in ("run_fingerprints.json", "run_fingerprints_backfill.json"):
        p = os.path.join(HERE, name)
        if os.path.exists(p):
            d = json.load(open(p))
            d["_file"] = name
            return d
    return {"available": False,
            "note": "run_fingerprints.json / run_fingerprints_backfill.json 均缺失：运行期指纹未记录"}


def cfg_for(seq_len: int, window: int) -> DepthARConfig:
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=seq_len,
                         mlp_mult=MLP, dense=True, train_window=window)


def attn_share_formula(T: int) -> float:
    """K10 公式：attn_share(T) = 512T / (512T + 786432)。"""
    return (2 * T * D) / (2 * T * D + OTHER_MACS)


def _load_w1():
    spec = importlib.util.spec_from_file_location(
        "w1_stability_driver_run",
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

def train_arm(spec: dict, steps: int, poll_s: int = 20, caps_min: float = 90.0) -> dict:
    """后台跑 `clean_driver.py`，≤110s 轮询日志 + 触摸心跳 + nvidia-smi 子进程峰值。

    调用方负责取/放锁（见 stage_train / stage_train_all）。
    """
    arm = spec["arm"]
    run_dir = os.path.join(HERE, "runs", arm)
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, DRIVER, arm,
           "--seq-len", str(spec["seq_len"]), "--window", str(spec["window"]),
           "--steps", str(steps), "--batch-size", str(BS), "--lr", str(LR),
           "--seed", str(spec["seed"]), "--candidate", "qk-normfrom512",
           "--out-root", os.path.join(HERE, "runs")]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] {arm}: started pid={proc.pid} steps={steps} "
          f"seq_len={spec['seq_len']} window={spec['window']} seed={spec['seed']}",
          flush=True)
    peak, t0 = 0.0, time.time()
    log_path = os.path.join(run_dir, "train_log.jsonl")
    while proc.poll() is None:
        gpu_touch()
        if time.time() - t0 > caps_min * 60:
            proc.terminate()
            raise TimeoutError(f"{arm} 训练超过 {caps_min} min cap")
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
        try:
            with open(log_path) as f:
                lines = f.readlines()
            if lines:
                st = int(json.loads(lines[-1])["step"])
                if st % 2000 == 0 and st > 0:
                    print(f"[run] {arm} {lines[-1].strip()[:150]}", flush=True)
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(poll_s)
    log.close()
    if proc.returncode != 0:
        tail = ""
        try:
            tail = open(os.path.join(run_dir, "stdout.log")).read()[-800:]
        except OSError:
            pass
        raise RuntimeError(f"{arm} 训练失败 rc={proc.returncode}"
                           f"\n--- stdout tail ---\n{tail}")
    wall = (time.time() - t0) / 60
    res = json.load(open(os.path.join(run_dir, "results.json")))
    out = {"arm": arm, "run_dir": os.path.relpath(run_dir, HERE),
           "wall_min": round(wall, 2),
           "seconds_per_step": round(wall * 60 / steps, 5),
           "peak_child_mem_gb_nvidia_smi": round(peak, 2),
           "peak_torch_gb": round(res["peak_gpu_mem_mb"] / 1024.0, 3),
           "params": res["params"], "tokens": res["tokens"],
           "patch": res["patch"],
           "gnorm_monitor": res["gnorm_monitor"],
           "sha256_model_pt": sha256_file(os.path.join(run_dir, "model.pt")),
           "size_bytes": os.path.getsize(os.path.join(run_dir, "model.pt")),
           "steps": steps, "seed": spec["seed"],
           # 运行期引擎指纹（修 P1-5）：训练**开始前**记一次，与 report 阶段的 HEAD 区分
           "code_fingerprints_run": _engine_fingerprints()}
    json.dump(out, open(os.path.join(HERE, f"{arm}-train_summary.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[run] {arm}: done wall={wall:.2f} min peak(nvidia-smi)={peak:.2f}GB "
          f"peak(torch)={out['peak_torch_gb']:.3f}GB "
          f"gnorm_max={out['gnorm_monitor']['gnorm_max']} "
          f"skips={out['gnorm_monitor']['grad_skips']}", flush=True)
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
    return {"median_ms": round(statistics.median(ts), 3),
            "min_ms": round(min(ts), 3), "n_iters": iters}


def _apply_qk_norm(model: DepthARModel, seq_len: int) -> dict:
    """按长度条件档给成本臂套 QK-Norm（T > min_t 才真的挂 hook）。"""
    if seq_len > QK_NORM_MIN_T:
        return _load_w1().apply_attn_scale(model, "qk-norm", 1.0,
                                           min_t=QK_NORM_MIN_T)
    return {"n_qkv_hooks": 0, "active": False}


def model_step_cost(seq_len: int, window: int, iters: int = 6) -> dict:
    """整模型一步 fwd+bwd 中位墙钟 + torch 峰值显存（套 QK-Norm，同实验配方）。

    `window > 0` ⇒ 走 SDPA 的**稠密 bool 掩码**路径（不兑现 O(T·w) 的算力节省）。
    """
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    model = DepthARModel(cfg_for(seq_len, window), 206).cuda()
    qk = _apply_qk_norm(model, seq_len)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    x = torch.randint(0, 206, (BS, seq_len)).cuda()
    km = (window_anchor_keep_mask(seq_len, window, 0, DEVICE)
          if window > 0 else None)
    am = (km.view(1, 1, seq_len, seq_len).expand(BS, H, seq_len, seq_len)
          if km is not None else None)

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if am is None:
                loss, _ = model(x, exit_depths=None)
            else:
                h = model._input_states(x)
                for blk in model.blocks:
                    h, _ = blk(h, attn_mask=am)
                loss = F.cross_entropy(model.readout(h)[:, :-1].reshape(-1, 206),
                                       x[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    timing = _timed(step, warmup=2, iters=iters)
    timing.update({"seq_len": seq_len, "window": window, "batch_size": BS,
                   "qk_norm_hooks": qk["n_qkv_hooks"],
                   "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)})
    del model, opt
    torch.cuda.empty_cache()
    return timing


def attn_island_cost(seq_len: int, window: int, iters: int = 8) -> dict:
    """注意力子层孤岛（qkv → 注意力 → out 投影）fwd+bwd 实测。"""
    dh = D // H
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn(BS, seq_len, D, device=DEVICE, requires_grad=True)
    qkv = torch.nn.Linear(D, 3 * D).to(DEVICE)
    proj = torch.nn.Linear(D, D).to(DEVICE)
    shape = lambda t: t.view(BS, seq_len, H, dh).transpose(1, 2)      # noqa: E731
    km = (window_anchor_keep_mask(seq_len, window, 0, DEVICE)
          if window > 0 else None)
    am = (km.view(1, 1, seq_len, seq_len).expand(BS, H, seq_len, seq_len)
          if km is not None else None)

    def step():
        q, k, v = qkv(x).chunk(3, dim=-1)
        a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                           attn_mask=am, is_causal=am is None)
        y = proj(a.transpose(1, 2).reshape(BS, seq_len, D))
        loss = y.float().pow(2).mean()
        qkv.zero_grad(); proj.zero_grad()
        loss.backward()

    out = _timed(step, warmup=2, iters=iters)
    out.update({"seq_len": seq_len, "window": window, "batch_size": BS,
                "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)})
    del x, qkv, proj
    torch.cuda.empty_cache()
    return out


def cost_theory(seq_lens=(256, 512, 1024, 2048)) -> dict:
    """纯算术的等效 FLOPs / 占比口径（不碰 GPU）——report 阶段可独立复算。"""
    theory = {"other_macs_per_token_per_layer": OTHER_MACS,
              "attn_share_formula": {str(T): round(attn_share_formula(T), 4)
                                     for T in seq_lens},
              "window_theory": {}}
    for T in seq_lens:
        w = T // 2
        live = window_mask_liveness(T, w, 0)
        attn_win, attn_full = 2 * w * D, 2 * T * D
        total_full, total_win = attn_full + OTHER_MACS, attn_win + OTHER_MACS
        theory["window_theory"][f"{w}@{T}"] = {
            "visible_frac_of_causal": live["visible_frac"],
            "attn_flops_reduction_relative_to_full": round(1 - attn_win / attn_full, 4),
            "layer_macs_reduction_all_in": round(1 - total_win / total_full, 4),
        }
    return theory


def cost_probe(iters: int = 6, seq_lens=(256, 512, 1024, 2048)) -> dict:
    measured = {"model_step": [], "attn_island": [], "ratio_by_T": [],
                "free_mb_before": free_gpu_mb()}
    for T in seq_lens:
        for w in (0, T // 2):
            measured["model_step"].append(model_step_cost(T, w, iters=iters))
            measured["attn_island"].append(attn_island_cost(T, w, iters=max(iters, 6)))
    by = {(r["seq_len"], r["window"]): r for r in measured["model_step"]}
    for T in seq_lens:
        w = T // 2
        f, win = by[(T, 0)], by[(T, w)]
        isl_f = next(r for r in measured["attn_island"]
                     if r["seq_len"] == T and r["window"] == 0)
        isl_w = next(r for r in measured["attn_island"]
                     if r["seq_len"] == T and r["window"] == w)
        measured["ratio_by_T"].append({
            "T": T, "window": w,
            "step_ms_full": f["median_ms"], "step_ms_window": win["median_ms"],
            "r_time_window_over_full": round(win["median_ms"] / f["median_ms"], 4),
            "qk_norm_hooks_full": f["qk_norm_hooks"],
            "peak_full_gib": f["peak_mem_gib"], "peak_window_gib": win["peak_mem_gib"],
            "r_mem_window_over_full": round(win["peak_mem_gib"] / f["peak_mem_gib"], 4),
            "attn_island_ms_full": isl_f["median_ms"],
            "attn_island_ms_window": isl_w["median_ms"],
            "attn_island_over_model_step": round(isl_f["median_ms"] / f["median_ms"], 4),
            "attn_island_over_model_step_window": round(
                isl_w["median_ms"] / win["median_ms"], 4),
            "attn_share_formula": round(attn_share_formula(T), 4),
        })
    measured["free_mb_after"] = free_gpu_mb()
    return {"theory": cost_theory(seq_lens), "measured": measured,
            "note": ("双口径：理论（等效 FLOPs）+ 实测（墙钟/显存）。公式值只作对照，"
                     "不得当判据（SCHEMA §17.11）。模型步成本已套 QK-Norm（同实验配方）。")}


# --------------------------------------------------------------- 电池

def _load_w1_hook(model: DepthARModel, seq_len: int) -> None:
    _apply_qk_norm(model, seq_len)


def load_arm(spec: dict, vocab: int) -> DepthARModel:
    torch.manual_seed(spec["seed"])
    m = DepthARModel(cfg_for(spec["seq_len"], spec["window"]), vocab).cuda()
    _load_w1_hook(m, spec["seq_len"])
    m.load_state_dict(torch.load(os.path.join(HERE, "runs", spec["arm"], "model.pt"),
                                 map_location="cuda"))
    m.eval()
    return m


def native_keep(spec: dict) -> torch.Tensor | None:
    if spec["window"] <= 0:
        return None
    return window_anchor_keep_mask(spec["seq_len"], spec["window"], 0, DEVICE)


def battery(spec: dict, eval_arr, model: DepthARModel) -> dict:
    T = spec["seq_len"]
    km = native_keep(spec)
    native = masked_bpc(model, eval_arr, km, n_batches=EVAL_BATCHES,
                        batch_size=EVAL_BS, window_seed=WINDOW_SEED, device=DEVICE)
    release = masked_bpc(model, eval_arr, None, n_batches=EVAL_BATCHES,
                         batch_size=EVAL_BS, window_seed=WINDOW_SEED, device=DEVICE)
    out = {"arm": spec["arm"], "seq_len": T, "window": spec["window"],
           "seed": spec["seed"],
           "bpc_native": round(native, 6), "bpc_release": round(release, 6),
           "n_batches": EVAL_BATCHES, "batch_size": EVAL_BS,
           "window_seed": WINDOW_SEED,
           "native_mask": "release(causal)" if km is None else "window"}
    out["native_mask_liveness"] = (window_mask_liveness(T, spec["window"], 0)
                                   if km is not None else None)
    return out


@torch.no_grad()
def instrument_selfcheck(model: DepthARModel, eval_arr, T: int,
                         n_batches: int = 4, bs: int = 8) -> dict:
    """仪器自检：显式精确因果掩码 vs 释放版路径，逐位一致（应 0）。"""
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


@torch.no_grad()
def synth_leg(spec: dict, model: DepthARModel) -> dict:
    T = spec["seq_len"]
    if T not in SYNTH_DISTS:
        return {"arm": spec["arm"], "seq_len": T, "skipped": True}
    km = native_keep(spec)
    out = {"arm": spec["arm"], "seq_len": T, "window": spec["window"],
           "native_mask": "window" if km is not None else "release(causal)",
           "by_distance": {}}
    for d in SYNTH_DISTS[T]:
        task = var_recall_batch(SYNTH_BATCH, T, d, sym_vocab=64, filler_token=64,
                                cue_token=205,
                                generator=torch.Generator().manual_seed(1000 + d))
        acc = accuracy(model, task, keep_mask=km, device=DEVICE, max_chunk=SYNTH_CHUNK)
        out["by_distance"][str(d)] = {
            "accuracy": acc["accuracy"], "n": acc["n"],
            "chance_answer_set": round(task.chance(), 6),
            "local_prior_accuracy_k8": local_prior_accuracy(task, k=8)["accuracy"],
            "local_prior_resistance": check_local_prior_resistance(task, k=8)["passed"],
        }
    return out


# --------------------------------------------------------------- 跨 run 引用

def _read_ref_battery(exp: str, arm: str) -> dict | None:
    p = os.path.join(EXP_ROOT, exp, "eval.json")
    if not os.path.exists(p):
        return None
    b = json.load(open(p))["batteries"].get(arm)
    if b is None:
        return None
    return {"exp": exp, "arm": arm, "bpc_native": b["bpc_native"],
            "bpc_release": b["bpc_release"], "n_batches": b["n_batches"],
            "batch_size": b["batch_size"], "window_seed": b["window_seed"]}


def ref_points(token_or_step: dict) -> list[dict]:
    rows = []
    for T, (exp, fa, wa) in token_or_step.items():
        fr, wr = _read_ref_battery(exp, fa), _read_ref_battery(exp, wa)
        if fr is None or wr is None:
            continue
        rows.append({"T": T, "exp": exp, "full_arm": fa, "win_arm": wa,
                     "bpc_full": fr["bpc_native"], "bpc_window": wr["bpc_native"],
                     "delta_native": round(fr["bpc_native"] - wr["bpc_native"], 6)})
    return sorted(rows, key=lambda r: r["T"])


# --------------------------------------------------------------- 阶段

def stage_preflight(cost_iters: int) -> None:
    """四臂（seed0）各 20 步冒烟 + 成本基准（四 T × full/window）。"""
    os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
    # 运行期引擎指纹（修 P1-5）：在**做任何训练前**落一次，与 report 阶段的 HEAD 区分。
    # 形状与人工补记（`run_fingerprints_backfill.json`）**同名键、可共存**：本写入器只写
    # `run_fingerprints.json`（preflight 快照），**不覆盖**人工补记文件。report 优先读本文件，
    # 缺失时回退到人工补记。
    with open(os.path.join(HERE, "run_fingerprints.json"), "w") as f:
        json.dump({"_what": "运行期引擎/仪器指纹（preflight 时快照）",
                   "_kind": "preflight_snapshot", "_when": "preflight",
                   **_engine_fingerprints()}, f,
                  indent=2, ensure_ascii=False)
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        out = {"note": "20 步冒烟：seed0 四臂各跑 20 步，检查活性量落盘与实测步时/显存"
                       "（不含任何 20 步外推时间字段，§17.5）",
               "free_mb_before": free_gpu_mb(), "arms": {}}
        for spec in ARMS_SEED0:
            arm = spec["arm"]
            r = train_arm(spec, steps=20, poll_s=3, caps_min=10.0)
            rows = [json.loads(l) for l in
                    open(os.path.join(HERE, "runs", arm, "train_log.jsonl"))]
            out["arms"][arm] = {
                "config": {"seq_len": spec["seq_len"], "window": spec["window"],
                           "steps_full": S},
                "smoke_wall_min": r["wall_min"],
                "smoke_sec_per_step": r["seconds_per_step"],
                "peak_torch_gb": r["peak_torch_gb"],
                "params": r["params"], "patch": r["patch"],
                "last_log_row": rows[-1] if rows else None,
                "window_mask_liveness": (window_mask_liveness(spec["seq_len"],
                                                              spec["window"], 0)
                                         if spec["window"] > 0 else None),
                "config_validates": True,
            }
            del r
        out["cost_measured"] = cost_probe(iters=max(cost_iters, 4))
        out["free_mb_after"] = free_gpu_mb()
        json.dump(out, open(os.path.join(HERE, "smoke", "smoke.json"), "w"),
                  indent=2, ensure_ascii=False)
        print(json.dumps([{"arm": a, "sps": v["smoke_sec_per_step"]}
                          for a, v in out["arms"].items()], indent=2))
    finally:
        gpu_release()


def stage_train(arm: str) -> None:
    spec = ARM_BY_NAME[arm]
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        free_before = free_gpu_mb()
        r = train_arm(spec, steps=S)
        r["free_before_mb"] = free_before
        r["free_after_mb"] = free_gpu_mb()
        json.dump(r, open(os.path.join(HERE, f"{arm}-train_summary.json"), "w"),
                  indent=2, ensure_ascii=False)
    finally:
        gpu_release()


def _has_ckpt(arm: str) -> bool:
    return os.path.exists(os.path.join(HERE, "runs", arm, "model.pt"))


def _available_arms() -> list[dict]:
    return [a for a in ARMS if _has_ckpt(a["arm"])]


def stage_train_all(gap_s: int = 60) -> None:
    """多臂驱动：**按臂释放锁**（§20.1），臂间留 ≥gap_s 秒窗口。

    已有 checkpoint 的臂跳过（支持断点续跑；seed1 臂可被预算砍掉——契约 §5）。
    """
    todo = [a for a in ARMS if not _has_ckpt(a["arm"])]
    for i, spec in enumerate(todo):
        stage_train(spec["arm"])
        if i < len(todo) - 1:
            print(f"[run] 释放锁，留 {gap_s}s 窗口（§20.1）", flush=True)
            time.sleep(gap_s)


def stage_eval(cost_iters: int = 6) -> None:
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, DATA))
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        out = {"batteries": {}, "selfchecks": {}, "synth": {}, "field_by_field": {}}
        for spec in _available_arms():
            arm = spec["arm"]
            model = load_arm(spec, vocab)
            out["batteries"][arm] = battery(spec, eval_arr, model)
            out["selfchecks"][arm] = instrument_selfcheck(model, eval_arr,
                                                          spec["seq_len"])
            out["synth"][arm] = synth_leg(spec, model)
            del model
            torch.cuda.empty_cache()
            print(f"[eval] {arm} done bpc_native="
                  f"{out['batteries'][arm]['bpc_native']}", flush=True)
        for spec in _available_arms():
            out["field_by_field"][spec["arm"]] = {
                "d_model": D, "n_layers": L, "n_heads": H, "mlp_mult": MLP,
                "dense": True, "topology": "sequential", "batch_size": BS,
                "lr": LR, "seed": spec["seed"], "steps": S,
                "exit_mode": "uniform", "seq_len": spec["seq_len"],
                "train_window": spec["window"], "qk_norm": "qk-normfrom512",
                "qk_norm_min_t": QK_NORM_MIN_T,
                "data": "enwik8 train 90M / eval 10M", "vocab_size": vocab,
            }
        out["cost"] = cost_probe(iters=cost_iters)
        json.dump(out, open(os.path.join(HERE, "eval.json"), "w"),
                  indent=2, ensure_ascii=False)
    finally:
        gpu_release()


def _band_class(x: float, band: float = 0.014) -> str:
    """带宽分类（有向，SCHEMA §17.3：Δ = full − window，正 = 窗口更好）。"""
    if abs(x) < band:
        return "不亏（噪声带内）"
    return "窗口更好（越界）" if x > 0 else "窗口更差（越界）"


def _late_slope(rows: list[dict], n: int = 5) -> float:
    xs = [r["step"] for r in rows[-n:]]
    ys = [r["loss"] for r in rows[-n:]]
    k = len(xs)
    mx, my = sum(xs) / k, sum(ys) / k
    den = sum((x - mx) ** 2 for x in xs)
    return round(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den * 1000, 5) \
        if den else float("nan")


def stage_report() -> None:
    smoke = json.load(open(os.path.join(HERE, "smoke", "smoke.json")))
    ev = json.load(open(os.path.join(HERE, "eval.json")))
    avail = _available_arms()
    trains = {a["arm"]: json.load(open(os.path.join(HERE,
                                                    f"{a['arm']}-train_summary.json")))
              for a in avail}
    b = ev["batteries"]
    band = 0.014
    dropped = [a["arm"] for a in ARMS if a["arm"] not in trains]

    # ---- 判据 C1：同 step 四点（T=256/512 本实验；T=1024/2048 只读复用）----
    step_rows = []
    for w, T in POINTS:
        if T <= 512:
            fa = f"full-{T}-s6k"
            wa = f"win{w}-{T}-s6k"
            bf, bw = b[fa]["bpc_native"], b[wa]["bpc_native"]
            src = "this_exp"
        else:
            exp, fa, wa = REF_STEP[T]
            fr = _read_ref_battery(exp, fa)
            wr = _read_ref_battery(exp, wa)
            bf, bw = fr["bpc_native"], wr["bpc_native"]
            src = f"ref:{exp}"
        dn = round(bf - bw, 6)
        step_rows.append({"T": T, "window": w, "source": src,
                          "full_arm": fa, "win_arm": wa,
                          "bpc_full": bf, "bpc_window": bw,
                          "delta_native": dn, "band_class": _band_class(dn, band),
                          "tokens": TOKENS_AT_S[T], "steps": S})
    shape_step = " → ".join(f"{r['band_class']}@{r['T']}" for r in step_rows)

    # ---- 判据 C5：同 token 副口径四点（全部只读复用）----
    token_rows = []
    for r in ref_points(REF_TOKEN):
        r = dict(r)
        r["band_class"] = _band_class(r["delta_native"], band)
        token_rows.append(r)
    shape_token = " → ".join(f"{r['band_class']}@{r['T']}" for r in token_rows)

    # ---- 判据 C4：本格噪声自测（seed0 vs seed1 同臂 bpc 极差）----
    noise = {"per_arm": {}, "note": "同配置、唯一差别 seed；单臂跨 seed 极差 = 本格噪声底"}
    for spec in ARMS_SEED0:
        a0, a1 = spec["arm"], spec["arm"] + "-s1"
        if a1 in b:
            spread = round(abs(b[a0]["bpc_native"] - b[a1]["bpc_native"]), 6)
            noise["per_arm"][f"{a0} vs {a1}"] = {
                "bpc_seed0": b[a0]["bpc_native"], "bpc_seed1": b[a1]["bpc_native"],
                "abs_spread": spread}
    for w, T in POINTS:
        if T > 512:
            continue
        d0 = next(r["delta_native"] for r in step_rows if r["T"] == T)
        fa1, wa1 = f"full-{T}-s6k-s1", f"win{w}-{T}-s6k-s1"
        if fa1 in b and wa1 in b:
            d1 = round(b[fa1]["bpc_native"] - b[wa1]["bpc_native"], 6)
            noise.setdefault("delta_spread", {})[str(T)] = {
                "delta_seed0": d0, "delta_seed1": d1,
                "abs_spread": round(abs(d0 - d1), 6)}
    if noise.get("per_arm"):
        noise["sigma_cell_single_arm_max"] = max(
            v["abs_spread"] for v in noise["per_arm"].values())
    else:
        noise["sigma_cell_single_arm_max"] = None
        noise["fallback_reference"] = (
            "未跑 seed1（预算砍臂）：引用 2625 已测本格噪声底——T=512 同配方两次重跑 "
            "clean bpc 极差 0.0045")
    # 判据量的本格噪声（修 P1-2）：要判的是 Δ（两臂之差），不是单臂 bpc；
    # 单臂极差是极值估计且量的是另一个随机变量。评测批内配对差因四/八臂共用同一评测流
    # （window_seed=2、同 20 批 bs16）在 Δ 中共模抵消（按构造 ≈0），不另计。
    ds = noise.get("delta_spread") or {}
    noise["cell_threshold_for_delta"] = (
        max(v["abs_spread"] for v in ds.values()) if ds else None)
    noise["cell_threshold_basis"] = (
        "判据量 Δ 的跨 seed 实测极差最大值（T=256/512）；单臂 bpc 极差仅作参考"
        "（极值估计 + 另一随机变量）；评测批内配对差共模抵消≈0")
    noise["seed_replicate_scope"] = (
        "seed 复本仅在 T=256/512 测得；T=1024/2048 无复本，其判读用工程带 0.014")

    # ---- 判据 C2：稳定性前置（本实验四臂 + 复用四臂的 gnorm）----
    gate = {"threshold_gnorm_max": 100, "threshold_grad_skips": 0, "per_arm": {}}
    for spec in avail:
        mon = trains[spec["arm"]]["gnorm_monitor"]
        gate["per_arm"][spec["arm"]] = {
            "gnorm_max": mon["gnorm_max"], "gnorm_mean": mon["gnorm_mean"],
            "grad_skips": mon["grad_skips"],
            "pass": bool(mon["gnorm_max"] < 100 and mon["grad_skips"] == 0)}
    gate["ref_arms"] = {
        "full-1024-clean (2626)": {"gnorm_max": 7.25, "grad_skips": 0, "pass": True},
        "full-2048-clean (2629)": {"gnorm_max": 7.22, "grad_skips": 0, "pass": True},
    }
    gate["all_seed0_full_arms_pass"] = all(
        gate["per_arm"][f"full-{T}-s6k"]["pass"] for T in (256, 512)
        if f"full-{T}-s6k" in gate["per_arm"])

    # ---- 判据 C3：四条腿（质量 = C1；FLOPs/墙钟/显存）----
    c3 = {"note": "每条腿单独报；公式值只作对照（§17.11）。r_time>1 = 窗口更慢。",
          "round_1_preflight": smoke["cost_measured"]["measured"],
          "round_2_eval": ev["cost"]["measured"],
          "theory": ev["cost"]["theory"]}
    c3["by_T"] = []
    for w, T in POINTS:
        r2 = next(r for r in ev["cost"]["measured"]["ratio_by_T"] if r["T"] == T)
        r1 = next(r for r in smoke["cost_measured"]["measured"]["ratio_by_T"]
                  if r["T"] == T)
        q = next(r for r in step_rows if r["T"] == T)
        th = ev["cost"]["theory"]["window_theory"][f"{w}@{T}"]
        c3["by_T"].append({
            "T": T, "window": w,
            "quality_delta_native": q["delta_native"],
            "quality_band_class": q["band_class"],
            "flops_attn_reduction": th["attn_flops_reduction_relative_to_full"],
            "flops_layer_all_in_reduction": th["layer_macs_reduction_all_in"],
            "r_time_round1": r1["r_time_window_over_full"],
            "r_time_round2": r2["r_time_window_over_full"],
            "r_mem_round2": r2["r_mem_window_over_full"],
            "peak_full_gib": r2["peak_full_gib"], "peak_window_gib": r2["peak_window_gib"],
            "attn_island_over_model_step_full": r2["attn_island_over_model_step"],
            "attn_share_formula": r2["attn_share_formula"],
        })

    liveness, stability = {}, {}
    for a in avail:
        arm = a["arm"]
        mon = trains[arm]["gnorm_monitor"]
        rows = [json.loads(l) for l in
                open(os.path.join(HERE, "runs", arm, "train_log.jsonl"))]
        liveness[arm] = {"window_mask": mon.get("window_mask"),
                         "window_mask_probe": mon.get("window_mask_probe"),
                         "qk_norm_hooks": trains[arm]["patch"].get("n_qkv_hooks"),
                         "qk_norm_params_added": trains[arm]["patch"].get("params_added")}
        stability[arm] = {"gnorm_max": mon["gnorm_max"], "gnorm_mean": mon["gnorm_mean"],
                          "grad_skips": mon["grad_skips"],
                          "final_logged_loss": rows[-1]["loss"],
                          "last_window_slope_nats_per_1k": _late_slope(rows)}

    cfg = {
        "contract": "axes/A-mixing/A6-mixing-operator/experiments/2636-longT-unified/README.md",
        "scope": ("本研究只做单轴单格实验，基底恒为 stock 对照；不做组合实验"
                  "（组合留白是显式声明，不是遗漏）。deps 只记录耦合方向。"),
        "intervention": {
            "factor": "训练期可见窗口（full vs T/2，同相对刻度 50%）× 序列长度 T；固定 S=6000 步",
            "arms": {a["arm"]: a["role"] for a in avail},
            "dropped_arms": dropped,
            "mask": ("keep[i,j] = (j<=i) & ((i-j) < train_window)；对角恒开；实现 = "
                     "pathlm.depth_ar.window_anchor_keep_mask"),
            "qk_norm": ("长度条件档：seq_len > 512 时逐头 L2 归一 q/k（乘 √d_h，零新增参数）；"
                        "seq_len ≤ 512 时不挂 hook（n_qkv_hooks == 0）"),
        },
        "model": {"d_model": D, "n_layers": L, "n_heads": H, "mlp_mult": MLP,
                  "dense": True, "topology": "sequential"},
        "data": {"name": "enwik8_full", "path": DATA,
                 "split": "前 90M train / 末 10M eval", "vocab_size": 206},
        "step_budget": S, "token_budget_by_T": TOKENS_AT_S,
        "token_budget_ref": TOKEN_BUDGET_REF,
        "batch_size": BS, "lr": LR,
        "arms": [{"arm": a["arm"], "seq_len": a["seq_len"], "window": a["window"],
                  "seed": a["seed"], "steps": S, "tokens": TOKENS_AT_S[a["seq_len"]],
                  "fields": ev["field_by_field"][a["arm"]]} for a in avail],
        "checkpoints": {a["arm"]: {
            "path": f"runs/{a['arm']}/model.pt",
            "sha256": trains[a["arm"]]["sha256_model_pt"],
            "size_bytes": trains[a["arm"]]["size_bytes"],
            "sha256_verified": True} for a in avail},
        "code_fingerprints_at_report": {
            "pathlm/depth_ar.py": sha256_file(os.path.join(ROOT, "pathlm/depth_ar.py")),
            "pathlm/model.py": sha256_file(os.path.join(ROOT, "pathlm/model.py")),
            "train_depth_ar.py": sha256_file(os.path.join(ROOT, "train_depth_ar.py")),
            "tools/evalkit/retention.py": sha256_file(
                os.path.join(ROOT, "tools/evalkit/retention.py")),
            "experiments/2625-longT-stability/stability_driver.py": sha256_file(
                os.path.join(EXP_ROOT, "2625-longT-stability", "stability_driver.py")),
            "note": ("**report 阶段**重算 ⇒ 记的是报告时 HEAD/工作树，**不是运行版本**；"
                     "本实验未改这些文件（只读）"),
        },
        # 运行期指纹（修 P1-5）：优先读 preflight 快照 `run_fingerprints.json`；
        # 缺失时回退到人工补记 `run_fingerprints_backfill.json`（即本次运行的来源）。
        "code_fingerprints_at_run": _load_run_fingerprints(),
    }

    results = {
        "contract": cfg["contract"], "scope": cfg["scope"], "config": cfg,
        "arms": {a["arm"]: {**trains[a["arm"]], "battery": b[a["arm"]],
                            "selfcheck": ev["selfchecks"][a["arm"]]} for a in avail},
        "criteria": {
            "criterion_1_quality_matched_step": {
                "delta_convention": "Δ = bpc_full − bpc_window（正 = 窗口更好）",
                "four_points": step_rows,
                "four_point_shape": shape_step,
                "variance_band_project": band,
                "variance_band_cell_delta": noise.get("cell_threshold_for_delta"),
                "variance_band_cell_single_arm_reference": noise.get("sigma_cell_single_arm_max"),
                "note": ("同 step S=6000；T=1024/2048 只读复用 2626/2629（同 step 6000、"
                         "同 bs8/seed0/qk-normfrom512）。token 预算随 T 线性（见 tokens）。"),
            },
            "criterion_1b_quality_matched_token": {
                "delta_convention": "Δ = bpc_full − bpc_window（正 = 窗口更好）",
                "four_points": token_rows,
                "four_point_shape": shape_token,
                "note": "同 token 49.152M 副口径，全部只读复用；steps = 24000/12000/6000/3000",
            },
            "criterion_2_stability_gate": gate,
            "criterion_3_four_legs": c3,
            "criterion_4_cell_noise": noise,
        },
        "cost": ev["cost"],
        "synthetic_leg": {
            "task": "var_recall_batch（tools/evalkit/longrange_tasks.py，未改工具）",
            "distances_by_T": {str(k): list(v) for k, v in SYNTH_DISTS.items()},
            "chance": 1.0 / 64, "local_prior_k": 8,
            "results": ev["synth"],
            "disclaimer": ("⚠️ var_recall 系列用**末位查询布局**，已由 A-AUDIT §2-P1-1 判为"
                           "**位置外推伪影**（2623 §9：同权重末位布局 full=0.0000 vs 同族多查询"
                           "0.3665）⇒ 本腿读数**不可解释**，只作流水记录，不得当长程/机制判据"),
        },
        "liveness": liveness, "stability": stability,
        "dropped_arms": dropped,
        "preflight_smoke": smoke["arms"],
        "wall_total_min": round(sum(trains[a["arm"]]["wall_min"] for a in avail), 2),
    }
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump(cfg, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({"shape_step": shape_step, "shape_token": shape_token,
                      "four_points": step_rows,
                      "cell_threshold_for_delta": noise.get("cell_threshold_for_delta")},
                     indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("preflight", "train", "train-all", "eval", "report", "all"))
    ap.add_argument("--arm", default=None)
    ap.add_argument("--cost-iters", type=int, default=6)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    if args.stage == "preflight":
        stage_preflight(args.cost_iters)
    elif args.stage == "train":
        if not args.arm:
            raise SystemExit("--stage train 需要 --arm")
        stage_train(args.arm)
    elif args.stage == "train-all":
        stage_train_all()
    elif args.stage == "eval":
        stage_eval(args.cost_iters)
    elif args.stage == "report":
        stage_report()
    elif args.stage == "all":
        if not args.skip_train:
            stage_train_all()
        stage_eval(args.cost_iters)
        stage_report()


if __name__ == "__main__":
    main()
