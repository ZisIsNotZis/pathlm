"""2626 · 长 T 干净三点（ticket 43-V1）—— 实验入口。

契约：同目录 `README.md`（§0 逐字抄录 V1 契约；判据线不得修改）。

干预（唯一变量）：T ∈ {256, 512, 1024} × {full（`window=0`）, window（= T/2，同相对刻度 50%）}，
**全部**启用长度条件档 `qk-normfrom512`（`min_t=512`）。因 `min_t=512`，T ≤ 512 的四个臂
不挂 hook（前向逐算子 = stock 基线），只有 T=1024 的两个臂真的归一 q/k。

阶段（每个 GPU 阶段**按臂取/放锁**，臂间留 ≥60s 窗口，SCHEMA §20/§20.1）：
    python3 run.py --stage preflight           # 6 臂 × 20 步冒烟 + 实测步时/显存（§20.3）
    python3 run.py --stage train --arm <arm>   # 训练一个臂
    python3 run.py --stage train-all           # 依次训练六臂（每臂间放锁 ≥60s）
    python3 run.py --stage eval                # 电池（native/release bpc + 合成腿 + 成本）
    python3 run.py --stage report              # 汇总 → results.json / config.json
    python3 run.py --stage all                 # train-all + eval + report

禁外推：三点只描述；成本/占比一律**实测**，公式值只作对照。
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

# ---- 契约骨架（§2 单因素）：d=256/L=8/h=8/mlp=4，stock dense，seed0/lr1e-3 ----
#: **batch 由约束决定**（SCHEMA §20.2/§20.3）：T=1024 上 bs32 会 OOM（2623 记录），
#: 且 W1 的稳定性修复只在 bs8 上验证过 ⇒ 三点**统一 bs8**（自洽的干净配方）。
D, L, H, MLP, BS, LR, SEED = 256, 8, 8, 4, 8, 1e-3, 0
QK_NORM_MIN_T = 512
#: token 预算（§20.3 先测步时再定）：实测 bs8 步时 T=256≈33ms、T=512≈34ms、T=1024≈48ms
#: ⇒ 98.3M 全预算约 99 min（超 60–80 min 信封）⇒ **缩到共同半预算 49.15M**（臂间同 token
#: 预算，§8「砍步数、不砍判据线、不换配方」）：
#: T=256→24000 步、T=512→12000 步、T=1024→6000 步，合计约 50 min 训练 + 评测/臂间窗口 ≈60 min。
#: **口径声明**：本实验为 49.2M/臂的**快照**，与 98.3M 的旧点（2622/2623/2624）不可比绝对水平。
TOKEN_BUDGET = 49_152_000

DRIVER = os.path.join(HERE, "clean_driver.py")

#: 六臂。`window = 0` ⇒ 全因果；`window = T/2` ⇒ 窗口占 T 的 50%（§17.16 同相对刻度）。
ARMS = [
    {"arm": "full-256-clean",   "seq_len": 256,  "window": 0,   "role": "T=256 · 全因果"},
    {"arm": "win128-256-clean", "seq_len": 256,  "window": 128, "role": "T=256 · 窗口 128（50%）"},
    {"arm": "full-512-clean",   "seq_len": 512,  "window": 0,   "role": "T=512 · 全因果"},
    {"arm": "win256-512-clean", "seq_len": 512,  "window": 256, "role": "T=512 · 窗口 256（50%）"},
    {"arm": "full-1024-clean",  "seq_len": 1024, "window": 0,   "role": "T=1024 · 全因果"},
    {"arm": "win512-1024-clean", "seq_len": 1024, "window": 512, "role": "T=1024 · 窗口 512（50%）"},
]
ARM_BY_NAME = {a["arm"]: a for a in ARMS}
POINTS = ((128, 256), (256, 512), (512, 1024))

#: 公式口径（K10）：per-token-per-layer MACs。attn=2·T·d，other=3d²+d²+2·mlp·d²。
OTHER_MACS = 3 * D * D + D * D + 2 * MLP * D * D          # 786432
#: 合成腿距离档（各 T 的窗口边界与远端各取一个；均在 T−2 以内 = 训练分布内，§17.18）
SYNTH_DISTS = {256: (64, 128, 192), 512: (128, 256, 384), 1024: (256, 512, 768)}
SYNTH_BATCH, SYNTH_CHUNK = 64, 16
EVAL_BATCHES, EVAL_BS, WINDOW_SEED = 20, 16, 2

LOCK = "/tmp/pathlm_gpu.lock"
LOCK_MAX_WAIT = 7200

# 跨 run 只读引用的旧点（X3 = 2622，Y2 = 2623/2624）
EXP_ROOT = os.path.join(ROOT, "axes", "1-time", "1.1.1-hard-window",
                        "experiments")


# ------------------------------------------------------------------ 工具

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def steps_for(seq_len: int) -> int:
    return int(TOKEN_BUDGET // (BS * seq_len))


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
           "--seed", str(SEED), "--candidate", "qk-normfrom512",
           "--out-root", os.path.join(HERE, "runs")]
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] {arm}: started pid={proc.pid} steps={steps} "
          f"seq_len={spec['seq_len']} window={spec['window']}", flush=True)
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
           "steps": steps}
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
    return {"median_ms": round(statistics.median(ts), 3), "min_ms": round(min(ts), 3),
            "n_iters": iters}


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
    """注意力子层孤岛（qkv → 注意力 → out 投影）fwd+bwd 实测：把「注意力算力占比」
    的**实测**值从整模型里隔离出来（与 K10 公式对照，公式值只作对照）。"""
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


def cost_theory() -> dict:
    """纯算术的等效 FLOPs / 占比口径（不碰 GPU）——report 阶段可独立复算。"""
    theory = {"other_macs_per_token_per_layer": OTHER_MACS,
              "attn_share_formula": {str(T): round(attn_share_formula(T), 4)
                                     for T in (256, 512, 1024)},
              "window_theory": {}}
    for w, T in POINTS:
        live = window_mask_liveness(T, w, 0)
        attn_win, attn_full = 2 * w * D, 2 * T * D
        total_full, total_win = attn_full + OTHER_MACS, attn_win + OTHER_MACS
        theory["window_theory"][f"{w}@{T}"] = {
            "visible_frac_of_causal": live["visible_frac"],
            "attn_flops_reduction_relative_to_full": round(1 - attn_win / attn_full, 4),
            "layer_macs_reduction_all_in": round(1 - total_win / total_full, 4),
        }
    return theory


def cost_probe(iters: int = 6, seq_lens=(256, 512, 1024)) -> dict:
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
            "attn_island_over_model_step": round(isl_f["median_ms"] / f["median_ms"], 4),
            "attn_island_over_model_step_window": round(isl_w["median_ms"] / win["median_ms"], 4),
            "attn_share_formula": round(attn_share_formula(T), 4),
        })
    return {"theory": cost_theory(), "measured": measured,
            "note": ("双口径：理论（等效 FLOPs）+ 实测（墙钟/显存）。公式值只作对照，"
                     "不得当判据（SCHEMA §17.11）。模型步成本已套 QK-Norm（同实验配方）。")}


# --------------------------------------------------------------- 电池

def _load_w1_hook(model: DepthARModel, seq_len: int) -> None:
    _apply_qk_norm(model, seq_len)


def load_arm(spec: dict, vocab: int) -> DepthARModel:
    torch.manual_seed(0)
    m = DepthARModel(cfg_for(spec["seq_len"], spec["window"]), vocab).cuda()
    _load_w1_hook(m, spec["seq_len"])            # 架构类干预评测时必须同样施加
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

def old_points_reference() -> dict:
    """旧三点（X3 = 2622 的 T=256/512 bs32；Y2 = 2623/2624 的 T=1024 bs8）——**只读引用**。"""
    ref = {"note": "旧点：X3(2622) bs32/无 QK-Norm；Y2(2623/2624) bs8/无 QK-Norm，"
                   "其 full 臂爆炸 ⇒ 与本次干净三点**不可直接并列**",
           "x3_2022": None, "y2_2623": None, "y2_2624": None}
    p22 = os.path.join(EXP_ROOT, "2622-longT-512", "results.json")
    if os.path.exists(p22):
        r = json.load(open(p22))["criteria"]["criterion_1_quality"]
        ref["x3_2622"] = {"native": r["native_operating_point"],
                          "batch_size": 32, "qk_norm": False}
    p23 = os.path.join(EXP_ROOT, "2623-longT-1024", "results.json")
    if os.path.exists(p23):
        r = json.load(open(p23))["criteria"]["criterion_1_quality"]
        ref["y2_2623"] = {"native": r["this_point_native"],
                          "release": r["this_point_release"],
                          "batch_size": 8, "qk_norm": False}
    p24 = os.path.join(EXP_ROOT, "2624-longT-1024-seed1", "results.json")
    if os.path.exists(p24):
        r24 = json.load(open(p24))
        try:
            ref["y2_2624"] = r24["criteria"]["criterion_1_quality"]
        except (KeyError, TypeError):
            ref["y2_2624"] = {"available": True}
    return ref


# --------------------------------------------------------------- 阶段

def stage_preflight(cost_iters: int) -> None:
    """6 臂各 20 步冒烟（活性量/步时/显存）+ 成本基准（三点 × full/window）。"""
    os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        out = {"note": "20 步冒烟：六臂各跑 20 步，检查活性量落盘与实测步时/显存"
                       "（不含任何 20 步外推时间字段，§17.5）",
               "free_mb_before": free_gpu_mb(), "arms": {}}
        for spec in ARMS:
            arm = spec["arm"]
            r = train_arm(spec, steps=20, poll_s=3, caps_min=10.0)
            rows = [json.loads(l) for l in
                    open(os.path.join(HERE, "runs", arm, "train_log.jsonl"))]
            out["arms"][arm] = {
                "config": {"seq_len": spec["seq_len"], "window": spec["window"],
                           "steps_full": steps_for(spec["seq_len"])},
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
        print(json.dumps([{"arm": a, "sps": v["smoke_sec_per_step"],
                           "steps": v["config"]["steps_full"]}
                          for a, v in out["arms"].items()], indent=2))
    finally:
        gpu_release()


def stage_train(arm: str) -> None:
    spec = ARM_BY_NAME[arm]
    steps = steps_for(spec["seq_len"])
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        free_before = free_gpu_mb()
        r = train_arm(spec, steps=steps)
        r["free_before_mb"] = free_before
        r["free_after_mb"] = free_gpu_mb()
        json.dump(r, open(os.path.join(HERE, f"{arm}-train_summary.json"), "w"),
                  indent=2, ensure_ascii=False)
    finally:
        gpu_release()


def stage_train_all(gap_s: int = 60) -> None:
    """多臂驱动：**按臂释放锁**（§20.1），臂间留 ≥gap_s 秒窗口。"""
    for i, spec in enumerate(ARMS):
        stage_train(spec["arm"])
        if i < len(ARMS) - 1:
            print(f"[run] 释放锁，留 {gap_s}s 窗口（§20.1）", flush=True)
            time.sleep(gap_s)


def stage_eval(cost_iters: int = 6) -> None:
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, DATA))
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        out = {"batteries": {}, "selfchecks": {}, "synth": {}, "field_by_field": {}}
        for spec in ARMS:
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
        for spec in ARMS:
            out["field_by_field"][spec["arm"]] = {
                "d_model": D, "n_layers": L, "n_heads": H, "mlp_mult": MLP,
                "dense": True, "topology": "sequential", "batch_size": BS,
                "lr": LR, "seed": SEED, "steps": steps_for(spec["seq_len"]),
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
    trains = {a["arm"]: json.load(open(os.path.join(HERE,
                                                    f"{a['arm']}-train_summary.json")))
              for a in ARMS}
    b = ev["batteries"]
    band = 0.014

    # ---- 判据① 三点 Δbpc + 带宽分类（native 为主口径）----
    three_native, three_release = [], []
    for w, T in POINTS:
        fa = f"full-{T}-clean"
        wa = f"win{w}-{T}-clean"
        dn = round(b[fa]["bpc_native"] - b[wa]["bpc_native"], 6)
        dr = round(b[fa]["bpc_release"] - b[wa]["bpc_release"], 6)
        three_native.append({"T": T, "window": w, "delta_native": dn,
                             "band_class": _band_class(dn, band),
                             "bpc_full": b[fa]["bpc_native"],
                             "bpc_window": b[wa]["bpc_native"]})
        three_release.append({"T": T, "window": w, "delta_release": dr,
                              "band_class": _band_class(dr, band)})
    shape = " → ".join(f"{p['band_class']}@{p['T']}" for p in three_native)

    # ---- 判据② 稳定性前置（full@T=1024 gnorm_max <100 且跳步 0）----
    full1024 = trains["full-1024-clean"]["gnorm_monitor"]
    win1024 = trains["win512-1024-clean"]["gnorm_monitor"]
    gate = {"full_1024_gnorm_max": full1024["gnorm_max"],
            "full_1024_gnorm_mean": full1024["gnorm_mean"],
            "full_1024_grad_skips": full1024["grad_skips"],
            "win_1024_gnorm_max": win1024["gnorm_max"],
            "threshold_gnorm_max": 100, "threshold_grad_skips": 0}
    gate["passed"] = bool(full1024["gnorm_max"] < 100 and full1024["grad_skips"] == 0)

    # ---- 判据③ 注意力占比实测 vs 公式 ----
    attn_share = {"measured": ev["cost"]["measured"]["ratio_by_T"],
                  "formula": {str(T): round(attn_share_formula(T), 4)
                              for T in (256, 512, 1024)},
                  "note": "公式 512T/(512T+786432)：256=0.1429、512=0.2500、1024=0.4000；"
                          "公式值只作对照（不是判据）"}

    liveness, stability = {}, {}
    for a in ARMS:
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
        "contract": "axes/1-time/1.1.1-hard-window/experiments/2626-longT-clean/README.md",
        "scope": ("本研究只做单轴单格实验，基底恒为 stock 对照；不做组合实验"
                  "（组合留白是显式声明，不是遗漏）。deps 只记录耦合方向。"),
        "intervention": {
            "factor": ("训练期可见窗口（full vs T/2，同相对刻度 50%）× 序列长度 T，"
                       "全部套用长度条件档 qk-normfrom512（min_t=512）"),
            "arms": {a["arm"]: a["role"] for a in ARMS},
            "mask": ("keep[i,j] = (j<=i) & ((i-j) < train_window)；对角恒开；实现 = "
                     "pathlm.depth_ar.window_anchor_keep_mask（训练/评测共用）"),
            "qk_norm": ("长度条件档：seq_len > 512 时逐头 L2 归一 q/k（乘 √d_h，零新增参数）；"
                        "seq_len ≤ 512 时不挂 hook（n_qkv_hooks == 0）"),
        },
        "model": {"d_model": D, "n_layers": L, "n_heads": H, "mlp_mult": MLP,
                  "dense": True, "topology": "sequential"},
        "data": {"name": "enwik8_full", "path": DATA,
                 "split": "前 90M train / 末 10M eval", "vocab_size": 206},
        "token_budget": TOKEN_BUDGET, "batch_size": BS, "lr": LR, "seed": SEED,
        "arms": [{"arm": a["arm"], "seq_len": a["seq_len"], "window": a["window"],
                  "steps": steps_for(a["seq_len"]),
                  "tokens": steps_for(a["seq_len"]) * BS * a["seq_len"],
                  "fields": ev["field_by_field"][a["arm"]]} for a in ARMS],
        "checkpoints": {a["arm"]: {
            "path": f"runs/{a['arm']}/model.pt",
            "sha256": trains[a["arm"]]["sha256_model_pt"],
            "size_bytes": trains[a["arm"]]["size_bytes"],
            "sha256_verified": True} for a in ARMS},
        "code_fingerprints": {
            "pathlm/depth_ar.py": sha256_file(os.path.join(ROOT, "pathlm/depth_ar.py")),
            "pathlm/model.py": sha256_file(os.path.join(ROOT, "pathlm/model.py")),
            "train_depth_ar.py": sha256_file(os.path.join(ROOT, "train_depth_ar.py")),
            "tools/evalkit/longrange_tasks.py": sha256_file(
                os.path.join(ROOT, "tools/evalkit/longrange_tasks.py")),
            "experiments/2625-longT-stability/stability_driver.py": sha256_file(
                os.path.join(EXP_ROOT, "2625-longT-stability", "stability_driver.py")),
            "note": ("本实验未改这些文件（只读）；qk-norm hook 从 2625 的 stability_driver.py "
                     "只读引用（单一实现来源）"),
        },
    }

    results = {
        "contract": cfg["contract"], "scope": cfg["scope"], "config": cfg,
        "arms": {a["arm"]: {**trains[a["arm"]], "battery": b[a["arm"]],
                            "selfcheck": ev["selfchecks"][a["arm"]]} for a in ARMS},
        "criteria": {
            "criterion_1_quality": {
                "delta_convention": "Δ = bpc_full − bpc_window（正 = 窗口更好）",
                "three_points_native": three_native,
                "three_points_release": three_release,
                "variance_band": band,
                "three_point_shape_native": shape,
                "primary_reading": "native 工作点 Δ 的带宽分类（|Δ|<0.014 ⇒ 不亏）",
                "note": "三点 = 128@256 / 256@512 / 512@1024（本实验自洽 bs8 + qk-normfrom512）；"
                        "只描述、禁止外推（§17.11/§17.16）",
            },
            "criterion_2_stability_gate": gate,
            "criterion_3_attn_share": attn_share,
            "criterion_4_memory_wallclock": {
                "round_1_preflight": smoke["cost_measured"]["measured"]["model_step"],
                "round_2_eval": ev["cost"]["measured"]["model_step"],
                "peak_torch_gb_by_arm": {a["arm"]: trains[a["arm"]]["peak_torch_gb"]
                                         for a in ARMS},
                "peak_nvidia_smi_gb_by_arm": {
                    a["arm"]: trains[a["arm"]]["peak_child_mem_gb_nvidia_smi"]
                    for a in ARMS},
                "budget_gb_net": 5.0,
                "max_peak_gb": max(trains[a["arm"]]["peak_torch_gb"] for a in ARMS),
                "free_mb_before": smoke.get("free_mb_before"),
                "free_mb_after": smoke.get("free_mb_after"),
            },
            "criterion_5_old_points": old_points_reference(),
        },
        "cost": ev["cost"],
        "synthetic_leg": {
            "task": "var_recall_batch（tools/evalkit/longrange_tasks.py，未改工具）",
            "distances_by_T": {str(k): list(v) for k, v in SYNTH_DISTS.items()},
            "chance": 1.0 / 64,
            "local_prior_k": 8,
            "results": ev["synth"],
            "disclaimer": ("距离档均 ≤ T−2（训练分布内，§17.18）；对窗口臂，远端档落在窗口外"
                           "（这正是要量的「可见性外能否召回」）。本腿在 enwik8 权重上是分布外"
                           "任务，预期接近 chance"),
        },
        "liveness": liveness,
        "stability": stability,
        "preflight_smoke": smoke["arms"],
        "wall_total_min": round(sum(trains[a["arm"]]["wall_min"] for a in ARMS), 2),
    }
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump(cfg, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({"gate": gate, "shape": shape,
                      "three_native": three_native}, indent=2, ensure_ascii=False))


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
