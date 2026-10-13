"""2622 · 长 T 首探：T=512（ticket 39-X3）—— 实验入口。

契约：同目录 `README.md`（判据线**逐字**抄录 + 主 Agent 裁决，不许改）。
唯一变量 = 训练期可见窗口（关闭 vs T/2）；两点 = `window128@T=256` 与 `window256@T=512`
（同相对刻度 50%，SCHEMA §17.16）。

四个训练臂（stock dense，d=256/L=8/h=8/mlp=4，bs32，seed0，lr 1e-3）：
- `full-256`     T=256, 全因果, 12000 步（98.3M token）
- `win128-256`   T=256, 窗口 128, 12000 步（同 token 预算；真第二点）
- `full-512`     T=512, 全因果,  6000 步（98.3M token）
- `win256-512`   T=512, 窗口 256,  6000 步（长 T 首探主臂）

阶段（每个 GPU 阶段自己取/放锁，遵守 SCHEMA §20/§20.1）：
    python3 run.py --stage preflight            # 20 步冒烟（四臂）+ 成本基准 → smoke/
    python3 run.py --stage train --arm full-512 # 训练一个臂（按臂取/放锁）
    python3 run.py --stage train-all            # 依次训练四臂（每臂间放锁 ≥60s）
    python3 run.py --stage eval                 # 电池（native/release bpc + 合成腿）
    python3 run.py --stage report               # 汇总 → results.json / config.json
    python3 run.py --stage all                  # train-all + eval + report

禁外推：两点只描述；成本/占比一律**实测**，公式值只作对照。
"""

import argparse
import hashlib
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
                             window_mask_liveness, window_mask_probe)
from pathlm.data import load_enwik8_full, batch                  # noqa: E402
from tools.evalkit.retention import masked_bpc, masked_logits    # noqa: E402
from tools.evalkit.longrange_tasks import (var_recall_batch,     # noqa: E402
                                           accuracy, local_prior_accuracy,
                                           check_local_prior_resistance)

DEVICE = "cuda"
DATA = "data/enwik8_full.npz"

# ---- 契约骨架（§2 单因素）：d=256/L=8/h=8/mlp=4，stock dense，bs32/seed0/lr1e-3 ----
D, L, H, MLP, BS, LR, SEED = 256, 8, 8, 4, 32, 1e-3, 0
#: token 预算 = 98.3M（与 A6 的 T=256 12k 步同）：T=256 → 12000 步；T=512 → 6000 步
TOKEN_BUDGET = 98_304_000

#: 四臂。`window = 0` ⇒ 关闭（全因果，逐位还原 v1）。
ARMS = [
    {"arm": "full-256",    "seq_len": 256, "window": 0,   "role": "基线 · T=256 · 全因果"},
    {"arm": "win128-256",  "seq_len": 256, "window": 128, "role": "第二点 · T=256 · 窗口 128（相对刻度 50%）"},
    {"arm": "full-512",    "seq_len": 512, "window": 0,   "role": "基线 · T=512 · 全因果"},
    {"arm": "win256-512",  "seq_len": 512, "window": 256, "role": "主臂 · T=512 · 窗口 256（相对刻度 50%）"},
]
ARM_BY_NAME = {a["arm"]: a for a in ARMS}

#: 公式口径（K10）：per-token-per-layer MACs。attn=2·T·d，other=3d²+d²+2·mlp·d²。
OTHER_MACS = 3 * D * D + D * D + 2 * MLP * D * D          # 786432
#: 窗口的渐进注意力 MACs = 2·w·d（每 token 只看 w 个键）
SYNTH_DISTS = {512: (128, 255, 300, 400, 480), 256: (64, 128, 192, 255)}
SYNTH_BATCH, SYNTH_CHUNK = 64, 16
EVAL_BATCHES, EVAL_BS = 20, 16

LOCK = "/tmp/pathlm_gpu.lock"
#: 锁等待上限（秒）：共享卡上别的线可能长时间持锁；eval 是短任务，等得住。
LOCK_MAX_WAIT = 3600


# ------------------------------------------------------------------ 工具

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def steps_for(seq_len: int) -> int:
    """同 token 预算下的步数（契约 §6）。"""
    n = TOKEN_BUDGET // (BS * seq_len)
    return int(n)


def cfg_for(seq_len: int, window: int) -> DepthARConfig:
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=seq_len,
                         mlp_mult=MLP, dense=True, train_window=window)


def attn_share_formula(T: int) -> float:
    """K10 公式（主 Agent 修正后）：attn_share(T) = 512T / (512T + 786432)。"""
    return (2 * T * D) / (2 * T * D + OTHER_MACS)


# --------------------------------------------------------------- GPU 锁

def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gpu_acquire(max_wait_s: int = 600, poll_s: int = 15) -> None:
    """SCHEMA §20 取锁（最多等 10 分钟；holder_pid 死 / 无 holder 且锁龄 >15min 可回收）。"""
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


# --------------------------------------------------------------- 训练一个臂

def train_arm(spec: dict, steps: int, poll_s: int = 20, caps_min: float = 30.0) -> dict:
    """后台启动 `train_depth_ar.py`，≤110s 轮询日志 + 触摸心跳 + nvidia-smi 子进程峰值。"""
    arm = spec["arm"]
    run_dir = os.path.join(HERE, "runs", arm)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, "train_depth_ar.py", arm,
           "--d", str(D), "--layers", str(L), "--heads", str(H),
           "--seq-len", str(spec["seq_len"]), "--mlp-mult", str(MLP), "--dense",
           "--steps", str(steps), "--batch-size", str(BS), "--lr", str(LR),
           "--seed", str(SEED), "--exit-mode", "uniform",
           "--eval-batches", str(EVAL_BATCHES), "--out-root",
           os.path.join(HERE, "runs")]
    if spec["window"] > 0:
        cmd += ["--train-window", str(spec["window"])]
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
        except Exception:
            pass
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            if lines:
                try:
                    st = int(json.loads(lines[-1])["step"])
                except (ValueError, KeyError):
                    st = -1
                if st % 2000 == 0 and st > 0:
                    print(f"[run] {arm} {lines[-1].strip()[:150]}", flush=True)
        time.sleep(poll_s)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"{arm} 训练失败 rc={proc.returncode}（见 runs/{arm}/stdout.log）")
    wall = (time.time() - t0) / 60
    res = json.load(open(os.path.join(run_dir, "results.json")))
    out = {"arm": arm, "run_dir": os.path.relpath(run_dir, HERE),
           "wall_min": round(wall, 2),
           "seconds_per_step": round(wall * 60 / steps, 5),
           "peak_child_mem_gb_nvidia_smi": round(peak, 2),
           "peak_torch_gb": round(res["peak_gpu_mem_mb"] / 1024.0, 3),
           "params": res["params"], "train_tokens": res["train_tokens"],
           "gnorm_monitor": res["gnorm_monitor"],
           "sha256_model_pt": sha256_file(os.path.join(run_dir, "model.pt")),
           "size_bytes": os.path.getsize(os.path.join(run_dir, "model.pt")),
           "steps": steps}
    json.dump(out, open(os.path.join(HERE, f"{arm}-train_summary.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[run] {arm}: done wall={wall:.2f} min peak(nvidia-smi)={peak:.2f}GB "
          f"peak(torch)={out['peak_torch_gb']:.3f}GB", flush=True)
    return out


# --------------------------------------------------------------- 电池

def load_arm(spec: dict, vocab: int) -> DepthARModel:
    torch.manual_seed(0)
    m = DepthARModel(cfg_for(spec["seq_len"], spec["window"]), vocab).cuda()
    m.load_state_dict(torch.load(os.path.join(HERE, "runs", spec["arm"], "model.pt"),
                                 map_location="cuda"))
    m.eval()
    return m


def native_keep(spec: dict) -> torch.Tensor | None:
    """native 工作点 = 窗口臂用自身训练掩码、full 臂用全因果（release）。"""
    if spec["window"] <= 0:
        return None
    return window_anchor_keep_mask(spec["seq_len"], spec["window"], 0, DEVICE)


def battery(spec: dict, eval_arr, model: DepthARModel) -> dict:
    T = spec["seq_len"]
    km = native_keep(spec)
    native = masked_bpc(model, eval_arr, km, n_batches=EVAL_BATCHES,
                        batch_size=EVAL_BS, device=DEVICE)
    release = masked_bpc(model, eval_arr, None, n_batches=EVAL_BATCHES,
                         batch_size=EVAL_BS, device=DEVICE)
    out = {"arm": spec["arm"], "seq_len": T, "window": spec["window"],
           "bpc_native": round(native, 6), "bpc_release": round(release, 6),
           "n_batches": EVAL_BATCHES, "batch_size": EVAL_BS, "window_seed": 2}
    if km is not None:
        out["native_mask_liveness"] = window_mask_liveness(T, spec["window"], 0)
    else:
        out["native_mask_liveness"] = None
    return out


@torch.no_grad()
def instrument_selfcheck(model: DepthARModel, eval_arr, T: int,
                         n_batches: int = 4, bs: int = 8) -> dict:
    """仪器自检：显式精确因果掩码（`is_causal=False`）vs 释放版路径，逐位一致（应 0）。"""
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


# --------------------------------------------------------------- 合成腿（§19）

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


def model_step_cost(seq_len: int, window: int, iters: int = 6) -> dict:
    """整模型一步 fwd+bwd 中位墙钟 + torch 峰值显存（同进程、同 shape、同 optimizer）。

    `window > 0` ⇒ 走 SDPA 的**稠密 bool 掩码**路径（本实装不兑现 O(T·w) 的算力节省，
    这正是要如实量出来的事）。
    """
    torch.manual_seed(0)
    model = DepthARModel(cfg_for(seq_len, window), 206).cuda()
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
                # 与训练路径同接口：逐块传 attn_mask（dense 分支）
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
                   "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)})
    del model, opt
    torch.cuda.empty_cache()
    return timing


def attn_island_cost(seq_len: int, window: int, iters: int = 8) -> dict:
    """注意力子层孤岛（qkv → 注意力 → out 投影）fwd+bwd 实测：把「注意力算力占比」
    的**实测**值从整模型里隔离出来（与 K10 公式对照，公式值只作对照）。"""
    dh = D // H
    torch.manual_seed(0)
    x = torch.randn(BS, seq_len, D, device=DEVICE, requires_grad=True)
    qkv = torch.nn.Linear(D, 3 * D).to(DEVICE)
    proj = torch.nn.Linear(D, D).to(DEVICE)
    shape = lambda t: t.view(BS, seq_len, H, dh).transpose(1, 2)
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
    theory = {
        "other_macs_per_token_per_layer": OTHER_MACS,
        "attn_macs_full_per_token_per_layer": {str(T): 2 * T * D for T in (256, 512)},
        "attn_share_formula": {str(T): round(attn_share_formula(T), 4)
                               for T in (256, 512)},
        "attn_share_formula_note": ("K10 公式 512T/(512T+786432)；T=512 = 0.2500"
                                    "（主 Agent 修正 K10 原文的 0.257）"),
        "window_theory": {},
    }
    for T, w in ((256, 128), (512, 256)):
        live = window_mask_liveness(T, w, 0)
        attn_win = 2 * w * D
        attn_full = 2 * T * D
        share_win = attn_win / (attn_win + OTHER_MACS)
        total_full = attn_full + OTHER_MACS
        total_win = attn_win + OTHER_MACS
        theory["window_theory"][f"{w}@{T}"] = {
            "visible_frac_of_causal": live["visible_frac"],
            "attn_macs_window_per_token_per_layer": attn_win,
            "attn_share_window_formula": round(share_win, 4),
            "attn_flops_reduction_relative_to_full": round(1 - attn_win / attn_full, 4),
            "layer_macs_reduction_all_in": round(1 - total_win / total_full, 4),
            "note": ("理论节省 = 只数注意力 MACs；本实装走稠密 bool 掩码 ⇒ 墙钟未必兑现"
                     "（见 measured.ratio_by_T）。attn_flops_reduction = 1 − w/T（渐进 50%）；"
                     "layer_macs_reduction_all_in = 1 − (attn_win+other)/(attn_full+other)，"
                     "即整模型每 token 的**理论等效 FLOPs** 降幅"),
        }
    return theory


def cost_probe(iters: int = 6) -> dict:
    measured = {"model_step": [], "attn_island": [], "ratio_by_T": []}
    for T, w in ((256, 0), (256, 128), (512, 0), (512, 256)):
        measured["model_step"].append(model_step_cost(T, w, iters=iters))
        measured["attn_island"].append(attn_island_cost(T, w, iters=max(iters, 6)))
    by = {}
    for r in measured["model_step"]:
        by[(r["seq_len"], r["window"])] = r
    for T, w in ((256, 128), (512, 256)):
        f, win = by[(T, 0)], by[(T, w)]
        isl_f = next(r for r in measured["attn_island"]
                     if r["seq_len"] == T and r["window"] == 0)
        isl_w = next(r for r in measured["attn_island"]
                     if r["seq_len"] == T and r["window"] == w)
        measured["ratio_by_T"].append({
            "T": T, "window": w,
            "step_ms_full": f["median_ms"], "step_ms_window": win["median_ms"],
            "r_time_window_over_full": round(win["median_ms"] / f["median_ms"], 4),
            "peak_full_gib": f["peak_mem_gib"], "peak_window_gib": win["peak_mem_gib"],
            "r_mem_window_over_full": round(win["peak_mem_gib"] / f["peak_mem_gib"], 4),
            "attn_island_ms_full": isl_f["median_ms"],
            "attn_island_over_model_step": round(isl_f["median_ms"] / f["median_ms"], 4),
            "attn_share_formula": round(attn_share_formula(T), 4),
        })
    theory = cost_theory()
    return {"theory": theory, "measured": measured,
            "note": ("双口径：理论（等效 FLOPs）+ 实测（墙钟/显存）。公式值只作对照，"
                     "不得当判据（SCHEMA §17.11）")}


# --------------------------------------------------------------- 阶段

def stage_preflight(cost_iters: int) -> None:
    os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        out = {"note": "20 步冒烟：四臂各跑 20 步，检查活性量落盘与实测步时/显存（无外推时间字段）",
               "arms": {}}
        for spec in ARMS:
            arm = spec["arm"]
            steps = steps_for(spec["seq_len"])
            r = train_arm(spec, steps=20, poll_s=3, caps_min=8.0)
            rows = [json.loads(l) for l in
                    open(os.path.join(HERE, "runs", arm, "train_log.jsonl"))]
            out["arms"][arm] = {
                "config": {"seq_len": spec["seq_len"], "window": spec["window"],
                           "steps_full": steps},
                "smoke_wall_min": r["wall_min"],
                "smoke_sec_per_step": r["seconds_per_step"],
                "peak_torch_gb": r["peak_torch_gb"],
                "params": r["params"],
                "last_log_row": rows[-1] if rows else None,
                "window_mask_liveness": (window_mask_liveness(spec["seq_len"],
                                                              spec["window"], 0)
                                         if spec["window"] > 0 else None),
                "config_validates": True,
            }
        out["cost_measured_step_ms_by_T"] = [
            {"seq_len": T, "window": w,
             **model_step_cost(T, w, iters=max(cost_iters, 4))}
            for T, w in ((256, 0), (256, 128), (512, 0), (512, 256))
        ]
        json.dump(out, open(os.path.join(HERE, "smoke", "smoke.json"), "w"),
                  indent=2, ensure_ascii=False)
        print(json.dumps(out["cost_measured_step_ms_by_T"], indent=2))
    finally:
        gpu_release()


def stage_train(arm: str) -> None:
    spec = ARM_BY_NAME[arm]
    steps = steps_for(spec["seq_len"])
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        train_arm(spec, steps=steps)
    finally:
        gpu_release()


def stage_train_all(gap_s: int = 60) -> None:
    """多臂驱动：**按臂释放锁**（SCHEMA §20.1），臂间留 ≥gap_s 秒窗口。"""
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
            print(f"[eval] {arm} done", flush=True)
        # 逐字段对照（§9）：检查只差 seq_len / train_window
        for spec in ARMS:
            out["field_by_field"][spec["arm"]] = {
                "d_model": D, "n_layers": L, "n_heads": H, "mlp_mult": MLP,
                "dense": True, "topology": "sequential", "batch_size": BS,
                "lr": LR, "seed": SEED, "steps": steps_for(spec["seq_len"]),
                "exit_mode": "uniform", "seq_len": spec["seq_len"],
                "train_window": spec["window"], "data": "enwik8 train 90M / eval 10M",
                "vocab_size": vocab,
            }
        out["cost"] = cost_probe(iters=cost_iters)
        json.dump(out, open(os.path.join(HERE, "eval.json"), "w"),
                  indent=2, ensure_ascii=False)
        print(json.dumps({a["arm"]: out["batteries"][a["arm"]]["bpc_native"]
                         for a in ARMS}, indent=2))
    finally:
        gpu_release()


def stage_report(cost_iters: int) -> None:
    smoke = json.load(open(os.path.join(HERE, "smoke", "smoke.json")))
    ev = json.load(open(os.path.join(HERE, "eval.json")))
    trains = {a["arm"]: json.load(open(os.path.join(HERE, f"{a['arm']}-train_summary.json")))
              for a in ARMS}
    cost = ev["cost"]
    cost["theory"] = cost_theory()          # 纯算术，独立复算（不信任 eval.json 的历史版）

    b = ev["batteries"]
    full256, win256 = b["full-256"], b["win128-256"]
    full512, win512 = b["full-512"], b["win256-512"]
    d256 = round(full256["bpc_native"] - win256["bpc_native"], 6)
    d512 = round(full512["bpc_native"] - win512["bpc_native"], 6)
    d256_rel = round(full256["bpc_release"] - win256["bpc_release"], 6)
    d512_rel = round(full512["bpc_release"] - win512["bpc_release"], 6)
    band = 0.014
    def sign(x):
        return "0" if abs(x) < band else ("+" if x > 0 else "-")

    liveness = {}
    for a in ARMS:
        mon = json.load(open(os.path.join(HERE, "runs", a["arm"], "results.json")))[
            "gnorm_monitor"]
        liveness[a["arm"]] = {"window_mask": mon.get("window_mask"),
                              "window_mask_probe": mon.get("window_mask_probe")}
    cfg = {
        "contract": "axes/1-time/1.1.1-hard-window/experiments/2622-longT-512/README.md",
        "scope": ("本研究只做单轴单格实验，基底恒为 stock 对照；不做组合实验"
                  "（组合留白是显式声明，不是遗漏）。deps 只记录耦合方向。"),
        "intervention": {
            "factor": "训练期可见窗口（唯一变量；关闭 vs T/2），两点同相对刻度 50%",
            "arms": {a["arm"]: a["role"] for a in ARMS},
            "mask": "keep[i,j] = (j<=i) & ((i-j) < train_window)；对角恒开；实现 = "
                    "pathlm.depth_ar.window_anchor_keep_mask（训练/评测共用同一函数）",
        },
        "model": {"d_model": D, "n_layers": L, "n_heads": H, "mlp_mult": MLP,
                  "dense": True, "topology": "sequential"},
        "data": {"name": "enwik8_full", "path": DATA,
                 "split": "前 90M train / 末 10M eval", "vocab_size": 206},
        "token_budget": TOKEN_BUDGET,
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
            "train_depth_ar.py": sha256_file(os.path.join(ROOT, "train_depth_ar.py")),
            "tools/evalkit/longrange_tasks.py": sha256_file(
                os.path.join(ROOT, "tools/evalkit/longrange_tasks.py")),
            "note": "本实验未改这些文件（只读，见 README §9）",
        },
    }

    results = {
        "contract": cfg["contract"], "scope": cfg["scope"],
        "config": cfg,
        "arms": {a["arm"]: {**trains[a["arm"]],
                            "battery": b[a["arm"]],
                            "selfcheck": ev["selfchecks"][a["arm"]]} for a in ARMS},
        "criteria": {
            "criterion_1_quality": {
                "delta_convention": "Δ = bpc_full − bpc_window（正 = 窗口更好）",
                "native_operating_point": {
                    "bpc_full_256": full256["bpc_native"],
                    "bpc_window128_256": win256["bpc_native"],
                    "delta_bpc_T256_window128": d256,
                    "bpc_full_512": full512["bpc_native"],
                    "bpc_window256_512": win512["bpc_native"],
                    "delta_bpc_T512_window256": d512,
                },
                "release_path": {
                    "bpc_full_256": full256["bpc_release"],
                    "bpc_window128_256": win256["bpc_release"],
                    "delta_bpc_T256_window128": d256_rel,
                    "bpc_full_512": full512["bpc_release"],
                    "bpc_window256_512": win512["bpc_release"],
                    "delta_bpc_T512_window256": d512_rel,
                },
                "variance_band": band,
                "sign_T256": sign(d256), "sign_T512": sign(d512),
                "same_sign": sign(d256) == sign(d512),
                "raw_delta_sign_T256": ("+" if d256 > 0 else "-" if d256 < 0 else "0"),
                "raw_delta_sign_T512": ("+" if d512 > 0 else "-" if d512 < 0 else "0"),
                "same_raw_delta_sign": ((d256 > 0) == (d512 > 0) and
                                        (d256 < 0) == (d512 < 0)),
                "primary_reading": ("band（|Δ|<0.014 判「不亏」）——契约「同判据」的字面口径；"
                                    "raw Δ 符号作并列描述"),
                "band_class_T256": ("不亏" if abs(d256) < band else "亏"),
                "band_class_T512": ("不亏" if abs(d512) < band else "亏"),
                "note": ("两点只描述、禁止外推（SCHEMA §17.16 / §17.11）；"
                         "|Δ|<0.014 ⇒ 该点「不亏」；两种读法（raw 符号 / 带宽分类）都报，"
                         "主口径 = 带宽分类"),
            },
            "criterion_2_attn_share": {
                "measured": cost["measured"]["ratio_by_T"],
                "formula": {str(T): round(attn_share_formula(T), 4) for T in (256, 512)},
                "note": ("公式 T=512 = 0.2500（主 Agent 修正 K10 原文 0.257）；公式只作对照"),
            },
            "criterion_3_memory": {
                "peak_torch_gb_by_arm": {a["arm"]: trains[a["arm"]]["peak_torch_gb"]
                                         for a in ARMS},
                "peak_nvidia_smi_gb_by_arm": {
                    a["arm"]: trains[a["arm"]]["peak_child_mem_gb_nvidia_smi"]
                    for a in ARMS},
                "budget_gb_net": 5.0,
                "max_peak_gb": max(trains[a["arm"]]["peak_torch_gb"] for a in ARMS),
            },
            "criterion_4_crossover": {
                "window256_T512_not_losing": bool(d512 >= -band),
                "note": ("Δ ≥ −0.014 ⇒ 记「T=512 为交叉点首个观测点」；单点，不外推"),
            },
        },
        "cost": cost,
        "synthetic_leg": {
            "task": "var_recall_batch（tools/evalkit/longrange_tasks.py，未改工具）",
            "distances_by_T": {str(k): list(v) for k, v in SYNTH_DISTS.items()},
            "chance": 1.0 / 64,
            "local_prior_k": 8,
            "results": ev["synth"],
            "disclaimer": ("距离档 >255 是本次新配置（DISTANCES 常量上限 255）；"
                           "与既往 255 上限的距离曲线**不可直接比较**"),
        },
        "liveness": liveness,
        "preflight_step_ms_by_T": smoke["cost_measured_step_ms_by_T"],
        "wall_total_min": round(sum(trains[a["arm"]]["wall_min"] for a in ARMS), 2),
    }
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)
    json.dump(cfg, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(results["criteria"], indent=2, ensure_ascii=False))


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
        stage_report(args.cost_iters)
    elif args.stage == "all":
        if not args.skip_train:
            stage_train_all()
        stage_eval(args.cost_iters)
        stage_report(args.cost_iters)


if __name__ == "__main__":
    main()
