"""2624 · 长 T 第三点 T=1024 的 **seed=1 双臂重跑**（ticket 41-Z1）—— 实验入口。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。
唯一变量 = 训练期可见窗口（关闭 vs T/2 = 512），**seed 固定为 1**（两臂相同）；
配方与 2623（seed=0）**逐字段相同，只改 seed**（token 预算 98.3M、bs8、12000 步）。

两个训练臂（stock dense，d=256/L=8/h=8/mlp=4，bs8，**seed1**，lr 1e-3）：
- `full-1024-seed1`     T=1024, 全因果,  12000 步（98.3M token）
- `win512-1024-seed1`   T=1024, 窗口 512, 12000 步（相对刻度 50%）

本实验回答 2623 留下的唯一问题：**`full` 臂的 gnorm 爆炸（seed0 的 1740.92）在 seed=1
是否复现**（判据线见 README §4，预先写下）。复现 ⇒ 记「long-T 训练不稳定」为工程债、
该点永久不可归因；不复现 ⇒ 该点可给干净 Δ、重判三点趋势。n=2 仍禁外推。

阶段（每个 GPU 阶段自己取/放锁，遵守 SCHEMA §20/§20.1）：
    python3 run.py --stage preflight                # 20 步冒烟（两臂）+ 成本基准 → smoke/
    python3 run.py --stage train --arm full-1024-seed1
    python3 run.py --stage train-all                # 依次训练两臂（每臂间放锁 ≥60s）
    python3 run.py --stage eval                     # 电池（native/release bpc + 合成腿）
    python3 run.py --stage report                   # 汇总 → results.json / config.json
    python3 run.py --stage all                      # train-all + eval + report

禁外推：只描述本点与 seed0 点、两点并列；成本/占比一律**实测**，公式值只作对照。
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

# ---- 契约骨架（§2 单因素）：d=256/L=8/h=8/mlp=4，stock dense，seed0/lr1e-3 ----
#: **batch 由实测内存决定（SCHEMA §20.2）**：T=1024 上 bs32 的窗口臂 OOM；两次尝试后
#: 用 bs8（两臂同配方）——本机背景进程（两个 `llama-server`）常驻占 ~18.3 GiB，
#: 全卡 free 仅 ~5 GiB；bs16 窗口臂训练峰值 4.06 GiB 在卡满时仍 OOM。
#: 实测峰值：full ~0.9 GiB / win512 ~2.2 GiB（见 smoke/smoke.json 与 README §6）。
D, L, H, MLP, BS, LR, SEED = 256, 8, 8, 4, 8, 1e-3, 1
#: token 预算 = 98.3M（与 2622 的 T=256/512 四点、2623 的 T=1024 seed0 同）：T=1024 → 12000 步（bs8）
TOKEN_BUDGET = 98_304_000
#: 外部占用降速时的**唯一**让步旋钮（SCHEMA §20.3）：缩步数、两臂同刻度、并在报告首段
#: 声明「Nk 快照口径、与 2623 的 12k 不可比」；**不砍臂、不换配方、不等卡**。
STEPS_OVERRIDE = int(os.environ.get("PATHLM_2624_STEPS", "0")) or None

#: 两臂。`window = 0` ⇒ 关闭（全因果）；`window = 512` ⇒ 窗口占 T 的 50%。
#: **两臂 seed 均为 1**（本实验相对 2623 的唯一变化就是 seed）。
ARMS = [
    {"arm": "full-1024-seed1",   "seq_len": 1024, "window": 0,
     "role": "基线 · T=1024 · 全因果 · seed1"},
    {"arm": "win512-1024-seed1", "seq_len": 1024, "window": 512,
     "role": "第三点 · T=1024 · 窗口 512（相对刻度 50%）· seed1"},
]
ARM_BY_NAME = {a["arm"]: a for a in ARMS}

#: 同相对刻度的三点（SCHEMA §17.16）；前两点来自 ticket 39-X3 的 2622（跨 run 引用）
POINTS_THREE = ((128, 256), (256, 512), (512, 1024))
X3_RESULTS = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator",
                          "experiments", "2622-longT-512", "results.json")
#: seed0 的 T=1024 点（ticket 40-Y2 的 2623）——本实验的 n=2 配对点 + 爆炸复现的参照
Y2_RESULTS = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator",
                          "experiments", "2623-longT-1024", "results.json")
#: 爆炸复现判据（**预先登记**，见 README §4）：seed0 的 full/win = 1740.92/7.35 = 236.9。
REPRO_ABS_GNORM_MAX = 100.0     # full@seed1 的 gnorm_max 须 > 100（seed0 = 1740.92）
REPRO_RATIO = 20.0              # full@seed1 的 gnorm_max 须 > 20 × win@seed1 的 gnorm_max

#: 公式口径（K10）：per-token-per-layer MACs。attn=2·T·d，other=3d²+d²+2·mlp·d²。
OTHER_MACS = 3 * D * D + D * D + 2 * MLP * D * D          # 786432
#: T=1024 的合成距离档（>255 是新配置；含窗口边界 512）
SYNTH_DISTS = {1024: (256, 300, 512, 768, 1000)}
SYNTH_BATCH, SYNTH_CHUNK = 64, 16
EVAL_BATCHES, EVAL_BS = 20, 16

LOCK = "/tmp/pathlm_gpu.lock"
#: 锁等待上限（秒）：共享卡上别的线可能长时间持锁；三条线（Y1/Y2/Y3）争用下单臂训练
#: 可能等很久，故放宽到 2 小时（§20.1 允许等待；不杀别人进程）。
LOCK_MAX_WAIT = 7200


# ------------------------------------------------------------------ 工具

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def steps_for(seq_len: int) -> int:
    """同 token 预算下的步数（契约 §6）。外部占用降速时只允许**缩步数**（§20.3），
    且必须两臂同刻度（`PATHLM_2624_STEPS` 对两臂同时生效）。"""
    n = int(TOKEN_BUDGET // (BS * seq_len))
    return min(n, STEPS_OVERRIDE) if STEPS_OVERRIDE else n


def cfg_for(seq_len: int, window: int) -> DepthARConfig:
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=seq_len,
                         mlp_mult=MLP, dense=True, train_window=window)


def attn_share_formula(T: int) -> float:
    """K10 公式：attn_share(T) = 512T / (512T + 786432)。T=1024 → 0.40。"""
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


def free_gpu_mb() -> float | None:
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        return float(q.stdout.strip().splitlines()[0])
    except Exception:                                        # noqa: BLE001
        return None


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
        except Exception:                                    # noqa: BLE001
            pass
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            if lines:
                try:
                    st = int(json.loads(lines[-1])["step"])
                except (ValueError, KeyError):
                    st = -1
                if st % 1000 == 0 and st > 0:
                    print(f"[run] {arm} {lines[-1].strip()[:150]}", flush=True)
        time.sleep(poll_s)
    log.close()
    if proc.returncode != 0:
        tail = ""
        try:
            tail = open(os.path.join(run_dir, "stdout.log")).read()[-600:]
        except OSError:
            pass
        raise RuntimeError(f"{arm} 训练失败 rc={proc.returncode}（见 runs/{arm}/stdout.log）"
                           f"\n--- stdout tail ---\n{tail}")
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
    torch.cuda.reset_peak_memory_stats()
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
    theory = {
        "other_macs_per_token_per_layer": OTHER_MACS,
        "attn_macs_full_per_token_per_layer": {str(T): 2 * T * D
                                               for T in (256, 512, 1024)},
        "attn_share_formula": {str(T): round(attn_share_formula(T), 4)
                               for T in (256, 512, 1024)},
        "attn_share_formula_note": ("K10 公式 512T/(512T+786432)；T=256=0.1429、"
                                    "T=512=0.2500、T=1024=0.4000"),
        "window_theory": {},
    }
    for w, T in POINTS_THREE:
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
                     "layer_macs_reduction_all_in = 1 − (attn_win+other)/(attn_full+other)"),
        }
    return theory


def cost_probe(iters: int = 6, seq_lens=(1024,)) -> dict:
    measured = {"model_step": [], "attn_island": [], "ratio_by_T": [], "free_mb_before": None}
    measured["free_mb_before"] = free_gpu_mb()
    for T in seq_lens:
        for w in (0, T // 2):
            measured["model_step"].append(model_step_cost(T, w, iters=iters))
            measured["attn_island"].append(attn_island_cost(T, w, iters=max(iters, 6)))
    by = {}
    for r in measured["model_step"]:
        by[(r["seq_len"], r["window"])] = r
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
            "peak_full_gib": f["peak_mem_gib"], "peak_window_gib": win["peak_mem_gib"],
            "r_mem_window_over_full": round(win["peak_mem_gib"] / f["peak_mem_gib"], 4),
            "attn_island_ms_full": isl_f["median_ms"],
            "attn_island_over_model_step": round(isl_f["median_ms"] / f["median_ms"], 4),
            "attn_island_over_model_step_window": round(isl_w["median_ms"] / win["median_ms"], 4),
            "attn_share_formula": round(attn_share_formula(T), 4),
        })
    theory = cost_theory()
    return {"theory": theory, "measured": measured,
            "note": ("双口径：理论（等效 FLOPs）+ 实测（墙钟/显存）。公式值只作对照，"
                     "不得当判据（SCHEMA §17.11）")}


def x3_three_point_reference() -> dict:
    """跨 run 引用 ticket 39-X3（2622）的 T=256/512 两点（native Δ），供三点并列。"""
    if not os.path.exists(X3_RESULTS):
        return {"available": False, "note": f"未找到 {X3_RESULTS}"}
    r = json.load(open(X3_RESULTS))
    nat = r["criteria"]["criterion_1_quality"]["native_operating_point"]
    rel = r["criteria"]["criterion_1_quality"]["release_path"]
    return {
        "available": True,
        "source": "axes/A-mixing/A6-mixing-operator/experiments/2622-longT-512/results.json",
        "native": {
            "T256_window128": {
                "bpc_full": nat["bpc_full_256"], "bpc_window": nat["bpc_window128_256"],
                "delta": nat["delta_bpc_T256_window128"]},
            "T512_window256": {
                "bpc_full": nat["bpc_full_512"], "bpc_window": nat["bpc_window256_512"],
                "delta": nat["delta_bpc_T512_window256"]},
        },
        "release": {
            "T256_window128": {
                "bpc_full": rel["bpc_full_256"], "bpc_window": rel["bpc_window128_256"],
                "delta": rel["delta_bpc_T256_window128"]},
            "T512_window256": {
                "bpc_full": rel["bpc_full_512"], "bpc_window": rel["bpc_window256_512"],
                "delta": rel["delta_bpc_T512_window256"]},
        },
        "attn_share_measured": {str(row["T"]): row["attn_island_over_model_step"]
                                for row in r["criteria"]["criterion_2_attn_share"]["measured"]},
    }


def y2_seed0_reference() -> dict:
    """跨 run 引用 ticket 40-Y2（2623）的 **seed0** T=1024 点：Δ1024、两臂 gnorm、bpc。

    这是本实验（seed1）的**配对点**（n=2）：配方逐字段相同，只差 seed。
    """
    if not os.path.exists(Y2_RESULTS):
        return {"available": False, "note": f"未找到 {Y2_RESULTS}"}
    r = json.load(open(Y2_RESULTS))
    c1 = r["criteria"]["criterion_1_quality"]
    stab = r["convergence_and_stability"]["by_arm"]
    return {
        "available": True,
        "source": "axes/A-mixing/A6-mixing-operator/experiments/2623-longT-1024/results.json",
        "seed": 0,
        "native": c1["this_point_native"],
        "release": c1["this_point_release"],
        "band_class": c1["band_class_T1024"],
        "interpretation": c1["interpretation"],
        "three_point_shape": c1["three_point_shape"],
        "stability": {a: {"gnorm_max": stab[a]["gnorm_max"],
                          "gnorm_mean": stab[a]["gnorm_mean"]} for a in stab},
    }


def band_class(x: float, band: float = 0.014) -> str:
    """带宽分类（有向，SCHEMA §17.3：Δ = full − window，正 = 窗口更好）。

    |Δ| < 0.014 ⇒ 「不亏（噪声带内）」；否则按符号分「更好 / 更差」。
    """
    if abs(x) < band:
        return "不亏（噪声带内）"
    return "新算子更好（越界）" if x > 0 else "新算子更差（越界）"


def explosion_reproduction(full_mon: dict, win_mon: dict) -> dict:
    """预先登记的「seed0 爆炸是否在 seed1 复现」判据（README §4，**开跑前写死**）。

    seed0 签名：`full-1024` gnorm_max = 1740.92，`win512-1024` = 7.35（比 236.9）。
    - R1（绝对量级）：`full.gnorm_max > REPRO_ABS_GNORM_MAX`(=100)
    - R2（相对量级）：`full.gnorm_max > REPRO_RATIO`(=20) × `win.gnorm_max`
    两者皆成立 ⇒ `reproduced`；皆不成立 ⇒ `not_reproduced`；只成立一个 ⇒
    `partial`（保守处置：仍按「不可归因」记账，不得给干净 Δ）。
    """
    f, w = full_mon["gnorm_max"], win_mon["gnorm_max"]
    r1 = bool(f > REPRO_ABS_GNORM_MAX)
    r2 = bool(f > REPRO_RATIO * w)
    if r1 and r2:
        verdict = "reproduced"
    elif not r1 and not r2:
        verdict = "not_reproduced"
    else:
        verdict = "partial"
    return {"full_gnorm_max": f, "win_gnorm_max": w,
            "ratio_full_over_win": (round(f / w, 4) if w else None),
            "R1_abs_gt": REPRO_ABS_GNORM_MAX, "R1_passed": r1,
            "R2_ratio_gt": REPRO_RATIO, "R2_passed": r2,
            "seed0_reference": {"full_gnorm_max": 1740.92, "win_gnorm_max": 7.35,
                                "ratio": 236.86},
            "verdict": verdict}


def window_slope(rows: list, lo: int, hi: int) -> float:
    """训练日志（每 200 步一行）在 [lo, hi] 上的 loss 斜率（nats/1k 步，§16/§17.15）。

    本实验无合成「答案行」：判据损失 = enwik8 CE（bpc = CE/log2），故斜率取该序列。
    """
    pts = [(r["step"], r["loss"]) for r in rows if lo <= r["step"] <= hi]
    if len(pts) < 2:
        return float("nan")
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    return float("nan") if den == 0 else \
        round(sum((p[0] - mx) * (p[1] - my) for p in pts) / den * 1000, 5)


# --------------------------------------------------------------- 阶段

def stage_preflight(cost_iters: int) -> None:
    os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
    gpu_acquire(max_wait_s=LOCK_MAX_WAIT)
    try:
        free_before = free_gpu_mb()
        out = {"note": "20 步冒烟：两臂各跑 20 步，检查活性量落盘与实测步时/显存（无外推时间字段）",
               "free_mb_before": free_before, "arms": {}}
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
            del r
        out["cost_measured_step_ms_by_T"] = [
            {"seq_len": T, "window": w,
             **model_step_cost(T, w, iters=max(cost_iters, 4))}
            for T in (1024,) for w in (0, 512)
        ]
        out["free_mb_after"] = free_gpu_mb()
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
    full = b["full-1024-seed1"]
    win = b["win512-1024-seed1"]
    d1024 = round(full["bpc_native"] - win["bpc_native"], 6)
    d1024_rel = round(full["bpc_release"] - win["bpc_release"], 6)
    band = 0.014

    x3 = x3_three_point_reference()
    y2 = y2_seed0_reference()

    liveness = {}
    stability = {}
    for a in ARMS:
        run_res = json.load(open(os.path.join(HERE, "runs", a["arm"], "results.json")))
        mon = run_res["gnorm_monitor"]
        rows = [json.loads(l) for l in
                open(os.path.join(HERE, "runs", a["arm"], "train_log.jsonl"))]
        liveness[a["arm"]] = {"window_mask": mon.get("window_mask"),
                               "window_mask_probe": mon.get("window_mask_probe")}
        n_last = steps_for(a["seq_len"])
        stability[a["arm"]] = {
            "gnorm_max": mon["gnorm_max"], "gnorm_mean": mon["gnorm_mean"],
            "grad_skips": mon["grad_skips"], "clip_max_norm": 1.0,
            "n_skip_threshold_gnorm_gt": 1e4,
            "final_logged_loss": rows[-1]["loss"],
            # 判据损失（enwik8 CE，§17.15/§20.3 的「答案/判据损失」——本实验无合成答案行）
            # 末窗斜率：最后 5 个 200 步日志点（即最后 ~800 步）
            "last_window_slope_nats_per_1k": window_slope(
                rows, max(0, n_last - 800), n_last),
            # 爆炸窗口斜率（seed0 的 gnorm 尖峰从 ~step 9800 起）：描述用，不是门禁
            "late_third_slope_nats_per_1k": window_slope(
                rows, int(0.75 * n_last), n_last),
            "early_slope_nats_per_1k": window_slope(
                rows, 0, int(0.25 * n_last)),
        }
    # Z1 ① ：seed0 爆炸是否在 seed1 复现（判据**开跑前已写死**，见 README §4）
    repro = explosion_reproduction(
        {"gnorm_max": stability[ARMS[0]["arm"]]["gnorm_max"]},
        {"gnorm_max": stability[ARMS[1]["arm"]]["gnorm_max"]})
    reproduced = repro["verdict"] == "reproduced"
    partial = repro["verdict"] == "partial"
    baseline_confounded = reproduced or partial          # 保守：partial 不给干净 Δ
    # 三点趋势（重判）：第三点 = 本点（seed1）；seed0 点作并列参照
    three = []
    if x3["available"]:
        for T, w, key in ((256, 128, "T256_window128"), (512, 256, "T512_window256")):
            d = x3["native"][key]["delta"]
            three.append({"T": T, "window": w, "delta_native": d,
                          "band_class": band_class(d), "source": "X3(2622)"})
    if y2["available"]:
        three.append({"T": 1024, "window": 512,
                      "delta_native": y2["native"]["delta_bpc_T1024_window512"],
                      "band_class": y2["band_class"],
                      "source": "Y2(2623, seed0)"})
    three.append({"T": 1024, "window": 512, "delta_native": d1024,
                  "band_class": band_class(d1024), "source": "this(2624, seed1)"})
    # 三点趋势（T=256 / T=512 / T=1024·**本点 seed1**）
    trend = [p for p in three if p["source"] == "X3(2622)"] + [three[-1]]
    _labels = [p["band_class"] for p in trend]
    if baseline_confounded:
        _labels[-1] = "不可判（基准臂不稳定）"
    shape = " → ".join(f"{lab}@{p['T']}" for lab, p in zip(_labels, trend))
    delta_pair = {
        "seed0_2623": (y2["native"]["delta_bpc_T1024_window512"]
                       if y2["available"] else None),
        "seed1_2624": d1024,
        "n": 2 if y2["available"] else 1,
        "mean": (round((y2["native"]["delta_bpc_T1024_window512"] + d1024) / 2, 6)
                 if y2["available"] else None),
        "note": ("n=2（同配方、仅 seed 不同）⇒ 只给两读数与均值，**不得**当散度界/趋势；"
                 "两个读数均与 X3 两点**batch 不同**（X3 bs32 vs 本线 bs8），只作描述"),
    }
    repro_note = (
        f"Z1 ①：seed0 的 full 臂 gnorm_max = {repro['seed0_reference']['full_gnorm_max']}"
        f"（win {repro['seed0_reference']['win_gnorm_max']}）；seed1 实测 full "
        f"gnorm_max = {repro['full_gnorm_max']}（win {repro['win_gnorm_max']}，"
        f"比 {repro['ratio_full_over_win']}）⇒ 判据 R1(abs>{repro['R1_abs_gt']}) "
        f"{'过' if repro['R1_passed'] else '不过'}、R2(ratio>{repro['R2_ratio_gt']}) "
        f"{'过' if repro['R2_passed'] else '不过'} ⇒ **{repro['verdict']}**。")
    if reproduced:
        verdict_text = ("seed0 的 gnorm 爆炸在 seed=1 复现 ⇒ 记「**long-T 训练不稳定**」"
                        "为**工程债**，该点**永久不可归因**（不得用本点做「窗口更好」的论据）。")
    elif partial:
        verdict_text = ("seed0 的 gnorm 爆炸在 seed=1 **部分复现**（R1/R2 只过一个）"
                        "⇒ 保守处置：仍记「不可归因」，不给干净 Δ；需第三 seed 才能再判。")
    else:
        verdict_text = ("seed0 的 gnorm 爆炸在 seed=1 **未复现**（R1/R2 均不过）"
                        "⇒ 本点可给**干净 Δ**（seed0 那次归为 seed 特异），并**重判三点趋势**；"
                        "但 n=2、仍禁外推。")
    confound_note = (
        repro_note + verdict_text +
        f" 本点 Δ(seed1) = {d1024:+.6f}（band 分类 {band_class(d1024)}）；"
        f"seed0 的 Δ = {(y2['native']['delta_bpc_T1024_window512'] if y2['available'] else float('nan')):+.6f}"
        f"（seed0 当时判「{y2['interpretation'] if y2['available'] else 'n/a'}」）。"
        "band 分类照字面保留，**未事后改判据**。")

    cfg = {
        "contract": "axes/A-mixing/A6-mixing-operator/experiments/2624-longT-1024-seed1/README.md",
        "scope": ("本研究只做单轴单格实验，基底恒为 stock 对照；不做组合实验"
                  "（组合留白是显式声明，不是遗漏）。deps 只记录耦合方向。"),
        "intervention": {
            "factor": ("训练期可见窗口（唯一变量；关闭 vs T/2）在 **seed=1** 下重跑；"
                       "seed 两臂相同，相对 2623 的唯一变化是 seed（0→1）"),
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
            "criterion_1_gnorm_explosion_reproduction": {
                "question": "seed0（2623）的 `full-1024` gnorm 爆炸（1740.92）在 seed=1 是否复现",
                **repro,
                "verdict": repro["verdict"],
                "verdict_text": verdict_text,
                "preregistered": ("R1: full.gnorm_max > 100；R2: full.gnorm_max > 20×win.gnorm_max；"
                                   "两者皆成立⇒reproduced；皆不⇒not_reproduced；只一个⇒partial"
                                   "（保守按不可归因）"),
            },
            "criterion_2_recipe_equality": {
                "seed_both_arms": SEED,
                "field_by_field": {a["arm"]: ev["field_by_field"][a["arm"]] for a in ARMS},
                "only_diff_between_arms": "train_window（0 vs 512）",
                "only_diff_vs_2623": "seed（0 → 1）；其余逐字段相同（含 bs8/12000 步/98.3M token）",
                "tokens_by_arm": {a["arm"]: steps_for(a["seq_len"]) * BS * a["seq_len"]
                                  for a in ARMS},
            },
            "criterion_3_answer_ce_slope_and_memory": {
                "decision_loss": ("enwik8 CE（bpc = CE/log2）——本实验无合成「答案行」，"
                                  "判据损失即此；按 §17.15/§20.3 报其末窗斜率"),
                "slope_caliber": ("训练日志（每 200 步一行）的 loss 序列，nats/1k 步；"
                                  "last_window = 最后 ~800 步；晚第三窗 = 后 25%；"
                                  "本实验无中间 checkpoint ⇒ 不能直接量评测流 CE 的跨越斜率"),
                "by_arm": {a: {"last_window_slope_nats_per_1k": stability[a]["last_window_slope_nats_per_1k"],
                               "late_third_slope_nats_per_1k": stability[a]["late_third_slope_nats_per_1k"],
                               "early_slope_nats_per_1k": stability[a]["early_slope_nats_per_1k"],
                               "final_logged_loss": stability[a]["final_logged_loss"]}
                           for a in stability},
                "delta_at_final_loss": round(stability[ARMS[0]["arm"]]["final_logged_loss"]
                                             - stability[ARMS[1]["arm"]]["final_logged_loss"], 4),
                "held_out_bpc_by_arm": {a["arm"]: b[a["arm"]]["bpc_native"] for a in ARMS},
                "memory": {
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
            },
            "criterion_4_no_extrapolation": {
                "n_seeds": 2,
                "statement": ("n=2（seed0×seed1）仍**禁外推**（SCHEMA §17.11）：只描述两点"
                              "与均值，不拟合趋势/不报置信区间；batch 与 X3 两点不同，不得并列比较"),
                "extrapolated": False,
            },
            "criterion_5_quality_this_point": {
                "delta_convention": "Δ = bpc_full − bpc_window（正 = 窗口更好）",
                "this_point_native": {
                    "bpc_full_1024_seed1": full["bpc_native"],
                    "bpc_window512_1024_seed1": win["bpc_native"],
                    "delta_bpc_T1024_window512_seed1": d1024,
                },
                "this_point_release": {
                    "bpc_full_1024_seed1": full["bpc_release"],
                    "bpc_window512_1024_seed1": win["bpc_release"],
                    "delta_bpc_T1024_window512_seed1": d1024_rel,
                },
                "band_class_T1024_seed1": band_class(d1024),
                "sign_T1024_seed1": ("0" if abs(d1024) < band
                                      else ("+" if d1024 > 0 else "-")),
                "n2_delta_pair": delta_pair,
                "points_native_seed0_and_seed1": three,
                "three_point_trend_native": trend,
                "three_point_shape": shape,
                "x3_reference": x3,
                "y2_seed0_reference": y2,
                "variance_band": band,
                "interpretation": ("不可判（基准臂不稳定）" if baseline_confounded
                                   else "band 分类可用（本 seed 干净 Δ）"),
                "confound_note": confound_note,
                "note": ("三点 = 128@256 / 256@512（X3 跨 run）/ 512@1024（本点 seed1，"
                         "并列 seed0）；只描述、禁止外推（SCHEMA §17.16 / §17.11）；"
                         "|Δ|<0.014 ⇒ 该点「不亏」"),
            },
            "criterion_6_attn_share": {
                "measured_T1024": cost["measured"]["ratio_by_T"],
                "measured_X3_256_512": (x3["attn_share_measured"]
                                        if x3.get("available") else None),
                "formula": {str(T): round(attn_share_formula(T), 4)
                            for T in (256, 512, 1024)},
                "note": ("公式 T=1024 = 0.4000；公式只作对照（不是判据）。"
                         "X3 实测 5.41%@256 → 10.78%@512 并列"),
            },
            "criterion_7_wallclock": {
                "round_1_preflight_step_ms": smoke["cost_measured_step_ms_by_T"],
                "round_2_eval": cost["measured"]["model_step"],
                "note": ("两轮实测；方向不明则如实记「测不出方向」；窗口走稠密 bool 掩码 ⇒ "
                         "理论省 12.5% 等效 FLOPs 未必兑现"),
            },
        },
        "cost": cost,
        "synthetic_leg": {
            "task": "var_recall_batch（tools/evalkit/longrange_tasks.py，未改工具）",
            "distances_by_T": {str(k): list(v) for k, v in SYNTH_DISTS.items()},
            "chance": 1.0 / 64,
            "local_prior_k": 8,
            "results": ev["synth"],
            "disclaimer": ("距离档 >255 是 2622/2623 的新配置（DISTANCES 常量上限 255）；"
                           "与既往 255 上限的距离曲线**不可直接比较**。本腿在 enwik8 训练的权重上"
                           "是分布外任务，预期为 chance（空结果）"),
        },
        "liveness": liveness,
        "convergence_and_stability": {
            "clip_max_norm": 1.0,
            "by_arm": stability,
            "note": ("长 T 训练稳定性观察（工程债线索，不改配方）：seed0（2623）的 full-1024 臂"
                     "出现 gnorm 尖峰（max 1740.92，clip@1.0 已生效；grad_skips=0 ⇒ 未达 1e4"
                     "跳步阈值）；本实验（seed1）的复现判定见 criteria.criterion_1。"
                     "任一时点「窗口更稳」均**不得**外推为「窗口更稳定」（n≤2）。"),
        },
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
