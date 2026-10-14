"""2655 · 2.3.1 逐层容量 schedule 最小判决 · 13M · dense · 纯 CLI.

契约 = 同目录 `README.md`（判据线 P1–P6 **预登记、不可改**）。本文件是唯一入口。

设计要点（为什么这样接线，见契约 §2.1）：

* **训练整段走共享代码**：本文件把 `train_depth_ar.DepthARModel` 全局名换成一个
  只做「把 `--mlp-hidden-sched` 透传进 cfg」的薄壳（引擎 E2 已实装 `mlp_hidden_sched`，
  提交 `4638882`），然后调 `train_depth_ar.main()` —— 参数解析、配置装配、
  `train()` 循环、checkpoint、`results.json` 全是共享实现，本文件不改共享代码。
  默认 cfg 路径与显式 uniform schedule 逐位一致由 `smoke.py` 的等值门单测钉住。
* **臂间唯一差别** = 每层 FFN 隐藏宽 `h_l`；`Σ_l h_l` 逐位相等 = 8192 ⇒
  FFN 参数（`Σ_l (2·h_l·d + h_l + d)`）与每 token FFN 等效 MACs（`Σ_l 2·h_l·d`）
  逐位相等（契约 §4-P4 的等预算门）。
* 逐层容量只由 CLI 表达，引擎无实验本地模型类。

用法：

    python3 run.py --arms uniform,front-big,growing --seed 0 --steps 12000
    python3 run.py --arms uniform --no-train          # 仅评测（复用 runs/…/model.pt）
    python3 run.py --aggregate                        # 汇总 P1–P6 → results.json

GPU 锁按 `SCHEMA.md` §20 取、**按臂释放**（§20.1，臂间 ≥60 s 窗口）。
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import train_depth_ar as TDA                                     # noqa: E402
from gpu_lock import GPUUnavailable, gpu_lock                    # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, depth_curve  # noqa: E402
from pathlm.data import load_enwik8_full                          # noqa: E402

DEVICE = "cuda"
LN2 = math.log(2.0)

# ---------------------------------------------------------------- 臂定义（契约 §3）

D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 8, 256, 4
SUM_HIDDEN = 8192                 # == 8 × 4 × 256（uniform 基准）
VARIANCE_BOUND = 0.014
N_WINDOWS, BATCHES_PER_WINDOW, EVAL_BATCH_SIZE = 5, 4, 16

#: `uniform` = 引擎缺省形态（h_l = mlp_mult·d = 1024），作等值门的显式 schedule 侧。
SCHEDULES = {
    "uniform":    [1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024],
    "front-big":  [1792, 1536, 1280, 1024, 768, 768, 512, 512],
    "growing":    [512, 512, 768, 768, 1024, 1280, 1536, 1792],
}
ARMS = list(SCHEDULES)

#: 历史同引擎 dense 13M 锚（契约 §4-P5 等值门用；同 config/seed/steps，仅 sched 不同）。
HIST_2613_SEQ_S0 = 1.5254         # 2613 sequential-12k-s0
HIST_2641_U4_S0 = 1.5314          # 2641 U4-s0（= 本 uniform 臂的同配方臂）


# ---------------------------------------------------------------- 训练（走共享 main）

def shared_argv(arm: str, seed: int, steps: int, batch_size: int, lr: float,
                out_root: str) -> list[str]:
    """与 `train_depth_ar.main()` 的 CLI 逐字对齐（dense 形态；唯一变量 = schedule）。"""
    return [os.path.join(ROOT, "train_depth_ar.py"),
            f"{arm}-s{seed}-{steps // 1000}k",
            "--d", str(D_MODEL), "--layers", str(N_LAYERS),
            "--heads", str(N_HEADS), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT),
            "--mlp-hidden-sched", ",".join(str(h) for h in SCHEDULES[arm]),
            "--dense", "--topology", "sequential",
            "--exit-mode", "uniform", "--steps", str(steps),
            "--batch-size", str(batch_size), "--lr", str(lr), "--seed", str(seed),
            "--eval-batches", "20", "--out-root", out_root]


def train_arm(arm: str, seed: int, steps: int, batch_size: int, lr: float,
              out_root: str) -> dict:
    argv_backup = list(sys.argv)
    try:
        sys.argv = shared_argv(arm, seed, steps, batch_size, lr, out_root)
        t0 = time.time()
        TDA.main()                       # ← 共享实现：解析 / 配置 / train() / 落盘
        wall = time.time() - t0
    finally:
        sys.argv = argv_backup
    return {"wall_s": round(wall, 1)}


def run_dir(arm: str, seed: int, steps: int, out_root: str) -> str:
    return os.path.join(out_root, f"{arm}-s{seed}-{steps // 1000}k")


# ---------------------------------------------------------------- 代价（等效理论量）

def analytic_cost(sched: list[int], d: int = D_MODEL, T: int = SEQ_LEN,
                  vocab: int = 206) -> dict:
    """每 token 等价 MACs（解析，**不含任何时间量**）+ FFN 参数（解析）。"""
    L = len(sched)
    attn_per_layer = 4 * d * d + 2 * T * d      # qkv 3d² + proj d² + QKᵀ T·d + AV T·d
    ffn_per_layer = [2 * h * d for h in sched]
    readout = d * vocab                          # tied 读出，每 token d×V
    total = L * attn_per_layer + sum(ffn_per_layer) + readout
    ffn_params = sum(2 * h * d + h + d for h in sched)
    return {
        "d_model": d, "seq_len": T, "n_layers": L, "vocab": vocab,
        "macs_per_token": {
            "attention": L * attn_per_layer,
            "ffn": sum(ffn_per_layer),
            "readout_tied": readout,
            "total": total,
        },
        "ffn_macs_per_token_by_layer": ffn_per_layer,
        "hidden_by_layer": list(sched),
        "sum_hidden": sum(sched),
        "ffn_params_analytic": ffn_params,
        "ffn_share_of_total_macs": round(sum(ffn_per_layer) / total, 6),
        "ffn_macs_rel_to_uniform": round(
            sum(ffn_per_layer) / sum(2 * h * d for h in [MLP_MULT * d] * L), 6),
        "note": ("等效理论量：MACs/token（乘加）。attention 的 T·d 项随序列长度线性、"
                 "FFN 的 h·d 项与 T 无关；两列分开报，不用任何时间量。"
                 "tokens-per-forward = 1（本格不涉及多 token 前向）。"
                 "各臂 Σh 逐位相等 ⇒ ffn 列与 ffn_params_analytic 逐位相等。"),
    }


# ---------------------------------------------------------------- 评测（本实验自持）

@torch.no_grad()
def windowed_bpc(model: DepthARModel, eval_arr, n_windows: int = N_WINDOWS,
                 batches_per_window: int = BATCHES_PER_WINDOW,
                 batch_size: int = EVAL_BATCH_SIZE, gen_seed: int = 2) -> dict:
    """把评测切成 `n_windows` 个**数据完全相同**的窗口（同一 generator seed 顺序取批），
    逐窗 bpc ⇒ 配对 Δ 的**窗口间 SE**（本仓 C1 口径：同 checkpoint 内的比较报窗间 SE）。"""
    from pathlm.data import batch
    gen = torch.Generator().manual_seed(gen_seed)
    T = model.cfg.seq_len
    model.eval()
    per_window = []
    for _ in range(n_windows):
        nats, n = 0.0, 0
        for _ in range(batches_per_window):
            x, _ = batch(eval_arr, batch_size, T, gen)
            x = x.to(DEVICE)
            _, aux = model(x, exit_depths=None)
            lg = aux["depth_logits"][-1][:, :-1]
            tgt = x[:, 1:]
            nats += torch.nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1),
                reduction="sum").item()
            n += tgt.numel()
        per_window.append(round(nats / n / LN2, 6))
    clean = round(sum(per_window) / len(per_window), 4)
    return {"window_bpc": per_window, "clean_bpc": clean,
            "n_windows": n_windows, "batches_per_window": batches_per_window,
            "batch_size": batch_size, "eval_generator_seed": gen_seed,
            "note": ("窗口 = 同一 generator seed 顺序下的连续数据批；所有臂共用该顺序 ⇒ "
                     "逐窗配对可比（窗口间 SE 是抽样噪声，0.014 是跨 run 训练漂移带）。")}


@torch.no_grad()
def ffn_liveness(model: DepthARModel, eval_arr, n_batches: int = 2,
                 batch_size: int = 8, seed: int = 1234) -> dict:
    """干预活性（`SCHEMA.md` §10）：在一次真实前向里用 forward hook 抓每层 `mlp` 的
    输入/输出，报逐层 `‖out‖/‖in‖` 的逐位置均值 —— 窄层不是死路径、宽层不是摆设；
    同时抓实际隐藏维（审计 schedule 是否真的到了模型）。"""
    from pathlm.data import batch
    gen = torch.Generator().manual_seed(seed)
    hidden = [int(blk.mlp[0].weight.shape[0]) for blk in model.blocks]
    acc = [[] for _ in model.blocks]
    handles = []

    def make_hook(l):
        def hook(_mod, inp, out):
            x = inp[0].detach().float()
            y = out.detach().float()
            xn = x.norm(dim=-1).clamp_min(1e-6)
            yn = y.norm(dim=-1)
            acc[l].append(float((yn / xn).mean()))
        return hook

    for l, blk in enumerate(model.blocks):
        handles.append(blk.mlp.register_forward_hook(make_hook(l)))
    was_training = model.training
    model.eval()
    try:
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
            model(x.to(DEVICE), exit_depths=None)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return {"hidden_by_layer": hidden,
            "ffn_out_rel_mag_by_layer": [round(statistics.mean(v), 6)
                                         for v in acc],
            "note": ("隐藏维 = 逐层审计（必须逐位等于 schedule）；相对幅度 = "
                     "真实前向中 ‖mlp_out‖/‖mlp_in‖ 的逐位置均值"
                     "（>0 ⇒ 该层 FFN 真的参与前向，不是死路径）")}


def param_audit(model: DepthARModel) -> dict:
    total = sum(p.numel() for p in model.parameters())
    ffn = sum(p.numel() for blk in model.blocks for p in blk.mlp.parameters())
    return {"params_total": total, "params_ffn": ffn,
            "params_attn_and_rest": total - ffn,
            "hidden_by_layer": [int(blk.mlp[0].weight.shape[0])
                                for blk in model.blocks]}


def loss_slope(rows: list[dict], steps: int, frac: float = 0.25) -> dict:
    """同方法 loss 斜率（`SCHEMA.md` §16）：末 frac 窗、最小二乘、×1000。"""
    start = int(steps * (1 - frac))
    win = [r for r in rows if start <= r["step"] <= steps]
    if len(win) < 2:
        return {"n_points": len(win), "slope_per_1k_steps": None}
    xs = [r["step"] for r in win]
    ys = [r["loss"] for r in win]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - xm) ** 2 for x in xs)
    slope = (sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / den
             if den else float("nan"))
    return {"n_points": len(win), "step_first": xs[0], "step_last": xs[-1],
            "slope_per_1k_steps": round(slope * 1000, 5),
            "window": f"末 {int(frac * 100)}%"}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate(arm: str, seed: int, steps: int, out_root: str,
             n_batches: int = 20) -> dict:
    rd = run_dir(arm, seed, steps, out_root)
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    cfg = DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                        seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True,
                        topology="sequential",
                        mlp_hidden_sched=list(SCHEDULES[arm]))
    model = DepthARModel(cfg, vocab).to(DEVICE)
    state = torch.load(os.path.join(rd, "model.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    curve = depth_curve(model, eval_arr, n_batches=n_batches,
                        batch_size=EVAL_BATCH_SIZE, device=DEVICE)
    win = windowed_bpc(model, eval_arr)
    rows = [json.loads(l) for l in open(os.path.join(rd, "train_log.jsonl"))
            if l.strip()]
    shared = json.load(open(os.path.join(rd, "results.json")))
    out = {
        "arm": arm, "seed": seed, "steps": steps,
        "hidden_by_layer": SCHEDULES[arm],
        "sum_hidden": sum(SCHEDULES[arm]),
        "eval": {"metric": "clean bpc（最终深度、无退出、序列级、越低越好）",
                 "n_batches": n_batches, "batch_size": EVAL_BATCH_SIZE,
                 "eval_generator_seed": 2,
                 "bpc_by_depth": curve,
                 "clean_bpc": win["clean_bpc"],
                 "clean_bpc_from_depth_curve": curve[-1]["bpc"],
                 "clean_bpc_two_paths_agree": (
                     abs(win["clean_bpc"] - curve[-1]["bpc"]) <= 5e-5),
                 "windows": win,
                 "acc_by_depth": [{"depth": r["depth"], "acc": r["acc"]}
                                  for r in curve]},
        "params": param_audit(model),
        "cost": analytic_cost(SCHEDULES[arm], vocab=vocab),
        "liveness": ffn_liveness(model, eval_arr),
        "loss_slope": loss_slope(rows, steps),
        "train_log_rows": len(rows),
        "shared_train_results": {
            "params": shared.get("params"),
            "train_tokens": shared.get("train_tokens"),
            "wall_minutes": shared.get("wall_minutes"),
            "peak_gpu_mem_mb": shared.get("peak_gpu_mem_mb"),
            "config": shared.get("config"),
            "gnorm_monitor": shared.get("gnorm_monitor"),
        },
        "resolved_config": shared.get("config"),
        "run_dir": os.path.relpath(rd, ROOT),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
    }
    out["params"]["hidden_matches_schedule"] = (
        out["params"]["hidden_by_layer"] == SCHEDULES[arm])
    out["cost"]["ffn_params_measured"] = out["params"]["params_ffn"]
    out["cost"]["ffn_params_analytic_matches_measured"] = (
        out["cost"]["ffn_params_analytic"] == out["params"]["params_ffn"])
    path = os.path.join(HERE, f"results_{arm}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[2655] {arm}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
          f"params_total={out['params']['params_total']} "
          f"hidden={out['params']['hidden_by_layer']}", flush=True)
    return out


# ---------------------------------------------------------------- 汇总（P1–P6）

def aggregate(steps: int, out_root: str, arms: list[str], seeds: list[int]) -> dict:
    recs = {}
    for arm in arms:
        for s in seeds:
            p = os.path.join(HERE, f"results_{arm}_s{s}.json")
            if os.path.exists(p):
                recs[(arm, s)] = json.load(open(p))

    def bpc(arm, s):
        return recs[(arm, s)]["eval"]["clean_bpc"] if (arm, s) in recs else None

    # P1/P2：配对 Δ = bpc_uniform − bpc_arm（正 = 该臂更好，SCHEMA §17.3 符号约定）
    deltas = {}
    for s in seeds:
        if ("uniform", s) not in recs:
            continue
        base = bpc("uniform", s)
        deltas[s] = {a: (round(base - bpc(a, s), 4) if (a, s) in recs else None)
                     for a in arms if a != "uniform"}

    # 窗间 SE（同 checkpoint：抽样噪声，与 0.014 跨 run 训练漂移带分开报）
    win_paired = {}
    if ("uniform", 0) in recs:
        for a in arms:
            if a == "uniform" or (a, 0) not in recs:
                continue
            wu = recs[("uniform", 0)]["eval"]["windows"]["window_bpc"]
            wa = recs[(a, 0)]["eval"]["windows"]["window_bpc"]
            d = [round(u - v, 6) for u, v in zip(wu, wa)]
            win_paired[a] = {
                "delta_bpc_uniform_minus_arm_by_window": d,
                "mean_delta": round(statistics.mean(d), 6),
                "se_across_windows": round(statistics.stdev(d) / math.sqrt(len(d)), 6),
                "n_windows": len(d)}

    def verdict(a, s):
        d = deltas.get(s, {}).get(a)
        if d is None:
            return None
        if d >= VARIANCE_BOUND:
            return f"可分：{a}（非均匀）在等 FFN 预算下优于 uniform"
        if d <= -VARIANCE_BOUND:
            return f"可分（反向）：{a} 在等 FFN 预算下劣于 uniform"
        return "不可排序（H0）：|Δ| < 0.014 ⇒ 逐层容量分配在本尺度无独立增益"

    eq_arms = [a for a in arms if (a, 0) in recs]
    eq_vals = {a: (recs[(a, 0)]["cost"]["macs_per_token"]["ffn"],
                   recs[(a, 0)]["params"]["params_ffn"],
                   recs[(a, 0)]["sum_hidden"]) for a in eq_arms}
    # 只有 ≥2 臂的都落盘时才能核对「逐位相等」；不足则**不判过也不判不过**
    # （`smoke.py` 的 `B_equal_budget` 已在模型层对三臂 schedule 做过该核对）
    if len(eq_vals) < 2:
        eq_status = "insufficient_arms"
    elif len(set(eq_vals.values())) == 1:
        eq_status = "pass"
    else:
        eq_status = "fail"

    p5 = None
    if ("uniform", 0) in recs:
        b = recs[("uniform", 0)]["eval"]["clean_bpc"]
        d_2613 = round(b - HIST_2613_SEQ_S0, 4)
        d_2641 = round(b - HIST_2641_U4_S0, 4)
        p5 = {"uniform_s0_clean_bpc": b,
              "vs_2613_sequential_s0": d_2613,
              "vs_2641_U4_s0": d_2641,
              "pass": bool(abs(d_2613) < 0.005 or abs(d_2641) < 0.005),
              "rule": ("显式 uniform schedule 与引擎缺省（`--mlp-mult 4`）逐位一致的"
                       "**模型层**证明见 `smoke.py`（state_dict + 前向逐位）；本行是"
                       "**训练层**旁证：与同配方历史臂之差 < 0.005 ⇒ 同引擎同配方")}

    out = {
        "experiment": "2655-layer-capacity-13m",
        "criterion": ("P1 Δ_front = bpc_uniform − bpc_front-big；"
                      "P2 Δ_grow = bpc_uniform − bpc_growing（正 = 该臂更好）；"
                      "P3 逐窗配对 Δ + 窗间 SE；P4 等 FFN 预算门（Σh / FFN 参数 / "
                      "FFN MACs 逐位相等）；P5 引擎等值门；P6 活性（隐藏维审计）。"
                      "方差带 0.014（限配对 Δ）。"),
        "variance_bound": VARIANCE_BOUND,
        "steps": steps,
        "arms": {f"{a}-s{s}": {"clean_bpc": bpc(a, s),
                               "sum_hidden": recs[(a, s)]["sum_hidden"],
                               "hidden_by_layer": recs[(a, s)]["hidden_by_layer"],
                               "macs_per_token": recs[(a, s)]["cost"]
                               ["macs_per_token"]["total"],
                               "macs_ffn": recs[(a, s)]["cost"]
                               ["macs_per_token"]["ffn"],
                               "params_total": recs[(a, s)]["params"]
                               ["params_total"],
                               "params_ffn": recs[(a, s)]["params"]["params_ffn"],
                               "slope_per_1k": recs[(a, s)]["loss_slope"]
                               ["slope_per_1k_steps"]}
                          for (a, s) in sorted(recs)},
        "P1_P2_deltas_by_seed": deltas,
        "P1_P2_verdict_by_seed": {str(s): {a: verdict(a, s)
                                           for a in deltas.get(s, {})}
                                  for s in seeds},
        "P3_window_paired_delta": win_paired,
        "P4_equal_ffn_budget_check": {
            "rule": "各臂 Σh = 8192、FFN 参数、每 token FFN MACs 必须逐位相等",
            "values_by_arm": {a: {"ffn_macs": v[0], "params_ffn": v[1],
                                  "sum_hidden": v[2]}
                              for a, v in eq_vals.items()},
            "status": eq_status, "pass": eq_status == "pass",
            "note": ("<2 臂落盘时 status=insufficient_arms（不判过也不判不过）；模型层的"
                     "三臂等预算核对见 `smoke_report.json:B_equal_budget`")},
        "P5_engine_equivalence_gate": p5,
        "P6_liveness": {f"{a}-s{s}": {"hidden_matches_schedule":
                                      recs[(a, s)]["params"]
                                      ["hidden_matches_schedule"],
                                      "ffn_params_analytic_matches_measured":
                                      recs[(a, s)]["cost"]
                                      ["ffn_params_analytic_matches_measured"],
                                      "ffn_out_rel_mag_by_layer":
                                      recs[(a, s)]["liveness"]
                                      ["ffn_out_rel_mag_by_layer"]}
                        for (a, s) in sorted(recs)},
        "source_files": [f"results_{a}_s{s}.json" for (a, s) in sorted(recs)],
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--no-train", action="store_true",
                    help="跳过训练，直接复用 runs/…/model.pt 评测")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="已完成的臂也重跑（缺省：有 results_<arm>_s<seed>.json 且 "
                         "model.pt 在则跳过 ⇒ 可续跑）")
    ap.add_argument("--lock-wait-s", type=float, default=600.0,
                    help="单臂等 GPU 锁上限（SCHEMA §20：最多 10 min）")
    ap.add_argument("--arm-gap-s", type=float, default=65.0,
                    help="臂间释放窗口（SCHEMA §20.1：≥60 s）")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a.strip() != ""]
    if args.aggregate:
        aggregate(args.steps, args.out_root, arms, [args.seed])
        return

    os.makedirs(args.out_root, exist_ok=True)
    log_path = os.path.join(HERE, "run_log.txt")
    failed = []
    for i, arm in enumerate(arms):
        done_path = os.path.join(HERE, f"results_{arm}_s{args.seed}.json")
        model_path = os.path.join(run_dir(arm, args.seed, args.steps,
                                          args.out_root), "model.pt")
        if (not args.force and os.path.exists(done_path)
                and (args.no_train or os.path.exists(model_path))):
            print(f"[2655] arm={arm} already done ⇒ skip（--force 可重跑）", flush=True)
            continue
        rec = {"arm": arm, "seed": args.seed, "steps": args.steps,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "held_lock": False}
        try:
            # 单臂 = 训练 + 评测一段，整段持锁；臂间释放（SCHEMA §20.1）
            with gpu_lock(name="wp2655-layer-capacity",
                          max_wait=args.lock_wait_s) as lk:
                rec["held_lock"] = True
                if not args.no_train:
                    rec["train"] = train_arm(arm, args.seed, args.steps,
                                             args.batch_size, args.lr,
                                             args.out_root)
                lk.heartbeat()
                rec["eval"] = {"clean_bpc": evaluate(arm, args.seed, args.steps,
                                                     args.out_root)
                               ["eval"]["clean_bpc"]}
            rec["status"] = "ok"
        except GPUUnavailable as e:
            rec["status"] = "gpu_lock_timeout"
            rec["error"] = str(e)
        finally:
            with open(log_path, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[2655] arm={arm} status={rec['status']}", flush=True)
        if rec["status"] != "ok":
            failed.append(arm)
        if i != len(arms) - 1 and rec["status"] == "ok":
            time.sleep(args.arm_gap_s)
    aggregate(args.steps, args.out_root, arms, [args.seed])
    if failed:
        raise SystemExit(9)          # 有臂没跑成（如等锁超时）⇒ 非零，便于 launcher 重试


if __name__ == "__main__":
    main()
