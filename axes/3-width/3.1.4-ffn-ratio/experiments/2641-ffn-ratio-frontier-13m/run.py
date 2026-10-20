"""2641 · 前馈隐藏比的质量-代价前沿 + 等 FFN 预算下的逐层非均匀宽度 · 13M · dense.

契约 = 同目录 `README.md`（判据线 P1–P6 **预登记、不可改**）。本文件是唯一入口。

设计要点（为什么这样接线，见契约 §2.1）：

* **训练整段走共享代码**：本文件把 `train_depth_ar` 模块里的全局名 `DepthARModel`
  换成一个**工厂**（构造完 stock 模型后按 r_schedule 对 FFN 做前缀切片），
  然后调用 `train_depth_ar.main()` —— 参数解析、配置装配、`train()` 循环、
  checkpoint、`results.json` 全部是**共享实现**，本文件不改共享代码。
* **臂间唯一的差别** = 每层 FFN 保留多少中间通道。所有臂的 embedding / attention /
  LN / 位置嵌入来自**同一次**随机抽样（同一 seed、同一构造顺序），FFN 权重也是
  **同一份抽样**的不同前缀切片 ⇒ 不是「两个不同初始化的模型」在比。
* 逐层 schedule 只写在本文件（实验目录内）；引擎没有「逐层 FFN 宽」开关，
  收编该开关的申请记在 `.scratch/64-axis3-width/TICKET.md`，本实验不擅改共享 `pathlm/**`。

用法：

    python3 run.py --arm PYR --seed 0 --steps 12000      # 训练 + 评测
    python3 run.py --arm PYR --seed 0 --steps 12000 --no-train   # 仅评测（复用 model.pt）
    python3 run.py --aggregate                           # 汇总 P1–P6 → results.json
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                   # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, depth_curve  # noqa: E402
from pathlm.data import load_enwik8_full                       # noqa: E402

DEVICE = "cuda"
LN2 = math.log(2.0)

# ---------------------------------------------------------------- 臂定义（契约 §3）

D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 8, 256, 4
VARIANCE_BOUND = 0.014

SCHEDULES = {
    "U4":   [4, 4, 4, 4, 4, 4, 4, 4],       # stock（Σr=32）= 代价 100% 参照
    "U2":   [2, 2, 2, 2, 2, 2, 2, 2],       # 等宽、等 FFN 预算（Σr=16）
    "PYR":  [4, 3, 2, 2, 2, 1, 1, 1],       # 金字塔（前宽后窄，Σr=16）
    "APYR": [1, 1, 1, 2, 2, 2, 3, 4],       # 倒金字塔（= PYR 镜像，Σr=16）
}

# 历史同引擎 dense 13M 锚（契约 §4-P5）：等值性门用
HIST_DENSE_BPC = 1.5287          # .scratch/23-depth-ar/evidence/23d-dense-B-12k
HIST_2613_BASELINE_S0 = 1.5257   # 2613 全新重训 baseline（同 config/seed/steps）
HIST_2613_BASELINE_S1 = 1.5298

_ACTIVE_SCHEDULE = None          # 由 main() 设置，供工厂读取


# ---------------------------------------------------------------- 干预：FFN 前缀切片

def narrow_ffn_prefix(model: DepthARModel, schedule: list[int],
                      d_model: int = D_MODEL) -> DepthARModel:
    """把每层 FFN 的中间维从 mlp_mult·d 收窄到 schedule[l]·d（**前缀切片**）。

    只碰 `blk.mlp[0].weight/bias`（外→内）与 `blk.mlp[2].weight`（内→外）的
    **前缀**行/列；GELU、Dropout、残差接线、attention 全不动。
    schedule[l] == mlp_mult 时该层是**无操作**（逐位保留原参数对象）。
    """
    if len(schedule) != len(model.blocks):
        raise ValueError(f"schedule 长度 {len(schedule)} != 层数 "
                         f"{len(model.blocks)}")
    for l, blk in enumerate(model.blocks):
        r = int(schedule[l])
        if not 1 <= r <= MLP_MULT:
            raise ValueError(f"第 {l + 1} 层 r={r} 不在 [1, {MLP_MULT}]")
        h = r * d_model
        lin1, lin2 = blk.mlp[0], blk.mlp[2]
        if r < MLP_MULT:
            lin1.weight = torch.nn.Parameter(lin1.weight[:h].clone())
            if lin1.bias is not None:
                lin1.bias = torch.nn.Parameter(lin1.bias[:h].clone())
            lin2.weight = torch.nn.Parameter(lin2.weight[:, :h].clone())
        assert lin1.weight.shape == (h, d_model), (lin1.weight.shape, h)
        assert lin2.weight.shape == (d_model, h), (lin2.weight.shape, h)
    return model


def build_model(schedule: list[int], vocab_size: int, seed: int,
                patch_rng: bool = True) -> DepthARModel:
    """所有臂共用同一构造路径：先按 mlp_mult=r_max 建**全宽**模型（同 seed ⇒
    同一次抽样），再按 schedule 取前缀切片。"""
    if patch_rng:
        torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                        seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True)
    model = DepthARModel(cfg, vocab_size)
    if schedule != [MLP_MULT] * len(schedule):
        narrow_ffn_prefix(model, schedule)
    return model


# ---------------------------------------------------------------- 代价（等效理论量）

def analytic_cost(schedule: list[int], d: int = D_MODEL, T: int = SEQ_LEN,
                  vocab: int = 206) -> dict:
    """每 token 等价 MACs（解析，**不含任何时间量**）+ FFN 参数（解析）。"""
    L = len(schedule)
    attn_per_layer = 4 * d * d + 2 * T * d        # qkv 3d² + proj d² + QKᵀ T·d + AV T·d
    ffn_per_layer = [2 * r * d * d for r in schedule]
    readout = d * vocab                           # tied 读出，每 token d×V
    total = L * attn_per_layer + sum(ffn_per_layer) + readout
    ffn_params = sum(2 * r * d * d + r * d + d for r in schedule)
    return {
        "d_model": d, "seq_len": T, "n_layers": L, "vocab": vocab,
        "macs_per_token": {
            "attention": L * attn_per_layer,
            "ffn": sum(ffn_per_layer),
            "readout_tied": readout,
            "total": total,
        },
        "ffn_macs_per_token_by_layer": ffn_per_layer,
        "ffn_params_analytic": ffn_params,
        "ffn_share_of_total_macs": round(sum(ffn_per_layer) / total, 6),
        "ffn_macs_rel_to_U4": round(sum(ffn_per_layer)
                                    / sum(2 * 4 * d * d for _ in range(L)), 6),
        "note": ("等效理论量：MACs/token（乘加）。attention 的 T·d 项随序列长度线性、"
                 "FFN 的 r·d² 项与 T 无关；两列分开报，不用任何时间量。"
                 "tokens-per-forward = 1（本格不涉及多 token 前向）。"),
    }


# ---------------------------------------------------------------- 训练（走共享 main）

def _factory_for(schedule: list[int]):
    """替换 `train_depth_ar.DepthARModel` 的工厂：共享 main() 调它建模型。"""
    def factory(cfg, vocab_size):
        model = DepthARModel(cfg, vocab_size)
        if schedule != [MLP_MULT] * len(schedule):
            narrow_ffn_prefix(model, schedule, d_model=cfg.d_model)
        return model
    return factory


def shared_argv(arm: str, seed: int, steps: int, batch_size: int, lr: float,
                out_root: str) -> list[str]:
    """与 `train_depth_ar.main()` 的 CLI 逐字对齐（dense 形态；唯一变量 = schedule）。"""
    return [os.path.join(ROOT, "train_depth_ar.py"),
            f"{arm}-s{seed}-{steps // 1000}k",
            "--d", str(D_MODEL), "--layers", str(N_LAYERS),
            "--heads", str(N_HEADS), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT), "--dense", "--topology", "sequential",
            "--exit-mode", "uniform", "--steps", str(steps),
            "--batch-size", str(batch_size), "--lr", str(lr), "--seed", str(seed),
            "--eval-batches", "20", "--out-root", out_root]


def train_arm(arm: str, seed: int, steps: int, batch_size: int, lr: float,
              out_root: str) -> dict:
    schedule = SCHEDULES[arm]
    orig = TDA.DepthARModel
    TDA.DepthARModel = _factory_for(schedule)
    argv_backup = list(sys.argv)
    try:
        sys.argv = shared_argv(arm, seed, steps, batch_size, lr, out_root)
        t0 = time.time()
        TDA.main()                       # ← 共享实现：解析 / 配置 / train() / 落盘
        wall = time.time() - t0
    finally:
        TDA.DepthARModel = orig
        sys.argv = argv_backup
    return {"wall_s": round(wall, 1)}


# ---------------------------------------------------------------- 评测（本实验自持）

def run_dir(arm: str, seed: int, steps: int, out_root: str) -> str:
    return os.path.join(out_root, f"{arm}-s{seed}-{steps // 1000}k")


@torch.no_grad()
def ffn_liveness(model: DepthARModel, eval_arr, n_batches: int = 2,
                 batch_size: int = 8, seed: int = 1234) -> dict:
    """干预活性（`harness/SCHEMA.md` §10）：**在一次真实前向**里用 forward hook 抓每层
    `mlp` 的输入/输出，报逐层 `‖out‖/‖in‖` 的逐位置均值 —— 窄层不是死路径、
    宽层不是摆设；同时抓实际隐藏维（审计 schedule 是否真的到了模型）。"""
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
    """同方法 loss 斜率（`harness/SCHEMA.md` §16）：末 frac 窗、最小二乘、×1000。"""
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
    train_arr, eval_arr, vocab = load_enwik8_full(".tmp/enwik8",
                                                  "data/enwik8_full.npz")
    model = build_model(SCHEDULES[arm], vocab, seed).to(DEVICE)
    state = torch.load(os.path.join(rd, "model.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    curve = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                        device=DEVICE)
    rows = [json.loads(l) for l in open(os.path.join(rd, "train_log.jsonl"))
            if l.strip()]
    shared = json.load(open(os.path.join(rd, "results.json")))
    out = {
        "arm": arm, "seed": seed, "steps": steps,
        "schedule_r_by_layer": SCHEDULES[arm],
        "sum_r": sum(SCHEDULES[arm]),
        "eval": {"metric": "clean bpc（最终深度、无退出、序列级、越低越好）",
                 "n_batches": n_batches, "batch_size": 16,
                 "eval_generator_seed": 2,
                 "bpc_by_depth": curve,
                 "clean_bpc": curve[-1]["bpc"],
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
        "run_dir": os.path.relpath(rd, ROOT),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
    }
    out["params"]["hidden_matches_schedule"] = (
        out["params"]["hidden_by_layer"] == [r * D_MODEL for r in SCHEDULES[arm]])
    out["cost"]["ffn_params_measured"] = out["params"]["params_ffn"]
    out["cost"]["ffn_params_analytic_matches_measured"] = (
        out["cost"]["ffn_params_analytic"] == out["params"]["params_ffn"])
    path = os.path.join(HERE, f"results_{arm}_s{seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[2641] {arm}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
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

    deltas = {}
    for s in seeds:
        d_up, d_dn = bpc("U2", s), bpc("APYR", s)
        if bpc("U2", s) is not None and bpc("PYR", s) is not None:
            d_up = round(bpc("U2", s) - bpc("PYR", s), 4)
        if bpc("U2", s) is not None and d_dn is not None:
            d_dn = round(bpc("U2", s) - bpc("APYR", s), 4)
        deltas[s] = {"delta_up_U2_minus_PYR": d_up,
                     "delta_down_U2_minus_APYR": d_dn}
    seeds_with = [s for s in seeds if (("U2", s) in recs and ("PYR", s) in recs
                                       and ("APYR", s) in recs)]
    up_vals = [deltas[s]["delta_up_U2_minus_PYR"] for s in seeds_with]
    same_sign = bool(len(up_vals) >= 2 and
                     (all(v > 0 for v in up_vals) or all(v < 0 for v in up_vals)))

    def verdict(s):
        d = deltas[s]["delta_up_U2_minus_PYR"]
        if d is None:
            return None
        if d >= VARIANCE_BOUND:
            return "可分：PYR（前宽后窄）在等 FFN 预算下优于等宽"
        if d <= -VARIANCE_BOUND:
            return "可分（反向）：PYR 在等 FFN 预算下劣于等宽"
        return "不可排序（H0）：|Δ| < 0.014 ⇒ 逐层 FFN 分配在本尺度无独立增益"

    p5 = None
    if ("U4", 0) in recs:
        d_hist = round(recs[("U4", 0)]["eval"]["clean_bpc"] - HIST_DENSE_BPC, 4)
        d_2613 = round(recs[("U4", 0)]["eval"]["clean_bpc"]
                       - HIST_2613_BASELINE_S0, 4)
        p5 = {"U4_s0_clean_bpc": recs[("U4", 0)]["eval"]["clean_bpc"],
              "vs_hist_23d_dense_B_12k": d_hist,
              "vs_2613_baseline_s0": d_2613,
              "pass": bool(abs(d_hist) < 0.005 or abs(d_2613) < 0.005),
              "rule": "与任一历史 dense 13M 锚之差 < 0.005 ⇒ 同引擎同配方"}

    out = {
        "experiment": "2641-ffn-ratio-frontier-13m",
        "criterion": ("P1 Δ_up = bpc_U2 − bpc_PYR（正 = PYR 更好）；"
                      "P2 Δ_down = bpc_U2 − bpc_APYR；P4 两 seed 同号；"
                      "P5 等值性门；P6 活性（隐藏维审计）。方差带 0.014。"),
        "variance_bound": VARIANCE_BOUND,
        "steps": steps,
        "seed_spread_reference": {"same_config_seed_spread_2613":
                                  round(HIST_2613_BASELINE_S1
                                        - HIST_2613_BASELINE_S0, 4),
                                  "note": "同 config 跨 seed bpc 散布（参照，不替代 0.014）"},
        "arms": {f"{a}-s{s}": {"clean_bpc": bpc(a, s),
                               "sum_r": recs[(a, s)]["sum_r"],
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
        "P1_verdict_by_seed": {str(s): verdict(s) for s in seeds},
        "P4_same_sign_across_seeds": bool(same_sign),
        "P5_engine_equivalence_gate": p5,
        "P6_liveness": {f"{a}-s{s}": {"hidden_matches_schedule":
                                      recs[(a, s)]["params"]
                                      ["hidden_matches_schedule"],
                                      "ffn_params_analytic_matches_measured":
                                      recs[(a, s)]["cost"]
                                      ["ffn_params_analytic_matches_measured"]}
                        for (a, s) in sorted(recs)},
        "equal_ffn_budget_check": {
            "rule": "U2 / PYR / APYR 的 Σr、FFN 参数、每 token FFN MACs 必须逐位相等",
            "sum_r": {a: recs[(a, 0)]["sum_r"] for a in arms
                      if (a, 0) in recs},
            "ffn_macs": {a: recs[(a, 0)]["cost"]["macs_per_token"]["ffn"]
                         for a in arms if (a, 0) in recs},
            "params_ffn": {a: recs[(a, 0)]["params"]["params_ffn"]
                           for a in arms if (a, 0) in recs}},
        "source_files": [f"results_{a}_s{s}.json" for (a, s) in sorted(recs)],
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    eq = out["equal_ffn_budget_check"]
    vals = {}
    for a in ("U2", "PYR", "APYR"):
        if a in eq["ffn_macs"]:
            vals[a] = (eq["ffn_macs"][a], eq["params_ffn"][a], eq["sum_r"][a])
    out["equal_ffn_budget_check"]["pass"] = bool(
        len(set(vals.values())) <= 1 and len(vals) >= 2)
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(SCHEDULES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--no-train", action="store_true",
                    help="跳过训练，直接复用 runs/…/model.pt 评测")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--arms", default="U4,U2,PYR,APYR")
    ap.add_argument("--seeds", default="0,1")
    args = ap.parse_args()

    if args.aggregate:
        aggregate(args.steps, args.out_root, args.arms.split(","),
                  [int(s) for s in args.seeds.split(",") if s.strip() != ""])
        return
    if not args.arm:
        raise SystemExit("--arm 或 --aggregate 必给其一")
    global _ACTIVE_SCHEDULE
    _ACTIVE_SCHEDULE = SCHEDULES[args.arm]
    if not args.no_train:
        os.makedirs(args.out_root, exist_ok=True)
        info = train_arm(args.arm, args.seed, args.steps, args.batch_size,
                         args.lr, args.out_root)
        rd = run_dir(args.arm, args.seed, args.steps, args.out_root)
        with open(os.path.join(rd, "train_wall.json"), "w") as f:
            json.dump(info, f)
    evaluate(args.arm, args.seed, args.steps, args.out_root)


if __name__ == "__main__":
    main()
