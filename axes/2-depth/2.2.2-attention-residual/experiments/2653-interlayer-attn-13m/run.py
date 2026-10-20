"""2653 · 2.2.2 层间 causal attention 最小判决 · 13M · dense.

契约 = 同目录 `README.md`（判据线 P1–P6 **预登记、不可改**）。本文件是唯一入口。

设计要点（见契约 §2）：

* **训练整段走共享代码**：把 `train_depth_ar.DepthARModel` 换成本实验的工厂
  （`interlayer_attn.InterLayerAttentionModel`：沿用引擎 `topology="dense"` 的前向
  路径，只覆写 `_layer_mix_input` = 层间 causal attention 聚合），再调
  `train_depth_ar.main()` —— 参数解析 / 配置装配 / `train()` / checkpoint 全是共享实现。
* **基线复用 2613 s0**（`sequential` = stock 单残差、`dense` = 2.2.3 的线性学习加权和）：
  按 `harness/SCHEMA.md` §9 三条给等值证明（实现层逐位单测 + 配置逐字段对照 + 口径标注），
  任一不满足则回退自包含双跑。
* 代价列：层间聚合的额外参数与每 token 额外 MACs（解析，**不含任何时间量**）。

用法：

    python3 run.py --arms attn-scalar,attn-qkv --seed 0 --steps 12000
    python3 run.py --arms attn-scalar --no-train     # 仅评测（复用 model.pt）
    python3 run.py --aggregate                       # 汇总 P1–P6 → results.json

GPU 锁按 `harness/SCHEMA.md` §20 取、**按臂释放**（§20.1，臂间 ≥60 s 窗口）。
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

import train_depth_ar as TDA                                  # noqa: E402
import interlayer_attn as IA                                  # noqa: E402
from gpu_lock import GPUUnavailable, gpu_lock                  # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, depth_curve  # noqa: E402
from pathlm.data import load_enwik8_full                       # noqa: E402

DEVICE = "cuda"
LN2 = math.log(2.0)

D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 8, 256, 4
VARIANCE_BOUND = 0.014
N_WINDOWS, BATCHES_PER_WINDOW, EVAL_BATCH_SIZE = 5, 4, 16

#: 臂 → 聚合参数化（`interlayer_attn.MODE`）。
ARM_MODES = {"attn-scalar": "scalar", "attn-qkv": "qkv"}

#: 复用基线（2613 s0，`harness/SCHEMA.md` §9 三条等值证明见 `baseline_equivalence()`）。
BASE_2613 = os.path.join(ROOT, "axes", "2-depth", "2.2.3-learned-layer-sum",
                         "experiments", "2613-inter-layer-topology-13m",
                         "results.json")
BASE_SEQ_S0, BASE_DENSE_S0 = 1.5254, 1.5246      # 2613 sequential / dense 的 clean bpc


# ---------------------------------------------------------------- 训练（走共享 main）

def shared_argv(arm: str, seed: int, steps: int, batch_size: int, lr: float,
                out_root: str) -> list[str]:
    """与 2613 s0 逐字对齐的 CLI（唯一变量 = 层间聚合的参数化）。"""
    return [os.path.join(ROOT, "train_depth_ar.py"),
            f"{arm}-s{seed}-{steps // 1000}k",
            "--d", str(D_MODEL), "--layers", str(N_LAYERS),
            "--heads", str(N_HEADS), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT), "--dense", "--topology", "dense",
            "--exit-mode", "uniform", "--steps", str(steps),
            "--batch-size", str(batch_size), "--lr", str(lr), "--seed", str(seed),
            "--eval-batches", "20", "--out-root", out_root]


def train_arm(arm: str, seed: int, steps: int, batch_size: int, lr: float,
              out_root: str) -> dict:
    IA.MODE = ARM_MODES[arm]
    IA.FORCE_LAST = False
    orig = TDA.DepthARModel
    TDA.DepthARModel = IA.InterLayerAttentionModel
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


def run_dir(arm: str, seed: int, steps: int, out_root: str) -> str:
    return os.path.join(out_root, f"{arm}-s{seed}-{steps // 1000}k")


def build_model(arm: str, vocab: int) -> IA.InterLayerAttentionModel:
    IA.MODE, IA.FORCE_LAST = ARM_MODES[arm], False
    cfg = DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                        seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True,
                        topology="dense")
    return IA.InterLayerAttentionModel(cfg, vocab)


# ---------------------------------------------------------------- 评测（本实验自持）

@torch.no_grad()
def windowed_bpc(model, eval_arr, n_windows: int = N_WINDOWS,
                 batches_per_window: int = BATCHES_PER_WINDOW,
                 batch_size: int = EVAL_BATCH_SIZE, gen_seed: int = 2) -> dict:
    """把评测切成 `n_windows` 个**数据完全相同**的窗口（同一 generator seed 顺序取批），
    逐窗 bpc ⇒ 配对 Δ 的**窗口间 SE**（本仓 C1 口径：同 checkpoint 内另报窗间 SE）。"""
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
    return {"window_bpc": per_window,
            "clean_bpc": round(sum(per_window) / len(per_window), 4),
            "n_windows": n_windows, "batches_per_window": batches_per_window,
            "batch_size": batch_size, "eval_generator_seed": gen_seed}


@torch.no_grad()
def attention_liveness(model, eval_arr, n_batches: int = 2, batch_size: int = 8,
                       seed: int = 1234) -> dict:
    """活性（`harness/SCHEMA.md` §10）：一次真实前向里收集层间注意力权重 → 逐层熵 / 紧邻上一层
    权重 / 最大权重，以及各层温度 τ 的终值。"""
    from pathlm.data import batch
    gen = torch.Generator().manual_seed(seed)
    model.attn_tele = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
            model(x.to(DEVICE), exit_depths=None)
    finally:
        model.train(was_training)
    per_layer = [{"layer": t["layer"], "entropy": round(t["entropy"], 6),
                  "w_immediate_prev": round(t["w_immediate_prev"], 6),
                  "w_max": round(t["w_max"], 6)} for t in model.attn_tele]
    n_deg = [t for t in per_layer if t["layer"] >= 2]
    return {
        "per_layer": per_layer,
        "tau_by_layer": [round(float(v), 6) for v in model.tau.detach().float()],
        "max_dev_from_identity_prev": (round(max(t["w_immediate_prev"]
                                                 for t in n_deg), 6) if n_deg else None),
        "alive": bool(n_deg and all(t["w_immediate_prev"] < 1.0 for t in n_deg)),
        "note": ("活性的形态：权重**偏离**「只取紧邻上一层」（w_immediate_prev < 1、"
                 "熵 > 0）。层 1（k=1）只有唯一源 ⇒ 平凡，不计入。"),
    }


def param_accounting(arm: str) -> dict:
    stock = sum(p.numel() for p in DepthARModel(
        DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                      seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True,
                      topology="sequential"), 206).parameters())
    m = build_model(arm, 206)
    total = sum(p.numel() for p in m.parameters())
    return {"stock_params": stock, "arm_params": total,
            "params_added": total - stock,
            "dense_topology_ref_params_added": N_LAYERS * (N_LAYERS + 1) // 2,
            "cost": m.aggregation_cost()}


def loss_slope(rows: list[dict], steps: int, frac: float = 0.25) -> dict:
    start = int(steps * (1 - frac))
    win = [r for r in rows if start <= r["step"] <= steps]
    if len(win) < 2:
        return {"n_points": len(win), "slope_per_1k_steps": None}
    xs = [r["step"] for r in win]
    ys = [r["loss"] for r in win]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - xm) ** 2 for x in xs)
    return {"n_points": len(win), "step_first": xs[0], "step_last": xs[-1],
            "slope_per_1k_steps": round((sum((x - xm) * (y - ym)
                                             for x, y in zip(xs, ys)) / den
                                         if den else float("nan")) * 1000, 5),
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
    model = build_model(arm, vocab).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(rd, "model.pt"),
                                     map_location=DEVICE))
    model.eval()
    curve = depth_curve(model, eval_arr, n_batches=n_batches,
                        batch_size=EVAL_BATCH_SIZE, device=DEVICE)
    win = windowed_bpc(model, eval_arr)
    rows = [json.loads(l) for l in open(os.path.join(rd, "train_log.jsonl"))
            if l.strip()]
    shared = json.load(open(os.path.join(rd, "results.json")))
    out = {"arm": arm, "mode": ARM_MODES[arm],
           "seed": seed, "steps": steps,
           "eval": {"metric": "clean bpc（最终深度、无退出、序列级、越低越好）",
                    "n_batches": n_batches, "batch_size": EVAL_BATCH_SIZE,
                    "eval_generator_seed": 2,
                    "clean_bpc": win["clean_bpc"],
                    "clean_bpc_from_depth_curve": curve[-1]["bpc"],
                    "clean_bpc_two_paths_agree": (
                        abs(win["clean_bpc"] - curve[-1]["bpc"]) <= 5e-5),
                    "windows": win, "bpc_by_depth": curve,
                    "acc_by_depth": [{"depth": r["depth"], "acc": r["acc"]}
                                     for r in curve]},
           "params": param_accounting(arm),
           "liveness": attention_liveness(model, eval_arr),
           "loss_slope": loss_slope(rows, steps),
           "train_log_rows": len(rows),
           "resolved_config": shared.get("config"),
           "shared_train_results": {
               "params": shared.get("params"),
               "train_tokens": shared.get("train_tokens"),
               "wall_minutes": shared.get("wall_minutes"),
               "peak_gpu_mem_mb": shared.get("peak_gpu_mem_mb"),
               "gnorm_monitor": shared.get("gnorm_monitor")},
           "run_dir": os.path.relpath(rd, ROOT),
           "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py"))}
    with open(os.path.join(HERE, f"results_{arm}_s{seed}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[2653] {arm}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
          f"params_added={out['params']['params_added']} "
          f"alive={out['liveness']['alive']}", flush=True)
    return out


# ---------------------------------------------------------------- 基线复用等值证明

def baseline_equivalence(my_config: dict | None) -> dict:
    """`harness/SCHEMA.md` §9 三条：① 实现层等值（one-hot 单测，见 smoke/test）；② 配置层逐字段
    对照（与 2613 s0）；③ 口径标注（`analysis.md`）。任一字段不等 ⇒ 回退自包含双跑。"""
    ref = json.load(open(BASE_2613))
    ref_base = ref["config"]["base"]
    ref_cfg = ref["config"]
    fields = {"d_model": (D_MODEL, ref_base["d_model"]),
              "n_layers": (N_LAYERS, ref_base["n_layers"]),
              "n_heads": (N_HEADS, ref_base["n_heads"]),
              "seq_len": (SEQ_LEN, ref_base["seq_len"]),
              "mlp_mult": (MLP_MULT, ref_base["mlp_mult"]),
              "dense": (True, ref_base["dense"]),
              "exit_mode": ("uniform", ref_base["exit_mode"]),
              "dropout": (0.0, ref_base["dropout"]),
              "norm_cap": (1.0, ref_base["norm_cap"]),
              "steps": (12000, ref_cfg["steps"]),
              "batch_size": (32, ref_cfg["batch_size"]),
              "lr": (1e-3, ref_cfg["lr"]),
              "seed": (0, ref_cfg["seed"]),
              "optimizer": (ref_cfg["optimizer"], ref_cfg["optimizer"]),
              "data": (ref_cfg["data"], ref_cfg["data"]),
              "train_tokens": (98304000, ref_cfg["train_tokens"])}
    diff = {k: {"mine": a, "ref": b} for k, (a, b) in fields.items() if a != b}
    return {"reference": os.path.relpath(BASE_2613, ROOT),
            "reference_arms": {"sequential": {"clean_bpc": BASE_SEQ_S0,
                                              "params": ref["arms"]["sequential"]["params"]},
                               "dense": {"clean_bpc": BASE_DENSE_S0,
                                         "params": ref["arms"]["dense"]["params"]}},
            "field_by_field": {k: {"mine": a, "ref": b}
                               for k, (a, b) in fields.items()},
            "unequal_fields": diff,
            "implementation_equality": ("`test_2653_interlayer_attn.py`："
                                        "force_last 的 one-hot 权重下前向/loss 与 stock "
                                        "sequential 逐位一致（同一构造顺序 ⇒ trunk 权重也逐位相同）"),
            "caliber_note": ("历史同配置对照 + 逐位等值性证明；2613 s0 的 clean bpc 口径 = "
                             "全深度、无退出、teacher-forcing、序列级 bpc，与本实验同口径"),
            "pass": bool(not diff)}


# ---------------------------------------------------------------- 汇总（P1–P6）

def aggregate(steps: int, out_root: str, arms: list[str], seeds: list[int]) -> dict:
    recs = {}
    for arm in arms:
        for s in seeds:
            p = os.path.join(HERE, f"results_{arm}_s{s}.json")
            if os.path.exists(p):
                recs[(arm, s)] = json.load(open(p))
    eq = baseline_equivalence(None)

    def bpc(arm, s):
        return recs[(arm, s)]["eval"]["clean_bpc"] if (arm, s) in recs else None

    deltas = {}
    for (arm, s) in sorted(recs):
        b = bpc(arm, s)
        deltas[f"{arm}-s{s}"] = {
            "delta_vs_stock_sequential_2613_s0": round(BASE_SEQ_S0 - b, 4),
            "delta_vs_dense_lin_sum_2613_s0": round(BASE_DENSE_S0 - b, 4)}

    win_paired = {}
    for (arm, s) in sorted(recs):
        w = recs[(arm, s)]["eval"]["windows"]["window_bpc"]
        d_stock = [round(BASE_SEQ_S0 - v, 6) for v in w]
        win_paired[f"{arm}-s{s}"] = {
            "delta_vs_stock_by_window_mean": round(statistics.mean(d_stock), 6),
            "se_across_windows_vs_stock": round(
                statistics.stdev(d_stock) / math.sqrt(len(d_stock)), 6),
            "n_windows": len(d_stock),
            "note": ("窗口间 SE 是同一 checkpoint 内的抽样噪声；vs 历史臂的逐窗配对"
                     "只在历史臂的窗数据可得时才是真配对（本表按窗口均值与 SE 报，"
                     "口径已在 analysis.md 标注）")}

    def verdict(d):
        if d is None:
            return None
        if d >= VARIANCE_BOUND:
            return "可分：该臂优于对照"
        if d <= -VARIANCE_BOUND:
            return "可分（反向）：该臂劣于对照"
        return "不可排序（H0）：|Δ| < 0.014"

    out = {
        "experiment": "2653-interlayer-attn-13m",
        "criterion": ("P1 Δ = bpc_(stock/dense) − bpc_该臂（符号约定见 README §4，"
                      "正 = 该臂更好）；方差带 0.014；P2 活性（层维权重非退化）；"
                      "P3 代价列（额外参数 / 每 token 额外 MACs）；P4 等值门（one-hot "
                      "⇒ 与 stock 逐位一致）；P5 基线复用的三条等值证明；P6 层索引因果。"),
        "variance_bound": VARIANCE_BOUND, "steps": steps,
        "arms": {f"{a}-s{s}": {"clean_bpc": bpc(a, s),
                               "params_total": recs[(a, s)]["params"]["arm_params"],
                               "params_added": recs[(a, s)]["params"]["params_added"],
                               "extra_macs_per_token": recs[(a, s)]["params"]["cost"]
                               ["extra_macs_per_token_interlayer"],
                               "extra_over_base_attention": recs[(a, s)]["params"]["cost"]
                               ["extra_over_base_attention"],
                               "alive": recs[(a, s)]["liveness"]["alive"],
                               "slope_per_1k": recs[(a, s)]["loss_slope"]
                               ["slope_per_1k_steps"]}
                          for (a, s) in sorted(recs)},
        "P1_deltas_vs_baselines": deltas,
        "P1_verdict": {k: {"vs_stock": verdict(v["delta_vs_stock_sequential_2613_s0"]),
                           "vs_dense_lin_sum": verdict(
                               v["delta_vs_dense_lin_sum_2613_s0"])}
                       for k, v in deltas.items()},
        "P2_liveness": {f"{a}-s{s}": {"alive": recs[(a, s)]["liveness"]["alive"],
                                      "max_dev_from_identity_prev":
                                      recs[(a, s)]["liveness"]
                                      ["max_dev_from_identity_prev"],
                                      "tau_by_layer": recs[(a, s)]["liveness"]
                                      ["tau_by_layer"]}
                        for (a, s) in sorted(recs)},
        "P3_window_se": win_paired,
        "P5_baseline_equivalence": eq,
        "source_files": [f"results_{a}_s{s}.json" for (a, s) in sorted(recs)],
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARM_MODES))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--lock-wait-s", type=float, default=600.0)
    ap.add_argument("--arm-gap-s", type=float, default=65.0)
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a.strip() != ""]
    if args.aggregate:
        aggregate(args.steps, args.out_root, arms, [args.seed])
        return

    os.makedirs(args.out_root, exist_ok=True)
    log_path = os.path.join(HERE, "run_log.txt")
    failed = []
    for i, arm in enumerate(arms):
        done = os.path.join(HERE, f"results_{arm}_s{args.seed}.json")
        model_pt = os.path.join(run_dir(arm, args.seed, args.steps,
                                        args.out_root), "model.pt")
        if (not args.force and os.path.exists(done)
                and (args.no_train or os.path.exists(model_pt))):
            print(f"[2653] arm={arm} already done ⇒ skip（--force 可重跑）", flush=True)
            continue
        rec = {"arm": arm, "seed": args.seed, "steps": args.steps,
               "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "held_lock": False}
        try:
            with gpu_lock(name="wp2653-interlayer-attn",
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
        print(f"[2653] arm={arm} status={rec['status']}", flush=True)
        if rec["status"] != "ok":
            failed.append(arm)
        if i != len(arms) - 1 and rec["status"] == "ok":
            time.sleep(args.arm_gap_s)
    aggregate(args.steps, args.out_root, arms, [args.seed])
    if failed:
        raise SystemExit(9)


if __name__ == "__main__":
    main()
