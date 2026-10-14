"""2652 · 残差形式最小判决 · 13M · depth-ar/dense（唯一入口）。

契约 = 同目录 `README.md`（判据线 P1–P6 **预登记、不可改**）。

设计要点（为什么这样接线）：

* **训练整段走共享代码**：本文件把 `train_depth_ar` 模块里的全局名 `DepthARModel`
  换成一个**工厂**（先按 stock 路径建完整模型，再把 `model.blocks` 重新接线成目标
  残差形式），然后调用 `train_depth_ar.main()` —— 参数解析、配置装配、`train()`
  循环、checkpoint、`results.json` 全部是**共享实现**，本文件不改共享代码。
* **臂间唯一的差别** = 残差形式（`residual_block.py`）；embedding / attention / LN /
  FFN 的初始化来自**同一次随机抽样**（同 seed、同构造顺序）⇒ 不是「两个不同初始化
  的模型」在比。
* **`base` 臂 = 引擎 stock `Block` 原样**（不包裹）⇒ 它就是引擎 stock 对照。
* 残差形式只实现在**实验目录内**（不改 `pathlm/**`、不改 `train_depth_ar.py`）。

用法：

    python3 smoke.py                                    # CPU 等值门（init 逐位一致）
    python3 run.py --smoke-gate                         # 同上（driver 内的入口）
    python3 run.py --smoke-steps 20                     # 20 步冒烟（四臂，一次持锁）
    python3 run.py --arm base --seed 0                  # 单臂：取锁 → 训练 → 评测 → 放锁
    python3 run.py --arm base --seed 0 --no-train       # 仅评测（复用 runs/…/model.pt）
    python3 run.py --aggregate                          # 汇总 P1–P6 → results.json
    python3 run.py 2>&1 | tee run_log.txt               # 全臂（按臂取/放 GPU 锁）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time

import torch

from common import (ARM_FORMS, DEFAULTS, EXEC_ORDER, HIST_23D_DENSE_B_12K,
                    HIST_2613_BASELINE_S0, HIST_2613_BASELINE_S1, HERE, ROOT,
                    VARIANCE_BOUND, make_cfg, param_count, run_name, shared_argv)
from gpu_lock import GPUUnavailable, gpu_lock
from residual_block import alpha_by_layer, build_model, gate_values

sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                   # noqa: E402
from pathlm.data import batch, load_enwik8_full                # noqa: E402
from pathlm.depth_ar import depth_curve                        # noqa: E402

DEVICE = "cuda"
LOCK_NAME = "wp2652-2.2.1-residual-form"


# ---------------------------------------------------------------- 训练（走共享 main）

def _factory(form: str):
    """替换 `train_depth_ar.DepthARModel` 的工厂：共享 main() 调它建模型。"""
    def factory(cfg, vocab_size):
        return build_model(cfg, vocab_size, form)
    return factory


def _argv_template(form: str, seed: int, steps: int, batch_size: int,
                   lr: float) -> list[str]:
    """共享 CLI 的逐字模板（run_name 归一成占位符 ⇒ 臂间可比对）。"""
    argv = shared_argv("<RUN>", seed, steps, batch_size, lr, "<OUT>")
    return argv


def train_arm(form: str, seed: int, steps: int, batch_size: int, lr: float,
              out_root: str, run: str) -> dict:
    # train() 以 append 模式写日志 ⇒ 重训前先清掉旧日志，否则同一 run 的行会叠加
    rd = os.path.join(out_root, run)
    for fn in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(rd, fn)
        if os.path.exists(p):
            os.remove(p)
    orig = TDA.DepthARModel
    TDA.DepthARModel = _factory(form)
    argv_backup = list(sys.argv)
    t0 = time.time()
    try:
        sys.argv = shared_argv(run, seed, steps, batch_size, lr, out_root)
        TDA.main()                       # ← 共享实现：解析 / 配置 / train() / 落盘
        wall = time.time() - t0
    finally:
        TDA.DepthARModel = orig
        sys.argv = argv_backup
    return {"train_wall_s": round(wall, 1)}


# ---------------------------------------------------------------- 评测（本实验自持）

def _run_dir(out_root: str, run: str) -> str:
    return os.path.join(out_root, run)


def _loss_slope(rows: list[dict], steps: int, frac: float = 0.25) -> dict:
    """同方法 loss 斜率（`SCHEMA.md` §16）：末 frac 窗、最小二乘、×1000。"""
    start = int(steps * (1 - frac))
    win = [r for r in rows if start <= r["step"] <= steps
           and math.isfinite(r["loss"])]
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


def _stream_rms_by_layer(model, eval_arr, n_batches: int = 2,
                         batch_size: int = 8, seed: int = 1234) -> list[float]:
    """**残差流 RMS 逐层**：在一次真实前向里 forward-hook 每块输出（`out[0]`），
    报 `sqrt(mean(x²))` 的逐批均值。`base` 是未归一温流、`postln` 是每子层后被 LN
    归一的流（RMS≈1）——这条曲线就是「归一化后置」的活性量，也是 `gated`/`scaled`
    的残差流诊断（`SCHEMA.md` §10）。"""
    acc: list[list[float]] = [[] for _ in model.blocks]
    handles = []

    def make_hook(l):
        def hook(_mod, _inp, out):
            y = out[0].detach().float()
            acc[l].append(float(y.pow(2).mean().sqrt()))
        return hook

    for l, blk in enumerate(model.blocks):
        handles.append(blk.register_forward_hook(make_hook(l)))
    gen = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(n_batches):
                x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
                model(x.to(DEVICE), exit_depths=None)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    return [round(statistics.mean(v), 4) if v else None for v in acc]


def _gate_grads(model, train_arr, batch_size: int = 8, seed: int = 1234) -> dict | None:
    """`gated` 的门控梯度（活动性证据：门真的拿得到梯度、且非零）。"""
    gates = [b for b in model.blocks if getattr(b, "form", None) == "gated"]
    if not gates:
        return None
    gen = torch.Generator().manual_seed(seed)
    x, _ = batch(train_arr, batch_size, model.cfg.seq_len, gen)
    was_training = model.training
    model.train()
    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad():
            loss, _ = model(x.to(DEVICE), exit_depths=None)
            loss.backward()
        ag = [b.attn_gate.grad for b in gates]
        mg = [b.mlp_gate.grad for b in gates]
        return {
            "attn_gate_grad_by_layer": [round(float(g), 8) for g in ag],
            "mlp_gate_grad_by_layer": [round(float(g), 8) for g in mg],
            "absmax_grad": round(max(abs(float(g)) for g in ag + mg), 8),
            "sublayer_keys": ["attn_gate", "mlp_gate"],
        }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


def _divergence(rows: list[dict]) -> dict:
    bad = [r["step"] for r in rows if not math.isfinite(r["loss"])]
    huge = [r["step"] for r in rows if r.get("gnorm") is not None
            and r["gnorm"] > 1e3]
    return {"first_nonfinite_loss_step": bad[0] if bad else None,
            "n_nonfinite_loss_rows": len(bad),
            "first_gnorm_over_1e3_step": huge[0] if huge else None,
            "diverged": bool(bad or huge)}


@torch.no_grad()
def evaluate(form: str, seed: int, steps: int, out_root: str, run: str,
             train_wall_s: float | None = None,
             argv_template: list[str] | None = None) -> dict:
    rd = _run_dir(out_root, run)
    train_arr, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DEFAULTS["data"])
    cfg = make_cfg(vocab)
    model = build_model(cfg, vocab, form).to(DEVICE)
    state = torch.load(os.path.join(rd, "model.pt"), map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    curve = depth_curve(model, eval_arr, n_batches=DEFAULTS["eval_batches"],
                        batch_size=DEFAULTS["eval_batch_size"], device=DEVICE)
    rows = [json.loads(l) for l in open(os.path.join(rd, "train_log.jsonl"))
            if l.strip()]
    shared = json.load(open(os.path.join(rd, "results.json")))
    n_layers = cfg.n_layers

    n_total = param_count(model)
    if form == "base":
        liveness = {"form": "base", "is_control": True, "params_added": 0,
                    "alive": None,
                    "note": "对照臂本身 = 引擎 stock pre-LN 单残差（无干预）"}
    elif form == "postln":
        liveness = {"form": "postln", "params_added": 0, "alive": True,
                    "normalization": "每个子层残差相加之后再 LN（归一化后置）",
                    "note": ("归一化后置：残差流在每子层后被 LN 归一 ⇒ "
                             "`stream_rms_by_layer` 应≈1 且与 base 的未归一温流不同")}
    elif form == "scaled":
        alphas = alpha_by_layer(form, n_layers)
        liveness = {"form": "scaled", "params_added": 0, "alive": True,
                    "alpha_by_layer": [round(a, 6) for a in alphas],
                    "alpha_formula": "α_l = 1/√(l+1)（l 从 0 起）",
                    "note": ("固定缩放：l=0 层 α=1.0（该层与 base 逐位一致），"
                             "l≥1 层 α<1 ⇒ 干预生效；无可学习参数")}
    else:
        gv = gate_values(model)
        g = _gate_grads(model, train_arr)
        dev = max(abs(v - 1.0) for v in gv["attn_by_layer"] + gv["mlp_by_layer"])
        allv = gv["attn_by_layer"] + gv["mlp_by_layer"]
        liveness = {
            "form": "gated", "params_added": 2 * n_layers,
            "params_added_frac": round(2 * n_layers / n_total, 9),
            "gates": {k: [round(v, 6) for v in vs] for k, vs in gv.items()},
            "init_value": 1.0,
            "gate_dev_from_init_absmax": round(dev, 6),
            "w_fro": round(math.sqrt(sum(v * v for v in allv)), 6),
            "w_absmax": round(max(abs(v) for v in allv), 6),
            "nonzero_frac": 1.0,
            "gate_grad": g,
            "alive": bool(dev > 0 and g["absmax_grad"] > 0),
            "note": ("可学习逐层标量门 init=1.0 ⇒ init 处与 base 逐位一致"
                     "（等值门见 smoke.py）；活性的两条腿 = 终值偏离 1 且梯度非零")}
    rms = _stream_rms_by_layer(model, eval_arr)
    liveness["stream_rms_by_layer"] = rms

    gnorm_win = [r["gnorm"] for r in rows
                 if r["step"] >= steps * 0.75 and r.get("gnorm") is not None]
    out = {
        "arm": form, "seed": seed, "steps": steps, "run_dir": os.path.relpath(rd, ROOT),
        "eval": {
            "metric": "clean bpc（最终深度、无退出、序列级、越低越好）",
            "protocol": ("engine depth_curve(n_batches=20, batch_size=16, "
                         "固定 generator seed 2)"),
            "clean_bpc": curve[-1]["bpc"],
            "bpc_by_depth": [{"depth": r["depth"], "bpc": r["bpc"], "acc": r["acc"]}
                             for r in curve],
        },
        "params": {"total": n_total, "form_added": liveness["params_added"],
                   "gates": 2 * n_layers if form == "gated" else 0},
        "residual_form_liveness": liveness,
        "stability": {
            "loss_slope": _loss_slope(rows, steps),
            "gnorm_mean_last_window": (round(statistics.mean(gnorm_win), 4)
                                       if gnorm_win else None),
            "gnorm_max": shared.get("gnorm_monitor", {}).get("gnorm_max"),
            "grad_skips": shared.get("gnorm_monitor", {}).get("grad_skips"),
            "divergence": _divergence(rows),
        },
        "shared_train_results": {
            "params": shared.get("params"),
            "train_tokens": shared.get("train_tokens"),
            "wall_minutes": shared.get("wall_minutes"),
            "peak_gpu_mem_mb": shared.get("peak_gpu_mem_mb"),
            "config": shared.get("config"),
            "topology_liveness": shared.get("topology_liveness"),
        },
        "argv_template": (argv_template if argv_template is not None
                          else _argv_template(form, seed, steps,
                                              DEFAULTS["batch_size"],
                                              DEFAULTS["lr"])),
        "train_wall_s": train_wall_s,
        "code_hash_run_py": _sha256(os.path.join(HERE, "run.py")),
        "code_hash_residual_block": _sha256(os.path.join(HERE, "residual_block.py")),
    }
    with open(os.path.join(HERE, f"results_{form}_s{seed}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[2652] {form}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
          f"params={n_total} form_added={liveness['params_added']} "
          f"alive={liveness.get('alive')}", flush=True)
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- 汇总（P1–P6）

def aggregate(steps: int, seeds: list[int], out_root: str) -> dict:
    recs: dict[tuple[str, int], dict] = {}
    for form in ARM_FORMS:
        for s in seeds:
            p = os.path.join(HERE, f"results_{form}_s{s}.json")
            if os.path.exists(p):
                recs[(form, s)] = json.load(open(p))

    def bpc(form, s):
        return recs[(form, s)]["eval"]["clean_bpc"] if (form, s) in recs else None

    def dcurve(form, s):
        if (form, s) not in recs:
            return None
        return {r["depth"]: r["bpc"] for r in recs[(form, s)]["eval"]["bpc_by_depth"]}

    deltas: dict[str, dict] = {}
    for form in ARM_FORMS:
        if form == "base":
            continue
        per_seed = {}
        for s in seeds:
            b, f = bpc("base", s), bpc(form, s)
            if b is None or f is None:
                per_seed[str(s)] = None
                continue
            db = dcurve("base", s)
            df = dcurve(form, s)
            per_seed[str(s)] = {
                "delta_base_minus_form": round(b - f, 4),
                "delta_by_depth": {str(k): round(db[k] - df[k], 4)
                                   for k in sorted(db)},
                "sign": "form_better" if b - f > 0 else
                        ("form_worse" if b - f < 0 else "tie"),
            }
        deltas[form] = per_seed

    def verdict(d):
        if d is None:
            return None
        if d >= VARIANCE_BOUND:
            return "可分：该形式优于 base"
        if d <= -VARIANCE_BOUND:
            return "可分（反向）：该形式劣于 base"
        return "不可排序（H0）：|Δ| < 0.014 ⇒ 在本尺度不是独立有增益的维度"

    verdicts = {form: {s: verdict(deltas[form][s]["delta_base_minus_form"]
                                  if deltas[form][s] else None)
                       for s in map(str, seeds)} for form in deltas}

    # 逐深度：中间深度（1..L-1）是否越带
    depth_band = {}
    for form in deltas:
        for s in map(str, seeds):
            dd = deltas[form][s]
            if not dd:
                continue
            mid = {k: v for k, v in dd["delta_by_depth"].items()
                   if 1 <= int(k) <= 7}
            depth_band.setdefault(form, {})[s] = {
                "delta_by_depth": dd["delta_by_depth"],
                "mid_depth_absmax": round(max(abs(v) for v in mid.values()), 4),
                "mid_depth_exceeds_band": bool(
                    max(abs(v) for v in mid.values()) >= VARIANCE_BOUND),
            }

    # 等值性门（P5）：base-s0 对历史同引擎 dense 13M 锚
    p5 = None
    if ("base", 0) in recs:
        b = recs[("base", 0)]["eval"]["clean_bpc"]
        d_hist = round(b - HIST_23D_DENSE_B_12K, 4)
        d_2613 = round(b - HIST_2613_BASELINE_S0, 4)
        p5 = {"base_s0_clean_bpc": b,
              "vs_hist_23d_dense_B_12k": d_hist,
              "vs_2613_baseline_s0": d_2613,
              "pass": bool(abs(d_hist) < 0.005 or abs(d_2613) < 0.005),
              "rule": "与任一历史 dense 13M 锚之差 < 0.005 ⇒ 同引擎同配方"}

    # 单因素机械核对：四臂的共享 CLI 模板逐字相同
    tmpls = {f"{a}-s{s}": json.dumps(recs[(a, s)]["argv_template"])
             for (a, s) in recs}
    single_factor = {"all_arms_same_shared_cli": len(set(tmpls.values())) <= 1,
                     "n_arms": len(recs)}

    out = {
        "experiment": "2652-residual-form-13m",
        "criterion": ("Δ = bpc_base − bpc_form（正 = 该形式更好）；判据带 = "
                      f"{VARIANCE_BOUND} bpc（配对 Δ、≥5 窗口）。"
                      "P1 clean bpc / P2 逐深度 bpc 曲线 / P3 稳定性 / "
                      "P4 活性 / P5 引擎等值门 / P6 单因素机械核对"),
        "variance_bound": VARIANCE_BOUND,
        "steps": steps,
        "seed_spread_reference": {
            "same_config_seed_spread_2613": round(HIST_2613_BASELINE_S1
                                                  - HIST_2613_BASELINE_S0, 4),
            "note": "同 config 跨 seed bpc 散布（对照参照，不替代 0.014 预登记门槛）"},
        "arms": {f"{a}-s{s}": {
            "clean_bpc": bpc(a, s),
            "params_total": recs[(a, s)]["params"]["total"],
            "form_added_params": recs[(a, s)]["params"]["form_added"],
            "slope_per_1k": recs[(a, s)]["stability"]["loss_slope"]
            ["slope_per_1k_steps"],
            "gnorm_mean_last_window": recs[(a, s)]["stability"]
            ["gnorm_mean_last_window"],
            "gnorm_max": recs[(a, s)]["stability"]["gnorm_max"],
            "diverged": recs[(a, s)]["stability"]["divergence"]["diverged"],
            "alive": recs[(a, s)]["residual_form_liveness"].get("alive"),
        } for (a, s) in sorted(recs)},
        "deltas": deltas,
        "verdict_by_form": verdicts,
        "depth_readability": depth_band,
        "P5_engine_equivalence_gate": p5,
        "single_factor_check": single_factor,
        "source_files": [f"results_{a}_s{s}.json" for (a, s) in sorted(recs)],
        "code_hash_run_py": _sha256(os.path.join(HERE, "run.py")),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in
                      ("arms", "deltas", "verdict_by_form", "P5_engine_equivalence_gate",
                       "single_factor_check")}, indent=2, ensure_ascii=False),
          flush=True)
    return out


# ---------------------------------------------------------------- 冒烟（20 步）

def smoke_steps(steps: int, seed: int, arms: list[str] | None = None,
                forms: tuple[str, ...] = ARM_FORMS) -> dict:
    """各臂跑 `steps` 步（一次持锁，整程 <10 min ⇒ 允许，SCHEMA §20.1），
    量实测步时并确认每臂都能跑通。"""
    out_root = os.path.join(HERE, "smoke")
    res = {}
    for form in (forms if arms is None else [a for a in ARM_FORMS if a in arms]):
        run = f"smoke-{form}-s{seed}"
        t = train_arm(form, seed, steps, DEFAULTS["batch_size"], DEFAULTS["lr"],
                      out_root, run)
        rd = _run_dir(out_root, run)
        rows = [json.loads(l) for l in open(os.path.join(rd, "train_log.jsonl"))
                if l.strip()]
        last = rows[-1] if rows else {}
        res[form] = {"wall_s": t["train_wall_s"],
                     "minutes_logged": last.get("min"),
                     "last_loss": last.get("loss"),
                     "last_gnorm": last.get("gnorm"),
                     "finite": bool(last.get("loss") is not None
                                    and math.isfinite(last["loss"]))}
        print(f"[2652 smoke] {form}: wall={t['train_wall_s']}s "
              f"loss={last.get('loss')} gnorm={last.get('gnorm')}", flush=True)
    with open(os.path.join(HERE, "smoke", "smoke_summary.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    return res


# ---------------------------------------------------------------- CLI

def _train_and_eval(form: str, seed: int, steps: int, batch_size: int, lr: float,
                    out_root: str, force: bool) -> dict:
    run = run_name(form, seed, steps)
    rd = _run_dir(out_root, run)
    ckpt = os.path.join(rd, "model.pt")
    train_wall = None
    if force or not os.path.exists(ckpt):
        t = train_arm(form, seed, steps, batch_size, lr, out_root, run)
        train_wall = t["train_wall_s"]
    else:
        print(f"[2652] {run}: 复用已有 checkpoint（--force 可强制重训）", flush=True)
    return evaluate(form, seed, steps, out_root, run, train_wall_s=train_wall,
                    argv_template=_argv_template(form, seed, steps, batch_size, lr))


def main():
    # 共享 `train_depth_ar.main()` 用**仓库根相对路径**读数据（`.tmp/enwik8` /
    # `data/enwik8_full.npz`）⇒ 无论从哪个目录调用本入口，先 chdir 到仓库根，
    # 否则共享训练会去实验目录里找 `data/`。
    os.chdir(ROOT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARM_FORMS)
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--no-train", action="store_true",
                    help="跳过训练，直接复用 runs/…/model.pt 评测")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--smoke-gate", action="store_true",
                    help="只跑 CPU 等值门（init 逐位一致）后退出")
    ap.add_argument("--smoke-steps", type=int, default=20,
                    help="冒烟步数（0 = 跳过；四臂一次持锁）")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--smoke-only", action="store_true",
                    help="只跑等值门 + 冒烟步数（不跑全臂）")
    ap.add_argument("--arms", default=",".join(EXEC_ORDER))
    ap.add_argument("--seeds", default=str(DEFAULTS["seed"]))
    ap.add_argument("--force", action="store_true", help="忽略已有 checkpoint 重训")
    ap.add_argument("--inter-arm-wait", type=float, default=60.0,
                    help="臂间释放锁后的礼貌窗口（秒；SCHEMA §20.1 ≥60）")
    args = ap.parse_args()

    if args.smoke_gate:
        import smoke as smoke_mod
        r = smoke_mod.init_equality_gate("cpu")
        if torch.cuda.is_available():
            r2 = smoke_mod.init_equality_gate("cuda")
            r = [r, r2]
        print(json.dumps(r, indent=2, ensure_ascii=False), flush=True)
        return

    if args.aggregate:
        aggregate(args.steps, [int(s) for s in args.seeds.split(",") if s.strip()],
                  args.out_root)
        return

    if args.arm:
        with gpu_lock(LOCK_NAME):
            _train_and_eval(args.arm, args.seed, args.steps, args.batch_size,
                            args.lr, args.out_root, args.force or not args.no_train)
        return

    # ---- 全臂 driver：等值门（CPU）→ 20 步冒烟（一次持锁）→ 逐臂（按臂取/放锁）
    import smoke as smoke_mod
    print("[2652] === 等值门（CPU）===", flush=True)
    r = smoke_mod.init_equality_gate("cpu")
    print(json.dumps(r, indent=2, ensure_ascii=False), flush=True)

    arms = [a for a in args.arms.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    if not args.skip_smoke and args.smoke_steps > 0:
        print(f"[2652] === 冒烟 {args.smoke_steps} 步 × {len(arms)} 臂"
              f"（一次持锁，整程 <10 min）===", flush=True)
        try:
            with gpu_lock(LOCK_NAME):
                smoke_steps(args.smoke_steps, seeds[0], arms)
        except GPUUnavailable as e:
            print(f"[2652] GPU 不可用：{e}", flush=True)

    if args.smoke_only:
        return

    for i, form in enumerate(arms):
        for s in seeds:
            print(f"[2652] === 臂 {form}-s{s}（取锁 → 训练 → 评测 → 放锁）===",
                  flush=True)
            try:
                with gpu_lock(LOCK_NAME):
                    _train_and_eval(form, s, args.steps, args.batch_size, args.lr,
                                    args.out_root, args.force)
            except GPUUnavailable as e:
                print(f"[2652] GPU 不可用（{form}-s{s} 未跑）：{e}；登记为 blocker",
                      flush=True)
                continue
            if i < len(arms) - 1 or s != seeds[-1]:
                print(f"[2652] 释放锁，礼貌窗口 {args.inter_arm_wait:.0f}s", flush=True)
                time.sleep(args.inter_arm_wait)
    aggregate(args.steps, seeds, args.out_root)


if __name__ == "__main__":
    main()
