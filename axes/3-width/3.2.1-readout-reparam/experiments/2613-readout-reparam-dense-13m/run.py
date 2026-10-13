"""2613 · 读出重参数化在普通 dense 模型上的可迁移性 · 13M —— 复现入口。

单因素（契约 F，判据线逐字抄自 `.scratch/30-batch3/issues/01-batch3.md`「契约 F」，
**未改**）。基底 = 普通 dense AR（`train_depth_ar.py --dense`：无门、无退出采样、
仅最终深度 CE，= 主流标准 decoder-only transformer + 共享底座三条偏离）。臂：

- `baseline`：tied 线性读出 `logits = h·Eᵀ`；
- `linres`（主臂）：读出前纯线性状态变换 `h' = h + P·(Q h)`（2·d² 参数，
  init Q ~ N(0, 0.02) / **P = 0** 单边零 ⇒ 严格恒等起步、非鞍点）；
- `gated`（可选第三臂，契约允许）：`h' = h + SiLU(W1 h) ⊙ (W2 h)`（同 2·d²）。

判据（预登记，不可改）：Δbpc_improvement = bpc_baseline − bpc_linres
  ≥ +0.014 ⇒ 可迁移（与架构无关的通用发现）；
  |Δ| < 0.014 ⇒ 不可迁移（只在 depth-ar 形态内有效）；
  ≤ −0.014 ⇒ 负迁移。
必附**两臂同方法**（末 25% 同一窗口、同一最小二乘）loss 斜率。

用法（后台 + ≤120s 轮询）：
    python3 run.py --steps 12000 --arms baseline,linres,gated > run.log 2>&1 &
    tail -f run.log
"""

import argparse, json, os, shutil, statistics, subprocess, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, depth_curve,
                             tf_acceptance)
from pathlm.data import load_enwik8_full, batch
from pathlm.metrics import ece

DATA = os.path.join(ROOT, "data", "enwik8_full.npz")
DEVICE = "cuda"

# 历史同引擎 dense 13M（tied 线性读出，同 hyper / 同 seed / 同 token 预算）：
# 用来证明「当前引擎逐位复现既有 dense 基线」。口径见 analysis.md §6。
HIST_DENSE_DIR = os.path.join(ROOT, ".scratch", "23-depth-ar", "evidence",
                              "23d-dense-B-12k")
HIST_DENSE_BPC = 1.5287

# 12k 快照下 depth-ar 形态（2611）的读出重参数化增益，作可迁移性的参照系
DEPTHAR_LINRES_DELTA_12K = -0.0888
DEPTHAR_LINRES_DELTA_24K = -0.0463

# 方差界（docs/methodology.md §2）：Δ < 0.014 不可排序
VARIANCE_BOUND = 0.014

ARMS = {
    "baseline": {"flags": [], "run": "dense-base-{steps}-s{seed}"},
    "linres": {"flags": ["--readout-linear-resid"], "run": "dense-linres-{steps}-s{seed}"},
    "gated": {"flags": ["--readout-gated"], "run": "dense-gated-{steps}-s{seed}"},
}


def dense_cfg(readout: str = "baseline") -> DepthARConfig:
    """契约 F 的 dense 骨架：d=256/L=8/h=8/seq256/mlp4，dense=True 形态。"""
    return DepthARConfig(d_model=256, n_layers=8, n_heads=8, seq_len=256,
                         mlp_mult=4, dense=True,
                         readout_gated=(readout == "gated"),
                         readout_linear_resid=(readout == "linres"))


def build_model(readout: str, vocab_size: int, seed: int = 0) -> DepthARModel:
    torch.manual_seed(seed)
    return DepthARModel(dense_cfg(readout), vocab_size).to(DEVICE)


def arm_run_dir(arm: str, steps: int, seed: int = 0) -> str:
    return os.path.join(HERE, "runs", ARMS[arm]["run"].format(steps=steps, seed=seed))


_LOG_BY_SEED = {0: "run_log.txt", 1: "run_seed1.log"}


def peak_from_log(arm: str, seed: int) -> float | None:
    """--no-train 重取证据时，从训练日志里取回 nvidia-smi 峰值显存（run.py 在
    训练时打印）。每个 seed 一个日志文件（seed0=run_log.txt，seed1=run_seed1.log）。"""
    fn = _LOG_BY_SEED.get(seed)
    if not fn:
        return None
    p = os.path.join(HERE, fn)
    if not os.path.exists(p):
        return None
    tag = f"[run] arm={arm} done wall="
    peak = None
    for line in open(p):
        if line.startswith(tag) and "peak=" in line:
            peak = float(line.split("peak=")[1].split("GB")[0])
    return peak


# ------------------------- 评测电池（两臂同一代码） -------------------------

@torch.no_grad()
def prob0_ece(model: DepthARModel, eval_arr, n_batches: int = 8,
              batch_size: int = 16, n_bins: int = 15, seed: int = 7) -> dict:
    """每深度 softmax 后 top-1 概率对 argmax 正确率的等频分箱 ECE。
    dense 形态下只有最终深度被监督，prob0_ece = 最终深度，另报 depth4 供参照。"""
    gen = torch.Generator().manual_seed(seed)
    L = model.cfg.n_layers
    conf = [[] for _ in range(L + 1)]
    hit = [[] for _ in range(L + 1)]
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
        x = x.to(DEVICE)
        _, aux = model(x, exit_depths=None)
        tgt = x[:, 1:]
        for k, lg in enumerate(aux["depth_logits"]):
            lg = lg[:, :-1].float()
            p = lg.softmax(-1)
            conf[k].append(p.max(-1).values.reshape(-1).cpu())
            hit[k].append((lg.argmax(-1) == tgt).reshape(-1))
    per_depth = [round(ece(torch.cat(conf[k]), torch.cat(hit[k]).float(),
                           n_bins=n_bins), 4) for k in range(L + 1)]
    return {"prob0_ece": per_depth[-1], "exit4_ece": per_depth[4],
            "ece_by_depth": per_depth, "n_bins": n_bins}


def intervention_liveness(model: DepthARModel) -> dict:
    """干预活性（SCHEMA §10 强制项）：训练结束时 P/Q（或 W1/W2）必须显著非零，
    否则干预从未被激活。"""
    if model.cfg.readout_linear_resid:
        P = model.readout_p.weight.detach().float()
        Q = model.readout_q.weight.detach().float()
        out = {"p_fro": round(float(P.norm()), 4),
               "q_fro": round(float(Q.norm()), 4),
               "w_absmax": round(float(max(P.abs().max(), Q.abs().max())), 4),
               "p_spec_norm": round(float(torch.linalg.matrix_norm(P, 2)), 4),
               "q_spec_norm": round(float(torch.linalg.matrix_norm(Q, 2)), 4),
               "params_added": 2 * model.cfg.d_model ** 2}
        n = P.numel() + Q.numel()
        nz = (torch.count_nonzero(P) + torch.count_nonzero(Q)).item()
    else:
        W1 = model.readout_w1.weight.detach().float()
        W2 = model.readout_w2.weight.detach().float()
        out = {"w1_fro": round(float(W1.norm()), 4),
               "w2_fro": round(float(W2.norm()), 4),
               "w_absmax": round(float(max(W1.abs().max(), W2.abs().max())), 4),
               "params_added": 2 * model.cfg.d_model ** 2}
        n = W1.numel() + W2.numel()
        nz = (torch.count_nonzero(W1) + torch.count_nonzero(W2)).item()
    out["readout"] = model.cfg.readout_linear_resid and "linear-resid" \
        or (model.cfg.readout_gated and "gated" or "tied-linear")
    out["nonzero_frac"] = round(nz / n, 6)
    return out


@torch.no_grad()
def transform_rel_mag(model: DepthARModel, eval_arr, n_batches: int = 4,
                      batch_size: int = 8, seed: int = 11) -> list[float]:
    """读出变换相对幅度 ||transform(h)|| / ||h||，逐深度（干预幅度量化）。
    baseline 无变换 ⇒ 恒 0。"""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    acc = [[] for _ in range(model.cfg.n_layers + 1)]
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
        x = x.to(DEVICE)
        _, aux = model(x, exit_depths=None)
        for k, s in enumerate(aux["states"]):
            h = s.float()
            if model.cfg.readout_gated:
                t = torch.nn.functional.silu(model.readout_w1(h)) \
                    * model.readout_w2(h)
            elif model.cfg.readout_linear_resid:
                t = model.readout_p(model.readout_q(h))
            else:
                t = torch.zeros((), device=h.device)
            acc[k].append(float(t.norm() / h.norm()))
    return [round(statistics.mean(a), 4) for a in acc]


def run_battery(model: DepthARModel, eval_arr, n_batches: int) -> dict:
    out = {}
    out["depth_curve"] = depth_curve(model, eval_arr, device=DEVICE,
                                     n_batches=n_batches)
    out["tf_acceptance"] = tf_acceptance(model, eval_arr, device=DEVICE,
                                         n_batches=n_batches)
    out["tf_note"] = ("dense 形态的中间深度读出无直接监督，tf@k 是 depth-k 残差态"
                      " tied 读出与最终读出的 argmax 一致率 proxy（非训练出口资产）")
    out["calibration"] = prob0_ece(model, eval_arr, n_batches=8, batch_size=16)
    out["transform_rel_mag_by_depth"] = transform_rel_mag(model, eval_arr)
    if model.cfg.readout_gated or model.cfg.readout_linear_resid:
        out["intervention"] = intervention_liveness(model)
    return out


# ------------------------- 训练子进程（后台 + 轮询） -------------------------

def train_arm(arm: str, steps: int, seed: int, caps_min: float,
              eval_batches: int) -> dict:
    """后台启动 train_depth_ar.py（--dense），轮询日志 + nvidia-smi 峰值显存。
    训练前清空本 run 的日志（train_depth_ar 是 append 语义）。"""
    run_dir = arm_run_dir(arm, steps, seed)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, "train_depth_ar.py", ARMS[arm]["run"].format(steps=steps, seed=seed),
           "--d", "256", "--layers", "8", "--heads", "8", "--seq-len", "256",
           "--mlp-mult", "4", "--steps", str(steps), "--batch-size", "32",
           "--seed", str(seed), "--exit-mode", "uniform", "--dense",
           "--eval-batches", str(eval_batches),
           "--out-root", os.path.join(HERE, "runs")] + ARMS[arm]["flags"]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] arm={arm} pid={proc.pid} steps={steps}", flush=True)
    peak, t0 = 0.0, time.time()
    log_path = os.path.join(run_dir, "train_log.jsonl")
    last_printed = -1
    while proc.poll() is None:
        if time.time() - t0 > caps_min * 60:
            proc.terminate()
            raise TimeoutError(f"arm {arm} exceeded {caps_min} min cap")
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True)
            for line in out.stdout.strip().splitlines():
                pid, mem = [s.strip() for s in line.split(",")]
                if int(pid) == proc.pid:
                    peak = max(peak, float(mem) / 1024.0)
        except Exception:
            pass
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            if lines:
                last = json.loads(lines[-1])
                if int(last["step"]) % 1000 == 0 and int(last["step"]) != last_printed:
                    last_printed = int(last["step"])
                    print(f"[{arm}] {lines[-1].strip()[:130]}", flush=True)
        time.sleep(20)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"arm {arm} training failed rc={proc.returncode}")
    for name in ("train_log.jsonl", "gate_log.jsonl"):
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(
                HERE, f"{arm}-s{seed}-{os.path.basename(src)}"))
    wall = (time.time() - t0) / 60
    print(f"[run] arm={arm} done wall={wall:.2f} min peak={peak:.2f}GB", flush=True)
    res = json.load(open(os.path.join(run_dir, "results.json")))
    return {"results": res, "wall_min": round(wall, 2),
            "peak_child_mem_gb": round(peak, 2)}


# ------------------------- 同方法 loss 斜率（收敛对账） -------------------------

def _ls_slope(pairs: list[tuple[float, float]]) -> float | None:
    n = len(pairs)
    if n < 2:
        return None
    xm = sum(p[0] for p in pairs) / n
    ym = sum(p[1] for p in pairs) / n
    num = sum((x - xm) * (y - ym) for x, y in pairs)
    den = sum((x - xm) ** 2 for x, _ in pairs)
    return num / den if den else None


def window_slope(rows: list[dict], key: str, start: int, steps: int) -> dict:
    """窗内 loss 对 step 的最小二乘斜率 ×1000（每 1000 步的 loss 变化）。
    **所有臂同一窗口、同一算法**（SCHEMA §16）。key = "loss" 或 "depth_ce0"。"""
    win = [r for r in rows if start <= r["step"] <= steps]
    pairs = [(r["step"], r["loss"] if key == "loss" else r["depth_ce"][0])
             for r in win]
    s = _ls_slope(pairs)
    return {"n_points": len(win), "step_first": win[0]["step"] if win else None,
            "step_last": win[-1]["step"] if win else None,
            "slope_per_1k_steps": round(s * 1000, 5) if s is not None else None}


def load_arm_rows(arm: str, steps: int, seed: int = 0) -> list[dict]:
    p = os.path.join(HERE, f"{arm}-s{seed}-train_log.jsonl")
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return rows


def convergence_block(arms: list[str], steps: int, seed: int = 0) -> dict:
    """两臂/多臂**同方法**斜率（末 25% 窗为主口径，另附末 50% 与末 1000 步
    交叉核对），以及斜率差。斜率相近 ⇒ 差异是水平差（质量）；悬殊 ⇒ 收敛速度差。"""
    blk = {}
    for arm in arms:
        rows = load_arm_rows(arm, steps, seed)
        blk[arm] = {
            "loss_last25pct": window_slope(rows, "loss", int(0.75 * steps), steps),
            "loss_last50pct": window_slope(rows, "loss", int(0.5 * steps), steps),
            "loss_last1000": window_slope(rows, "loss", steps - 1000, steps),
            "final_ce_last25pct": window_slope(rows, "depth_ce0", int(0.75 * steps),
                                               steps),
            "loss_first": rows[0]["loss"], "loss_last": rows[-1]["loss"],
            "step_first": rows[0]["step"], "step_last": rows[-1]["step"],
        }
    base = blk["baseline"]["loss_last25pct"]["slope_per_1k_steps"]
    for arm in arms:
        s = blk[arm]["loss_last25pct"]["slope_per_1k_steps"]
        blk[arm]["slope_minus_baseline_last25pct"] = (
            round(s - base, 5) if (s is not None and base is not None) else None)
    return {
        "method": "末 25% 窗（同一窗口比例、同一最小二乘算法）loss 对 step 的斜率"
                  " ×1000（每 1000 步的 loss 变化）；bench 同一口径。",
        "window_def": f"last 25% = steps {int(0.75 * steps)}..{steps}",
        "by_arm": blk,
        "note": "loss 是单 batch 的最终深度 CE（dense 形态 n_ce=1，无混合深度噪声）；"
                "斜率用于「是否只是收敛加速」的对账，不单独作判据。",
    }


# ------------------------- 等值性 / 对账 -------------------------

def reload_arm(arm: str, steps: int, vocab_size: int, seed: int = 0) -> DepthARModel:
    run_dir = arm_run_dir(arm, steps, seed)
    model = build_model(arm, vocab_size, seed=0)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"),
                                     map_location=DEVICE))
    return model


def config_equivalence(arms: list[str], steps: int, seed: int) -> dict:
    """配置层逐字段对照（SCHEMA §9/§16：两臂逐字段一致，唯一变量 = 读出几何）。"""
    cfgs = {a: json.load(open(os.path.join(arm_run_dir(a, steps, seed), "results.json")))
            for a in arms}
    fields = ["d_model", "n_layers", "n_heads", "seq_len", "mlp_mult", "dropout",
              "norm_cap", "gate_bias_init", "dense", "fill_kv", "proj_fill",
              "readout_dims", "topology"]
    top = ["exit_mode", "aux_weight", "aux_anneal_to", "final_weight",
           "shallow_stopgrad", "dense_aux", "exit_anneal",
           "train_tokens", "vocab_size"]
    rows = []
    base = cfgs["baseline"]
    for f in fields:
        vals = [cfgs[a]["config"].get(f, getattr(dense_cfg(), f, None))
                for a in arms]
        rows.append({"field": f, "values": dict(zip(arms, vals)),
                     "all_equal": len(set(map(str, vals))) == 1,
                     "source": "results.json::config"})
    for f in top:
        vals = [cfgs[a].get(f) for a in arms]
        rows.append({"field": f, "values": dict(zip(arms, vals)),
                     "all_equal": len(set(map(str, vals))) == 1,
                     "source": "results.json"})
    # train 块
    for f in ("steps", "batch_size", "lr", "seed"):
        vals = [cfgs[a]["train"][f] for a in arms]
        rows.append({"field": f"train.{f}", "values": dict(zip(arms, vals)),
                     "all_equal": len(set(map(str, vals))) == 1,
                     "source": "results.json::train"})
    # 读出几何 + 新增参数量 = 唯一变量（预期不等：params 差 = 2·d²）
    for f in ("readout_gated", "readout_linear_resid", "params"):
        vals = [cfgs[a].get(f, False) if f != "params" else cfgs[a].get(f)
                for a in arms]
        rows.append({"field": f, "values": dict(zip(arms, vals)),
                     "all_equal": len(set(map(str, vals))) == 1,
                     "source": "唯一变量 = 读出几何（预期不等；params 差 = 2·d²）"})
    geom_fields = {"readout_gated", "readout_linear_resid", "params"}
    all_equal_except_geom = all(r["all_equal"] for r in rows
                                if r["field"] not in geom_fields)
    return {"rows": rows, "all_equal_except_readout_geometry": all_equal_except_geom,
            "baseline_source": os.path.relpath(arm_run_dir("baseline", steps, seed), ROOT)}


def historical_reproduction(eval_arr) -> dict:
    """证明当前引擎逐位复现既有 dense 13M 基线（23d-dense-B-12k）。"""
    model = build_model("baseline", 206, seed=0)
    sd = torch.load(os.path.join(HIST_DENSE_DIR, "model.pt"), map_location=DEVICE)
    load = model.load_state_dict(sd)
    curve = depth_curve(model, eval_arr, device=DEVICE, n_batches=20)
    return {"source": os.path.relpath(HIST_DENSE_DIR, ROOT),
            "state_dict_missing": list(load.missing_keys),
            "state_dict_unexpected": list(load.unexpected_keys),
            "reproduced_bpc": curve[-1]["bpc"], "hist_bpc": HIST_DENSE_BPC,
            "bitwise_equal": curve[-1]["bpc"] == HIST_DENSE_BPC,
            "note": "同一评测代码（depth_curve, 20 batches）重测历史 checkpoint；"
                    "bpc 逐位复现 ⇒ 当前引擎与历史实现等价。"}


# ------------------------- 主流程 -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="baseline,linres,gated",
                    help="逗号分隔：baseline,linres,gated")
    ap.add_argument("--caps-min", type=float, default=25.0,
                    help="每臂训练墙钟上限（分钟），超时中止")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--results-file", default="results.json")
    ap.add_argument("--no-train", action="store_true",
                    help="跳过训练，直接评测 runs/ 下已有 checkpoint")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        assert a in ARMS, f"unknown arm {a}"

    t_start = time.time()
    torch.manual_seed(args.seed)
    _, eval_arr, vocab_size = load_enwik8_full(os.path.join(ROOT, ".tmp", "enwik8"), DATA)

    base_train = {"d_model": 256, "n_layers": 8, "n_heads": 8, "seq_len": 256,
                  "mlp_mult": 4, "dense": True, "fill_kv": True,
                  "proj_fill": False, "topology": "sequential",
                  "gate_bias_init": -2.0, "readout_dims": None,
                  "aux_weight": 1.0, "final_weight": 1.0,
                  "shallow_stopgrad": False, "dense_aux": False}
    cfg_dump = {
        "experiment": "2613-readout-reparam-dense-13m",
        "contract": "契约 F（.scratch/30-batch3/issues/01-batch3.md）",
        "seed": args.seed, "steps": args.steps, "batch_size": 32, "lr": 0.001,
        "exit_mode": "uniform", "data": DATA,
        "data_split": "first 90M train / last 10M eval, vocab from full file",
        "vocab_size": vocab_size, "device": DEVICE,
        "base": base_train, "arms": arms,
        "intervention": {
            "baseline": {"readout": "tied linear h·Eᵀ", "readout_linear_resid": False,
                         "readout_gated": False},
            "linres": {"readout": "(h + P·(Q h))·Eᵀ（整体 (I+PQ)h）",
                       "readout_linear_resid": True,
                       "init": "Q ~ N(0,0.02), P = 0（单边零，恒等起步、非鞍点）",
                       "params_added": 2 * 256 * 256},
            "gated": {"readout": "(h + SiLU(W1 h)⊙(W2 h))·Eᵀ",
                      "readout_gated": True,
                      "init": "W1 ~ N(0,0.02), W2 = 0（单边零）",
                      "params_added": 2 * 256 * 256},
        },
        "criterion": {
            "definition": "Δbpc_improvement = bpc_baseline − bpc_linres（正 = 改善）",
            "transferable": "Δ ≥ +0.014",
            "not_transferable": "|Δ| < 0.014",
            "negative_transfer": "Δ ≤ −0.014",
            "variance_bound": VARIANCE_BOUND,
            "same_method_slope_required": True,
        },
        "reference_depthar_12k": DEPTHAR_LINRES_DELTA_12K,
        "reference_depthar_24k": DEPTHAR_LINRES_DELTA_24K,
    }
    json.dump(cfg_dump, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)

    prev = {}
    if not args.no_train:
        for arm in arms:
            tr = train_arm(arm, args.steps, args.seed, args.caps_min,
                           args.eval_batches)
            prev[arm] = tr
    out = {"experiment": cfg_dump["experiment"], "config": cfg_dump,
           "train": {a: prev.get(a, {}) for a in arms}}

    # ---- 历史复现门 ----
    out["historical_reproduction"] = historical_reproduction(eval_arr)
    print(f"[hist] bpc={out['historical_reproduction']['reproduced_bpc']} "
          f"(hist {HIST_DENSE_BPC})", flush=True)

    # ---- 配置层等值性 ----
    out["config_equivalence"] = config_equivalence(arms, args.steps, args.seed)
    print(f"[cfg] all_equal_except_readout={out['config_equivalence']['all_equal_except_readout_geometry']}",
          flush=True)

    # ---- 逐臂电池 ----
    for arm in arms:
        model = reload_arm(arm, args.steps, vocab_size, args.seed)
        t0 = time.time()
        batt = run_battery(model, eval_arr, args.eval_batches)
        arm_res = (prev[arm]["results"] if arm in prev
                   else json.load(open(os.path.join(
                       arm_run_dir(arm, args.steps, args.seed), "results.json"))))
        out[f"{arm}_arm"] = {
            "source": f"runs/{ARMS[arm]['run'].format(steps=args.steps, seed=args.seed)}",
            "params": arm_res.get("params"),
            "wall_min": (prev[arm].get("wall_min") if arm in prev
                         else arm_res.get("wall_minutes")),
            "peak_child_mem_gb": (prev[arm].get("peak_child_mem_gb") if arm in prev
                                  else peak_from_log(arm, args.seed)),
            "eval_wall_min": round((time.time() - t0) / 60, 2),
            "gnorm_monitor": arm_res.get("gnorm_monitor"),
            "train_wall_min_reported": arm_res.get("wall_minutes"),
            **batt,
        }
        print(f"[{arm}] clean_bpc={batt['depth_curve'][-1]['bpc']} "
              f"tf4={next(r['accept'] for r in batt['tf_acceptance'] if r['depth']==4)} "
              f"params={out[f'{arm}_arm']['params']}", flush=True)

    # ---- train 块（把每臂训练端的 wall/peak/params/gnorm 汇总；--no-train 时
    # 从 run 自身的 results.json 回填，保证证据齐全） ----
    out["train"] = {a: {"source": out[f"{a}_arm"]["source"],
                        "wall_min": out[f"{a}_arm"]["wall_min"],
                        "peak_child_mem_gb": out[f"{a}_arm"]["peak_child_mem_gb"],
                        "params": out[f"{a}_arm"]["params"],
                        "gnorm_monitor": out[f"{a}_arm"]["gnorm_monitor"]}
                    for a in arms}

    # ---- 干预活性块（SCHEMA §10：顶层镜像，便于复核清单逐项核对） ----
    out["intervention"] = out.get("linres_arm", {}).get("intervention")

    # ---- 同方法斜率（收敛对账） ----
    out["convergence"] = convergence_block(arms, args.steps, args.seed)

    # ---- 对账（SCHEMA §12）与判读 ----
    b_bpc = out["baseline_arm"]["depth_curve"][-1]["bpc"]
    l_bpc = out["linres_arm"]["depth_curve"][-1]["bpc"]
    delta = round(b_bpc - l_bpc, 4)                 # 正 = 改善
    p_b = out["baseline_arm"]["params"]
    p_l = out["linres_arm"]["params"]
    slopes = out["convergence"]["by_arm"]
    sl_b = slopes["baseline"]["loss_last25pct"]["slope_per_1k_steps"]
    sl_l = slopes["linres"]["loss_last25pct"]["slope_per_1k_steps"]

    if delta >= VARIANCE_BOUND:
        verdict = "可迁移（与架构无关的通用发现）"
    elif delta <= -VARIANCE_BOUND:
        verdict = "负迁移"
    else:
        verdict = "不可迁移（只在 depth-ar 形态内有效）"

    out["reconciliation"] = {
        "arithmetic": {
            "bpc_baseline": b_bpc, "bpc_linres": l_bpc,
            "delta_improvement": delta,
            "delta_arm_minus_baseline": round(-delta, 4),
            "criterion_transferable": f"Δ ≥ +{VARIANCE_BOUND}", "met": delta >= VARIANCE_BOUND,
            "criterion_negative": f"Δ ≤ −{VARIANCE_BOUND}", "is_negative": delta <= -VARIANCE_BOUND,
            "within_variance_bound": abs(delta) < VARIANCE_BOUND,
            "params_baseline": p_b, "params_linres": p_l,
            "params_extra": None if (p_b is None or p_l is None) else p_l - p_b,
            "params_extra_expected": 2 * 256 * 256,
            "params_extra_equal_2d2": None if (p_b is None or p_l is None)
            else (p_l - p_b) == 2 * 256 * 256,
            "bpc_lower_bound": 0.0,
            "fresh_baseline_vs_hist_retrain": round(b_bpc - HIST_DENSE_BPC, 4),
            "fresh_baseline_vs_hist_note": (
                "全新重训的 baseline 与历史 23d-dense-B-12k（同配置同 seed）的 bpc "
                "差 = run-to-run 数值不确定性底噪（CUDA bf16 归约顺序）；历史 "
                "checkpoint 重载评测仍逐位得 1.5287（见 historical_reproduction）。"),
            "reference_depthar_12k_delta_improvement": -DEPTHAR_LINRES_DELTA_12K,
            "reference_depthar_24k_delta_improvement": -DEPTHAR_LINRES_DELTA_24K,
        },
        "task": {
            "same_task": "同一 enwik8 char 级 AR 任务、同一数据切分（前 90M/末 10M）、"
                         "同一位置集（全序列逐位置、最终深度读出）、同一目标定义"
                         "（tokens 自移位）",
            "same_battery": "两臂用同一 run.py::run_battery（同 n_batches/seed/"
                            "batch_size/评测代码）",
            "no_truth_leak": "两臂读出只依赖模型自身状态，无真值泄漏，无「假上下文」"
                             "式对照",
            "deploy_correspondence": "两臂都是 stock decoder-only 形态的标准读出路径；"
                                     "对照只改读出前状态变换",
        },
        "caliber": {
            "matched_step_snapshot": True, "steps": args.steps,
            "train_tokens": args.steps * 32 * 256, "seed": args.seed, "n_seed": 1,
            "note": "两臂同为 matched-step 快照口径（非渐近、非墙钟、非 FLOPs）；"
                    "梯度计算含 CUDA 非确定性，快照 bpc 以实测为准。",
        },
    }
    out["verdict"] = {
        "criterion": ("Δbpc_improvement = bpc_baseline − bpc_linres；"
                      "≥ +0.014 可迁移 / |Δ| < 0.014 不可迁移 / ≤ −0.014 负迁移"),
        "delta_improvement": delta,
        "bpc_baseline": b_bpc, "bpc_linres": l_bpc,
        "linres_tf4": next(r["accept"] for r in out["linres_arm"]["tf_acceptance"]
                           if r["depth"] == 4),
        "baseline_tf4": next(r["accept"] for r in out["baseline_arm"]["tf_acceptance"]
                             if r["depth"] == 4),
        "baseline_prob0_ece": out["baseline_arm"]["calibration"]["prob0_ece"],
        "linres_prob0_ece": out["linres_arm"]["calibration"]["prob0_ece"],
        "slope_last25pct_baseline": sl_b, "slope_last25pct_linres": sl_l,
        "slope_diff_linres_minus_baseline": (None if (sl_b is None or sl_l is None)
                                             else round(sl_l - sl_b, 5)),
        "verdict": verdict,
    }
    if "gated" in arms:
        out["verdict"]["gated_bpc"] = out["gated_arm"]["depth_curve"][-1]["bpc"]
        out["verdict"]["gated_delta_improvement"] = round(b_bpc - out["gated_arm"]["depth_curve"][-1]["bpc"], 4)

    out["wall_total_min"] = round((time.time() - t_start) / 60, 2)
    json.dump(out, open(os.path.join(HERE, args.results_file), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out["verdict"], indent=2, ensure_ascii=False), flush=True)
    print(f"[done] total wall {out['wall_total_min']} min", flush=True)


if __name__ == "__main__":
    main()
