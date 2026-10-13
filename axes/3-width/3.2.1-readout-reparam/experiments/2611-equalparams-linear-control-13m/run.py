"""2611 · 读出几何的等参数纯线性对照 · 13M · depth-ar —— 复现入口。

单因素（契约 B，判据线预先写下未改）：读出几何。三个臂共用同一配方
（全深度并行监督 + carry 门 + 随机退出 + fill_kv + 同 seed + 同 token 预算）：

- **A 臂**（基底）：tied 线性读出 `logits = h·Eᵀ`；
- **gated 臂**（2610 主臂，复用其 checkpoint）：读出前非线性状态变换
  `h' = h + SiLU(W1 h) ⊙ (W2 h)`，新增参数 2·d²；
- **本对照臂**（线性等参数）：读出前**纯线性**状态变换
  `h' = h + P·(Q h)`（整体 = (I + PQ) h），新增参数同为 2·d²，
  init Q ~ N(0, 0.02) / P = 0（单边零 ⇒ 严格恒等起步、非鞍点）。

判读：Δbpc（本对照 vs A）与 Δbpc（gated vs A）差距 < 0.02 ⇒ 增益来自参数量
（gated 结论降级）；本对照 |Δbpc| ≤ 0.014 而 gated ≤ −0.05 ⇒ 非线性本身有效。
另交 `stability.md`（三臂 gnorm / NaN / loss 突跳 / 逐深度 ECE 表）。

用法（后台运行 + ≤120s 轮询日志）：
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup python3 run.py --steps 12000 --seed 0 --caps-min 20 > run.log 2>&1 &
    tail -f run.log
"""

import argparse, json, os, shutil, statistics, subprocess, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, depth_curve,
                             tf_acceptance, workspace_probe, spec_cost_model,
                             generation_quality, readout_workspace_energy)
from pathlm.data import load_enwik8_full, batch
from pathlm.metrics import ece

# 历史同配置 A 臂（tied 线性，同 seed / 同 hyper / 同 token 预算 / fill）
A_ARM_DIR = os.path.join(ROOT, ".scratch", "23-depth-ar", "evidence",
                         "23c-H2-w0-12k")
A_LOG = os.path.join(A_ARM_DIR, "train_log.jsonl")
# 2610 gated 主臂（同配置 + 唯一变量 = 非线性读出），复用其 checkpoint 与训练日志
GATED_DIR = os.path.join(ROOT, "axes", "W-width", "W4-readout-geometry",
                         "experiments", "2610-gated-readout-13m-depthar")
GATED_RUN = os.path.join(GATED_DIR, "runs", "gated-12k-s0")
GATED_LOG = os.path.join(GATED_DIR, "train_log.jsonl")
# 同引擎 dense 13M matched 税分母（历史参照，口径见 analysis.md §6）
DENSE_BPC_13M = 1.5287
DATA = "data/enwik8_full.npz"
DEVICE = "cuda"
RUN_NAME = "linear-12k-s0"


def base_cfg(**kw) -> DepthARConfig:
    """契约 §2 骨架：d=256/L=8/h=8/seq256/mlp4，carry 门、随机退出、fill_kv。"""
    return DepthARConfig(d_model=256, n_layers=8, n_heads=8, seq_len=256,
                         mlp_mult=4, dense=False, fill_kv=True, **kw)


# ------------------------- prob0 ECE（与 2610 同实现） -------------------------

@torch.no_grad()
def prob0_ece(model: DepthARModel, eval_arr, n_batches: int = 8,
              batch_size: int = 16, n_bins: int = 15, seed: int = 7) -> dict:
    """prob0 校准：每深度 softmax 后 top-1 概率对 argmax 正确率的等频分箱
    ECE（`pathlm.metrics.ece`）。prob0_ece = 最终深度；exit4_ece = 出口深度 4。"""
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
    d4 = 4 if 4 <= L else L
    return {"prob0_ece": per_depth[-1], "exit4_ece": per_depth[d4],
            "ece_by_depth": per_depth, "n_bins": n_bins}


# ------------------------------ 评测电池 ------------------------------

def intervention_liveness(model: DepthARModel) -> dict:
    """干预活性判据（SCHEMA §10 强制项）：训练结束时 P/Q 必须显著非零，
    否则干预从未被激活。"""
    P = model.readout_p.weight.detach().float()
    Q = model.readout_q.weight.detach().float()
    n = P.numel() + Q.numel()
    nz = (torch.count_nonzero(P) + torch.count_nonzero(Q)).item()
    return {"p_fro": round(float(P.norm()), 4),
            "q_fro": round(float(Q.norm()), 4),
            "w_absmax": round(float(max(P.abs().max(), Q.abs().max())), 4),
            "nonzero_frac": round(nz / n, 6),
            "p_spec_norm": round(float(torch.linalg.matrix_norm(P, 2)), 4),
            "q_spec_norm": round(float(torch.linalg.matrix_norm(Q, 2)), 4),
            "params_added": 2 * model.cfg.d_model ** 2}


@torch.no_grad()
def transform_rel_mag(model: DepthARModel, eval_arr, n_batches: int = 4,
                      batch_size: int = 8, seed: int = 11) -> list[float]:
    """读出变换相对幅度 ||transform(h)|| / ||h||，逐深度（干预是否只是小修正
    的量化证据）。与 2610 的 readout_gate_diag.json 同口径。"""
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
            else:                    # A 臂：读出前无状态变换（transform ≡ 0）
                t = torch.zeros((), device=h.device)
            acc[k].append(float(t.norm() / h.norm()))
    return [round(statistics.mean(a), 4) for a in acc]


def run_battery(model: DepthARModel, eval_arr: dict, eval_batches: int) -> dict:
    out = {}
    out["depth_curve"] = depth_curve(model, eval_arr, device=DEVICE,
                                     n_batches=eval_batches)
    out["tf_acceptance"] = tf_acceptance(model, eval_arr, device=DEVICE,
                                         n_batches=eval_batches)
    out["spec_cost_model"] = spec_cost_model(out["tf_acceptance"],
                                             model.cfg.n_layers,
                                             draft_depths=(4, 6))
    out["workspace"] = workspace_probe(model, eval_arr, device=DEVICE,
                                       n_batches=4, batch_size=8)
    out["readout_workspace_energy"] = readout_workspace_energy(
        model, eval_arr, device=DEVICE, n_batches=4, batch_size=8)
    out["generation"] = generation_quality(model, eval_arr, device=DEVICE,
                                           prompt_len=64, n_new=128)
    out["calibration"] = prob0_ece(model, eval_arr, n_batches=8, batch_size=16)
    out["transform_rel_mag_by_depth"] = transform_rel_mag(model, eval_arr)
    return out


# ------------------------------ 训练子进程 ------------------------------

def train_linear(steps: int, seed: int, caps_min: float) -> dict:
    """后台启动 train_depth_ar.py（--readout-linear-resid），轮询日志 +
    nvidia-smi 峰值显存；超时/失败即中止并抛出。训练前清空本 run 的日志
    （train_depth_ar 是 append 语义；2610 曾因重跑把两段日志串在一起）。"""
    run_dir = os.path.join(HERE, "runs", RUN_NAME)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, "train_depth_ar.py", RUN_NAME,
           "--d", "256", "--layers", "8", "--heads", "8", "--seq-len", "256",
           "--mlp-mult", "4", "--steps", str(steps), "--batch-size", "32",
           "--seed", str(seed), "--exit-mode", "uniform",
           "--readout-linear-resid", "--eval-batches", "20",
           "--out-root", os.path.join(HERE, "runs")]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] linear training started pid={proc.pid} steps={steps}",
          flush=True)
    peak, t0 = 0.0, time.time()
    log_path = os.path.join(run_dir, "train_log.jsonl")
    while proc.poll() is None:
        if time.time() - t0 > caps_min * 60:
            proc.terminate()
            raise TimeoutError(f"training exceeded {caps_min} min cap")
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
                if int(last["step"]) % 1000 == 0:
                    print(f"[run] {lines[-1].strip()[:130]}", flush=True)
        time.sleep(20)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"training failed rc={proc.returncode}")
    for name in ("train_log.jsonl", "gate_log.jsonl"):
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HERE, name))
    wall = (time.time() - t0) / 60
    print(f"[run] linear training done wall={wall:.2f} min "
          f"peak_child_mem={peak:.2f}GB", flush=True)
    res = json.load(open(os.path.join(run_dir, "results.json")))
    return {"results": res, "wall_min": round(wall, 2),
            "peak_child_mem_gb": round(peak, 2)}


# ---------------------- 等值性门（逐字段对照） ----------------------

def reload_model(directory: str, **cfg_kw) -> DepthARModel:
    hist = json.load(open(os.path.join(directory, "results.json")))
    torch.manual_seed(0)
    model = DepthARModel(base_cfg(**cfg_kw), hist["vocab_size"]).cuda()
    model.load_state_dict(torch.load(os.path.join(directory, "model.pt"),
                                     map_location="cuda"))
    return model, hist


def equivalence_table(hist: dict, gated: bool) -> list[dict]:
    """对照臂等值性门（SCHEMA §9）：对照 run 的 resolved config 与本对照臂
    逐字段对照；唯一允许不等的字段是读出几何（单因素）。"""
    cfg, tr = hist["config"], hist["train"]
    this = {"readout_gated": False, "readout_linear_resid": True}
    rows = [
        ("d_model", cfg["d_model"], 256, "results.json::config"),
        ("n_layers", cfg["n_layers"], 8, "results.json::config"),
        ("n_heads", cfg["n_heads"], 8, "results.json::config"),
        ("seq_len", cfg["seq_len"], 256, "results.json::config"),
        ("mlp_mult", cfg["mlp_mult"], 4, "results.json::config"),
        ("dropout", cfg["dropout"], 0.0, "results.json::config"),
        ("norm_cap", cfg["norm_cap"], 1.0, "results.json::config"),
        ("gate_bias_init", cfg["gate_bias_init"], -2.0, "results.json::config"),
        ("dense", cfg["dense"], False, "results.json::config"),
        ("readout_dims", cfg.get("readout_dims"), None, "results.json::config"),
        ("exit_mode", hist["exit_mode"], "uniform", "results.json"),
        ("fill_kv", hist.get("fill_kv", True), True, "results.json"),
        ("proj_fill", hist.get("proj_fill", False), False, "results.json"),
        ("aux_weight", hist["aux_weight"], 1.0, "results.json"),
        ("aux_anneal_to", hist["aux_anneal_to"], None, "results.json"),
        ("final_weight", hist.get("final_weight", 1.0), 1.0, "results.json"),
        ("shallow_stopgrad", hist["shallow_stopgrad"], False, "results.json"),
        ("dense_aux", hist["dense_aux"], False, "results.json"),
        ("steps", tr["steps"], 12000, "results.json::train"),
        ("batch_size", tr["batch_size"], 32, "results.json::train"),
        ("lr", tr["lr"], 0.001, "results.json::train"),
        ("seed", tr["seed"], 0, "results.json::train"),
        ("train_tokens", hist["train_tokens"], 98_304_000, "results.json"),
        ("vocab_size", hist["vocab_size"], 206, "results.json"),
        ("readout_gated", hist.get("readout_gated", False), this["readout_gated"],
         "唯一变量 = 读出几何（gated 臂为 True）"),
        ("readout_linear_resid", hist.get("readout_linear_resid", False),
         this["readout_linear_resid"], "唯一变量 = 读出几何（本对照臂为 True）"),
    ]
    return [{"field": f, "ref_arm": h, "linear_arm": t, "equal": h == t,
             "source": s} for f, h, t, s in rows]


# ---------------------- 稳定性统计（stability.md 数据源） ----------------------

def split_runs(path: str) -> list[list[dict]]:
    """train_log.jsonl 逐行解析并按 step 回绕切成多次 run（train_depth_ar 是
    append 语义：重跑会把两段串在同一文件里，2610 的日志就是两段）。"""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    runs, cur = [], []
    for r in rows:
        if cur and r["step"] <= cur[-1]["step"]:
            runs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    return runs


def stability_stats(rows: list[dict], gnorm_monitor: dict | None) -> dict:
    """一个臂的稳定性量：gnorm max / 中位数 / 尖峰次数（尖峰 = 该臂 200 步
    采样点上 gnorm > 3×中位数）/ NaN 检查 / loss 突跳（相邻采样点 loss 上升
    > 0.25）/ 全程 gnorm_monitor（12000 步真值，含跳步数）。"""
    g = [r["gnorm"] for r in rows]
    loss = [r["loss"] for r in rows]
    med = statistics.median(g)
    inc = [b - a for a, b in zip(loss, loss[1:])]
    finite = all(torch.isfinite(torch.tensor(x)).item()
                 for x in g + loss + [c for r in rows for c in r["depth_ce"]])
    return {
        "n_logged_samples": len(rows),
        "step_first": rows[0]["step"], "step_last": rows[-1]["step"],
        "gnorm_logged_max": max(g), "gnorm_logged_median": round(med, 3),
        "gnorm_spikes_gt3x_median": sum(1 for x in g if x > 3 * med),
        "gnorm_spikes_gt5x_median": sum(1 for x in g if x > 5 * med),
        "all_finite": finite,
        "loss_first": loss[0], "loss_last": loss[-1],
        "loss_max_step_increase": round(max(inc), 4),
        "loss_jumps_gt_0.25": sum(1 for x in inc if x > 0.25),
        "gnorm_monitor_all_steps": gnorm_monitor,
    }


def stability_block() -> dict:
    a_rows = split_runs(A_LOG)[-1]
    g_rows = split_runs(GATED_LOG)[-1]        # 末段 = 2610 主臂（首段是作废的零初始臂）
    l_rows = split_runs(os.path.join(HERE, "train_log.jsonl"))[-1]
    g_res = json.load(open(os.path.join(GATED_RUN, "results.json")))
    l_res = json.load(open(os.path.join(HERE, "runs", RUN_NAME, "results.json")))
    return {
        "note": "2610 的 train_log.jsonl 含两段 run（首段是作废的零初始鞍点臂，"
                "末段才是主臂）；此处只取各文件末段，段长见 step_first/step_last。",
        "A_arm_23c": stability_stats(a_rows, None),
        "gated_arm_2610": stability_stats(g_rows, g_res.get("gnorm_monitor")),
        "linear_arm_2611": stability_stats(l_rows, l_res.get("gnorm_monitor")),
    }


# ------------------------------ 主流程 ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--caps-min", type=float, default=20.0,
                    help="本对照臂训练墙钟上限（分钟），超时中止")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--eval-only", action="store_true",
                    help="只重算三臂电池（不训练），用于核对既有结果")
    args = ap.parse_args()

    t_start = time.time()
    torch.manual_seed(args.seed)
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", DATA)

    cfg_dump = {
        "experiment": "2611-equalparams-linear-control-13m",
        "seed": args.seed, "steps": args.steps, "batch_size": 32, "lr": 0.001,
        "exit_mode": "uniform", "data": DATA,
        "data_split": "first 90M train / last 10M eval, vocab from full file",
        "vocab_size": vocab_size,
        "A_arm": {"readout": "tied linear h·Eᵀ", "readout_gated": False,
                  "readout_linear_resid": False,
                  "source": os.path.relpath(A_ARM_DIR, ROOT)},
        "gated_arm": {"readout": "h + SiLU(W1 h)·(W2 h) then ·Eᵀ",
                      "readout_gated": True,
                      "readout_w1_w2_init": "W1 ~ N(0, 0.02), W2 = 0",
                      "params_extra": 2 * 256 * 256,
                      "source": os.path.relpath(GATED_RUN, ROOT)},
        "linear_arm": {"readout": "h + P·(Q h) then ·Eᵀ（整体 = (I + PQ) h）",
                       "readout_linear_resid": True,
                       "readout_p_q_init": "Q ~ N(0, 0.02), P = 0 "
                                           "(identity start, non-saddle)",
                       "params_extra": 2 * 256 * 256},
        "base": {"d_model": 256, "n_layers": 8, "n_heads": 8, "seq_len": 256,
                 "mlp_mult": 4, "dense": False, "fill_kv": True,
                 "proj_fill": False, "gate_bias_init": -2.0, "aux_weight": 1.0,
                 "final_weight": 1.0, "shallow_stopgrad": False,
                 "dense_aux": False},
    }
    json.dump(cfg_dump, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    out = {"experiment": cfg_dump["experiment"], "config": cfg_dump}

    # ---- A 臂：等值性门 + 当前代码重测 ----
    a_model, hist_a = reload_model(A_ARM_DIR)
    out["A_arm_equivalence"] = equivalence_table(hist_a, gated=False)
    out["A_arm_equivalence_all_equal_except_readout"] = all(
        r["equal"] for r in out["A_arm_equivalence"]
        if r["field"] not in ("readout_gated", "readout_linear_resid"))
    t0 = time.time()
    a_batt = run_battery(a_model, eval_arr, args.eval_batches)
    out["A_arm"] = {"source": os.path.relpath(A_ARM_DIR, ROOT),
                    "hist_clean_bpc": hist_a["depth_curve"][-1]["bpc"],
                    **a_batt, "eval_wall_min": round((time.time() - t0) / 60, 2)}
    print(f"[A] clean_bpc={a_batt['depth_curve'][-1]['bpc']} "
          f"tf4={next(r['accept'] for r in a_batt['tf_acceptance'] if r['depth']==4)} "
          f"({out['A_arm']['eval_wall_min']} min)", flush=True)

    # ---- gated 臂（2610 主臂 checkpoint）：等值性门 + 同口径重测 ----
    g_model, hist_g = reload_model(GATED_RUN, readout_gated=True)
    out["gated_arm_equivalence"] = equivalence_table(hist_g, gated=True)
    out["gated_arm_equivalence_all_equal_except_readout"] = all(
        r["equal"] for r in out["gated_arm_equivalence"]
        if r["field"] not in ("readout_gated", "readout_linear_resid"))
    t0 = time.time()
    g_batt = run_battery(g_model, eval_arr, args.eval_batches)
    out["gated_arm"] = {"source": os.path.relpath(GATED_RUN, ROOT),
                        "hist_clean_bpc": hist_g["depth_curve"][-1]["bpc"],
                        **g_batt, "eval_wall_min": round((time.time() - t0) / 60, 2),
                        "gnorm_monitor": hist_g.get("gnorm_monitor"),
                        "params": hist_g["params"],
                        "hist_wall_min": hist_g["wall_minutes"]}
    print(f"[gated] clean_bpc={g_batt['depth_curve'][-1]['bpc']} "
          f"tf4={next(r['accept'] for r in g_batt['tf_acceptance'] if r['depth']==4)} "
          f"({out['gated_arm']['eval_wall_min']} min)", flush=True)

    if args.eval_only:
        json.dump(out, open(os.path.join(HERE, "results.json"), "w"),
                  indent=2, ensure_ascii=False)
        print("[done] eval-only", flush=True)
        return

    # ---- 本对照臂：全新训练 12k（唯一变量 = 线性读出几何） ----
    tr = train_linear(args.steps, args.seed, args.caps_min)
    l_res = tr["results"]
    torch.manual_seed(args.seed)
    l_model = DepthARModel(base_cfg(readout_linear_resid=True), vocab_size).cuda()
    l_model.load_state_dict(torch.load(
        os.path.join(HERE, "runs", RUN_NAME, "model.pt"), map_location="cuda"))
    t0 = time.time()
    l_batt = run_battery(l_model, eval_arr, args.eval_batches)
    out["intervention"] = intervention_liveness(l_model)
    out["linear_arm"] = {"source": f"runs/{RUN_NAME}",
                         **l_batt, "wall_min": tr["wall_min"],
                         "peak_child_mem_gb": tr["peak_child_mem_gb"],
                         "eval_wall_min": round((time.time() - t0) / 60, 2),
                         "gnorm_monitor": l_res.get("gnorm_monitor"),
                         "params": l_res["params"],
                         "intervention": out["intervention"]}
    print(f"[linear] clean_bpc={l_batt['depth_curve'][-1]['bpc']} "
          f"tf4={next(r['accept'] for r in l_batt['tf_acceptance'] if r['depth']==4)} "
          f"params={l_res['params']} wall={tr['wall_min']} min", flush=True)

    # ---- 对账（SCHEMA §12）----
    a_bpc = out["A_arm"]["depth_curve"][-1]["bpc"]
    g_bpc = out["gated_arm"]["depth_curve"][-1]["bpc"]
    l_bpc = out["linear_arm"]["depth_curve"][-1]["bpc"]
    d_gated = round(g_bpc - a_bpc, 4)
    d_linear = round(l_bpc - a_bpc, 4)
    gap = round(abs(d_linear - d_gated), 4)
    params_a = hist_a["params"]
    params_g = hist_g["params"]
    params_l = l_res["params"]
    out["reconciliation"] = {
        "arithmetic": {
            "params_A": params_a, "params_gated": params_g, "params_linear": params_l,
            "params_extra_gated": params_g - params_a,
            "params_extra_linear": params_l - params_a,
            "params_extra_equal": (params_g - params_a) == (params_l - params_a)
                                  == 2 * 256 * 256,
            "variance_bound": 0.014,
            "decision_threshold_gap": 0.02,
            "threshold_above_noise_floor": 0.02 > 0.014,
            "delta_bpc_gated": d_gated, "delta_bpc_linear": d_linear,
            "delta_gap": gap,
        },
        "task": {
            "same_task": "同一 enwik8 char 级 AR 任务、同一 eval 切分（末 10M）、"
                         "同一位置集（全序列逐位置）、同一目标定义（tokens 自移位）",
            "same_battery": "三臂用同一 run.py::run_battery（同 n_batches/seed/"
                            "batch_size）",
            "deploy_correspondence": "三臂都是 depth-AR 的标准读出路径，"
                                     "无额外输入、无真值泄漏；对照只改读出前状态变换",
        },
        "caliber": {
            "matched_step_snapshot": True, "steps": args.steps,
            "train_tokens": l_res["train_tokens"], "seed": args.seed, "n_seed": 1,
            "note": "三臂同为 12k 步 / 98.3M token / seed 0 的快照口径（非渐近、"
                    "非墙钟、非 FLOPs）",
        },
    }

    # ---- 判读（契约 B 判据线预先写下，不许事后调整） ----
    g_tf4 = next(r["accept"] for r in out["gated_arm"]["tf_acceptance"]
                 if r["depth"] == 4)
    l_tf4 = next(r["accept"] for r in out["linear_arm"]["tf_acceptance"]
                 if r["depth"] == 4)
    a_tf4 = next(r["accept"] for r in out["A_arm"]["tf_acceptance"]
                 if r["depth"] == 4)
    if gap < 0.02:
        ruling = ("增益来自参数量而非非线性（本对照臂 Δbpc 与 gated 臂差距 "
                  f"{gap} < 0.02）⇒ gated 结论降级为「加参数的同价效果」")
    elif d_linear >= -0.014 and d_gated <= -0.05:
        ruling = (f"非线性本身有效（本对照臂 Δbpc {d_linear} ≥ −0.014 而 gated "
                  f"{d_gated} ≤ −0.05）⇒ gated 结论加固")
    else:
        ruling = (f"中间情况（Δ_linear {d_linear}、Δ_gated {d_gated}、"
                  f"gap {gap}）⇒ 如实报，不裁决")
    out["verdict"] = {
        "criterion": "① |Δbpc(linear) − Δbpc(gated)| < 0.02 ⇒ 参数量解释；"
                     "② Δbpc(linear) ≥ −0.014 且 Δbpc(gated) ≤ −0.05 ⇒ 非线性有效",
        "A_clean_bpc": a_bpc, "gated_clean_bpc": g_bpc, "linear_clean_bpc": l_bpc,
        "delta_bpc_gated_vs_A": d_gated, "delta_bpc_linear_vs_A": d_linear,
        "delta_gap": gap,
        "A_tf4": a_tf4, "gated_tf4": g_tf4, "linear_tf4": l_tf4,
        "A_prob0_ece": out["A_arm"]["calibration"]["prob0_ece"],
        "gated_prob0_ece": out["gated_arm"]["calibration"]["prob0_ece"],
        "linear_prob0_ece": out["linear_arm"]["calibration"]["prob0_ece"],
        "dense_bpc_ref_13m": DENSE_BPC_13M,
        "A_tax_vs_dense": round(a_bpc - DENSE_BPC_13M, 4),
        "gated_tax_vs_dense": round(g_bpc - DENSE_BPC_13M, 4),
        "linear_tax_vs_dense": round(l_bpc - DENSE_BPC_13M, 4),
        "ruling": ruling,
    }
    # ---- 稳定性块（stability.md 数据源）----
    out["stability"] = stability_block()
    ece_by_depth = {"A_arm": out["A_arm"]["calibration"]["ece_by_depth"],
                    "gated_arm": out["gated_arm"]["calibration"]["ece_by_depth"],
                    "linear_arm": out["linear_arm"]["calibration"]["ece_by_depth"]}
    out["stability"]["ece_by_depth"] = ece_by_depth
    out["stability"]["gnorm_note"] = (
        "gnorm_monitor_all_steps 是 train_depth_ar 全程 12000 步的统计（真值）；"
        "gnorm_logged_* 只是每 200 步采样点上可见的值。")

    out["wall_total_min"] = round((time.time() - t_start) / 60, 2)
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out["verdict"], indent=2, ensure_ascii=False), flush=True)
    print(f"[done] total wall {out['wall_total_min']} min", flush=True)


if __name__ == "__main__":
    main()
