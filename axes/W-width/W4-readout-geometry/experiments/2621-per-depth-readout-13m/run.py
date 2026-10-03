"""2621 · 逐深度独立读出 vs 共享 tied 读出 · 13M · depth-ar —— 复现入口（分阶段）。

单因素（契约 W4，判据线预先写下未改）：读出是否逐深度独立。三臂共用同一配方
（全深度并行监督 + carry 门 + 随机退出 + fill_kv + 同 seed + 同 token 预算）：

- **A 臂**（基底，复用 23c-H2-w0-12k）：共享 tied 读出 `logits_k = h_k·Eᵀ`；
- **PD 臂**（主臂，全新训练）：逐深度独立读出——读出前逐深度线性变换
  `h_k → h_k + P_k(Q_k h_k)`（每个深度一套 d×d），再走共享 tied 表；
  新增参数 (L+1)·2d²；
- **EP 臂**（§15 等参数对照，全新训练）：单个共享线性变换 `h_k → h_k + A(B h_k)`，
  m = (L+1)·d ⇒ 新增参数 2·d·m = (L+1)·2d²（与 PD 名义等参数），无逐深度自由度。

另同口径重测 W4 既有参照臂（2610 gated / 2611 linear，各共享 2d²），把本臂放在
「共享小重参数化 → 逐深度」这条线上。

用法（**分阶段**，每阶段单独取 GPU 锁，见 SCHEMA §20.1）：
    python3 run.py --stage arms          # 复用臂电池（A/gated/linear），不需训练
    python3 run.py --stage perdepth      # 训练 PD + 电池
    python3 run.py --stage wide_shared   # 训练 EP + 电池
    python3 run.py --stage assemble      # 纯 CPU：合并 + 对账 + 判读（写 results.json）
"""

import argparse, json, os, shutil, statistics, subprocess, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, depth_curve,
                             tf_acceptance, workspace_probe, spec_cost_model,
                             generation_quality)
from pathlm.data import load_enwik8_full, batch
from pathlm.metrics import ece

# 复用臂（SCHEMA §9：开关关闭逐位等值单测 + 逐字段对照 + 口径标注三道门）
A_DIR = os.path.join(ROOT, ".scratch", "23-depth-ar", "evidence", "23c-H2-w0-12k")
GATED_RUN = os.path.join(ROOT, "axes", "W-width", "W4-readout-geometry",
                         "experiments", "2610-gated-readout-13m-depthar",
                         "runs", "gated-12k-s0")
LINEAR_RUN = os.path.join(ROOT, "axes", "W-width", "W4-readout-geometry",
                          "experiments", "2611-equalparams-linear-control-13m",
                          "runs", "linear-12k-s0")
DENSE_BPC_13M = 1.5287     # 同引擎 dense 13M matched 税分母（2611 口径）
DATA = "data/enwik8_full.npz"
DEVICE = "cuda"
D, L = 256, 8
STEPS = 12000
# PD 的逐深度变换 rank 与 EP 的宽中间维（m = (L+1)·d ⇒ 名义等参数）
EP_M = (L + 1) * D


def base_cfg(**kw) -> DepthARConfig:
    """契约 §2 骨架：d=256/L=8/h=8/seq256/mlp4，carry 门、随机退出、fill_kv。"""
    return DepthARConfig(d_model=D, n_layers=L, n_heads=8, seq_len=256,
                         mlp_mult=4, dense=False, fill_kv=True, **kw)


# ------------------------- prob0 ECE（与 2610/2611 同实现） -------------------------

@torch.no_grad()
def prob0_ece(model: DepthARModel, eval_arr, n_batches: int = 8,
              batch_size: int = 16, n_bins: int = 15, seed: int = 7) -> dict:
    gen = torch.Generator().manual_seed(seed)
    Ln = model.cfg.n_layers
    conf = [[] for _ in range(Ln + 1)]
    hit = [[] for _ in range(Ln + 1)]
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
                           n_bins=n_bins), 4) for k in range(Ln + 1)]
    d4 = 4 if 4 <= Ln else Ln
    return {"prob0_ece": per_depth[-1], "exit4_ece": per_depth[d4],
            "ece_by_depth": per_depth, "n_bins": n_bins}


# ------------------------------ 评测电池 ------------------------------

@torch.no_grad()
def readout_liveness(model: DepthARModel) -> dict:
    """干预活性判据（SCHEMA §10 强制项）：终值必须显著非零。"""
    if model.cfg.readout_perdepth:
        p = [float(m.weight.norm()) for m in model.readout_pd_p]
        q = [float(m.weight.norm()) for m in model.readout_pd_q]
        return {"mode": "perdepth",
                "p_fro_by_depth": [round(v, 4) for v in p],
                "q_fro_by_depth": [round(v, 4) for v in q],
                "p_fro_min": round(min(p), 4), "q_fro_min": round(min(q), 4),
                "alive": min(p) > 0 and min(q) > 0,
                "params_added": (L + 1) * 2 * D * D}
    if model.cfg.readout_wide_shared > 0:
        a = float(model.readout_shared_a.weight.norm())
        b = float(model.readout_shared_b.weight.norm())
        return {"mode": "wide_shared", "m": int(model.cfg.readout_wide_shared),
                "a_fro": round(a, 4), "b_fro": round(b, 4),
                "alive": a > 0 and b > 0,
                "params_added": 2 * D * int(model.cfg.readout_wide_shared)}
    if model.cfg.readout_gated:
        w1 = float(model.readout_w1.weight.norm())
        w2 = float(model.readout_w2.weight.norm())
        return {"mode": "gated", "w1_fro": round(w1, 4), "w2_fro": round(w2, 4),
                "alive": w1 > 0 and w2 > 0, "params_added": 2 * D * D}
    if model.cfg.readout_linear_resid:
        p = float(model.readout_p.weight.norm())
        q = float(model.readout_q.weight.norm())
        return {"mode": "linear_resid", "p_fro": round(p, 4),
                "q_fro": round(q, 4), "alive": p > 0 and q > 0,
                "params_added": 2 * D * D}
    return {"mode": "shared_tied", "alive": True, "params_added": 0}


@torch.no_grad()
def transform_rel_mag(model: DepthARModel, eval_arr, n_batches: int = 4,
                      batch_size: int = 8, seed: int = 11) -> list[float]:
    """读出变换相对幅度 ‖transform(h)‖ / ‖h‖，逐深度（干预是否只是小修正的量化
    证据）。per-depth 臂逐深度一套变换；其余臂共享一套（各深度数值仍不同）。"""
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
            elif model.cfg.readout_perdepth:
                t = model.readout_pd_p[k](model.readout_pd_q[k](h))
            elif model.cfg.readout_wide_shared > 0:
                t = model.readout_shared_a(model.readout_shared_b(h))
            else:
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
    out["generation"] = generation_quality(model, eval_arr, device=DEVICE,
                                           prompt_len=64, n_new=128)
    out["calibration"] = prob0_ece(model, eval_arr, n_batches=8, batch_size=16)
    out["transform_rel_mag_by_depth"] = transform_rel_mag(model, eval_arr)
    return out


# ------------------------------ 训练子进程 ------------------------------

def train_arm(name: str, flag: list[str], caps_min: float) -> dict:
    """后台启动 train_depth_ar.py，轮询日志 + nvidia-smi 峰值显存；超时/失败即中止。
    训练前清空本 run 的日志（train_depth_ar 是 append 语义）。"""
    run_dir = os.path.join(HERE, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, "train_depth_ar.py", name,
           "--d", str(D), "--layers", str(L), "--heads", "8", "--seq-len", "256",
           "--mlp-mult", "4", "--steps", str(STEPS), "--batch-size", "32",
           "--seed", "0", "--exit-mode", "uniform", *flag,
           "--eval-batches", "20", "--out-root", os.path.join(HERE, "runs")]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] {name} training started pid={proc.pid}", flush=True)
    peak, t0 = 0.0, time.time()
    log_path = os.path.join(run_dir, "train_log.jsonl")
    while proc.poll() is None:
        if time.time() - t0 > caps_min * 60:
            proc.terminate()
            raise TimeoutError(f"{name} exceeded {caps_min} min cap")
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
            if lines and int(json.loads(lines[-1])["step"]) % 2000 == 0:
                print(f"[run] {name} {lines[-1].strip()[:140]}", flush=True)
        time.sleep(30)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"{name} training failed rc={proc.returncode}")
    for nm in ("train_log.jsonl", "gate_log.jsonl"):
        src = os.path.join(run_dir, nm)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HERE, f"{name}.{nm}"))
    wall = (time.time() - t0) / 60
    print(f"[run] {name} done wall={wall:.2f} min peak_child_mem={peak:.2f}GB",
          flush=True)
    res = json.load(open(os.path.join(run_dir, "results.json")))
    return {"results": res, "wall_min": round(wall, 2),
            "peak_child_mem_gb": round(peak, 2)}


# ---------------------- 等值性门（逐字段对照） ----------------------

def reload_model(directory: str, **cfg_kw) -> tuple[DepthARModel, dict]:
    hist = json.load(open(os.path.join(directory, "results.json")))
    torch.manual_seed(0)
    model = DepthARModel(base_cfg(**cfg_kw), hist["vocab_size"]).cuda()
    model.load_state_dict(torch.load(os.path.join(directory, "model.pt"),
                                    map_location="cuda"))
    return model, hist


def equivalence_table(hist: dict, expected: dict) -> list[dict]:
    """对照臂等值性门（SCHEMA §9）：对照 run 的 resolved config 与期望值逐字段
    对照；唯一允许不等的字段是读出几何（单因素）。"""
    cfg, tr = hist["config"], hist["train"]
    rows = [
        ("d_model", cfg["d_model"], 256),
        ("n_layers", cfg["n_layers"], 8),
        ("n_heads", cfg["n_heads"], 8),
        ("seq_len", cfg["seq_len"], 256),
        ("mlp_mult", cfg["mlp_mult"], 4),
        ("dropout", cfg["dropout"], 0.0),
        ("norm_cap", cfg["norm_cap"], 1.0),
        ("gate_bias_init", cfg["gate_bias_init"], -2.0),
        ("dense", cfg["dense"], False),
        ("readout_dims", cfg.get("readout_dims"), None),
        ("exit_mode", hist["exit_mode"], "uniform"),
        ("fill_kv", hist.get("fill_kv", True), True),
        ("proj_fill", hist.get("proj_fill", False), False),
        ("aux_weight", hist["aux_weight"], 1.0),
        ("final_weight", hist.get("final_weight", 1.0), 1.0),
        ("shallow_stopgrad", hist["shallow_stopgrad"], False),
        ("dense_aux", hist["dense_aux"], False),
        ("steps", tr["steps"], STEPS),
        ("batch_size", tr["batch_size"], 32),
        ("lr", tr["lr"], 0.001),
        ("seed", tr["seed"], 0),
        ("train_tokens", hist["train_tokens"], STEPS * 32 * 256),
        ("vocab_size", hist["vocab_size"], 206),
    ]
    rows += [(f, hist.get(f, False), v) for f, v in expected.items()]
    # 读出几何四字段全列出来核（不只看本臂那一个）：臂自己的字段应等于其设定值、
    # 其余三个应为关（单因素：一次只开一个）。注：readout_wide_shared 缺省是 0（int）。
    for f in READOUT_FIELDS:
        want = expected.get(f, 0 if f == "readout_wide_shared" else False)
        rows.append((f, hist.get(f, 0 if f == "readout_wide_shared" else False), want))
    def _eq(h, t):
        return h == t and isinstance(h, bool) == isinstance(t, bool)
    return [{"field": f, "ref_arm": h, "expected": t, "equal": _eq(h, t),
             "source": "results.json"} for f, h, t in rows]


READOUT_FIELDS = ("readout_gated", "readout_linear_resid", "readout_perdepth",
                  "readout_wide_shared")


def equiv_ok(table: list[dict]) -> bool:
    """§9 等值性门：**所有**字段（含读出几何四字段）都必须等于本臂的预期设定。
    臂与臂之间的差异由各自的 `expected` 承载（唯一变量 = 读出几何）。"""
    return all(r["equal"] for r in table)


# ---------------------- 稳定性统计（时序数据源） ----------------------

def read_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def split_runs(rows: list[dict]) -> list[list[dict]]:
    runs, cur = [], []
    for r in rows:
        if cur and r["step"] <= cur[-1]["step"]:
            runs.append(cur)
            cur = []
        cur.append(r)
    if cur:
        runs.append(cur)
    return runs


def loss_slope(rows: list[dict], frac: float = 0.25) -> dict:
    """同方法斜率（SCHEMA §16）：末 frac 窗内对 loss 做最小二乘线性拟合，
    返回斜率（loss/1k step）与窗内首末 loss。三臂必须用同一 frac 与同一算法。"""
    n = len(rows)
    k0 = int(n * (1 - frac))
    win = rows[k0:]
    xs = [r["step"] / 1000.0 for r in win]
    ys = [r["loss"] for r in win]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0
    return {"window": f"last {frac:.0%} of logged samples",
            "step_first": win[0]["step"], "step_last": win[-1]["step"],
            "slope_loss_per_1k": round(slope, 5),
            "loss_first": win[0]["loss"], "loss_last": win[-1]["loss"]}


def stability_stats(rows: list[dict], gnorm_monitor: dict | None) -> dict:
    g = [r["gnorm"] for r in rows]
    loss = [r["loss"] for r in rows]
    med = statistics.median(g)
    inc = [b - a for a, b in zip(loss, loss[1:])]
    finite = all(torch.isfinite(torch.tensor(x)).item()
                 for x in g + loss + [c for r in rows for c in r["depth_ce"]])
    return {"n_logged_samples": len(rows),
            "step_first": rows[0]["step"], "step_last": rows[-1]["step"],
            "gnorm_logged_max": max(g), "gnorm_logged_median": round(med, 3),
            "gnorm_spikes_gt3x_median": sum(1 for x in g if x > 3 * med),
            "gnorm_spikes_gt5x_median": sum(1 for x in g if x > 5 * med),
            "all_finite": finite,
            "loss_first": loss[0], "loss_last": loss[-1],
            "loss_max_step_increase": round(max(inc), 4),
            "loss_jumps_gt_0.25": sum(1 for x in inc if x > 0.25),
            "gnorm_monitor_all_steps": gnorm_monitor,
            "loss_slope_last25pct": loss_slope(rows)}


# ------------------------------ 主流程 ------------------------------

def load_eval():
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", DATA)
    return eval_arr, vocab_size


def stage_arms(args):
    eval_arr, _ = load_eval()
    out = {}
    specs = [
        ("A", A_DIR, {}, "shared tied 读出 `h_k·Eᵀ`"),
        ("gated", GATED_RUN, {"readout_gated": True},
         "共享非线性重参数化 `h + SiLU(W1h)⊙(W2h)` 后 tied（2d²）"),
        ("linear", LINEAR_RUN, {"readout_linear_resid": True},
         "共享线性重参数化 `h + P(Qh)` 后 tied（2d²）"),
    ]
    for name, d, kw, desc in specs:
        model, hist = reload_model(d, **kw)
        table = equivalence_table(hist, kw)
        t0 = time.time()
        batt = run_battery(model, eval_arr, args.eval_batches)
        out[name] = {"source": os.path.relpath(d, ROOT), "readout": desc,
                     "equivalence": table, "equivalence_ok": equiv_ok(table),
                     "hist_clean_bpc": hist["depth_curve"][-1]["bpc"],
                     "params": hist["params"], **batt,
                     "eval_wall_min": round((time.time() - t0) / 60, 2)}
        json.dump(out[name], open(os.path.join(HERE, f"arm_{name}.json"), "w"),
                  indent=2, ensure_ascii=False)
        print(f"[{name}] clean_bpc={batt['depth_curve'][-1]['bpc']} "
              f"tf4={next(r['accept'] for r in batt['tf_acceptance'] if r['depth'] == 4)} "
              f"eq_ok={out[name]['equivalence_ok']} "
              f"({out[name]['eval_wall_min']} min)", flush=True)


def stage_train_arm(name: str, flag: list[str], cfg_kw: dict, desc: str, args):
    eval_arr, vocab_size = load_eval()
    tr = train_arm(name, flag, args.caps_min)
    res = tr["results"]
    torch.manual_seed(0)
    model = DepthARModel(base_cfg(**cfg_kw), vocab_size).cuda()
    model.load_state_dict(torch.load(
        os.path.join(HERE, "runs", name, "model.pt"), map_location="cuda"))
    t0 = time.time()
    batt = run_battery(model, eval_arr, args.eval_batches)
    out = {"source": f"runs/{name}", "readout": desc,
           "equivalence": equivalence_table(res, cfg_kw),
           "intervention": readout_liveness(model),
           "params": res["params"], **batt,
           "wall_min": tr["wall_min"], "peak_child_mem_gb": tr["peak_child_mem_gb"],
           "eval_wall_min": round((time.time() - t0) / 60, 2),
           "gnorm_monitor": res.get("gnorm_monitor"),
           "hist_clean_bpc": batt["depth_curve"][-1]["bpc"]}
    out["equivalence_ok"] = equiv_ok(out["equivalence"])
    json.dump(out, open(os.path.join(HERE, f"arm_{name}.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[{name}] clean_bpc={batt['depth_curve'][-1]['bpc']} "
          f"tf4={next(r['accept'] for r in batt['tf_acceptance'] if r['depth'] == 4)} "
          f"params={res['params']} liveness={out['intervention']['alive']} "
          f"wall={tr['wall_min']} min", flush=True)


def stage_assemble(args):
    arms = {}
    for name in ("A", "gated", "linear", "perdepth", "wide_shared"):
        p = os.path.join(HERE, f"arm_{name}.json")
        if not os.path.exists(p):
            print(f"[assemble] missing arm_{name}.json", flush=True)
            return
        arms[name] = json.load(open(p))
    a_bpc = arms["A"]["depth_curve"][-1]["bpc"]
    pd_bpc = arms["perdepth"]["depth_curve"][-1]["bpc"]
    ep_bpc = arms["wide_shared"]["depth_curve"][-1]["bpc"]
    g_bpc = arms["gated"]["depth_curve"][-1]["bpc"]
    l_bpc = arms["linear"]["depth_curve"][-1]["bpc"]
    d_pd = round(pd_bpc - a_bpc, 4)
    d_ep = round(ep_bpc - a_bpc, 4)
    d_g = round(g_bpc - a_bpc, 4)
    d_l = round(l_bpc - a_bpc, 4)
    tf = lambda nm, k: next(r["accept"] for r in arms[nm]["tf_acceptance"]
                            if r["depth"] == k)
    params_a = arms["A"]["params"]
    params_pd = arms["perdepth"]["params"]
    params_ep = arms["wide_shared"]["params"]
    out = {"experiment": "2621-per-depth-readout-13m",
           "config": json.load(open(os.path.join(HERE, "config.json"))),
           "arms": {k: {kk: vv for kk, vv in v.items()}
                    for k, v in arms.items()},
           "reconciliation": {
               "arithmetic": {
                   "params_A": params_a, "params_PD": params_pd,
                   "params_EP": params_ep,
                   "params_PD_minus_A": params_pd - params_a,
                   "params_EP_minus_A": params_ep - params_a,
                   "expected_extra": (L + 1) * 2 * D * D,
                   "equal_params": (params_pd - params_a) == (params_ep - params_a)
                                   == (L + 1) * 2 * D * D,
                   "variance_bound": 0.014,
                   "delta_bpc_PD_vs_A": d_pd, "delta_bpc_EP_vs_A": d_ep,
                   "delta_bpc_gated_vs_A": d_g, "delta_bpc_linear_vs_A": d_l,
                   "PD_minus_EP": round(d_pd - d_ep, 4)},
               "task": {
                   "same_task": "同一 enwik8 char 级 AR 任务、同一 eval 切分（末 10M）、"
                                "同一位置集（全序列逐位置）、同一目标定义（tokens 自移位）",
                   "same_battery": "全部臂用同一 run.py::run_battery（同 n_batches/"
                                   "seed/batch_size）",
                   "deploy_correspondence": "五臂都是 depth-AR 的标准读出路径，"
                                            "无额外输入、无真值泄漏；唯一差别 = 读出几何"},
               "caliber": {
                   "matched_step_snapshot": True, "steps": STEPS,
                   "train_tokens": STEPS * 32 * 256, "seed": 0, "n_seed": 1,
                   "note": "五臂同为 12k 步 / 98.3M token / seed 0 的快照口径"
                           "（非渐近、非墙钟、非 FLOPs）"}},
           }
    verdict = {
        "criterion": "C1 Δbpc(PD)≤−0.014 ⇒ 逐深度读出有效（支持 W4 归因）；"
                     "|Δbpc(PD)|<0.014 ⇒ 否证该归因。"
                     "C2 |Δbpc(PD)−Δbpc(EP)|<0.014 ⇒ 容量解释；"
                     "Δbpc(PD)≤Δbpc(EP)−0.014 ⇒ 逐深度独立性有效。",
        "A_clean_bpc": a_bpc, "PD_clean_bpc": pd_bpc, "EP_clean_bpc": ep_bpc,
        "gated_clean_bpc": g_bpc, "linear_clean_bpc": l_bpc,
        "delta_bpc_PD_vs_A": d_pd, "delta_bpc_EP_vs_A": d_ep,
        "delta_bpc_gated_vs_A": d_g, "delta_bpc_linear_vs_A": d_l,
        "PD_minus_EP": round(d_pd - d_ep, 4),
        "A_tf4": tf("A", 4), "PD_tf4": tf("perdepth", 4), "EP_tf4": tf("wide_shared", 4),
        "A_tf6": tf("A", 6), "PD_tf6": tf("perdepth", 6), "EP_tf6": tf("wide_shared", 6),
        "A_prob0_ece": arms["A"]["calibration"]["prob0_ece"],
        "PD_prob0_ece": arms["perdepth"]["calibration"]["prob0_ece"],
        "EP_prob0_ece": arms["wide_shared"]["calibration"]["prob0_ece"],
        "dense_bpc_ref_13m": DENSE_BPC_13M,
        "A_tax_vs_dense": round(a_bpc - DENSE_BPC_13M, 4),
        "PD_tax_vs_dense": round(pd_bpc - DENSE_BPC_13M, 4),
    }
    # C1
    if d_pd <= -0.014:
        verdict["C1"] = ("逐深度读出有效（Δbpc(PD)=%.4f ≤ −0.014）⇒ 支持"
                         "「共享读出是本形态缺陷」的 W4 归因" % d_pd)
    elif abs(d_pd) < 0.014:
        verdict["C1"] = ("逐深度读出无效（|Δbpc(PD)|=%.4f < 0.014）⇒ 否证"
                         "「共享读出是缺陷」的归因" % abs(d_pd))
    else:
        verdict["C1"] = "落在方差界内（Δbpc(PD)=%.4f）⇒ 不可排序" % d_pd
    # C2
    gap = abs(d_pd - d_ep)
    if gap < 0.014:
        verdict["C2"] = ("增益归因于容量（|Δbpc(PD)−Δbpc(EP)|=%.4f < 0.014）："
                         "共享的大变换已足够，逐深度独立性无额外贡献" % gap)
    elif d_pd <= d_ep - 0.014:
        verdict["C2"] = ("逐深度独立性本身有效（Δbpc(PD)=%.4f ≤ Δbpc(EP)−0.014="
                         "%.4f）" % (d_pd, d_ep - 0.014))
    else:
        verdict["C2"] = ("PD 明显坏于 EP（Δbpc(PD)=%.4f > Δbpc(EP)=%.4f）⇒ 逐深度"
                         "独立性引入税" % (d_pd, d_ep))
    verdict["C2_note"] = ("EP 的「名义等参数」靠冗余达成（一个 d×d 映射），因此 PD vs EP"
                          " 含「有效容量」混杂——见 analysis.md §6。")
    # C1/C2 的组合读数（两个判据各自的预登记分支不变，此处只把二者合起来说一次）：
    # C1 只问「逐深度读出相对 A 有没有效」；C2 是 §15 的归因判别（同名义参数的共享对照）。
    if d_pd <= -0.014 and d_pd > d_ep + 0.014:
        verdict["combined"] = (
            "C1 的字面分支（逐深度读出有效）成立，但 C2 的§15 等参数对照把所有增益归到"
            "容量：同名义参数的**共享**变换（EP）反而严格更好（Δbpc %.4f vs %.4f，差 "
            "%.4f ≥ 0.014）⇒ 「一个共享读出服务所有深度 = 本形态缺陷」这一具体机制"
            "**不被支持**（逐深度独立反而引入税）；能站住的只是弱化版 W4 归因"
            "「本形态的读出容量/重参数化不足」。" % (d_ep, d_pd, abs(d_ep - d_pd)))
    else:
        verdict["combined"] = "C1 与 C2 一致（见各自分支），无需额外调和。"
    out["verdict"] = verdict
    out["stability"] = {
        "A_arm": stability_stats(split_runs(read_rows(
            os.path.join(A_DIR, "train_log.jsonl")))[-1],
            arms["A"].get("gnorm_monitor")),
        "perdepth_arm": stability_stats(split_runs(read_rows(
            os.path.join(HERE, "runs", "perdepth", "train_log.jsonl")))[-1],
            arms["perdepth"].get("gnorm_monitor")),
        "wide_shared_arm": stability_stats(split_runs(read_rows(
            os.path.join(HERE, "runs", "wide_shared", "train_log.jsonl")))[-1],
            arms["wide_shared"].get("gnorm_monitor")),
        "note": "斜率用同一算法（末 25% 窗最小二乘）同 frac 计算；"
                "gnorm_monitor_all_steps 是全程真值，gnorm_logged_* 只是 200 步采样。"}
    out["readout_activity_by_200_steps"] = {
        nm: [r.get("readout") for r in read_rows(
             os.path.join(HERE, "runs", nm, "train_log.jsonl"))]
        for nm in ("perdepth", "wide_shared")}
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=2,
              ensure_ascii=False)
    print(json.dumps(verdict, indent=2, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["arms", "perdepth", "wide_shared", "assemble"])
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--caps-min", type=float, default=45.0)
    args = ap.parse_args()
    # config.json（resolved）只写一次
    cfg_path = os.path.join(HERE, "config.json")
    if not os.path.exists(cfg_path):
        json.dump({
            "experiment": "2621-per-depth-readout-13m", "seed": 0,
            "steps": STEPS, "batch_size": 32, "lr": 0.001, "data": DATA,
            "data_split": "first 90M train / last 10M eval, vocab from full file",
            "vocab_size": 206, "d_model": D, "n_layers": L, "n_heads": 8,
            "seq_len": 256, "mlp_mult": 4,
            "A_arm": {"readout": "shared tied h·Eᵀ",
                      "source": os.path.relpath(A_DIR, ROOT)},
            "perdepth_arm": {"readout": "h + P_k(Q_k h) per depth, then tied Eᵀ",
                             "init": "Q_k ~ N(0,0.02), P_k = 0 (identity start)",
                             "params_extra": (L + 1) * 2 * D * D},
            "wide_shared_arm": {"readout": "h + A(B h) shared, then tied Eᵀ",
                                "m": EP_M, "init": "B ~ N(0,0.02), A = 0",
                                "params_extra": 2 * D * EP_M,
                                "note": "§15 名义等参数对照，无逐深度自由度"},
            "base": {"dense": False, "fill_kv": True, "proj_fill": False,
                     "gate_bias_init": -2.0, "aux_weight": 1.0, "final_weight": 1.0,
                     "shallow_stopgrad": False, "dense_aux": False,
                     "exit_mode": "uniform"},
        }, open(cfg_path, "w"), indent=2, ensure_ascii=False)
    torch.manual_seed(0)
    if args.stage == "arms":
        stage_arms(args)
    elif args.stage == "perdepth":
        stage_train_arm("perdepth", ["--readout-perdepth"],
                        {"readout_perdepth": True},
                        "逐深度独立读出 h + P_k(Q_k h) 后 tied（(L+1)·2d²）", args)
    elif args.stage == "wide_shared":
        stage_train_arm("wide_shared", ["--readout-wide-shared", str(EP_M)],
                        {"readout_wide_shared": EP_M},
                        f"共享线性重参数化 h + A(Bh) 后 tied（m={EP_M}，(L+1)·2d²）",
                        args)
    else:
        stage_assemble(args)


if __name__ == "__main__":
    main()
