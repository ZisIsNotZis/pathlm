"""2612 · gated 读出 24k 收敛复测（两臂）· 13M · depth-ar —— 复现入口。

契约 C（`.scratch/29-verify-batch/issues/01-verify-batch.md`）：
把 2610 的 A 臂（tied 线性读出）与主臂（gated 残差读出）都跑到 **24000 步**，
其余配置逐字段一致。判据线预先写下、不得修改：

    Δbpc(24k) = bpc_gated − bpc_A（越低越好，负 = gated 更好）
      ≤ −0.05      ⇒ 增益在更长训练下保持（成功）
      |Δ| ≤ 0.014  ⇒ 12k 的增益是欠拟合伪影（撤回）
      中间         ⇒ 如实报，并附两臂末 1000 步 loss 斜率作为收敛证据

单因素：唯一变量 = 读出几何（`train_depth_ar.py --readout-gated` 开/关）。
两臂同 seed / 同超参 / 同数据切分 / 同步数（24k cosine LR 调度）/ 同评测电池。
两臂都在本脚本内**全新训练**（不是续训）：2610 的 A 臂是 12k 历史 checkpoint，
不含 optimizer state，无法从 12k 续训；两臂同为 fresh-24k 才能保证「逐字段一致」
（SCHEMA §9：任一字段不等 → 回退自包含双跑）。

- A 臂（基底）：`logits = h·Eᵀ`（tied 线性读出，readout_gated=False）；
- gated 臂（2610 主臂）：`h' = h + SiLU(W1 h) ⊙ (W2 h)` 再 `·Eᵀ`
  （W1 ~ N(0, 0.02) / W2 = 0 ⇒ 严格恒等起步、非鞍点；新增 2·d² = 131,072 参数）。

收敛证据（契约要求「同一方法」）：两臂都取训练日志（每 200 步一落）的**末 1000 步
loss 线性斜率**（末 6 个采样点最小二乘拟合，斜率 ×1000 = 每 1000 步的 loss 变化），
并另报末 25% / 末 50% 窗口与末深度 CE 斜率作为交叉核对。日志 loss 是单 batch 的
随机退出混合 loss，逐点噪声 ~±0.05，斜率只作「是否仍在大幅下降 / 有无发散」的证据。

用法（后台运行 + ≤120s 分次轮询日志）：

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup python3 run.py --steps 24000 --seed 0 --caps-min 50 > run.log 2>&1 &
    tail -n 40 run.log
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

# 12k 快照口径参照（2610/2611 同代码重测，用于把 24k Δ 接到 12k Δ 上）
SNAP_12K = {"A_clean_bpc": 1.8582, "gated_clean_bpc": 1.7770,
            "delta_gated_vs_A": -0.0812,
            "source": "2611-equalparams-linear-control-13m/results.json::verdict"
                      "（2610 的 12k checkpoint 在当前代码下重测）"}
# 同引擎 dense 13M matched 税分母（历史参照，口径见 analysis.md）
DENSE_BPC_13M = 1.5287
DATA = "data/enwik8_full.npz"
DEVICE = "cuda"
ARM_A = "A-24k-s0"
ARM_GATED = "gated-24k-s0"


def base_cfg(**kw) -> DepthARConfig:
    """契约骨架：d=256/L=8/h=8/seq256/mlp4，carry 门、随机退出、fill_kv。"""
    return DepthARConfig(d_model=256, n_layers=8, n_heads=8, seq_len=256,
                         mlp_mult=4, dense=False, fill_kv=True, **kw)


# ------------------------- prob0 ECE（与 2610/2611 同实现） -------------------------

@torch.no_grad()
def prob0_ece(model: DepthARModel, eval_arr, n_batches: int = 8,
              batch_size: int = 16, n_bins: int = 15, seed: int = 7) -> dict:
    """prob0 校准：每深度 softmax top-1 概率对 argmax 正确率的等频分箱 ECE。"""
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


# ------------------------------ 干预活性 / 读出诊断 ------------------------------

def intervention_liveness(model: DepthARModel) -> dict:
    """干预活性判据（SCHEMA §10）：训练结束时 W1/W2 必须显著非零，否则干预
    从未被激活（= 零初始鞍点，测不到机制）。A 臂无这两组参数。"""
    if not model.cfg.readout_gated:
        return {"readout_gated": False, "params_added": 0,
                "note": "A 臂（tied 线性）无读出状态变换参数"}
    W1 = model.readout_w1.weight.detach().float()
    W2 = model.readout_w2.weight.detach().float()
    n = W1.numel() + W2.numel()
    nz = (torch.count_nonzero(W1) + torch.count_nonzero(W2)).item()
    return {"readout_gated": True,
            "params_added": 2 * model.cfg.d_model ** 2,
            "w1_fro": round(float(W1.norm()), 4),
            "w2_fro": round(float(W2.norm()), 4),
            "w_absmax": round(float(max(W1.abs().max(), W2.abs().max())), 4),
            "nonzero_frac": round(nz / n, 6),
            "w1_spec_norm": round(float(torch.linalg.matrix_norm(W1, 2)), 4),
            "w2_spec_norm": round(float(torch.linalg.matrix_norm(W2, 2)), 4)}


@torch.no_grad()
def transform_rel_mag(model: DepthARModel, eval_arr, n_batches: int = 4,
                      batch_size: int = 8, seed: int = 11) -> list[float]:
    """读出变换相对幅度 ||transform(h)|| / ||h||，逐深度（干预是否只是小修正
    的量化证据）。A 臂 transform ≡ 0。"""
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
    out["readout_workspace_energy"] = readout_workspace_energy(
        model, eval_arr, device=DEVICE, n_batches=4, batch_size=8)
    out["generation"] = generation_quality(model, eval_arr, device=DEVICE,
                                           prompt_len=64, n_new=128)
    out["calibration"] = prob0_ece(model, eval_arr, n_batches=8, batch_size=16)
    out["transform_rel_mag_by_depth"] = transform_rel_mag(model, eval_arr)
    return out


# ------------------------------ 训练子进程 ------------------------------

def train_arm(run_name: str, gated: bool, steps: int, seed: int,
              caps_min: float) -> dict:
    """后台启动 train_depth_ar.py（两臂唯一差别 = --readout-gated），轮询日志 +
    nvidia-smi 峰值显存；超时/失败即中止并抛出。训练前清空本 run 的日志
    （train_depth_ar 是 append 语义；2610 曾因重跑把两段日志串在一起）。"""
    run_dir = os.path.join(HERE, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, "train_depth_ar.py", run_name,
           "--d", "256", "--layers", "8", "--heads", "8", "--seq-len", "256",
           "--mlp-mult", "4", "--steps", str(steps), "--batch-size", "32",
           "--seed", str(seed), "--exit-mode", "uniform",
           "--eval-batches", "20", "--out-root", os.path.join(HERE, "runs")]
    if gated:
        cmd.append("--readout-gated")
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] {run_name} training started pid={proc.pid} steps={steps} "
          f"gated={gated}", flush=True)
    peak, t0 = 0.0, time.time()
    log_path = os.path.join(run_dir, "train_log.jsonl")
    while proc.poll() is None:
        if time.time() - t0 > caps_min * 60:
            proc.terminate()
            raise TimeoutError(f"{run_name} exceeded {caps_min} min cap")
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
                if int(last["step"]) % 2000 == 0:
                    print(f"[run] {run_name} {lines[-1].strip()[:140]}",
                          flush=True)
        time.sleep(20)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"{run_name} training failed rc={proc.returncode}")
    # 训练日志入库（runs/ 被 .gitignore 覆盖）：拷到实验根供交付
    for name in ("train_log.jsonl", "gate_log.jsonl"):
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HERE, f"{name[:-6]}_{run_name}.jsonl"))
    shutil.copy(os.path.join(run_dir, "stdout.log"),
                os.path.join(HERE, f"stdout_{run_name}.txt"))
    wall = (time.time() - t0) / 60
    print(f"[run] {run_name} done wall={wall:.2f} min "
          f"peak_child_mem={peak:.2f}GB", flush=True)
    res = json.load(open(os.path.join(run_dir, "results.json")))
    return {"results": res, "wall_min": round(wall, 2),
            "peak_child_mem_gb": round(peak, 2)}


def load_arm(run_name: str, gated: bool) -> tuple[DepthARModel, dict]:
    hist = json.load(open(os.path.join(HERE, "runs", run_name, "results.json")))
    torch.manual_seed(0)
    model = DepthARModel(base_cfg(readout_gated=gated), hist["vocab_size"]).cuda()
    model.load_state_dict(torch.load(os.path.join(HERE, "runs", run_name,
                                                  "model.pt"),
                                     map_location="cuda"))
    return model, hist


# ---------------------- 逐字段对照（单因素门） ----------------------

def field_table(a_hist: dict, g_hist: dict) -> list[dict]:
    """两臂 resolved config 逐字段对照（SCHEMA §9）：唯一允许不等的字段 =
    readout_gated（单因素）；其余全部必须相等。"""
    cfg_a, tr_a = a_hist["config"], a_hist["train"]
    cfg_g, tr_g = g_hist["config"], g_hist["train"]
    rows = []
    for f in ("d_model", "n_layers", "n_heads", "seq_len", "mlp_mult",
              "dropout", "norm_cap", "gate_bias_init", "dense", "fill_kv",
              "proj_fill", "readout_dims",
              "readout_gated", "readout_linear_resid"):
        rows.append((f"config.{f}", cfg_a.get(f), cfg_g.get(f)))
    for f in ("steps", "batch_size", "lr", "seed"):
        rows.append((f"train.{f}", tr_a.get(f), tr_g.get(f)))
    for f in ("exit_mode", "fill_kv", "proj_fill", "aux_weight",
              "aux_anneal_to", "final_weight", "readout_dims", "readout_gated",
              "readout_linear_resid", "exit_anneal", "exit_anneal_to",
              "exit_anneal_frac", "final_anneal_to", "shallow_stopgrad",
              "dense_aux", "vocab_size", "train_tokens"):
        rows.append((f, a_hist.get(f), g_hist.get(f)))
    return [{"field": f, "A_arm": a, "gated_arm": b, "equal": a == b}
            for f, a, b in rows]


# ---------------------- 收敛证据（同一方法，两臂） ----------------------

def window_slope(rows: list[dict], key, start: int, end: int):
    """采样点最小二乘线性斜率 ×1000（每 1000 步的指标变化）。key 可为字段名
    或取值的函数（如 depth_ce[-1]）。"""
    pts = [(r["step"], key(r) if callable(key) else r[key])
           for r in rows if start <= r["step"] <= end]
    if len(pts) < 2:
        return None
    n = len(pts)
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((x - mx) ** 2 for x, _ in pts)
    num = sum((x - mx) * (y - my) for x, y in pts)
    return round(num / den * 1000, 4)


def window_mean(rows: list[dict], key, start: int, end: int):
    vals = [key(r) if callable(key) else r[key]
            for r in rows if start <= r["step"] <= end]
    return round(statistics.mean(vals), 4) if vals else None


def convergence_block(rows: list[dict], steps: int, gnorm_monitor) -> dict:
    """一臂的收敛/稳定性证据。末 1000 步 = 契约要求的收敛证据（两臂同方法）。"""
    loss = [r["loss"] for r in rows]
    inc = [b - a for a, b in zip(loss, loss[1:])]
    dce = lambda r: r["depth_ce"][-1]
    finite = all(torch.isfinite(torch.tensor(x)).item()
                 for x in loss + [r["gnorm"] for r in rows]
                 + [c for r in rows for c in r["depth_ce"]])
    return {
        "n_logged_samples": len(rows),
        "step_first": rows[0]["step"], "step_last": rows[-1]["step"],
        # —— 契约要求的收敛证据（同一方法：末 1000 步，末 6 个采样点最小二乘）——
        "loss_slope_last1000_per_1k": window_slope(rows, "loss",
                                                   steps - 1000, steps),
        "loss_slope_last25pct_per_1k": window_slope(rows, "loss",
                                                    0.75 * steps, steps),
        "loss_slope_last50pct_per_1k": window_slope(rows, "loss",
                                                    0.5 * steps, steps),
        "depthL_ce_slope_last1000_per_1k": window_slope(rows, dce,
                                                        steps - 1000, steps),
        "depthL_ce_slope_last25pct_per_1k": window_slope(rows, dce,
                                                         0.75 * steps, steps),
        # —— 12k→24k 轨迹（窗口均值，比单点斜率稳）——
        "loss_mean_12k_window": window_mean(rows, "loss", 11000, 12000),
        "loss_mean_24k_window": window_mean(rows, "loss", steps - 1000, steps),
        "depthL_ce_mean_12k_window": window_mean(rows, dce, 11000, 12000),
        "depthL_ce_mean_24k_window": window_mean(rows, dce, steps - 1000, steps),
        # —— 稳定性 ——
        "loss_first": loss[0], "loss_last": loss[-1],
        "loss_max_step_increase": round(max(inc), 4),
        "loss_jumps_gt_0.25": sum(1 for x in inc if x > 0.25),
        "all_finite": finite,
        "gnorm_monitor_all_steps": gnorm_monitor,
    }


# ------------------------------ 主流程 ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--caps-min", type=float, default=50.0,
                    help="每臂训练墙钟上限（分钟），超时中止")
    ap.add_argument("--eval-batches", type=int, default=20)
    args = ap.parse_args()

    t_start = time.time()
    torch.manual_seed(args.seed)
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", DATA)
    steps = args.steps

    cfg_dump = {
        "experiment": "2612-gated-readout-24k-13m",
        "ticket": "29-verify-batch", "contract": "C",
        "seed": args.seed, "steps": steps, "batch_size": 32, "lr": 0.001,
        "exit_mode": "uniform", "data": DATA,
        "data_split": "first 90M train / last 10M eval, vocab from full file",
        "vocab_size": vocab_size,
        "train_tokens": steps * 32 * 256,
        "A_arm": {"readout": "tied linear h·Eᵀ", "readout_gated": False,
                  "run": ARM_A},
        "gated_arm": {"readout": "h + SiLU(W1 h)·(W2 h) then ·Eᵀ",
                      "readout_gated": True,
                      "readout_w1_w2_init": "W1 ~ N(0, 0.02), W2 = 0 "
                                            "(identity start, non-saddle)",
                      "params_extra": 2 * 256 * 256, "run": ARM_GATED},
        "base": {"d_model": 256, "n_layers": 8, "n_heads": 8, "seq_len": 256,
                 "mlp_mult": 4, "dense": False, "fill_kv": True,
                 "proj_fill": False, "gate_bias_init": -2.0, "aux_weight": 1.0,
                 "final_weight": 1.0, "shallow_stopgrad": False,
                 "dense_aux": False},
        "gpu_wall_budget": "两臂 ≈50 min（共享卡 2× 折扣口径）",
    }
    json.dump(cfg_dump, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    out = {"experiment": cfg_dump["experiment"], "ticket": "29-verify-batch",
           "contract": "C", "config": cfg_dump}

    arms = {}
    for arm_name, gated in ((ARM_A, False), (ARM_GATED, True)):
        tr = train_arm(arm_name, gated, steps, args.seed, args.caps_min)
        model, hist = load_arm(arm_name, gated)
        t0 = time.time()
        batt = run_battery(model, eval_arr, args.eval_batches)
        arms[arm_name] = {
            "source": f"runs/{arm_name}", "run_name": arm_name,
            "readout_gated": gated,
            "params": hist["params"],
            "wall_min": tr["wall_min"],
            "peak_child_mem_gb": tr["peak_child_mem_gb"],
            "eval_wall_min": round((time.time() - t0) / 60, 2),
            "gnorm_monitor": hist.get("gnorm_monitor"),
            "avg_train_depth_ce": hist.get("avg_train_depth_ce"),
            **batt,
        }
        print(f"[{arm_name}] clean_bpc={batt['depth_curve'][-1]['bpc']} "
              f"tf4={next(r['accept'] for r in batt['tf_acceptance'] if r['depth']==4)} "
              f"params={hist['params']} wall={tr['wall_min']} min", flush=True)

    out["A_arm"] = arms[ARM_A]
    out["gated_arm"] = arms[ARM_GATED]
    # 干预活性量（SCHEMA §10）来自训练后的 gated checkpoint
    g_model, g_hist = load_arm(ARM_GATED, True)
    out["intervention"] = intervention_liveness(g_model)
    out["gated_arm"]["intervention"] = out["intervention"]

    # —— 单因素门：两臂逐字段对照 ——
    a_hist = json.load(open(os.path.join(HERE, "runs", ARM_A, "results.json")))
    out["arm_field_equivalence"] = field_table(a_hist, g_hist)
    out["arm_field_equivalence_all_equal_except_readout"] = all(
        r["equal"] for r in out["arm_field_equivalence"]
        if not r["field"].endswith("readout_gated"))

    # —— 收敛证据（同一方法，两臂）——
    def arm_rows(run_name):
        rows = [json.loads(l) for l in
                open(os.path.join(HERE, f"train_log_{run_name}.jsonl"))
                if l.strip()]
        return rows
    out["convergence"] = {
        "method": "末 1000 步 = 采样点（每 200 步一落）在 [steps−1000, steps] "
                  "区间的 loss 最小二乘线性斜率 ×1000；两臂同一方法。"
                  "另报末 25% / 末 50% 窗口与末深度 CE 斜率交叉核对。",
        "A_arm": convergence_block(arm_rows(ARM_A), steps,
                                   a_hist.get("gnorm_monitor")),
        "gated_arm": convergence_block(arm_rows(ARM_GATED), steps,
                                       g_hist.get("gnorm_monitor")),
    }
    out["convergence"]["loss_slope_last1000_delta"] = round(
        out["convergence"]["gated_arm"]["loss_slope_last1000_per_1k"]
        - out["convergence"]["A_arm"]["loss_slope_last1000_per_1k"], 4)

    # —— 判读（契约 C 判据线预先写下，不许事后调整）——
    a_bpc = out["A_arm"]["depth_curve"][-1]["bpc"]
    g_bpc = out["gated_arm"]["depth_curve"][-1]["bpc"]
    delta = round(g_bpc - a_bpc, 4)
    a_tf4 = next(r["accept"] for r in out["A_arm"]["tf_acceptance"]
                 if r["depth"] == 4)
    g_tf4 = next(r["accept"] for r in out["gated_arm"]["tf_acceptance"]
                 if r["depth"] == 4)
    a_ece = out["A_arm"]["calibration"]["prob0_ece"]
    g_ece = out["gated_arm"]["calibration"]["prob0_ece"]
    if delta <= -0.05:
        ruling = f"成功：Δbpc(24k) = {delta} ≤ −0.05，增益在更长训练下保持"
    elif abs(delta) <= 0.014:
        ruling = (f"撤回：|Δbpc(24k)| = {abs(delta)} ≤ 0.014，12k 的增益是"
                  f"欠拟合伪影")
    elif delta < -0.014:
        ruling = (f"中间情况（弱正向）：−0.05 < Δbpc(24k) = {delta} < −0.014，"
                  f"如实报，不裁决")
    else:
        ruling = (f"中间情况（增益反转）：Δbpc(24k) = {delta} > +0.014，如实报")
    out["verdict"] = {
        "criterion": "Δbpc(24k) ≤ −0.05 ⇒ 保持（成功）；|Δ| ≤ 0.014 ⇒ 撤回；"
                     "中间如实报（附两臂末 1000 步 loss 斜率）",
        "A_clean_bpc_24k": a_bpc, "gated_clean_bpc_24k": g_bpc,
        "delta_clean_bpc_24k": delta,
        "snapshot_12k_reference": SNAP_12K,
        "A_tf4": a_tf4, "gated_tf4": g_tf4,
        "A_prob0_ece": a_ece, "gated_prob0_ece": g_ece,
        "delta_ece": round(g_ece - a_ece, 4),
        "dense_bpc_ref_13m": DENSE_BPC_13M,
        "A_tax_vs_dense_24k": round(a_bpc - DENSE_BPC_13M, 4),
        "gated_tax_vs_dense_24k": round(g_bpc - DENSE_BPC_13M, 4),
        "params_A": out["A_arm"]["params"], "params_gated": out["gated_arm"]["params"],
        "params_extra": out["gated_arm"]["params"] - out["A_arm"]["params"],
        "ruling": ruling,
    }

    # —— 对账（SCHEMA §12：算术 / 任务 / 口径三道）——
    a_slope = out["convergence"]["A_arm"]["loss_slope_last1000_per_1k"]
    g_slope = out["convergence"]["gated_arm"]["loss_slope_last1000_per_1k"]
    out["reconciliation"] = {
        "arithmetic": {
            "params_A": out["A_arm"]["params"],
            "params_gated": out["gated_arm"]["params"],
            "params_extra": out["gated_arm"]["params"] - out["A_arm"]["params"],
            "params_extra_equals_2d2":
                (out["gated_arm"]["params"] - out["A_arm"]["params"])
                == 2 * 256 * 256,
            "delta_bpc_definition": "bpc_gated − bpc_A（负 = gated 更好）",
            "variance_bound": 0.014,
            "criterion_kept_threshold": -0.05,
            "criterion_withdraw_threshold": 0.014,
            "delta_bpc_24k": delta,
            "delta_bpc_12k_reference": SNAP_12K["delta_gated_vs_A"],
            "delta_change_12k_to_24k": round(
                delta - SNAP_12K["delta_gated_vs_A"], 4),
            "loss_slope_last1000_per_1k_A": a_slope,
            "loss_slope_last1000_per_1k_gated": g_slope,
            "slope_based_12k_extrapolation": {
                "note": "粗算术核对：若两臂末 1000 步斜率线性外推 12k 步，"
                        "Δ 的变化 ≈ (gated_slope − A_slope)×12；与实测 "
                        "delta_change_12k_to_24k 对照，判断 Δ 变化是否与"
                        "「两臂仍在以不同速度下降」一致。",
                "predicted_delta_change": round((g_slope - a_slope) * 12, 4),
                "observed_delta_change": round(
                    delta - SNAP_12K["delta_gated_vs_A"], 4),
                "caveat": "日志 loss 是单 batch 随机退出混合 loss，逐点噪声大，"
                          "该外推只作量级核对，不作判据。",
            },
        },
        "task": {
            "same_task": "同一 enwik8 char 级 AR 任务、同一 eval 切分（末 10M）",
            "same_position_set": "全序列逐位置、同一目标定义（tokens 自移位）",
            "same_battery": "两臂用同一 run.py::run_battery（同 n_batches/"
                            "seed/batch_size）",
            "deploy_correspondence":
                "两臂都是 depth-AR 的标准读出路径，无额外输入、无真值泄漏；"
                "对照只改读出前状态变换（gated 新增 2·d² 参数）",
        },
        "caliber": {
            "matched_step_snapshot": True, "steps": steps,
            "train_tokens": steps * 32 * 256, "seed": args.seed, "n_seed": 1,
            "note": "两臂同为 fresh 24k 步 / 196.6M token / seed 0 的 matched-step "
                    "快照口径（非渐近、非墙钟、非 FLOPs）；A 臂不是 2610 的 12k "
                    "历史 checkpoint 续训（无 optimizer state），而是同配方全新"
                    "训练——保证两臂逐字段一致。",
        },
        "background_slope_audit": {
            "note": "契约背景「主臂末 25% loss 斜率 −0.0298/1k、A 臂 −0.0054/1k」"
                    "口径可疑：−0.0054 确实是 A 臂末 25% 窗（9000–12000）的斜率，"
                    "但 −0.0298 是该损失在**末 50% 窗（6000–12000）**的斜率"
                    "（末 25% 窗同法得 −0.0039）。两腿窗口不一致 ⇒ 背景里「主臂"
                    "下降快 5.5×」的比较不成立。本实验用同一方法（末 1000 步）重算。",
            "recomputed_2610_12k": {
                "gated_loss_slope_last25pct_per_1k": -0.0039,
                "A_loss_slope_last25pct_per_1k": -0.0054,
                "gated_loss_slope_last50pct_per_1k": -0.0298,
                "A_loss_slope_last50pct_per_1k": -0.0333,
            },
        },
    }

    # —— 成本 ——
    peak = max(out["A_arm"]["peak_child_mem_gb"],
               out["gated_arm"]["peak_child_mem_gb"])
    out["cost"] = {
        "A_train_wall_min": out["A_arm"]["wall_min"],
        "gated_train_wall_min": out["gated_arm"]["wall_min"],
        "eval_wall_min": round(out["A_arm"]["eval_wall_min"]
                               + out["gated_arm"]["eval_wall_min"], 2),
        "peak_child_mem_gb": peak,
        "wall_total_min": round((time.time() - t_start) / 60, 2),
    }
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("verdict", "cost")}, indent=2,
                     ensure_ascii=False), flush=True)
    print(f"[done] total wall {out['cost']['wall_total_min']} min", flush=True)


if __name__ == "__main__":
    main()
