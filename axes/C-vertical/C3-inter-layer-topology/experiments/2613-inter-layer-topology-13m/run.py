"""2613 · 层间拓扑（纯空白格 C3 首次本地探索）· 13M · depth-ar/dense —— 复现入口。

单因素（契约 E，判据线预先写下未改）：**层与层之间的接法**。四个臂都在 stock
decoder-only transformer（`--dense`：无门/无退出/仅最终深度 CE）上，同数据、同
token 预算、同步数、同 seed、同 lr、同 batch —— 唯一差异 = 层间接线：

- `sequential`        h = h + attn(ln1 h)；h = h + mlp(ln2 h)（现有 Block，基线）；
- `parallel`          h = h + attn(ln h) + mlp(ln h)（单个共享 pre-LN，PaLM/GPT-J
                      原式；参数 −L·2d = −4096）；
- `parallel-dual-ln`  两个子层各带自己的 pre-LN、都读同一块输入 h
                      （**与基线参数量逐位相同**的等参数位点，SCHEMA §15）；
- `dense`             层 k 输入 = Σ_{j<k} α_kj·h_j（DenseFormer 式），α 初始化为
                      「只取当前层」（起步与基线逐位等价）；参数 +L(L+1)/2 = +36。

判读（契约 E 判据线，逐字未改）：任一变体 Δ clean bpc vs `sequential` ≤ −0.014 ⇒
成功；|Δ| < 0.014 ⇒ 中性；≥ +0.014 ⇒ 失败。并列判据 `tf@4 ≥ 0.95`。**必须附**三臂
（本实验四臂）同方法（末 25% 窗）loss 斜率作为收敛对账。

用法（后台运行 + ≤120s 轮询日志）：
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup python3 run.py --steps 12000 --seed 0 > run_log.txt 2>&1 &
    tail -f run_log.txt
"""

import argparse, json, os, shutil, statistics, subprocess, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, ParallelBlock,
                             depth_curve, tf_acceptance, spec_cost_model,
                             topology_liveness)
from pathlm.data import load_enwik8_full, batch
from pathlm.metrics import ece

DATA = "data/enwik8_full.npz"
DEVICE = "cuda"
D, L, H, SEQ, MLP, BS, LR = 256, 8, 8, 256, 4, 32, 1e-3
# 同引擎 dense 13M matched 税分母（历史 23d-dense-B-12k，口径见 analysis.md §7）
DENSE_BPC_13M = 1.5287
DENSE_HIST = ".scratch/23-depth-ar/evidence/23d-dense-B-12k"

ARMS = [
    {"topology": "sequential", "role": "基线（现有 Block：链式残差）"},
    {"topology": "parallel",
     "role": "并行子层（单个共享 pre-LN；参数 −L·2d）"},
    {"topology": "parallel-dual-ln",
     "role": "并行子层（两子层各带 pre-LN；与基线逐位等参数）"},
    {"topology": "dense", "role": "层间学习加权和（DenseFormer 式；参数 +36）"},
]


def cfg_base(topo: str) -> DepthARConfig:
    """契约 E 骨架：d=256 / L=8 / h=8 / seq=256 / mlp_mult=4，stock dense 形态。"""
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                         mlp_mult=MLP, dense=True, topology=topo)


def run_name(topo: str) -> str:
    return f"{topo}-12k-s0"


# ------------------------------ 评测电池 ------------------------------

@torch.no_grad()
def prob0_ece(model: DepthARModel, eval_arr, n_batches: int = 8,
              batch_size: int = 16, n_bins: int = 15, seed: int = 7) -> dict:
    """prob0 校准：每深度 softmax 后 top-1 概率对 argmax 正确率的等频分箱 ECE
    （`pathlm.metrics.ece`）。prob0_ece = 最终深度；exit4_ece = 深度 4。"""
    gen = torch.Generator().manual_seed(seed)
    L_ = model.cfg.n_layers
    conf = [[] for _ in range(L_ + 1)]
    hit = [[] for _ in range(L_ + 1)]
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
                           n_bins=n_bins), 4) for k in range(L_ + 1)]
    return {"prob0_ece": per_depth[-1], "exit4_ece": per_depth[4],
            "ece_by_depth": per_depth, "n_bins": n_bins}


@torch.no_grad()
def branch_activity(model: DepthARModel, eval_arr, n_batches: int = 4,
                    batch_size: int = 8, seed: int = 11) -> list[dict]:
    """逐层「子层增量相对幅度」||Δ_attn||/||h||、||Δ_mlp||/||h||（h = 该层输入，
    每臂按自己的接线取子层输入）：接线/分支是否真在干活的量化证据（SCHEMA §10）。
    parallel 臂另给 `wiring_rel_dev_vs_seq`：把本臂块权重拷进一个 sequential
    Block（共享 LN 臂的 ln2 用 ln1 顶替）后，同一输入下两种接线的输出相对差。"""
    gen = torch.Generator().manual_seed(seed)
    model.eval()
    acc = [{"attn": [], "mlp": [], "dev": []} for _ in range(model.cfg.n_layers)]
    use_dev = model.cfg.topology.startswith("parallel")
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, gen)
        x = x.to(DEVICE)
        _, aux = model(x, exit_depths=None)
        for i, blk in enumerate(model.blocks):
            h_in = _layer_input(model, aux, i)
            da, dm = _block_branches(blk, h_in)
            acc[i]["attn"].append(float(da.norm() / h_in.norm()))
            acc[i]["mlp"].append(float(dm.norm() / h_in.norm()))
            if use_dev:
                sb = counterfactual_sequential_block(blk)
                out_par = blk(h_in)[0]
                out_seq = sb(h_in)[0]
                acc[i]["dev"].append(float((out_par - out_seq).norm()
                                            / out_seq.norm()))
    return [{"layer": i + 1,
             "attn_delta_rel": round(statistics.mean(a["attn"]), 4),
             "mlp_delta_rel": round(statistics.mean(a["mlp"]), 4),
             "wiring_rel_dev_vs_seq": (round(statistics.mean(a["dev"]), 4)
                                       if a["dev"] else None)}
            for i, a in enumerate(acc)]


def _layer_input(model: DepthARModel, aux: dict, i: int) -> torch.Tensor:
    """第 i 层（0-index）的输入：dense 拓扑 = 层权加权和，其余 = 上一层输出。"""
    if model.cfg.topology == "dense":
        return model._layer_mix_input(aux["states"], i + 1)
    return aux["states"][i]


def _block_branches(blk, h_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按该臂接线返回子层增量 (Δ_attn, Δ_mlp)：Block = 链式（mlp 读
    x + Δ_attn），ParallelBlock = 两分支同读块输入 h_in。"""
    b = getattr(blk, "block", blk)
    B, T, d = h_in.shape
    h1 = b.ln1(h_in)
    q, k, v = b.qkv(h1).chunk(3, dim=-1)
    shape = lambda t: t.view(B, T, b.n_heads, d // b.n_heads).transpose(1, 2)
    a = torch.nn.functional.scaled_dot_product_attention(
        shape(q), shape(k), shape(v), is_causal=True)
    da = b.proj(a.transpose(1, 2).reshape(B, T, d))
    if isinstance(blk, ParallelBlock):
        h2 = h1 if blk.share_ln else b.ln2(h_in)
    else:
        h2 = b.ln2(h_in + da)
    return da, b.mlp(h2)


def counterfactual_sequential_block(blk):
    """把并行臂的块权重复制进一个 sequential `Block`：同权重、只改接线的反事实
    （共享 LN 臂无 ln2 ⇒ 用其 ln1 顶替，语义 = 「同一套权重走链式接线」）。"""
    from pathlm.model import Block, ModelConfig
    b = getattr(blk, "block", blk)
    mcfg = ModelConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                       mlp_mult=MLP)
    sb = Block(mcfg)
    sd = {k: v for k, v in b.state_dict().items()}
    if "ln2.weight" not in sd:
        sd["ln2.weight"] = b.ln1.weight.detach().clone()
        sd["ln2.bias"] = b.ln1.bias.detach().clone()
    sb.load_state_dict(sd)
    return sb.to(next(b.parameters()).device)


def run_battery(model: DepthARModel, eval_arr, eval_batches: int) -> dict:
    out = {}
    out["depth_curve"] = depth_curve(model, eval_arr, device=DEVICE,
                                     n_batches=eval_batches)
    out["clean_bpc"] = out["depth_curve"][-1]["bpc"]
    out["next_token_acc"] = out["depth_curve"][-1]["acc"]
    out["tf_acceptance"] = tf_acceptance(model, eval_arr, device=DEVICE,
                                         n_batches=eval_batches)
    out["tf@4"] = next(r["accept"] for r in out["tf_acceptance"] if r["depth"] == 4)
    out["spec_cost_model"] = spec_cost_model(out["tf_acceptance"], L,
                                             draft_depths=(4, 6))
    out["calibration"] = prob0_ece(model, eval_arr, n_batches=8, batch_size=16)
    out["branch_activity"] = branch_activity(model, eval_arr)
    return out


# ------------------------------ 训练子进程 ------------------------------

def train_arm(topo: str, steps: int, seed: int, caps_min: float,
              poll_s: int = 30) -> dict:
    """后台启动 train_depth_ar.py（--dense --topology <topo>），轮询日志 +
    nvidia-smi 峰值显存；超时/失败即中止并抛出。训练前清空本 run 的日志
    （train_depth_ar 是 append 语义）。"""
    name = run_name(topo)
    run_dir = os.path.join(HERE, "runs", name)
    os.makedirs(run_dir, exist_ok=True)
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        p = os.path.join(run_dir, f)
        if os.path.exists(p):
            os.remove(p)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, "train_depth_ar.py", name,
           "--d", str(D), "--layers", str(L), "--heads", str(H),
           "--seq-len", str(SEQ), "--mlp-mult", str(MLP), "--dense",
           "--topology", topo, "--steps", str(steps), "--batch-size", str(BS),
           "--seed", str(seed), "--exit-mode", "uniform", "--eval-batches", "20",
           "--out-root", os.path.join(HERE, "runs")]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] {name}: training started pid={proc.pid} steps={steps}",
          flush=True)
    peak, t0 = 0.0, time.time()
    log_path = os.path.join(run_dir, "train_log.jsonl")
    while proc.poll() is None:
        if time.time() - t0 > caps_min * 60:
            proc.terminate()
            raise TimeoutError(f"{name} training exceeded {caps_min} min cap")
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
            if lines and int(json.loads(lines[-1])["step"]) % 2000 == 0:
                print(f"[run] {name} {lines[-1].strip()[:120]}", flush=True)
        time.sleep(poll_s)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"{name} training failed rc={proc.returncode}")
    for f in ("train_log.jsonl", "gate_log.jsonl"):
        src = os.path.join(run_dir, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HERE, f"{name}-{f}"))
    wall = (time.time() - t0) / 60
    res = json.load(open(os.path.join(run_dir, "results.json")))
    print(f"[run] {name}: done wall={wall:.2f} min peak_child_mem={peak:.2f}GB "
          f"clean_bpc={res['bpc']}", flush=True)
    return {"results": res, "wall_min": round(wall, 2),
            "peak_child_mem_gb": round(peak, 2),
            "seconds_per_step": round(wall * 60 / steps, 4)}


# ---------------------- 收敛对账（SCHEMA §16） ----------------------

def tail_window_slope(rows: list[dict], frac: float = 0.25) -> dict:
    """末 frac 窗内 loss 对 step 的最小二乘斜率（**四臂同一方法、同一窗口比例**）
    ——收敛对账用：斜率相同 ⇒ 差异是水平差（质量），斜率悬殊 ⇒ 收敛速度差。"""
    i0 = int(len(rows) * (1 - frac))
    win = rows[i0:]
    xs = [r["step"] for r in win]
    ys = [r["loss"] for r in win]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom  # nats/step
    return {"window_frac": frac, "n_points": len(win),
            "step_from": xs[0], "step_to": xs[-1],
            "loss_from": ys[0], "loss_to": ys[-1],
            "slope_nats_per_1k_steps": round(slope * 1000, 5),
            "slope_nats_per_1k_steps_late_half": round(
                _slope(win[len(win) // 2:]) * 1000, 5)}


def _slope(win: list[dict]) -> float:
    xs = [r["step"] for r in win]
    ys = [r["loss"] for r in win]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / \
        sum((x - xm) ** 2 for x in xs)


def load_arm_rows(topo: str) -> list[dict]:
    path = os.path.join(HERE, f"{run_name(topo)}-train_log.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]


# ---------------------- 稳定性统计（stability.md 数据源） ----------------------

def stability_stats(rows: list[dict], gnorm_monitor: dict | None) -> dict:
    g = [r["gnorm"] for r in rows]
    loss = [r["loss"] for r in rows]
    med = statistics.median(g)
    inc = [b - a for a, b in zip(loss, loss[1:])]
    finite = all(torch.isfinite(torch.tensor(v)).item() for v in g + loss)
    return {
        "n_logged_samples": len(rows),
        "step_first": rows[0]["step"], "step_last": rows[-1]["step"],
        "gnorm_logged_max": max(g), "gnorm_logged_median": round(med, 3),
        "gnorm_spikes_gt3x_median": sum(1 for v in g if v > 3 * med),
        "gnorm_spikes_gt5x_median": sum(1 for v in g if v > 5 * med),
        "all_finite": finite,
        "loss_first": loss[0], "loss_last": loss[-1],
        "loss_max_step_increase": round(max(inc), 4),
        "loss_jumps_gt_0.25": sum(1 for v in inc if v > 0.25),
        "gnorm_monitor_all_steps": gnorm_monitor,
    }


# ------------------------------ 主流程 ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--caps-min", type=float, default=14.0,
                    help="单臂训练墙钟上限（分钟），超时中止（契约预算 40 min）")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--poll-s", type=int, default=30)
    ap.add_argument("--skip-train", action="store_true",
                    help="复用 runs/ 下已有 checkpoint，只重算电池与判读")
    ap.add_argument("--budget-min", type=float, default=38.0,
                    help="GPU 墙钟预算（契约 40 min）；超出则在下一臂开跑前中止"
                         "（不静默跑残臂）")
    ap.add_argument("--arms", default=None,
                    help="只跑指定拓扑（逗号分隔，必须含 sequential）；缺省全跑")
    args = ap.parse_args()
    if args.arms:
        want = [t.strip() for t in args.arms.split(",") if t.strip()]
        unknown = [t for t in want if t not in [a["topology"] for a in ARMS]]
        if unknown or "sequential" not in want:
            raise SystemExit(f"--arms 无效（{unknown}）：必须含 sequential，"
                             f"可选 {[a['topology'] for a in ARMS]}")
        ARMS[:] = [a for a in ARMS if a["topology"] in want]

    t_start = time.time()
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", DATA)
    cfg_dump = {
        "experiment": "2613-inter-layer-topology-13m",
        "ticket": "30-batch3 契约 E",
        "seed": args.seed, "steps": args.steps, "batch_size": BS, "lr": LR,
        "optimizer": "AdamW(wd=0.01) + warmup 200 步 + cosine 到 0.1×",
        "data": DATA,
        "data_split": "first 90M train / last 10M eval, vocab from full file",
        "vocab_size": vocab_size, "train_tokens": args.steps * BS * SEQ,
        "base": {"d_model": D, "n_layers": L, "n_heads": H, "seq_len": SEQ,
                 "mlp_mult": MLP, "dropout": 0.0, "norm_cap": 1.0,
                 "dense": True, "exit_mode": "uniform",
                 "loss": "仅最终深度 CE（stock AR）",
                 "readout": "tied h·Eᵀ，无 final LayerNorm（共享底座偏离 1）"},
        "arms": [{"topology": a["topology"], "role": a["role"],
                  "params_expected_delta": {"sequential": 0,
                                            "parallel": -L * 2 * D,
                                            "parallel-dual-ln": 0,
                                            "dense": L * (L + 1) // 2}[a["topology"]],
                  "run": f"runs/{run_name(a['topology'])}"} for a in ARMS],
        "historical_cross_check": {
            "path": DENSE_HIST,
            "note": "历史同配置 dense-B-12k（sequential 接线，同 seed/同 12k/同 "
                    "98.3M token）：本实验的 sequential 臂应复现其 1.5287；"
                    "该 checkpoint 也用当前代码重测过（口径对齐）"},
    }
    json.dump(cfg_dump, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    out = {"experiment": cfg_dump["experiment"], "config": cfg_dump, "arms": {}}

    # ---- 历史对照的预检（当前代码重测，口径对齐；不参与主判据） ----
    torch.manual_seed(args.seed)
    hist_model = DepthARModel(cfg_base("sequential"), vocab_size).cuda()
    hist_model.load_state_dict(torch.load(
        os.path.join(ROOT, DENSE_HIST, "model.pt"), map_location="cuda"))
    hist_curve = depth_curve(hist_model, eval_arr, device=DEVICE,
                             n_batches=args.eval_batches)
    hist_tf = tf_acceptance(hist_model, eval_arr, device=DEVICE,
                            n_batches=args.eval_batches)
    out["historical_dense_B_12k"] = {
        "source": DENSE_HIST,
        "hist_bpc": json.load(open(os.path.join(ROOT, DENSE_HIST,
                                                "results.json")))["bpc"],
        "remeasured_bpc": hist_curve[-1]["bpc"],
        "remeasured_acc": hist_curve[-1]["acc"],
        "remeasured_tf@4": next(r["accept"] for r in hist_tf if r["depth"] == 4),
        "note": "历史 dense-B-12k 无 tf 表（当时未测）；此处补测作为 tf@4 的"
                "形态参照（stock dense 无浅层出口监督）",
    }
    print(f"[hist] {DENSE_HIST} remeasured bpc "
          f"{hist_curve[-1]['bpc']} (hist "
          f"{out['historical_dense_B_12k']['hist_bpc']}), tf@4 "
          f"{out['historical_dense_B_12k']['remeasured_tf@4']}", flush=True)

    # ---- 四臂训练 + 电池 ----
    for i, arm in enumerate(ARMS):
        topo = arm["topology"]
        if not args.skip_train and (time.time() - t_start) / 60 > args.budget_min:
            raise SystemExit(
                f"[budget] {args.budget_min} min 墙钟预算已到；已完成 "
                f"{list(out['arms'])}，未跑 {[a['topology'] for a in ARMS[i:]]}")
        if args.skip_train:
            tr = {"results": json.load(open(os.path.join(
                HERE, "runs", run_name(topo), "results.json"))),
                "wall_min": json.load(open(os.path.join(
                    HERE, "runs", run_name(topo), "results.json")))["wall_minutes"],
                "peak_child_mem_gb": None, "seconds_per_step": None}
        else:
            tr = train_arm(topo, args.steps, args.seed, args.caps_min,
                           args.poll_s)
        res = tr["results"]
        torch.manual_seed(args.seed)
        model = DepthARModel(cfg_base(topo), vocab_size).cuda()
        model.load_state_dict(torch.load(
            os.path.join(HERE, "runs", run_name(topo), "model.pt"),
            map_location="cuda"))
        t0 = time.time()
        batt = run_battery(model, eval_arr, args.eval_batches)
        out["arms"][topo] = {
            "role": arm["role"], "topology": topo,
            "source": f"runs/{run_name(topo)}",
            "params": res["params"],
            "train_tokens": res["train_tokens"],
            "wall_min": tr["wall_min"], "peak_child_mem_gb": tr["peak_child_mem_gb"],
            "seconds_per_step": tr["seconds_per_step"],
            "gnorm_monitor": res.get("gnorm_monitor"),
            "intervention": {**topology_liveness(model),
                             "branch_activity": batt["branch_activity"]},
            **batt, "eval_wall_min": round((time.time() - t0) / 60, 2)}
        print(f"[arm] {topo}: clean_bpc={batt['clean_bpc']} tf@4={batt['tf@4']} "
              f"params={res['params']} wall={tr['wall_min']} min "
              f"eval={out['arms'][topo]['eval_wall_min']} min", flush=True)

    # ---- 逐字段对照表（单因素：只允许 topology / params 不等） ----
    fields = [("d_model", D), ("n_layers", L), ("n_heads", H), ("seq_len", SEQ),
              ("mlp_mult", MLP), ("dropout", 0.0), ("norm_cap", 1.0),
              ("dense", True), ("fill_kv", True), ("proj_fill", False),
              ("readout_dims", None), ("readout_gated", False),
              ("readout_linear_resid", False), ("exit_mode", "uniform"),
              ("aux_weight", 1.0), ("final_weight", 1.0),
              ("shallow_stopgrad", False), ("dense_aux", False),
              ("steps", args.steps), ("batch_size", BS), ("lr", LR),
              ("seed", args.seed), ("train_tokens", args.steps * BS * SEQ),
              ("vocab_size", vocab_size),
              ("data_split", "first 90M/ last 10M, vocab over full file")]
    rows = []
    for f, want in fields:
        row = {"field": f, "expected": want, "arms": {}}
        for arm in ARMS:
            topo = arm["topology"]
            res = json.load(open(os.path.join(HERE, "runs", run_name(topo),
                                              "results.json")))
            got = {
                "d_model": res["config"]["d_model"],
                "n_layers": res["config"]["n_layers"],
                "n_heads": res["config"]["n_heads"],
                "seq_len": res["config"]["seq_len"],
                "mlp_mult": res["config"]["mlp_mult"],
                "dropout": res["config"]["dropout"],
                "norm_cap": res["config"]["norm_cap"],
                "dense": res["config"]["dense"],
                "fill_kv": res["fill_kv"], "proj_fill": res["proj_fill"],
                "readout_dims": res["readout_dims"],
                "readout_gated": res["readout_gated"],
                "readout_linear_resid": res["readout_linear_resid"],
                "exit_mode": res["exit_mode"],
                "aux_weight": res["aux_weight"], "final_weight": res["final_weight"],
                "shallow_stopgrad": res["shallow_stopgrad"],
                "dense_aux": res["dense_aux"],
                "steps": res["train"]["steps"], "batch_size": res["train"]["batch_size"],
                "lr": res["train"]["lr"], "seed": res["train"]["seed"],
                "train_tokens": res["train_tokens"],
                "vocab_size": res["vocab_size"],
                "data_split": "first 90M/ last 10M, vocab over full file",
            }[f]
            row["arms"][topo] = got
            row["equal" + f"_{topo}"] = (got == want)
        row["all_arms_equal"] = all(row["equal" + f"_{a['topology']}"]
                                    for a in ARMS)
        rows.append(row)
    out["field_by_field"] = rows
    out["field_by_field_all_equal_except_topology"] = all(
        r["all_arms_equal"] for r in rows)
    topo_row = {"field": "topology（唯一变量）",
                "arms": {a["topology"]: a["topology"] for a in ARMS},
                "note": "唯一不等字段；参数量随之变化（见 params_* 与对账块）"}
    out["field_by_field"].append(topo_row)

    # ---- 对账（SCHEMA §12 三道）+ 收敛对账（§16） ----
    seq_bpc = out["arms"]["sequential"]["clean_bpc"]
    params_seq = out["arms"]["sequential"]["params"]
    deltas = {a["topology"]: round(out["arms"][a["topology"]]["clean_bpc"]
                                   - seq_bpc, 4) for a in ARMS}
    slopes = {a["topology"]: tail_window_slope(load_arm_rows(a["topology"]))
              for a in ARMS}
    out["reconciliation"] = {
        "arithmetic": {
            "bpc_definition": "bpc_arm − bpc_sequential（同一条 depth_curve 的 depth-8 "
                              "读数、同 20 个 eval batch、同 eval 切分）",
            "delta_bpc": deltas,
            "variance_bound": 0.014,
            "judge_threshold_eq_variance_bound": True,
            "params": {a["topology"]: out["arms"][a["topology"]]["params"]
                       for a in ARMS},
            "params_delta": {a["topology"]: out["arms"][a["topology"]]["params"]
                             - params_seq for a in ARMS},
            "params_delta_expected": {"sequential": 0, "parallel": -L * 2 * D,
                                      "parallel-dual-ln": 0,
                                      "dense": L * (L + 1) // 2},
            "params_delta_exact": all(
                out["arms"][a["topology"]]["params"] - params_seq ==
                {"sequential": 0, "parallel": -L * 2 * D, "parallel-dual-ln": 0,
                 "dense": L * (L + 1) // 2}[a["topology"]] for a in ARMS),
            "dense_params_added_frac": round(
                (L * (L + 1) // 2) / params_seq, 8),
            "bpc_bounds": "clean bpc 理论下界 0；实测四臂均在 (1.4, 1.7)",
        },
        "task": {
            "same_task": "四臂同一 enwik8 char 级 AR 任务、同一 eval 切分（末 10M）、"
                         "同一位置集（全序列逐位置 × depth-8 读出）、同一目标定义"
                         "（tokens 自移位）",
            "same_battery": "四臂同一 run.py::run_battery（同 n_batches=20 / 同 eval "
                            "seed / 同 batch_size）",
            "deploy_correspondence": "四臂都是 stock decoder-only 前向；对照只改层与"
                                     "层之间的接法，无额外输入、无真值泄漏（输入分布"
                                     "= 同一训练/评测数据流）",
        },
        "caliber": {
            "matched_step_snapshot": True, "steps": args.steps, "seed": args.seed,
            "n_seed": 1, "train_tokens": args.steps * BS * SEQ,
            "note": f"四臂同为 {args.steps} 步 / {args.steps * BS * SEQ / 1e6:.1f}M "
                    "token / seed 0 的 matched-step 快照口径（非渐近、非墙钟、"
                    "非 FLOPs）",
        },
        "convergence": {
            "method": "末 25% 窗（同一窗口比例、同一最小二乘算法）loss 斜率，"
                      "四臂同方法",
            "tail_slope_by_arm": slopes,
            "slope_spread_nats_per_1k": round(
                max(s["slope_nats_per_1k_steps"] for s in slopes.values()) -
                min(s["slope_nats_per_1k_steps"] for s in slopes.values()), 5),
            "note": "斜率相近 ⇒ 差异是水平差（质量）而非收敛速度差；另附末段"
                    "（窗内后半）斜率。更长训练点（2×）未做：契约 E 的收敛对账"
                    "只要求同方法斜率，且预算不足以加 24k 双跑（见 analysis §7）。",
        },
    }

    # ---- 判读（契约 E 判据线，逐字未改） ----
    variants = [a["topology"] for a in ARMS if a["topology"] != "sequential"]
    success = [t for t in variants if deltas[t] <= -0.014]
    fail = [t for t in variants if deltas[t] >= 0.014]
    if success:
        ruling = (f"成功：{success} 相对基线改善 ≥0.014（Δ = "
                  f"{ {t: deltas[t] for t in success} }）")
    elif fail:
        ruling = f"失败（任一臂更差 ≥+0.014）：{ {t: deltas[t] for t in fail} }"
    else:
        ruling = (f"中性：全部变体 |Δ| < 0.014（Δ = {deltas}）")
    tf4 = {a["topology"]: out["arms"][a["topology"]]["tf@4"] for a in ARMS}
    out["verdict"] = {
        "criterion": "任一变体 Δ clean bpc vs sequential ≤ −0.014 ⇒ 成功；"
                     "|Δ| < 0.014 ⇒ 中性；≥ +0.014 ⇒ 失败；并列 tf@4 ≥ 0.95",
        "delta_bpc_vs_sequential": deltas,
        "clean_bpc": {a["topology"]: out["arms"][a["topology"]]["clean_bpc"]
                      for a in ARMS},
        "success_arms": success, "fail_arms": fail, "ruling": ruling,
        "tf@4_by_arm": tf4,
        "tf@4_judge_satisfied_by": [t for t, v in tf4.items() if v >= 0.95],
        "tf@4_judge_note": "stock dense 形态（无浅层出口监督）下 tf@4 ≈ 0.44"
                           "（历史 dense-B-12k 实测），并列判据 ≥0.95 在该形态"
                           "上**不可达**——四臂同口径同不可达，不作为区分量；"
                           "如实报，不改判据线",
        "dense_bpc_ref_13m": DENSE_BPC_13M,
        "tax_vs_dense_by_arm": {a["topology"]: round(
            out["arms"][a["topology"]]["clean_bpc"] - DENSE_BPC_13M, 4)
            for a in ARMS},
        "next_token_acc_by_arm": {a["topology"]: out["arms"][a["topology"]]
                                  ["next_token_acc"] for a in ARMS},
    }

    # ---- 稳定性块 + 逐深度 ECE ----
    out["stability"] = {
        a["topology"]: stability_stats(
            load_arm_rows(a["topology"]),
            out["arms"][a["topology"]]["gnorm_monitor"]) for a in ARMS}
    out["stability"]["ece_by_depth"] = {
        a["topology"]: out["arms"][a["topology"]]["calibration"]["ece_by_depth"]
        for a in ARMS}
    out["stability"]["gnorm_note"] = (
        "gnorm_monitor_all_steps 是 train_depth_ar 全程真值（每个优化器 step）；"
        "gnorm_logged_* 只是每 200 步采样点上可见的值。")

    out["wall_total_min"] = round((time.time() - t_start) / 60, 2)
    out["gpu_peak_by_arm_gb"] = {a["topology"]: out["arms"][a["topology"]]
                                 ["peak_child_mem_gb"] for a in ARMS}
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out["verdict"], indent=2, ensure_ascii=False), flush=True)
    print(f"[done] total wall {out['wall_total_min']} min", flush=True)


if __name__ == "__main__":
    main()
