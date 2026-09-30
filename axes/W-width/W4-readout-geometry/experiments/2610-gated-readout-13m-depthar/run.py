"""2610 · gated 残差读出（保 tied 桥）· 13M · depth-ar —— 复现入口。

单因素：只改读出几何。基线 A = tied 线性读出（logits = h·Eᵀ）；主臂 =
gated 残差读出（logits = (h + SiLU(W1 h) ⊙ (W2 h))·Eᵀ）。其余配置（全深度
并行监督 + carry 门 + 随机退出 + fill_kv + 同 seed + 同 token 预算）逐位一致。

A 臂按契约 §5「同引擎同配方同 token 预算」用历史同配置对照
（`.scratch/23-depth-ar/evidence/23c-H2-w0-12k`，d=256/L=8/bs32/seq256/12k/seed0/
fill；见 analysis.md §1 等值性门），并在本脚本下用**当前评测代码重测电池**
（含 prob0 ECE），以与主臂口径对齐。主臂全新训练 12k 步（同一 CLI）。

用法（后台运行 + 轮询日志）：
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup python3 run.py --steps 12000 > run.log 2>&1 &
    tail -f run.log                       # 轮询
"""

import argparse, json, math, os, shutil, subprocess, sys, time

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
DATA = "data/enwik8_full.npz"
DEVICE = "cuda"


def base_cfg(**kw) -> DepthARConfig:
    """契约 §2 骨架：d=256/L=8/h=8/seq256/mlp4，carry 门、随机退出、fill_kv。"""
    return DepthARConfig(d_model=256, n_layers=8, n_heads=8, seq_len=256,
                         mlp_mult=4, dense=False, fill_kv=True, **kw)


# ------------------------- prob0 ECE（契约 §3 可控性） -------------------------

@torch.no_grad()
def prob0_ece(model: DepthARModel, eval_arr, n_batches: int = 20,
              batch_size: int = 16, n_bins: int = 15, seed: int = 7) -> dict:
    """prob0 校准：每深度 softmax 后 top-1 概率（prob0）对 argmax 正确率，
    等频分箱 ECE（pathlm.metrics.ece）。prob0_ece = 最终深度（该 token 的
    提交置信度）；exit4_ece = 出口深度 4。"""
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
    """干预活性判据（P3 强制项）：训练结束时 W1/W2 必须显著非零，否则干预
    从未被激活（= 零初始鞍点，测不到机制）。"""
    W1 = model.readout_w1.weight.detach().float()
    W2 = model.readout_w2.weight.detach().float()
    n = W1.numel() + W2.numel()
    nz = (torch.count_nonzero(W1) + torch.count_nonzero(W2)).item()
    return {"w1_fro": round(float(W1.norm()), 4),
            "w2_fro": round(float(W2.norm()), 4),
            "nonzero_frac": round(nz / n, 6),
            "w1_spec_norm": round(float(torch.linalg.matrix_norm(W1, 2)), 4),
            "w2_spec_norm": round(float(torch.linalg.matrix_norm(W2, 2)), 4)}


def run_battery(model: DepthARModel, eval_arr: dict, eval_batches: int) -> dict:
    out = {}
    out["depth_curve"] = depth_curve(model, eval_arr, device=DEVICE,
                                     n_batches=eval_batches)
    out["tf_acceptance"] = tf_acceptance(model, eval_arr, device=DEVICE,
                                         n_batches=eval_batches)
    out["spec_cost_model"] = spec_cost_model(out["tf_acceptance"],
                                             model.cfg.n_layers, draft_depths=(4, 6))
    out["workspace"] = workspace_probe(model, eval_arr, device=DEVICE,
                                       n_batches=4, batch_size=8)
    out["readout_workspace_energy"] = readout_workspace_energy(
        model, eval_arr, device=DEVICE, n_batches=4, batch_size=8)
    out["generation"] = generation_quality(model, eval_arr, device=DEVICE,
                                           prompt_len=64, n_new=128)
    out["calibration"] = prob0_ece(model, eval_arr, n_batches=8, batch_size=16)
    return out


# ------------------------------ 训练子进程 ------------------------------

def train_gated(run_name: str, steps: int, seed: int, caps_min: float) -> dict:
    """后台启动 train_depth_ar.py（--readout-gated），轮询日志 + nvidia-smi
    峰值显存；超时/失败即中止并抛出。"""
    run_dir = os.path.join(HERE, "runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    log = open(os.path.join(run_dir, "stdout.log"), "w")
    cmd = [sys.executable, "train_depth_ar.py", run_name,
           "--d", "256", "--layers", "8", "--heads", "8", "--seq-len", "256",
           "--mlp-mult", "4", "--steps", str(steps), "--batch-size", "32",
           "--seed", str(seed), "--exit-mode", "uniform", "--readout-gated",
           "--eval-batches", "20", "--out-root", os.path.join(HERE, "runs")]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] gated training started pid={proc.pid} steps={steps}", flush=True)
    peak = 0.0
    t0 = time.time()
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
            if lines and int(json.loads(lines[-1])["step"]) % 1000 == 0:
                print(f"[run] {lines[-1].strip()[:120]}", flush=True)
        time.sleep(20)
    log.close()
    if proc.returncode != 0:
        raise RuntimeError(f"training failed rc={proc.returncode}")
    # 训练日志入库（`runs/` 被 .gitignore 覆盖）：拷到实验根供交付
    for name in ("train_log.jsonl", "gate_log.jsonl"):
        src = os.path.join(run_dir, name)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HERE, name))
    wall = (time.time() - t0) / 60
    print(f"[run] gated training done wall={wall:.2f} min peak_child_mem="
          f"{peak:.2f}GB", flush=True)
    res = json.load(open(os.path.join(run_dir, "results.json")))
    return {"results": res, "wall_min": round(wall, 2),
            "peak_child_mem_gb": round(peak, 2)}


# ---------------------- A 臂等值性门 + 重测电池 ----------------------

def load_a_arm() -> tuple[DepthARModel, dict]:
    """载入历史同配置 A 臂 checkpoint（tied 线性），返回模型与历史 results。"""
    hist = json.load(open(os.path.join(A_ARM_DIR, "results.json")))
    torch.manual_seed(0)
    model = DepthARModel(base_cfg(readout_gated=False), hist["vocab_size"]).cuda()
    sd = torch.load(os.path.join(A_ARM_DIR, "model.pt"), map_location="cuda")
    model.load_state_dict(sd)
    return model, hist


def equivalence_table(hist: dict) -> list[dict]:
    """A 臂等值性门（supervisor 裁决）：历史 run 的 resolved config 与本契约
    A 臂逐字段对照。`source` 标注历史值的出处（results.json 字段 / 后续 ticket
    的分解文件 / 代码谱系）。"""
    cfg = hist["config"]
    tr = hist["train"]
    rows = [
        ("d_model", cfg["d_model"], 256, "results.json"),
        ("n_layers", cfg["n_layers"], 8, "results.json"),
        ("n_heads", cfg["n_heads"], 8, "results.json"),
        ("seq_len", cfg["seq_len"], 256, "results.json"),
        ("mlp_mult", cfg["mlp_mult"], 4, "results.json"),
        ("dropout", cfg["dropout"], 0.0, "results.json"),
        ("norm_cap", cfg["norm_cap"], 1.0, "results.json"),
        ("gate_bias_init", cfg["gate_bias_init"], -2.0, "results.json"),
        ("dense", cfg["dense"], False, "results.json"),
        ("exit_mode", hist["exit_mode"], "uniform", "results.json"),
        ("aux_weight", hist["aux_weight"], 1.0, "results.json"),
        ("aux_anneal_to", hist["aux_anneal_to"], None, "results.json"),
        ("final_weight", hist.get("final_weight", 1.0), 1.0,
         "23c 无 final_weight 字段；等权 v1 = ticket24 final_weight 1.0"),
        ("shallow_stopgrad", hist["shallow_stopgrad"], False, "results.json"),
        ("dense_aux", hist["dense_aux"], False, "results.json"),
        ("steps", tr["steps"], 12000, "results.json"),
        ("batch_size", tr["batch_size"], 32, "results.json"),
        ("lr", tr["lr"], 0.001, "results.json"),
        ("seed", tr["seed"], 0, "results.json"),
        ("train_tokens", hist["train_tokens"], 98_304_000, "results.json"),
        ("vocab_size", hist["vocab_size"], 206, "results.json"),
        ("fill_kv", True, True,
         "开关 23e 才加；23c-H2 是 v1 混合填充语义（23e-decomposition：W0-12k = 门+退出+混合填充）"),
        ("proj_fill", False, False, "未引入（23e 才加）；v1 缺省 False"),
        ("readout_dims", None, None, "未引入（ticket 25 才加）"),
        ("readout_gated", False, False, "本实验新增；A 臂 = 关（tied 线性）"),
    ]
    return [{"field": f, "hist_A": h, "this_A": t, "equal": h == t,
             "source": s} for f, h, t, s in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--caps-min", type=float, default=16.0,
                    help="主臂训练墙钟上限（分钟），超时中止")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--eval-a-only", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    torch.manual_seed(args.seed)
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", DATA)

    # config.json（resolved，含 seed）
    gated_cfg = base_cfg(readout_gated=True)
    cfg_dump = {"experiment": "2610-gated-readout-13m-depthar",
                "seed": args.seed, "steps": args.steps, "batch_size": 32,
                "lr": 0.001, "exit_mode": "uniform", "data": DATA,
                "data_split": "first 90M train / last 10M eval, vocab from full file",
                "vocab_size": vocab_size,
                "A_arm": {"readout": "tied linear h·Eᵀ", "readout_gated": False,
                          "source": os.path.relpath(A_ARM_DIR, ROOT)},
                "main_arm": {"readout": "h + SiLU(W1 h)·(W2 h) then ·Eᵀ",
                             "readout_gated": True,
                             "readout_w1_w2_init": "W1 ~ N(0, 0.02), W2 = 0 "
                                                   "(identity start, non-saddle)",
                             "params_extra": 2 * 256 * 256},
                "base": {"d_model": 256, "n_layers": 8, "n_heads": 8,
                         "seq_len": 256, "mlp_mult": 4, "dense": False,
                         "fill_kv": True, "proj_fill": False,
                         "gate_bias_init": -2.0, "aux_weight": 1.0,
                         "final_weight": 1.0, "shallow_stopgrad": False,
                         "dense_aux": False}}
    json.dump(cfg_dump, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)

    out = {"experiment": cfg_dump["experiment"], "config": cfg_dump}

    # ---- A 臂：等值性门 + 当前代码重测 ----
    a_model, hist_a = load_a_arm()
    out["A_arm_equivalence"] = equivalence_table(hist_a)
    out["A_arm_equivalence_all_equal"] = all(r["equal"] for r in out["A_arm_equivalence"])
    print(f"[A] equivalence gate all_equal={out['A_arm_equivalence_all_equal']}",
          flush=True)
    t0 = time.time()
    a_batt = run_battery(a_model, eval_arr, args.eval_batches)
    out["A_arm"] = {"source": os.path.relpath(A_ARM_DIR, ROOT),
                    "hist_clean_bpc": hist_a.get("depth_curve", [{}])[-1].get("bpc"),
                    "hist_tf4": next((r["accept"] for r in hist_a.get("tf_acceptance", [])
                                      if r["depth"] == 4), None),
                    **a_batt,
                    "eval_wall_min": round((time.time() - t0) / 60, 2)}
    print(f"[A] battery done clean_bpc="
          f"{a_batt['depth_curve'][-1]['bpc']} tf4="
          f"{next(r['accept'] for r in a_batt['tf_acceptance'] if r['depth']==4)} "
          f"({round((time.time()-t0)/60,2)} min)", flush=True)

    if args.eval_a_only:
        json.dump(out, open(os.path.join(HERE, "results.json"), "w"),
                  indent=2, ensure_ascii=False)
        return

    # ---- 主臂：全新训练 12k ----
    tr = train_gated("gated-12k-s0", args.steps, args.seed, args.caps_min)
    g_res = tr["results"]
    # 主臂电池已由 train_depth_ar.py 写出；补 ECE（需 checkpoint）
    torch.manual_seed(args.seed)
    g_model = DepthARModel(gated_cfg, vocab_size).cuda()
    g_model.load_state_dict(torch.load(os.path.join(HERE, "runs", "gated-12k-s0",
                                                    "model.pt"), map_location="cuda"))
    t0 = time.time()
    g_batt = run_battery(g_model, eval_arr, args.eval_batches)
    out["intervention"] = intervention_liveness(g_model)
    out["main_arm"] = {"source": "runs/gated-12k-s0",
                       **g_batt, "wall_min": tr["wall_min"],
                       "peak_child_mem_gb": tr["peak_child_mem_gb"],
                       "eval_wall_min": round((time.time() - t0) / 60, 2),
                       "gnorm_monitor": g_res.get("gnorm_monitor"),
                       "params": g_res["params"],
                       "intervention": out["intervention"]}

    # ---- 判读（判据线预先写下，不许事后调整） ----
    a_bpc = out["A_arm"]["depth_curve"][-1]["bpc"]
    g_bpc = out["main_arm"]["depth_curve"][-1]["bpc"]
    delta = round(g_bpc - a_bpc, 4)
    g_tf4 = next(r["accept"] for r in out["main_arm"]["tf_acceptance"]
                 if r["depth"] == 4)
    a_tf4 = next(r["accept"] for r in out["A_arm"]["tf_acceptance"]
                 if r["depth"] == 4)
    a_ece = out["A_arm"]["calibration"]["prob0_ece"]
    g_ece = out["main_arm"]["calibration"]["prob0_ece"]
    ece_delta = round(g_ece - a_ece, 4)
    if delta <= -0.05 and g_tf4 >= 0.95 and ece_delta <= 0.003:
        verdict = "成功"
    elif delta <= -0.05:
        verdict = "失败（bpc 过线但 tf@4/ECE 判据未过）"
    elif delta < -0.014:
        verdict = "弱正向（未过 −0.05，记录方向性证据，不采纳）"
    elif delta <= 0.014:
        verdict = "中性/无头寸"
    else:
        verdict = "失败"
    # 同引擎 dense 税分母（历史 13M matched 参照）
    dense_bpc = 1.5287    # .scratch/23-depth-ar/evidence/23d-dense-B-12k
    out["verdict"] = {
        "criterion": "Δclean_bpc ≤ −0.05 且 tf@4 ≥ 0.95 且 ECE 不劣化 >0.003",
        "A_clean_bpc": a_bpc, "main_clean_bpc": g_bpc, "delta_clean_bpc": delta,
        "A_tf4": a_tf4, "main_tf4": g_tf4,
        "A_prob0_ece": a_ece, "main_prob0_ece": g_ece, "delta_ece": ece_delta,
        "dense_bpc_ref_13m": dense_bpc,
        "A_tax_vs_dense": round(a_bpc - dense_bpc, 4),
        "main_tax_vs_dense": round(g_bpc - dense_bpc, 4),
        "ruling": verdict,
    }
    out["wall_total_min"] = round((time.time() - t_start) / 60, 2)
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out["verdict"], indent=2, ensure_ascii=False), flush=True)
    print(f"[done] total wall {out['wall_total_min']} min", flush=True)


if __name__ == "__main__":
    main()
