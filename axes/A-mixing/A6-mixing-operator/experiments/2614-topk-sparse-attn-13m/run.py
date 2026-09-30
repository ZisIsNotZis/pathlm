"""2614 · 内容 top-k 稀疏注意力 vs 全注意力（A6 混合算子首探）· 13M · depth-ar/dense。

单因素（ticket 31 契约 J，判据线预先写下未改）：**唯一变量 = 注意力算子**。
三个臂都在同一 stock decoder-only transformer 上（`--dense`：无 carry 门、无随机
退出、仅最终深度 CE、tied 读出），同数据、同 token 预算、同步数、同 seed、同 lr、
同 batch，唯一差异 = 注意力算子：

- `full`   （基线）全因果 softmax 注意力（stock `Block` + SDPA）；
- `topk32` 每个查询在**因果掩码内**只保留内容相似度（q·kᵀ）最大的 32 个键，
  其余置 −inf 后 softmax（选择按内容、不按位置）；
- `topk64` 同上，k = 64。

Δ 符号（全篇统一，SCHEMA §17.3）：**Δ = bpc_对照 − bpc_实验 = bpc_full − bpc_topk**
（正 = 改善）。判据线逐字见 README.md §0/§4。

用法（后台运行 + ≤110s 轮询日志）：
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      nohup python3 run.py --steps 12000 --seed 0 > run_log.txt 2>&1 &
    tail -f run_log.txt
"""

import argparse, json, os, statistics, shutil, subprocess, sys, time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import (DepthARConfig, DepthARModel, TopKBlock, Block,
                             depth_curve, tf_acceptance, attn_sparse_probe,
                             causal_topk_keep_mask, sparse_attn_telemetry)
from pathlm.data import load_enwik8_full, batch

DATA = "data/enwik8_full.npz"
DEVICE = "cuda"
D, L, H, SEQ, MLP, BS, LR = 256, 8, 8, 256, 4, 32, 1e-3

ARMS = [
    {"arm": "full", "k": 0,
     "role": "基线：全因果 softmax 注意力（stock Block + SDPA）"},
    {"arm": "topk32", "k": 32,
     "role": "内容 top-k（k=32）：因果内按 q·kᵀ 保留最大 32 个键"},
    {"arm": "topk64", "k": 64, "role": "内容 top-k（k=64）：同算子、另一档 k"},
]
RERUN_ARMS = ("full",)   # 重跑散度（契约 J 要求）：主 Agent 裁决只重跑基线臂


def cfg_base(k: int) -> DepthARConfig:
    """契约 J 骨架：d=256 / L=8 / h=8 / seq=256 / mlp_mult=4，stock dense 形态。"""
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ,
                         mlp_mult=MLP, dense=True, attn_sparse_k=k)


def run_name(arm: str, rep: int = 1, suffix: str = "") -> str:
    base = f"{arm}-12k-s0" if rep == 1 else f"{arm}-12k-s0-r{rep}"
    return base + suffix


# ------------------------------ 训练子进程 ------------------------------

def train_arm(arm: str, k: int, steps: int, seed: int, caps_min: float,
              rep: int = 1, poll_s: int = 20, eval_batches: int = 20,
              suffix: str = "") -> dict:
    """后台启动 train_depth_ar.py（--dense --attn-sparse-k k），轮询日志 +
    nvidia-smi 峰值显存；超时/失败即中止并抛出。训练前清空本 run 的日志
    （train_depth_ar 是 append 语义）。"""
    name = run_name(arm, rep, suffix)
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
           "--attn-sparse-k", str(k), "--steps", str(steps),
           "--batch-size", str(BS), "--seed", str(seed),
           "--exit-mode", "uniform", "--eval-batches", str(eval_batches),
           "--out-root", os.path.join(HERE, "runs")]
    env = dict(os.environ, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            env=env)
    print(f"[run] {name}: training started pid={proc.pid} steps={steps} k={k}",
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
            if lines and int(json.loads(lines[-1])["step"]) % 4000 == 0:
                print(f"[run] {name} {lines[-1].strip()[:150]}", flush=True)
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
          f"bpc={res['bpc']}", flush=True)
    return {"results": res, "wall_min": round(wall, 2),
            "peak_child_mem_gb": round(peak, 2),
            "seconds_per_step": round(wall * 60 / steps, 5)}


# ------------------------------ 评测电池 ------------------------------

@torch.no_grad()
def operator_liveness(model: DepthARModel, eval_arr, k: int,
                      batch_size: int = 8, seed: int = 21) -> dict | None:
    """同一权重、只改注意力算子的反事实（SCHEMA §10 活性判据）：

    - `per_layer_rel_dev_vs_full`：把 `TopKBlock.topk` 临时设为 ≥ T（选择退化为
      恒等 ⇒ 与 stock `Block` 逐位一致的因果 SDPA，单测
      `test_topk_block_k_ge_T_bitwise_equals_block` 钉住）后重算该层输出，
      两者相对偏差 = 稀疏算子真实改变该层计算的幅度（>0 ⇒ 干预活着）；
    - `sel_in_lastk_frac`：被选键里落在「最近 k 个位置」窗口内的比例——位置选择
      基线的重合度，≪1 说明选择是按内容而非位置窗口；
    - `sel_dist_mean`：被选键的平均距离。
    """
    if k <= 0:
        return None
    gen = torch.Generator().manual_seed(seed)
    x, _ = batch(eval_arr, batch_size, SEQ, gen)
    x = x.to(DEVICE)
    _, aux = model(x, exit_depths=None)
    per_layer, devs = [], []
    for i, blk in enumerate(model.blocks):
        assert isinstance(blk, TopKBlock)
        h_in = aux["states"][i]
        out_sparse, tele = blk(h_in)
        saved = blk.topk
        blk.topk = SEQ                       # 退化 = 全因果注意力（同权重）
        out_full, tele_full = blk(h_in)
        blk.topk = saved
        assert tele_full is None
        dev = float(((out_sparse - out_full).norm() / out_full.norm()).detach())
        devs.append(dev)
        per_layer.append({"layer": i + 1,
                          "rel_dev_vs_full": round(dev, 6),
                          "kept_mean": round(float(tele["kept_mean"]), 4),
                          "sel_dist_mean": round(float(tele["sel_dist_mean"]), 4),
                          "sel_in_lastk_frac": round(
                              float(tele["sel_in_lastk_frac"]), 4)})
    return {"k": k, "eval_batch_size": batch_size, "eval_seed": seed,
            "per_layer": per_layer,
            "rel_dev_vs_full_mean": round(statistics.mean(devs), 6),
            "rel_dev_vs_full_min": round(min(devs), 6),
            "alive": bool(min(devs) > 0)}


def run_battery(model: DepthARModel, eval_arr, k: int, eval_batches: int) -> dict:
    out = {}
    out["depth_curve"] = depth_curve(model, eval_arr, device=DEVICE,
                                     n_batches=eval_batches)
    out["clean_bpc"] = out["depth_curve"][-1]["bpc"]
    out["next_token_acc"] = out["depth_curve"][-1]["acc"]
    out["tf_acceptance"] = tf_acceptance(model, eval_arr, device=DEVICE,
                                         n_batches=eval_batches)
    out["tf@4"] = next(r["accept"] for r in out["tf_acceptance"]
                       if r["depth"] == 4)
    out["attn_sparse_liveness"] = attn_sparse_probe(model, eval_arr, device=DEVICE)
    out["operator_liveness"] = operator_liveness(model, eval_arr, k)
    return out


# ---------------------- 成本口径（理论 + 实测，契约 J 并列判据） ----------------------

def flops_accounting(k: int, T: int = SEQ, d: int = D, mlp_mult: int = MLP) -> dict:
    """每 token 每层的乘加对（MAC = 2 FLOPs）口径：

    - 全注意力：QKᵀ 2·T·d + PV 2·T·d = 4·T·d；
    - 内容 top-k（**选择必须算全分数矩阵**）：QKᵀ 2·T·d（不可省）+ PV 2·k·d
      + 选择开销 O(T·log k) 比较（不计入 matmul 口径）；
    - 其余部件与 k 无关：qkv 投影 6d²、out 投影 2d²、MLP 4·mlp_mult·d²。
    """
    attn_full = 4 * T * d
    attn_topk = 2 * T * d + 2 * k * d
    other = 6 * d * d + 2 * d * d + 4 * mlp_mult * d * d
    total_full = attn_full + other
    total_topk = attn_topk + other
    return {
        "per_token_per_layer_flops": {"attn_full": attn_full,
                                      "attn_topk": attn_topk, "other": other,
                                      "total_full": total_full,
                                      "total_topk": total_topk},
        "attn_flops_reduction_frac": round((attn_full - attn_topk) / attn_full, 4),
        "attn_reduction_formula": "(T − k) / (2T)：QKᵀ 全矩阵因内容选择不可省",
        "attn_share_of_layer_flops_full": round(attn_full / total_full, 4),
        "total_flops_reduction_frac": round(
            (total_full - total_topk) / total_full, 4),
        "naive_key_count_reduction_frac": round(1 - k / T, 4),
        "kv_cache_bytes_reduction_frac": 0.0,
        "kv_cache_note": ("内容选择需要**全部** K/V 才能算分数 ⇒ KV 缓存/显存不能减少"
                          "（位置窗口式稀疏才会减到 k/T）；这是与「位置稀疏」的本质"
                          "区别，不是实现缺陷"),
    }


def scaling_table(Ts=(256, 512, 1024, 2048, 4096, 8192)) -> list[dict]:
    """注意力占比与 top-k 上限的 T 依赖（**纯公式**，非实测）：T 越大，注意力在
    每 token FLOPs 中的占比越高 ⇒ 内容 top-k 的上限越高。"""
    rows = []
    for T in Ts:
        for k in (32, 64):
            fa = flops_accounting(k, T=T)
            rows.append({"T": T, "k": k,
                         "attn_share_of_layer": fa["attn_share_of_layer_flops_full"],
                         "attn_reduction_frac": fa["attn_flops_reduction_frac"],
                         "total_layer_reduction_frac": fa["total_flops_reduction_frac"]})
    return rows


def _timed(fn, warmup: int = 3, iters: int = 8) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": round(statistics.median(ts), 3),
            "min_ms": round(min(ts), 3), "n_iters": iters}


def model_step_cost(k: int, steps: int = 8) -> dict:
    """实测（同一进程、同一 shape、同一 optimizer）：整模型一步 fwd+bwd 的
    中位墙钟与峰值显存（torch.cuda.max_memory_allocated）。"""
    torch.manual_seed(0)
    model = DepthARModel(cfg_base(k), 206).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    x = torch.randint(0, 206, (BS, SEQ)).cuda()
    timing = {}

    def step():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = model(x, exit_depths=None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    timing = _timed(step, warmup=3, iters=steps)
    timing["peak_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
    del model, opt
    torch.cuda.empty_cache()
    return timing


def attn_island_cost(mode: str, k: int, B: int = BS, T: int = SEQ, d: int = D,
                     n_heads: int = H, iters: int = 8) -> dict:
    """注意力子层孤岛（qkv→注意力→out 投影）fwd+bwd 实测，同一输入、同一 buffer；
    `mode` ∈ {full, topk_dense, topk_sdpa, topk_gather}：三种 top-k 实装分别代表
    （a）本实验训练实装（显式 softmax）、（b）保留 SDPA 融合核的掩码实装、
    （c）**参考实现**（真的只对 k 个键做混合，gather 选出 K/V，非训练实装）。"""
    dh = d // n_heads
    torch.manual_seed(0)
    x = torch.randn(B, T, d, device=DEVICE, requires_grad=True)
    qkv = torch.nn.Linear(d, 3 * d).to(DEVICE)
    proj = torch.nn.Linear(d, d).to(DEVICE)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=DEVICE))
    shape = lambda t: t.view(B, T, n_heads, dh).transpose(1, 2)

    def step():
        q, kk, v = qkv(x).chunk(3, dim=-1)
        qs, ks, vs = shape(q), shape(kk), shape(v)
        if mode == "full":
            a = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
        else:
            scores = qs @ ks.transpose(-2, -1) / dh ** 0.5
            keep = causal_topk_keep_mask(scores, k, causal)
            if mode == "topk_dense":
                w = scores.masked_fill(~keep, float("-inf")).softmax(-1)
                a = w @ vs
            elif mode == "topk_sdpa":
                bias = torch.zeros_like(scores).masked_fill_(~keep, float("-inf"))
                a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=bias)
            elif mode == "topk_gather":
                idx = torch.topk(scores.masked_fill(~keep, float("-inf")), k,
                                 dim=-1).indices              # [B,H,T,k]
                take = idx.unsqueeze(-1).expand(-1, -1, -1, -1, dh)
                ks_g = torch.gather(ks.unsqueeze(-3).expand(-1, -1, T, -1, -1),
                                    3, take)                  # [B,H,T,k,dh]
                vs_g = torch.gather(vs.unsqueeze(-3).expand(-1, -1, T, -1, -1),
                                    3, take)
                sc = (qs.unsqueeze(-2) * ks_g).sum(-1) / dh ** 0.5
                a = (sc.softmax(-1).unsqueeze(-1) * vs_g).sum(-2)
            else:
                raise ValueError(mode)
        a = a.transpose(1, 2).reshape(B, T, d)
        y = proj(a)
        loss = y.float().pow(2).mean()
        qkv.zero_grad(); proj.zero_grad()
        loss.backward()

    out = _timed(step, warmup=2, iters=iters)
    out.update({"mode": mode, "k": k, "batch_size": B, "seq_len": T,
                "d_model": d,
                "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)})
    del x, qkv, proj
    torch.cuda.empty_cache()
    return out


def cost_probe(iters: int = 8) -> dict:
    """契约 J 并列判据的完整口径：理论（k/T、FLOPs 公式）+ 实测（整模型步时/
    显存 + 注意力孤岛三种实装）。**实测值含实现代价**（本实装离开 SDPA 融合核），
    与理论值的差距如实报出，不做「理论 vs 实测取优」的选择性汇报。"""
    model_steps = {f"k={k}": model_step_cost(k, steps=iters) for k in (0, 32, 64)}
    ref = model_steps["k=0"]["median_ms"]
    for key, val in model_steps.items():
        val["step_time_ratio_vs_full"] = round(val["median_ms"] / ref, 4)
        val["peak_mem_ratio_vs_full"] = round(
            val["peak_mem_gib"] / model_steps["k=0"]["peak_mem_gib"], 4)
    islands = [attn_island_cost("full", 0, iters=iters)]
    islands += [attn_island_cost("topk_dense", k, iters=iters) for k in (32, 64)]
    islands += [attn_island_cost("topk_sdpa", k, iters=iters) for k in (32, 64)]
    # 参考实现（真稀疏 gather）用小 batch，避免 idx/KV  gather 张量过大：单独标注 shape
    islands += [attn_island_cost("topk_gather", k, B=8, iters=iters)
                for k in (32, 64)]
    iref = islands[0]["median_ms"]
    for row in islands:
        row["island_time_ratio_vs_full"] = round(row["median_ms"] / iref, 4)
        row["shape_note"] = ("B=8（参考实现用小 batch，否则 gather 的 idx/KV "
                             "张量过大）" if row["mode"] == "topk_gather"
                             else "B=32（= 训练 batch）")
    return {
        "theory": {f"k={k}": flops_accounting(k) for k in (32, 64)},
        "theory_scaling_vs_T": scaling_table(),
        "measured_model_step": model_steps,
        "measured_attn_island": islands,
        "measured_note": ("整模型步时/显存 = 训练实装（显式 softmax 的 top-k）；"
                          "注意力孤岛另报三种实装，用于把「算法本征代价」与"
                          "「离开 SDPA 融合核的实现代价」分开"),
    }


# ---------------------- 收敛对账（SCHEMA §16）+ 稳定性 ----------------------

def _slope(win: list[dict]) -> float:
    xs = [r["step"] for r in win]
    ys = [r["loss"] for r in win]
    xm, ym = sum(xs) / len(xs), sum(ys) / len(ys)
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / \
        sum((x - xm) ** 2 for x in xs)


def tail_window_slope(rows: list[dict], frac: float = 0.25) -> dict:
    """末 frac 窗内 loss 对 step 的最小二乘斜率（**三臂同一方法、同一窗口比例**）。
    样本不足（日志点 < 3 或 step 跨度 = 0）时如实返回 None，不外推。"""
    i0 = int(len(rows) * (1 - frac))
    win = rows[i0:]
    if len(win) < 3 or win[-1]["step"] == win[0]["step"]:
        win = rows[-min(10, len(rows)):]
    span_ok = len(win) >= 3 and win[-1]["step"] != win[0]["step"]
    return {"window_frac": frac, "n_points": len(win),
            "step_from": win[0]["step"], "step_to": win[-1]["step"],
            "loss_from": win[0]["loss"], "loss_to": win[-1]["loss"],
            "slope_nats_per_1k_steps": (round(_slope(win) * 1000, 5)
                                        if span_ok else None),
            "slope_nats_per_1k_steps_late_half": (round(
                _slope(win[len(win) // 2:]) * 1000, 5)
                if span_ok and len(win) >= 6 else None),
            "note": None if span_ok else "日志点不足，斜率不可估（未外推）"}


def load_arm_rows(name: str) -> list[dict]:
    path = os.path.join(HERE, f"{name}-train_log.jsonl")
    return [json.loads(l) for l in open(path) if l.strip()]


def stability_stats(rows: list[dict], gnorm_monitor: dict | None) -> dict:
    g = [r["gnorm"] for r in rows]
    loss = [r["loss"] for r in rows]
    med = statistics.median(g)
    inc = [b - a for a, b in zip(loss, loss[1:])]
    return {"n_logged_samples": len(rows),
            "step_first": rows[0]["step"], "step_last": rows[-1]["step"],
            "gnorm_logged_max": max(g), "gnorm_logged_median": round(med, 3),
            "gnorm_spikes_gt3x_median": sum(1 for v in g if v > 3 * med),
            "gnorm_spikes_gt5x_median": sum(1 for v in g if v > 5 * med),
            "all_finite": all(torch.isfinite(torch.tensor(v)).item()
                              for v in g + loss),
            "loss_first": loss[0], "loss_last": loss[-1],
            "loss_max_step_increase": round(max(inc), 4),
            "loss_jumps_gt_0.25": sum(1 for v in inc if v > 0.25),
            "gnorm_monitor_all_steps": gnorm_monitor}


# ------------------------------ 主流程 ------------------------------

def main():
    os.chdir(ROOT)          # 数据/入口脚本都是仓库根相对路径（单一来源）
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--caps-min", type=float, default=26.0,
                    help="单臂训练墙钟上限（分钟），超时中止")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--poll-s", type=int, default=20)
    ap.add_argument("--skip-train", action="store_true",
                    help="复用 runs/ 下已有 checkpoint，只重算电池与判读")
    ap.add_argument("--budget-min", type=float, default=70.0,
                    help="GPU 墙钟预算（契约 §6 = 45 min；主 Agent 裁决显式批准超支到"
                         "≈65–70 min）：超出则在下一臂开跑前中止（不静默跑残臂）；"
                         "砍项顺序 = 重跑臂 → cost-iters，**不砍判据线、不砍评测批次**")
    ap.add_argument("--rerun", dest="rerun", action="store_true", default=True,
                    help="跑重跑臂（契约 J 要求「重跑散度」）")
    ap.add_argument("--no-rerun", dest="rerun", action="store_false")
    ap.add_argument("--rerun-arms", default=",".join(RERUN_ARMS))
    ap.add_argument("--arms", default=None,
                    help="只跑指定臂（逗号分隔，必须含 full）")
    ap.add_argument("--cost-iters", type=int, default=8)
    ap.add_argument("--suffix", default="",
                    help="run 名后缀（调试用，例如 -dry；缺省空 = 正式证据目录名）")
    args = ap.parse_args()

    arms = list(ARMS)
    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in want if a not in [x["arm"] for x in ARMS]]
        if unknown or "full" not in want:
            raise SystemExit(f"--arms 无效（{unknown}）：必须含 full，"
                             f"可选 {[x['arm'] for x in ARMS]}")
        arms = [a for a in arms if a["arm"] in want]
    rerun_arms = [a.strip() for a in args.rerun_arms.split(",")
                  if a.strip()] if args.rerun else []

    t_start = time.time()
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", DATA)
    cfg_dump = {
        "experiment": "2614-topk-sparse-attn-13m",
        "ticket": "31-batch4 契约 J",
        "seed": args.seed, "steps": args.steps, "batch_size": BS, "lr": LR,
        "optimizer": "AdamW(wd=0.01) + warmup 200 步 + cosine 到 0.1×",
        "data": DATA,
        "data_split": "first 90M train / last 10M eval, vocab from full file",
        "vocab_size": vocab_size, "train_tokens": args.steps * BS * SEQ,
        "base": {"d_model": D, "n_layers": L, "n_heads": H, "seq_len": SEQ,
                 "mlp_mult": MLP, "dropout": 0.0, "norm_cap": 1.0,
                 "dense": True, "topology": "sequential",
                 "exit_mode": "uniform", "loss": "仅最终深度 CE（stock AR）",
                 "readout": "tied h·Eᵀ，无 final LayerNorm（共享底座偏离 1）"},
        "arms": [{"arm": a["arm"], "attn_sparse_k": a["k"], "role": a["role"],
                  "run": f"runs/{run_name(a['arm'], 1, args.suffix)}"}
                 for a in arms],
        "rerun_arms": rerun_arms,
        "rerun_note": ("重跑散度（契约 J 要求）：主 Agent 裁决「3 主臂 + 只重跑基线 "
                        "full 臂」（Δ 的噪声由基线与实验臂共同决定，基线是被比较的"
                        "参照，重跑它同时能跨批校验可比性）；评测批次保持 20 不砍。"),
        "variable": "attn_sparse_k（唯一不等字段；其余逐字段一致，见 field_by_field）",
        "delta_convention": "Δ = bpc_full − bpc_topk（正 = 改善；SCHEMA §17.3）",
    }
    json.dump(cfg_dump, open(os.path.join(HERE, "config.json"), "w"),
              indent=2, ensure_ascii=False)
    out = {"experiment": cfg_dump["experiment"], "config": cfg_dump, "arms": {}}

    # ---- 成本口径：理论 + 实测（与训练解耦，先测，避免 GPU 抖动时被训练日志污染） ----
    out["cost"] = cost_probe(iters=args.cost_iters)
    print("[cost] model step ms: "
          f"{ {k: v['median_ms'] for k, v in out['cost']['measured_model_step'].items()} }",
          flush=True)

    def budget_ok(what: str) -> bool:
        el = (time.time() - t_start) / 60
        if not args.skip_train and el > args.budget_min:
            print(f"[budget] {el:.1f} min > {args.budget_min} min：中止在 {what} 之前",
                  flush=True)
            return False
        return True

    # ---- 三臂训练 + 电池 ----
    for i, arm in enumerate(arms):
        name = arm["arm"]
        if not budget_ok(f"arm {name}"):
            out["stopped_by_budget_before"] = [a["arm"] for a in arms[i:]]
            break
        if args.skip_train:
            rp = os.path.join(HERE, "runs", run_name(name, 1, args.suffix),
                              "results.json")
            tr = {"results": json.load(open(rp)),
                  "wall_min": json.load(open(rp))["wall_minutes"],
                  "peak_child_mem_gb": None, "seconds_per_step": None}
        else:
            tr = train_arm(name, arm["k"], args.steps, args.seed, args.caps_min,
                           poll_s=args.poll_s, eval_batches=args.eval_batches,
                           suffix=args.suffix)
        res = tr["results"]
        torch.manual_seed(args.seed)
        model = DepthARModel(cfg_base(arm["k"]), vocab_size).cuda()
        model.load_state_dict(torch.load(
            os.path.join(HERE, "runs", run_name(name, 1, args.suffix), "model.pt"),
            map_location="cuda"))
        t0 = time.time()
        batt = run_battery(model, eval_arr, arm["k"], args.eval_batches)
        out["arms"][name] = {
            "role": arm["role"], "attn_sparse_k": arm["k"],
            "source": f"runs/{run_name(name)}",
            "params": res["params"], "train_tokens": res["train_tokens"],
            "bpc_from_train_process": res["bpc"],
            "wall_min": tr["wall_min"],
            "peak_child_mem_gb": tr["peak_child_mem_gb"],
            "seconds_per_step": tr["seconds_per_step"],
            "gnorm_monitor": res.get("gnorm_monitor"),
            "intervention": res.get("attn_sparse_liveness"),
            **batt, "eval_wall_min": round((time.time() - t0) / 60, 2)}
        print(f"[arm] {name} (k={arm['k']}): clean_bpc={batt['clean_bpc']} "
              f"acc={batt['next_token_acc']} alive="
              f"{batt['attn_sparse_liveness']['alive']} wall={tr['wall_min']} min "
              f"eval={out['arms'][name]['eval_wall_min']} min", flush=True)

    # ---- 重跑散度（契约 J 要求）：同 seed 重训，量实现层非确定性 ----
    if rerun_arms:
        out["reruns"] = {}
        for name in rerun_arms:
            arm = next(a for a in ARMS if a["arm"] == name)
            if not budget_ok(f"rerun {name}"):
                out["reruns"][name] = {"status": "skipped_budget"}
                continue
            tr = train_arm(name, arm["k"], args.steps, args.seed, args.caps_min,
                           rep=2, poll_s=args.poll_s,
                           eval_batches=args.eval_batches, suffix=args.suffix)
            res = tr["results"]
            torch.manual_seed(args.seed)
            model = DepthARModel(cfg_base(arm["k"]), vocab_size).cuda()
            model.load_state_dict(torch.load(
                os.path.join(HERE, "runs", run_name(name, 2, args.suffix),
                             "model.pt"),
                map_location="cuda"))
            batt = run_battery(model, eval_arr, arm["k"], args.eval_batches)
            base = out["arms"].get(name, {})
            out["reruns"][name] = {
                "status": "done", "source": f"runs/{run_name(name, 2)}",
                "clean_bpc": batt["clean_bpc"],
                "next_token_acc": batt["next_token_acc"],
                "wall_min": tr["wall_min"],
                "peak_child_mem_gb": tr["peak_child_mem_gb"],
                "seconds_per_step": tr["seconds_per_step"],
                "bpc_abs_diff_vs_primary":
                    round(abs(batt["clean_bpc"] - base.get("clean_bpc", float("nan"))), 4)
                    if base else None,
                "attn_sparse_liveness": batt["attn_sparse_liveness"],
            }
            print(f"[rerun] {name}: clean_bpc={batt['clean_bpc']} "
                  f"(primary {base.get('clean_bpc')})", flush=True)

    # ---- 逐字段对照表（单因素：只允许 attn_sparse_k / 参数量相等项不同） ----
    fields = [("d_model", D), ("n_layers", L), ("n_heads", H), ("seq_len", SEQ),
              ("mlp_mult", MLP), ("dropout", 0.0), ("norm_cap", 1.0),
              ("dense", True), ("topology", "sequential"), ("fill_kv", True),
              ("proj_fill", False), ("readout_dims", None),
              ("readout_gated", False), ("readout_linear_resid", False),
              ("exit_mode", "uniform"), ("aux_weight", 1.0),
              ("final_weight", 1.0), ("shallow_stopgrad", False),
              ("dense_aux", False), ("steps", args.steps),
              ("batch_size", BS), ("lr", LR), ("seed", args.seed),
              ("train_tokens", args.steps * BS * SEQ),
              ("vocab_size", vocab_size),
              ("data_split", "first 90M/ last 10M, vocab over full file")]
    rows = []
    for f, want in fields:
        row = {"field": f, "expected": want, "arms": {}}
        for arm in arms:
            res = json.load(open(os.path.join(
                HERE, "runs", run_name(arm["arm"], 1, args.suffix),
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
                "topology": res["topology"],
                "fill_kv": res["fill_kv"], "proj_fill": res["proj_fill"],
                "readout_dims": res["readout_dims"],
                "readout_gated": res["readout_gated"],
                "readout_linear_resid": res["readout_linear_resid"],
                "exit_mode": res["exit_mode"],
                "aux_weight": res["aux_weight"],
                "final_weight": res["final_weight"],
                "shallow_stopgrad": res["shallow_stopgrad"],
                "dense_aux": res["dense_aux"],
                "steps": res["train"]["steps"],
                "batch_size": res["train"]["batch_size"],
                "lr": res["train"]["lr"], "seed": res["train"]["seed"],
                "train_tokens": res["train_tokens"],
                "vocab_size": res["vocab_size"],
                "data_split": "first 90M/ last 10M, vocab over full file",
            }[f]
            row["arms"][arm["arm"]] = got
            row["equal_" + arm["arm"]] = (got == want)
        row["all_arms_equal"] = all(row["equal_" + a["arm"]] for a in arms)
        rows.append(row)
    out["field_by_field"] = rows
    out["field_by_field_all_equal_except_attn_sparse_k"] = all(
        r["all_arms_equal"] for r in rows)
    out["field_by_field"].append({
        "field": "attn_sparse_k（唯一变量）",
        "arms": {a["arm"]: a["k"] for a in arms},
        "note": "唯一不等字段；该开关不新增参数（TopKBlock 与 Block 参数量逐位相等，"
                "单测钉住），因此本实验天然通过「等参数对照」（SCHEMA §15）"})

    # ---- 三道对账（SCHEMA §12）+ 收敛对账（§16） ----
    has_full = "full" in out["arms"]
    base_bpc = out["arms"]["full"]["clean_bpc"] if has_full else None
    variants = [a["arm"] for a in arms if a["arm"] != "full"]
    deltas = ({v: round(base_bpc - out["arms"][v]["clean_bpc"], 4)
               for v in variants} if has_full else {})
    slopes = {a["arm"]: tail_window_slope(load_arm_rows(
        run_name(a["arm"], 1, args.suffix)))
              for a in arms}
    params = {a["arm"]: out["arms"][a["arm"]]["params"] for a in arms
              if a["arm"] in out["arms"]}
    out["reconciliation"] = {
        "arithmetic": {
            "delta_definition": "Δ = bpc_full − bpc_topk（正 = 改善；SCHEMA §17.3）",
            "bpc_by_arm": {a["arm"]: out["arms"][a["arm"]]["clean_bpc"]
                           for a in arms if a["arm"] in out["arms"]},
            "delta_bpc": deltas,
            "variance_bound": 0.014,
            "judge_threshold_eq_variance_bound": True,
            "params": params,
            "params_all_equal": len(set(params.values())) <= 1,
            "kept_keys_identity": {
                "formula": "mean_i min(k, i+1) = (k(k+1)/2 + (T−k)·k) / T",
                "theory": {f"k={k}": round(
                    sum(min(k, i + 1) for i in range(SEQ)) / SEQ, 6)
                    for k in (32, 64)},
                "measured": {a["arm"]: out["arms"][a["arm"]]
                             ["attn_sparse_liveness"].get("kept_mean")
                             for a in arms if a["arm"] in out["arms"]},
                "note": "实测保留键数必须逐位等于理论式（活性量对账；不等 ⇒ 掩码没按 k 生效）",
            },
            "cost_arithmetic": {
                "attn_reduction_formula": "(T − k)/(2T)（QKᵀ 全矩阵不可省）",
                "theory_vs_measured_gap_note": (
                    "理论 FLOPs 下降 ≠ 实测墙钟下降：本实装离开 SDPA 融合核并物化 "
                    "[B,H,T,T] 中间量 ⇒ 实测更慢，如实并列报出（见 cost 块）"),
            },
        },
        "task": {
            "same_task": "三臂同一 enwik8 char 级 AR 任务、同一 eval 切分（末 10M）、"
                         "同一位置集（全序列逐位置 × depth-8 读出）、同一目标定义"
                         "（tokens 自移位）",
            "same_battery": "三臂同一 run.py::run_battery（同 n_batches=20 / 同 "
                            "eval seed / 同 batch_size）",
            "deploy_correspondence": "三臂都是 stock decoder-only 前向；对照只改"
                                     "注意力算子，输入分布完全相同（同一数据流），"
                                     "无额外输入、无真值泄漏",
        },
        "caliber": {
            "matched_step_snapshot": True, "steps": args.steps, "seed": args.seed,
            "n_seed": 1, "train_tokens": args.steps * BS * SEQ,
            "note": f"三臂同为 {args.steps} 步 / {args.steps * BS * SEQ / 1e6:.1f}M "
                    "token / seed 0 的 matched-step 快照口径（非渐近、非墙钟、非 FLOPs）",
        },
        "convergence": {
            "method": "末 25% 窗（同一窗口比例、同一最小二乘算法）loss 斜率，三臂同方法",
            "tail_slope_by_arm": slopes,
            "slope_spread_nats_per_1k": round(
                max(s["slope_nats_per_1k_steps"] for s in slopes.values()) -
                min(s["slope_nats_per_1k_steps"] for s in slopes.values()), 5)
                if all(s["slope_nats_per_1k_steps"] is not None
                       for s in slopes.values()) else None,
            "rerun_divergence": {k: v for k, v in out.get("reruns", {}).items()},
            "note": "斜率相近 ⇒ Δ 是水平差（质量）；斜率悬殊 ⇒ 收敛速度差，只能作快照"
                    "结论。2× 更长训练点未做（契约 J 未列入预算，见 analysis §7）",
        },
    }

    # ---- 判读（契约 J 判据线，逐字未改） ----
    better = [v for v in variants if deltas.get(v, 0) >= 0.014]
    worse = [v for v in variants if deltas.get(v, 0) <= -0.014]
    neutral = [v for v in variants if abs(deltas.get(v, 0)) < 0.014]
    if better:
        ruling = (f"更好（少见，记录）：{ {v: deltas[v] for v in better} }（Δ ≥ +0.014）")
    elif worse:
        ruling = (f"稀疏选择在该规模有害：{ {v: deltas[v] for v in worse} }"
                  "（Δ ≤ −0.014，记录 k 与劣化幅度）")
    else:
        ruling = f"与全注意力同质量（|Δ| < 0.014）：{deltas}"
    out["verdict"] = {
        "criterion": ("主判据：|Δ| < 0.014 ⇒ 与全注意力同质量；Δ ≥ +0.014 ⇒ 更好；"
                      "Δ ≤ −0.014 ⇒ 稀疏有害（记录 k 与幅度）。"
                      "Δ = bpc_full − bpc_topk（正 = 改善）"),
        "delta_bpc": deltas,
        "clean_bpc_by_arm": {a["arm"]: out["arms"][a["arm"]]["clean_bpc"]
                             for a in arms if a["arm"] in out["arms"]},
        "better_arms": better, "neutral_arms": neutral, "worse_arms": worse,
        "ruling": ruling,
        "cost_measured": {
            "model_step_time_ratio_vs_full": {
                k: v["step_time_ratio_vs_full"]
                for k, v in out["cost"]["measured_model_step"].items()},
            "model_step_peak_mem_ratio_vs_full": {
                k: v["peak_mem_ratio_vs_full"]
                for k, v in out["cost"]["measured_model_step"].items()},
            "attn_island_time_ratio_vs_full": {
                f"{r['mode']}_k{r['k']}": r["island_time_ratio_vs_full"]
                for r in out["cost"]["measured_attn_island"]},
            "theory_by_k": {k: {"naive_key_count_reduction": v[
                "naive_key_count_reduction_frac"],
                "attn_flops_reduction": v["attn_flops_reduction_frac"],
                "total_flops_reduction": v["total_flops_reduction_frac"],
                "kv_cache_reduction": v["kv_cache_bytes_reduction_frac"]}
                for k, v in out["cost"]["theory"].items()},
        },
    }

    # ---- 稳定性块 ----
    out["stability"] = {a["arm"]: stability_stats(
        load_arm_rows(run_name(a["arm"], 1, args.suffix)),
        out["arms"][a["arm"]]["gnorm_monitor"]) for a in arms
        if a["arm"] in out["arms"]}
    for name, rr in out.get("reruns", {}).items():
        if rr.get("status") == "done":
            out["stability"][f"{name}-r2"] = stability_stats(
                load_arm_rows(run_name(name, 2, args.suffix)), None)
    out["wall_total_min"] = round((time.time() - t_start) / 60, 2)
    out["gpu_peak_by_arm_gb"] = {a["arm"]: out["arms"][a["arm"]]
                                 ["peak_child_mem_gb"] for a in arms
                                 if a["arm"] in out["arms"]}
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out["verdict"], indent=2, ensure_ascii=False), flush=True)
    print(f"[done] total wall {out['wall_total_min']} min", flush=True)


if __name__ == "__main__":
    main()
