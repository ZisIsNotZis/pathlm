"""2644 · 通道稀疏激活（per-token top-k 条件计算 vs 固定前缀常开）· 13M · dense.

契约 = 同目录 `README.md`（判据线 P1–P4 **预登记、不可改**）。本文件是唯一入口。

设计要点（为什么这样接线，见契约 §2.1）：

* **训练整段走共享代码**：本文件把 `train_depth_ar` 模块里的全局名
  `DepthARModel` 换成一个工厂（构造 stock 模型后把每块 `blk.mlp` 包成
  `SparseFFN`），然后调用 `train_depth_ar.main()` —— 参数解析、配置装配、
  `train()` 循环、checkpoint、`results.json` 全部是**共享实现**。
* **臂间唯一的差别 = 每 token 参与前向的 FFN 隐藏通道子集**；所有臂都保留完整
  FFN 权重（总参数逐臂相同）。
* `mask ≡ 1`（`keep_frac=1.0`）时前向与 stock **逐位相同**（`x * 1.0` 精确）：
  等值门由单测 `T1` 钉住、变异 `T2` 钉住它不是恒真。
* **代价实现口径**（契约 §3）：按内容选通道必须先算全部 pre-activation ⇒ `lin1`
  省不掉，`K50` 真实省 25%、`F50`（固定前缀）才省 50%。两列都报。

用法：

    python3 run.py --arm K50 --seed 0 --steps 12000             # 训练 + 评测
    python3 run.py --arm K50 --seed 0 --steps 12000 --no-train   # 仅评测
    python3 run.py --aggregate                                   # 汇总 P1–P4 → results.json
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
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as TDA                                   # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel, depth_curve  # noqa: E402
from pathlm.data import load_enwik8_full, batch                # noqa: E402

DEVICE = "cuda"
LN2 = math.log(2.0)

# ---------------------------------------------------------------- 臂定义（契约 §3）

D_MODEL, N_LAYERS, N_HEADS, SEQ_LEN, MLP_MULT = 256, 8, 8, 256, 4
VOCAB = 206
HIDDEN = MLP_MULT * D_MODEL                                # 1024
ARMS = ("K50", "K25", "F50")
KEEP_FRAC = {"K50": 0.5, "K25": 0.25, "F50": 0.5}
VARIANCE_BOUND = 0.014

# 复用锚（2641 U4-s0 / U2-s0，见契约 §5）
D_ANCHOR_BPC = 1.5314
U2_ANCHOR_BPC = 1.5709       # 2641 U2-s0：同算力位点、FFN 参数减半
U2_ANCHOR_MACS = 5295616
D_ANCHOR_MACS = 7392768


# ---------------------------------------------------------------- 干预：FFN 通道掩码

class SparseFFN(nn.Module):
    """把 stock `blk.mlp`（[Linear, GELU, Linear, Dropout]）包成
    `out = drop(lin2(gelu(lin1(x)) * mask))`。

    `mask` 与激活**同 dtype**（`x * 1.0` 在 IEEE 下精确 ⇒ `keep_frac=1.0` 时
    前向与 stock **逐位相同**）。选择器**无参数**：`K*` 用 `|pre|` 的 top-k，
    `F50` 用固定前缀。`record=True` 时把 mask 计数累积到 `self.stats`（活性列）。
    """

    KEEP = KEEP_FRAC

    def __init__(self, mlp: nn.Sequential, arm: str, keep_frac: float | None = None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
        self.mlp = mlp
        self.arm = arm
        self.keep = KEEP_FRAC[arm] if keep_frac is None else float(keep_frac)
        self.counts = None            # [H] float64：逐通道被选中次数（None = 不记录）
        self.n_tokens = 0
        self.mode_masks = None        # bool [H]：与该层众数集比对（None = 不比对）
        self.mode_batch = []          # 每次前向一条 (matched_frac, n)（选择一致率）

    def k_active(self) -> int:
        return int(round(self.keep * HIDDEN))

    def _mask(self, pre: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        k = self.k_active()
        if self.arm == "F50":
            m = torch.zeros_like(g)
            m[..., :k] = 1
            return m
        idx = pre.detach().float().abs().topk(k, dim=-1).indices
        m = torch.zeros_like(g)
        m.scatter_(-1, idx, 1)
        return m

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin1, act, lin2, drop = self.mlp[0], self.mlp[1], self.mlp[2], self.mlp[3]
        pre = lin1(x)
        g = act(pre)
        mask = self._mask(pre, g)
        if self.counts is not None:
            self.counts += mask.detach().float().sum(dim=(0, 1)).cpu().double()
            self.n_tokens += int(mask.shape[0] * mask.shape[1])
        if self.mode_masks is not None:
            eq = (mask.bool() == self.mode_masks.to(mask.device)).all(dim=-1)
            self.mode_batch.append((float(eq.float().mean()), int(eq.numel())))
        return drop(lin2(g * mask))


def patch_ffn(model: DepthARModel, arm: str) -> DepthARModel:
    """把每块 `blk.mlp` 换成 `SparseFFN`（只改前向的通道子集，不加参数）。"""
    for blk in model.blocks:
        blk.mlp = SparseFFN(blk.mlp, arm)
    return model


def sparse_ffns(model: DepthARModel) -> list[SparseFFN]:
    return [blk.mlp for blk in model.blocks]


def build_model(arm: str, vocab_size: int = VOCAB, seed: int = 0) -> DepthARModel:
    """所有臂共用同一构造路径（同 seed ⇒ 同一份主干抽样 + 同一份 FFN 权重）。"""
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS,
                        seq_len=SEQ_LEN, mlp_mult=MLP_MULT, dense=True)
    return patch_ffn(DepthARModel(cfg, vocab_size), arm)


# ---------------------------------------------------------------- 代价（等效理论量）

def analytic_cost(arm: str, d: int = D_MODEL, h: int = HIDDEN, T: int = SEQ_LEN,
                  L: int = N_LAYERS, vocab: int = VOCAB) -> dict:
    """每 token 等价 MACs（解析，**不含任何时间量**）——契约 §3 的两列口径。

    * `ffn_macs_impl`（primary）：按内容选通道必须先算全部 pre-activation ⇒
      `K*` = `d·h`（lin1 全）+ `k·d`（lin2 按 k）；`F50`（固定前缀）可同时切两个矩阵
      ⇒ `(h/2)·d + d·(h/2)`；`D`（dense 锚）= `2·h·d`。
    * `ffn_macs_contract`（并列）：按激活比例缩放 FFN 项 `2·h·d·(k/h)`。
    """
    if arm == "D":
        k = h
        ffn_impl = L * (2 * h * d)
    elif arm == "F50":
        k = h // 2
        ffn_impl = L * (d * k + k * d)                 # 前缀切片：两个矩阵都省一半
    else:
        k = int(round(KEEP_FRAC[arm] * h))
        ffn_impl = L * (d * h + k * d)                 # lin1 全算，lin2 按 k
    ffn_contract = L * (2 * h * d) * (k / h)
    attn = L * (4 * d * d + 2 * T * d)
    readout = d * vocab
    total = attn + ffn_impl + readout
    return {
        "d_model": d, "hidden": h, "seq_len": T, "n_layers": L, "vocab": vocab,
        "k_active_channels": k,
        "ffn_macs_impl": ffn_impl, "ffn_macs_contract": ffn_contract,
        "macs_per_token": {"attention": attn, "ffn": ffn_impl,
                           "readout_tied": readout, "total": total},
        "macs_total_rel_to_D": round(total / (attn + L * 2 * h * d + readout), 6),
        "ffn_macs_rel_to_D": round(ffn_impl / (L * 2 * h * d), 6),
        "note": ("等效理论量：MACs/token（乘加），不含时间量。实现口径（primary）= "
                 "lin1 全算 + lin2 按 k（K*）/ 前缀切片（F50）；契约口径 = 按激活比例"
                 "缩放 FFN 项。选择器的比较/排序开销不计 MACs（见 analysis）。"),
    }


# ---------------------------------------------------------------- 训练（走共享 main）

def _factory_for(arm: str):
    def factory(cfg, vocab_size):
        return patch_ffn(DepthARModel(cfg, vocab_size), arm)
    return factory


def shared_argv(run_name: str, seed: int, steps: int, batch_size: int, lr: float,
                out_root: str) -> list[str]:
    return [os.path.join(ROOT, "train_depth_ar.py"), run_name,
            "--d", str(D_MODEL), "--layers", str(N_LAYERS),
            "--heads", str(N_HEADS), "--seq-len", str(SEQ_LEN),
            "--mlp-mult", str(MLP_MULT), "--dense", "--topology", "sequential",
            "--exit-mode", "uniform", "--steps", str(steps),
            "--batch-size", str(batch_size), "--lr", str(lr), "--seed", str(seed),
            "--eval-batches", "20", "--out-root", out_root]


def run_dir(arm: str, seed: int, steps: int, out_root: str) -> str:
    return os.path.join(out_root, f"{arm}-s{seed}-{steps // 1000}k")


def train_arm(arm: str, seed: int, steps: int, batch_size: int, lr: float,
              out_root: str) -> dict:
    orig = TDA.DepthARModel
    TDA.DepthARModel = _factory_for(arm)
    argv_backup = list(sys.argv)
    try:
        sys.argv = shared_argv(f"{arm}-s{seed}-{steps // 1000}k", seed, steps,
                               batch_size, lr, out_root)
        t0 = time.time()
        TDA.main()
        wall = time.time() - t0
    finally:
        TDA.DepthARModel = orig
        sys.argv = argv_backup
    return {"wall_s": round(wall, 1)}


# ---------------------------------------------------------------- 评测（本实验自持）

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def param_audit(model: DepthARModel) -> dict:
    total = sum(p.numel() for p in model.parameters())
    ffn = sum(p.numel() for blk in model.blocks
              for p in blk.mlp.parameters())
    return {"params_total": total, "params_ffn": ffn,
            "params_attn_and_rest": total - ffn}


@torch.no_grad()
def activation_profile(model: DepthARModel, eval_arr, n_batches: int = 20,
                       batch_size: int = 16, gen_seed: int = 2) -> dict:
    """P2 活性列：逐层逐通道激活频率 + 使用熵 + 逐 token 选择一致率。

    两趟：① 累积逐通道选中次数 ⇒ 众数集（每层 top-k）与 `q` 的熵；
    ② 统计每个 (层, token) 的选择集**恰好等于**该层众数集的比例（坍缩上界：`F50` = 1）。
    """
    T = model.cfg.seq_len
    ffns = sparse_ffns(model)
    k = ffns[0].k_active()
    model.eval()
    # --- pass 1（累积逐通道选中次数）
    gen = torch.Generator().manual_seed(gen_seed)
    for f in ffns:
        f.counts = torch.zeros(HIDDEN, dtype=torch.float64)
        f.n_tokens = 0
        f.mode_masks = None
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, gen)
            model(x.to(DEVICE), exit_depths=None)
    counts = [f.counts for f in ffns]
    n_tok = [f.n_tokens for f in ffns]
    for f in ffns:
        f.counts = None
    mode_sets = [torch.topk(c, k).indices.sort().values for c in counts]
    layer_stats = []
    for l in range(len(ffns)):
        q = (counts[l] / (n_tok[l] * k)).clamp_min(1e-12)
        ent = float(-(q * q.log()).sum())
        used = int((counts[l] > 0).sum())
        srt, _ = torch.sort(counts[l])
        layer_stats.append({
            "entropy_nats": round(ent, 6),
            "entropy_over_logH": round(ent / math.log(HIDDEN), 6),
            "uniform_value": 1.0,
            "collapse_value_k": round(math.log(k) / math.log(HIDDEN), 6),
            "channels_ever_used": used,
            "freq_deciles": [round(float(v) / (n_tok[l] * k), 6)
                             for v in torch.stack([srt[int(i * (HIDDEN - 1) / 9)]
                                                   for i in range(10)])],
        })
    # --- pass 2（选择一致率：mask 与逐位置众数集的比对**走同一段前向代码**）
    gen = torch.Generator().manual_seed(gen_seed)
    mode_masks = []
    for l in range(len(ffns)):
        m = torch.zeros(HIDDEN, dtype=torch.bool)
        m[mode_sets[l]] = True
        mode_masks.append(m)
    for f, m in zip(ffns, mode_masks):
        f.mode_masks = m
        f.mode_batch = []
    with torch.no_grad():
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, gen)
            model(x.to(DEVICE), exit_depths=None)
    per_batch_layer = [[float(r) for r, _ in f.mode_batch] for f in ffns]
    n_per_layer = [[int(n) for _, n in f.mode_batch] for f in ffns]
    for f in ffns:
        f.mode_masks = None
    per_batch = [sum(r * n for r, n in zip(per_batch_layer[l], n_per_layer[l]))
                 / max(sum(n_per_layer[l]), 1) for l in range(len(ffns))]
    matches = sum(r * n for l in range(len(ffns))
                  for r, n in zip(per_batch_layer[l], n_per_layer[l]))
    tot = sum(sum(nl) for nl in n_per_layer)
    layer_mode_share = [round(float(statistics.mean(per_batch_layer[l])), 6)
                        for l in range(len(ffns))]
    return {
        "k_active_channels": k, "hidden": HIDDEN,
        "n_batches": n_batches, "batch_size": batch_size, "n_tokens_per_layer": n_tok,
        "by_layer": layer_stats,
        "mode_set_share": round(matches / max(tot, 1), 6),
        "mode_set_share_by_layer": layer_mode_share,
        "mode_set_share_per_batch": [round(r, 6) for r in per_batch],
        "mode_set_share_wilson_95ci": wilson_ci(int(matches), int(tot)),
        "mode_set_share_cluster_bootstrap_95ci": cluster_bootstrap_ci(per_batch),
        "note": ("entropy_over_logH 两端：完全坍缩 = log(k)/log(H)（K50=0.9, K25=0.8）、"
                 "均匀随机 = 1.0；F50 构造性地落在坍缩端且 mode_set_share = 1.0。"
                 "mode_set_share = 每个 (层,token) 的选择集恰好等于该层众数集的比例。"),
    }


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(centre - half, 6), round(centre + half, 6)]


def cluster_bootstrap_ci(rates: list, n_iter: int = 2000, seed: int = 1234) -> list:
    if not rates:
        return [None, None]
    g = torch.Generator().manual_seed(seed)
    n = len(rates)
    ms = sorted(sum(rates[int(i)] for i in torch.randint(0, n, (n,), generator=g)) / n
                for _ in range(n_iter))
    return [round(ms[int(0.025 * n_iter)], 6),
            round(ms[min(int(0.975 * n_iter), n_iter - 1)], 6)]


@torch.no_grad()
def causal_check(model: DepthARModel, gen_seed: int = 7) -> dict:
    g = torch.Generator().manual_seed(gen_seed)
    T = model.cfg.seq_len
    x = torch.randint(0, VOCAB, (2, T), generator=g).to(DEVICE)
    x2 = x.clone()
    j = T // 2
    x2[:, j:] = torch.randint(0, VOCAB, (2, T - j), generator=g).to(DEVICE)
    model.eval()
    _, a1 = model(x, exit_depths=None)
    _, a2 = model(x2, exit_depths=None)
    lg1 = a1["depth_logits"][-1][:, :j].float()
    lg2 = a2["depth_logits"][-1][:, :j].float()
    return {"max_abs_diff_before_j": float((lg1 - lg2).abs().max()),
            "bitwise_equal_before_j": bool(torch.equal(lg1, lg2))}


@torch.no_grad()
def ffn_dead_zone(model: DepthARModel) -> dict:
    """F50 的未用通道梯度恒 0 的实测版：一次性真实前向 + backward，看逐通道梯度范数。"""
    g = torch.Generator().manual_seed(5)
    x = torch.randint(0, VOCAB, (2, SEQ_LEN), generator=g).to(DEVICE)
    model.train()
    loss, _ = model(x)
    model.zero_grad()
    loss.backward()
    out = []
    for blk in model.blocks:
        lin1, lin2 = blk.mlp.mlp[0], blk.mlp.mlp[2]
        g1 = lin1.weight.grad
        g2 = lin2.weight.grad
        out.append({"rows_nonzero_lin1": int((g1.abs().sum(-1) > 0).sum()),
                    "cols_nonzero_lin2": int((g2.abs().sum(0) > 0).sum())})
    model.zero_grad()
    return {"per_layer": out, "hidden": HIDDEN}


@torch.no_grad()
def evaluate(arm: str, seed: int, steps: int, out_root: str,
             n_batches: int = 20) -> dict:
    rd = run_dir(arm, seed, steps, out_root)
    ckpt = os.path.join(rd, "model.pt")
    train_arr, eval_arr, vocab = load_enwik8_full(".tmp/enwik8",
                                                  "data/enwik8_full.npz")
    model = build_model(arm, vocab, seed).to(DEVICE)
    state = torch.load(ckpt, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    curve = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                        device=DEVICE)
    curve2 = depth_curve(model, eval_arr, n_batches=n_batches, batch_size=16,
                         device=DEVICE)
    rows = [json.loads(l) for l in open(os.path.join(rd, "train_log.jsonl"))
            if l.strip()]
    shared = json.load(open(os.path.join(rd, "results.json")))
    out = {
        "arm": arm, "seed": seed, "steps": steps,
        "keep_frac": KEEP_FRAC[arm],
        "eval": {"metric": "clean bpc（最终深度、无退出、序列级、越低越好）",
                 "n_batches": n_batches, "batch_size": 16,
                 "eval_generator_seed": 2,
                 "bpc_by_depth": curve, "clean_bpc": curve[-1]["bpc"]},
        "params": param_audit(model),
        "cost": analytic_cost(arm, vocab=vocab),
        "activation_profile": activation_profile(model, eval_arr,
                                                 n_batches=n_batches),
        "causal": causal_check(model),
        "dead_zone_probe": ffn_dead_zone(model),
        "reeval_bitwise_identical": bool(curve == curve2),
        "loss_slope": loss_slope(rows, steps),
        "checkpoint": {"path": os.path.relpath(ckpt, ROOT),
                       "sha256": sha256_file(ckpt),
                       "missing_keys": list(missing),
                       "unexpected_keys": list(unexpected)},
        "shared_train_results": {"params": shared.get("params"),
                                 "train_tokens": shared.get("train_tokens"),
                                 "wall_minutes": shared.get("wall_minutes"),
                                 "peak_gpu_mem_mb": shared.get("peak_gpu_mem_mb")},
        "run_dir": os.path.relpath(rd, ROOT),
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
    }
    with open(os.path.join(HERE, f"results_{arm}_s{seed}.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[2644] {arm}-s{seed} clean_bpc={out['eval']['clean_bpc']} "
          f"params_total={out['params']['params_total']} "
          f"mode_set_share={out['activation_profile']['mode_set_share']} "
          f"entropy/logH={out['activation_profile']['by_layer'][0]['entropy_over_logH']}",
          flush=True)
    return out


def loss_slope(rows: list, steps: int, frac: float = 0.25) -> dict:
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
    return {"n_points": len(win), "slope_per_1k_steps": round(slope * 1000, 5),
            "window": f"末 {int(frac * 100)}%"}


# ---------------------------------------------------------------- 汇总（P1–P4）

def aggregate(steps: int, out_root: str, arms: list, seeds: list) -> dict:
    recs = {}
    for arm in arms:
        for s in seeds:
            p = os.path.join(HERE, f"results_{arm}_s{s}.json")
            if os.path.exists(p):
                recs[(arm, s)] = json.load(open(p))
    s0 = seeds[0]
    rows = {}
    for arm in arms:
        if (arm, s0) not in recs:
            rows[arm] = {"clean_bpc": None, "note": "未跑"}
            continue
        r = recs[(arm, s0)]
        rows[arm] = {"clean_bpc": r["eval"]["clean_bpc"],
                     "delta_vs_D": round(r["eval"]["clean_bpc"] - D_ANCHOR_BPC, 4),
                     "ffn_macs_impl": r["cost"]["ffn_macs_impl"],
                     "ffn_macs_contract": r["cost"]["ffn_macs_contract"],
                     "macs_total": r["cost"]["macs_per_token"]["total"],
                     "macs_total_rel_to_D": r["cost"]["macs_total_rel_to_D"],
                     "params_total": r["params"]["params_total"],
                     "entropy_over_logH": r["activation_profile"]["by_layer"][0]
                     ["entropy_over_logH"],
                     "mode_set_share": r["activation_profile"]["mode_set_share"],
                     "reeval_bitwise_identical": r["reeval_bitwise_identical"],
                     "slope_per_1k": r["loss_slope"]["slope_per_1k_steps"]}
    def verdict(arm):
        if rows.get(arm, {}).get("clean_bpc") is None:
            return "未跑"
        d = rows[arm]["delta_vs_D"]
        if d <= -VARIANCE_BOUND:
            return "可排序更优（相对 dense 锚）"
        if d >= VARIANCE_BOUND:
            return "可排序更差（相对 dense 锚）"
        return "不可排序（H0）：|Δ| < 0.014"
    k50 = rows.get("K50", {})
    f50 = rows.get("F50", {})
    p4 = None
    if k50.get("clean_bpc") is not None and f50.get("clean_bpc") is not None:
        dkf = round(k50["clean_bpc"] - f50["clean_bpc"], 4)
        if dkf <= -VARIANCE_BOUND:
            p4 = {"delta_K50_minus_F50": dkf,
                  "verdict": "条件性有效（K50 可排序优于无条件 F50）"}
        elif dkf >= VARIANCE_BOUND:
            p4 = {"delta_K50_minus_F50": dkf,
                  "verdict": "条件性**有害**（K50 可排序劣于 F50）：固定前缀更优"}
        else:
            p4 = {"delta_K50_minus_F50": dkf,
                  "verdict": ("机制无效：条件性无独立收益（|Δ| < 0.014），"
                              "本尺度判「保参数换计算 = 减参数换计算」的退化解")}
        p4["collapse"] = {"K50_mode_set_share": k50["mode_set_share"],
                          "F50_mode_set_share": f50["mode_set_share"],
                          "K50_entropy_over_logH": k50["entropy_over_logH"],
                          "K50_collapse_value": 0.9}
    out = {
        "experiment": "2644-channel-sparsity-13m",
        "criterion": ("P1 clean bpc（0.014 带）；P2 活性列（频率/熵/mode_set_share，"
                      "率类走 bootstrap/Wilson CI）；P3 与 U2-s0（减参数换计算）同框；P4 否证线。"),
        "variance_bound": VARIANCE_BOUND,
        "band_declaration": ("0.014 仅限 bpc 口径；使用率/熵/mode_set_share 等率类指标单独定门，"
                             "不跨量纲借用。"),
        "anchors": {"D_U4_s0_clean_bpc": D_ANCHOR_BPC, "D_U4_s0_macs_total": D_ANCHOR_MACS,
                    "U2_s0_clean_bpc": U2_ANCHOR_BPC, "U2_s0_macs_total": U2_ANCHOR_MACS,
                    "source": "2641-ffn-ratio-frontier-13m（复用，按 SCHEMA §9 三道门）"},
        "n_seeds": 1, "seed_note": "n=1（seed 0）：越带结论为单点快照，只报增量方向。",
        "cost_note": ("实现口径（primary）：按内容选通道必须先算全部 pre-activation ⇒ K50 真实省 "
                      "FFN 的 25%（不是 50%）；F50 固定前缀可同时切两个矩阵 ⇒ 省 50%。"),
        "arms": rows,
        "P1_verdict_vs_D": {a: verdict(a) for a in arms},
        "P3_K50_vs_U2": ({"K50_clean_bpc": k50.get("clean_bpc"),
                          "U2_s0_clean_bpc": U2_ANCHOR_BPC,
                          "delta_K50_minus_U2": (round(k50["clean_bpc"] - U2_ANCHOR_BPC, 4)
                                                 if k50.get("clean_bpc") is not None else None),
                          "macs_total_equal": (k50.get("macs_total") == U2_ANCHOR_MACS
                                               if k50.get("macs_total") else None),
                          "params_K50": k50.get("params_total"),
                          "params_U2_s0": 4861440,
                          "note": "保参数（K50/F50）vs 减参数（U2）在同一每 token 总 MACs 位点"}
                         if k50.get("macs_total") else None),
        "P4_falsification": p4,
        "source_files": [f"results_{a}_s{s}.json" for (a, s) in sorted(recs)],
        "code_hash_run_py": sha256_file(os.path.join(HERE, "run.py")),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: out[k] for k in ("arms", "P1_verdict_vs_D", "P4_falsification")},
                     indent=2, ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out-root", default=os.path.join(HERE, "runs"))
    ap.add_argument("--no-train", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--arms", default="K50,K25,F50")
    ap.add_argument("--seeds", default="0")
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.steps, args.out_root, args.arms.split(","),
                  [int(s) for s in args.seeds.split(",") if s.strip() != ""])
        return
    if not args.arm:
        raise SystemExit("--arm 或 --aggregate 必给其一")
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
