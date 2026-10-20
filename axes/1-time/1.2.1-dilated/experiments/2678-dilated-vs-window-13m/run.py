"""2678-dilated-vs-window-13m · 空洞采样 vs 连续窗口（`1.2.1` 的 WP1 判决腿）。

预登记判据见 `README.md` §4，判据常数在 `config.json`（两者与 `run.py` 一起被
`tools/prereg_bind.py --contract README.md config.json run.py` 绑住）。本脚本做三件事：

1. `--stage train --arm dil32 --seed N`：用**同一个训练函数** `train_depth_ar.train()`
   训一支空洞臂。掩码在实验目录内实现（`DilatedBlock` 把静态掩码 buffer 挂到 stock
   `Block` 上、走它既有的 `attn_mask` 路径），`pathlm/**` 与 `train_depth_ar.py` 一字不改；
   转换后逐参数核对「与 stock 构造逐位相同」（唯一变量 = 掩码）。
2. `--stage eval`：载入四臂（`win32` / `win32_s1` 复用 `2676`、`full` 复用 `2614`、
   `dil32` / `dil32_s1` 本实验训练），做对齐核查（判据④⑤）、等值门（复现 `2676` 的
   逐批 bpc）、配对 bpc（判据①⑥）与变距离召回电池（判据②③），写 `results.json`。
3. 判词由 `config.json::verdict_rules` 机械决定，脚本不另加解释。

用法：
    python3 run.py --stage train --arm dil32 --seed 0 --steps 12000
    python3 run.py --stage train --arm dil32 --seed 1 --steps 12000
    python3 run.py --stage eval
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import train_depth_ar as tdt                                   # noqa: E402
from pathlm.config import ModelConfig                          # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,      # noqa: E402
                             window_anchor_keep_mask)
from pathlm.data import load_enwik8_full, batch                # noqa: E402  (batch 用于 bpc 采样)
from tools.evalkit.longrange_tasks import var_recall_batch     # noqa: E402
from tools.evalkit.retention import masked_logits              # noqa: E402
from tools.prereg_bind import require as require_prereg        # noqa: E402

CFG = json.loads(Path(HERE, "config.json").read_text(encoding="utf-8"))
RECIPE = CFG["recipe"]
K = CFG["k"]
LOCAL, N_DIL = CFG["mask"]["local"], CFG["mask"]["n_dilated"]
D, L, H = RECIPE["d_model"], RECIPE["n_layers"], RECIPE["n_heads"]
SEQ, MLP = RECIPE["seq_len"], RECIPE["mlp_mult"]
NB, BSIZE, DSEED = CFG["eval"]["n_batches"], CFG["eval"]["batch_size"], CFG["eval"]["window_seed"]
LN2 = 0.6931471805599453
DEVICE = "cuda"
DATA = RECIPE["data"]

EXP_2614 = os.path.join(ROOT, "axes/1-time/1.2.3-content-topk/experiments/2614-topk-sparse-attn-13m")
EXP_2676 = os.path.join(ROOT, "axes/1-time/1.2.3-content-topk/experiments/2676-content-vs-window-13m")

ARMS = [
    {"name": "full", "kind": "full", "seed": 0,
     "ckpt": os.path.join(EXP_2614, "runs/full-12k-s0/model.pt"),
     "role": "参照线：全因果（每行 i+1 个键），只算「少看键的代价」"},
    {"name": "win32", "kind": "window", "seed": 0,
     "ckpt": os.path.join(EXP_2676, "runs/win32/model.pt"),
     "role": "对照：连续窗口 32（最近的 32 个位置），复用 2676"},
    {"name": "win32_s1", "kind": "window", "seed": 1,
     "ckpt": os.path.join(EXP_2676, "runs/win32_s1/model.pt"),
     "role": "对照的 seed 1（配对口径同 2676），复用 2676"},
    {"name": "dil32", "kind": "dilated", "seed": 0,
     "ckpt": os.path.join(HERE, "runs/dil32-s0/model.pt"),
     "role": "实验：近端窗口 16 + 16 个空洞点铺满前缀（同 32 键）"},
    {"name": "dil32_s1", "kind": "dilated", "seed": 1,
     "ckpt": os.path.join(HERE, "runs/dil32-s1/model.pt"),
     "role": "实验的 seed 1"},
]
PAIRS = {"s0": ("win32", "dil32"), "s1": ("win32_s1", "dil32_s1")}
BAND = float(CFG["band_bpc"])


# --------------------------------------------------------------------- 掩码


def dilated_keep_mask(T: int, local: int = LOCAL, n_dil: int = N_DIL,
                      device="cpu") -> torch.Tensor:
    """空洞采样掩码 `[T, T]`（bool，True = 该 (query, key) 对进入 softmax）。

    `keep[i, j] = (j <= i) ∧ ((i − j) < local ∨ j ∈ picks_i)`，其中
    `picks_i = {round(t·(P−1)/(n_dil−1)) : t = 0..n_dil−1}`、`P = i − local + 1`
    是可用前缀长度（`j ∈ [0, i−local]`）。行 `i + 1 <= local + n_dil` 退化为
    全因果 ⇒ 每行可见键数恒为 `min(local + n_dil, i + 1)`，与同 k 的连续窗口
    逐行相等（判据④）。

    逐行 dilation = `(P−1)/(n_dil−1)`：固定 stride 在 `i < local + 8·n_dil` 时
    取不满 `n_dil` 个点、会让两臂可见键数不等，故按行均匀铺满前缀。
    """
    idx = torch.arange(T, device=device)
    i, j = idx.unsqueeze(1), idx.unsqueeze(0)
    causal = j <= i
    keep = causal & ((i - j) < local)
    short = (i + 1) <= (local + n_dil)
    t = torch.arange(n_dil, device=device, dtype=torch.float).view(1, n_dil)
    prefix = (idx - local + 1).clamp_min(1).float().view(T, 1)     # P_i
    picks = torch.round(t * (prefix - 1) / (n_dil - 1)).long()     # [T, n_dil]
    dil = torch.zeros(T, T, dtype=torch.bool, device=device)
    dil.scatter_(1, picks.clamp(0, T - 1), True)
    keep = ((keep | dil) & causal) | (short.expand(T, T) & causal)
    return keep


class DilatedBlock(nn.Module):
    """stock `Block` + 一张静态空洞掩码（唯一差别 = 注意力可见集）。

    不重写注意力：把掩码按 SDPA 需要的形状展开后交给 `Block.forward(attn_mask=...)`
    的既有掩码路径（与 `DepthARModel._window_mask` 同款 expand），因此前向语义由
    共享实现保证、本类只负责「挑哪些键」。掩码 buffer 是 `persistent=False`，不进
    state_dict ⇒ 权重可从 stock `Block` 逐位搬过来（转换后由 `convert_to_dilated`
    逐参数核对）。
    """

    def __init__(self, cfg: ModelConfig, keep: torch.Tensor, mlp_hidden: int | None = None):
        super().__init__()
        from pathlm.model import Block
        self.block = Block(cfg, mlp_hidden=mlp_hidden)
        self.register_buffer("keep", keep, persistent=False)
        self.n_heads = cfg.n_heads

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                dist_pen: float = 0.0, kv_from: torch.Tensor | None = None,
                gate_vec: torch.Tensor | None = None):
        B, T, _ = x.shape
        if attn_mask is None:
            attn_mask = self.keep.view(1, 1, T, T).expand(B, self.n_heads, T, T)
        return self.block(x, attn_mask=attn_mask, dist_pen=dist_pen,
                          kv_from=kv_from, gate_vec=gate_vec)


def masked_module_names(model: nn.Module) -> list[str]:
    return [n for n, m in model.named_modules() if isinstance(m, DilatedBlock)]


def convert_to_dilated(model: DepthARModel, keep: torch.Tensor) -> dict:
    """把 stock `Block` 换成 `DilatedBlock`（权重逐参数搬过去），返回核对读数。

    构造 `DilatedBlock` 会消耗随机数，但权重随即被 state_dict 覆盖 ⇒ 主干参数与
    stock 构造逐位相同（判据⑤的证据：唯一变量 = 掩码）。
    """
    mcfg = ModelConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ, mlp_mult=MLP,
                       dropout=RECIPE["dropout"], norm_cap=RECIPE["norm_cap"])
    diffs, dropped = {}, []
    blocks = []
    for i, blk in enumerate(model.blocks):
        mh = None if getattr(blk, "mlp", None) is None else blk.mlp[0].out_features
        new = DilatedBlock(mcfg, keep, mlp_hidden=mh)
        new.block.load_state_dict(blk.state_dict())
        for (name, a), (name_b, b) in zip(blk.named_parameters(),
                                         new.block.named_parameters()):
            if name != name_b or not torch.equal(a, b):
                diffs[f"{i}.{name}"] = True
        dropped.append(sorted(set(blk.state_dict().keys())
                              ^ set(new.block.state_dict().keys())))
        blocks.append(new)
    model.blocks = nn.ModuleList(blocks)
    return {"params_identical_to_stock": not diffs, "diff_params": sorted(diffs),
            "state_dict_key_symmetric_diff": dropped,
            "masked_blocks": len(masked_module_names(model))}


# --------------------------------------------------------------------- 训练


def build_model(seed: int, vocab: int, dilated: bool) -> tuple[DepthARModel, dict]:
    torch.manual_seed(seed)                                    # 与 train_depth_ar.main 同序
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ, mlp_mult=MLP,
                        dropout=RECIPE["dropout"], norm_cap=RECIPE["norm_cap"],
                        dense=True, topology="sequential")
    model = DepthARModel(cfg, vocab)
    info = {"converted": None}
    if dilated:
        keep = dilated_keep_mask(SEQ, LOCAL, N_DIL)
        info["converted"] = convert_to_dilated(model, keep)
        assert info["converted"]["params_identical_to_stock"], "转换后参数不一致"
    return model.cuda(), info


def train_arm(arm: str, seed: int, steps: int) -> None:
    name = f"{arm}-s{seed}"
    run_dir = os.path.join(HERE, "runs", name)
    if arm != "dil32":
        raise SystemExit(f"本实验只训练空洞臂（dil32），收到 {arm}")
    os.makedirs(run_dir, exist_ok=True)
    train_arr, _, vocab = load_enwik8_full(".tmp/enwik8", DATA)
    model, info = build_model(seed, vocab, dilated=True)
    n_params = sum(p.numel() for p in model.parameters())
    rows = _row_counts()
    print(f"{name}: {n_params/1e6:.2f}M params  mask local={LOCAL} n_dil={N_DIL} "
          f"kept/row[{rows['min']},{rows['max']}] equals_min_k={rows['equals_min_k']} "
          f"steps {steps}", flush=True)
    tcfg = {"steps": steps, "batch_size": RECIPE["batch_size"], "lr": RECIPE["lr"],
            "seed": seed}
    torch.cuda.reset_peak_memory_stats()
    wall, _, _, stats = tdt.train(
        model, train_arr, tcfg,
        os.path.join(run_dir, "train_log.jsonl"),
        os.path.join(run_dir, "gate_log.jsonl"),
        exit_mode="uniform")
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    out = {
        "run": name, "arm": arm, "seed": seed, "steps": steps,
        "recipe": RECIPE, "mask": CFG["mask"],
        "n_params": n_params,
        "wall_minutes": round(wall / 60, 2),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "conversion_check": info["converted"],
        "row_counts": rows,
        "train_stats": {k: v for k, v in stats.items()
                        if k in ("gnorm_max", "gnorm_mean", "skipped_steps")},
    }
    json.dump(out, open(os.path.join(run_dir, "results.json"), "w"), indent=2,
              ensure_ascii=False)
    print(f"{name}: wall {out['wall_minutes']} min  peak {out['peak_gpu_mem_mb']} MiB",
          flush=True)


def _row_counts() -> dict:
    keep = dilated_keep_mask(SEQ, LOCAL, N_DIL)
    counts = keep.sum(-1).tolist()
    return {"per_row": counts, "min": min(counts), "max": max(counts),
            "equals_min_k": counts == [min(K, i + 1) for i in range(SEQ)]}


# --------------------------------------------------------------------- 评测


def cfg_for_eval() -> DepthARConfig:
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ, mlp_mult=MLP,
                         dropout=RECIPE["dropout"], norm_cap=RECIPE["norm_cap"],
                         dense=True, topology="sequential")


def load_arm(arm: dict, vocab: int) -> DepthARModel:
    torch.manual_seed(0)
    model = DepthARModel(cfg_for_eval(), vocab)
    if arm["kind"] == "dilated":
        keep = dilated_keep_mask(SEQ, LOCAL, N_DIL)
        info = convert_to_dilated(model, keep)
        assert info["params_identical_to_stock"] and info["masked_blocks"] == L
    model = model.to(DEVICE)
    state = torch.load(arm["ckpt"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def keep_mask_for(arm: dict):
    """评测时的可见集：窗口臂用 `window_anchor_keep_mask`、空洞臂用本实验的掩码
    （两臂各自与自己的训练期可见集逐位一致），`full` 交给 stock 因果路径。"""
    if arm["kind"] == "window":
        return window_anchor_keep_mask(SEQ, K, 0, device=DEVICE)
    if arm["kind"] == "dilated":
        return dilated_keep_mask(SEQ, LOCAL, N_DIL, device=DEVICE)
    return None


def per_batch_bpc(model: DepthARModel, keep, eval_arr) -> list[float]:
    gen = torch.Generator().manual_seed(DSEED)
    out = []
    with torch.no_grad():
        for _ in range(NB):
            x, _ = batch(eval_arr, BSIZE, SEQ, gen)
            x = x.to(DEVICE)
            logits = masked_logits(model, x, keep)[:, :-1].float()
            tgt = x[:, 1:]
            nats = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   tgt.reshape(-1), reduction="sum").item()
            out.append(round(nats / tgt.numel() / LN2, 5))
    return out


def paired_stats(comparator: list[float], arm: list[float]) -> dict:
    """Δ = 对照 − 实验（正 = 实验更好）。"""
    d = [c - a for c, a in zip(comparator, arm)]
    n = len(d)
    mean, sd = statistics.mean(d), statistics.stdev(d)
    se = sd / n ** 0.5
    return {"n_batches": n, "delta_bpc": round(mean, 5),
            "sd_of_paired_delta": round(sd, 5), "se_of_mean": round(se, 5),
            "t_stat": round(mean / se, 2) if se > 0 else None,
            "frac_batches_delta_gt_0": round(sum(1 for v in d if v > 0) / n, 3),
            "min_delta": round(min(d), 5), "max_delta": round(max(d), 5)}


def recall_curve(model: DepthARModel, keep, distances: list[int]) -> dict:
    """变距离回忆：逐批 top-1 准确率（率类指标，按批聚类）。"""
    rc = CFG["recall"]
    gen = torch.Generator().manual_seed(rc["task_seed"])
    out = {}
    for dist in distances:
        per_batch, chance = [], None
        for _ in range(rc["n_batches"]):
            task = var_recall_batch(rc["batch_size"], SEQ, dist, generator=gen)
            chance = 1.0 / task.n_answers
            logits = masked_logits(model, task.tokens.to(DEVICE), keep)
            pred = logits.argmax(-1)
            sel = torch.gather(pred, 1, task.rows.to(DEVICE))
            hit = (sel == task.answers.to(DEVICE)).float()
            per_batch.append([float(hit.sum()), float(hit.numel())])
        hits = sum(h for h, _ in per_batch)
        ns = sum(n for _, n in per_batch)
        out[f"d{dist}"] = {"per_batch": per_batch,
                           "accuracy": round(hits / ns, 4),
                           "chance": round(chance, 4),
                           "n": int(ns)}
    return out


def boot_ci_paired(a: list[float], b: list[float], n_boot: int, seed: int) -> dict:
    """按批聚类的配对 bootstrap（百分位法）——差值 `Δ = a − b`（a = 对照读数）。"""
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    g = torch.Generator().manual_seed(seed)
    means = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g).tolist()
        means.append(sum(diffs[i] for i in idx) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot) - 1]
    return {"delta_acc": round(statistics.mean(diffs), 5),
            "ci95": [round(lo, 5), round(hi, 5)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_batches": n,
            "n_boot": n_boot}


def alignment_checks() -> dict:
    """判据④：两臂逐行可见键数相等、掩码恒为因果子集、最远可达距离列。"""
    causal = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
    win = window_anchor_keep_mask(SEQ, K, 0)
    dil = dilated_keep_mask(SEQ, LOCAL, N_DIL)
    cw, cd = win.sum(-1).tolist(), dil.sum(-1).tolist()
    idx = torch.arange(SEQ)
    dist = idx.unsqueeze(1) - idx.unsqueeze(0)
    reach = {"win32_max": int(((win * dist).max())),
             "dil32_max": int(((dil * dist).max())),
             "first_row_where_they_differ": next(
                 (i for i in range(SEQ) if not torch.equal(win[i], dil[i])), None),
             "rows_differing": int(sum(1 for i in range(SEQ)
                                       if not torch.equal(win[i], dil[i])))}
    return {"per_row_count_equal": cw == cd,
            "mismatch_rows": int(sum(1 for a, b in zip(cw, cd) if a != b)),
            "per_row_count_equals_min_k": cw == [min(K, i + 1) for i in range(SEQ)],
            "win_kept_per_row": cw,
            "dil_kept_per_row": cd,
            "mask_offending_future_pairs": int((win & ~causal).sum() + (dil & ~causal).sum()),
            "reach": reach,
            "note": "逐行可见键数两臂相等且都 = min(32, i+1)；本字段即判据④的落盘列"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "eval"), required=True)
    ap.add_argument("--arm", default="dil32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=RECIPE["steps"])
    args = ap.parse_args()
    os.chdir(ROOT)

    if require_prereg(Path(HERE)) != 0:
        raise SystemExit("预登记未绑定：先跑 python3 tools/prereg_bind.py "
                         "axes/1-time/1.2.1-dilated/experiments/2678-dilated-vs-window-13m "
                         "--contract README.md config.json run.py")

    if args.stage == "train":
        train_arm(args.arm, args.seed, args.steps)
        return

    # ---- eval ----
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)
    align = alignment_checks()
    print(f"[align] per_row_equal={align['per_row_count_equal']} "
          f"mismatch={align['mismatch_rows']} future={align['mask_offending_future_pairs']} "
          f"reach={align['reach']}", flush=True)
    if not align["per_row_count_equal"] or align["mismatch_rows"] or \
            align["mask_offending_future_pairs"]:
        raise SystemExit("对齐核查失败：判决作废（README §5）")

    rc = CFG["recall"]
    distances = list(rc["near_distances"]) + list(rc["far_distances"])
    per_batch, params, bpc_mean, recall = {}, {}, {}, {}
    for arm in ARMS:
        if not os.path.exists(arm["ckpt"]):
            raise SystemExit(f"缺 checkpoint：{arm['ckpt']}（先跑 train_arms.sh）")
        model = load_arm(arm, vocab)
        keep = keep_mask_for(arm)
        params[arm["name"]] = sum(p.numel() for p in model.parameters())
        per_batch[arm["name"]] = per_batch_bpc(model, keep, eval_arr)
        bpc_mean[arm["name"]] = round(statistics.mean(per_batch[arm["name"]]), 5)
        recall[arm["name"]] = recall_curve(model, keep, distances)
        del model
        torch.cuda.empty_cache()
        print(f"[eval] {arm['name']}: bpc {bpc_mean[arm['name']]} params {params[arm['name']]}",
              flush=True)

    # 等值门：复用臂读数必须复现 2676 的落盘逐批 bpc（口径未漂移）
    ref = json.load(open(os.path.join(EXP_2676, "results.json")))["per_batch_bpc"]
    reuse_ref = {}
    for name in ("win32", "win32_s1"):
        same = per_batch[name] == [round(v, 5) for v in ref[name]]
        reuse_ref[name] = {"reproduced": same,
                           "ref": "2676-content-vs-window-13m/results.json::per_batch_bpc"}
        if not same:
            raise SystemExit(f"等值门失败：{name} 的逐批 bpc 未复现 2676 落盘值")

    vs_full = {name: paired_stats(per_batch["full"], per_batch[name])
               for name in ("win32", "dil32")}
    by_seed = {seed: paired_stats(per_batch[w], per_batch[d]) for seed, (w, d) in PAIRS.items()}
    d0, d1 = by_seed["s0"]["delta_bpc"], by_seed["s1"]["delta_bpc"]
    same_sign = (d0 > 0) == (d1 > 0)
    both_beyond = abs(d0) >= BAND and abs(d1) >= BAND

    recall_delta = {}
    for seed, (w, d) in PAIRS.items():
        recall_delta[seed] = {}
        for dist in distances:
            a = [h / n for h, n in recall[w][f"d{dist}"]["per_batch"]]
            b = [h / n for h, n in recall[d][f"d{dist}"]["per_batch"]]
            recall_delta[seed][f"d{dist}"] = boot_ci_paired(
                a, b, rc["n_boot"], rc["boot_seed"])
    far_sep = {seed: [k for k, v in recall_delta[seed].items()
                      if k in [f"d{x}" for x in rc["far_distances"]] and v["excludes_zero"]]
               for seed in PAIRS}
    near_sep = {seed: [k for k, v in recall_delta[seed].items()
                       if k in [f"d{x}" for x in rc["near_distances"]] and v["excludes_zero"]]
                for seed in PAIRS}
    recall_separable = any(far_sep[s] for s in PAIRS)

    if same_sign and both_beyond and d0 > 0:
        main_verdict = "空洞采样占优（两 seed 同向且各自越带 ⇒ n=2）"
    elif same_sign and both_beyond and d0 < 0:
        main_verdict = "空洞采样更差（两 seed 同向且各自越带 ⇒ n=2）：按位置最近仍占优"
    elif not same_sign:
        main_verdict = "两 seed 不同向 ⇒ 本 n 下不可排序（不解释成噪声、不取平均掩盖）"
    else:
        main_verdict = "两 seed 同向但至少一 seed 未越带 ⇒ 本 n 下不可排序"

    if recall_separable:
        recall_verdict = "远档召回可分（至少一 seed 的远档 CI 不含 0）"
    else:
        recall_verdict = "远档召回不可分（两 seed 的全部远档 CI 都含 0）"

    falsified = (not recall_separable) and (not (same_sign and both_beyond)) \
        and all(abs(x) < BAND for x in (d0, d1))
    if falsified:
        overall = "判『空洞采样在本尺度无净增量、不成立』：远档召回不可分 且 两 seed 的 bpc 差都落 0.014 带内"
    elif same_sign and both_beyond and d0 < 0:
        overall = "判『按位置最近仍占优』：bpc 两 seed 同向越带为负"
    elif same_sign and both_beyond and d0 > 0:
        overall = "判『空洞采样占优』：bpc 两 seed 同向越带为正"
    else:
        overall = "本 n 下不可排序（判据①未达两 seed 同向越带、判据②远档不可分）"

    out = {
        "experiment": "2678-dilated-vs-window-13m",
        "delta_directions": {
            "delta_bpc": "comparator_minus_arm",
            "delta_acc": "comparator_minus_arm",
            "sd_of_paired_delta": "comparator_minus_arm",
            "min_delta": "comparator_minus_arm",
            "max_delta": "comparator_minus_arm",
            "frac_batches_delta_gt_0": "comparator_minus_arm",
            "ci95": "comparator_minus_arm",
        },
        "delta_direction_note": (
            "所有 Δ 一律 Δ = 读数_对照 − 读数_实验：对照 = win32（连续窗口），实验 = dil32"
            "（空洞采样）⇒ 正 = 空洞占优。`vs_full` 的对照 = full、实验 = 该臂。"),
        "band_bpc": BAND,
        "band_note": CFG["band_note"],
        "config": {"recipe": RECIPE, "k": K, "mask": CFG["mask"],
                   "eval": CFG["eval"], "recall": rc,
                   "seeds": CFG["seeds"]},
        "arms": [{"name": a["name"], "kind": a["kind"], "seed": a["seed"],
                  "role": a["role"], "ckpt": os.path.relpath(a["ckpt"], ROOT),
                  "params": params[a["name"]]} for a in ARMS],
        "alignment_checks": align,
        "params_all_equal": len(set(params.values())) == 1,
        "params_value": params["win32"],
        "mean_bpc_by_arm": bpc_mean,
        "per_batch_bpc": per_batch,
        "paired_vs_full": vs_full,
        "paired_dilated_vs_window": by_seed,
        "recall_by_arm": recall,
        "recall_delta_window_minus_dilated": recall_delta,
        "recall_separable": {"far": far_sep, "near": near_sep},
        "cross_seed_reference_line": {
            "note": ("同一臂跨 seed 之差是两个独立训练 run 之差，只作参考线，"
                     "不与 0.014 配对带混用（docs/methodology.md §2）"),
            "win32_abs_diff": round(abs(bpc_mean["win32_s1"] - bpc_mean["win32"]), 5),
            "dil32_abs_diff": round(abs(bpc_mean["dil32_s1"] - bpc_mean["dil32"]), 5),
        },
        "main_verdict": {
            "criterion": ("① Δ = bpc_win32 − bpc_dil32（越 ±0.014 带才算可分）；"
                          "② 远档召回按批聚类配对 bootstrap 95% CI 不含 0 才算可分"),
            "quality": main_verdict,
            "recall": recall_verdict,
            "overall": overall,
            "n_seed": len(PAIRS),
            "falsified": bool(falsified),
        },
        "wall_reference": {"wall_in_criterion": False,
                           "note": "训练墙钟与峰值显存逐臂申报（train_declaration），不进判据（docs/methodology.md §3）"},
        "reuse_equivalence": reuse_ref,
        "train_declaration": {a["name"]: _decl(a["name"]) for a in ARMS},
    }
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=2,
              ensure_ascii=False)
    print(json.dumps({"mean_bpc_by_arm": bpc_mean,
                      "paired": by_seed,
                      "verdict": out["main_verdict"]}, ensure_ascii=False, indent=1))


def _decl(name: str) -> dict:
    """逐臂训练申报（墙钟 / 峰值显存，不进判据）；复用臂读它们的落盘 run。"""
    for arm in ARMS:
        if arm["name"] == name:
            p = os.path.join(os.path.dirname(arm["ckpt"]), "results.json")
            if os.path.exists(p):
                r = json.load(open(p))
                return {"wall_minutes": r.get("wall_minutes"),
                        "peak_gpu_mem_mb": r.get("peak_gpu_mem_mb"),
                        "source": os.path.relpath(p, ROOT)}
            return {"wall_minutes": None, "peak_gpu_mem_mb": None,
                    "source": os.path.relpath(os.path.dirname(arm["ckpt"]), ROOT)}
    return {}


if __name__ == "__main__":
    main()
