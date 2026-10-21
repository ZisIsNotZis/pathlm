"""2690-dilated-density-k144-13m · 长 T 上「远处点的数目」买不买得到质量（`1.2.1` 的支持读数）。

判据见 `README.md` §4，常数在 `config.json`（两文件与 `run.py` / `train_arms.sh` 一起被
`tools/prereg_bind.py --contract` 绑住）。本实验只改一件事：把 `2689` 的 `dil32` 掩码
（近端 16 + 16 个远处点）的远处点数从 16 提到 128（近端 16 不变 ⇒ 每行 32 → 144 个键）。
训练配方、token 预算、评测口径都与 `2689` 的 T=2048 点逐字段相同，对照臂的读数在**本实验的仪器上
重算**（等值门要求它复现 `2689` 的逐批 bpc）。

用法：
    python3 run.py --stage train --arm dil144-t2048-s0
    python3 run.py --stage eval
"""

from __future__ import annotations

import argparse
import json
import math
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

import train_depth_ar as tdt                                     # noqa: E402
from pathlm.config import ModelConfig                            # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,        # noqa: E402
                             window_anchor_keep_mask)
from pathlm.data import load_enwik8_full, batch                  # noqa: E402
from tools.evalkit.retention import masked_logits                # noqa: E402
from tools.prereg_bind import require as require_prereg          # noqa: E402
from tools.run_provenance import write_results                   # noqa: E402

CFG = json.loads(Path(HERE, "config.json").read_text(encoding="utf-8"))
RECIPE = CFG["recipe"]
LOCAL, NDIL = int(CFG["mask"]["local"]), int(CFG["mask"]["n_dilated"])
D, L, H = RECIPE["d_model"], RECIPE["n_layers"], RECIPE["n_heads"]
MLP, DATA, SEQ = RECIPE["mlp_mult"], RECIPE["data"], RECIPE["seq_len"]
BAND = float(CFG["band_bpc"])
NB, BSIZE, DSEED = CFG["eval"]["n_batches"], CFG["eval"]["batch_size"], CFG["eval"]["window_seed"]
LN2 = 0.6931471805599453
DEVICE = "cuda"
K_ARM = LOCAL + NDIL                      # 本实验每行的可见键数（144）
ARMS = {a["name"]: a for a in CFG["arms"]}
EXP_2689 = os.path.join(ROOT, "axes/1-time/1.2.1-dilated/experiments/2689-longT-layerdil-13m")


def dil_keep(seq: int, local: int = LOCAL, n_dil: int = NDIL, device="cpu") -> torch.Tensor:
    """远处点数 = `n_dil` 的摊开掩码（与 `2689` 的 `dil_keep` 同一条规则）。

    `keep[i,j] = (j <= i) ∧ ((i-j) < local ∨ j ∈ {round(t·(P-1)/(n_dil-1))})`，
    `P = i-local+1`；`i+1 <= local+n_dil` 的行退化为全因果 ⇒ 每行 = `min(local+n_dil, i+1)`。
    """
    idx = torch.arange(seq, device=device)
    i, j = idx.unsqueeze(1), idx.unsqueeze(0)
    causal = j <= i
    keep = causal & ((i - j) < local)
    short = (i + 1) <= (local + n_dil)
    t = torch.arange(n_dil, device=device, dtype=torch.float).view(1, n_dil)
    prefix = (idx - local + 1).clamp_min(1).float().view(seq, 1)
    picks = torch.round(t * (prefix - 1) / (n_dil - 1)).long()
    dil = torch.zeros(seq, seq, dtype=torch.bool, device=device)
    dil.scatter_(1, picks.clamp(0, seq - 1), True)
    return ((keep | dil) & causal) | (short.expand(seq, seq) & causal)


class DilatedBlock(nn.Module):
    """stock `Block` + 一张静态掩码（唯一差别 = 注意力可见集，同 `2689` 的实现）。"""

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


def convert_to_dilated(model: DepthARModel, n_dil: int) -> dict:
    """把 stock `Block` 换成带掩码的 `DilatedBlock`（权重逐参数搬过去）并核对（判据②）。"""
    keep = dil_keep(SEQ, LOCAL, n_dil)
    mcfg = ModelConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ, mlp_mult=MLP,
                       dropout=RECIPE["dropout"], norm_cap=RECIPE["norm_cap"])
    diffs = {}
    blocks = []
    for i, blk in enumerate(model.blocks):
        mh = None if getattr(blk, "mlp", None) is None else blk.mlp[0].out_features
        new = DilatedBlock(mcfg, keep, mlp_hidden=mh)
        new.block.load_state_dict(blk.state_dict())
        for (name, a), (name_b, b) in zip(blk.named_parameters(),
                                          new.block.named_parameters()):
            if name != name_b or not torch.equal(a, b):
                diffs[f"{i}.{name}"] = True
        blocks.append(new)
    model.blocks = nn.ModuleList(blocks)
    return {"params_identical_to_stock": not diffs, "diff_params": sorted(diffs),
            "masked_blocks": len(blocks), "n_dilated": n_dil}


def model_cfg() -> DepthARConfig:
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ, mlp_mult=MLP,
                         dropout=RECIPE["dropout"], norm_cap=RECIPE["norm_cap"],
                         dense=True, topology="sequential", train_window=0, train_anchors=0)


def far_spacing(seq: int, n_dil: int, local: int = LOCAL) -> float:
    """远处点的平均间距（前缀长度 / (n_dil−1)，按行平均）——密度这个「唯一变量」的量化。"""
    prefix = torch.arange(seq).clamp_min(local).float() - local + 1
    return round(float((prefix / (n_dil - 1)).mean()), 2)


def row_counts(n_dil: int) -> dict:
    counts = dil_keep(SEQ, LOCAL, n_dil).sum(-1).tolist()
    k = LOCAL + n_dil
    return {"per_row": counts, "min": min(counts), "max": max(counts),
            "equals_min_k": counts == [min(k, i + 1) for i in range(SEQ)]}


def train_arm(name: str, steps: int | None = None, out_root: str = "runs") -> None:
    spec = ARMS[name]
    if not spec["trained"]:
        raise SystemExit(f"{name} 是复用臂（不训练）")
    seed = int(spec["seed"])
    steps = int(steps) if steps else int(RECIPE["steps"])
    run_dir = os.path.join(HERE, out_root, name)
    os.makedirs(run_dir, exist_ok=True)
    train_arr, _, vocab = load_enwik8_full(".tmp/enwik8", DATA)
    torch.manual_seed(seed)
    model = DepthARModel(model_cfg(), vocab)
    info = convert_to_dilated(model, NDIL)
    assert info["params_identical_to_stock"], "转换后参数不一致"
    n_params = sum(p.numel() for p in model.parameters())
    model = model.cuda()
    rows = row_counts(NDIL)
    print(f"{name}: {n_params/1e6:.2f}M params  seq {SEQ}  steps {steps}  "
          f"kept/row[{rows['min']},{rows['max']}] equals_min_k={rows['equals_min_k']}", flush=True)
    tcfg = {"steps": steps, "batch_size": RECIPE["batch_size"], "lr": RECIPE["lr"], "seed": seed}
    torch.cuda.reset_peak_memory_stats()
    wall, _, _, stats = tdt.train(model, train_arr, tcfg,
                                 os.path.join(run_dir, "train_log.jsonl"),
                                 os.path.join(run_dir, "gate_log.jsonl"),
                                 exit_mode="uniform")
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    out = {"run": name, "arm": name, "seed": seed, "steps": steps,
           "tokens": steps * RECIPE["batch_size"] * SEQ,
           "recipe": RECIPE, "mask": CFG["mask"], "n_params": n_params,
           "wall_minutes": round(wall / 60, 2),
           "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
           "conversion_check": info, "row_counts": rows,
           "train_stats": {k: v for k, v in stats.items()
                           if k in ("gnorm_max", "gnorm_mean", "grad_skips")}}
    write_results(os.path.join(run_dir, "results.json"), out, source=__file__)
    print(f"{name}: wall {out['wall_minutes']} min  peak {out['peak_gpu_mem_mb']} MiB  "
          f"gnorm_max {out['train_stats'].get('gnorm_max')}", flush=True)


def load_arm(name: str, vocab: int) -> DepthARModel:
    spec = ARMS[name]
    torch.manual_seed(0)
    model = DepthARModel(model_cfg(), vocab)
    if spec["trained"]:
        info = convert_to_dilated(model, NDIL)
    else:                       # 对照臂是 2689 的 dil32（n_dil = 16）
        keep = dil_keep(SEQ, LOCAL, 16)
        mcfg = ModelConfig(d_model=D, n_layers=L, n_heads=H, seq_len=SEQ, mlp_mult=MLP,
                           dropout=RECIPE["dropout"], norm_cap=RECIPE["norm_cap"])
        blocks = []
        for blk in model.blocks:
            mh = None if getattr(blk, "mlp", None) is None else blk.mlp[0].out_features
            new = DilatedBlock(mcfg, keep, mlp_hidden=mh)
            new.block.load_state_dict(blk.state_dict())
            blocks.append(new)
        model.blocks = nn.ModuleList(blocks)
        info = {"params_identical_to_stock": True, "n_dilated": 16}
    assert info["params_identical_to_stock"]
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(ROOT, spec["ckpt"]), map_location=DEVICE))
    model.eval()
    return model


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
    return {"n_batches": n, "delta_bpc": round(mean, 5), "sd_of_paired_delta": round(sd, 5),
            "se_of_mean": round(se, 5), "two_se": round(1.96 * se, 5),
            "t_stat": round(mean / se, 2) if se > 0 else None,
            "frac_batches_delta_gt_0": round(sum(1 for v in d if v > 0) / n, 3),
            "n_req_for_band": int(math.ceil((1.96 * sd / BAND) ** 2)) if sd > 0 else None}


def band_class(delta: float) -> str:
    if delta >= BAND:
        return "beyond_positive"
    if delta <= -BAND:
        return "beyond_negative"
    return "in_band"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "eval"), required=True)
    ap.add_argument("--arm", default="")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()
    os.chdir(ROOT)

    if require_prereg(Path(HERE)) != 0:
        raise SystemExit("预登记未绑定：先跑 python3 tools/prereg_bind.py "
                         "axes/1-time/1.2.1-dilated/experiments/2690-dilated-density-k144-13m "
                         "--contract README.md config.json run.py train_arms.sh")

    if args.stage == "train":
        train_arm(args.arm, args.steps or None, args.out_root)
        return

    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)
    win = window_anchor_keep_mask(SEQ, 32, 0)
    old = dil_keep(SEQ, LOCAL, 16)
    new = dil_keep(SEQ, LOCAL, NDIL)
    causal = torch.tril(torch.ones(SEQ, SEQ, dtype=torch.bool))
    structure = {
        "seq": SEQ,
        "kept_per_row_arm": new.sum(-1).tolist()[-1],
        "kept_per_row_comparator": old.sum(-1).tolist()[-1],
        "kept_rows_equal_to_min_k_arm": new.sum(-1).tolist() == [min(K_ARM, i + 1) for i in range(SEQ)],
        "kept_rows_equal_to_min_k_comparator": old.sum(-1).tolist() == [min(32, i + 1) for i in range(SEQ)],
        "mask_offending_future_pairs": int((new & ~causal).sum() + (old & ~causal).sum()),
        "same_local_window": bool(torch.equal(new & ((torch.arange(SEQ).view(-1, 1)
                                                     - torch.arange(SEQ).view(1, -1)) < LOCAL),
                                            old & ((torch.arange(SEQ).view(-1, 1)
                                                    - torch.arange(SEQ).view(1, -1)) < LOCAL))),
        "far_spacing_mean_arm": far_spacing(SEQ, NDIL),
        "far_spacing_mean_comparator": far_spacing(SEQ, 16),
        "note": "两臂的近端 16 窗口逐位相同（同一 local），差别只在远处点的数目与间距；"
                "逐行计数 = min(local + n_dil, i+1)",
    }
    print(f"[structure] {structure}", flush=True)
    if not (structure["kept_rows_equal_to_min_k_arm"]
            and structure["kept_rows_equal_to_min_k_comparator"]
            and structure["mask_offending_future_pairs"] == 0
            and structure["same_local_window"]):
        raise SystemExit("结构核查失败：判决作废（README §5）")

    per_batch, bpc_mean, params, not_run = {}, {}, {}, []
    for name, spec in ARMS.items():
        if not os.path.exists(os.path.join(ROOT, spec["ckpt"])):
            not_run.append(name)
            print(f"[eval] {name}: 缺 checkpoint ⇒ 登记为未跑", flush=True)
            continue
        model = load_arm(name, vocab)
        keep = dil_keep(SEQ, LOCAL, NDIL if spec["trained"] else 16, device=DEVICE)
        params[name] = sum(p.numel() for p in model.parameters())
        per_batch[name] = per_batch_bpc(model, keep, eval_arr)
        bpc_mean[name] = round(statistics.mean(per_batch[name]), 5)
        del model
        torch.cuda.empty_cache()
        print(f"[eval] {name}: bpc {bpc_mean[name]} params {params[name]}", flush=True)

    # 等值门：本实验仪器重算的 2689 对照臂读数必须复现 2689 的落盘逐批 bpc
    ref = json.load(open(os.path.join(EXP_2689, "results.json")))["per_batch_bpc"]
    reuse_ref = {}
    for name in ("dil32-t2048-s0", "dil32-t2048-s1"):
        same = name in per_batch and per_batch[name] == [round(v, 5) for v in ref[name]]
        reuse_ref[name] = {"reproduced": same,
                           "ref": "2689-longT-layerdil-13m/results.json::per_batch_bpc"}
        if not same:
            raise SystemExit(f"等值门失败：{name} 的逐批 bpc 未复现 2689 落盘值（README §5）")

    pairs = {}
    for pid, p in CFG["pairs"].items():
        c, a = p["comparator"], p["arm"]
        if c not in per_batch or a not in per_batch:
            pairs[pid] = {"not_run": True,
                          "missing": [n for n in (c, a) if n not in per_batch]}
            continue
        st = paired_stats(per_batch[c], per_batch[a])
        pairs[pid] = {"comparator": c, "arm": a, **st,
                      "band_class": band_class(st["delta_bpc"]),
                      "params_equal": params.get(c) == params.get(a)}
    deltas = [v["delta_bpc"] for v in pairs.values() if "not_run" not in v]
    classes = [v["band_class"] for v in pairs.values() if "not_run" not in v]
    if len(deltas) < 2:
        verdict = "未跑（掉臂规则）：两个 seed 的对照或实验臂缺 checkpoint，本项不作判词"
    else:
        same_sign = len({d > 0 for d in deltas}) == 1
        beyond = all(abs(d) >= BAND for d in deltas)
        if not same_sign:
            verdict = "两 seed 不同向 ⇒ 本 n 下不可排序（不解释成噪声、不取平均掩盖）"
        elif beyond and deltas[0] > 0:
            verdict = ("远处点数买得到质量（两 seed 同向且各自越带 ⇒ n=2）："
                       "⇒ 2689 的固定 k 答案是「本 k 分辨不出」")
        elif beyond:
            verdict = "加远处点反而更差（两 seed 同向且各自越负带）"
        else:
            verdict = ("加远处点买不到质量（两 seed 同向但未同时越带）："
                       "⇒ 长 T 上远处信息在本尺度不值钱")
    arms_over = []
    for name in ARMS:
        p = os.path.join(ROOT, os.path.dirname(ARMS[name]["ckpt"]), "results.json")
        if os.path.exists(p):
            g = (json.load(open(p)).get("train_stats") or {}).get("gnorm_max")
            if g is not None and g >= 100:
                arms_over.append(name)

    out = {
        "experiment": CFG["experiment"],
        "cell": CFG["cell"],
        "question": CFG["question"],
        "band_bpc": BAND,
        "delta_directions": {"delta_bpc": "comparator_minus_arm"},
        "delta_direction_note": ("Δ = bpc(16 个远处点) − bpc(128 个远处点)（对照 − 实验）⇒ "
                                 "正 = 远处点多的那臂更好"),
        "recipe": RECIPE,
        "mask": CFG["mask"],
        "structure_checks": structure,
        "per_batch_bpc": per_batch,
        "mean_bpc": bpc_mean,
        "n_params": params,
        "arms_not_run": not_run,
        "criteria": {
            "c1_density": {
                "note": "Δ = bpc(16 个远处点的臂) − bpc(128 个远处点的臂)，正 = 远处点多更好；"
                        "0.014 配对带；每行可见键数 32 → 144 是成本列、不进判词",
                "verdict": verdict,
                "pairs": pairs,
                "n_seed": len(deltas),
            },
            "c2_must_not_change": {
                "structure": structure,
                "params_equal_between_arms": all(v.get("params_equal", False)
                                                 for v in pairs.values() if "not_run" not in v),
            },
            "c3_reuse_equivalence": reuse_ref,
            "c4_stability_gate": {"threshold": 100, "arms_over_threshold": arms_over,
                                  "note": "gnorm_max < 100 且 grad_skips == 0 才入场"},
        },
        "main_verdict": {"density": verdict, "n_seed": len(deltas),
                         "band_classes": classes, "deltas": deltas},
        "train_declaration": {n: _decl(n) for n in ARMS},
        "wall_reference": {"wall_in_criterion": False,
                           "note": "训练墙钟与峰值显存逐臂申报，不进判据（harness/methodology.md §3）"},
    }
    write_results(os.path.join(HERE, "results.json"), out, source=__file__)
    print(json.dumps({"mean_bpc": bpc_mean, "verdict": verdict}, ensure_ascii=False, indent=1))


def _decl(name: str) -> dict:
    p = os.path.join(ROOT, os.path.dirname(ARMS[name]["ckpt"]), "results.json")
    if os.path.exists(p):
        r = json.load(open(p))
        return {"wall_minutes": r.get("wall_minutes"),
                "peak_gpu_mem_mb": r.get("peak_gpu_mem_mb"),
                "gnorm_max": (r.get("train_stats") or {}).get("gnorm_max"),
                "source": os.path.relpath(p, ROOT)}
    return {"wall_minutes": None, "peak_gpu_mem_mb": None, "gnorm_max": None,
            "source": os.path.relpath(p, ROOT), "note": "该臂未跑"}


if __name__ == "__main__":
    main()
