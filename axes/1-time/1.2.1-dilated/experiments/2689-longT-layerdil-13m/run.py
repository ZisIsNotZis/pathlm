"""2689-longT-layerdil-13m · 长 T 上的空洞采样 + 逐层递增 dilation（`1.2.1` 的第二轮判决腿）。

预登记判据见 `README.md` §4，判据常数（臂清单 / 配对表 / 判词规则）在 `config.json`，
两者与 `run.py` / `train_arms.sh` 一起被 `tools/prereg_bind.py --contract` 绑住。本脚本做两件事：

1. `--stage train --arm <臂名>`：用**同一个训练函数** `train_depth_ar.train()` 训一支新臂。
   掩码在实验目录内实现（`DilatedBlock` 把静态掩码 buffer 挂到 stock `Block` 上、走它既有的
   `attn_mask` 路径），`pathlm/**` 与 `train_depth_ar.py` 一字不改；转换后逐参数核对
   「与 stock 构造逐位相同」（唯一变量 = 可见集）。
2. `--stage eval`：载入三个 T 点的全部臂（T=256 四臂复用 `2676` / `2678`），做
   - 结构核查（判据④）、参数量与构造路径核查（判据⑤）、等值门（判据⑥）；
   - 每个臂的配对 clean bpc（判据①②）与「去掉 far 点」的消融读数（判据③）；
   写 `results.json`（含 `delta_directions` 字段级映射）。判词由 `config.json::verdict_rules`
   机械决定，脚本不另加解释。

用法：
    python3 run.py --stage train --arm dil32-t2048-s0
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
K = int(CFG["k"])
LOCAL, NDIL = int(CFG["mask"]["local"]), int(CFG["mask"]["n_dilated"])
STRIDES = [int(s) for s in CFG["mask"]["layer_strides"]]
D, L, H = RECIPE["d_model"], RECIPE["n_layers"], RECIPE["n_heads"]
MLP, DATA = RECIPE["mlp_mult"], RECIPE["data"]
BAND = float(CFG["band_bpc"])
EVAL = CFG["eval"]
NB, BSIZE, DSEED = EVAL["n_batches"], EVAL["batch_size"], EVAL["window_seed"]
LN2 = 0.6931471805599453
DEVICE = "cuda"
ARMS = {a["name"]: a for a in CFG["arms"]}
EXP_2678 = os.path.join(ROOT, "axes/1-time/1.2.1-dilated/experiments/2678-dilated-vs-window-13m")


# --------------------------------------------------------------------- 掩码


def window_keep(seq: int, k: int = K, device="cpu") -> torch.Tensor:
    """连续窗口（对照臂）：最近的 `k` 个位置。走引擎既有的 `window_anchor_keep_mask`。"""
    return window_anchor_keep_mask(seq, k, 0, device=device)


def dil_keep(seq: int, local: int = LOCAL, n_dil: int = NDIL, device="cpu") -> torch.Tensor:
    """平铺空洞掩码 `[T, T]`（bool，True = 该 (query, key) 对进入 softmax）。

    `keep[i, j] = (j <= i) ∧ ((i − j) < local ∨ j ∈ picks_i)`，其中
    `picks_i = {round(t·(P−1)/(n_dil−1)) : t = 0..n_dil−1}`、`P = i − local + 1`
    是可用前缀长度。行 `i + 1 <= local + n_dil` 退化为全因果 ⇒ 每行可见键数恒为
    `min(local + n_dil, i + 1)`（判据④）。与 `2678` 的 `dil32` 是同一条规则
    （等值门⑥ 用 T=256 的逐批 bpc 复现来钉住这一点）。
    """
    idx = torch.arange(seq, device=device)
    i, j = idx.unsqueeze(1), idx.unsqueeze(0)
    causal = j <= i
    keep = causal & ((i - j) < local)
    short = (i + 1) <= (local + n_dil)
    t = torch.arange(n_dil, device=device, dtype=torch.float).view(1, n_dil)
    prefix = (idx - local + 1).clamp_min(1).float().view(seq, 1)     # P_i
    picks = torch.round(t * (prefix - 1) / (n_dil - 1)).long()       # [T, n_dil]
    dil = torch.zeros(seq, seq, dtype=torch.bool, device=device)
    dil.scatter_(1, picks.clamp(0, seq - 1), True)
    keep = ((keep | dil) & causal) | (short.expand(seq, seq) & causal)
    return keep


def layered_keep(seq: int, local: int = LOCAL, n_dil: int = NDIL,
                 strides: list[int] | None = None,
                 device="cpu") -> list[torch.Tensor]:
    """逐层递增 dilation 掩码：第 ℓ 层的 far 点按 stride `s_ℓ` 往远处取。

    `span_ℓ = min(P−1, s_ℓ·(n_dil−1))`，`picks_ℓ = {i−local−round(t·span_ℓ/(n_dil−1))
    : t = 0..n_dil−1}`；`i + 1 <= local + n_dil` 的行退化为全因果 ⇒ 每行仍是
    `min(k, i+1)` 个键、且 ⊆ 因果（判据④）。

    `s_ℓ = 1` 时 far 点 = 紧邻本地窗口外的 `n_dil−1` 个连续位置 ⇒ 该层掩码与
    `window_keep(seq, k)` **逐位相同**（判据④ 机检这一条）；`s_ℓ` 越大摊得越远
    ⇒ 逐层可达距离单调不减。
    """
    strides = STRIDES if strides is None else strides
    idx = torch.arange(seq, device=device)
    i, j = idx.unsqueeze(1), idx.unsqueeze(0)
    causal = j <= i
    inner = causal & ((i - j) < local)
    short = ((i + 1) <= (local + n_dil)).expand(seq, seq) & causal
    prefix = (idx - local + 1).clamp_min(1).float().view(seq, 1)      # P_i
    base = (idx - local).clamp_min(0).view(seq, 1)                    # i − local
    t = torch.arange(n_dil, device=device, dtype=torch.float).view(1, n_dil)
    out = []
    for step in strides:
        span = torch.minimum(prefix - 1,
                             torch.tensor(float(step * (n_dil - 1)), device=device))
        picks = (base - torch.round(t * span / (n_dil - 1)).long()).clamp_min(0)
        dil = torch.zeros(seq, seq, dtype=torch.bool, device=device)
        dil.scatter_(1, picks, True)
        out.append(((inner | dil) & causal) | short)
    return out


def masks_for(kind: str, seq: int, device="cpu"):
    """臂族的可见集：窗口臂走引擎的 `train_window`（返回 None），另两族给掩码。"""
    if kind == "window":
        return None
    if kind == "dilated":
        return dil_keep(seq, device=device)
    if kind == "layered":
        return layered_keep(seq, device=device)
    raise SystemExit(f"未知臂族：{kind!r}（只允许 window / dilated / layered）")


class DilatedBlock(nn.Module):
    """stock `Block` + 一张静态掩码（唯一差别 = 注意力可见集）。

    不重写注意力：把掩码按 SDPA 需要的形状展开后交给 `Block.forward(attn_mask=...)`
    的既有掩码路径（与 `DepthARModel._window_mask` 同款 expand），因此前向语义由共享
    实现保证、本类只负责「挑哪些键」。掩码 buffer 是 `persistent=False`，不进
    state_dict ⇒ 权重可从 stock `Block` 逐位搬过来（转换后由 `convert_to_dilated` 核对）。
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


def convert_to_dilated(model: DepthARModel, keeps) -> dict:
    """把 stock `Block` 换成 `DilatedBlock`（权重逐参数搬过去），返回核对读数。

    `keeps` 可以是**一张**掩码（所有层同一套，`dil32`）或**逐层**的列表（`lad32`）。
    构造 `DilatedBlock` 会消耗随机数，但权重随即被 state_dict 覆盖 ⇒ 主干参数与 stock
    构造逐位相同（判据⑤ 的证据：唯一变量 = 可见集）。
    """
    per_layer = list(keeps) if isinstance(keeps, (list, tuple)) else [keeps] * len(model.blocks)
    if len(per_layer) != len(model.blocks):
        raise SystemExit(f"逐层掩码数 {len(per_layer)} != 层数 {len(model.blocks)}")
    mcfg = ModelConfig(d_model=D, n_layers=L, n_heads=H, seq_len=model.cfg.seq_len,
                       mlp_mult=MLP, dropout=RECIPE["dropout"], norm_cap=RECIPE["norm_cap"])
    diffs, dropped = {}, []
    blocks = []
    for i, blk in enumerate(model.blocks):
        mh = None if getattr(blk, "mlp", None) is None else blk.mlp[0].out_features
        new = DilatedBlock(mcfg, per_layer[i], mlp_hidden=mh)
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
            "state_dict_key_symmetric_diff": dropped, "masked_blocks": len(blocks),
            "per_layer_mask": not isinstance(keeps, torch.Tensor)}


# --------------------------------------------------------------------- 训练


def model_cfg(seq: int, kind: str) -> DepthARConfig:
    return DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=seq, mlp_mult=MLP,
                         dropout=RECIPE["dropout"], norm_cap=RECIPE["norm_cap"],
                         dense=True, topology="sequential",
                         train_window=(K if kind == "window" else 0), train_anchors=0)


def train_arm(name: str, steps: int | None = None, out_root: str = "runs") -> None:
    spec = ARMS[name]
    if not spec["trained"]:
        raise SystemExit(f"{name} 是复用臂（不训练）")
    seq, seed, kind = int(spec["seq"]), int(spec["seed"]), spec["kind"]
    steps = int(steps) if steps else int(CFG["points"][str(seq)]["steps"])
    run_dir = os.path.join(HERE, out_root, name)
    os.makedirs(run_dir, exist_ok=True)
    train_arr, _, vocab = load_enwik8_full(".tmp/enwik8", DATA)
    torch.manual_seed(seed)                                    # 与 train_depth_ar.main 同序
    model = DepthARModel(model_cfg(seq, kind), vocab)
    keeps = masks_for(kind, seq)
    info = {"converted": None}
    if keeps is not None:
        info["converted"] = convert_to_dilated(model, keeps)
        assert info["converted"]["params_identical_to_stock"], "转换后参数不一致"
    n_params = sum(p.numel() for p in model.parameters())
    model = model.cuda()
    rows = mask_row_counts(kind, seq)
    print(f"{name}: {n_params/1e6:.2f}M params  seq {seq}  steps {steps}  "
          f"kept/row[{rows['min']},{rows['max']}] equals_min_k={rows['equals_min_k']}",
          flush=True)
    tcfg = {"steps": steps, "batch_size": int(CFG["points"][str(seq)]["batch_size"]),
            "lr": RECIPE["lr"], "seed": seed}
    torch.cuda.reset_peak_memory_stats()
    wall, _, _, stats = tdt.train(
        model, train_arr, tcfg,
        os.path.join(run_dir, "train_log.jsonl"),
        os.path.join(run_dir, "gate_log.jsonl"),
        exit_mode="uniform")
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    out = {
        "run": name, "arm": name, "kind": kind, "seq": seq, "seed": seed, "steps": steps,
        "tokens": steps * tcfg["batch_size"] * seq,
        "recipe": RECIPE, "mask": CFG["mask"] if keeps is not None else {"kind": "window", "k": K},
        "n_params": n_params,
        "wall_minutes": round(wall / 60, 2),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "conversion_check": info["converted"],
        "row_counts": rows,
        "train_stats": {k: v for k, v in stats.items()
                        if k in ("gnorm_max", "gnorm_mean", "skipped_steps", "grad_skips")},
    }
    write_results(os.path.join(run_dir, "results.json"), out, source=__file__)
    print(f"{name}: wall {out['wall_minutes']} min  peak {out['peak_gpu_mem_mb']} MiB  "
          f"gnorm_max {out['train_stats'].get('gnorm_max')}", flush=True)


def mask_row_counts(kind: str, seq: int) -> dict:
    """逐行可见键数（判据④ 的臂级落盘列）。逐层臂给逐层列表。"""
    keep = masks_for(kind, seq)
    if keep is None:
        keep = window_keep(seq)
    ms = keep if isinstance(keep, list) else [keep]
    counts = [m.sum(-1).tolist() for m in ms]
    expect = [min(K, i + 1) for i in range(seq)]
    return {"per_row": counts if len(ms) > 1 else counts[0],
            "min": min(min(c) for c in counts), "max": max(max(c) for c in counts),
            "equals_min_k": all(c == expect for c in counts)}


# --------------------------------------------------------------------- 评测


def load_arm(name: str, vocab: int) -> DepthARModel:
    spec = ARMS[name]
    seq, kind = int(spec["seq"]), spec["kind"]
    torch.manual_seed(0)
    model = DepthARModel(model_cfg(seq, kind), vocab)
    keeps = masks_for(kind, seq)
    if keeps is not None:
        info = convert_to_dilated(model, keeps)
        assert info["params_identical_to_stock"] and info["masked_blocks"] == L
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(ROOT, spec["ckpt"]), map_location=DEVICE))
    model.eval()
    return model


def eval_keep(name: str):
    """评测时的可见集：窗口臂给窗口掩码、空洞臂给自己的掩码、逐层臂交给块自己的 buffer。"""
    spec = ARMS[name]
    seq, kind = int(spec["seq"]), spec["kind"]
    if kind == "window":
        return window_keep(seq, device=DEVICE)
    if kind == "layered":
        return None                      # 每层掩码不同 ⇒ 走 DilatedBlock 自己的 buffer
    return dil_keep(seq, device=DEVICE)


def per_batch_bpc(model: DepthARModel, keep, eval_arr, seq: int) -> list[float]:
    gen = torch.Generator().manual_seed(DSEED)
    out = []
    with torch.no_grad():
        for _ in range(NB):
            x, _ = batch(eval_arr, BSIZE, seq, gen)
            x = x.to(DEVICE)
            logits = masked_logits(model, x, keep)[:, :-1].float()
            tgt = x[:, 1:]
            nats = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                   tgt.reshape(-1), reduction="sum").item()
            out.append(round(nats / tgt.numel() / LN2, 5))
    return out


def paired_stats(comparator: list[float], arm: list[float]) -> dict:
    """Δ = 对照 − 实验（正 = 实验更好）。`two_se` = 1.96·SE = 本读数在本 T 的解析力。"""
    d = [c - a for c, a in zip(comparator, arm)]
    n = len(d)
    mean, sd = statistics.mean(d), statistics.stdev(d)
    se = sd / n ** 0.5
    return {"n_batches": n, "delta_bpc": round(mean, 5),
            "sd_of_paired_delta": round(sd, 5), "se_of_mean": round(se, 5),
            "two_se": round(1.96 * se, 5),
            "t_stat": round(mean / se, 2) if se > 0 else None,
            "frac_batches_delta_gt_0": round(sum(1 for v in d if v > 0) / n, 3),
            "min_delta": round(min(d), 5), "max_delta": round(max(d), 5),
            "n_req_for_band": int(math.ceil((1.96 * sd / BAND) ** 2)) if sd > 0 else None}


def band_class(delta: float) -> str:
    if delta >= BAND:
        return "beyond_positive"
    if delta <= -BAND:
        return "beyond_negative"
    return "in_band"


def mask_structure(seq: int) -> dict:
    """判据④：三族掩码逐行计数、因果子集、`lad32` 第 0 层 == `win32`、逐层可达距离。"""
    win, dil, lad = window_keep(seq), dil_keep(seq), layered_keep(seq)
    causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
    expect = [min(K, i + 1) for i in range(seq)]
    fams = {"win32": [win], "dil32": [dil], "lad32": lad}
    counts = {f: [m.sum(-1).tolist() for m in ms] for f, ms in fams.items()}
    idx = torch.arange(seq)
    dist = idx.unsqueeze(1) - idx.unsqueeze(0)
    reach = {f: [int(((m & (dist > 0)).mul(dist)).max()) for m in ms]
             for f, ms in fams.items()}
    mismatch = sum(1 for f, cc in counts.items() for c in cc
                   if c != counts["win32"][0])
    return {
        "seq": seq,
        "per_row_count_equal_across_families": mismatch == 0,
        "mismatch_rows": mismatch,
        "per_row_count_equals_min_k": all(c == expect for cc in counts.values() for c in cc),
        "mask_offending_future_pairs": int(sum(int((m & ~causal).sum())
                                               for ms in fams.values() for m in ms)),
        "lad_layer0_equals_win32": bool(torch.equal(lad[0], win)),
        "reach_max_distance": reach,
        "lad_reach_monotone": all(b >= a for a, b in zip(reach["lad32"], reach["lad32"][1:])),
        "note": "逐行可见键数三族相等且都 = min(32, i+1)；本字段即判据④ 的落盘列",
    }


def _decl(name: str) -> dict:
    """逐臂训练申报（墙钟 / 峰值显存，不进判据）。"""
    spec = ARMS[name]
    p = os.path.join(ROOT, os.path.dirname(spec["ckpt"]), "results.json")
    if os.path.exists(p):
        r = json.load(open(p))
        return {"wall_minutes": r.get("wall_minutes"),
                "peak_gpu_mem_mb": r.get("peak_gpu_mem_mb"),
                "gnorm_max": (r.get("train_stats") or {}).get("gnorm_max"),
                "source": os.path.relpath(p, ROOT)}
    return {"wall_minutes": None, "peak_gpu_mem_mb": None, "gnorm_max": None,
            "source": os.path.relpath(p, ROOT), "note": "该臂未跑（掉臂规则），申报项为空"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "eval"), required=True)
    ap.add_argument("--arm", default="", help="训练阶段的臂名（见 config.json::arms）")
    ap.add_argument("--steps", type=int, default=0, help="覆盖该 T 点的步数（只用于冒烟）")
    ap.add_argument("--out-root", default="runs",
                    help="训练产物根（相对本目录；给绝对路径可把冒烟写进 /tmp）")
    args = ap.parse_args()
    os.chdir(ROOT)

    if require_prereg(Path(HERE)) != 0:
        raise SystemExit("预登记未绑定：先跑 python3 tools/prereg_bind.py "
                         "axes/1-time/1.2.1-dilated/experiments/2689-longT-layerdil-13m "
                         "--contract README.md config.json run.py train_arms.sh")

    if args.stage == "train":
        train_arm(args.arm, args.steps or None, args.out_root)
        return

    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", DATA)
    structure = {str(seq): mask_structure(seq) for seq in (256, 1024, 2048)}
    for s, st in structure.items():
        print(f"[structure T={s}] counts_equal={st['per_row_count_equal_across_families']} "
              f"min_k={st['per_row_count_equals_min_k']} future={st['mask_offending_future_pairs']} "
              f"lad0==win={st['lad_layer0_equals_win32']} monotone={st['lad_reach_monotone']}",
              flush=True)
        if not (st["per_row_count_equal_across_families"] and st["per_row_count_equals_min_k"]
                and st["mask_offending_future_pairs"] == 0
                and st["lad_layer0_equals_win32"] and st["lad_reach_monotone"]):
            raise SystemExit(f"结构核查失败（T={s}）：判决作废（README §6）")

    per_batch, bpc_mean, ablate, params, not_run = {}, {}, {}, {}, []
    for name, spec in ARMS.items():
        seq, kind = int(spec["seq"]), spec["kind"]
        if not os.path.exists(os.path.join(ROOT, spec["ckpt"])):
            not_run.append(name)
            print(f"[eval] {name}: 缺 checkpoint ⇒ 登记为未跑（掉臂规则）", flush=True)
            continue
        model = load_arm(name, vocab)
        keep = eval_keep(name)
        params[name] = sum(p.numel() for p in model.parameters())
        per_batch[name] = per_batch_bpc(model, keep, eval_arr, seq)
        bpc_mean[name] = round(statistics.mean(per_batch[name]), 5)
        if kind in ("dilated", "layered"):
            local_only = window_keep(seq, LOCAL, device=DEVICE)
            ablate[name] = paired_stats(per_batch_bpc(model, local_only, eval_arr, seq),
                                        per_batch[name])
        del model
        torch.cuda.empty_cache()
        print(f"[eval] {name}: bpc {bpc_mean[name]} params {params[name]}", flush=True)

    # 判据⑥ 等值门：T=256 的逐批 bpc 必须逐位复现 2678 的落盘值
    ref = json.load(open(os.path.join(EXP_2678, "results.json")))["per_batch_bpc"]
    reuse_ref = {}
    for name in ("win32", "win32_s1", "dil32", "dil32_s1"):
        same = name in per_batch and per_batch[name] == [round(v, 5) for v in ref[name]]
        reuse_ref[name] = {"reproduced": same,
                           "ref": "2678-dilated-vs-window-13m/results.json::per_batch_bpc"}
        if not same:
            raise SystemExit(f"等值门失败：{name} 的逐批 bpc 未复现 2678 落盘值（README §6）")

    # 判据① / ②：逐对的配对 Δ
    pairs = {}
    for pid, p in CFG["pairs"].items():
        cmp_name, arm_name = p["comparator"], p["arm"]
        if cmp_name not in per_batch or arm_name not in per_batch:
            pairs[pid] = {"not_run": True,
                          "missing": [n for n in (cmp_name, arm_name) if n not in per_batch]}
            continue
        st = paired_stats(per_batch[cmp_name], per_batch[arm_name])
        pairs[pid] = {"seq": int(p["seq"]), "comparator": cmp_name, "arm": arm_name,
                      **st, "band_class": band_class(st["delta_bpc"]),
                      "params_equal": params.get(cmp_name) == params.get(arm_name)}

    # 判据③：去掉 far 点后的边际价值 D(T)（描述量）
    dependence = {}
    for name, ab in ablate.items():
        seq = str(ARMS[name]["seq"])
        dependence.setdefault(seq, {})[name] = {
            "delta_bpc": ab["delta_bpc"], "two_se": ab["two_se"],
            "band_class": band_class(ab["delta_bpc"]),
            "note": "D = bpc(只留近端 16) − bpc(原掩码)，正 = far 点对该 checkpoint 有价值"}
    d_growth = None
    if "256" in dependence and "2048" in dependence \
            and "dil32-t2048-s0" in dependence["2048"] and "dil32" in dependence["256"]:
        d256 = dependence["256"]["dil32"]["delta_bpc"]
        d2048 = dependence["2048"]["dil32-t2048-s0"]["delta_bpc"]
        d_growth = {"D_t256_seed0": d256, "D_t2048_seed0": d2048,
                    "delta_bpc": round(d2048 - d256, 5),
                    "band_class": band_class(round(d2048 - d256, 5)),
                    "note": "D(2048) − D(256)：跨 run 描述量，带只作参照线（不是配对判据）"}

    # ---- 判词（config.json::verdict_rules 的机械应用）----
    def per_T(prefix: str) -> dict:
        got = {k: v for k, v in pairs.items() if k.startswith(prefix) and "not_run" not in v}
        if not got:
            return {"verdict": "未跑（掉臂规则）", "n_seed": 0, "deltas": {}}
        deltas = {k: v["delta_bpc"] for k, v in got.items()}
        cls = {k: v["band_class"] for k, v in got.items()}
        n = len(got)
        if n >= 2:
            same_sign = len({v > 0 for v in deltas.values()}) == 1
            beyond = all(abs(v) >= BAND for v in deltas.values())
            singles = all(v["params_equal"] for v in got.values())
            if not singles:
                v = "参数量不等 ⇒ 判据⑤ 失败，判决作废"
            elif not same_sign:
                v = "两 seed 不同向 ⇒ 本 n 下不可排序（不解释成噪声、不取平均掩盖）"
            elif beyond:
                v = ("空洞采样占优（两 seed 同向且各自越带 ⇒ n=2）" if list(deltas.values())[0] > 0
                     else "空洞采样更差（两 seed 同向且各自越带 ⇒ n=2）：按位置最近仍占优")
            else:
                v = "两 seed 同向但至少一 seed 未越带 ⇒ 本 n 下不可排序"
        else:
            d = list(deltas.values())[0]
            v = (f"单 seed（{list(deltas)[0]}）：Δ = {d:+.5f} ⇒ {cls[list(cls)[0]]}"
                 "（只见一次，不作 n=2 结论）")
        return {"verdict": v, "n_seed": n, "deltas": deltas, "band_classes": cls,
                "max_two_se": max(v["two_se"] for v in got.values()),
                "max_n_req_for_band": max(v["n_req_for_band"] for v in got.values())}

    by_T = {"256": per_T("c1_t256"), "1024": per_T("c1_t1024"), "2048": per_T("c1_t2048")}

    # 「分辨不出（仪器侧）」出口：Δ 落带内但 1.96·SE > 带
    instrument_limited = {T: (all(c == "in_band" for c in by_T[T].get("band_classes", {}).values())
                              and by_T[T].get("max_two_se", 0) > BAND)
                          for T in by_T if by_T[T].get("deltas")}

    main_2048 = by_T["2048"]["verdict"]
    no_gain_2048 = (by_T["2048"].get("band_classes")
                    and all(c == "in_band" for c in by_T["2048"]["band_classes"].values())
                    and by_T["2048"]["n_seed"] >= 2)
    if no_gain_2048:
        if instrument_limited["2048"]:
            quality_answer = ("长 T 上判『分辨不出（仪器侧）』：两 seed 的 Δ 都落 0.014 带内，"
                              f"但本 T 的解析力 1.96·SE = {by_T['2048']['max_two_se']} > 0.014"
                              f" ⇒ 要判到带内需 n ≈ {by_T['2048']['max_n_req_for_band']} 批/臂")
        elif d_growth is not None and d_growth["delta_bpc"] >= BAND:
            quality_answer = ("长 T 上判『本 k 分辨不出』：Δ 在带内，但远处键的边际价值随 T 上升"
                              f"（D(2048) − D(256) = {d_growth['delta_bpc']:+.5f} ≥ 0.014）"
                              " ⇒ 32 键里只有 16 个远处点、覆盖率 12.5% → 1.5%，固定 k 抓不住")
        else:
            quality_answer = ("长 T 上判『真实文本不需要远处信息』：Δ 两 seed 都落带内，"
                              "且远处键的边际价值未随 T 上升"
                              + (f"（D(2048) − D(256) = {d_growth['delta_bpc']:+.5f} < 0.014）"
                                 if d_growth is not None else "（判据③ 未跑）"))
    elif by_T["2048"].get("deltas"):
        deltas = by_T["2048"]["deltas"]
        all_negative = all(c == "beyond_negative"
                           for c in by_T["2048"]["band_classes"].values())
        onset = next((T for T in ("256", "1024", "2048")
                      if by_T[T].get("band_classes")
                      and all(c == "beyond_positive" for c in by_T[T]["band_classes"].values())), None)
        quality_answer = ((f"长 T 上空洞臂越负带（按位置最近在长 T 上占优）：T=2048 的 Δ = "
                           if all_negative else f"长 T 上出现质量增量：T=2048 的 Δ = ")
                          + f"{', '.join(f'{k} {v:+.5f}' for k, v in deltas.items())}"
                          + ("；没有任何 T 点越正带（起点 T = 无）；跨 T 的走向只按观测点描述、不外推"
                             if all_negative else
                             f"；起点 T = {onset}（第一个两 seed 都越带的 T 点）；"
                             "增长率只按观测点描述、不外推"))
    else:
        quality_answer = "未跑（掉臂规则）：长 T 腿的 checkpoint 缺失，本项不作质量判词"

    conversion_checks = {}
    for name, spec in ARMS.items():
        p = os.path.join(ROOT, os.path.dirname(spec["ckpt"]), "results.json")
        conversion_checks[name] = (json.load(open(p)).get("conversion_check")
                                   if os.path.exists(p) else None)
    arms_not_ok = [n for n in ARMS if (_decl(n).get("gnorm_max") or 0) >= 100]

    out = {
        "experiment": CFG["experiment"],
        "cell": CFG["cell"],
        "question": CFG["question"],
        "band_bpc": BAND,
        "delta_directions": {
            "delta_bpc": "comparator_minus_arm",
            "deltas": "comparator_minus_arm",
        },
        "delta_direction_note": ("全部 Δ 都是「对照 − 实验」：判据① 对照 = 窗口臂、实验 = 空洞臂；"
                                 "判据② 对照 = 平铺空洞臂、实验 = 逐层递增臂；判据③ 对照 = 去掉 far 点"
                                 "（只留近端 16）的重评测、实验 = 原掩码 ⇒ 正 = 后者更好"),
        "recipe": RECIPE,
        "mask": CFG["mask"],
        "structure_checks": structure,
        "per_batch_bpc": per_batch,
        "mean_bpc": bpc_mean,
        "n_params": params,
        "arms_not_run": not_run,
        "criteria": {
            "c1_quality_by_T": {
                "note": "Δ = bpc_对照 − bpc_实验（对照 = 窗口臂、实验 = 空洞臂），正 = 空洞更好；"
                        "0.014 配对带",
                "by_T": by_T,
                "instrument_limited": instrument_limited,
                "answer": quality_answer,
            },
            "c2_layered_arm": {
                "note": "Δ = bpc_平铺 − bpc_逐层递增（对照 = 平铺空洞臂、实验 = 逐层臂），正 = 逐层更好",
                "pair": pairs.get("c3_ladder_s0"),
                "verdict": ("未跑（掉臂规则）" if pairs.get("c3_ladder_s0", {}).get("not_run")
                            else pairs["c3_ladder_s0"]["band_class"]),
            },
            "c3_far_key_dependence": {
                "note": "D(T) = bpc(只留近端 16) − bpc(原掩码)（对照 = 去掉 far 点、实验 = 原掩码），"
                        "正 = far 点有价值；跨 T 的差是描述量、带只作参照线",
                "by_T": dependence,
                "growth": d_growth,
            },
            "c4_must_not_change": {
                "structure": {k: {kk: vv for kk, vv in v.items() if kk != "reach_max_distance"}
                              for k, v in structure.items()},
                "params_equal_within_T": all(v.get("params_equal", False)
                                             for k, v in pairs.items()
                                             if k.startswith("c1_") and "not_run" not in v),
                "conversion_checks": conversion_checks,
            },
            "c5_reuse_equivalence": reuse_ref,
            "c6_stability_gate": {"threshold": 100, "arms_over_threshold": arms_not_ok,
                                 "note": "gnorm_max < 100 且 skips == 0 才入场（判据⑦）"},
        },
        "main_verdict": {
            "criterion": ("① Δ = 对照 − 实验（越 ±0.014 带才算可分）；② 逐层递增相对平铺；"
                          "③ 远处键的边际价值随 T 的走向"),
            "t2048_quality": main_2048,
            "answer": quality_answer,
            "n_seed_by_T": {T: by_T[T].get("n_seed", 0) for T in by_T},
            "falsified_prior": bool(no_gain_2048 and d_growth is not None
                                    and d_growth["delta_bpc"] < BAND),
        },
        "train_declaration": {a: _decl(a) for a in ARMS},
        "wall_reference": {"wall_in_criterion": False,
                           "note": "训练墙钟与峰值显存逐臂申报，不进判据（harness/methodology.md §3）"},
    }
    write_results(os.path.join(HERE, "results.json"), out, source=__file__)
    print(json.dumps({"mean_bpc": bpc_mean, "verdict": out["main_verdict"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
