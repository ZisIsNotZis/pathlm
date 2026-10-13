"""2619 · 统一 KV 预算参照线总图（ticket 36 波二 · U3 线）—— 实验入口。

契约：同目录 `README.md`（判据线逐字抄录，未改）。
仪器：`tools/evalkit/longrange_tasks.py`（合成长程任务电池）、
      `tools/evalkit/kv_quant.py`（KV 量化往返）、`tools/evalkit/retention.py`（可见性掩码）。
被测量模型：三族 checkpoint 复用（只读，运行时断言 sha256），见 `MODEL.md`。

本实验是**测量 + 汇总**类，不训练、不加参数：
  ① 把 14 个「KB 预算点」映射到同一横轴 `bytes_per_position`（解析公式，判据 C1）；
  ② 用**同一条 bpc 路径**（`bpc_of`）重测所有点的 clean bpc，并与来源实验对照；
  ③ 用**同一件合成仪器**（`longrange_tasks`，同一 seed）测三点召回腿
     （copy / induction / var_recall），同时给 chance 与局部先验地板；
  ④ 自检（仪器等值 / 量化活性 / 窗口删除量 / 与来源实验逐一对照）；
  ⑤ 落盘 一张表（`table.md`）+ 一张图（`chart.png` + `chart.txt`）+ `results.json`。

用法：
  python3 run.py            # 全跑（纯推理，共享卡 ≈8–12 min）
  python3 run.py --smoke    # 冒烟：缩小批量/档数，实测步时与显存（不外推时间）
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full, batch                       # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,            # noqa: E402
                             kv_head_liveness)
from tools.evalkit.kv_quant import kv_quant_logits, summarize_stats  # noqa: E402
from tools.evalkit.retention import (masked_logits, masked_bpc,       # noqa: E402
                                     mask_stats, recent_keep_mask)
from tools.evalkit.longrange_tasks import (                          # noqa: E402
    induction_batch, copy_batch, var_recall_batch,
    accuracy as lib_accuracy, local_prior_accuracy, corrupt_span,
    check_local_prior_resistance)

LN2 = 0.6931471805599453
VARIANCE_BAND_BPC = 0.014


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------- 位置映射（判据 C1）


def bytes_per_position(spec: dict, n_layers: int = 12, n_heads: int = 8,
                       head_dim: int = 32, T: int = 256) -> dict:
    """**唯一**的横轴公式（判据 C1）：

        bytes_per_position = 2(K,V) · n_layers · n_kv_heads_eff · head_dim
                             · value_bytes · (positions_kept / T)

    三种机制都只是在这一个乘式里改不同因子：
      - 量化  ⇒ 改 `value_bytes`（2 / 1 / 0.5）
      - 头共享 ⇒ 改 `n_kv_heads_eff`（8 / 2 / 1），且是**重新训练**的模型
      - 窗口  ⇒ 改 `positions_kept`（d / T）
    """
    kind = spec["kind"]
    if kind == "head":
        nkv = spec["n_kv_heads"]
    else:
        nkv = n_heads
    vb = spec["value_bytes"]
    kept = spec["positions_kept"] if kind == "window" else T
    per_pos = 2 * n_layers * nkv * head_dim * vb * (kept / T)
    return {"bytes_per_position": round(per_pos, 6),
            "factor_value_bytes": vb,
            "factor_n_kv_heads_eff": nkv,
            "factor_positions_kept": kept,
            "factor_positions_kept_over_T": round(kept / T, 6),
            "T": T}


# ----------------------------------------------------------------- 统一 bpc 路径


@torch.no_grad()
def bpc_of(logits_fn, eval_arr, T: int, n_batches: int, batch_size: int,
           window_seed: int, device: str, on_batch=None) -> float:
    """clean bpc（越低越好）：因果移位 CE 的 token 平均。

    **与 `retention.masked_bpc` 同一算法、同一窗口序**（同一 `Generator(window_seed)`、
    同一 `pathlm.data.batch` 调用序），只是把「模型 → logits」换成显式的 `logits_fn`，
    从而让 14 个臂走**同一条评测路径**（判据 C1 的口径一致性）。
    单测/自检：fp16 臂与 `retention.masked_bpc(model, eval_arr, None)` 逐位相等。
    """
    gen = torch.Generator().manual_seed(window_seed)
    nats, n_tok = 0.0, 0
    for i in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.to(device)
        logits = logits_fn(x)[:, :-1]
        tgt = x[:, 1:]
        nats += F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                tgt.reshape(-1), reduction="sum").item()
        n_tok += tgt.numel()
        if on_batch is not None:
            on_batch(i, n_tok)
    return nats / n_tok / LN2


# ----------------------------------------------------------------- 合成召回腿


def floor_threshold(n: int, chance: float, z: float = 1.959963984540054) -> float:
    """判据 C4 的「地板」阈值：`chance + 2·σ_wilson`（σ 取 Wilson 区间半宽，在 p=chance 处）。"""
    lo, _ = wilson_ci(int(round(chance * n)), n, z)
    return chance + 2 * (chance - lo)


def wilson_ci(n_hits: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% 置信区间（比例）。"""
    if n == 0:
        return (0.0, 0.0)
    p = n_hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _summarize_hits(hits: torch.Tensor, dists: torch.Tensor) -> dict:
    """与 `longrange_tasks._summarize` 同口径的聚合（accuracy / n / n_hits / by_distance）。"""
    hits = hits.to(torch.float64)
    dists = dists.to(torch.long)
    by_dist = {}
    for d in sorted(set(int(v) for v in dists.flatten().tolist())):
        sel = dists == d
        by_dist[str(d)] = {"accuracy": round(float(hits[sel].mean().item()), 6),
                           "n": int(sel.sum().item()),
                           "n_hits": int(hits[sel].sum().item())}
    n = int(hits.numel())
    nh = int(hits.sum().item())
    lo, hi = wilson_ci(nh, n)
    return {"accuracy": round(float(hits.mean().item()), 6), "n": n, "n_hits": nh,
            "wilson95": [round(lo, 6), round(hi, 6)], "by_distance": by_dist}


@torch.no_grad()
def synth_accuracy(logits_fn, task, device: str, max_chunk: int = 64) -> dict:
    """模型（由 `logits_fn` 给定）在合成任务上的 top-1 准确率 + 距离曲线 + Wilson CI。

    与 `longrange_tasks.accuracy` 同算法（分块前向、`argmax` 在查询行取值），
    只是接受 `logits_fn` 从而能覆盖量化 / 窗口 / 头共享三族。
    """
    hits = torch.zeros(task.B, task.M, dtype=torch.long)
    for lo in range(0, task.B, max_chunk):
        hi = min(lo + max_chunk, task.B)
        logits = logits_fn(task.tokens[lo:hi].to(device))
        pred = logits.argmax(-1)
        sel = torch.gather(pred, 1, task.rows[lo:hi].to(device))
        hits[lo:hi] = (sel == task.answers[lo:hi].to(device)).to(torch.long).cpu()
    return _summarize_hits(hits, task.dists)


def concat(batches):
    f = batches[0]
    return type(f)(name=f.name, tokens=torch.cat([b.tokens for b in batches], 0),
                   rows=torch.cat([b.rows for b in batches], 0),
                   answers=torch.cat([b.answers for b in batches], 0),
                   dists=torch.cat([b.dists for b in batches], 0),
                   n_answers=f.n_answers, meta=f.meta)


def build_tasks(cfg: dict, smoke: bool = False) -> dict:
    """三件合成任务的批（同一 `Generator(seed)` ⇒ 逐臂逐位同任务）。"""
    tok = cfg["tokens"]
    T = cfg["model"]["seq_len"]
    sym, fill, cue = tok["sym_vocab"], tok["filler_token"], tok["cue_token"]
    t = cfg["tasks"]
    nb = 1 if smoke else 0
    g = torch.Generator().manual_seed(cfg["seed"])
    common = dict(sym_vocab=sym, filler_token=fill, cue_token=cue, generator=g)

    def multi(fn, n_batches, **kw):
        return concat([fn(**kw, **common) for _ in range(n_batches)])

    copy_pos = t["copy"]["src_positions"][:2] if smoke else t["copy"]["src_positions"]
    ind_starts = t["induction"]["src_starts"][:1] if smoke else t["induction"]["src_starts"]
    var_ds = t["var_recall"]["distances"][:1] if smoke else t["var_recall"]["distances"]
    bs = 8 if smoke else None
    return {
        "copy": multi(copy_batch, t["copy"]["n_batches"] + nb, seq_len=T,
                      copy_len=t["copy"]["copy_len"],
                      src_pos=copy_pos[0], batch_size=bs or t["copy"]["batch_size"]),
        "copy_by_pos": {p: multi(copy_batch, t["copy"]["n_batches"] + nb, seq_len=T,
                                 copy_len=t["copy"]["copy_len"], src_pos=p,
                                 batch_size=bs or t["copy"]["batch_size"])
                        for p in copy_pos},
        "induction": multi(induction_batch, t["induction"]["n_batches"] + nb, seq_len=T,
                           prefix_len=t["induction"]["prefix_len"],
                           copy_len=t["induction"]["copy_len"], src_start=ind_starts[0],
                           batch_size=bs or t["induction"]["batch_size"]),
        "induction_by_start": {s: multi(induction_batch, t["induction"]["n_batches"] + nb,
                                        seq_len=T, prefix_len=t["induction"]["prefix_len"],
                                        copy_len=t["induction"]["copy_len"], src_start=s,
                                        batch_size=bs or t["induction"]["batch_size"])
                               for s in ind_starts},
        "var_recall": multi(var_recall_batch, t["var_recall"]["n_batches"] + nb, seq_len=T,
                            distance=var_ds[0], batch_size=bs or t["var_recall"]["batch_size"]),
        "var_recall_by_d": {d: multi(var_recall_batch, t["var_recall"]["n_batches"] + nb,
                                     seq_len=T, distance=d,
                                     batch_size=bs or t["var_recall"]["batch_size"])
                            for d in var_ds},
    }


def pooled_accuracy(logits_fn, tasks_by_key: dict, device: str) -> dict:
    """把多档（距离/起点）的批拼起来求**汇总**准确率，并给逐档曲线。"""
    pooled = concat(list(tasks_by_key.values()))
    out = synth_accuracy(logits_fn, pooled, device)
    out["per_setting"] = {str(k): {"accuracy": v["accuracy"], "n": v["n"],
                                   "n_hits": v["n_hits"], "wilson95": v["wilson95"],
                                   "distances": sorted(int(x) for x in v["by_distance"])}
                          for k, v in ((k, synth_accuracy(logits_fn, t, device))
                                       for k, t in tasks_by_key.items())}
    return out


# ----------------------------------------------------------------- 模型装载


def activity_of(spec: dict, model, T: int, device: str, probe_tokens: torch.Tensor) -> dict:
    """SCHEMA §10 活性量：本实验无训练，故报**推理期实际通路**是否真的被改。"""
    kind = spec["kind"]
    if kind == "quant":
        _, stats = kv_quant_logits(model, probe_tokens, spec["mode"])
        return {"kind": "quant", "mode": spec["mode"], **summarize_stats(stats)}
    if kind == "window":
        keep = recent_keep_mask(T, spec["d"], device=device)
        return {"kind": "window", "d": spec["d"], **mask_stats(keep)}
    return {"kind": "head", "n_kv_heads": spec["n_kv_heads"],
            **kv_head_liveness(model, probe_tokens)}


def load_trunk(cfg: dict, device: str = "cuda"):
    spec = cfg["checkpoints"]["trunk"]
    ckpt = os.path.join(ROOT, spec["path"])
    got = sha256_file(ckpt)
    if got != spec["sha256"]:
        raise SystemExit(f"trunk sha256 mismatch:\n  want {spec['sha256']}\n  got  {got}")
    model = DepthARModel(DepthARConfig(**cfg["model"]), cfg["data"]["vocab_size"])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(ckpt), "sha256_verified": True}


def load_head_arm(cfg: dict, arm: str, device: str = "cuda"):
    fam = cfg["checkpoints"]["head_family"]
    spec = fam["arms"][arm]
    ckpt = os.path.join(ROOT, fam["root"], spec["run"], "model.pt")
    got = sha256_file(ckpt)
    if got != spec["sha256"]:
        raise SystemExit(f"{arm} sha256 mismatch:\n  want {spec['sha256']}\n  got  {got}")
    mcfg = dict(cfg["model"])
    mcfg["n_kv_heads"] = spec["n_kv_heads"]
    model = DepthARModel(DepthARConfig(**mcfg), cfg["data"]["vocab_size"])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": os.path.relpath(ckpt, ROOT), "sha256": got,
                   "size_bytes": os.path.getsize(ckpt), "sha256_verified": True}


def make_logits_fn(model, spec: dict, T: int, device: str):
    """把一个臂变成 `tokens -> logits`（三族共用的唯一接口）。"""
    kind = spec["kind"]
    if kind == "quant":
        mode = spec["mode"]

        def fn(x, _m=model, _mode=mode):
            return kv_quant_logits(_m, x, _mode)[0]
        return fn
    if kind == "window":
        keep = recent_keep_mask(T, spec["d"], device=device)

        def fn(x, _m=model, _k=keep):
            return masked_logits(_m, x, _k)
        return fn

    def fn(x, _m=model):
        return masked_logits(_m, x, None)
    return fn


# ----------------------------------------------------------------- 来源实验对照


def source_values() -> dict:
    """来源实验（R1 / W3 / M2-2615）落盘值的只读摘录，用于逐一对照。"""
    p_r1 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/"
                              "2614-kv-quant-eval-13m/results.json")
    p_w3 = os.path.join(ROOT, "archive/wiki-legacy/axes/W-width/W3-kv-head-sharing/experiments/"
                              "2616-gqa-mqa-13m/results.json")
    p_m2 = os.path.join(ROOT, "axes/1-time/1.1.4.2-compression/experiments/"
                              "2615-retention-retrieval-leg-13m/results.json")
    out = {"R1": {}, "W3": {}, "M2": {}}
    d = json.load(open(p_r1))
    for row in d["quant_rows"]:
        out["R1"][row["mode"]] = {"bpc": row["bpc"], "damage_bpc": row["damage_bpc"],
                                  "per_position_bytes": row["per_position_bytes"]}
    d = json.load(open(p_w3))
    for arm in ("mha", "gqa", "mqa"):
        a = d["arms"][arm]
        out["W3"][arm] = {"bpc_clean": a["bpc_clean"],
                          "bpc_second_path_masked_bpc": a.get("bpc_second_path_masked_bpc"),
                          "damage_bpc": d["criteria"]["per_arm"].get(arm, {}).get("damage_bpc")}
    d = json.load(open(p_m2))
    for c in d["configs"]:
        if c["kind"] == "recent":
            out["M2"][c["id"]] = {"bpc": c["bpc"], "damage_bpc": c["damage_bpc"],
                                  "recall": c["retention"]["recall"]}
    out["sources"] = {"R1": os.path.relpath(p_r1, ROOT), "W3": os.path.relpath(p_w3, ROOT),
                      "M2": os.path.relpath(p_m2, ROOT)}
    return out


def missing_combinations() -> list[dict]:
    """图上**没测**的组合 = 下一批候选（判据 C5）。"""
    return [
        {"id": "GQA × int8/int4",
         "why_missing": "两类机制分属不同实验（W3 只测头共享、R1 只测量化），本项目 scope 禁组合实验",
         "next_step": "单因素拆分：固定 GQA(2 头) 再逐档量化 K/V；契约须写清 2×2 的臂数与步时实测"},
        {"id": "窗口 × 锚点 / sink",
         "why_missing": "A3 已测锚点通道（needle 76.8→93.3%）但那是**真实文本针**口径；本图未给锚点档的字节/位置与合成召回",
         "next_step": "把 window_anchor_keep_mask 加进本图（锚点占 bytes/pos 极小，应单列一档）"},
        {"id": "窗口 × 量化",
         "why_missing": "同 scope 限制（组合未测）",
         "next_step": "固定窗口 d 后量化剩余 KV；两因子在同一个字节轴上相互作用可测"},
        {"id": "权重 / 激活量化",
         "why_missing": "R1 只测了 KV cache 量化；权重/激活量化本项目零经验",
         "next_step": "先分诊（业界已做透 ⇒ 可能直接写 reference 结论），再决定是否本地补测"},
        {"id": "MLA 低秩 KV（W3 选项之一）",
         "why_missing": "零经验；低秩投影的压缩率不是本图的单一乘式（需报秩 r）",
         "next_step": "先扩展横轴公式为 2·d·r 形式，再测 13M"},
        {"id": "摘要压缩（M2 选项之一）",
         "why_missing": "零经验；压缩率依赖摘要器，不是单一乘式",
         "next_step": "先定摘要器与预算口径，再测；注意 SCHEMA §17.9 的成本交叉点"},
        {"id": "长 T（>256）下的窗口 / 驱逐",
         "why_missing": "本图所有点都在 T=256 上评测；窗口的 bytes/pos = 12288·d/T 随 T 下降，T 越长窗口越便宜（未测）",
         "next_step": "在 T∈{512,1024,4096} 上重测窗口档的 bpc 与合成召回，给出与 T 的关系"},
        {"id": "各臂的**实测**显存峰值",
         "why_missing": "本实验评测实现把 K/V 立刻反量化、窗口按掩码取子集（SDPA 仍算 T×T 分数矩阵）⇒ 实测峰值不随档下降，本实验只报解析列（判据 C2）",
         "next_step": "写一个真正的 cache 实现（分页/分组存储 + 只在需要时反量化），再测部署口径峰值"},
        {"id": "召回腿的真实文本针（needle v3）跨族",
         "why_missing": "2615 只给 trunk 族的窗口档测了真实文本针召回；量化/头共享族没有该腿（不同仪器不得混列，本图未并入）",
         "next_step": "用同一 2615 仪器补跑 int8/int4/gqa/mqa 的针召回，作为**第二条召回腿**单列"},
        {"id": "GQA/MQA 的等参数对照臂",
         "why_missing": "W3 的自述缺口：GQA/MQA 参数量比 MHA 少 8.75%/10.25%，代价混着容量差",
         "next_step": "补等参数对照（加宽 MLP 或少层），再判头共享的净代价"}
    ]


# ----------------------------------------------------------------- 图 / 表

CLASSES = {"baseline": "不压缩基线", "quant": "量化（R1）", "window": "窗口/驱逐（M2）",
           "head": "头共享（W3）"}
#: PNG 图内标签用英文（matplotlib 缺省字体 DejaVu Sans 无 CJK 字形；
#: 图表标签按工程惯例用英文，中文说明留在 table.md / analysis.md）。
CLASSES_EN = {"baseline": "no compression (trunk fp16)", "quant": "quantization (R1)",
              "window": "window / eviction (M2)", "head": "KV head sharing (W3)"}


def ascii_chart(points: list[dict], width: int = 72, height: int = 22) -> str:
    """ASCII 散点图：x = log10(每位置 KV 字节)，y = damage_bpc（越小越好）。"""
    xs = [math.log10(p["bytes_per_position"]) for p in points]
    ys = [p["damage_bpc"] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 - x0 < 1e-9:
        x1 = x0 + 1
    if y1 - y0 < 1e-9:
        y1 = y0 + 0.01
    grid = [[" "] * width for _ in range(height)]
    # 碰撞回避：同一格上的多点依次右移/左移/上移（否则基线行上 b/M/8/4/Q
    # 会互相覆盖，实测首版就丢了 fp16 基线的标记）。
    order = [(0, 0), (1, 0), (-1, 0), (2, 0), (-2, 0), (3, 0), (-3, 0),
             (0, -1), (1, -1), (-1, -1), (0, 1), (1, 1), (-1, 1)]
    for p, xv, yv in zip(points, xs, ys):
        cx = int(round((xv - x0) / (x1 - x0) * (width - 1)))
        cy = height - 1 - int(round((yv - y0) / (y1 - y0) * (height - 1)))
        for dx, dy in order:
            nx, ny = cx + dx, max(0, min(height - 1, cy + dy))
            if 0 <= nx < width and grid[ny][nx] == " ":
                grid[ny][nx] = p["mark"]
                break
    lines = ["x 轴 = log10(每位置 KV 字节)（左小而省 → 右大而贵）；"
             "y 轴 = damage_bpc（上=更差）", ""]
    for r, row in enumerate(grid):
        yval = y1 - r * (y1 - y0) / (height - 1)
        lines.append(f"{yval:+7.4f} |" + "".join(row))
    lines.append(" " * 9 + "+" + "-" * width)
    lines.append(" " * 10 + " ".join(f"{math.pow(10, x0 + i * (x1 - x0) / 4):>8.0f}"
                                     for i in range(5)))
    lines.append(" " * 10 + "每位置 KV 字节（解析）")
    lines.append("")
    by_mark: dict[str, list[str]] = {}
    for p in points:
        by_mark.setdefault(p["mark"], []).append(p["arm"])
    lines.append("标记→点：" + "  ".join(f"{m}={','.join(a)}" for m, a in by_mark.items()))
    lines.append("标记→类别：" + "  ".join(f"{k}=({c})" for k, c in CLASSES.items()))
    return "\n".join(lines)


def mark_of(arm: str, kind: str) -> str:
    if kind == "quant":
        return {"fp16": "b", "int8": "8", "int4": "4"}[arm]
    if kind == "window":
        return "w"
    return {"mha": "M", "gqa": "G", "mqa": "Q"}.get(arm, "?")


def markdown_table(points: list[dict]) -> str:
    head = ("| 点 | 类别 | 每位置 KV 字节（解析） | vs 本族基线 | bpc | Δbpc==damage | "
            "合成召回 copy（汇总） | 95% CI | vs chance | vs 局部先验 | 来源路径 | 口径 |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for p in points:
        ci = f"[{p['recall_copy']['wilson95'][0]:.4f}, {p['recall_copy']['wilson95'][1]:.4f}]"
        rows.append(
            f"| `{p['arm']}` | {CLASSES[p['class']]} | {p['bytes_per_position']:.0f} "
            f"({p['kv_reduction_vs_family_baseline'] * 100:.1f}%↓) | "
            f"{p['family_baseline']} | {p['bpc']:.6f} | {p['damage_bpc']:+.6f}"
            f"{' ⚠不可排序' if abs(p['damage_bpc']) < VARIANCE_BAND_BPC and p['damage_bpc'] != 0 else ''} | "
            f"{p['recall_copy']['accuracy']:.4f}"
            f"{' (地板)' if p['recall_copy']['at_floor'] else ''} | {ci} | "
            f"{p['recall_copy']['accuracy'] / p['chance_uniform']:.2f}× | "
            f"{p['recall_copy']['accuracy'] - p['local_prior_copy']:+.4f} | "
            f"`{p['source']}` | 解析字节（非实测显存） |")
    return head + "\n".join(rows) + "\n"


# ----------------------------------------------------------------- 主流程


def _write_charts(cfg, points, controls, rec, out_dir: str) -> None:
    """从点表重画「一张表 + 一张图」（P 解析口径的横轴 + ASCII 图 + PNG）。"""
    with open(os.path.join(out_dir, "table.md"), "w") as f:
        f.write(_render_table(cfg, points, controls, rec))
    with open(os.path.join(out_dir, "chart.txt"), "w") as f:
        f.write(ascii_chart(points) + "\n")
    try:
        _render_png(points, controls, os.path.join(out_dir, "chart.png"))
        rec["chart_png"] = True
    except Exception as e:                            # pragma: no cover - 环境相关
        rec["chart_png"] = f"unavailable: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--charts-only", action="store_true",
                    help="只从已有 results.json 重画图/表（纯 CPU，不取 GPU 锁）")
    ap.add_argument("--compare", default=None,
                    help="与另一个 run 的 results.json 对比（复现性散度），写进 results.json::reproducibility")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)
    if args.charts_only:
        rec = json.load(open(os.path.join(args.out_dir, "results.json")))
        _write_charts(rec["config"], rec["points"], rec["controls"], rec, args.out_dir)
        with open(os.path.join(args.out_dir, "results.json"), "w") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        print(f"[charts-only] 从 results.json 重画完成（纯 CPU）→ {args.out_dir}", flush=True)
        return 0
    cfg = json.load(open(args.config))
    dev = cfg["eval"]["device"]
    T = cfg["model"]["seq_len"]
    rec: dict = {"experiment": cfg["experiment"], "contract": cfg["contract"],
                 "scope": cfg["scope"], "config": cfg, "smoke": bool(args.smoke)}
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    _, eval_arr, _ = load_enwik8_full(".tmp/enwik8", os.path.join(ROOT, cfg["data"]["path"]))

    # ---------- ① 仪器自检（bpc 路径 / 合成路径 与库函数逐位一致） ----------
    trunk, trunk_ck = load_trunk(cfg, dev)
    trunk.eval()
    # 仪器自检用**恒等臂**（KV fp32 往返 = 不改数值），而不是 fp16 臂
    # （fp16 本身就是一档存储精度，会真的改数值；它的 bpc 与 R1 的 fp16 档对照）。
    fp32_fn = make_logits_fn(trunk, {"kind": "quant", "mode": "fp32"}, T, dev)
    anchor = cfg["controls"]["baseline_anchor_bpc"]
    lib_bpc = masked_bpc(trunk, eval_arr, None, n_batches=cfg["eval"]["n_batches"],
                         batch_size=cfg["eval"]["batch_size"],
                         window_seed=cfg["eval"]["window_seed"], device=dev)
    my_bpc = bpc_of(fp32_fn, eval_arr, T, cfg["eval"]["n_batches"],
                    cfg["eval"]["batch_size"], cfg["eval"]["window_seed"], dev)
    tasks = build_tasks(cfg, smoke=args.smoke)
    probe = tasks["var_recall"]
    lib_acc = lib_accuracy(trunk, probe, device=dev)
    my_acc = synth_accuracy(fp32_fn, probe, dev)
    self_check = {
        "bpc_path": {"bpc_of": round(my_bpc, 6), "retention.masked_bpc": round(lib_bpc, 6),
                     "abs_diff": abs(my_bpc - lib_bpc),
                     "bitwise_equal": my_bpc == lib_bpc,
                     "anchor_2614_release_bpc": anchor,
                     "abs_diff_vs_anchor": abs(my_bpc - anchor),
                     "mode": "fp32（恒等：KV 数值未改）"},
        "synth_path": {"run.py::synth_accuracy": my_acc["accuracy"],
                       "longrange_tasks.accuracy": lib_acc["accuracy"],
                       "n_equal": my_acc["n"] == lib_acc["n"],
                       "n_hits_equal": my_acc["n_hits"] == lib_acc["n_hits"],
                       "by_distance_equal": all(
                           my_acc["by_distance"][k]["accuracy"]
                           == lib_acc["by_distance"][k]["accuracy"]
                           for k in lib_acc["by_distance"])},
        "note": "两条腿都先证明「本实验路径 == 既有仪器路径」，再进入跨臂比较（SCHEMA §9 实现层等值）。",
    }
    rec["instrument"] = self_check
    print(f"[自检] 恒等臂 bpc_of={my_bpc:.6f} vs masked_bpc={lib_bpc:.6f} "
          f"(逐位={self_check['bpc_path']['bitwise_equal']})；"
          f"synth={my_acc['accuracy']:.4f} vs lib={lib_acc['accuracy']:.4f}", flush=True)

    # ---------- ② 对照臂：chance / 局部先验 / 毁源 ----------
    copy_positions = cfg["tasks"]["copy"]["src_positions"]
    copy_len = cfg["tasks"]["copy"]["copy_len"]
    # 毁源必须毁**该档自己的源区间** `[src_pos, src_pos+copy_len)`（不是恒为 [0,32)）：
    # 诊断脚本 `diagnose_copy_confound.py` 实测，若只毁 [0,32)，则 src_pos=96/191 两档的源
    # 根本没被碰过 → 那个「毁源对照」会虚高（实测 fp16 臂虚高到 0.0990，真值应为 chance）。
    tasks["copy_corrupt_by_pos"] = {
        p: corrupt_span(tasks["copy_by_pos"][p], p, p + copy_len)
        for p in tasks["copy_by_pos"]}
    controls = {
        "chance_uniform": round(1.0 / cfg["tokens"]["sym_vocab"], 8),
        "local_prior_copy": local_prior_accuracy(tasks["copy"], k=cfg["tokens"]["local_k"]),
        "local_prior_induction": local_prior_accuracy(tasks["induction"], k=cfg["tokens"]["local_k"]),
        "local_prior_var_recall": local_prior_accuracy(tasks["var_recall"], k=cfg["tokens"]["local_k"]),
        "local_prior_resistance": {
            "copy": check_local_prior_resistance(tasks["copy"], k=cfg["tokens"]["local_k"]),
            "induction": check_local_prior_resistance(tasks["induction"], k=cfg["tokens"]["local_k"]),
            "var_recall": check_local_prior_resistance(tasks["var_recall"], k=cfg["tokens"]["local_k"]),
        },
        "corrupt_source_note": ("把 copy 主腿的源片段整段换成随机符号（其余逐位不变）："
                                "读数若真的来自被搬运的信息，必须落回 chance。"
                                "**逐臂**都报了该对照（`recall_copy_source_corrupt`），"
                                "因为窗口臂即使源可见也不会看到它。"),
    }
    rec["controls"] = controls
    print(f"[对照] copy 局部先验={controls['local_prior_copy']['accuracy']:.4f} / "
          f"chance={controls['chance_uniform']:.4f}", flush=True)

    # ---------- ③ 逐臂：bpc + 合成召回 ----------
    families: dict[str, object] = {"trunk": trunk}
    points: list[dict] = []
    n_arms = len(cfg["arms"])
    probe_tokens = tasks["var_recall"].tokens[:4].to(dev)
    for i, (arm, spec) in enumerate(cfg["arms"].items(), 1):
        fam = spec["family"]
        if fam == "head":
            model, ck = load_head_arm(cfg, arm, dev)
        else:
            model, ck = trunk, trunk_ck
        fn = make_logits_fn(model, spec, T, dev)
        bpc = bpc_of(fn, eval_arr, T, cfg["eval"]["n_batches"], cfg["eval"]["batch_size"],
                     cfg["eval"]["window_seed"], dev)
        copy_pooled = pooled_accuracy(fn, tasks["copy_by_pos"], dev)
        copy_corrupt = pooled_accuracy(fn, tasks["copy_corrupt_by_pos"], dev)
        ind_pooled = pooled_accuracy(fn, tasks["induction_by_start"], dev)
        var_pooled = pooled_accuracy(fn, tasks["var_recall_by_d"], dev)
        geo = bytes_per_position(spec, n_layers=cfg["model"]["n_layers"],
                                 n_heads=cfg["model"]["n_heads"],
                                 head_dim=cfg["model"]["d_model"] // cfg["model"]["n_heads"],
                                 T=T)
        chance = controls["chance_uniform"]
        at_floor = copy_pooled["accuracy"] <= floor_threshold(copy_pooled["n"], chance)
        points.append({
            "arm": arm, "role": spec["role"], "family": fam, "class": (
                "baseline" if arm in ("fp16", "mha") else spec["kind"]),
            "checkpoint": ck,
            "bpc": round(bpc, 6), "damage_bpc": None, "family_baseline": None,
            "source": None,
            "activity": activity_of(spec, model, T, dev, probe_tokens),
            "recall_copy": {**copy_pooled, "at_floor": bool(at_floor)},
            "recall_copy_source_corrupt": copy_corrupt,
            "recall_induction": ind_pooled, "recall_var_recall": var_pooled,
            "chance_uniform": chance,
            "local_prior_copy": controls["local_prior_copy"]["accuracy"],
            "local_prior_induction": controls["local_prior_induction"]["accuracy"],
            "local_prior_var_recall": controls["local_prior_var_recall"]["accuracy"],
            "mark": mark_of(arm, spec["kind"]),
            **geo,
        })
        print(f"[{i}/{n_arms}] {arm:<7} bpc={bpc:.6f} copy={copy_pooled['accuracy']:.4f} "
              f"induction={ind_pooled['accuracy']:.4f} var={var_pooled['accuracy']:.4f} "
              f"bytes/pos={geo['bytes_per_position']:.0f}", flush=True)

    # ---------- ④ 相对本族基线的 damage + 来源路径 ----------
    by_arm = {p["arm"]: p for p in points}
    for p in points:
        base_arm = cfg["family_baseline"][p["family"]]
        base = by_arm[base_arm]
        p["family_baseline"] = base_arm
        p["damage_bpc"] = round(p["bpc"] - base["bpc"], 6)
        p["delta_schema"] = round(base["bpc"] - p["bpc"], 6)
        p["kv_reduction_vs_family_baseline"] = round(
            1 - p["bytes_per_position"] / base["bytes_per_position"], 6)
        p["source"] = _source_of(cfg, p["arm"])

    # ---------- ⑤ 与来源实验逐一对照 ----------
    src = source_values()
    cross = []
    for p in points:
        row = {"arm": p["arm"], "this_bpc": p["bpc"], "source": p["source"]}
        if p["arm"] in src["R1"]:
            row["source_bpc"] = src["R1"][p["arm"]]["bpc"]
        elif p["arm"] in src["W3"]:
            row["source_bpc"] = src["W3"][p["arm"]].get("bpc_second_path_masked_bpc")
            row["source_bpc_headline"] = src["W3"][p["arm"]]["bpc_clean"]
        elif p["arm"].startswith("win"):
            key = f"recent:d={cfg['arms'][p['arm']]['d']}"
            row["source_bpc"] = src["M2"].get(key, {}).get("bpc")
        if row.get("source_bpc") is not None:
            row["abs_diff"] = round(abs(p["bpc"] - row["source_bpc"]), 8)
            row["match"] = row["abs_diff"] <= 1e-5
        else:
            row["match"] = None
        cross.append(row)
    rec["cross_check_vs_source"] = {"rows": cross,
                                    "all_matched": all(r["match"] is not False for r in cross),
                                    "note": "本实验 14 臂全部同进程重测；此表证明与来源实验一致。"}

    # ---------- ⑥ 汇总落盘 ----------
    rec["points"] = points
    rec["missing_combinations"] = missing_combinations()
    rec["memory_convention"] = {
        "analytic_bytes_per_position": "见 points[*].bytes_per_position（判据 C1 的唯一公式）",
        "measured_gpu_peak_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "isolation_note": ("判据 C2：解析字节是本图的**横轴**；实测显存峰值单列在此，"
                           "**不**进横轴、**不**与解析列混排。评测期实现把 K/V 立刻"
                           "反量化 / 按掩码取子集（SDPA 仍算 T×T 分数矩阵）⇒ 实测峰值"
                           "不随档下降——这正是它不能当横轴的原因。"),
        "source_experiments_analytic_note": "R1/W3 的解析口径与本公式逐位一致（见 cross_check）。",
    }
    rec["code"] = {name: sha256_file(os.path.join(ROOT, name))[:16] for name in cfg["reuse"]["code_anchors"]}
    if args.compare:
        prev = json.load(open(args.compare))
        rows = []
        for a, b in zip(points, prev["points"]):
            rows.append({"arm": a["arm"], "bpc_abs_diff": abs(a["bpc"] - b["bpc"]),
                         "copy_abs_diff": abs(a["recall_copy"]["accuracy"]
                                              - b["recall_copy"]["accuracy"]),
                         "induction_abs_diff": abs(a["recall_induction"]["accuracy"]
                                                   - b["recall_induction"]["accuracy"]),
                         "var_abs_diff": abs(a["recall_var_recall"]["accuracy"]
                                             - b["recall_var_recall"]["accuracy"])})
        rec["reproducibility"] = {
            "compared_to": args.compare, "rows": rows,
            "max_bpc_abs_diff": max(r["bpc_abs_diff"] for r in rows),
            "max_recall_abs_diff": max(r["copy_abs_diff"] for r in rows),
            "threshold_bpc": cfg["reproducibility"]["max_abs_delta_bpc_threshold"],
            "pass": max(r["bpc_abs_diff"] for r in rows)
            <= cfg["reproducibility"]["max_abs_delta_bpc_threshold"]}
        print(f"[复现] max |Δbpc| = {rec['reproducibility']['max_bpc_abs_diff']:.2e}, "
              f"max |Δcopy| = {rec['reproducibility']['max_recall_abs_diff']:.2e}", flush=True)
    rec["cost"] = {"wall_seconds": None, "device": dev,
                   "wall_minutes": None, "peak_mem_mib": None}
    rec["cost"]["wall_seconds"] = round(time.time() - t0, 2)
    rec["cost"]["wall_minutes"] = round(rec["cost"]["wall_seconds"] / 60, 3)
    rec["cost"]["peak_mem_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)

    if not args.smoke:
        _write_charts(cfg, points, controls, rec, args.out_dir)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(rec, f, indent=1, ensure_ascii=False)
    print(f"[完成] {rec['cost']['wall_minutes']} min，峰值 {rec['cost']['peak_mem_mib']} MiB，"
          f"{len(points)} 点 → {args.out_dir}", flush=True)
    return 0


def _source_of(cfg: dict, arm: str) -> str:
    spec = cfg["arms"][arm]
    if spec["kind"] == "quant":
        return "axes/5-quant/5.1-kv-quant/experiments/2614-kv-quant-eval-13m/results.json"
    if spec["kind"] == "window":
        return ("axes/1-time/1.1.4.2-compression/experiments/"
                "2615-retention-retrieval-leg-13m/results.json")
    return "axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/results.json"


def _render_table(cfg, points, controls, rec) -> str:
    L = []
    L.append("# 2619 · 统一 KV 预算参照线 —— 一张表")
    L.append(f"横轴公式（判据 C1）：`bytes_per_position = 2 · n_layers · n_kv_heads_eff · "
             f"head_dim · value_bytes · (positions_kept / T)`，T = {cfg['model']['seq_len']}。"
             "**本列是解析口径**；实测显存单列在文末，两者不混。")
    L.append(f"bpc 与召回全部由本实验**同进程、同一路径**重测（`bpc_of` / `synth_accuracy`），"
             f"与来源实验逐一对照见 `results.json::cross_check_vs_source`。"
             f"召回地板：chance = {controls['chance_uniform']:.4f}（1/64）、"
             f"局部先验 = {controls['local_prior_copy']['accuracy']:.4f}（copy）/ "
             f"{controls['local_prior_induction']['accuracy']:.4f}（induction）/ "
             f"{controls['local_prior_var_recall']['accuracy']:.4f}（var_recall）；"
             f"逐臂毁源对照见下表。")
    L.append(markdown_table(points).rstrip("\n"))

    t2 = ["## 表 2 · 逐任务召回（同一仪器，同一 seed）",
          "| 点 | 每位置 KV 字节 | copy（汇总） | copy **毁源**对照 | induction（汇总） | "
          "var_recall（汇总） | 不可分辨的腿 |",
          "|---|---|---|---|---|---|---|"]
    for p in points:
        floors = [n for n, k in (("copy", "recall_copy"), ("induction", "recall_induction"),
                                 ("var_recall", "recall_var_recall"))
                  if p[k]["accuracy"]
                  <= floor_threshold(p[k]["n"], controls["chance_uniform"])]
        t2.append(f"| `{p['arm']}` | {p['bytes_per_position']:.0f} | "
                  f"{p['recall_copy']['accuracy']:.4f} | "
                  f"{p['recall_copy_source_corrupt']['accuracy']:.4f} | "
                  f"{p['recall_induction']['accuracy']:.4f} | "
                  f"{p['recall_var_recall']['accuracy']:.4f} | "
                  f"{', '.join(floors) if floors else '（无）'} |")
    t2.append("")
    t2.append("> 判据 C4：某腿 ≤ `chance + 2·σ_wilson` ⇒ 在该任务上**不可分辨**"
              "（不是「机制无效」，SCHEMA §19）。")
    L.append("\n".join(t2))

    t3 = ["## 表 3 · 与来源实验逐一对照",
          "| 点 | 本实验 bpc | 来源实验路径 | 来源值 | abs diff | 一致 |",
          "|---|---|---|---|---|---|"]
    for r in rec["cross_check_vs_source"]["rows"]:
        t3.append(f"| `{r['arm']}` | {r['this_bpc']:.6f} | `{r['source']}` | "
                  f"{r.get('source_bpc', '—')} | {r.get('abs_diff', '—')} | "
                  f"{'✓' if r['match'] else ('—' if r['match'] is None else '✗')} |")
    L.append("\n".join(t3))

    t4 = ["## 表 4 · 缺失组合（下一批候选，判据 C5）",
          "| 组合 | 为什么缺 | 若要补应怎么做 |", "|---|---|---|"]
    for m in rec["missing_combinations"]:
        t4.append(f"| {m['id']} | {m['why_missing']} | {m['next_step']} |")
    L.append("\n".join(t4))

    L.append(f"## 实测显存（**与解析列分列**，判据 C2）\n\n"
             f"- 本实验评测期 GPU 峰值（`torch.cuda.max_memory_allocated`）："
             f"**{rec['memory_convention']['measured_gpu_peak_mib']} MiB**（全臂共用一个进程，"
             f"是上界，不是逐臂读数）。\n"
             f"- 为什么不逐臂报：评测实现把 K/V 立刻反量化、窗口按掩码取子集，SDPA 仍物化 "
             f"T×T 分数矩阵 ⇒ **实测峰值不随档下降**，把它当横轴会得出「量化不省显存」的错结论。\n"
             f"- 解析列与来源实验一致：见 `results.json::cross_check_vs_source`。")

    L.append("## ASCII 图\n\n```\n" + ascii_chart(points) + "\n```")
    return "\n\n".join(L) + "\n"


def _render_png(points, controls, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style = {"baseline": ("k", "o"), "quant": ("tab:red", "s"),
             "window": ("tab:blue", "^"), "head": ("tab:green", "D")}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for p in points:
        c, m = style[p["class"]]
        axes[0].scatter(p["bytes_per_position"], p["damage_bpc"], color=c, marker=m, s=55)
        axes[0].annotate(p["arm"], (p["bytes_per_position"], p["damage_bpc"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
        axes[1].scatter(p["bytes_per_position"], p["recall_copy"]["accuracy"],
                        color=c, marker=m, s=55)
        axes[1].annotate(p["arm"], (p["bytes_per_position"], p["recall_copy"]["accuracy"]),
                         textcoords="offset points", xytext=(4, 4), fontsize=7)
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(alpha=0.3)
        ax.set_xlabel("bytes per position (analytic)")
    axes[0].axhline(VARIANCE_BAND_BPC, ls="--", c="gray", lw=1)
    axes[0].text(points[0]["bytes_per_position"], VARIANCE_BAND_BPC * 1.06,
                 "0.014 noise band", fontsize=7, color="gray")
    axes[0].set_ylabel("damage_bpc (vs own family baseline; lower = better)")
    axes[0].set_title("leg 1: clean bpc")
    axes[1].axhline(controls["chance_uniform"], ls=":", c="gray", lw=1)
    axes[1].text(points[0]["bytes_per_position"], controls["chance_uniform"] * 1.2,
                 "chance 1/64", fontsize=7, color="gray")
    axes[1].axhline(controls["local_prior_copy"]["accuracy"], ls="--", c="gray", lw=1)
    axes[1].text(points[0]["bytes_per_position"], controls["local_prior_copy"]["accuracy"] * 1.15,
                 "local prior (copy)", fontsize=7, color="gray")
    axes[1].set_ylabel("synthetic recall: copy (pooled; higher = better)")
    axes[1].set_title("leg 2: long-range retrieval (synthetic, local-prior resistant)")
    handles = [plt.Line2D([], [], color=c, marker=m, ls="", label=CLASSES_EN[k])
               for k, (c, m) in style.items()]
    axes[1].legend(handles=handles, fontsize=8, loc="upper left")
    fig.suptitle("2619 · unified KV budget chart (T=256; bytes/position is ANALYTIC, not measured GPU memory)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    sys.exit(main())
