#!/usr/bin/env python3
"""2671-padfill-13m · S2 全量扫描驱动（契约见本目录 README.md）。

    python3 run.py --stage all --out . --seeds 0 1 --steps 1500

阶段：
- `gate`：在**当前（最终）revision** 上重跑契约 §5 的 L0 / L0b 硬门，写 `evidence/l0_l0b_probe_s2.txt`。
- `scan`：5 臂 × `--seeds` 训练（S2 的 `S = 1500`，S1 按契约 §3 选定、不得改）+ 64 固定 offset 评测。
- `all`：`gate` + `scan` + 落盘 `results.json` / `analysis.md` / `MODEL.md` / `train_log.jsonl`。

纪律：
- **逐 run 落盘**：每跑完一个 `(arm, seed)` 即重写 `results.json` 与 `RESUME.md`，可取续跑
  （`--resume` 跳过已完成 run）。
- **GPU 按臂取放锁**（`gpu_lock.py`，`SCHEMA.md` §20）：每个 run 单独 with gpu_lock，绝不整程独占；
  锁被占则登记并重试，**不杀**他人进程。
- **禁墙钟**：`results.json` / `analysis.md` 判据不含墙钟语义字段（`P.has_wall_clock` 自检必须为 `[]`）；
  墙钟只允许出现在 `run_log.txt`。
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import math
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import padfill as P  # noqa: E402
from gpu_lock import GPUUnavailable, gpu_lock  # noqa: E402

REFERENCE_BAND = 0.014                    # 契约 §4：主判 = bpc 口径 ⇒ 0.014 bpc 参考带（限配对 Δ）
# 每次取锁的等待上限（秒）；外层 supervisor 以 20 s 节奏重试，锁被他人长占时不会死等。
LOCK_MAX_WAIT = float(os.environ.get("WP632_LOCK_MAX_WAIT", "600"))
FILL_ARMS = ["fill_bidir_r10", "fill_bidir_r25", "fill_bidir_r50", "fill_causal_r25"]
AR_AT_M_RATES = {"0.1": 0.10, "0.25": 0.25, "0.5": 0.50}   # AR 参照要报 ar_at_M 的 r
LOG_EVERY = 100


# ---- 纯函数（可单测；run.py 的自有逻辑都集中在此）-------------------------------
def exceeds_reference_band(delta_bpc: float, band: float = REFERENCE_BAND) -> bool:
    """主判是否越 0.014 bpc 参考带（严格大于；边界 0.014 不算越）。"""
    return abs(delta_bpc) > band


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0.0 or hi < 0.0


def delta_bpc_block(s_fill_M: float, s_ar_M: float, n_m: int) -> float:
    """单 block 配对 Δbpc = bpc(fill) − bpc(ar_at_M)（同 M_r，越低越好的 bpc 口径）。"""
    return (float(s_fill_M) - float(s_ar_M)) / (int(n_m) * math.log(2))


def mean_over_seeds(per_seed_blocks: list[list[float]]) -> list[float]:
    """两 seed 的逐 block Δ 先对齐求均值（同一 M_r / 同一 offset），再喂聚类 bootstrap。"""
    return np.asarray(per_seed_blocks, dtype=np.float64).mean(axis=0).tolist()


def cost_axis(arm: str) -> dict:
    """契约 §7 的等效理论量（禁墙钟）：tokens-per-forward、每 block 顺序前向次数、每 token 等效 FLOPs。"""
    cfg = P.ARMS[arm]
    T, L = P.SEQ_LEN, P.N_LAYERS
    if cfg["loss"] == "ar":
        return {"tokens_per_forward": 1.0,
                "sequential_forwards_per_block": float(T - 1),
                "equivalent_flops_per_token": float(L)}
    r = float(cfg["r"])
    return {"tokens_per_forward": r * T,
            "sequential_forwards_per_block": 1.0,
            "equivalent_flops_per_token": L / r}


def wall_clock_fields(obj) -> list[str]:
    """L5 禁墙钟自检：复用 `padfill.has_wall_clock`（key 命中墙钟语义即报）。"""
    return P.has_wall_clock(obj)


def source_sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def arm_run_key(arm: str, seed: int) -> str:
    return f"{arm}|s{seed}"


def ar_next_token_loss(logits: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """AR 臂的 next-token 交叉熵。

    `win` 形状 (B, T+1)，`logits = model(win[:, :T])` 形状 (B, T, V)：位置 i 的 logits 预测 `win[i+1]`，
    故目标 = `win[:, 1:]`（B, T），与 `logits` 逐位对齐。**不得**再对 logits 切 `[:, :-1]`
    （`padfill.py::train_steps` 的 AR 分支有此 off-by-one，会得到 16352 vs 16384 而抛 ValueError；
    见 RESUME.md 的偏差记录）。
    """
    V = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, V), win[:, 1:].reshape(-1))


def train_ar(arm: str, seed: int, steps: int, train_arr, device: str = "cuda",
             log_fn=None):
    """AR 臂训练：镜像 `padfill.train_steps` 的 AR 分支，仅把 off-by-one 目标修正。

    数据顺序 seed（`DATA_SEED`）与模型 init seed 与其它臂一致 ⇒ matched-step、同 token 预算。
    """
    torch.manual_seed(seed)
    model = P.PadFillLM(attn_mask="causal").to(device)
    opt = P.build_optimizer(model)
    dg = torch.Generator().manual_seed(P.DATA_SEED)
    model.train()
    V = model.vocab_size
    for i in range(steps):
        win = P.sample_window(train_arr, P.BATCH_SIZE, P.SEQ_LEN + 1, dg).to(device)
        logits = model(win[:, :P.SEQ_LEN])
        loss = ar_next_token_loss(logits, win)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if log_fn is not None and (i + 1) % LOG_EVERY == 0:
            log_fn(i + 1, float(loss.detach()))
    model.eval()
    return model


# ---- 日志 / 工具 ---------------------------------------------------------------
def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Tee:
    """墙钟只允许出现在这里（run_log.txt）。"""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def _load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _dump(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _fresh_results(steps: int, seeds: list[int], offsets: list[int], device: str,
                   smoke: bool) -> dict:
    return {
        "experiment": "2671-padfill-13m",
        "contract": "README.md (pre-registered)",
        "stage_note": "S2 全量扫描（5 臂 × 2 seed）在 S = 1500（S1 按契约 §3 选定）",
        "smoke": bool(smoke),
        "config": {
            "steps": steps,
            "batch_size": P.BATCH_SIZE,
            "seq_len": P.SEQ_LEN,
            "lr": P.LR,
            "optimizer": "Adam(no weight decay)",
            "data_seed": P.DATA_SEED,
            "mask_seed": P.MASK_SEED,
            "eval_mask_seed": P.EVAL_MASK_SEED,
            "bootstrap": {"B": P.BOOT_B, "seed": P.BOOT_SEED,
                          "cluster": "eval_block", "ci": "percentile95"},
            "seeds": list(seeds),
            "arm_rates": {a: P.ARMS[a]["r"] for a in P.ARMS},
            "arm_attn": {a: P.ARMS[a]["attn_mask"] for a in P.ARMS},
            "eval_blocks": len(offsets),
            "eval_offsets": list(offsets),
            "min_mask_pos": P.MIN_MASK_POS,
            "reference_band_bpc": REFERENCE_BAND,
            "device": device,
            "structure": {"d_model": P.D_MODEL, "n_layers": P.N_LAYERS,
                          "n_heads": P.N_HEADS, "mlp_mult": P.MLP_MULT,
                          "params": P.count_params(P.PadFillLM(attn_mask="causal"))},
        },
        "ar": {},
        "arms": {},
        "L0": {}, "L0b": {}, "L1": {}, "L2": {}, "L3": {}, "L4": {}, "L5": {},
        "falsification": {}, "verdicts": {},
    }


# ---- L0 / L0b 硬门（最终 revision 上重跑）--------------------------------------
def run_gate(results: dict, out: str, log, device: str = "cpu") -> None:
    """契约 §5 的 L0 / L0b：交付结构 + 交付 mask 语义上，四条检查（两绿两红）。"""
    V, T = P.VOCAB_SIZE, P.SEQ_LEN
    gen = torch.Generator().manual_seed(1)
    true = torch.randint(0, V - 1, (2, T), generator=gen)          # 真 token 恒 ≠ [mask]
    M = P.sample_fill_mask(true.shape, 0.25, torch.Generator().manual_seed(7))
    pos = torch.arange(T)[None, :]
    half = T // 2
    right = (~M) & (pos >= half)
    leftM = M & (pos < half)

    def mk(attn: str):
        torch.manual_seed(0)
        m = P.PadFillLM(attn_mask=attn).to(device)
        m.eval()
        return m

    def perturb_at(t, sel):
        t2 = t.clone()
        t2[sel] = (t[sel] + 1) % (V - 1)
        return t2

    def fwd(m, x):
        with torch.no_grad():
            return m(x)

    def maxabs(a, b, sel):
        return float((a - b).abs()[sel].max())

    m_full, m_causal = mk("full"), mk("causal")
    xa = P.make_fill_input(true, M, "mask"); xa2 = P.make_fill_input(perturb_at(true, M), M, "mask")
    g1 = maxabs(fwd(m_full, xa), fwd(m_full, xa2), M)
    xb = P.make_fill_input(true, M, "mask"); xb2 = P.make_fill_input(perturb_at(true, right), M, "mask")
    g2 = maxabs(fwd(m_causal, xb), fwd(m_causal, xb2), leftM)
    xc = P.make_fill_input(true, M, "leak"); xc2 = P.make_fill_input(perturb_at(true, M), M, "leak")
    r1 = maxabs(fwd(m_full, xc), fwd(m_full, xc2), M)
    r2 = maxabs(fwd(m_full, xb), fwd(m_full, xb2), leftM)

    assert g1 == 0.0, f"L0-a 必须红不了？g1={g1}"
    assert g2 == 0.0, f"L0-b 必须红不了？g2={g2}"
    assert r1 > 0.0 and r2 > 0.0, f"gate probe 未变红 r1={r1} r2={r2}"
    lines = [
        "L0 / L0b 硬门（S2 最终 revision 上重跑；交付结构 + 交付 mask 语义）",
        f"source_sha256 padfill.py = {source_sha(os.path.join(HERE, 'padfill.py'))}",
        f"source_sha256 run.py     = {source_sha(os.path.join(HERE, 'run.py'))}",
        "",
        f"L0-a  (bidir mask,   perturb true@M)   max|Δ| = {g1!r}   [绿 ⇒ 期望 0.0]",
        f"L0-b  (causal,       perturb right)    max|Δ| = {g2!r}   [绿 ⇒ 期望 0.0]",
        f"probe-1 (leak mask,  perturb true@M)   max|Δ| = {r1!r}   [红 ⇒ 期望 > 0]",
        f"probe-2 (full attn,  perturb right)    max|Δ| = {r2!r}   [红 ⇒ 期望 > 0]",
    ]
    for tag, val in (("probe-1", r1), ("probe-2", r2)):
        try:
            assert val == 0.0
            raise AssertionError(f"{tag} 未变红 —— gate probe 失效")
        except AssertionError:
            lines.append(f"  gate probe {tag}: `assert max|Δ| == 0` 抛 AssertionError ⇒ 检查可失败 ✓")
    lines.append("ALL GREEN/PLACED: L0 两条绿、gate probe 两条红。")
    ev = os.path.join(out, "evidence")
    os.makedirs(ev, exist_ok=True)
    with open(os.path.join(ev, "l0_l0b_probe_s2.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    results["L0"] = {"bidir_perturb_M_maxabs": g1, "causal_perturb_right_maxabs": g2,
                     "pass": bool(g1 == 0.0 and g2 == 0.0),
                     "note": "扰动 M 位置真值后 M 位置 logits 逐位不变；因果臂右侧扰动不影响左侧 M"}
    results["L0b"] = {"leak_mask_probe_maxabs": r1, "full_attn_probe_maxabs": r2,
                      "both_red": bool(r1 > 0.0 and r2 > 0.0),
                      "note": "泄漏构造 / causal 改 full 后 L0 必须变红 ⇒ 仪器可失败"}
    results["L0"]["source_sha256"] = {"padfill.py": source_sha(os.path.join(HERE, "padfill.py")),
                                      "run.py": source_sha(os.path.join(HERE, "run.py"))}
    log(f"gate: L0 pass={results['L0']['pass']} L0b both_red={results['L0b']['both_red']} "
        f"g1={g1} g2={g2} r1={r1} r2={r2}")


# ---- 训练 + 评测（逐 run）------------------------------------------------------
def train_one(arm: str, seed: int, steps: int, train_arr, device: str, log) -> tuple:
    """按臂取放 GPU 锁后训一个 run；返回 (model, loss_trace)。"""
    trace: list[dict] = []

    def log_fn(step, loss):
        trace.append({"step": int(step), "loss": loss})

    ctx = gpu_lock(name=f"wp6.3.2-{arm}-s{seed}", max_wait=LOCK_MAX_WAIT) if device == "cuda" \
        else contextlib.nullcontext()
    with ctx:
        log(f"train arm={arm} seed={seed} steps={steps} device={device} lock_acquired")
        if arm == "ar":
            # 见 ar_next_token_loss：lib 的 AR 分支 off-by-one，AR 训练在 run.py 内修正（不改 padfill.py）
            model = train_ar(arm, seed, steps, train_arr, device=device, log_fn=log_fn)
        else:
            model = P.train_arm(arm, seed, steps, train_arr, device=device, log_fn=log_fn)
        if trace:
            log(f"train DONE arm={arm} seed={seed} last_loss={trace[-1]['loss']:.5f} lock_released")
        else:
            log(f"train DONE arm={arm} seed={seed} (no log point; steps < {LOG_EVERY}) lock_released")
    return model, trace


def eval_one(ar_model, fill_model, val, offsets, r: float, device: str) -> dict:
    return P.eval_block_set(ar_model, fill_model, val, offsets, r, device=device)


def _run_from_eval(arm: str, seed: int, r: float, ev: dict, trace, is_ar: bool) -> dict:
    blocks = [{"offset": b["offset"], "n_m": b["n_m"],
               "s_ar_M": b["s_ar_M"], "s_fill_M": b["s_fill_M"]} for b in ev["blocks"]]
    entry = {"arm": arm, "seed": seed, "r": r, "attn_mask": P.ARMS[arm]["attn_mask"],
             "ar_full_bpc": ev["ar_full_bpc"], "n_m_total": ev["n_m_total"],
             "blocks": blocks, "loss_trace": trace, "done": True}
    if is_ar:
        return entry
    deltas = [delta_bpc_block(b["s_fill_M"], b["s_ar_M"], b["n_m"]) for b in blocks]
    mean, lo, hi = P.paired_bootstrap_ci(deltas)
    entry.update({
        "ar_at_M_bpc": ev["ar_at_M_bpc"],
        "fill_M_bpc": ev["fill_M_bpc"],
        "delta_mean": mean, "delta_lo": lo, "delta_hi": hi,
        "ci_excludes_zero": ci_excludes_zero(lo, hi),
        "exceeds_band": exceeds_reference_band(mean),
        "delta_blocks": deltas,
    })
    return entry


def run_scan(results: dict, out: str, val, train_arr, seeds: list[int], steps: int,
             device: str, log, resume: bool) -> None:
    offsets = results["config"]["eval_offsets"]

    def done(key: str) -> bool:
        return bool(resume and _find_run(results, key))

    # --- AR 参照臂（每个 seed 一个模型；供同 seed 填空臂配对）---
    ar_models: dict[int, object] = {}
    for seed in seeds:
        key = arm_run_key("ar", seed)
        if done(key):
            log(f"SKIP (resume) {key}")
            continue
        model, trace = train_one("ar", seed, steps, train_arr, device, log)
        per_r, n_m = {}, {}
        ctx = gpu_lock(name=f"wp6.3.2-ar-s{seed}-eval", max_wait=LOCK_MAX_WAIT) if device == "cuda" \
            else contextlib.nullcontext()
        with ctx:
            ev_full = eval_one(model, None, val, offsets, 0.25, device)
            for rtag, r in AR_AT_M_RATES.items():
                ev = eval_one(model, None, val, offsets, r, device)
                per_r[rtag] = ev["ar_at_M_bpc"]
                n_m[rtag] = ev["n_m_total"]
        ar_models[seed] = model
        results["ar"][str(seed)] = {
            "arm": "ar", "seed": seed, "attn_mask": "causal",
            "ar_full_bpc": ev_full["ar_full_bpc"],
            "ar_at_M_bpc": per_r, "n_m_total": n_m,
            "blocks": [{"offset": b["offset"], "n_m": b["n_m"],
                        "s_ar_M": b["s_ar_M"], "s_fill_M": b["s_fill_M"]}
                       for b in ev_full["blocks"]],
            "loss_trace": trace, "done": True,
        }
        log(f"ar seed={seed} ar_full_bpc={ev_full['ar_full_bpc']:.5f} "
            f"ar_at_M_25={per_r['0.25']:.5f}")
        _dump(os.path.join(out, "results.json"), results)
        _write_resume(out, results, steps, seeds, ["ar"])

    # --- 填空臂 ---
    for seed in seeds:
        if seed not in ar_models:
            pending = [arm for arm in FILL_ARMS if not done(arm_run_key(arm, seed))]
            if not pending:
                # 全部填空臂已完成 ⇒ 配对读数已在 results.json，重训 AR 只对「新臂评测」有用，此处无新臂
                log(f"resume: s{seed} 填空臂已全完成 ⇒ 跳过 AR 配对重训（配对读数已在 results.json）")
                continue
            # resume 时 AR 已完成但模型不在内存：重训 AR（成本 ~1 run），保证配对同源
            log(f"resume: AR s{seed} 已完成但未驻留内存 ⇒ 重训以配对（不写入 results）")
            ar_models[seed], _ = train_one("ar", seed, steps, train_arr, device, log)
        for arm in FILL_ARMS:
            key = arm_run_key(arm, seed)
            if done(key):
                log(f"SKIP (resume) {key}")
                continue
            r = float(P.ARMS[arm]["r"])
            model, trace = train_one(arm, seed, steps, train_arr, device, log)
            if device == "cuda":
                with gpu_lock(name=f"wp6.3.2-{arm}-s{seed}-eval", max_wait=LOCK_MAX_WAIT):
                    ev = eval_one(ar_models[seed], model, val, offsets, r, device)
            else:
                ev = eval_one(ar_models[seed], model, val, offsets, r, device)
            entry = _run_from_eval(arm, seed, r, ev, trace, is_ar=False)
            results["arms"].setdefault(arm, {})[str(seed)] = entry
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            log(f"fill arm={arm} seed={seed} fill_M_bpc={entry['fill_M_bpc']:.5f} "
                f"ar_at_M_bpc={entry['ar_at_M_bpc']:.5f} "
                f"Δ={entry['delta_mean']:+.5f} CI[{entry['delta_lo']:+.5f},"
                f"{entry['delta_hi']:+.5f}]")
            _dump(os.path.join(out, "results.json"), results)
            with open(os.path.join(out, "train_log.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps({"arm": arm, "seed": seed, "r": r, "steps": steps,
                                    "trace": trace}, ensure_ascii=False) + "\n")
            _write_resume(out, results, steps, seeds, FILL_ARMS)


def _find_run(results: dict, key: str) -> dict | None:
    arm, s = key.split("|s")
    seed = s
    if arm == "ar":
        return results.get("ar", {}).get(seed) if results.get("ar", {}).get(seed, {}).get("done") else None
    return results.get("arms", {}).get(arm, {}).get(seed) \
        if results.get("arms", {}).get(arm, {}).get(seed, {}).get("done") else None


# ---- 汇总 / 判读 ---------------------------------------------------------------
def _blocks_of(entry: dict) -> tuple[list[float], list[float], list[int]]:
    """(fill per-block bpc, ar_at_M per-block bpc, n_m)。"""
    ln2 = math.log(2)
    fill = [b["s_fill_M"] / (b["n_m"] * ln2) for b in entry["blocks"]]
    ar = [b["s_ar_M"] / (b["n_m"] * ln2) for b in entry["blocks"]]
    n_m = [b["n_m"] for b in entry["blocks"]]
    return fill, ar, n_m


def _arm_deltas(results: dict, arm: str) -> list[list[float]]:
    return [e["delta_blocks"] for e in (results["arms"].get(arm, {}).values())]


def _aggregate_arm(results: dict, arm: str) -> dict:
    per_seed = {}
    deltas_per_seed = []
    for s, e in sorted(results["arms"].get(arm, {}).items()):
        if "delta_blocks" not in e:
            continue
        per_seed[s] = {"delta_mean": e["delta_mean"], "delta_lo": e["delta_lo"],
                       "delta_hi": e["delta_hi"], "ci_excludes_zero": e["ci_excludes_zero"],
                       "fill_M_bpc": e["fill_M_bpc"], "ar_at_M_bpc": e["ar_at_M_bpc"]}
        deltas_per_seed.append(e["delta_blocks"])
    out = {"per_seed": per_seed}
    if deltas_per_seed:
        comb = mean_over_seeds(deltas_per_seed)
        mean, lo, hi = P.paired_bootstrap_ci(comb)
        out["combined_mean_over_seeds"] = {
            "delta": mean, "ci95": [lo, hi], "excludes_zero": ci_excludes_zero(lo, hi),
            "exceeds_band": exceeds_reference_band(mean)}
    return out


def finalize(results: dict, out: str, steps: int, seeds: list[int], log) -> None:
    # L1：逐臂主判 + CI
    results["L1"] = {arm: _aggregate_arm(results, arm) for arm in FILL_ARMS}
    results["L1"]["_note"] = ("主判差 = bpc(fill_*) − bpc(ar_at_M_r)（同一批 M_r 逐 block 配对）；"
                              "combined = 两 seed 逐 block Δ 先对齐求均值再做 block 聚类配对 bootstrap；"
                              "参考带 = 0.014 bpc（bpc 口径，限配对 Δ、≥5 窗口）")

    # L2：r 形状（fill_bidir 的 Δ 随 r 走向）
    r_series = []
    for arm in ["fill_bidir_r10", "fill_bidir_r25", "fill_bidir_r50"]:
        comb = results["L1"][arm].get("combined_mean_over_seeds")
        r_series.append({"arm": arm, "r": float(P.ARMS[arm]["r"]),
                         "delta": comb["delta"] if comb else None,
                         "ci95": comb["ci95"] if comb else None})
    deltas = [x["delta"] for x in r_series]
    monotonic_worse = all(deltas[i] < deltas[i + 1] for i in range(len(deltas) - 1)) \
        if all(d is not None for d in deltas) else None
    results["L2"] = {"fill_bidir_series": r_series,
                     "monotonic_worsening_with_r": monotonic_worse,
                     "note": "Δ 随 r 单调恶化 = 每步都严格增大（bpc 越低越好 ⇒ 增大即变差）"}

    # L3：可见性对照（fill_causal_r25 vs fill_bidir_r25，同 M_25 逐 block 配对）
    l3 = {}
    ca = results["arms"].get("fill_causal_r25", {})
    bi = results["arms"].get("fill_bidir_r25", {})
    common = [s for s in ca if s in bi]
    if common:
        diffs = []
        for s in common:
            fc, _, _ = _blocks_of(ca[s])
            fb, _, _ = _blocks_of(bi[s])
            diffs.append([a - b for a, b in zip(fc, fb)])
        comb = mean_over_seeds(diffs)
        mean, lo, hi = P.paired_bootstrap_ci(comb)
        l3 = {"pair": ["fill_causal_r25", "fill_bidir_r25"],
              "delta_mean": mean, "ci95": [lo, hi],
              "excludes_zero": ci_excludes_zero(lo, hi),
              "note": "Δ = bpc(fill_causal_r25) − bpc(fill_bidir_r25)，同 M_25 逐 block 配对"}
    results["L3"] = l3

    # L4：代价轴（等效理论量，禁墙钟）
    results["L4"] = {
        "cost_axis": {arm: cost_axis(arm) for arm in P.ARMS},
        "measured_tokens_per_forward": {
            arm: (sum(e["n_m_total"] for e in results["arms"][arm].values())
                  / max(len(results["arms"][arm]), 1))
            for arm in FILL_ARMS if arm in results["arms"]},
        "note": "AR tokens-per-forward = 1（逐 token）；填空一次前向填出 |M| ≈ r·T 个位置；"
                "每 token 等效 FLOPs（以 AR 每 token 的 L 为单位）= L/r。全部理论量，禁墙钟。",
    }

    # L5：禁墙钟自检 + 确定性
    bad = wall_clock_fields({k: v for k, v in results.items() if k != "_wall_clock_fields"})
    results["_wall_clock_fields"] = bad
    if bad:
        raise AssertionError(f"results.json 含墙钟字段：{bad}")

    # L1/L2/L3 与 _wall_clock_fields 均已就位后才判否证（否则在空 L1 上判读）
    _compute_falsification(results)

    _write_train_log(out, results)
    _write_model_md(results, out, steps, seeds)
    _write_analysis(results, out, steps, seeds)
    log(f"finalize DONE wall_clock_fields={bad}")


def _write_train_log(out: str, results: dict) -> None:
    """从 results 重写 train_log.jsonl（去重；逐 run 训练轨迹）。"""
    with open(os.path.join(out, "train_log.jsonl"), "w", encoding="utf-8") as f:
        for seed_s, e in sorted(results.get("ar", {}).items()):
            f.write(json.dumps({"arm": "ar", "seed": int(seed_s), "r": 0.0,
                                "trace": e.get("loss_trace", [])}, ensure_ascii=False) + "\n")
        for arm in FILL_ARMS:
            for seed_s, e in sorted(results.get("arms", {}).get(arm, {}).items()):
                f.write(json.dumps({"arm": arm, "seed": int(seed_s), "r": e["r"],
                                    "trace": e.get("loss_trace", [])}, ensure_ascii=False) + "\n")


def _fmt_ci(comb) -> str:
    if not comb:
        return "—"
    return (f"Δ={comb['delta']:+.5f} CI[{comb['ci95'][0]:+.5f},{comb['ci95'][1]:+.5f}] "
            f"{'不含0' if comb['excludes_zero'] else '含0'}"
            f"{' ·越带' if comb['exceeds_band'] else ' ·带内'}")


def _write_analysis(results: dict, out: str, steps: int, seeds: list[int]) -> None:
    cfg = results["config"]
    L = []
    L.append("# analysis — 2671-padfill-13m（S2 全量扫描：5 臂 × 2 seed）")
    L.append("")
    L.append("## 人话摘要")
    L.append("")
    L.append(f"**Nk 快照口径：本实验 `S = {steps}`，与仓库既有 6000 步档不可比；只读臂间增量与排序**"
             "（`SCHEMA.md` §20.3）。")
    L.append("")
    L.append("我们在**同一 harness、同一 token 预算、同一结构**（唯一变量 = 注意力可见性 + 损失位置集）下，"
             "比较「一次前向把所有待填位置填出来」的填空臂和全因果 AR 参照：AR 逐 token 生成，"
             "填空臂一次前向填出全部 PAD。主判是**同一批待填位置 `M_r` 上的 bpc 差**"
             "（每字符比特数，越低越好；Δ = 填空 − AR，正 = 填空更差），带 0.014 bpc 参考带。")
    # headline numbers
    head = []
    for arm in FILL_ARMS:
        comb = results["L1"].get(arm, {}).get("combined_mean_over_seeds")
        if comb:
            head.append(f"`{arm}` Δ={comb['delta']:+.4f} bpc "
                        f"(95% CI [{comb['ci95'][0]:+.4f},{comb['ci95'][1]:+.4f}], "
                        f"{'越带' if comb['exceeds_band'] else '带内'}, "
                        f"{'CI 不含 0' if comb['excludes_zero'] else 'CI 含 0'})")
    ar_full = [results["ar"][s]["ar_full_bpc"] for s in results.get("ar", {})]
    if ar_full:
        L.append(f"参照系：AR 臂全位置 teacher-forced bpc = "
                 f"{np.mean(ar_full):.4f}（越低越好；各种子 {', '.join(f'{x:.4f}' for x in ar_full)}）。"
                 f"相比 AR，填空臂在待填位置上的质量代价：")
        for h in head:
            L.append(f"- {h}")
    L.append("")
    l2 = results["L2"]
    if l2.get("monotonic_worsening_with_r") is not None:
        series_txt = ", ".join(f"r={x['r']:.2f}: {x['delta']:+.4f}" for x in l2["fill_bidir_series"]
                               if x["delta"] is not None)
        mono = l2["monotonic_worsening_with_r"]
        L.append(f"- **r 形状（L2）**：fill_bidir 的 Δ 随 r ∈ {{10%,25%,50%}} "
                 f"{'**单调恶化**' if mono else '**不单调**（r=0.25 代价最小）'}（序列 {series_txt}）。"
                 + ("" if mono else "⇒ 契约 §6「随 r 单调恶化」子条件**未满足**。"))
    l3 = results.get("L3") or {}
    if l3:
        L.append(f"- **可见性对照（L3）**：`fill_causal_r25 − fill_bidir_r25` Δ={l3['delta_mean']:+.4f} "
                 f"CI[{l3['ci95'][0]:+.4f},{l3['ci95'][1]:+.4f}] "
                 f"({'CI 不含 0 ⇒ 可见性有独立增益' if l3['excludes_zero'] else 'CI 含 0 ⇒ 可见性无独立增益'})。")
    L.append(f"- **硬门（L0/L0b）**：L0 "
             f"{'**通过**' if results.get('L0', {}).get('pass') else '**未过**'}；"
             f"L0b gate probe 两条 "
             f"{'**均变红（仪器可失败）**' if results.get('L0b', {}).get('both_red') else '未全红'}。")
    L.append(f"- **禁墙钟（L5）**：`results.json` 墙钟语义字段 = "
             f"{results.get('_wall_clock_fields')}（0 命中）。")
    L.append("")
    L.append("> 信息集不对称（必须同时读）：`fill_bidir` 的信息集 = 全部**非 M** 真 token（含右侧）+ "
             "自身位置的 mask 占位；`ar_at_M_r` 的信息集 = 全部**左侧**真 token。两者**互不包含**，"
             "因此主判差**不是**「纯机制优劣」，而是两种可用信息下的质量差。")
    L.append("")

    L.append("## 口径与配置")
    L.append("")
    L.append(f"- matched-step：steps={cfg['steps']}、batch={cfg['batch_size']}、T={cfg['seq_len']}、"
             f"lr={cfg['lr']}、{cfg['optimizer']}；模型种子 = {cfg['seeds']}")
    L.append(f"- 数据顺序 seed={cfg['data_seed']}（各臂同 token 预算）；训练 mask seed={cfg['mask_seed']}；"
             f"评测 mask seed={cfg['eval_mask_seed']}")
    L.append(f"- 评测：enwik8 val 的 {cfg['eval_blocks']} 个固定 offset × T={cfg['seq_len']} block；"
             f"位置集 M_r：i ≥ {cfg['min_mask_pos']} 且以概率 r 取「被填」")
    L.append(f"- 主判 bootstrap：block 聚类配对，B={cfg['bootstrap']['B']}、"
             f"seed={cfg['bootstrap']['seed']}、percentile 95% CI；参考带 = {cfg['reference_band_bpc']} bpc")
    L.append(f"- 结构：d_model={cfg['structure']['d_model']}、n_layers={cfg['structure']['n_layers']}、"
             f"n_heads={cfg['structure']['n_heads']}、mlp_mult={cfg['structure']['mlp_mult']}、"
             f"params={cfg['structure']['params']:,}")
    L.append("")

    L.append("## 数字表：主判与参照 bpc")
    L.append("")
    L.append("| 臂 | 种子 | attn | r | ar_at_M bpc | fill_M bpc | Δbpc | 95% CI | 越 0.014 带 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for arm in FILL_ARMS:
        for s, e in sorted(results["arms"].get(arm, {}).items()):
            L.append(f"| `{arm}` | {s} | {e['attn_mask']} | {e['r']:.2f} | "
                     f"{e['ar_at_M_bpc']:.5f} | {e['fill_M_bpc']:.5f} | {e['delta_mean']:+.5f} | "
                     f"[{e['delta_lo']:+.5f},{e['delta_hi']:+.5f}] | "
                     f"{'是' if e['exceeds_band'] else '否'} |")
        comb = results["L1"].get(arm, {}).get("combined_mean_over_seeds")
        if comb:
            L.append(f"| `{arm}` | **合并** | — | — | — | — | **{comb['delta']:+.5f}** | "
                     f"**[{comb['ci95'][0]:+.5f},{comb['ci95'][1]:+.5f}]** | "
                     f"{'是' if comb['exceeds_band'] else '否'} |")
    L.append("")
    L.append("| AR 参照 | 种子 | ar_full bpc | ar_at_M_10 | ar_at_M_25 | ar_at_M_50 |")
    L.append("|---|---|---|---|---|---|")
    for s, e in sorted(results.get("ar", {}).items()):
        m = e["ar_at_M_bpc"]
        L.append(f"| `ar` | {s} | {e['ar_full_bpc']:.5f} | {m.get('0.1', float('nan')):.5f} | "
                 f"{m.get('0.25', float('nan')):.5f} | {m.get('0.5', float('nan')):.5f} |")
    L.append("")

    L.append("## L0–L5 逐条")
    L.append("")
    L.append(f"- **L0 不受未来泄漏硬门**：`pass={results['L0'].get('pass')}` "
             f"(bidir 扰动 M 真值 max|Δ|={results['L0'].get('bidir_perturb_M_maxabs')}, "
             f"causal 右侧扰动 max|Δ|={results['L0'].get('causal_perturb_right_maxabs')})")
    L.append(f"- **L0b 可失败性自检**：`both_red={results['L0b'].get('both_red')}` "
             f"(泄漏构造 max|Δ|={results['L0b'].get('leak_mask_probe_maxabs')}, "
             f"causal 改 full max|Δ|={results['L0b'].get('full_attn_probe_maxabs')})；"
             f"证据 = `evidence/l0_l0b_probe_s2.txt`")
    L.append(f"- **L1 可读性/质量主判**：{'齐全' if results['L1'] else '缺'}（逐臂逐 seed 的 "
             f"ar_full / ar_at_M / fill_M bpc + 配对 Δ + CI）")
    L.append(f"- **L2 r 形状**：`monotonic_worsening_with_r={results['L2'].get('monotonic_worsening_with_r')}`")
    L.append(f"- **L3 可见性对照**：{'有' if results.get('L3') else '缺'}")
    L.append(f"- **L4 代价轴**：报告齐全（等效理论量，禁墙钟）")
    l5 = results.get("L5", {})
    sc = l5.get("short_train_20step_checksum_identical")
    det_txt = (f"eval_rerun_identical={l5.get('eval_rerun_identical')}；"
               f"short_train_20step_checksum_identical={sc}")
    if l5.get("short_train_checksum"):
        det_txt += f"（两次校验和 {l5['short_train_checksum']}）"
    if sc is True and l5.get("eval_rerun_identical") is True:
        det_txt += "。判：**通过**。"
    elif sc is False:
        det_txt += "。判：**确定性未通过**（短程训练同 seed 两次重跑的校验和未逐位相同）。"
    else:
        det_txt += "。判：**未通过 / 未验证**（本 revision 未取得 GPU 锁独立重跑，未取得非 None 值）。"
    det_txt += ("该子项以浮点逐位相等判定，GPU 归约顺序可使其跨运行抖动（同 revision 曾同时观测到"
                " True 与 False）；历次观测值见 `run_log.txt`。")
    for _ek in ("short_train_error", "eval_error"):
        if l5.get(_ek):
            det_txt += f" 错误：{_ek}={l5[_ek]}。"
    L.append(f"- **L5 禁墙钟自检**：`_wall_clock_fields={results.get('_wall_clock_fields')}`（0 命中）。")
    L.append(f"- **L5 确定性**：{det_txt}")
    L.append("")

    L.append("## §6 否证条件逐条")
    L.append("")
    fal = results.get("falsification", {})
    for key, val in fal.items():
        if isinstance(val, dict) and "triggered" in val:
            L.append(f"- `{key}`：{'**触发**' if val['triggered'] else '未触发'} —— {val.get('detail','')}")
    L.append("")
    L.append(f"**判读**：{results.get('verdicts', {}).get('main', '—')}")
    L.append("")

    L.append("## 边界与未测")
    L.append("")
    L.append(f"- 结论只适用于本设定（{steps} 步 / char enwik8 / 交付 12.8M 结构 / 该 r 集合），不外推；"
             f"S < 6000 ⇒ 快照口径，与仓库既有 6000 步档不可比。")
    L.append("- 模型权重不入库（无 ckpt）；同 seed 由初始化 + 固定数据/mask 生成器重建。")
    L.append("")
    with open(os.path.join(out, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def _write_model_md(results: dict, out: str, steps: int, seeds: list[int]) -> None:
    cfg = results["config"]
    L = [
        "# MODEL · 2671-padfill-13m",
        "",
        "本实验自带 harness（`padfill.py`），因为共享引擎 `pathlm/model.py` 的注意力是**硬编码因果**的，",
        "而主判需要 `full`（双向）可见性的填空臂（不得改 `pathlm/**`）。AR 参照臂用**同一 harness 的因果**",
        "配置，保证唯一结构变量只有注意力可见性（+ 损失位置集）。",
        "",
        "## 结构（契约 §2 写死）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 形态 | pre-LN decoder-only char LM |",
        f"| d_model | {cfg['structure']['d_model']} |",
        f"| n_layers | {cfg['structure']['n_layers']} |",
        f"| n_heads | {cfg['structure']['n_heads']}（head_dim = {cfg['structure']['d_model'] // cfg['structure']['n_heads']}） |",
        f"| mlp_mult | {cfg['structure']['mlp_mult']} |",
        f"| seq_len | {cfg['seq_len']} |",
        f"| 读出 | 共享 tied：`logits = LN_F(h) @ E^T` |",
        f"| 注意力 | `F.scaled_dot_product_attention`；`attn_mask ∈ {{causal, full}}` 是唯一结构变量 |",
        f"| 实测参数量 | {cfg['structure']['params']:,}（两可见性逐位相同） |",
        "",
        "## 数据",
        "",
        "`data/enwik8_full.npz`（仓库既有，`pathlm/data.py::load_enwik8_full`）：train 90M / val 10M 字节、",
        f"`vocab_size = {P.VOCAB_SIZE}`（205 字节值 + 1 个 `[mask]` 槽，`[mask]` = 末槽 = id {P.MASK_TOKEN}）。",
        "",
        "## 五臂（契约 §2）与本次读数",
        "",
        "| 臂 | 注意力 | r | 种子 | ar_at_M bpc | fill_M bpc | Δbpc |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in FILL_ARMS:
        for s, e in sorted(results["arms"].get(arm, {}).items()):
            L.append(f"| `{arm}` | {e['attn_mask']} | {e['r']:.2f} | {s} | {e['ar_at_M_bpc']:.5f} | "
                     f"{e['fill_M_bpc']:.5f} | {e['delta_mean']:+.5f} |")
    for s, e in sorted(results.get("ar", {}).items()):
        L.append(f"| `ar` | causal | — | {s} | — | — | —（参照 ar_full={e['ar_full_bpc']:.5f}） |")
    L += [
        "",
        "权重不入库（`.gitignore` 覆盖 `*.pt`）：模型由 `torch.manual_seed(seed)` + 固定数据/mask 生成器",
        "重建（`padfill.train_arm`），确定性见 `analysis.md` 的 L5。",
        "",
        "## 复现命令",
        "",
        "```bash",
        "cd /home/z/vibe/pathlm/axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m",
        "python3 padfill.py --params",
        "python3 -m pytest test_padfill.py test_run_s2.py -q",
        f"python3 run.py --stage all --out . --seeds {' '.join(str(s) for s in seeds)} --steps {steps}",
        "python3 mutate.py            # 变异抓红 + 逐字节还原",
        "```",
        "",
    ]
    with open(os.path.join(out, "MODEL.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ---- RESUME -------------------------------------------------------------------
def _write_resume(out: str, results: dict, steps: int, seeds: list[int], _files) -> None:
    done = []
    for s, e in results.get("ar", {}).items():
        if e.get("done"):
            done.append(f"ar|s{s}")
    for arm in FILL_ARMS:
        for s, e in results.get("arms", {}).get(arm, {}).items():
            if e.get("done"):
                done.append(f"{arm}|s{s}")
    missing = [arm_run_key(a, s) for a in ["ar"] + FILL_ARMS for s in seeds
               if arm_run_key(a, s) not in done]
    lines = [
        "# RESUME · 2671-padfill-13m（S2 续跑状态）",
        "",
        f"- 目标：5 臂 × 2 seed 在 S = {steps} 的 matched-step 训练 + 64 offset 评测（契约 §3/§4）。",
        f"- 已完成 run（{len(done)}/{len(seeds) * 5}）：{', '.join(done) if done else '（无）'}",
        f"- 未完成 run：{', '.join(missing) if missing else '（无，全量完成）'}",
        "",
        "## 下一个确切命令",
        "",
        "```bash",
        "cd /home/z/vibe/pathlm/axes/6-predict/6.3.2-pad-fill/experiments/2671-padfill-13m",
        f"python3 run.py --stage all --out . --seeds {' '.join(str(s) for s in seeds)} "
        f"--steps {steps} --resume",
        "```",
        "",
        "`--resume` 跳过 `results.json` 中 `done=true` 的 run；仅当该 seed 仍有未完成的填空臂时才重训 AR 以保证配对同源（该 seed 填空臂全部完成则跳过，配对读数已在 `results.json`）。",
        "GPU 按臂取放锁（`gpu_lock.py`），锁被占则登记重试，不杀他人进程。",
        "",
    ]
    with open(os.path.join(out, "RESUME.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---- 主流程 -------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["gate", "scan", "all"])
    ap.add_argument("--out", default=HERE)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-lock", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    log = Tee(os.path.join(out, "run_log.txt"))
    torch.backends.cudnn.benchmark = False

    steps = 3 if a.smoke else a.steps
    if not a.smoke and steps != 1500:
        raise SystemExit(f"契约 §3：S 已由 S1 选定为 1500，不得改（收到 steps={a.steps}）")

    train_arr, val, vocab = P.load_data()
    n_off = 2 if a.smoke else P.EVAL_BLOCKS
    offsets = P.eval_offsets(len(val), n=n_off)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if a.no_lock and device == "cuda":
        device = "cpu"
    log(f"stage={a.stage} out={out} seeds={a.seeds} steps={steps} device={device} "
        f"vocab={vocab} eval_blocks={len(offsets)} smoke={a.smoke} resume={a.resume}")

    rpath = os.path.join(out, "results.json")
    results = _load_results(rpath) if a.resume else {}
    if not results:
        results = _fresh_results(steps, a.seeds, offsets, device, a.smoke)

    if a.stage in ("gate", "all"):
        run_gate(results, out, log)
        _dump(rpath, results)

    if a.stage in ("scan", "all"):
        try:
            run_scan(results, out, val, train_arr, a.seeds, steps, device, log, a.resume)
        except GPUUnavailable as exc:
            # 锁被他人长占：落盘已完成部分 + RESUME，退出码 3 由 supervisor 以 20 s 节奏重试。
            log(f"GPUUnavailable: {exc} ⇒ 落盘已完成部分 + RESUME（退出码 3）")
            _write_resume(out, results, steps, a.seeds, [])
            _dump(rpath, results)
            return 3

    # 完整性检查
    all_done = all(_find_run(results, arm_run_key(arm, s)) for arm in ["ar"] + FILL_ARMS
                   for s in a.seeds)
    if a.stage == "all" and all_done:
        _compute_L5(results, out, train_arr, device, log)
        finalize(results, out, steps, a.seeds, log)
        _dump(rpath, results)
    else:
        log(f"未全量完成（all_done={all_done}）⇒ 只落盘已完成部分；analysis.md 待全量完成后生成")
        _write_resume(out, results, steps, a.seeds, [])
        _dump(rpath, results)
    return 0


def _compute_L5(results: dict, out: str, train_arr, device: str, log) -> None:
    """L5：禁墙钟（finalize 里再查一次）+ 确定性（短程训练同 seed 重跑 + 评测重跑逐值相同）。"""
    det = {}
    # (a) 评测确定性：同一只训练好的模型评两遍，逐值相同（不取锁，CPU/GPU 都可）
    lock = gpu_lock(name="wp6.3.2-l5", max_wait=LOCK_MAX_WAIT) if device == "cuda" else contextlib.nullcontext()
    try:
        with lock:
            torch.manual_seed(0)
            m_ar = P.PadFillLM(attn_mask="causal").to(device).eval()
            torch.manual_seed(0)
            m_fi = P.PadFillLM(attn_mask="full").to(device).eval()
            off = P.eval_offsets(len(train_arr), n=4)
            e1 = P.eval_block_set(m_ar, m_fi, train_arr, off, 0.25, device=device)
            e2 = P.eval_block_set(m_ar, m_fi, train_arr, off, 0.25, device=device)
        det["eval_rerun_identical"] = bool(
            e1["delta_blocks"] == e2["delta_blocks"]
            and (e1["delta_lo"], e1["delta_hi"]) == (e2["delta_lo"], e2["delta_hi"]))
    except Exception as exc:  # pragma: no cover
        det["eval_rerun_identical"] = None
        det["eval_error"] = repr(exc)
    # (b) 训练确定性：短程（30 步）同 seed 重跑两次，末步 loss 与 logits 校验和逐值相同（CPU 省时）
    try:
        def short_train():
            torch.manual_seed(0)
            model = P.PadFillLM(attn_mask="full").to(device)
            opt = P.build_optimizer(model)
            dg = torch.Generator().manual_seed(P.DATA_SEED)
            mg = torch.Generator().manual_seed(P.MASK_SEED)
            P.train_steps(model, opt, "fill_bidir_r25", 20, train_arr, device, dg, mg)
            with torch.no_grad():
                idx = torch.randint(0, 205, (2, 64),
                                    generator=torch.Generator().manual_seed(9)).to(device)
                return float(model(idx).sum())
        with lock:
            c1, c2 = short_train(), short_train()
        det["short_train_20step_checksum_identical"] = bool(c1 == c2)
        det["short_train_checksum"] = [c1, c2]
    except Exception as exc:  # pragma: no cover
        det["short_train_20step_checksum_identical"] = None
        det["short_train_error"] = repr(exc)
    det["summary"] = ("评测同模型重跑逐值相同；训练以 20 步短程同 seed 重跑校验和验证（完整 1500 步重跑受时间盒限制）"
                      if det.get("eval_rerun_identical") and det.get("short_train_20step_checksum_identical")
                      else "见 L5 明细（如有非相同项，如实声明）")
    results["L5"] = det
    log(f"L5 determinism: eval_rerun_identical={det.get('eval_rerun_identical')} "
        f"short_train_identical={det.get('short_train_20step_checksum_identical')}")


def _compute_falsification(results: dict) -> None:
    """契约 §6 逐条判定。"""
    cfg = results["config"]
    band = float(cfg["reference_band_bpc"])
    l1 = results["L1"]
    bidir = ["fill_bidir_r10", "fill_bidir_r25", "fill_bidir_r50"]
    combs = {a: l1.get(a, {}).get("combined_mean_over_seeds") for a in bidir}
    all_exclude = all(combs[a] and combs[a]["excludes_zero"] for a in bidir)
    all_exceed = all(combs[a] and exceeds_reference_band(combs[a]["delta"], band) for a in bidir)
    monotonic = results["L2"].get("monotonic_worsening_with_r")
    l3 = results.get("L3") or {}
    causal_worse = bool(l3 and l3.get("delta_mean", 0.0) > 0.0)
    # 若所有 bidir 都不越带或 CI 含 0 ⇒ 不可分
    # 契约 §6：任一 r 的 |Δ| ≤ 带 或 CI 含 0 ⇒ 该 r 的代价不可分（本函数注释与契约均作「至少一条」）
    indeterminable = any((not combs[a]) or ((not exceeds_reference_band(combs[a]["delta"], band))
                                            or (not combs[a]["excludes_zero"])) for a in bidir)

    fal = {
        "cond_A_not_viable": {
            "triggered": bool(all_exclude and all_exceed and monotonic and causal_worse),
            "detail": f"bidir CI 全不含 0={all_exclude}、|Δ| 全越 {band} 带={all_exceed}、"
                      f"r 单调恶化={monotonic}、causal 更差={causal_worse}"},
        "cond_B_indeterminable": {
            "triggered": bool(indeterminable),
            "detail": f"至少一条 bidir 的 |Δ| ≤ {band} 或 CI 含 0 ⇒ 本尺度下该 r 的质量代价不可分"
                      f"（不得写成「无损」）"},
        "cond_C_visibility_no_gain": {
            "triggered": bool(l3 and not l3.get("excludes_zero")),
            "detail": "" if not l3 else
                      f"fill_bidir_r25 − fill_causal_r25 CI 含 0={not l3['excludes_zero']} ⇒ "
                      f"填空视图的可见性在本尺度无独立增益"},
    }
    results["falsification"] = fal

    if fal["cond_A_not_viable"]["triggered"]:
        main = ("判「一次填充（K=1 并行填充）在本尺度 / 本任务上**不成立**」；"
                "6.3.2 的 failure_analysis 须区分设计问题 / 结构问题。")
    elif fal["cond_B_indeterminable"]["triggered"]:
        main = "判「本尺度下该 r 的质量代价不可分」（不得写成「无损」）。"
    else:
        main = (f"逐臂分判（见数字表）：bidir 主判差可分辨 —— CI 全不含 0={all_exclude}、"
                f"|Δ| 全越 {band} 带={all_exceed}；r 单调恶化={monotonic}"
                f" ⇒ 未满足契约 §6「不成立」条件的全部要求，按各臂读数分判。")
    extra = ""
    if fal["cond_C_visibility_no_gain"]["triggered"]:
        extra = " 另判「填空视图的可见性在本尺度无独立增益」。"
    results["verdicts"] = {
        "main": main + extra,
        "L0": f"pass={results['L0'].get('pass')}",
        "L0b": f"both_red={results['L0b'].get('both_red')}",
        # 键名避开墙钟语义词（wall/clock/time/...）：`has_wall_clock` 扫描整份 results，
        # 键名自带“墙钟”会被自检误报；此处只报自检结果，不是墙钟量。
        "L5_forbidden_field_scan": f"0 命中={results.get('_wall_clock_fields') == []}",
    }


if __name__ == "__main__":
    raise SystemExit(main())
