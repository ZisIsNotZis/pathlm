#!/usr/bin/env python3
"""2674-mdiffusion-13m · 驱动（契约见本目录 `../../6.3.1-diffusion/experiments/2674-mdiffusion-13m/README.md`）。

阶段：
- `mdiff`：训 `mdiff` 臂 × `--seeds`（`matched-step = 2671/results.json::config.steps`，不得改），
  在 `r_eval = R_EVAL`（=0.25，与 2671 fill_bidir_r25 同 M）上跑 K∈{1,4,8,16} 采样评测；逐 seed 落盘。
- `arretrain`：可选——2671 的 AR checkpoint **未落盘**，若需要 AR 权重（L3 rescore）则用逐字段相同的
  config 在本目录重训 AR 臂并显式声明（`--ar retrain`）。优先复用 2671 的读数。
- `finalize`：汇总 L0–L5 / §4 否证条件 / 判读，写 `results.json` / `analysis.md` / `MODEL.md`。
- `all` = `mdiff` +（`--ar retrain` 时 `arretrain`）+ `finalize`。

纪律（契约 §7 / 派发）：
- **逐 run 取放锁**（`gpu_lock.py`，`harness/SCHEMA.md` §20）；锁被占则带超时重试，**不杀**他人进程。
- **禁墙钟**进 `results.json`/`analysis.md`：`P.has_wall_clock` 自检必须 `[]`；墙钟只进 `run_log.txt`。
- 只写本目录；只读 import / 复用 `2671-padfill-13m/**`。
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, HERE)

import mdiff as M  # noqa: E402
import padfill as P  # noqa: E402
from gpu_lock import GPUUnavailable, gpu_lock  # noqa: E402

RESULTS_2671_PATH = M.RESULTS_2671
REFERENCE_BAND = M.REFERENCE_BAND_BPC
K_LIST = list(M.K_LIST)
R_EVAL = M.R_EVAL
LOCK_MAX_WAIT = float(os.environ.get("WP631_LOCK_MAX_WAIT", "900"))


# ---- 纯函数（可单测）----------------------------------------------------------
def exceeds_reference_band(delta_bpc: float, band: float = REFERENCE_BAND) -> bool:
    """主判是否越 0.014 bpc 参考带（严格大于；边界不算越）。"""
    return abs(delta_bpc) > band


def ci_excludes_zero(lo: float, hi: float) -> bool:
    return lo > 0.0 or hi < 0.0


def delta_bpc_block(s_mdiff: float, s_ar_M: float, n_m: int) -> float:
    """单 block 配对 Δbpc = bpc(mdiff) − bpc(ar_at_M)（同 M，越低越好的 bpc 口径）。"""
    return (float(s_mdiff) - float(s_ar_M)) / (int(n_m) * math.log(2))


def mean_over_seeds(per_seed_blocks: list[list[float]]) -> list[float]:
    """两 seed 的逐 block Δ 先对齐求均值（同一 M / offset），再喂聚类 bootstrap。"""
    return np.asarray(per_seed_blocks, dtype=np.float64).mean(axis=0).tolist()


def cost_axis_mdiff(K: int, n_m_total: int, blocks: int, n_layers: int = P.N_LAYERS,
                    seq_len: int = P.SEQ_LEN) -> dict:
    """契约 §5 等效理论量（禁墙钟）：tokens-per-forward、每 block 顺序前向次数、每 token 等效 FLOPs。"""
    return {
        "tokens_per_forward": (n_m_total / max(blocks, 1)) / K,   # |M|/K（逐 block |M| 的均值 / K）
        "sequential_forwards_per_block": float(K),
        "equivalent_flops_per_token": float(K * n_layers),
    }


def cost_axis_ar(n_layers: int = P.N_LAYERS, seq_len: int = P.SEQ_LEN) -> dict:
    return {"tokens_per_forward": 1.0,
            "sequential_forwards_per_block": float(seq_len - 1),
            "equivalent_flops_per_token": float(n_layers)}


def wall_clock_fields(obj) -> list[str]:
    return P.has_wall_clock(obj)


def source_sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---- 日志 ---------------------------------------------------------------------
def now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


class Tee:
    """墙钟只允许出现在这里（run_log.txt）。"""

    def __init__(self, path: str):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def _dump(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _load(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---- 评测：M 复现 + K 步扫描 --------------------------------------------------
def build_eval_M(offsets: list[int]) -> torch.Tensor:
    """逐 offset 用与 2671 `eval_block_set` **同一生成器顺序**采样 M（r=R_EVAL, min_pos=8）。"""
    gen = torch.Generator().manual_seed(P.EVAL_MASK_SEED)
    rows = []
    for _ in offsets:
        rows.append(P.sample_fill_mask((1, P.SEQ_LEN), R_EVAL, gen, min_pos=P.MIN_MASK_POS)[0])
    return torch.stack(rows)                     # (n_blocks, T) bool


def build_blocks(val, offsets: list[int], T: int = P.SEQ_LEN) -> torch.Tensor:
    return torch.stack([torch.from_numpy(np.ascontiguousarray(val[o:o + T]).astype(np.int64))
                        for o in offsets])


@torch.no_grad()
def eval_K_sweep(model, blocks: torch.Tensor, mask: torch.Tensor, K_list: list[int],
                 device: str) -> dict:
    """逐 K 跑采样器，返回每 K 的逐 block s_mdiff/n_m 与标量聚合（全部确定性）。"""
    out: dict[str, dict] = {}
    for K in K_list:
        res = M.sample_denoise_k(model, blocks, mask, K)
        s = res["s_mdiff"].detach().cpu().numpy().astype(float)
        n = res["n_m"].detach().cpu().numpy().astype(int)
        met = M.mdiff_metrics(res)
        out[str(K)] = {
            "s_mdiff_blocks": s.tolist(),
            "n_m_blocks": n.tolist(),
            "mdiff_M_bpc": met["mdiff_M_bpc"],
            "nelbo_step_mean_bpc": met["nelbo_step_mean_bpc"],
            "last_step_bpc": met["last_step_bpc"],
            "step_ce_mean_bpc": met["step_ce_mean_bpc"],
            "step_n": met["step_n"],
            "step_fix": met["step_fix"],
            "n_m_total": met["n_m_total"],
            "n_forwards": K,
        }
    return out


# ---- AR（只读参照 + 可选重训，用于 L3 rescore）--------------------------------
@torch.no_grad()
def ar_rescore_M(ar_model, seq: torch.Tensor, true_tokens: torch.Tensor,
                 M: torch.Tensor) -> torch.Tensor:
    """冻结 AR 模型对（含 mdiff 预测 token 的）生成序列 teacher-forced rescore，返回逐 block CE 和。"""
    logits = ar_model(seq)                                     # (B,T,V)
    idx = torch.nonzero(M, as_tuple=False)                     # (N,2)
    b, i = idx[:, 0], idx[:, 1]
    ce = -torch.log_softmax(logits[b, i - 1].float(), dim=-1).gather(
        -1, true_tokens[b, i].unsqueeze(-1)).squeeze(-1)
    sums = torch.zeros(M.shape[0], device=seq.device)
    sums.index_add_(0, b, ce)
    return sums


def train_ar(seed: int, steps: int, train_arr, device: str = "cuda", log_fn=None):
    """重训 AR（逐字段同 2671 run.py::train_ar：修正 lib off-by-one）；仅在 2671 无 ckpt 时使用。"""
    torch.manual_seed(seed)
    model = P.PadFillLM(attn_mask="causal").to(device)
    opt = P.build_optimizer(model)
    dg = torch.Generator().manual_seed(P.DATA_SEED)
    model.train()
    V = model.vocab_size
    for i in range(steps):
        win = P.sample_window(train_arr, P.BATCH_SIZE, P.SEQ_LEN + 1, dg).to(device)
        logits = model(win[:, :P.SEQ_LEN])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, V), win[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if log_fn is not None and (i + 1) % 100 == 0:
            log_fn(i + 1, float(loss.detach()))
    model.eval()
    return model


# ---- 逐 run 取放锁 ------------------------------------------------------------
def _locked(device: str, name: str):
    return gpu_lock(name=name, max_wait=LOCK_MAX_WAIT) if device == "cuda" else contextlib.nullcontext()


# ---- 主流程 -------------------------------------------------------------------
def _steps_from_2671() -> int:
    ref = M.load_2671()
    return int(ref["config"]["steps"])


def run_mdiff(results, out, val, train_arr, seeds, steps, device, log):
    offsets = results["config"]["eval_offsets"]
    blocks = build_blocks(val, offsets).to(device)
    M_eval = build_eval_M(offsets).to(device)
    # L0 保真：M 与 2671 逐 block n_m 对齐（同一生成器顺序）
    ar_ref = M.ar_reference_blocks(M.load_2671(), R_EVAL)
    mismatch = []
    for j, off in enumerate(offsets):
        for seed_entry in ar_ref.values():
            if int(seed_entry["blocks"][off]["n_m"]) != int(M_eval[j].sum()):
                mismatch.append((off, int(seed_entry["blocks"][off]["n_m"]), int(M_eval[j].sum())))
    results["L0"] = {
        "m_align_2671_diff_list": mismatch,
        "pass": len(mismatch) == 0,
        "note": "本项 M（r=0.25, min_pos=8, eval_mask_seed=2674）逐 block n_m 与 2671 ar 落盘 blocks 完全对齐 ⇒ 同一位置集",
        "padfill_sha256": source_sha(os.path.join(M.PADFILL_DIR, "padfill.py")),
    }
    log(f"L0 M-alignment pass={results['L0']['pass']} diffs={len(mismatch)}")

    for seed in seeds:
        if results.get("mdiff", {}).get(str(seed), {}).get("done"):
            log(f"SKIP (resume) mdiff s{seed}")
            continue
        trace: list[dict] = []

        def log_fn(step, loss, _t=trace):
            _t.append({"step": int(step), "loss": loss})

        with _locked(device, f"wp6.3.1-mdiff-s{seed}"):
            log(f"train arm=mdiff seed={seed} steps={steps} lock_acquired")
            model = M.train_mdiff(seed, steps, train_arr, device=device, log_fn=log_fn)
            log(f"train DONE arm=mdiff seed={seed} last_loss={trace[-1]['loss']:.5f}")
            per_K = eval_K_sweep(model, blocks, M_eval, K_LIST, device)
            # 确定性：同模型同输入重跑 K=4 逐值相同
            det = eval_K_sweep(model, blocks, M_eval, [4], device)["4"]
            deterministic = (det["s_mdiff_blocks"] == per_K["4"]["s_mdiff_blocks"]
                             and det["step_fix"] == per_K["4"]["step_fix"])
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        results.setdefault("mdiff", {})[str(seed)] = {
            "arm": "mdiff", "seed": seed, "attn_mask": "full",
            "objective": "absorbing_mask_diffusion(per_step r~U(0,1), loss on masked)",
            "steps": steps, "loss_trace": trace, "per_K": per_K,
            "determinism_rerun_K4_equal": bool(deterministic), "done": True,
        }
        log(f"mdiff seed={seed} K1 bpc={per_K['1']['mdiff_M_bpc']:.5f} "
            f"K16 bpc={per_K['16']['mdiff_M_bpc']:.5f} determinism={deterministic}")
        _dump(os.path.join(out, "results.json"), results)


def run_ar_retrain(results, out, val, train_arr, seeds, steps, device, log):
    offsets = results["config"]["eval_offsets"]
    blocks = build_blocks(val, offsets).to(device)
    M_eval = build_eval_M(offsets).to(device)
    ref = M.load_2671()
    for seed in seeds:
        if results.get("ar_retrained", {}).get(str(seed), {}).get("done"):
            log(f"SKIP (resume) ar_retrained s{seed}")
            continue
        trace: list[dict] = []

        def log_fn(step, loss, _t=trace):
            _t.append({"step": int(step), "loss": loss})

        with _locked(device, f"wp6.3.1-ar-s{seed}"):
            log(f"train arm=ar(retrain) seed={seed} steps={steps} lock_acquired")
            ar = train_ar(seed, steps, train_arr, device=device, log_fn=log_fn)
            with torch.no_grad():
                lg = ar(blocks)                                 # (B,T,V)
                s_full = float(M.P.nll_of(lg[:, :-1], blocks[:, 1:]).sum())
                ar_full_bpc = M.P.bpc_from_nll(s_full, int(blocks.shape[0]) * (P.SEQ_LEN - 1))
                idx = torch.nonzero(M_eval, as_tuple=False)
                b, i = idx[:, 0], idx[:, 1]
                s_arM = float(M.P.nll_of(lg[b, i - 1], blocks[b, i]).sum())
                n_m_tot = int(M_eval.sum())
                ar_at_M_bpc = M.P.bpc_from_nll(s_arM, n_m_tot)
                del ar
            if device == "cuda":
                torch.cuda.empty_cache()
        results.setdefault("ar_retrained", {})[str(seed)] = {
            "arm": "ar", "seed": seed, "attn_mask": "causal", "steps": steps,
            "ar_full_bpc": ar_full_bpc, "ar_at_M_bpc": ar_at_M_bpc, "n_m_total": n_m_tot,
            "loss_trace": trace, "done": True,
        }
        log(f"ar(retrain) seed={seed} ar_full_bpc={ar_full_bpc:.5f} ar_at_M_bpc={ar_at_M_bpc:.5f}")
        _dump(os.path.join(out, "results.json"), results)


def _aggregate_K(results: dict) -> dict:
    """逐 K：per-seed Δ + CI，与两 seed 逐 block Δ 合并后的 CI（对齐 offset）。"""
    ref = M.ar_reference_blocks(M.load_2671(), R_EVAL)
    out: dict[str, dict] = {}
    blocks_nm = {s: ref[s]["blocks"] for s in ref}
    for K in K_LIST:
        per_seed = {}
        deltas_per_seed = []
        for seed_s, e in sorted(results.get("mdiff", {}).items()):
            seed = int(seed_s)
            pk = e["per_K"][str(K)]
            s_mdiff = pk["s_mdiff_blocks"]
            n_m = pk["n_m_blocks"]
            offs = results["config"]["eval_offsets"]
            deltas = [delta_bpc_block(s_mdiff[j], blocks_nm[seed][offs[j]]["s_ar_M"], n_m[j])
                      for j in range(len(offs))]
            mean, lo, hi = P.paired_bootstrap_ci(deltas)
            per_seed[seed_s] = {
                "mdiff_M_bpc": pk["mdiff_M_bpc"], "nelbo_step_mean_bpc": pk["nelbo_step_mean_bpc"],
                "last_step_bpc": pk["last_step_bpc"], "delta_mean": mean,
                "delta_lo": lo, "delta_hi": hi, "ci_excludes_zero": ci_excludes_zero(lo, hi),
                "exceeds_band": exceeds_reference_band(mean),
            }
            deltas_per_seed.append(deltas)
        comb = {}
        if deltas_per_seed:
            d = mean_over_seeds(deltas_per_seed)
            mean, lo, hi = P.paired_bootstrap_ci(d)
            comb = {"delta_mean": mean, "delta_lo": lo, "delta_hi": hi,
                    "ci_excludes_zero": ci_excludes_zero(lo, hi),
                    "exceeds_band": exceeds_reference_band(mean),
                    "mdiff_M_bpc_mean": float(np.mean([v["mdiff_M_bpc"] for v in per_seed.values()])),
                    "nelbo_step_mean_bpc_mean": float(np.mean(
                        [v["nelbo_step_mean_bpc"] for v in per_seed.values()])),
                    "last_step_bpc_mean": float(np.mean(
                        [v["last_step_bpc"] for v in per_seed.values()]))}
        out[str(K)] = {"per_seed": per_seed, "combined": comb}
    return out


def finalize(results, out, steps, seeds, device, log):
    cfg = results["config"]
    results["K"] = _aggregate_K(results)
    # L1
    results["L1"] = {
        "per_K": results["K"],
        "note": ("NELBO 逐步均值 = 每步在仍被 mask 位置上 CE 均值（bpc）的均值；末步 bpc = 第 K 步该均值；"
                 "Δ = bpc(mdiff 首定格) − bpc(ar_at_M)（同 M、同 64 block 配对）；"
                 "combined = 两 seed 逐 block Δ 对齐求均值后做 block 聚类配对 bootstrap；参考带 0.014 bpc"),
        "info_set_asymmetry": ("AR 见左侧真值（因果、teacher-forced）；mdiff 见全部非 M 真 token"
                               "（双向）+ 未定格位置 [mask]，且以自身 argmax 预测续填。信息集互不包含。"),
    }
    # L2：K 形状
    series = []
    for K in K_LIST:
        comb = results["K"][str(K)]["combined"]
        if comb:
            series.append({"K": K, "delta_mean": comb["delta_mean"],
                           "mdiff_M_bpc_mean": comb["mdiff_M_bpc_mean"],
                           "nelbo_step_mean_bpc_mean": comb["nelbo_step_mean_bpc_mean"],
                           "last_step_bpc_mean": comb["last_step_bpc_mean"]})
    deltas = [x["delta_mean"] for x in series]
    dK16_lt_dK1 = (deltas[-1] < deltas[0]) if len(deltas) >= 2 else None
    results["L2"] = {"K_series": series, "delta_K16_less_than_K1": dK16_lt_dK1,
                     "note": "Δ（bpc）随 K 的走向；bpc 越低越好 ⇒ 下降即改善。判断『提高 K 能否拉回带内』"}
    # L3：外部锚（需 AR 权重）
    if results.get("ar_retrained"):
        results["L3"] = _l3_rescore(results)
    else:
        results["L3"] = {"status": "gap",
                         "reason": "2671 的 AR checkpoint 未落盘 ⇒ 无法用冻结 AR 对生成序列 rescore；"
                                   "本轮以只读复用 2671 的 AR 读数（ar_full / ar_at_M）替代，L3 记为 gap"}
    # L4：代价轴
    nm_tot = {str(K): sum(sum(e["per_K"][str(K)]["n_m_blocks"])
                          for e in results.get("mdiff", {}).values()) for K in K_LIST}
    blocks_each = len(cfg["eval_offsets"]) * max(len(results.get("mdiff", {})), 1)
    results["L4"] = {
        "ar": cost_axis_ar(),
        "mdiff": {str(K): cost_axis_mdiff(K, nm_tot[str(K)], blocks_each) for K in K_LIST},
        "measured_n_m_total": nm_tot,
        "note": ("tokens-per-forward = |M|/K；每 block 顺序前向 = K（AR = T−1）；每 token 等效 FLOPs = K·L"
                 "（以 AR 每 token 的 L 为单位）。全部等效理论量，禁墙钟。"),
    }
    # L5：禁墙钟 + n≥2 + 确定性
    bad = wall_clock_fields({k: v for k, v in results.items()
                             if k not in ("_wall_clock_fields", "L5")})
    results["L5"] = {
        "wall_clock_fields": bad,
        "n_seeds": len(results.get("mdiff", {})),
        "determinism_rerun_K4_equal": all(e.get("determinism_rerun_K4_equal")
                                          for e in results.get("mdiff", {}).values()),
    }
    if bad:
        raise AssertionError(f"results.json 含墙钟字段：{bad}")
    results["_wall_clock_fields"] = bad
    # §4 否证条件
    results["falsification"] = _falsification(results)
    results["verdicts"] = {"main": _verdict(results)}
    _write_train_log(out, results)
    _write_model_md(results, out, steps, seeds)
    _write_analysis(results, out, steps, seeds)
    _dump(os.path.join(out, "results.json"), results)
    log(f"finalize DONE wall_clock_fields={bad} verdict={results['verdicts']['main'][:80]}")


def _l3_rescore(results: dict) -> dict:
    """L3：冻结 AR 对 mdiff 生成序列的 rescore bpc 差（需 ar_retrained 的模型权重）。

    说明：本轮 AR 权重在重训进程中用完即弃，未跨进程落盘；L3 因此在 `arretrain` 阶段若未与
    mdiff 结果同进程计算则记为 gap（见 analysis）。
    """
    return {"status": "gap", "reason": "AR 权重未跨进程持久化；L3 rescore 需与 mdiff K 扫描同进程计算"}


def _falsification(results: dict) -> dict:
    ref = M.load_2671()
    ar_full = {s: ref["ar"][s]["ar_full_bpc"] for s in ref.get("ar", {})}
    tri = []
    any_trigger = False
    for K in K_LIST:
        comb = results["K"][str(K)]["combined"]
        if not comb:
            continue
        c1 = comb["ci_excludes_zero"] and abs(comb["delta_mean"]) > REFERENCE_BAND
        tri.append({"K": K, "ci_excludes_zero": comb["ci_excludes_zero"],
                    "delta_mean": comb["delta_mean"], "exceeds_band": comb["exceeds_band"],
                    "cond_single": bool(c1)})
        any_trigger |= bool(c1)
    d1 = next((x["delta_mean"] for x in tri if x["K"] == 1), None)
    d16 = next((x["delta_mean"] for x in tri if x["K"] == 16), None)
    cond2 = (d1 is not None and d16 is not None and d16 >= d1)   # K=16 不见小于 K=1
    contradict = bool(any_trigger and cond2)
    all_ci_include_zero = all(not x["ci_excludes_zero"] for x in tri) if tri else None
    verdict = {}
    verdict["mask_diffusion_parallel_generation_fails"] = {
        "triggered": contradict,
        "detail": (f"任一 K 的 Δ CI 不含 0 且 |Δ|>0.014：{any_trigger}；"
                   f"Δ(K=16) 不见小于 Δ(K=1)：{cond2}（Δ1={d1}, Δ16={d16}）"),
    }
    verdict["indistinguishable_at_this_scale"] = {
        "triggered": bool(all_ci_include_zero),
        "detail": "全部 K 的配对 95% CI 含 0 ⇒ 本尺度不可分（不得写成『与 AR 无损』）",
    }
    verdict["K1_connects_to_2671_fill_bidir"] = {
        "triggered": None,
        "detail": "见 analysis 的 K=1 与 2671 fill_bidir_r25 连接点行",
    }
    return {"ar_full_bpc_2671": ar_full, "per_K": tri,
            "delta_K16_less_than_K1": (d16 < d1) if (d1 is not None and d16 is not None) else None,
            "conditions": verdict}


def _verdict(results: dict) -> str:
    f = results.get("falsification", {}).get("conditions", {})
    if f.get("mask_diffusion_parallel_generation_fails", {}).get("triggered"):
        return "掩码扩散并行生成在本尺度 / 本任务上不成立（§4 触发）"
    if f.get("indistinguishable_at_this_scale", {}).get("triggered"):
        return "本尺度下不可分（全部 K 的 Δ 配对 CI 含 0）；不得写成「与 AR 无损」"
    return "逐 K 的 Δ 与带/CI 见 L1；未触发 §4 的整体否证"


# ---- 落盘 ---------------------------------------------------------------------
def _write_train_log(out: str, results: dict) -> None:
    with open(os.path.join(out, "train_log.jsonl"), "w", encoding="utf-8") as f:
        for s, e in sorted(results.get("mdiff", {}).items()):
            f.write(json.dumps({"arm": "mdiff", "seed": int(s), "r_eval": R_EVAL,
                                "trace": e.get("loss_trace", [])}, ensure_ascii=False) + "\n")
        for s, e in sorted(results.get("ar_retrained", {}).items()):
            f.write(json.dumps({"arm": "ar_retrained", "seed": int(s), "r_eval": R_EVAL,
                                "trace": e.get("loss_trace", [])}, ensure_ascii=False) + "\n")


def _write_model_md(results: dict, out: str, steps: int, seeds: list[int]) -> None:
    cfg = results["config"]
    L = [
        "# MODEL · 2674-mdiffusion-13m",
        "",
        "本实验**只读复用** 2671-padfill-13m 的 harness `padfill.py`（不修改），在自己的 `mdiff.py` 里",
        "新增**吸收态掩码扩散训练目标**与 **K 步置信度解掩码采样器**。模型结构逐字段同 2671（唯一结构变量",
        "= 注意力可见性 `attn_mask`；本项用 `full`=双向）。",
        "",
        "## 结构（同 2671，契约 §2）",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 形态 | pre-LN decoder-only char LM |",
        f"| d_model | {cfg['structure']['d_model']} |",
        f"| n_layers | {cfg['structure']['n_layers']} |",
        f"| n_heads | {cfg['structure']['n_heads']} |",
        f"| mlp_mult | {cfg['structure']['mlp_mult']} |",
        f"| seq_len | {cfg['seq_len']} |",
        f"| 读出 | 共享 tied：`logits = LN_F(h) @ E^T` |",
        f"| 注意力 | SDPA；`attn_mask=full`（本项唯一结构变量） |",
        f"| 实测参数量 | {cfg['structure']['params']:,} |",
        "",
        "## 只读依赖 / sha256",
        "",
        f"- `padfill.py`（只读 import，来源 = `2671-padfill-13m`）：`{cfg['padfill_sha256']}`",
        f"- `mdiff.py`（本项库）：`{source_sha(os.path.join(HERE, 'mdiff.py'))}`",
        f"- `run.py`（本项驱动）：`{source_sha(os.path.join(HERE, 'run.py'))}`",
        "",
        "## AR 参照臂来源",
        "",
        f"- `{cfg['ar_weights']['source']}`",
        f"- 策略：{cfg['ar_weights']['policy']}",
    ]
    for s, e in sorted(results.get("ar_retrained", {}).items()):
        L.append(f"  - 重训 AR seed {s}：`ar_full_bpc={e['ar_full_bpc']:.5f}`、"
                 f"`ar_at_M_bpc={e['ar_at_M_bpc']:.5f}`（对照 2671 落盘读数）")
    L += [
        "",
        "## 数据",
        "",
        "`data/enwik8_full.npz`（仓库既有）：train 90M / val 10M 字节、`vocab_size=206`",
        f"（205 字节值 + 1 个 `[mask]` 槽，`[mask]` = 末槽 = id {P.MASK_TOKEN}）。",
        "",
        "权重不入库（`.gitignore` 覆盖 `*.pt`）：`mdiff` 臂由 `torch.manual_seed(seed)` + 固定数据/mask",
        "生成器重建；同 seed 采样器重跑逐值相同（见 `analysis.md` L5）。",
        "",
    ]
    with open(os.path.join(out, "MODEL.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def _fmt_ci(comb) -> str:
    if not comb:
        return "—"
    return (f"Δ={comb['delta_mean']:+.5f} CI[{comb['delta_lo']:+.5f},{comb['delta_hi']:+.5f}] "
            f"{'不含0' if comb['ci_excludes_zero'] else '含0'}"
            f"{' ·越带' if comb['exceeds_band'] else ' ·带内'}")


def _write_analysis(results: dict, out: str, steps: int, seeds: list[int]) -> None:
    cfg = results["config"]
    ref = M.load_2671()
    L = []
    L.append("# analysis — 2674-mdiffusion-13m（掩码扩散 K 步去噪最小判决）")
    L.append("")
    L.append("## 人话摘要")
    L.append("")
    if steps < 6000:
        L.append(f"**Nk 快照口径：本实验 `S = {steps}`（与 2671 matched），与仓库既有 6000 步档不可比；"
                 "只读臂间增量与排序**（`harness/SCHEMA.md` §20.3）。")
        L.append("")
    ar_full = [ref["ar"][s]["ar_full_bpc"] for s in ref.get("ar", {})]
    L.append("我们在**同一 harness、同一 token 预算、同一结构**下问一件事：把「逐 token 因果生成」换成"
             "「K 步双向去噪、一次并行填一批位置」，质量代价有多大、随 K 怎么走。参照是同一个 2671 的"
             "全因果 AR 臂（只读复用其读数）。")
    L.append("")
    L.append(f"参照系：AR 全位置 teacher-forced bpc = {np.mean(ar_full):.4f}"
             f"（越低越好；种子 {', '.join(f'{x:.4f}' for x in ar_full)}）。mdiff 臂在同位置集合 M 上：")
    for K in K_LIST:
        comb = results["K"][str(K)]["combined"]
        if comb:
            L.append(f"- **K={K}**：mdiff bpc={comb['mdiff_M_bpc_mean']:.4f}、"
                     f"NELBO 逐步均值={comb['nelbo_step_mean_bpc_mean']:.4f}、末步 bpc={comb['last_step_bpc_mean']:.4f}；"
                     f"{_fmt_ci(comb)}")
    L.append("")
    l2 = results["L2"]
    if l2.get("delta_K16_less_than_K1") is not None:
        L.append(f"- **K 的形状（L2）**：Δ(K=16) "
                 f"{'< ' if l2['delta_K16_less_than_K1'] else '≥ '}Δ(K=1) ⇒ "
                 f"{'提高 K 有改善' if l2['delta_K16_less_than_K1'] else '提高 K 不能把质量拉回（甚至更差）'}；"
                 f"Δ 随 K：" + ", ".join(f"K={x['K']}:{x['delta_mean']:+.4f}"
                                        for x in l2["K_series"]) + "。")
    L.append(f"- **K=1 与 2671 `fill_bidir_r25` 的连接点**：见下「K=1 连接点」行（单独成行）。")
    L.append(f"- **禁墙钟（L5）**：`results.json` 墙钟语义字段 = {results.get('_wall_clock_fields')}（0 命中）；"
             f"n_seeds={results['L5']['n_seeds']}。")
    L.append("")
    L.append("> 信息集不对称（必须同时读）：AR 臂对 M 位置的预测只依赖**左侧真值**（因果 teacher-forced）；"
             "mdiff 臂对 M 位置的预测依赖**全部非 M 真 token（含右侧）** + 未定格位置的 `[mask]` 占位，"
             "且已定格位置是**它自己的 argmax 预测**而非真值。两者信息集互不包含。")
    L.append("")
    L.append("## K=1 连接点（6.3.1 ↔ 6.3.2）")
    L.append("")
    k1 = results["K"]["1"]["combined"]
    fb = ref["arms"]["fill_bidir_r25"]
    fb_delta = [(_fb_delta(ref, s)) for s in sorted(fb)]
    L.append(f"- K=1 采样器退化为「一次前向填出全部 M」；其逐位置 bpc 与 2671 `fill_bidir_r25` 同 M 口径。"
             f"本项 mdiff K=1 合并 Δ（相对 ar_at_M）={_fmt_ci(k1)}。")
    L.append(f"- 2671 `fill_bidir_r25` 同 M 的合并 Δ（相对 ar_at_M）= "
             f"{np.mean([d[0] for d in fb_delta]):+.5f}（逐 seed "
             f"{', '.join(f'{d[0]:+.5f}' for d in fb_delta)}）——作为**跨项一致性**对照：mdiff（变 r 训练）"
             f"在 K=1 上的质量应落在 fill_bidir（固定 r=0.25 训练）附近。")
    L.append("")
    L.append("## 口径与配置")
    L.append("")
    L.append(f"- matched-step：steps={cfg['steps']}（从 2671 `config.steps` 读，不得改）、batch={cfg['batch_size']}、"
             f"T={cfg['seq_len']}、lr={cfg['lr']}、{cfg['optimizer']}；模型种子={cfg['seeds']}")
    L.append(f"- mdiff 目标：每 step 抽 r~U(0,1)，逐位置独立以概率 r 置 `[mask]`，损失只在被 mask 位置；"
             f"mask 调度 seed={cfg['mask_seed']}；数据顺序 seed={cfg['data_seed']}")
    L.append(f"- 评测位置集 M：r_eval={cfg['r_eval']}（=2671 fill_bidir_r25 同 M）、min_pos={cfg['min_mask_pos']}、"
             f"eval_mask_seed={cfg['eval_mask_seed']}、{cfg['eval_blocks']} 个固定 offset × T={cfg['seq_len']}")
    L.append(f"- K 扫描 = {cfg['k_list']}；主判 bootstrap：block 聚类配对、B={cfg['bootstrap']['B']}、"
             f"seed={cfg['bootstrap']['seed']}、percentile 95%；参考带={cfg['reference_band_bpc']} bpc")
    L.append(f"- 结构：d_model={cfg['structure']['d_model']}、n_layers={cfg['structure']['n_layers']}、"
             f"params={cfg['structure']['params']:,}；`padfill.py` sha256={cfg['padfill_sha256'][:16]}…")
    L.append("")
    L.append("## 数字表：逐 K × seed")
    L.append("")
    L.append("| K | 种子 | mdiff bpc | NELBO 逐步均值 | 末步 bpc | ar_at_M bpc | Δbpc | 95% CI | 越 0.014 带 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for K in K_LIST:
        for s, v in sorted(results["K"][str(K)]["per_seed"].items()):
            L.append(f"| {K} | {s} | {v['mdiff_M_bpc']:.5f} | {v['nelbo_step_mean_bpc']:.5f} | "
                     f"{v['last_step_bpc']:.5f} | — | {v['delta_mean']:+.5f} | "
                     f"[{v['delta_lo']:+.5f},{v['delta_hi']:+.5f}] | {'是' if v['exceeds_band'] else '否'} |")
        comb = results["K"][str(K)]["combined"]
        if comb:
            L.append(f"| {K} | **合并** | {comb['mdiff_M_bpc_mean']:.5f} | "
                     f"{comb['nelbo_step_mean_bpc_mean']:.5f} | {comb['last_step_bpc_mean']:.5f} | "
                     f"{np.mean([ref['ar'][s]['ar_at_M_bpc']['0.25'] for s in ref['ar']]):.5f} | "
                     f"**{comb['delta_mean']:+.5f}** | **[{comb['delta_lo']:+.5f},{comb['delta_hi']:+.5f}]** | "
                     f"{'是' if comb['exceeds_band'] else '否'} |")
    L.append("")
    L.append("| AR 参照（只读 2671） | 种子 | ar_full bpc | ar_at_M_25 bpc |")
    L.append("|---|---|---|---|")
    for s, e in sorted(ref.get("ar", {}).items()):
        L.append(f"| `ar` | {s} | {e['ar_full_bpc']:.5f} | {e['ar_at_M_bpc']['0.25']:.5f} |")
    L.append("")
    L.append("## 代价轴（L4，等效理论量，禁墙钟）")
    L.append("")
    L.append("| 臂 | tokens-per-forward | 每 block 顺序前向 | 每 token 等效 FLOPs |")
    L.append("|---|---|---|---|")
    L.append(f"| `ar` | 1 | {results['L4']['ar']['sequential_forwards_per_block']:.0f}（T−1） | "
             f"{results['L4']['ar']['equivalent_flops_per_token']:.0f}（L） |")
    for K in K_LIST:
        c = results["L4"]["mdiff"][str(K)]
        L.append(f"| `mdiff` K={K} | |M|/K = {c['tokens_per_forward']:.2f} | {c['sequential_forwards_per_block']:.0f} | "
                 f"{c['equivalent_flops_per_token']:.0f}（K·L） |")
    L.append("")
    L.append("## L0–L5 逐条")
    L.append("")
    L.append(f"- **L0 实现保真**：M 逐 block n_m 与 2671 ar 落盘对齐差异清单 = "
             f"{results['L0']['m_align_2671_diff_list']}（空 = 同一位置集）；`padfill.py` sha256 见 MODEL.md。")
    L.append(f"- **L1 主判**：逐 K 的 mdiff bpc / NELBO 逐步均值 / 末步 bpc / Δ 配对 CI 齐全（见上表）；"
             f"AR 信息集不对称已显式声明。")
    L.append(f"- **L2 K 形状**：Δ(K=16) < Δ(K=1) = {l2.get('delta_K16_less_than_K1')}")
    L.append(f"- **L3 外部锚**：{results['L3'].get('status', '有')} —— {results['L3'].get('reason','')}")
    L.append(f"- **L4 代价轴**：报告齐全（等效理论量）。")
    L.append(f"- **L5**：禁墙钟字段={results['_wall_clock_fields']}（0 命中）；n_seeds="
             f"{results['L5']['n_seeds']}；确定性（同模型同输入 K=4 重跑逐值相同）= "
             f"{results['L5']['determinism_rerun_K4_equal']}")
    L.append("")
    L.append("## §4 否证条件逐条")
    L.append("")
    for key, val in results["falsification"]["conditions"].items():
        if val.get("triggered") is None:
            L.append(f"- `{key}`：{val['detail']}")
        else:
            L.append(f"- `{key}`：{'**触发**' if val['triggered'] else '未触发'} —— {val['detail']}")
    L.append("")
    L.append(f"**判读**：{results['verdicts']['main']}")
    L.append("")
    L.append("## 边界与未测（gaps）")
    L.append("")
    for s in cfg.get("gaps", []):
        L.append(f"- {s}")
    L.append(f"- 结论只适用于本设定（{steps} 步 / char enwik8 / 交付双向结构 / r_eval={cfg['r_eval']} / K∈{cfg['k_list']}），"
             f"不外推；S < 6000 ⇒ 快照口径。")
    L.append("- 模型权重不入库；同 seed 由初始化 + 固定数据/mask 生成器重建。")
    L.append("")
    with open(os.path.join(out, "analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def _fb_delta(ref: dict, seed_s: str) -> tuple:
    """2671 fill_bidir_r25 与 ar_at_M 的合并 Δ（逐 seed 用其 delta_mean）。"""
    e = ref["arms"]["fill_bidir_r25"][seed_s]
    return (e["delta_mean"], e["delta_lo"], e["delta_hi"])


# ---- 入口 ---------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "mdiff", "arretrain", "finalize"])
    ap.add_argument("--out", default=".")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--ar", default="reuse", choices=["reuse", "retrain"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    log = Tee(os.path.join(out, "run_log.txt"))
    steps = a.steps if a.steps is not None else _steps_from_2671()
    device = a.device if torch.cuda.is_available() else "cpu"

    train, val, vocab = P.load_data()
    offsets = P.eval_offsets(len(val))
    ref = M.load_2671()
    log(f"stage={a.stage} out={out} seeds={a.seeds} steps={steps} device={device} ar={a.ar}")

    results = _load(os.path.join(out, "results.json"))
    if not results:
        results = {
            "experiment": "2674-mdiffusion-13m",
            "contract": "README.md (pre-registered)",
            "stage_note": "WP 6.3.1-WP1（掩码扩散最小判决）；matched-step 与 2671 一致",
            "config": {
                "steps": steps, "batch_size": P.BATCH_SIZE, "seq_len": P.SEQ_LEN, "lr": P.LR,
                "optimizer": "Adam(no weight decay)",
                "data_seed": P.DATA_SEED, "mask_seed": P.MASK_SEED,
                "eval_mask_seed": P.EVAL_MASK_SEED,
                "bootstrap": {"B": P.BOOT_B, "seed": P.BOOT_SEED, "cluster": "eval_block",
                              "ci": "percentile95"},
                "seeds": list(a.seeds), "r_eval": R_EVAL, "k_list": K_LIST,
                "eval_blocks": len(offsets), "eval_offsets": [int(o) for o in offsets],
                "min_mask_pos": P.MIN_MASK_POS, "reference_band_bpc": REFERENCE_BAND,
                "arms": {"ar": {"attn_mask": "causal", "role": "reference (read-only from 2671)"},
                         "mdiff": {"attn_mask": "full", "objective": "absorbing_mask_diffusion"}},
                "structure": {"d_model": P.D_MODEL, "n_layers": P.N_LAYERS, "n_heads": P.N_HEADS,
                              "mlp_mult": P.MLP_MULT,
                              "params": P.count_params(P.PadFillLM(attn_mask="full"))},
                "padfill_sha256": source_sha(os.path.join(M.PADFILL_DIR, "padfill.py")),
                "ar_weights": {
                    "source": "2671-padfill-13m AR checkpoint 未落盘（无 ckpt）；只读复用其 results.json 读数",
                    "policy": ("reuse 2671 读数" if a.ar == "reuse"
                               else "2671 无 ckpt ⇒ 本目录逐字段相同 config 重训 AR（声明于 analysis.md）"),
                },
                "gaps": [],
            },
            "ar_reference": ref.get("ar", {}),
            "mdiff": {},
        }
    # 记录 AR 参照读数（只读）
    for s, e in ref.get("ar", {}).items():
        results["ar_reference"] = results.get("ar_reference", {})
        results["ar_reference"][s] = {"ar_full_bpc": e["ar_full_bpc"],
                                      "ar_at_M_bpc": e["ar_at_M_bpc"],
                                      "n_m_total": e["n_m_total"], "source": "2671/results.json"}

    try:
        if a.stage in ("all", "mdiff"):
            run_mdiff(results, out, val, train, a.seeds, steps, device, log)
        if a.stage in ("all",) and a.ar == "retrain":
            run_ar_retrain(results, out, val, train, a.seeds, steps, device, log)
        if a.stage in ("arretrain",):
            run_ar_retrain(results, out, val, train, a.seeds, steps, device, log)
        if a.stage in ("all", "finalize"):
            # gaps：未测试项（契约 §4 明列）
            results["config"]["gaps"] = _default_gaps(results, steps)
            finalize(results, out, steps, a.seeds, device, log)
    except GPUUnavailable as e:
        log(f"GPUUnavailable: {e} —— 登记后退出，不杀他人进程")
        _dump(os.path.join(out, "results.json"), results)
        return 2
    log("DONE")
    return 0


def _default_gaps(results: dict, steps: int) -> list[str]:
    gaps = []
    if len(results.get("mdiff", {})) < 2:
        gaps.append("n_seeds < 2：部分种子未跑完（见 results.json::mdiff）")
    if not results.get("ar_retrained"):
        gaps.append("L3 外部锚（冻结 AR rescore）未做：2671 AR checkpoint 未落盘，本轮只读复用其读数；"
                    "最小成本 = 同进程重训 AR ×2 seed（≈2×1500 step）后对 mdiff 生成序列 rescore")
    gaps.append("离散扩散**均匀核**臂（uniform transition）未做：契约 §4 明列未做项")
    gaps.append("**噪声调度扫描**（cosine / linear / per-token t）未做：契约 §4 明列未做项")
    gaps.append("**与 6.3.3 同框**未做：契约 §4 明列未做项")
    if steps < 6000:
        gaps.append(f"S={steps} 快照口径，与仓库既有 6000 步档不可比（非能力结论，只读增量/排序）")
    return gaps


if __name__ == "__main__":
    raise SystemExit(main())
