"""轴 2 · 2.1.1 early exit · WP1-e2e：逐 token 退出决策的**序列级**实测（纯推理）。

口径（预登记见同目录 README.md）：
- 主判据 = `stream_bpc`：teacher-forcing、逐位置**在线**决策下的整段 bpc
  （每个位置按自己的退出深度读出；异构深度 KV 填充语义按因果顺序传播）；
- 算力账 = layer-passes/token 与跳过层占比（等效理论量）；
- 样本 = **与历史 `depth_curve` / `tf_acceptance` 完全同一个**窗口样本
  （`pathlm.data.batch` + `seed=2`、20 批 × 16、T=256）⇒ prescribed-k 臂可直接与
  历史 depth_curve[k] 对锚，检验本口径与既有判据同刻度；
- 统计 = 同 checkpoint 内比较是**配对**的（各臂同一批窗口），故报「逐窗口配对差 Δ 的
  SE」，另报绝对 bpc 的窗口间 SE 作描述；判据门槛仍用仓内跨 run 噪声带 0.014；
- 校准 = 每深度 top-1 概率 vs 两种标签（`tf` / `true`）的 ECE 与 risk–coverage；
- 诊断 = 自由生成（生产解码器 `step_auto`）与全深度贪心流的逐 token 一致率。

**不训练**（复用已入账带 sha256 的 checkpoint）；**CPU** 纯推理（§20：纯 CPU 不取锁）。
输出：results.json / run_log.txt（墙钟只落在 run_log.txt 的过程行，不进 results.json、
不进任何判据）。

用法：
    python3 run.py --out . [--smoke] [--device cpu]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import torch.nn.functional as F

from pathlm.data import load_enwik8_full, batch                     # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,           # noqa: E402
                             depth_curve, generate)
from pathlm.exit_policy import (Top1ProbPolicy, PrescribedDepths,   # noqa: E402
                                NeverExit, stream_bpc, risk_coverage,
                                generate_with_exit, ece_from_bins,
                                merge_bin_tables)

LN2 = 0.6931471805599453
CKPT = ".scratch/23-depth-ar/evidence/23c-H2-w0-12k/model.pt"
CKPT_SHA = "2b054ab7368757fe79cf36c32bacacb5e7aafe52f718a02f0558dbb3aa5446e6"
CKPT_CFG = dict(d_model=256, n_layers=8, n_heads=8, seq_len=256, mlp_mult=4)
DATA_RAW, DATA_CACHE = ".tmp/enwik8", "data/enwik8_full.npz"

T = 256
CANON = dict(n_batches=20, batch_size=16, seed=2)   # = depth_curve / tf_acceptance 的样本约定
CHUNK = 64                                          # 只影响内存与分块，不影响结果
THRESHOLDS = (0.3, 0.6, 0.9, 0.95, 0.99, 0.999)
NOISE_BAND = 0.014
# 历史锚点（同 checkpoint，`23c-H2-w0-12k/results.json`，同一 seed=2 样本）
HIST_DEPTH_CURVE = {0: 5.4191, 1: 2.3819, 2: 1.9988, 3: 1.8863, 4: 1.8628,
                    5: 1.8593, 6: 1.8585, 7: 1.8584, 8: 1.8582}
HIST_TF = {0: 0.057, 1: 0.7187, 2: 0.8597, 3: 0.9348, 4: 0.9694,
           5: 0.9871, 6: 0.9938, 7: 0.998}
GEN = dict(seed=5, batch_size=4, prompt_len=64, n_new=32, commit="soft")


def log(msg, fh=None):
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_windows(eval_arr, n_batches, batch_size, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([batch(eval_arr, batch_size, T, g)[0]
                      for _ in range(n_batches)], dim=0)


def build_arms(L, W):
    """臂表：全深度基线 + 固定前缀（对照/口径锚点）+ 逐 token 阈值扫描（主判据）。"""
    arms = [("never-exit", NeverExit(), "全深度基线（退化臂：永不过早退出）")]
    for k in (2, 3, 4, 5, 6, 7):
        arms.append((f"prescribed-{k}", PrescribedDepths(torch.full((W, T), k)),
                     f"固定前缀退 {k} 层（对照臂 / 口径自洽锚点）"))
    for tau in THRESHOLDS:
        arms.append((f"token-thr-{tau}", Top1ProbPolicy(tau, min_depth=1, max_depth=L),
                     f"逐 token top-1 概率 ≥ {tau} 即停"))
    arms.append(("token-thr-0.999-min4", Top1ProbPolicy(0.999, min_depth=4, max_depth=L),
                 "高阈值 + 硬下界 4 层（浅层地板臂）"))
    return arms


def full_argmax(model, X, chunk=CHUNK):
    """全深度 argmax（[N,T-1]），跨块预计算一次供各臂复用。"""
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            _, aux = model(X[i:i + chunk], exit_depths=None)
            outs.append(aux["depth_logits"][-1][:, :-1].argmax(-1))
    return torch.cat(outs, dim=0)


def full_ce_per_pos(model, X, chunk=CHUNK):
    """全深度（无退出）下的**逐位置** CE —— 「同位置配对比较」的参照面。"""
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            xc = X[i:i + chunk]
            _, aux = model(xc, exit_depths=None)
            lg = aux["depth_logits"][-1][:, :-1].float()
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                 xc[:, 1:].reshape(-1), reduction="none")
            outs.append(ce.view(lg.shape[0], lg.shape[1]))
    return torch.cat(outs, dim=0)


def eval_arm(model, X, policy, fa, L, chunk=CHUNK):
    """分块评测一条臂，返回合并后的指标（分块只影响内存，不影响结果）。"""
    n_pred = 0
    ce_sum = 0.0
    lp_sum = 0
    n_pos = 0
    acc_tf_sum = 0.0
    acc_true_sum = 0.0
    per_win: list[float] = []
    buckets = {k: [0, 0.0, 0.0, 0.0] for k in range(L + 1)}   # n, ce_sum, true_hits, tf_hits
    hist = [0] * (L + 1)
    depth_parts = []
    ce_parts = []
    for i in range(0, X.shape[0], chunk):
        xc = X[i:i + chunk]
        out = stream_bpc(model, xc, policy.slice_for(i, i + xc.shape[0]),
                         full_argmax=fa[i:i + chunk])
        n_posc = out["depths"].numel()
        npredc = n_posc - out["depths"].shape[0]
        n_pred += npredc
        ce_sum += out["bpc"] * npredc * LN2
        lp_sum += out["layer_passes_per_token"] * n_posc
        n_pos += n_posc
        acc_tf_sum += out["exit_acc_tf"] * npredc
        acc_true_sum += out["exit_acc_true"] * npredc
        per_win += out["bpc_per_row"].tolist()
        for r in out["by_exit_depth"]:
            b = buckets[r["depth"]]
            b[0] += r["n"]
            if r["n"]:
                b[1] += r["bpc"] * r["n"] * LN2
                b[2] += r["acc_true"] * r["n"]
                b[3] += r["acc_tf"] * r["n"]
        depth_parts.append(out["depths"].clone())
        ce_parts.append(out["ce_per_pos"].clone())
        for h in out["depths_hist"]:
            hist[h["depth"]] += h["n"]
    by_depth = [{"depth": k, "n": b[0],
                 "frac": (b[0] / n_pred if n_pred else 0.0),
                 "bpc": (round(b[1] / b[0] / LN2, 6) if b[0] else None),
                 "acc_true": (round(b[2] / b[0], 6) if b[0] else None),
                 "acc_tf": (round(b[3] / b[0], 6) if b[0] else None)}
                for k, b in buckets.items()]
    mean_lp = lp_sum / n_pos if n_pos else 0.0
    return {
        "bpc": ce_sum / n_pred / LN2,
        "bpc_per_window": per_win,
        "layer_passes_per_token": mean_lp,
        "skip_frac": 1.0 - mean_lp / L,
        "exit_acc_tf": acc_tf_sum / n_pred,
        "exit_acc_true": acc_true_sum / n_pred,
        "by_exit_depth": by_depth,
        "depths_hist": [{"depth": k, "n": c, "frac": c / n_pos if n_pos else 0.0}
                        for k, c in enumerate(hist)],
        "n_pred": n_pred, "n_pos": n_pos,
        "depths": torch.cat(depth_parts, dim=0) if depth_parts else None,
        "ce_per_pos": torch.cat(ce_parts, dim=0) if ce_parts else None,
    }


def se(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    logf = open(os.path.join(args.out, "run_log.txt"), "a")
    t_start = time.perf_counter()

    sha = sha256_file(args.ckpt)
    log(f"[ckpt] {args.ckpt} sha256={sha}", logf)
    assert sha == CKPT_SHA, f"checkpoint sha256 不符：{sha} != {CKPT_SHA}"

    _, ev, V = load_enwik8_full(DATA_RAW, DATA_CACHE)
    cb = dict(CANON)
    if args.smoke:
        cb["n_batches"] = 2
    X = canonical_windows(ev, cb["n_batches"], cb["batch_size"], cb["seed"])
    W = X.shape[0]
    log(f"[data] enwik8 val；样本 = batch(seed={cb['seed']}) × {cb['n_batches']} × "
        f"{cb['batch_size']}，T={T} ⇒ W={W} 窗口，预测位置={W * (T - 1)}", logf)

    model = DepthARModel(DepthARConfig(**CKPT_CFG), V)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    L = model.cfg.n_layers
    log(f"[model] params={n_params} cfg={CKPT_CFG} dense={model.cfg.dense} "
        f"fill_kv={model.cfg.fill_kv} proj_fill={model.cfg.proj_fill} "
        f"carry_gate={model.cfg.carry_gate} L={L}", logf)

    fa = full_argmax(model, X)
    base_ce = full_ce_per_pos(model, X)

    arms = build_arms(L, W)
    if args.smoke:
        arms = arms[:3]
    rows = []
    all_depths = {}
    all_ce = {}
    for name, pol, note in arms:
        t0 = time.perf_counter()
        r = eval_arm(model, X, pol, fa, L)
        dt = time.perf_counter() - t0
        all_depths[name] = r["depths"]
        all_ce[name] = r["ce_per_pos"]
        r.update({"arm": name, "note": note,
                  "bpc": round(r["bpc"], 6),
                  "bpc_se_over_windows": round(se(r["bpc_per_window"]), 6),
                  "layer_passes_per_token": round(r["layer_passes_per_token"], 6),
                  "skip_frac": round(r["skip_frac"], 6),
                  "exit_acc_tf": round(r["exit_acc_tf"], 6),
                  "exit_acc_true": round(r["exit_acc_true"], 6)})
        rows.append(r)
        log(f"[arm] {name}: bpc={r['bpc']:.4f} ±{r['bpc_se_over_windows']:.4f}(窗口SE) "
            f"layer-passes={r['layer_passes_per_token']:.3f} "
            f"acc_tf={r['exit_acc_tf']:.4f} acc_true={r['exit_acc_true']:.4f} "
            f"[{dt:.1f}s 过程指标，不进判据]", logf)

    # 配对统计（各臂同一批窗口）：Δ 的 SE 远小于绝对 bpc 的 SE
    base = rows[0]
    for r in rows:
        d = [a - b for a, b in zip(r["bpc_per_window"], base["bpc_per_window"])]
        r["delta_bpc_vs_full"] = round(r["bpc"] - base["bpc"], 6)
        r["delta_bpc_paired_se"] = round(se(d), 6)
        r["delta_bpc_in_bands"] = round((r["bpc"] - base["bpc"]) / NOISE_BAND, 3)
        r["delta_bpc_paired_significant"] = bool(
            abs(r["bpc"] - base["bpc"]) > 2 * max(se(d), 1e-12))
        r["bpc_per_window_rounded"] = [round(b, 6) for b in r["bpc_per_window"]]
        r.pop("depths", None)
        r.pop("ce_per_pos", None)

    # 固定前缀前沿（同算力比较的对手）：Δbpc 随 layer-passes 的折线插值
    # 折线端点 = prescribed-2..7 + 全深度基线（8 层，Δ=0）
    fxd = sorted([(r["layer_passes_per_token"], r["delta_bpc_vs_full"])
                  for r in rows
                  if r["arm"].startswith("prescribed-") or r["arm"] == "never-exit"])

    def frontier_delta(lp):
        if lp <= fxd[0][0] or lp >= fxd[-1][0]:
            return None
        for (x0, y0), (x1, y1) in zip(fxd, fxd[1:]):
            if x0 <= lp <= x1:
                return y0 + (lp - x0) / (x1 - x0) * (y1 - y0)
        return None

    frontier = []
    for r in rows:
        if not r["arm"].startswith("token-thr"):
            continue
        lp, d = r["layer_passes_per_token"], r["delta_bpc_vs_full"]
        fd_ = frontier_delta(lp)
        frontier.append({
            "arm": r["arm"], "layer_passes_per_token": round(lp, 6),
            "delta_bpc": d, "fixed_depth_frontier_delta": (round(fd_, 6) if fd_ is not None else None),
            "frontier_excess": (round(d - fd_, 6) if fd_ is not None else None),
            "extra_layers_vs_dominated_fixed_arm": (
                round(lp - min(x for x, y in fxd if y <= d), 6)
                if any(y <= d for x, y in fxd) else None),
            "dominated_by_fixed": bool(any(y <= d for x, y in fxd)),
        })
    log("[frontier] " + "; ".join(
        f"{r['arm']}: lp={r['layer_passes_per_token']:.3f} Δ={r['delta_bpc']:+.4f} "
        + (f"vs 固定前缀同算力 {r['fixed_depth_frontier_delta']:+.4f} ⇒ 超出 "
           f"{r['frontier_excess']:+.4f}") for r in frontier), logf)

    # 口径自洽锚点：本实现的 prescribed-k / never-exit 必须复现历史 depth_curve
    anchor = []
    for r in rows:
        if r["arm"] == "never-exit":
            k = L
        elif r["arm"].startswith("prescribed-"):
            k = int(r["arm"].split("-")[1])
        else:
            continue
        anchor.append({"arm": r["arm"], "depth": k,
                       "bpc_this_run": r["bpc"], "bpc_hist_depth_curve": HIST_DEPTH_CURVE[k],
                       "delta": round(r["bpc"] - HIST_DEPTH_CURVE[k], 6),
                       "acc_tf_this_run": r["exit_acc_tf"],
                       "acc_tf_hist": HIST_TF.get(k)})
    log("[anchor] " + "; ".join(
        f"{a['arm']}: Δbpc={a['delta']:+.4f} Δtf={a['acc_tf_this_run'] - a['acc_tf_hist']:+.4f}"
        if a["acc_tf_hist"] else f"{a['arm']}: Δbpc={a['delta']:+.4f}" for a in anchor), logf)

    # 校准（分块累积原表后合并；ECE 只依赖每箱 n 与两个均值 ⇒ 合并后等价）
    cal_tables_tf, cal_tables_true = [], []
    for i in range(0, W, CHUNK):
        from pathlm.exit_policy import exit_calibration
        c = exit_calibration(model, X[i:i + CHUNK])
        cal_tables_tf.append([r["bins_tf"] for r in c["rows"]])
        cal_tables_true.append([r["bins_true"] for r in c["rows"]])
    cal_rows = []
    for ki in range(len(cal_tables_tf[0])):
        bt = merge_bin_tables([t[ki] for t in cal_tables_tf], 10)
        btr = merge_bin_tables([t[ki] for t in cal_tables_true], 10)
        n = sum(b["n"] for b in bt)
        mp = sum(b["n"] * b["mean_prob"] for b in bt if b["n"]) / n
        atf = sum(b["n"] * b["acc"] for b in bt if b["n"]) / n
        atr = sum(b["n"] * b["acc"] for b in btr if b["n"]) / n
        cal_rows.append({"depth": ki, "n": n,
                         "mean_top1_prob": round(mp, 6),
                         "acc_tf": round(atf, 6), "acc_true": round(atr, 6),
                         "overconf_tf": round(mp - atf, 6),
                         "overconf_true": round(mp - atr, 6),
                         "ece_tf": round(ece_from_bins(bt), 6),
                         "ece_true": round(ece_from_bins(btr), 6)})
    log("[calib] " + "; ".join(f"k={r['depth']} p̄={r['mean_top1_prob']:.3f} "
                               f"tf={r['acc_tf']:.3f} true={r['acc_true']:.3f} "
                               f"ECE_tf={r['ece_tf']:.3f} ECE_true={r['ece_true']:.3f}"
                               for r in cal_rows), logf)

    # risk–coverage（分块合并：n_kept/n_total 直接相加）
    rc_parts = []
    for i in range(0, W, CHUNK):
        rc_parts.append(risk_coverage(model, X[i:i + CHUNK], THRESHOLDS))
    rc = []
    for j in range(len(rc_parts[0])):
        acc = {"depth": rc_parts[0][j]["depth"], "tau": rc_parts[0][j]["tau"],
               "n_kept": 0, "n_total": 0, "tf_sum": 0.0, "true_sum": 0.0,
               "dropped_sum": 0.0, "n_dropped": 0}
        ok = True
        for part in rc_parts:
            r = part[j]
            acc["n_kept"] += r["n_kept"]
            acc["n_total"] += r["n_total"]
            if r["acc_tf_kept"] is not None:
                acc["tf_sum"] += r["acc_tf_kept"] * r["n_kept"]
                acc["true_sum"] += r["acc_true_kept"] * r["n_kept"]
            nd = r["n_total"] - r["n_kept"]
            if r["acc_true_dropped"] is not None:
                acc["dropped_sum"] += r["acc_true_dropped"] * nd
                acc["n_dropped"] += nd
            if r["n_kept"] == 0:
                ok = False
        rc.append({"depth": acc["depth"], "tau": acc["tau"],
                   "coverage": round(acc["n_kept"] / max(acc["n_total"], 1), 6),
                   "n_kept": acc["n_kept"], "n_total": acc["n_total"],
                   "acc_tf_kept": (round(acc["tf_sum"] / acc["n_kept"], 6) if ok else None),
                   "acc_true_kept": (round(acc["true_sum"] / acc["n_kept"], 6) if ok else None),
                   "acc_true_dropped": (round(acc["dropped_sum"] / acc["n_dropped"], 6)
                                        if acc["n_dropped"] else None)})

    # ---- 归因诊断（都是**诊断臂**，不进主判据；都便宜、都同一样本）----
    # D1 深度分布置换对照：保留每臂的退出深度**边缘分布**、打散「内容↔深度」的相关
    #    ⇒ 若置换臂明显好于原臂，说明损伤来自「难位置恰好吃到更浅上下文 + 陈旧填充」，
    #    而不是深度分布本身。
    # D2 fill 关（no-fill 语义）：把「陈旧填充」这一项单独拆开（注意：该臂偏离训练
    #    语义，故一律标为语义迁移臂，不当主判据）。
    diag = []
    diagnosis_arms = [n for n in ("token-thr-0.6", "token-thr-0.9") if n in all_depths]
    if args.smoke:
        diagnosis_arms = []
    gperm = torch.Generator().manual_seed(1234)
    model_nf = None
    if diagnosis_arms:
        model_nf = DepthARModel(DepthARConfig(fill_kv=False, **CKPT_CFG), V)
        model_nf.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        model_nf.eval()
    for nm in diagnosis_arms:
        d0 = all_depths[nm]
        perm = torch.randperm(d0.numel(), generator=gperm)
        dsh = d0.reshape(-1)[perm].reshape(d0.shape)
        arm_perm = eval_arm(model, X, PrescribedDepths(dsh), fa, L)
        same_marginal = bool(torch.equal(torch.sort(d0.reshape(-1)).values,
                                         torch.sort(dsh.reshape(-1)).values))
        diag.append({"kind": "depth-shuffled", "base_arm": nm,
                     "arm": f"shuffled-{nm}", "same_depth_marginal": same_marginal,
                     "bpc": round(arm_perm["bpc"], 6),
                     "layer_passes_per_token": round(arm_perm["layer_passes_per_token"], 6),
                     "delta_bpc_vs_base_arm": round(arm_perm["bpc"] -                          [r for r in rows if r["arm"] == nm][0]["bpc"], 6),
                     "note": "同边缘分布、打散内容↔深度相关（控制臂，需知道未来 ⇒ 不可部署）"})
        log(f"[diag] {nm} 深度置换：bpc={arm_perm['bpc']:.4f} "
            f"（原臂 {[r for r in rows if r['arm'] == nm][0]['bpc']:.4f}）", logf)
        if model_nf is not None:
            arm_nf = eval_arm(model_nf, X, PrescribedDepths(d0), fa, L)
            diag.append({"kind": "no-fill", "base_arm": nm, "arm": f"nofill-{nm}",
                         "bpc": round(arm_nf["bpc"], 6),
                         "layer_passes_per_token": round(arm_nf["layer_passes_per_token"], 6),
                         "delta_bpc_vs_base_arm": round(arm_nf["bpc"] -                              [r for r in rows if r["arm"] == nm][0]["bpc"], 6),
                         "note": "no-fill 语义（偏离训练语义，语义迁移臂）"})
            log(f"[diag] {nm} + no-fill：bpc={arm_nf['bpc']:.4f}", logf)
    # D3 结构零假设：全同深度的 prescribed 臂在 fill / no-fill 下必须逐位相同
    #    （混合深度语义未触发 ⇒ 开关不该有影响）。活性判据（SCHEMA §10）。
    if model_nf is not None:
        d4 = torch.full((W, T), 4)
        a_fill = eval_arm(model, X, PrescribedDepths(d4), fa, L)
        a_nofill = eval_arm(model_nf, X, PrescribedDepths(d4), fa, L)
        diag.append({"kind": "structural-null", "arm": "prescribed-4×{fill,nofill}",
                     "bpc_fill": round(a_fill["bpc"], 6),
                     "bpc_nofill": round(a_nofill["bpc"], 6),
                     "delta": round(a_nofill["bpc"] - a_fill["bpc"], 6),
                     "note": "全同深度下填充语义不触发 ⇒ 两臂必须相同（开关只影响混合深度）"})
        log(f"[diag] 结构零假设 prescribed-4 fill vs no-fill: "
            f"{a_fill['bpc']:.4f} vs {a_nofill['bpc']:.4f}", logf)

    # D4 **同位置**配对分析（决定性的机制证据）：把每个退出深度桶里的位置拿出来，
    #    比较「该臂自己的 CE」与「全深度基线在**同一批位置**上的 CE」。
    #    若最深桶（跑到 L 层的位置）在基线里本来就好、在该臂里变差 ⇒ 早退损害的是
    #    **仍然深跑的相邻位置**（它们的上下文里少了两层以上被跳过位置的深层表征）。
    same_pos = []
    for nm in diagnosis_arms:
        d = all_depths[nm][:, :-1]
        for k in range(L + 1):
            m = d == k
            n = int(m.sum())
            if n == 0:
                continue
            same_pos.append({
                "arm": nm, "depth_bucket": k, "n": n,
                "bpc_this_arm": round(float(all_ce[nm][m].mean()) / LN2, 6),
                "bpc_full_depth_same_positions": round(float(base_ce[m].mean()) / LN2, 6),
                "delta_same_positions": round(float(
                    (all_ce[nm][m].mean() - base_ce[m].mean())) / LN2, 6),
            })
    log("[same-pos] " + "; ".join(
        f"{r['arm']}@d{r['depth_bucket']}: {r['bpc_this_arm']:.3f} vs 同位置全深度 "
        f"{r['bpc_full_depth_same_positions']:.3f} (Δ{r['delta_same_positions']:+.3f}, "
        f"n={r['n']})" for r in same_pos), logf)

    # 自由生成：与全深度贪心流的逐 token 一致率（诊断列，不是「必须不变」判据）
    gen_rows = []
    if not args.smoke:
        g = torch.Generator().manual_seed(GEN["seed"])
        prompt = batch(ev, GEN["batch_size"], GEN["prompt_len"], g)[0]
        full_streams = [generate(model, prompt[b], GEN["n_new"], commit=GEN["commit"],
                                 forced_depth=None)[0] for b in range(prompt.shape[0])]
        for name, pol, _ in arms:
            if not name.startswith("token-thr"):
                continue
            agree = tot = 0
            lps = []
            for b in range(prompt.shape[0]):
                gs, st = generate_with_exit(model, prompt[b], GEN["n_new"], pol,
                                            commit=GEN["commit"])
                agree += sum(int(a == c) for a, c in zip(gs, full_streams[b]))
                tot += len(gs)
                lps.append(st["layer_passes_per_token"])
            gen_rows.append({"arm": name, "agree_with_full_greedy": round(agree / tot, 6),
                             "n_tokens": tot,
                             "layer_passes_per_token": round(sum(lps) / len(lps), 6)})
            log(f"[gen] {name}: 一致率 {agree}/{tot} = {agree / tot:.4f} "
                f"layer-passes={sum(lps) / len(lps):.3f}", logf)

    res = {
        "experiment": "2631-pertoken-exit-seqlevel",
        "item": "2.1.1 early exit",
        "ticket": "61（轴 2 lead, Phase A · WP1-e2e）",
        "engine": "depth-ar（pathlm.depth_ar）+ pathlm.exit_policy",
        "ckpt": {"path": args.ckpt, "sha256": sha, "params": n_params,
                 "config": CKPT_CFG},
        "protocol": {
            "device": args.device,
            "dtype": "fp32（CPU 纯推理；不训练、不分片、不 autocast）",
            "sample": {**cb, "window_len": T, "windows": W,
                       "n_pred_positions": W * (T - 1),
                       "note": "与历史 depth_curve/tf_acceptance 同一 seed=2 样本约定 ⇒ 可对锚"},
            "dataset": "data/enwik8_full.npz val 尾段（byte 级，V=206）",
            "primary_metric": ("stream_bpc：teacher-forcing、逐位置**在线**退出决策下的整段 "
                               "bpc（bits/char）；位置 i 用其退出深度读出预测 t_{i+1}"),
            "cost_metric": "layer-passes/token 与跳过层占比（等效理论量）",
            "noise_band": NOISE_BAND,
            "stats": ["delta_bpc_vs_full：与全深度基线的配对差（各臂同一批窗口）",
                      "delta_bpc_paired_se：逐窗口配对差的 SE",
                      "bpc_se_over_windows：绝对 bpc 的窗口间 SE（描述列）",
                      "delta_bpc_in_bands：Δbpc / 跨 run 噪声带 0.014"],
            "wallclock_disclaimer": "墙钟/吞吐不进 results.json、不进任何判据（SCHEMA §7/§8）",
            "thresholds": list(THRESHOLDS),
            "calibration_labels": ["tf = argmax_k == argmax_L",
                                   "true = argmax_k == 真实下一字节"],
            "gen_agreement": GEN | {"note": "诊断列；early exit 本就会改流，只有退化臂须为 1.000"},
        },
        "fixed_depth_frontier": fxd,
        "frontier_comparison": frontier,
        "sanity_anchor": anchor,
        "arms": rows,
        "calibration": {"n_bins": 10, "rows": cal_rows},
        "risk_coverage": rc,
        "generation_agreement": gen_rows,
        "attribution_diagnostics": diag,
        "same_position_paired": same_pos,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    log(f"[done] {time.perf_counter() - t_start:.1f}s（过程指标，不进判据）；results.json 已写",
        logf)
    logf.close()


if __name__ == "__main__":
    main()
