"""Collapse-domain monitor probe (ticket 22b): text repetition rate as the
monitoring signal for the attractor-collapse blind spot of the path
disagreement js (ticket 22: fully collapsed windows read js=0.000 /
agree=1.000 — "perfectly healthy" — and injected corruption does not move
them; js monitors semantic disagreement, not collapse).

Sections:
 1. natural   raw enwik8 windows (CPU only): the natural repetition level
              of the data itself — any collapse threshold must clear this
              control (enwik8 repeats naturally; report, don't assume).
 2. windows   free-running greedy 400-token segments, 3 prompt kinds x N.
              Reuses ticket-22 ``run_generation`` VERBATIM: real/corrupt
              windows with pi < 6 are byte-identical to ticket 22 (same seed
              draws); the random-kind prompts are drawn AFTER the prefix
              draws, so they differ from ticket 22's n_prompts=6 run —
              fresh windows, verified reproducible run-to-run.
 3. monitor   per-window repetition metrics on the GENERATED region
              (distinct-2 ratio / unique-token ratio / max token share),
              window-level 2-path re-score js (generated rows, ticket-22
              seed formula) and decode-time mean prob0. Labels: PRIMARY =
              manual semantic labels via --labels-json (recorded as
              label_source=manual; threshold labels are a function of the
              metric itself -> circular AUC, kept only as the pre-registered
              rule audit — greedy output sits wholly below the natural
              distinct-2 level, so that rule degenerates to all-collapse).
              Criterion 22b: repetition AUC (collapse vs healthy) >= 0.9
              AND js decoupled on the collapsed windows (their mean js
              below the healthy mean and at least half at/below the pooled
              median). A d2-axis threshold sweep documents the label
              counts (the repetition continuum has no natural gap).

    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 \
        probe_collapse_monitor.py --config .tmp/INT2.json \
        --ckpt .scratch/15-slider-rung3/evidence/INT2_r3/model.pt \
        --out .scratch/22-gen-quality/evidence/collapse_monitor_INT2r3.json \
        --labels-json .scratch/22-gen-quality/evidence/collapse_labels_22b.json
"""

import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig
from pathlm.data import load_enwik8_full
from pathlm.gen_quality import (collapse_summary, discrimination_auc,
                                distinct_ngram_ratio, label_collapse,
                                max_token_share, rescore_disagreement,
                                unique_ratio)
from pathlm.model import PathLM
from probe_gen_quality import KINDS, run_generation

THR_DISTINCT2 = 0.5
THR_UNIQUE = 0.15
SWEEP = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def natural_baseline(ev, n_windows: int, win_len: int, seed: int) -> dict:
    """Repetition statistics of RAW enwik8 windows — the natural level the
    collapse thresholds must sit below (CPU only)."""
    g = torch.Generator().manual_seed(seed)
    hi = len(ev) - win_len - 1
    rows = []
    for _ in range(n_windows):
        i = int(torch.randint(0, hi, (1,), generator=g))
        w = torch.from_numpy(ev[i:i + win_len].astype("int64"))
        rows.append((distinct_ngram_ratio(w), unique_ratio(w),
                     max_token_share(w)))
    d2 = torch.tensor([r[0] for r in rows])
    uq = torch.tensor([r[1] for r in rows])
    ms = torch.tensor([r[2] for r in rows])
    return {"n_windows": n_windows, "window_len": win_len,
            "distinct2_mean": round(float(d2.mean()), 4),
            "distinct2_min": round(float(d2.min()), 4),
            "unique_mean": round(float(uq.mean()), 4),
            "unique_min": round(float(uq.min()), 4),
            "max_share_mean": round(float(ms.mean()), 4),
            "max_share_max": round(float(ms.max()), 4)}


@torch.no_grad()
def monitor_rows(model, segs: list[dict], seed: int, thr_d2: float = THR_DISTINCT2,
                 thr_uq: float = THR_UNIQUE) -> list[dict]:
    """Per-window repetition metrics on the generated region + the ticket-22
    2-path re-score js over the generated rows + decode-time mean prob0."""
    rows = []
    for s in segs:
        gen = torch.tensor(s["gen"])
        rescore_seed = seed + 31 + KINDS.index(s["kind"]) * 97 + s["pi"]
        r = rescore_disagreement(model, s["window"].unsqueeze(0), n_paths=2,
                                 seed=rescore_seed)
        lo = len(s["delivered"]) - 1          # row predicting gen[0]
        hi_r = lo + len(s["gen"])             # .. row predicting gen[-1]
        d2 = distinct_ngram_ratio(gen)
        uq = unique_ratio(gen)
        rows.append({
            "kind": s["kind"], "pi": s["pi"], "n_gen": len(s["gen"]),
            "distinct2": round(d2, 4), "unique": round(uq, 4),
            "max_share": round(max_token_share(gen), 4),
            "threshold_label": label_collapse(d2, uq, thr_d2, thr_uq),
            "window_js": round(float(r["js"][0, lo:hi_r].mean()), 5),
            "window_agree": round(float(r["agree"][0, lo:hi_r].mean()), 4),
            "mean_prob0_decode": s["mean_prob0_decode"],
        })
        print(f"  [{s['kind']}#{s['pi']}] distinct2={d2:.3f} unique={uq:.3f} "
              f"share={rows[-1]['max_share']:.3f} js={rows[-1]['window_js']:.5f} "
              f"thr_label={rows[-1]['threshold_label']}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--n-new", type=int, default=400)
    ap.add_argument("--n-baseline", type=int, default=64)
    ap.add_argument("--thr-distinct2", type=float, default=THR_DISTINCT2)
    ap.add_argument("--thr-unique", type=float, default=THR_UNIQUE)
    ap.add_argument("--labels-json", default=None,
                    help="manual semantic labels: JSON map kind#pi -> "
                         "(collapse 0/1, class str); when given, the PRIMARY "
                         "criterion uses them (label_source=manual) — "
                         "threshold labels are a function of the metric "
                         "itself (circular AUC)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    mcfg, pcap = ModelConfig(**cfg["model"]), PathConfig(**cfg["path"])
    _, ev, V = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    torch.manual_seed(args.seed)
    model = PathLM(mcfg, pcap, V).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    model.eval()
    print(f"loaded {args.ckpt} (V={V}, seq_len={mcfg.seq_len})", flush=True)

    print("== 1. natural enwik8 repetition baseline (CPU) ==", flush=True)
    nat = natural_baseline(ev, args.n_baseline, args.n_new, args.seed + 3)
    print(f"  natural: distinct2={nat['distinct2_mean']} "
          f"(min {nat['distinct2_min']}) unique={nat['unique_mean']} "
          f"max_share={nat['max_share_mean']} (max {nat['max_share_max']})",
          flush=True)

    print(f"== 2. free generation ({3 * args.n_prompts} windows x "
          f"{args.n_new} tok) ==", flush=True)
    t0 = time.perf_counter()
    segs = run_generation(model, ev, args.n_prompts, args.n_new, args.seed)

    print("== 3. repetition monitor + js decoupling ==", flush=True)
    rows = monitor_rows(model, segs, args.seed, args.thr_distinct2,
                        args.thr_unique)
    manual = None
    if args.labels_json:
        manual = json.load(open(args.labels_json))["windows"]
        for r in rows:
            key = f"{r['kind']}#{r['pi']}"
            r["manual_class"] = manual[key]["class"]
            r["manual_label"] = int(manual[key]["collapse"])
    d2 = [r["distinct2"] for r in rows]
    uq = [r["unique"] for r in rows]
    js = [r["window_js"] for r in rows]
    summary = collapse_summary(d2, uq, js, args.thr_distinct2,
                               args.thr_unique,
                               labels=([r["manual_label"] for r in rows]
                                       if manual else None))
    try:
        threshold_summary = collapse_summary(d2, uq, js, args.thr_distinct2,
                                             args.thr_unique)
    except ValueError as e:                  # expected failure mode: the
        threshold_summary = {"error": str(e),  # fixed thresholds degenerate to
                             "label_source": "threshold",  # one class under
                             "note": "fixed thresholds label ALL windows "  # the global repetition bias
                                     "collapse (greedy output sits wholly "
                                     "below the natural distinct-2 level)"}
    sweep = {}
    for t in SWEEP:                     # d2-axis audit of the threshold rule
        lab = [1 if d < t else 0 for d in d2]   # (unique arm disabled: ALL
        n_c = sum(lab)                  # generated windows sit below it)
        entry = {"n_collapse": n_c, "n_healthy": len(lab) - n_c,
                 "auc_rep_distinct2": None}
        if 0 < n_c < len(lab):
            entry["auc_rep_distinct2"] = discrimination_auc(
                torch.tensor([-d for d in d2]),
                torch.tensor(lab, dtype=torch.float32))
        sweep[str(t)] = entry
    kind_means = {}
    for kind in KINDS:
        rs = [r for r in rows if r["kind"] == kind]
        kind_means[kind] = {
            "n": len(rs),
            "n_collapse_manual": (sum(r["manual_label"] for r in rs)
                                  if manual else None),
            "distinct2_mean": round(sum(r["distinct2"] for r in rs) / len(rs), 4),
            "js_mean": round(sum(r["window_js"] for r in rs) / len(rs), 5)}

    out = {"model": args.ckpt, "config": args.config, "ticket": "22b",
           "n_prompts": args.n_prompts, "n_new": args.n_new,
           "label_rule": "distinct2 < thr OR unique < thr (generated region)",
           "thresholds": {"distinct2": args.thr_distinct2,
                          "unique": args.thr_unique},
           "natural_baseline": nat, "kind_means": kind_means,
           "threshold_rule_d2_axis_sweep": sweep,
           "criterion_22b": summary,
           "threshold_rule_summary": threshold_summary,
           "threshold_rule_caveat": "threshold labels are a function of the "
                                    "repetition metric itself -> circular AUC; "
                                    "kept for the pre-registered rule audit only",
           "per_window": rows,
           "generation_seconds": round(time.perf_counter() - t0, 1)}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"criterion 22b: pass={summary['pass']} "
          f"auc_rep={summary['auc_rep_distinct2']} "
          f"auc_js_alarm={summary['auc_js_alarm_direction']} "
          f"js collapse/healthy={summary['mean_js_collapse']}/"
          f"{summary['mean_js_healthy']} "
          f"n_collapse={summary['n_collapse']}/{summary['n_windows']} "
          f"sweep={sweep}", flush=True)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
