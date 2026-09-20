"""Free-generation quality probe (ticket 22): teacher-forced multi-path
disagreement as the deployable output-quality signal where mean prob0 fails
(ticket 15: free-running corrupted generation reads prob0 REVERSED,
0.976 corrupted vs 0.954 clean — repetition attractors self-endorse).

Sections:
 1. tf_calib   eval stream, clean + corrupted-input profiles: 3-path
               re-score (corruption seed replayed per path) -> per-position
               js / agreement / prob0 vs realized next-token hit ->
               error-rate curves for BOTH signals, AUC, rank corr, and the
               joint prob0 x js bucketing. Criterion A: disagreement bins'
               error monotonicity >= prob0 bins' (or complementary: the
               joint bucketing stratifies within prob0's confident half).
 2. gen_free   free-running greedy 400-token segments from 3 prompt kinds
               (real eval prefix / random bytes / stream-corrupted 15%),
               then a 2-path teacher-forced RE-SCORE of each emitted
               segment (read-only) -> window-level disagreement vs the
               decode-time mean prob0. Criterion B: window disagreement
               follows the known degradation order real < corrupt < random
               while prob0's same-window ordering is reported alongside
               (known reversed/confused).
 3. injection  the SAME generated windows re-scored under injected
               wrong-token corruption 0/7.5/15% (same paths per window,
               only the dose varies) -> dose-response. Criterion C: window
               disagreement rises monotonically with the injected rate
               (signal tracks true degradation, not a prompt confound).

Re-score windows carry the tokens AS DELIVERED (prompt as read by the
channel + emitted ids); the re-score itself reads the window cleanly
(corruption off) — the channel-dose axis is isolated in section 3.

    python3 probe_gen_quality.py --config .tmp/INT2.json \
        --ckpt .scratch/15-slider-rung3/evidence/INT2_r3/model.pt \
        --out .scratch/22-gen-quality/evidence/gen_quality_INT2r3.json
"""

import argparse, json, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig
from pathlm.data import batch, load_enwik8_full
from pathlm.decode import decode, decode_spec
from pathlm.gen_quality import (discrimination_auc, joint_high_conf_bucketing,
                                monotone_frac, rank_corr, rescore_disagreement,
                                signal_bins)
from pathlm.model import PathLM
from pathlm.slider import proxy_summary
from slider import stream_corruptor  # ticket-15 channel semantics, verbatim

RATES = (0.0, 0.075, 0.15)
KINDS = ("real", "random", "corrupt")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ----------------------------------------------------- 1. TF calibration (A)

def tf_calibration(model, ev, n_batches: int, batch_size: int, n_paths: int,
                   seed: int) -> dict:
    """Per-position disagreement vs prob0 as error predictors on
    teacher-forced eval windows (clean and corrupted-input profiles)."""
    T = model.mcfg.seq_len
    out: dict = {}
    for profile in ("clean", "corrupt"):
        cw, cm = ((0.075, 0.075) if profile == "corrupt" else (0.0, 0.0))
        js, agree, p0, hit = [], [], [], []
        t0 = time.perf_counter()
        for bi in range(n_batches):
            g = torch.Generator().manual_seed(seed * 1000 + bi)
            x, _ = batch(ev, batch_size, T, g)
            r = rescore_disagreement(model, x, n_paths=n_paths,
                                     corrupt_wrong=cw, corrupt_mask=cm,
                                     seed=seed * 1000 + bi)
            js.append(r["js"].reshape(-1))
            agree.append(r["agree"].reshape(-1))
            p0.append(r["prob0"].reshape(-1))
            hit.append(r["hit"].reshape(-1))
        js = torch.cat(js)
        agree = torch.cat(agree)
        p0 = torch.cat(p0)
        miss = 1 - torch.cat(hit)
        score_p0 = -p0  # detecting direction: LOW prob0 marks misses
        js_bins = signal_bins(js, miss)
        p0_bins = signal_bins(score_p0, miss)
        res = {
            "n_positions": int(js.numel()),
            "miss_rate": round(float(miss.mean()), 4),
            "mean_js_bits": round(float(js.mean()), 5),
            "mean_agree": round(float(agree.mean()), 4),
            "mean_prob0": round(float(p0.mean()), 4),
            "js_bins": js_bins,
            "prob0_bins_ascending_in_detecting_score": p0_bins,
            "auc_js": discrimination_auc(js, miss),
            "auc_prob0_flipped": discrimination_auc(score_p0, miss),
            "rank_corr_js": round(rank_corr(js, miss), 4),
            "rank_corr_prob0_flipped": round(rank_corr(score_p0, miss), 4),
            "mono_js": monotone_frac(js_bins),
            "mono_prob0": monotone_frac(p0_bins),
            "joint_high_conf": joint_high_conf_bucketing(js, p0, miss),
        }
        out[profile] = res
        print(f"  [{profile}] n={res['n_positions']} miss={res['miss_rate']} "
              f"js={res['mean_js_bits']} p0={res['mean_prob0']} "
              f"auc js/p0={res['auc_js']}/{res['auc_prob0_flipped']} "
              f"mono js/p0={res['mono_js']}/{res['mono_prob0']} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
    c = out["corrupt"]
    joint_ok = bool(c["joint_high_conf"]["monotone"]
                    and c["joint_high_conf"]["spread"] >= 0.02)
    out["criterion_A"] = {
        "primary_profile": "corrupt",
        "decision_rule": "mono_js >= mono_prob0 (adjacent-bin strict-rise "
                         "fraction, equal-count bins) OR joint prob0 x js "
                         "bucketing monotone with spread >= 0.02 inside the "
                         "confident half",
        "mono_js": c["mono_js"], "mono_prob0": c["mono_prob0"],
        "auc_js": c["auc_js"], "auc_prob0_flipped": c["auc_prob0_flipped"],
        "rank_corr_js": c["rank_corr_js"],
        "rank_corr_prob0_flipped": c["rank_corr_prob0_flipped"],
        "joint_high_conf": c["joint_high_conf"], "joint_ok": joint_ok,
        "pass": bool(c["mono_js"] >= c["mono_prob0"] or joint_ok),
    }
    print(f"  criterion A: pass={out['criterion_A']['pass']} "
          f"(mono js {c['mono_js']} vs p0 {c['mono_prob0']}; "
          f"auc js {c['auc_js']} vs p0 {c['auc_prob0_flipped']}; "
          f"joint {c['joint_high_conf']['err_by_js_tercile']} "
          f"spread {c['joint_high_conf']['spread']})", flush=True)
    return out


# --------------------------------------------- 2. free generation + rescore (B)

@torch.no_grad()
def run_generation(model, ev, n_prompts: int, n_new: int, seed: int) -> list[dict]:
    """Free-running greedy segments for the 3 prompt kinds. `feed` is what
    decode consumes; `delivered` is what the world ends up with (the channel
    output for the corrupt kind) and becomes the re-score window context."""
    n_real = model.n_real_tokens
    mask_id = model.mask_token
    g = torch.Generator().manual_seed(seed)
    SL = model.mcfg.seq_len
    hi = len(ev) - SL - 1
    idx = torch.randint(0, hi, (n_prompts,), generator=g)
    real = [torch.from_numpy(ev[i:i + 64].astype("int64")) for i in idx]
    segs = []
    for pi in range(n_prompts):
        rnd = torch.randint(0, n_real, (64,), generator=g)
        fn = stream_corruptor(n_real, mask_id, seed + 100 + pi)
        delivered = torch.tensor([fn(int(t)) for t in real[pi]])  # prefill replay
        segs.append({"kind": "real", "pi": pi, "feed": real[pi],
                     "delivered": real[pi].clone(), "fn": None})
        segs.append({"kind": "random", "pi": pi, "feed": rnd,
                     "delivered": rnd.clone(), "fn": None})
        segs.append({"kind": "corrupt", "pi": pi, "feed": real[pi],
                     "delivered": delivered, "fn": fn})
    for s in segs:
        log: list = []
        _sync()
        t0 = time.perf_counter()
        gen, _st = decode(model, s["feed"].cuda(), n_new, model.pcap,
                          prob0_log=log, input_fn=s["fn"])
        _sync()
        s["gen"] = gen
        s["gen_seconds"] = round(time.perf_counter() - t0, 1)
        s["mean_prob0_decode"] = proxy_summary(log)["proxy_mean_prob0"]
        s["window"] = torch.cat([s["delivered"], torch.tensor(gen)])
        print(f"  [{s['kind']}#{s['pi']}] {len(gen)} tok {s['gen_seconds']}s "
              f"prob0={s['mean_prob0_decode']}", flush=True)
    return segs


@torch.no_grad()
def gen_free_rows(model, segs: list[dict], n_paths: int, seed: int) -> dict:
    """2-path teacher-forced re-score of each emitted window (generated
    rows only) -> window-level disagreement + the decode-time prob0 view."""
    rows = []
    for s in segs:
        r = rescore_disagreement(model, s["window"].unsqueeze(0), n_paths=n_paths,
                                 seed=seed + 31 + KINDS.index(s["kind"]) * 97 + s["pi"])
        lo = len(s["delivered"]) - 1                 # row predicting gen[0]
        hi_r = lo + len(s["gen"])                    # .. row predicting gen[-1]
        row = {"kind": s["kind"], "pi": s["pi"],
               "window_mean_js": round(float(r["js"][0, lo:hi_r].mean()), 5),
               "window_mean_agree": round(float(r["agree"][0, lo:hi_r].mean()), 4),
               "mean_prob0_decode": s["mean_prob0_decode"]}
        rows.append(row)
        print(f"  [{s['kind']}#{s['pi']}] js={row['window_mean_js']} "
              f"agree={row['window_mean_agree']} prob0={row['mean_prob0_decode']}",
              flush=True)
    kinds = {}
    for kind in KINDS:
        rs = [r for r in rows if r["kind"] == kind]
        kinds[kind] = {
            "n": len(rs),
            "mean_window_js": round(sum(r["window_mean_js"] for r in rs) / len(rs), 5),
            "mean_window_agree": round(sum(r["window_mean_agree"] for r in rs) / len(rs), 4),
            "mean_prob0_decode": round(sum(r["mean_prob0_decode"] for r in rs) / len(rs), 4),
            "per_window": rs,
        }
    j = {k: kinds[k]["mean_window_js"] for k in KINDS}
    p0 = {k: kinds[k]["mean_prob0_decode"] for k in KINDS}
    return {"kinds": kinds,
            "js_order": j, "prob0_order": p0,
            "criterion_B": {
                "decision_rule": "mean window js strictly ordered real < corrupt < random",
                "js_order_ok": bool(j["real"] < j["corrupt"] < j["random"]),
                "pass": bool(j["real"] < j["corrupt"] < j["random"])},
            }


# ---------------------------------- 2b. ticket-15 reversal replica (B context)

@torch.no_grad()
def ticket15_replication(model, n_prompts: int, seed: int) -> dict:
    """Ticket-15 reversal replica, exact protocol (proxy.degradation_demo):
    random-byte prompts, k=1 SPEC decode with retry gate tau=0.9, 192 tokens,
    corrupt = streaming 15% channel on every fed token, proxy = mean PRE-retry
    prob0 of the window. Ticket 15 measured ONE prompt: clean 0.954 (fire
    9.6%) vs corrupt 0.9757 (fire 5.8%) — prob0 REVERSED. Here: 6 prompts for
    stability + the 2-path re-score disagreement on the same windows — does
    js separate the arms that prob0 confuses?"""
    n_real, mask_id = model.n_real_tokens, model.mask_token
    g = torch.Generator().manual_seed(seed + 555)
    rows = []
    for pi in range(n_prompts):
        rnd = torch.randint(0, n_real, (64,), generator=g)
        for tag, fn in (("clean", None),
                        ("corrupt", stream_corruptor(n_real, mask_id,
                                                     seed + 700 + pi))):
            delivered = rnd.clone()
            if fn is not None:
                delivered = torch.tensor([fn(int(t)) for t in rnd])
            log: list = []
            gen, st = decode_spec(model, rnd.cuda(), 192, model.pcap,
                                  max_drafts=1, retry_threshold=0.9,
                                  prob0_log=log, input_fn=fn)
            proxy = proxy_summary(log)
            window = torch.cat([delivered, torch.tensor(gen)])
            r = rescore_disagreement(model, window.unsqueeze(0), n_paths=2,
                                     seed=seed + 900 + pi * 2 + int(tag == "corrupt"))
            lo, hi_r = len(delivered) - 1, len(delivered) + len(gen) - 1
            rows.append({"tag": tag, "pi": pi,
                         "mean_prob0": proxy["proxy_mean_prob0"],
                         "fire_fraction": proxy["fire_fraction"],
                         "window_mean_js": round(float(r["js"][0, lo:hi_r].mean()), 5),
                         "window_mean_agree": round(float(r["agree"][0, lo:hi_r].mean()), 4)})
        print(f"  [repl#{pi}] clean p0={rows[-2]['mean_prob0']} "
              f"js={rows[-2]['window_mean_js']} | corrupt "
              f"p0={rows[-1]['mean_prob0']} js={rows[-1]['window_mean_js']}",
              flush=True)
    out = {}
    for tag in ("clean", "corrupt"):
        rs = [r for r in rows if r["tag"] == tag]
        out[tag] = {"n": len(rs),
                    "mean_prob0": round(sum(r["mean_prob0"] for r in rs) / len(rs), 4),
                    "fire_fraction": round(sum(r["fire_fraction"] for r in rs) / len(rs), 4),
                    "mean_window_js": round(sum(r["window_mean_js"] for r in rs) / len(rs), 5),
                    "per_window": rs}
    out["prob0_reversed"] = bool(out["corrupt"]["mean_prob0"] > out["clean"]["mean_prob0"])
    out["js_separates"] = bool(out["corrupt"]["mean_window_js"] > out["clean"]["mean_window_js"])
    out["ticket15_reference"] = {"n_prompts": 1, "clean_prob0": 0.954,
                                 "corrupt_prob0": 0.9757,
                                 "source": "slider_INT2r3.json proxy.degradation_demo"}
    return out


# ------------------------------------------------- 3. injection control (C)

@torch.no_grad()
def injection_rows(model, segs: list[dict], n_paths: int, seed: int) -> dict:
    """Same windows, same paths (per-window seed fixed across rates, the
    same seed as the section-2 re-score so rate 0 reproduces that js
    exactly), only the injected wrong-token dose varies: 0 / 7.5 / 15%."""
    rows = []
    for s in segs:
        lo = len(s["delivered"]) - 1
        hi_r = lo + len(s["gen"])
        rec = {"kind": s["kind"], "pi": s["pi"], "js_by_rate": {}}
        for rate in RATES:
            r = rescore_disagreement(model, s["window"].unsqueeze(0),
                                     n_paths=n_paths, corrupt_wrong=rate,
                                     corrupt_mask=0.0,
                                     seed=seed + 31 + KINDS.index(s["kind"]) * 97 + s["pi"])
            rec["js_by_rate"][str(rate)] = round(float(r["js"][0, lo:hi_r].mean()), 5)
        vals = [rec["js_by_rate"][str(t)] for t in RATES]
        rec["monotone"] = bool(vals[0] < vals[1] < vals[2])
        rows.append(rec)
        print(f"  [{s['kind']}#{s['pi']}] js@0/.075/.15 = {vals} "
              f"monotone={rec['monotone']}", flush=True)
    pooled = {str(t): round(sum(r["js_by_rate"][str(t)] for r in rows) / len(rows), 5)
              for t in RATES}
    kind_means = {kind: {str(t): round(sum(r["js_by_rate"][str(t)]
                                             for r in rows if r["kind"] == kind)
                                       / max(1, sum(1 for r in rows if r["kind"] == kind)), 5)
                         for t in RATES}
                  for kind in KINDS}
    mono_frac_windows = round(sum(r["monotone"] for r in rows) / len(rows), 4)
    ordered = all(pooled[str(a)] < pooled[str(b)] for a, b in zip(RATES, RATES[1:]))
    return {"per_window": rows, "pooled_mean_js": pooled,
            "kind_mean_js": kind_means,
            "monotone_window_frac": mono_frac_windows,
            "criterion_C": {
                "decision_rule": "pooled mean window js strictly increasing "
                                 "in the injected rate 0 < 7.5% < 15%",
                "pooled_order_ok": bool(ordered),
                "pass": bool(ordered)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--n-new", type=int, default=400)
    ap.add_argument("--n-batches", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--n-paths-tf", type=int, default=3)
    ap.add_argument("--n-paths-rescore", type=int, default=2)
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

    out = {"model": args.ckpt, "config": args.config,
           "n_paths_tf": args.n_paths_tf,
           "n_paths_rescore": args.n_paths_rescore,
           "n_prompts": args.n_prompts, "n_new": args.n_new,
           "injection_rates": list(RATES)}

    print("== 1. tf calibration ==", flush=True)
    out["tf_calib"] = tf_calibration(model, ev, args.n_batches, args.batch_size,
                                     args.n_paths_tf, args.seed)

    print("== 2. free generation ==", flush=True)
    segs = run_generation(model, ev, args.n_prompts, args.n_new, args.seed)
    out["gen_free"] = gen_free_rows(model, segs, args.n_paths_rescore, args.seed)
    print(f"  criterion B: pass={out['gen_free']['criterion_B']['pass']} "
          f"js={out['gen_free']['js_order']} "
          f"prob0={out['gen_free']['prob0_order']}", flush=True)

    print("== 2b. ticket-15 replica (random prompts, k=1 tau=0.9, 192 tok) ==", flush=True)
    out["ticket15_replication"] = ticket15_replication(model, args.n_prompts,
                                                       args.seed)
    tr = out["ticket15_replication"]
    print(f"  replica: prob0_reversed={tr['prob0_reversed']} "
          f"js_separates={tr['js_separates']} "
          f"p0 clean/corrupt={tr['clean']['mean_prob0']}/{tr['corrupt']['mean_prob0']} "
          f"js clean/corrupt={tr['clean']['mean_window_js']}/{tr['corrupt']['mean_window_js']}",
          flush=True)

    print("== 3. injection control ==", flush=True)
    out["injection_control"] = injection_rows(model, segs, args.n_paths_rescore,
                                              args.seed)
    ic = out["injection_control"]
    print(f"  criterion C: pass={ic['criterion_C']['pass']} "
          f"pooled={ic['pooled_mean_js']} "
          f"monotone_frac={ic['monotone_window_frac']}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
