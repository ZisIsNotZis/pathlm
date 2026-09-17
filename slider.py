"""Slider end-to-end probe (ticket 15, Rung 3): calibration -> frontier ->
budget solve -> validated allocation decode -> online-quality proxy.

Implements docs/mental_model.md §3 on a real checkpoint:

 1. calibrate   gate_probe on the CALIBRATION half of the eval stream
                (clean + corrupted input profiles) -> fire curve P(prob0<tau),
                teacher-forced spec-accept estimate a2, reliability.
 2. frontier    per profile: same probe on the QUALITY half -> gated quality
                per tau; decode bench on held-out prompts per (k, tau)
                -> wall-clock tok/s + measured forwards/token.
 3. solve       budgets in TWO cost currencies -> (k, tau):
                "forwards" = forward count per emitted token (mental_model §1:
                budget = forwards x depth; spec amortizes below 1),
                "flop" = equivalent width-1 forward units (FLOP-normalized;
                spec floor 1.0). Validation decodes on FRESH prompts compare
                predicted vs measured cost in each currency.
 4. proxy       online quality: effective mean prob0 (round-2 on fired
                positions) vs realized accuracy; generation-time degradation
                detection (clean vs corrupted prompt, no ground truth).

Cost currencies: BOTH are predicted from the calibration (fire rate, a2) and
validated against the decode bench; wall-clock tok/s is reported alongside,
never mixed in.

    python3 slider.py --config .tmp/INT2.json \
        --ckpt .scratch/15-slider-rung3/evidence/INT2_r3/model.pt \
        --out .scratch/15-slider-rung3/evidence/slider_INT2r3.json
"""

import argparse, json, os, random, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig
from pathlm.data import load_enwik8_full
from pathlm.model import PathLM
from pathlm.decode import decode, decode_spec
from pathlm.slider import (Frontier, FrontierPoint, fire_curve, forwards_cost,
                           gate_probe, gated_quality, pearson, proxy_summary,
                           reliability, spec_cost)

TAUS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98]
BUDGETS = {"forwards": [0.5, 0.6, 0.75, 1.0],     # mental_model §1 currency
           "flop": [1.0, 1.1, 1.25, 1.5]}         # FLOP-normalized view


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def corrupt_ids(ids: torch.Tensor, mask_rate: float, wrong_rate: float,
                n_real: int, mask_id: int, seed: int) -> torch.Tensor:
    """Stage-0 style input corruption on a 1-D id tensor (deployment view:
    corruption is applied by the environment, the model never knows).
    Works on CPU and returns a CPU tensor (the decoder reads ints only)."""
    g = torch.Generator().manual_seed(seed)
    ids = ids.detach().cpu().clone()
    m = torch.rand(ids.shape, generator=g) < mask_rate
    w = (torch.rand(ids.shape, generator=g) < wrong_rate) & ~m
    ids[m] = mask_id
    wr = torch.randint(0, n_real, ids.shape, generator=g)
    wr = (wr + (wr == ids).long()) % n_real          # no collision with orig
    ids[w] = wr[w]
    return ids


def stream_corruptor(n_real: int, mask_id: int, seed: int,
                     mask_rate: float = 0.075, wrong_rate: float = 0.075):
    """Environment-side noisy channel: corrupts every token the decoder reads
    (prompt + each generated token before it is fed back). Matches the
    stage-0 corruption the TF calibration probes see (iid 15%), so the
    corrupt-profile calibration and deployment see the same input law."""
    g = torch.Generator().manual_seed(seed)

    def fn(t: int) -> int:
        r = torch.rand(1, generator=g).item()
        if r < mask_rate:
            return mask_id
        if r < mask_rate + wrong_rate:
            w = torch.randint(0, n_real, (1,), generator=g).item()
            return (w + (w == t)) % n_real
        return t

    return fn


@torch.no_grad()
def bench(model: PathLM, pcap: PathConfig, prompts: list[torch.Tensor],
          n_new: int, k: int, tau: float | None, corrupted_input: bool,
          seed: int = 11) -> dict:
    """Decode bench over prompts (greedy, B=1). Returns throughput, measured
    forwards/token (the v1 cost currency, counted not timed), spec accept
    rate, retry fraction and the online proxy from prob0_log."""
    tok_s_list, fwd_list, flop_list, proxies = [], [], [], []
    retries_frac, a2_num, a2_den = 0.0, 0, 0
    n_tokens = 0
    for pi, prompt in enumerate(prompts):
        inp_fn = None
        if corrupted_input:
            inp_fn = stream_corruptor(model.n_real_tokens, model.mask_token,
                                      seed + 100 + pi)
        log: list = []
        _sync()
        t0 = time.perf_counter()
        if k >= 1 and model.pcap.n_mtp >= 2:
            gen, st = decode_spec(model, prompt, n_new, pcap, max_drafts=k,
                                  retry_threshold=tau, prob0_log=log,
                                  input_fn=inp_fn)
            fwd = st["n_forward"] + st["retries"]
            flop_units = st["n_forward"] * (1 + k) + st["retries"]
            a2_num += st["drafts_accepted"]
            a2_den += st["drafts_offered"]
        else:
            gen, st = decode(model, prompt, n_new, pcap,
                             retry_threshold=tau, prob0_log=log,
                             input_fn=inp_fn)
            fwd = len(gen) + sum(st["retries"])
            flop_units = fwd
        _sync()
        dt = time.perf_counter() - t0
        tok_s_list.append(len(gen) / dt)
        fwd_list.append(fwd / max(len(gen), 1))
        flop_list.append(flop_units / max(len(gen), 1))
        retries_frac += sum(e["fired"] for e in log) / max(len(log), 1)
        n_tokens += len(gen)
        proxies.append(proxy_summary(log))
    mean = lambda xs: sum(xs) / len(xs)
    def mean_or_none(key):
        vals = [p[key] for p in proxies if p[key] is not None]
        return round(mean(vals), 4) if vals else None
    row = {
        "tok_s": round(mean(tok_s_list), 1),
        "forwards_per_tok": round(mean(fwd_list), 4),
        "flop_units_per_tok": round(mean(flop_list), 4),
        "retry_frac_loops": round(retries_frac / len(prompts), 4),
        "n_tokens": n_tokens,
        "proxy": {key: mean_or_none(key) for key in
                  ("proxy_mean_prob0", "proxy_mean_prob0_post",
                   "fire_fraction")},
    }
    if a2_den:
        row["a2_measured"] = round(a2_num / a2_den, 4)
    return row


def build_frontier(model: PathLM, pcap: PathConfig, calib, qual, prompts,
                   n_new: int, profile: str, probe_batches: int) -> Frontier:
    """One profile's frontier: calibration -> predicted cost; quality probe ->
    quality axis; decode bench -> measured behaviour."""
    corr = profile == "corrupt"
    cal = gate_probe(model, calib, model.vocab_size, n_batches=probe_batches,
                     corrupted=corr, seed=0)
    ql = gate_probe(model, qual, model.vocab_size, n_batches=probe_batches,
                    corrupted=corr, seed=1)
    # oracle reference: training-time mixture re-entry for round 2 — the
    # decode engine currently produces only the overwrite transport
    # (Decoder.retry); this row quantifies that engine gap
    qo = gate_probe(model, qual, model.vocab_size, n_batches=probe_batches,
                    corrupted=corr, seed=1, mixture_round2=True)
    gq_oracle = gated_quality(qo, TAUS)
    fire_cal = fire_curve(cal["prob0"], TAUS)
    gq = gated_quality(ql, TAUS)
    gq_calib = gated_quality(cal, TAUS)   # split-consistency reference
    a2_pred = float(cal["hit_spec"].mean())
    rel = reliability(ql["prob0"], ql["hit_r1"])
    points = []
    for k in (0, 1):
        if k >= 1 and pcap.n_mtp < 2:
            continue
        for tau in [None] + TAUS:
            fire = 0.0 if tau is None else fire_cal[tau]
            a2 = 1.0 if k == 0 else a2_pred
            qrow = gq["off" if tau is None else tau]
            quality = {"gated_bpc": qrow["gated_bpc"], "gated_acc": qrow["gated_acc"]}
            if "gated_acc_corr" in qrow:
                quality["gated_acc_corr"] = qrow["gated_acc_corr"]
            m = bench(model, pcap, prompts, n_new, k, tau, corrupted_input=corr)
            points.append(FrontierPoint(
                k=k, tau=tau, fire_rate=fire, a2=a2,
                cost_forwards=forwards_cost(k, fire, a2),
                cost_flop=spec_cost(k, fire, a2),
                quality=quality, measured=m))
            print(f"  [{profile}] k={k} tau={tau}  fwd={points[-1].cost_forwards:.3f} "
                  f"flop={points[-1].cost_flop:.3f}  measured={m['forwards_per_tok']:.3f} "
                  f"fwd/tok  tok/s={m['tok_s']}  bpc={qrow['gated_bpc']:.4f}  "
                  f"acc={qrow['gated_acc']:.4f}", flush=True)
    return Frontier(points), {
        "profile": profile,
        "fire_curve_calib": {str(t): round(v, 5) for t, v in fire_cal.items()},
        "a2_pred_tf": round(a2_pred, 4),
        "gated_quality_split": {str(t): v for t, v in gq.items()},
        "gated_quality_calib": {str(t): v for t, v in gq_calib.items()},
        "oracle_gated_quality_mixture": {str(t): v for t, v in gq_oracle.items()},
        "reliability_prob0_vs_hit0": reliability(ql["prob0"], ql["hit0_r1"]),
        "reliability_prob0r2_vs_hit0_fired": (
            reliability(ql["prob0_r2"][ql["prob0"] < 0.98],
                        ql["hit0_r2"][ql["prob0"] < 0.98])
            if (ql["prob0"] < 0.98).any() else None),
        "bpc_split_gap_max": round(max(
            abs(gq[t]["gated_bpc"] - gq_calib[t]["gated_bpc"])
            for t in gq), 4),
        "fire_curve_split_gap_max": round(max(
            abs(gq[t]["fire_rate"] - gq_calib[t]["fire_rate"])
            for t in gq), 4),
        "reliability_r1": rel,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-new", type=int, default=256)
    ap.add_argument("--probe-batches", type=int, default=16)
    ap.add_argument("--bench-prompts", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    mcfg, pcap = ModelConfig(**cfg["model"]), PathConfig(**cfg["path"])
    _, ev, V = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    half = len(ev) // 2
    calib, qual = ev[:half], ev[half:]
    torch.manual_seed(args.seed)
    model = PathLM(mcfg, pcap, V).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    model.eval()
    print(f"model {args.ckpt}  V={V}  calib={len(calib)} qual={len(qual)}", flush=True)

    T = model.mcfg.seq_len
    g = torch.Generator().manual_seed(7)
    frontier_prompts = [torch.randint(0, model.n_real_tokens, (64,), generator=g).cuda()
                        for _ in range(args.bench_prompts)]
    g2 = torch.Generator().manual_seed(99)   # fresh prompts for validation
    val_prompts = [torch.randint(0, model.n_real_tokens, (64,), generator=g2).cuda()
                   for _ in range(args.bench_prompts)]

    out: dict = {"model": args.ckpt, "config": args.config,
                 "taus": TAUS, "budgets": BUDGETS,
                 "cost_currencies": {
                     "forwards": "v1 budget currency: forward count per emitted "
                                 "token (mental_model §1: budget = forwards x depth)",
                     "flop": "FLOP-normalized: equivalent width-1 forward units per "
                             "emitted token (width-(1+k) verify = 1+k units)"}}
    frontiers, meta = {}, {}
    for profile in ("clean", "corrupt"):
        print(f"== profile {profile} ==", flush=True)
        f, m = build_frontier(model, pcap, calib, qual, frontier_prompts,
                              args.n_new, profile, args.probe_batches)
        frontiers[profile], meta[profile] = f, m
    out["calibration"] = meta

    # ------------------------------------------------ budget solve + validate
    solves = []
    measured_field = {"forwards": "forwards_per_tok", "flop": "flop_units_per_tok"}
    for profile, f in frontiers.items():
        for currency, budgets in BUDGETS.items():
            for b in budgets:
                try:
                    p = f.solve_budget(b, key="gated_bpc", currency=currency)
                except ValueError as e:
                    solves.append({"profile": profile, "budget": b,
                                   "currency": currency, "infeasible": str(e)})
                    print(f"  [{profile}] {currency} b={b} infeasible", flush=True)
                    continue
                val = bench(model, pcap, val_prompts, args.n_new, p.k, p.tau,
                            corrupted_input=(profile == "corrupt"))
                meas = val[measured_field[currency]]
                pred = p.cost_forwards if currency == "forwards" else p.cost_flop
                rel_err = (meas - pred) / pred
                solves.append({
                    "profile": profile, "budget": b, "currency": currency,
                    "solved": {"k": p.k, "tau": p.tau,
                               "cost_forwards": round(p.cost_forwards, 4),
                               "cost_flop": round(p.cost_flop, 4),
                               "fire_rate": p.fire_rate, "a2": p.a2,
                               "quality": p.quality},
                    "validation": {**val, "cost_rel_err": round(rel_err, 4)}})
                print(f"  [{profile}] {currency} b={b} -> {p.label()}  "
                      f"predicted {pred:.3f}  measured {meas:.3f} "
                      f"(rel {rel_err:+.1%})", flush=True)
    out["budget_solve"] = solves

    # ------------------------------------------------------ quality solve
    qsolves = []
    for profile, f in frontiers.items():
        off_bpc = [p.quality["gated_bpc"] for p in f.points
                   if p.tau is None and p.k == 0][0]
        target = off_bpc - 0.005   # beat retry-off by a variance-scale margin
        try:
            p = f.solve_quality(target, key="gated_bpc")
            qsolves.append({"profile": profile, "target_gated_bpc": round(target, 4),
                            "solved": {"k": p.k, "tau": p.tau,
                                       "cost_forwards": round(p.cost_forwards, 4),
                                       "cost_flop": round(p.cost_flop, 4),
                                       "quality": p.quality}})
            print(f"  [{profile}] q<={target:.4f} -> {p.label()} "
                  f"fwd {p.cost_forwards:.3f} / flop {p.cost_flop:.3f}", flush=True)
        except ValueError as e:
            qsolves.append({"profile": profile,
                            "target_gated_bpc": round(target, 4),
                            "infeasible": str(e)})
            print(f"  [{profile}] q<={target:.4f} infeasible: {e}", flush=True)
    out["quality_solve"] = qsolves

    # ------------------------------------------------------- online proxy
    proxy_sec: dict = {}
    for profile in ("clean", "corrupt"):
        ql = gate_probe(model, qual, model.vocab_size,
                        n_batches=args.probe_batches, corrupted=(profile == "corrupt"),
                        seed=1)
        gq = gated_quality(ql, TAUS)
        xs, ys = [], []
        for t, row in gq.items():
            xs.append(row["proxy_eff_mean"])
            ys.append(row["gated_acc"])
        # effective proxy at the flagship tau=0.9: post-retry prob0 on fired
        # positions, pre-retry elsewhere -- the online observable after the
        # gate has acted, against realized gated next-token accuracy
        gate = ql["prob0"] < 0.9
        eff = torch.where(gate, ql["prob0_r2"], ql["prob0"])
        eff_hit = torch.where(gate, ql["hit_r2"], ql["hit_r1"])
        proxy_sec[profile] = {
            # the honest proxy claim: prob0 estimates P(current committed
            # token correct) -- validated against node-0 realized hit
            "reliability_prob0_vs_hit0": reliability(ql["prob0"], ql["hit0_r1"]),
            "reliability_eff_tau09_vs_hit0": reliability(
                torch.where(gate, ql["prob0_r2"], ql["prob0"]),
                torch.where(gate, ql["hit0_r2"], ql["hit0_r1"])),
            "gate_relevance_vs_nexttok": {
                "reliability_r1": reliability(ql["prob0"], ql["hit_r1"]),
                "reliability_eff_tau09": reliability(eff, eff_hit),
                "proxy_vs_acc_over_taus": {
                    "pearson_r": round(pearson(xs, ys), 4),
                    "points": [{"proxy": x, "acc": y} for x, y in zip(xs, ys)]}},
        }
    # generation-time degradation detection: clean vs stream-corrupted decode,
    # k=1 tau=0.9, same prompts -- the online proxy must flag the bad channel
    det = {}
    for profile in ("clean", "corrupt"):
        det[profile] = bench(model, pcap, frontier_prompts[:1], 192, 1, 0.9,
                             corrupted_input=(profile == "corrupt"))
    proxy_sec["degradation_demo"] = {
        "setup": "greedy decode 192 tokens, k=1 tau=0.9; corrupt = stream "
                 "corruption (every fed token, iid 15%); proxy = mean prob0 "
                 "of the window (no ground truth at decode time)",
        "clean": det["clean"]["proxy"], "corrupt": det["corrupt"]["proxy"],
        "proxy_gap": round(det["clean"]["proxy"]["proxy_mean_prob0"]
                           - det["corrupt"]["proxy"]["proxy_mean_prob0"], 4),
        "fire_fraction_gap": round(
            det["corrupt"]["proxy"]["fire_fraction"]
            - det["clean"]["proxy"]["fire_fraction"], 4)}
    out["proxy"] = proxy_sec

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
