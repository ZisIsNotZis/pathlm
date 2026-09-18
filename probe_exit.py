"""Exit-probe measurements (ticket 18): the exit lever calibrated via a
probe-style depth head (trunk stop-grad), retry coexistence, and the
two-sided frontier (exit points BELOW 1.0 flop + plain + spec points).

Sections:
 1. depth curve     per-depth node-0+1 CE on clean AND corrupted streams —
                    the exit quality axis (depth d costs d/n_layers flop).
 2. exit sweep      decode with exit_threshold sweep: mean depth (cost),
                    wall-clock tok/s; optionally composed with prob0-gated
                    retry (retry_threshold) — exit and retry compose in the
                    plain (k=0) decode path (exit x spec stays engine-blocked).
 3. retry coex      gate_probe-gated quality per tau on the EX1 model vs the
                    INT2_r3 reference (ticket 15): did the probe head keep the
                    retry lever alive where dense-exit (ALLOC) killed it?

    python3 probe_exit.py --config configs/EX1.json \
        --ckpt .scratch/18-exit-probe/evidence/EX1/model.pt \
        --ref .scratch/15-slider-rung3/evidence/slider_INT2r3.json \
        --out .scratch/18-exit-probe/evidence/exit_probe.json
"""

import argparse, json, os, random, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_full, batch
from pathlm.model import PathLM
from pathlm.decode import decode
from pathlm.eval import eval_pc
from pathlm.slider import gate_probe, gated_quality, reliability

LN2 = 0.6931471805599453
EXITS = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def depth_bpc(model, eval_arr, vocab_size, corrupted: bool,
              n_batches=24, batch_size=32, seed=1):
    """Per-depth node-1 bpc (the exit quality axis). Uses the model's own
    dense-exit hook (collect_depth_logits) so depths 1..n report their
    readout quality; clean and corrupted curves share the unit (bpc)."""
    T = model.mcfg.seq_len
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(1234)  # eval corruption convention
    sums: dict[int, float] = {}
    with eval_pc(model, w_dense_exit=1.0, collect_depth_logits=True,
                 p_retry=0.0, p_token_retry=0.0,
                 corrupt_wrong=(model.pcap.corrupt_wrong if corrupted else 0.0),
                 corrupt_mask=(model.pcap.corrupt_mask if corrupted else 0.0)):
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, gen)
            x = x.to(model.embed.weight.device)
            tgt = x[:, 1:]
            _, aux = model(x, [sample_path(model.pcap, rng, model.mcfg.n_layers)], x)
            for d, lg in enumerate(aux["depth_logits"], start=1):
                ce = F.cross_entropy(lg[:, :tgt.shape[1]].reshape(-1, vocab_size),
                                     tgt.reshape(-1), reduction="sum")
                sums[d] = sums.get(d, 0.0) + ce.item()
    n_tok = n_batches * (T - 1) * batch_size
    return {d: round(v / n_tok / LN2, 4) for d, v in sorted(sums.items())}


@torch.no_grad()
def exit_sweep(model, pcap, prompts, n_new=256, taus_retry=(None, 0.9)):
    """decode() with exit sweep x retry gate: mean depth (flop cost =
    mean_depth/n_layers), wall-clock, retries."""
    rows = []
    for tau_r in taus_retry:
        for te in [None] + EXITS:
            tok_s_list, depth_list, fwd_list, flop_list = [], [], [], []
            retr = 0
            for prompt in prompts:
                log = []
                _sync(); t0 = time.perf_counter()
                gen, st = decode(model, prompt, n_new, pcap,
                                 exit_threshold=te, retry_threshold=tau_r,
                                 prob0_log=log)
                _sync(); dt = time.perf_counter() - t0
                tok_s_list.append(len(gen) / dt)
                depths = st.get("depths") or [model.mcfg.n_layers]
                depth_list.append(sum(depths) / len(depths))
                # flop accounting: exited steps cost their executed depth; a
                # fired retry round re-runs the FULL stack (Decoder.retry has
                # no exit) — count it as n_layers
                retr += sum(st["retries"])
                total_flop_units = sum(depths) + sum(st["retries"]) * model.mcfg.n_layers
                flop_list.append(total_flop_units / max(len(gen), 1) / model.mcfg.n_layers)
                fwd_list.append((len(gen) + sum(st["retries"])) / max(len(gen), 1))
            n = len(prompts)
            rows.append({
                "tau_retry": tau_r, "tau_exit": te,
                "tok_s": round(sum(tok_s_list) / n, 1),
                "mean_depth": round(sum(depth_list) / n, 2),
                "flop_per_tok": round(sum(flop_list) / n, 4),
                "forwards_per_tok": round(sum(fwd_list) / n, 4),
                "retries": retr,
            })
            print(f"  tau_retry={tau_r} tau_exit={te}  flop={rows[-1]['flop_per_tok']:.3f} "
                  f"(depth {rows[-1]['mean_depth']:.1f}/12)  tok/s={rows[-1]['tok_s']} "
                  f"fwd={rows[-1]['forwards_per_tok']:.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default=None, help="ticket-15 slider JSON (INT2_r3) for the retry-coexistence comparison")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-new", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    mcfg, pcap = ModelConfig(**cfg["model"]), PathConfig(**cfg["path"])
    _, ev, V = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    torch.manual_seed(args.seed)
    model = PathLM(mcfg, pcap, V).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    model.eval()
    assert pcap.w_dense_exit > 0, "this probe expects a dense-exit-supervised model"

    g = torch.Generator().manual_seed(7)
    prompts = [torch.randint(0, model.n_real_tokens, (64,), generator=g).cuda()
               for _ in range(2)]

    out = {"model": args.ckpt, "config": args.config}

    # ---- 1. depth curves (the exit quality axis) ----
    clean = depth_bpc(model, ev, V, corrupted=False, n_batches=24)
    corr = depth_bpc(model, ev, V, corrupted=True, n_batches=24)
    out["depth_curve_clean"] = clean
    out["depth_curve_corrupt"] = corr
    print(f"depth clean : {clean}", flush=True)
    print(f"depth corrupt: {corr}", flush=True)

    # ---- 2. exit sweep (composed with retry gate) ----
    print("== exit sweep ==", flush=True)
    out["exit_sweep"] = exit_sweep(model, pcap, prompts, n_new=args.n_new)

    # ---- 3. retry coexistence ----
    half = len(ev) // 2
    cal = gate_probe(model, ev[:half], V, n_batches=16, corrupted=True, seed=0)
    ql = gate_probe(model, ev[half:], V, n_batches=16, corrupted=True, seed=1)
    fire = {t: float((cal["prob0"] < t).float().mean()) for t in (0.5, 0.9, 0.98)}
    gq = gated_quality(ql, [0.5, 0.9, 0.98])
    gq = {("off" if k == "off" else str(k)): v for k, v in gq.items()}
    out["retry_coexistence"] = {
        "fire_curve_calib": {str(t): round(v, 5) for t, v in fire.items()},
        "gated_quality": {str(t): v for t, v in gq.items()},
        "reliability_prob0_vs_hit0": reliability(ql["prob0"], ql["hit0_r1"]),
    }
    off = gq["off"]["gated_bpc"]
    print("== retry coexistence (corrupt, EX1) ==", flush=True)
    for t in (0.5, 0.9, 0.98):
        print(f"  tau={t}: gated_bpc {gq[str(t)]['gated_bpc']:.4f} (off {off:.4f}, "
              f"delta {gq[str(t)]['gated_bpc'] - off:+.4f}, fire {gq[str(t)]['fire_rate']})",
              flush=True)
    if args.ref and os.path.exists(args.ref):
        ref = json.load(open(args.ref))
        rg = ref["calibration"]["corrupt"]["gated_quality_split"]
        out["retry_coexistence"]["ref_INT2r3"] = {
            str(t): {"gated_bpc": rg[t]["gated_bpc"], "fire_rate": rg[t]["fire_rate"]}
            for t in ("off", "0.9", "0.98")}
        print(f"  ref INT2_r3: off {rg['off']['gated_bpc']:.4f} "
              f"tau.9 {rg['0.9']['gated_bpc']:.4f} tau.98 {rg['0.98']['gated_bpc']:.4f}",
              flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
