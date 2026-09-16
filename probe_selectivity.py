"""Allocator selectivity probe (ticket 14, experiment 1).

The vision's core claim: compute should be allocated per position by confidence —
low-confidence positions gain more quality per FLOP from an extra pass than
high-confidence ones. If the gain is flat across confidence, uniform allocation
suffices and selective allocation has no value.

Measurement: on corrupted input, bucket positions by ROUND-0 prob0 (the
allocator's own signal, available before deciding to spend) into deciles, then
measure per-bucket node-1 next-token bpc/accuracy at each retry round. The
per-bucket improvement r1->r2 (and r1->r3) divided by its FLOP cost is the
allocation value.

Also reports the gated picture: if only the lowest-confidence 15% of positions
received a second pass, what fraction of the total r1->r2 benefit would be
retained? (Dense computation is used for the measurement — building sparse
machinery is only justified if this value is there.)

    python3 probe_selectivity.py --config .tmp/ALLOC.json \
        --ckpt .tmp/p5/ALLOC_s0/model.pt --out .tmp/curves4/ALLOC_sel.jsonl
"""

import argparse, json, os, random, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_full, batch
from pathlm.model import PathLM
from pathlm.eval import eval_pc

DECILES = 10
LN2 = 0.6931471805599453


@torch.no_grad()
def selectivity_point(model: PathLM, eval_arr, vocab_size: int, max_rounds: int,
                      n_batches: int, batch_size: int, seed: int) -> dict:
    T = model.mcfg.seq_len
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    torch.manual_seed(1234)  # same corruption-draw convention as eval.repair

    # per-decile accumulators: [decile][round] = nats, tokens, hits-on-corrupt
    nats = [[0.0] * (max_rounds + 1) for _ in range(DECILES)]
    toks = [[0] * (max_rounds + 1) for _ in range(DECILES)]
    corr_hits = [[0] * (max_rounds + 1) for _ in range(DECILES)]
    corr_n = [0] * DECILES
    fire = [0, 0]                      # [gated positions, total] at the chosen gate
    tok_corr = 0
    total_nats = [0.0] * (max_rounds + 1)
    total_toks = [0] * (max_rounds + 1)
    edges = torch.linspace(0, 1, DECILES + 1)

    with eval_pc(model, w_dense_exit=0.0, p_retry=0.0, p_token_retry=0.0):
        pc = model.pcap
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, gen)
            x = x.to(model.embed.weight.device)
            tgt = x[:, 1:]
            paths = [sample_path(pc, rng, model.mcfg.n_layers) for _ in range(max_rounds)]
            for p in paths:
                p.n_retries = 0
                p.token_retry = False
            _, aux = model(x, paths, x)
            mask = aux["corrupt_mask"][:, :T - 1]
            # round-0 prob0: the allocator's signal, known BEFORE spending round 2
            conf0 = torch.sigmoid(aux["rounds"][0][0]["conf"][:, :T - 1])
            dec = torch.bucketize(conf0, edges[1:-1].to(conf0.device),
                                  right=False).clamp_(0, DECILES - 1)

            for r, nodes in enumerate(aux["rounds"], start=1):
                logits = nodes[1]["logits"][:, :T - 1]
                nll = -F.log_softmax(logits.float(), dim=-1).gather(
                    -1, tgt.unsqueeze(-1)).squeeze(-1)          # [B, T-1]
                hit = (logits.argmax(-1) == tgt)
                for d in range(DECILES):
                    sel = (dec == d)
                    nats[d][r] += nll[sel].sum().item()
                    toks[d][r] += sel.sum().item()
                    corr_hits[d][r] += (hit & mask & sel).sum().item()
                    corr_n[d] += (mask & sel).sum().item()   # BUGFIX: was never
                                                             # accumulated -> ntok None
                total_nats[r] += nll.sum().item()
                total_toks[r] += tgt.numel()                 # BUGFIX: was sel (decile)
            # selectivity gate: fire on the lowest-confidence 15% of positions
            thr = torch.quantile(conf0.flatten(), 0.15)
            gate = conf0 < thr
            fire[0] += gate.sum().item()
            fire[1] += gate.numel()
            tok_corr += int(mask.sum().item())

    out = {"decile": []}
    for d in range(DECILES):
        row = {"decile": d, "prob0_lo": round(edges[d].item(), 2),
               "prob0_hi": round(edges[d + 1].item(), 2),
               "tokens": toks[d][1],
               "n_corrupt": corr_n[d]}
        for r in range(1, max_rounds + 1):
            row[f"bpc_r{r}"] = (nats[d][r] / toks[d][r] / LN2) if toks[d][r] else None
            row[f"ntok_corr_r{r}"] = (corr_hits[d][r] / corr_n[d]) if corr_n[d] else None
        # r1->r2 improvement in bpc (negative = better)
        if row.get("bpc_r2") is not None and row["bpc_r1"] is not None:
            row["gain_r2_bpc"] = row["bpc_r1"] - row["bpc_r2"]
        if row.get("ntok_corr_r2") is not None:
            row["gain_r2_ntok"] = row["ntok_corr_r2"] - (row["ntok_corr_r1"] or 0)
        out["decile"].append(row)
    out["fire_rate_15pct"] = fire[0] / max(fire[1], 1)
    out["tokens_total"] = sum(toks[d][1] for d in range(DECILES))
    out["n_corrupt_total"] = tok_corr

    # gated retention: the benefit that lives in the lowest-confidence 15%
    def agg(row_key, r):
        num = sum(row[f"gain_{row_key}_r{r}"] * row["tokens"] for row in out["decile"]
                  if row.get(f"gain_{row_key}_r{r}") is not None)
        den = sum(row["tokens"] for row in out["decile"])
        return num / den if den else None
    out["avg_gain_r2_bpc"] = agg("r2_bpc", 2)
    out["avg_gain_r2_ntok"] = agg("r2_ntok", 2)
    # retention: gain captured by positions in the lowest-15% prob0 gate
    lo_rows = [row for row in out["decile"] if row["decile"] <= 1]  # bottom 2 deciles ≈ 15-20%
    if all(row.get("gain_r2_bpc") is not None for row in lo_rows) and abs(out["avg_gain_r2_bpc"]) > 1e-9:
        lo_tok = sum(row["tokens"] for row in lo_rows)
        lo_gain = sum(row["gain_r2_bpc"] * row["tokens"] for row in lo_rows) / lo_tok
        out["retention_low20_share"] = round(lo_gain / out["avg_gain_r2_bpc"], 4)
    else:
        out["retention_low20_share"] = None  # avg gain ~0: r1->r2 benefit is absent overall
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    mcfg, pcap = ModelConfig(**cfg["model"]), PathConfig(**cfg["path"])
    _, ev, V = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    torch.manual_seed(cfg["train"]["seed"])
    model = PathLM(mcfg, pcap, V).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    model.eval()
    rec = selectivity_point(model, ev, V, args.rounds, args.batches, args.batch_size, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"{'dec':>3} {'prob0':>9} {'tokens':>8} {'r1':>7} {'r2':>7} {'r3':>7} "
          f"{'gain_r2':>8} {'gain_r3':>8} {'ntok r1→r2':>10}")
    for row in rec["decile"]:
        g2 = f"{row['gain_r2_bpc']:+.4f}" if row.get("gain_r2_bpc") is not None else "—"
        g3 = f"{row['gain_r3_bpc']:+.4f}" if row.get("gain_r3_bpc") is not None else "—"
        n2 = f"{row.get('gain_r2_ntok', 0):+.4f}" if row.get("ntok_corr_r2") is not None else "—"
        print(f"{row['decile']:>3} {row['prob0_lo']:.1f}-{row['prob0_hi']:.1f} "
              f"{row['tokens']:>8} {row['bpc_r1']:>7.4f} {row['bpc_r2']:>7.4f} "
              f"{row['bpc_r3']:>7.4f} {g2:>8} {g3:>8} {n2:>10}")
    print(f"\nfire_rate@15% = {rec['fire_rate_15pct']:.3f}")
    print(f"avg gain r1→r2: bpc {rec['avg_gain_r2_bpc']:+.4f}, ntok {rec['avg_gain_r2_ntok']:+.4f}")
    if rec.get("retention_low20_share") is not None:
        print(f"lowest-2-decile share of total gain = {rec['retention_low20_share']:.2%} "
              f"(~15% gate uses bottom 2 deciles)")
    else:
        print("retention: N/A — avg r1->r2 gain is ~0 (no benefit to allocate)")


if __name__ == "__main__":
    main()
