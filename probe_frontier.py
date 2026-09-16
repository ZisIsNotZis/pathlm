"""Slider frontier probe (ticket 14, experiment 1 parts 3-4).

Measures the first real quality-vs-compute frontier for the Slider: for each
combination of speculative width k and prob0-gated retry threshold tau_retry,
report inference throughput (decode tok/s on a fixed prompt) and quality
(teacher-forced node-1 bpc under the SAME gating, on clean and corrupted
held-out text). Also exit-threshold rows for the depth lever.

Gating semantics match `decode_spec(retry_threshold=...)`: the prediction of
t_{i+1} comes from the refined row i iff prob0(row i) < tau_retry.

    python3 probe_frontier.py --config .tmp/INT2.json \
        --ckpt .tmp/p3/INT2_s0/model.pt \
        --out .scratch/14-allocator/evidence/composition/frontier_INT2.json
"""

import argparse, json, os, random, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_full, batch
from pathlm.model import PathLM
from pathlm.decode import decode, decode_spec
from pathlm.eval import eval_pc

LN2 = 0.6931471805599453


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def gated_teacher_forced(model, eval_arr, vocab_size, tau, n_batches, batch_size,
                         corrupted: bool, seed: int) -> dict:
    """Teacher-forced node-1 quality under prob0-gating at threshold tau.

    Per position i: the prediction of t_{i+1} comes from the round-2 state iff
    prob0(i) < tau, else from the round-1 state — the same gate the decode loop
    applies before emitting. tau=None -> round 1 everywhere (retry off).

    Returns gated bpc, the gated next-token accuracy on corrupted positions
    (the repair-relevant view), the round-1 (ungated) version of the same for
    reference, and the fire rate."""
    T = model.mcfg.seq_len
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    torch.manual_seed(1234)  # eval.repair corruption convention
    nats = 0.0
    toks = 0
    corr_n = 0
    gated_hits_corr = 0
    r1_hits_corr = 0
    fired = 0

    with eval_pc(model, w_dense_exit=0.0, p_retry=0.0, p_token_retry=0.0,
                 corrupt_wrong=(0.075 if corrupted else 0.0),
                 corrupt_mask=(0.075 if corrupted else 0.0)):
        pc = model.pcap
        for _ in range(n_batches):
            x, _ = batch(eval_arr, batch_size, T, gen)
            x = x.to(model.embed.weight.device)
            tgt = x[:, 1:]
            paths = [sample_path(pc, rng, model.mcfg.n_layers) for _ in range(2)]
            for p in paths:
                p.n_retries = 0
                p.token_retry = False
            _, aux = model(x, paths, x)
            mask = aux["corrupt_mask"][:, :T - 1]
            conf0 = torch.sigmoid(aux["rounds"][0][0]["conf"][:, :T - 1])
            nll_by_r, arg_by_r = {}, {}
            for r in (1, 2):
                logits = aux["rounds"][r - 1][1]["logits"][:, :T - 1]
                nll_by_r[r] = -F.log_softmax(logits.float(), -1).gather(
                    -1, tgt.unsqueeze(-1)).squeeze(-1)
                arg_by_r[r] = logits.argmax(-1)
            if tau is None:
                nll = nll_by_r[1]
                arg = arg_by_r[1]
            else:
                gate = (conf0 < tau)
                nll = torch.where(gate, nll_by_r[2], nll_by_r[1])
                arg = torch.where(gate, arg_by_r[2], arg_by_r[1])
                fired += gate.sum().item()
            nats += nll.sum().item()
            toks += tgt.numel()
            corr_n += mask.sum().item()
            gated_hits_corr += ((arg == tgt) & mask).sum().item()
            r1_hits_corr += ((arg_by_r[1] == tgt) & mask).sum().item()

    return {"gated_bpc": round(nats / toks / LN2, 4),
            "fire_rate": round(fired / toks, 4) if tau is not None else None,
            "ntok_corr": round(gated_hits_corr / corr_n, 4) if corr_n else None,
            "ntok_corr_r1": round(r1_hits_corr / corr_n, 4) if corr_n else None,
            "n_tokens": toks, "n_corrupt": corr_n}


@torch.no_grad()
def decode_bench(model, pcap, prompt, n_new, k, tau) -> dict:
    """Throughput of one decode configuration (greedy, B=1)."""
    _sync()
    t0 = time.perf_counter()
    if k >= 1 and model.pcap.n_mtp >= 2:
        gen, st = decode_spec(model, prompt, n_new, pcap, max_drafts=k,
                              retry_threshold=tau)
    else:
        gen, st = decode(model, prompt, n_new, pcap, retry_threshold=tau)
    _sync()
    dt = time.perf_counter() - t0
    tok_s = round(n_new / dt, 1) if dt > 0 else None
    if "n_forward" in st:
        passes = st["n_forward"] / max(len(gen), 1)
        extra = {"spec_accepts": st.get("drafts_accepted")}
    else:
        passes = 1 + sum(st["retries"]) / max(len(gen), 1)
        extra = {"retries": sum(st["retries"])}
    return {"tok_s": tok_s, "passes_per_tok": round(passes, 3), **extra}


@torch.no_grad()
def decode_bench_exit(model, pcap, prompt, n_new, tau_exit) -> dict:
    _sync()
    t0 = time.perf_counter()
    gen, st = decode(model, prompt, n_new, pcap, exit_threshold=tau_exit)
    _sync()
    dt = time.perf_counter() - t0
    depths = st.get("depths", [])
    return {"tok_s": round(n_new / dt, 1) if dt > 0 else None,
            "mean_depth": round(sum(depths) / len(depths), 2) if depths else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-new", type=int, default=200)
    ap.add_argument("--tf-batches", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    mcfg, pcap = ModelConfig(**cfg["model"]), PathConfig(**cfg["path"])
    _, ev, V = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    torch.manual_seed(args.seed)
    model = PathLM(mcfg, pcap, V).cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda"))
    model.eval()

    prompt = torch.randint(0, model.n_real_tokens, (64,),
                           generator=torch.Generator().manual_seed(7)).cuda()
    grid = []
    for k in (0, 1):
        if k >= 1 and model.pcap.n_mtp < 2:
            continue
        for tau in (None, 0.5, 0.7, 0.8, 0.9):
            row = {"k": k, "tau_retry": tau}
            row.update(decode_bench(model, pcap, prompt, args.n_new, k, tau))
            q_clean = gated_teacher_forced(model, ev, V, tau, args.tf_batches,
                                           args.batch_size, corrupted=False,
                                           seed=args.seed)
            q_corr = gated_teacher_forced(model, ev, V, tau, args.tf_batches,
                                          args.batch_size, corrupted=True,
                                          seed=args.seed)
            row["clean_bpc_gated"] = q_clean["gated_bpc"]
            row["corrupt_bpc_gated"] = q_corr["gated_bpc"]
            row["corrupt_ntok_gated"] = q_corr["ntok_corr"]
            row["corrupt_ntok_r1"] = q_corr["ntok_corr_r1"]
            row["fire_rate_tf"] = q_clean["fire_rate"]
            grid.append(row)
            t = tau if tau is not None else "off"
            print(f"  k={k} tau_retry={t:>4}  tok/s {row['tok_s']}  "
                  f"passes {row['passes_per_tok']}  clean_bpc {row['clean_bpc_gated']}  "
                  f"corr_bpc {row['corrupt_bpc_gated']}  "
                  f"ntok_corr {row['corrupt_ntok_gated']} (r1 {row['corrupt_ntok_r1']})",
                  flush=True)
    for te in (0.9, 0.8):
        row = {"k": 0, "tau_retry": None, "tau_exit": te}
        row.update(decode_bench_exit(model, pcap, prompt, args.n_new, te))
        row["quality"] = "UNCALIBRATED (INT2 has no dense-exit training) — speed only"
        grid.append(row)
        print(f"  tau_exit={te}  tok/s {row['tok_s']}  mean_depth {row['mean_depth']}"
              f"  (quality uncalibrated)", flush=True)

    out = {"model": args.ckpt, "config": args.config, "grid": grid,
           "notes": {"quality": "teacher-forced node-1 bpc under prob0-gating at tau; "
                                "corrupted = stage-0 corruption on (seed 1234)",
                     "speed": f"greedy decode tok/s, B=1, n_new={args.n_new}",
                     "exit_caveat": "INT2 has no dense-exit training: exit saves "
                                    "compute but the depth-wise quality is uncalibrated",
                     "exit_x_spec": "not composable in this engine: the batched "
                                    "verification must run full depth "
                                    "(_run_stack_multi), see docs/mental_model.md §4"}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("saved", args.out)


if __name__ == "__main__":
    main()
