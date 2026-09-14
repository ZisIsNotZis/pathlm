"""E2E effectiveness-vs-training-step curves (ticket 11).

The tax-only view is misleading: clean 1-round bpc is the one operating point
where repair machinery has nothing to do, so it charges a mechanism full price
and gives it no chance to earn. This probe measures what a mechanism BUYS at
the operating point it was built for, and what it costs at inference.

Per checkpoint, per retry depth n:
  bpc_round_n   node-1 CE of round n's OWN output, corrupted input — the
                quality-vs-compute curve the tax view cannot show
  repair_rn     self-node repair accuracy at round n (round 1 included)
  clean_bpc     no corruption, 1 round — the capacity-tax view
  tok_per_s     measured throughput of the 1-round and the k-round path

Every row is cost + gain. Raw tax makes no sense without the paired benefit
column; and repair machinery has an accelerated-decoding angle too (prob0-gated
retry is a spec-decode accept-rate problem: 77.3% accept given verify-right).

    python3 probe_e2e.py --config configs/C1.json --ckpt-dir .tmp/ckpts2/C1long \
        --out .tmp/curves2/C1long_e2e.jsonl
    python3 probe_e2e.py --compare .tmp/curves2/C1long_e2e.jsonl \
        .tmp/curves2/B0long_e2e.jsonl --n 3
"""

import argparse, glob, json, os, random, re, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_full, batch
from pathlm.model import PathLM
from pathlm.eval import eval_pc

STEP_RE = re.compile(r"model_step(\d+)\.pt$")


def _load_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read config {path}: {e}") from e


def steps_in(ckpt_dir: str) -> list[tuple[int, str]]:
    found = []
    for p in glob.glob(os.path.join(ckpt_dir, "model_step*.pt")):
        m = STEP_RE.search(p)
        if m is None:
            continue
        try:
            found.append((int(m.group(1)), p))
        except ValueError:
            continue
    return sorted(found)


def ece_of(conf: torch.Tensor, hit: torch.Tensor, bins: int = 15) -> float:
    conf, hit = conf.flatten(), hit.flatten()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        sel = (conf > edges[i]) & (conf <= edges[i + 1])
        if sel.any():
            ece += (sel.float().sum().item() / conf.numel()
                    * abs(hit[sel].mean().item() - conf[sel].mean().item()))
    return round(float(ece), 4)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def e2e_point(model: PathLM, eval_arr, vocab_size: int, max_rounds: int,
              n_batches: int, batch_size: int, seed: int) -> dict:
    """One checkpoint, every retry depth, on the SAME batches. Round n's CE is
    accumulated over round n's own output only — never mixed across rounds, or
    "bpc after n rounds" would be a meaningless round-mixture."""
    T = model.mcfg.seq_len
    gen = torch.Generator().manual_seed(seed)
    batches = [batch(eval_arr, batch_size, T, gen)[0].to(model.embed.weight.device)
               for _ in range(n_batches)]
    rng = random.Random(seed)
    torch.manual_seed(1234)  # same corruption-draw convention as eval.repair

    ce = {n: 0.0 for n in range(1, max_rounds + 1)}
    rep = {n: [0, 0] for n in range(1, max_rounds + 1)}   # [hits, n_corrupt]
    nats_clean = 0.0
    toks = tok_corr = 0
    t_k = t_1 = 0.0          # wall time of the k-round pass and the 1-round pass
    confs, hits = [], []

    with eval_pc(model, w_dense_exit=0.0, p_retry=0.0, p_token_retry=0.0):
        pc = model.pcap
        for x in batches:
            tgt = x[:, 1:]

            # --- k-round pass, corruption ON (the mechanism's operating point)
            paths = [sample_path(pc, rng, model.mcfg.n_layers) for _ in range(max_rounds)]
            for p in paths:
                p.n_retries = 0
                p.token_retry = False
            _sync(); t0 = time.perf_counter()
            _, aux = model(x, paths, x)
            _sync(); t_k += time.perf_counter() - t0
            mask = aux["corrupt_mask"]
            for r, nodes in enumerate(aux["rounds"], start=1):
                logits = nodes[1]["logits"][:, :-1]
                ce[r] += F.cross_entropy(logits.reshape(-1, vocab_size),
                                         tgt.reshape(-1), reduction="sum").item()
                mask_r = mask[:, :logits.shape[1]]  # corrupt_mask covers all T rows
                rep[r][0] += ((logits.argmax(-1) == tgt) & mask_r).sum().item()
                rep[r][1] += mask_r.sum().item()
                if r == 1:
                    confs.append(nodes[1]["conf"][:, :-1].cpu())
                    hits.append((logits.argmax(-1) == tgt).float().cpu())

            # --- 1-round pass, corruption OFF (the tax view + the speed base)
            with eval_pc(model, corrupt_wrong=0.0, corrupt_mask=0.0,
                         perturb_noise=0.0, pure_noise=0.0):
                p1 = sample_path(pc, rng, model.mcfg.n_layers)
                p1.n_retries = 0
                p1.token_retry = False
                _sync(); t0 = time.perf_counter()
                _, aux_c = model(x, [p1], x)
                _sync(); t_1 += time.perf_counter() - t0
            logits = aux_c["rounds"][0][1]["logits"][:, :-1]
            nats_clean += F.cross_entropy(logits.reshape(-1, vocab_size),
                                          tgt.reshape(-1), reduction="sum").item()

            toks += tgt.numel()
            tok_corr += int(mask[:, :logits.shape[1]].sum().item())
            del aux, aux_c

    bpc = {n: ce[n] / toks / 0.6931471805599453 for n in ce}
    return {
        "bpc_round": {str(n): v for n, v in bpc.items()},
        "clean_bpc": nats_clean / toks / 0.6931471805599453,
        "repair": {str(n): (rep[n][0] / rep[n][1]) if rep[n][1] else None
                   for n in rep},
        "n_corrupt": tok_corr,
        "tok_per_s_1round": toks / t_1 if t_1 > 0 else None,
        "tok_per_s_kround": (toks * max_rounds) / t_k if t_k > 0 else None,
        "k_round_slowdown": (t_k / max(t_1, 1e-9)) / max(max_rounds, 1),
        "ece_round1": ece_of(torch.cat(confs), torch.cat(hits)),
        "n_tokens": toks,
    }


def build_curve(config_path: str, ckpt_dir: str, out_path: str, max_rounds: int,
                n_batches: int, batch_size: int, seed: int) -> None:
    cfg = _load_json(config_path)
    mcfg, pcap = ModelConfig(**cfg["model"]), PathConfig(**cfg["path"])
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    pts = steps_in(ckpt_dir)
    if not pts:
        raise SystemExit(f"no model_step*.pt in {ckpt_dir}")
    try:
        f = open(out_path, "w")
    except OSError as e:
        raise SystemExit(f"cannot write {out_path}: {e}") from e
    with f:
        for step, path in pts:
            torch.manual_seed(cfg["train"]["seed"])
            model = PathLM(mcfg, pcap, vocab_size).cuda()
            model.load_state_dict(torch.load(path, map_location="cuda"))
            model.eval()
            rec = {"step": step, "ckpt": os.path.basename(path)}
            rec.update(e2e_point(model, eval_arr, vocab_size, max_rounds,
                                 n_batches, batch_size, seed))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            r, rep = rec["bpc_round"], rec["repair"]
            print(f"  step {step:6d}  clean {rec['clean_bpc']:.4f}"
                  + "".join(f"  r{n} {r[str(n)]:.4f}" for n in range(1, max_rounds + 1))
                  + "".join(f"  rep{n} {rep[str(n)]:.4f}"
                            for n in range(1, max_rounds + 1)
                            if rep.get(str(n)) is not None), flush=True)
            del model
            torch.cuda.empty_cache()


def compare(mech_path: str, base_path: str, n: int) -> None:
    """E2E ledger: the gain column BESIDE the tax column, per step. First file
    is the mechanism run, second is the B0 baseline."""
    def load(p):
        try:
            with open(p) as f:
                return {json.loads(l)["step"]: json.loads(l) for l in f}
        except (OSError, json.JSONDecodeError, KeyError) as e:
            raise SystemExit(f"cannot read curve {p}: {e}") from e
    A, B = load(mech_path), load(base_path)
    shared = sorted(set(A) & set(B))
    if not shared:
        raise SystemExit("no shared steps between the two curves")
    rk = str(n)
    print(f"{'step':>6} | {'tax(clean)':>10} | {'E2E@r'+rk:>9} {'B0@r'+rk:>9} "
          f"| {'bpc gain':>9} | {'repair gain':>11} | {'slowdown':>8}")
    for s in shared:
        a, b = A[s], B[s]
        tax = a["clean_bpc"] - b["clean_bpc"]
        gain = b["bpc_round"][rk] - a["bpc_round"][rk]
        ra, rb = a["repair"].get(rk), b["repair"].get(rk)
        rg = (ra - rb) if (ra is not None and rb is not None) else float("nan")
        print(f"{s:>6} | {tax:>+10.4f} | {a['bpc_round'][rk]:>9.4f} "
              f"{b['bpc_round'][rk]:>9.4f} | {gain:>+9.4f} | {rg:>+11.4f} | "
              f"{a['k_round_slowdown']:>7.2f}x")
    a, b = A[shared[-1]], B[shared[-1]]
    ra, rb = a["repair"].get(rk), b["repair"].get(rk)
    rep_gain = (ra - rb) if (isinstance(ra, float) and isinstance(rb, float)) else float("nan")
    print(f"\n@step {shared[-1]}: tax {a['clean_bpc']-b['clean_bpc']:+.4f} bpc | "
          f"E2E bpc gain {b['bpc_round'][rk]-a['bpc_round'][rk]:+.4f} | "
          f"repair gain {rep_gain:+.4f} | "
          f"{a['tok_per_s_kround']:.0f} tok/s "
          f"({b['tok_per_s_kround']/a['tok_per_s_kround']:.2f}x B0)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--out")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", nargs=2, metavar=("MECH.jsonl", "B0.jsonl"))
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    if args.compare:
        compare(args.compare[0], args.compare[1], args.n)
        return
    for req in ("config", "ckpt_dir", "out"):
        if not getattr(args, req):
            raise SystemExit(f"--{req.replace('_','-')} is required to build a curve")
    build_curve(args.config, args.ckpt_dir, args.out, args.rounds,
                args.batches, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
