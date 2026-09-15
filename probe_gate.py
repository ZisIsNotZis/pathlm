"""Eval-only prob0 retry-gate sweep (ticket 13, experiment 1).

The gate only changes the re-entry DECISION, so it can be applied at eval time
to an EXISTING retry checkpoint — no retraining. For each tau this measures, on
corrupted input:

  fire_rate   fraction of positions whose previous-round prob0 < tau
  bpc_r1/r2   node-1 CE per round (corrupted input)
  ntok_corr   node-1 NEXT-TOKEN accuracy on corrupted positions (this is NOT
              eval.repair's node-0 self-repair; see probe_e2e.py's header)

`retention_bpc` = (bpc_r1 - bpc_r2)(tau) / (bpc_r1 - bpc_r2)(ungated), and
`retention_ntok` = (ntok_r2 - ntok_r1)(tau) / (ungated). The ungated reference
is the tau where the gate is off (0.0 / 1.0, identical by construction).

    python3 probe_gate.py --config configs/C1M.json \
        --ckpt .tmp/ckpts4/C1M_s0/model_step012000.pt \
        --out .tmp/gating/C1M_s0_gate.jsonl
"""

import argparse, glob, json, os, random, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_full, batch
from pathlm.model import PathLM
from pathlm.eval import eval_pc

STEP_RE = __import__("re").compile(r"model_step(\d+)\.pt$")
LN2 = 0.6931471805599453


def _load_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read config {path}: {e}") from e


def _resolve_ckpt(spec: str) -> tuple[int, str]:
    """A .pt file, or a directory -> its latest model_step*.pt."""
    if os.path.isdir(spec):
        found = [(int(m.group(1)), p)
                 for p in glob.glob(os.path.join(spec, "model_step*.pt"))
                 if (m := STEP_RE.search(p))]
        if not found:
            raise SystemExit(f"no model_step*.pt in {spec}")
        return max(found)
    m = STEP_RE.search(spec)
    return (int(m.group(1)) if m else -1), spec


@torch.no_grad()
def sweep_ckpt(config_path: str, ckpt_spec: str, taus: list[float],
               n_batches: int, batch_size: int, seed: int) -> list[dict]:
    cfg = _load_json(config_path)
    mcfg, pcap = ModelConfig(**cfg["model"]), PathConfig(**cfg["path"])
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    step, ckpt = _resolve_ckpt(ckpt_spec)
    torch.manual_seed(cfg["train"]["seed"])
    model = PathLM(mcfg, pcap, vocab_size).cuda()
    model.load_state_dict(torch.load(ckpt, map_location="cuda"))
    model.eval()

    T = model.mcfg.seq_len
    gen = torch.Generator().manual_seed(seed)
    xs = [batch(eval_arr, batch_size, T, gen)[0].cuda() for _ in range(n_batches)]
    # one fixed pair of layer paths per batch, shared across taus
    prng = random.Random(seed)
    path_pairs = []
    for _ in range(n_batches):
        ps = []
        for _ in range(2):
            p = sample_path(pcap, prng, model.mcfg.n_layers)
            p.n_retries = 0
            p.token_retry = False
            ps.append(p)
        path_pairs.append(ps)

    recs = []
    for tau in taus:
        torch.manual_seed(1234)  # identical corruption draws for every tau
        ce1 = ce2 = 0.0
        hit1 = hit2 = 0
        n_corr = 0
        fire = 0.0
        toks = 0
        for x, ps in zip(xs, path_pairs):
            with eval_pc(model, retry_gate=tau, p_token_retry=0.0, w_dense_exit=0.0):
                _, aux = model(x, ps, x)
            conf0 = aux["rounds"][0][0]["conf"]  # [B, T] previous-round prob0 logit
            if tau > 0:
                fire += (conf0.sigmoid() < tau).float().mean().item()
            tgt = x[:, 1:]
            mask = aux["corrupt_mask"][:, :tgt.shape[1]]
            n_corr += int(mask.sum().item())
            for r, nodes in enumerate(aux["rounds"], start=1):
                logits = nodes[1]["logits"][:, :tgt.shape[1]]
                ce = F.cross_entropy(logits.reshape(-1, vocab_size), tgt.reshape(-1),
                                     reduction="sum").item()
                hits = ((logits.argmax(-1) == tgt) & mask).sum().item()
                if r == 1:
                    ce1 += ce
                    hit1 += hits
                else:
                    ce2 += ce
                    hit2 += hits
            toks += tgt.numel()
        bpc1, bpc2 = ce1 / toks / LN2, ce2 / toks / LN2
        nt1 = hit1 / n_corr if n_corr else float("nan")
        nt2 = hit2 / n_corr if n_corr else float("nan")
        recs.append({"step": step, "ckpt": os.path.basename(ckpt), "tau": tau,
                     "fire_rate": fire / n_batches, "bpc_r1": bpc1, "bpc_r2": bpc2,
                     "ntok_corr_r1": nt1, "ntok_corr_r2": nt2,
                     "n_corrupt": n_corr, "n_tokens": toks})
        print(f"  {os.path.basename(ckpt)} tau={tau:.2f} fire={fire/n_batches:.4f} "
              f"r1 {bpc1:.4f} r2 {bpc2:.4f}  ntok {nt1:.4f}->{nt2:.4f}", flush=True)
    del model
    torch.cuda.empty_cache()
    return recs


def _add_retention(recs: list[dict]) -> None:
    """Retention relative to the ungated reference at the same checkpoint."""
    by_step: dict[int, list[dict]] = {}
    for r in recs:
        by_step.setdefault(r["step"], []).append(r)
    for _step, rows in by_step.items():
        base = min(rows, key=lambda r: r["fire_rate"])  # tau=0.0 -> ungated
        b_gain = base["bpc_r1"] - base["bpc_r2"]
        n_gain = base["ntok_corr_r2"] - base["ntok_corr_r1"]
        for r in rows:
            r["retention_bpc"] = ((r["bpc_r1"] - r["bpc_r2"]) / b_gain
                                  if abs(b_gain) > 1e-9 else None)
            r["retention_ntok"] = ((r["ntok_corr_r2"] - r["ntok_corr_r1"]) / n_gain
                                   if abs(n_gain) > 1e-9 else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    all_recs = []
    for spec in args.ckpt:
        print(f"[{spec}]", flush=True)
        all_recs.extend(sweep_ckpt(args.config, spec, args.taus,
                                   args.batches, args.batch_size, args.seed))
    _add_retention(all_recs)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(all_recs)} rows to {args.out}", flush=True)


if __name__ == "__main__":
    main()
