"""P1 (ticket 12): conditional-chain A/B probe.

bpc-first rule (user's measurement law): the chain has an obvious bpc claim —
does conditioning node-2 on the actual t_{i+1} beat the direct head on t+2?

Per checkpoint, paired batches (same seed), four estimators of t_{i+2}:
  node2_direct      the direct transform head
  chain_oracle      token-conditioned on the TRUE t_{i+1} (teacher forcing) —
                    an upper bound; never quoted alone as the result
  chain_deploy      token-conditioned on node-1's argmax — what inference
                    actually has; THIS is the honest number
  chain latent      (control mode) the old T1@T1 composition

Spec-decode row (speed side): draft = the estimator's argmax for t_{i+2};
verify = node-1 at row i+1. Report P(draft == verify) and P(accept | verify
correct) — the deployable accept rate.

    python3 probe_chain.py --ckpt-dir .tmp/ckpts3/CH_token_long \
        --config .tmp/CH_token.json --out .tmp/curves3/CH_token.jsonl
    python3 probe_chain.py --report CH_latent.jsonl CH_token.jsonl
"""

import argparse, glob, json, os, random, re, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_full, batch
from pathlm.model import PathLM

STEP_RE = re.compile(r"model_step(\d+)\.pt$")


def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise SystemExit(f"cannot read {path}: {e}") from e


def steps_in(ckpt_dir):
    found = []
    for p in glob.glob(os.path.join(ckpt_dir, "model_step*.pt")):
        m = STEP_RE.search(p)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


@torch.no_grad()
def chain_point(model, eval_arr, vocab_size, n_batches, batch_size, seed):
    T = model.mcfg.seq_len
    gen = torch.Generator().manual_seed(seed)
    rng = random.Random(seed)
    token_mode = model.pcap.chain_mode == "token"
    torch.manual_seed(1234)

    acc = {"node1": [], "direct": [], "chain": [], "chain_oracle": [], "chain_deploy": []}
    ce = {"node1": [], "direct": [], "chain": [], "chain_oracle": [], "chain_deploy": []}
    acc_agree, acc_right = {"direct": [], "chain_deploy": []}, {"direct": [], "chain_deploy": []}

    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, gen)
        x = x.cuda()
        t2 = x[:, 2:]
        path = sample_path(model.pcap, rng, model.mcfg.n_layers)
        path.n_retries = 0
        path.token_retry = False

        # forward 1: oracle condition (teacher forcing) for token mode; plain for latent
        cond = x[:, 1:] if token_mode else None
        _, aux = model(x, [path], x, condition=cond)
        r = aux["rounds"][0]
        l1 = r[1]["logits"][:, :T - 2]
        l2d = r[2]["logits"][:, :T - 2]
        l2c = r[1]["chain2"]["logits"][:, :T - 2]
        acc["node1"].append((r[1]["logits"][:, :T - 1].argmax(-1) == x[:, 1:]).float().mean().item())
        ce["node1"].append(F.cross_entropy(r[1]["logits"][:, :T - 1].reshape(-1, vocab_size),
                                           x[:, 1:].reshape(-1)).item())
        acc["direct"].append((l2d.argmax(-1) == t2).float().mean().item())
        ce["direct"].append(F.cross_entropy(l2d.reshape(-1, vocab_size), t2.reshape(-1)).item())
        key = "chain_oracle" if token_mode else "chain"
        acc[key].append((l2c.argmax(-1) == t2).float().mean().item())
        ce[key].append(F.cross_entropy(l2c.reshape(-1, vocab_size), t2.reshape(-1)).item())

        if token_mode:
            # forward 2: deploy condition (node-1's argmax), same path for pairing
            model.deploy_chain = True
            _, aux_d = model(x, [path], x)
            model.deploy_chain = False
            l2cd = aux_d["rounds"][0][1]["chain2"]["logits"][:, :T - 2]
            acc["chain_deploy"].append((l2cd.argmax(-1) == t2).float().mean().item())
            ce["chain_deploy"].append(F.cross_entropy(l2cd.reshape(-1, vocab_size),
                                                      t2.reshape(-1)).item())
            ver_src = l2cd
        else:
            ver_src = l2c

        # spec-decode: draft (estimator at row i for t+2) vs verify (node-1 at row i+1)
        for name, ld in (("direct", l2d), ("chain_deploy", ver_src)):
            draft = ld.argmax(-1)
            verify = r[1]["logits"].argmax(-1)[:, 1:T - 1]  # node-1 row i+1 → t_{i+2}
            right = (verify == x[:, 2:])
            ok = (draft == verify)
            acc_agree[name].append(ok.float().mean().item())
            if right.any():
                acc_right[name].append(ok[right].float().mean().item())

    def m(v):
        return round(sum(v) / len(v), 4) if v else None

    out = {
        "acc_node1_t1": m(acc["node1"]), "ce_node1_t1": m(ce["node1"]),
        "acc_direct_t2": m(acc["direct"]), "ce_direct_t2": m(ce["direct"]),
        "accept_direct": m(acc_agree["direct"]),
        "accept_direct_given_right": m(acc_right["direct"]),
    }
    for k in ("chain", "chain_oracle", "chain_deploy"):
        if acc[k]:
            out[f"acc_{k}_t2"] = m(acc[k])
            out[f"ce_{k}_t2"] = m(ce[k])
    if acc_agree["chain_deploy"]:
        out["accept_chain_deploy"] = m(acc_agree["chain_deploy"])
        out["accept_chain_deploy_given_right"] = m(acc_right["chain_deploy"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ckpt-dir")
    ap.add_argument("--out")
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", nargs=2, metavar=("LATENT.jsonl", "TOKEN.jsonl"))
    args = ap.parse_args()

    if args.report:
        def last(p):
            return json.loads(open(p).readlines()[-1])
        a, b = last(args.report[0]), last(args.report[1])
        rows = [
            ("node1 t+1 acc", "acc_node1_t1"), ("node1 t+1 ce", "ce_node1_t1"),
            ("direct t+2 acc", "acc_direct_t2"), ("direct t+2 ce", "ce_direct_t2"),
            ("chain t+2 acc", "acc_chain_t2"), ("chain t+2 ce", "ce_chain_t2"),
            ("chain-oracle t+2 acc", "acc_chain_oracle_t2"),
            ("chain-oracle t+2 ce", "ce_chain_oracle_t2"),
            ("chain-deploy t+2 acc", "acc_chain_deploy_t2"),
            ("chain-deploy t+2 ce", "ce_chain_deploy_t2"),
            ("accept direct", "accept_direct"),
            ("accept direct|right", "accept_direct_given_right"),
            ("accept chain-deploy", "accept_chain_deploy"),
            ("accept chain-deploy|right", "accept_chain_deploy_given_right"),
        ]
        print(f"{'metric':30} {'latent':>12} {'token':>12} {'delta':>10}")
        for name, k in rows:
            va, vb = a.get(k), b.get(k)
            if va is None and vb is None:
                continue
            d = f"{vb - va:+.4f}" if (va is not None and vb is not None) else ""
            print(f"{name:30} {va if va is not None else '—':>12} "
                  f"{vb if vb is not None else '—':>12} {d:>10}")
        return

    for req in ("config", "ckpt_dir", "out"):
        if not getattr(args, req):
            raise SystemExit(f"--{req.replace('_','-')} is required")
    cfg = _load_json(args.config)
    mcfg, pcap = ModelConfig(**cfg["model"]), PathConfig(**cfg["path"])
    _, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    pts = steps_in(args.ckpt_dir)
    if not pts:
        raise SystemExit(f"no model_step*.pt in {args.ckpt_dir}")
    try:
        f = open(args.out, "w")
    except OSError as e:
        raise SystemExit(f"cannot write {args.out}: {e}") from e
    with f:
        for step, path in pts:
            torch.manual_seed(cfg["train"]["seed"])
            model = PathLM(mcfg, pcap, vocab_size).cuda()
            model.load_state_dict(torch.load(path, map_location="cuda"))
            model.eval()
            rec = {"step": step}
            rec.update(chain_point(model, eval_arr, vocab_size,
                                   args.batches, args.batch_size, args.seed))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  step {step:6d}  node1 {rec['acc_node1_t1']:.4f}  "
                  f"direct {rec['acc_direct_t2']:.4f}" +
                  (f"  oracle {rec.get('acc_chain_oracle_t2', 0):.4f}"
                   f"  deploy {rec.get('acc_chain_deploy_t2', 0):.4f}"
                   f"  accept_d {rec.get('accept_chain_deploy_given_right', 0):.4f}"
                   if 'acc_chain_oracle_t2' in rec else
                   f"  chain {rec.get('acc_chain_t2', 0):.4f}"), flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
