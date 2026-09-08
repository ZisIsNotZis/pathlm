"""M0 probe: train the micro PathLM and answer gates (a)-(d) of docs/experiments.md.

Usage: python train_m0.py <run_name> [--w-consistency F] [--p-retry F] [--steps N]
Writes checkpoint + metrics JSON to .scratch/02-engine-m0/evidence/<run_name>/.
"""

import argparse, json, math, os, random, sys, time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_subset, batch
from pathlm.metrics import ece, ensemble
from pathlm.model import PathLM

CORRUPT_RATE = 0.15  # lambda_total for the M0 probe (wrong-token only)


def make_model(pcap: PathConfig, vocab_size: int) -> PathLM:
    mcfg = ModelConfig()
    torch.manual_seed(0)
    return PathLM(mcfg, pcap, vocab_size)


def sample_rounds(pcap: PathConfig, rng: random.Random, n_layers: int) -> list:
    """Base path decides the retry count; each round gets its own fresh path."""
    base = sample_path(pcap, rng, n_layers)
    return [base] + [sample_path(pcap, rng, n_layers) for _ in range(base.n_retries)]


def train(run_dir: str, pcap: PathConfig, data, vocab_size: int, steps: int):
    train_arr, val_arr, _ = data
    model = make_model(pcap, vocab_size).cuda()
    mcfg = model.mcfg
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200, (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))))
    rng, torch_rng = random.Random(0), torch.Generator().manual_seed(0)
    log_path = os.path.join(run_dir, "train_log.jsonl")
    t0 = time.time()
    for step in range(steps):
        model.train()
        x, y = batch(train_arr, 64, 128, torch_rng)
        paths = sample_rounds(pcap, rng, mcfg.n_layers)
        loss, _ = model(x.cuda(), paths, x.cuda())  # targets = clean input tokens
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == steps - 1:
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": round(loss.item(), 4),
                                    "lr": sched.get_last_lr()[0], "min": round((time.time()-t0)/60, 1)}) + "\n")
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    return model


@torch.no_grad()
def evaluate(model: PathLM, val_arr, vocab_size: int, pcap: PathConfig, n_batches=40):
    """Gate quantities on the validation stream (corruption on, retry always
    evaluated). Eval corruption is seed-controlled via torch.manual_seed for
    run-to-run reproducibility."""
    model.eval()
    torch.manual_seed(1234)  # seed the global RNG that stage-0 corruption draws from
    eval_pc = PathConfig(**{**pcap.__dict__, "corrupt_wrong": CORRUPT_RATE, "p_retry": 0.0})
    rng, torch_rng = random.Random(1), torch.Generator().manual_seed(1)
    names = ("node0_r0", "node0_r1", "node1", "node2_direct", "node2_chain")
    streams = {name: {"logits": [], "conf": [], "tgt": []} for name in names}
    corrupt_masks, cosines, disagreements = [], [], []
    T = model.mcfg.seq_len
    T0, T1, T2 = T, T - 1, T - 2
    for _ in range(n_batches):
        x, _ = batch(val_arr, 64, T, torch_rng)
        x = x.cuda()  # targets = the clean tokens themselves
        paths = [sample_path(eval_pc, rng, model.mcfg.n_layers) for _ in range(2)]
        _, aux = model(x, paths, x)
        cm = aux["corrupt_mask"]
        corrupt_masks.append(cm.cpu())
        streams["node0_r0"]["logits"].append(aux["rounds"][0][0]["logits"].cpu())
        streams["node0_r1"]["logits"].append(aux["rounds"][1][0]["logits"].cpu())
        for r, name in ((0, "node0_r0"), (1, "node0_r1")):
            streams[name]["conf"].append(aux["rounds"][r][0]["conf"].cpu())
            streams[name]["tgt"].append(x.cpu())
        streams["node1"]["logits"].append(aux["rounds"][0][1]["logits"][:, :T1].cpu())
        streams["node1"]["conf"].append(aux["rounds"][0][1]["conf"][:, :T1].cpu())
        streams["node1"]["tgt"].append(x[:, 1:].cpu())
        for tag, r in (("direct", 0), ("direct", 1)):
            node = aux["rounds"][r][2]
            streams["node2_direct"]["logits"].append(node["logits"][:, :T2].cpu())
            streams["node2_direct"]["conf"].append(node["conf"][:, :T2].cpu())
            streams["node2_direct"]["tgt"].append(x[:, 2:].cpu())
        for r in (0, 1):
            ch = aux["rounds"][r][1]["chain2"]
            streams["node2_chain"]["logits"].append(ch["logits"][:, :T2].cpu())
            streams["node2_chain"]["conf"].append(ch["conf"][:, :T2].cpu())
            streams["node2_chain"]["tgt"].append(x[:, 2:].cpu())
        # gate (c): dispersion between the two node-2 estimates (round 0)
        d_lat = aux["rounds"][0][2]["latent"]; c_lat = aux["rounds"][0][1]["chain2"]["latent"]
        cosines.append(F.cosine_similarity(d_lat, c_lat, dim=-1).mean().cpu())
        disagreements.append((aux["rounds"][0][2]["logits"][:, :T2].argmax(-1) !=
                              ch["logits"][:, :T2].argmax(-1)).float().mean().cpu())

    def stack(name):
        s = streams[name]
        return torch.cat(s["logits"]), torch.cat(s["conf"]), torch.cat(s["tgt"])

    out = {"node_acc": {}, "node_ece": {}, "ens_vs_best": {}, "retry": {},
           "consistency_dispersion": {}}
    ests = {}
    for name in names:
        logits, conf, tgt = stack(name)
        hit = (logits.argmax(-1) == tgt).float()
        ests[name] = (logits, conf, hit)
        out["node_acc"][name] = round(hit.mean().item(), 4)
        out["node_ece"][name] = round(ece(conf.sigmoid(), hit), 4)
    # (b): node-2 ensemble (direct + chain, both rounds) vs best single estimate
    pairs = [(ests[n][0], ests[n][1]) for n in ("node2_direct", "node2_chain")]
    mix, conf_mix = ensemble(pairs)
    tgt2 = stack("node2_direct")[2]
    hit_mix = (mix.argmax(-1) == tgt2).float()
    out["ens_vs_best"] = {
        "ensemble_acc": round(hit_mix.mean().item(), 4),
        "ensemble_ece": round(ece(conf_mix, hit_mix), 4),
        "best_single_acc": round(max(out["node_acc"][n] for n in ("node2_direct", "node2_chain")), 4),
    }
    # (c): dispersion between the node-2 estimates
    out["consistency_dispersion"] = {
        "latent_cosine": round(torch.stack(cosines).mean().item(), 4),
        "argmax_disagreement": round(torch.stack(disagreements).mean().item(), 4),
    }
    # (d): retry effect on repair (self-head accuracy on actually-corrupted positions)
    cm = torch.cat(corrupt_masks)
    for r in (0, 1):
        logits, conf, tgt = stack(f"node0_r{r}")
        hit = (logits.argmax(-1) == tgt).float()
        out["retry"][f"round{r}"] = {
            "self_acc_all": round(hit.mean().item(), 4),
            "self_acc_corrupted": round(hit[cm].mean().item(), 4),
            "ece": round(ece(conf.sigmoid(), hit), 4),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--w-consistency", type=float, default=0.0)
    ap.add_argument("--p-retry", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()

    run_dir = os.path.join(".scratch", "02-engine-m0", "evidence", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    data = load_enwik8_subset(".tmp/enwik8", 12_000_000, "data/m0_subset.npz")
    vocab_size = data[2]
    pcap = PathConfig(corrupt_wrong=CORRUPT_RATE, n_mtp=2, transport="direct",
                      p_retry=args.p_retry, w_consistency=args.w_consistency)
    model = train(run_dir, pcap, data, vocab_size, args.steps)
    results = evaluate(model, data[1], vocab_size, pcap)
    results["config"] = {"w_consistency": args.w_consistency, "p_retry": args.p_retry,
                         "steps": args.steps, "corrupt_wrong": CORRUPT_RATE}
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
