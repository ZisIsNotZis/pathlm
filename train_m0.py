"""M0 probe: train the micro PathLM and answer gates (a)-(d) of docs/experiments.md.

Usage: python train_m0.py <run_name> [--w-consistency F] [--p-retry F] [--steps N]
Writes checkpoint + metrics JSON to .scratch/02-engine-m0/evidence/<run_name>/.
"""

import argparse, json, math, os, random, sys, time

import torch

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


def train(run_dir: str, pcap: PathConfig, data, vocab_size: int, steps: int):
    train_arr, val_arr, _ = data
    model = make_model(pcap, vocab_size).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200, (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))))
    rng, torch_rng = random.Random(0), torch.Generator().manual_seed(0)
    log_path = os.path.join(run_dir, "train_log.jsonl")
    t0 = time.time()
    for step in range(steps):
        model.train()
        x, y = batch(train_arr, 64, 128, torch_rng)
        path = sample_path(pcap, rng)
        loss, _ = model(x.cuda(), path, y.cuda())
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
    """Gate quantities on the validation stream (corruption on, retry always evaluated)."""
    model.eval()
    eval_pc = PathConfig(**{**pcap.__dict__, "corrupt_wrong": CORRUPT_RATE, "p_retry": 0.0})
    rng, torch_rng = random.Random(1), torch.Generator().manual_seed(1)
    streams = {name: {"logits": [], "conf": [], "tgt": []} for name in
               ("node0_r0", "node0_r1", "node1", "node2_direct", "node2_chain")}
    corrupt_masks = []
    for _ in range(n_batches):
        x, y = batch(val_arr, 64, 128, torch_rng)
        x, y = x.cuda(), y.cuda()
        path = sample_path(eval_pc, rng)
        path.n_retries = 1  # always evaluate the retry round for gate (d)
        _, aux = model(x, path, y)
        T = y.shape[1]
        streams["node0_r0"]["logits"].append(aux["rounds"][0][0]["logits"].cpu())
        streams["node0_r1"]["logits"].append(aux["rounds"][1][0]["logits"].cpu())
        streams["node1"]["logits"].append(aux["rounds"][0][1]["logits"][:, :T-1].cpu())
        streams["node1"]["conf"].append(aux["rounds"][0][1]["conf"][:, :T-1].cpu())
        streams["node1"]["tgt"].append(y[:, 1:].cpu())
        streams["node2_direct"]["logits"].append(aux["rounds"][0][2]["logits"][:, :T-2].cpu())
        streams["node2_direct"]["conf"].append(aux["rounds"][0][2]["conf"][:, :T-2].cpu())
        streams["node2_direct"]["tgt"].append(y[:, 2:].cpu())
        ch = aux["rounds"][0][1]["chain2"]
        streams["node2_chain"]["logits"].append(ch["logits"][:, :T-2].cpu())
        streams["node2_chain"]["conf"].append(ch["conf"][:, :T-2].cpu())
        streams["node2_chain"]["tgt"].append(y[:, 2:].cpu())
        streams["node0_r0"]["conf"].append(aux["rounds"][0][0]["conf"].cpu())
        streams["node0_r0"]["tgt"].append(y.cpu())
        streams["node0_r1"]["conf"].append(aux["rounds"][1][0]["conf"].cpu())
        streams["node0_r1"]["tgt"].append(y.cpu())
        corrupt_masks.append(aux["corrupt_mask"].cpu())
        # node2 estimates from the retry round join the ensemble
        T2 = T - 2
        streams["node2_direct"]["logits"].append(aux["rounds"][1][2]["logits"][:, :T2].cpu())
        streams["node2_direct"]["conf"].append(aux["rounds"][1][2]["conf"][:, :T2].cpu())
        streams["node2_direct"]["tgt"].append(y[:, 2:].cpu())
        ch1 = aux["rounds"][1][1]["chain2"]
        streams["node2_chain"]["logits"].append(ch1["logits"][:, :T2].cpu())
        streams["node2_chain"]["conf"].append(ch1["conf"][:, :T2].cpu())
        streams["node2_chain"]["tgt"].append(y[:, 2:].cpu())

    def stack(name):
        s = streams[name]
        return (torch.cat(s["logits"]), torch.cat(s["conf"]), torch.cat(s["tgt"]))

    out = {"node_acc": {}, "node_ece": {}, "ens_vs_best": {}, "retry": {}}
    ests = {}
    for name in streams:
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
    # (d): retry effect on repair (self-head accuracy on corrupted positions)
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
