import sys, os, json, random, torch
sys.path.insert(0, os.getcwd())
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_full, batch
from pathlm.model import PathLM

def probe(run, transp, mix, n_rounds=3):
    cfg = json.load(open(f'configs/{run}.json'))
    mcfg = ModelConfig(**cfg['model']); pcap = PathConfig(**cfg['path'])
    pcap.transport = transp; pcap.reentry_mix = mix
    pcap.p_retry=0.0; pcap.p_token_retry=0.0; pcap.w_dense_exit=0.0
    pcap.corrupt_wrong=0.15; pcap.corrupt_mask=0.0
    model = PathLM(mcfg, pcap, 206).cuda()
    sd = torch.load(f'.scratch/10-gap-sweep/evidence/{run}/model.pt', map_location='cuda')
    model.load_state_dict(sd); model.eval()
    train, ev, V = load_enwik8_full('.tmp/enwik8','data/enwik8_full.npz')
    g = torch.Generator().manual_seed(1)
    x,_ = batch(ev, 16, 512, g); x=x.cuda()
    rng = random.Random(0)
    rec = {}
    orig_mix = model._mixture_reentry
    def spy(mixd):
        out = orig_mix(mixd)
        rec.setdefault('mixre',[]).append((out.norm(dim=-1).mean().item(), out.norm(dim=-1).max().item()))
        return out
    model._mixture_reentry = spy
    orig_run = model._run_layers
    def spy2(h, *a, **k):
        rec.setdefault('in',[]).append(h.norm(dim=-1).mean().item())
        out = orig_run(h, *a, **k)
        rec.setdefault('out',[]).append(out[0].norm(dim=-1).mean().item())
        return out
    model._run_layers = spy2
    torch.manual_seed(1234)
    paths=[sample_path(pcap, rng, mcfg.n_layers) for _ in range(n_rounds)]
    with torch.no_grad():
        _, aux = model(x, paths, x)
    emb = model.embed.weight.norm(dim=-1)
    emb0 = model.embed(x).norm(dim=-1)
    print(f"--- {run} transp={transp} mix={mix}")
    print(f"  emb row norm mean={emb.mean():.3f}  |E[t]| mean={emb0.mean():.3f} max={emb0.max():.3f}  norm_cap={mcfg.norm_cap}")
    print("  layer-input norms :", [round(v,3) for v in rec.get('in',[])])
    print("  layer-output norms:", [round(v,3) for v in rec.get('out',[])])
    print("  mixture-reentry norms (mean,max):", [(round(a,3),round(b,3)) for a,b in rec.get('mixre',[])])
    # round-0 vs round-1 node-0 logits: identical?
    r0 = aux['rounds'][0][0]['logits']; r1 = aux['rounds'][1][0]['logits']
    print(f"  r0 vs r1 node0 logits: max|d|={(r0-r1).abs().max().item():.4f}  cos={torch.nn.functional.cosine_similarity(r0.flatten(),r1.flatten(),dim=0).item():.5f}")

for mix in (False, True):
    probe('C1M','direct',mix)
for mix in (False, True):
    probe('C1M','soft',mix)
