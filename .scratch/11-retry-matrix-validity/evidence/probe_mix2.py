import sys, os, json, random, torch, torch.nn.functional as F
sys.path.insert(0, os.getcwd())
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import load_enwik8_full, batch
from pathlm.model import PathLM

def probe(run, transp, mix, n_rounds=4):
    cfg = json.load(open(f'configs/{run}.json'))
    mcfg = ModelConfig(**cfg['model']); pcap = PathConfig(**cfg['path'])
    pcap.transport = transp; pcap.reentry_mix = mix
    pcap.p_retry=0.0; pcap.p_token_retry=0.0; pcap.w_dense_exit=0.0
    pcap.corrupt_wrong=0.15; pcap.corrupt_mask=0.0
    model = PathLM(mcfg, pcap, 206).cuda()
    model.load_state_dict(torch.load(f'.scratch/10-gap-sweep/evidence/{run}/model.pt', map_location='cuda')); model.eval()
    _, ev, V = load_enwik8_full('.tmp/enwik8','data/enwik8_full.npz')
    g = torch.Generator().manual_seed(1); x,_ = batch(ev, 16, 512, g); x=x.cuda()
    rng = random.Random(0); rec=[]
    orig = model._mixture_reentry
    def spy(d):
        out = orig(d)
        rec.append(out.detach())
        return out
    model._mixture_reentry = spy
    torch.manual_seed(1234)
    paths=[sample_path(pcap, rng, mcfg.n_layers) for _ in range(n_rounds)]
    with torch.no_grad(): _, aux = model(x, paths, x)
    E = model.embed.weight
    print(f"--- {run} {transp} mix={mix}")
    # clean-pass reference: norm of a depth-12 latent vs embedding
    for i, st in enumerate(rec):
        n = st.norm(dim=-1)
        # how much of the re-entry state lies in span(E)?  proj = (h@E.T)@E
        proj = (st @ E.T) @ E
        ratio = (proj.norm(dim=-1)/n.clamp_min(1e-6)).mean().item()
        print(f"  reentry r{i+1}: norm mean={n.mean():.2f}  ||proj_E||/||h||={ratio:.3f}")
    print(f"  final-round layer-output norm (what the model itself emits): "
          f"{aux['rounds'][-1][0]['latent'].norm(dim=-1).mean():.2f}")

probe('C1M','direct',False)
probe('C1M','direct',True)
probe('C1M','linear',False)
probe('C1M','linear',True)
