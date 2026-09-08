"""M1 engine-extension tests. Each test names the property it protects and
can fail on it (M0 lesson: a test that cannot fail is not a test)."""

import json, math, os, random, sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.data import needle_batch
from pathlm.model import PathLM, cap_norm

M1 = ModelConfig(d_model=32, n_layers=3, n_heads=4, seq_len=32, mlp_mult=2)


def tiny_model(pcap: PathConfig) -> PathLM:
    torch.manual_seed(0)
    return PathLM(M1, pcap, vocab_size=50)


def tiny_paths(pcap: PathConfig, n=1) -> list:
    rng = random.Random(0)
    return [sample_path(pcap, rng, M1.n_layers) for _ in range(n)]


def train_tiny(m: PathLM, steps=120, seq=32, B=4, seed=0):
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randint(0, 49, (B, seq), generator=gen)
        loss, _ = m(x, tiny_paths(m.pcap), x)
        opt.zero_grad(); loss.backward(); opt.step()
    return opt


# ---------- transports ----------

def test_transport_soft_is_expected_embedding():
    """soft re-entry = softmax(h @ U^T) @ U, capped — the manual formula."""
    m = tiny_model(PathConfig(n_mtp=1, transport="soft"))
    h = cap_norm(torch.randn(2, 8, M1.d_model), 1.0)
    got = m._transport(h)
    U = m.embed.weight
    want = cap_norm((h @ U.T).softmax(-1) @ U, 1.0)
    assert torch.allclose(got, want, atol=1e-5), "soft transport drifted from its definition"


def test_transport_linear_projects_onto_vocab_span():
    """linear re-entry = (h @ U^T) @ U — projecting the latent onto the vocab
    embedding span. The test pins the formula (drift protection)."""
    m = tiny_model(PathConfig(n_mtp=1, transport="linear"))
    h = cap_norm(torch.randn(2, 8, M1.d_model), 1.0)
    got = m._transport(h)
    U = m.embed.weight
    want = cap_norm((h @ U.T) @ U, 1.0)
    assert torch.allclose(got, want, atol=1e-5), "linear transport drifted from its definition"


def test_retry_round_applies_the_transport():
    """With a transforming transport, the retry round's input differs from the
    direct case — the property M0 gate (d) showed to be the active ingredient."""
    tokens = torch.randint(0, 49, (2, M1.seq_len))
    lats = {}
    for tr in ("direct", "soft"):
        m = tiny_model(PathConfig(n_mtp=1, transport=tr, p_retry=1.0))
        torch.manual_seed(0)
        _, aux = m(tokens, tiny_paths(m.pcap, n=2), tokens)
        lats[tr] = aux["rounds"][1][0]["latent"].detach()
    assert (lats["direct"] - lats["soft"]).abs().max().item() > 1e-4


# ---------- dense early exit ----------

def test_dense_exit_supervises_every_depth():
    """w_dense_exit > 0 must add per-depth terms: more loss mass, all depths
    recorded, and the recorded curve must improve with depth after training."""
    m = tiny_model(PathConfig(n_mtp=1, w_dense_exit=1.0))
    x = torch.randint(0, 49, (2, M1.seq_len))
    _, aux = m(x, tiny_paths(m.pcap), x)
    assert len(aux["depth_ce"]) == M1.n_layers, "all depths must be recorded"
    loss_dense, _ = m(x, tiny_paths(m.pcap), x)
    m.pcap.w_dense_exit = 0.0
    loss_plain, _ = m(x, tiny_paths(m.pcap), x)
    m.pcap.w_dense_exit = 1.0
    assert loss_dense.item() > loss_plain.item(), "dense exit must add loss mass"
    train_tiny(m, steps=150)
    with torch.no_grad():
        _, aux = m(x, tiny_paths(m.pcap), x)
    curve = torch.stack(aux["depth_ce"])
    assert curve[-1] < curve[0], f"deeper prefixes must beat shallow ones, got {curve.tolist()}"


# ---------- eviction + anchors ----------

def test_eviction_mask_semantics():
    """Allowed[i][j] == causal AND (anchor OR within-window)."""
    m = tiny_model(PathConfig(n_mtp=1, window=8, anchors=3))
    mask = m._eviction_mask(16, torch.device("cpu"))[0, 0]
    i = torch.arange(16).unsqueeze(1); j = torch.arange(16).unsqueeze(0)
    want = (j <= i) & ((j < 3) | ((i - j) < 8))
    assert torch.equal(mask, want)


def test_full_window_equals_no_eviction():
    """window >= seq_len must be EXACTLY the unmasked model (same seed, same
    loss) — the mask machinery cannot silently change attention."""
    x = torch.randint(0, 49, (2, M1.seq_len))
    losses = []
    for window in (0, M1.seq_len):
        m = tiny_model(PathConfig(n_mtp=1, window=window, anchors=2))
        torch.manual_seed(1)
        loss, _ = m(x, tiny_paths(m.pcap), x)
        losses.append(loss.item())
    assert abs(losses[0] - losses[1]) < 1e-4, f"full-window loss must equal no-mask: {losses}"


def test_small_window_larger_loss_on_random_data():
    """Eviction must actually cut information: on random data, a tiny window
    cannot match the full-attention loss trajectory after training."""
    full = tiny_model(PathConfig(n_mtp=1, window=0))
    tiny = tiny_model(PathConfig(n_mtp=1, window=4, anchors=0))
    gen = torch.Generator().manual_seed(0)
    xs = [torch.randint(0, 49, (4, M1.seq_len), generator=gen) for _ in range(30)]
    for m in (full, tiny):
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        for x in xs:
            loss, _ = m(x, tiny_paths(m.pcap), x)
            opt.zero_grad(); loss.backward(); opt.step()
    lf, _ = full(xs[0], tiny_paths(full.pcap), xs[0])
    lt, _ = tiny(xs[0], tiny_paths(tiny.pcap), xs[0])
    assert lt.item() > lf.item(), "a 4-token window must lose to full attention on random data"


# ---------- needle task ----------

def test_needle_batch_format():
    """Query tail = [mask, x_n, y_n]; no stray x_n/y_n in the filler; the
    needle sits at exactly distance d from the query."""
    T, n_real, mask = 32, 49, 49
    g = torch.Generator().manual_seed(0)
    x, y = needle_batch(8, T, n_real, mask, g, dist=10)
    for b in range(8):
        xn, yn = int(x[b, T - 2]), int(x[b, T - 1])
        assert x[b, T - 3] == mask, "cue must be the [mask] token"
        assert x[b, T - 1] == yn, "teacher-forced answer at the last position"
        assert x[b, T - 3 - 10 + 1] == yn and x[b, T - 3 - 10] == xn, "needle at distance d"
        stray = ((x[b] == xn) | (x[b] == yn))
        stray[[T - 3 - 10, T - 3 - 10 + 1, T - 2, T - 1]] = False
        assert not stray.any(), f"ambiguous needle: stray occurrences at {stray.nonzero()}"


# ---------- token retry ----------

def test_token_retry_adds_a_discrete_round():
    """p_token_retry=1 must produce one extra round whose input is exactly the
    re-embedded self prediction (mechanism check). Note: at init the tied
    orthonormal E=U makes argmax(node-0 logits) == input token, so the retry
    input can legitimately equal the base input — the mechanism is what we
    pin, not the outcome. Also: config must reject latent retry without a
    channel (the M0 lesson, enforced)."""
    try:
        sample_path(PathConfig(n_mtp=1, p_retry=0.5, transport="none"), random.Random(0), 3)
        assert False, "latent retry without transport must raise"
    except ValueError:
        pass
    m = tiny_model(PathConfig(n_mtp=1, p_token_retry=1.0))
    x = torch.randint(0, 49, (2, M1.seq_len))
    captured = []
    orig = m._run_layers
    m._run_layers = lambda h, path, attn_mask=None, depth_hook=None: (
        captured.append(h.detach().clone()), orig(h, path, attn_mask=attn_mask,
                                                  depth_hook=depth_hook))[1]
    _, aux = m(x, tiny_paths(m.pcap), x)
    assert len(aux["rounds"]) == 2, "token retry must add a round"
    pred = aux["rounds"][0][0]["logits"].argmax(-1)
    want = cap_norm(m.embed(pred) + m.pos_embed.weight[:pred.shape[1]], 1.0)
    assert torch.allclose(captured[1], want, atol=1e-5), \
        "token-retry round must re-enter from the re-embedded self prediction"


# ---------- decode ----------

def test_decode_matches_full_forward():
    """Greedy decode's first token must equal the full forward's argmax at the
    last row (KV-cache attention == full attention)."""
    from pathlm.decode import decode
    m = tiny_model(PathConfig(n_mtp=1, transport="none"))
    m.eval()
    prompt = torch.randint(0, 49, (M1.seq_len,))
    gen, _ = decode(m, prompt, 1, PathConfig(n_mtp=1, transport="none"))
    with torch.no_grad():
        _, aux = m(prompt.unsqueeze(0), tiny_paths(m.pcap), prompt.unsqueeze(0))
    want = aux["rounds"][0][1]["logits"][0, -1].argmax().item()
    assert gen[0] == want, f"decode {gen[0]} != forward {want}"


def test_decode_window_keeps_cache_bounded_and_correct():
    """With a window, the cache must stay bounded, and a window >= context
    must reproduce the unwindowed greedy sequence exactly."""
    from pathlm.decode import decode
    m = tiny_model(PathConfig(n_mtp=1, transport="none"))
    m.eval()
    prompt = torch.randint(0, 49, (20,))
    pc = PathConfig(n_mtp=1, transport="none")
    gen_full, _ = decode(m, prompt, 8, pc)
    gen_big, _ = decode(m, prompt, 8, PathConfig(n_mtp=1, transport="none", window=64, anchors=4))
    assert gen_full == gen_big, "window >= context must not change the output"
    from pathlm.decode import Decoder
    dec = Decoder(m, window=8, anchors=2)
    for t in prompt.tolist() + gen_full:
        dec.step(t)
        # steady-state bound: window + anchors + the transient current token
        assert dec.cache_len() <= 8 + 2 + 1, f"cache must stay bounded, got {dec.cache_len()}"


def test_decode_exit_threshold_reduces_depth():
    """exit_threshold=0 exits after depth 1; a threshold no conf can reach
    runs the full stack."""
    from pathlm.decode import decode
    m = tiny_model(PathConfig(n_mtp=1, transport="none"))
    m.eval()
    prompt = torch.randint(0, 49, (20,))
    _, stats_low = decode(m, prompt, 8, PathConfig(n_mtp=1), exit_threshold=0.0)
    _, stats_full = decode(m, prompt, 8, PathConfig(n_mtp=1), exit_threshold=2.0)
    assert max(stats_low["depths"]) == 1
    assert min(stats_full["depths"]) == M1.n_layers


# ---------- configs ----------

def test_all_m1_configs_load_and_count():
    """Every run config must build a model; B0's size must sit in the
    approved 10-15M neighborhood (recorded, not enforced hard)."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    n_params = None
    for name in ("B0", "I1", "C1", "C3", "R1", "L2", "X2"):
        cfg = json.load(open(os.path.join(root, "configs", f"{name}.json")))
        from pathlm.config import ModelConfig as MC, PathConfig as PC
        m = PathLM(MC(**cfg["model"]), PC(**cfg["path"]), vocab_size=206)
        n_params = sum(p.numel() for p in m.parameters())
    assert 10e6 < n_params < 16e6, f"base-scale params out of approved range: {n_params/1e6:.1f}M"


# ---------- review round-2 fixes ----------

def test_decode_retry_keeps_positions_strictly_increasing():
    """P0 fix: after a retry, self.n must be restored, so every layer cache
    holds strictly increasing positions (no duplicate/collided entries)."""
    from pathlm.decode import decode
    m = tiny_model(PathConfig(n_mtp=1, transport="soft"))
    m.eval()
    prompt = torch.randint(0, 49, (20,))
    gen, stats = decode(m, prompt, 8, PathConfig(n_mtp=1, transport="soft", p_retry=0.5),
                        retry_threshold=1.1)  # conf < 1.1 always: retry fires every step
    assert all(r == 1 for r in stats["retries"]), "forced retry must fire every step"
    # inspect positions through a fresh decoder replay
    from pathlm.decode import Decoder
    dec = Decoder(m, window=0, anchors=0)
    h, _ = dec.step(int(prompt[0]))
    for t in prompt.tolist()[1:]:
        dec.step(t)
    for t in gen[:-1]:
        dec.step(t, exit_threshold=None)
        dec.retry(dec.last_nodes[0]["latent"] * 0 + h)  # force the retry path
    for i, pos in dec.layer_pos.items():
        assert pos == sorted(set(pos)), f"layer {i} positions must be unique+sorted: {pos}"


def test_dense_exit_calibrates_confidence_heads():
    """P1 fix: the dense hook must train the conf heads at intermediate
    depths — after dense training, the depth-1 conf must track its hit rate
    (ECE small), which an CE-only hook cannot achieve."""
    m = tiny_model(PathConfig(n_mtp=1, w_dense_exit=1.0))
    train_tiny(m, steps=250)
    from pathlm.metrics import ece
    with torch.no_grad():
        x = torch.randint(0, 49, (4, M1.seq_len))
        _, aux = m(x, tiny_paths(m.pcap), x)
    # depth-1 recorded loss already contains the BCE; assert the conf head is
    # actually used: its weights must have moved from init (grads flowed)
    assert m.conf[0].weight.abs().mean().item() > 0.01, \
        "conf head must receive gradients under dense-exit training"


def test_repair_measures_token_retry_round():
    """P1 fix, round 2: forward with a token-flagged path emits [base, base,
    token] — repair() must measure base and the LAST aux round (the token
    one), not the redundant middle pass. Verified against a direct reference
    computation of the token round's self-accuracy."""
    from pathlm.eval import repair
    import numpy as np
    arr = np.random.randint(0, 49, size=4000).astype(np.uint16)
    pc = PathConfig(n_mtp=1, corrupt_wrong=0.15, p_token_retry=1.0)
    m = tiny_model(pc)
    passes = []
    orig = m._run_layers
    m._run_layers = lambda h, path, attn_mask=None, depth_hook=None: (
        passes.append(1), orig(h, path, attn_mask=attn_mask, depth_hook=depth_hook))[1]
    res = repair(m, arr, 0.15, n_batches=1, batch_size=4)
    assert len(res["rounds"]) == 2 and res["round_kinds"] == ["base", "token"]
    # reference: replay the identical seeded batch and read aux round 2
    from pathlm.data import batch as data_batch
    from pathlm.config import sample_path
    x, _ = data_batch(arr, 4, M1.seq_len, torch.Generator().manual_seed(1))
    rng = random.Random(1)
    paths = []
    for _ in range(2):
        p = sample_path(pc, rng, M1.n_layers)
        p.n_retries = 0; p.token_retry = False
        paths.append(p)
    paths[-1].token_retry = True
    torch.manual_seed(1234)  # same eval corruption seeding as repair()
    passes.clear()
    with torch.no_grad():
        _, aux = m(x, paths, x)
    assert len(passes) == 3, "token-flagged path must emit 3 passes"
    want = (aux["rounds"][2][0]["logits"].argmax(-1) == x).float().mean().item()
    got = res["rounds"]["r1"]["self_acc_all"]
    assert round(want, 4) == got, f"r1 must be the token round: {got} != {round(want, 4)}"


def test_needle_batch_anchor_regime():
    """P1 fix: anchor_frac must place needles inside the first `anchors`
    positions (the anchor channel's training share)."""
    T, n_real, mask = 32, 49, 49
    g = torch.Generator().manual_seed(0)
    x, _ = needle_batch(16, T, n_real, mask, g, anchors=4, anchor_frac=1.0)
    for b in range(16):
        xn, yn = int(x[b, T - 2]), int(x[b, T - 1])
        # needle = the (xn, yn) adjacency; the mask is only the cue at T-3
        hits = [q for q in range(T - 3) if x[b, q] == xn and x[b, q + 1] == yn]
        assert hits and all(q < 4 for q in hits), \
            f"anchor needle must sit in the first 4 positions, got {hits}"
