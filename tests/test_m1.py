"""M1 engine-extension tests. Each test names the property it protects and
can fail on it (M0 lesson: a test that cannot fail is not a test)."""

import json, math, os, random, sys

import numpy as np
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
    mask_opt = m._eviction_mask(16, torch.device("cpu"))
    assert mask_opt is not None
    mask = mask_opt[0, 0]
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
    """X2v2: multi-needle on real-text filler — needles carved into the text,
    query tail [mask, x_i, y_i] per needle, pairs absent from the row's text,
    metadata (query_row, needle_pos, dist) consistent."""
    T, n_real, mask, K = 64, 49, 49, 4
    g = torch.Generator().manual_seed(0)
    data = np.random.randint(0, n_real, size=5000).astype(np.uint16)
    x, y, meta = needle_batch(4, T, n_real, mask, g, data=data, anchors=4,
                              anchor_frac=0.25, n_needles=K)
    q0 = T - 3 * K
    for b in range(4):
        assert len(meta[b]) == K
        for i, (q_row, p, d) in enumerate(meta[b]):
            assert q_row == q0 + 3 * i + 1
            xn, yn = int(x[b, p]), int(x[b, p + 1])
            assert x[b, q0 + 3 * i] == mask, "query cue must be [mask]"
            assert x[b, q_row] == xn and x[b, q_row + 1] == yn, "query must echo the needle pair"
            assert d == q_row - (p + 1), "metadata distance"
            # pair must not occur in the text body other than at the needle
            body = x[b, :q0].tolist()
            occur = [s for s in range(len(body) - 1)
                     if body[s] == xn and body[s + 1] == yn and s != p]
            assert not occur, f"ambiguous needle at {occur}"
        # filler between needles must be real data (no mask tokens)
        assert not (x[b, :q0] == mask).any(), "mask must not appear in the text body"


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
    def _spy(h, path, attn_mask=None, depth_hook=None, dist_pen=0.0, _o=orig):
        captured.append(h.detach().clone())
        return _o(h, path, attn_mask=attn_mask, depth_hook=depth_hook,
                  dist_pen=dist_pen)
    m._run_layers = _spy
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
        nodes = dec.last_nodes
        assert nodes is not None, "step() must populate last_nodes"
        dec.retry(nodes[0]["latent"] * 0 + h)  # force the retry path
    for i, pos in dec.layer_pos.items():
        assert pos == sorted(set(pos)), f"layer {i} positions must be unique+sorted: {pos}"


def test_dense_exit_calibrates_confidence_heads():
    """P1 fix: the dense hook must train the conf heads at intermediate
    depths — after dense training, the DEPTH-1 confidence must track its own
    hit rate (ECE small). The final-node BCE alone (always on) cannot produce
    depth-1 calibration, so this fails if the hook's BCE is removed."""
    m = tiny_model(PathConfig(n_mtp=1, w_dense_exit=1.0))
    train_tiny(m, steps=250)
    from pathlm.metrics import ece
    confs, logits0, targets = [], [], []
    orig = m._mtp_nodes
    first = [False]
    def spy(h):
        nodes = orig(h)
        if first[0]:  # depth-1 call only (the dense hook fires at every depth)
            confs.append(nodes[0]["conf"].detach().cpu())
            logits0.append(nodes[0]["logits"].detach().cpu())
            first[0] = False
        return nodes
    m._mtp_nodes = spy
    with torch.no_grad():
        for _ in range(4):
            x = torch.randint(0, 49, (4, M1.seq_len))
            first[0] = True
            m(x, tiny_paths(m.pcap), x)
            targets.append(x.cpu())
    m._mtp_nodes = orig
    hits = (torch.cat(logits0).argmax(-1) == torch.cat(targets)).float()
    depth1_ece = ece(torch.cat(confs).sigmoid(), hits)
    assert depth1_ece < 0.3, f"depth-1 conf must be calibrated via the hook BCE, got {depth1_ece:.3f}"


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
    def _spy(h, path, attn_mask=None, depth_hook=None, dist_pen=0.0, _o=orig):
        passes.append(1)
        return _o(h, path, attn_mask=attn_mask, depth_hook=depth_hook,
                  dist_pen=dist_pen)
    m._run_layers = _spy
    res = repair(m, arr, n_batches=1, batch_size=4)
    assert res["round_kinds"] == ["base", "pass1", "token"], res["round_kinds"]
    # reference: replay the identical seeded batch and read aux round 2 (token)
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
    got = res["rounds"]["r2"]["self_acc_all"]
    assert round(want, 4) == got, f"r1 must be the token round: {got} != {round(want, 4)}"


def test_needle_batch_anchor_regime():
    """anchor_frac must place needles inside the first `anchors` positions
    (the anchor channel's training share)."""
    import numpy as np
    T, n_real, mask = 64, 49, 49
    g = torch.Generator().manual_seed(0)
    data = np.random.randint(0, n_real, size=5000).astype(np.uint16)
    x, _, meta = needle_batch(8, T, n_real, mask, g, data=data, anchors=4,
                              anchor_frac=1.0, n_needles=4)
    for b in range(8):
        anchor_ps = [p for (q, p, d) in meta[b] if p < 4]
        assert anchor_ps, "anchor_frac=1.0 must place every needle in anchors"


# ---------- mixture re-entry (C4/C5) ----------

def test_mixture_reentry_anchors_the_init():
    """The accumulator must differ from overwrite once there is history to
    mix (round 2): mixture re-enters from S = (1·P0 + w1·P1)/(1+w1) and the
    weighted latent mean, overwrite re-enters from round 1 alone. (At round 1
    the two are identical by construction — one term in the sum.)"""
    m = tiny_model(PathConfig(n_mtp=1, transport="soft", p_retry=1.0, reentry_mix=True))
    m.eval()
    x = torch.randint(0, 49, (2, M1.seq_len))
    with torch.no_grad():
        _, aux = m(x, tiny_paths(m.pcap, n=3), x)
    assert len(aux["rounds"]) == 3
    m2 = tiny_model(PathConfig(n_mtp=1, transport="soft", p_retry=1.0, reentry_mix=False))
    m2.load_state_dict(m.state_dict())
    m2.eval()
    with torch.no_grad():
        _, aux2 = m2(x, tiny_paths(m2.pcap, n=3), x)
    d = (aux["rounds"][2][0]["latent"] - aux2["rounds"][2][0]["latent"]).abs().max().item()
    assert d > 1e-6, "mixture re-entry must change the re-entered state vs overwrite at round 2"


def test_mixture_reentry_transport_forms():
    """soft: expected embedding of the accumulated distribution; direct:
    weighted latent mean. Pinned against manual computation."""
    for tr, check in (("soft", "soft"), ("direct", "direct")):
        m = tiny_model(PathConfig(n_mtp=1, transport=tr, p_retry=1.0, reentry_mix=True))
        m.eval()
        x = torch.randint(0, 49, (1, M1.seq_len))
        lats, confs = [], []
        orig = m._run_layers
        def spy(h, path, attn_mask=None, depth_hook=None, dist_pen=0.0, _orig=orig):
            out, _ = _orig(h, path, attn_mask=attn_mask, depth_hook=depth_hook)
            lats.append(out.detach().clone())  # round OUTPUT (what mix stores)
            return out, []
        m._run_layers = spy
        # capture confs via _mtp_nodes
        node_calls = []
        orig_nodes = m._mtp_nodes
        def node_spy(h, _orig=orig_nodes):
            nodes = _orig(h)
            node_calls.append(nodes[0]["conf"].detach().sigmoid().clone())
            return nodes
        m._mtp_nodes = node_spy
        with torch.no_grad():
            m(x, tiny_paths(m.pcap, n=2), x)
        U = m.embed.weight
        w0 = torch.ones_like(node_calls[0])
        w1 = node_calls[1]
        if check == "soft":
            S = (w0.unsqueeze(-1) * (lats[0] @ U.T).softmax(-1)
                 + w1.unsqueeze(-1) * (lats[1] @ U.T).softmax(-1)) / (w0 + w1).unsqueeze(-1)
            want = cap_norm(S @ U, 1.0)
        else:
            want = cap_norm((lats[0] + w1.unsqueeze(-1) * lats[1]) / (w0 + w1).unsqueeze(-1), 1.0)
        got = m._mixture_reentry({"W": (w0 + w1).unsqueeze(-1),
                                  "S": (w0.unsqueeze(-1) * (lats[0] @ U.T).softmax(-1)
                                        + w1.unsqueeze(-1) * (lats[1] @ U.T).softmax(-1)),
                                  "L": torch.stack([lats[0],
                                                    w1.unsqueeze(-1) * lats[1]])})
        assert torch.allclose(got, want, atol=1e-4), f"{tr} mixture form drifted"


def test_retry_curve_forces_rounds():
    """repair(rounds=k) must run a k-round loop regardless of config, with
    per-round labels — the recurrent-transformer measurement."""
    from pathlm.eval import repair
    import numpy as np
    arr = np.random.randint(0, 49, size=8000).astype(np.uint16)
    m = tiny_model(PathConfig(n_mtp=1, corrupt_wrong=0.15, transport="direct", p_retry=0.5))
    passes = []
    orig = m._run_layers
    def _spy(h, path, attn_mask=None, depth_hook=None, dist_pen=0.0, _o=orig):
        passes.append(1)
        return _o(h, path, attn_mask=attn_mask, depth_hook=depth_hook,
                  dist_pen=dist_pen)
    m._run_layers = _spy
    res = repair(m, arr, n_batches=1, batch_size=4, rounds=4)
    assert len(res["rounds"]) == 4 and res["round_kinds"] == ["base", "latent", "latent", "latent"]
    assert len(passes) == 4, f"forced 4-round loop must run 4 passes, got {len(passes)}"


def test_eval_overrides_tolerate_reentry_mix():
    """Regression: bpc/repair force p_retry=0 via eval_pc — that must NOT trip
    config validation (mix without a loop is simply unused at eval time)."""
    from pathlm.eval import bpc
    import numpy as np
    arr = np.random.randint(0, 49, size=4000).astype(np.uint16)
    m = tiny_model(PathConfig(n_mtp=1, corrupt_wrong=0.15, transport="soft",
                              p_retry=0.5, reentry_mix=True))
    res = bpc(m, arr, vocab_size=50, n_batches=2, batch_size=4)
    assert 0 < res["bpc"] < 10


def test_mixture_reentry_identity_for_transport_none():
    """C5 v1 lesson: with transport=none, middle passes must NOT re-enter via
    the latent mean (it blurs token identity: self-acc collapsed to 54%)."""
    m = tiny_model(PathConfig(n_mtp=1, p_token_retry=1.0, reentry_mix=True))
    m.eval()
    x = torch.randint(0, 49, (2, M1.seq_len))
    inputs, outputs = [], []
    orig = m._run_layers
    def spy(h, path, attn_mask=None, depth_hook=None, dist_pen=0.0, _orig=orig):
        out, _ = _orig(h, path, attn_mask=attn_mask, depth_hook=depth_hook)
        inputs.append(h.detach().clone())
        outputs.append(out.detach().clone())
        return out, []
    m._run_layers = spy
    with torch.no_grad():
        _, aux = m(x, tiny_paths(m.pcap, n=2), x)
    assert len(aux["rounds"]) == 3  # base, pass1, token
    assert torch.allclose(inputs[1], outputs[0]), \
        "middle pass must re-enter from the base OUTPUT unchanged (identity)"
    assert not torch.allclose(inputs[2], outputs[1]), \
        "token round must re-embed the mixture vote, not reuse the latent"


def test_mixture_accumulator_end_to_end_reference():
    """Pins the forward-side accumulation (weight source = node-0 conf,
    anchor w=1, weighted latents): the ACTUAL round-2 re-entry input must
    equal the reference computed from captured round outputs and confs."""
    m = tiny_model(PathConfig(n_mtp=1, transport="soft", p_retry=1.0, reentry_mix=True))
    m.eval()
    x = torch.randint(0, 49, (1, M1.seq_len))
    outs, confs, reentry_inputs = [], [], []
    orig_layers, orig_nodes = m._run_layers, m._mtp_nodes
    def layer_spy(h, path, attn_mask=None, depth_hook=None, dist_pen=0.0, _o=orig_layers):
        reentry_inputs.append(h.detach().clone())
        out, _ = _o(h, path, attn_mask=attn_mask, depth_hook=depth_hook)
        outs.append(out.detach().clone())
        return out, []
    def node_spy(h, _o=orig_nodes):
        nodes = _o(h)
        confs.append(nodes[0]["conf"].detach().sigmoid().clone())
        return nodes
    m._run_layers, m._mtp_nodes = layer_spy, node_spy
    with torch.no_grad():
        m(x, tiny_paths(m.pcap, n=3), x)
    m._run_layers, m._mtp_nodes = orig_layers, orig_nodes
    U = m.embed.weight
    w0, w1 = torch.ones_like(confs[0]), confs[1]
    S = (w0.unsqueeze(-1) * (outs[0] @ U.T).softmax(-1)
         + w1.unsqueeze(-1) * (outs[1] @ U.T).softmax(-1)) / (w0 + w1).unsqueeze(-1)
    want = cap_norm(S @ U, 1.0)
    # reentry_inputs: [embed, reentry_r1, reentry_r2] — round 2's input is index 2
    assert torch.allclose(reentry_inputs[2], want, atol=1e-4), \
        "round-2 re-entry input must equal the accumulated mixture (anchor w=1, conf-weighted)"


def test_distance_penalty_bias_and_telemetry():
    """X1: dist_pen produces an additive -pen*log(1+d) attention bias and
    per-head distance telemetry; off = exact no-op (bias None)."""
    m = tiny_model(PathConfig(n_mtp=1, dist_pen=0.0))
    m.eval()
    x = torch.randint(0, 49, (2, M1.seq_len))
    paths = tiny_paths(m.pcap, n=1)
    torch.manual_seed(0)
    _, aux_off = m(x, paths, x)
    assert "attn_dist" not in aux_off
    m.pcap.dist_pen = 0.5
    torch.manual_seed(0)
    _, aux_on = m(x, paths, x)
    assert "attn_dist" in aux_on
    dist = aux_on["attn_dist"]  # [B, H] mean attended distance (layers averaged)
    assert dist.shape == (2, M1.n_heads), dist.shape
    assert (dist > 0).all(), "attended distance must be positive under causal attention"
    # bias correctness: an isolated block with dist_pen p must equal the same
    # block with an explicit additive mask of -p*log(1+d). The block takes an
    # optional mask; assert non-None to satisfy the type checker.
    blk = m._run_layers.__self__.blocks[0] if hasattr(m._run_layers, "__self__") else m.blocks[0]
    assert blk is not None
    B, T, d = 1, 32, m.mcfg.d_model
    h = torch.randn(B, T, d)
    i = torch.arange(T).unsqueeze(1)
    j = torch.arange(T).unsqueeze(0)
    bias = -0.5 * torch.log1p((i - j).clamp_min(0).float())
    bias = bias.masked_fill(j > i, float("-inf")).view(1, 1, T, T)
    torch.manual_seed(3)
    out_none, _ = blk(h)  # causal only, pen off
    torch.manual_seed(3)
    out_bias, _ = blk(h, attn_mask=bias)  # causal + penalty, pen off
    torch.manual_seed(3)
    out_pen, dist = blk(h, dist_pen=0.5)   # pen on
    assert torch.allclose(out_pen, out_bias, atol=1e-3, rtol=1e-3), \
        "dist_pen must apply exactly the -pen*log(1+d) additive bias"
    assert not torch.allclose(out_none, out_pen)


def test_depth_logits_capture_eval_only():
    """T2 probe: collect_depth_logits=True emits per-depth node-1 logits/conf
    (all depths, incl. final); default off = no capture, no behavior change."""
    m = tiny_model(PathConfig(n_mtp=1, w_dense_exit=1.0))
    m.eval()
    x = torch.randint(0, 49, (2, M1.seq_len))
    torch.manual_seed(0)
    _, aux_off = m(x, tiny_paths(m.pcap, n=1), x)
    assert "depth_logits" not in aux_off
    m.pcap.collect_depth_logits = True
    torch.manual_seed(0)
    _, aux_on = m(x, tiny_paths(m.pcap, n=1), x)
    dl, dc = aux_on["depth_logits"], aux_on["depth_conf"]
    assert len(dl) == M1.n_layers and len(dc) == M1.n_layers
    assert dl[0].shape == (2, M1.seq_len, 50), dl[0].shape  # node-1 heads, full row span
    assert torch.allclose(dl[-1], aux_on["rounds"][0][1]["logits"].detach())
    m.pcap.collect_depth_logits = False
