"""Fast CPU tests for the M0 PathLM skeleton. Run: python -m pytest tests/ -q"""

import os, random, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.metrics import ece, ensemble
from pathlm.model import PathLM


def tiny_model(pcap=None):
    pcap = pcap or PathConfig(n_mtp=2, transport="direct", p_retry=0.5)
    torch.manual_seed(0)
    return PathLM(ModelConfig(d_model=32, n_layers=2, n_heads=2, seq_len=16), pcap, vocab_size=50)


def tiny_paths(pcap, n=1, seed=0):
    rng = random.Random(seed)
    return [sample_path(pcap, rng, 2) for _ in range(n)]


def test_causal_masking_blocks_future_targets():
    """A target that depends only on a FUTURE token must stay at chance for a
    causal model — this is the property the mask guarantees."""
    m = tiny_model(PathConfig(n_mtp=0))
    B, T = 8, 16
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    for _ in range(60):
        x = torch.randint(0, 49, (B, T), generator=gen)   # fresh data: nothing to memorize
        shifted = torch.roll(x, -1, dims=1)               # target[i] = x[i+1]: future-only
        paths = tiny_paths(m.pcap)
        loss, _ = m(x, paths, shifted)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, aux = m(x, paths, shifted)
    hit = aux["rounds"][0][0]["hit"].float().mean().item()
    assert hit < 0.15, f"node0 reached {hit:.2f} on a future-only target — mask is leaking"


def test_corruption_leaves_targets_untouched():
    """Corruption mutates only the internal input; the tokens argument must be
    unchanged, and rate 1.0 corrupts every position."""
    m = tiny_model(PathConfig(corrupt_wrong=1.0, n_mtp=1, transport="none"))
    tokens = torch.randint(0, 49, (8, 16))
    before = tokens.clone()
    _, aux = m(tokens, tiny_paths(m.pcap), tokens)
    assert torch.equal(tokens, before), "forward must not mutate its input"
    assert aux["corrupt_mask"].all()


def test_node_targets_are_shift_correctly():
    """Node k at row i must predict the clean token t_{i+k}: on a 2-cycle
    sequence (A B A B ...), node0 must reproduce the input, node1 must predict
    the opposite token, node2 the same token. An off-by-one shift fails this."""
    m = tiny_model(PathConfig(n_mtp=2, transport="none"))
    B, T = 4, 16
    tokens = torch.tensor([[0, 1] * (T // 2)] * B)
    paths = tiny_paths(m.pcap)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(120):
        loss, _ = m(tokens, paths, tokens)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, aux = m(tokens, paths, tokens)
    for k in range(3):
        pred = aux["rounds"][0][k]["logits"].argmax(-1)[:, :T - k]
        expected = tokens[:, k:]
        acc = (pred == expected).float().mean().item()
        assert acc > 0.95, f"node{k} predicts the wrong offset (acc {acc:.2f})"


def test_confidence_head_tracks_the_argmax_event_under_uncertainty():
    """Confidence BCE trains sigmoid(conf) toward P(argmax == target) — with
    fresh random data each step the model stays uncertain, so this checks
    tracking under uncertainty, not just saturation on a trivial sequence."""
    m = tiny_model(PathConfig(n_mtp=1, transport="none", corrupt_wrong=0.5))
    B, T = 4, 16
    paths = tiny_paths(m.pcap)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    for _ in range(60):
        x = torch.randint(0, 49, (B, T), generator=gen)
        loss, _ = m(x, paths, x)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        x = torch.randint(0, 49, (B, T), generator=gen)
        _, aux = m(x, paths, x)
    hit = aux["rounds"][0][0]["hit"]
    conf = aux["rounds"][0][0]["conf"].sigmoid()
    assert ece(conf, hit) < 0.15, "confidence must be calibrated in the ECE sense"
    assert 0.05 < hit.float().mean().item() < 0.95, "model must be uncertain for this test to mean anything"


def test_ece_perfect_calibration_is_low():
    conf = torch.tensor([0.1] * 10 + [0.9] * 10)
    corr = torch.tensor([1.] * 1 + [0.] * 9 + [1.] * 9 + [0.] * 1)  # 10% and 90% accurate
    assert ece(conf, corr) < 0.02
    assert ece(torch.ones(100), torch.zeros(100)) > 0.9            # fully wrong but confident


def test_ensemble_follows_the_confident_member():
    """Weights are normalized calibrated confidences: a confident-correct member
    must dominate an unconfident-wrong one."""
    logits_good = torch.tensor([[[5.0, 0.0, 0.0, 0.0]]])
    conf_good = torch.tensor([[3.0]])   # sigmoid ~0.95
    logits_bad = torch.tensor([[[0.0, 5.0, 0.0, 0.0]]])
    conf_bad = torch.tensor([[-3.0]])   # sigmoid ~0.05
    mix, conf_mix = ensemble([(logits_good, conf_good), (logits_bad, conf_bad)])
    assert mix.argmax(-1).reshape(-1)[0].item() == 0
    assert conf_mix.reshape(-1)[0].item() > 0.8


def test_ensemble_weights_are_normalized_probabilities():
    """Design §5: weights from calibrated confidences, not raw-logit softmax."""
    lg = torch.zeros(1, 1, 4); lb = torch.zeros(1, 1, 4)
    cg = torch.tensor([[20.0]])   # sigmoid = 1.0
    cb = torch.tensor([[0.0]])    # sigmoid = 0.5
    _, conf_mix = ensemble([(lg, cg), (lb, cb)])
    assert abs(conf_mix.reshape(-1)[0].item() - (2 * 1.0 + 1 * 0.5) / 3) < 1e-4  # 2/3·1 + 1/3·0.5 = 5/6


def test_retry_uses_fresh_paths_and_transforms_the_latent():
    m = tiny_model()
    tokens = torch.randint(0, 49, (2, 16))
    paths = tiny_paths(m.pcap, n=2)
    _, aux = m(tokens, paths, tokens)
    assert len(aux["rounds"]) == 2
    d = (aux["rounds"][1][0]["latent"] - aux["rounds"][0][0]["latent"]).abs().max().item()
    assert d > 0, "retry round must transform the latent"


def test_sample_rounds_gives_each_round_its_own_path():
    """The freshness mechanism itself: with full shuffle, two rounds of one
    sample_rounds call must (almost surely) carry different layer orders."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from train_m0 import sample_rounds
    pcap = PathConfig(shuffle_locality=1.0, p_retry=1.0, n_mtp=2)
    rounds = sample_rounds(pcap, random.Random(0), n_layers=4)
    assert len(rounds) == 2 and rounds[0].n_retries == 1
    assert rounds[0].layer_order != rounds[1].layer_order, "rounds must not share one path"


def test_sample_path_extremes():
    rng = random.Random(0)
    normal = sample_path(PathConfig(), rng, n_layers=2)
    assert normal.layer_order == [0, 1]  # base config: identity path
    # locality=1 must actually permute: many distinct orders across draws
    # (the previous formula could only ever produce the identity)
    orders = {tuple(sample_path(PathConfig(shuffle_locality=1.0), rng, n_layers=4).layer_order)
              for _ in range(20)}
    assert len(orders) >= 10, f"locality=1 barely permutes: {orders}"
    chaos = sample_path(PathConfig(shuffle_locality=1.0, p_skip=0.9, p_redo=0.9, p_retry=1.0),
                        rng, n_layers=2)
    assert len(chaos.layer_order) >= 1 and chaos.n_retries == 1
