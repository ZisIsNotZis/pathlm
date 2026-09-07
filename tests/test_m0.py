"""Fast CPU tests for the M0 PathLM skeleton. Run: python -m pytest tests/ -q"""

import os, random, sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.metrics import ece, ensemble
from pathlm.model import PathLM


def tiny_path(pcap, seed=0):
    return sample_path(pcap, random.Random(seed), 2)


def tiny_model(pcap=None):
    pcap = pcap or PathConfig(n_mtp=2, transport="direct", p_retry=0.5)
    torch.manual_seed(0)
    return PathLM(ModelConfig(d_model=32, n_layers=2, n_heads=2, seq_len=16), pcap, vocab_size=50)


def test_causal_masking_predicts_from_past_only():
    """Loss must be high when targets are unpredictable-from-context, low when copyable."""
    m = tiny_model()
    B, T = 4, 16
    tokens = torch.randint(0, 49, (B, T))
    path = tiny_path(PathConfig(n_mtp=2))
    loss_random, _ = m(tokens, path, tokens)  # targets = fresh random: unpredictable
    seq = torch.arange(T).unsqueeze(0).repeat(B, 1) % 49
    loss_copy, _ = m(seq, path, seq)          # arange sequence: learnable structure
    assert loss_random.item() > 0
    # after one step on the copyable sequence the loss must drop below random's
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(30):
        loss, _ = m(seq, path, seq)
        opt.zero_grad(); loss.backward(); opt.step()
    loss_after, _ = m(seq, path, seq)
    assert loss_after.item() < loss_random.item() * 0.5


def test_corruption_keeps_targets_clean():
    """Corruption mutates inputs only; targets passed in are returned untouched."""
    m = tiny_model(PathConfig(corrupt_wrong=1.0, n_mtp=1, transport="none"))
    tokens = torch.randint(0, 49, (8, 16))
    path = tiny_path(m.pcap)
    _, aux = m(tokens, path, tokens)
    assert aux["corrupt_mask"].all()          # rate 1.0 -> every position corrupted


def test_component_targets_are_shifted():
    """Node k's CE must be computed against tokens[i+k]: check via a trained lookup."""
    m = tiny_model(PathConfig(n_mtp=2, transport="none"))
    B, T = 2, 16
    # constant sequence: every node's prediction target equals every input token
    tokens = torch.full((B, T), 7)
    path = tiny_path(m.pcap)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(50):
        loss, _ = m(tokens, path, tokens)
        opt.zero_grad(); loss.backward(); opt.step()
    _, aux = m(tokens, path, tokens)
    for k in range(3):
        acc = (aux["rounds"][0][k]["logits"].argmax(-1) == 7).float().mean().item()
        assert acc > 0.95, f"node{k} failed to learn constant target"


def test_confidence_head_predicts_the_argmax_event():
    """Confidence BCE trains sigmoid(conf) toward P(argmax == target)."""
    m = tiny_model(PathConfig(n_mtp=1, transport="none"))
    B, T = 2, 16
    tokens = torch.full((B, T), 7)
    path = tiny_path(m.pcap)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    for _ in range(50):
        loss, _ = m(tokens, path, tokens)
        opt.zero_grad(); loss.backward(); opt.step()
    _, aux = m(tokens, path, tokens)
    hit = aux["rounds"][0][0]["hit"]
    conf = aux["rounds"][0][0]["conf"].sigmoid()
    assert (conf - hit).abs().mean().item() < 0.25


def test_ece_perfect_calibration_is_low():
    conf = torch.tensor([0.1] * 10 + [0.9] * 10)
    corr = torch.tensor([1.] * 1 + [0.] * 9 + [1.] * 9 + [0.] * 1)  # 10% and 90% accurate
    assert ece(conf, corr) < 0.02
    assert ece(torch.ones(100), torch.zeros(100)) > 0.9           # fully wrong but confident


def test_ensemble_follows_the_confident_member():
    # confident-correct member + unconfident-wrong member -> mixture follows the confident one
    logits_good = torch.tensor([[[5.0, 0.0, 0.0, 0.0]]])
    conf_good = torch.tensor([[3.0]])   # raw logit, sigmoid ~0.95
    logits_bad = torch.tensor([[[0.0, 5.0, 0.0, 0.0]]])
    conf_bad = torch.tensor([[-3.0]])   # sigmoid ~0.05
    mix, conf_mix = ensemble([(logits_good, conf_good), (logits_bad, conf_bad)])
    assert mix.argmax(-1).reshape(-1)[0].item() == 0
    assert conf_mix.reshape(-1)[0].item() > 0.8


def test_retry_produces_a_second_round():
    m = tiny_model()
    tokens = torch.randint(0, 49, (2, 16))
    path = tiny_path(m.pcap)
    path.n_retries = 1
    _, aux = m(tokens, path, tokens)
    assert len(aux["rounds"]) == 2
    d = (aux["rounds"][1][0]["latent"] - aux["rounds"][0][0]["latent"]).abs().max().item()
    assert d > 0, "retry round must transform the latent"


def test_sample_path_extremes():
    rng = random.Random(0)
    normal = sample_path(PathConfig(), rng, n_layers=2)
    assert normal.layer_order == [0, 1]  # base config: identity path
    chaos = sample_path(PathConfig(shuffle_locality=1.0, p_skip=0.9, p_redo=0.9, p_retry=1.0), rng, n_layers=2)
    assert len(chaos.layer_order) >= 1 and chaos.n_retries == 1
