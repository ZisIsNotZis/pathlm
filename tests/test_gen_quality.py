"""Ticket-22 quality-signal tests: pure signal metrics + a CPU-model test of
the multi-path re-score probe wiring (shared corruption draw, transport pin)."""

import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig, PathSample, sample_path
from pathlm.eval import eval_pc
from pathlm.gen_quality import (discrimination_auc, js_bits, monotone_frac,
                                pairwise_agreement, pairwise_js, rank_corr,
                                rescore_disagreement, signal_bins)
from pathlm.model import PathLM

MC = ModelConfig(d_model=32, n_layers=3, n_heads=4, seq_len=32, mlp_mult=2)


def _dist(*rows):
    return torch.tensor(rows, dtype=torch.float32)


# ------------------------------------------------------------------ js_bits

def test_js_bits_identical_is_zero():
    p = _dist([0.2, 0.5, 0.3], [0.7, 0.1, 0.2])
    assert js_bits(p, p).abs().max() < 1e-6


def test_js_bits_disjoint_one_hots_is_one_bit():
    p = _dist([1.0, 0.0], [0.0, 1.0])
    j = js_bits(p[0], p[1])
    assert j == pytest.approx(1.0, abs=1e-6)   # JS bounded by 1 bit


def test_js_bits_hand_computed_half_overlap():
    # p=(.5,.5,0), q=(1,0,0): KL(p||m)=0.20752, KL(q||m)=0.41504 -> 0.31128 bits
    p, q = _dist([0.5, 0.5, 0.0]), _dist([1.0, 0.0, 0.0])
    assert float(js_bits(p, q)) == pytest.approx(0.31128, abs=1e-4)


def test_js_bits_symmetric_and_bounded():
    g = torch.Generator().manual_seed(0)
    p = torch.softmax(torch.randn(7, 11, generator=g), -1)
    q = torch.softmax(torch.randn(7, 11, generator=g), -1)
    jpq, jqp = js_bits(p, q), js_bits(q, p)
    assert torch.allclose(jpq, jqp, atol=1e-6)
    assert ((jpq >= -1e-6) & (jpq <= 1.0 + 1e-6)).all()
    with pytest.raises(ValueError):
        js_bits(p, q[:, :5])


# --------------------------------------------------------- pairwise helpers

def test_pairwise_js_means_over_all_pairs():
    a = _dist([1.0, 0.0])
    b = _dist([0.0, 1.0])
    stacked = torch.stack([a, b, a])             # path C == path A
    expected = (js_bits(a, b) + js_bits(a, a) + js_bits(b, a)) / 3
    assert float(pairwise_js(stacked)) == pytest.approx(float(expected), abs=1e-7)
    assert float(pairwise_js(stacked)) > 0.5
    with pytest.raises(ValueError):
        pairwise_js(torch.stack([a]))            # single path


def test_pairwise_agreement_counts_matching_argmaxes():
    hard = torch.tensor([[1, 2, 3], [1, 2, 2], [1, 0, 0]])
    # pairs: (0,1) 2/3, (0,2) 1/3, (1,2) 1/3 -> mean 4/9
    assert float(pairwise_agreement(hard).mean()) == pytest.approx(4 / 9)


# -------------------------------------------------------- rank_corr / AUC

def test_rank_corr_perfect_and_constant():
    assert rank_corr([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)
    assert rank_corr([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]) == pytest.approx(-1.0)
    assert rank_corr([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]) == 0.0   # pearson degenerate


def test_rank_corr_ties_get_average_ranks():
    # x=[1,1,2] -> ranks (1.5,1.5,3); pearson vs (1,2,3) = 0.8660
    assert rank_corr([1.0, 1.0, 2.0], [1.0, 2.0, 3.0]) == pytest.approx(0.86603, abs=1e-4)


def test_discrimination_auc_extremes_ties_and_single_class():
    assert discrimination_auc([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(1.0)
    assert discrimination_auc([1, 2, 3, 4], [1, 1, 0, 0]) == pytest.approx(0.0)
    assert discrimination_auc([7, 7, 7, 7], [0, 0, 1, 1]) == pytest.approx(0.5)
    assert discrimination_auc([1, 2, 3], [0, 1, 0]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        discrimination_auc([1, 2, 3], [0, 0, 0])   # no miss class


# ------------------------------------------------------- bins / monotonicity

def test_signal_bins_equal_count_and_err_rates():
    s = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    m = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    bins = signal_bins(s, m, n_bins=2)
    assert [b["n"] for b in bins] == [5, 5]
    assert bins[0]["mean_signal"] == pytest.approx(3.0)
    assert bins[1]["mean_signal"] == pytest.approx(8.0)
    assert bins[0]["err_rate"] == 0.0 and bins[1]["err_rate"] == 1.0
    with pytest.raises(ValueError):
        signal_bins([1, 2], [1, 2, 3])


def test_monotone_frac_directions():
    # strict-increase definition: a tie between adjacent bins is NOT "up"
    assert monotone_frac([{"err_rate": 0.1}, {"err_rate": 0.2},
                          {"err_rate": 0.3}]) == pytest.approx(1.0)
    assert monotone_frac([{"err_rate": 0.3}, {"err_rate": 0.2},
                          {"err_rate": 0.1}]) == pytest.approx(0.0)
    assert monotone_frac([{"err_rate": 0.1}, {"err_rate": 0.1},
                          {"err_rate": 0.2}]) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        monotone_frac([{"err_rate": 0.1}])


# --------------------------------------------------- rescore_disagreement

def _tiny_model():
    pcap = PathConfig(n_mtp=2, transport="soft", p_retry=0.5, reentry_mix=True,
                      corrupt_wrong=0.075, corrupt_mask=0.075, p_skip=0.1)
    torch.manual_seed(0)
    return PathLM(MC, pcap, 50).eval()


def test_rescore_disagreement_shapes_determinism_and_overrides():
    model = _tiny_model()
    x = torch.randint(0, 49, (2, MC.seq_len))
    r = rescore_disagreement(model, x, n_paths=3, seed=0)
    assert r["js"].shape == (2, MC.seq_len - 1)
    assert ((r["js"] >= -1e-6) & torch.isfinite(r["js"])).all()
    assert ((r["agree"] >= 0) & (r["agree"] <= 1)).all()
    assert ((r["prob0"] > 0) & (r["prob0"] < 1)).all()
    assert ((r["hit"] == 0) | (r["hit"] == 1)).all()
    r2 = rescore_disagreement(model, x, n_paths=3, seed=0)
    for k in r:
        assert torch.allclose(r[k], r2[k]), f"same seed must reproduce {k}"
    # injected-corruption mode and short windows run and stay finite
    r3 = rescore_disagreement(model, x[:, :20], n_paths=2, corrupt_wrong=0.15, seed=3)
    assert torch.isfinite(r3["js"]).all() and r3["js"].shape == (2, 19)
    with pytest.raises(ValueError):
        rescore_disagreement(model, x[:, :1])      # window too short
    with pytest.raises(ValueError):
        rescore_disagreement(model, x, n_paths=1)  # need >= 2 paths


def test_rescore_protocol_replayed_seed_pins_input_across_paths():
    # The probe scores each path with its OWN forward and REPLAYS the
    # corruption seed, so paths see a bitwise-identical input: identical
    # paths must give exactly-zero disagreement, a genuinely different
    # path (a skipped layer) must give positive disagreement. (Passing
    # several paths to ONE forward would re-enter from the previous
    # round's latent — deepening, not re-scoring.)
    model = _tiny_model()
    x = torch.randint(0, 49, (2, MC.seq_len))
    with eval_pc(model, reentry_mix=False, p_retry=0.0, p_token_retry=0.0,
                 w_dense_exit=0.0), torch.no_grad():
        same = sample_path(model.pcap, random.Random(1), MC.n_layers)
        same.n_retries, same.token_retry = 0, False
        outs = []
        for _ in range(2):
            torch.manual_seed(7)                       # the replay step
            _, aux = model(x, [same], x)               # corrupt draws active
            outs.append(aux["rounds"][0][1]["logits"])
        assert torch.equal(*outs)                      # identical input, bitwise

        p_full = PathSample(layer_order=[0, 1, 2], layer_repeats=[1, 1, 1])
        p_skip = PathSample(layer_order=[0, 2], layer_repeats=[1, 1])
        ls = []
        for p in (p_full, p_skip):
            torch.manual_seed(7)
            _, aux = model(x, [p], x)
            ls.append(aux["rounds"][0][1]["logits"][:, :-1].float().softmax(-1))
        # positive but small at random init (near-uniform logits); the pin is
        # the contrast with the exactly-zero identical-path case above
        assert float(pairwise_js(torch.stack(ls)).mean()) > 0.0
