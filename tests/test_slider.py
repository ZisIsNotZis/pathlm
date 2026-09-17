"""Slider machinery tests. Pure-function properties + one CPU-model test of
the decode-time prob0_log wiring (the online-quality probe input)."""

import math
import os
import random
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlm.config import ModelConfig, PathConfig, sample_path
from pathlm.slider import (Frontier, FrontierPoint, fire_curve, forwards_cost,
                           gated_quality, pearson, proxy_summary, reliability,
                           spec_cost)

MC = ModelConfig(d_model=32, n_layers=3, n_heads=4, seq_len=32, mlp_mult=2)


# ------------------------------------------------------------- fire_curve

def test_fire_curve_matches_manual_count():
    probs = torch.tensor([0.05, 0.2, 0.45, 0.6, 0.9])
    fc = fire_curve(probs, [0.1, 0.5, 0.95])
    assert fc[0.1] == pytest.approx(1 / 5)
    assert fc[0.5] == pytest.approx(3 / 5)   # 0.05, 0.2, 0.45
    assert fc[0.95] == pytest.approx(1.0)    # strict <


def test_fire_curve_monotone_in_tau():
    probs = torch.rand(1000)
    taus = [0.1, 0.3, 0.5, 0.7, 0.9]
    fc = fire_curve(probs, taus)
    vals = [fc[t] for t in taus]
    assert vals == sorted(vals)
    assert all(0.0 <= v <= 1.0 for v in vals)


# --------------------------------------------------------------- spec_cost

def test_spec_cost_plain_is_one_plus_fire():
    assert spec_cost(0, 0.0, 1.0) == pytest.approx(1.0)
    assert spec_cost(0, 0.0152, 1.0) == pytest.approx(1.0152)
    # a2 must not matter for k=0 (no spec loop)
    assert spec_cost(0, 0.3, 0.0) == pytest.approx(1.3)


def test_spec_cost_spec_equal_accept_is_compute_neutral():
    # a2 = 1 (every draft accepted), no retries: 2 units / 2 tokens = 1.0
    assert spec_cost(1, 0.0, 1.0) == pytest.approx(1.0)
    # ticket-14 INT2 operating point: fire 0.0152, a2 0.84 -> (2.0152)/1.84
    assert spec_cost(1, 0.0152, 0.84) == pytest.approx(2.0152 / 1.84)


def test_spec_cost_monotone_and_bounded():
    for fire in (0.0, 0.2, 0.9):
        assert spec_cost(1, fire, 1.0) <= spec_cost(1, fire, 0.5) <= spec_cost(1, fire, 0.0)
    assert spec_cost(1, 0.0, 1.0) >= 1.0  # floor: plain-equivalent compute
    with pytest.raises(ValueError):
        spec_cost(-1, 0.0, 1.0)
    with pytest.raises(ValueError):
        spec_cost(1, 0.0, 1.5)


def test_forwards_cost_spec_amortizes_below_one():
    # v1 budget currency = forward count: spec with a2=1 emits 2 tokens per
    # verify forward -> 0.5 forwards/token (BELOW the plain floor of 1.0)
    assert forwards_cost(1, 0.0, 1.0) == pytest.approx(0.5)
    assert forwards_cost(0, 0.0, 1.0) == pytest.approx(1.0)
    assert forwards_cost(1, 0.0152, 0.84) == pytest.approx(1.0152 / 1.84)
    # retry adds exactly one forward per fired loop
    assert forwards_cost(1, 1.0, 1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        forwards_cost(1, 0.0, 1.5)


# ---------------------------------------------------------- Frontier solve

def _pt(k, tau, fire, a2, corrupt, clean=1.64, **measured):
    return FrontierPoint(k=k, tau=tau, fire_rate=fire, a2=(a2 if k else None),
                         cost_forwards=forwards_cost(k, fire, a2 if k else 1.0),
                         cost_flop=spec_cost(k, fire, a2 if k else 1.0),
                         quality={"corrupt_bpc": corrupt, "clean_bpc": clean},
                         measured=measured)


def test_solve_budget_hits_exact_cost_with_interpolation():
    # k=0 series: tau .5 (fire .1) -> tau .9 (fire .5); corrupt 2.04 -> 2.03
    f = Frontier([_pt(0, 0.5, 0.1, 1.0, 2.040),
                  _pt(0, 0.9, 0.5, 1.0, 2.030)])
    p = f.solve_budget(1.2)          # fire .2 at tau = .5 + .1/(.4)*.4 = .6
    assert p.k == 0
    assert p.tau == pytest.approx(0.6)
    assert p.cost_forwards == pytest.approx(1.2)
    assert p.quality["corrupt_bpc"] == pytest.approx(2.0375)


def test_solve_budget_prefers_better_quality_within_budget():
    # k=1 forwards cost = 1.5/1.5 = 1.0 (flop view 5/3); both in budget 1.7
    f = Frontier([_pt(0, 0.9, 0.5, 1.0, 2.030),
                  _pt(1, 0.9, 0.5, 0.5, 2.000)])
    p = f.solve_budget(1.7)
    assert p.k == 1 and p.tau == 0.9
    assert p.quality["corrupt_bpc"] == pytest.approx(2.0)
    # flop currency: k=1 costs 5/3 > 1.5 -> plain wins
    p2 = f.solve_budget(1.5, currency="flop")
    assert p2.k == 0 and p2.tau == 0.9


def test_solve_budget_clean_key_prefers_less_firing():
    # gating slightly hurts clean quality: lower tau = better clean bpc
    f = Frontier([_pt(0, 0.5, 0.1, 1.0, 2.040, clean=1.6395),
                  _pt(0, 0.9, 0.5, 1.0, 2.030, clean=1.6431)])
    # clean key: the measured low-tau point wins (quality 1.6395 < interp 1.63995)
    p = f.solve_budget(1.2, key="clean_bpc")
    assert p.tau == pytest.approx(0.5)
    assert p.quality["clean_bpc"] == pytest.approx(1.6395)
    # corrupt key: the interpolated interior point wins (more repair, still in budget)
    p2 = f.solve_budget(1.2, key="corrupt_bpc")
    assert p2.tau == pytest.approx(0.6)
    assert p2.quality["corrupt_bpc"] == pytest.approx(2.040 + 0.25 * (2.030 - 2.040))


def test_solve_budget_below_floor_raises():
    f = Frontier([_pt(0, None, 0.0, 1.0, 2.040)])
    with pytest.raises(ValueError, match="budget"):
        f.solve_budget(0.5)


def test_solve_quality_picks_cheapest_feasible():
    f = Frontier([_pt(0, 0.9, 0.5, 1.0, 2.030),      # forwards cost 1.5
                  _pt(1, 0.9, 0.5, 0.5, 2.000),      # forwards cost 1.0
                  _pt(1, 0.5, 0.1, 0.5, 2.035)])     # forwards cost 0.733
    p = f.solve_quality(2.01)                        # need corrupt <= 2.01
    assert p.k == 1 and p.tau == 0.9
    assert p.cost_forwards == pytest.approx(1.0)
    # flop currency: k=0 tau=.9 (2.030) misses the 2.01 target; only k=1
    # reaches it, so the currency does not change the answer here
    p2 = f.solve_quality(2.01, currency="flop")
    assert p2.k == 1
    assert p2.cost_flop == pytest.approx(5 / 3)
    # at a looser target both reach it and flop prefers the cheaper FLOP point
    p3 = f.solve_quality(2.035, currency="flop")
    assert p3.k == 1 and p3.tau == 0.5 and p3.cost_flop == pytest.approx(1.4)


def test_solve_quality_unreachable_raises():
    f = Frontier([_pt(0, 0.9, 0.5, 1.0, 2.030)])
    with pytest.raises(ValueError, match="target"):
        f.solve_quality(1.9)


def test_frontier_tau_none_is_retry_off_endpoint():
    off = _pt(0, None, 0.0, 1.0, 2.040)
    assert off.tau_eff == 0.0
    f = Frontier([off, _pt(0, 0.9, 0.5, 1.0, 2.030)])
    # tau=0 behaves like retry-off (fire 0, quality clamps to the off point)
    assert f.cost_at(0, 0.0) == pytest.approx(1.0)
    assert f.cost_at(0, 0.0, currency="flop") == pytest.approx(1.0)
    assert f.quality_at(0, 0.0, "corrupt_bpc") == pytest.approx(2.040)


# ------------------------------------------------------- reliability/proxy

def test_reliability_perfect_calibration_low_ece():
    torch.manual_seed(0)
    probs = torch.rand(4000)
    hits = (torch.rand(4000) < probs).float()   # realized acc == prob in expect.
    rep = reliability(probs, hits, n_bins=10)
    assert rep["ece"] < 0.05
    assert sum(b["n"] for b in rep["bins"]) == 4000
    assert rep["acc"] == pytest.approx(rep["mean_prob"], abs=0.05)


def test_reliability_miscalibration_is_detected():
    torch.manual_seed(1)
    probs = torch.rand(4000) * 0.5 + 0.5     # always claims >= .5
    hits = (torch.rand(4000) < 0.3).float()  # always wrong 70%
    rep = reliability(probs, hits)
    assert rep["ece"] > 0.15


def test_proxy_summary_weights_by_n_emitted():
    log = [{"prob0_pre": 0.9, "prob0_post": None, "fired": False, "n_emitted": 3},
           {"prob0_pre": 0.1, "prob0_post": 0.8, "fired": True, "n_emitted": 1}]
    s = proxy_summary(log)
    assert s["proxy_mean_prob0"] == pytest.approx((0.9 * 3 + 0.1) / 4)
    assert s["proxy_mean_prob0_post"] == pytest.approx(0.8)
    assert s["fire_fraction"] == pytest.approx(0.5)
    assert s["n_emitted"] == 4
    with pytest.raises(ValueError):
        proxy_summary([])


# ------------------------------------------------------ gated_quality etc.

def _probe(n=1000, seed=3, hit_p=0.6):
    """Synthetic gate_probe arrays: prob0 uniform, round-2 better."""
    g = torch.Generator().manual_seed(seed)
    p0 = torch.rand(n, generator=g)
    p0r2 = torch.clamp(p0 + 0.2, max=1.0)
    hit1 = (torch.rand(n, generator=g) < p0 * hit_p).float()
    hit2 = (torch.rand(n, generator=g) < p0r2 * hit_p).float()
    nll1 = torch.where(hit1 > 0, torch.tensor(0.1), torch.tensor(5.0))
    nll2 = torch.where(hit2 > 0, torch.tensor(0.1), torch.tensor(5.0))
    corr = (torch.rand(n, generator=g) < 0.1).float()
    return {"prob0": p0, "prob0_r2": p0r2, "hit_r1": hit1, "hit_r2": hit2,
            "nll_r1": nll1, "nll_r2": nll2, "corrupt": corr, "hit_spec": hit2}


def test_gated_quality_matches_gate_semantics():
    probe = _probe(2000)
    rows = gated_quality(probe, [0.5])
    p0, gate = probe["prob0"], probe["prob0"] < 0.5
    # fire rate == P(prob0 < tau)
    assert rows[0.5]["fire_rate"] == pytest.approx(float(gate.float().mean()), abs=1e-4)
    # gated hit mixes rounds by the gate
    gated_hit = torch.where(gate, probe["hit_r2"], probe["hit_r1"])
    assert rows[0.5]["gated_acc"] == pytest.approx(float(gated_hit.mean()), abs=1e-4)
    # retry-off row (tau "off") uses round 1 everywhere
    assert rows["off"]["fire_rate"] == 0.0
    assert rows["off"]["gated_acc"] == pytest.approx(float(probe["hit_r1"].mean()), abs=1e-4)
    # effective proxy substitutes round-2 prob0 on fired positions
    eff = torch.where(gate, probe["prob0_r2"], probe["prob0"])
    assert rows[0.5]["proxy_eff_mean"] == pytest.approx(float(eff.mean()), abs=1e-4)
    # corrupted-position accuracy is reported when corruption exists
    assert "gated_acc_corr" in rows[0.5]


def test_pearson_recovers_known_correlations():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(500, generator=g)
    assert pearson(x, x) == pytest.approx(1.0)
    assert pearson(x, -x) == pytest.approx(-1.0)
    noise = torch.randn(500, generator=g) * 0.1
    assert pearson(x, x + noise) > 0.9
    assert pearson(x, torch.randn(500, generator=g)) < 0.2
    with pytest.raises(ValueError):
        pearson([1.0], [1.0, 2.0])


# ------------------------------------------------- decode wiring (CPU model)

def _spec_model(n_mtp=2):
    pcap = PathConfig(n_mtp=n_mtp, chain_mode="latent", transport="soft",
                      p_retry=0.0, reentry_mix=False, corrupt_wrong=0.0,
                      corrupt_mask=0.0)
    from pathlm.model import PathLM
    torch.manual_seed(0)
    return PathLM(MC, pcap, vocab_size=50)


def test_decode_prob0_log_records_gate_decisions():
    from pathlm.decode import decode_spec
    m = _spec_model()
    prompt = torch.randint(0, 49, (8,))
    log: list = []
    got, st = decode_spec(m, prompt, 12, m.pcap, retry_threshold=0.9, prob0_log=log)
    # spec k=1: accepted drafts make loops emit 2 tokens -> fewer entries than tokens
    assert len(got) == 12
    assert sum(e["n_emitted"] for e in log) == 12
    assert 6 <= len(log) <= 12
    assert all(e["fired"] for e in log)  # prob0 ~0.5 < 0.9 fires every loop
    for e in log:
        assert 0.0 <= e["prob0_pre"] <= 1.0
        assert (e["prob0_post"] is None) == (not e["fired"])
    # retry-off: nothing fires, post is None everywhere
    log2: list = []
    decode_spec(m, prompt, 12, m.pcap, retry_threshold=None, prob0_log=log2)
    assert not any(e["fired"] for e in log2)


def test_decode_prob0_log_plain_path_matches_spec():
    from pathlm.decode import decode, decode_spec
    m = _spec_model(n_mtp=1)   # no spec loop -> decode() plain path
    prompt = torch.randint(0, 49, (8,))
    log: list = []
    got, st = decode(m, prompt, 10, m.pcap, retry_threshold=0.5, prob0_log=log)
    assert len(log) == 10
    assert all(e["n_emitted"] == 1 for e in log)
    assert len(got) == 10


def test_decode_input_fn_identity_is_noop_and_real_fn_changes_stream():
    from pathlm.decode import decode, decode_spec
    m = _spec_model()
    prompt = torch.randint(0, 49, (8,))
    base, _ = decode_spec(m, prompt, 10, m.pcap)
    ident, _ = decode_spec(m, prompt, 10, m.pcap,
                           input_fn=lambda t: t)          # identity = no-op
    assert ident == base
    shifted, _ = decode_spec(m, prompt, 10, m.pcap,
                             input_fn=lambda t: (t + 7) % 49)
    assert shifted != base          # the model really reads the transformed ids
    base0, _ = decode(m, prompt, 10, m.pcap)
    ident0, _ = decode(m, prompt, 10, m.pcap, input_fn=lambda t: t)
    assert ident0 == base0
    shifted0, _ = decode(m, prompt, 10, m.pcap, input_fn=lambda t: (t + 7) % 49)
    assert shifted0 != base0
