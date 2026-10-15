"""WP 6.2.4 · 2673-robustness-13m 的本地可失败单测（全部 CPU，不取 GPU 锁）。

跑法（仓库根）：

    python3 -m pytest \
      axes/6-predict/6.2-speculative-decoding/experiments/2673-robustness-13m/test_spec_lossless.py -q

覆盖契约 §3 的必要自检：

1. **L0 gate probe（硬门）**：接受–残差重采的 Monte Carlo 边缘分布 == 目标分布；**故意改错**
   （接受概率恒为 1 / 残差重采写成直接用草稿分布）⇒ 边缘分布 ≠ 目标分布（必须**变红**）。
2. **微型模型端到端无损性**：`spec_T1` 与 `target_sample` 的经验 token 分布 / 目标 NLL 一致；
   改错规则 ⇒ 不一致。
3. **cache/carry 一致性**：`spec_T1` 记录的每个发出 token 的**目标条件 log 概率**，等于把发出流
   重新喂进模型算出的 log 概率（carry 结转没有污染缓存）。
4. **代价轴**：`spec_T1` 的 `n_forward == n_rounds`（每轮 1 次前向）、tokens-per-forward > 1。
5. **产出口径**：`results.json` 无墙钟字段（机器可核标记）。
6. **变异**见 `mutate.py`（须抓红且逐字节还原）。
"""

from __future__ import annotations

import json
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
for p in (HERE, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import spec_lossless as S  # noqa: E402
from pathlm.config import ModelConfig, PathConfig  # noqa: E402
from pathlm.model import PathLM  # noqa: E402

VOCAB = 64


def tiny_model(seed: int = 0, scale: float = 2.0, divergent: bool = False) -> PathLM:
    """非退化微型模型（embedding 行归一化到同范数 × scale）。

    `divergent=True` 时把 node-2 / node-3 直连头换成**大权重随机映射**：随机初值下三个头
    近乎等价（q ≈ p），gate probe 没有鉴别力；放大后的草稿头与目标头明显不同（实测
    TV(p,q) ≈ 0.95），改错的接受规则才能被检验抓住。
    """
    torch.manual_seed(seed)
    m = PathLM(ModelConfig(d_model=64, n_layers=4, n_heads=4, seq_len=128, mlp_mult=2),
               PathConfig(n_mtp=3, chain_mode="token", transport="none"),
               vocab_size=VOCAB).eval()
    with torch.no_grad():
        w = m.embed.weight.data
        m.embed.weight.data = w / w.norm(dim=1, keepdim=True) * scale
        if divergent:
            g = torch.Generator().manual_seed(seed + 77)
            for k in (1, 2):
                tr = m.transforms[k]
                tr.mlp[0].weight.data = torch.randn(128, 64, generator=g) * 3.0
                tr.mlp[0].bias.data = torch.randn(128, generator=g) * 3.0
                tr.mlp[2].weight.data = torch.randn(64, 128, generator=g) * 3.0
                tr.mlp[2].bias.data = torch.randn(64, generator=g) * 3.0
    return m


def make_prompt(seed: int = 0, length: int = 16) -> torch.Tensor:
    torch.manual_seed(700 + seed)
    return torch.randint(0, VOCAB, (length,))


def _simplex(seed: int, vocab: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.softmax(torch.randn(vocab, generator=g), -1)


def _mc_marginal(p, q, mode: str, n: int = 120000, vocab: int = 16, seed: int = 0):
    """在固定 (p,q) 上做 n 次接受–重采，返回发出 token 的经验分布。"""
    g = torch.Generator().manual_seed(seed)
    hist = torch.zeros(vocab)
    for _ in range(n):
        x = int(torch.multinomial(q, 1, generator=g))
        if mode == "correct":
            u = float(torch.rand((), generator=g))
            ok = S.accept_test(p, q, x, u)
        elif mode == "accept_always":
            ok = True
        else:
            raise ValueError(mode)
        if ok:
            y = x
        elif mode == "correct":
            y = int(torch.multinomial(S.residual_dist(p, q), 1, generator=g))
        else:
            y = x
        hist[y] += 1
    return hist / hist.sum()


# ---------------------------------------------------------------- ① L0 gate probe（单元级）

def test_accept_resample_marginal_equals_target():
    """接受–残差重采的发出边缘分布 == 目标分布 p（Monte Carlo，V=16，n=12 万）。"""
    p, q = _simplex(1, 16), _simplex(2, 16)
    assert float((p - q).abs().sum()) > 0.5, "测试前提：p 与 q 必须明显不同"
    emp = _mc_marginal(p, q, "correct")
    max_err = float((emp - p).abs().max())
    assert max_err < 0.006, f"无损性失败：经验分布与 p 的最大偏差 {max_err}"


def test_gate_probe_accept_always_breaks_marginal():
    """**故意改错（接受恒为 1）** ⇒ 发出边缘分布 == 草稿 q ≠ p（必须变红）。"""
    p, q = _simplex(1, 16), _simplex(2, 16)
    emp = _mc_marginal(p, q, "accept_always")
    err_vs_p = float((emp - p).abs().sum())
    err_vs_q = float((emp - q).abs().sum())
    assert err_vs_p > 0.3, f"改错规则竟仍接近 p（偏差 {err_vs_p}）——检验无鉴别力"
    assert err_vs_q < 0.02, f"改错规则应等于草稿 q，实测 {(emp - q).abs().sum()}"


def test_gate_probe_resample_from_draft_breaks_marginal():
    """**故意改错（重采直接用 q）** ⇒ 边缘分布 ≠ p（必须变红）。"""
    p, q = _simplex(3, 16), _simplex(4, 16)
    g = torch.Generator().manual_seed(5)
    hist = torch.zeros(16)
    n = 120000
    for _ in range(n):
        x = int(torch.multinomial(q, 1, generator=g))
        u = float(torch.rand((), generator=g))
        if S.accept_test(p, q, x, u):
            y = x
        else:
            y = int(torch.multinomial(q, 1, generator=g))   # 改错：用 q 不用残差
        hist[y] += 1
    emp = hist / hist.sum()
    assert float((emp - p).abs().sum()) > 0.3, "改错重采竟仍接近 p —— 检验无鉴别力"


# ---------------------------------------------------------------- ② 微型模型端到端

def _tv(hs: torch.Tensor, ht: torch.Tensor) -> float:
    return 0.5 * float((hs - ht).abs().sum())


def _target_pool(model, prompt, n_new, pcap, seeds):
    hist = torch.zeros(VOCAB)
    logps = []
    for s in seeds:
        seq, st = S.target_sample_T1(model, prompt, n_new, pcap, s)
        for t in seq:
            hist[t] += 1
        logps.extend(st["logp"])
    return hist / hist.sum(), -sum(logps) / len(logps)


def test_tiny_model_spec_is_lossless():
    """`spec_T1`（正确规则）的经验 token 分布 ≈ `target_sample`（有限样本噪声内）。"""
    m = tiny_model(seed=0, divergent=True)
    prompt = make_prompt(0)
    pcap = m.pcap
    seeds = list(range(32))
    n_new = 64
    hs, _, _ = _gen_pool(m, prompt, n_new, pcap, seeds)
    ht, _ = _target_pool(m, prompt, n_new, pcap, seeds)
    # 噪声地板 = 两个独立 target_sample 子样本之间的 TV
    h_even, _ = _target_pool(m, prompt, n_new, pcap, seeds[0::2])
    h_odd, _ = _target_pool(m, prompt, n_new, pcap, seeds[1::2])
    tv_null = _tv(h_even, h_odd)
    tv = _tv(hs, ht)
    assert tv < max(0.20, 2.0 * tv_null), \
        f"spec_T1(正确) 与 target_sample 的 TV={tv} 明显超过噪声地板 {tv_null}（应无损）"


def test_tiny_model_gate_probe_breaks_losslessness():
    """end-to-end gate probe：改错的接受规则 ⇒ 经验分布明显偏离 target_sample（变红）。"""
    m = tiny_model(seed=0, divergent=True)
    prompt = make_prompt(0)
    pcap = m.pcap
    seeds = list(range(32))
    n_new = 64
    ht, _ = _target_pool(m, prompt, n_new, pcap, seeds)
    tv_correct = _tv(_gen_pool(m, prompt, n_new, pcap, seeds)[0], ht)
    for broken in ({"accept_always": True}, {"resample_from_draft": True}):
        hb, _, _ = _gen_pool(m, prompt, n_new, pcap, seeds, broken=broken)
        tv = _tv(hb, ht)
        assert tv > tv_correct + 0.2, \
            f"改错 {broken} 竟仍无损（TV={tv} vs 正确 {tv_correct}）——检验无鉴别力"


def _gen_pool(model, prompt, n_new, pcap, seeds, broken=None):
    """按 seed 多次生成，池化发出 token 直方图 + 目标 NLL。"""
    hist = torch.zeros(VOCAB)
    logps = []
    n_forward = 0
    for s in seeds:
        if broken is None:
            seq, st = S.spec_sample_T1(model, prompt, n_new, pcap, s)
        else:
            seq, st = S.spec_sample_T1(model, prompt, n_new, pcap, s, **broken)
        for t in seq:
            hist[t] += 1
        logps.extend(st["logp"])
        n_forward += st["n_forward"]
    return hist / hist.sum(), -sum(logps) / max(len(logps), 1), n_forward


def test_target_sample_first_token_matches_node1_distribution():
    """参照臂健全性：`target_sample` 的**第一个**发出 token 分布 ≈ node-1 分布。"""
    m = tiny_model(seed=1)
    pcap = m.pcap
    hist = torch.zeros(VOCAB)
    ref = torch.zeros(VOCAB)
    for pi in range(4):
        prompt = make_prompt(pi)
        ref += S.target_dist_at_here(_last_nodes(m, prompt, pcap))
        for s in range(300):
            seq, _ = S.target_sample_T1(m, prompt, 1, pcap, s)
            hist[seq[0]] += 1
    hist = hist / hist.sum()
    ref = ref / ref.sum()
    tv = 0.5 * float((hist - ref).abs().sum())
    # 1200 个样本 / V=64 ⇒ 有限样本噪声地板 ~0.1；阈值 0.25 留余量
    assert tv < 0.25, f"target_sample 首 token 分布与 node-1 分布 TV={tv} 偏大"


def _last_nodes(m, prompt, pcap):
    import spec_tree as T
    dec = T.Decoder(m, window=pcap.window, anchors=pcap.anchors)
    dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        dec.step(int(tok))
    return dec.last_nodes


# ---------------------------------------------------------------- ③ cache/carry 一致性

def test_emitted_logp_matches_fresh_forward():
    """发出流的每个 token 的目标条件 log 概率 == 重新喂进模型逐 token 算出的值。

    证明 carry 结转与接受行写回没有污染缓存（否则位置对不上，log 概率会错）。
    """
    import spec_tree as T
    m = tiny_model(seed=2)
    prompt = make_prompt(2)
    pcap = m.pcap
    seq, st = S.spec_sample_T1(m, prompt, 40, pcap, seed=3)
    dec = T.Decoder(m, window=pcap.window, anchors=pcap.anchors)
    dec.step(int(prompt[0]))
    for tok in prompt.tolist()[1:]:
        dec.step(int(tok))
    for j, tok in enumerate(seq):
        p = S.target_dist_at_here(dec.last_nodes)
        got = float(torch.log(p[tok] + 1e-30))
        assert abs(got - st["logp"][j]) < 1e-4, \
            f"位置 {j}: 记录 logp={st['logp'][j]} vs 重算 {got}（缓存/carry 不一致）"
        if j < len(seq) - 1:
            dec.step(int(tok))


# ---------------------------------------------------------------- ④ 代价轴

def test_cost_axis_one_forward_per_round_and_tpf_gt_one():
    m = tiny_model(seed=0)
    prompt = make_prompt(0)
    seq, st = S.spec_sample_T1(m, prompt, 96, m.pcap, seed=0)
    assert st["n_forward"] == st["n_rounds"], "每轮必须恰 1 次主干前向"
    tpf = st["tokens_per_forward"]
    assert tpf is not None and tpf > 1.0, f"tokens-per-forward={tpf} 应 > 1"
    assert abs(st["tau"] - (tpf - 1.0)) < 1e-9, "τ 与 tokens-per-forward 必须自洽"


def test_draft_dists_are_normalized():
    m = tiny_model(seed=0)
    nodes = _last_nodes(m, make_prompt(0), m.pcap)
    q2, q3 = S.draft_dists(nodes)
    for q in (q2, q3):
        assert abs(float(q.sum()) - 1.0) < 1e-5
    p1 = S.target_dist_at_here(nodes)
    assert abs(float(p1.sum()) - 1.0) < 1e-5


# ---------------------------------------------------------------- ⑤ 产出口径

def test_results_json_has_no_wall_clock_fields():
    """禁墙钟判据的机器可核标记：`results.json` 里不得出现墙钟字段。"""
    path = os.path.join(HERE, "results.json")
    if not os.path.exists(path):
        pytest.skip("results.json 尚未生成（跑 run.py 之后才检查）")
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    bad = []

    def walk(node, prefix=""):
        import re as _re
        if isinstance(node, dict):
            for k, v in node.items():
                toks = _re.split(r"[_]+", k.lower())
                if ("wall" in k.lower() or "speedup" in k.lower()
                        or any(t in ("ms", "seconds", "toks", "tok_s", "wallclock") for t in toks)):
                    bad.append(f"{prefix}.{k}")
                walk(v, f"{prefix}.{k}")
        elif isinstance(node, list):
            for v in node:
                walk(v, prefix)

    walk(blob)
    assert not bad, f"results.json 出现墙钟字段: {bad}"
