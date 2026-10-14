#!/usr/bin/env python3
"""L0 / L0b 的红绿数值证据（S1 留证；可重跑）。

打印四条检查的 `max|Δ logits|`：L0 的两条应为 **0.0**（绿），gate probe 的两条应 **> 0**（红）。
同时演示 gate probe 确实让对应的 `assert delta == 0` **抛断言**（可失败性）。

    python3 evidence/probe_l0l0b.py
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import padfill as P  # noqa: E402

TV, TD, TL, TH, TT = 32, 64, 2, 4, 64


def mk(kind, seed=0):
    torch.manual_seed(seed)
    m = P.PadFillLM(vocab_size=TV, attn_mask=kind, d_model=TD, n_layers=TL,
                    n_heads=TH, mlp_mult=4, seq_len=TT)
    m.eval()
    return m


def true_tokens(seed=1):
    return torch.randint(0, TV - 1, (2, TT), generator=torch.Generator().manual_seed(seed))


def maxabs(a, b, sel=None):
    d = (a - b).abs()
    return float(d[sel].max()) if sel is not None else float(d.max())


def perturb_at(true, sel):
    t2 = true.clone()
    t2[sel] = (true[sel] + 1) % (TV - 1)
    return t2


def main():
    true = true_tokens()
    M = P.sample_fill_mask(true.shape, 0.25, torch.Generator().manual_seed(7))
    pos = torch.arange(TT)[None, :]
    half = TT // 2
    right = (~M) & (pos >= half)
    leftM = M & (pos < half)
    m_full, m_causal = mk("full"), mk("causal")

    def fwd(m, x):
        with torch.no_grad():
            return m(x)

    # L0-a（绿）：bidir，扰动被填位置真值 → 被填位置 logits 逐位不变
    xa, xa2 = (P.make_fill_input(t, M, "mask", mask_token=TV - 1) for t in (true, perturb_at(true, M)))
    g1 = maxabs(fwd(m_full, xa), fwd(m_full, xa2), M)
    # L0-b（绿）：causal，扰动右侧非 M → 左侧 M logits 逐位不变
    xb, xb2 = (P.make_fill_input(t, M, "mask", mask_token=TV - 1) for t in (true, perturb_at(true, right)))
    g2 = maxabs(fwd(m_causal, xb), fwd(m_causal, xb2), leftM)
    # probe-1（红）：泄漏构造（真 token 喂进 M）→ L0-a 变红
    xc, xc2 = (P.make_fill_input(t, M, "leak", mask_token=TV - 1) for t in (true, perturb_at(true, M)))
    r1 = maxabs(fwd(m_full, xc), fwd(m_full, xc2), M)
    # probe-2（红）：causal 改 full → L0-b 变红
    r2 = maxabs(fwd(m_full, xb), fwd(m_full, xb2), leftM)

    print(f"L0-a  (bidir mask,   perturb true@M)   max|Δ| = {g1!r}   [绿 ⇒ 期望 0.0]")
    print(f"L0-b  (causal,       perturb right)    max|Δ| = {g2!r}   [绿 ⇒ 期望 0.0]")
    print(f"probe-1 (leak mask,  perturb true@M)   max|Δ| = {r1!r}   [红 ⇒ 期望 > 0]")
    print(f"probe-2 (full attn,  perturb right)    max|Δ| = {r2!r}   [红 ⇒ 期望 > 0]")

    assert g1 == 0.0 and g2 == 0.0, "L0 must be green"
    assert r1 > 0.0 and r2 > 0.0, "gate probes must be red"
    # 可失败性：把红当作绿来断言，必须抛断言（证明检查真的能失败）
    for tag, val in (("probe-1", r1), ("probe-2", r2)):
        try:
            assert val == 0.0
            raise SystemExit(f"{tag} 没有变红 —— gate probe 失效")
        except AssertionError:
            print(f"  gate probe {tag}: `assert max|Δ| == 0` 抛 AssertionError ⇒ 检查可失败 ✓")
    print("ALL GREEN/PLACED: L0 两条绿、gate probe 两条红。")


if __name__ == "__main__":
    main()
