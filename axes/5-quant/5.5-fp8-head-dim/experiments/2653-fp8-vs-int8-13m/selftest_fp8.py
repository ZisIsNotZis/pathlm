"""2653 · fp8 / 大 head_dim 的可失败单测 + 变异（在本目录内，不写 `tests/**`）。

契约：`README.md` §5/§6。每个护栏都带一个**会红的变异**：
- fp8 往返不是恒等（否证「量化器没干活也算近无损」）；
- E4M3 在 448 饱和、E5M2 在 57344 饱和（否证「格式上限写错也能过」）；
- per-token 标尺**严格前缀确定**（否证非因果标尺混进来）；
- 前缀不变性仪器**能抓到非因果前向**（否证仪器本身没有牙）；
- 字节账与 `5.1` 的 12288 / 6528 / 6144 逐项对账（否证口径漂移）；
- `G` 必须整除 `head_dim`（否证「G=64 也能在 head_dim=32 上摊薄」）；
- 实测标尺个数 = 解析值数 / G（否证 `2/G` 摊薄项只写在文档里）。

用法：`python3 -m pytest selftest_fp8.py -q`（或 `python3 selftest_fp8.py`）。
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import fp8_arms as fa                                                        # noqa: E402

P2621 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2621-kv-budget-rest")
P2631 = os.path.join(ROOT, "axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc2631 = _load(os.path.join(P2631, "causal_check.py"), "causal_check_2631")
kvq = _load(os.path.join(ROOT, "tools/evalkit/kv_quant.py"), "kv_quant_t")


def _cfg():
    cfg = json.load(open(os.path.join(HERE, "config.json")))
    parent = json.load(open(os.path.join(P2621, "config.json")))
    for k in ("model", "data", "tokens", "checkpoints"):
        cfg[k] = parent[k]
    cfg["head_dim"] = cfg["model"]["d_model"] // cfg["model"]["n_heads"]
    return cfg


# --------------------------------------------------------------- ① fp8 往返真的改了数值


def test_fp8_roundtrip_is_not_identity():
    """变异：把 fp8 往返实现成恒等（`return x`）⇒ 本测试红。"""
    x = torch.randn(2, 8, 64, 32) * 0.5
    for fmt in fa.FP8_FORMATS:
        dq, _, _ = fa.fp8_roundtrip(x, fmt, "none")
        changed = (dq != x).float().mean().item()
        assert changed > 0.5, f"{fmt} 往返几乎没改数值（changed_frac={changed}）⇒ 量化没生效"
        assert (dq - x).abs().max().item() > 0.0


def test_e4m3_saturates_at_448_and_e5m2_at_57344():
    """变异：格式上限写错（用 2**15 之类）⇒ 本测试红。"""
    sat4 = torch.tensor([[[[500.0, -500.0, 1e5, -1e5]]]])
    d4, _, n_sat4 = fa.fp8_roundtrip(sat4, "fp8_e4m3")
    assert fa.FP8_MAX["fp8_e4m3"] == 448.0
    assert d4[0, 0, 0, 0].item() == 448.0 and d4[0, 0, 0, 1].item() == -448.0
    assert n_sat4 == 4, "E4M3 的饱和计数应把 |x|>448 的值都数上"
    wide = torch.tensor([[[[384.0, -384.0, 1e5, -1e5]]]])
    d5, _, n_sat5 = fa.fp8_roundtrip(wide, "fp8_e5m2")
    assert fa.FP8_MAX["fp8_e5m2"] == 57344.0
    assert d5[0, 0, 0, 0].item() == 384.0, "E5M2 的动态范围应容下 384（且 384 可精确表示）"
    assert d5[0, 0, 0, 2].item() == float("inf") and n_sat5 == 2, \
        "E5M2 超 57344 溢出为 inf（实测 torch 语义），同时计入 saturated 计数"


def test_over_range_semantics_differ_between_e4m3_and_e5m2():
    """实兑口径：E4M3（fn，无 inf）超范围**饱和**到 448；E5M2 超范围**溢出为 inf**。

    这不是格式理论，而是 `torch` CPU cast 的实际语义（已实测）；本实验在结果里只报告
    `saturated_frac`（`|x| > 格式 max` 的占比），**不**把溢出后的数值当结论。
    """
    x = torch.tensor([[[[640.0]]]])
    d4, _, _ = fa.fp8_roundtrip(x, "fp8_e4m3")
    assert d4.item() == 448.0, "640 超 E4M3 上限 448 ⇒ 饱和"
    d5, _, n5 = fa.fp8_roundtrip(x, "fp8_e5m2")
    assert d5.item() == 640.0, "640 在 E5M2 里可精确表示（尾数 2 位 ⇒ 步长 128）"
    assert n5 == 0
    over, _, _ = fa.fp8_roundtrip(torch.tensor([[[[600.0]]]]), "fp8_e5m2")
    assert over.item() == 640.0, "600 在 E5M2 里四舍五入到 640（不是 1024）"
    d5h, _, n5h = fa.fp8_roundtrip(torch.tensor([[[[1e5]]]]), "fp8_e5m2")
    assert d5h.item() == float("inf") and n5h == 1, "E5M2 超范围应溢出为 inf 且计入 saturated"
    n4 = fa.fp8_roundtrip(torch.tensor([[[[1e5]]]]), "fp8_e4m3")[2]
    assert n4 == 1, "E4M3 同样把 |x|>448 计入 saturated（只是数值饱和而非溢出）"


def test_e4m3_flushes_subnormals_below_2_pow_minus_9():
    """E4M3 的最小次正规 = 2^-9 ≈ 0.00195；更小的值被冲成 0（动态范围的另一面）。"""
    x = torch.full((1, 1, 1, 4), 1e-6)
    dq, _, _ = fa.fp8_roundtrip(x, "fp8_e4m3")
    assert torch.equal(dq, torch.zeros_like(dq)), "E4M3 应把 1e-6 冲成 0"
    x2 = torch.full((1, 1, 1, 4), 0.5)
    dq2, _, _ = fa.fp8_roundtrip(x2, "fp8_e4m3")
    assert torch.equal(dq2, x2), "0.5 在 E4M3 里可精确表示"


# --------------------------------------------------------------- ② 标尺的因果性


def test_tokenscale_is_prefix_causal():
    """per-token 标尺只由当前 token 自己的值决定 ⇒ 改后续位置不改过去位置的反量化值。

    变异：把标尺换成**全序列 amax**（非因果）⇒ 本测试红。
    """
    g = torch.Generator().manual_seed(3)
    x = torch.randn(1, 2, 8, 32, generator=g) * 0.5
    dq, s, _ = fa.fp8_roundtrip(x, "fp8_e4m3", "tokenscale")
    assert s.shape == (1, 2, 8, 1), "per-token 标尺形状应为 [..., H, T, 1]"

    x2 = x.clone()
    x2[0, :, -1, :] *= 100.0                      # 只改最后一个位置的数值
    dq2, _, _ = fa.fp8_roundtrip(x2, "fp8_e4m3", "tokenscale")
    assert torch.equal(dq[..., :-1, :], dq2[..., :-1, :]), "改未来位置改动了过去位置的反量化值"

    def noncausal(t):
        amax = t.abs().amax().clamp_min(1e-12)
        sc = amax / fa.FP8_MAX["fp8_e4m3"]
        return ((t / sc).to(torch.float8_e4m3fn).to(t.dtype)) * sc

    bad, good = noncausal(x), noncausal(x2)
    assert not torch.equal(bad[..., :-1, :], good[..., :-1, :]), \
        "非因果变异本该被抓到——说明本测试没有牙"


def test_prefix_invariance_instrument_catches_noncausal_forward():
    """仪器（`2631/causal_check`）必须能抓到非因果前向，否则它证不了任何事。"""
    g = torch.Generator().manual_seed(5)
    tokens = torch.randint(0, 40, (2, 8), generator=g)

    def causal(x):
        # 过去位置只依赖过去 token
        return torch.cumsum(x.float(), dim=1).unsqueeze(-1)

    ok = cc2631.prefix_invariance_of_logits(causal, tokens)
    assert ok["future_token_past_logits_equal"] and ok["peer_request_logits_equal"]

    def noncausal(x):
        # 过去位置被最后一个 token 污染（同批邻居也进归约）
        total = x.float().sum()
        return (x.float() + total).unsqueeze(-1)

    bad = cc2631.prefix_invariance_of_logits(noncausal, tokens)
    assert not bad["future_token_past_logits_equal"], "仪器没抓到『改未来 token』"
    assert not bad["peer_request_logits_equal"], "仪器没抓到『改同批邻居』"


# --------------------------------------------------------------- ③ 字节账 / G 与 head_dim


def test_deployment_ledger_reconciles_5_1():
    """变异：把 1 + 2/G 写成 1 + 1/G，或把每位置值数写成别的数 ⇒ 本测试红。"""
    led = fa.deployment_ledger(_cfg())
    assert led["values_per_position"] == 6144.0
    assert led["fp16_bytes_per_position"] == 12288.0
    assert led["per_arm"]["int8_block_g32"]["bytes_per_value"] == 1.0625
    assert led["per_arm"]["int8_block_g32"]["bytes_per_position"] == 6528.0
    assert led["per_arm"]["fp8_e4m3"]["bytes_per_value"] == 1.0
    assert led["per_arm"]["fp8_e4m3"]["bytes_per_position"] == 6144.0
    # 逐项因子必须能重算出同一个数
    for name, e in led["per_arm"].items():
        assert abs(e["values_per_position"] * e["bytes_per_value"]
                   - e["bytes_per_position"]) < 1e-6, name
        assert abs(e["value_bytes_per_value"] + e["scale_bytes_per_value"]
                   - e["bytes_per_value"]) < 1e-9, name
    assert fa.bytes_per_value_with_scale(8) == 1.25
    assert fa.bytes_per_value_with_scale(16) == 1.125
    assert fa.bytes_per_value_with_scale(64) == 1.03125


def test_head_dim_cap_is_46_875_percent_at_head_dim_32():
    """head_dim=32 ⇒ G ≤ 32 ⇒ 省幅封顶 46.875%；head_dim=64 ⇒ G=64 ⇒ 48.4375%。"""
    cap = {str(h): fa.head_dim_cap(h) for h in (32, 64)}
    assert cap["32"]["max_group"] == 32 and cap["32"]["reduction_vs_fp16"] == 0.46875
    assert cap["64"]["max_group"] == 64 and cap["64"]["reduction_vs_fp16"] == 0.484375
    ladder = {r["group"]: r for r in fa.head_dim_ladder(head_dims=(32,), groups=(8, 16, 32, 64))}
    assert ladder[32]["legal"] and ladder[32]["bytes_per_value"] == 1.0625
    assert not ladder[64]["legal"], "head_dim=32 时 G=64 必须非法（块跨 token，违反前缀确定）"


def test_group_must_divide_head_dim():
    """变异：让 quantize_kv 接受 G=64 / head_dim=32 ⇒ 本测试红。"""
    x = torch.randn(1, 2, 8, 32)
    kvq.quantize_kv(x, "int8", "block", 32)                       # 合法
    for bad in (64, 48, 5):
        try:
            kvq.quantize_kv(x, "int8", "block", bad)
        except ValueError:
            continue
        raise AssertionError(f"G={bad} 不整除 head_dim=32，本该报错")


def test_measured_scale_count_matches_analytic_2_over_G():
    """实测标尺个数 / 值数 必须恰为 1/G（否则 `2/G` 摊薄项只是文档里的传说）。"""
    x = torch.randn(1, 2, 8, 32)
    for g in (8, 16, 32):
        _, acc = kvq.quantize_kv(x, "int8", "block", g)
        assert acc["n_scales"] == x.numel() // g
        assert abs(2.0 * acc["n_scales"] / x.numel() - 2.0 / g) < 1e-9
    _, acc_ts = fa.quantize_fp8_kv(x, "fp8_e4m3", "tokenscale")
    assert acc_ts["n_scales"] == 1 * 2 * 8          # 每 (batch, head, token) 一把
    assert abs(2.0 * acc_ts["n_scales"] / x.numel() - 2.0 / 32) < 1e-9
    _, acc_none = fa.quantize_fp8_kv(x, "fp8_e4m3", "none")
    assert acc_none["n_scales"] == 0, "免标尺档不该有标尺"


def test_fp8_activity_reports_saturation_and_change():
    """fp8 活性量：`saturated_frac` / `changed_frac` / `rel_err` 必须有值且非零（防「只改标签」）。"""
    x = torch.randn(1, 2, 8, 32) * 0.5
    x[0, 0, 0, 0] = 1000.0
    _, acc = fa.quantize_fp8_kv(x, "fp8_e4m3", "none")
    s = fa.summarize_fp8(acc)
    assert s["n_values"] == x.numel()
    assert 0 < s["changed_frac"] <= 1.0
    assert s["rel_err"] > 0
    assert s["n_saturated"] == 1 and abs(s["saturated_frac"] - 1 / x.numel()) < 1e-12


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}", flush=True)
    print(f"{len(fns)} tests passed")
