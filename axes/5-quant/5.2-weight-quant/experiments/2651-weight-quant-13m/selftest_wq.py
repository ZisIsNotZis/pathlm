"""2651 · 权重量化 WP1 —— 语义层 `wq_arms.py` 的**自带单测 + 变异验证**（纯 CPU，无 GPU / 无 checkpoint）。

用法：

    python3 -m pytest selftest_wq.py -q      # 单测
    python3 selftest_wq.py                   # 单测 + 进程内变异电池 → mutation_report.json

变异电池（`M1..M9`）对 `wq_arms.py` 打**进程内猴子补丁**，故意破坏「量化 / 标尺 / 位数 / 字节账 /
还原 / 死权重过滤」，要求对应单测**报红**；只有 `test_failure` 算抓红（`tools.evalkit.mutation`
的分类），未抓住 ⇒ 非零退出。补丁在 `finally` 里还原，本文件不改任何磁盘文件。
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import wq_arms as wq                                                   # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel                 # noqa: E402
from tools.evalkit.retention import masked_logits                      # noqa: E402

#: 小模型（纯 CPU，秒级）：12 层 × 256 的 trunk 的缩比，只用来验语义层的**性质**，
#: 不产生任何结论数字。
TINY = dict(d_model=32, n_layers=2, n_heads=2, seq_len=64, mlp_mult=4, dense=True)


def tiny_model() -> DepthARModel:
    torch.manual_seed(0)
    m = DepthARModel(DepthARConfig(**TINY), 206)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def tiny_tokens() -> torch.Tensor:
    return torch.randint(0, 200, (2, 64), generator=torch.Generator().manual_seed(3))


# ----------------------------------------------------------------- 单测


def test_declared_caliber_is_fp16_scale():
    """判据线声明：标尺 fp16 存储 = 2 B/把；四个臂的 `(bits, scheme, group)` 逐字钉住。"""
    assert wq.SCALE_BYTES == 2.0
    assert wq.qmax_of(8) == 127 and wq.qmax_of(4) == 7
    assert [(s.name, s.bits, s.scheme, s.group) for s in wq.ARM_SPECS.values()] == [
        ("fp16", 16, "identity", None),
        ("int8_per_channel", 8, "per_channel", None),
        ("int8_group32", 8, "group", 32),
        ("int4_group32", 4, "group", 32)]
    assert wq.PRIMARY_GROUP == 32


def test_fake_quantize_roundtrip_error_bound():
    """对称量化的取整误差上界 `|ŵ − w| ≤ 0.5·s_max`（不触发额外 clip 误差）。"""
    torch.manual_seed(1)
    w = torch.randn(64, 128) * 0.02
    for bits in (8, 4):
        for scheme, group in (("per_channel", None), ("group", 32)):
            _, st = wq.fake_quantize_weight(w, bits, scheme, group)
            assert st["rounding_bound_ok"], (bits, scheme, group, st["max_abs_err"])
            assert st["max_abs_err"] <= 0.5 * st["scale_max"] * (1 + 1e-6)
            assert st["changed_frac"] > 0.0, "量化没有改任何权重 ⇒ 死干预"


def test_per_channel_scale_count_is_n_out():
    """`per_channel` 的标尺个数必须**正好等于输出通道数**。"""
    for n_out, n_in in ((768, 256), (256, 1536)):
        assert wq.scale_count(n_out, n_in, "per_channel") == n_out
        assert wq.matrix_bytes(n_out, n_in, 8, "per_channel")["n_scales"] == n_out


def test_group_scale_count_and_divisibility():
    """`group G` 标尺个数 = `n_out·n_in/G`；不能整除 ⇒ 必须报错（块不整齐就不成立）。"""
    for n_out, n_in, g in ((768, 256, 32), (256, 1536, 128), (256, 256, 256)):
        assert wq.scale_count(n_out, n_in, "group", g) == n_out * n_in // g
    try:
        wq.scale_count(64, 100, "group", 32)
    except ValueError:
        pass
    else:
        raise AssertionError("group=32 不整除 in_features=100 时必须报错")


def test_byte_account_matches_analytic_formula():
    """逐矩阵字节账 = `bits/8 + 2·n_scales/n_params`（标尺**不得**当免费）。"""
    m = tiny_model()
    for name, spec in wq.ARM_SPECS.items():
        acct = wq.byte_account(m, spec)
        for r in acct["per_matrix"]:
            n_out, n_in = r["shape"]
            expect_scales = (0 if spec.scheme == "identity" else
                             n_out if spec.scheme == "per_channel" else
                             n_out * n_in // spec.group)
            assert r["n_scales"] == expect_scales, (name, r["name"], r["n_scales"])
            expect_bpp = spec.bits / 8.0 + 2.0 * expect_scales / (n_out * n_in)
            assert abs(r["bytes_per_param"] - expect_bpp) < 1e-12, (name, r["name"])
        assert abs(sum(r["total_bytes"] for r in acct["per_matrix"])
                   - acct["scope_linear_all"]["total_bytes"]) < 1e-6
    # 解析式单点：`per_channel` = 1 + 2/n_in（每行一把标尺）；`group32` int4 = 0.5 + 2/32
    assert abs(wq.matrix_bytes(768, 256, 8, "per_channel")["bytes_per_param"]
               - (1.0 + 2.0 / 256)) < 1e-12
    assert abs(wq.matrix_bytes(768, 256, 4, "group", 32)["bytes_per_param"]
               - (0.5 + 2.0 / 32)) < 1e-12


def test_fp16_arm_is_bitwise_identity():
    """非量化臂（`identity`）必须**逐位**等于 stock 前向：logits 不变、权重不变、无补丁。"""
    m = tiny_model()
    tok = tiny_tokens()
    base = masked_logits(m, tok, None)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    spec = wq.arm_spec("fp16")
    with wq.arm_context(m, spec):
        got = masked_logits(m, tok, None)
        assert all("forward" not in mod.__dict__
                   for _, mod in wq.linear_weight_matrices(m)), "fp16 臂不该装任何补丁"
        assert torch.equal(base, got), "fp16 臂的 logits 必须逐位等于 stock"
    assert torch.equal(masked_logits(m, tok, None), base)
    assert all(torch.equal(before[n], p.detach()) for n, p in m.named_parameters())
    act = wq.activity_of(m, spec)
    assert act["changed_frac"] == 0.0 and act["max_abs_err"] == 0.0
    acct = wq.byte_account(m, spec)
    assert acct["scope_linear_all"]["bytes_per_param"] == 2.0
    assert acct["scope_whole_model"]["bytes_per_param"] == 2.0


def test_quantized_arms_change_logits_and_restore_bitwise():
    """量化臂必须真的改 logits（干预活着），且退出后**逐位还原**（权重张量没被改）。"""
    m = tiny_model()
    tok = tiny_tokens()
    base = masked_logits(m, tok, None)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    for name in ("int8_per_channel", "int8_group32", "int4_group32"):
        spec = wq.arm_spec(name)
        with wq.arm_context(m, spec):
            assert not torch.equal(masked_logits(m, tok, None), base), f"{name} 的干预是死的"
        assert torch.equal(masked_logits(m, tok, None), base), f"{name} 未逐位还原"
        assert all(torch.equal(before[n], p.detach()) for n, p in m.named_parameters())
        assert all("forward" not in mod.__dict__
                   for _, mod in wq.linear_weight_matrices(m)), f"{name} 的补丁没拆掉"
        act = wq.activity_of(m, spec)
        assert act["changed_frac"] > 0.0 and act["rel_err"] > 0.0
        assert act["scales_match_analytic"] and not act["bound_failures"]


def test_bits_and_group_monotonicity():
    """位数越低误差越大；同一码位下更细的标尺（group32）误差 ≤ per_channel。"""
    m = tiny_model()
    e8g = wq.activity_of(m, wq.arm_spec("int8_group32"))["rel_err"]
    e4g = wq.activity_of(m, wq.arm_spec("int4_group32"))["rel_err"]
    e8p = wq.activity_of(m, wq.arm_spec("int8_per_channel"))["rel_err"]
    assert e4g > e8g, f"int4 的相对误差应大于 int8：{e4g} vs {e8g}"
    assert e8g <= e8p + 1e-9, f"group32 标尺更细，误差不应大于 per_channel：{e8g} vs {e8p}"


def test_dead_gates_are_excluded_from_quality_evidence():
    """`gates.*` 是 dense 前向的死权重：清零不改 logits ⇒ 只进字节账，不进质量读数。"""
    m = tiny_model()
    probe = wq.dead_gate_probe(m, tiny_tokens())
    assert probe["zeroed_logits_bitwise_equal"] is True
    assert probe["restored_bitwise"] is True
    assert probe["n_dead_modules"] > 0
    all_names = [n for n, _ in wq.linear_weight_matrices(m, include_dead_gates=True)]
    live_names = [n for n, _ in wq.linear_weight_matrices(m, include_dead_gates=False)]
    assert len(live_names) == len(all_names) - probe["n_dead_modules"]
    assert all(wq.is_dead_linear(n) for n in set(all_names) - set(live_names))
    acct = wq.byte_account(m, wq.arm_spec("int8_group32"))
    assert acct["scope_linear_all"]["n_params"] > acct["scope_linear_live"]["n_params"]
    assert acct["scope_linear_all"]["n_scales"] > acct["scope_linear_live"]["n_scales"]


def test_group_ladder_flags_invalid_groups():
    """阶梯把「不能整除所有 in_features 的 G」标成无效（不许给出一个虚假的字节数）。"""
    m = tiny_model()
    ladder = wq.byte_account(m, wq.arm_spec("fp16"))["ladder"]
    by_key = {(r["scheme"], r["group"]): r for r in ladder}
    assert by_key[("fp16", None)]["bytes_per_param"] == 2.0
    for g in wq.GROUP_LADDER:
        row = by_key[("group", g)]
        if all(int(mod.weight.shape[1]) % g == 0
               for _, mod in wq.linear_weight_matrices(m)):
            assert abs(row["bytes_per_param"] - (1.0 + 2.0 / g)) < 1e-12, g
        else:
            assert row["bytes_per_param"] is None, g


# ----------------------------------------------------------------- 进程内变异电池

_ORIG_FAKE_QUANT = wq.fake_quantize_weight
_ORIG_SCALE_COUNT = wq.scale_count
_ORIG_ARM_SPEC = wq.arm_spec


def _broken_no_round(w, bits, scheme, group=None):
    """M1：反量化不取整（量化退化为恒等）⇒ 活性 / 误差界检查必须报红。"""
    _, st = _ORIG_FAKE_QUANT(w, bits, scheme, group)
    st = dict(st)
    st.update(changed_frac=0.0, rel_err=0.0, max_abs_err=0.0)
    return w.clone(), st


def _broken_per_channel_count(n_out, n_in, scheme, group=None):
    """M2：per_channel 的标尺个数写成 1（全张量一把）⇒ 字节账与实现脱钩。"""
    if scheme == "per_channel":
        return 1
    return _ORIG_SCALE_COUNT(n_out, n_in, scheme, group)


def _broken_group_no_check(n_out, n_in, scheme, group=None):
    """M7：`group` 去掉整除校验，用向上取整算标尺个数（不整齐的块也能出数）。"""
    if scheme == "group":
        if not group or group < 1:
            raise ValueError(f"group 方案要求 group >= 1，收到 {group!r}")
        return n_out * max(1, -(-n_in // group))
    return _ORIG_SCALE_COUNT(n_out, n_in, scheme, group)


def _broken_apply_arm_noop(model, spec, include_dead_gates=True):
    """M5：施加器变成空操作（臂根本没装上）。"""
    return lambda: None


def _broken_apply_arm_leaky(model, spec, include_dead_gates=True):
    """M6：照旧装补丁，但**不拆**（返回空的 restore）⇒ 逐位还原证据恒真。"""
    if spec.scheme == "identity":
        return lambda: None
    for _name, mod in wq.linear_weight_matrices(model, include_dead_gates):
        w_hat, _ = _ORIG_FAKE_QUANT(mod.weight.detach(), spec.bits, spec.scheme, spec.group)
        bias = mod.bias

        def fwd(x, _w=w_hat, _b=bias):
            return torch.nn.functional.linear(x, _w, _b)

        mod.forward = fwd
    return lambda: None                                  # 变异：不拆补丁


def _broken_fp16_is_quantized(name):
    """M9：`fp16` 臂被当成 int8 per-channel 装上（基线不再逐位等于 stock）。"""
    if name == "fp16":
        return wq.ArmSpec("fp16", 8, "per_channel")
    return _ORIG_ARM_SPEC(name)


def _mutated(attr: str, value, test_fn):
    """打补丁 → 跑测试 → `finally` 还原；异常由 `mutation.inprocess_outcome` 分类。"""
    orig = getattr(wq, attr)
    setattr(wq, attr, value)
    try:
        test_fn()
    finally:
        setattr(wq, attr, orig)


def _no_patch_test():
    m = tiny_model()
    tok = tiny_tokens()
    base = masked_logits(m, tok, None)
    with wq.arm_context(m, wq.arm_spec("int4_group32")):
        assert not torch.equal(masked_logits(m, tok, None), base), "补丁没生效"


def _restore_leak_test():
    m = tiny_model()
    tok = tiny_tokens()
    with wq.arm_context(m, wq.arm_spec("int8_group32")):
        masked_logits(m, tok, None)
    assert all("forward" not in mod.__dict__
               for _, mod in wq.linear_weight_matrices(m)), "补丁没拆掉"


def mutation_specs() -> list[dict]:
    """`M1..M9`：每条 = 目标属性 + 破坏值 + 期望报红的单测。每条测试**自建模型**（不共享状态）。"""
    return [
        {"id": "M1", "attr": "fake_quantize_weight", "value": _broken_no_round,
         "what": "反量化不取整（量化退化成恒等 / 死干预）",
         "test": lambda: test_fake_quantize_roundtrip_error_bound()},
        {"id": "M2", "attr": "scale_count", "value": _broken_per_channel_count,
         "what": "per_channel 标尺个数写成 1（字节账与实现脱钩）",
         "test": lambda: test_per_channel_scale_count_is_n_out()},
        {"id": "M3", "attr": "SCALE_BYTES", "value": 0.0,
         "what": "把标尺当免费（历史 5.1 的口径错误复现）",
         "test": lambda: test_byte_account_matches_analytic_formula()},
        {"id": "M4", "attr": "QMAX", "value": {8: 127, 4: 127},
         "what": "int4 用 int8 的上界（位数不再起作用）",
         "test": lambda: test_bits_and_group_monotonicity()},
        {"id": "M5", "attr": "apply_arm", "value": _broken_apply_arm_noop,
         "what": "施加器变成空操作（臂没装上）", "test": _no_patch_test},
        {"id": "M6", "attr": "apply_arm", "value": _broken_apply_arm_leaky,
         "what": "还原不拆补丁（逐位还原证据恒真）", "test": _restore_leak_test},
        {"id": "M7", "attr": "scale_count", "value": _broken_group_no_check,
         "what": "去掉 G | n_in 的整除校验（不整齐的块也算得出字节）",
         "test": lambda: test_group_scale_count_and_divisibility()},
        {"id": "M8", "attr": "is_dead_linear", "value": lambda name: False,
         "what": "死权重过滤失效（把 gates 当活权重报质量证据）",
         "test": lambda: test_dead_gates_are_excluded_from_quality_evidence()},
        {"id": "M9", "attr": "arm_spec", "value": _broken_fp16_is_quantized,
         "what": "fp16 臂被装上量化（基线不再逐位等于 stock）",
         "test": lambda: test_fp16_arm_is_bitwise_identity()},
    ]


def run_mutations() -> int:
    from tools.evalkit import mutation

    report = {"targets": {os.path.relpath(__file__, ROOT):
                          {"note": "进程内猴子补丁，磁盘字节未改（无需 sha256 对照）"}},
              "mutations": []}
    ok = True
    for spec in mutation_specs():
        caught, detail, status = mutation.inprocess_outcome(
            _mutated, args=(spec["attr"], spec["value"], spec["test"]))
        ok = ok and caught
        report["mutations"].append({"id": spec["id"], "what": spec["what"],
                                    "status": status, "caught_red": caught,
                                    "detail": detail})
        print(f"[mutation] {spec['id']}: {status} caught={caught} :: {detail}", flush=True)
    report["all_caught"] = ok
    mutation.exit_if_uncaught(ok, report, os.path.join(HERE, "mutation_report.json"))
    return 0


# ----------------------------------------------------------------- 入口


def main() -> int:
    import pytest
    rc = int(pytest.main([os.path.abspath(__file__), "-q"]))
    if rc != 0:
        print(f"[selftest] 单测未过（rc={rc}）；变异电池不跑", file=sys.stderr)
        return rc
    return run_mutations()


if __name__ == "__main__":
    sys.exit(main())
