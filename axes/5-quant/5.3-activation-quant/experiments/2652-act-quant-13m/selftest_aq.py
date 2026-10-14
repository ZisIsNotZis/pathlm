"""2652 · `aq_arms.py` 的可失败单测 + 变异验证（纯 CPU，不取 GPU 锁）。

用法：
  python3 selftest_aq.py            # 跑全部单测 + 变异验证（要求全部抓红）
  python3 selftest_aq.py --tests    # 只跑单测（应全绿）
  python3 -m pytest -q selftest_aq.py

变异纪律（task §4 必报护栏 ③）：对 `aq_arms.py` 做**故意破坏**（量化 / 标尺 / 位数 /
忽略干预），每次破坏后跑对应单测，要求**报红**；随后逐字节还原（sha256 校验）。
未抓到 ⇒ 非零退出。这正是「断言必须能失败」的可失败性证据（policy::proof.gate_probe）。
"""

import hashlib
import json
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import aq_arms as aq                                                    # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel                 # noqa: E402
from tools.evalkit.retention import masked_logits                       # noqa: E402

CFG = json.load(open(os.path.join(HERE, "config.json")))


def tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=32, n_layers=2, n_heads=4, seq_len=16,
                        mlp_mult=2, dense=True)
    m = DepthARModel(cfg, vocab_size=32)
    m.eval()
    return m


# --------------------------------------------------------------- 单测


def test_fp16_mode_is_bitwise_identity():
    torch.manual_seed(0)
    x = torch.randn(2, 7, 16)
    dq, scale, ratio = aq.fake_quant_activation(x, "fp16")
    assert dq is x, "fp16 档必须是逐位恒等（同一对象），不能引入任何往返"
    assert scale is None and ratio is None


def test_scale_equals_row_amax_over_qmax():
    torch.manual_seed(1)
    x = torch.randn(3, 5, 8)
    _, scale, _ = aq.fake_quant_activation(x, "int8")
    ref = x.abs().amax(dim=-1, keepdim=True) / 127.0
    assert torch.allclose(scale, ref.clamp_min(1e-12), atol=1e-12), \
        "标尺必须恰好等于该行 amax/qmax"


def test_per_token_scale_is_one_per_row():
    torch.manual_seed(2)
    x = torch.randn(4, 6, 32)
    _, scale, _ = aq.fake_quant_activation(x, "int8")
    assert scale.shape == (4, 6, 1), f"per-token 标尺形状应为 (B,T,1)，得 {scale.shape}"
    assert scale.numel() == 24


def test_int8_roundtrip_error_bound():
    torch.manual_seed(3)
    x = torch.randn(5, 9, 24) * 3.0
    dq, scale, _ = aq.fake_quant_activation(x, "int8")
    err = (dq - x).abs()
    assert float(err.max()) <= 0.5 * float(scale.max()) + 1e-9, \
        "对称取整往返误差上界 = 0.5·scale"


def test_int8_changes_values():
    torch.manual_seed(4)
    x = torch.randn(4, 4, 16)
    dq, _, _ = aq.fake_quant_activation(x, "int8")
    assert float((dq != x).float().mean()) > 0.5, "int8 必须真的改数值（干预是活的）"


def test_int4_uses_seven_levels_and_coarser_quantum():
    torch.manual_seed(5)
    x = torch.randn(3, 3, 16)
    _, s8, _ = aq.fake_quant_activation(x, "int8")
    _, s4, _ = aq.fake_quant_activation(x, "int4")
    assert torch.allclose(s4, s8 * 127 / 7, rtol=1e-6), \
        "int4 的标尺必须是 int8 的 127/7 倍（同一个 amax，位数不同）"
    dq4, _, _ = aq.fake_quant_activation(x, "int4")
    assert float((dq4 - x).abs().max()) <= 0.5 * float(s4.max()) + 1e-9


def test_qmax_values_are_the_bit_width_definition():
    assert aq._qmax("int8") == 127 and aq._qmax("int4") == 7
    assert aq._BITS == {"int8": 8, "int4": 4}


def test_activation_bytes_per_token_matches_formula():
    b = aq.activation_bytes_per_token(CFG, "int8")
    assert b["elements_per_token_total"] == 27648
    assert b["n_tensors_total"] == 48
    assert b["bytes_per_token_activation"] == 27744.0, b["bytes_per_token_activation"]
    assert aq.activation_bytes_per_token(CFG, "int4")["bytes_per_token_activation"] == 13920.0
    assert aq.activation_bytes_per_token(CFG, "fp16")["bytes_per_token_activation"] == 55296.0


def test_activation_bytes_increase_with_bits():
    a4 = aq.activation_bytes_per_token(CFG, "int4")["bytes_per_token_activation"]
    a8 = aq.activation_bytes_per_token(CFG, "int8")["bytes_per_token_activation"]
    f16 = aq.activation_bytes_per_token(CFG, "fp16")["bytes_per_token_activation"]
    assert a4 < a8 < f16


def test_scale_overhead_is_two_bytes_per_tensor_per_token():
    b = aq.activation_bytes_per_token(CFG, "int8")
    assert b["scale_bytes_per_token"] == 2.0 * 4 * 12, b["scale_bytes_per_token"]
    assert aq.activation_bytes_per_token(CFG, "fp16")["scale_bytes_per_token"] == 0.0
    # 值字节部分 = fp16 的一半（int8）
    assert (b["bytes_per_token_activation"] - b["scale_bytes_per_token"]) * 2 \
        == aq.activation_bytes_per_token(CFG, "fp16")["bytes_per_token_activation"]


def test_weight_bytes_split_and_per_channel_scale():
    w8 = aq.weight_bytes(CFG, "int8")
    w16 = aq.weight_bytes(CFG, "fp16")
    assert w8["n_params_quantized_linears"] == 12582912
    assert w8["bytes"] == 12582912 + 2816 * 2 * 12, w8["bytes"]
    assert w16["bytes"] == 2 * 12582912
    w = torch.randn(6, 4)
    s = aq.weight_quant_scale(w, "int8")
    assert s.shape == (6, 1), f"权重标尺必须 per-output-channel，得 {s.shape}"
    assert torch.allclose(s, (w.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-12),
                          atol=1e-12)


def test_aq_forward_fp16_matches_masked_logits_bitwise():
    m = tiny_model()
    torch.manual_seed(11)
    tok = torch.randint(0, 32, (2, 16))
    fwd = aq.AqForward(m, {"activation_mode": "fp16", "weight_mode": "fp16"})
    with torch.no_grad():
        a = fwd.logits(tok)
        b = masked_logits(m, tok, None)
    assert torch.equal(a, b), "fp16 档的前向必须与既有仪器逐位相同（硬护栏 ②）"


def test_aq_forward_int8_changes_logits():
    m = tiny_model()
    torch.manual_seed(11)
    tok = torch.randint(0, 32, (2, 16))
    f8 = aq.AqForward(m, {"activation_mode": "fp16", "weight_mode": "fp16"})
    fq = aq.AqForward(m, {"activation_mode": "int8", "weight_mode": "fp16"})
    with torch.no_grad():
        a, b = f8.logits(tok), fq.logits(tok)
    assert not torch.equal(a, b), "int8 激活必须真的改变前向（干预不是死的）"


def test_aq_forward_w8a8_changes_logits_and_weight_cache_used():
    m = tiny_model()
    torch.manual_seed(11)
    tok = torch.randint(0, 32, (2, 16))
    fq = aq.AqForward(m, {"activation_mode": "int8", "weight_mode": "fp16"})
    fw = aq.AqForward(m, {"activation_mode": "int8", "weight_mode": "int8"})
    with torch.no_grad():
        a, b = fq.logits(tok), fw.logits(tok)
    assert not torch.equal(a, b), "组合臂 W8A8 必须真的改变前向"
    ws = fw.weight_activity()
    # tiny 模型（d=32/L=2/mlp2）被量化 Linear 的参数量 = 2·(3d²+d²+2d·2d) = 16384
    assert ws["w_mode"] == "int8" and ws["changed_frac"] > 0.5
    assert ws["n_values"] == 16384, ws["n_values"]


def test_activity_stats_reconcile_scale_count_and_changed_frac():
    m = tiny_model()
    torch.manual_seed(12)
    tok = torch.randint(0, 32, (2, 16))
    spec = {"activation_mode": "int8", "weight_mode": "fp16"}
    fwd = aq.AqForward(m, spec)
    stats = aq.new_stats(spec)
    with torch.no_grad():
        fwd.logits(tok, stats)
    out = aq.summarize_stats(stats, n_layers=2, n_tensors_per_layer=4)
    B, T = tok.shape
    assert out["aggregate"]["n_tensors_observed"] == 8
    assert out["aggregate"]["n_scales"] == 8 * B * T, out["aggregate"]["n_scales"]
    assert out["aggregate"]["changed_frac"] > 0.5


def test_outlier_index_and_effective_bits():
    x = torch.ones(4, 32)                       # 32 个通道，全部量级 1
    x[:, 0] = 100.0                             # 通道 0 是离群通道（100×）
    dq, scale, ratio = aq.fake_quant_activation(x, "int8")
    spec = {"activation_mode": "int8", "weight_mode": "fp16"}
    stats = aq.new_stats(spec)
    aq._record_activation(x, dq, scale, ratio, "int8", stats, 0, "qkv")
    out = aq.summarize_stats(stats, n_layers=1, n_tensors_per_layer=1)
    t = out["per_tensor"]["L00.qkv"]
    assert abs(t["chan_amax_outlier_index"] - 100.0) < 1e-3, t["chan_amax_outlier_index"]
    assert abs(t["effective_bits_median_channel"] - (8 - 6.643856189774725)) < 1e-4
    assert abs(t["revert_chan_frac_6x"] - 1 / 32) < 1e-9, t["revert_chan_frac_6x"]


def test_revert_byte_penalty_math():
    spec = {"activation_mode": "int8", "weight_mode": "fp16"}
    x = torch.ones(2, 64) * 0.5
    x[:, 3] = 50.0
    dq, scale, ratio = aq.fake_quant_activation(x, "int8")
    stats = aq.new_stats(spec)
    aq._record_activation(x, dq, scale, ratio, "int8", stats, 0, "mlp0")
    out = aq.summarize_stats(stats, n_layers=1, n_tensors_per_layer=1)
    pen = aq.revert_byte_penalty(out, CFG)
    assert pen["applicable"] is True
    assert pen["n_revert_channels_total"] == 1
    # 每 token 回退 1 个通道 ⇒ 额外 1 B/token（fp16 2 B − int8 1 B）
    assert abs(pen["revert_extra_bytes_per_token"] - 1.0) < 1e-9
    assert 0 < pen["saving_eaten_frac"] < 1e-3


# --------------------------------------------------------------- 变异验证

MUTATIONS = [
    {"id": "M1", "what": "位数定义被改（int8 当成 16 位）",
     "old": '_BITS = {"int8": 8, "int4": 4}',
     "new": '_BITS = {"int8": 16, "int4": 4}  # 变异',
     "tests": ["test_qmax_values_are_the_bit_width_definition",
               "test_activation_bytes_per_token_matches_formula"]},
    {"id": "M2", "what": "per-token 标尺退化成 per-tensor（整批一把标尺）",
     "old": "    amax = x.abs().amax(dim=-1, keepdim=True)",
     "new": "    amax = x.abs().amax()  # 变异：整批一把标尺",
     "tests": ["test_per_token_scale_is_one_per_row",
               "test_scale_equals_row_amax_over_qmax"]},
    {"id": "M3", "what": "标尺忘了除 qmax（量化台阶错）",
     "old": "    scale = (amax / qmax).clamp_min(1e-12)",
     "new": "    scale = amax.clamp_min(1e-12)  # 变异：忘了除 qmax",
     "tests": ["test_scale_equals_row_amax_over_qmax",
               "test_int4_uses_seven_levels_and_coarser_quantum"]},
    {"id": "M4", "what": "标尺当免费（激活字节账不含标尺）",
     "old": '    scale_per_tensor_per_token = 0.0 if act_mode == "fp16" else SCALE_BYTES',
     "new": "    scale_per_tensor_per_token = 0.0  # 变异：标尺当免费",
     "tests": ["test_scale_overhead_is_two_bytes_per_tensor_per_token"]},
    {"id": "M5", "what": "整型档根本不量化（干预是死的）",
     "old": "        dq, scale, ratio = fake_quant_activation(x, mode)",
     "new": "        return x  # 变异：整型档不量化\n        dq, scale, ratio = fake_quant_activation(x, mode)",
     "tests": ["test_aq_forward_int8_changes_logits"]},
    {"id": "M6", "what": "权重标尺退化成 per-tensor（per-output-channel 语义丢失）",
     "old": "    amax = w.abs().amax(dim=1, keepdim=True)",
     "new": "    amax = w.abs().amax()  # 变异：整张权重一把标尺",
     "tests": ["test_weight_bytes_split_and_per_channel_scale"]},
    {"id": "M7", "what": "fp16 档偷偷走一次 fp16 往返（恒等性被破坏）",
     "old": '    if mode == "fp16":\n        return x, None, None',
     "new": ('    if mode == "fp16":\n'
             '        return x.to(torch.float16).to(x.dtype), None, None  # 变异'),
     "tests": ["test_fp16_mode_is_bitwise_identity",
               "test_aq_forward_fp16_matches_masked_logits_bitwise"]},
]
TARGET = os.path.join(HERE, "aq_arms.py")


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _run_pytest(names):
    out = {}
    for n in names:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q",
                            f"{os.path.abspath(__file__)}::{n}"],
                           cwd=HERE, capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or ["(no stdout)"])[-1][:160]
        out[n] = {"returncode": r.returncode,
                  "verdict": "FAILED（抓红）" if r.returncode != 0 else "passed（未抓到）",
                  "tail": tail}
    return out


def run_mutations() -> dict:
    original = open(TARGET).read()
    sha_before = _sha(TARGET)
    report = {"target": os.path.relpath(TARGET, ROOT), "sha256_before": sha_before,
              "mutations": []}
    try:
        for m in MUTATIONS:
            assert original.count(m["old"]) == 1, f"{m['id']}: 定位失败"
            open(TARGET, "w").write(original.replace(m["old"], m["new"]))
            assert _sha(TARGET) != sha_before, f"{m['id']}: 文件未变"
            res = _run_pytest(m["tests"])
            caught = all(v["returncode"] != 0 for v in res.values())
            report["mutations"].append({"id": m["id"], "what": m["what"],
                                        "tests": res, "all_expected_tests_caught": caught})
            print(f"[{m['id']}] {m['what']}: {'全部抓红 ✓' if caught else '未抓到 ✗'}",
                  flush=True)
            open(TARGET, "w").write(original)
            assert _sha(TARGET) == sha_before, f"{m['id']}: 还原失败"
    finally:
        open(TARGET, "w").write(original)
    report["sha256_after"] = _sha(TARGET)
    report["restored_sha256_equal"] = report["sha256_after"] == sha_before
    report["all_caught"] = all(m["all_expected_tests_caught"]
                               for m in report["mutations"])
    with open(os.path.join(HERE, "mutation_report.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    return report


def main() -> int:
    if "--tests" in sys.argv:
        return subprocess.run([sys.executable, "-m", "pytest", "-q",
                               os.path.abspath(__file__)], cwd=HERE).returncode
    green = subprocess.run([sys.executable, "-m", "pytest", "-q",
                            os.path.abspath(__file__)], cwd=HERE,
                           capture_output=True, text=True)
    print(green.stdout.strip().splitlines()[-1], flush=True)
    if green.returncode != 0:
        print(green.stdout)
        print("单测未全绿 ⇒ 先修单测，不做变异验证", flush=True)
        return 1
    rep = run_mutations()
    print(json.dumps({"green_before": True,
                      "restored_sha256_equal": rep["restored_sha256_equal"],
                      "all_caught": rep["all_caught"]}, indent=2), flush=True)
    return 0 if (rep["all_caught"] and rep["restored_sha256_equal"]) else 2


if __name__ == "__main__":
    sys.exit(main())
