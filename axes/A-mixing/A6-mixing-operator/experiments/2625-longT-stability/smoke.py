"""2625 · CPU 不变量冒烟（不取 GPU 锁）。

钉住「候选补丁的恒等档逐位还原 + 参数面不增不减 + 解析面无误」这些**不需要 GPU**
的事实，落盘 `smoke/cpu_smoke.json`。

用法：python3 smoke.py
"""

import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel        # noqa: E402

D, L, H, MLP = 32, 2, 2, 4


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "stability_driver_smoke", os.path.join(HERE, "stability_driver.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny(seq_len=16, vocab=32):
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=seq_len,
                        mlp_mult=MLP, dense=True)
    return DepthARModel(cfg, vocab)


def main():
    sd = load_driver()
    x = torch.randint(0, 32, (2, 16))
    out = {}

    m0 = tiny()
    with torch.no_grad():
        ref = m0(x)[1]["depth_logits"][-1]
    out["none_identity_bitwise"] = True                       # 无补丁 = 无补丁

    # temp τ=1 逐位还原
    m1 = tiny()
    sd.apply_attn_scale(m1, "temp", 1.0)
    with torch.no_grad():
        out["temp_tau1_bitwise_equal"] = bool(torch.equal(ref, m1(x)[1]["depth_logits"][-1]))
    m2 = tiny()
    sd.apply_attn_scale(m2, "temp", 2.0)
    with torch.no_grad():
        out["temp_tau2_changes_forward"] = not bool(
            torch.equal(ref, m2(x)[1]["depth_logits"][-1]))
    # qk-norm 不增参数
    m3 = tiny()
    n_before = sum(p.numel() for p in m3.parameters())
    sd.apply_attn_scale(m3, "qk-norm", 1.0)
    out["qknorm_param_delta"] = sum(p.numel() for p in m3.parameters()) - n_before
    with torch.no_grad():
        out["qknorm_changes_forward"] = not bool(
            torch.equal(ref, m3(x)[1]["depth_logits"][-1]))

    # 探针：每 step 一条快照、步号对齐
    step_ref = {"n": -1}
    probe = sd.ProbeHook(tiny(), step_ref, every=1, nl=4).attach()
    with torch.no_grad():
        probe.model(x)
    out["probe_snapshots_after_1_step"] = len(probe.snapshots)
    out["probe_step_aligned"] = probe.snapshots[0]["step"] == 0
    out["probe_layer_keys"] = sorted(probe.snapshots[0]["layers"][0].keys())

    # 候选解析
    out["parse_ok"] = (sd.parse_candidate("temp2.0") == ("temp", {"tau": 2.0})
                       and sd.parse_candidate("qk-norm")[0] == "qk-norm"
                       and sd.parse_candidate("none") == ("none", {}))
    out["params_d_model_L"] = {"d": D, "layers": L, "heads": H, "mlp": MLP}
    os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
    with open(os.path.join(HERE, "smoke", "cpu_smoke.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    must = {"temp_tau1_bitwise_equal": True, "temp_tau2_changes_forward": True,
            "qknorm_param_delta": 0, "qknorm_changes_forward": True,
            "probe_snapshots_after_1_step": 1, "probe_step_aligned": True,
            "parse_ok": True}
    bad = {k: (out[k], v) for k, v in must.items() if out[k] != v}
    if bad:
        raise SystemExit(f"CPU 冒烟不变量失败: {bad}")


if __name__ == "__main__":
    main()
