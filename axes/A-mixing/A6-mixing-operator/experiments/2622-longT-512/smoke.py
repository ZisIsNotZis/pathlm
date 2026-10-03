"""2622 · CPU 侧冒烟（不取 GPU 锁）：四臂配置可构造 + 关键不变量 + 理论表。

GPU 侧的 20 步冒烟/步时/显存见 `run.py --stage preflight`（它写 `smoke/smoke.json`）。
本脚本只跑**不需要 GPU** 的检查，便于在任何时候复核契约骨架。

用法：python3 smoke.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

import torch                                                     # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,        # noqa: E402
                             window_anchor_keep_mask, window_mask_liveness)


def main():
    sys.path.insert(0, HERE)
    import run as R                                              # noqa: E402

    out = {"note": "CPU 冒烟：四臂 config 可构造 + 不变量 + 理论表（无 GPU）", "arms": {}}
    for spec in R.ARMS:
        cfg = R.cfg_for(spec["seq_len"], spec["window"])
        m = DepthARModel(cfg, 206)
        n = sum(p.numel() for p in m.parameters())
        steps = R.steps_for(spec["seq_len"])
        out["arms"][spec["arm"]] = {
            "seq_len": spec["seq_len"], "window": spec["window"],
            "steps": steps, "tokens": steps * R.BS * spec["seq_len"],
            "params": n,
            "window_mask_liveness": (window_mask_liveness(spec["seq_len"],
                                                          spec["window"], 0)
                                     if spec["window"] > 0 else None),
        }
        print(f"{spec['arm']:14s} T={spec['seq_len']} w={spec['window']:3d} "
              f"steps={steps} params={n} "
              f"live={out['arms'][spec['arm']]['window_mask_liveness']}", flush=True)

    # 关键不变量（与 tests/test_longT_512.py 同源，独立再算一遍）
    causal = torch.tril(torch.ones(256, 256, dtype=torch.bool))
    out["invariants"] = {
        "window256@T256_is_identity": bool(torch.equal(
            window_anchor_keep_mask(256, 256, 0), causal)),
        "window256@T512_visible_frac": window_mask_liveness(512, 256, 0)["visible_frac"],
        "window128@T256_visible_frac": window_mask_liveness(256, 128, 0)["visible_frac"],
        "token_budget_equal_across_T": (
            out["arms"]["full-256"]["tokens"] == out["arms"]["full-512"]["tokens"]),
        "param_delta_T512_minus_T256": (out["arms"]["full-512"]["params"]
                                        - out["arms"]["full-256"]["params"]),
    }
    out["theory"] = {
        "attn_share_formula": {str(T): round(R.attn_share_formula(T), 4)
                               for T in (256, 512)},
        "note": "公式只作对照；K10 原 T=512=25.7% 已由主 Agent 修正为 25.0%",
    }
    os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "smoke", "cpu_smoke.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out["invariants"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
