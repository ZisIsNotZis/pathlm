"""2624 · CPU 侧冒烟（不取 GPU 锁）：两臂配置可构造 + 关键不变量 + 种子活性。

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

    out = {"note": "CPU 冒烟：两臂 config 可构造 + 不变量 + 种子活性（无 GPU）",
           "seed": R.SEED, "arms": {}}
    full_arm, win_arm = R.ARMS[0]["arm"], R.ARMS[1]["arm"]
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
        print(f"{spec['arm']:20s} T={spec['seq_len']} w={spec['window']:3d} "
              f"steps={steps} params={n} "
              f"live={out['arms'][spec['arm']]['window_mask_liveness']}", flush=True)

    # 种子活性（§10）：同配置、不同 seed ⇒ 初始参数必须不同（seed 是活因子）
    def _init_norm(seed):
        torch.manual_seed(seed)
        return float(sum(p.detach().double().abs().sum()
                         for p in DepthARModel(R.cfg_for(1024, 512), 206).parameters()))
    out["seed_liveness"] = {"abs_sum_seed0": _init_norm(0), "abs_sum_seed1": _init_norm(1),
                            "alive": _init_norm(0) != _init_norm(1)}

    # 关键不变量（与 tests/test_longT_1024_seed1.py 同源，独立再算一遍）
    out["invariants"] = {
        "window512@T1024_is_nondegenerate": not bool(torch.equal(
            window_anchor_keep_mask(1024, 512, 0),
            torch.tril(torch.ones(1024, 1024, dtype=torch.bool)))),
        "window1024@T1024_is_identity": bool(torch.equal(
            window_anchor_keep_mask(1024, 1024, 0),
            torch.tril(torch.ones(1024, 1024, dtype=torch.bool)))),
        "window512@T1024_visible_frac": window_mask_liveness(1024, 512, 0)["visible_frac"],
        "token_budget_equal_across_arms": (
            out["arms"][full_arm]["tokens"] == out["arms"][win_arm]["tokens"]),
        "seed_equal_across_arms": R.SEED == 1,
        "param_delta_T1024_minus_T512": (out["arms"][full_arm]["params"]
                                         - 7_028_224),
    }
    # 与前两点同相对刻度（SCHEMA §17.16）
    out["three_points"] = [{"window": w, "T": T, "ratio": w / T,
                            "visible_frac": window_mask_liveness(T, w, 0)["visible_frac"]}
                           for w, T in R.POINTS_THREE]
    out["theory"] = {
        "attn_share_formula": {str(T): round(R.attn_share_formula(T), 4)
                               for T in (256, 512, 1024)},
        "note": "公式只作对照；T=1024 = 40.0%",
    }
    os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "smoke", "cpu_smoke.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out["invariants"], indent=2, ensure_ascii=False))
    print(json.dumps(out["seed_liveness"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
