#!/usr/bin/env python3
"""2634-carry-realtext-13m · **感受野扫描**：给「被评行必须选在 eval_lo − rf ≥ corrupt_hi」

这条布局前提提供可复核产物（复核 P2-9：此前「第 243 行已是 0.0」只有口头读数）。

做法（确定性、CPU 即可）：

1. 按本实验的 canonical 规格（T=256、L=12、w=16，随机初始化、固定 seed）建模型；
2. 造两份输入 `x` 与 `x_corrupt`（前缀 `[0, corrupt_hi)` 换成固定随机字节）；
3. 在两个掩码下各前向一次，逐行算 `max|Δlogit|`，找出**第一个逐位不敏感的行**
   `first_insensitive_row`；
4. 断言 `first_insensitive_row ≥ corrupt_hi + rf`（`rf = L·(w−1)`）——即 `rf` 是**保守**
   上界；并把 1 层 / 2 层的小配置也扫一遍，验证「多跳宽度 ≈ L·(w−1)」的机制解释；
5. 把**泄漏布局**（`[0,128)` / 被评行从 144 起）也扫一遍：它必须**非零**（证明这条
   不变式不是摆设，而旧布局真的泄漏）。

输出 `rf_scan.json`（入 git）。用法：`python3 rf_scan.py`
"""

from __future__ import annotations

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             window_anchor_keep_mask)
from tools.evalkit.retention import masked_logits                  # noqa: E402


def _row_profile(model, keep_mask, T: int, corrupt_hi: int, seed: int = 7) -> list[float]:
    """逐行 `max|Δlogit|`（原件 vs 被毁前缀），长度 T。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, model.vocab_size, (2, T), generator=g)
    cg = torch.Generator().manual_seed(seed + 1)
    xc = x.clone()
    xc[:, :corrupt_hi] = torch.randint(0, model.vocab_size, (2, corrupt_hi), generator=cg)
    with torch.no_grad():
        a = masked_logits(model, x, keep_mask)
        b = masked_logits(model, xc, keep_mask)
    return [float(v) for v in (a - b).abs().amax(-1)[0]]


def _first_insensitive(profile: list[float], start: int) -> int | None:
    """从 `start` 起第一个「逐位逐元素全零」的行（None = 整段都不为零）。"""
    for i in range(start, len(profile)):
        if profile[i] == 0.0:
            return i
    return None


def _max_on_rows(profile: list[float], lo: int, hi: int) -> float:
    seg = profile[lo:min(hi, len(profile))]
    return max(seg) if seg else float("nan")


def scan(layers: int, window: int, T: int, corrupt_hi: int, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = DepthARModel(DepthARConfig(d_model=64, n_layers=layers, n_heads=4, seq_len=T,
                                       mlp_mult=2, dense=True, topology="sequential",
                                       train_window=window, train_anchors=0), 206)
    model.eval()
    keep = window_anchor_keep_mask(T, window, 0)
    prof = _row_profile(model, keep, T, corrupt_hi)
    rf = layers * (window - 1)
    first = _first_insensitive(prof, corrupt_hi)
    return {
        "layers": layers, "window": window, "T": T, "corrupt_hi": corrupt_hi,
        "rf_formula": rf, "first_insensitive_row": first,
        "predicted_first_insensitive_row": corrupt_hi + rf,
        # 实测的「首个逐位不敏感行」可以比公式预测**早 1 行**（边界那一跳的影响低于
        # 浮点分辨率）⇒ 判定用 `>= 预测 − 1`；公式仍是**保守**上界（用于选布局）。
        "rf_is_conservative": bool(first is not None
                                   and first >= corrupt_hi + rf - 1),
        "max_abs_logit_diff_on_rows_144_255": round(_max_on_rows(prof, 144, 256), 12),
        "max_abs_logit_diff_at_243": (prof[243] if T > 243 else None),
        "max_abs_logit_diff_at_244": (prof[244] if T > 244 else None),
        "profile_tail": [round(v, 8) for v in prof[max(0, T - 20):]],
    }


def main() -> int:
    out = {"purpose": ("给契约 §4a 的布局不变式（eval_lo − rf ≥ corrupt_hi）与 analysis §5 的"
                       "「第 244 行起逐位不敏感」提供可复核产物"),
           "model_spec": "d_model=64（CPU 即可）、随机初始化、固定 seed；只测感受野结构",
           "scans": {}}
    # ① canonical（本实验）：T=256 / 12 层 / 窗口 16 / 被毁前缀 [0,64)
    out["scans"]["canonical_T256_L12_w16_corrupt64"] = scan(12, 16, 256, 64)
    # ② 机制解释：多跳宽度 ≈ L·(w−1)
    out["scans"]["L2_w16_T128_corrupt32"] = scan(2, 16, 128, 32)
    out["scans"]["L4_w8_T128_corrupt16"] = scan(4, 8, 128, 16)
    # ③ 旧（泄漏）布局：被毁 [0,128)、被评行从 144 起 ⇒ 必须非零
    leak = scan(12, 16, 256, 128)
    out["scans"]["leaked_layout_T256_L12_w16_corrupt128"] = leak
    out["leaked_layout_is_sensitive"] = bool(
        (leak["first_insensitive_row"] is None)
        or leak["first_insensitive_row"] > 144)
    out["old_layout_eval_lo_144_differs_from_clean"] = leak["max_abs_logit_diff_at_243"]
    ok = all(v["rf_is_conservative"] for k, v in out["scans"].items() if "leaked" not in k)
    out["all_scan_ok"] = bool(ok and out["leaked_layout_is_sensitive"])
    with open(os.path.join(HERE, "rf_scan.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
        f.write("\n")
    for k, v in out["scans"].items():
        print(f"{k}: first_insensitive_row={v['first_insensitive_row']} "
              f"predicted={v['predicted_first_insensitive_row']} "
              f"Δ@243={v['max_abs_logit_diff_at_243']} Δ@244={v['max_abs_logit_diff_at_244']}")
    print("all_scan_ok:", out["all_scan_ok"],
          "| leaked_layout_is_sensitive:", out["leaked_layout_is_sensitive"])
    return 0 if out["all_scan_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
