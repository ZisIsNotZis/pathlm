"""2685 · 线性核质量腿的训练驱动：**只改注意力算子**，配方与同批全因果臂逐字一致。

本目录不改共享文件（`pathlm/**`、`train_depth_ar.py`、`tools/**`、`tests/**` 全部只读调用）。
干预点只有一个：本进程内把引擎的 φ 换成无消去求值、把分母 guard 的 eps 降到 1e-30
（`stable_linear.install()`），其余一切（CLI、`train()`、AdamW + 余弦退火 + 梯度裁剪、bf16 autocast、
日志与 `results.json` 落盘）都是共享训练器原样。

训练结束后，本驱动在**该臂自己的** `results.json` 里追加一项声明（不改共享训练器的任何读数）：
`linear_phi_numeric` —— 本次用的是哪一档 φ、声明的 eps、`attn_linear` 的活性量（`phi_min/den_min`）
与按 README §3 判的健康读数（`grad_skips` / `gnorm_max` / `den_min` vs 声明 eps）。健康门的最终判定与
落盘在 `run.py`（它另做一次前向的 φ 零占比审计）。

用法（GPU 串行，单臂 ≈9–18 min；臂名之后的所有参数原样转给共享训练器）：

    python3 train_linear.py linear_s0 --d 256 --layers 12 --heads 8 --mlp-mult 6 \
        --seq-len 256 --dense --topology sequential --steps 12000 --batch-size 32 \
        --lr 1e-3 --seed 0 --exit-mode uniform --attn-linear --eval-batches 4 \
        --out-root <本目录>/train
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)

import pathlm.depth_ar as da  # noqa: E402  —— 引擎模块（只读；替换只发生在进程内的模块属性上）
import train_depth_ar  # noqa: E402  —— 共享训练器（只读调用）

import stable_linear  # noqa: E402  —— 本实验唯一的干预点（φ 的无消去求值与分母 guard）


def _cli_value(flag: str) -> str | None:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == flag and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return None


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法：python3 train_linear.py <run_name> <共享训练器的参数...>")
    run_name = sys.argv[1]
    out_root = _cli_value("--out-root")
    if out_root is None:
        raise SystemExit("必须显式给 --out-root（本臂的落盘位置要可核对）")
    if "--attn-linear" not in sys.argv:
        raise SystemExit("本驱动只训线性注意力臂：必须带 --attn-linear（否则是死干预）")
    if _cli_value("--train-window") not in (None, "0") or _cli_value("--train-anchors") not in (None, "0"):
        raise SystemExit("线性注意力与窗口/锚点互斥（引擎硬校验；单因素）")

    stable_linear.install()
    if da.linear_attn_phi is not stable_linear.stable_phi:
        raise SystemExit("φ 没有被换成无消去档：引擎模块的属性没被替换（死干预）")
    if da.LINEAR_ATTN_EPS != stable_linear.LINEAR_ATTN_EPS_DECLARED:
        raise SystemExit("分母 guard 的 eps 没被替换（死干预）")

    train_depth_ar.main()

    res_path = Path(out_root) / run_name / "results.json"
    res = json.loads(res_path.read_text(encoding="utf-8"))
    g = res.get("gnorm_monitor") or {}
    alive = (g.get("attn_linear_mean") or {})
    res["linear_phi_numeric"] = {
        "phi": "stable evaluation of elu(x)+1: exp(min(x,0)) + relu(x)",
        "phi_engine_default": "F.elu(x) + 1",
        "linear_attn_eps_declared": stable_linear.LINEAR_ATTN_EPS_DECLARED,
        "linear_attn_eps_engine_default": 1e-6,
        "kernel_changed": False,
        "note": ("只改 φ 的求值方式与分母 guard 的生效阈值；φ 的数学形式、归一化式、参数量、配方不变"
                 "（`stable_linear.py` 的模块文档给了为什么这不是换核）。"),
        "liveness_at_train": {"phi_min": alive.get("phi_min"), "phi_max": alive.get("phi_max"),
                              "den_min": alive.get("den_min"),
                              "inter_mass_frac": alive.get("inter_mass_frac"),
                              "state_norm": alive.get("state_norm")},
        "grad_skips": g.get("grad_skips"), "gnorm_max": g.get("gnorm_max"),
        "den_never_below_declared_eps": (
            None if alive.get("den_min") is None
            else bool(alive["den_min"] > stable_linear.LINEAR_ATTN_EPS_DECLARED)),
        "source": (HERE / "stable_linear.py").relative_to(ROOT).as_posix(),
    }
    res_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[linear-stable] {res_path}: phi_min {alive.get('phi_min')} / den_min "
          f"{alive.get('den_min')} / gnorm_max {g.get('gnorm_max')} / skips {g.get('grad_skips')}",
          flush=True)


if __name__ == "__main__":
    main()
