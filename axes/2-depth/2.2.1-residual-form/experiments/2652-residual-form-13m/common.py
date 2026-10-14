"""2652 · 共享常量与工厂参数（臂定义、resolved config、共享训练入口的 CLI）。

单一事实来源：臂名、规格、`shared_argv`（与 `train_depth_ar.main()` 逐字对齐）
只在这里写一份，`run.py` / `smoke.py` 都从这里取。
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig  # noqa: E402

#: 轴 2 的 13M 标准档（与 2613 / 2641 逐字段相同）
DEFAULTS = {
    "d_model": 256,
    "n_layers": 8,
    "n_heads": 8,
    "seq_len": 256,
    "mlp_mult": 4,
    "vocab_size": 206,
    "steps": 12000,
    "batch_size": 32,
    "lr": 1e-3,
    "seed": 0,
    "eval_batches": 20,
    "eval_batch_size": 16,
    "data": "data/enwik8_full.npz",
}

#: 契约 §9.2 WP-2652 的四臂
ARM_FORMS = ("base", "postln", "scaled", "gated")

#: 执行顺序：`scaled` 排最后（契约的「先砍 scaled」= 直接不跑最后这一臂）
EXEC_ORDER = ("base", "postln", "gated", "scaled")

VARIANCE_BOUND = 0.014

#: 历史同引擎 dense 13M 锚（等值性门用；皆 = `.scratch/23-depth-ar/evidence`）
HIST_23D_DENSE_B_12K = 1.5287     # 23d-dense-B-12k
HIST_2613_BASELINE_S0 = 1.5257    # 2613 sequential-12k-s0（seed 0）
HIST_2613_BASELINE_S1 = 1.5298    # 2613 sequential-12k-s1（seed 1）


def make_cfg(vocab_size: int) -> DepthARConfig:
    """resolved config（dense + sequential + 全深度最终 CE）——四臂逐字段相同。"""
    return DepthARConfig(
        d_model=DEFAULTS["d_model"], n_layers=DEFAULTS["n_layers"],
        n_heads=DEFAULTS["n_heads"], seq_len=DEFAULTS["seq_len"],
        mlp_mult=DEFAULTS["mlp_mult"], dense=True, topology="sequential",
        dropout=0.0, norm_cap=1.0, fill_kv=True)


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def run_name(form: str, seed: int, steps: int) -> str:
    return f"{form}-s{seed}-{steps // 1000}k"


def shared_argv(run: str, seed: int, steps: int, batch_size: int, lr: float,
                out_root: str) -> list[str]:
    """与 `train_depth_ar.main()` 的 CLI 逐字对齐（dense 形态；唯一变量 = 残差形式，
    而残差形式**不进 CLI**，由工厂替换注入）。"""
    return [os.path.join(ROOT, "train_depth_ar.py"), run,
            "--d", str(DEFAULTS["d_model"]), "--layers", str(DEFAULTS["n_layers"]),
            "--heads", str(DEFAULTS["n_heads"]),
            "--seq-len", str(DEFAULTS["seq_len"]),
            "--mlp-mult", str(DEFAULTS["mlp_mult"]),
            "--dense", "--topology", "sequential", "--exit-mode", "uniform",
            "--steps", str(steps), "--batch-size", str(batch_size),
            "--lr", str(lr), "--seed", str(seed),
            "--eval-batches", str(DEFAULTS["eval_batches"]),
            "--out-root", out_root]
