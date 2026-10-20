"""2683 · 块条带臂的训练驱动：**只换掩码算子**，配方逐字沿用条带臂。

本目录不改共享文件（`pathlm/**`、`train_depth_ar.py`、`tools/**`、`tests/**` 全部只读调用）。
干预点只有一个：把 `train_depth_ar` 里的模型类换成 `block_sparse.BlockStripeModel`，它只覆写
`DepthARModel._window_mask`（训练前向构造可见掩码的那唯一一处），掩码来自同一个 `block_sparse.py`。
其余一切（CLI、`train()`、AdamW + 余弦退火 + 梯度裁剪 + 随机退出采样、bf16 autocast、
日志与 `results.json` 落盘）都是共享训练器原样 ⇒ 与 `2677` 的条带臂逐字段同一配方。

用法（GPU 串行，单臂 ≈8–9 min；臂名之后的所有参数原样转给共享训练器）：

    python3 train_block.py block8l4_s0 --d 256 --layers 12 --heads 8 --mlp-mult 6 \
        --seq-len 256 --dense --topology sequential --steps 12000 --batch-size 32 \
        --lr 1e-3 --seed 0 --exit-mode uniform --train-window 24 --train-anchors 0 \
        --eval-batches 4 --out-root <本目录>/train

训练结束后，本驱动在**该臂自己的** `results.json` 里追加两项声明（不改共享训练器的任何读数）：

- `block_sparse_mask`：这张掩码的自证（逐行键数、因果子集、距离结构、sha256 指纹、`alive`）；
- `block_sparse_mask.mask_calls_during_training`：训练期真的调用过掩码的层次数
  （= 0 就是「开关开着、掩码没跑」的死干预，本驱动会直接非零退出）。

引擎自报的 `window_mask` 字段描述的是 `(train_window, train_anchors) = (24, 0)` 的**条带**
计数（共享训练器不知道掩码算子被换过）。它与本掩码的 `n_visible_pairs` 恰好相等是设计使然
（逐行等键），但**它不是本臂掩码的证据**：证据是 `block_sparse_mask`。
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

import train_depth_ar  # noqa: E402  —— 共享训练器（只读调用）

import block_sparse  # noqa: E402  —— 掩码与模型子类（本实验唯一干预点）


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
        raise SystemExit("用法：python3 train_block.py <run_name> <共享训练器的参数...>")
    run_name = sys.argv[1]
    out_root = _cli_value("--out-root")
    train_window = _cli_value("--train-window")
    if out_root is None:
        raise SystemExit("必须显式给 --out-root（本臂的落盘位置要可核对）")
    if train_window != str(block_sparse.K_EFF):
        raise SystemExit(f"本驱动只训「每行 {block_sparse.K_EFF} 个可见键」的块条带臂："
                         f"--train-window 必须 = {block_sparse.K_EFF}（收到 {train_window}）——"
                         "否则共享前向不会走掩码分支（死干预）")

    train_depth_ar.DepthARModel = block_sparse.BlockStripeModel   # 唯一干预点
    train_depth_ar.main()

    run_dir = Path(out_root) / run_name
    if block_sparse.MASK_CALLS["layers"] <= 0:
        raise SystemExit(f"块条带掩码在训练期一次都没被调用（{run_dir}）⇒ 干预是死的，该臂作废")
    res_path = run_dir / "results.json"
    res = json.loads(res_path.read_text(encoding="utf-8"))
    if int(res.get("train_window", -1)) != block_sparse.K_EFF:
        raise SystemExit(f"{res_path}: train_window != {block_sparse.K_EFF}，掩码分支可能没开")
    check = block_sparse.self_check(int(res["config"]["seq_len"]))
    res["block_sparse_mask"] = {
        **check,
        "mask_calls_during_training": block_sparse.MASK_CALLS["layers"],
        "source": (HERE / "block_sparse.py").relative_to(ROOT).as_posix(),
        "note": ("本臂的注意力掩码 = 块条带（最近 16 + 滞后 4 块的一块对齐 8 键），由 "
                 "`BlockStripeModel._window_mask` 在训练前向里构造。引擎自报的 `window_mask` / "
                 "`window_mask_probe` 描述的是 `(train_window, train_anchors) = (24, 0)` 的条带"
                 "计数与「掩码开/关」探针，不是本掩码的证据；本掩码的证据是这里的 "
                 "`n_visible_pairs` / `per_row_counts_equal_band24` / `fingerprint_sha256` / "
                 "`mask_calls_during_training`。评测侧用同一函数构造显式掩码（`run.py` 的 "
                 "训练-评测同源门比指纹）。"),
    }
    res_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[block-sparse] {run_dir}: n_visible_pairs "
          f"{check['n_visible_pairs']}、逐行等键 {check['per_row_counts_equal_band24']}、"
          f"指纹 {check['fingerprint_sha256'][:12]}、训练期掩码调用 "
          f"{block_sparse.MASK_CALLS['layers']}", flush=True)


if __name__ == "__main__":
    main()
