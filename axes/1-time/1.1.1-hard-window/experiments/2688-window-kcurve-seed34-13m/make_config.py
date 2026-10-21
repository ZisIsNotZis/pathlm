#!/usr/bin/env python3
"""2688-window-kcurve-seed34-13m · 训练后生成 `config.json`（臂路径 + 检查点 sha256）。

`config.json` 是**产物**，不是判据面：判据线与口径的唯一家是 `README.md` + `run.py`。
本脚本做三件事：
  1. 按契约顺序写 14 个臂（10 复用 + 4 新训），每条算 sha256（复用臂的 sha256 也当场算，不抄别人的记录）；
  2. 缺臂的处理照 `README.md` §5a：任一臂缺 `model.pt` ⇒ 非零退出（判据只弱不假，但读数不能缺臂出）；
  3. 把新臂的训练墙钟与返回码从驱动日志 `train_driver.log` 的 `TRAIN START/END <臂>` 行取出来
     落进 `config.json`（`*.log` 是 git-ignored 派生物，取不到就写 `null`，不臆造）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

HERE_REL = "axes/1-time/1.1.1-hard-window/experiments/2688-window-kcurve-seed34-13m"
E111 = "axes/1-time/1.1.1-hard-window/experiments"
D2682 = f"{E111}/2682-window-kcurve-13m/train"
D2686 = f"{E111}/2686-window-kcurve-seed2-13m/train"

#: (name, kind, window, anchors, seed, path, reused_from)；顺序 = `run.py::CONTRACT_ARMS`。
ARMS: list[tuple] = [
    ("full", "full", 256, 0, 0,
     "axes/6-predict/6.2-speculative-decoding/experiments/"
     "2610-frozen-trunk-jumphead-13m/trunk.pt", "2610"),
    ("win64_s0", "window", 64, 0, 0, f"{D2682}/win64_s0/model.pt", "2682"),
    ("win64_s1", "window", 64, 0, 1, f"{D2682}/win64_s1/model.pt", "2682"),
    ("win64_s2", "window", 64, 0, 2, f"{D2686}/win64_s2/model.pt", "2686"),
    ("win64_s3", "window", 64, 0, 3, f"{HERE_REL}/train/win64_s3/model.pt", None),
    ("win64a_s0", "window", 64, 8, 0, f"{D2682}/win64a_s0/model.pt", "2682"),
    ("win64a_s1", "window", 64, 8, 1, f"{D2682}/win64a_s1/model.pt", "2682"),
    ("win64a_s2", "window", 64, 8, 2, f"{D2686}/win64a_s2/model.pt", "2686"),
    ("win64a_s3", "window", 64, 8, 3, f"{HERE_REL}/train/win64a_s3/model.pt", None),
    ("win128_s0", "window", 128, 0, 0, f"{D2682}/win128_s0/model.pt", "2682"),
    ("win128_s1", "window", 128, 0, 1, f"{D2682}/win128_s1/model.pt", "2682"),
    ("win128_s2", "window", 128, 0, 2, f"{D2686}/win128_s2/model.pt", "2686"),
    ("win128_s3", "window", 128, 0, 3, f"{HERE_REL}/train/win128_s3/model.pt", None),
    ("win128_s4", "window", 128, 0, 4, f"{HERE_REL}/train/win128_s4/model.pt", None),
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_ledger(name: str) -> dict:
    """该臂在 `train_driver.log` 里的训练墙钟与返回码（只有本实验训练的臂才有）。"""
    out = {"train_seconds": None, "train_rc": None}
    log = os.path.join(HERE, "train_driver.log")
    if not os.path.exists(log):
        return out
    txt = open(log, encoding="utf-8", errors="replace").read()
    m = re.findall(r"(\d{4}-\d\d-\d\dT[\d:+-]+)\]\s+TRAIN (START|END) "
                   + re.escape(name) + r"\b", txt)
    if len(m) >= 2:
        try:
            out["train_seconds"] = round(
                (datetime.fromisoformat(m[1][0]) - datetime.fromisoformat(m[0][0])).total_seconds(), 1)
        except ValueError:
            pass
    rc = re.search(r"TRAIN END " + re.escape(name) + r"\s+rc=(-?\d+)", txt)
    if rc:
        out["train_rc"] = int(rc.group(1))
    return out


def main() -> int:
    arms, missing = [], []
    for name, kind, window, anchors, seed, path, reused in ARMS:
        full = os.path.join(ROOT, path)
        if not os.path.exists(full):
            missing.append(name)
            continue
        ledger = {"train_seconds": None, "train_rc": None} if reused else train_ledger(name)
        arms.append({"name": name, "kind": kind, "window": window, "anchors": anchors,
                     "seed": seed, "path": path, "sha256": sha256_file(full),
                     "reused_from": reused, "train_seconds": ledger["train_seconds"],
                     "train_rc": ledger["train_rc"]})
    if missing:
        print(f"缺臂的 model.pt：{missing} ⇒ 不能生成 config.json（缺臂不出判，见 README §5a）",
              file=sys.stderr)
        return 3

    cfg = {
        "experiment": HERE_REL,
        "contract": f"{HERE_REL}/README.md",
        "model": {"d_model": 256, "n_layers": 12, "n_heads": 8, "seq_len": 256,
                  "mlp_mult": 6, "dense": True, "topology": "sequential"},
        "data": {"name": "enwik8_full", "path": "data/enwik8_full.npz",
                 "split": "前 90M 字符训练 / 末 10M 字符评测（本实验只用尾段做评测）",
                 "vocab_size": 206, "n_real_tokens": 205, "mask_token": 205},
        "eval": {"bpc": {"n_batches": 96, "batch_size": 16, "window_seed": 2,
                         "note": "与 2662 / 2681 / 2682 / 2686 同一函数、同一批数、同一窗口序；"
                                 "bpc token 数 = 96×16×255 = 391,680"}},
        "seeds": [0, 1, 2, 3, 4],
        "arms": arms,
    }
    out = os.path.join(HERE, "config.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n_new = sum(1 for a in arms if a["reused_from"] is None)
    print(f"写出 {out}：{len(arms)} 臂（{n_new} 新训 / {len(arms) - n_new} 复用）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
