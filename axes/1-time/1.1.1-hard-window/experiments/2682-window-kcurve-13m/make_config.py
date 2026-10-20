#!/usr/bin/env python3
"""2682-window-kcurve-13m · 训练后生成 `config.json`（臂路径 + 检查点 sha256）。

`config.json` 是**产物**，不是判据面：判据线与口径的唯一家是 `README.md` + `run.py`。
本脚本做三件事：
  1. 按契约顺序把 17 个必到臂（含 9 个复用臂）+ 2 个可选臂写进 `config.json`，
     每条算 sha256（复用臂的 sha256 也当场算，不抄别人的记录）；
  2. 训练臂缺 `model.pt` 时：必到臂 ⇒ 非零退出（缺必到臂不能出判）；可选臂 ⇒ 记入
     `missing_new_arms`（README §4b 预登记的回落）；
  3. 打印实耗训练墙钟与返回码（从驱动日志 `train_driver.log` 的 `TRAIN START/END <臂>` 行取，
     取不到就如实写 `null`，不臆造）；`*.log` 是 git-ignored 派生物，故这两项必须落进 `config.json`。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

HERE_REL = "axes/1-time/1.1.1-hard-window/experiments/2682-window-kcurve-13m"
E1141 = "axes/1-time/1.1.4.1-explicit-memory/experiments"
E111 = "axes/1-time/1.1.1-hard-window/experiments"

#: (name, kind, window, anchors, seed, path, reused_from)
#: 顺序 = `run.py::CONTRACT_ARMS` 的顺序（run.py 会逐项核对）。
ARMS: list[tuple] = [
    ("full", "full", 256, 0, 0,
     "axes/6-predict/6.2-speculative-decoding/experiments/"
     "2610-frozen-trunk-jumphead-13m/trunk.pt", "2610"),
    ("win16_s0", "window", 16, 0, 0, f"{E1141}/2634-carry-realtext-13m/train/rt_none_s0/model.pt",
     "2634（跨项复本）"),
    ("win16_s1", "window", 16, 0, 1, f"{E1141}/2634-carry-realtext-13m/train/rt_none_s1/model.pt",
     "2634（跨项复本）"),
    ("win16_s2", "window", 16, 0, 2, f"{E111}/2681-anchor-recall-seeds23-13m/train/win16_s2/model.pt",
     "2681（本项）"),
    ("win16_s3", "window", 16, 0, 3, f"{E111}/2681-anchor-recall-seeds23-13m/train/win16_s3/model.pt",
     "2681（本项）"),
    ("win16a_s0", "window", 16, 8, 0,
     f"{E111}/2662-hard-window-anchor-13m/train/win16a_s0/model.pt", "2662（本项）"),
    ("win16a_s1", "window", 16, 8, 1,
     f"{E111}/2662-hard-window-anchor-13m/train/win16a_s1/model.pt", "2662（本项）"),
    ("win16a_s2", "window", 16, 8, 2,
     f"{E111}/2681-anchor-recall-seeds23-13m/train/win16a_s2/model.pt", "2681（本项）"),
    ("win16a_s3", "window", 16, 8, 3,
     f"{E111}/2681-anchor-recall-seeds23-13m/train/win16a_s3/model.pt", "2681（本项）"),
    ("win32_s0", "window", 32, 0, 0, f"{HERE_REL}/train/win32_s0/model.pt", None),
    ("win32_s1", "window", 32, 0, 1, f"{HERE_REL}/train/win32_s1/model.pt", None),
    ("win64_s0", "window", 64, 0, 0, f"{HERE_REL}/train/win64_s0/model.pt", None),
    ("win64_s1", "window", 64, 0, 1, f"{HERE_REL}/train/win64_s1/model.pt", None),
    ("win128_s0", "window", 128, 0, 0, f"{HERE_REL}/train/win128_s0/model.pt", None),
    ("win128_s1", "window", 128, 0, 1, f"{HERE_REL}/train/win128_s1/model.pt", None),
    ("win192_s0", "window", 192, 0, 0, f"{HERE_REL}/train/win192_s0/model.pt", None),
    ("win192_s1", "window", 192, 0, 1, f"{HERE_REL}/train/win192_s1/model.pt", None),
]
OPTIONAL: list[tuple] = [
    ("win64a_s0", "window", 64, 8, 0, f"{HERE_REL}/train/win64a_s0/model.pt", None),
    ("win64a_s1", "window", 64, 8, 1, f"{HERE_REL}/train/win64a_s1/model.pt", None),
]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_ledger(name: str) -> dict:
    """该臂在驱动日志 `train_driver.log` 里的训练流水：墙钟秒数与返回码。

    （只有**本实验训练的**臂才有；复用臂的训练墙钟与 rc 不在本目录，故返回 null 而不臆造。
    `*.log` 是 git-ignored 的派生物 ⇒ 这两项必须落进仓内的 `config.json` 才可回溯。）
    """
    out = {"train_seconds": None, "train_rc": None}
    log = os.path.join(HERE, "train_driver.log")
    if not os.path.exists(log):
        return out
    txt = open(log, encoding="utf-8", errors="replace").read()
    m = re.findall(r"(\d{4}-\d\d-\d\dT[\d:+-]+)\]\s+TRAIN (START|END) " + re.escape(name) + r"\b", txt)
    if len(m) >= 2:
        from datetime import datetime
        try:
            a = datetime.fromisoformat(m[0][0])
            b = datetime.fromisoformat(m[1][0])
            out["train_seconds"] = round((b - a).total_seconds(), 1)
        except ValueError:
            pass
    rc = re.search(r"TRAIN END " + re.escape(name) + r"\s+rc=(-?\d+)", txt)
    if rc:
        out["train_rc"] = int(rc.group(1))
    return out


def main() -> int:
    arms, missing_required, missing_optional = [], [], []
    for group, bucket in ((ARMS, missing_required), (OPTIONAL, missing_optional)):
        for name, kind, window, anchors, seed, path, reused in group:
            full = os.path.join(ROOT, path)
            if not os.path.exists(full):
                bucket.append(name)
                continue
            ledger = train_ledger(name) if reused is None else {
                "train_seconds": None, "train_rc": None}
            arms.append({
                "name": name, "kind": kind, "window": window,
                "anchors": anchors, "seed": seed, "path": path,
                "sha256": sha256_file(full), "reused_from": reused,
                "train_seconds": ledger["train_seconds"],
                "train_rc": ledger["train_rc"],
            })
    if missing_required:
        print(f"缺必到臂的 model.pt：{missing_required} ⇒ 不能生成 config.json（缺必到臂不出判）",
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
                         "note": "与 2662 / 2681 同一函数、同一批数、同一窗口序；"
                                 "bpc token 数 = 96×16×255 = 391,680"}},
        "curves": {"recent_ds": [16, 32, 64, 128, 192, 240],
                   "distance_ds": [1, 2, 4, 8, 16, 32, 64, 128, 255],
                   "arms": ["full", "win16_s0", "win64_s0", "win192_s0"],
                   "note": "recent = 累计预算列（直接测）；distance = 单距离列（只描述形状）"},
        "seeds": [0, 1, 2, 3],
        "arms": arms,
        "missing_new_arms": missing_optional,
        "missing_new_arms_note": ("可选臂只服务判据⑥（锚点杠杆在 k=64）；缺它们时判据⑥ 记 not_produced，"
                                  "其余判据不受影响（README §4b）"),
    }
    out = os.path.join(HERE, "config.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
        f.write("\n")
    n_new = sum(1 for a in arms if a["reused_from"] is None)
    print(f"写出 {out}：{len(arms)} 臂（{n_new} 新训 / {len(arms) - n_new} 复用）"
          f"，可选臂缺失 {missing_optional}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
