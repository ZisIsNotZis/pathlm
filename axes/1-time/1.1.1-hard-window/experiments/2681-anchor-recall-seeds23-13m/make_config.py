#!/usr/bin/env python3
"""2681-anchor-recall-seeds23-13m · 生成 `config.json`（逐臂 path + sha256）。

契约：同目录 `README.md` §3/§4（判据线与口径常数**写在 README 与 run.py**，本文件不含判据）。
本文件只做一件事：把 9 臂检查点的来源路径与 sha256 落盘，供 `run.py` 读取并进 `results.json`。

来源分工：

- **新臂**（本实验训练）`win16a_s2 / win16a_s3`（窗口 16 + 锚点 8）与 `win16_s2 / win16_s3`
  （窗口 16 + 锚点 0），落 `<本目录>/train/<arm>/model.pt`；
- **复用臂**（只评测）`win16a_s0 / win16a_s1`（`2662` 的 `train/win16a_s*`）、
  `win16_s0 / win16_s1`（`2634` 的 `train/rt_none_s*`）、`full`（`2610` 的 `trunk.pt`），
  sha256 必须逐位等于落盘记录值（本文件的 `REUSE_SHA`），否则非零退出。

`config.json` 在训练之后生成（新臂的 sha256 只有训完才有）⇒ 它**不是**预登记契约文件：
预登记绑定覆盖 `README.md` + `run.py`（`prereg_binding.json::contract_files`），而 `run.py`
开跑前逐字段核对 `config.json` 的臂清单与口径常数（`_check_config_against_contract`），
两者不一致即拒绝启动。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))

EXP2634 = ("axes/1-time/1.1.4.1-explicit-memory/experiments/"
           "2634-carry-realtext-13m")
EXP2662 = ("axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m")
TRUNK2610 = ("axes/6-predict/6.2-speculative-decoding/experiments/"
             "2610-frozen-trunk-jumphead-13m/trunk.pt")
NEW = os.path.join(HERE, "train")

# 复用臂的落盘 sha256（2662 / 2634 / 2610 记录值；本文件写下后运行前不依赖训练）。
REUSE_SHA = {
    "win16a_s0": "5a41ed069ee0ba784187b8d2adc054d334ed418c57ba5ff05fbf6347141cdb2d",
    "win16a_s1": "eb6c9d4a8693c45848d55c672f466aa42f92e4c8586011862ca2f2cdf0a55ffb",
    "win16_s0": "c3f6f08a4fd1dfe9b5c72cb433fe20a2390052b1eda11b8ab9a6d0fe468deca7",
    "win16_s1": "2bfe167c133c496f42864a191f2fc8833f20ba4ec8f9f4f1dceba43dd84fb162",
    "full": "86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2",
}


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: str) -> str:
    return os.path.relpath(path, ROOT)


def arm(name, kind, window, anchors, rule, seed, path, *, reused_from=None,
        train_log=None, train_results_json=None):
    p_abs = path if os.path.isabs(path) else os.path.join(ROOT, path)
    if not os.path.exists(p_abs):
        raise SystemExit(f"{name}: checkpoint 不存在：{path}（先训练）")
    got = sha256_file(p_abs)
    want = REUSE_SHA.get(name)
    if want is not None and got != want:
        raise SystemExit(f"{name}: 复用 checkpoint sha256 与记录值不一致\n "
                         f"want {want}\n got  {got}")
    d = {"name": name, "kind": kind, "window": window, "anchors": anchors,
         "rule": rule, "seed": seed, "path": rel(p_abs), "sha256": got,
         "size_bytes": os.path.getsize(p_abs)}
    if reused_from:
        d["reused_from"] = reused_from
    if train_log:
        d["train_log"] = train_log
    if train_results_json:
        d["train_results_json"] = train_results_json
    return d


def build() -> dict:
    new_arms, missing_new = [], []
    for s in (2, 3):
        for run, anchors in ((f"win16a_s{s}", 8), (f"win16_s{s}", 0)):
            d = os.path.join(NEW, run)
            ck = os.path.join(d, "model.pt")
            if not os.path.exists(ck):
                # 逐臂独立判定：缺哪臂只记哪臂，别的臂照写（不整体降级）。
                missing_new.append(run)
                continue
            new_arms.append(arm(run, "window", 16, anchors, "none", s, ck,
                                train_log=rel(os.path.join(d, "train_log.jsonl")),
                                train_results_json=rel(os.path.join(d, "results.json"))))

    reused = []
    for s in (0, 1):
        d = os.path.join(ROOT, EXP2662, f"train/win16a_s{s}")
        reused.append(arm(f"win16a_s{s}", "window", 16, 8, "none", s,
                          os.path.join(d, "model.pt"),
                          reused_from=f"2662-hard-window-anchor-13m/win16a_s{s}",
                          train_log=f"{EXP2662}/train/win16a_s{s}/train_log.jsonl",
                          train_results_json=f"{EXP2662}/train/win16a_s{s}/results.json"))
    for s in (0, 1):
        d = os.path.join(ROOT, EXP2634, f"train/rt_none_s{s}")
        reused.append(arm(f"win16_s{s}", "window", 16, 0, "none", s,
                          os.path.join(d, "model.pt"),
                          reused_from=f"2634-carry-realtext-13m/rt_none_s{s}",
                          train_log=f"{EXP2634}/train/rt_none_s{s}/train_log.jsonl",
                          train_results_json=f"{EXP2634}/train/rt_none_s{s}/results.json"))
    reused.append(arm("full", "full", 256, 0, "none", 0, os.path.join(ROOT, TRUNK2610),
                      reused_from="2610-frozen-trunk-jumphead-13m/trunk.pt"))

    cfg = {
        "device": "cuda",
        "model": {
            "d_model": 256, "n_layers": 12, "n_heads": 8, "seq_len": 256,
            "mlp_mult": 6, "dense": True, "topology": "sequential",
            "train_window": 16,
            "note": ("train_window/train_anchors 是探针派生 rf 的模型规格（本实验全体干预臂"
                     "都开窗 16）；逐臂的真实可见性由 arms[*].window/anchors + 掩码构造决定"),
        },
        "data": {
            "name": "enwik8_full", "path": "data/enwik8_full.npz",
            "split": "前 90M 字符训练 / 末 10M 字符评测（本实验只用尾段做评测）",
            "vocab_size": 206, "n_real_tokens": 205, "mask_token": 205,
            "note": "真实文本口径：训练与评测同一份 enwik8 字节流（与 train_depth_ar.py 同一路径）",
        },
        "eval": {
            "bpc": {"n_batches": 96, "batch_size": 16, "window_seed": 2,
                    "note": ("与 2662 / 2634 同一函数、同一批数、同一窗口序 ⇒ 复用臂 bpc 必须"
                             "逐位复现（判据①）")},
            "probe": {"n_batches": 96, "batch_size": 16, "window_seed": 2,
                      "corrupt_lo": 0, "corrupt_hi": 64,
                      "eval_lo": 244, "eval_hi": 255, "corrupt_seed": 2634, "rf": 180,
                      "note": ("毁源探针默认布局（=2634）：被毁 [0,64)，被评行 [244,255)，"
                               "rf=12·15=180（244−180=64 ≥ 64）。变体 B（beyond_anchor）"
                               "在 run.py 里用 corrupt_lo=anchors=8 运行，保住锚点槽完整")},
        },
        "needle": {
            "n_needles": 8, "phrase_len": 6, "anchors": 8, "anchor_frac": 0.25,
            "n_batches": 64, "batch_size": 16, "seed": 3, "bucket_anchors": 8,
            "note": ("needle v3 真实文本短语针（retention.phrase_needle_batch），逐字段与 2662 "
                     "相同 ⇒ 复用臂的 anchor/beyond 桶读数必须逐位复现（判据①）；placement "
                     "anchors=8，归属桶**一律 anchors=8**（九臂同索引）"),
        },
        "seeds": [0, 1, 2, 3],
        "arms": reused[:2] + [a for a in new_arms if a["anchors"] == 8]
                + reused[2:4] + [a for a in new_arms if a["anchors"] == 0] + reused[4:],
        "missing_new_arms": missing_new,
        "train_recipe": {
            "cmd": ("python3 train_depth_ar.py <arm> --d 256 --layers 12 --heads 8 "
                    "--mlp-mult 6 --seq-len 256 --dense --topology sequential --steps 12000 "
                    "--batch-size 32 --lr 1e-3 --seed {2,3} --exit-mode uniform "
                    "--train-window 16 --train-anchors {8,0} --eval-batches 4 --out-root "
                    "axes/1-time/1.1.1-hard-window/experiments/"
                    "2681-anchor-recall-seeds23-13m/train"),
            "note": ("与 2662 的 win16a_s{0,1} / 2634 的 rt_none_s{0,1} **逐字段一致**，"
                     "唯一差别 = seed ∈ {2,3}"),
            "steps": 12000, "batch_size": 32, "seq_len": 256, "tokens_per_arm": 98304000,
        },
        "steptime_measured": None,
    }
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-new", action="store_true",
                    help=("新臂缺失时把告警同时打到 stderr；"
                          "缺臂一律记入 missing_new_arms，其余臂照写"))
    ap.add_argument("--out", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args()
    cfg = build()
    for a in cfg["arms"]:
        print(f"  {a['name']:10s} kind={a['kind']:6s} w={a['window']:3d} "
              f"anchors={a['anchors']} seed={a['seed']} sha256={a['sha256'][:16]}…")
    missing = cfg["missing_new_arms"]
    if missing:
        warn = (f"[warn] 新臂 checkpoint 未训练：{missing} ⇒ 记入 config.json 的 "
                f"missing_new_arms；其余 {len(cfg['arms'])} 臂照常写入（不整体降级）")
        print(warn)
        if args.require_new:
            print(warn, file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"写入 {rel(args.out)}（{len(cfg['arms'])} 臂，缺 {len(missing)} 个新臂）")


if __name__ == "__main__":
    main()
