#!/usr/bin/env python3
"""2662-hard-window-anchor-13m · 生成 `config.json`（逐臂 path + sha256）。

契约：同目录 `README.md` §3/§4/§4a/§4b。基线配方逐字段取 2634
（`2634/config.json` 的 `train_recipe`）；本文件只做两件事：

1. 落 5 臂的来源路径与 sha256（复用臂指向 2634 / 2610 的现成 checkpoint）；
2. 把评测口径常量写死（bpc 96×16 / window_seed=2；探针 96×16 / rf=180 / 被评行
   [244,255)；needle n_needles=8 / phrase_len=6 / anchor_frac=0.25 / seed=3 / bucket_anchors=8）。

`--require-new` 的语义（逐臂独立判定，不再整体降级）：**已存在且 sha256 可核的臂一律写入
`arms`**（新臂 `win16a_s0/s1` 现算 sha256，复用臂逐位校验写死值）；**缺失的新臂只记入
`missing_new_arms` 并告警**（`--require-new` 时同时打到 stderr），**不因此丢弃已训好的臂**。
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
TRUNK2610 = ("axes/6-predict/6.2-speculative-decoding/experiments/"
             "2610-frozen-trunk-jumphead-13m/trunk.pt")
NEW = os.path.join(HERE, "train")

# 复用臂的落盘 sha256（2634/config.json 记录值；本文件写下后运行前不依赖训练）。
REUSE_SHA = {
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
    for s in (0, 1):
        run = f"win16a_s{s}"
        d = os.path.join(NEW, run)
        ck = os.path.join(d, "model.pt")
        if not os.path.exists(ck):
            # 逐臂独立判定：缺哪臂只记哪臂，别的臂照写（不整体降级）。
            missing_new.append(run)
            continue
        new_arms.append(arm(run, "window", 16, 8, "none", s, ck,
                            train_log=rel(os.path.join(d, "train_log.jsonl")),
                            train_results_json=rel(os.path.join(d, "results.json"))))

    reused = [
        arm("win16_s0", "window", 16, 0, "none", 0,
            os.path.join(ROOT, EXP2634, "train/rt_none_s0/model.pt"),
            reused_from="2634-carry-realtext-13m/rt_none_s0",
            train_log=f"{EXP2634}/train/rt_none_s0/train_log.jsonl",
            train_results_json=f"{EXP2634}/train/rt_none_s0/results.json"),
        arm("win16_s1", "window", 16, 0, "none", 1,
            os.path.join(ROOT, EXP2634, "train/rt_none_s1/model.pt"),
            reused_from="2634-carry-realtext-13m/rt_none_s1",
            train_log=f"{EXP2634}/train/rt_none_s1/train_log.jsonl",
            train_results_json=f"{EXP2634}/train/rt_none_s1/results.json"),
        arm("full", "full", 256, 0, "none", 0, os.path.join(ROOT, TRUNK2610),
            reused_from="2610-frozen-trunk-jumphead-13m/trunk.pt"),
    ]

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
                    "note": "与 2634 同一函数/批数/窗口序 ⇒ 复用臂应与 2634 落盘值逐位相同"},
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
            "note": ("needle v3 真实文本短语针（retention.phrase_needle_batch）；placement "
                     "anchors=8，归属桶**一律 anchors=8**（五臂同索引）⇒ 桶 = anchor "
                     "(p<8) / window（可见）/ beyond（有位置不可见）"),
        },
        "seeds": [0, 1],
        "arms": new_arms + reused,
        "missing_new_arms": missing_new,
        "bands": {
            "sigma_cell": None,
            "sigma_cell_note": ("留空 ⇒ 由 run.py 用本实验自己的换种子极差填（本格带为主判）；"
                                "跨格 0.014 带为参考，两带并列报、不混用（C1）"),
            "cross_cell_band": 0.014,
        },
        "train_recipe": {
            "cmd": ("python3 train_depth_ar.py <arm> --d 256 --layers 12 --heads 8 "
                    "--mlp-mult 6 --seq-len 256 --dense --topology sequential --steps 12000 "
                    "--batch-size 32 --lr 1e-3 --seed {0,1} --exit-mode uniform "
                    "--train-window 16 --train-anchors 8 --eval-batches 4 --out-root "
                    "axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/train"),
            "note": "唯一变量 = seed；与 2634 的 rt_none 臂只差 --train-anchors 8 vs 0",
            "steps": 12000, "batch_size": 32, "seq_len": 256, "tokens_per_arm": 98304000,
        },
        "steptime_measured": None,
    }
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-new", action="store_true",
                    help=("新臂（win16a_s0/s1）缺失时把告警同时打到 stderr；"
                          "不再非零退出——缺臂一律记入 missing_new_arms，其余臂照写"))
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
