#!/usr/bin/env python3
"""生成 2639 的 `config.json`（评测臂表 + 口径）。

不改共享文件；只读盘（2638 的 config.json 提供**复用臂**的路径与 sha256，
本目录 `train/` 下的三个新臂的 sha256 现算）并落盘本目录 config.json。
用法：python3 make_config.py [--require-new]   （--require-new：三个新臂缺一即非零退出）
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
REF = os.path.join(ROOT, "axes/1-time/1.1.4.1-explicit-memory/experiments/"
                         "2638-carry-frontier-13m/config.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-new", action="store_true")
    args = ap.parse_args()
    ref = json.load(open(REF))
    ref_arms = {a["name"]: a for a in ref["arms"]}

    reuse = ["full", "win16_s0", "win16_s1", "win17_s0", "win18_s0", "win24_s0",
             "win32_s0", "win128_s0", "carry16_s0", "carry16_s1"]
    arms = []
    for name in reuse:
        a = dict(ref_arms[name])
        a["reused_from"] = ("2610 trunk（全因果锚）" if name == "full" else
                            ("2634-carry-realtext-13m"
                             if a["path"].find("2634-carry-realtext-13m") >= 0
                             else "2638-carry-frontier-13m"))
        if name.startswith("carry16"):
            a["carry_chunk"] = 64          # 训练时实现细节（2634 口径）
            a["carry_per_layer"] = False
            a["carry_decay"] = 1.0
        got = sha256_file(os.path.join(ROOT, a["path"]))
        assert got == a["sha256"], f"{name}: 复用臂 sha256 与 2638 config 不符"
        arms.append(a)

    new = [("carrypl16_s0", dict(carry_chunk=256, carry_per_layer=True,
                                 carry_decay=1.0, seed=0,
                                 role="**完备机制**（E1）：逐层独立状态 + 内容写门（取自当前层 "
                                      "hidden），γ=1；与 win32_s0 **逐字节相等**（393,216 B）")),
           ("c16c256_s0", dict(carry_chunk=256, carry_per_layer=False, carry_decay=1.0,
                               seed=0,
                               role="单共享槽 delta，chunk=256：与完备臂 chunk 匹配的**退化机制**"
                                    "对照（也充当 chunk 敏感性探针）")),
           ("carrypl16d_s0", dict(carry_chunk=256, carry_per_layer=True,
                                  carry_decay=0.99, seed=0,
                                  role="完备机制 + 衰减 γ=0.99（GLA 式遗忘，E1 可选开关）"))]
    missing = []
    for name, extra in new:
        rel = f"axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/train/{name.replace('carrypl16d_s0','pl16d').replace('carrypl16_s0','pl16').replace('c16c256_s0','c16')}/model.pt"
        if not os.path.exists(os.path.join(ROOT, rel)):
            missing.append(name)
            continue
        arms.append({
            "name": name, "kind": "carry", "path": rel,
            "sha256": sha256_file(os.path.join(ROOT, rel)),
            "window": 16, "anchors": 0, "rule": "delta", "size_bytes":
                os.path.getsize(os.path.join(ROOT, rel)),
            "role": extra.pop("role"), "reused_from": None, **extra})
    if args.require_new and missing:
        print(f"缺新臂 checkpoint：{missing}", file=sys.stderr)
        raise SystemExit(3)

    cfg = {
        "contract": "axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/README.md",
        "scope": ("只做单轴单格实验，基底恒为 stock decoder-only 全因果；组合默认留白。"
                  "本实验把**完整机制**（逐层独立状态 + 内容写门 ± 衰减 γ，E1 开关）放进"
                  "『每位置等效 KV 字节 vs clean bpc』前沿，与 ① 单共享槽退化版（2634/2638）"
                  "② 同预算纯窗口基线 对照。"),
        "device": "cuda",
        "model": ref["model"],
        "data": ref["data"],
        "eval": ref["eval"],
        "cost_axis": {
            "bytes_per_elem": 2,
            "note": ("每位置等效 KV 字节 = n_layers × 2(K,V) × d_model × bytes_per_elem × k；"
                     "carry 状态 = （单共享槽）n_heads×hd² floats；（逐层）n_layers × n_heads×hd² "
                     "floats —— run.py 从加载的模型上实际读（state_floats_actual）。"
                     "横轴另报等效窗口位置数 k_eff = 总字节 / 12288。"),
            "kv_bytes_per_position_unit": 12288},
        "bands": ref.get("bands", {}),
        "arms": arms,
        "missing_new_arms": missing,
    }
    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"config.json 落盘：{len(arms)} 臂；缺 {missing}")


if __name__ == "__main__":
    main()
