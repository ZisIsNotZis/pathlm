#!/usr/bin/env python3
"""2633-carry-seed2 · 把三臂新 checkpoint 的 sha256 / size_bytes 回填进 `config.json`。

用法：python3 fill_config_hashes.py     # 训练完成后跑；缺任一**新臂** checkpoint 即非零退出
（复用臂的 sha256 来自 2631 的 config.json，本次不重写——它们由 `load_arm` 逐次断言。）
"""

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cfg_path = os.path.join(HERE, "config.json")
    cfg = json.load(open(cfg_path))
    missing = []
    for a in cfg["arms"]:
        if a["name"] not in cfg["new_in_2633"]:
            continue
        p = os.path.join(ROOT, a["path"])
        if not os.path.exists(p):
            missing.append(a["path"])
            continue
        a["sha256"] = sha256_file(p)
        a["size_bytes"] = os.path.getsize(p)
        print(f"{a['name']}: {a['sha256'][:16]}… {a['size_bytes']} B")
    if missing:
        print("缺 checkpoint（先训练）：" + ", ".join(missing), file=sys.stderr)
        return 1
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("config.json 已回填三臂新 checkpoint 的 sha256")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
