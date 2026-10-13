#!/usr/bin/env python3
"""训练后回填 `config.json::arms.<arm>.sha256` / `size_bytes`（权重不入库，哈希入 config）。

用法：python3 fill_config_hashes.py
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cfg_path = os.path.join(HERE, "config.json")
    cfg = json.load(open(cfg_path))
    changed = []
    for name, arm in cfg["arms"].items():
        p = os.path.join(ROOT, arm["path"])
        if not os.path.exists(p):
            print(f"[skip] {name}: 缺 {arm['path']}")
            continue
        arm["sha256"] = sha256_file(p)
        arm["size_bytes"] = os.path.getsize(p)
        changed.append(name)
        print(f"[ok] {name}: {arm['sha256'][:16]}… {arm['size_bytes']} B")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"回填 {len(changed)} 臂：{changed}")


if __name__ == "__main__":
    main()
