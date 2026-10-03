#!/usr/bin/env python3
"""把 `train/` 下新臂的权重 sha256 回填进 `config.json`（权重不入库，哈希入账）。

用法：python3 fill_hashes.py            # 幂等；只改 config.json 的 checkpoint.sha256
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    cfg_path = os.path.join(HERE, "config.json")
    cfg = json.load(open(cfg_path))
    filled = {}
    for arm, spec in cfg["arms"].items():
        if spec["source"] != "2624":
            continue                                     # 6k 四臂复用 2623 的哈希
        p = os.path.join(HERE, "train", arm, "model.pt")
        if not os.path.exists(p):
            print(f"跳过 {arm}：{os.path.relpath(p, HERE)} 不存在")
            continue
        cfg["checkpoint"]["sha256"][arm] = sha256_file(p)
        filled[arm] = {"sha256": cfg["checkpoint"]["sha256"][arm],
                       "size_bytes": os.path.getsize(p)}
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(json.dumps(filled, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
