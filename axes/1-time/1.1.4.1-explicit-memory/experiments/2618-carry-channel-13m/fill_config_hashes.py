"""把各臂 `model.pt` 的 sha256 回填进 `config.json`（权不入库 ⇒ 哈希必须逐次核）。

用法：python3 fill_config_hashes.py [--config config.json]
"""

import argparse
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    for name, spec in cfg["arms"].items():
        path = spec["path"]
        path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        if not os.path.exists(path):
            print(f"{name}: 缺 {spec['path']}（跳过）")
            continue
        spec["sha256"] = sha256_file(path)
        spec["size_bytes"] = os.path.getsize(path)
        print(f"{name}: {spec['sha256']}  ({spec['size_bytes']} B)")
    with open(args.config, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
