"""2620 · 训练完成后回填 config.json 的权重 sha256 / size_bytes（`run.py` 加载前逐臂断言）。

用法：python3 fill_config_hashes.py    # 在训练落盘之后、跑 run.py 之前
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
    for arm, spec in cfg["arms"].items():
        path = os.path.join(ROOT, spec["path"])
        if not os.path.exists(path):
            print(f"{arm}: MISSING {spec['path']}（先训练）")
            continue
        spec["sha256"] = sha256_file(path)
        spec["size_bytes"] = os.path.getsize(path)
        changed.append(arm)
        print(f"{arm}: {spec['sha256'][:16]}… {spec['size_bytes']} B")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"updated {len(changed)}/{len(cfg['arms'])} arms in config.json")


if __name__ == "__main__":
    main()
