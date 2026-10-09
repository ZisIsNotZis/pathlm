"""把本轮权重的 sha256/体积回填进 `config.json`（权重不入库 ⇒ 哈希必须逐次核）。

用法（训练完成后）：python3 fill_config_hashes.py
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TRAIN_DIR = "train13m"
ARMS = ("stock", "l1", "repr01")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    cfg_path = os.path.join(HERE, "config.json")
    cfg = json.load(open(cfg_path))
    cfg.setdefault("arms", {})
    for arm in ARMS:
        p = os.path.join(HERE, TRAIN_DIR, arm, "model.pt")
        if not os.path.exists(p):
            print(f"SKIP 13m/{arm}: 尚未训练（{p}）")
            continue
        cfg["arms"][f"13m/{arm}"] = {
            "sha256": sha256_file(p), "size_bytes": os.path.getsize(p),
            "path": os.path.relpath(p, ROOT)}
        print(f"13m/{arm}: {cfg['arms'][f'13m/{arm}']['sha256'][:16]}… "
              f"{cfg['arms'][f'13m/{arm}']['size_bytes']} B")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
