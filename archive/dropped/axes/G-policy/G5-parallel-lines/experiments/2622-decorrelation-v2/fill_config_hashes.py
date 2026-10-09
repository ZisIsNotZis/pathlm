"""把本轮权重的 sha256/体积回填进 `config.json`（权重不入库 ⇒ 哈希必须逐次核）。

- 培训臂：`train{13m,50m}/<arm>/model.pt` 写入 `config.json::arms["<scale>/<arm>"]`；
- 复用臂（50m/stock）：沿用其来源 `2620` 的 sha256（`run.py` 会逐次重算并断言）。

用法（训练完成后）：python3 fill_config_hashes.py
"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
SCALES = {"13m": ("train13m", ("stock", "l1", "repr", "repr01")),
          "50m": ("train50m", ("repr01",))}
REUSE = {"50m/stock": "axes/G-policy/G5-parallel-lines/experiments/2620-decorrelation-at-scale/train/stock/model.pt",
         "50m/l1": "axes/G-policy/G5-parallel-lines/experiments/2620-decorrelation-at-scale/train/l1/model.pt"}


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
    for scale, (train_dir, arms) in SCALES.items():
        for arm in arms:
            p = os.path.join(HERE, train_dir, arm, "model.pt")
            if not os.path.exists(p):
                print(f"SKIP {scale}/{arm}: 尚未训练（{p}）")
                continue
            cfg["arms"][f"{scale}/{arm}"] = {
                "sha256": sha256_file(p), "size_bytes": os.path.getsize(p),
                "path": os.path.relpath(p, ROOT)}
            print(f"{scale}/{arm}: {cfg['arms'][f'{scale}/{arm}']['sha256'][:16]}… "
                  f"{cfg['arms'][f'{scale}/{arm}']['size_bytes']} B")
    for key, rel in REUSE.items():
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            cfg["arms"][key] = {"sha256": sha256_file(p), "size_bytes": os.path.getsize(p),
                                "path": rel, "reused_from": "2620-decorrelation-at-scale"}
            print(f"{key}: {cfg['arms'][key]['sha256'][:16]}… (reused)")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
