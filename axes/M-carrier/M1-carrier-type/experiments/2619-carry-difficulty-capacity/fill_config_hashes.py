"""训练完成后回填 config.json 的 checkpoint sha256（权重不入库，哈希必须逐次核）。"""
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


cfg_path = os.path.join(HERE, "config.json")
cfg = json.load(open(cfg_path))
for name, spec in cfg["arms"].items():
    p = os.path.join(ROOT, spec["path"])
    if os.path.exists(p):
        spec["sha256"] = sha256_file(p)
        spec["size_bytes"] = os.path.getsize(p)
        print(f"{name}: {spec['sha256'][:16]}… ({spec['size_bytes']} B)")
    else:
        print(f"{name}: MISSING {spec['path']}")
with open(cfg_path, "w") as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print("config.json updated")
