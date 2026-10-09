"""把本轮权重的 sha256/体积回填进 `config.json`（权重不入库 ⇒ 哈希必须逐次核）。

`run.py` 每次加载都重算 sha256 并与 config.json 记录断言一致（`TO_FILL` 视为未填）。
用法（训练完成后）：python3 fill_config_hashes.py
"""

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


def main() -> None:
    cfg_path = os.path.join(HERE, "config.json")
    cfg = json.load(open(cfg_path))
    for key, entry in cfg["arms"].items():
        p = os.path.join(ROOT, entry["ckpt_file"])
        if not os.path.exists(p):
            print(f"SKIP {key}: 尚未训练（{entry['ckpt_file']}）")
            continue
        want = entry.get("sha256")
        got = sha256_file(p)
        entry["sha256"] = got
        entry["size_bytes"] = os.path.getsize(p)
        tag = "（与既有记录一致）" if want == got else ("" if not want or want == "TO_FILL"
                                                        else "（!! 与既有记录不同）")
        print(f"{key}: {got[:16]}… {entry['size_bytes']} B {tag}")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
