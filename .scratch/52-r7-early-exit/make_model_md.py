"""生成 MODEL.md 的权重清单表（sha256 / 体积 / 生成命令）。权重本身不入库。"""

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EV = os.path.join(ROOT, ".scratch", "52-r7-early-exit", "evidence")
EXP_REL = "axes/R-references/R7-existing-early-exit/experiments/2623-standard-early-exit-13m"


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    rows = []
    for name in sorted(os.listdir(EV)):
        mp = os.path.join(EV, name, "model.pt")
        if not os.path.exists(mp):
            continue
        rj = os.path.join(EV, name, "results.json")
        r = json.load(open(rj)) if os.path.exists(rj) else {}
        c = r.get("config", {})
        t = r.get("train", {})
        rows.append({
            "run": name, "sha256": sha256(mp),
            "mb": round(os.path.getsize(mp) / 2 ** 20, 2),
            "params": r.get("params"),
            "cfg": f"d={c.get('d_model')} L={c.get('n_layers')} h={c.get('n_heads')} "
                   f"seq={c.get('seq_len')} mlp={c.get('mlp_mult')} "
                   f"dense={c.get('dense')} gate={c.get('carry_gate')} "
                   f"aux={c.get('aux_mode')} mod={c.get('mod_ratio')}",
            "train": f"{t.get('steps')} steps × bs{t.get('batch_size')} × seq"
                     f"{c.get('seq_len')}, seed {t.get('seed')}",
            "wall": r.get("wall_minutes"), "peak_mb": r.get("peak_gpu_mem_mb"),
        })
    print("| run | sha256 | 体积 MB | 参数 | 形态 | 训练 | 墙钟 min | 峰值 MB |")
    print("|---|---|---|---|---|---|---|---|")
    for x in rows:
        print(f"| `{x['run']}` | `{x['sha256'][:16]}…` | {x['mb']} | {x['params']} "
              f"| {x['cfg']} | {x['train']} | {x['wall']} | {x['peak_mb']} |")
    print()
    for x in rows:
        print(f"{x['run']}: {x['sha256']}")


if __name__ == "__main__":
    main()
