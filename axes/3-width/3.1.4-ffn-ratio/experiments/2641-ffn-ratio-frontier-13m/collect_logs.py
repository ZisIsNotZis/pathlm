#!/usr/bin/env python3
"""2641 · 训练后证据归集（把被 .gitignore 挡掉的日志与哈希清单落到可入库的文件）。

跑在**全部臂训练结束之后**（不参与训练路径；不改 run.py，保证臂间代码哈希一致）：

  * `runs/<arm>-s<seed>-12k/{train_log,gate_log}.jsonl` → `<EXP>/<arm>-s<seed>-{train_log,gate_log}.jsonl`
  * 驱动日志与逐臂 stdout（`*.log` 被 .gitignore 挡）→ `<EXP>/run_log.txt`（含 §20 锁取放与显存读数）
  * `<EXP>/config.json`：底座配置 + 臂定义 + §20 锁协议声明
  * `<EXP>/MODEL.md`：逐臂权重/代码/数据哈希 + 参数量 + 逐层隐藏维（权重本身不入库）
  * 校验：各臂记录的 `code_hash_run_py` 必须**逐位相同**（证明训练期间 run.py 未被改过）

用法：python3 collect_logs.py
"""
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import (D_MODEL, MLP_MULT, N_HEADS, N_LAYERS, SCHEDULES, SEQ_LEN,  # noqa: E402
                 analytic_cost)

ARMS = [("U4", 0), ("U2", 0), ("PYR", 0), ("APYR", 0), ("U2", 1), ("PYR", 1),
        ("APYR", 1), ("U4", 1)]
STEPS = 12000


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    run_root = os.path.join(HERE, "runs")
    present, hashes = [], {}
    for arm, seed in ARMS:
        rd = os.path.join(run_root, f"{arm}-s{seed}-{STEPS // 1000}k")
        if not os.path.exists(rd):
            print(f"[skip] {arm}-s{seed}（无 run 目录）")
            continue
        present.append((arm, seed))
        for name in ("train_log.jsonl", "gate_log.jsonl"):
            src = os.path.join(rd, name)
            if os.path.exists(src):
                dst = os.path.join(HERE, f"{arm}-s{seed}-{name}")
                with open(src, "rb") as fi, open(dst, "wb") as fo:
                    fo.write(fi.read())
        mp = os.path.join(rd, "model.pt")
        if os.path.exists(mp):
            hashes[f"{arm}-s{seed}/model.pt"] = sha256_file(mp)
            hashes[f"{arm}-s{seed}/model.pt_bytes"] = os.path.getsize(mp)

    # 证据哈希（代码 / 配置 / 数据）
    for f in ("run.py", "test_2641_ffn_ratio.py", "README.md", "train_arms.sh"):
        hashes[f] = sha256_file(os.path.join(HERE, f))
    hashes["data/enwik8_full.npz"] = sha256_file(
        os.path.join(HERE, "..", "..", "..", "..", "..", "data",
                     "enwik8_full.npz"))

    # 逐臂结果里的 run.py 哈希必须一致（训练期间代码冻结）
    code_hashes = {}
    for arm, seed in present:
        p = os.path.join(HERE, f"results_{arm}_s{seed}.json")
        if os.path.exists(p):
            code_hashes[f"{arm}-s{seed}"] = json.load(open(p))["code_hash_run_py"]
    frozen = len(set(code_hashes.values())) == 1

    # 日志归集（.log 被 gitignore ⇒ 落到 run_log.txt）
    chunks = []
    for f in ("train_arms_driver.log",):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            chunks.append(f"===== {f} =====\n" + open(p, errors="replace").read())
    for arm, seed in present:
        p = os.path.join(HERE, f"train_{arm}_s{seed}.log")
        if os.path.exists(p):
            body = open(p, errors="replace").read()
            chunks.append(f"===== train_{arm}_s{seed}.log（stdout，末 4000 字符）=====\n"
                          + body[-4000:])
    with open(os.path.join(HERE, "run_log.txt"), "w") as f:
        f.write("\n\n".join(chunks))

    cfg = {
        "experiment": "2641-ffn-ratio-frontier-13m",
        "base": {"d_model": D_MODEL, "n_layers": N_LAYERS, "n_heads": N_HEADS,
                 "seq_len": SEQ_LEN, "mlp_mult": MLP_MULT, "dense": True,
                 "dropout": 0.0, "norm_cap": 1.0, "topology": "sequential",
                 "exit_mode": "uniform", "fill_kv": True},
        "training": {"steps": STEPS, "batch_size": 32, "lr": 1e-3,
                     "train_tokens": STEPS * 32 * SEQ_LEN,
                     "data": "data/enwik8_full.npz（前 90M train / 末 10M eval，词表 206）"},
        "intervention": {
            "what": "逐层 FFN 隐藏维 r_l（构造后对 mlp[0]/mlp[2] 做**前缀切片**）",
            "arms": SCHEDULES,
            "matched_cost_rule": "U2/PYR/APYR 对齐 Σ_l r_l ⇒ FFN 参数与每 token FFN MACs 逐位相等",
            "analytic_cost": {a: analytic_cost(SCHEDULES[a]) for a in SCHEDULES},
        },
        "eval": {"metric": "clean bpc（depth_curve 最终深度，n_batches=20, bs=16, "
                           "eval generator seed=2）"},
        "gpu_lock": "SCHEMA.md §20：按臂取/放 /tmp/pathlm_gpu.lock，臂间 ≥61 s；"
                    "锁取放时间戳与 free 读数见 run_log.txt",
        "arms_run": [f"{a}-s{s}" for a, s in present],
        "code_frozen_during_training": frozen,
        "code_hashes_run_py": code_hashes,
    }
    with open(os.path.join(HERE, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    lines = ["# 2641 · 权重与证据清单（权重本身不入库）", "",
             f"本清单由 `collect_logs.py` 在训练结束后生成；`code_frozen_during_training = "
             f"{frozen}`（各臂记录的 `run.py` 哈希逐位相同）。", "",
             "## 臂与逐层 FFN 隐藏维", "",
             "| 臂 | Σr | r_l | 每 token FFN MACs | FFN 参数 | hidden units（实测审计） |",
             "|---|---|---|---|---|---|"]
    for arm, seed in present:
        p = os.path.join(HERE, f"results_{arm}_s{seed}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        lines.append(f"| {arm}-s{seed} | {r['sum_r']} | {r['schedule_r_by_layer']} | "
                     f"{r['cost']['macs_per_token']['ffn']} | "
                     f"{r['params']['params_ffn']} | {r['params']['hidden_by_layer']} |")
    lines += ["", "## 哈希（sha256）", "", "| 文件 | sha256 | 字节 |", "|---|---|---|"]
    for k, v in sorted(hashes.items()):
        size = hashes.get(f"{k}_bytes", "")
        if k.endswith("_bytes"):
            continue
        lines.append(f"| `{k}` | `{v}` | {size} |")
    with open(os.path.join(HERE, "MODEL.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # 门的复跑记录（等值性 + 变异）
    proc = subprocess.run([sys.executable, "-m", "pytest",
                           os.path.join(HERE, "test_2641_ffn_ratio.py"), "-q"],
                          capture_output=True, text=True, cwd=HERE)
    with open(os.path.join(HERE, "gates_report.txt"), "w") as f:
        f.write(f"$ python3 -m pytest test_2641_ffn_ratio.py -q\n"
                f"exit={proc.returncode}\n\n{proc.stdout}\n{proc.stderr}\n")
    print(f"归集完成：{len(present)} 臂；code_frozen={frozen}；pytest rc={proc.returncode}")
    if not frozen:
        print("警告：各臂 run.py 哈希不一致（训练期间代码被改过）")
        sys.exit(2)


if __name__ == "__main__":
    main()
