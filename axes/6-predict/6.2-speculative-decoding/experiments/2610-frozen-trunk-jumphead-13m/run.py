"""2610 冻结主干 + 只训跳跃头 —— 实验入口（薄封装）。

契约：同目录 README.md（判据线 §4，不许改）。
真实实现 = 仓库根 `train_jumphead.py`；本脚本只固定 config / out-dir / 联合臂步数，
便于按契约 §9 的「每格实验目录自带入口」复现。

用法：
  python3 run.py                 # 主结果（主干 12k + 头 2k×2 seed + 联合臂 1.5k）
  python3 run.py --smoke         # 20 步冒烟
  python3 run.py --capacity      # 容量消融（复用 trunk.pt，见 MODEL.md）
"""

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
CONFIG = os.path.join(HERE, "config.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="20 步冒烟路径")
    ap.add_argument("--capacity", action="store_true",
                    help="容量消融：复用 trunk.pt 训宽头（hidden=4d × 2 block）")
    args, extra = ap.parse_known_args()

    if args.capacity:
        cmd = [sys.executable, os.path.join(ROOT, "train_jumphead.py"),
               "--config", CONFIG, "--out-dir", os.path.join(HERE, "capacity-h4x2"),
               "--trunk-ckpt", os.path.join(HERE, "trunk.pt"), "--trunk-steps", "0",
               "--head-hidden-mult", "4", "--head-blocks", "2", "--head-steps", "3000"]
    else:
        cmd = [sys.executable, os.path.join(ROOT, "train_jumphead.py"),
               "--config", CONFIG, "--out-dir", HERE]
        if args.smoke:
            cmd.append("--smoke")          # 冒烟不跑联合臂（--smoke 已把步数降到 20）
        else:
            cmd += ["--joint-steps", "1500"]
    cmd += extra
    print("+", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
