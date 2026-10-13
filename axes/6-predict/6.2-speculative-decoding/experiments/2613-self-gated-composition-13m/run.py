"""2613 自门控多跳组合（契约 30-G，H3 纯空白格）—— 实验入口（薄封装）。

契约：同目录 README.md（判据线 §4 逐字保留，不许改）。
真实实现 = 仓库根 `train_jumphead.py --composition`；本脚本只固定 config / out-dir，
便于按契约 §9 的「每格实验目录自带入口」复现。

用法：
  python3 run.py            # 主结果：冻结主干上 4 臂 × 2 seed（2000 步）+ 4000 步收敛点
  python3 run.py --smoke    # 20 步冒烟（实测步时与显存）
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
    ap.add_argument("--smoke", action="store_true", help="20 步冒烟（打印步时与显存）")
    args, extra = ap.parse_known_args()

    cmd = [sys.executable, os.path.join(ROOT, "train_jumphead.py"),
           "--config", CONFIG, "--out-dir", HERE, "--composition"]
    if args.smoke:
        cmd.append("--smoke")
    cmd += extra
    print("+", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
