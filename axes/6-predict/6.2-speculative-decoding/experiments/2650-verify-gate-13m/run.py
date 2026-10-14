"""薄入口：README §7 的 `python3 run.py ...` 直接转调 `verify_gate.main()`。

实现与判据都在 `verify_gate.py`（本实验的落盘驱动）；本文件只做 CLI 转发。
"""

from __future__ import annotations

import sys

from verify_gate import main

if __name__ == "__main__":
    sys.exit(main())
