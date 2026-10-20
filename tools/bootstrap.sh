#!/usr/bin/env bash
# tools/bootstrap.sh — 干净克隆后一条命令把「机检环境」建好。
#
# 做什么（三道门，任一不过即非零退出，不静默降级）：
#   1. 版本门：Python >= 3.10（`tools/` 与 `tests/` 用了 `X | None` 注解）。
#   2. 依赖门：按 `requirements.txt` 逐个 import 校验；缺哪个就用 `python3 -m pip` 装。
#   3. 报版本：打通后打印实测的解释器与依赖版本，最后一行是 OK。
#
# 不做（干净克隆重建不了的东西，README「快速开始」里如实写）：
#   - `data/enwik8_full.npz`：由 `.tmp/enwik8`（enwik8 原始字节，不在库内、也不会自动下载）
#     首次运行时生成，`.gitignore` 排除；机检四件套不需要它。
#   - `*.pt` 训练权重：派生物，按各实验目录 `README.md` 的命令重训，不入库。
#   - `tests/` 的 torch 依赖：见 `requirements-dev.txt`，与机检四件套无关。
#
# 用法：
#   tools/bootstrap.sh                          # 用 $PYTHON 或 python3
#   PYTHON=/tmp/venv/bin/python tools/bootstrap.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-python3}"

# 1) 版本门
if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "FAIL：需要 Python >= 3.10，$PY 报的是 $("$PY" -V 2>&1)" >&2
  echo "下一步：装一个 3.10+ 的解释器，然后 PYTHON=/path/to/python3.12 tools/bootstrap.sh" >&2
  exit 2
fi

# 2) 依赖门：requirements.txt 的包名 → import 名（只列包名与模块名不同的）
declare -A IMPORT_NAME=( [PyYAML]=yaml )
declare -a MISSING=()
while IFS= read -r spec; do
  case "$spec" in ''|'#'*) continue ;; esac
  pkg="${spec%%[<>=!~ ]*}"
  mod="${IMPORT_NAME[$pkg]:-${pkg,,}}"
  if ! "$PY" -c "import $mod" >/dev/null 2>&1; then
    MISSING+=("$spec")
  fi
done < requirements.txt

if ((${#MISSING[@]})); then
  if ! "$PY" -m pip --version >/dev/null 2>&1; then
    echo "FAIL：缺依赖 ${MISSING[*]}，而 $PY 没有 pip" >&2
    echo "下一步：用一个带 pip 的解释器建 venv，再用它重跑本脚本：" >&2
    echo "  python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt" >&2
    echo "  PYTHON=.venv/bin/python tools/bootstrap.sh" >&2
    exit 3
  fi
  echo "装依赖：${MISSING[*]}"
  "$PY" -m pip install --disable-pip-version-check -r requirements.txt
else
  echo "依赖已满足（requirements.txt 无须安装）"
fi

# 3) 报版本
"$PY" - <<'PY'
import importlib, sys
print(f"python {sys.version.split()[0]}")
for mod, pkg in (("yaml", "PyYAML"),):
    m = importlib.import_module(mod)
    print(f"{pkg} {getattr(m, '__version__', '?')}")
PY

if ! "$PY" -c 'import pytest, numpy, torch' >/dev/null 2>&1; then
  echo "note：$PY 里没有 pytest/numpy/torch ⇒ 跑不了 tests/（见 requirements-dev.txt）；机检四件套不需要它们。"
fi

echo "OK：机检环境就绪。下一步：$PY tools/summary.py --check"
