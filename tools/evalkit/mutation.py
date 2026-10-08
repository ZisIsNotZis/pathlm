"""统一变异验证框架（F3 / 复核 B4）。

**问题（B4）**：各实验的 `mutate.py` 把 pytest 的**任意非零返回码**当「抓红」——
收集错误（import/语法）、环境错误（pytest 没装、超时、OOM）都会被误记成「变异被抓」，
而「未抓到」也不总是非零退出。

**本模块把分类、判红与退出码统一**：

| 类别 | 判据 | 语义 |
|---|---|---|
| `passed` | rc == 0 | 变异**没被抓到**（不算抓红） |
| `test_failure` | rc == 1（或输出含 `failed`/`FAILED`） | **抓红成功**（唯一算作抓红的类别） |
| `collection_error` | 输出含 `errors during collection` / `ERROR collecting` / `no tests ran`，或 rc ∈ {2, 5} | 测试根本没跑起来 / 被中断 |
| `environment_error` | rc ∉ {0,1,2,5}（3 内部错误、4 用法错误、127、负信号…），或启动失败/超时 | 环境问题 |

用法（每个实验的 `mutate.py`）：

    from tools.evalkit import mutation
    r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q", TESTS], cwd=ROOT)
    # r.returncode 非零 ⟺ 分类 == test_failure（收集/环境错误已在 pytest_run 内抛异常）
    caught = r.returncode != 0
    mutation.exit_if_uncaught(all_caught, report, os.path.join(HERE, "mutation_report.json"))

`pytest_run` 对 `collection_error` / `environment_error` **抛 `ClassifiedError`**（并打印
分类与原始输出尾部），所以脚本立刻非零退出，而不是把「测试没跑起来」当成「抓红成功」。
进程内（monkeypatch 型）变异用 `inprocess_outcome`，分类规则见该函数。
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys

PASSED = "passed"
TEST_FAILURE = "test_failure"
COLLECTION_ERROR = "collection_error"
ENVIRONMENT_ERROR = "environment_error"

#: 只有这个类别算「抓红」
CAUGHT_STATUS = TEST_FAILURE
_COLLECTION_MARKERS = ("errors during collection", "ERROR collecting",
                       "no tests ran", "error during collection")
_ENV_MARKERS = ("ModuleNotFoundError: No module named 'pytest'",
                "command not found", "No such file or directory")


class ClassifiedError(RuntimeError):
    """变异验证里出现「测试没跑起来」——不计抓红，且必须以非零码结束脚本。"""

    def __init__(self, status: str, detail: str):
        super().__init__(f"[{status}] {detail}")
        self.status = status
        self.detail = detail


def classify_pytest(returncode: int, stdout: str = "", stderr: str = "") -> str:
    """pytest 退出码 + 输出 → 四个类别之一（判据见模块文档）。"""
    out = f"{stdout}\n{stderr}"
    if returncode == 0:
        return PASSED
    for marker in _COLLECTION_MARKERS:
        if marker in out:
            return COLLECTION_ERROR
    for marker in _ENV_MARKERS:
        if marker in out:
            return ENVIRONMENT_ERROR
    if returncode in (2, 5):
        return COLLECTION_ERROR
    if returncode in (3, 4):
        return ENVIRONMENT_ERROR
    if returncode == 1:
        return TEST_FAILURE
    return ENVIRONMENT_ERROR


def pytest_run(cmd, cwd=None, capture_output=True, text=True, timeout=None, **kw):
    """`subprocess.run` 的同形替身（**只用于 pytest 调用**）。

    返回的对象与 `CompletedProcess` 同形（`returncode/stdout/stderr`）；区别是：
    `collection_error` / `environment_error` 直接抛 `ClassifiedError`（脚本非零退出），
    于是调用点的 `returncode != 0` 就**恰好**等价于「测试失败被抓红」。
    """
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    try:
        r = subprocess.run(argv, cwd=cwd, capture_output=capture_output, text=text,
                           timeout=timeout, **kw)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise ClassifiedError(ENVIRONMENT_ERROR,
                              f"pytest 启动失败/超时：{exc!r}；cmd={argv}") from exc
    status = classify_pytest(r.returncode, r.stdout or "", r.stderr or "")
    tail = (r.stdout or r.stderr or "").strip().splitlines()
    print(f"[mutation] pytest rc={r.returncode} → {status}"
          + (f" :: {tail[-1][:160]}" if tail else ""), flush=True)
    if status in (COLLECTION_ERROR, ENVIRONMENT_ERROR):
        raise ClassifiedError(status,
                              f"变异验证的 pytest 以 {status} 结束（**不计抓红**）："
                              f"cmd={argv}\n{(r.stdout or '')[-800:]}\n{(r.stderr or '')[-400:]}")
    return r


def pytest_main(argv, plugins=None) -> int:
    """进程内 `pytest.main` 的同形替身（用于「同一个进程里打猴子补丁」的变异脚本）。

    返回退出码；但 `collection_error` / `environment_error` 直接抛 `ClassifiedError`
    （**不计抓红**），于是调用点的 `rc != 0` 恰好等价于「测试失败被抓红」。
    """
    import pytest as _pytest                                   # noqa: PLC0415
    rc = int(_pytest.main(list(argv), plugins=plugins))
    status = classify_pytest(rc)
    print(f"[mutation] pytest.main rc={rc} → {status}", flush=True)
    if status in (COLLECTION_ERROR, ENVIRONMENT_ERROR):
        raise ClassifiedError(status,
                              f"进程内 pytest 以 {status} 结束（**不计抓红**）：argv={argv}")
    return rc


def inprocess_outcome(fn, args=(), truncated: int = 160) -> tuple[bool, str, str]:
    """进程内（monkeypatch 型）变异的判定：`(caught, detail, status)`。

    - 无异常 ⇒ `passed`（没抓到）；
    - `AssertionError` ⇒ `test_failure`（干净抓红）；
    - 其它异常 ⇒ `test_failure`（等价 pytest 里 FAILED），但 detail 标注异常名；
    - `ImportError` / `ModuleNotFoundError` / `OSError` / `MemoryError` / `SystemExit` /
      `KeyboardInterrupt` ⇒ `environment_error`（**抛 ClassifiedError**，不计抓红）。
    """
    try:
        fn(*args)
    except AssertionError as exc:
        return True, str(exc)[:truncated], TEST_FAILURE
    except (ImportError, ModuleNotFoundError, OSError, MemoryError,
            SystemExit, KeyboardInterrupt) as exc:
        raise ClassifiedError(ENVIRONMENT_ERROR,
                              f"进程内变异遇到环境类异常 {type(exc).__name__}: {exc}") from exc
    except Exception as exc:                                     # noqa: BLE001
        return True, f"{type(exc).__name__}: {exc}"[:truncated], TEST_FAILURE
    return False, "test 未抓红（不变量未被钉住）", PASSED


def exit_if_uncaught(ok: bool, report: dict | None = None, report_path: str | None = None,
                     message: str = "有变异未被抓红") -> None:
    """统一收尾：写报告 + 未抓到/未还原 ⇒ 非零退出（F3 的硬要求）。"""
    if report is not None and report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    if not ok:
        print(f"[mutation] FAIL：{message}", file=sys.stderr)
        sys.exit(1)
    print("[mutation] OK：全部变异抓红且逐字节还原")


def run_inprocess_specs(specs, load_testmod) -> tuple[dict, bool]:
    """进程内变异的一站式跑法（可选）：`specs` = [(id, desc, broken_fn, test_name, args)]。

    每个 spec 在 `load_testmod()` 返回的测试模块上打补丁（调用方负责 `setattr` 与还原，
    由 `broken_fn` 自己实现——见各实验 `mutate.py` 的既有写法）。
    """
    report = {"mutations": []}
    ok = True
    for spec in specs:
        mid, desc, run, restore = spec
        caught, detail, status = run(load_testmod, restore)
        ok = ok and caught
        report["mutations"].append({"id": mid, "desc": desc, "status": status,
                                    "caught_red": caught, "detail": detail})
        print(f"[mutation] {mid}: {status} caught={caught} :: {detail}", flush=True)
    report["all_caught"] = ok
    return report, ok


def summarize(report: dict) -> dict:
    """把报告折成「类别直方图」，便于 CI 汇总（F3：未抓到必须可汇总）。"""
    hist: dict[str, int] = {}
    for m in report.get("mutations", []):
        hist[m.get("status", "unknown")] = hist.get(m.get("status", "unknown"), 0) + 1
    return {"n_mutations": len(report.get("mutations", [])), "by_status": hist,
            "all_caught": bool(report.get("all_caught", False))}


def hunt_repo_mutate_scripts(repo_root: str) -> list[str]:
    """列出仓库里所有 `*/mutate.py`（供守卫测试用）。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in (".git", "__pycache__", "node_modules")]
        for fn in filenames:
            if fn == "mutate.py":
                out.append(os.path.relpath(os.path.join(dirpath, fn), repo_root))
    return sorted(out)
