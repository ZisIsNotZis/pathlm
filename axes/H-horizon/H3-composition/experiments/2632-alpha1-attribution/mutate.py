"""2632 · 变异验证（复用统一框架 `tools/evalkit/mutation.py`）。

对本实验 claim-critical 的接线/统计各打一处源码变异，跑 `test_diag.py`，只认
`test_failure`（收集/环境错误由 `mutation.pytest_run` 抛 `ClassifiedError` ⇒ 非零退出）。
每条变异后逐字节还原（sha256 前后相等，不等即非零退出）。

跑法：`python3 axes/H-horizon/H3-composition/experiments/2632-alpha1-attribution/mutate.py`
"""

from __future__ import annotations

import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _repo_root(start: str) -> str:
    d = start
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, "pathlm")):
            return d
        d = os.path.dirname(d)
    raise RuntimeError(f"找不到仓库根（含 pathlm/ 的祖先目录）: {start}")


ROOT = _repo_root(HERE)
sys.path.insert(0, ROOT)

from tools.evalkit import mutation  # noqa: E402

TESTS = os.path.join(HERE, "test_diag.py")

#: 每条 = 打一处源码变异；`old` 必须在文件里**唯一**出现。
CASES = [
    {"id": "M1", "file": "diag.py",
     "old": "            d_chain = _argmax(lc)",
     "new": "            d_chain = _argmax(l2)  # 错接到 direct 头",
     "desc": "chain_deploy 错读 node2（臂接线被抓红）"},
    {"id": "M2", "file": "diag.py",
     "old": "            onodes = model._mtp_nodes(h, condition=torch.tensor([[true_next]]))",
     "new": "            onodes = model._mtp_nodes(h, condition=torch.tensor([[v1]]))",
     "desc": "oracle 条件退化成 v1（条件线被抓红）"},
    {"id": "M3", "file": "diag.py",
     "old": "            _, fn = dec.step_multi([v1])",
     "new": "            _, fn = dec.step_multi([true_next])  # 目标前缀喂错",
     "desc": "T2*_deploy 用了 teacher 前缀（部署目标被抓红）"},
    {"id": "M4", "file": "diag.py",
     "old": "    return float(np.mean([1.0 if r[a] == r[b] else 0.0 for r in rows])) if rows else None",
     "new": "    return float(np.mean([1.0 if r[a] != r[b] else 0.0 for r in rows])) if rows else None",
     "desc": "_mean_eq 被反转（把所有命中率统计打反，被抓红）"},
    {"id": "M5", "file": "diag.py",
     "old": "    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d",
     "new": "    h = z * math.sqrt(p * (1.0 - p) / n) / d  # 丢掉 Wilson 的小 n 修正",
     "desc": "Wilson CI 退化（k=0 下界不再为 0，被抓红）"},
    {"id": "M6", "file": "diag.py",
     "old": "            row[\"t2teacher\"] = _argmax(dec.last_nodes[1][\"logits\"][0, 0].float())",
     "new": "            row[\"t2teacher\"] = v1  # 用本位置而非下一位置",
     "desc": "t2teacher 取位错（teacher 目标被抓红）"},
    {"id": "M7", "file": "diag.py",
     "old": "    return wilson_ci(sum(1 for r in rows if r[a] == r[b]), len(rows))",
     "new": "    return wilson_ci(sum(1 for r in rows if r[a] == r[b]), len(rows) + 1)",
     "desc": "Wilson 分母错（落盘 wilson_ci 与原始记录不再一致，被抓红）"},
]


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> int:
    report = {"mutations": [], "n_cases": len(CASES)}
    ok = True
    for c in CASES:
        path = os.path.join(HERE, c["file"])
        before = sha256(path)
        text = open(path, encoding="utf-8").read()
        n = text.count(c["old"])
        if n != 1:
            print(f"[mutation] {c['id']}: 锚点出现 {n} 次（须恰好 1 次），中止", file=sys.stderr)
            mutation.exit_if_uncaught(False, report,
                                      os.path.join(HERE, "mutation_report.json"),
                                      message=f"{c['id']} 锚点不唯一")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(c["old"], c["new"], 1))
        try:
            r = mutation.pytest_run([sys.executable, "-m", "pytest", "-q", TESTS], cwd=ROOT)
            caught = r.returncode != 0
        finally:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        after = sha256(path)
        restored = before == after
        ok = ok and caught and restored
        report["mutations"].append({
            "id": c["id"], "file": c["file"], "desc": c["desc"],
            "caught_red": bool(caught),
            "status": mutation.TEST_FAILURE if caught else mutation.PASSED,
            "restored_byte_identical": bool(restored),
            "sha256_before": before, "sha256_after": after,
        })
        print(f"[mutation] {c['id']} {c['file']}: caught={caught} restored={restored}",
              flush=True)
    report["all_caught"] = ok
    report["by_status"] = mutation.summarize(report)["by_status"]
    mutation.exit_if_uncaught(ok, report, os.path.join(HERE, "mutation_report.json"),
                              message="有变异未被抓红或未逐字节还原")
    return 0


if __name__ == "__main__":
    sys.exit(main())
