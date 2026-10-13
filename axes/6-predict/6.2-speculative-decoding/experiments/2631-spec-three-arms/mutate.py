"""R6 · 变异验证（复用统一框架 `tools/evalkit/mutation.py`，F3 / 复核 B4）。

**本脚本做三件事**（supervisor 裁决 ① 的第 2 条硬要求）：
1. 逐条把**源码文本**改成坏版本，跑 `test_spec_arms.py`，只看 `test_failure` 算「抓红」
   （收集/环境错误由 `mutation.pytest_run` 抛 `ClassifiedError` ⇒ 本脚本非零退出）；
2. **逐字节还原**（sha256 前后相等，不等即非零退出）；
3. 至少一条变异打在**引擎修复那一行**（`pathlm/model.py` 的 `cond.shape[1] >= h.shape[1]`），
   保证「修复真的被单测钉住」。

跑法：`python3 axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/mutate.py`
"""

from __future__ import annotations

import hashlib
import json
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

TESTS = os.path.join(HERE, "test_spec_arms.py")

#: 每条 = 打一处源码变异；`old` 必须在文件里**唯一**出现。
CASES = [
    {
        "id": "M1",
        "file": "pathlm/model.py",
        "old": "t_next = (cond if cond.shape[1] >= h.shape[1]\n"
               "                              else torch.cat([cond, cond[:, -1:]], dim=1))",
        "new": "t_next = cond  # 永远不右垫",
        "desc": "F_ENG：去掉右垫 ⇒ T>=2 的批量路径行数失配（引擎修复被抓红）",
    },
    {
        "id": "M2",
        "file": "pathlm/model.py",
        "old": "t_next = (cond if cond.shape[1] >= h.shape[1]\n"
               "                              else torch.cat([cond, cond[:, -1:]], dim=1))",
        "new": "t_next = torch.cat([cond, cond[:, -1:]], dim=1)  # 永远右垫（修复前）",
        "desc": "F_ENG：退回修复前行为 ⇒ T==1 单位置解码崩溃（回归被抓红）",
    },
    {
        "id": "M3",
        "file": "spec_arms.py", "base": "exp",
        "old": "    if arm == \"chain\":\n        return nodes[1][\"chain2\"][\"logits\"][0, -1].float()",
        "new": "    if arm == \"chain\":\n        return nodes[2][\"logits\"][0, -1].float()",
        "desc": "chain 臂错接到 direct 头（臂定义被抓红）",
    },
    {
        "id": "M4",
        "file": "spec_arms.py", "base": "exp",
        "old": "COMBINED_LAM = 0.5",
        "new": "COMBINED_LAM = 0.0",
        "desc": "预登记混合系数被改（判据线被抓红）",
    },
    {
        "id": "M5",
        "file": "spec_arms.py", "base": "exp",
        "old": "            drafts.append(int(nodes[k][\"logits\"][0, -1].argmax()))",
        "new": "            drafts.append(int(second_slot_logits(nodes, arm, lam).argmax()))",
        "desc": "第二个草稿位也随臂变（唯一变量被破坏 ⇔ 固定背景被抓红）",
    },
    {
        "id": "M6",
        "file": "spec_arms.py", "base": "exp",
        "old": "        \"tokens_per_forward\": n_new / n_rounds,",
        "new": "        \"tokens_per_forward\": 1.0 + n_new / n_rounds,",
        "desc": "tok/fwd 恒等式被破（账目被抓红）",
    },
    {
        "id": "M7",
        "file": "spec_arms.py", "base": "exp",
        "old": "        return lam * pc + (1.0 - lam) * pd",
        "new": "        return lam * nodes[1][\"chain2\"][\"logits\"][0, -1].float() \\\n"
               "            + (1.0 - lam) * nodes[2][\"logits\"][0, -1].float()",
        "desc": "combined 退化成 logits 线性组合（概率混合被抓红）",
    },
]


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> int:
    report = {"mutations": [], "n_cases": len(CASES)}
    ok = True
    for c in CASES:
        path = os.path.join(HERE if c.get("base") == "exp" else ROOT, c["file"])
        before = sha256(path)
        text = open(path, encoding="utf-8").read()
        n = text.count(c["old"])
        if n != 1:
            print(f"[mutation] {c['id']}: 锚点出现 {n} 次（必须恰好 1 次），中止",
                  file=sys.stderr)
            mutation.exit_if_uncaught(False, report,
                                      os.path.join(HERE, "mutation_report.json"),
                                      message=f"{c['id']} 锚点不唯一")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text.replace(c["old"], c["new"], 1))
        try:
            r = mutation.pytest_run(
                [sys.executable, "-m", "pytest", "-q", TESTS], cwd=ROOT)
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
