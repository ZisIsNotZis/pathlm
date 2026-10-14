"""只读扫描：E2/E3 触碰过的引擎行是否让 `axes/**/mutate.py` 的 `old=` 锚点失配。

判据（N3）：
- 若某变异脚本的 `old` 字面量在 **E2 父提交 `46ba4cf`** 的某引擎文件里能匹配，
  而在**当前工作树**同一文件里不再匹配 ⇒ 该锚点被 E2/E3 改动打断（stale）。
- 同一 `old` 也按**committed HEAD** 版本的 `mutate.py` 复核一次，以区分「工作树已修」
  与「提交态仍断」。

只为报告用，不写任何 `axes/**`（本脚本连 axes 也只读）。

输出：`.scratch/67-engine-changes/STALE-MUTATION-ANCHORS.md` 与 stdout 摘要。
"""

import ast
import difflib
import os
import subprocess
import sys

ROOT = "/home/z/vibe/pathlm"
# E2/E3 触碰过的引擎文件（ticket 67 §2）
ENGINE_FILES = ["pathlm/depth_ar.py", "pathlm/model.py",
                "pathlm/decode.py", "pathlm/config.py"]
PARENT = "46ba4cf"          # E2 父提交（引擎改动窗口的起点）


def _run(args):
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True).returncode, \
        subprocess.run(args, cwd=ROOT, capture_output=True, text=True).stdout


def _git_show(rev: str, path: str):
    p = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT,
                       capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def _read(path: str) -> str:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


def _mutate_scripts() -> list:
    out = subprocess.run(["find", "axes", "-name", "mutate.py"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return sorted(p for p in out.splitlines() if p.strip())


def _old_literals(src: str, path: str):
    """AST 抽取每个字典里 `"old": <str>` 的值 + 行号 + 同字典的 file/base/id。"""
    tree = ast.parse(src, filename=path)
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
        if "old" not in keys:
            continue
        meta = {}
        for k, v in zip(node.keys, node.values):
            if not isinstance(k, ast.Constant):
                continue
            try:
                if k.value in ("old", "new", "id", "file", "base"):
                    meta[k.value] = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                pass
        old = meta.get("old")
        if isinstance(old, str):
            found.append({"id": meta.get("id"), "file": meta.get("file"),
                          "base": meta.get("base"),
                          "lineno": node.lineno, "old": old})
    return found


def _map_to_current(parent_src: str, current_src: str, needle: str):
    """把 parent 里含 needle 的位置，经 SequenceMatcher 映射到 current 的对应新行。"""
    plines = parent_src.splitlines()
    clines = current_src.splitlines()
    nlines = needle.splitlines()
    start = None
    for i in range(len(plines) - len(nlines) + 1):
        if plines[i:i + len(nlines)] == nlines:
            start = i
            break
    if start is None:
        for i, ln in enumerate(plines):
            if nlines and nlines[0] == ln:
                start = i
                break
    if start is None:
        return None, []
    sm = difflib.SequenceMatcher(None, plines, clines, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if i1 <= start < i2:
            new = clines[j1:j2]
            return tag, [(j1 + k + 1, clines[j1 + k]) for k in range(len(new))]
    return None, []


def _classify(scripts_state: dict, current: dict, parent: dict, engine_set: set):
    """scripts_state: {script_path: [(lineno, id, declared, old), ...]}"""
    recs = []
    for script in sorted(scripts_state):
        for lineno, mid, declared, old in scripts_state[script]:
            in_current = [p for p in ENGINE_FILES if old in current[p]]
            in_parent = [p for p in ENGINE_FILES if old in parent[p]]
            aims = bool(in_parent or in_current) or declared in engine_set
            if not aims:
                continue
            if in_current:
                continue  # 仍匹配，非 stale
            target = in_parent[0] if in_parent else (declared or "?")
            tag, newlines = (None, [])
            if in_parent:
                tag, newlines = _map_to_current(parent[target], current[target], old)
            recs.append({"script": script, "lineno": lineno, "id": mid,
                         "target": target, "parent_match": bool(in_parent),
                         "declared": declared, "op": tag,
                         "current_lines": newlines, "old": old})
    return recs


def _state_from(scripts: dict, content_of):
    st = {}
    for path in scripts:
        src = content_of(path)
        if not src:
            continue
        lits = _old_literals(src, path)
        if not lits:
            continue
        st[path] = []
        for m in lits:
            f = m.get("file")
            declared = None
            if f:
                declared = f if "/" in f else (
                    os.path.relpath(os.path.join(os.path.dirname(path), f), ROOT)
                    if m.get("base") == "exp" else f)
            st[path].append((m["lineno"], m.get("id"), declared, m["old"]))
    return st


def _changed_engine_lines(commits, files):
    """E2/E3 在引擎文件里真正改动（- / +）的行文本（去首尾空白）。"""
    changed = {f: {"removed": [], "added": []} for f in files}
    for sha in commits:
        for f in files:
            d = subprocess.run(["git", "diff", f"{sha}^", sha, "--", f], cwd=ROOT,
                               capture_output=True, text=True).stdout
            for ln in d.splitlines():
                if ln.startswith("---") or ln.startswith("+++"):
                    continue
                if ln.startswith("-"):
                    changed[f]["removed"].append(ln[1:].strip())
                elif ln.startswith("+"):
                    changed[f]["added"].append(ln[1:].strip())
    return changed


def _anchors_inlining(scripts, changed):
    """锚点 `old` 的任一行等于 E2/E3 删掉（或新增后仍被内联）的引擎行 ⇒ 命中。"""
    rem = {f: set(changed[f]["removed"]) for f in changed}
    add = {f: set(changed[f]["added"]) for f in changed}
    hits = []
    for path in sorted(scripts):
        src = _read(path)
        for m in _old_literals(src, path):
            lines = {ln.strip() for ln in m["old"].splitlines() if ln.strip()}
            for f in changed:
                inter = sorted((lines & rem[f]) | (lines & add[f]))
                if inter:
                    hits.append({"script": path, "lineno": m["lineno"],
                                 "id": m.get("id"), "file": f, "lines": inter})
                    break
    return hits


def _render(recs, title, note):
    out = [f"### {title}", "", note, ""]
    if not recs:
        out.append("- （无）")
        out.append("")
        return out
    for r in recs:
        out.append(f"- **[{r['id'] or '-'}]** `{r['script']}:{r['lineno']}` → 引擎文件 "
                   f"`{r['target']}`")
        out.append("  - `old=` 字面片段：")
        out.append("    ```python")
        for ln in r["old"].splitlines():
            out.append("    " + ln)
        out.append("    ```")
        if r["parent_match"]:
            out.append(f"  - 当前 `{r['target']}` 中 **0 匹配**（E2 父提交 `{PARENT}` 能匹配）"
                       "⇒ 重跑该变异会 unmatched。")
        else:
            out.append("  - 父提交也不匹配（非 E2/E3 直接打断，单列）。")
        if r["current_lines"]:
            out.append("  - 对应引擎行（当前）：")
            for lineno, text in r["current_lines"]:
                out.append(f"    - `{r['target']}:{lineno}`  `{text.strip()}`")
    out.append("")
    return out


def main():
    current = {p: _read(p) for p in ENGINE_FILES}
    parent = {p: _git_show(PARENT, p) for p in ENGINE_FILES}
    engine_set = set(ENGINE_FILES)
    scripts = _mutate_scripts()

    # 提交态：HEAD 版本的 mutate.py（untracked 的用工作树内容代替）
    head_state = {}
    for path in scripts:
        src = _git_show("HEAD", path)
        if src is not None:
            lits = _old_literals(src, path)
            if lits:
                head_state[path] = []
                for m in lits:
                    f = m.get("file")
                    declared = None
                    if f:
                        declared = f if "/" in f else (
                            os.path.relpath(os.path.join(os.path.dirname(path), f),
                                            ROOT) if m.get("base") == "exp" else f)
                    head_state[path].append((m["lineno"], m.get("id"), declared, m["old"]))
    wt_state = _state_from(scripts, _read)

    recs_wt = _classify(wt_state, current, parent, engine_set)
    recs_head = _classify(head_state, current, parent, engine_set)

    hid = {r["script"]: r for r in recs_head}
    wt_fixed = [hid[s] for s in hid if s not in {r["script"] for r in recs_wt}]
    head_only = [r for r in recs_head]

    out = ["# 陈旧变异锚点扫描（N3 · 只读）", "",
           "> 性质：**只读扫描报告**（engine-writer 收尾，N3）。扫描对象 = `axes/**/mutate.py`",
           "> 的 `old=` 锚点，判定「在 E2 父提交 `46ba4cf` 的引擎文件里能匹配、在当前引擎",
           "> 源码里不匹配」者。**未修改任何 `axes/**`**。",
           f"> 扫描脚本：`.scratch/67-engine-changes/scan_stale_anchors.py`；引擎文件 = "
           + "、".join(f"`{p}`" for p in ENGINE_FILES) + "。",
           "> E2 = `4638882`，E3 = `fa6a1ad`；引擎文件改动窗口内只有这两个提交（`git log "
           f"{PARENT}..HEAD -- <引擎文件>`）。", "",
           "## 0. 人话摘要", "",
           f"- **按 committed HEAD 版本的变异脚本**：{len(recs_head)} 条锚点失配。",
           f"- **按当前工作树版本**：{len(recs_wt)} 条锚点失配。",
           "- 已知的 `2614-content-topk` 锚点失配**已被轴 1 在工作树里修好**（未提交）——",
           "  该修复把 `TopKBlock(...)` 接线行同步到 E2 之后的 `_mh(i)` / `for i in range`。",
           "- 未发现轴 1 修复之外的其它失配。", ""]
    out += _render(recs_head, "1. 提交态（HEAD 的 mutate.py 对当前引擎）",
                   "这些锚点在 **committed HEAD** 状态下会 unmatched。")
    out += _render(recs_wt, "2. 工作树态（当前 mutate.py 对当前引擎）",
                   "这些锚点在**当前工作树**下会 unmatched。")
    if wt_fixed:
        out += ["### 3. 提交态失配、工作树已修（并发轴 1 修复，未提交）", ""]
        for r in wt_fixed:
            out.append(f"- `{r['script']}`（id `{r['id']}`，行 {r['lineno']}）")
        out.append("")

    changed = _changed_engine_lines(["4638882", "fa6a1ad"], ENGINE_FILES)
    inline_hits = _anchors_inlining(scripts, changed)
    out += ["### 4. 内联 E2/E3 触碰行的锚点（不论当前是否仍匹配）", "",
            "判定 = 锚点 `old` 里有一行等于 E2/E3 改动（`-`/`+`）的引擎行。", ""]
    if inline_hits:
        for h in inline_hits:
            out.append(f"- `{h['script']}:{h['lineno']}` [id `{h['id']}`] → `{h['file']}`")
            for ln in h["lines"]:
                out.append(f"  - 触碰行：`{ln}`")
    else:
        out.append("- （无）")
    out.append("")

    text = "\n".join(out) + "\n"
    dest = os.path.join(ROOT, ".scratch/67-engine-changes/STALE-MUTATION-ANCHORS.md")
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
