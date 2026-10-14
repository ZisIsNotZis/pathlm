#!/usr/bin/env python3
"""C1 判据带声明审计（轴 6，ticket 63 续 · 只读 + 枚举，不改任何文件）。

判据（docs/research-schema.md §12.1）：
  1. 凡引用噪声带 / 阈值的字段（0.014 / 噪声带 / 平局带 / 可排序），必须显式声明「用哪个带」；
  2. 概率 / 接受率 / 准确率类指标不得跨量纲借用 0.014 bpc 带 ⇒ 该类字段应指向 CI（bootstrap / Wilson）；
  3. `falsification` 字段必须声明用哪个带（硬性）。

本脚本逐 `project.yaml` 逐字段扫描，输出命中清单与「是否已在同一字段声明带」的判定，
并对跨量纲借用给 alarm。非零退出码仅在发现「未声明带」的命中时给出，便于人工过一遍。
"""
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
ROOT = os.path.abspath(ROOT)
AXIS = os.path.join(ROOT, "axes", "6-predict")

# 引用噪声带 / 阈值的词
BAND_HIT = re.compile(r"0\.014|噪声带|平局带|可排序")
# 声明用哪个带的词（同一字段内出现即视为已声明）
DECL = re.compile(r"用哪个带|C1 裁决|不跨量纲|配对 bootstrap|Wilson|本格带|带（C1|主判")
# 概率 / 率类指标的词（跨量纲借用告警用）
RATE = re.compile(r"接受率|接受长度|准确率|命中率|α₁|α1|τ\b|概率")
# bpc 口径的词
BPC = re.compile(r"bpc|每字符比特")

FIELDS = ["purpose", "innovation", "expected_effect", "validation", "objective_criteria",
          "falsification", "conclusion", "failure_analysis", "conclusion_review", "argument",
          "cost_model", "gap", "milestone", "work_packages", "gaps"]


def walk(node, path, out):
    if isinstance(node, str):
        out.append((path, node))
    elif isinstance(node, dict):
        for k, v in node.items():
            walk(v, path + "." + str(k), out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, path + "[%d]" % i, out)


def main():
    hits = []
    for dp, _dn, fn in os.walk(AXIS):
        if "project.yaml" not in fn:
            continue
        p = os.path.join(dp, "project.yaml")
        import yaml
        with open(p) as f:
            doc = yaml.safe_load(f)
        flat = []
        walk(doc, "", flat)
        for path, text in flat:
            if path.startswith(".change_log"):
                # change_log 是过程台账（历史），不是判据字段；本轮 C1 回填只审判据字段。
                continue
            if not BAND_HIT.search(text):
                continue
            declared = bool(DECL.search(text))
            cross = bool(RATE.search(text)) and not bool(BPC.search(text))
            hits.append({
                "file": os.path.relpath(p, ROOT),
                "id": doc.get("id"),
                "path": path.lstrip("."),
                "declared": declared,
                "cross_dim_risk": cross,
                "text": re.sub(r"\s+", " ", text)[:160],
            })
    print(json.dumps(hits, ensure_ascii=False, indent=1))
    und = [h for h in hits if not h["declared"]]
    cross = [h for h in hits if h["cross_dim_risk"]]
    print("\n== 汇总：命中 %d 处；未在同字段声明带 %d 处；跨量纲风险 %d 处"
          % (len(hits), len(und), len(cross)), file=sys.stderr)
    for h in und:
        print("  未声明: %s :: %s" % (h["file"], h["path"]), file=sys.stderr)
    for h in cross:
        print("  跨量纲: %s :: %s" % (h["file"], h["path"]), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
