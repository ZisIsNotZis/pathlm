#!/usr/bin/env python3
"""轴 3 · C1 判据带声明审计（docs/research-schema.md §12.1 的声明义务）。

用途：枚举本轴 8 份 `project.yaml` 的 `validation` / `objective_criteria` / `falsification` / `conclusion`
四个字段，分别判两件事——
  mentions_band：该字段是否引用了噪声带 / 阈值 / 率类 CI（如 0.014、"噪声带"、bootstrap、Wilson）；
  declares_band：该字段是否**显式声明了用哪个带**（"用哪个带" 或 "判据带（C1 裁决）"）。
按 §12.1，**声明义务在 `falsification`**；本脚本把四列都打出来，便于人工核对哪些字段引用了带却没声明。

用法：python3 .scratch/64-axis3-width/c1_band_audit.py
退出码：0 = 全部 `falsification` 都已声明；1 = 存在缺声明的项（并打印清单）。
"""
import glob
import re
import sys

import yaml

BAND = re.compile(r"(0\.014|噪声带|判据带|用哪个带|bootstrap|Wilson|95% ?CI|置信区间)")
DECLARE = re.compile(r"(用哪个带|判据带（C1|判据带\(C1|C1 裁决)")
FIELDS = ("validation", "objective_criteria", "falsification", "conclusion")

missing = []
rows = []
for path in sorted(glob.glob("axes/3-width/*/project.yaml")):
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    item = doc.get("id")
    for field in FIELDS:
        text = doc.get(field) or ""
        mentions = bool(BAND.search(text))
        declares = bool(DECLARE.search(text))
        rows.append((item, field, mentions, declares))
        if field == "falsification" and not declares:
            missing.append(item)

for item, field, mentions, declares in rows:
    print(f"{item:7s} {field:20s} mentions_band={str(mentions):5s} "
          f"declares_band={str(declares):5s}")

if missing:
    print("\n缺 falsification 带声明的项：" + ", ".join(missing), file=sys.stderr)
    sys.exit(1)
print("\nOK：本轴全部 falsification 均已声明用哪个带。")
