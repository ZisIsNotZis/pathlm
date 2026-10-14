#!/usr/bin/env python3
"""C1 声明义务扫描（ticket 66 · Phase B）。

用途：枚举本轴（`axes/5-quant/**/project.yaml`）所有**引用判据带 / 阈值**的字段
（`validation` / `objective_criteria` / `falsification` / `conclusion` / `milestone.*`），
并对每个字段给出三个布尔量：

- `refs_band`：该字段是否引用某条带 / 阈值（`0.014`、`噪声带`、`在力带`、`chance`、`门槛`、`阈值`、
  `Wilson`、`CI`、`可排序`、`越带`、`不越带`）；
- `names_band`：该字段是否**点明**用哪条带（跨格 `0.014` 带 / 本格在力带 / 概率类 CI）；
- `is_falsification`：是否为 `falsification`（`docs/research-schema.md` §12.1 对其**硬性**要求声明）。

判定：`refs_band and not names_band` ⇒ **缺声明**（硬性项若为 `falsification` 则标 `[HARD]`）。

只读；不改任何文件。用法：`python3 c1_band_declaration_scan.py`（非零退出 = 存在缺声明）。
"""
import glob
import os
import sys

import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

REF_TOKENS = ("0.014", "噪声带", "在力带", "chance", "0.0156", "门槛", "阈值",
              "Wilson", "可排序", "越带", "带内", "不越")
# 点名「用哪条带」的写法（任一条即可）
NAME_TOKENS = ("用哪个带", "在力带", "0.014 bpc 带", "0.014 噪声带", "Wilson 95%", "Wilson 95% CI",
               "配对 bootstrap", "本格在力带", "本格带")


def fields(doc):
    out = {}
    for k in ("validation", "objective_criteria", "falsification", "conclusion", "conclusion_review"):
        v = doc.get(k)
        out[k] = v if isinstance(v, str) else ("" if not v else str(v))
    ms = doc.get("milestone") or {}
    for k in ("output", "acceptance"):
        if k in ms:
            out["milestone." + k] = str(ms.get(k) or "")
    return out


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "axes", "5-quant", "*", "project.yaml")))
    missing = []
    print(f"scan {len(files)} project.yaml under axes/5-quant/")
    for f in files:
        doc = yaml.safe_load(open(f))
        rel = os.path.relpath(f, ROOT)
        for k, v in fields(doc).items():
            if not v:
                continue
            refs = any(t in v for t in REF_TOKENS)
            names = any(t in v for t in NAME_TOKENS) or ("0.014" in v and "带" in v)
            if refs and not names:
                tag = "[HARD]" if k == "falsification" else "[soft]"
                print(f"  {rel} :: {k} ⇒ 引用带/阈值但未点明用哪条带 {tag}")
                missing.append((rel, k, tag))
            elif refs and names:
                print(f"  {rel} :: {k} ⇒ 已声明（refs+names）")
    print(f"\nMISSING={len(missing)}")
    for rel, k, tag in missing:
        print(f"  {rel} :: {k} {tag}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
