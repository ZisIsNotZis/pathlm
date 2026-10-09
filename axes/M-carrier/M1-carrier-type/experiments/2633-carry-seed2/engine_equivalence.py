#!/usr/bin/env python3
"""2633-carry-seed2 · **语义级**实现等价证据（SCHEMA §9 第①项的加强版）。

背景（为什么文件哈希不够）：本仓有多条线并行，`pathlm/**` 可能有**别的线的未提交改动**
（本线执行期间确实出现过）。文件哈希只说明「字节是否一样」，回答不了「换了字节之后，
**本实验用到的那条路径**行为是否一样」。

做法：把 `HEAD` 提交版的 `pathlm/` 整包解到临时目录、以独立包名 `old_pathlm` 导入，
与本实验的三档配置（`carry_rule` ∈ {none, plain, delta}，dense + topology=sequential +
窗口 16 + 锚 0，13M 主干）各建一个模型：

1. 同 seed 建两个版本 ⇒ **参数量与参数张量逐位相同**（构造层等价）；
2. 把同一份 `state_dict` 装进两版模型 ⇒ 固定输入上 **logits 逐位相同**（前向等价）；
3. 结论落 `engine_equivalence.json`；`bitwise_equivalent_on_this_experiments_configs`
   为假 ⇒ `equality_vs_2631.compare()` 判 `reuse_rejected`（不许拿「换过引擎」的臂当对照）。

用法：python3 engine_equivalence.py        # 退出码 0 = 等价
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)          # 让 `pathlm` 可导入（本实验只读导入仓库包）
CFG = json.load(open(os.path.join(HERE, "config.json")))

#: 本实验用到的主干规格（来自 config.json，不另写一套）
BASE = dict(CFG["model"])
VOCAB = CFG["data"]["vocab_size"]
RULES = ("none", "plain", "delta")
PROBE_SEED = 1234


def _extract_head_package(dst: str) -> str:
    """把 `HEAD` 版的 `pathlm/*.py` 解到 `dst/old_pathlm/`，返回包目录。"""
    pkg = os.path.join(dst, "old_pathlm")
    os.makedirs(pkg, exist_ok=True)
    files = subprocess.run(["git", "-C", ROOT, "ls-tree", "--name-only", "HEAD", "pathlm/"],
                           capture_output=True, text=True).stdout.split()
    for rel in files:
        if not rel.endswith(".py"):
            continue
        blob = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{rel}"],
                              capture_output=True, text=True).stdout
        with open(os.path.join(pkg, os.path.basename(rel)), "w") as f:
            f.write(blob)
    return pkg


def check() -> dict:
    out: dict = {"configs": [None if r == "none" else r for r in RULES],
                 "probe_seed": PROBE_SEED, "per_rule": {}}
    with tempfile.TemporaryDirectory() as tmp:
        _extract_head_package(tmp)
        if tmp not in sys.path:
            sys.path.insert(0, tmp)
        # `HEAD` 版的整包已解到 tmp/old_pathlm/（含 `__init__.py`）⇒ 直接按包导入
        # （相对导入 `.config` 等就能解析；不要手工注册无 `__path__` 的伪模块）。
        import old_pathlm.depth_ar as old_da

        import pathlm.depth_ar as new_da
        g = torch.Generator().manual_seed(PROBE_SEED)
        x = torch.randint(0, VOCAB, (2, BASE["seq_len"]), generator=g)

        def build(mod, rule):
            rule_val = None if rule == "none" else rule
            torch.manual_seed(0)
            m = mod.DepthARModel(mod.DepthARConfig(**{**BASE, "carry_rule": rule_val}), VOCAB)
            m.eval()
            return m

        for rule in RULES:
            mo, mn = build(old_da, rule), build(new_da, rule)
            so = {k: v.detach().clone() for k, v in mo.state_dict().items()}
            sn = {k: v.detach().clone() for k, v in mn.state_dict().items()}
            keys_same = list(so) == list(sn)
            params_bitwise = keys_same and all(torch.equal(so[k], sn[k]) for k in so)
            npo = sum(p.numel() for p in mo.parameters())
            npn = sum(p.numel() for p in mn.parameters())
            mn.load_state_dict(so, strict=True)          # 同一份权重装进新版
            with torch.no_grad():
                lo = mo(x)[1]["depth_logits"][-1]
                ln = mn(x)[1]["depth_logits"][-1]
            max_dev = float((lo - ln).abs().max().item())
            out["per_rule"][rule] = {
                "n_params_head": npo, "n_params_working_tree": npn,
                "n_params_same": bool(npo == npn),
                "params_bitwise_same_at_same_seed": bool(params_bitwise),
                "state_dict_keys_same": bool(keys_same),
                "max_abs_logit_diff_same_weights": max_dev,
                "bitwise_equal_forward": bool(max_dev == 0.0),
            }
    out["bitwise_equivalent_on_this_experiments_configs"] = all(
        v["bitwise_equal_forward"] and v["n_params_same"] for v in out["per_rule"].values())
    out["note"] = ("工作树引擎与 HEAD 提交版引擎在**本实验用到的**三档配置上逐位等价；"
                   "这只覆盖本实验的配置，不代表两版在所有配置上等价")
    return out


def main() -> int:
    r = check()
    for rule, v in r["per_rule"].items():
        print(f"{rule:6s} params {v['n_params_head']}/{v['n_params_working_tree']} "
              f"bitwise_params={v['params_bitwise_same_at_same_seed']} "
              f"max|Δlogit|={v['max_abs_logit_diff_same_weights']:.3e}")
    print("等价:", r["bitwise_equivalent_on_this_experiments_configs"])
    out = os.path.join(HERE, "engine_equivalence.json")
    with open(out, "w") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return 0 if r["bitwise_equivalent_on_this_experiments_configs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
