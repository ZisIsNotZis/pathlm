#!/usr/bin/env python3
"""2642 · 跨家族同字节预算重画（块标尺口径）—— 纯再分析入口。

契约：同目录 `README.md` + `config.json::criterion`（运行前写入）。
职责：① 从 `2632` 的落盘读数**独立重算**每位置部署字节并对账（门禁 G1）；
      ② 把跨家族点按「部署字节」分组，给出同字节两腿读数；
      ③ 用磁盘 checkpoint 真实参数量拆开头共享族的参数混杂；
      ④ 把 fp8 的解析价签放进同一横轴（质量未测）。
不做：前向、训练、插值造点、GPU 锁、联网。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]  # .../pathlm
PARENT = HERE.parent / "2632-kv-block-scale"

# ---- 记录口径常量（逐字来自 config.json::axis_definition） ----
GROUP_G = 32                      # 块标尺分组（沿 head_dim）
SCALE_BYTES = 2.0                 # 每个标尺占 2 B（fp16）
QUANT_MODES = ("int8", "int4")

FAMILY_OF_CLASS = {
    "baseline": "stock fp16（trunk / MHA）",
    "quant": "trunk × 量化",
    "window": "trunk × 窗口",
    "window_anchor": "trunk × 窗口+锚点",
    "head": "头共享 GQA/MQA",
    "quant_head": "头共享 × 量化",
    "combo_quant_window": "窗口 × 量化",
    "combo_head_window": "头共享 × 窗口",
}

CHECKPOINTS = {
    "trunk": ROOT / "axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt",
    "mha": ROOT / "axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/mha-12k-s0/model.pt",
    "gqa": ROOT / "axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/gqa-12k-s0/model.pt",
    "mqa": ROOT / "axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/runs/mqa-12k-s0/model.pt",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(rel_or_abs: str | Path) -> dict:
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = ROOT / rel_or_abs
    with open(path) as fh:
        return json.load(fh)


def param_count(path: Path, d_model: int = 256, head_dim: int = 32) -> tuple[int, int]:
    """返回 (总参数量, n_kv_heads)。

    checkpoint 用**融合** qkv 权重：行数 = n_heads·head_dim + 2·n_kv·head_dim
    （同一套 trunk 架构，MHA/GQA/MQA 只改 n_kv）⇒ n_kv = (rows − d_model) / (2·head_dim)。
    """
    import torch

    sd = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    total = sum(v.numel() for v in sd.values())
    n_kv = None
    for key, val in sd.items():
        if key.endswith("qkv.weight"):
            n_kv = (int(val.shape[0]) - d_model) // (2 * head_dim)
            break
        if key.endswith("k_proj.weight"):
            n_kv = int(val.shape[0]) // head_dim
            break
    if n_kv is None:
        raise RuntimeError(f"未找到融合 qkv / k_proj 权重：{path}")
    return int(total), int(n_kv)


def parse_2632_table(path: Path) -> dict:
    """解析父实验**另一份产物** `2632/table.md` 表 1（markdown 表格）。

    与 `results.json` 不同格式、由 2632 自己的代码路径生成 ⇒ 用于跨文档独立核对读数。
    列序：臂 | 类别 | 值字节 | 部署字节(block) | ×fp16 | 旧部署(a) | ×fp16(a) | bpc | damage | 量化增量 | copy 召回 | 95% CI | 前缀不变。
    """
    import re

    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 13:
            continue
        try:
            arm = cells[0].strip("`")
            deploy = float(cells[3])
            bpc = float(cells[7])
            recall = float(cells[10].split()[0])
            lo, hi = (float(x) for x in cells[11].strip("[]").split(","))
        except (ValueError, IndexError):
            continue
        out[arm] = {"deployment_bytes_per_position": deploy, "value_bytes": float(cells[2]),
                    "bpc": bpc, "recall_copy": recall, "recall_copy_ci": [lo, hi]}
    return out


def main() -> int:
    log: list[str] = []
    cfg = json.loads((HERE / "config.json").read_text())
    parent_raw = (PARENT / "results.json").read_bytes()
    cfg_raw = (PARENT / "config.json").read_bytes()
    log.append(f"parent results.json sha256 = {hashlib.sha256(parent_raw).hexdigest()}")
    log.append(f"parent config.json  sha256 = {hashlib.sha256(cfg_raw).hexdigest()}")
    for rel, want in cfg["inputs_sha256"].items():
        got = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
        status = "OK" if got == want else "MISMATCH"
        log.append(f"input {rel}: {status}")
        if status != "OK":
            log.append("FATAL：输入指纹与预登记不符 ⇒ 本实验作废")
            print("\n".join(log))
            return 2

    parent = json.loads(parent_raw)
    model = parent["config"]["model"]
    L = int(model["n_layers"])
    head_dim = int(model["d_model"]) // int(model["n_heads"])
    T = int(parent["config"]["axis_definition"]["seq_len_T"])
    points = parent["points"]
    per_arm = parent["deployment_caliber"]["per_arm"]
    band = parent["ranking_band"]["in_force_band_bpc"]

    # ---- ① 独立重算部署字节（门禁 G1） ----
    rows, max_diff, value_only_mismatch = [], 0.0, []
    for p in points:
        arm = p["arm"]
        rec = per_arm[arm]["primary"]
        mode = rec["mode"]
        quantized = mode in QUANT_MODES
        sbv = SCALE_BYTES / GROUP_G if quantized else 0.0
        vbytes = float(p["factor_value_bytes"])
        kept = float(p["factor_positions_kept"])
        n_kv = float(p["factor_n_kv_heads_eff"])
        values_per_position = 2.0 * L * n_kv * head_dim * (kept / T)
        value_bytes = values_per_position * vbytes
        scale_bytes = values_per_position * sbv
        deploy = value_bytes + scale_bytes
        diff = abs(deploy - float(rec["bytes_per_position"]))
        max_diff = max(max_diff, diff)
        vo = abs(value_bytes - float(p["bytes_per_position"]))
        if vo > 1e-9:
            value_only_mismatch.append((arm, value_bytes, p["bytes_per_position"]))
        rows.append({
            "arm": arm, "class": p["class"], "family": FAMILY_OF_CLASS[p["class"]],
            "scheme": rec["scheme"], "group": rec.get("group"), "mode": mode,
            "quantized": quantized,
            "values_per_position": values_per_position,
            "value_bytes_per_position": value_bytes,
            "scale_bytes_per_value": sbv,
            "scale_bytes_per_position": scale_bytes,
            "deployment_bytes_per_position": deploy,
            "recorded_deployment_bytes": float(rec["bytes_per_position"]),
            "recorded_value_only_bytes": float(p["bytes_per_position"]),
            "abs_diff_vs_recorded": diff,
            "bpc": p["bpc"], "damage_bpc": p.get("damage_bpc"),
            "quant_increment_bpc": p.get("quant_increment_bpc"),
            "recall_copy": p["recall_copy"]["accuracy"],
            "recall_copy_ci": p["recall_copy"]["wilson95"],
            "positions_kept": kept, "n_kv_heads_eff": n_kv,
            "source": p["source"],
        })
    rows.sort(key=lambda r: (r["deployment_bytes_per_position"], r["bpc"]))

    # 「必须不变」（预登记第 4 条）——两条证据，分开记，不混：
    #   (a) 透传事实：本实验对读数是**逐值透传、无任何数值变换**（不是独立检验，只是过程陈述）；
    #   (b) 跨文档独立核：把父实验**另一份产物** `2632/table.md` 表 1（markdown，由 2632 自己的代码路径生成、
    #       与 results.json 不同格式）解析出来，与 2642 的 rows 逐臂比部署字节 / bpc / copy 召回 / CI。
    #       这是「读数未被改动」的真检验：若 2632 的 json 与 md 不一致、或 2642 改动了读数，此处必红。
    readings_passed_through_untouched = True  # (a) 构造性事实：rows 的数值字段直接取自同一 points
    table_rows = parse_2632_table(PARENT / "table.md")
    # 容差按 md 的**显示精度**定：字节与 bpc 逐位（md 印整数/6 位小数）；召回与 CI md 只印 4 位小数 ⇒ 容差 5e-5
    tol = {"deployment_bytes_per_position": 1e-9, "bpc": 1e-9, "recall_copy": 5e-5, "recall_copy_ci": 5e-5}
    cross = {"source": str((PARENT / "table.md").relative_to(ROOT)),
             "n_arms_parsed": len(table_rows), "tolerance_by_field": tol,
             "max_abs_diff": {}, "mismatches": [], "rounding_note": "2632/table.md 的召回与 CI 只印 4 位小数 ⇒ 容差 5e-5（不是口径差异）"}
    for field in ("deployment_bytes_per_position", "bpc", "recall_copy"):
        cross["max_abs_diff"][field] = 0.0
    cross["max_abs_diff"]["recall_copy_ci"] = 0.0
    for r in rows:
        t = table_rows.get(r["arm"])
        if t is None:
            cross["mismatches"].append((r["arm"], "missing_in_table_md"))
            continue
        for field, ref in (("deployment_bytes_per_position", t["deployment_bytes_per_position"]),
                           ("bpc", t["bpc"]), ("recall_copy", t["recall_copy"])):
            d = abs(r[field] - ref)
            cross["max_abs_diff"][field] = max(cross["max_abs_diff"][field], d)
            if d > tol[field]:
                cross["mismatches"].append((r["arm"], field, r[field], ref))
        ci_d = max(abs(a - b) for a, b in zip(r["recall_copy_ci"], t["recall_copy_ci"]))
        cross["max_abs_diff"]["recall_copy_ci"] = max(cross["max_abs_diff"]["recall_copy_ci"], ci_d)
        if ci_d > tol["recall_copy_ci"]:
            cross["mismatches"].append((r["arm"], "recall_copy_ci", r["recall_copy_ci"], t["recall_copy_ci"]))
    cross["ok"] = not cross["mismatches"] and cross["n_arms_parsed"] == len(rows)
    log.append(f"必须不变 (a) 透传事实 = {readings_passed_through_untouched}（非独立检验）")
    log.append(f"必须不变 (b) 跨文档独立核（2632/table.md 表 1 vs 2642 rows）：解析 {cross['n_arms_parsed']} 臂，"
               f"max|Δ| = {cross['max_abs_diff']}，不一致 = {len(cross['mismatches'])} ⇒ "
               f"{'PASS' if cross['ok'] else 'FAIL'}")
    log.append(f"上游独立检验：2632::reuse.nonquant_bitwise_preserved = "
               f"{parent['reuse']['nonquant_bitwise_preserved']}（max|Δbpc| = {parent['reuse']['nonquant_max_abs_delta_bpc']}，"
               f"相对 2631 的检验）")

    gate_g1 = max_diff < 1e-9
    bad_scheme = [r["arm"] for r in rows if r["quantized"] and (r["scheme"] != "block" or r["group"] != GROUP_G)]
    gate_g2 = not bad_scheme
    log.append(f"G1 独立重算 vs 2632 记录：max|Δ部署字节| = {max_diff:g} ⇒ {'PASS' if gate_g1 else 'FAIL'}")
    log.append(f"G2 量化臂标尺口径：非块标尺量化臂 = {bad_scheme} ⇒ {'PASS' if gate_g2 else 'FAIL'}")
    log.append(f"仅值口径核对：points[*].bytes_per_position 与独立重算的仅值字节不一致条目 = {len(value_only_mismatch)}")
    log.append(f"在力排序带 = {band} bpc（2632::ranking_band）")

    # ---- ② 同字节分组（跨家族） ----
    buckets: dict[float, list[dict]] = {}
    for r in rows:
        buckets.setdefault(round(r["deployment_bytes_per_position"], 6), []).append(r)
    same_byte_groups = []
    for b in sorted(buckets):
        members = buckets[b]
        fams = sorted({m["family"] for m in members})
        if len(members) >= 2 and len(fams) >= 2:
            bpc_sorted = sorted(m["bpc"] for m in members)
            same_byte_groups.append({
                "bytes_per_position": b,
                "n_members": len(members),
                "families": fams,
                "members": [{**{k: m[k] for k in ("arm", "family", "bpc", "recall_copy", "recall_copy_ci",
                                               "quantized", "positions_kept")},
                             "param_confounded": m["class"] in ("head", "quant_head", "combo_head_window")} for m in
                            sorted(members, key=lambda m: m["bpc"])],
                "bpc_spread": bpc_sorted[-1] - bpc_sorted[0],
                "bpc_spread_exceeds_band": (bpc_sorted[-1] - bpc_sorted[0]) > band,
                "bpc_best_arm": min(members, key=lambda m: m["bpc"])["arm"],
                "bpc_best_is_param_confounded": min(members, key=lambda m: m["bpc"])["class"] in
                                                ("head", "quant_head", "combo_head_window"),
            })

    # ---- ③ 参数量混杂 ----
    params, nkv = {}, {}
    for name, path in CHECKPOINTS.items():
        params[name], nkv[name] = param_count(path)
    d_model, n_heads = 256, 8
    param_table = {
        "d_model": d_model, "n_heads": n_heads, "head_dim": head_dim, "n_layers": L,
        "measured": {k: {"params": v, "n_kv_heads": nkv[k], "path": str(CHECKPOINTS[k].relative_to(ROOT))}
                     for k, v in params.items()},
        "analytic_delta_vs_mha": {},
        "measured_delta_vs_mha": {},
        "analytic_formula": "L · (2·d_model + 2) · (H − n_kv) · head_dim",
    }
    for name in ("gqa", "mqa"):
        a = L * (2 * d_model + 2) * (n_heads - nkv[name]) * head_dim
        m = params["mha"] - params[name]
        param_table["analytic_delta_vs_mha"][name] = a
        param_table["measured_delta_vs_mha"][name] = m
        param_table.setdefault("rel_delta_vs_mha", {})[name] = m / params["mha"]
        log.append(f"参数混杂 {name}: 磁盘实测 MHA−{name.upper()} = {m}（= {m/params['mha']:.4%}），解析 = {a} ⇒ {'一致' if a == m else '不一致'}")
    param_table["analytic_matches_measured"] = all(
        param_table["analytic_delta_vs_mha"][n] == param_table["measured_delta_vs_mha"][n] for n in ("gqa", "mqa"))
    log.append(f"MHA 总参数量 = {params['mha']}，trunk 参数量 = {params['trunk']}（同架构，差 {params['trunk']-params['mha']}）")

    # ---- ④ Pareto（原始 + 越带严格） ----
    def dominator(r, others, strict_band):
        """返回支配 r 的点：字节 ≤ r 且 bpc ≤ r − strict_band，且两者不同时相等。"""
        for o in others:
            if o is r:
                continue
            if (o["deployment_bytes_per_position"] <= r["deployment_bytes_per_position"]
                    and o["bpc"] <= r["bpc"] - strict_band
                    and not (o["deployment_bytes_per_position"] == r["deployment_bytes_per_position"]
                             and o["bpc"] == r["bpc"])):
                return o
        return None

    pareto_raw = [r["arm"] for r in rows if dominator(r, rows, 0.0) is None]
    pareto_band = [r["arm"] for r in rows if dominator(r, rows, band) is None]
    dominated_by = {}
    for r in rows:
        d = dominator(r, rows, band)
        if d is not None and r["arm"] not in pareto_band:
            dominated_by[r["arm"]] = d["arm"]
    dominated_by_raw = {}
    for r in rows:
        d = dominator(r, rows, 0.0)
        if d is not None and r["arm"] not in pareto_raw:
            dominated_by_raw[r["arm"]] = d["arm"]
    log.append(f"Pareto（原始 bpc 序）：{len(pareto_raw)} 臂 = {pareto_raw}")
    log.append(f"Pareto（越 {band} 带才算更好）：{len(pareto_band)} 臂 = {pareto_band}")

    fp8 = parent["deployment_caliber"]["fp8_reference"]
    fp8_row = {
        "arm": "fp8（E4M3/E5M2，解析）",
        "deployment_bytes_per_position": fp8["bytes_per_position"],
        "ratio_vs_fp16": fp8["ratio_vs_fp16"],
        "quality_measured": fp8["quality_measured"],
        "note": fp8["note"],
        "vs_int8_block": {
            "int8_bytes_per_position": next(r["deployment_bytes_per_position"] for r in rows if r["arm"] == "int8"),
            "saving_vs_int8_block_bytes": next(r["deployment_bytes_per_position"] for r in rows if r["arm"] == "int8") - fp8["bytes_per_position"],
        },
    }

    result = {
        "experiment": cfg["experiment"],
        "kind": cfg["kind"],
        "contract": "axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/README.md",
        "criterion": cfg["criterion"],
        "gates": {
            "G1_independent_recompute_matches_2632": gate_g1,
            "G1_max_abs_diff_deployment_bytes": max_diff,
            "G2_all_quantized_arms_use_block_scale": gate_g2,
            "G2_offenders": bad_scheme,
            "value_only_caliber_agrees": not value_only_mismatch,
            "readings_passed_through_untouched": readings_passed_through_untouched,
            "cross_artifact_table_md_ok": cross["ok"],
            "cross_artifact_table_md": cross,
            "parent_nonquant_bitwise_preserved": parent["reuse"]["nonquant_bitwise_preserved"],
            "parent_nonquant_max_abs_delta_bpc": parent["reuse"]["nonquant_max_abs_delta_bpc"],
            "verdict": "PROCEED" if (gate_g1 and gate_g2) else "VOID",
        "verdict_basis": "只由 G1（独立重算字节账）与 G2（块标尺口径）决定；跨文档独立核另列（cross_artifact_table_md）",
        },
        "inputs_sha256": cfg["inputs_sha256"],
        "rows": rows,
        "same_byte_groups": same_byte_groups,
        "param_confound": param_table,
        "pareto": {"raw_arm_order": pareto_raw, "strict_over_band": pareto_band,
                   "dominated_with_dominator": dominated_by,
                   "raw_dominated_with_dominator": dominated_by_raw,
                   "band_bpc": band},
        "fp8_reference": fp8_row,
        "withdrawn_claim": {
            "withdrawn": "「同 −75% 档 GQA 便宜 3.2×」——该说法建立在已作废的逐值标尺 int4 数字上（55-review W-08 / R8）。",
            "replacement_reading": [
                f"块标尺口径下 −71.875% 的档位是 trunk int4 = 3456 B/位置（bpc {next(r['bpc'] for r in rows if r['arm']=='int4')}）；"
                f"头共享 fp16 GQA = {next(r['deployment_bytes_per_position'] for r in rows if r['arm']=='gqa'):g} B/位置"
                f"（bpc {next(r['bpc'] for r in rows if r['arm']=='gqa')}），但其参数量比 MHA 少 "
                f"{param_table['measured_delta_vs_mha']['gqa']}（{param_table['rel_delta_vs_mha']['gqa']:.2%}）。",
                "两族不再落在同一字节水平上（3456 vs 3072），故旧「同档便宜 3.2×」不能用；同字节比较改按 same_byte_groups 读。",
            ],
        },
        "answer": {
            "q": "在块标尺口径的同一「每位置部署字节」横轴上，跨家族可比什么？",
            "g1_gate": "PASS" if gate_g1 else "FAIL",
            "n_same_byte_groups": len(same_byte_groups),
            "n_groups_with_bpc_spread_over_band": sum(1 for g in same_byte_groups if g["bpc_spread_exceeds_band"]),
            "param_confound_resolved": param_table["analytic_matches_measured"],
            "cannot_claim": [
                "不得把 points[*].bytes_per_position（仅值口径）当部署字节用",
                "不得在两点之间插值造点",
                "不得在无参数量列时断言头共享族的机制优势",
                "不得断言 int8+块标尺是 8 位方案里字节最优的（fp8 解析更省、质量未测）",
                "本实验不产生新的质量证据：结论强度上限 = 落盘读数在统一口径下的正确重排",
            ],
        },
        "code": {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))},
    }
    (HERE / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    # ---- table.md ----
    def fmt_ci(ci):
        return f"[{ci[0]:.4f}, {ci[1]:.4f}]"

    lines = ["# 2642 · 跨家族同字节预算（块标尺口径）—— 表", "",
             f"判据带（在力带） = **{band} bpc**；横轴 = 每位置部署字节（值 + 块标尺，G={GROUP_G}）；"
             f"T = {T}，L = {L}，head_dim = {head_dim}。", "",
             "## 表 1 · 27 臂（按部署字节升序）", "",
             "| 臂 | 族 | 部署字节/位置 | 仅值字节/位置 | 标尺 B/值 | bpc | Δbpc vs 本族基线 | 纯量化增量 | copy 召回 [95% CI] | checkpoint |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dmg = "—（本族基线）" if r["damage_bpc"] in (0.0, None) else f"{r['damage_bpc']:+.6f}"
        qinc = "—" if r["quant_increment_bpc"] is None else f"{r['quant_increment_bpc']:+.6f}"
        ckpt = ("2616（头族 12.35M/12.16M）" if r["class"] in ("head", "quant_head", "combo_head_window")
                else "2610（trunk 13.54M）")
        lines.append(f"| `{r['arm']}` | {r['family']} | {r['deployment_bytes_per_position']:g} | "
                     f"{r['value_bytes_per_position']:g} | {r['scale_bytes_per_value']:g} | {r['bpc']:.6f} | "
                     f"{dmg} | {qinc} | {r['recall_copy']:.4f} {fmt_ci(r['recall_copy_ci'])} | {ckpt} |")
    lines += ["", "## 表 2 · 跨家族同字节分组（不靠插值的直接比较）", ""]
    if not same_byte_groups:
        lines.append("**空**：本网格上没有恰好同字节的跨家族点对 ⇒ 跨家族同字节比较在本网格上不可做（须补臂）。")
    for g in same_byte_groups:
        lines.append(f"### {g['bytes_per_position']:g} B/位置（{g['n_members']} 臂 / {len(g['families'])} 族；"
                     f"bpc 极差 {g['bpc_spread']:.6f}，{'越带（可排序）' if g['bpc_spread_exceeds_band'] else '在带内（不可排序）'}）")
        lines += ["", "| 臂 | 族 | bpc | copy 召回 [95% CI] | 参数混杂 |", "|---|---|---|---|---|"]
        for m in g["members"]:
            flag = "是（头族少 8.75%/10.21% 参数）" if m["param_confounded"] else "否"
            lines.append(f"| `{m['arm']}` | {m['family']} | {m['bpc']:.6f} | {m['recall_copy']:.4f} {fmt_ci(m['recall_copy_ci'])} | {flag} |")
        n_conf = sum(1 for m in g["members"] if m["param_confounded"])
        best = min(g["members"], key=lambda m: m["bpc"])
        head_best = "头族点在本组 bpc 最低" if best["param_confounded"] else "本组 bpc 最低点**不是**头族"
        lines.append("")
        lines.append(f"> 本组头族点数 = {n_conf}/{g['n_members']}；{head_best}。"
                     f"头族点的判读受参数量混杂（少 8.75%/10.21% 参数）污染 ⇒ **不构成纯机制优势**（见表 3）；"
                     f"要读「同字节纯机制」须补等参数臂（登记为下一步）。")
        lines.append("")
    lines += ["## 表 3 · 参数量混杂（头共享族）", "",
              "| checkpoint | 参数量（磁盘实测） | n_kv_heads | 与 MHA 的差（实测） | 与 MHA 的差（解析） | 相对 |",
              "|---|---|---|---|---|---|"]
    for name in ("trunk", "mha", "gqa", "mqa"):
        d_m = param_table["measured_delta_vs_mha"].get(name)
        d_a = param_table["analytic_delta_vs_mha"].get(name)
        rel = param_table.get("rel_delta_vs_mha", {}).get(name)
        lines.append(f"| {name} | {params[name]} | {nkv[name]} | "
                     f"{'—' if d_m is None else d_m} | {'—' if d_a is None else d_a} | "
                     f"{'—' if rel is None else f'{rel:.2%}'} |")
    lines += ["", f"解析式 = `{param_table['analytic_formula']}`；解析与实测一致："
                  f"**{param_table['analytic_matches_measured']}**。", "",
              "## 表 4 · Pareto", "",
              f"- 原始 bpc 序非支配：{', '.join('`%s`' % a for a in pareto_raw)}",
              f"- 越 {band} 带才算更好：{', '.join('`%s`' % a for a in pareto_band)}",
              f"- 原始序被支配点（含支配者）：{json.dumps(dominated_by_raw, ensure_ascii=False)}",
              f"- 越带序被支配点（含支配者）：{json.dumps(dominated_by, ensure_ascii=False)}", "",
              "## 表 5 · fp8 同价对手（质量未测）", "",
              f"- fp8 解析：{fp8_row['deployment_bytes_per_position']:g} B/位置（×{fp8_row['ratio_vs_fp16']:.2f} vs fp16），"
              f"比 int8+块标尺省 {fp8_row['vs_int8_block']['saving_vs_int8_block_bytes']:g} B/位置；"
              f"`quality_measured = {fp8_row['quality_measured']}`。", ""]
    (HERE / "table.md").write_text("\n".join(lines) + "\n")

    log.append(f"输出：results.json（{len(rows)} 臂）、table.md（{len(same_byte_groups)} 个同字节组）")
    log.append(f"结论摘要：G1 {'PASS' if gate_g1 else 'FAIL'}；同字节跨家族组 = {len(same_byte_groups)}；"
               f"其中 bpc 越带 = {sum(1 for g in same_byte_groups if g['bpc_spread_exceeds_band'])}；"
               f"参数量混杂已拆 = {param_table['analytic_matches_measured']}")
    (HERE / "run_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))
    return 0 if (gate_g1 and gate_g2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
