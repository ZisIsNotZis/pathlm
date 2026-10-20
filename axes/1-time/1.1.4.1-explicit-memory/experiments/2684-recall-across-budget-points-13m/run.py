#!/usr/bin/env python3
"""2684-recall-across-budget-points-13m · 评测入口（纯推理，0 训练 token）。

契约：同目录 `README.md`（§4 判据线**预登记、写下后不得改**）；配置 `config.json`。

做三件事：

  (a) 十三臂逐批跑 needle v3 真实文本短语针（`plant=True` 植入 / `plant=False` 配对地板，
      同一 generator 序列 ⇒ 针位逐一相同），**每臂用自己训练时的可见性**
      `window_anchor_keep_mask(T, w, 0)` ⇒ 桶边界也是各臂自己的 w；
  (b) 仪器复现门（判据①）：六个 `2661` 复用臂在窗口 16 可见性下重算同索引硬子集
      （`needle_stats.hard_index_sets` 的交集），与 `2661/results.json::needle_leg` 逐位比对；
  (c) 固定远端集（针距离 > 128，网格最宽的纯窗臂也看不见）∩ 十三臂 control 全答不出 ⇒
      在该集上逐臂算植入召回（Wilson CI）与 `carrypl16_s0 − 该臂` 的两比例差（判据②③⑤）。

仪器只读复用 `2661` 的 `run.load_arm` / `run.resolve` / `run.sha256_file` 与 `needle_stats`；
不复制实现，两文件 sha256 记进 `results.json::code_hashes`。

用法：
  python3 run.py            # 全电池（纯推理，共享卡 ~7 min）
  python3 run.py --smoke    # 冒烟：少批（结果写 smoke/，不作判据）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
INSTRUMENT = os.path.join(
    ROOT, "axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/"
          "2661-implicit-vs-explicit-recall-13m")
for _p in (ROOT, INSTRUMENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pathlm.data import load_enwik8_full                          # noqa: E402
from pathlm.depth_ar import window_anchor_keep_mask                # noqa: E402
from tools.evalkit.retention import needle_recall, phrase_needle_batch  # noqa: E402
import needle_stats as ns                                          # noqa: E402
import run as instrument                                           # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or (os.path.join(HERE, "smoke") if args.smoke else HERE)
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(args.config))
    if args.smoke:
        cfg["needle"]["n_batches"] = 2
        cfg["needle"]["batch_size"] = 8
    dev = cfg["device"]
    if dev.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("config 要求 cuda，但本机不可用")
    T = cfg["model"]["seq_len"]
    ncfg = {k: cfg["needle"][k] for k in
            ("n_needles", "phrase_len", "anchors", "anchor_frac")}
    L = cfg["needle"]["phrase_len"]
    roles = cfg["arm_roles"]
    log: list[dict] = []
    t0 = time.time()
    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def note(msg: str) -> None:
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载 enwik8 尾段做评测")
    _, eval_arr, vocab = load_enwik8_full(os.path.join(ROOT, ".tmp/enwik8"),
                                          os.path.join(ROOT, "data/enwik8_full.npz"))
    assert vocab == cfg["data"]["vocab_size"], \
        f"vocab {vocab} != {cfg['data']['vocab_size']}"
    n_real, mask_tok = cfg["data"]["n_real_tokens"], cfg["data"]["mask_token"]

    note("加载十三臂（逐臂 sha256 断言）+ 逐臂可见性掩码")
    arms, ck, keeps = {}, {}, {}
    for spec in cfg["arms"]:
        model, info = instrument.load_arm(spec, cfg["model"], vocab, dev)
        arms[spec["name"]], ck[spec["name"]] = model, info
        keeps[spec["name"]] = window_anchor_keep_mask(
            T, int(spec["window"]), int(spec.get("anchors", 0)), device=dev)
        note(f"  {spec['name']}: sha={info['sha256'][:12]}… w={spec['window']} "
             f"rule={spec['rule']} per_layer={info['carry_per_layer']} "
             f"state_floats={info['state_floats_actual']} params={info['n_params']}")
    none_names = [s["name"] for s in cfg["arms"] if s["rule"] in (None, "none")]
    carry_names = [s["name"] for s in cfg["arms"] if s["rule"] not in (None, "none")]
    assert len({ck[n]["n_params"] for n in none_names}) == 1, "无通道臂参数量不一致"
    assert len({ck[n]["n_params"] for n in carry_names}) == 1, "通道臂参数量不一致"

    note(f"针腿：{cfg['needle']['n_batches']}×{cfg['needle']['batch_size']}，"
         f"{cfg['needle']['n_needles']} 针/行，seed={cfg['needle']['seed']}（与 2661 同一序列）")
    gen = torch.Generator().manual_seed(cfg["needle"]["seed"])
    records = {f"{n}|{k}": [] for n in arms for k in ("planted", "control")}
    n_needles_total = 0
    for bi in range(cfg["needle"]["n_batches"]):
        state = gen.get_state()
        x, meta = phrase_needle_batch(cfg["needle"]["batch_size"], T, eval_arr, n_real,
                                      mask_tok, gen, plant=True, **ncfg)
        gen.set_state(state)
        x0, meta0 = phrase_needle_batch(cfg["needle"]["batch_size"], T, eval_arr, n_real,
                                        mask_tok, gen, plant=False, **ncfg)
        if meta0 != meta:
            raise SystemExit("配对控制臂的针位必须与主臂逐一相同")
        n_needles_total += sum(len(m) for m in meta)
        for name, model in arms.items():
            for kind, xx in (("planted", x), ("control", x0)):
                rec = needle_recall(model, xx, meta, keeps[name],
                                    anchors=cfg["needle"]["bucket_anchors"],
                                    phrase_len=L, device=dev)
                records[f"{name}|{kind}"].extend(rec["per_needle"])
        if bi % 8 == 0:
            note(f"  needle batch {bi + 1}/{cfg['needle']['n_batches']}")

    def beyond(name: str, kind: str) -> list[dict]:
        return [r for r in records[f"{name}|{kind}"] if r["bucket"] == "beyond"]

    # ---------------- ① 仪器复现门：与 2661 逐位比对 ----------------
    gate_arms = list(roles["equality_gate_arms"])
    bplanted_gate = {n: beyond(n, "planted") for n in gate_arms}
    bcontrol_gate = {n: beyond(n, "control") for n in gate_arms}
    for n in gate_arms:
        assert len(bplanted_gate[n]) == len(bcontrol_gate[n]), f"{n}: 桶内配对长度不一致"
    idx_gate = ns.hard_index_sets(bcontrol_gate)
    eg = cfg["equality_gate"]
    gate = {"n_indexed": idx_gate["n_indexed"],
            "n_intersection": idx_gate["n_intersection"],
            "n_intersection_2661": eg["n_intersection"],
            "per_arm": {}}
    for n in gate_arms:
        got = ns.recall_on_indices(bplanted_gate[n], idx_gate["intersection"])
        want = eg["hits"][n]
        gate["per_arm"][n] = {"hits": got["hits"], "hits_2661": want,
                              "recall": got["recall"],
                              "wilson_ci": got["wilson_ci"],
                              "equal": bool(got["hits"] == want)}
    gate["all_equal"] = bool(gate["n_intersection"] == eg["n_intersection"]
                             and all(v["equal"] for v in gate["per_arm"].values()))
    note(f"判据① 仪器复现门：n={gate['n_intersection']}（2661 {eg['n_intersection']}）；"
         f"hits 逐臂相等={gate['all_equal']}")

    # ---------------- ② 固定远端集：距离 > 128 且十三臂 control 全答不出 ----------------
    # 对齐口径：每臂的 needle 记录按「批 → 行 → 针」同一顺序追加（同一 generator 序列、同一 meta）
    # ⇒ **列表下标跨臂对齐**。桶（window / beyond）是逐臂每记录自己的属性，不参与这里取交。
    d_min = int(cfg["fixed_set"]["distance_min_exclusive"])
    primary = roles["primary_carry"]
    lead = records[f"{primary}|planted"]
    far_pos = [i for i, r in enumerate(lead) if r["dist"] > d_min]
    miss_all = {i for i, r in enumerate(records[f"{primary}|control"]) if r["hit"] == 0}
    for name in arms:
        miss_all &= {i for i, r in enumerate(records[f"{name}|control"]) if r["hit"] == 0}
    far_idx = sorted(set(far_pos) & miss_all)
    note(f"固定远端集：距离>{d_min} 的针 {len(far_pos)} 个；再交十三臂 control 全答不出 "
         f"⇒ {len(far_idx)} 个")
    if not far_idx:
        raise SystemExit("固定远端集为空：判据②③⑤ 无法判（如实报，不插值）")

    armed = {}
    for name in arms:
        planted_all, control_all = records[f"{name}|planted"], records[f"{name}|control"]
        armed[name] = {
            "n_records": len(planted_all),
            "fixed_set": ns.recall_on_indices(planted_all, far_idx),
            "fixed_set_floor": ns.recall_on_indices(control_all, far_idx),
            "by_bucket": {
                b: {"n": sum(1 for r in planted_all if r["bucket"] == b),
                    "recall": (round(sum(r["hit"] for r in planted_all
                                         if r["bucket"] == b)
                                     / max(1, sum(1 for r in planted_all if r["bucket"] == b)), 6))}
                for b in sorted({r["bucket"] for r in planted_all})},
        }
    contrasts = {}
    for name in roles["window_grid"]:
        e, i = armed[primary]["fixed_set"], armed[name]["fixed_set"]
        # `beyond_above_floor(第一组 hits, n, 第二组 hits, n)` 返回的 diff = 第一组 − 第二组
        # （函数原名来自「植入 − 地板」，这里用它当「carry − 纯窗」的两比例差，参数位意义写在下面）
        contrasts[f"{primary}_minus_{name}"] = {
            "param_positions": "第一组 = carrypl16_s0，第二组 = 该纯窗臂；delta_pp 正 = carry 更高",
            "window": next(s["window"] for s in cfg["arms"] if s["name"] == name),
            **ns.beyond_above_floor(e["hits"], e["n"], i["hits"], i["n"]),
        }
    degraded = {}
    for name in roles["degraded_carry"]:
        d = armed[name]["fixed_set"]
        degraded[name] = {
            "fixed_set": d,
            "param_positions": "第一组 = carrypl16_s0（完备），第二组 = 该退化臂；delta_pp 正 = 完备更好",
            "vs_primary": ns.beyond_above_floor(
                armed[primary]["fixed_set"]["hits"], armed[primary]["fixed_set"]["n"],
                d["hits"], d["n"]),
        }
    # 判据④：纯窗臂召回随 w 的结构单调性（自检，不作判据）
    curve = [(next(s["window"] for s in cfg["arms"] if s["name"] == n),
              armed[n]["fixed_set"]["recall"]) for n in roles["window_grid"]]
    curve_sorted = sorted((w, r) for w, r in curve if r is not None)

    payload = {
        "experiment": cfg["experiment"],
        "contract": cfg["contract"],
        "config": cfg,
        "budget": cfg["budget"],
        "checkpoints": ck,
        "arm_roles": roles,
        "selfchecks": {
            "n_params_by_role": {"window": ck[none_names[0]]["n_params"],
                                 "carry": ck[carry_names[0]]["n_params"]},
            "n_needles_total": n_needles_total,
            "pairing_asserted": True,
        },
        "equality_gate": gate,
        "fixed_set": {
            "distance_min_exclusive": d_min,
            "n_far_positions": len(far_pos),
            "n_indexed": len(far_idx),
            "definition": ("针距离 > 128（网格最宽的纯窗臂 win128 也看不见）∩ 十三臂 control 全部答不出"
                           "（该子集上地板结构性为 0，仍逐臂按配对控制报地板）"),
        },
        "per_arm": armed,
        "delta_directions": {
            "delta_pp": "arm_minus_comparator",
        },
        "delta_direction_note": (
            "本文件里所有 `delta_pp` / `diff_ci_95` 都是**两比例差**：`contrasts[*]` 是 "
            "carrypl16_s0 − 该纯窗臂（第一组 carry 当「臂」、第二组纯窗户当「对照」⇒ "
            "arm_minus_comparator，正 = carry 召回更高）；`degraded_vs_complete[*].vs_primary` "
            "同向（carrypl16_s0 − 退化臂）。bpc 类 Δ 本文件一个都没有（质量轴的家是 "
            "2638 / 2639 / 2681）。"),
        "contrasts": contrasts,
        "degraded_vs_complete": degraded,
        "window_recall_curve": [{"window": w, "recall": r} for w, r in curve_sorted],
        "code_hashes": {f: sha256_file(os.path.join(INSTRUMENT, f))
                        for f in cfg["instrument"]["files"]}
                       | {"run.py": sha256_file(os.path.abspath(__file__))},
        "cost": {
            "wall_minutes_total": None,
            "eval_peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                                     if dev.startswith("cuda") else None),
            "n_training_tokens": 0,
            "lock_note": ("纯推理评测：整程 <10 min ⇒ 按 harness/SCHEMA.md §20.1 整程持 "
                          "/tmp/pathlm_gpu.lock；取/放时间戳见 run_log.jsonl 的首末条"),
            "note": "墙钟与显存峰值只作过程指标，不进判据",
        },
    }
    payload["cost"]["wall_minutes_total"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(os.path.join(out_dir, "run_log.jsonl"), "w") as f:
        for row in log:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    note(f"落盘：{out_dir}/results.json + run_log.jsonl")
    print(f"\n判据① all_equal={gate['all_equal']}（n={gate['n_intersection']}）")
    print(f"固定远端集 n={len(far_idx)}；{primary} 召回 "
          f"{armed[primary]['fixed_set']['recall']} "
          f"CI={armed[primary]['fixed_set']['wilson_ci']}")
    for k, v in contrasts.items():
        print(f"  {k}: Δ={v['delta_pp']:+}pp CI={v['diff_ci_95_pp']} passes={v['passes']}")


if __name__ == "__main__":
    main()
