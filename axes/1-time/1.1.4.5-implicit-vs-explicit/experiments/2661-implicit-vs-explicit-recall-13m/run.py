#!/usr/bin/env python3
"""2661-implicit-vs-explicit-recall-13m · 评测入口（纯推理）：implicit（纯窗口 16）vs
explicit（完备机制：逐层独立状态 + 内容写门）正面对撞。

契约：同目录 `README.md`（§4 判据线**预登记、写下后未改**）。本入口是一份自
`2634-carry-realtext-13m/run.py` fork 的副本：**探针/仪器/掩码口径一字不改**，只把
**臂集**换成 `config.json` 里的 6 臂、把臂配置从 2639 抄成「逐臂 E1 开关」形式。

做五条腿：

  (a) 毁源式窗外源敏感度探针 `realtext_probe.far_source_probe`（kwargs 一律经
      `rp.probe_kwargs(cfg)` 派生，`rf` 必填）——逐臂 `Δbpc_far` + 逐簇配对 bootstrap
      CI95 + `max_abs_logit_diff_on_eval_rows`；
  (b) needle v3 真实文本短语针：`beyond` 桶 vs **配对无针地板**（整体）+
      **同索引硬子集**（六臂 `plant=False` 全答不出）上的差值 + CI（跨臂直接对撞）；
  (c) clean bpc（窗口 16 掩码，96×16）与 2638/2639 记录值逐位复现；
  (d) 诊断（不作判据）：`channel_off` 消融抓力 + `retention.prefix_keep_mask` /
      `recent_keep_mask` 的 damage_bpc / safe_radius 损伤曲线；
  (e) 因果性（必须不变）：只改后半段 token ⇒ 前半段 logits 逐位不变 + 掩码非退化。

用法：
  python3 run.py            # 全电池（纯推理，共享卡 ~10 min）
  python3 run.py --smoke    # 冒烟：少批
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pathlm.data import load_enwik8_full                          # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             carry_probe, carry_read_mass_stats,
                             window_anchor_keep_mask)
from tools.evalkit.retention import (masked_bpc, masked_logits,     # noqa: E402
                                     needle_recall, needle_summary,
                                     phrase_needle_batch,
                                     prefix_keep_mask, recent_keep_mask,
                                     safe_radius)
from tools.evalkit.carry_channel import (loss_slope, param_equality,  # noqa: E402
                                         trunk_equality, wilson_ci)
import realtext_probe as rp                                        # noqa: E402
import needle_stats as ns                                          # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _strict(obj):
    """把非有限浮点（NaN/Inf）换成 None，保证落盘 JSON 严格可解析。

    来源：共享 runner 在 `carry_decay != 1.0` 时会把一个遥测标量写成 `NaN`
    （`2639/train/pl16d/results.json` 的 `carry_liveness.rel_dev_vs_other_rule`）；
    本文件把它转抄进 `train_meta[*].carry_liveness_training` ⇒ 若直接 `json.dump`
    会写出裸 `NaN`。本轴不改共享代码，只在自己的落盘边界做严格化。
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strict(v) for v in obj]
    return obj


def model_kwargs(spec: dict, mcfg: dict) -> dict:
    """逐臂 `DepthARConfig` 构造参数：窗口/锚点 + 写入规则 + E1 完备度开关。

    与 `2639/run.py` 同款（逐臂用自己的训练时实现细节 chunk + `carry_per_layer`
    / `carry_decay`），**不改共享默认**。
    """
    kw = dict(mcfg)
    kw["train_window"] = int(spec["window"])
    kw["train_anchors"] = int(spec.get("anchors", 0))
    kw["carry_rule"] = None if spec["rule"] in (None, "none") else spec["rule"]
    if kw["carry_rule"] is not None:
        kw["carry_chunk"] = int(spec.get("carry_chunk", 64))
        kw["carry_per_layer"] = bool(spec.get("carry_per_layer", False))
        kw["carry_decay"] = float(spec.get("carry_decay", 1.0))
    return kw


def load_arm(spec: dict, mcfg: dict, vocab: int, device: str):
    path = resolve(spec["path"])
    got = sha256_file(path)
    want = spec.get("sha256")
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{spec['path']}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"{spec['name']}: checkpoint sha256 mismatch:\n "
                         f"want {want}\n got  {got}")
    cfg = DepthARConfig(**model_kwargs(spec, mcfg))
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # 状态字节：单共享槽 = carry_heads × hd²；逐层 = n_layers × 上面那个。
    state_floats = 0
    if model.cfg.carry_rule is not None:
        per_layer_state = (model.carry.n_heads * model.carry.head_dim ** 2)
        state_floats = per_layer_state * (model.cfg.n_layers
                                          if model.cfg.carry_per_layer else 1)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "kind": spec["kind"], "window": spec["window"],
                   "anchors": spec.get("anchors", 0),
                   "rule": spec["rule"], "carry_rule": cfg.carry_rule,
                   "seed": spec["seed"],
                   "carry_chunk": spec.get("carry_chunk"),
                   "carry_per_layer": bool(spec.get("carry_per_layer", False)),
                   "carry_decay": float(spec.get("carry_decay", 1.0)),
                   "state_floats_actual": state_floats,
                   "block_class": type(model.blocks[0]).__name__,
                   "n_params": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def channel_off_grip(model, tokens: torch.Tensor, keep_mask=None) -> dict:
    """`channel_off` 的**抓力**直接读数：置零每层读出增益前后最后一层 logits 的最大差。

    `> 0` ⇔ `realtext_probe.channel_off` 真的改了前向（读出增益被置零）；
    `== 0` ⇔ 消融没接上（死消融）。无通道臂上应为 `applicable=False`。
    """
    if getattr(model.cfg, "carry_rule", None) is None:
        return {"applicable": False, "max_abs_logit_diff_off_vs_on": None,
                "grip": False,
                "note": "无通道臂：channel_off 是空操作"}
    was_training = model.training
    model.eval()
    try:
        on = masked_logits(model, tokens, keep_mask).float()
        with rp.channel_off(model):
            off = masked_logits(model, tokens, keep_mask).float()
    finally:
        model.train(was_training)
    d = float((on - off).abs().max())
    return {"applicable": True, "max_abs_logit_diff_off_vs_on": round(d, 6),
            "grip": bool(d > 0),
            "note": ("置零每层 carry_scale（+ 共享 gate_w）前后最后一层 logits 最大差；"
                     "> 0 ⇔ 消融真的把每层读出增益置零（E1 逐层模式覆盖检查）")}


@torch.no_grad()
def causality_probe(model, x: torch.Tensor, keep_mask=None) -> dict:
    """**严格因果性探针（必须不变项，腿 e）**：只改后半段 token ⇒ 前半段 logits 逐位不变。

    逐层状态是因果递推（位置 t 的读出只用 S_t，而 S_t 只含 ≤ t 的写入）⇒ 必须逐位不变；
    后半段必须真的变（否则探针没抓力）。
    """
    B, T = x.shape
    half = T // 2
    g = torch.Generator().manual_seed(7)
    x2 = x.clone()
    x2[:, half:] = torch.randint(1, model.vocab_size, (B, T - half), generator=g)
    was_training = model.training
    model.eval()
    try:
        a = masked_logits(model, x, keep_mask).float()
        b = masked_logits(model, x2, keep_mask).float()
    finally:
        model.train(was_training)
    d = (a[:, :half] - b[:, :half]).abs().max()
    return {"perturbed_from": half,
            "max_abs_logit_diff_prefix": round(float(d), 9),
            "causal_no_future_leak": bool(float(d) == 0.0),
            "changed_suffix_max_abs_logit_diff": round(
                float((a[:, half:] - b[:, half:]).abs().max()), 6),
            "note": ("只改后半段 token ⇒ 前半段 logits 逐位不变 ⇔ 严格因果（无未来泄露）；"
                     "后半段必须真的变（否则探针没抓力）")}


def _restrict(records: list[dict], idx: list[int]) -> list[dict]:
    return [records[i] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke", "eval") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)
    cfg = json.load(open(args.config))
    if args.smoke:
        for k in ("bpc", "probe"):
            cfg["eval"][k]["n_batches"] = 1
            cfg["eval"][k]["batch_size"] = 8
        cfg["needle"]["n_batches"] = 1
        cfg["needle"]["batch_size"] = 8
        cfg["retention"]["n_batches"] = 1
        cfg["retention"]["batch_size"] = 8
    dev = cfg["device"]
    T = cfg["model"]["seq_len"]
    window, anchors = cfg["visibility"]["train_window"], cfg["visibility"]["train_anchors"]
    n_real, mask_tok = cfg["data"]["n_real_tokens"], cfg["data"]["mask_token"]
    vcfg = cfg["needle"]
    roles = cfg["arm_roles"]
    log: list[dict] = []
    t0 = time.time()
    use_cuda = dev.startswith("cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载 enwik8 尾段做评测")
    _, eval_arr, vocab = load_enwik8_full(os.path.join(ROOT, ".tmp/enwik8"),
                                          os.path.join(ROOT, "data/enwik8_full.npz"))
    assert vocab == cfg["data"]["vocab_size"], f"vocab {vocab} != {cfg['data']['vocab_size']}"

    note("加载六臂（sha256 断言）")
    arms, ck = {}, {}
    for spec in cfg["arms"]:
        arms[spec["name"]], ck[spec["name"]] = load_arm(
            spec, cfg["model"], vocab, dev)
        note(f"  {spec['name']}: {ck[spec['name']]['sha256'][:12]}… kind={spec['kind']} "
             f"rule={spec['rule']} per_layer={ck[spec['name']]['carry_per_layer']} "
             f"chunk={ck[spec['name']]['carry_chunk']} "
             f"state_floats={ck[spec['name']]['state_floats_actual']} "
             f"params={ck[spec['name']]['n_params']}")
    results = {"config": cfg, "checkpoints": ck,
               "visibility": {"train_window": window, "train_anchors": anchors},
               "arm_roles": roles}

    # ---------------- ① 仪器自检（含 E1 逐层通路接线） ----------------
    note("自检①：窗口掩码非退化 + none/delta(共享+逐层) 主干逐位 + 单边零起步 + 参数量账")
    keep = window_anchor_keep_mask(T, window, anchors, device=dev)
    n_visible = int(keep.sum().item())
    expected_visible = sum(min(i + 1, window) for i in range(T))
    assert n_visible == expected_visible, f"可见对数 {n_visible} != {expected_visible}"
    assert n_visible < T * (T + 1) // 2, "窗口掩码在 T=256 上是恒等算子（退化点）"
    g = torch.Generator().manual_seed(7)
    x_probe = torch.randint(0, vocab, (2, T), generator=g).to(dev)

    def _build(seed, extra):
        torch.manual_seed(seed)
        m = DepthARModel(DepthARConfig(**{**cfg["model"], **extra}), vocab)
        m.eval()
        return m

    trunk, id_start, built_params = {}, {}, {}
    for seed in cfg["seeds"]:
        none_m = _build(seed, {"carry_rule": None})
        shared_m = _build(seed, {"carry_rule": "delta", "carry_chunk": 64,
                                 "carry_per_layer": False, "carry_decay": 1.0})
        trunk[f"shared_seed{seed}"] = trunk_equality(shared_m, none_m)
        with torch.no_grad():
            l0 = none_m(x_probe.cpu())[1]["depth_logits"][-1]
            id_start[f"shared_seed{seed}"] = {
                "delta_matches_none_bitwise": torch.equal(
                    l0, shared_m(x_probe.cpu())[1]["depth_logits"][-1])}
        built_params[f"none_seed{seed}"] = sum(p.numel() for p in none_m.parameters())
        built_params[f"shared_seed{seed}"] = sum(p.numel() for p in shared_m.parameters())
    pl_m = _build(0, {"carry_rule": "delta", "carry_chunk": 256,
                      "carry_per_layer": True, "carry_decay": 1.0})
    none0 = _build(0, {"carry_rule": None})
    trunk["perlayer_seed0"] = trunk_equality(pl_m, none0)
    with torch.no_grad():
        l0 = none0(x_probe.cpu())[1]["depth_logits"][-1]
        id_start["perlayer_seed0"] = {
            "delta_matches_none_bitwise": torch.equal(
                l0, pl_m(x_probe.cpu())[1]["depth_logits"][-1])}
    built_params["perlayer_seed0"] = sum(p.numel() for p in pl_m.parameters())
    # 主干逐位 + 单边零起步：**硬断言**
    for k, v in trunk.items():
        assert v["trunk_bitwise_equal"], f"{k}: 主干参数不逐位相等"
    for k, v in id_start.items():
        assert v["delta_matches_none_bitwise"], f"{k}: delta 单边零起步不逐位等于无通道"
    none_names = [s["name"] for s in cfg["arms"] if s["rule"] in (None, "none")]
    carry_names = [s["name"] for s in cfg["arms"] if s["rule"] not in (None, "none")]
    assert len({ck[n]["n_params"] for n in none_names}) == 1, "无通道臂参数量不一致"
    assert len({ck[n]["n_params"] for n in carry_names}) == 1, "通道臂参数量不一致"
    results["selfchecks"] = {
        "window_mask_visible_pairs": n_visible,
        "window_mask_expected_pairs": expected_visible,
        "window_mask_nondegenerate": True,
        "trunk_bitwise_equal_carry_vs_none": trunk,
        "identity_start_logits_bitwise": id_start,
        "built_params": built_params,
        "loaded_n_params": {n: ck[n]["n_params"] for n in ck},
    }
    note(f"  可见对数 {n_visible}；主干逐位={all(v['trunk_bitwise_equal'] for v in trunk.values())}"
         f"；单边零起步={all(v['delta_matches_none_bitwise'] for v in id_start.values())}"
         f"；参数 纯窗={ck[none_names[0]]['n_params']} 通道={ck[carry_names[0]]['n_params']}")

    # ---------------- ② 质量腿（腿 c）+ 通道消融抓力 ----------------
    eb, ep = cfg["eval"]["bpc"], cfg["eval"]["probe"]
    note(f"质量腿：bpc_win16（窗口 {window}/{anchors}，{eb['n_batches']}×{eb['batch_size']}，"
         f"~{eb['n_batches']*eb['batch_size']*(T-1)} token）+ 通道消融抓力")
    quality = {}
    per_window_bpc: dict[str, tuple[list[float], list[int]]] = {}
    for name, model in arms.items():
        row = {}
        bpc_val, bits_w, n_w = rp.win16_bpc(model, eval_arr, keep, **{
            k: eb[k] for k in ("n_batches", "batch_size", "window_seed")},
            device=dev, per_window=True)
        per_window_bpc[name] = (bits_w, n_w)
        row["bpc_win16"] = round(bpc_val, 6)
        with rp.channel_off(model):
            row["bpc_win16_channel_off"] = round(rp.win16_bpc(model, eval_arr, keep, **{
                k: eb[k] for k in ("n_batches", "batch_size", "window_seed")},
                device=dev), 6)
        row["channel_off_grip"] = channel_off_grip(model, x_probe, keep)
        quality[name] = row
        note(f"  {name}: bpc_win16={row['bpc_win16']} "
             f"通道消融={row['bpc_win16_channel_off']} "
             f"抓力={row['channel_off_grip']['grip']}")

    # 判据⑤：与 2638/2639 记录值逐位复现
    rep = cfg["reproduction"]["bpc_win16"]
    reproduction = {"checked_at_6dp": {}, "all_equal": True, "differences": {},
                    "smoke": bool(args.smoke),
                    "note": ("smoke 模式下批数极小 ⇒ 该比对无意义（不作判据）；"
                             "全电池（非 --smoke）下才硬断言逐位一致")}
    for name in quality:
        if name not in rep:
            continue
        want = rep[name]
        got = quality[name]["bpc_win16"]
        eq = abs(got - want) < 5e-7
        reproduction["checked_at_6dp"][name] = {"expected": want, "got": got, "equal": eq}
        if not eq:
            reproduction["all_equal"] = False
            reproduction["differences"][name] = round(got - want, 6)
        note(f"  复现 {name}: 期望 {want} 实测 {got} ⇒ {'一致' if eq else '差异 %+0.6f' % (got-want)}")
    results["quality_leg"] = quality
    results["reproduction_check"] = reproduction
    if not args.smoke and not reproduction["all_equal"]:
        raise SystemExit(f"判据⑤失败：clean bpc 与记录值不逐位一致：{reproduction['differences']}")

    # ---------------- ③ 腿 (a)：毁源式窗外源敏感度探针 ----------------
    pkw = rp.probe_kwargs(cfg)
    assert "rf" in pkw and pkw["rf"] == cfg["model"]["n_layers"] * (window - 1)
    note(f"腿(a)：Δbpc_far 探针（毁前缀 [0,{pkw['corrupt_hi']})，被评行 "
         f"[{pkw['eval_lo']},{pkw['eval_hi']})，rf={pkw['rf']}，"
         f"{pkw['n_batches']}×{pkw['batch_size']}）")
    probe = {}
    for name, model in arms.items():
        row = rp.far_source_probe(model, eval_arr, keep, **pkw, device=dev)
        with rp.channel_off(model):
            row["with_channel_off"] = rp.far_source_probe(model, eval_arr, keep, **pkw,
                                                          device=dev)
        probe[name] = row
        note(f"  {name}: Δbpc_far={row['delta_bpc_far']:+.6f} "
             f"CI{row['delta_ci95_clustered']} clean={row['ce_far_clean']:.4f} "
             f"max|Δlogit|被评行={row['max_abs_logit_diff_on_eval_rows']:.3e} | "
             f"通道消融后 Δ={row['with_channel_off']['delta_bpc_far']:+.6f}")
    results["far_source_probe"] = probe
    # 仪器一致性旁证：carry16_s0 的 Δbpc_far 应与 2634 逐位一致（同 corrupt_seed）
    if "carry16_s0" in probe:
        probe_rep = {"carry16_s0": {
            "expected": cfg["reproduction"]["probe_delta_bpc_far"]["carry16_s0"],
            "got": probe["carry16_s0"]["delta_bpc_far"]}}
        probe_rep["carry16_s0"]["equal"] = abs(
            probe_rep["carry16_s0"]["got"] - probe_rep["carry16_s0"]["expected"]) < 5e-7
        results["reproduction_check"]["probe_delta_bpc_far"] = probe_rep

    # ---------------- ④ 腿 (b)：needle v3 ----------------
    ncfg = dict(n_needles=vcfg["n_needles"], phrase_len=vcfg["phrase_len"],
                anchors=vcfg["anchors"], anchor_frac=vcfg["anchor_frac"])
    L = vcfg["phrase_len"]
    note(f"腿(b)：真实文本短语针（needle v3，{vcfg['n_batches']}×{vcfg['batch_size']}，"
         f"{vcfg['n_needles']} 针/行，seed={vcfg['seed']}）")
    gen = torch.Generator().manual_seed(vcfg["seed"])
    records = {f"{n}|{kind}": [] for n in arms for kind in ("planted", "control")}
    n_needles_total = 0
    for bi in range(vcfg["n_batches"]):
        state = gen.get_state()
        x, meta = phrase_needle_batch(vcfg["batch_size"], T, eval_arr, n_real,
                                      mask_tok, gen, **ncfg, plant=True)
        gen.set_state(state)
        x0, meta0 = phrase_needle_batch(vcfg["batch_size"], T, eval_arr, n_real,
                                        mask_tok, gen, **ncfg, plant=False)
        if meta0 != meta:
            raise SystemExit("配对控制臂的针位必须与主臂逐一相同")
        n_needles_total += sum(len(m) for m in meta)
        for name, model in arms.items():
            for kind, xx in (("planted", x), ("control", x0)):
                rec = needle_recall(model, xx, meta, keep,
                                    anchors=vcfg["bucket_anchors"], phrase_len=L, device=dev)
                records[f"{name}|{kind}"].extend(rec["per_needle"])
        log.append({"t": round(time.time() - t0, 2), "msg": f"needle batch {bi}"})
    needle = {"conditions": {}, "n_needles_total": n_needles_total,
              "seed": vcfg["seed"], "n_batches": vcfg["n_batches"],
              "bucket_rule": ("anchors=0 ⇒ 桶只有 window（短语全可见）与 beyond（有位置不可见）；"
                              "本实验锚点 0，窗 16 ⇒ beyond 占绝大多数")}
    bands = [(0, 32), (32, 64), (64, 128), (128, 256)]

    def band_of(d):
        for lo, hi in bands:
            if lo <= d < hi:
                return f"[{lo},{hi})"
        return f"[{bands[-1][1]},)"

    for key, recs in records.items():
        s = needle_summary(recs)
        by_band: dict[str, list[int]] = {}
        for r in recs:
            if r["bucket"] == "beyond":
                by_band.setdefault(band_of(r["dist"]), []).append(r["hit"])
        needle["conditions"][key] = {
            "recall": s["recall"], "n": s["n"], "buckets": s["buckets"],
            "beyond_by_distance_band": {b: {"recall": round(sum(v) / len(v), 6),
                                            "n": len(v)} for b, v in sorted(by_band.items())},
        }
    # ---- beyond 桶内配对读数 + **同索引**硬子集 ----
    bplanted = {n: [r for r in records[f"{n}|planted"] if r["bucket"] == "beyond"] for n in arms}
    bcontrol = {n: [r for r in records[f"{n}|control"] if r["bucket"] == "beyond"] for n in arms}
    for n in arms:
        assert len(bplanted[n]) == len(bcontrol[n]), "桶内配对长度不一致"
    idx_sets = ns.hard_index_sets(bcontrol)
    paired = {}
    for name in arms:
        p, c = bplanted[name], bcontrol[name]
        kp, kc, n = sum(r["hit"] for r in p), sum(r["hit"] for r in c), len(p)
        row = {"n": n, "planted_hits": kp, "control_hits": kc,
               "planted_recall": round(kp / n, 6) if n else None,
               "control_recall": round(kc / n, 6) if n else None,
               "delta_pp": round(100 * (kp - kc) / n, 3) if n else None,
               "hard_subset_same_index": {
                   "intersection": {**ns.recall_on_indices(p, idx_sets["intersection"]),
                                    "definition": "六臂 control 全答不出（同索引）"},
                   "union": {**ns.recall_on_indices(p, idx_sets["union"]),
                             "definition": "至少一臂 control 答不出（同索引）"}}}
        paired[name] = row
    needle["paired_beyond_vs_floor"] = paired
    needle["hard_index_sets"] = {"n_indexed": idx_sets["n_indexed"],
                                 "per_arm_control_miss": idx_sets["per_arm_control_miss"],
                                 "n_intersection": idx_sets["n_intersection"],
                                 "n_union": idx_sets["n_union"],
                                 "intersection": idx_sets["intersection"],
                                 "union": idx_sets["union"],
                                 "definition": idx_sets["definition"]}
    # **同索引**上的跨臂比较（explicit − implicit，正 = explicit 更好）
    hard_cmp = {}
    explicit_all = list(roles["explicit_complete"]) + list(roles["explicit_degraded"])
    for set_name in ("intersection", "union"):
        idx = idx_sets[set_name]
        per_arm = {n: ns.recall_on_indices(bplanted[n], idx) for n in arms}
        cmp_rows = {}
        for ename in explicit_all:
            for iname in roles["implicit"]:
                e, i = per_arm[ename], per_arm[iname]
                cmp_rows[f"{ename}_minus_{iname}"] = ns.beyond_above_floor(
                    e["hits"], e["n"], i["hits"], i["n"])
        hard_cmp[set_name] = {"per_arm_recall": per_arm, "explicit_minus_implicit": cmp_rows}
    needle["hard_subset_same_index_cross_arm"] = hard_cmp
    # 主读数：primary explicit 在硬同索引子集上 vs 配对地板（该子集地板恒为 0）
    inter_idx = idx_sets["intersection"]
    pname = roles["primary_explicit"]
    p_hits = ns.recall_on_indices(bplanted[pname], inter_idx)
    primary = {
        "arm": pname, "subset": "intersection",
        "n": p_hits["n"], "hits": p_hits["hits"], "recall": p_hits["recall"],
        "wilson_ci": p_hits["wilson_ci"],
        "vs_floor": (ns.beyond_above_floor(p_hits["hits"], p_hits["n"], 0, p_hits["n"])
                     if p_hits["n"] else None),
        "vs_primary_implicit": hard_cmp["intersection"]["explicit_minus_implicit"].get(
            f"{pname}_minus_{roles['primary_implicit']}"),
        "note": ("硬同索引子集 = 六臂 plant=False 全部答不出的针（该子集上地板恒为 0）；"
                 "vs_floor = 植入 − 地板两比例差（判据①）；vs_primary_implicit = 跨臂对撞"),
    }
    needle["primary_hard_subset"] = primary
    # 逐臂判定（判据②的负对照用整体 beyond 桶）
    intervention = tuple(n for n in arms if ck[n]["carry_rule"])
    needle["judgement"] = ns.judge(
        {n: {"hits": paired[n]["planted_hits"], "n": paired[n]["n"]} for n in arms},
        {n: {"hits": paired[n]["control_hits"], "n": paired[n]["n"]} for n in arms},
        intervention)
    needle["criterion_rule"] = (
        "判据①② = 逐臂：explicit（carrypl16_s0）的 beyond 植入召回 > 其配对地板、"
        "差值 95% CI 下界 > 0（硬同索引子集上）；implicit（win16_*）不越地板（负对照）")
    for name in arms:
        r = needle["judgement"]["per_arm"][name]
        hi = paired[name]["hard_subset_same_index"]["intersection"]
        note(f"  {name}: beyond 植入 {r['planted_recall']:.4f} vs 地板 {r['control_recall']:.4f} "
             f"⇒ Δ={r['delta_pp']}pp CI{r['diff_ci_95_pp']}pp passes={r['passes']}；"
             f"同索引交子集 n={hi['n']} 植入召回={hi['recall']}")
    note(f"  硬同索引子集 n={idx_sets['n_intersection']}；"
         f"{pname} 植入召回={p_hits['recall']} Wilson{p_hits['wilson_ci']}；"
         f"vs 地板 diff_ci={primary['vs_floor']['diff_ci_95'] if primary['vs_floor'] else None}")
    if primary["vs_primary_implicit"]:
        note(f"  {pname} − {roles['primary_implicit']}（同索引子集）："
             f"diff={primary['vs_primary_implicit']['delta_pp']}pp "
             f"CI{primary['vs_primary_implicit']['diff_ci_95_pp']}pp "
             f"passes={primary['vs_primary_implicit']['passes']}")
    results["needle_leg"] = needle

    # ---------------- ⑤ 腿 (e)：因果性 / 必须不变 ----------------
    note("腿(e)：严格因果性探针（只改后半段 token ⇒ 前半段 logits 逐位不变）")
    caus = {}
    for name, model in arms.items():
        caus[name] = causality_probe(model, x_probe, keep)
        note(f"  {name}: 前半段 max|Δlogit|={caus[name]['max_abs_logit_diff_prefix']:.3e} "
             f"因果={caus[name]['causal_no_future_leak']} "
             f"后半段抓力={caus[name]['changed_suffix_max_abs_logit_diff']}")
        assert caus[name]["changed_suffix_max_abs_logit_diff"] > 0, f"{name}: 因果探针无抓力"
    results["causality"] = caus

    # ---------------- ⑥ 腿 (d)：通道消融 + KV 损伤曲线（诊断，不作判据） ----------------
    rcfg = cfg["retention"]
    retention = {"enabled": bool(rcfg["enabled"]),
                 "config": {k: rcfg[k] for k in ("n_batches", "batch_size", "window_seed",
                                                  "recent_d", "prefix_drop")},
                 "caliber_note": ("显式状态固定大小、结构上不可截断 ⇒ prefix/recent 掩码只删"
                                  "attention 的 KV，carrypl16 臂每层状态仍完整携带前缀信息；"
                                  "damage_bpc = bpc_掩码 − bpc_clean；正 = 变差"),
                 "curves": {}}
    if rcfg["enabled"]:
        note(f"腿(d) 诊断：KV 损伤曲线（{rcfg['n_batches']}×{rcfg['batch_size']}，"
             f"recent_d={rcfg['recent_d']}，prefix_drop={rcfg['prefix_drop']}）")
        for name, model in arms.items():
            clean = quality[name]["bpc_win16"]
            recent_rows, prefix_rows = [], []
            for d in rcfg["recent_d"]:
                km = recent_keep_mask(T, d, device=dev)
                b = masked_bpc(model, eval_arr, km, n_batches=rcfg["n_batches"],
                               batch_size=rcfg["batch_size"],
                               window_seed=rcfg["window_seed"], device=dev)
                recent_rows.append({"d": d, "bpc": round(b, 6),
                                    "damage_bpc": round(b - clean, 6),
                                    "delta_bpc": round(clean - b, 6)})
            for N in rcfg["prefix_drop"]:
                km = prefix_keep_mask(T, N, device=dev)
                b = masked_bpc(model, eval_arr, km, n_batches=rcfg["n_batches"],
                               batch_size=rcfg["batch_size"],
                               window_seed=rcfg["window_seed"], device=dev)
                prefix_rows.append({"d": N, "bpc": round(b, 6),
                                    "damage_bpc": round(b - clean, 6),
                                    "delta_bpc": round(clean - b, 6)})
            retention["curves"][name] = {
                "bpc_clean_window16": clean,
                "recent": recent_rows, "prefix": prefix_rows,
                "safe_radius_recent": {
                    "sigma_cell": safe_radius(recent_rows, band=cfg["bands"]["sigma_cell"],
                                              key="d", delta="delta_bpc"),
                    "band_0.014": safe_radius(recent_rows, band=cfg["bands"]["cross_cell_band"],
                                              key="d", delta="delta_bpc")},
                "safe_radius_prefix": {
                    "sigma_cell": safe_radius(prefix_rows, band=cfg["bands"]["sigma_cell"],
                                              key="d", delta="delta_bpc"),
                    "band_0.014": safe_radius(prefix_rows, band=cfg["bands"]["cross_cell_band"],
                                              key="d", delta="delta_bpc")},
            }
            note(f"  {name}: recent damage=" +
                 ",".join(f"d{r['d']}:{r['damage_bpc']:+.4f}" for r in recent_rows) +
                 " | prefix damage=" +
                 ",".join(f"N{r['d']}:{r['damage_bpc']:+.4f}" for r in prefix_rows))
    results["retention"] = retention

    # ---------------- ⑦ 噪声带 + 跨臂效应 ----------------
    note("噪声带（σ_cell 同配置换种子极差）与跨臂配对效应")
    pair_spreads = {f"{a}|{b}": round(abs(quality[a]["bpc_win16"] - quality[b]["bpc_win16"]), 6)
                    for a, b in (("win16_s0", "win16_s1"), ("carry16_s0", "carry16_s1"))
                    if a in quality and b in quality}
    sigma_bpc = max(pair_spreads.values()) if pair_spreads else None
    paired_pairs = []
    for ename in explicit_all:
        for iname in roles["implicit"]:
            bits_i, n_i = per_window_bpc[iname]
            bits_e, _ = per_window_bpc[ename]
            ci = rp.paired_cluster_bootstrap_ci([a - b for a, b in zip(bits_i, bits_e)], n_i)
            paired_pairs.append({
                "explicit": ename, "implicit": iname,
                "bpc_implicit": quality[iname]["bpc_win16"],
                "bpc_explicit": quality[ename]["bpc_win16"],
                "delta_bpc": round(quality[iname]["bpc_win16"] - quality[ename]["bpc_win16"], 6),
                "paired_delta_bpc_ci95_bits": ci,
                "paired_note": ("配对 CI（同窗口序）：只含同批窗口的测量噪声，不含训练种子方差"
                                "（后者由 σ_cell 描述）；正 = explicit 更好")})
    effects = {"bpc_win16": {n: quality[n]["bpc_win16"] for n in arms},
               "explicit_minus_implicit": paired_pairs,
               "delta_bpc_far": {n: probe[n]["delta_bpc_far"] for n in arms}}
    results["sigma_cell"] = {"bpc_win16": sigma_bpc, "pair_spreads": pair_spreads,
                             "note": "同配置换种子的极差（2 样本 ⇒ 描述量，不是方差上界）"}
    results["effects"] = effects

    # ---------------- ⑧ 判据块 ----------------
    premise_ok = all(probe[n]["max_abs_logit_diff_on_eval_rows"] == 0.0
                     for n in roles["implicit"])
    c1 = bool(primary["vs_floor"] and primary["vs_floor"]["passes"]
              and primary["wilson_ci"][0] > 0)
    c1b = {f"{pname}_minus_{iname}":
           bool(hard_cmp["intersection"]["explicit_minus_implicit"][f"{pname}_minus_{iname}"]["passes"])
           for iname in roles["implicit"]}
    c2 = {n: {"passes_above_floor": needle["judgement"]["per_arm"][n]["passes"],
              "stops_at_floor": not needle["judgement"]["per_arm"][n]["passes"]}
          for n in roles["implicit"]}
    c3 = {n: {"delta_bpc_far": probe[n]["delta_bpc_far"],
              "ci95": probe[n]["delta_ci95_clustered"],
              "positive_ci": bool(probe[n]["delta_bpc_far"] > 0
                                  and probe[n]["delta_ci95_clustered"][0] > 0)}
          for n in roles["explicit_complete"]}
    c6 = {n: bool(caus[n]["causal_no_future_leak"]) for n in arms}
    criteria = {
        "arms": roles,
        "c1_explicit_recall_above_floor_hard_subset": c1,
        "c1b_explicit_minus_implicit_hard_subset": c1b,
        "c2_implicit_stops_at_floor_negative_control": c2,
        "c3_explicit_delta_bpc_far_positive_ci": c3,
        "c4_premise_implicit_bitwise_insensitive": bool(premise_ok),
        "c5_quality_reproduction_all_equal": bool(reproduction["all_equal"]),
        "c6_causality_prefix_bitwise_invariant": c6,
        "c6_all_causal": bool(all(c6.values())),
        "bands": cfg["bands"],
        "no_extrapolation": True,
        "bands_note": ("bpc 类：本格 σ_cell 为主判、0.014 为参考，两带并列报不得混用；"
                       "召回/接受率类只走配对 bootstrap / Wilson 95% CI，不跨量纲借用 0.014"),
    }
    results["criteria"] = criteria
    note(f"判据① explicit 硬子集召回越地板：{c1}（跨臂 {c1b}）")
    note(f"判据② implicit 负对照停在地板：{ {k: v['stops_at_floor'] for k, v in c2.items()} }")
    note(f"判据③ explicit Δbpc_far>0 且 CI 排除 0：{c3}")
    note(f"判据④ 前提（implicit 逐位不敏感）：{premise_ok}；判据⑤ 逐位复现："
         f"{reproduction['all_equal']}；判据⑥ 因果：{c6}")

    # ---------------- ⑨ 活性 / 训练侧读数 ----------------
    note("活性量与训练侧读数")
    slopes, tail, train_meta, read_mass = {}, {}, {}, {}
    for name, model in arms.items():
        spec = next(a for a in cfg["arms"] if a["name"] == name)
        res_p = json.load(open(resolve(spec["train_results_json"])))
        lines = [json.loads(l) for l in open(resolve(spec["train_log"])) if l.strip()]
        slopes[name] = loss_slope(lines)
        tail[name] = {"last_step": lines[-1]["step"], "last_loss": lines[-1]["loss"],
                      "last_read_share": lines[-1].get("attn_carry")}
        train_meta[name] = {"wall_minutes": res_p.get("wall_minutes"),
                            "peak_gpu_mem_mb": res_p.get("peak_gpu_mem_mb"),
                            "params": res_p.get("params"),
                            "seed": res_p.get("train", {}).get("seed"),
                            "carry_write": res_p.get("carry_write"),
                            "window_mask": res_p.get("window_mask")}
        if ck[name]["carry_rule"]:
            read_mass[name] = carry_read_mass_stats(model, x_probe, window=window)
            train_meta[name]["carry_liveness_training"] = _strict(res_p.get("carry_liveness"))
            read_mass[name]["eval_recheck"] = {
                "note": "评测期反事实：关通道 / 换写入规则是否改变前向",
                "probe": _strict(carry_probe(model, x_probe))}
        note(f"  {name}: wall={train_meta[name]['wall_minutes']}min "
             f"peak={train_meta[name]['peak_gpu_mem_mb']}MB "
             f"loss末={tail[name]['last_loss']:.4f} "
             f"read_share={read_mass.get(name, {}).get('read_share_per_layer')}")
    results.update({
        "liveness": read_mass, "loss_slope": slopes, "train_log_tail": tail,
        "train_meta": train_meta,
    })

    # ---------------- ⑩ 成本 + 落盘 ----------------
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "eval_peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                                  if use_cuda else None),
        "n_eval_windows_bpc": eb["n_batches"] * eb["batch_size"],
        "n_probe_windows": ep["n_batches"] * ep["batch_size"],
        "n_needle_batches": vcfg["n_batches"],
        "retention_enabled": bool(rcfg["enabled"]),
        "lock_note": ("纯推理评测，由驱动整程持 /tmp/pathlm_gpu.lock（SCHEMA §20），"
                      "取/放时间戳见 run_log.txt"),
    }
    results["code_hashes"] = {
        rel: sha256_file(os.path.join(ROOT, rel)) for rel in
        ("pathlm/depth_ar.py", "tools/evalkit/retention.py",
         "tools/evalkit/carry_channel.py", "pathlm/data.py")
        if os.path.exists(os.path.join(ROOT, rel))}
    results["code_hashes"].update({
        rel: sha256_file(os.path.join(HERE, rel)) for rel in
        ("run.py", "realtext_probe.py", "needle_stats.py", "config.json")})

    payload = _strict(results)
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=False)
    with open(os.path.join(args.out_dir, "needle_raw.json"), "w") as f:
        json.dump(records, f)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"落盘：{args.out_dir}/results.json + needle_raw.json + run_log.jsonl")


if __name__ == "__main__":
    main()
