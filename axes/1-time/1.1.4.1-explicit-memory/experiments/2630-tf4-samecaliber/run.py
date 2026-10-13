#!/usr/bin/env python3
"""2630-tf4-samecaliber · 同口径出口资产判决 —— 评测入口（纯推理）。

契约：同目录 `README.md`（§4 判据线**逐字**抄录任务下发文本）。本入口只做**评测**
（不训练、不改权重），把两臂（`base` = 随机退出 + proj-fill；`carry` = 同上 + delta
写入通道；**两臂都训在 enwik8 上**）在下列腿上量出来：

  ① 主判据 = **同口径 tf@4**（enwik8 训练、enwik8 评测）——engine `tf_acceptance`
     的 depth-4 出口 argmax 与全深度 argmax 一致率。判据：|Δ(carry−base)| < 1.4pp
     ⇒ 记「carry 不弄坏出口资产」；
  ② clean bpc（**同口径**）——engine `depth_curve` 的全深度（depth=L）bpc；
  ③ 参数量（逐臂声明 + 差）；
  ④ 活性量（carry 臂读通道遥测；每 200 步训练日志的系列）；
  ⑤ **口径不混用**：本线与 2629（合成训练 + enwik8 跨分布代理）是两个不同分布的
     口径，数字**不得混用**；本入口另报一条**反向跨分布诊断**（enwik8 训好的模型在
     synth 上评 tf@4），仅用于解释「口径为何会打架」，**不作判据**。

**非 dense 评测的口径声明**：两臂都是非 dense（RAP 形态：逐通道门 + 混合填充 +
随机退出），因此**不能**用 `retention.masked_logits`（那是 dense 主干的读法，会静默
丢掉门与填充）——本入口统一走引擎自己的 `tf_acceptance` / `depth_curve`（它们内部走
`model(x, exit_depths=None)` 全深度前向）。这与训练入口 `train_depth_ar.py` 在
`synth is None and not dense` 分支里跑的**是同一对函数、同一批参数**（同 seed / 同批数）
⇒ 两只入口的 tf@4 应逐位一致（`reconcile_numbers.py` 交叉核对）。

用法：
  python3 run.py                # 全电池（纯推理）
  python3 run.py --smoke        # 冒烟：少批（实测耗时/显存，不外推）
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from pathlm.data import load_enwik8_full                    # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,   # noqa: E402
                             depth_curve, tf_acceptance)
from tools.evalkit.longrange_tasks import assoc_recall_multi_batch  # noqa: E402
from samecaliber_arms import (ARMS, ARM_NAMES, BASELINE_ARM,  # noqa: E402
                              DATA_SOURCE, TARGET_ARM, VARIANCE_BAND_PP,
                              arm_spec, capability_probe, expected_block_class,
                              model_config_kwargs, single_factor_increment,
                              single_factor_increment_pp, tf4_holds)

# 与训练入口 `train_depth_ar.py`（`synth is None and not dense` 分支）逐参数对齐：
N_BATCHES = 20        # = args.eval_batches 缺省
BATCH_SIZE = 16
DEPTHS = (4, 6)       # 出口资产主口径 tf@4 / 副口径 tf@6


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(name: str, mcfg: dict, vocab: int, device: str, ckpt: dict):
    """加载一臂的权重并**断言 sha256**（权重不入库，哈希逐次核）。"""
    spec = arm_spec(name)
    path = resolve(os.path.join(ckpt["dir"], name, "model.pt"))
    got = sha256_file(path)
    want = ckpt.get("sha256", {}).get(name)
    if not want or want == "TO_FILL_AFTER_TRAINING":
        raise SystemExit(f"{name}: config 里没有可用 sha256（先训练并回填）")
    if got != want:
        raise SystemExit(f"{name}: checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    cfg = DepthARConfig(**{**mcfg, **model_config_kwargs(spec)})
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": os.path.relpath(path, ROOT), "sha256": got,
                   "size_bytes": os.path.getsize(path), "sha256_verified": True,
                   "block_class": type(model.blocks[0]).__name__,
                   "block_class_expected": expected_block_class(spec),
                   "n_params": sum(p.numel() for p in model.parameters()),
                   "carry_rule": spec["carry"],
                   "dense": bool(model.cfg.dense),
                   "proj_fill": bool(model.cfg.proj_fill),
                   "fill_kv": bool(model.cfg.fill_kv)}


def carry_intervention_block(name: str, model) -> dict:
    """SCHEMA §10 的 `intervention:` 活性块（参数侧：w_fro / w_absmax / nonzero_frac）。

    无 carry 臂同样落块，但 `enabled=False`、量全为 None —— 「没有这条通道」是基准臂
    的一部分，不是一个未激活的干预。
    """
    spec = arm_spec(name)
    if not spec["carry"]:
        return {"name": "delta_carry_write_channel", "enabled": False,
                "params_added": 0, "w_fro": None, "w_absmax": None,
                "nonzero_frac": None, "read_scale_absmean": None,
                "alive": False,
                "note": "无通道（carry_rule=None）：读出量为 0"}
    ps = [p.detach().float() for p in model.carry.parameters()]
    flat = torch.cat([p.reshape(-1) for p in ps])
    rs = [float(blk.carry_scale.detach().float().abs().mean())
          for blk in model.blocks]
    return {"name": "delta_carry_write_channel", "enabled": True,
            "params_added": sum(p.numel() for p in model.carry.parameters())
            + sum(blk.carry_scale.numel() for blk in model.blocks),
            "w_fro": round(float(sum(float(p.norm()) for p in ps)), 6),
            "w_absmax": round(float(flat.abs().max()), 6),
            "nonzero_frac": round(float((flat != 0).float().mean()), 6),
            "read_scale_absmean": round(sum(rs) / len(rs), 6),
            "alive": True}


@torch.no_grad()
def per_depth_agreement(model, T, sym, fill, cue, K, Q, gap, device, depths,
                        bs, nb, seed_base, L):
    """synth（训练分布外，**诊断腿**）的 depth-k 出口 argmax 与全深度 argmax 一致率。

    与 engine `tf_acceptance` 完全同一算法（同关键的一步是 `exit_depths=None` 的
    全深度前向 + 逐位置 argmax 比对），只是换到 synth 批次上 ⇒ 口径可比、分布不同。
    """
    agree = {k: 0 for k in depths}
    n = 0
    model.eval()
    for i in range(nb):
        tb = assoc_recall_multi_batch(
            bs, T, K, Q, sym_vocab=sym, filler_token=fill, cue_token=cue, gap=gap,
            generator=torch.Generator().manual_seed(seed_base + i))
        x = tb.tokens.to(device)
        _, aux = model(x, exit_depths=None)
        full = aux["depth_logits"][-1][:, :-1].argmax(-1)
        for k in depths:
            draft = aux["depth_logits"][k][:, :-1].argmax(-1)
            agree[k] += int((draft == full).sum())
        n += full.numel()
    return {str(k): round(agree[k] / n, 6) for k in depths}, n


def _slope_per_1k(rows, key, frac=0.25):
    """末窗（最后 frac 比例）线性斜率（每 1k 步），同方法同窗（SCHEMA §16/§17.15）。"""
    if not rows:
        return None
    k = max(2, int(len(rows) * frac))
    win = rows[-k:]
    xs = [r["step"] for r in win]
    ys = [r[key] for r in win]
    if any(y is None for y in ys):
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    return round(slope * 1000, 6)


def train_activity(name: str, train_dir: str, L: int) -> dict:
    """从 `train/<arm>/train_log.jsonl` 抽活性量（每 200 步）与收敛门禁量。"""
    p = resolve(os.path.join(train_dir, name, "train_log.jsonl"))
    rows = [json.loads(ln) for ln in open(p) if ln.strip()]
    carry_rows = [r for r in rows if r.get("attn_carry")]
    per200 = [r for r in rows if r["step"] % 200 == 0]
    full_ce = [{"step": r["step"], "ce": r["depth_ce"][L]} for r in per200]
    d4_ce = [{"step": r["step"], "ce": r["depth_ce"][4]} for r in per200]
    loss = [{"step": r["step"], "ce": r["loss"]} for r in per200]
    series = [{"step": r["step"],
               "read_share": r["attn_carry"].get("read_share"),
               "read_scale_absmean": r["attn_carry"].get("read_scale_absmean"),
               "state_fro": r["attn_carry"].get("state_fro"),
               "beta_mean": r["attn_carry"].get("beta_mean")}
              for r in carry_rows]
    return {"n_log_lines": len(rows),
            "n_rows_step_multiple_of_200": len(per200),
            "final_step": rows[-1]["step"], "final_loss": rows[-1]["loss"],
            "last_attn_carry": (carry_rows[-1]["attn_carry"] if carry_rows else None),
            "attn_carry_series_present": len(carry_rows),
            "carry_series": series,
            "carry_series_last200": series[-1] if series else None,
            "full_depth_ce_slope_nats_per_1k_lastquarter":
                _slope_per_1k(full_ce, "ce"),
            "depth4_ce_slope_nats_per_1k_lastquarter": _slope_per_1k(d4_ce, "ce"),
            "loss_slope_nats_per_1k_lastquarter": _slope_per_1k(loss, "ce"),
            "n_exit_ce": len(rows[-1]["depth_ce"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="少批，快速验证接线/耗时")
    args = ap.parse_args()
    nb = 2 if args.smoke else N_BATCHES
    bs = 8 if args.smoke else BATCH_SIZE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    cfg_path = os.path.join(HERE, "config.json")
    cfg = json.load(open(cfg_path))
    mcfg = cfg["model"]
    ckpt = cfg["checkpoint"]
    train_dir = cfg["checkpoint"]["dir"]
    log = []

    def say(msg):
        log.append({"t": round(time.time() - t0, 2), "msg": msg})
        print(msg, flush=True)

    say("CPU 能力面自检（两臂可建 / 块类 / 主干逐位一致）")
    cap = capability_probe()
    say(f"  combo_buildable={cap['combo_buildable']} "
        f"trunk_bit_identical={cap['trunk_bit_identical']} param_delta={cap['param_delta']}")

    # 训练数据（enwik8 全量）——同口径评测用它的 eval 段（引擎 `data.batch` 同约定）。
    train_arr, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8",
                                                       "data/enwik8_full.npz")
    say(f"enwik8 vocab={vocab_size}（模型 vocab={cfg['data']['vocab_size']}）")

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    free_before = (torch.cuda.mem_get_info()[0] // 2 ** 20) if device == "cuda" else None

    models, ckpts, params, arch_eq = {}, {}, {}, {}
    for name in ARM_NAMES:
        m, info = load_arm(name, mcfg, vocab_size, device, ckpt)
        models[name] = m
        ckpts[name] = info
        params[name] = {"n_params": info["n_params"]}
        arch_eq.setdefault("block_class", {})[name] = info["block_class"]
        arch_eq.setdefault("block_class_expected", {})[name] = \
            info["block_class_expected"]
        arch_eq.setdefault("dense", {})[name] = info["dense"]
        arch_eq.setdefault("proj_fill", {})[name] = info["proj_fill"]
        arch_eq.setdefault("fill_kv", {})[name] = info["fill_kv"]
        arch_eq.setdefault("data_source", {})[name] = DATA_SOURCE
        say(f"  {name:6s} {info['sha256'][:16]}… block={info['block_class']} "
            f"params={info['n_params']}")
    for name in ARM_NAMES:
        params[name]["delta_vs_base"] = params[name]["n_params"] - \
            params[BASELINE_ARM]["n_params"]
        params[name]["delta_pct_vs_base"] = round(
            100.0 * params[name]["delta_vs_base"]
            / params[BASELINE_ARM]["n_params"], 3)

    # 训练入口 `train_depth_ar.py` 的 enwik8 电池产物（若有）——只作**交叉核对**：
    # 它与本入口跑同两个引擎函数（tf_acceptance / depth_curve），应逐位一致。
    # **引擎已知限制**：`generation_quality` 构造 `DepthARDecoder` 时对 `carry_rule`
    # 直接 `raise ValueError`（增量解码尚未接入 `γ ⊙ mem_t`）⇒ **carry 臂**的训练
    # 入口在电量末尾中止、`results.json` 未落盘（`model.pt` 已在更早处保存）。
    # 主电池因此由本入口独立重算；缺档如实登记，**不**用估算值填补。
    train_entry = {}
    for name in ARM_NAMES:
        tp = resolve(os.path.join(train_dir, name, "results.json"))
        entry = {"results_json": os.path.relpath(tp, ROOT),
                 "present": os.path.exists(tp)}
        if entry["present"]:
            TR = json.load(open(tp))
            dep = {r["depth"]: r["accept"] for r in TR["tf_acceptance"]}
            entry.update({"tf4": dep[4], "tf6": dep[6],
                          "clean_bpc": TR["depth_curve"][-1]["bpc"],
                          "peak_gpu_mem_mb": TR["peak_gpu_mem_mb"],
                          "wall_minutes": TR["wall_minutes"]})
        else:
            entry["note"] = ("训练入口 enwik8 电池在 generation_quality 处中止"
                              "（DepthARDecoder 拒绝 carry_rule，引擎已知限制）；"
                              "model.pt 已保存，主电池由本入口独立重算")
        train_entry[name] = entry

    # ---------------- ① 主判据：同口径 tf@4（enwik8 训练、enwik8 评测） ----------------
    say(f"判据①（主）：同口径 tf@4/tf@6 —— enwik8 训练、enwik8 评测 "
        f"（n_batches={nb} bs={bs}）")
    tf_enw = {}
    for name in ARM_NAMES:
        table = tf_acceptance(models[name], eval_arr, n_batches=nb, batch_size=bs,
                              device=device)
        by = {r["depth"]: r["accept"] for r in table}
        tf_enw[name] = {"by_depth": table,
                        "tf4": by[4], "tf6": by[6],
                        "caliber": "enwik8（**同口径**：训练与评测同分布；TF 代理）"}
        say(f"  {name:6s} tf@4={by[4]:.4f} tf@6={by[6]:.4f}")

    # ---------------- ② clean bpc（同口径） ----------------
    say("判据②：clean bpc（同口径 enwik8 全深度前向，depth 曲线末点）")
    bpc = {}
    for name in ARM_NAMES:
        curve = depth_curve(models[name], eval_arr, n_batches=nb, batch_size=bs,
                            device=device)
        bpc[name] = {"bpc": curve[-1]["bpc"], "by_depth": curve,
                     "n_batches": nb, "batch_size": bs}
        say(f"  {name:6s} clean bpc(depth={len(curve)-1})={curve[-1]['bpc']:.6f}")

    # ---------------- 反向跨分布诊断（不作判据） ----------------
    say("诊断腿（**不作判据**）：enwik8 训好的模型在 synth 上评 tf@4（反向跨分布）")
    tsk = cfg["synth_diag"]
    tf_syn = {}
    for name in ARM_NAMES:
        acc, n = per_depth_agreement(
            models[name], cfg["model"]["seq_len"], tsk["sym_vocab"],
            tsk["filler_token"], tsk["cue_token"], tsk["K"], tsk["Q"], tsk["gap"],
            device, DEPTHS, tsk["batch_size"], tsk["n_batches"], tsk["seed"],
            cfg["model"]["n_layers"])
        tf_syn[name] = {"accept": acc, "n_tokens": n,
                        "caliber": "synth-assoc-mqar（**反向跨分布诊断**，不作判据）"}
        say(f"  {name:6s} tf@4(synth)={acc['4']:.4f} tf@6(synth)={acc['6']:.4f}")

    # ---------------- 增量 / 判定 ----------------
    increments = {
        "tf4_enwik8_samecaliber": single_factor_increment_pp(
            {n: tf_enw[n]["tf4"] for n in ARM_NAMES}),
        "tf6_enwik8_samecaliber": single_factor_increment_pp(
            {n: tf_enw[n]["tf6"] for n in ARM_NAMES}),
        "tf4_synth_crosscaliber_diag": single_factor_increment_pp(
            {n: tf_syn[n]["accept"]["4"] for n in ARM_NAMES}),
        "bpc_samecaliber": single_factor_increment(
            {n: -bpc[n]["bpc"] for n in ARM_NAMES}),   # raw bpc: bpc_base − bpc_carry
    }
    delta_tf4 = increments["tf4_enwik8_samecaliber"]["carry_effect"]
    decision = {
        "primary_metric": "tf4_enwik8_samecaliber",
        "baseline_arm": BASELINE_ARM, "target_arm": TARGET_ARM,
        "delta_pp": delta_tf4, "band_pp": VARIANCE_BAND_PP,
        "holds": tf4_holds(delta_tf4),
        "rule": "|Δ(carry−base)| < 1.4pp ⇒ 记「carry 不弄坏出口资产」（同口径 tf@4）",
        "caliber_note": ("**同口径** = enwik8 训练 + enwik8 评测；与 2629 的"
                         "「合成训练 + enwik8 跨分布代理」是两个不同分布的口径，"
                         "数字**不得混用**"),
    }
    say(f"判定：Δtf@4(同口径)={delta_tf4:+.2f}pp ⇒ holds={decision['holds']}")

    # ---------------- ③ 参数量 / ④ 活性量 ----------------
    intervention = {name: carry_intervention_block(name, models[name])
                    for name in ARM_NAMES}
    activity = {name: train_activity(name, train_dir, cfg["model"]["n_layers"])
                for name in ARM_NAMES}

    peak = (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
            if device == "cuda" else None)
    free_after = (torch.cuda.mem_get_info()[0] // 2 ** 20) if device == "cuda" else None

    results = {
        "config": cfg,
        "capability_probe": cap,
        "checkpoints": ckpts,
        "param_declaration": params,
        "architecture_equality": arch_eq,
        "train_entry_battery": train_entry,
        "tf_enwik8_samecaliber": tf_enw,
        "clean_bpc_samecaliber": bpc,
        "tf_synth_crosscaliber_diag": tf_syn,
        "increments": increments,
        "tf4_decision": decision,
        "intervention": intervention[TARGET_ARM],
        "intervention_by_arm": intervention,
        "train_activity": activity,
        "cost": {"pure_inference": True, "device": device,
                 "peak_gpu_mem_mb": peak,
                 "free_mb_before": free_before, "free_mb_after": free_after,
                 "wall_minutes_total_eval": round((time.time() - t0) / 60, 3)},
        "provenance": {"run_log": log, "smoke": args.smoke,
                       "n_batches": nb, "batch_size": bs},
    }
    outp = (os.path.join(HERE, "smoke", "results.json") if args.smoke
            else os.path.join(HERE, "results.json"))
    with open(outp, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n写出 {os.path.relpath(outp, ROOT)}")
    json.dump({k: results[k] for k in
               ("tf_enwik8_samecaliber", "increments", "tf4_decision",
                "param_declaration", "cost")},
              sys.stdout, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
