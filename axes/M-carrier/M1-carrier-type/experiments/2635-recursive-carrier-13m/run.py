#!/usr/bin/env python3
"""2635-recursive-carrier-13m · 评测入口（纯推理）：质量腿 + 活性 + 长程侧腿。

契约：同目录 `README.md`（§4 判据线预登记，未改一字）。本入口做：

  ① 仪器自检：两臂参数量**逐位相同**（`attn_linear` 不新增参数）；同 seed 重建的
     开关开/关两臂**主干参数逐位相等**；clean bpc 同权重再评一遍（纯推理确定性）；
  ② 质量腿（判据②）：clean bpc（全因果、enwik8 尾段），matched-step 快照口径；
  ③ 活性（判据③）：`attn_linear_liveness` / `attn_linear_probe`（φ>0、den>0、
     块间状态质量份额、同权重换算子的相对偏差）；
  ④ 侧腿（判据④，不作判据）：长程电池四任务零样本准确率（与 2617 的 stock 基线同一构造）；
  ⑤ 训练侧：loss 斜率、末窗斜率、停摆门报告（`stall_report.json`）。

用法：
  python3 run.py            # 全电池（纯推理，共享卡 ~3 min）
  python3 run.py --smoke    # 冒烟：少批
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
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pathlm.data import load_enwik8_full                          # noqa: E402
from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             attn_linear_liveness, attn_linear_probe)
from tools.evalkit.retention import masked_bpc                     # noqa: E402
from tools.evalkit.longrange_tasks import (accuracy,               # noqa: E402
                                           assoc_recall_batch, copy_batch,
                                           induction_batch, var_recall_batch)
from tools.evalkit.carry_channel import loss_slope, param_equality  # noqa: E402


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_arm(spec: dict, mcfg: dict, vocab: int, device: str):
    path = resolve(spec["path"])
    got = sha256_file(path)
    if got != spec.get("sha256"):
        raise SystemExit(f"{spec['path']}: sha256 mismatch\n want {spec.get('sha256')}"
                         f"\n got  {got}")
    cfg = DepthARConfig(**{**mcfg, "attn_linear": spec["attn_linear"]})
    model = DepthARModel(cfg, vocab)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": spec["path"], "sha256": got,
                   "size_bytes": os.path.getsize(path), "attn_linear": spec["attn_linear"],
                   "block_class": type(model.blocks[0]).__name__,
                   "n_params": sum(p.numel() for p in model.parameters())}


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
        cfg["eval"]["bpc"]["n_batches"] = 1
        cfg["eval"]["bpc"]["batch_size"] = 8
        for k in ("assoc_recall", "induction", "copy", "var_recall"):
            cfg["battery"][k]["n_batches"] = 1
        cfg["battery"]["var_recall"]["distances"] = [64, 255]
    dev = cfg["device"]
    log: list[dict] = []
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()

    def note(msg: str):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)
        log.append({"t": round(time.time() - t0, 2), "msg": msg})

    note("加载 enwik8 尾段 + 两臂（sha256 断言）")
    _, eval_arr, vocab = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    arms, ck = {}, {}
    for spec in cfg["arms"]:
        arms[spec["name"]], ck[spec["name"]] = load_arm(spec, cfg["model"], vocab, dev)
        note(f"  {spec['name']}: {ck[spec['name']]['sha256'][:16]}… "
             f"attn_linear={spec['attn_linear']} params={ck[spec['name']]['n_params']} "
             f"block={ck[spec['name']]['block_class']}")
    results = {"config": cfg, "checkpoints": ck}

    # ---------------- ① 仪器自检 ----------------
    note("自检①：参数量逐位相同 + 同 seed 重建主干逐位相等 + 评测确定性")
    assert ck["stock"]["n_params"] == ck["linear"]["n_params"], \
        "attn_linear 不新增参数（引擎不变量）被破坏"
    built = {}
    torch.manual_seed(cfg["seeds"][0])
    built["stock"] = DepthARModel(DepthARConfig(**{**cfg["model"], "attn_linear": False}),
                                  vocab)
    torch.manual_seed(cfg["seeds"][0])
    built["linear"] = DepthARModel(DepthARConfig(**{**cfg["model"], "attn_linear": True}),
                                   vocab)
    for m in built.values():
        m.eval()
    peq = param_equality(built["stock"], built["linear"])
    g = torch.Generator().manual_seed(21)
    x_probe = torch.randint(0, vocab, (2, cfg["model"]["seq_len"]), generator=g).cuda()
    with torch.no_grad():
        o1 = arms["linear"](x_probe)[1]["depth_logits"][-1]
        o2 = arms["linear"](x_probe)[1]["depth_logits"][-1]
    results["selfchecks"] = {
        "n_params_equal": ck["stock"]["n_params"] == ck["linear"]["n_params"],
        "n_params": ck["stock"]["n_params"],
        "params_bitwise_equal_built_same_seed": peq,
        "eval_determinism_bitwise": bool(torch.equal(o1, o2)),
    }
    note(f"  参数量 {ck['stock']['n_params']}（两臂相同）；同 seed 重建逐位等参数="
         f"{peq['bitwise_equal']}；评测确定性={results['selfchecks']['eval_determinism_bitwise']}")

    # ---------------- ② 质量腿 ----------------
    eb = cfg["eval"]["bpc"]
    note(f"质量腿（判据②）：clean bpc = masked_bpc(keep_mask=None, {eb['n_batches']}×"
         f"{eb['batch_size']}, window_seed={eb['window_seed']})")
    bpc = {}
    for name, model in arms.items():
        bpc[name] = round(masked_bpc(model, eval_arr, None, n_batches=eb["n_batches"],
                                     batch_size=eb["batch_size"],
                                     window_seed=eb["window_seed"], device=dev), 6)
        note(f"  {name}: bpc={bpc[name]}")
    delta_bpc = round(bpc["stock"] - bpc["linear"], 6)

    # ---------------- ③ 活性 ----------------
    lv = cfg["eval"]["liveness"]
    note("活性（判据③）：φ/den/块间质量/同权重换算子的相对偏差")
    liveness = {}
    for name, model in arms.items():
        if ck[name]["attn_linear"]:
            liveness[name] = attn_linear_probe(model, eval_arr, n_batches=lv["n_batches"],
                                              batch_size=lv["batch_size"],
                                              seed=lv["seed"], device=dev)
        else:
            liveness[name] = attn_linear_liveness(model)
        note(f"  {name}: alive={liveness[name].get('alive')} "
             f"den_min={liveness[name].get('den_min')} "
             f"phi_min={liveness[name].get('phi_min')} "
             f"inter_mass_frac={liveness[name].get('inter_mass_frac')} "
             f"rel_dev={liveness[name].get('rel_dev_vs_full_mean')}")

    # ---------------- ④ 侧腿：长程电池（不作判据） ----------------
    b = cfg["battery"]
    tok = b["tokens"]
    T = cfg["model"]["seq_len"]
    gen = torch.Generator().manual_seed(2617)
    note("侧腿（判据④，不作判据）：长程电池四任务零样本")
    tasks: list[tuple[str, object]] = []
    for k in b["assoc_recall"]["n_pairs"]:
        tasks.append((f"assoc_recall_K{k}", assoc_recall_batch(
            b["assoc_recall"]["batch_size"], T, k, **tok,
            generator=torch.Generator().manual_seed(2617 + k))))
    for sp in b["induction"]["src_starts"]:
        tasks.append((f"induction_src{sp}", induction_batch(
            b["induction"]["batch_size"], T, b["induction"]["prefix_len"],
            b["induction"]["copy_len"], **tok,
            generator=torch.Generator().manual_seed(2617 + sp))))
    for sp in b["copy"]["src_positions"]:
        tasks.append((f"copy_src{sp}", copy_batch(
            b["copy"]["batch_size"], T, b["copy"]["copy_len"], sp, **tok,
            generator=torch.Generator().manual_seed(2617 + sp))))
    for d in b["var_recall"]["distances"]:
        tasks.append((f"var_recall_d{d}", var_recall_batch(
            b["var_recall"]["batch_size"], T, d, **tok,
            generator=torch.Generator().manual_seed(2617 + d))))
    battery = {}
    for tname, task in tasks:
        row = {}
        for name, model in arms.items():
            row[name] = accuracy(model, task, None, device=dev)
        battery[tname] = {k: {"accuracy": round(v["accuracy"], 6),
                              "n": v["n"], "chance": round(task.chance(), 6),
                              "distance_mean": v.get("distance_mean")}
                          for k, v in row.items()}
    note("  电池（accuracy）：" + " | ".join(
        f"{t}: stock={v['stock']['accuracy']:.4f} linear={v['linear']['accuracy']:.4f}"
        for t, v in list(battery.items())[:6]) + " …")

    # ---------------- ⑤ 训练侧 ----------------
    note("训练侧：loss 斜率 / 停摆门报告")
    slopes, tail, train_meta = {}, {}, {}
    for name in arms:
        spec = next(a for a in cfg["arms"] if a["name"] == name)
        res_p = json.load(open(resolve(spec["train_results_json"])))
        lines = [json.loads(l) for l in open(resolve(spec["train_log"])) if l.strip()]
        slopes[name] = loss_slope(lines, lo=200, hi=cfg["steps"])
        tail[name] = {"last_step": lines[-1]["step"], "last_loss": lines[-1]["loss"],
                      "last_gnorm": lines[-1].get("gnorm")}
        train_meta[name] = {"wall_minutes": res_p.get("wall_minutes"),
                            "peak_gpu_mem_mb": res_p.get("peak_gpu_mem_mb"),
                            "params": res_p.get("params"),
                            "attn_linear": res_p.get("attn_linear"),
                            "linear_chunk": res_p.get("linear_chunk")}
    stall_path = os.path.join(HERE, "stall_report.json")
    stall = json.load(open(stall_path)) if os.path.exists(stall_path) else None
    results.update({
        "quality_leg": {"bpc": bpc, "delta_bpc_stock_minus_linear": delta_bpc,
                        "caliber": f"matched-step 快照（{cfg['steps']} 步 / "
                                   f"{cfg['steps']*32*256/1e6:.1f}M token）；全因果（无窗口）",
                        "note": ("单点：本实验只在 seed=0 上跑一次 ⇒ **无方差界**，"
                                 "只作「不可分辨 / 有代价」的描述，不称等价（README §9.3）")},
        "liveness": liveness, "battery": battery, "loss_slope": slopes,
        "train_log_tail": tail, "train_meta": train_meta,
        "stall_report": stall,
        "criteria": {
            "c1_runnable": bool(stall is None or stall.get("status") == "finished"),
            "c2_quality_reported": bool(stall is None or stall.get("status") == "finished"),
            "c2_delta_bpc_stock_minus_linear": delta_bpc,
            "c3_liveness_alive": bool(liveness["linear"].get("alive")),
            "c4_battery_reported": True, "c5_no_extrapolation": True,
            "c1_note": ("停摆门（stall_gate.py）判定；status=stalled 时判据①不通过，"
                        "本格记「实现层再次停摆」而不给质量结论"),
        },
    })
    results["cost"] = {
        "wall_minutes_total_eval": round((time.time() - t0) / 60, 2),
        "train_wall_minutes": {n: train_meta[n]["wall_minutes"] for n in arms},
        "eval_peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "train_peak_gpu_mem_mb": {n: train_meta[n]["peak_gpu_mem_mb"] for n in arms},
        "lock_note": "训练按臂释放锁（§20.1，臂间 ≥60s）；评测整程 <10 min 持锁",
    }
    results["code_hashes"] = {
        rel: sha256_file(os.path.join(ROOT, rel)) for rel in
        ("pathlm/depth_ar.py", "train_depth_ar.py", "tools/evalkit/longrange_tasks.py",
         "tools/evalkit/carry_channel.py", "tools/evalkit/retention.py",
         "pathlm/data.py") if os.path.exists(os.path.join(ROOT, rel))}
    results["code_hashes"].update({
        rel: sha256_file(os.path.join(HERE, rel)) for rel in
        ("run.py", "stall_gate.py", "config.json")})
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    note(f"落盘：{args.out_dir}/results.json")


if __name__ == "__main__":
    main()
