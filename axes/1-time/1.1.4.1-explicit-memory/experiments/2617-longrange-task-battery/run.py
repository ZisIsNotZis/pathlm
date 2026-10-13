"""2617 · 合成长程任务电池（ticket 34 契约 A）—— 实验入口（纯推理，不训练）。

契约：同目录 `README.md`（判据线**逐字**抄录，不许改）。
仪器实现：`tools/evalkit/longrange_tasks.py`；单测：`tests/test_longrange_tasks.py`。
被测量模型：`2610-frozen-trunk-jumphead-13m/trunk.pt`（复用，运行时断言 sha256）。

跑三件事：① 仪器自检（释放版前向 == 显式因果掩码前向，逐位）；② 四个合成任务
的 chance / 局部先验 / 13M 主干基线准确率（含距离-准确率曲线）；③ 变距离回忆 ×
`retention.recent_keep_mask` 可见性曲线 + 打乱答案的变异验证。

用法：
  python3 run.py                 # 全电池（纯推理，共享卡 ~1–3 min）
  python3 run.py --smoke         # 冒烟：缩小批量/档数，实测步时与显存
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

from pathlm.depth_ar import DepthARConfig, DepthARModel            # noqa: E402
from tools.evalkit.retention import masked_logits                  # noqa: E402
from tools.evalkit.longrange_tasks import (                        # noqa: E402
    TaskBatch, assoc_recall_batch, induction_batch, copy_batch,
    var_recall_batch, task_summary, accuracy, local_prior_accuracy,
    recall_under_visibility, shuffle_answers, corrupt_span,
    check_local_prior_resistance)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_trunk(cfg: dict, device: str):
    """加载被测量主干并**断言 sha256**（复用他人 checkpoint 的强制项）。"""
    ckpt = os.path.join(ROOT, cfg["checkpoint"]["path"])
    got = sha256_file(ckpt)
    want = cfg["checkpoint"]["sha256"]
    if got != want:
        raise SystemExit(f"checkpoint sha256 mismatch:\n  want {want}\n  got  {got}")
    model = DepthARModel(DepthARConfig(**cfg["model"]), cfg["data"]["vocab_size"])
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {"path": cfg["checkpoint"]["path"], "sha256": got,
                   "size_bytes": os.path.getsize(ckpt), "sha256_verified": True}


def concat(batches: list[TaskBatch]) -> TaskBatch:
    """把同任务的多个批拼成一批（距离曲线的分桶由 `dists` 决定）。"""
    f = batches[0]
    return TaskBatch(
        name=f.name, tokens=torch.cat([b.tokens for b in batches], 0),
        rows=torch.cat([b.rows for b in batches], 0),
        answers=torch.cat([b.answers for b in batches], 0),
        dists=torch.cat([b.dists for b in batches], 0),
        n_answers=f.n_answers, meta=f.meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--out-dir", default=None,
                    help="落盘目录；缺省 = 本目录（--smoke 时 = 本目录/smoke）")
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：缩小批量/档数，实测步时与显存（不外推时间）")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = os.path.join(HERE, "smoke") if args.smoke else HERE
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = json.load(open(args.config))
    if args.smoke:
        cfg["tasks"]["assoc_recall"]["n_pairs"] = [4]
        cfg["tasks"]["assoc_recall"]["n_batches"] = 1
        cfg["tasks"]["induction"]["src_starts"] = [0]
        cfg["tasks"]["copy"]["src_positions"] = [0]
        cfg["tasks"]["var_recall"]["distances"] = [16, 255]
        for t in cfg["tasks"].values():
            t["batch_size"] = 8
            t["n_batches"] = 1
        cfg["retention_curve"]["n_batches"] = 1
        cfg["retention_curve"]["batch_size"] = 8
        cfg["mutation"]["n_batches"] = 1
        cfg["mutation"]["batch_size"] = 8

    dev = cfg["device"]
    tok = cfg["tokens"]
    sym, fill, cue = tok["sym_vocab"], tok["filler_token"], tok["cue_token"]
    local_k = tok["local_k"]
    T = cfg["model"]["seq_len"]
    V = cfg["data"]["vocab_size"]
    g = torch.Generator().manual_seed(cfg["seed"])

    model, ck = load_trunk(cfg, dev)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    log_lines: list[dict] = []

    def make(kind: str, **kw) -> TaskBatch:
        common = dict(sym_vocab=sym, filler_token=fill, cue_token=cue, generator=g)
        return {"assoc": assoc_recall_batch, "induction": induction_batch,
                "copy": copy_batch, "var": var_recall_batch}[kind](**kw, **common)

    def batch_of(kind: str, n_batches: int, **kw) -> TaskBatch:
        """同一配置重复生成 `n_batches` 批再拼接（扩大样本量）。"""
        return concat([make(kind, **kw) for _ in range(n_batches)])

    # ---------- ① 仪器自检：释放版前向 == 显式因果掩码前向 ----------
    probe = make("var", batch_size=cfg["tasks"]["var_recall"]["batch_size"],
                 seq_len=T, distance=cfg["tasks"]["var_recall"]["distances"][0])
    x = probe.tokens.to(dev)
    logits_release = masked_logits(model, x, None)
    tril = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
    logits_mask = masked_logits(model, x, tril)
    instrument = {
        "max_abs_logit_diff": round(float((logits_release - logits_mask).abs().max()
                                          .item()), 8),
        "bitwise_equal": bool(torch.equal(logits_release, logits_mask)),
        "note": ("释放版 is_causal=True 路径 vs 显式精确因果掩码路径；两者应逐位相等"
                 "（与 2614 仪器同款自检）。"),
    }
    print(f"[自检] 释放版 vs 显式因果掩码 max|Δlogit| = "
          f"{instrument['max_abs_logit_diff']}", flush=True)

    # ---------- ② 四个任务 ----------
    tasks: dict = {}
    controls: dict = {}

    # 关联回忆：K（对数）与距离（被问哪一对）两轴
    ar_cfg = cfg["tasks"]["assoc_recall"]
    ar_by_len = {}
    ar_all = []
    for K in ar_cfg["n_pairs"]:
        per_qi = [batch_of("assoc", ar_cfg["n_batches"], batch_size=ar_cfg["batch_size"],
                           seq_len=T, n_pairs=K, query_index=qi) for qi in range(K)]
        pooled = concat(per_qi)
        ar_all.append(pooled)
        ar_by_len[str(K)] = task_summary(model, pooled, local_k=local_k,
                                         vocab_size=V, device=dev)
        log_lines.append({"task": "assoc_recall", "n_pairs": K,
                          "n": pooled.B, "model_acc": ar_by_len[str(K)]["model"]["accuracy"],
                          "local_acc": ar_by_len[str(K)]["local_prior"]["accuracy"]})
        print(f"[assoc_recall] K={K} model={ar_by_len[str(K)]['model']['accuracy']:.4f} "
              f"local={ar_by_len[str(K)]['local_prior']['accuracy']:.4f} "
              f"chance={pooled.chance():.4f}", flush=True)
    tasks["assoc_recall"] = {"by_n_pairs": ar_by_len,
                             "pooled": task_summary(model, concat(ar_all), local_k=local_k,
                                                    vocab_size=V, device=dev)}

    # Induction：源片段起点（距离）曲线 + 源对照臂（毁掉被拷贝的那段）
    ind_cfg = cfg["tasks"]["induction"]
    ind_batches, ind_control = [], []
    for a in ind_cfg["src_starts"]:
        b = batch_of("induction", ind_cfg["n_batches"],
                     batch_size=ind_cfg["batch_size"], seq_len=T,
                     prefix_len=ind_cfg["prefix_len"], copy_len=ind_cfg["copy_len"],
                     src_start=a)
        ind_batches.append(b)
        ind_control.append(corrupt_span(b, a, a + ind_cfg["copy_len"], g))
    ind_pooled = concat(ind_batches)
    tasks["induction"] = {"pooled": task_summary(model, ind_pooled, local_k=local_k,
                                                 vocab_size=V, device=dev)}
    controls["induction_source_destroyed"] = accuracy(model, concat(ind_control),
                                                      device=dev)
    print(f"[induction] model={tasks['induction']['pooled']['model']['accuracy']:.4f} "
          f"local={tasks['induction']['pooled']['local_prior']['accuracy']:.4f} "
          f"ctrl(src destroyed)={controls['induction_source_destroyed']['accuracy']:.4f}",
          flush=True)

    # 复制：源位置（距离）曲线 + 源对照臂（毁掉源片段）
    cp_cfg = cfg["tasks"]["copy"]
    cp_batches, cp_control = [], []
    for p in cp_cfg["src_positions"]:
        b = batch_of("copy", cp_cfg["n_batches"], batch_size=cp_cfg["batch_size"],
                     seq_len=T, copy_len=cp_cfg["copy_len"], src_pos=p)
        cp_batches.append(b)
        cp_control.append(corrupt_span(b, p, p + cp_cfg["copy_len"], g))
    cp_pooled = concat(cp_batches)
    tasks["copy"] = {"pooled": task_summary(model, cp_pooled, local_k=local_k,
                                            vocab_size=V, device=dev)}
    controls["copy_source_destroyed"] = accuracy(model, concat(cp_control), device=dev)
    print(f"[copy] model={tasks['copy']['pooled']['model']['accuracy']:.4f} "
          f"local={tasks['copy']['pooled']['local_prior']['accuracy']:.4f} "
          f"ctrl(src destroyed)={controls['copy_source_destroyed']['accuracy']:.4f}",
          flush=True)

    # 变距离回忆：距离曲线
    vr_cfg = cfg["tasks"]["var_recall"]
    vr_batches = [batch_of("var", vr_cfg["n_batches"], batch_size=vr_cfg["batch_size"],
                           seq_len=T, distance=d) for d in vr_cfg["distances"]]
    vr_pooled = concat(vr_batches)
    tasks["var_recall"] = {"pooled": task_summary(model, vr_pooled, local_k=local_k,
                                                  vocab_size=V, device=dev)}
    print(f"[var_recall] by_distance model="
          f"{ {k: v['accuracy'] for k, v in tasks['var_recall']['pooled']['model']['by_distance'].items()} }",
          flush=True)

    # ---------- ③ 变距离回忆 × retention 可见性 ----------
    rc = cfg["retention_curve"]
    # 每个距离单独造批：源位置随 distance 变，可见性曲线只在同一距离上固定变化
    keep_curve = {}
    for d in rc["distances"]:
        b = batch_of("var", rc["n_batches"], batch_size=rc["batch_size"],
                     seq_len=T, distance=d)
        keep_curve[str(d)] = {
            "full": accuracy(model, b, device=dev),
            "by_keep_d": recall_under_visibility(model, b, keep_ds=rc["keep_ds"],
                                                 device=dev),
        }
    retention_curve = {"distance_fixed": rc["distances"], "keep_ds": rc["keep_ds"],
                       "note": ("复用 retention.recent_keep_mask（只保留最近 d 个"
                                "位置的可见性）；源 (key,value) 对在 d < distance 时"
                                "不可见 ⇒ 若模型真靠长程记忆，召回会掉到 chance。"),
                       "curve": keep_curve}
    print("[retention] by_distance×keep_d 曲线已生成", flush=True)

    # ---------- ④ 变异验证：打乱答案 → 降到 chance ----------
    mu = cfg["mutation"]
    mb = batch_of("var", mu["n_batches"], batch_size=mu["batch_size"],
                  seq_len=T, distance=mu["distance"])
    acc_full = accuracy(model, mb, device=dev)
    acc_mut = accuracy(model, shuffle_answers(mb, generator=g), device=dev)
    mutation = {"task": "var_recall", "distance": mu["distance"],
                "chance": round(mb.chance(), 6),
                "accuracy_true_answers": acc_full["accuracy"],
                "accuracy_shuffled_answers": acc_mut["accuracy"],
                "dropped_to_chance": acc_mut["accuracy"] < 3 * mb.chance()}
    print(f"[mutation] true={acc_full['accuracy']:.4f} shuffled="
          f"{acc_mut['accuracy']:.4f} chance={mb.chance():.4f}", flush=True)

    # ---------- 判据 (a)：局部先验抗性 ----------
    guard = {}
    guard["assoc_recall"] = check_local_prior_resistance(
        concat(ar_all), k=local_k)
    guard["induction"] = check_local_prior_resistance(ind_pooled, k=local_k)
    guard["copy"] = check_local_prior_resistance(cp_pooled, k=local_k)
    guard["var_recall"] = check_local_prior_resistance(vr_pooled, k=local_k)
    guard_all_pass = all(guard[t]["passed"] for t in guard)

    # ---------- 落盘 ----------
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    results = {
        "contract": ("axes/1-time/1.1.4.1-explicit-memory/experiments/"
                     "2617-longrange-task-battery/README.md"),
        "smoke": bool(args.smoke),
        "checkpoint": ck,
        "config": cfg,
        "instrument": instrument,
        "intervention": {
            "what": ("评测期合成任务（无权重改动、无训练）：测模型在局部先验无法"
                     "补全的长程任务上的 top-1 准确率。"),
            "params_added": 0, "weights_changed": False, "training_steps": 0,
            "alive": guard_all_pass,
            "note": ("活性量 = 局部先验抗性（四任务均须 < max(3*chance, 0.02)）与"
                     "掩码可见性自检；本实验无训练步，故无「每 200 步」训练活性日志。"),
        },
        "tasks": tasks,
        "controls": controls,
        "retention_curve": retention_curve,
        "mutation": mutation,
        "criteria": {
            "a_local_prior_resistance": guard,
            "a_all_pass": guard_all_pass,
            "b_trunk_baseline_accuracies": {
                "assoc_recall": tasks["assoc_recall"]["pooled"]["model"]["accuracy"],
                "induction": tasks["induction"]["pooled"]["model"]["accuracy"],
                "copy": tasks["copy"]["pooled"]["model"]["accuracy"],
                "var_recall": tasks["var_recall"]["pooled"]["model"]["accuracy"],
            },
            "b_trunk_var_recall_by_distance":
                tasks["var_recall"]["pooled"]["model"]["by_distance"],
            "c_retention_compatible": True,
            "c_reused": "tools.evalkit.retention.recent_keep_mask",
        },
        "code": {
            "run_py_sha256": sha256_file(os.path.abspath(__file__)),
            "longrange_tasks_py_sha256": sha256_file(
                os.path.join(ROOT, "tools", "evalkit", "longrange_tasks.py")),
            "tests_sha256": sha256_file(
                os.path.join(ROOT, "tests", "test_longrange_tasks.py")),
            "retention_py_sha256": sha256_file(
                os.path.join(ROOT, "tools", "evalkit", "retention.py")),
        },
        "cost": {"wall_minutes": round(wall / 60, 3), "wall_seconds": round(wall, 2),
                 "peak_mem_mib": round(peak, 1), "device": dev},
    }
    out = os.path.join(args.out_dir, "results.json")
    json.dump(results, open(out, "w"), indent=2, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "run_log.jsonl"), "w") as f:
        for r in log_lines:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n判据 (a) 全过：{guard_all_pass}"
          f"{'（冒烟：样本量太小，此项仅供跑通，不作准）' if args.smoke else ''}\n"
          f"落盘：{out}\n墙钟 {wall/60:.2f} min，峰值显存 {peak:.0f} MiB", flush=True)


if __name__ == "__main__":
    main()
