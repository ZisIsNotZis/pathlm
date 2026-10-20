"""2613 · 契约 F 的 20 步冒烟（SCHEMA §3 / harness/methodology.md §10：先冒烟再全跑）。

落 `smoke/smoke.json`，回答五件事：
1. **恒等起步（全尺寸 13M）**：dense+linres 的 readout_p=0 ⇒ 与同 seed 的 dense
   基线在一整个真实 batch 上全深度 logits 逐位一致；state_dict 只多 readout_p/q；
2. **参数量账**：linres = baseline + 2·d²（部署 GPU 上的真实参数量）；
3. **干预活性**：20 步后 P 已离开零点、Q 非零（单边零非鞍点）；
4. **步时与显存**（全跑预算的输入）：实测每步秒数与 torch 峰值分配显存；
5. **与 gated 臂参数逐位相同**（可选第三臂的等参数性）。

用法（<1 min）：python3 smoke.py
"""

import json, os, sys, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from pathlm.data import load_enwik8_full, batch
from run import build_model, run_battery  # noqa: E402

DATA = os.path.join(ROOT, "data", "enwik8_full.npz")
DEVICE = "cuda"


def twenty_steps(model, train_arr, seed=0, steps=20, lr=1e-3):
    """复刻 train_depth_ar.train 的单因素子集（dense 形态：final CE，无混合深度）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    seq = model.cfg.seq_len
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    losses = []
    for _ in range(steps):
        model.train()
        x, _ = batch(train_arr, 32, seq, gen)
        x = x.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            loss, _ = model(x, exit_depths=None)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    return {"steps": steps, "sec_per_step": round((time.time() - t0) / steps, 4),
            "wall_s": round(time.time() - t0, 2),
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "torch_peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "torch_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 3)}


def main():
    os.makedirs(os.path.join(HERE, "smoke"), exist_ok=True)
    out = {"experiment": "2613-readout-reparam-dense-13m", "device": DEVICE,
           "visible_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}
    train_arr, eval_arr, vocab = load_enwik8_full(os.path.join(ROOT, ".tmp", "enwik8"), DATA)
    out["vocab_size"] = vocab

    base = build_model("baseline", vocab, seed=0)
    lr_m = build_model("linres", vocab, seed=0)
    g_m = build_model("gated", vocab, seed=0)
    p_base = sum(p.numel() for p in base.parameters())
    p_lin = sum(p.numel() for p in lr_m.parameters())
    p_g = sum(p.numel() for p in g_m.parameters())
    out["params"] = {"baseline": p_base, "linres": p_lin, "gated": p_g,
                     "linres_minus_baseline": p_lin - p_base,
                     "expected_2d2": 2 * base.cfg.d_model ** 2,
                     "linres_equals_gated": p_lin == p_g}

    sb, sr = base.state_dict(), lr_m.state_dict()
    out["state_dict_extra_keys_linres"] = sorted(set(sr) - set(sb))
    out["state_dict_extra_keys_gated"] = sorted(set(g_m.state_dict()) - set(sb))
    out["common_params_bitwise_equal"] = all(torch.equal(sb[k], sr[k]) for k in sb)
    out["readout_p_all_zero_at_init"] = int(torch.count_nonzero(sr["readout_p.weight"])) == 0
    out["readout_q_nonzero_at_init"] = int(torch.count_nonzero(sr["readout_q.weight"])) > 0

    # 恒等起步：真实 batch 上全深度 logits 逐位一致
    torch.manual_seed(7)
    x, _ = batch(eval_arr, 4, base.cfg.seq_len, torch.Generator().manual_seed(7))
    x = x.to(DEVICE)
    with torch.no_grad():
        _, ab = base(x, exit_depths=None)
        _, ar = lr_m(x, exit_depths=None)
    out["identity_start_logits_bitwise_equal"] = all(
        torch.equal(a, b) for a, b in zip(ab["depth_logits"], ar["depth_logits"]))

    # 20 步活性 + 步时/显存（init 的 P fro 必须在训练前取：state_dict 是引用）
    init_p_fro = round(float(sr["readout_p.weight"].float().norm()), 4)
    out["twenty_steps_baseline"] = twenty_steps(base, train_arr, seed=0)
    out["twenty_steps_linres"] = twenty_steps(lr_m, train_arr, seed=0)
    P = lr_m.readout_p.weight.detach().float()
    Q = lr_m.readout_q.weight.detach().float()
    out["linres_activity_after_20"] = {
        "p_fro": round(float(P.norm()), 4), "q_fro": round(float(Q.norm()), 4),
        "p_nonzero": int(torch.count_nonzero(P)),
        "p_at_init_fro": init_p_fro}
    out["escaped_saddle"] = out["linres_activity_after_20"]["p_nonzero"] > 0
    out["smoke_battery_linres"] = run_battery(lr_m, eval_arr, n_batches=2)

    json.dump(out, open(os.path.join(HERE, "smoke", "smoke.json"), "w"),
              indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
