"""深度自适应 AR 训练 runner（ticket 23，方向修订后第一个实验）。

用法：python train_depth_ar.py <run_name> --d 256 --layers 8 [--dense]
                              [--steps 3600] [--seed 0] [--out-root DIR]
训练 + 测量电池（depth 曲线 / TF 接受率 / workspace 探针 / 出口提交生成 /
门曲线）→ <out-root>/<run_name>/（缺省 .scratch/23-depth-ar/evidence/）。

预登记门：out-root 归属 `axes/**` 的某个实验目录（含 `README.md` 的那一层）时，那格的契约必须
已绑定（`tools/prereg_bind.py`），否则拒绝启动——本脚本自己就是产物写入点，直跑它能绕过各实验
自己的 `run.py` / `train_arms.sh` 两道门。`.scratch/`、`/tmp/` 下的 out-root 不受此约束。
"""

import argparse, json, math, os, sys, time
from dataclasses import asdict

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))
from pathlm.data import load_enwik8_full, batch
# 预登记门：out-root 归属 axes/** 的实验时，那格的契约必须先绑定（tools/prereg_bind.py）。
from tools.prereg_bind import owner_for_output, require as require_prereg
from tools.evalkit.longrange_tasks import assoc_recall_multi_batch
from pathlm.depth_ar import (DepthARConfig, DepthARModel, sample_exit_depths,
                             depth_curve, tf_acceptance, workspace_probe,
                             generation_quality, readout_workspace_energy,
                             spec_cost_model, topology_liveness,
                             attn_sparse_probe, window_mask_liveness,
                             window_mask_probe, attn_linear_probe,
                             kv_head_liveness, carry_liveness, carry_probe,
                             carry_read_mass_stats)
# 重建训练信号（ticket 36 波二 U2 线，F4）：逻辑全在 `pathlm/recon_signal.py`，
# 这里只做「开关 → 损失接线 → 活性量落盘」。
from pathlm.recon_signal import recon_loss, recon_liveness


def add_recon_term(loss, model, hidden, tokens, recon_lambda, filler_token=None):
    """把重建项接到主干损失上（本契约的**唯一干预接线**）。

    `recon_lambda <= 0` 或 `recon_window == 0` ⇒ **原样返回传入的 `loss` 对象**
    （不是 `loss + 0·x`）：不建图、不做浮点运算、不消耗随机数 ⇒ λ=0 与基线逐位
    等价（单测 `test_recon_lambda_zero_returns_same_loss_object` 钉住对象同一性；
    头参数 `p.grad is None` ⇒ AdamW 整条跳过，连 weight decay 都不作用）。

    返回 `(loss, stats | None)`；`stats` 的键见 `recon_signal.recon_loss`。
    """
    if recon_lambda <= 0 or model.cfg.recon_window <= 0:
        return loss, None
    stats = recon_loss(model.recon_head, hidden, tokens,
                       model.cfg.recon_window, filler_token)
    return loss + recon_lambda * stats["loss"], stats


def _linear_anneal(v0: float, v1: float | None, step: int, steps: int,
                   anneal_frac: float) -> float:
    """通用线性退火：前 (1−frac)·steps 步恒 v0，末 frac·steps 步线性到 v1
    （起点连续）；v1=None 恒 v0。"""
    if v1 is None:
        return v0
    start = (1.0 - anneal_frac) * steps
    if step <= start:
        return v0
    frac = min((step - start) / max(steps - start, 1.0), 1.0)
    return v0 + (v1 - v0) * frac


def aux_weight_at(step: int, steps: int, aux_weight: float,
                  anneal_to: float | None = None,
                  anneal_frac: float = 0.2) -> float:
    """浅层 aux 权重时间表（ticket 23b）：常数 aux_weight；anneal_to 给定时
    在最后 anneal_frac 比例的步数内从 aux_weight 线性退火到 anneal_to
    （W3 = 1.0 → 0.1，最后 20% 步；ticket 25 C 臂 --exit-anneal 用
    anneal_frac=0.25 → 0.3）。起点连续：step ≤ start 恒 aux_weight。"""
    return _linear_anneal(aux_weight, anneal_to, step, steps, anneal_frac)


def final_weight_at(step: int, steps: int, final_weight: float,
                    anneal_to: float | None = None,
                    anneal_frac: float = 0.25) -> float:
    """最终深度权重时间表（ticket 25 C 臂 --exit-anneal）：final 权重与浅层
    aux 同窗线性退火（2× → 1.0，纯 trunk CE 权重终点恒 ≥ 1）；
    anneal_to=None 恒 final_weight（ticket 24 语义）。"""
    return _linear_anneal(final_weight, anneal_to, step, steps, anneal_frac)


def train(model: DepthARModel, train_arr, tcfg: dict, log_path: str,
          gate_log_path: str, exit_mode: str = "uniform",
          aux_weight: float = 1.0, aux_anneal_to: float | None = None,
          final_weight: float = 1.0,
          shallow_stopgrad: bool = False, dense_aux: bool = False,
          exit_anneal_to: float | None = None,
          exit_anneal_frac: float = 0.25,
          final_anneal_to: float | None = None,
          recon_lambda: float = 0.0,
          synth: dict | None = None):
    """随机退出深度训练：每步均匀采样 d_i ∈ {0..L}；记录逐深度 CE 与门均值。
    浅层 CE 权重按 aux_weight_at(step) 调度（默认恒 1 = v1 等权）；最终深度
    权重 final_weight（ticket 24：默认 1.0 = v1，2.0 = 契约「最终深度 2×」）。
    ticket 23c 开关：shallow_stopgrad = H3（浅头纯读出）；
    dense_aux = H1（dense 臂全深度监督，需 model.cfg.dense）。
    ticket 25 C 臂 --exit-anneal：exit_anneal_to 给定时浅层权重在末
    exit_anneal_frac 步线性降到该值，final 权重同窗线性降到 final_anneal_to
    （2× → 1.0）；调度关断时行为与 ticket-24 逐位一致。
    梯度范数监控（ticket 24）：全程累计 gnorm max/mean 与跳步数，随结果落盘。
    recon_lambda（ticket 36 波二 U2 线，F4 训练信号）：> 0 时把重建项
    λ·CE(recon_head(hidden), tokens[i−W]) 加在主损失上（fp32，走**同一个** pre-step
    grad-norm gate）；= 0 时 `add_recon_term` **原样返回**主损失对象 ⇒ 不建图、
    逐位等价基线（单测钉住）。"""
    steps, bs = tcfg["steps"], tcfg["batch_size"]
    seq = model.cfg.seq_len
    L = model.cfg.n_layers
    n_ce = 1 if (model.cfg.dense and not dense_aux) else L + 1
    # 训练期可见窗口（ticket 32 契约 O）的活性量：掩码是静态的（不随步变化），
    # 每个 200 步日志行都写同一份；另做一个一次性功能探针（掩码开/关两次前向的
    # 最大 logit 差 > 0）证明这个开关不是死的。掩码关闭时两者都是 None，
    # 不增加任何计算、不消耗随机数（逐位还原 v1）。
    win_live = (window_mask_liveness(seq, model.cfg.train_window,
                                     model.cfg.train_anchors)
                if model.cfg.train_window > 0 else None)
    win_probe = None
    if win_live is not None:
        x_probe = torch.randint(0, model.vocab_size, (2, seq),
                                generator=torch.Generator().manual_seed(1234))
        win_probe = window_mask_probe(model, x_probe.cuda())
        win_live = {**win_live, **win_probe}
    # KV 头共享（ticket 33 契约 S）的活性量：形状是**静态**的（不随步变化），每个
    # 200 步日志行都写同一份；同时把「各 KV 头的 K 投影行范数 / KV 行 Frobenius
    # 范数」做成**动态**量随日志落盘（证明变窄的 K/V 投影真的在学，不是死路径）。
    # 开关关闭时为 None，不增加任何计算、不消耗随机数（逐位还原 v1）。
    gqa_active = (model.cfg.n_kv_heads is not None
                  and model.cfg.n_kv_heads < model.cfg.n_heads)

    def kv_live() -> dict | None:
        if not gqa_active:
            return None
        blocks = model.blocks
        rows = []
        for blk in blocks:
            w = blk.qkv.weight.detach().float()
            d = model.cfg.d_model
            kv_dim = model.cfg.n_kv_heads * (d // model.cfg.n_heads)
            rows.append(w[d:d + 2 * kv_dim])
        return {
            "n_heads": model.cfg.n_heads,
            "n_kv_heads": model.cfg.n_kv_heads,
            "head_dim": model.cfg.d_model // model.cfg.n_heads,
            "qkv_out_width": int(blocks[0].qkv.weight.shape[0]),
            "kv_elems_per_position": 2 * model.cfg.n_kv_heads
            * (model.cfg.d_model // model.cfg.n_heads),
            "kv_reduction_frac_vs_mha": round(
                1 - model.cfg.n_kv_heads / model.cfg.n_heads, 6),
            "k_proj_per_kv_head_row_norm": [
                round(float(blk.qkv.weight.detach().float()[
                    model.cfg.d_model + i * (model.cfg.d_model
                                             // model.cfg.n_heads):
                    model.cfg.d_model + (i + 1) * (model.cfg.d_model
                                                   // model.cfg.n_heads)].norm()),
                      4)
                for i in range(model.cfg.n_kv_heads)],
            "kv_rows_fro_mean": round(
                float(sum(float(r.norm()) for r in rows) / len(rows)), 4),
        }
    # 写入通道（ticket 35 契约 U）的活性量（SCHEMA §10/§17.4）：与线性注意力同款，
    # 每步**按张量**累加、每 200 步随同一行日志落盘（不在每步做 `.item()`）。
    # carry 参数（写门权重/读出增益）是**静态块**：每个 200 步行写同一份，
    # 其中 read_scale_absmean 是「单边零初始化是否已逃逸」的直接证据。
    carry_keys = ("read_scale_absmean", "read_share", "mem_write_to_resid_ratio")
    carry_sum = {k: None for k in carry_keys}
    carry_layer_sum: list[dict] = []          # 逐层累计（每层每步一份）
    n_carry_layers = 0
    # 合成任务的**答案行** CE（契约 U 的训练侧诊断，不是判据）：只有这些行真的
    # 依赖通道；整段 loss 的其余部分是 filler（学得很快）与不可预测的随机符号
    # （恒在 ln64 地板）⇒ 只看总 loss 会看不到通道到底有没有在学。
    ans_ce_sum = 0.0
    ans_ce_n = 0

    def carry_log() -> dict | None:
        """每 200 步落盘的通道活性量：扫描侧（state_fro/β/写入量/读出范数）+ 读出侧
        （read_scale 逃逸、每层读出占注意力输出的比例 read_share）。"""
        if model.cfg.carry_rule is None:
            return None
        out = {k: round(float(v) / n_carry_layers, 6)
               for k, v in carry_sum.items() if v is not None} if n_carry_layers else {}
        scan = getattr(model, "_carry_tele", None)
        if scan is not None:
            out.update({k: round(float(scan[k]), 6)
                        for k in ("state_fro", "beta_mean", "write_rel",
                                  "read_norm")})
        out["n_layer_samples"] = n_carry_layers
        out["rule"] = model.cfg.carry_rule
        out["carry_chunk"] = int(model.cfg.carry_chunk)
        return out

    def data_batch():
        """一个训练批的 token `[bs, seq]`（cuda）+ 合成任务的判决行（用于诊断）。"""
        if synth is not None:
            tb = assoc_recall_multi_batch(
                bs, seq, synth["pairs"], synth["queries"],
                sym_vocab=synth["sym_vocab"], filler_token=synth["filler_token"],
                cue_token=synth["cue_token"], gap=synth["gap"],
                generator=torch_rng)
            return tb.tokens.cuda(), tb.rows.cuda(), tb.answers.cuda()
        x, _ = batch(train_arr, bs, seq, torch_rng)
        return x.cuda(), None, None

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))))
    torch_rng = torch.Generator().manual_seed(tcfg["seed"])
    t0 = time.time()
    depth_ce_acc = [0.0] * n_ce
    gate_acc = [0.0] * L
    # 内容 top-k 稀疏注意力的活性量（SCHEMA §10/§17.4）：每步累加、每 200 步
    # 随同一行训练日志落盘（cumulative mean，与 gate_mean 同口径）。
    sparse_keys = ("kept_mean", "kept_frac_of_T", "sel_dist_mean",
                   "sel_in_lastk_frac")
    sparse_acc = {kk: 0.0 for kk in sparse_keys}
    n_sparse_layers = 0
    # 线性注意力的活性量（SCHEMA §10/§17.4，契约 R）：每步**按张量**累加、每 200 步
    # 随同一行训练日志落盘（φ/分母取累计最小/最大，其余为累计均值）。
    # 不在每步做 `.item()`：8 层 × 6 次设备同步在共享卡上会变成同步风暴。
    lin_min = None
    lin_max = None
    lin_sum = {"inter_mass_frac": None, "state_norm": None}
    n_lin_layers = 0

    def lin_log() -> dict | None:
        """把累计张量活性量转成日志用的 Python 浮点（每 200 步调一次）。"""
        if not n_lin_layers or lin_min is None:
            return None
        return {
            "phi_min": round(float(lin_min["phi_min"]), 8),
            "phi_max": round(float(lin_max["phi_max"]), 6),
            "den_min": round(float(lin_min["den_min"]), 8),
            "inter_mass_frac": round(float(lin_sum["inter_mass_frac"])
                                     / n_lin_layers, 6),
            "state_norm": round(float(lin_sum["state_norm"])
                                / n_lin_layers, 6),
            "n_layer_samples": n_lin_layers,
        }
    n_logged = 0
    n_log_writes = 0
    gnorm_sum = 0.0
    gnorm_max = 0.0
    n_grad_skip = 0
    # 重建训练信号（ticket 36 波二 U2 线）的活性量（SCHEMA §10/§17.4）：每步按张量
    # 累加、每 200 步随同一行日志落盘；`head_gnorm` 是该解码头 **真的拿到梯度**
    # 的证据（λ=0 臂恒 None）。
    recon_acc: dict = {}

    def recon_log() -> dict | None:
        n = recon_acc.get("n_steps", 0)
        if not n:
            return None
        n_sym, n_fill = recon_acc.get("n_symbol"), recon_acc.get("n_filler")
        g = recon_acc.get("head_gnorm")
        return {
            "window": int(model.cfg.recon_window),
            "ce_all": round(float(recon_acc["ce_all"]) / n, 4),
            "ce_symbol": round(float(recon_acc["ce_symbol"]) / n, 4),
            "ce_filler": round(float(recon_acc["ce_filler"]) / n, 4),
            "acc_all": round(float(recon_acc["acc_all"]) / n, 6),
            "acc_symbol": round(float(recon_acc["acc_symbol"]) / n, 6),
            "head_gnorm": (round(float(g) / n, 6) if g is not None else None),
            "n_symbol_frac": (round(float(n_sym / (n_sym + n_fill)), 4)
                              if n_sym is not None else None),
            "n_steps": n,
        }
    # 读出几何干预的活性量（SCHEMA §10/§17.4，ticket 38 W4 线）：每 200 步随同一
    # 行训练日志落盘。per-depth 臂报**逐深度**的 P_k/Q_k fro 范数（证明每个深度
    # 的头都真的在学、不是死开关）；等参数共享对照臂报 A/B fro 范数。
    def readout_log() -> dict | None:
        if model.cfg.readout_perdepth:
            return {"mode": "perdepth",
                    "p_fro_by_depth": [round(float(m.weight.norm()), 4)
                                       for m in model.readout_pd_p],
                    "q_fro_by_depth": [round(float(m.weight.norm()), 4)
                                       for m in model.readout_pd_q]}
        if model.cfg.readout_wide_shared > 0:
            return {"mode": "wide_shared",
                    "m": int(model.cfg.readout_wide_shared),
                    "a_fro": round(float(model.readout_shared_a.weight.norm()), 4),
                    "b_fro": round(float(model.readout_shared_b.weight.norm()), 4)}
        return None
    for step in range(steps):
        model.train()
        x, ans_rows, ans_vals = data_batch()
        if exit_mode == "full":
            depths = torch.full((bs, seq), L, dtype=torch.long).cuda()
        else:
            depths = sample_exit_depths(bs, seq, L, torch_rng).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            if exit_anneal_to is not None:
                w = aux_weight_at(step, steps, aux_weight, exit_anneal_to,
                                  exit_anneal_frac)
                fw = final_weight_at(step, steps, final_weight, final_anneal_to,
                                     exit_anneal_frac)
            else:
                w = aux_weight_at(step, steps, aux_weight, aux_anneal_to)
                fw = final_weight
            loss, aux = model(x, exit_depths=depths, shallow_weight=w,
                              final_weight=fw,
                              shallow_stopgrad=shallow_stopgrad,
                              dense_aux=dense_aux)
            # 重建项（λ>0 才建图；目标 = 刚掉出可见窗口的那一格 tokens[i−W]）
            loss, recon_term = add_recon_term(
                loss, model, aux["states"][-1], x, recon_lambda,
                filler_token=(synth["filler_token"] if synth is not None
                              else None))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gnorm_f = float(gnorm)
        gnorm_sum += gnorm_f
        gnorm_max = max(gnorm_max, gnorm_f)
        if not torch.isfinite(gnorm) or gnorm > 1e4:
            n_grad_skip += 1
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.step()
        sched.step()
        for k in range(n_ce):
            depth_ce_acc[k] += float(aux["depth_ce"][k].detach())
        for k in range(len(aux["gate_mean"])):
            gate_acc[k] += float(aux["gate_mean"][k])
        for s in aux.get("attn_sparse", ()):
            for kk in sparse_keys:
                sparse_acc[kk] += float(s[kk])
            n_sparse_layers += 1
        for s in aux.get("attn_linear", ()):
            if lin_min is None:
                lin_min = {k: s[k].detach() for k in ("phi_min", "den_min")}
                lin_max = {"phi_max": s["phi_max"].detach()}
            else:
                lin_min["phi_min"] = torch.minimum(lin_min["phi_min"],
                                                   s["phi_min"].detach())
                lin_min["den_min"] = torch.minimum(lin_min["den_min"],
                                                   s["den_min"].detach())
                lin_max["phi_max"] = torch.maximum(lin_max["phi_max"],
                                                   s["phi_max"].detach())
            lin_sum["inter_mass_frac"] = (s["inter_mass_frac"].detach()
                                           if lin_sum["inter_mass_frac"] is None
                                           else lin_sum["inter_mass_frac"]
                                           + s["inter_mass_frac"].detach())
            lin_sum["state_norm"] = (s["state_norm"].detach()
                                     if lin_sum["state_norm"] is None
                                     else lin_sum["state_norm"]
                                     + s["state_norm"].detach())
            n_lin_layers += 1
        for i, s in enumerate(aux.get("attn_carry", ())):
            while len(carry_layer_sum) <= i:
                carry_layer_sum.append({k: None for k in carry_keys})
            for kk in carry_keys:
                v = s[kk].detach()
                carry_sum[kk] = v if carry_sum[kk] is None else carry_sum[kk] + v
                cur = carry_layer_sum[i][kk]
                carry_layer_sum[i][kk] = v if cur is None else cur + v
            n_carry_layers += 1
        if recon_term is not None:
            for kk in ("ce_all", "ce_symbol", "ce_filler", "acc_all",
                       "acc_symbol", "n_symbol", "n_filler"):
                v = recon_term[kk].detach()
                cur = recon_acc.get(kk)
                recon_acc[kk] = v if cur is None else cur + v
            recon_acc["n_steps"] = recon_acc.get("n_steps", 0) + 1
            g = model.recon_head.weight.grad
            if g is not None:
                gn = g.detach().norm()
                cur = recon_acc.get("head_gnorm")
                recon_acc["head_gnorm"] = gn if cur is None else cur + gn
        if ans_rows is not None:
            with torch.no_grad():
                lg = aux["depth_logits"][-1].float()               # [B,T,V]
                ar_ = torch.arange(lg.shape[0], device=lg.device).unsqueeze(1)
                sel = lg[ar_, ans_rows]                            # [B,Q,V]
                assert sel.shape[:2] == ans_vals.shape
                ce = F.cross_entropy(sel.reshape(-1, sel.shape[-1]),
                                     ans_vals.reshape(-1))
                ans_ce_sum += float(ce)
                ans_ce_n += 1
        n_logged += 1
        if step % 200 == 0 or step == steps - 1:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "loss": round(float(loss.detach()), 4),
                    "depth_ce": [round(float(c), 4) for c in aux["depth_ce"]],
                    "gate_mean": [round(float(g), 4) for g in aux["gate_mean"]],
                    "gnorm": round(float(gnorm), 2),
                    "aux_w": round(w, 4),
                    "final_w": round(fw, 4),
                    "attn_sparse": ({kk: round(v / n_sparse_layers, 6)
                                    for kk, v in sparse_acc.items()}
                                   if n_sparse_layers else None),
                    "attn_linear": lin_log(),
                    "attn_carry": carry_log(),
                    "recon": recon_log(),
                    "readout": readout_log(),
                    "synth_answer_ce": (round(ans_ce_sum / max(ans_ce_n, 1), 4)
                                        if ans_ce_n else None),
                    "synth_chance_ce": (round(math.log(synth["sym_vocab"]), 4)
                                        if synth is not None else None),
                    "window_mask": win_live,
                    "kv_heads": kv_live(),
                    "lr": round(float(sched.get_last_lr()[0]), 6),
                    "min": round((time.time() - t0) / 60, 2)}) + "\n")
            n_log_writes += 1
            with open(gate_log_path, "a") as f:
                f.write(json.dumps({"step": step,
                                    "gate_mean": [round(float(g), 4)
                                                  for g in aux["gate_mean"]]}) + "\n")
    return time.time() - t0, [c / max(n_logged, 1) for c in depth_ce_acc], \
        [g / max(n_logged, 1) for g in gate_acc], {
            "gnorm_max": round(gnorm_max, 2),
            "gnorm_mean": round(gnorm_sum / max(steps, 1), 3),
            "grad_skips": n_grad_skip,
            "attn_sparse_mean": ({kk: round(v / n_sparse_layers, 6)
                                  for kk, v in sparse_acc.items()}
                                 if n_sparse_layers else None),
            "attn_sparse_layer_samples": n_sparse_layers,
            "attn_linear_mean": (lambda d: ({k: v for k, v in d.items()
                                             if k != "n_layer_samples"}
                                            if d else None))(lin_log()),
            "attn_linear_layer_samples": n_lin_layers,
            "attn_carry_mean": (lambda d: ({k: v for k, v in d.items()
                                            if k != "n_layer_samples"}
                                           if d else None))(carry_log()),
            "attn_carry_layer_samples": n_carry_layers,
            "attn_carry_layer_stats": (
                [{k: round(float(v) / max(steps, 1), 6) for k, v in row.items()
                  if v is not None} for row in carry_layer_sum]
                if n_carry_layers else None),
            "attn_carry_scan_mean": (lambda d: ({k: round(float(v), 6) for k, v in
                                                  d.items()} if d else None))(
                (model._carry_tele if getattr(model, "_carry_tele", None) else None)),
            "recon_window": int(model.cfg.recon_window),
            "recon_lambda": float(recon_lambda),
            "recon_mean": recon_log(),
            "synth_answer_ce_mean": (round(ans_ce_sum / max(ans_ce_n, 1), 4)
                                     if ans_ce_n else None),
            "synth_answer_ce_n_steps": ans_ce_n,
            "synth_chance_ce": (round(math.log(synth["sym_vocab"]), 4)
                                if synth is not None else None),
            "window_mask": win_live,
            "window_mask_probe": win_probe,
            "kv_heads": kv_live(),
            "kv_heads_n_logged_lines": n_log_writes if gqa_active else 0}


def _parse_int_sched(raw):
    """逗号分隔的正整数 schedule → `list[int]`（None/空 → None）。

    E2/E4（ticket 67）：`--mlp-hidden-sched "1728,1729"` 这类 CLI 串转逐层隐藏宽；
    空串/None 一律回 None（= 逐层回落 `--mlp-mult`，逐位还原现状），绝不返回空列表
    （空列表会与「长度 == n_layers」校验冲突而误报）。非法项交给
    `DepthARModel.__init__` 的正整数校验建时报错。
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(v) for v in raw] or None
    parts = [p for p in str(raw).replace(" ", "").split(",") if p != ""]
    return [int(p) for p in parts] or None


def build_depthar_config(args) -> DepthARConfig:
    """CLI 参数 → `DepthARConfig`（单测钉住「CLI 开关不是死的」）。

    把这一段从 `main()` 里抽出来只为可测：`--n-kv-heads`（ticket 33 契约 S）这类
    新增通道必须有一条可失败单测证明它真的到了模型配置（否则「开关是死的」）。
    """
    return DepthARConfig(d_model=args.d, n_layers=args.layers, n_heads=args.heads,
                         seq_len=args.seq_len, mlp_mult=args.mlp_mult,
                         # E2/E4（ticket 67）：历史单测用不带该字段的 Namespace ⇒ getattr。
                         mlp_hidden_sched=_parse_int_sched(
                             getattr(args, "mlp_hidden_sched", None)),
                         dense=args.dense,
                         gate_bias_init=args.gate_bias_init,
                         topology=args.topology,
                         fill_kv=args.fill_kv, proj_fill=args.proj_fill,
                         readout_dims=args.readout_dims,
                         readout_gated=args.readout_gated,
                         readout_linear_resid=args.readout_linear_resid,
                         # ticket 38 W4 线：新开关用 getattr ⇒ 历史用不带这两个字段
                         # 的 `argparse.Namespace` 的单测逐字不动（同 recon_window）。
                         readout_perdepth=getattr(args, "readout_perdepth", False),
                         readout_wide_shared=getattr(args, "readout_wide_shared", 0),
                         attn_sparse_k=args.attn_sparse_k,
                         attn_linear=args.attn_linear,
                         linear_chunk=args.linear_chunk,
                         train_window=args.train_window,
                         train_anchors=args.train_anchors,
                         n_kv_heads=args.n_kv_heads,
                         carry_rule=(None if args.carry_write == "none"
                                     else args.carry_write),
                         carry_chunk=args.carry_chunk,
                         # 缺省 0（= 关闭）：`build_depthar_config` 的历史单测用的是不带
                         # 这个新字段的 `argparse.Namespace`，用 getattr 保证它们逐字不动；
                         # 「CLI 开关不是死的」由 `tests/test_recon_signal.py::
                         # test_recon_cli_plumbed_to_config` 单独钉住。
                         recon_window=getattr(args, "recon_window", 0),
                         carry_state_dim=args.carry_state_dim,
                         # R7 新开关：历史单测用不带这些字段的 Namespace ⇒ getattr。
                         carry_gate=getattr(args, "carry_gate", True),
                         aux_mode=getattr(args, "aux_mode", "renorm"),
                         mod_ratio=getattr(args, "mod_ratio", 0.0))


def _kv_head_liveness(model: DepthARModel, vocab_size: int,
                      seq_len: int) -> dict:
    """KV 头共享的活性块（SCHEMA §10）：在**一次真实前向**上抓实际 K/V 张量形状。

    探针输入用固定 seed 的随机 token（形状类活性量与输入分布无关），与训练数据流
    无关、不消耗训练随机数（在训练结束后、落盘结果前调用）。
    """
    g = torch.Generator().manual_seed(1234)
    x = torch.randint(0, vocab_size, (2, min(seq_len, 64)), generator=g)
    x = x.cuda() if torch.cuda.is_available() else x
    was_training = model.training
    out = kv_head_liveness(model, x)
    model.train(was_training)
    out["probe_tokens_shape"] = list(x.shape)
    out["probe_seed"] = 1234
    return out


def _probe_tokens(model: DepthARModel, vocab_size: int, args, synth) -> torch.Tensor:
    """通道功能探针（`carry_probe`）的输入：与训练**同分布**的一批固定 seed 输入。

    形态：合成任务（契约 U）时用 `assoc_recall_multi_batch`（seed 1234）的两行；
    否则用固定 seed 的随机 token（形状类活性量与输入分布无关，同
    `_kv_head_liveness`）。不消耗训练随机数（在训练结束后、落盘前调用）。
    """
    if synth is not None:
        tb = assoc_recall_multi_batch(
            2, args.seq_len, synth["pairs"], synth["queries"],
            sym_vocab=synth["sym_vocab"], filler_token=synth["filler_token"],
            cue_token=synth["cue_token"], gap=synth["gap"],
            generator=torch.Generator().manual_seed(1234))
        x = tb.tokens
    else:
        g = torch.Generator().manual_seed(1234)
        x = torch.randint(0, vocab_size, (2, min(args.seq_len, 64)), generator=g)
    return x.cuda() if torch.cuda.is_available() else x


def _r7_intervention(args, cfg, results) -> dict:
    """R7 早退对照臂的 `intervention:` 活性块（SCHEMA §10）。

    三件必须能回读的事：① 真正的干预是什么；② 它是否活着（活性量非零）；
    ③ 与基线的差异在哪里（无门「无门」用结构证据：参数少 L·(d²+d)）。
    """
    gate_means = results.get("avg_train_gate_mean") or [None]
    if cfg.mod_ratio > 0:
        me = results.get("mod_eval") or {}
        return {
            "name": "mod_token_routing",
            "alive": True,
            "params_added": me.get("router_params"),
            "mod_ratio": cfg.mod_ratio,
            "router_weight": getattr(args, "mod_router_weight", None),
            "routed_frac_mean": me.get("routed_frac_mean"),
            "layer_apps_per_token_measured": me.get(
                "layer_apps_per_token_measured"),
            "layer_apps_per_token_analytic": me.get("layer_apps_per_token_analytic"),
            "note": "每块放行每序列 top-k（k = round(ratio·T)）个 token，其余残差"
                    "旁路；活性量 = 逐层放行比例与逐 token 块作用次数（解析账并报）",
        }
    if not cfg.carry_gate:
        return {
            "name": "no_carry_gate",
            "alive": True,
            "params_added": 0,
            "params_removed": cfg.n_layers
            * (cfg.d_model * cfg.d_model + cfg.d_model),
            "w_fro": None, "w_absmax": None, "nonzero_frac": None,
            "gate_mean": gate_means,
            "note": "无门变体：参数里没有门（−L·(d²+d)），块输出原样写入；"
                    "avg_train_gate_mean 全 0 是「本臂无门」，不从门开度读机制",
        }
    if not cfg.dense:
        return {
            "name": "variable_depth_form",
            "alive": True,
            "params_added": 0,
            "carry_gate": cfg.carry_gate,
            "fill_kv": cfg.fill_kv, "proj_fill": cfg.proj_fill,
            "w_fro": None, "w_absmax": None, "nonzero_frac": None,
            "gate_mean": gate_means,
            "note": ("本臂的干预 = 深度形态本身（非 dense：随机退出深度 + 混合填充）"
                     "＋损失配方；活性量 = 逐层门均值（每 200 步另落 gate_log.jsonl，"
                     "SCHEMA §17.4）。`avg_train_gate_mean` 全 0 只在无门臂出现。"),
        }
    return {
        "name": "aux_mode_additive",
        "alive": True,
        "params_added": 0,
        "aux_mode": cfg.aux_mode,
        "aux_weight": args.aux_weight, "final_weight": args.final_weight,
        "gate_mean": gate_means,
        "note": "公平配方：主目标保持绝对权重，浅层项是轻量加法项（不除分母）；"
                "唯一变量 = 损失归一，形态与 c1-legacy 臂逐字段相同",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--mlp-mult", type=int, default=4,
                    help="MLP 隐层倍数（ticket 24：100M 规格用 6）")
    ap.add_argument("--mlp-hidden-sched", default=None,
                    help="逐层 FFN 隐藏宽（逗号分隔正整数，长度=层数；缺省 None = 各层"
                         "回落 --mlp-mult×d，逐位还原现状）。E2/E4 统一标杆：值 = mult×d"
                         " 即逐层 mlp_mult，任意整数即等参数对照所需宽度")
    ap.add_argument("--dense", action="store_true")
    # ---- R7「既有早退实现」对照开关（默认全为历史行为，逐位还原 v1）----
    ap.add_argument("--no-carry-gate", dest="carry_gate", action="store_false",
                    help="R7 无门变体（LayerSkip 式形态）：不建 carry 门，块输出"
                         "原样写入（参数 −L·(d²+d)）；只对非 dense 的变深度路径有意义")
    ap.add_argument("--aux-mode", choices=("renorm", "additive"),
                    default="renorm",
                    help="R7 损失归一：renorm（缺省，逐位还原 v1）= 加权平均，主目标"
                         "被摊薄到 f/(w(n−1)+f)；additive = 公平配方，主目标保持权重 f"
                         "再加 w·mean(浅层 CE)")
    ap.add_argument("--mod-ratio", type=float, default=0.0,
                    help="R7 MoD 式路由（需 --dense）：每块放行每序列 top-k 个 token，"
                         "k = round(ratio·T)；0 = 关闭（缺省）。"
                         "isoFLOP 设计点 = --layers 2n --mod-ratio 0.5")
    ap.add_argument("--mod-router-weight", type=float, default=0.01,
                    help="MoD 路由器辅助损失的系数 α（缺省 0.01；论文系数记记忆·未核）")
    ap.add_argument("--topology", choices=("sequential", "parallel",
                                          "parallel-dual-ln", "dense"),
                    default="sequential",
                    help="层间拓扑（ticket 30 契约 E）：sequential = 现有 Block（链式"
                         "残差）；parallel = 并行子层（单个共享 pre-LN，PaLM 原式，"
                         "参数 −2d）；parallel-dual-ln = 两子层各带 pre-LN、同读块"
                         "输入（与 Block 逐位等参数）；dense = 层输入 = 对所有此前"
                         "层输出的学习加权和（α 初始化为只取当前层）。后三者需"
                         "配合 --dense（单因素）")
    ap.add_argument("--exit-mode", choices=("uniform", "full"), default="uniform",
                    help="uniform = 随机退出深度 dᵢ~U{0..L}（混合深度填充）；"
                         "full = 全部 dᵢ=L（无混合填充，仅保留全深度并行监督——"
                         "税归因 ablation）")
    ap.add_argument("--steps", type=int, default=3600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gate-bias-init", type=float, default=-2.0)
    ap.add_argument("--aux-weight", type=float, default=1.0,
                    help="浅层（depth 0..L-1）CE 权重，最终深度恒 1；1.0 = "
                         "v1 等权（ticket 23b）")
    ap.add_argument("--aux-anneal-to", type=float, default=None,
                    help="给定则浅层权重在最后 20%% 步从 --aux-weight 线性退火"
                         "到该值（W3：--aux-weight 1.0 --aux-anneal-to 0.1）")
    ap.add_argument("--final-weight", type=float, default=1.0,
                    help="最终深度 CE 权重（ticket 24：1.0 = v1 等权；"
                         "2.0 = 契约「权重均匀 + 最终深度 2×」）")
    ap.add_argument("--shallow-stopgrad", action="store_true",
                    help="浅层（depth 0..L-1）读出前 detach 状态：浅层 CE 只训练 "
                         "tied U 头，梯度不穿 trunk（ticket 23c，H3）")
    ap.add_argument("--dense-aux", action="store_true",
                    help="dense 臂开启全深度并行监督（L+1 个 CE，梯度穿 trunk；"
                         "ticket 23c，H1；需配合 --dense）")
    ap.add_argument("--attn-sparse-k", type=int, default=0,
                    help="内容 top-k 稀疏注意力（ticket 31 契约 J）：每个查询"
                         "在因果掩码内只保留内容相似度最高的 k 个键（按内容"
                         "选择，其余置 -inf）。0 = 关闭（缺省，逐位还原 v1）；"
                         "需配合 --dense 且 topology=sequential（单因素）")
    ap.add_argument("--attn-linear", dest="attn_linear", action="store_true",
                    help="线性注意力（ticket 33 契约 R）：注意力算子换成归一化线性核"
                         " out_i = Σ_{j≤i}(φ(q_i)·φ(k_j))v_j / Σ φ(q_i)·φ(k_j)，φ=elu+1；"
                         "用累积状态实现 ⇒ 每层时间/显存对 T 线性。缺省关闭（逐位还原 v1）；"
                         "需配合 --dense 且 topology=sequential（单因素）")
    ap.add_argument("--linear-chunk", type=int, default=64,
                    help="线性注意力分块大小（实现细节，不是第二个干预档；需整除"
                         " --seq-len）")
    ap.add_argument("--train-window", type=int, default=0,
                    help="训练期可见窗口（ticket 32 契约 O）：每个查询只看最近"
                         " N 个位置；0 = 关闭（缺省，逐位还原 v1 的全因果路径）。"
                         "需配合 --dense 且 topology=sequential（单因素）")
    ap.add_argument("--train-anchors", type=int, default=0,
                    help="窗口训练时额外恒可见的最前面 N 个锚点位置（需 "
                         "--train-window > 0）")
    ap.add_argument("--n-kv-heads", type=int, default=None,
                    help="KV 头共享（ticket 33 契约 S，W3）：注意力 K/V 的**头数**"
                         "（不是头维度）。缺省 None = MHA（= --heads，stock Block"
                         " 路径，逐位还原 v1）；< --heads 时 K/V 投影只产生这么多个"
                         "头（= 2 即 GQA，= 1 即 MQA），由 SDPA enable_gqa 按组广播。"
                         "需配合 --dense 且 topology=sequential（单因素），必须整除"
                         " --heads")
    ap.add_argument("--no-fill-kv", dest="fill_kv", action="store_false",
                    help="no-fill ragged 语义（ticket 23e）：早退位置在未跑层"
                         "的 KV 中缺席（原引擎 per-layer cache 语义）；缺省 = "
                         "v1 fill（W0 复现点）")
    ap.add_argument("--carry-write", choices=("none", "plain", "delta"),
                    default="none",
                    help="显式写入通道（ticket 35 契约 U）：none = 无通道（U0，"
                         "缺省，逐位还原 v1）；plain = 逐位置累加写 S←S+βvk̂ᵀ（U1）；"
                         "delta = 残差写 S←S(I−βk̂k̂ᵀ)+βvk̂ᵀ（U2）。U1/U2 逐位等参数，"
                         "唯一变量 = 写入规则；读出增益单边零 ⇒ 起步恒等于 U0。"
                         "需 topology=sequential（单因素），可与 --train-window 同时开"
                         "（窗口造需求、通道提供路径）；ENG2 后也可与非 dense 的随机退出"
                         "+ ragged/fill（含 --proj-fill）同开。")
    ap.add_argument("--carry-chunk", type=int, default=64,
                    help="写入通道状态扫描的分块大小（实现细节：只改浮点累加顺序，"
                         "不改语义；缺省 64）")
    ap.add_argument("--carry-state-dim", type=int, default=None,
                    help="写入通道的**状态宽度** dh（ticket 36 契约 U1 的容量轴）："
                         "缺省 None = v1 口径 dh = d_model/n_heads（与注意力头同宽）；"
                         "给整数则通道头数 = d_model/dh、与注意力 n_heads 解耦。"
                         "必须整除 d_model，且只在 --carry-write 非 none 时有效")
    ap.add_argument("--recon-window", type=int, default=0,
                    help="重建训练信号（ticket 36 波二 U2 线，F4）：0 = 关闭（缺省，"
                         "逐位还原 v1）；> 0 = 建单层线性解码头，要求位置 i 的隐状态重建"
                         " tokens[i−recon-window]（刚掉出可见窗口的那一格）。需配合 "
                         "--dense、topology=sequential 且 --train-window > 0；本实验"
                         "用 --recon-window 16（= train_window）")
    ap.add_argument("--recon-lambda", type=float, default=0.0,
                    help="重建项权重 λ：> 0 = CE + λ·重建 CE（fp32，走同一个 pre-step "
                         "grad-norm gate）；= 0（缺省）= 不建图、逐位等价基线")
    ap.add_argument("--synth-task", choices=("none", "assoc-mqar"),
                    default="none",
                    help="训练数据源：none = enwik8（缺省，逐位还原历史）；"
                         "assoc-mqar = 多查询关联回忆合成任务（契约 U 的主任务，"
                         "见 tools/evalkit/longrange_tasks.py::assoc_recall_multi_batch）")
    ap.add_argument("--synth-pairs", type=int, default=8,
                    help="合成任务的 (key,value) 对数 K")
    ap.add_argument("--synth-queries", type=int, default=8,
                    help="合成任务末尾的查询数 Q（= 每行依赖记忆的监督行数）")
    ap.add_argument("--synth-gap", type=int, default=14,
                    help="合成任务里查询之间的 filler 数（≥16−3 ⇒ 判决行的可见窗口"
                         "与单查询版同一局部模式、窗口内无符号）")
    ap.add_argument("--proj-fill", action="store_true",
                    help="fill 基础上加逐层线性 adapter（恒等初始化，学得 "
                         "remap；ticket 23e 可选臂，需 fill_kv）")
    ap.add_argument("--readout-dims", type=int, default=None,
                    help="结构切分（ticket 25 B 臂）：unembed/出口头只读前 r 个"
                         "通道（tied 表列切片 E[:, :r]），trunk/embedding 全维度；"
                         "缺省 = 全维度读出")
    ap.add_argument("--readout-gated", action="store_true",
                    help="gated 残差读出（W4 主臂）：读出前加非线性状态变换 "
                         "h + SiLU(W1 h) ⊙ (W2 h)，再走 tied 线性读出（init "
                         "W1~N(0,0.02)/W2=0 ⇒ 严格恒等起步、非鞍点）；缺省 = "
                         "tied 线性（v1 逐位还原）")
    ap.add_argument("--readout-linear-resid", action="store_true",
                    help="线性等参数对照读出（W4 契约 B，2611）：读出前加纯线性"
                         "状态变换 h + P·(Q h)（init Q~N(0,0.02)/P=0 ⇒ 严格恒等"
                         "起步、非鞍点）；参数与 --readout-gated 相同（2·d²）但"
                         "无非线性；两者互斥", dest="readout_linear_resid")
    ap.add_argument("--readout-perdepth", action="store_true",
                    help="逐深度独立读出（W4 ticket 38 主臂）：每个深度 k 一套"
                         "线性状态变换 h + P_k·(Q_k h)，再走共享 tied 读出 ⇒ 每个"
                         "深度的有效读出头不同（= 逐深度一个线性头）；init "
                         "Q_k~N(0,0.02)/P_k=0 ⇒ 严格恒等起步、非鞍点；新增参数 "
                         "(L+1)·2d²", dest="readout_perdepth")
    ap.add_argument("--readout-wide-shared", type=int, default=0,
                    help="等参数深度共享读出对照（W4 ticket 38，SCHEMA §15）：读"
                         "出前加单个共享线性变换 h + A·(B h)（A [d,m]、B [m,d]，"
                         "m = 本值）；与 per-depth 臂**名义等参数**（取 m=(L+1)·d）"
                         "但无逐深度自由度。0 = 关闭（缺省）",
                    dest="readout_wide_shared")
    ap.add_argument("--exit-anneal", action="store_true",
                    help="C 臂冷却退火（ticket 25）：浅层 aux 权重均匀到 "
                         "(1−frac) 处、末 frac 线性降到 --exit-anneal-to；"
                         "final 权重从 --final-weight 同窗线性降到 "
                         "--final-anneal-to（纯 trunk CE 权重终点 1.0）")
    ap.add_argument("--exit-anneal-to", type=float, default=0.3,
                    help="浅层 aux 权重退火终点（缺省 0.3）")
    ap.add_argument("--exit-anneal-frac", type=float, default=0.25,
                    help="退火窗口占总步数比例（缺省末 25%%）")
    ap.add_argument("--final-anneal-to", type=float, default=1.0,
                    help="final 权重退火终点（缺省 1.0 = 2× 降回等权）")
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--out-root", default=".scratch/23-depth-ar/evidence")
    args = ap.parse_args()

    # 重建训练信号（ticket 36 波二 U2 线）：λ>0 但没开窗 = 「死的干预」⇒ 直接报错
    # （SCHEMA §10：不允许交付一个看起来开了、实际没接线的开关）。
    if args.recon_lambda > 0 and args.recon_window <= 0:
        raise SystemExit("--recon-lambda > 0 需要 --recon-window > 0（否则重建项恒不生效）")
    if args.recon_window > 0 and args.recon_lambda < 0:
        raise SystemExit(f"--recon-lambda 必须 >= 0，收到 {args.recon_lambda}")

    run_dir = os.path.join(args.out_root, args.run_name)
    # 预登记门（docs/methodology.md §8）：本脚本自己就是产物写入点（<out-root>/<run>/results.json），
    # 所以门必须在这里——只放在各实验的 run.py / train_arms.sh 里，直跑本脚本就能绕过。
    # out-root 归属 axes/** 的某个实验（含 README.md 的那一层）⇒ 那格的契约必须先绑定；
    # .scratch/ 与 /tmp/ 下的 out-root 不归属任何实验，不受此约束。
    run_owner = owner_for_output(run_dir)
    if run_owner is not None:
        owner_rel = os.path.relpath(run_owner, ROOT)
        if require_prereg(run_owner) != 0:
            raise SystemExit(f"预登记未绑定：{owner_rel}"
                             f"（先跑 python3 tools/prereg_bind.py {owner_rel}）")
    os.makedirs(run_dir, exist_ok=True)
    # 诊断开关（默认关闭）：共享卡上出现「无进展但 CPU 满载」时，用它定期把**全线程
    # Python 栈**打进 stderr（子进程的 stdout.log），定位卡在哪一步。
    dump_secs = int(os.environ.get("PATHLM_DUMP_STACKS", "0"))
    if dump_secs > 0:
        import faulthandler
        faulthandler.enable()
        faulthandler.dump_traceback_later(dump_secs, repeat=True)
        print(f"[diag] 每 {dump_secs}s 打一次全线程 Python 栈（PATHLM_DUMP_STACKS）",
              flush=True)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    synth = (None if args.synth_task == "none" else
             {"task": args.synth_task, "pairs": args.synth_pairs,
              "queries": args.synth_queries, "gap": args.synth_gap,
              "sym_vocab": 64, "filler_token": 64, "cue_token": 205})
    if synth is None:
        train_arr, eval_arr, vocab_size = load_enwik8_full(
            ".tmp/enwik8", "data/enwik8_full.npz")
    else:
        # 合成任务（契约 U）：vocab 仍用主干的 206（cue = vocab-1 = 205 是本仓
        # 既有 needle 约定），不加载 enwik8
        train_arr, eval_arr, vocab_size = None, None, 206
    cfg = build_depthar_config(args)
    if cfg.mod_ratio > 0:
        from pathlm.mod_router import MoDModel, mod_eval
        model = MoDModel(cfg, vocab_size,
                         router_weight=args.mod_router_weight).cuda()
    else:
        model = DepthARModel(cfg, vocab_size).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.run_name}: {n_params/1e6:.2f}M params, vocab {vocab_size}, "
          f"topology {args.topology}, attn_sparse_k {args.attn_sparse_k}, "
          f"attn_linear {args.attn_linear} (chunk {args.linear_chunk}), "
          f"train_window {args.train_window}/+{args.train_anchors}, "
          f"n_kv_heads {args.n_kv_heads}, carry {args.carry_write} "
          f"(chunk {args.carry_chunk}, state_dim {args.carry_state_dim}), "
          f"synth {args.synth_task}, "
          f"steps {args.steps} x bs{args.batch_size} x seq{args.seq_len} "
          f"= {args.steps*args.batch_size*args.seq_len/1e6:.1f}M tokens", flush=True)

    tcfg = {"steps": args.steps, "batch_size": args.batch_size, "lr": args.lr,
            "seed": args.seed}
    wall, avg_depth_ce, avg_gate, train_stats = train(
        model, train_arr, tcfg,
        os.path.join(run_dir, "train_log.jsonl"),
        os.path.join(run_dir, "gate_log.jsonl"),
        exit_mode=args.exit_mode,
        aux_weight=args.aux_weight,
        aux_anneal_to=args.aux_anneal_to,
        final_weight=args.final_weight,
        shallow_stopgrad=args.shallow_stopgrad,
        dense_aux=args.dense_aux,
        exit_anneal_to=(args.exit_anneal_to if args.exit_anneal else None),
        exit_anneal_frac=args.exit_anneal_frac,
        final_anneal_to=(args.final_anneal_to if args.exit_anneal else None),
        recon_lambda=args.recon_lambda,
        synth=synth)
    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))

    model.eval()
    results = {
        "run": args.run_name, "config": asdict(cfg), "train": tcfg,
        "topology": args.topology,
        "attn_sparse_k": args.attn_sparse_k,
        "attn_linear": args.attn_linear,
        "linear_chunk": args.linear_chunk,
        "train_window": args.train_window,
        "train_anchors": args.train_anchors,
        "n_kv_heads": args.n_kv_heads,
        "carry_write": args.carry_write,
        "carry_chunk": args.carry_chunk,
        "carry_state_dim": args.carry_state_dim,
        "synth": synth,
        "recon_window": args.recon_window,
        "recon_lambda": args.recon_lambda,
        "synth_task_config": (assoc_recall_multi_batch(
            2, args.seq_len, args.synth_pairs, args.synth_queries,
            gap=args.synth_gap,
            generator=torch.Generator().manual_seed(1234)).meta
            if synth is not None else None),
        "kv_head_liveness": _kv_head_liveness(model, vocab_size,
                                              seq_len=args.seq_len),
        "carry_liveness": carry_liveness(
            model, getattr(model, "_carry_tele", None),
            train_stats.get("attn_carry_layer_stats"),
            prob=carry_probe(model, _probe_tokens(model, vocab_size, args, synth))),
        "carry_read_mass": carry_read_mass_stats(
            model, _probe_tokens(model, vocab_size, args, synth),
            window=args.train_window if args.train_window > 0 else 16),
        "carry_probe_input": ("synth assoc-mqar（seed 1234）" if synth is not None
                              else "enwik8 eval 批（data.batch, seed 1234）"),
        "recon_liveness": recon_liveness(
            (model.recon_head if model.cfg.recon_window > 0 else None),
            model.cfg.recon_window, args.recon_lambda,
            train_stats.get("recon_mean")),
        "topology_liveness": topology_liveness(model),
        "exit_mode": args.exit_mode,
        "carry_gate": cfg.carry_gate, "aux_mode": cfg.aux_mode,
        "mod_ratio": cfg.mod_ratio,
        "mod_router_weight": (args.mod_router_weight
                              if cfg.mod_ratio > 0 else None),
        "fill_kv": args.fill_kv, "proj_fill": args.proj_fill,
        "aux_weight": args.aux_weight, "aux_anneal_to": args.aux_anneal_to,
        "final_weight": args.final_weight,
        "readout_dims": args.readout_dims,
        "readout_gated": args.readout_gated,
        "readout_linear_resid": args.readout_linear_resid,
        "readout_perdepth": getattr(args, "readout_perdepth", False),
        "readout_wide_shared": getattr(args, "readout_wide_shared", 0),
        "exit_anneal_to": args.exit_anneal_to if args.exit_anneal else None,
        "exit_anneal_frac": args.exit_anneal_frac if args.exit_anneal else None,
        "final_anneal_to": args.final_anneal_to if args.exit_anneal else None,
        "shallow_stopgrad": args.shallow_stopgrad, "dense_aux": args.dense_aux,
        "gnorm_monitor": train_stats,
        "params": n_params, "vocab_size": vocab_size,
        "train_tokens": args.steps * args.batch_size * args.seq_len,
        "wall_minutes": round(wall / 60, 2),
        "peak_gpu_mem_mb": (round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                            if torch.cuda.is_available() else None),
        "avg_train_depth_ce": [round(c, 4) for c in avg_depth_ce],
        "avg_train_gate_mean": [round(g, 4) for g in avg_gate],
    }
    if not args.dense and synth is None:
        results["depth_curve"] = depth_curve(model, eval_arr, device="cuda",
                                             n_batches=args.eval_batches)
        results["tf_acceptance"] = tf_acceptance(model, eval_arr, device="cuda",
                                                 n_batches=args.eval_batches)
        results["spec_cost_model"] = spec_cost_model(
            results["tf_acceptance"], cfg.n_layers,
            draft_depths=(4, 6))
        if cfg.readout_dims is not None and cfg.readout_dims < cfg.d_model:
            results["readout_workspace_energy"] = readout_workspace_energy(
                model, eval_arr, device="cuda", n_batches=4, batch_size=8)
        results["workspace"] = workspace_probe(model, eval_arr, device="cuda",
                                               n_batches=4, batch_size=8)
        results["generation"] = generation_quality(model, eval_arr, device="cuda",
                                                   prompt_len=64, n_new=128)
    elif cfg.mod_ratio > 0:
        # R7 MoD 臂：enwik8 电池（最终深度 bpc + 路由活性量 + 解析算力账）。
        results["mod_eval"] = mod_eval(model, eval_arr, device="cuda",
                                       n_batches=args.eval_batches)
    else:
        # 合成任务（契约 U）或 dense 臂：enwik8 电池不适用（bpc/生成段都定义在
        # 真实文本上，且 dense 臂不走变深度解码）；本臂的评测在实验目录的
        # `run.py` 里跑（合成长程任务电池 + 毁源对照臂）。ENG2：非 dense + 合成
        # 任务也走这里（此前会在 `depth_curve(eval_arr=None)` 上崩）。
        results["synth_note"] = ("训练用合成多查询关联回忆（契约 U）或 dense 臂，"
                                 "未在本入口跑 enwik8 电池；机制读数见实验目录 "
                                 "run.py/results.json")
    # SCHEMA §10：results.json 必须落一个 `intervention:` 活性块（与 recon_liveness
    # 同一份，不重复计算）。λ=0 臂同样落块，但 `alive=False`、`grad_flowed=None` ——
    # 「头存在但未被训练」是基线的一部分，不是一个未激活的干预。
    results["intervention"] = {**results["recon_liveness"],
                               "name": "reconstruction_sufficiency_loss",
                               "note": ("目标 = 位置 i 重建 tokens[i−recon_window]"
                                        "（刚掉出可见窗口的那一格）；λ=0 时整条项不建图"
                                        "（add_recon_term 原样返回 loss 对象）")}
    r7_arm = (cfg.mod_ratio > 0 or not cfg.carry_gate
              or cfg.aux_mode != "renorm"
              or (not cfg.dense and getattr(args, "recon_lambda", 0.0) <= 0))
    if r7_arm:
        # R7 早退对照臂：干预不是重建项 ⇒ 活性块必须换成该臂真正的干预
        # （SCHEMA §10 不允许交一个「看起来活着、其实没接线」的活性量）。
        results["intervention"] = _r7_intervention(args, cfg, results)
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("config", "train")}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
