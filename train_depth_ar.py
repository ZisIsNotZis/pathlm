"""深度自适应 AR 训练 runner（ticket 23，方向修订后第一个实验）。

用法：python train_depth_ar.py <run_name> --d 256 --layers 8 [--dense]
                              [--steps 3600] [--seed 0] [--out-root DIR]
训练 + 测量电池（depth 曲线 / TF 接受率 / workspace 探针 / 出口提交生成 /
门曲线）→ .scratch/23-depth-ar/evidence/<run_name>/。
"""

import argparse, json, math, os, sys, time
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlm.depth_ar import (DepthARConfig, DepthARModel, sample_exit_depths,
                             depth_curve, tf_acceptance, workspace_probe,
                             generation_quality, readout_workspace_energy,
                             spec_cost_model, topology_liveness,
                             attn_sparse_probe, window_mask_liveness,
                             window_mask_probe, attn_linear_probe,
                             kv_head_liveness)
from pathlm.data import load_enwik8_full, batch


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
          final_anneal_to: float | None = None):
    """随机退出深度训练：每步均匀采样 d_i ∈ {0..L}；记录逐深度 CE 与门均值。
    浅层 CE 权重按 aux_weight_at(step) 调度（默认恒 1 = v1 等权）；最终深度
    权重 final_weight（ticket 24：默认 1.0 = v1，2.0 = 契约「最终深度 2×」）。
    ticket 23c 开关：shallow_stopgrad = H3（浅头纯读出）；
    dense_aux = H1（dense 臂全深度监督，需 model.cfg.dense）。
    ticket 25 C 臂 --exit-anneal：exit_anneal_to 给定时浅层权重在末
    exit_anneal_frac 步线性降到该值，final 权重同窗线性降到 final_anneal_to
    （2× → 1.0）；调度关断时行为与 ticket-24 逐位一致。
    梯度范数监控（ticket 24）：全程累计 gnorm max/mean 与跳步数，随结果落盘。"""
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
    for step in range(steps):
        model.train()
        x, _ = batch(train_arr, bs, seq, torch_rng)
        x = x.cuda()
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
            "window_mask": win_live,
            "window_mask_probe": win_probe,
            "kv_heads": kv_live(),
            "kv_heads_n_logged_lines": n_log_writes if gqa_active else 0}


def build_depthar_config(args) -> DepthARConfig:
    """CLI 参数 → `DepthARConfig`（单测钉住「CLI 开关不是死的」）。

    把这一段从 `main()` 里抽出来只为可测：`--n-kv-heads`（ticket 33 契约 S）这类
    新增通道必须有一条可失败单测证明它真的到了模型配置（否则「开关是死的」）。
    """
    return DepthARConfig(d_model=args.d, n_layers=args.layers, n_heads=args.heads,
                         seq_len=args.seq_len, mlp_mult=args.mlp_mult,
                         dense=args.dense,
                         gate_bias_init=args.gate_bias_init,
                         topology=args.topology,
                         fill_kv=args.fill_kv, proj_fill=args.proj_fill,
                         readout_dims=args.readout_dims,
                         readout_gated=args.readout_gated,
                         readout_linear_resid=args.readout_linear_resid,
                         attn_sparse_k=args.attn_sparse_k,
                         attn_linear=args.attn_linear,
                         linear_chunk=args.linear_chunk,
                         train_window=args.train_window,
                         train_anchors=args.train_anchors,
                         n_kv_heads=args.n_kv_heads)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_name")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--mlp-mult", type=int, default=4,
                    help="MLP 隐层倍数（ticket 24：100M 规格用 6）")
    ap.add_argument("--dense", action="store_true")
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

    run_dir = os.path.join(args.out_root, args.run_name)
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
    train_arr, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    cfg = build_depthar_config(args)
    model = DepthARModel(cfg, vocab_size).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.run_name}: {n_params/1e6:.2f}M params, vocab {vocab_size}, "
          f"topology {args.topology}, attn_sparse_k {args.attn_sparse_k}, "
          f"attn_linear {args.attn_linear} (chunk {args.linear_chunk}), "
          f"train_window {args.train_window}/+{args.train_anchors}, "
          f"n_kv_heads {args.n_kv_heads}, "
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
        final_anneal_to=(args.final_anneal_to if args.exit_anneal else None))
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
        "kv_head_liveness": _kv_head_liveness(model, vocab_size,
                                               seq_len=args.seq_len),
        "topology_liveness": topology_liveness(model),
        "exit_mode": args.exit_mode,
        "fill_kv": args.fill_kv, "proj_fill": args.proj_fill,
        "aux_weight": args.aux_weight, "aux_anneal_to": args.aux_anneal_to,
        "final_weight": args.final_weight,
        "readout_dims": args.readout_dims,
        "readout_gated": args.readout_gated,
        "readout_linear_resid": args.readout_linear_resid,
        "exit_anneal": args.exit_anneal,
        "exit_anneal_to": args.exit_anneal_to if args.exit_anneal else None,
        "exit_anneal_frac": args.exit_anneal_frac if args.exit_anneal else None,
        "final_anneal_to": args.final_anneal_to if args.exit_anneal else None,
        "shallow_stopgrad": args.shallow_stopgrad, "dense_aux": args.dense_aux,
        "gnorm_monitor": train_stats,
        "params": n_params, "vocab_size": vocab_size,
        "train_tokens": args.steps * args.batch_size * args.seq_len,
        "wall_minutes": round(wall / 60, 2),
        "avg_train_depth_ce": [round(c, 4) for c in avg_depth_ce],
        "avg_train_gate_mean": [round(g, 4) for g in avg_gate],
    }
    if not args.dense:
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
    else:
        curve = depth_curve(model, eval_arr, device="cuda", n_batches=args.eval_batches)
        # 23c H1（dense+aux）也需全深度曲线（depth-L 主数字 + depth-4 出口 bpc）
        results["depth_curve"] = curve
        results["bpc"] = curve[-1]["bpc"]
        results["next_token_acc"] = curve[-1]["acc"]
        # ticket 25 电池：dense 也报 tf（中间读出无监督，截断资产 proxy 口径）
        results["tf_acceptance"] = tf_acceptance(model, eval_arr, device="cuda",
                                                 n_batches=args.eval_batches)
        results["tf_note"] = ("dense 中间深度读出无直接监督：tf 系 depth-k "
                              "残差态 tied 读出与 depth-L 读出的 argmax 一致率 "
                              "proxy（截断资产口径，非训练出口资产）")
        # 内容 top-k 稀疏的活性块（SCHEMA §10）：真实前向的保留键数 + 选择统计
        results["attn_sparse_liveness"] = attn_sparse_probe(
            model, eval_arr, device="cuda")
        # 线性注意力的活性块（SCHEMA §10，契约 R）：φ/分母实测量程 + 同权重换算子的
        # 反事实偏差（rel_dev > 0 ⇒ 不是换个名字的全注意力）+ 核质量分布
        results["attn_linear_liveness"] = attn_linear_probe(
            model, eval_arr, device="cuda")
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("config", "train")}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
