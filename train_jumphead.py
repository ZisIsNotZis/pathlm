"""冻结主干 + 只训跳跃头（H4 线 A，ticket 28-pilot-batch1）。

契约：axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/README.md

命题：在一个**冻结**的 13M dense decoder-only 主干上只训一个预测 t+2 的跳跃
头，回答等 FLOPs 问题——「1 次前向 + 头」（成本 1）能否达到「2 次前向」
（成本 2）质量的 ≥0.81×？

实现要点（契约 §7 / 任务实现要点）：
- 主干 = 现有 dense 预设（DepthARModel(dense=True)，tied 读出、无 final LN）：
  复现 `train_depth_ar.py --dense` 的 dense 臂，不另起架构。
- 跳跃头 = d→d 残差变换（复用 `pathlm.model.TransformHead`）+ tied 读出
  （`logits = z @ embed.weightᵀ`，与主干 readout 同一几何/概率语义）。
- 监督目标：位置 i 的读出预测 t_{i+2}（shift=2；off-by-one 由 shift_pairs 单测钉住）。
- 冻结断言：主干全部参数 requires_grad=False，且一次 backward 后 .grad is None。
- 对照臂：主干自身 readout 的 t+1（等 FLOPs 分母）；主干 readout 直接预测 t+2
  （无头，头的能力分母）；同主干直接头 t+1；2 次前向（提交 t+1 再预测 t+2）。

用法：
  python train_jumphead.py --config <exp>/config.json --out-dir <exp> [--smoke]
"""

import argparse, hashlib, json, math, os, sys, time
from collections import defaultdict
from dataclasses import asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlm.data import batch, load_enwik8_full
from pathlm.depth_ar import DepthARConfig, DepthARModel, depth_curve
from pathlm.metrics import ece
from pathlm.model import TransformHead
from train_depth_ar import train as train_trunk_fn

LN2 = 0.6931471805599453


# ----------------------------------------------------------------- 冻结与主干


def trunk_hidden(trunk: DepthARModel, x: torch.Tensor) -> torch.Tensor:
    """dense 主干的最终层表征 [B,T,d]。

    与 DepthARModel.forward 的 dense 分支逐位一致（无门、无退出、纯残差堆叠），
    但跳过 forward 里的全深度读出头（13 个 fp32 读出头对头训练是纯浪费）。
    tests/test_jumphead.py::test_trunk_hidden_matches_forward_states 钉住等价。
    """
    h = trunk._input_states(x)
    for blk in trunk.blocks:
        h, _ = blk(h)
    return h


def freeze_trunk(trunk: nn.Module) -> None:
    """冻结主干：全部参数 requires_grad=False，并清掉训练残留的 .grad。

    清 .grad 是断言的一部分——主干训练后 .grad 仍挂在参数上，若不清，头反传后
    assert_frozen 会看到陈旧梯度而误判「未冻结」。
    """
    for p in trunk.parameters():
        p.requires_grad_(False)
        p.grad = None


def assert_frozen(trunk: nn.Module) -> tuple[bool, dict]:
    """冻结断言：主干无参数 requires_grad，且无参数字段 .grad 非 None。

    返回 (ok, offenders)。变异验证（放开主干梯度）时 ok 必须为 False。
    """
    unfrozen = [n for n, p in trunk.named_parameters() if p.requires_grad]
    graded = [n for n, p in trunk.named_parameters() if p.grad is not None]
    return (not unfrozen and not graded,
            {"requires_grad_true": unfrozen, "grad_not_none": graded})


# ----------------------------------------------------------------- 跳跃头


def shift_pairs(logits: torch.Tensor, tokens: torch.Tensor,
                shift: int) -> tuple[torch.Tensor, torch.Tensor]:
    """监督配对：行 i 的 logits 预测 tokens[:, i+shift]（shift=1 = t+1，shift=2 = 跳跃 t+2）。

    返回 (logits[:, :-shift], tokens[:, shift:])。方向由该函数唯一决定，
    单测 test_shift2_targets_are_t_plus_2 钉住（shift=2 的 CE≈0，shift=1 必须大）。
    """
    if not 1 <= shift < tokens.shape[1]:
        raise ValueError(f"shift must be in [1, T), got {shift}")
    return logits[:, :-shift], tokens[:, shift:]


class ResidualMLP(nn.Module):
    """d→d 残差 MLP 栈：h + Σ_b (Linear(hidden) → GELU → Linear(d))。

    容量消融用（契约 §8 失败决策树第一步）：默认 head（hidden_mult=2，
    n_blocks=1）与仓库 `pathlm.model.TransformHead` 同形；放大即单因素加容量。
    """

    def __init__(self, d_model: int, hidden_mult: int, n_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, hidden_mult * d_model), nn.GELU(),
                          nn.Linear(hidden_mult * d_model, d_model))
            for _ in range(n_blocks))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            h = h + b(h)
        return h


class JumpHead(nn.Module):
    """跳跃头：d→d 变换 + tied 读出。

    读出复用主干表 E（`logits = transform(h) @ Eᵀ`，fp32），与主干 readout
    保持概率语义一致——head 不持有自己的 unembed，语义上只是「主干外挂的
    状态变换 + 同一读出几何」。
    默认（hidden_mult=2，n_blocks=1）= 仓库现有 TransformHead（主臂）。
    """

    def __init__(self, d_model: int, hidden_mult: int = 2, n_blocks: int = 1):
        super().__init__()
        if hidden_mult == 2 and n_blocks == 1:
            self.transform = TransformHead(d_model)   # h + MLP(h)
        else:
            self.transform = ResidualMLP(d_model, hidden_mult, n_blocks)

    def forward(self, h: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        z = self.transform(h)
        with torch.autocast("cuda", enabled=False):
            return z.float() @ embed_weight.float().T


class HeadBank(nn.Module):
    """两条对照头共用一个冻结主干表征：t1（直接头，t+1）/ t2（跳跃头，t+2）。"""

    def __init__(self, d_model: int, hidden_mult: int = 2, n_blocks: int = 1):
        super().__init__()
        self.t1 = JumpHead(d_model, hidden_mult, n_blocks)
        self.t2 = JumpHead(d_model, hidden_mult, n_blocks)


def init_heads(bank: HeadBank) -> None:
    """与仓库其它头一致的初始化（std=0.02，bias 0）。"""
    for m in bank.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ----------------------------------------------------------------- 训练


def train_heads(trunk: DepthARModel, train_arr, hcfg: dict, seed: int,
                log_path: str, device: str = "cuda", frozen: bool = True) -> tuple[HeadBank, float, dict]:
    """在主干上训 t1/t2 两条头（一次前向同时监督两臂）。

    frozen=True（主臂）：主干前向 no_grad，优化器只含头参数，反传后断言主干无梯度；
    frozen=False（对照 3 联合臂）：主干参与梯度，优化器含头 + 主干参数。
    返回 (bank, wall, stats)。
    """
    steps, bs = hcfg["steps"], hcfg["batch_size"]
    seq = trunk.cfg.seq_len
    torch.manual_seed(seed)
    bank = HeadBank(trunk.cfg.d_model, hcfg.get("hidden_mult", 2),
                    hcfg.get("n_blocks", 1)).to(device)
    init_heads(bank)
    params = list(bank.parameters()) + ([] if frozen else list(trunk.parameters()))
    opt = torch.optim.AdamW(params, lr=hcfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200,
                           0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))))
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    loss_acc = 0.0
    frozen_checked = False
    for step in range(steps):
        bank.train()
        x, _ = batch(train_arr, bs, seq, gen)
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            if frozen:
                with torch.no_grad():
                    h = trunk_hidden(trunk, x)
            else:
                h = trunk_hidden(trunk, x)
            l1, y1 = shift_pairs(bank.t1(h, trunk.embed.weight), x, 1)
            l2, y2 = shift_pairs(bank.t2(h, trunk.embed.weight), x, 2)
            loss = F.cross_entropy(l1.reshape(-1, trunk.vocab_size), y1.reshape(-1)) \
                + F.cross_entropy(l2.reshape(-1, trunk.vocab_size), y2.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if frozen and not frozen_checked:
            ok, info = assert_frozen(trunk)
            if not ok:
                raise AssertionError(f"trunk not frozen after backward: {info}")
            frozen_checked = True
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        loss_acc += float(loss.detach())
        if step % 200 == 0 or step == steps - 1:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "step": step, "loss": round(float(loss.detach()), 4),
                    "lr": round(float(sched.get_last_lr()[0]), 6),
                    "min": round((time.time() - t0) / 60, 3)}) + "\n")
    wall = time.time() - t0
    return bank, wall, {"loss_mean": round(loss_acc / max(steps, 1), 4),
                        "steps": steps, "seed": seed, "frozen": frozen}


# ----------------------------------------------------------------- 评测


def _bins(conf: torch.Tensor, hit: torch.Tensor, n_bins: int = 10) -> list[dict]:
    """按置信度分层的命中率（分层单调性检查）。"""
    out = []
    edges = torch.linspace(0, 1, n_bins + 1)
    for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        n = int(m.sum())
        out.append({"bin": k, "lo": round(float(lo), 2), "hi": round(float(hi), 2),
                    "n": n,
                    "mean_conf": round(float(conf[m].mean()), 4) if n else None,
                    "hit": round(float(hit[m].float().mean()), 4) if n else None})
    return out


@torch.no_grad()
def evaluate(trunk: DepthARModel, bank: HeadBank, eval_arr, ecfg: dict,
             device: str = "cuda") -> dict:
    """全部对照臂的 argmax 命中率 + 校准。行 i 统一定义为预测 t_{i+1}/t_{i+2}。"""
    trunk.eval()
    bank.eval()
    gen = torch.Generator().manual_seed(ecfg["seed"])
    T = trunk.cfg.seq_len
    r = T - 2                                    # 行 0..T-3（t+1 与 t+2 共同可用行）
    nb, bs = ecfg["n_batches"], ecfg["batch_size"]
    E = trunk.embed.weight
    acc = defaultdict(float)
    n_pos = 0
    conf_chunks, hit_chunks = [], []
    trunk_t1_hit_chunks, jump_hit_chunks = [], []
    for _ in range(nb):
        x, _ = batch(eval_arr, bs, T, gen)
        x = x.to(device)
        h = trunk_hidden(trunk, x)
        L = trunk.readout(h)                     # [B,T,V] fp32
        xt1, xt2 = x[:, 1:r + 1], x[:, 2:r + 2]  # t+1 / t+2 目标（行 0..r-1）
        row = slice(0, r)
        # 主干自身 readout
        a1 = L[:, row].argmax(-1)
        acc["trunk_t1"] += (a1 == xt1).sum().item()
        acc["trunk_t2_naive"] += (a1 == xt2).sum().item()   # 无头、1 次前向、直接目标 t+2
        t1_hit = (a1 == xt1)
        # 直接头 t+1
        l1 = bank.t1(h, E)[:, row].argmax(-1)
        acc["head_t1"] += (l1 == xt1).sum().item()
        # 跳跃头 t+2
        jl = bank.t2(h, E)                       # [B,T,V]
        j = jl[:, row].argmax(-1)
        jhit = (j == xt2)
        acc["jump_t2"] += jhit.sum().item()
        # 2 次前向（提交 t+1 再预测 t+2）：把模型自己预测的 t+1 放回输入，再跑一次
        xp = x.clone()
        xp[:, 1:] = L[:, :-1].argmax(-1)
        L2 = trunk.readout(trunk_hidden(trunk, xp))
        acc["two_forward_deploy_t2"] += (L2[:, 1:T - 1].argmax(-1) == xt2).sum().item()
        acc["two_forward_oracle_t2"] += (L[:, 1:T - 1].argmax(-1) == xt2).sum().item()
        # 条件接受率 + 校准收集
        acc["jump_t2_given_trunk_t1"] += (jhit & t1_hit).sum().item()
        acc["n_trunk_t1"] += t1_hit.sum().item()
        conf_chunks.append(jl[:, row].float().softmax(-1).max(-1).values.cpu())
        hit_chunks.append(jhit.cpu())
        trunk_t1_hit_chunks.append(t1_hit.cpu())
        jump_hit_chunks.append(jhit.cpu())
        n_pos += (bs * r)
    conf = torch.cat(conf_chunks).flatten()
    jhit_all = torch.cat(hit_chunks).flatten()
    t1_all = torch.cat(trunk_t1_hit_chunks).flatten()
    out = {
        "n_positions": n_pos,
        "trunk_t1_hit": round(acc["trunk_t1"] / n_pos, 4),
        "trunk_t2_naive_hit": round(acc["trunk_t2_naive"] / n_pos, 4),
        "head_t1_hit": round(acc["head_t1"] / n_pos, 4),
        "jump_t2_hit": round(acc["jump_t2"] / n_pos, 4),
        "two_forward_deploy_t2_hit": round(acc["two_forward_deploy_t2"] / n_pos, 4),
        "two_forward_oracle_t2_hit": round(acc["two_forward_oracle_t2"] / n_pos, 4),
        "conditional_accept_given_trunk_t1": round(
            acc["jump_t2_given_trunk_t1"] / max(acc["n_trunk_t1"], 1.0), 4),
        "jump_t2_ece": round(ece(conf, jhit_all), 4),
        "jump_t2_conf_bins": _bins(conf, jhit_all),
        "corr_jump_hit__trunk_t1_hit": round(
            float(((t1_all.float() - t1_all.float().mean())
                   * (jhit_all.float() - jhit_all.float().mean())).mean()
                  / (t1_all.float().std() * jhit_all.float().std() + 1e-12)), 4),
    }
    return out


# ----------------------------------------------------------------- 主流程


def _verdict(numer: float, threshold: float, bpc_delta: float) -> str:
    """契约 §4 的三分判决（预先写下的线，代码只做归类）。

    成功 = 命中 ≥ 0.81×分母 且 |Δbpc| < 0.014；
    中性 = 命中 ∈ [0.557, 0.60)；否则失败（无头寸 / 零税不成立）。
    """
    if abs(bpc_delta) >= 0.014:
        return "失败：零税不成立"
    if numer >= threshold:
        return "成功"
    if numer >= 0.557:
        return "中性"
    return "失败：冻结主干上跳跃头无头寸"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trunk-steps", type=int, default=None)
    ap.add_argument("--head-steps", type=int, default=None)
    ap.add_argument("--joint-steps", type=int, default=0,
                    help="对照 3（可选）：从已训主干出发、放开主干联合微调 t+2 头的步数；0 = 跳过")
    ap.add_argument("--trunk-ckpt", default=None,
                    help="已训主干权重路径；给定则跳过主干训练（容量消融复用同一冻结主干）")
    ap.add_argument("--head-hidden-mult", type=int, default=None, help="头 MLP 隐层倍数（缺省 2）")
    ap.add_argument("--head-blocks", type=int, default=None, help="头残差块数（缺省 1）")
    ap.add_argument("--smoke", action="store_true",
                    help="20 步冒烟：主干 20 步 + 头 20 步 + 小评测，打印步时与显存")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    if args.smoke:
        cfg["trunk_train"]["steps"] = 20
        cfg["head_train"]["steps"] = 20
        cfg["eval"]["n_batches"] = 2
        cfg["eval"]["batch_size"] = 4
    if args.head_hidden_mult is not None:
        cfg["head_train"]["hidden_mult"] = args.head_hidden_mult
    if args.head_blocks is not None:
        cfg["head_train"]["n_blocks"] = args.head_blocks
    if args.trunk_steps is not None:
        cfg["trunk_train"]["steps"] = args.trunk_steps
    if args.head_steps is not None:
        cfg["head_train"]["steps"] = args.head_steps

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(cfg["trunk_train"]["seed"])
    train_arr, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")

    tcfg = dict(cfg["trunk_train"])
    tcfg["seed"] = int(tcfg["seed"])
    trunk = DepthARModel(DepthARConfig(**cfg["trunk"]), vocab_size).cuda()
    n_params = sum(p.numel() for p in trunk.parameters())
    print(f"trunk: {n_params/1e6:.3f}M params, vocab {vocab_size}, "
          f"{tcfg['steps']} steps x bs{tcfg['batch_size']} x seq{cfg['trunk']['seq_len']}",
          flush=True)

    # --- 主干训练（复用 train_depth_ar 的 dense 臂）；--trunk-ckpt 时直接加载 ---
    torch.cuda.reset_peak_memory_stats()
    if args.trunk_ckpt:
        trunk.load_state_dict(torch.load(args.trunk_ckpt, map_location="cuda"))
        wall_trunk, tstats = 0.0, {"loaded_from": args.trunk_ckpt}
        print(f"trunk loaded from {args.trunk_ckpt} (training skipped)", flush=True)
    else:
        wall_trunk, avg_ce, _, tstats = train_trunk_fn(
            trunk, train_arr, tcfg,
            os.path.join(args.out_dir, "trunk_train_log.jsonl"),
            os.path.join(args.out_dir, "trunk_gate_log.jsonl"),
            exit_mode="uniform")
    peak_train_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    trunk_ckpt = os.path.join(args.out_dir, "trunk.pt")
    torch.save(trunk.state_dict(), trunk_ckpt)
    trunk_sha = sha256_file(trunk_ckpt)
    trunk_size = os.path.getsize(trunk_ckpt)
    print(f"trunk done in {wall_trunk/60:.2f} min, peak {peak_train_mb:.0f} MiB, "
          f"sha256 {trunk_sha[:12]} ({trunk_size/1e6:.1f} MB)", flush=True)

    # --- 冻结 + 挂头前的主干 bpc（零税基线）---
    curve_before = depth_curve(trunk, eval_arr, n_batches=cfg["eval"]["n_batches"],
                               device="cuda")
    bpc_before = curve_before[-1]["bpc"]
    freeze_trunk(trunk)
    ok, info = assert_frozen(trunk)
    print(f"freeze assertion: ok={ok} info={info}", flush=True)

    # --- 头训练（2 seeds）---
    per_seed = {}
    head_ckpts = {}
    head_cfg = cfg["head_train"]
    for seed in head_cfg["seeds"]:
        bank, wall_head, hstats = train_heads(
            trunk, train_arr, head_cfg, int(seed),
            os.path.join(args.out_dir, f"head_train_log_s{seed}.jsonl"))
        bank_ckpt = os.path.join(args.out_dir, f"head_s{seed}.pt")
        torch.save(bank.state_dict(), bank_ckpt)
        head_ckpts[str(seed)] = {"path": os.path.basename(bank_ckpt),
                                 "sha256": sha256_file(bank_ckpt),
                                 "size_bytes": os.path.getsize(bank_ckpt)}
        ev = evaluate(trunk, bank, eval_arr, cfg["eval"])
        per_seed[str(seed)] = {"head_train": hstats,
                               "wall_head_minutes": round(wall_head / 60, 2), **ev}
        print(f"seed {seed}: jump_t2={ev['jump_t2_hit']:.4f} "
              f"trunk_t1={ev['trunk_t1_hit']:.4f} head={wall_head/60:.2f}min", flush=True)

    # --- 挂头后主干 bpc（零税验证：主干冻结，bpc 必须逐位不变）---
    curve_after = depth_curve(trunk, eval_arr, n_batches=cfg["eval"]["n_batches"],
                              device="cuda")
    bpc_after = curve_after[-1]["bpc"]

    mean = lambda key: round(sum(per_seed[s][key] for s in per_seed) / len(per_seed), 4)
    denom = mean("trunk_t1_hit")
    numer = mean("jump_t2_hit")
    threshold = round(0.81 * denom, 4)

    # --- 对照 3（可选）：联合微调 ---
    joint = None
    if args.joint_steps > 0:
        for p in trunk.parameters():
            p.requires_grad_(True)
        jbank = HeadBank(trunk.cfg.d_model).cuda()
        init_heads(jbank)
        jcfg = dict(head_cfg)
        jcfg["steps"] = args.joint_steps
        jcfg["lr"] = 3e-4
        jbank, wall_joint, jstats = train_heads(
            trunk, train_arr, jcfg, int(head_cfg["seeds"][0]),
            os.path.join(args.out_dir, "joint_train_log.jsonl"), frozen=False)
        joint = {"wall_minutes": round(wall_joint / 60, 2), "train": jstats, **evaluate(
            trunk, jbank, eval_arr, cfg["eval"])}
        # 记录联合臂的主干 bpc（主干已被改动，冻结臂的 zero-tax 结论不适用）
        joint["trunk_bpc"] = depth_curve(trunk, eval_arr,
                                         n_batches=cfg["eval"]["n_batches"],
                                         device="cuda")[-1]["bpc"]

    results = {
        "contract": ("axes/H-horizon/H4-head-trunk-relation/experiments/"
                     "2610-frozen-trunk-jumphead-13m/README.md"),
        "config": cfg,
        "trunk": {
            "params": n_params, "sha256": trunk_sha, "size_bytes": trunk_size,
            "steps": tcfg["steps"], "batch_size": tcfg["batch_size"],
            "seq_len": cfg["trunk"]["seq_len"],
            "tokens": tcfg["steps"] * tcfg["batch_size"] * cfg["trunk"]["seq_len"],
            "wall_minutes": round(wall_trunk / 60, 2),
            "peak_mem_mib": round(peak_train_mb, 1),
            "train_stats": tstats,
            "bpc": bpc_before, "t1_acc": curve_before[-1]["acc"],
        },
        "zero_tax": {
            "bpc_before_head": bpc_before, "bpc_after_head": bpc_after,
            "delta_bpc": round(bpc_after - bpc_before, 6),
            "frozen_assertion_ok": bool(ok),
        },
        "heads": head_ckpts,
        "arms": per_seed,
        "arms_mean": {k: mean(k) for k in
                      ("trunk_t1_hit", "trunk_t2_naive_hit", "head_t1_hit",
                       "jump_t2_hit", "two_forward_deploy_t2_hit",
                       "two_forward_oracle_t2_hit",
                       "conditional_accept_given_trunk_t1", "jump_t2_ece")},
        "criteria": {
            "denominator_trunk_t1_hit": denom,
            "threshold_0.81x": threshold,
            "numerator_jump_t2_hit": numer,
            "ratio_vs_denominator": round(numer / denom, 4) if denom else None,
            "zero_tax_ok": abs(bpc_after - bpc_before) < 0.014,
            "verdict": _verdict(numer, threshold, bpc_after - bpc_before),
        },
        "joint_arm": joint,
        "cost": {"trunk_min": round(wall_trunk / 60, 2),
                 "peak_mem_mib": round(peak_train_mb, 1)},
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("config", "arms")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
