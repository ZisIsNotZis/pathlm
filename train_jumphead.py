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
from pathlm.model import TransformHead, cap_norm
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


def trunk_layer_inputs(trunk: DepthARModel, x: torch.Tensor
                       ) -> tuple[list[torch.Tensor], torch.Tensor]:
    """逐层输入表征 + 末态。inputs[l] = 第 l 个 block 的输入 [B,T,d]，final = 末态。

    trunk_layer_inputs(...)[1] 与 trunk_hidden(...) 逐位一致（单测钉住）。
    AR-1 对照臂用 inputs 拿真值前缀的逐层 K/V。
    """
    h = trunk._input_states(x)
    inputs: list[torch.Tensor] = []
    for blk in trunk.blocks:
        inputs.append(h)
        h, _ = blk(h)
    return inputs, h


@torch.no_grad()
def ar1_second_step_readout(trunk: DepthARModel, x: torch.Tensor,
                            a1: torch.Tensor) -> torch.Tensor:
    """诚实 1-step AR 的第二次前向：**只把位置 i+1 替换**为 pass-1 自身 argmax。

    与 ticket 28 的旧 deploy 臂（`xp[:, 1:] = L[:, :-1].argmax(-1)`，把**整段
    上下文**都换成自预测）不同：这里对每个目标行 i，位置 0..i 保持**真值**，
    只有紧邻的前一个位置 i+1 被替换为 `a1[:, i]`，再读出位置 i+1 的 logits
    （预测 t_{i+2}）。这正是「自回归跑两次、第二次只吃自己生成的那**一个**
    token」的真实对手。

    工程实现：因果注意力下，真值前缀位置 ≤ i 的表征只依赖 x[0..i]，故它们的
    逐层 K/V 由第一次前向的逐层输入唯一确定；新位置 i+1 的表征只需在本层
    attention 里查询真值前缀 0..i 加上自身（KV-cache 语义一次算完所有行 i）。
    与「逐行各跑一次标准前向」逐位等价（test_ar1_matches_bruteforce_rowloop）。

    a1: [B,T]，a1[:,i] = pass-1 对 t_{i+1} 的 argmax。返回 logits [B,T-2,V]，
    行 i = 对 x[i+2] 的预测。
    """
    B, T = x.shape
    r = T - 2
    if r < 1:
        raise ValueError(f"seq_len must be >= 3, got {T}")
    cfg = trunk.cfg
    d, H = cfg.d_model, cfg.n_heads
    dh = d // H
    scale = dh ** -0.5
    inputs, _ = trunk_layer_inputs(trunk, x)     # 真值前缀逐层输入（K/V 来源）
    tok = a1[:, :r].to(x.device)                 # 新位置 i+1 的 token = a1[:,i]
    pos = torch.arange(1, r + 1, device=x.device)
    new = cap_norm(trunk.embed(tok) + trunk.pos_embed.weight[pos], cfg.norm_cap)
    causal = (torch.arange(T, device=x.device).unsqueeze(0)
              <= torch.arange(r, device=x.device).unsqueeze(1))   # [r,T] 键 j ≤ i
    for l, blk in enumerate(trunk.blocks):
        qp, kp, vp = blk.qkv(blk.ln1(inputs[l])).chunk(3, dim=-1)
        qn, kn, vn = blk.qkv(blk.ln1(new)).chunk(3, dim=-1)
        qn_h = qn.view(B, r, H, dh).transpose(1, 2)             # [B,H,r,dh]
        kp_h = kp.view(B, T, H, dh).permute(0, 2, 3, 1)         # [B,H,dh,T]
        vp_h = vp.view(B, T, H, dh).transpose(1, 2)            # [B,H,T,dh]
        pref = (qn_h @ kp_h) * scale                           # [B,H,r,T]
        pref = pref.masked_fill(~causal.view(1, 1, r, T), float("-inf"))
        kn_h = kn.view(B, r, H, dh).transpose(1, 2)
        self_s = (qn_h * kn_h).sum(-1, keepdim=True) * scale   # [B,H,r,1]
        w = torch.cat([pref, self_s], dim=-1).softmax(-1)      # [B,H,r,T+1]
        out = (torch.einsum("bhij,bhjd->bhid", w[..., :T], vp_h)
               + w[..., T:] * vn.view(B, r, H, dh).transpose(1, 2))
        a = out.transpose(1, 2).reshape(B, r, d)
        new = new + blk.proj(a)
        new = new + blk.mlp(blk.ln2(new))
    return trunk.readout(new)                    # [B,r,V]


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


# ------------------------------------------- 契约 30-G：自门控多跳组合（H3）
#
# 命题：预测 t+2 可以「先后跳 1 两次」（T∘T），也可以「直接跳两步」；由逐通道门
# 在两者之间插值。三臂（direct / composed / gated）参数量对齐（direct 与 composed
# 逐位相同；gated 多一个门 G），外加一个 §15 等参数对照臂 direct_eq。

COMPOSE_MODES = ("direct", "composed", "gated", "direct_eq")


def tied_logits(z: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
    """tied 读出（fp32，autocast 之外）：logits = z @ Eᵀ。与 JumpHead 同几何。"""
    with torch.autocast("cuda", enabled=False):
        return z.float() @ embed_weight.float().T


class WideDirectHead(nn.Module):
    """§15 等参数对照：去掉门控/组合结构，把参数花在更宽的 MLP 上。

    参数量 = MLP 隐层 3d（6d²+4d）+ 逐通道偏置（d）= 6d²+5d，与 gated 臂
    （T 的 4d²+3d + G 的 2d²+2d = 6d²+5d）**逐位相同**；d=256 ⇒ 394,496。
    逐通道偏置是为逐位对齐而加的通用容量项（不是门控结构）。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, 3 * d_model), nn.GELU(),
                                 nn.Linear(3 * d_model, d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.mlp(h) + self.bias


class BranchGate(nn.Module):
    """逐通道自门控 G(h) = fc2(GELU(fc1(h)))，g = sigmoid(G(h)) ∈ (0,1)。

    单边零初始化（SCHEMA §10）：输出侧 fc2 = 0，输入侧 fc1 ~ N(0, 0.02)。
    起点 G(h) ≡ 0 ⇒ g ≡ 0.5（direct 与 composed 的 50/50 混合，**不是**任一臂
    的逐位等价——sigmoid 门下要 g≡0 只能靠负偏置，那就不是单边零；见 README §4）。
    fc2 在首步即拿到非零梯度（∂L/∂fc2 ∝ g'(0)·(T∘T(h)−T(h))·upstream ≠ 0，
    两分支不同 ⇒ 非零），所以不是鞍点。
    tests/test_jumphead.py::test_gate_single_sided_zero_escapes 钉住（含变异版）。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(h)))


class CompositionHead(nn.Module):
    """契约 30-G 的组合头：一个 T（+ 可选门）作用一次/两次，tied 读出 t+2。

    direct    : z = T(h)                     （T 作用一次）
    composed  : z = T(T(h))                  （同一个 T 作用两次，参数与 direct 逐位相同）
    gated     : z = g⊙T(T(h)) + (1−g)⊙T(h)   （g = sigmoid(G(h)) 逐通道）
    direct_eq : z = WideDirectHead(h)         （§15 等参数对照：无门、无组合）
    """

    def __init__(self, d_model: int, mode: str):
        super().__init__()
        if mode not in COMPOSE_MODES:
            raise ValueError(f"unknown composition mode {mode!r} (expect {COMPOSE_MODES})")
        self.mode = mode
        self.T = WideDirectHead(d_model) if mode == "direct_eq" else TransformHead(d_model)
        self.G = BranchGate(d_model) if mode == "gated" else None

    def branches(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(z1, z2) = (T(h), T(T(h)))；非组合臂的 z2 为 None。"""
        z1 = self.T(h)
        z2 = self.T(z1) if self.mode in ("composed", "gated") else None
        return z1, z2

    def mixture(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """返回 (z_mix, g)：g 只对 gated 臂非 None（逐通道门开度 [.., d]）。"""
        z1, z2 = self.branches(h)
        if z2 is None:
            return z1, None
        if self.mode == "composed":
            return z2, None
        g = torch.sigmoid(self.G(h))
        return g * z2 + (1.0 - g) * z1, g

    def forward(self, h: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        z, _ = self.mixture(h)
        return tied_logits(z, embed_weight)


def init_composition_head(head: CompositionHead) -> None:
    """头初始化（std=0.02，bias 0）+ 门的单边零（输出侧 fc2 清零）。"""
    for m in head.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    if head.G is not None:
        nn.init.zeros_(head.G.fc2.weight)
        nn.init.zeros_(head.G.fc2.bias)


def param_stats(module: nn.Module) -> dict:
    """§10 活性量：参数量 / Frobenius 范数 / 逐位最大绝对值 / 非零占比。"""
    flat = torch.cat([p.detach().float().flatten() for p in module.parameters()])
    return {"params": int(flat.numel()), "w_fro": round(float(flat.norm()), 4),
            "w_absmax": round(float(flat.abs().max()), 6),
            "nonzero_frac": round(float((flat != 0).float().mean()), 6)}


def head_activity(head: CompositionHead) -> dict:
    """干预活性块（T 与门分开报；门的开度分布在评测里报）。"""
    out = {"mode": head.mode,
           "params": sum(p.numel() for p in head.parameters()),
           "T": param_stats(head.T)}
    if head.G is not None:
        out["G"] = param_stats(head.G)
    return out


# ----------------------------------------------------------------- 训练


def train_composition_head(trunk: DepthARModel, train_arr, hcfg: dict, seed: int,
                           mode: str, log_path: str,
                           device: str = "cuda") -> tuple[CompositionHead, float, dict]:
    """在冻结主干上只训一个 t+2 组合头（契约 30-G）。loss = CE(t+2)，无 t+1 头。

    与 train_heads 的区别只有两处：① 只监督 t+2；② 头是 CompositionHead(mode)。
    其余（冻结主干 / AdamW / 同一 cosine 调度 / 同一 batch 流 / 同一日志格式）
    逐字段一致，使三个臂与等参数对照臂互为单因素对照。
    """
    steps, bs = hcfg["steps"], hcfg["batch_size"]
    seq = trunk.cfg.seq_len
    torch.manual_seed(seed)
    head = CompositionHead(trunk.cfg.d_model, mode).to(device)
    init_composition_head(head)
    opt = torch.optim.AdamW(head.parameters(), lr=hcfg["lr"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200,
                           0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps))))
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    loss_acc = 0.0
    frozen_checked = False
    for step in range(steps):
        head.train()
        x, _ = batch(train_arr, bs, seq, gen)
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=torch.cuda.is_available()):
            with torch.no_grad():
                h = trunk_hidden(trunk, x)
            z, _ = head.mixture(h)
            l2, y2 = shift_pairs(tied_logits(z, trunk.embed.weight), x, 2)
            loss = F.cross_entropy(l2.reshape(-1, trunk.vocab_size), y2.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not frozen_checked:
            ok, info = assert_frozen(trunk)
            if not ok:
                raise AssertionError(f"trunk not frozen after backward: {info}")
            frozen_checked = True
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
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
    return head, wall, {"loss_mean": round(loss_acc / max(steps, 1), 4),
                        "steps": steps, "seed": seed, "mode": mode,
                        "frozen": True, "activity": head_activity(head)}


def loss_slope_from_log(log_path: str, tail_frac: float = 0.25) -> dict:
    """§16 收敛对账：同一窗口（末 tail_frac）、同一算法（对日志点做 OLS）的 loss 斜率。

    所有臂共用同一日志间隔（每 200 步）与同一窗口定义，斜率可直接横比。
    返回斜率（每 1k 步的 loss 变化，负 = 还在下降）与该窗口的点数。
    """
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    if not rows:
        return {"slope_per_1k": None, "n_points": 0, "window": None}
    steps = max(r["step"] for r in rows)
    lo = (1.0 - tail_frac) * steps
    win = [(r["step"], r["loss"]) for r in rows if r["step"] >= lo]
    n = len(win)
    if n < 2:
        return {"slope_per_1k": None, "n_points": n,
                "window": [round(lo, 1), steps]}
    mx = sum(s for s, _ in win) / n
    my = sum(v for _, v in win) / n
    den = sum((s - mx) ** 2 for s, _ in win)
    slope = sum((s - mx) * (v - my) for s, v in win) / den if den else 0.0
    return {"slope_per_1k": round(slope * 1000, 5), "n_points": n,
            "window": [round(lo, 1), steps],
            "tail_mean_loss": round(my, 4)}


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


# ------------------------------------------------------ 契约 29-A：诚实 AR-1 对照


def equiv_speedup_from_dims(trunk: DepthARModel, head: JumpHead) -> dict:
    """产 1 个 token 的等效成本比（contract A 口径）。

    跳跃头臂（1 次主干前向 + 头前向）同时给出 t+1 与 t+2 ⇒ 每 token 成本
    = trunk_macs + head_macs；AR 臂（2 次主干前向）给出同样两个 token ⇒
    每 token 成本 = 2·trunk_macs。比值 > 1 表示跳跃头更便宜。
    """
    cfg = trunk.cfg
    d, L, ml, V = cfg.d_model, cfg.n_layers, cfg.mlp_mult, trunk.vocab_size
    trunk_macs = L * (3 * d * d + d * d + 2 * ml * d * d) + d * V   # qkv+proj+mlp+readout
    head_macs = 4 * d * d + d * V                                   # TransformHead(2d)+tied readout
    return {"trunk_macs_per_token": trunk_macs,
            "head_macs_per_token": head_macs,
            "forward_calls_jump": 1, "forward_calls_ar": 2,
            "equiv_speedup": round(2 * trunk_macs / (trunk_macs + head_macs), 4)}


@torch.no_grad()
def evaluate_ar1(trunk: DepthARModel, bank: HeadBank, eval_arr, ecfg: dict,
                 device: str = "cuda") -> dict:
    """契约 29-A 的量（全部在「行 i 预测 t_{i+2}」的同一位置集上打分）。

    - jump_t2_hit：跳跃头 1 前向 argmax == 真值 t+2（质量口径）
    - ar1_deploy_t2_hit：诚实 1-step AR（只替换位置 i+1）argmax == 真值 t+2（主对手）
    - accept_jump_t2：P(跳跃头 argmax == 主干自身 1-step AR argmax)（草稿接受率，无作弊）
    - oracle_t2_hit：喂真 t+1 的上界（仅参考，不作判据）
    """
    trunk.eval()
    bank.eval()
    gen = torch.Generator().manual_seed(ecfg["seed"])
    T = trunk.cfg.seq_len
    r = T - 2
    E = trunk.embed.weight
    acc = defaultdict(float)
    n_pos = 0
    for _ in range(ecfg["n_batches"]):
        x, _ = batch(eval_arr, ecfg["batch_size"], T, gen)
        x = x.to(device)
        h = trunk_hidden(trunk, x)
        L = trunk.readout(h)                     # [B,T,V] fp32
        xt1, xt2 = x[:, 1:r + 1], x[:, 2:r + 2]  # 行 i 的 t+1 / t+2 目标
        row = slice(0, r)
        a1 = L.argmax(-1)                        # [B,T] pass-1 自身 argmax（行 i → t_{i+1}）
        acc["trunk_t1_hit"] += (a1[:, row] == xt1).sum().item()
        # 跳跃头 t+2（1 前向）
        j = bank.t2(h, E)[:, row].argmax(-1)     # [B,r]
        jhit = (j == xt2)
        acc["jump_t2"] += jhit.sum().item()
        # 诚实 1-step AR：只替换位置 i+1，保留真值前缀
        ar1 = ar1_second_step_readout(trunk, x, a1).argmax(-1)   # [B,r]
        acc["ar1_deploy_t2"] += (ar1 == xt2).sum().item()
        acc["accept_jump"] += (j == ar1).sum().item()           # 草稿接受率（对照物 = 主干自身）
        # oracle（喂真 t+1，仅上界参考）
        orc = L[:, 1:r + 1].argmax(-1)
        acc["oracle_t2"] += (orc == xt2).sum().item()
        acc["accept_jump_vs_oracle"] += (j == orc).sum().item()
        n_pos += x.shape[0] * r
    out = {
        "n_positions": n_pos,
        "trunk_t1_hit": round(acc["trunk_t1_hit"] / n_pos, 4),
        "jump_t2_hit": round(acc["jump_t2"] / n_pos, 4),
        "ar1_deploy_t2_hit": round(acc["ar1_deploy_t2"] / n_pos, 4),
        "accept_jump_t2": round(acc["accept_jump"] / n_pos, 4),
        "oracle_t2_hit": round(acc["oracle_t2"] / n_pos, 4),
        "accept_jump_t2_vs_oracle": round(acc["accept_jump_vs_oracle"] / n_pos, 4),
        "ratio_jump_over_ar1": round(acc["jump_t2"] / max(acc["ar1_deploy_t2"], 1.0), 4),
    }
    out["equiv_speedup"] = equiv_speedup_from_dims(trunk, bank.t2)["equiv_speedup"]
    return out


# -------------------------------------- 契约 30-G：三臂同位置集评测


@torch.no_grad()
def evaluate_composition(trunk: DepthARModel, heads: dict[str, CompositionHead],
                         eval_arr, ecfg: dict, g_bins: int = 10,
                         device: str = "cuda") -> dict:
    """三臂 + 等参数对照臂在**同一位置集**（同一 eval seed / 批次顺序）上打分。

    位置集：行 i 预测 t_{i+2}（i = 0..T-3），与 2610/2611 逐位相同。
    对照物（§11/§13）：
    - ar1_deploy：主干自身 1-step AR（只替换紧邻位置 i+1）——诚实对手，草稿接受率的
      分母就是它（= 头 argmax == 主干自身 argmax），无真值参与；
    - oracle：喂真 t+1 —— 只是上界参考，不作判据。
    两臂共享的量（trunk_t1 / ar1 / oracle）只算一次，所有臂吃同一批位置。
    """
    trunk.eval()
    for hd in heads.values():
        hd.eval()
    gen = torch.Generator().manual_seed(ecfg["seed"])
    T = trunk.cfg.seq_len
    r = T - 2
    V = trunk.vocab_size
    E = trunk.embed.weight
    shared = defaultdict(float)
    per_arm = {m: defaultdict(float) for m in heads}
    ce_sum = {m: 0.0 for m in heads}
    g_scalar, g_hit, g_b1, g_b2, g_chan = [], [], [], [], []
    n_pos = 0
    for _ in range(ecfg["n_batches"]):
        x, _ = batch(eval_arr, ecfg["batch_size"], T, gen)
        x = x.to(device)
        h = trunk_hidden(trunk, x)
        L = trunk.readout(h)
        xt1, xt2 = x[:, 1:r + 1], x[:, 2:r + 2]
        row = slice(0, r)
        a1 = L.argmax(-1)
        shared["trunk_t1_hit"] += (a1[:, row] == xt1).sum().item()
        ar1 = ar1_second_step_readout(trunk, x, a1).argmax(-1)     # [B,r]
        shared["ar1_deploy_t2_hit"] += (ar1 == xt2).sum().item()
        orc = L[:, 1:r + 1].argmax(-1)
        shared["oracle_t2_hit"] += (orc == xt2).sum().item()
        for mode, head in heads.items():
            z, g = head.mixture(h)
            logits = tied_logits(z, E)[:, row]                     # [B,r,V]
            j = logits.argmax(-1)
            per_arm[mode]["t2_hit"] += (j == xt2).sum().item()
            per_arm[mode]["accept_vs_ar1"] += (j == ar1).sum().item()
            per_arm[mode]["accept_vs_oracle"] += (j == orc).sum().item()
            ce_sum[mode] += F.cross_entropy(
                logits.reshape(-1, V), xt2.reshape(-1), reduction="sum").item()
            if g is not None:
                z1, z2 = head.branches(h)
                b1 = tied_logits(z1, E)[:, row].argmax(-1)
                b2 = tied_logits(z2, E)[:, row].argmax(-1)
                gv = g[:, row]                                     # [B,r,d]
                g_scalar.append(gv.mean(-1).flatten().cpu())
                g_chan.append(gv.mean(dim=(0, 1)).cpu())
                g_hit.append((j == xt2).flatten().cpu())
                g_b1.append((b1 == xt2).flatten().cpu())
                g_b2.append((b2 == xt2).flatten().cpu())
        n_pos += x.shape[0] * r

    out = {"n_positions": n_pos, "shared": {k: round(v / n_pos, 4) for k, v in shared.items()},
           "arms": {}}
    for mode in heads:
        out["arms"][mode] = {k: round(v / n_pos, 4) for k, v in per_arm[mode].items()}
        out["arms"][mode]["eval_ce_t2"] = round(ce_sum[mode] / n_pos, 4)
        out["arms"][mode]["activity"] = head_activity(heads[mode])
    if g_scalar:
        gs = torch.cat(g_scalar)
        hit = torch.cat(g_hit)
        b1_ok = torch.cat(g_b1)
        b2_ok = torch.cat(g_b2)
        cm = torch.stack(g_chan).mean(0)                          # 逐通道跨位置均值 [d]
        q = torch.quantile(gs, torch.tensor([0.1, 0.5, 0.9]))
        edges = torch.linspace(0, 1, g_bins + 1)
        bins = []
        for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            m = (gs >= lo) & (gs <= hi) if k == 0 else (gs > lo) & (gs <= hi)
            n = int(m.sum())
            bins.append({"bin": k, "g_lo": round(float(lo), 2), "g_hi": round(float(hi), 2),
                         "n": n,
                         "g_mean": round(float(gs[m].mean()), 4) if n else None,
                         "t2_hit": round(float(hit[m].float().mean()), 4) if n else None,
                         "direct_branch_hit": round(float(b1_ok[m].float().mean()), 4) if n else None,
                         "composed_branch_hit": round(float(b2_ok[m].float().mean()), 4) if n else None})
        # 可控性：门能不能指向「那一步正确的分支」
        comp_only = b2_ok & (~b1_ok)          # composed 分支对、direct 分支错
        dir_only = b1_ok & (~b2_ok)           # direct 分支对、composed 分支错
        out["gated_g"] = {
            "n_positions": int(gs.numel()),
            "direct_branch_hit_overall": round(float(b1_ok.float().mean()), 4),
            "composed_branch_hit_overall": round(float(b2_ok.float().mean()), 4),
            "mean": round(float(gs.mean()), 4), "std": round(float(gs.std()), 4),
            "q10": round(float(q[0]), 4), "q50": round(float(q[1]), 4),
            "q90": round(float(q[2]), 4),
            "frac_prefer_two_hops": round(float((gs > 0.5).float().mean()), 4),
            "channel_mean_min": round(float(cm.min()), 4),
            "channel_mean_max": round(float(cm.max()), 4),
            "dead_channels_frac": round(float(((cm < 0.05) | (cm > 0.95)).float().mean()), 4),
            "n_composed_branch_better": int(comp_only.sum()),
            "n_direct_branch_better": int(dir_only.sum()),
            "g_mean_when_composed_branch_better": (
                round(float(gs[comp_only].mean()), 4) if int(comp_only.sum()) else None),
            "g_mean_when_direct_branch_better": (
                round(float(gs[dir_only].mean()), 4) if int(dir_only.sum()) else None),
            "note": "g 按位置取通道均值后再分桶；channel_mean_* 是逐通道跨位置的均值",
            "bins": bins,
        }
    return out


def compose_verdict(delta: float, margin: float = 0.014) -> str:
    """契约 G §4 的预登记判据线归类（代码只做归类，不改线）。

    成功 = gated t+2 命中 ≥ max(direct, composed) + 0.014；
    |Δ| < 0.014 ⇒ 中性（本项目方差界内不可排序）；Δ ≤ −0.014 ⇒ 失败（自门控组合有害）。
    """
    if delta >= margin:
        return "成功：自门控组合有效"
    if delta <= -margin:
        return "失败：自门控组合劣于 max(direct, composed)"
    return "中性：Δ 在方差界内，不可排序"


def run_composition(cfg: dict, args) -> None:
    """契约 30-G 主流程：冻结主干上训 direct / composed / gated / direct_eq 四臂。

    direct_eq 是 §15 的等参数对照（与 gated 参数量逐位相同、无门无组合），
    不是判据线的一部分；判据线只有 README §4 那一条。
    """
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    rel = cfg["trunk_reuse"]["path"]
    trunk_path = rel if os.path.isabs(rel) else os.path.normpath(os.path.join(cfg_dir, rel))
    sha_expect = cfg["trunk_reuse"].get("sha256")
    sha_actual = sha256_file(trunk_path)
    if sha_expect and sha_actual != sha_expect:
        raise SystemExit(f"trunk sha256 不符：{sha_actual} != {sha_expect}")

    hcfg = dict(cfg["head_train"])
    ecfg = dict(cfg["eval"])
    ccfg = dict(cfg.get("convergence_check", {}) or {})
    modes = list(hcfg["arms"])
    seeds = [int(s) for s in hcfg["seeds"]]
    if args.head_steps is not None:
        hcfg["steps"] = args.head_steps
    if args.smoke:
        ccfg = {}

    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seeds[0])
    train_arr, eval_arr, vocab_size = load_enwik8_full(".tmp/enwik8", "data/enwik8_full.npz")
    trunk = DepthARModel(DepthARConfig(**cfg["trunk"]), vocab_size).cuda()
    trunk.load_state_dict(torch.load(trunk_path, map_location="cuda"))
    freeze_trunk(trunk)
    ok, info = assert_frozen(trunk)
    n_trunk_params = sum(p.numel() for p in trunk.parameters())
    print(f"trunk {trunk_path}\n  sha256 {sha_actual} ({os.path.getsize(trunk_path)} B), "
          f"params {n_trunk_params}, frozen ok={ok} {info}", flush=True)

    t_start = time.time()
    arms, ckpts = {m: {} for m in modes}, {}
    gate_dist = {}
    last_shared = None
    for seed in seeds:
        heads = {}
        for mode in modes:
            log = os.path.join(out_dir, f"compose_log_{mode}_s{seed}.jsonl")
            head, wall, stats = train_composition_head(trunk, train_arr, hcfg, seed, mode, log)
            ckpt = os.path.join(out_dir, f"compose_head_{mode}_s{seed}.pt")
            torch.save(head.state_dict(), ckpt)
            ckpts[f"{mode}_s{seed}"] = {
                "path": os.path.basename(ckpt), "sha256": sha256_file(ckpt),
                "size_bytes": os.path.getsize(ckpt)}
            arms[mode][str(seed)] = {"train": stats, "wall_minutes": round(wall / 60, 2)}
            heads[mode] = head
            print(f"seed {seed} {mode}: loss_mean {stats['loss_mean']} "
                  f"({wall / 60:.2f} min)", flush=True)
        ev = evaluate_composition(trunk, heads, eval_arr, ecfg)
        for mode in modes:
            arms[mode][str(seed)].update(ev["arms"][mode])
            arms[mode][str(seed)]["n_positions"] = ev["n_positions"]
        last_shared = ev["shared"]
        if "gated_g" in ev:
            gate_dist[str(seed)] = ev["gated_g"]
        print(f"seed {seed} eval: " + " ".join(
            f"{m}={ev['arms'][m]['t2_hit']:.4f}/{ev['arms'][m]['accept_vs_ar1']:.4f}"
            for m in modes) + f"  ar1={ev['shared']['ar1_deploy_t2_hit']:.4f}", flush=True)

    # --- §16 收敛对账：同一主干、同一方法（同窗口同算法），只改步数 ---
    conv = {}
    if ccfg.get("steps"):
        conv_cfg = dict(hcfg)
        conv_cfg["steps"] = int(ccfg["steps"])
        for seed in [int(s) for s in ccfg.get("seeds", [])]:
            heads = {}
            for mode in modes:
                log = os.path.join(out_dir, f"compose_log_{mode}_s{seed}_{conv_cfg['steps']}step.jsonl")
                head, wall, stats = train_composition_head(trunk, train_arr, conv_cfg,
                                                           seed, mode, log)
                ckpt = os.path.join(out_dir, f"compose_head_{mode}_s{seed}_{conv_cfg['steps']}step.pt")
                torch.save(head.state_dict(), ckpt)
                ckpts[f"{mode}_s{seed}_{conv_cfg['steps']}step"] = {
                    "path": os.path.basename(ckpt), "sha256": sha256_file(ckpt),
                    "size_bytes": os.path.getsize(ckpt)}
                heads[mode] = head
                print(f"conv seed {seed} {mode}: {stats['loss_mean']} ({wall/60:.2f} min)", flush=True)
            ev = evaluate_composition(trunk, heads, eval_arr, ecfg)
            conv[str(seed)] = {"steps": conv_cfg["steps"], **ev}
            print(f"conv seed {seed} eval: " + " ".join(
                f"{m}={ev['arms'][m]['t2_hit']:.4f}" for m in modes), flush=True)

    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
    mean = lambda mode, key: round(
        sum(arms[mode][s][key] for s in arms[mode]) / len(arms[mode]), 4)
    dm, cmo, gm = mean("direct", "t2_hit"), mean("composed", "t2_hit"), mean("gated", "t2_hit")
    best = max(dm, cmo)
    acc_gated = mean("gated", "accept_vs_ar1")
    shared0 = last_shared

    def slopes(mode: str, steps: int, seed: int = 0) -> dict:
        tag = "" if steps == hcfg["steps"] else f"_{steps}step"
        return loss_slope_from_log(os.path.join(out_dir, f"compose_log_{mode}_s{seed}{tag}.jsonl"))

    conv_summary = {}
    if conv:
        for steps in sorted({v["steps"] for v in conv.values()}):
            conv_summary[str(steps)] = {
                "t2_hit": {m: round(sum(conv[s]["arms"][m]["t2_hit"] for s in conv) / len(conv), 4)
                           for m in modes},
                "delta_gated_vs_best": round(
                    sum(conv[s]["arms"]["gated"]["t2_hit"] for s in conv) / len(conv)
                    - max(sum(conv[s]["arms"][m]["t2_hit"] for s in conv) / len(conv)
                          for m in ("direct", "composed")), 4),
                "slope_per_1k": {m: slopes(m, steps)["slope_per_1k"] for m in modes},
                "window": slopes("gated", steps)["window"],
            }

    results = {
        "contract": ("axes/H-horizon/H3-composition/experiments/"
                     "2613-self-gated-composition-13m/README.md"),
        "config": cfg,
        "reused_trunk": {"path": trunk_path, "sha256": sha_actual,
                         "sha256_verified": bool(sha_expect) and sha_actual == sha_expect,
                         "size_bytes": os.path.getsize(trunk_path),
                         "params": n_trunk_params, "frozen_assertion_ok": bool(ok),
                         "frozen_offenders": info},
        "intervention": {
            "variable": "组合结构（唯一变量）",
            "arms": {m: (arms[m][str(seeds[0])]["train"]["activity"]) for m in modes},
            "gated_g_at_eval": gate_dist,
            "alive": all(arms[m][str(seeds[0])]["train"]["activity"]["T"]["nonzero_frac"] > 0
                         for m in modes),
        },
        "arms": arms,
        "arms_mean": {m: {k: mean(m, k) for k in
                           ("t2_hit", "accept_vs_ar1", "accept_vs_oracle", "eval_ce_t2")}
                      for m in modes},
        "params": {m: arms[m][str(seeds[0])]["train"]["activity"]["params"] for m in modes},
        "convergence": {
            "main": {m: {"steps": hcfg["steps"], **slopes(m, hcfg["steps"])} for m in modes},
            "longer": conv_summary,
            "raw": conv,
            "note": ("每臂日志每 200 步一点；斜率 = 末 25% 窗口内对日志点做 OLS，"
                     "所有臂同窗口同算法（§16）"),
        },
        "criteria": {
            "main": {"metric": "t2_hit（行 i 的 argmax == 真值 t_{i+2}）",
                     "gated": gm, "direct": dm, "composed": cmo,
                     "best_baseline": best, "margin": 0.014,
                     "delta_vs_best": round(gm - best, 4),
                     "ok": gm >= best + 0.014},
            "parallel_draft_accept": {
                "metric": "accept_vs_ar1 = P(头 argmax == 主干自身 1-step AR argmax)",
                "gated": acc_gated, "threshold": 0.90, "ok": acc_gated >= 0.90},
            "oracle": {"t2_hit": shared0["oracle_t2_hit"], "used_as_criterion": False,
                      "note": "喂真 t+1 的上界，只作参考（SCHEMA §11）"},
            "verdict": compose_verdict(gm - best),
        },
        "reuse_anchors": {
            "note": "同主干上的历史锚点（不同训练配方：2610/2611 同时训 t1+t2 两头）",
            "2610_head_t2_hit": 0.4667, "2611_ar1_deploy_t2_hit": 0.5489,
            "2611_accept_jump_t2": 0.5998, "2611_oracle_t2_hit": 0.6953,
        },
        "heads": ckpts,
        "cost": {"gpu_wall_minutes": round((time.time() - t_start) / 60, 2),
                 "peak_mem_mib": round(peak_mb, 1),
                 "runs": len(seeds) * len(modes) + len(ccfg.get("seeds", [])) * len(modes)},
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in results.items()
                      if k not in ("config", "heads", "arms")},
                     indent=2, ensure_ascii=False))


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
    ap.add_argument("--composition", action="store_true",
                    help="契约 30-G：冻结主干上训 direct/composed/gated/direct_eq 四臂（H3 自门控组合）")
    ap.add_argument("--smoke", action="store_true",
                    help="20 步冒烟：主干 20 步 + 头 20 步 + 小评测，打印步时与显存")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    if args.smoke:
        cfg.setdefault("trunk_train", {})["steps"] = 20
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

    if args.composition:
        return run_composition(cfg, args)

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
