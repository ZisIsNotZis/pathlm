"""ticket 42-W1（long-T 训练稳定性修复）的可失败单测。

钉住的是**本实验新增的、自己负责的语义面**：实验本地驱动器 `stability_driver.py`
的候选补丁（每个语义开关都必须「真的生效」且「缺省/恒等档逐位还原」）。
全部 CPU，不取 GPU 锁。

Run: python3 -m pytest tests/test_longT_stability.py -q
"""

import importlib.util
import json
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evidence_anchor                                       # noqa: E402
from pathlm.depth_ar import DepthARConfig, DepthARModel          # noqa: E402

DRIVER = os.path.join(ROOT, "axes", "A-mixing", "A6-mixing-operator",
                      "experiments", "2625-longT-stability",
                      "stability_driver.py")
EXP = os.path.dirname(DRIVER)

D, L, H, MLP = 32, 2, 2, 4


def _load_driver():
    spec = importlib.util.spec_from_file_location("stability_driver_t", DRIVER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sd = _load_driver()


def _tiny_model(seed: int = 0, seq_len: int = 16, vocab: int = 32):
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=D, n_layers=L, n_heads=H, seq_len=seq_len,
                        mlp_mult=MLP, dense=True)
    return DepthARModel(cfg, vocab)


# ------------------------------------------------------------------ 候选解析

def test_parse_candidate_maps_names_to_kinds():
    assert sd.parse_candidate("none") == ("none", {})
    assert sd.parse_candidate("temp2.0") == ("temp", {"tau": 2.0})
    assert sd.parse_candidate("qk-norm") == ("qk-norm", {"tau": 1.0,
                                                     "min_t": 0})
    assert sd.parse_candidate("qk-normfrom512") == ("qk-norm",
                                                    {"tau": 1.0, "min_t": 512})
    assert sd.parse_candidate("wd0.1") == ("wd", {"wd": 0.1})
    assert sd.parse_candidate("warmup2000") == ("warmup", {"warmup": 2000.0})
    assert sd.parse_candidate("clip0.5") == ("clip", {"clip": 0.5})
    assert sd.parse_candidate("initpos0.05") == ("initpos", {"pos_std": 0.05})
    with pytest.raises(SystemExit):
        sd.parse_candidate("bogus1")


# ------------------------------------------------------------------ temp 候选

def test_qk_norm_length_conditional_is_inert_at_or_below_min_t():
    """长度条件开关：`min_t=512` 时 T=512 的模型**不挂任何 hook**（逐位不变）。"""
    x = torch.randint(0, 32, (2, 16))
    m0 = _tiny_model(seq_len=16)
    with torch.no_grad():
        base = m0(x)[1]["depth_logits"][-1]
    m1 = _tiny_model(seq_len=16)
    info = sd.apply_attn_scale(m1, "qk-norm", 1.0, min_t=512)
    assert info["active"] is False and info["n_qkv_hooks"] == 0
    with torch.no_grad():
        assert torch.equal(base, m1(x)[1]["depth_logits"][-1]), \
            "长度条件档在 T ≤ min_t 时不是逐位恒等"
    # 反向：T > min_t 时必须真的挂上 hook
    m2 = _tiny_model(seq_len=16)
    info2 = sd.apply_attn_scale(m2, "qk-norm", 1.0, min_t=8)
    assert info2["active"] is True and info2["n_qkv_hooks"] == L


def test_temp_candidate_actually_changes_logits():
    """语义开关必须真的生效：τ=2 时同一输入的前向输出必须变。"""
    x = torch.randint(0, 32, (2, 16))
    m0 = _tiny_model()
    with torch.no_grad():
        base = m0(x)[1]["depth_logits"][-1]
    m1 = _tiny_model()
    info = sd.apply_attn_scale(m1, "temp", 2.0)
    assert info["n_qkv_hooks"] == L
    with torch.no_grad():
        alt = m1(x)[1]["depth_logits"][-1]
    assert not torch.equal(base, alt), "temp 候选是死的（前向输出未变）"


def test_temp_tau_one_is_bitwise_identity():
    """恒等档（τ=1）必须逐位还原无 hook 的前向。"""
    x = torch.randint(0, 32, (2, 16))
    m0 = _tiny_model()
    with torch.no_grad():
        base = m0(x)[1]["depth_logits"][-1]
    m1 = _tiny_model()
    sd.apply_attn_scale(m1, "temp", 1.0)
    with torch.no_grad():
        alt = m1(x)[1]["depth_logits"][-1]
    assert torch.equal(base, alt), "τ=1 不是逐位恒等"


# ------------------------------------------------------------------ qk-norm 候选

def test_qk_norm_gives_unit_scale_heads():
    """q/k 按 head L2 归一后，每个头的范数必须恰为 √d_h。"""
    dh = D // H
    seen = []

    def check(module, inp, out):
        seen.append(out)

    m = _tiny_model()
    sd.apply_attn_scale(m, "qk-norm", 1.0)
    for blk in m.blocks:
        blk.qkv.register_forward_hook(check)
    x = torch.randint(0, 32, (2, 16))
    with torch.no_grad():
        m(x)
    assert len(seen) == L
    for out in seen:
        d = out.shape[-1] // 3
        q = out[..., :d].reshape(-1, H, dh).float()
        k = out[..., d:2 * d].reshape(-1, H, dh).float()
        assert torch.allclose(q.norm(dim=-1), torch.full_like(q.norm(dim=-1),
                                                              math.sqrt(dh)),
                              atol=1e-4)
        assert torch.allclose(k.norm(dim=-1), torch.full_like(k.norm(dim=-1),
                                                              math.sqrt(dh)),
                              atol=1e-4)


# ------------------------------------------------------------------ warmup 候选

def test_warmup_patch_only_extends_warmup():
    """N 步预热 + 同一余弦；N=200 时必须与引擎原式逐位相同。"""
    steps = 1000
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    orig_cls = sd.torch.optim.lr_scheduler.LambdaLR
    try:
        sd.patch_warmup(200, steps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.0)
        got200 = [sched.lr_lambdas[0](s) for s in range(steps)]
    finally:
        sd.torch.optim.lr_scheduler.LambdaLR = orig_cls
    want = [min((s + 1) / 200, 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / steps)))
            for s in range(steps)]
    assert got200 == want, "N=200 未逐位还原引擎原调度"
    try:
        sd.patch_warmup(500, steps)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.0)
        got500 = [sched.lr_lambdas[0](s) for s in range(steps)]
    finally:
        sd.torch.optim.lr_scheduler.LambdaLR = orig_cls
    assert got500[0] < got200[0], "预热拉长后第 0 步 LR 必须更小"
    for s in range(500, steps):
        assert got500[s] == want[s], "预热结束后必须与余弦逐位相同"


# ------------------------------------------------------------------ wd / clip

def test_weight_decay_patch_sets_optimizer_arg():
    orig = sd.torch.optim.AdamW
    try:
        info = sd.patch_weight_decay(0.1)
        opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3,
                                weight_decay=0.01)
    finally:
        sd.torch.optim.AdamW = orig
    assert info["weight_decay"] == 0.1
    assert opt.param_groups[0]["weight_decay"] == 0.1


def test_clip_threshold_patch_replaces_max_norm():
    orig = sd.torch.nn.utils.clip_grad_norm_
    try:
        sd.patch_clip_threshold(0.5)
        p = torch.nn.Parameter(torch.zeros(4))
        p.grad = torch.full((4,), 10.0)
        got = float(torch.nn.utils.clip_grad_norm_([p], 1.0).detach())
    finally:
        sd.torch.nn.utils.clip_grad_norm_ = orig
    assert got == pytest.approx(math.sqrt(4 * 100.0)), "预裁剪范数读数被改"
    assert float(p.grad.norm()) == pytest.approx(0.5, abs=1e-6), "阈值未被替换"


# ------------------------------------------------------------------ init 候选

def test_initpos_changes_pos_embed_scale():
    m = _tiny_model()
    before = float(m.pos_embed.weight.detach().std())
    sd.apply_init_scale(m, 0.5, None)
    after = float(m.pos_embed.weight.detach().std())
    assert abs(after - before) > 0.1 and abs(after - 0.5) < 0.1


# ------------------------------------------------------------------ 探针

def test_probe_records_all_layer_stats():
    m = _tiny_model()
    step_ref = {"n": -1}
    probe = sd.ProbeHook(m, step_ref, every=1, nl=4).attach()
    with torch.no_grad():
        m(torch.randint(0, 32, (2, 16)))
    snap = probe.snapshots[0]
    assert snap["step"] == 0 and snap["resid_in"] is not None
    assert len(snap["layers"]) == L
    for row in snap["layers"]:
        for key in ("qkv_w_norm", "logit_rms", "logit_max", "top1", "entropy",
                    "resid_out"):
            assert key in row and row[key] == row[key]     # 键齐全且非 NaN
        assert 0.0 < row["top1"] <= 1.0
        assert row["entropy"] >= -1e-6


# --------------------------------------------- 驱动器保真度（对 2623 逐位）

def test_driver_preflight_step0_matches_engine_2623():
    """驱动器 `none` 候选的 step0 读数必须与引擎 2623 的 full-1024 逐位相同。

    证据文件：本目录 `runs/preflight-20/train_log.jsonl`（20 步、seed0、T=1024、bs8、
    候选 `none`）与 `2623-longT-1024/runs/full-1024/train_log.jsonl` 的首行。

    F_KEY（复核 B12）：`runs/` 不入 git ⇒ 比对所需的**最小证据**（两侧 step0 行）已入
    git（`tests/evidence/longT_step0_anchors.json`）；本地原始证据不在时 `xfail`，
    不再静默 `pytest.skip`。
    """
    anchors = evidence_anchor.load_anchors()
    ours_a = evidence_anchor.anchor(anchors, "stability_preflight_step0")
    theirs_a = evidence_anchor.anchor(anchors, "engine_2623_full1024_step0")
    # ① 入 git 的锚点自洽（从 git 即可复现该断言）
    assert ours_a["step"] == theirs_a["step"] == 0
    assert ours_a["loss"] == theirs_a["loss"], "锚点记录的两侧 step0 loss 不一致"
    assert ours_a["gnorm"] == theirs_a["gnorm"], "锚点记录的两侧 step0 gnorm 不一致"
    # ② 本地原始证据在时逐位复算（并与锚点对齐，防证据被换）
    ours = evidence_anchor.local_step0(
        os.path.join("axes", "A-mixing", "A6-mixing-operator", "experiments",
                     "2625-longT-stability", "runs", "preflight-20",
                     "train_log.jsonl"), "stability_preflight_step0", anchors)
    theirs = evidence_anchor.local_step0(
        os.path.join("axes", "A-mixing", "A6-mixing-operator", "experiments",
                     "2623-longT-1024", "runs", "full-1024", "train_log.jsonl"),
        "engine_2623_full1024_step0", anchors)
    a, b = ours, theirs
    assert a["step"] == b["step"] == 0
    assert a["loss"] == b["loss"], "驱动器 step0 loss 与引擎不一致"
    assert a["gnorm"] == b["gnorm"], "驱动器 step0 gnorm 与引擎不一致"
