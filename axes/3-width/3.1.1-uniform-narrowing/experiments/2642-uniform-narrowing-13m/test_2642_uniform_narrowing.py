"""2642 · 等值性门与审计单测（自包含，不改共享 `tests/`）。

`SCHEMA.md` §9 / `docs/methodology.md` §5 / 契约 `README.md` §2.2：

- **T1**：ρ=1.0（w=d=256）的**前缀前向** == stock 引擎前向 —— logits 与 loss **逐位相同**；
- **T2**：推理期**切片模型**（原生 d=w）前向 == 同宽度训练期**前缀子网络**前向 —— 逐位相同，
  w ∈ {128,192,256}（证明「省算力」是真的可达，不是掩码假象）；
- **T3**：变异（故意切错列 / 取错 LN 张量 / 头数取错 / 掩码式错切）必须**抓红**；
- **T4**：参数与维数审计（切片模型里不存在全宽张量；实测参数 == 解析参数；活跃维 == 目标 w）。

    python3 -m pytest test_2642_uniform_narrowing.py -q
"""

import importlib.util
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARConfig, DepthARModel          # noqa: E402

_spec = importlib.util.spec_from_file_location("run_2642", os.path.join(HERE, "run.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

VOCAB = 206
WIDTHS = (128, 192, 256)


def full_model(seed: int = 0) -> DepthARModel:
    torch.manual_seed(seed)
    cfg = DepthARConfig(d_model=R.D_MODEL, n_layers=R.N_LAYERS, n_heads=R.N_HEADS,
                        seq_len=R.SEQ_LEN, mlp_mult=R.MLP_MULT, dense=True,
                        topology="sequential", fill_kv=True)
    m = DepthARModel(cfg, VOCAB)
    m.eval()
    return m


def tokens(B: int = 2, T: int = 32, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (B, T), generator=g)


# ---------------------------------------------------------------- T1：ρ=1.0 逐位还原 stock

def test_t1_prefix_forward_at_full_width_matches_stock_bitwise():
    m = full_model()
    x = tokens()
    with torch.no_grad():
        logits_p, states = R.prefix_forward_full(m, x, R.D_MODEL)
        logits_s = m(x, exit_depths=None)[1]["depth_logits"][-1]
        loss_p, _ = R.prefix_loss_full(m, x, R.D_MODEL)
        loss_s = m(x, exit_depths=None)[0]
    assert logits_p.shape == logits_s.shape
    assert torch.equal(logits_p, logits_s), "T1：前缀前向在 w=d 时必须逐位等于 stock"
    assert torch.equal(loss_p, loss_s), "T1：dense 损失也必须逐位相同"
    assert len(states) == R.N_LAYERS + 1


def test_t1_prefix_forward_matches_stock_under_bf16_autocast():
    if not torch.cuda.is_available():
        pytest.skip("bf16 autocast 门需要 CUDA")
    m = full_model().cuda()
    x = tokens().cuda()
    with torch.no_grad():
        for enabled in (True, False):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                logits_p, _ = R.prefix_forward_full(m, x, R.D_MODEL)
                logits_s = m(x, exit_depths=None)[1]["depth_logits"][-1]
                loss_p, _ = R.prefix_loss_full(m, x, R.D_MODEL)
                loss_s = m(x, exit_depths=None)[0]
            assert torch.equal(logits_p, logits_s), f"autocast={enabled}"
            assert torch.equal(loss_p, loss_s), f"autocast={enabled}"


# ---------------------------------------------------------------- T2：切片模型 == 前缀前向

@pytest.mark.parametrize("w", WIDTHS)
def test_t2_sliced_model_matches_prefix_forward_bitwise(w):
    m = full_model()
    sd = m.state_dict()
    x = tokens()
    sl = R.build_sliced_model(sd, w).eval()
    with torch.no_grad():
        logits_p, _ = R.prefix_forward_full(m, x, w)
        logits_s = sl(x, exit_depths=None)[1]["depth_logits"][-1]
        loss_p, _ = R.prefix_loss_full(m, x, w)
        loss_s = sl(x, exit_depths=None)[0]
    assert logits_p.shape == (x.shape[0], x.shape[1], VOCAB)
    assert torch.equal(logits_p, logits_s), f"T2：w={w} 切片模型必须逐位等于前缀前向"
    assert torch.equal(loss_p, loss_s), f"T2：w={w} 的损失也必须逐位相同"


# ---------------------------------------------------------------- T4：参数与维数审计

@pytest.mark.parametrize("w", WIDTHS)
def test_t4_sliced_model_is_really_narrower(w):
    m = full_model()
    sd = m.state_dict()
    sl = R.build_sliced_model(sd, w)
    assert sl.cfg.d_model == w
    assert sl.cfg.n_heads == w // R.HEAD_DIM
    assert [int(b.mlp[0].weight.shape[0]) for b in sl.blocks] == [R.MLP_MULT * w] * R.N_LAYERS
    tensors = [t for t in sl.state_dict().values() if t.dim() >= 2]
    assert max(int(max(t.shape)) for t in tensors) <= R.MLP_MULT * w, \
        "切片模型里不得存在全宽（d=256）张量 ⇒ 省算力是真的"
    measured = sum(p.numel() for p in sl.parameters())
    assert measured == R.analytic_params(w, vocab=VOCAB)["params_total"]
    if w < R.D_MODEL:
        assert measured < sum(p.numel() for p in m.parameters())


@pytest.mark.parametrize("d,heads", [(256, 8), (192, 6), (128, 4)])
def test_t4_uniform_arm_params_match_analytic(d, heads):
    torch.manual_seed(0)
    m = DepthARModel(R.width_cfg(d), VOCAB)
    measured = sum(p.numel() for p in m.parameters())
    ap = R.analytic_params(d, vocab=VOCAB)
    assert measured == ap["params_total"]
    assert [int(b.mlp[0].weight.shape[0]) for b in m.blocks] == [R.MLP_MULT * d] * R.N_LAYERS


def test_t4_analytic_cost_matches_2641_recorded_values():
    """代价口径自检：U4/U2 的解析 MACs 必须与 2641 落盘值逐位相同（同一只尺子）。"""
    assert R.analytic_cost(256)["macs_per_token"]["total"] == 7_392_768
    u2 = R.load_anchor("U2")
    assert u2["macs_per_token"] == 5_295_616
    assert R.analytic_cost(192)["macs_per_token"]["total"] == 4_364_928
    assert R.analytic_cost(128)["macs_per_token"]["total"] == 2_123_520


# ---------------------------------------------------------------- T3：变异必须抓红

def test_t3a_wrong_proj_row_slice_breaks_t2():
    m = full_model()
    sd = m.state_dict()
    x = tokens()
    w = 128
    bad = R.convert_state_dict(sd, w)
    bad["blocks.0.proj.weight"] = sd["blocks.0.proj.weight"][w:2 * w, :w].contiguous()
    sl = DepthARModel(R.width_cfg(w), VOCAB)
    sl.load_state_dict(bad)
    sl.eval()
    with torch.no_grad():
        logits_p, _ = R.prefix_forward_full(m, x, w)
        logits_bad = sl(x, exit_depths=None)[1]["depth_logits"][-1]
    assert not torch.equal(logits_p, logits_bad), "T3a 变异必须被 T2 抓红"


def test_t3b_wrong_ln_slice_breaks_t1():
    """LN 权重取错（形状相同、取值不同）⇒ T1 必须抓红。

    注：stock 里 ln1/ln2 的 weight 初值都是 ones，所以「拿 ln2 的权重冒充 ln1」不改变
    数值、抓不红 —— 变异必须是**改变取值**的错误切片（这里取门矩阵的一列，取值不同）。
    """
    m = full_model()
    x = tokens()
    w = R.D_MODEL
    wrong_ln = m.gates[0].weight[:w, 0].detach().clone()
    assert not torch.equal(wrong_ln, m.blocks[0].ln1.weight[:w])
    with torch.no_grad():
        bad, _ = R.prefix_forward_full(m, x, w, mutate={"ln1_w": wrong_ln})
        ref = m(x, exit_depths=None)[1]["depth_logits"][-1]
    assert not torch.equal(bad, ref), "T3b 变异（LN 取错取值）必须被 T1 抓红"


def test_t3c_wrong_proj_column_slice_breaks_t2():
    m = full_model()
    x = tokens()
    w = 128
    wrong_proj = m.blocks[0].proj.weight[1:w + 1, 1:w + 1].detach().clone()
    with torch.no_grad():
        bad, _ = R.prefix_forward_full(m, x, w, mutate={"proj_w": wrong_proj})
        sl = R.build_sliced_model(m.state_dict(), w).eval()
        good = sl(x, exit_depths=None)[1]["depth_logits"][-1]
    assert not torch.equal(bad, good), "T3c 变异（proj 取错行列）必须被 T2 抓红"


def test_t3d_wrong_head_count_breaks_or_raises():
    m = full_model()
    x = tokens()
    with pytest.raises(Exception):
        R.prefix_forward_full(m, x, 192, mutate={"n_heads": 7})


def test_t3e_masking_style_slice_fails_t2():
    """掩码式错切（把 QKV 输出当连续前缀 [:3w]）必须与真前缀子网络不同。"""
    m = full_model()
    x = tokens()
    w = 128
    with torch.no_grad():
        logits_p, _ = R.prefix_forward_full(m, x, w)
        sl = R.build_sliced_model(m.state_dict(), w, mask_instead_of_slice=True).eval()
        logits_bad = sl(x, exit_depths=None)[1]["depth_logits"][-1]
    assert not torch.equal(logits_p, logits_bad), "T3e：掩码式错切必须被 T2 抓红"


# ---------------------------------------------------------------- P4-③ 活性 hook 回归门

def _toy_text(n: int = 8192, seed: int = 7):
    import numpy as np
    return np.random.default_rng(seed).integers(0, VOCAB, size=n, dtype=np.int64)


def _live_model(d: int, heads: int):
    torch.manual_seed(0)
    return DepthARModel(R.width_cfg(d), VOCAB).eval()


def test_p4_liveness_hook_accepts_block_tuple_output():
    """回归门（D192）+ WD256 三档切片：`block_liveness` 的 forward hook 必须接受
    `Block.forward` 的元组输出 `(block_out, telemetry)`。

    若 hook 退回只支持 tensor（`out.detach()`），每个臂都会抛 `AttributeError` ⇒
    本用例必须变红。
    """
    if not torch.cuda.is_available():
        pytest.skip("block_liveness 走 cuda 设备")
    arr = _toy_text()
    m192 = _live_model(192, 6).cuda()
    live = R.block_liveness(m192, arr, w=192, n_batches=1, batch_size=2)
    assert live["active_width"] == 192
    assert len(live["block_out_over_in_by_layer"]) == R.N_LAYERS, "hook 必须每层都触发"
    assert live["min_ratio"] > 0.0, "每层的真实前向必须活性非 0"

    sd = full_model().state_dict()
    for w in WIDTHS:
        sl = R.build_sliced_model(sd, w).cuda().eval()
        lv = R.block_liveness(sl, arr, w=w, n_batches=1, batch_size=2)
        assert lv["active_width"] == w
        assert lv["hidden_by_layer"] == [R.MLP_MULT * w] * R.N_LAYERS
        assert len(lv["block_out_over_in_by_layer"]) == R.N_LAYERS
        assert lv["min_ratio"] > 0.0


# ---------------------------------------------------------------- 因果性（P4-②）

@pytest.mark.parametrize("w", (256, 192, 128))
def test_causality_prefix_and_sliced(w):
    m = full_model()
    x = tokens(B=2, T=32)
    with torch.no_grad():
        logits1, _ = R.prefix_forward_full(m, x, w)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % VOCAB
        logits2, _ = R.prefix_forward_full(m, x2, w)
    d_past = (logits1[:, :-1] - logits2[:, :-1]).abs().max().item()
    d_last = (logits1[:, -1] - logits2[:, -1]).abs().max().item()
    assert d_past == 0.0, f"w={w}：改末位 token 不得影响之前位置（因果掩码）"
    assert d_last > 0.0, f"w={w}：末位必须变（活性对照）"


# ---------------------------------------------------------------- 复用纪律 / 配置对照

def test_anchor_reuse_reads_2641_files_and_config_parity():
    u4, u2 = R.load_anchor("U4"), R.load_anchor("U2")
    assert u4["clean_bpc"] == 1.5314 and u4["params_total"] == 6_962_688
    assert u4["macs_per_token"] == 7_392_768
    assert u2["clean_bpc"] == 1.5709 and u2["macs_per_token"] == 5_295_616
    par = R.anchor_config_parity(12000, 32, 1e-3)
    assert par["all_equal"], par["mismatched_fields"]


def test_arm_argv_differs_from_u4_only_in_width_fields():
    def kw(arm):
        a = R.shared_argv(arm, 0, 12000, 32, 1e-3, "x")
        out, i = {}, 2
        while i < len(a):
            if a[i].startswith("--"):
                out[a[i]] = a[i + 1]
                i += 2
            else:
                i += 1
        return out

    base = kw("WD256")            # d=256, heads=8 = U4 的轴配置
    for arm, d, heads in (("D192", "192", "6"), ("D128", "128", "4")):
        cur = kw(arm)
        diff = {k: (base[k], cur[k]) for k in base if base[k] != cur[k]}
        assert diff == {"--d": ("256", d), "--heads": ("8", heads)}, diff
    assert kw("WD256") == base


def test_width_dropout_model_stock_init_and_prefix_forward():
    """WD256 的构造路径 = stock 构造（同 seed ⇒ 同一次抽样），前向走前缀子网络。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=R.D_MODEL, n_layers=R.N_LAYERS, n_heads=R.N_HEADS,
                        seq_len=R.SEQ_LEN, mlp_mult=R.MLP_MULT, dense=True,
                        topology="sequential", fill_kv=True)
    torch.manual_seed(0)
    stock = DepthARModel(cfg, VOCAB)
    torch.manual_seed(0)
    wd = R.WidthDropoutModel(cfg, VOCAB, seed=0)
    for k in stock.state_dict():
        assert torch.equal(stock.state_dict()[k], wd.state_dict()[k]), \
            "同一 seed 下 WD256 的初始化必须与 stock 逐位相同"
    x = tokens()
    wd.eval()
    with torch.no_grad():
        loss, aux = wd(x, exit_depths=None)
    seen = set(wd.wd_samples)
    assert len(wd.wd_samples) == 1 and seen <= set(WIDTHS)
    w = wd.wd_samples[0]
    with torch.no_grad():
        ref, _ = R.prefix_forward_full(wd, x, w)
    assert torch.equal(aux["depth_logits"][-1], ref)
    assert loss.ndim == 0 and float(loss) > 0


# ---------------------------------------------------------------- P5b 带内判据（复核 N-5）
# 这三条是本 WP 期内新增判据（`p5b_frontier_verdict`）的红-绿门：带内 ⇒ **取消**方向性宣称，
# 越带才允许方向性判词。若有人把带内分支改回「占前沿优势 / 不落下风」，本组必红。
def test_p5b_in_band_cancels_directional_claim():
    txt = R.p5b_frontier_verdict({"D192": 0.001287}, 0.001392, in_band=False,
                                 delta_bpc_gap=0.0132, bound=0.014)
    assert "不可排序" in txt
    assert "0.0132" in txt
    for bad in ("占前沿优势", "不占前沿优势", "略优于", "不落下风"):
        assert bad not in txt, f"带内判词不得出现方向性措辞：{bad}"


def test_p5b_out_of_band_keeps_directional_claim_worse():
    txt = R.p5b_frontier_verdict({"D192": 0.003000}, 0.001392, in_band=False,
                                 delta_bpc_gap=0.0500, bound=0.014)
    assert "不占前沿优势" in txt and "不可排序" not in txt


def test_p5b_out_of_band_keeps_directional_claim_better():
    txt = R.p5b_frontier_verdict({"D192": 0.000500}, 0.001392, in_band=False,
                                 delta_bpc_gap=0.0500, bound=0.014)
    assert "至少一臂优于 FFN 旋钮" in txt and "不可排序" not in txt
