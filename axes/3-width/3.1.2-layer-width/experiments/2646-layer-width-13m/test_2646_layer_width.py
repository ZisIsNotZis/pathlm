"""2646 · 等值性门与审计单测（自包含，不改共享 `tests/`）。

`SCHEMA.md` §9 / `docs/methodology.md` §5 / 契约 `README.md` §2.3：

- **T1**：全层 `w_l = d = 256` 时，前缀子空间模型与 stock 引擎**逐位相同**
  （state_dict 逐位 + logits/loss 逐位；fp32 与 bf16 autocast 各一次）；
- **T2**：逐层**实测**活跃维 == 目标 schedule、实测参数量 == 解析值、
  qkv 前缀是**分块**前缀（`q|k|v` 各取前 w 行）而不是输出行连续前缀；
- **T3**：变异（切错行 / 切错列 / 换层宽 / 宽度不整除 head_dim / 手改宽度标签）
  必须**抓红**；
- **P3**：因果性（改末位 token ⇒ 之前位置 logits 逐位不变）；
- **复用纪律**：解析代价与 2641 U4/U2 落盘值逐位相同；逐字段配置对照全等。

    python3 -m pytest test_2646_layer_width.py -q
"""

import importlib.util
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, ROOT)

from pathlm.depth_ar import DepthARModel                          # noqa: E402

_spec = importlib.util.spec_from_file_location("run_2646", os.path.join(HERE, "run.py"))
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

VOCAB = 206
ARMS = ("EQ", "PYR", "HG", "APYR")


def full_schedule_model(schedule=None, seed: int = 0) -> DepthARModel:
    return R.build_model(schedule or [R.D_MODEL] * R.N_LAYERS, VOCAB, seed)


def stock_model(seed: int = 0) -> DepthARModel:
    torch.manual_seed(seed)
    m = DepthARModel(R.stock_cfg(), VOCAB)
    m.eval()
    return m


def tokens(B: int = 2, T: int = 32, seed: int = 1234) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (B, T), generator=g)


# ---------------------------------------------------------------- 臂与 schedule 的自身审计

def test_all_arms_share_sum_w2_and_multiset():
    for arm in ARMS:
        assert R.sum_w2(R.SCHEDULES[arm]) == 8 * 192 ** 2, arm
    pyr, hg, apyr = (R.SCHEDULES[a] for a in ("PYR", "HG", "APYR"))
    assert sorted(pyr) == sorted(hg) == sorted(apyr), "三重集必须逐位相同"
    assert sum(pyr) == sum(hg) == sum(apyr), "三臂 Σw 也必须逐位相同（参数与 MACs 相同）"
    assert apyr == list(reversed(pyr))
    assert R.SCHEDULES["EQ"] == [192] * R.N_LAYERS
    # 形态
    assert all(a >= b for a, b in zip(pyr, pyr[1:])), "PYR 必须非增"
    assert all(a <= b for a, b in zip(apyr, apyr[1:])), "APYR 必须非减"
    peak = hg.index(max(hg))
    assert 0 < peak < len(hg) - 1, "HG 的峰必须在中间"
    assert all(a <= b for a, b in zip(hg, hg[1:peak + 1])), "HG 峰前必须非减"
    assert all(a >= b for a, b in zip(hg[peak:], hg[peak + 1:])), "HG 峰后必须非增"
    assert hg[0] < max(hg) and hg[-1] < max(hg), "HG 两端必须窄于中段"


def test_search_schedule_json_matches_codes_schedules():
    """落盘的搜索结果与代码里的 schedule 必须一致（搜索脚本是它们的家）。"""
    path = os.path.join(HERE, "schedule_search.json")
    if not os.path.exists(path):
        pytest.skip("先跑 search_schedule.py")
    import json
    d = json.load(open(path))
    assert d["arms"]["PYR"]["schedule"] == R.SCHEDULES["PYR"]
    assert d["arms"]["HG"]["schedule"] == R.SCHEDULES["HG"]
    assert d["arms"]["APYR"]["schedule"] == R.SCHEDULES["APYR"]
    assert d["arms"]["EQ"]["sum_w2"] == 8 * 192 ** 2
    assert d["n_symmetric_unimodal_nonuniform"] == 0, \
        "非均匀的严格对称沙漏若存在，HG 的定义必须改用它"


# ---------------------------------------------------------------- T1：全宽 ⇒ 逐位还原 stock

def test_t1_prefix_model_at_full_width_is_bitwise_stock():
    torch.manual_seed(0)
    prefix = full_schedule_model([R.D_MODEL] * R.N_LAYERS, seed=0)
    assert all(isinstance(b, R.PrefixSubspaceBlock) for b in prefix.blocks), \
        "全宽臂也必须走前缀代码路径（否则门是空的）"
    assert all(b.width == R.D_MODEL for b in prefix.blocks)
    stock = stock_model(0)
    sd_p, sd_s = prefix.state_dict(), stock.state_dict()
    assert sd_p.keys() == sd_s.keys()
    for k in sd_s:
        assert torch.equal(sd_p[k], sd_s[k]), f"state_dict 键 {k} 不逐位相同"
    x = tokens()
    with torch.no_grad():
        lp, ls = prefix(x, exit_depths=None), stock(x, exit_depths=None)
    assert torch.equal(lp[1]["depth_logits"][-1], ls[1]["depth_logits"][-1]), \
        "T1：w=d 时 logits 必须逐位相同"
    assert torch.equal(lp[0], ls[0]), "T1：dense 损失也必须逐位相同"


def test_t1_matches_stock_under_bf16_autocast_and_train_mode():
    if not torch.cuda.is_available():
        pytest.skip("bf16 autocast 门需要 CUDA")
    prefix = full_schedule_model([R.D_MODEL] * R.N_LAYERS).cuda()
    stock = stock_model(0).cuda()
    x = tokens().cuda()
    for enabled in (True, False):
        for training in (False, True):
            prefix.train(training)
            stock.train(training)
            with torch.no_grad():
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                    lp, ls = prefix(x, exit_depths=None), stock(x, exit_depths=None)
            assert torch.equal(lp[1]["depth_logits"][-1], ls[1]["depth_logits"][-1]), \
                f"autocast={enabled} training={training}"
            assert torch.equal(lp[0], ls[0]), f"autocast={enabled} training={training}"


# ---------------------------------------------------------------- T2：实测宽度 / 参数 / 切法

@pytest.mark.parametrize("arm", ARMS)
def test_t2_measured_widths_and_params_match_target(arm):
    schedule = R.SCHEDULES[arm]
    m = full_schedule_model(schedule)
    assert m.cfg.d_model == R.D_MODEL, "残差流恒 256 宽（读出恒在 256 上）"
    assert [b.width for b in m.blocks] == schedule
    for l, (blk, w) in enumerate(zip(m.blocks, schedule)):
        assert blk.qkv.weight.shape == (3 * w, w), (l, blk.qkv.weight.shape)
        assert blk.proj.weight.shape == (w, w), (l, blk.proj.weight.shape)
        assert blk.mlp[0].weight.shape == (R.MLP_MULT * w, w), (l,)
        assert blk.mlp[2].weight.shape == (w, R.MLP_MULT * w), (l,)
        assert blk.n_heads == w // R.HEAD_DIM, (l, blk.n_heads)
    measured = sum(p.numel() for p in m.parameters())
    ap = R.analytic_params(schedule, vocab=VOCAB)
    assert measured == ap["params_total"], (measured, ap["params_total"])
    assert m.embed.weight.shape == (VOCAB, R.D_MODEL)
    assert sum(p.numel() for p in m.gates.parameters()) == ap["gates_dead_in_dense"]


def test_t2_qkv_prefix_is_blockwise_not_contiguous():
    """`qkv` 的前缀 = `q|k|v` 各取前 w 行（不是输出行的连续前缀 `[:3w]`）。"""
    torch.manual_seed(0)
    stock = DepthARModel(R.stock_cfg(), VOCAB)
    w, d = 192, R.D_MODEL
    blk = R.prefix_block_from_stock(stock.blocks[0], w)
    src = stock.blocks[0].qkv.weight
    blockwise = torch.cat([src[0:w, :w], src[d:d + w, :w], src[2 * d:2 * d + w, :w]], 0)
    assert torch.equal(blk.qkv.weight, blockwise)
    assert not torch.equal(blk.qkv.weight, src[:3 * w, :w]), \
        "分块前缀与掩码式连续前缀不是一回事（后者必须被本断言区分开）"


def test_t2_liveness_reports_measured_width_from_real_forward():
    """`prefix_liveness` 的实测活跃维来自 LayerNorm 的**实际输入宽度**。"""
    if not torch.cuda.is_available():
        pytest.skip("prefix_liveness 走 cuda")
    import numpy as np
    arr = np.random.default_rng(7).integers(0, VOCAB, size=8192, dtype=np.int64)
    for arm in ARMS:
        m = full_schedule_model(R.SCHEDULES[arm]).cuda().eval()
        lv = R.prefix_liveness(m, arr, n_batches=1, batch_size=2)
        assert lv["measured_ln_input_width_by_layer"] == R.SCHEDULES[arm], arm
        assert lv["measured_active_width_matches_target"], arm
        assert lv["qkv_out_matches_3x_target"], arm
        assert lv["hidden_matches_4x_target"], arm
        assert lv["min_ratio"] > 0.0, f"{arm}: 每层真实前向必须活性非 0"


# ---------------------------------------------------------------- T3：变异必须抓红

def _block_out(blk, x):
    with torch.no_grad():
        return blk(x)[0]


def test_t3a_wrong_row_slice_is_not_caught_by_shapes_but_changes_forward():
    """切错**行**（q 块整体上移 1 行）：形状审计看不出来 ⇒ 只有逐位比对能抓。"""
    torch.manual_seed(0)
    stock = DepthARModel(R.stock_cfg(), VOCAB)
    src, w, d = stock.blocks[0].qkv.weight, 192, R.D_MODEL
    good = R.prefix_block_from_stock(stock.blocks[0], w)
    bad = R.prefix_block_from_stock(stock.blocks[0], w)
    with torch.no_grad():
        bad.qkv.weight.copy_(torch.cat(
            [src[1:1 + w, :w], src[d:d + w, :w], src[2 * d:2 * d + w, :w]], 0))
    assert bad.qkv.weight.shape == good.qkv.weight.shape, "变异不改形状"
    x = torch.randn(2, 4, R.D_MODEL, generator=torch.Generator().manual_seed(3))
    assert not torch.equal(_block_out(good, x), _block_out(bad, x)), \
        "T3a 变异必须被逐位比对抓红"


def test_t3b_wrong_column_slice_changes_forward():
    torch.manual_seed(0)
    stock = DepthARModel(R.stock_cfg(), VOCAB)
    w = 192
    good = R.prefix_block_from_stock(stock.blocks[0], w)
    bad = R.prefix_block_from_stock(stock.blocks[0], w)
    with torch.no_grad():
        bad.proj.weight.copy_(stock.blocks[0].proj.weight[:w, w // 6:w // 6 + w])
    assert bad.proj.weight.shape == good.proj.weight.shape
    x = torch.randn(2, 4, R.D_MODEL, generator=torch.Generator().manual_seed(3))
    assert not torch.equal(_block_out(good, x), _block_out(bad, x)), \
        "T3b 变异（切错列）必须被逐位比对抓红"


def test_t3c_swapping_two_layer_widths_changes_forward():
    """把某一层宽度换到另一层（多重集不变、分配变了）⇒ 前向必须变。"""
    base = [192] * R.N_LAYERS
    swapped = [224] + [192] * (R.N_LAYERS - 1)
    a = full_schedule_model(base, seed=0)
    b = full_schedule_model(swapped, seed=0)
    assert R.sum_w2(swapped) != R.sum_w2(base)
    x = tokens()
    with torch.no_grad():
        assert not torch.equal(a(x, exit_depths=None)[1]["depth_logits"][-1],
                               b(x, exit_depths=None)[1]["depth_logits"][-1])


def test_t3d_width_not_multiple_of_head_dim_is_rejected():
    torch.manual_seed(0)
    stock = DepthARModel(R.stock_cfg(), VOCAB)
    with pytest.raises(ValueError):
        R.prefix_block_from_stock(stock.blocks[0], 200)


def test_t3e_relabelling_width_without_reslicing_raises():
    """只改宽度标签不重切权重 ⇒ 前向必须**报错**（形状约束），不能被静默接受。"""
    m = full_schedule_model(R.SCHEDULES["EQ"])
    m.blocks[0].width = 224
    with pytest.raises(Exception):
        m(tokens(), exit_depths=None)


def test_t3f_narrowing_the_schedule_changes_the_model():
    """EQ（192）相对全宽 stock 必须真的不同（否则「干预没到模型」）。"""
    stock = stock_model(0)
    eq = full_schedule_model(R.SCHEDULES["EQ"], seed=0)
    x = tokens()
    with torch.no_grad():
        assert not torch.equal(eq(x, exit_depths=None)[1]["depth_logits"][-1],
                               stock(x, exit_depths=None)[1]["depth_logits"][-1])
    assert sum(p.numel() for p in eq.blocks.parameters()) < \
        sum(p.numel() for p in stock.blocks.parameters())


# ---------------------------------------------------------------- P3-②：因果性

@pytest.mark.parametrize("arm", ARMS)
def test_p3_causality_for_every_arm(arm):
    m = full_schedule_model(R.SCHEDULES[arm])
    x = tokens(B=2, T=32)
    with torch.no_grad():
        lg1 = m(x, exit_depths=None)[1]["depth_logits"][-1]
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % VOCAB
        lg2 = m(x2, exit_depths=None)[1]["depth_logits"][-1]
    assert (lg1[:, :-1] - lg2[:, :-1]).abs().max().item() == 0.0, \
        f"{arm}：改末位 token 不得影响之前位置（因果掩码）"
    assert (lg1[:, -1] - lg2[:, -1]).abs().max().item() > 0.0, \
        f"{arm}：末位必须变（活性对照）"


# ---------------------------------------------------------------- 复用纪律 / 配置对照

def test_analytic_cost_and_params_match_2641_U4_exactly():
    """代价与参数口径自检：全宽 schedule 必须逐位还原 2641 U4-s0 的落盘值。"""
    full = [R.D_MODEL] * R.N_LAYERS
    ac = R.analytic_cost(full, vocab=VOCAB)
    ap = R.analytic_params(full, vocab=VOCAB)
    u4 = R.load_anchor("U4")
    assert ac["macs_per_token"]["total"] == u4["macs_per_token"] == 7_392_768
    assert ac["macs_per_token"]["attention"] == u4["macs_attn"] == 3_145_728
    assert ac["macs_per_token"]["ffn"] == u4["macs_ffn"] == 4_194_304
    assert ac["macs_per_token"]["readout_tied"] == u4["macs_readout"] == 52_736
    assert ap["params_total"] == u4["params_total"] == 6_962_688


def test_anchor_reuse_reads_2641_files_and_config_parity():
    u4, u2 = R.load_anchor("U4"), R.load_anchor("U2")
    assert u4["clean_bpc"] == 1.5314 and u4["params_total"] == 6_962_688
    assert u2["clean_bpc"] == 1.5709
    par = R.anchor_config_parity(12000, 32, 1e-3)
    assert par["all_equal"], par["mismatched_fields"]


def test_shared_argv_is_identical_across_arms():
    """干预不在 CLI（引擎无逐层 d_model）⇒ 所有臂 argv 逐位相同（单因素声明）。"""
    argvs = {a: R.shared_argv(a, 0, 12000, 32, 1e-3, "x") for a in ARMS}
    # 唯一允许的差异 = run_name（argv[1]）——其余 CLI 逐字对齐。
    assert len({tuple(v[2:]) for v in argvs.values()}) == 1
    assert {v[1] for v in argvs.values()} == {f"{a}-s0-12k" for a in ARMS}
    joined = " ".join(argvs["EQ"])
    assert "--d 256" in joined and "--layers 8" in joined and "--heads 8" in joined
    assert "--dense" in joined and "--topology sequential" in joined


def test_analytic_params_are_history_independent_but_arm_dependent():
    """PYR/HG/APYR 解析参数**逐位相同**（同一多重集 ⇒ 同一 Σw² 与 Σw）；
    EQ 只差在 Σw（Σw² 逐位相等，差 = 13·ΔΣw，必须在分析里逐位报告）。"""
    p = {a: R.analytic_params(R.SCHEDULES[a], vocab=VOCAB) for a in ARMS}
    assert p["PYR"]["params_total"] == p["HG"]["params_total"] == p["APYR"]["params_total"]
    assert p["PYR"]["params_total"] != p["EQ"]["params_total"]
    for a in ARMS:
        assert p[a]["sum_w2"] == 8 * 192 ** 2
    delta_w = sum(R.SCHEDULES["EQ"]) - sum(R.SCHEDULES["PYR"])
    assert p["EQ"]["params_total"] - p["PYR"]["params_total"] == 13 * delta_w
    assert delta_w == 64
