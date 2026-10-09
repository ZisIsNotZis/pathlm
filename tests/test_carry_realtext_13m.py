"""2634-carry-realtext-13m · 真实文本口径仪器的**可失败单测**（CPU、秒级）。

靶子 = 本实验自己的仪器 `realtext_probe.py` 与 `config.json` 的预登记布局：

① **布局前提**（契约判据①）：被评行的 L 层多跳感受野必须完全落在未毁区 —— 比它小的
   `eval_lo` 必须被**拒绝**（否则「通道搬来的」与「注意力看见的」不可分）；
② **毁源探针的语义**：无通道臂对被毁前缀逐位不敏感（`max|Δlogit| == 0`、`Δbpc_far == 0`）；
   通道臂在**单边零起步**（γ=0）时同样逐位恒等；把 γ 手动置为非零后**必须**变敏感
   ——否则这台仪器证明不了任何「搬运」；
③ **通道消融**（`channel_off`）：参数逐位还原、消融后与无通道一致；
④ 统计层：按窗口聚类的配对 bootstrap 的算术；
⑤ 配置不变量：`rf == n_layers·(window−1)`、`eval_lo − rf ≥ corrupt_hi`、
   四臂 = {none, delta} × {0, 1}、可见性与模型配置一致。
"""

import json
import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = os.path.join(ROOT, "axes", "M-carrier", "M1-carrier-type", "experiments",
                   "2634-carry-realtext-13m")
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP)

from pathlm.depth_ar import (DepthARConfig, DepthARModel,          # noqa: E402
                             window_anchor_keep_mask)
from tools.evalkit.retention import masked_bpc                     # noqa: E402
import realtext_probe as rp                                        # noqa: E402
import needle_stats as ns                                          # noqa: E402

CFG = json.load(open(os.path.join(EXP, "config.json")))
T = 128                     # 测试用小 T（比主干 256 小 ⇒ 只测语义，不测尺度）
WINDOW = 8                  # 测试用小窗口
LAYERS = 2
RF = LAYERS * (WINDOW - 1)
CORRUPT_HI = 16
EVAL_LO = CORRUPT_HI + RF          # 恰好等于前提边界
VOCAB = 206


def _eval_arr(n: int = 40000) -> np.ndarray:
    rng = np.random.RandomState(0)
    return rng.randint(0, VOCAB - 1, n).astype(np.int64)


def _model(rule: str | None, *, layers: int = LAYERS, window: int = WINDOW):
    torch.manual_seed(0)
    m = DepthARModel(DepthARConfig(d_model=32, n_layers=layers, n_heads=2, seq_len=T,
                                   mlp_mult=2, dense=True, topology="sequential",
                                   train_window=window, train_anchors=0,
                                   carry_state_dim=None, carry_rule=rule), VOCAB)
    m.eval()
    return m


def _probe(model, keep, **kw):
    args = dict(n_batches=2, batch_size=4, corrupt_hi=CORRUPT_HI, eval_lo=EVAL_LO,
                eval_hi=T - 1, rf=RF, device="cpu")
    args.update(kw)
    return rp.far_source_probe(model, _eval_arr(), keep, **args)


# ---------------- ① 布局前提必须可失败 ----------------

def test_layout_leak_is_rejected():
    keep = window_anchor_keep_mask(T, WINDOW, 0)
    m = _model(None)
    with pytest.raises(AssertionError, match="布局泄漏"):
        rp.far_source_probe(m, _eval_arr(), keep, n_batches=1, batch_size=2,
                            corrupt_hi=CORRUPT_HI, eval_lo=EVAL_LO - 1, eval_hi=T - 1,
                            rf=RF, device="cpu")


def test_layout_accepts_exact_boundary():
    keep = window_anchor_keep_mask(T, WINDOW, 0)
    out = _probe(_model(None), keep)                      # eval_lo − rf == corrupt_hi
    assert out["layout"]["min_distance_of_corrupted_prefix_from_eval_rows"] == RF


# ---------------- ② 探针语义 ----------------

def test_no_channel_is_bitwise_insensitive():
    """前提①：无通道臂对被毁前缀逐位不敏感（多跳注意力也够不着）。"""
    keep = window_anchor_keep_mask(T, WINDOW, 0)
    out = _probe(_model(None), keep)
    assert out["max_abs_logit_diff_on_eval_rows"] == 0.0
    assert out["delta_bpc_far"] == 0.0


def test_channel_identity_start_is_insensitive():
    """单边零起步（γ=0）⇒ 通道臂也逐位恒等（SCHEMA §10 的恒等起步在仪器上可见）。"""
    keep = window_anchor_keep_mask(T, WINDOW, 0)
    out = _probe(_model("delta"), keep)
    assert out["max_abs_logit_diff_on_eval_rows"] == 0.0
    assert out["delta_bpc_far"] == 0.0


def test_channel_with_nonzero_gain_is_sensitive():
    """把读出增益置为非零 ⇒ 被毁前缀必须改变被评行（否则仪器测不出搬运）。"""
    keep = window_anchor_keep_mask(T, WINDOW, 0)
    m = _model("delta")
    with torch.no_grad():
        for blk in m.blocks:
            blk.carry_scale.fill_(1.0)
    out = _probe(m, keep)
    assert out["max_abs_logit_diff_on_eval_rows"] > 0.0
    assert out["ce_far_corrupt"] != out["ce_far_clean"]


def test_multi_layer_receptive_field_is_respected():
    """把 eval_lo 设到边界内一步 ⇒ 单层可见集虽干净、多跳仍可达 ⇒ 必须拒绝。"""
    keep = window_anchor_keep_mask(T, WINDOW, 0)
    m = _model(None)
    with pytest.raises(AssertionError):
        rp.far_source_probe(m, _eval_arr(), keep, n_batches=1, batch_size=2,
                            corrupt_hi=CORRUPT_HI, eval_lo=T - 1 - RF + CORRUPT_HI - 1,
                            eval_hi=T - 1, rf=RF, device="cpu")


# ---------------- ③ 通道消融 ----------------

def test_channel_off_restores_bitwise_and_matches_none():
    keep = window_anchor_keep_mask(T, WINDOW, 0)
    m = _model("delta")
    with torch.no_grad():
        for blk in m.blocks:
            blk.carry_scale.fill_(1.0)
    before = [blk.carry_scale.detach().clone() for blk in m.blocks]
    gate_before = m.carry.gate_w.detach().clone()
    with rp.channel_off(m):
        abl = _probe(m, keep)
        assert all(bool((p == 0).all()) for p in
                   [blk.carry_scale for blk in m.blocks])
    assert all(torch.equal(a, b) for a, b in zip(before,
                                                 [blk.carry_scale for blk in m.blocks]))
    assert torch.equal(gate_before, m.carry.gate_w)
    assert abl["max_abs_logit_diff_on_eval_rows"] == 0.0      # 消融 ⇒ 与无通道一致


def test_channel_off_is_noop_without_channel():
    m = _model(None)
    with rp.channel_off(m):
        pass
    assert not hasattr(m, "carry")


# ---------------- ④ 统计层 ----------------

def test_paired_cluster_bootstrap_zero_difference():
    d = [0.0] * 8
    n = [10] * 8
    ci = rp.paired_cluster_bootstrap_ci(d, n, n_boot=200, seed=1)
    assert ci == [0.0, 0.0]


def test_paired_cluster_bootstrap_positive_shift():
    d = [1.0] * 8
    n = [10] * 8
    ci = rp.paired_cluster_bootstrap_ci(d, n, n_boot=200, seed=1)
    assert ci[0] > 0.09 and ci[1] == pytest.approx(0.1, abs=1e-9)


def test_paired_cluster_bootstrap_rejects_mismatch():
    with pytest.raises(ValueError):
        rp.paired_cluster_bootstrap_ci([1.0, 2.0], [10])


# ---------------- ⑥ rf 必填 / probe_kwargs 接线（复核 P1-1） ----------------

def test_far_source_probe_requires_rf():
    """`rf` 是必填关键字参数：漏传必须报错（默认 0 会让泄漏断言退化成恒真式）。"""
    keep = window_anchor_keep_mask(T, WINDOW, 0)
    m = _model(None)
    with pytest.raises(TypeError):
        rp.far_source_probe(m, _eval_arr(), keep, n_batches=1, batch_size=2)
    with pytest.raises(ValueError, match="rf"):
        rp.far_source_probe(m, _eval_arr(), keep, rf=0, n_batches=1, batch_size=2)


def test_probe_kwargs_derives_rf_and_cross_checks_config():
    pkw = rp.probe_kwargs(CFG)
    assert pkw["rf"] == CFG["model"]["n_layers"] * (CFG["model"]["train_window"] - 1) == 180
    assert pkw["eval_lo"] - pkw["rf"] >= pkw["corrupt_hi"]
    # config 里的 rf 与模型规格不一致 ⇒ 报错（不接受手工填空）
    bad = json.loads(json.dumps(CFG))
    bad["eval"]["probe"]["rf"] = 999
    with pytest.raises(ValueError, match="rf"):
        rp.probe_kwargs(bad)
    # 无窗模型：本仪器无定义
    nowin = json.loads(json.dumps(CFG))
    nowin["model"]["train_window"] = 0
    nowin["eval"]["probe"].pop("rf", None)
    with pytest.raises(ValueError, match="train_window"):
        rp.probe_kwargs(nowin)


def test_module_defaults_are_leak_free_for_canonical_rf():
    rf = CFG["model"]["n_layers"] * (CFG["model"]["train_window"] - 1)
    assert rp.DEFAULT_EVAL_LO - rf >= rp.DEFAULT_CORRUPT_HI
    # 且 canonical 布局必须与 config 记录的一致
    assert (rp.DEFAULT_CORRUPT_HI, rp.DEFAULT_EVAL_LO, rp.DEFAULT_EVAL_HI) == (
        CFG["eval"]["probe"]["corrupt_hi"], CFG["eval"]["probe"]["eval_lo"],
        CFG["eval"]["probe"]["eval_hi"])


def test_run_py_calls_probe_only_via_probe_kwargs():
    """**接线钉住**：run.py 的每次 `far_source_probe(` 都必须吃 `**pkw`（含 rf）。

    这条测试的理由：rf 曾被默认值 0 漏传 ⇒ 硬断言退化成恒真。断言 rf 必填之外，
    再把「调用点接线」也钉住（两个调用点都要）。
    """
    src = open(os.path.join(EXP, "run.py")).read()
    assert "rp.probe_kwargs(cfg)" in src
    n_calls = src.count("far_source_probe(")
    n_pkw = src.count("**pkw")
    assert n_calls == 2 and n_pkw >= n_calls, f"calls={n_calls} pkw={n_pkw}"


def test_rf_scan_artifact_is_consistent():
    p = os.path.join(EXP, "rf_scan.json")
    if not os.path.exists(p):
        pytest.skip("rf_scan.json 尚未生成")
    r = json.load(open(p))
    assert r["all_scan_ok"] is True
    canon = r["scans"]["canonical_T256_L12_w16_corrupt64"]
    assert canon["first_insensitive_row"] >= canon["predicted_first_insensitive_row"] - 1
    assert canon["max_abs_logit_diff_at_244"] == 0.0
    assert r["leaked_layout_is_sensitive"] is True


# ---------------- ⑦ needle 统计层（复核 P1-2 / P1-4） ----------------

def _needle(hits: list[int]) -> list[dict]:
    return [{"bucket": "beyond", "hit": h, "dist": 200 + i} for i, h in enumerate(hits)]


def test_beyond_above_floor_direction_and_can_pass():
    """判据④必须**可达**：构造一个「植入明显高于地板」的输入，它必须通过。"""
    ok = ns.beyond_above_floor(planted_hits=600, planted_n=1000,
                               control_hits=500, control_n=1000)
    assert ok["passes"] is True and ok["delta_pp"] == pytest.approx(10.0, abs=0.01)
    assert ok["diff_ci_95"][0] > 0
    null = ns.beyond_above_floor(500, 1000, 500, 1000)
    assert null["passes"] is False and null["delta_pp"] == 0.0
    worse = ns.beyond_above_floor(400, 1000, 500, 1000)
    assert worse["passes"] is False and worse["delta_pp"] < 0
    # 旧实现（把 planted 传在 a、control 传在 b）在同一份「应通过」输入上会判否
    # ⇒ 这条断言把「方向传反 ⇒ 判据永不可达」这个 bug 钉住。
    from tools.evalkit.carry_channel import two_proportion
    reversed_old = two_proportion(600, 1000, 500, 1000)["diff_ci_95"][0]
    assert reversed_old < 0


def test_hard_index_sets_are_shared_across_arms():
    a = _needle([0, 1, 1, 0, 1])          # control 失败索引 {0, 3}
    b = _needle([0, 0, 1, 0, 1])          # control 失败索引 {0, 1, 3}
    sets = ns.hard_index_sets({"a": a, "b": b})
    assert sets["intersection"] == [0, 3]
    assert sets["union"] == [0, 1, 3]
    assert sets["per_arm_control_miss"] == {"a": 2, "b": 3}
    with pytest.raises(ValueError):
        ns.hard_index_sets({"a": a, "b": b[:3]})


def test_recall_on_indices_is_index_aligned():
    """同一索引集下两臂的 n 必须相同（否则跨臂比较不可比），hits 各按自己的记录算。"""
    a = _needle([1, 0, 1, 1, 0])
    b = _needle([0, 0, 1, 0, 0])
    idx = [0, 2, 3]
    ra, rb = ns.recall_on_indices(a, idx), ns.recall_on_indices(b, idx)
    assert ra["n"] == rb["n"] == 3
    assert (ra["hits"], ra["recall"]) == (3, 1.0)
    assert (rb["hits"], rb["recall"]) == (1, 0.333333)
    assert ns.recall_on_indices(a, [])["n"] == 0


def test_judge_aggregation_rule_is_per_intervention_arm():
    planted = {"none_s0": {"hits": 10, "n": 100}, "delta_s0": {"hits": 20, "n": 100}}
    control = {"none_s0": {"hits": 10, "n": 100}, "delta_s0": {"hits": 10, "n": 100}}
    j = ns.judge(planted, control, ("delta_s0",))
    assert j["per_arm"]["delta_s0"]["passes"] is True
    assert j["per_arm"]["none_s0"]["passes"] is False
    assert j["intervention_arms_all_pass"] is True and j["all_arms_pass"] is False
    assert j["any_arm_passes"] is True


# ---------------- ⑤ bpc 仪器对齐与配置不变量 ----------------

def test_win16_bpc_matches_repo_instrument():
    keep = window_anchor_keep_mask(T, WINDOW, 0)
    m = _model(None)
    arr = _eval_arr()
    a = rp.win16_bpc(m, arr, keep, n_batches=2, batch_size=4, window_seed=2, device="cpu")
    b = masked_bpc(m, arr, keep, n_batches=2, batch_size=4, window_seed=2, device="cpu")
    # 同一算法、同一窗口序；容差 1e-6 是因为逐窗口比特数走 `reduction="none"` 再
    # 自己求和（配对 bootstrap 需要逐窗口值），浮点累加顺序与仓库版不同。
    assert a == pytest.approx(b, abs=1e-6)
    # per_window=True 的返回必须自洽：总比特/总 token == 标量版
    val, bits, n_tok = rp.win16_bpc(m, arr, keep, n_batches=2, batch_size=4,
                                    window_seed=2, device="cpu", per_window=True)
    assert val == pytest.approx(b, abs=1e-6)
    assert sum(bits) / sum(n_tok) == pytest.approx(a, abs=1e-6)


def test_config_layout_invariants():
    probe = CFG["eval"]["probe"]
    model = CFG["model"]
    w, L = model["train_window"], model["n_layers"]
    assert probe["rf"] == L * (w - 1)
    assert probe["eval_lo"] - probe["rf"] >= probe["corrupt_hi"]
    assert probe["eval_hi"] < model["seq_len"]
    assert CFG["arms"] and len(CFG["arms"]) == 4
    assert {(a["rule"], a["seed"]) for a in CFG["arms"]} == {
        ("none", 0), ("delta", 0), ("none", 1), ("delta", 1)}
    assert CFG["visibility"]["train_window"] == model["train_window"] == 16
    assert CFG["visibility"]["train_anchors"] == model["train_anchors"] == 0
    assert CFG["train_recipe"]["tokens_per_arm"] == 12000 * 32 * 256


def test_config_needle_leg_is_paired_and_anchorless():
    n = CFG["needle"]
    assert n["bucket_anchors"] == 0, "无锚区可见性 ⇒ 桶只能是 window/beyond"
    assert n["phrase_len"] + 1 <= CFG["model"]["seq_len"]
    assert n["n_batches"] >= 32, "地板/主臂要配对到足够针数"
