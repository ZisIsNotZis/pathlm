"""ENG1（ticket 39-X2）· 引擎可组合性的**可失败单测**。

ENG1 前：`carry_rule`、`n_kv_heads < n_heads`（GQA/MQA）、`train_window > 0` 三特性
硬互斥，且块类二选一（`GQABlock` vs `CarryChannelBlock`）⇒ `GQA × carry × window`
在引擎里不存在。本文件钉住四件事：

① 三种两两组合（GQA×carry / GQA×window / carry×window）与三者齐开都能建起来，
   且每个特性都**真的活着**（不是只把守卫删掉）；
② 缺省路径（全关）与 stock `Block` 逐位一致（ENG1 不改缺省行为）；
③ 单特性路径的块类不变（`GQABlock` / `CarryChannelBlock` 仍是快路径）。

变异验证（每处抓红 + 逐字节还原）见
`axes/1-time/1.1.4.1-explicit-memory/experiments/2622-eng1-composable/mutate.py`。
"""

import torch

from pathlm.config import ModelConfig
from pathlm.depth_ar import ComposableBlock, DepthARConfig, DepthARModel
from pathlm.model import Block


def _model(**kw):
    """ENG1 的测试位点：d=32 / 8 头 / dh=4，dense + sequential（单因素）。"""
    torch.manual_seed(0)
    cfg = DepthARConfig(d_model=32, n_layers=3, n_heads=8, seq_len=12,
                        dense=True, **kw)
    return DepthARModel(cfg, vocab_size=10)


def _tokens():
    return torch.randint(0, 10, (2, 12))


def _logits(model, x):
    with torch.no_grad():
        return model(x)[1]["depth_logits"][-1].float()


def _activate_carry(model, value=0.5):
    """把读出增益从单边零激活（否则通道在 init 处恒等、看不出「被读」）。用**同一
    常量**填充（而不是随机）⇒ 两臂（窗口臂 / 全窗口臂）激活后权重仍逐位相同。"""
    for blk in model.blocks:
        blk.carry_scale.data.fill_(value)


def _assert_same_weights(m1, m2):
    """同 seed ⇒ 逐字段权重逐位相同（窗口开关不消耗随机数）。"""
    p1, p2 = list(m1.named_parameters()), list(m2.named_parameters())
    assert [n for n, _ in p1] == [n for n, _ in p2]
    for (n1, w1), (_, w2) in zip(p1, p2):
        assert torch.equal(w1, w2), n1


# ---------------- ① GQA × carry（ENG1 前块类二选一 ⇒ 组合不存在） ----------------

def test_eng1_gqa_x_carry_composes_and_both_active():
    """GQA × carry 现在 = `ComposableBlock`，且 KV 头投影真的变窄 + 写入通道真的被读。"""
    x = _tokens()
    m = _model(carry_rule="delta", carry_chunk=64, n_kv_heads=2)
    assert type(m.blocks[0]).__name__ == "ComposableBlock"
    d, nkv, dh = 32, 2, 4
    # ① KV 头投影活着：qkv 输出宽度 = d + 2·n_kv_heads·dh（不是 3d）
    assert tuple(m.blocks[0].qkv.weight.shape) == (d + 2 * nkv * dh, d)
    assert m.blocks[0].kv_telemetry is None                # 还没前向
    _logits(m, x)
    assert m.blocks[0].kv_telemetry["k_shape"][1] == nkv   # 真进 SDPA 的 K/V 头数
    # ② 写入通道活着：激活 γ 后读出量非零，置零 γ 会改变 logits
    _activate_carry(m)
    with torch.no_grad():
        _, aux = m(x)
    assert float(aux["attn_carry"][0]["read_share"]) > 0.0
    with torch.no_grad():
        with_carry = _logits(m, x)
        scales = [b.carry_scale.detach().clone() for b in m.blocks]
        for b in m.blocks:
            b.carry_scale.zero_()
        without = _logits(m, x)
        for b, s in zip(m.blocks, scales):
            b.carry_scale.copy_(s)
    assert not torch.equal(with_carry, without)


# ---------------- ② GQA × window（窗口是与块类正交的装饰层） ----------------

def test_eng1_gqa_x_window_composes_and_window_active():
    """GQA × train_window 能同开：块类仍是 `GQABlock`，窗口经 `attn_mask` 正交接入，
    且掩码真的改变前向（同 seed、同权重、全窗口对照逐位不同）。"""
    x = _tokens()
    m_win = _model(n_kv_heads=2, train_window=4)
    m_full = _model(n_kv_heads=2, train_window=12)         # 12 = seq_len ⇒ 恒等因果
    assert type(m_win.blocks[0]).__name__ == "GQABlock"    # 单特性仍走快路径
    assert tuple(m_win.blocks[0].qkv.weight.shape)[0] == 32 + 2 * 2 * 4  # GQA 活着
    _assert_same_weights(m_win, m_full)
    assert not torch.equal(_logits(m_win, x), _logits(m_full, x))  # 窗口真的生效
    assert m_win.blocks[0].kv_telemetry["attn_mask"] is True       # 掩码进了 SDPA


# ---------------- ③ carry × window（旧护栏已允许，此处作回归钉） ----------------

def test_eng1_carry_x_window_composes_and_both_active():
    """carry × train_window：窗口造需求、通道提供路径——两道干预同开且各自活着。"""
    x = _tokens()
    m_win = _model(carry_rule="plain", carry_chunk=64, train_window=4)
    m_full = _model(carry_rule="plain", carry_chunk=64, train_window=12)
    assert type(m_win.blocks[0]).__name__ == "CarryChannelBlock"
    _activate_carry(m_win)
    _activate_carry(m_full)
    with torch.no_grad():
        _, aux = m_win(x)
    assert float(aux["attn_carry"][0]["read_share"]) > 0.0      # 通道活着
    _assert_same_weights(m_win, m_full)
    assert not torch.equal(_logits(m_win, x), _logits(m_full, x))  # 窗口活着


# ---------------- ④ 三者齐开（ENG1 前完全不存在的组合） ----------------

def test_eng1_gqa_x_carry_x_window_all_compose():
    """三者齐开：KV 头投影变窄 + 写入通道读出非零 + 窗口掩码生效——三个活性量同时成立。"""
    x = _tokens()
    m = _model(carry_rule="delta", carry_chunk=64, n_kv_heads=2, train_window=4)
    m_full = _model(carry_rule="delta", carry_chunk=64, n_kv_heads=2,
                    train_window=12)
    assert type(m.blocks[0]).__name__ == "ComposableBlock"
    d, nkv, dh = 32, 2, 4
    assert tuple(m.blocks[0].qkv.weight.shape) == (d + 2 * nkv * dh, d)  # ① GQA
    _activate_carry(m)
    _activate_carry(m_full)
    with torch.no_grad():
        _, aux = m(x)
    assert float(aux["attn_carry"][0]["read_share"]) > 0.0   # ② carry
    assert m.blocks[0].kv_telemetry["k_shape"][1] == nkv     # ① 真进 SDPA 的形状
    assert m.blocks[0].kv_telemetry["attn_mask"] is True     # ③ 窗口进了 SDPA
    _assert_same_weights(m, m_full)
    assert not torch.equal(_logits(m, x), _logits(m_full, x))  # ③ 窗口真的生效


# ---------------- ⑤ 缺省路径逐位还原 v1（ENG1 不改缺省行为） ----------------

def test_eng1_composable_default_matches_stock_block_bitwise():
    """`ComposableBlock`（MHA、无 carry）与 stock `Block` 同 seed 下状态字典与前向
    **逐位**一致 ⇒ 缺省路径逐位还原 v1。"""
    mcfg = ModelConfig(d_model=32, n_layers=1, n_heads=8, seq_len=12)
    torch.manual_seed(0)
    ref = Block(mcfg)
    torch.manual_seed(0)
    comp = ComposableBlock(mcfg)                # n_kv_heads=None, carry_on=False    assert list(ref.state_dict().keys()) == list(comp.state_dict().keys())
    for (n1, p1), (_, p2) in zip(ref.state_dict().items(), comp.state_dict().items()):
        assert torch.equal(p1, p2), n1
    x = torch.randn(2, 12, 32)
    with torch.no_grad():
        assert torch.equal(ref(x)[0], comp(x)[0])


def test_eng1_single_feature_fast_paths_unchanged():
    """单特性路径的块类不变：GQA-only → `GQABlock`、carry-only → `CarryChannelBlock`、
    缺省 → `Block`（ENG1 只把「组合」接到新块，不动快路径）。"""
    assert type(_model(n_kv_heads=2).blocks[0]).__name__ == "GQABlock"
    assert type(_model(carry_rule="plain", carry_chunk=64).blocks[0]).__name__ \
        == "CarryChannelBlock"
    assert type(_model().blocks[0]).__name__ == "Block"
    assert type(_model(carry_rule="plain", carry_chunk=64,
                       n_kv_heads=2).blocks[0]).__name__ == "ComposableBlock"
