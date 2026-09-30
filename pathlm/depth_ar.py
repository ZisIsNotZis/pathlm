"""深度自适应 AR（方向修订后的新形态，mental_model §0 / ticket 23）。

纯 AR（无 corruption / 无 retry / 无 mask 站），渐进精化 + 早退 + 自 spec：

- **全深度并行监督**：depth 0..L 每层都有 tied-unembed fp32 读出
  （depth 0 = embedding 直连读出），各深度 CE 等权平均（v1；ticket 23b 起
  支持浅层 aux 降权 `shallow_weight`，最终深度恒权重 1）。
- **逐通道携带门**：h_{k+1} = h_k + g_k ⊙ Refine_k(h_k)，
  g_k = σ(Gate_k(h_k))，Gate 偏置负初始化（carry：g≈0，恒等起步）。
- **随机退出深度（训练）**：每个 position 采样 d_i ∈ {0..L}。位置 i 在
  层 k ≤ d_i 正常精化；k > d_i 后状态冻结在 h_{d_i}[i]，且它在层 k 的
  KV = 层 k 的 kv 投影作用于冻结表征（ln1 → W_qkv 的 k/v 半边）——
  混合深度 context。fill 语义由 tests/test_depth_ar.py 钉住。
- **推理提交**：soft（Σp·E[t]，默认）/ hard（argmax→embed）/
  latent（ablation：直通退出深度状态，预期漂移）。

对照 arm：dense=True 退化为普通 AR transformer（无门/无退出/仅最终 CE），
即 dense-B 参照。训练入口 train_depth_ar.py。
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .model import Block, cap_norm

LN2 = 0.6931471805599453  # nats → bits

#: 线性注意力的归一化分母下限（φ = elu+1 ≥ 0 且对角恒贡献 ⇒ 分母恒 > 0）。
LINEAR_ATTN_EPS = 1e-6
#: 线性注意力训练实装的分块大小（chunk）缺省。它是**实现细节**（单测钉住：改
#: chunk 只改浮点累加顺序，不改语义），不是第二个干预档。
DEFAULT_LINEAR_CHUNK = 64


@dataclass
class DepthARConfig:
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    seq_len: int = 256
    mlp_mult: int = 4
    dropout: float = 0.0
    # 输入侧 norm cap（与 PathLM 约定一致；只 cap embedding 输入，不 cap 残差流）
    norm_cap: float = 1.0
    # carry 门偏置初始化：σ(-2)≈0.12 起步（恒等近似 + 梯度可流）；
    # 测试用 -30 验证精确恒等极限。
    gate_bias_init: float = -2.0
    # 对照 arm：纯 dense AR（无门 / 无退出采样 / 仅最终深度 CE）
    dense: bool = False
    # 混合深度 context 填充策略（ticket 23e）：True = v1 fill（已退出行 KV 用
    # 冻结表征经本层 kv 投影填充，W0 复现点）；False = no-fill ragged（早退
    # 位置在未跑层的 KV 中缺席，原引擎 per-layer cache 语义，decode.Decoder 同款）
    fill_kv: bool = True
    # proj-fill 臂（ticket 23e 可选）：fill 基础上给已退出行 KV 源加逐层线性
    # adapter（恒等初始化——起步严格等于 fill，remap 从 fill 出发学）。需 fill_kv
    proj_fill: bool = False
    # 结构切分（ticket 25 B 臂）：读出维度数 r ≤ d_model。unembed（全部出口头
    # 与最终头共用同一 tied 读出）只读 h[..., :r]，tied 表取列切片 E[:, :r]；
    # trunk 与 embedding 端保持全维度——workspace 维（r..d）经残差参与全层
    # 计算但不进 logits。None = 全维度读出（v1 复现点，逐位一致）。
    readout_dims: int | None = None
    # gated 残差读出（W4 主臂）：读出前加一个非线性状态变换
    # h → h + SiLU(W1 h) ⊙ (W2 h)（W1/W2 无偏置），再走 tied 线性读出
    # h·Eᵀ。tied 表仍是唯一读出基（prob0 / Σp·E 凸包 / latent 投影语义不变）；
    # 参数量 2·d² 可忽略且可 scale（非 vocab×hidden MLP 读出）。缺省 False = v1
    # 逐位还原。
    # init：W1 ~ N(0, 0.02)、**W2 = 0** —— 起步 transform = SiLU(W1 h)⊙0 = 0
    # ⇒ 严格恒等起步（单测钉住）。**不能用 W1=W2=0**：该点两端梯度
    # ∂L/∂W2 ∝ SiLU(W1h)⊗h = SiLU(0)⊗h = 0、∂L/∂W1 ∝ diag(W2h)·SiLU'(W1h)·(h⊗I) = 0
    # 同时为零 ⇒ 永不可学的鞍点（已实测：零初始 12k 步后 |W1|=|W2|=0）。
    readout_gated: bool = False
    # 线性等参数对照读出（W4 契约 B，2611）：读出前加一个**纯线性**状态变换
    # h → h + P·(Q h)（P、Q 各 d×d 无偏置），再走 tied 线性读出。与 gated 臂
    # **参数量完全相同**（2·d²），但整体等价于一个线性映射
    # (I + PQ) h——用来判定 gated 臂的增益来自参数量还是非线性。
    # init：Q ~ N(0, 0.02)、**P = 0**（单边零，与 gated 臂同款）⇒ 起步
    # transform = P(Qh) = 0 严格恒等起步（单测钉住）；∂L/∂P = grad ⊗ (Qh) ≠ 0
    # ⇒ 非鞍点。**不能 P=Q=0**（同 gated 的零初始鞍点）。与 readout_gated 互斥。
    readout_linear_resid: bool = False
    # 层间拓扑（ticket 30 契约 E，唯一变量 = 层与层之间的接法）。只在
    # dense=True（stock decoder-only 形态：无门/无退出/仅最终 CE）下开放——
    # 层间接法与退出/填充语义耦合时无法单因素归因。缺省 "sequential" 逐位还原 v1。
    #   sequential        h = h + attn(ln1 h)；h = h + mlp(ln2 h)（现有 Block）
    #   parallel          h = h + attn(ln h) + mlp(ln h)（**单个共享 pre-LN**，
    #                     PaLM/GPT-J 原式；参数比 sequential 少 2d = 512）
    #   parallel-dual-ln  两个子层各用自己的 pre-LN（ln1 给 attn、ln2 给 mlp），
    #                     **都作用于同一块输入 h**——与 Block 参数量逐位相同的
    #                     并行子层变体（SCHEMA §15 的等参数位点）
    #   dense             DenseFormer 式：层 k 输入 = Σ_{j<k} α_kj·h_j，α 打包为
    #                     L(L+1)/2 标量，初始化为「只取 j=k−1」（= 顺序残差）。
    #                     该点非鞍点（α 非全零 + h_j 非零），无需单边零技巧。
    topology: str = "sequential"
    # 内容 top-k 稀疏注意力（ticket 31 契约 J，A6 混合算子首探）。0 = 关闭
    # （缺省，逐位还原 v1：stock 因果 SDPA）。k > 0 = 每个查询在**因果掩码内**
    # 只保留内容相似度（q·kᵀ）最大的 k 个键，其余置 −inf 后 softmax —— 选择
    # **按内容、不按位置**（位置选择 = 最近的 k 个键，只作为对照基线报重合度）。
    # 只在 dense=True（stock decoder-only：无门/无退出/无填充）下开放：退出/填充
    # 路径的稀疏注意力语义未定义，混在一起会破坏单因素归因。
    attn_sparse_k: int = 0
    # 训练期可见窗口（ticket 32 契约 O，A9/A3 「信息能否往右压」的直接检验）。
    # 唯一变量 = 训练时每个查询能看到多远的 key：
    #   keep[i, j] = (j ≤ i) ∧ ((i − j) < train_window ∨ j < train_anchors)
    # 评测用**同一个** `window_anchor_keep_mask` 构造掩码（训练/评测掩码一致，
    # 契约 P 第 ③ 项）。train_window = 0（缺省）= 关闭 ⇒ 走 stock 因果 SDPA
    # 路径，逐位还原 v1（单测钉住）。只在 dense=True + topology="sequential"
    # 下开放（退出/填充/并行接线下的窗口语义未定义，单因素）。
    train_window: int = 0
    train_anchors: int = 0
    # 线性注意力（ticket 33 契约 R，A6 混合算子续）。唯一变量 = 注意力算子：
    #   False（缺省）= stock 全因果 softmax 注意力（SDPA 融合核），逐位还原 v1；
    #   True = 归一化线性核注意力 out_i = Σ_{j≤i} (φ(q_i)·φ(k_j)) v_j / Σ φ(q_i)·φ(k_j)，
    #          φ = elu+1（`linear_attn_phi`）。用**累积状态**实现 ⇒ 每层时间/显存
    #          对 T 线性（不物化 [T,T] 分数矩阵，也不需要全量 KV）。
    # 不新增任何参数（块与 Block 逐位同构，单测钉住）⇒ SCHEMA §15 的等参数要求
    # 自动满足。只在 dense=True + topology="sequential" 下开放（单因素），且与
    # attn_sparse_k / train_window 互斥（一次只改一个因素）。
    attn_linear: bool = False
    # 线性注意力训练实装的分块大小（实现细节；见 DEFAULT_LINEAR_CHUNK 注释）
    linear_chunk: int = DEFAULT_LINEAR_CHUNK
    # KV 头共享（ticket 33 契约 S，W3 参照线）：注意力里 K/V 的**头数**（不是头
    # 维度）。缺省 None = MHA，即 n_kv_heads == n_heads，走 stock `Block` 路径、
    # 逐位还原 v1；< n_heads 时走 `GQABlock`：K/V 投影只产生 n_kv_heads 个头
    # （GQA；n_kv_heads = 1 即 MQA），由 SDPA 的 `enable_gqa` 按组广播给 query 头，
    # 因此 KV cache 的每位置元素数从 2·d 降到 2·n_kv_heads·(d/n_heads)
    # （n_kv_heads = 2 时 −75%、= 1 时 −87.5%）。头维度恒 = d/n_heads（与 query
    # 头同维，否则点积无定义）⇒ 必须整除 n_heads。只在 dense=True +
    # topology="sequential" 下开放（退出/填充/并行接线下的 KV 语义未定义，
    # 单因素）。n_kv_heads == n_heads 等价于缺省（同样走 `Block`）。
    n_kv_heads: int | None = None


TOPOLOGIES = ("sequential", "parallel", "parallel-dual-ln", "dense")


def sample_exit_depths(B: int, T: int, n_layers: int, generator: torch.Generator,
                       device: str = "cpu") -> torch.Tensor:
    """训练时均匀采样退出深度 d_i ∈ {0..L}（契约：均匀分布起步）。"""
    return torch.randint(0, n_layers + 1, (B, T), generator=generator,
                         device=device)


class ParallelBlock(nn.Module):
    """并行子层拓扑（契约 E 的 `parallel` 臂）：两个子层读**同一个块输入**，
    `h' = h + attn(ln_a h) + mlp(ln_b h)`（区别于 Block 的链式
    `h + mlp(ln2(h + attn(ln1 h)))`）。

    - `share_ln=True`（契约公式原式 / PaLM-GPT-J）：`ln_a = ln_b = ln1`，单个
      共享 pre-LN ⇒ 参数比 Block 少 2d（无 ln2）；
    - `share_ln=False`（`parallel-dual-ln`）：保留 ln1/ln2 两个 pre-LN、都作用于
      块输入 h ⇒ 参数集合与 Block **逐位相同**（SCHEMA §15 的等参数位点）。

    子模块创建顺序与 Block 一致（ln1 → [ln2] → qkv → proj → mlp），因此两种变体
    消耗的初始化随机数与 Block 相同——同 seed 下 embed/pos/gates 及 trunk 权重
    与 sequential 臂逐位一致（单测钉住），臂间唯一差异 = 接线。
    只支持 stock 路径（无 mask/dist_pen/kv_from/gate_vec）：层间拓扑只在
    dense=True 形态下开放，退出/填充语义下的并行接线未定义。
    """

    def __init__(self, cfg: ModelConfig, share_ln: bool = True):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.share_ln = share_ln
        self.ln1 = nn.LayerNorm(d)
        if not share_ln:
            self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.mlp_mult * d), nn.GELU(),
            nn.Linear(cfg.mlp_mult * d, d), nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                dist_pen: float = 0.0, kv_from: torch.Tensor | None = None,
                gate_vec: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, None]:
        for name, given in (("attn_mask", attn_mask is not None),
                            ("dist_pen", dist_pen > 0),
                            ("kv_from", kv_from is not None),
                            ("gate_vec", gate_vec is not None)):
            if given:
                raise ValueError(
                    f"ParallelBlock 只支持 stock 注意力路径，收到 {name}："
                    "并行子层拓扑只在 dense=True（无退出/无填充）形态下开放")
        B, T, d = x.shape
        h1 = self.ln1(x)                       # attn 分支的 pre-LN
        q, k, v = self.qkv(h1).chunk(3, dim=-1)
        shape = lambda t: t.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        a = F.scaled_dot_product_attention(shape(q), shape(k), shape(v),
                                           is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, d)
        h2 = h1 if self.share_ln else self.ln2(x)   # mlp 分支：同一块输入 x
        return x + self.proj(a) + self.mlp(h2), None


def causal_topk_keep_mask(scores: torch.Tensor, k: int,
                          causal: torch.Tensor | None = None) -> torch.Tensor:
    """内容 top-k 的因果保留掩码 [.., T, T]（bool，True = 进入 softmax）。

    - `causal` 缺省 = 现算下三角（`causal[i, j] = j ≤ i`）；
    - 每行在**因果允许的键**里保留分数最大的 k 个（`torch.topk` 语义，同分按
      索引定序）；行 i 的允许键数为 i+1，故 i+1 < k 时保留全部 i+1 个
      （`& causal` 合取把 topk 在 −inf 上的任意选择剔除，保证掩码恒为「因果 ∩
      top-k」，不会放出非因果键）。
    - k ≥ T 时返回恒等因果掩码（选择无信息量，调用方应直接走 stock SDPA 路径）。
    """
    T = scores.shape[-1]
    if causal is None:
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                       device=scores.device))
    if k >= T:
        return causal
    allowed = scores.masked_fill(~causal, float("-inf"))
    idx = allowed.topk(k, dim=-1).indices
    keep = torch.zeros_like(allowed, dtype=torch.bool).scatter_(-1, idx, True)
    return keep & causal


def sparse_attn_telemetry(keep: torch.Tensor, k: int) -> dict:
    """top-k 注意力的活性量（SCHEMA §10）：保留键数与「被选键是不是就是最近
    的 k 个键（位置选择）」的重合度。返回 0 维张量（调用方决定何时 `.item()`）。

    - `kept_mean/min/max`：每个查询实际保留的因果键数；
    - `sel_dist_mean`：被选键的平均距离 i − j；
    - `sel_in_lastk_frac`：被选键中落在「最近 k 个位置」窗口内的比例——它是
      **位置选择基线**的重合度；≪ 1 说明选择确实按内容、不像位置窗口。
    """
    T = keep.shape[-1]
    idx = torch.arange(T, device=keep.device)
    dist = (idx.unsqueeze(1) - idx.unsqueeze(0))          # [T,T]，因果区 ≥ 0
    kept = keep.sum(-1).float()                            # [.., T]
    total = kept.sum().clamp_min(1.0)
    return {
        "kept_mean": kept.mean(),
        "kept_min": kept.min(),
        "kept_max": kept.max(),
        "kept_frac_of_T": kept.mean() / T,
        "sel_dist_mean": (keep * dist).sum().float() / total,
        "sel_in_lastk_frac": (keep & (dist < k)).sum().float() / total,
    }


class TopKBlock(Block):
    """内容 top-k 稀疏注意力块（ticket 31 契约 J）。



    与 `Block` 的唯一差别 = 注意力算子：先算**完整**分数矩阵
    `scores = q·kᵀ/√dh`，在因果掩码内保留每个查询分数最高的 k 个键、其余置
    −inf 后 softmax（数学语义 = 只用被选键做混合的稀疏注意力；选择按内容）。

    - 子模块创建顺序与 `Block` 完全一致（ln1 → ln2 → qkv → proj → mlp），因此
      同 seed 下参数与全注意力臂**逐位相同**、参数量增量为 0（单因素：唯一变量
      是算子）；因果掩码是 `persistent=False` 的 buffer，不进 state_dict；
    - `topk >= T` 时因果允许键数恒 ≤ T ⇒ 选择退化为恒等 ⇒ 直接走 stock SDPA
      路径，与 `Block` 逐位一致（单测钉住）；
    - 只支持 stock 路径（无 mask/dist_pen/kv_from/gate_vec）——稀疏注意力只在
      dense=True（无退出/无填充）形态下开放。
    """

    #: 该块的 aux 遥测归类（`DepthARModel.forward` 依此把活性量放进
    #: `aux["attn_sparse"]`）；线性注意力块同名字段用 "attn_linear"。
    telemetry_kind = "attn_sparse"

    def __init__(self, cfg: ModelConfig, topk: int):
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        super().__init__(cfg)
        self.topk = int(topk)
        self.register_buffer(
            "causal", torch.tril(torch.ones(cfg.seq_len, cfg.seq_len,
                                            dtype=torch.bool)),
            persistent=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                dist_pen: float = 0.0, kv_from: torch.Tensor | None = None,
                gate_vec: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict | None]:
        for name, given in (("attn_mask", attn_mask is not None),
                            ("dist_pen", dist_pen > 0),
                            ("kv_from", kv_from is not None),
                            ("gate_vec", gate_vec is not None)):
            if given:
                raise ValueError(
                    f"TopKBlock 只支持 stock 注意力路径，收到 {name}：内容 "
                    "top-k 稀疏只在 dense=True（无退出/无填充）形态下开放")
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        shape = lambda t: t.view(B, T, self.n_heads, d // self.n_heads).transpose(1, 2)
        qs, ks, vs = shape(q), shape(k), shape(v)
        if self.topk >= T:
            a = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
            tele = None
        else:
            scores = qs @ ks.transpose(-2, -1) / (d // self.n_heads) ** 0.5
            keep = causal_topk_keep_mask(scores, self.topk, self.causal[:T, :T])
            w = scores.masked_fill(~keep, float("-inf")).softmax(-1)
            a = w @ vs
            tele = sparse_attn_telemetry(keep, self.topk)
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x, tele


# ---------- KV 头共享（ticket 33 契约 S，W3 参照线） ----------


class GQABlock(nn.Module):
    """KV 头共享注意力块（ticket 33 契约 S，W3 参照线）：GQA / MQA。

    与 stock `Block` 的唯一差别 = K/V 的**头数**（query 头数不变）：n_kv_heads 个
    K/V 头，每组 `n_heads / n_kv_heads` 个 query 头共享同一组 K/V（n_kv_heads=1
    即 MQA）。n_kv_heads == n_heads（MHA）由 `Block` 承担，本类不参与 ⇒ 缺省路径
    逐位还原 v1（单测钉住）。

    - 投影输出宽度 = `d + 2·n_kv_heads·dh`（不是 3d）⇒ **KV cache 的每位置元素数
      真的从 2·d 降到 2·n_kv_heads·dh**，这是机制本身，不是标签；
    - 注意力用 `scaled_dot_product_attention(..., enable_gqa=True)`：K/V 以
      `[B, n_kv_heads, T, dh]` 进内核、由内核按组广播（**不物化**复制到 n_heads；
      实测与手工 `repeat_interleave` 展开逐位一致，单测钉住），该形状就是部署侧
      KV cache 的形状（`kv_telemetry` 逐次前向记录，作活性证据）；
    - 子模块创建顺序与 `Block` 一致（ln1 → qkv → proj → mlp）；
    - 只支持 stock 路径（无 dist_pen/kv_from/gate_vec）——KV 头共享只在 dense=True
      （无退出/无填充）形态下开放；评测期可见性掩码 `attn_mask` 支持。
    """

    def __init__(self, cfg: ModelConfig, n_kv_heads: int):
        super().__init__()
        d, nh = cfg.d_model, cfg.n_heads
        if not 0 < n_kv_heads < nh or nh % n_kv_heads:
            raise ValueError(
                f"n_kv_heads 必须满足 0 < n_kv_heads < n_heads 且整除 n_heads，"
                f"收到 n_kv_heads={n_kv_heads} / n_heads={nh}（n_kv_heads == "
                "n_heads 是 MHA，走 stock Block，不用本块）")
        self.n_heads = nh
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = d // nh
        self.kv_dim = self.n_kv_heads * self.head_dim
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        # K/V 半边只有 kv_dim 宽（唯一的结构差异；q 半边恒 d）
        self.qkv = nn.Linear(d, d + 2 * self.kv_dim)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.mlp_mult * d), nn.GELU(),
            nn.Linear(cfg.mlp_mult * d, d), nn.Dropout(cfg.dropout),
        )
        #: 每次前向记录实际进 SDPA 的 K/V 张量形状（活性证据，不进 state_dict）
        self.kv_telemetry: dict | None = None

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                dist_pen: float = 0.0, kv_from: torch.Tensor | None = None,
                gate_vec: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, None]:
        for name, given in (("dist_pen", dist_pen > 0),
                            ("kv_from", kv_from is not None),
                            ("gate_vec", gate_vec is not None)):
            if given:
                raise ValueError(
                    f"GQABlock 只支持 stock 注意力路径，收到 {name}：KV 头共享只在"
                    "dense=True（无退出/无填充）形态下开放")
        if attn_mask is not None and attn_mask.dim() not in (2, 4):
            raise ValueError("attn_mask 必须是 [T,T] 或 [B,H,T,T]（评测期可见性掩码）")
        B, T, d = x.shape
        qkv = self.qkv(self.ln1(x))
        q, kv = qkv[..., :d], qkv[..., d:]
        k, v = kv[..., :self.kv_dim], kv[..., self.kv_dim:]

        def shape(t: torch.Tensor, nh: int) -> torch.Tensor:
            return t.view(B, T, nh, self.head_dim).transpose(1, 2)

        qs, ks, vs = shape(q, self.n_heads), shape(k, self.n_kv_heads), shape(v, self.n_kv_heads)
        a = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=attn_mask,
                                           is_causal=attn_mask is None,
                                           enable_gqa=True)
        self.kv_telemetry = {
            "q_shape": tuple(qs.shape), "k_shape": tuple(ks.shape),
            "v_shape": tuple(vs.shape), "qkv_out_width": int(qkv.shape[-1]),
            "n_kv_heads": self.n_kv_heads, "head_dim": self.head_dim,
            "enable_gqa": True, "expanded_to_n_heads": False,
            "attn_mask": attn_mask is not None,
        }
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x, None


@torch.no_grad()
def kv_head_liveness(model: "DepthARModel", tokens: torch.Tensor) -> dict:
    """KV 头共享的活性块（SCHEMA §10 强制项，契约 S）：报**实际张量形状**。

    形状类干预（改 KV 头数）不能只改标签，因此本函数在**一次真实前向**里抓：
    ① 每层 `qkv` 投影的**实际输出宽度**（MHA = 3d；GQA/MQA = d + 2·n_kv_heads·dh，
    由 forward hook 抓，与配置无关）；② GQA/MQA 层真正送进 SDPA 的 K/V 张量形状
    `[B, n_kv_heads, T, dh]`（块自己在 forward 里记录）。两者与 `cfg` 期望不符时
    直接抛 `AssertionError`——活性证据是张量本身，不是标签。

    解析 KV 内存：每位置元素数 = `n_layers · 2 · n_kv_heads · head_dim`，字节数按
    fp16（部署默认 KV 精度）与 fp32 两个口径给出；`kv_reduction_frac` 是相对 MHA
    （2·d/层/位置）的下降比。**这是解析内存**，不是实测显存（见实验 analysis.md）。
    """
    cfg = model.cfg
    d, nh = cfg.d_model, cfg.n_heads
    dh = d // nh
    nkv = cfg.n_kv_heads if cfg.n_kv_heads is not None else nh
    widths: dict[int, tuple] = {}
    handles = [blk.qkv.register_forward_hook(
        (lambda k: lambda mod, inp, out: widths.__setitem__(k, tuple(out.shape)))(k))
        for k, blk in enumerate(model.blocks)]
    was_training = model.training
    model.eval()
    try:
        model(tokens)
    finally:
        for h in handles:
            h.remove()
        model.train(was_training)
    assert len(widths) == cfg.n_layers, "有层的前向没被 hook 到（活性证据不完整）"
    layers = []
    for k, blk in enumerate(model.blocks):
        w_shape = tuple(blk.qkv.weight.shape)
        qkv_w = widths[k][-1]
        assert w_shape[0] == qkv_w, (
            f"层 {k}: qkv 权重输出宽度 {w_shape[0]} != 前向实际输出宽度 {qkv_w}")
        if cfg.n_kv_heads is not None and cfg.n_kv_heads < nh:
            tele = blk.kv_telemetry
            assert tele is not None, f"层 {k}: GQA 块没有落 K/V 张量形状（活性证据缺失）"
            k_shape, v_shape = tele["k_shape"], tele["v_shape"]
            q_shape = tele["q_shape"]
        else:
            # stock Block：K/V 半边 = u 的 1/3 宽，reshape 后头数 = 半边宽 / dh
            kv_width = qkv_w // 3
            assert qkv_w == 3 * d and kv_width % dh == 0, \
                f"层 {k}: 非 MHA 布局（qkv 输出宽 {qkv_w}）却走 MHA 分支"
            k_shape = (tokens.shape[0], kv_width // dh, tokens.shape[1], dh)
            v_shape = k_shape
            q_shape = k_shape
        assert k_shape[1] == nkv and v_shape[1] == nkv, (
            f"层 {k}: 实际 K/V 头数 {k_shape[1]}/{v_shape[1]} != 配置的 {nkv}"
            "（只改标签没改张量）")
        assert k_shape == v_shape and k_shape[0] == tokens.shape[0] \
            and k_shape[2] == tokens.shape[1] and k_shape[3] == dh, \
            f"层 {k}: K/V 张量形状异常 {k_shape} / {v_shape}"
        kv_elems = 2 * nkv * dh
        w = blk.qkv.weight.detach().float()
        kv_dim = nkv * dh
        k_rows = w[:d + kv_dim][d:] if cfg.n_kv_heads is not None and cfg.n_kv_heads < nh \
            else w[d:2 * d]
        # 每个 KV 头自己的 K 投影行：两个头若逐位相同（或退化成广播），就是「只改标签」
        v_rows = w[d + kv_dim:d + 2 * kv_dim] if cfg.n_kv_heads is not None \
            and cfg.n_kv_heads < nh else w[2 * d:3 * d]
        k_heads_w = [k_rows[i * dh:(i + 1) * dh] for i in range(nkv)]
        head_norms = [round(float(hh.norm()), 6) for hh in k_heads_w]
        pair_diff = max((float((k_heads_w[i] - k_heads_w[j]).abs().max())
                         for i in range(nkv) for j in range(i + 1, nkv)),
                        default=None)
        layers.append({
            "layer": k, "class": type(blk).__name__,
            "qkv_weight_shape": w_shape, "qkv_out_width": qkv_w,
            "k_shape": list(k_shape), "v_shape": list(v_shape),
            "q_shape": list(q_shape),
            "n_kv_heads": nkv, "head_dim": dh, "q_heads": nh,
            "kv_elems_per_position": kv_elems,
            "kv_bytes_per_position_fp16": kv_elems * 2,
            "kv_bytes_per_position_fp32": kv_elems * 4,
            "k_proj_per_kv_head_row_norm": head_norms,
            "kv_head_pair_max_abs_weight_diff": (round(pair_diff, 6)
                                                 if pair_diff is not None else None),
            "k_rows_nonzero_frac": round(float((k_rows != 0).float().mean()), 6),
            "kv_rows_fro": round(float(torch.cat([k_rows, v_rows]).norm()), 6),
        })
    per_layer_elems = 2 * nkv * dh
    total_elems = cfg.n_layers * per_layer_elems
    mha_total_elems = cfg.n_layers * 2 * d
    kv_part = [p for name, p in model.named_parameters()
               if name.endswith(".qkv.weight")]
    kv_nonzero = sum(int((p.detach() != 0).sum()) for p in kv_part)
    kv_numel = sum(p.numel() for p in kv_part)
    return {
        "enabled": bool(cfg.n_kv_heads is not None and cfg.n_kv_heads < nh),
        "mha": bool(cfg.n_kv_heads is None or cfg.n_kv_heads == nh),
        "n_heads": nh, "n_kv_heads": nkv, "head_dim": dh,
        "group_size": nh // nkv,
        "block_python_class": type(model.blocks[0]).__name__,
        "qkv_out_width": widths[0][-1],
        "qkv_out_width_mha_equivalent": 3 * d,
        "k_batch_shape": layers[0]["k_shape"],
        "params_per_layer_kv_elems": per_layer_elems,
        "params_added": -2 * (d - nkv * dh) * (d + 1) * cfg.n_layers,
        "params_added_note": ("KV 头共享**减少**参数（K/V 投影半边变窄），故为负；"
                              "= −2·(d − n_kv_heads·dh)·(d+1)·n_layers（含 qkv 偏置的"
                              "半边差）⇒ 容量差异，见 analysis.md 混杂声明（SCHEMA §15 "
                              "的剪枝类情形）"
                              if nkv < nh else
                              "MHA（n_kv_heads == n_heads）：无共享，参数与 v1 逐位相同"),
        "kv_qkv_weight_nonzero_frac": round(kv_nonzero / max(kv_numel, 1), 6),
        "kv_qkv_weight_fro": round(float(sum(float(p.detach().float().norm())
                                             for p in kv_part)), 6),
        "kv_elems_per_position_per_layer": per_layer_elems,
        "kv_elems_per_position": total_elems,
        "kv_bytes_per_position_fp16": total_elems * 2,
        "kv_bytes_per_position_fp32": total_elems * 4,
        "kv_reduction_frac": round(1 - nkv / nh, 6),
        "mha_kv_bytes_per_position_fp16": mha_total_elems * 2,
        "k_proj_per_kv_head_row_norm": layers[0]["k_proj_per_kv_head_row_norm"],
        "kv_head_pair_max_abs_weight_diff": (
            min(r["kv_head_pair_max_abs_weight_diff"] for r in layers)
            if nkv > 1 else None),
        "kv_rows_fro_mean": round(
            sum(r["kv_rows_fro"] for r in layers) / cfg.n_layers, 6),
        "per_layer": layers,
        "alive": bool(kv_nonzero > 0 and layers[0]["k_shape"][1] == nkv
                       and (nkv == 1 or all(
                           r["kv_head_pair_max_abs_weight_diff"]
                           for r in layers))),
        "alive_note": ("活性证据 = 前向实际 K/V 张量形状 [B, n_kv_heads, T, dh]"
                       "（不是标签）+ qkv 投影权重非零 + 各 KV 头的 K 投影行不全同"
                       if cfg.n_kv_heads is not None and cfg.n_kv_heads < nh else
                       "MHA（n_kv_heads == n_heads）：无共享，KV 形状 = [B, n_heads, T, dh]"),
    }


# ---------- 线性注意力（ticket 33 契约 R，A6 混合算子续） ----------


def linear_attn_phi(x: torch.Tensor) -> torch.Tensor:
    """线性注意力的特征映射 φ(x) = **elu(x) + 1**（Katharopoulos et al. 2020 的
    「elu+1」，单测钉住公式本身）。两条性质都是机制的一部分：

    - **非负**：核分数 `φ(q)·φ(k) ≥ 0` ⇒ 注意力权重可解释为归一化的非负平均，
      不会出现负权重相互抵消（φ = elu 会给出负值）；
    - **恒有兜底**：数学上 φ > 0；fp32 下 `x ≲ −16` 时 `elu(x)` 舍入为 −1 ⇒
      φ 恰好下溢为 0（实测 φ(−20) = 0.0），因此分母必须 clamp（见
      `LINEAR_ATTN_EPS`）——对角项恒在，clamp 后分母恒 > 0，不会出 NaN。
    """
    return F.elu(x) + 1.0


def linear_attn_reference(qf: torch.Tensor, kf: torch.Tensor, v: torch.Tensor,
                          keep: torch.Tensor | None = None) -> torch.Tensor:
    """线性注意力的**参考实现**（显式 `[B,H,T,T]` 核分数，O(T²) 时间/显存）。

    把线性注意力写成「softmax 换成归一化线性核」的等价形式：
    `out_i = Σ_{j∈allow} (φ(q_i)·φ(k_j)) v_j / Σ_{j∈allow} (φ(q_i)·φ(k_j))`。
    因为打分是显式的，它**可以接受任意的可见性掩码**（`[T,T]` 或 `[B,H,T,T]`
    bool，语义 = 被删位置的 φ(k) 从分子分母中一起去掉，与全注意力掩码同一件仪器
    的可见性语义）——这是「评测期依赖曲线」能在同一个算子上跑的前提（契约 R 判据 ③）。

    只用于测试与评测期仪器；训练的 O(T) 实装见 `linear_attn_chunked`，两者的
    数值一致性由单测钉住（同一 φ、同一归一化，差异仅浮点累加顺序）。
    """
    T = qf.shape[-2]
    scores = qf @ kf.transpose(-2, -1)                       # [B,H,T,T]，非负
    if keep is None:
        keep = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                     device=scores.device))
    scores = scores.masked_fill(~keep.to(torch.bool), 0.0)
    den = scores.sum(-1, keepdim=True)
    return (scores @ v) / den.clamp_min(LINEAR_ATTN_EPS)


def linear_attn_chunked(qf: torch.Tensor, kf: torch.Tensor, v: torch.Tensor,
                        chunk: int = DEFAULT_LINEAR_CHUNK
                        ) -> tuple[torch.Tensor, dict]:
    """线性注意力的 **O(T) 训练实装**（契约 R 的「必须证明真的是 O(T)」）。

    分块形式（chunk = C）：块**间**用累积状态、块**内**用显式因果打分。

        out_i = [ φ(q_i)ᵀ·S_{n(i)} + Σ_{j∈块 n(i), j≤i} (φ(k_j)·φ(q_i))·v_j ]
              / [ φ(q_i)·z_{n(i)} + Σ_{j∈块 n(i), j≤i} φ(k_j)·φ(q_i) ]

    其中 `S_n = Σ_{m<n} Σ_{j∈块 m} φ(k_j) v_jᵀ`、`z_n = Σ_{m<n} Σ_{j∈块 m} φ(k_j)`
    只依赖块号，用一次 exclusive `cumsum` 得到。因此：

    - **时间**：每 token 每 head 的 MAC 数 = `dh²`（状态×查询）+ `C·dh`（块内打分
      与混合），**两者都与 T 无关** ⇒ 每层 O(T)（对照：全注意力每 token 每 head
      = `2·T·dh`，即 O(T²)）；
    - **显存**：状态 `[B,H,T/C,dh,dh]` + 块内分数 `[B,H,T/C,C,C]` ⇒ O(T)
      （对照：全注意力的分数矩阵 O(T²)）；**全程不物化任何 `[T,T]` 张量**。

    返回 `(out, telemetry)`；`telemetry` 是训练日志每 200 步落盘的活性量
    （SCHEMA §10）：`phi_min/phi_max`（φ 的实测量程，机制前提）、`den_min`
    （归一化分母最小值）、`inter_mass_frac`（输出范数中来自**块间状态** =
    更远过去的份额）、`state_norm`（状态矩阵范数）。

    **活性量以 0 维张量返回**（不在前向里 `.item()`/`float()`）：调用方按张量
    累加、每 200 步落盘时再转 Python 浮点；每步 8 层 × 6 次设备同步在共享卡上
    会拖垮训练（本实验第一次跑 linear 臂时实测卡在 step ~1000）。
    """
    B, H, T, dh = qf.shape
    C = min(int(chunk), T)
    if C < 1 or T % C:
        raise ValueError(f"chunk={chunk} 必须整除 seq_len={T}（本实装的分块要求）")
    n = T // C
    qc = qf.reshape(B, H, n, C, dh)
    kc = kf.reshape(B, H, n, C, dh)
    vc = v.reshape(B, H, n, C, dh)
    S_chunk = torch.einsum("bhncd,bhnce->bhnde", kc, vc)     # [B,H,n,dh,dh]
    z_chunk = kc.sum(3)                                      # [B,H,n,dh]
    # exclusive prefix（不含本块）：用「cumsum 后右移一位、首位补 0」而不是
    # `cumsum - S_chunk`——后者的相减会把**本块自己的状态**以舍入误差的形式带回
    # 前一块的输出（实测 1 ulp 的未来泄漏），破坏严格因果；右移形式无此问题，
    # 单测 `test_linear_attn_causal_only_no_future_leak` 钉住逐位因果。
    zero_s = torch.zeros_like(S_chunk[:, :, :1])
    zero_z = torch.zeros_like(z_chunk[:, :, :1])
    S_prev = torch.cat([zero_s, S_chunk.cumsum(2)[:, :, :-1]], dim=2)
    z_prev = torch.cat([zero_z, z_chunk.cumsum(2)[:, :, :-1]], dim=2)
    num_inter = torch.einsum("bhncd,bhnde->bhnce", qc, S_prev)   # 块间（远过去）
    den_inter = torch.einsum("bhncd,bhnd->bhnc", qc, z_prev)
    intra = torch.einsum("bhnid,bhnjd->bhnij", qc, kc).tril()   # 块内因果核分数
    num_intra = torch.einsum("bhnij,bhnjd->bhnid", intra, vc)
    den = den_inter + intra.sum(-1)
    den_clamped = den.clamp_min(LINEAR_ATTN_EPS)
    out = (num_inter + num_intra) / den_clamped.unsqueeze(-1)
    ni = num_inter.float().norm(dim=-1)
    nj = num_intra.float().norm(dim=-1)
    tele = {
        # 活性量以 **0 维张量** 形式返回（不在前向里做 `.item()`/`float()`）：
        # 每步做 8 层 × 6 次设备同步在共享卡上会变成同步风暴（本实验第一次
        # 跑 linear 臂时实测卡在 step ~1000、CPU 满载 / GPU 无进展），训练循环
        # 改为按张量累加、只在每 200 步落盘时转成 Python 浮点（见 train_depth_ar.py）。
        "phi_min": qf.detach().min().minimum(kf.detach().min()),
        "phi_max": qf.detach().max().maximum(kf.detach().max()),
        "den_min": den_clamped.detach().min(),
        "inter_mass_frac": (ni / (ni + nj + LINEAR_ATTN_EPS)).detach().mean(),
        "state_norm": S_chunk.detach().float().norm(dim=(-2, -1)).mean(),
    }
    return out.reshape(B, H, T, dh), tele


class LinearAttnBlock(Block):
    """线性注意力块（ticket 33 契约 R，A6 混合算子续）。

    与 `Block` 的唯一差别 = 注意力算子：把全因果 softmax 注意力换成**归一化线性
    核**注意力 `out_i = Σ_{j≤i} (φ(q_i)·φ(k_j)) v_j / Σ_{j≤i} φ(q_i)·φ(k_j)`，
    φ = elu+1。用累积状态实现 ⇒ 每层时间/显存对 T **线性**：不需要 `[T,T]` 分数
    矩阵，也不需要保留全量 KV（流式推理只需每层每头一个 `dh×dh` 状态 + `dh` 归一化
    向量）。训练实装 = 分块并行形式（`linear_attn_chunked`）。

    - 子模块创建顺序与 `Block` 完全一致（ln1 → ln2 → qkv → proj → mlp）⇒ 同 seed
      下参数与全注意力臂**逐位相同**、参数量增量 0（单测钉住：单因素，且 SCHEMA §15
      的等参数要求自动满足）；
    - `attn_mask` 只在**评测期仪器路径**出现（`tools/evalkit/retention.py` 的
      `masked_bpc` / `needle_recall`）：语义 = 把被删位置的 φ(k) 从分子分母中一起
      去掉（与全注意力掩码同一件仪器的可见性语义），走 `linear_attn_reference`；
      `attn_mask=None`（训练路径）= 因果线性注意力，走 O(T) 分块实装；
    - 只支持 stock 路径（无 dist_pen/kv_from/gate_vec）——线性注意力只在 dense=True
      （无退出/无填充）形态下开放。
    """

    telemetry_kind = "attn_linear"

    def __init__(self, cfg: ModelConfig, chunk: int = DEFAULT_LINEAR_CHUNK):
        super().__init__(cfg)
        self.chunk = int(chunk)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None,
                dist_pen: float = 0.0, kv_from: torch.Tensor | None = None,
                gate_vec: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, dict | None]:
        for name, given in (("dist_pen", dist_pen > 0),
                            ("kv_from", kv_from is not None),
                            ("gate_vec", gate_vec is not None)):
            if given:
                raise ValueError(
                    f"LinearAttnBlock 只支持 stock 注意力路径，收到 {name}："
                    "线性注意力只在 dense=True（无退出/无填充）形态下开放")
        if attn_mask is not None and attn_mask.dim() not in (2, 4):
            raise ValueError("attn_mask 必须是 [T,T] 或 [B,H,T,T]（评测期仪器掩码）")
        B, T, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        shape = lambda t: t.view(B, T, self.n_heads,
                                 d // self.n_heads).transpose(1, 2)
        qf, kf = linear_attn_phi(shape(q)), linear_attn_phi(shape(k))
        vs = shape(v)
        if attn_mask is None:
            a, tele = linear_attn_chunked(qf, kf, vs, self.chunk)
        else:
            a, tele = linear_attn_reference(qf, kf, vs, keep=attn_mask), None
        a = a.transpose(1, 2).reshape(B, T, d)
        x = x + self.proj(a)
        x = x + self.mlp(self.ln2(x))
        return x, tele


def attn_linear_liveness(model: "DepthARModel",
                         layer_stats: list[dict] | None = None,
                         prob: dict | None = None) -> dict:
    """线性注意力的活性块（SCHEMA §10 强制项，契约 R）。

    `params_added` 恒 0：该干预**不新增任何参数**（`LinearAttnBlock` 与 `Block`
    逐位同构，单测与实测参数量逐位对照钉住），所以「干预活着」必须由**算子自身
    的运行统计**证明：

    - `phi_min/phi_max/den_min/inter_mass_frac/state_norm`：一次真实 forward 的
      逐层 telemetry（`layer_stats`）聚合。φ 恒 > 0 与 den > 0 是机制前提；
      `inter_mass_frac` = 输出范数中来自块间状态（= 更远过去）的份额，它为 0
      则说明模型根本没读状态（「活的干预」的最直接证据）；
    - `rel_dev_vs_full_*`（`prob` 由 `attn_linear_probe` 提供）：同一权重、同一
      输入下把该层算子换回全因果 softmax 注意力的相对偏差 —— **> 0 才证明这不是
      「换个名字的全注意力」**（契约 R 实现要点）。
    """
    cfg = model.cfg
    out = {"attn_linear": bool(cfg.attn_linear), "enabled": bool(cfg.attn_linear),
           "params_added": 0, "parametrized": False,
           "linear_chunk": int(cfg.linear_chunk),
           "block_python_class": type(model.blocks[0]).__name__}
    if not cfg.attn_linear:
        out.update({"phi_min": None, "phi_max": None, "den_min": None,
                    "inter_mass_frac": None, "state_norm": None,
                    "rel_dev_vs_full_min": None, "alive": False,
                    "alive_note": "开关关闭（attn_linear=False）：无异于基线"})
        return out
    if layer_stats:
        import statistics as _st
        out.update({
            "phi_min": round(float(min(s["phi_min"] for s in layer_stats)), 8),
            "phi_max": round(float(max(s["phi_max"] for s in layer_stats)), 6),
            "den_min": round(float(min(s["den_min"] for s in layer_stats)), 8),
            "inter_mass_frac": round(float(_st.mean(
                [float(s["inter_mass_frac"]) for s in layer_stats])), 6),
            "state_norm": round(float(_st.mean(
                [float(s["state_norm"]) for s in layer_stats])), 6),
            "n_layers_sampled": len(layer_stats)})
    else:
        out.update({"phi_min": None, "den_min": None, "inter_mass_frac": None,
                    "state_norm": None,
                    "alive_note": "未提供 layer_stats（无真实前向的活性样本）"})
    if prob:
        out["rel_dev_vs_full_min"] = prob["rel_dev_vs_full_min"]
        out["rel_dev_vs_full_mean"] = prob["rel_dev_vs_full_mean"]
        out["per_layer"] = prob["per_layer"]
    out["alive"] = bool(out.get("phi_min") is not None
                        and out["phi_min"] > 0 and out["den_min"] > 0
                        and (prob is None or prob["alive"]))
    out["alive_note"] = (
        "φ = elu+1 实测 > 0、归一化分母 > 0（无 NaN 通道），且同权重把算子换回全"
        "注意力时输出有偏差（rel_dev > 0）⇒ 干预活着，不是换个名字的全注意力")
    return out


@torch.no_grad()
def attn_linear_probe(model: "DepthARModel", eval_arr, n_batches: int = 2,
                      batch_size: int = 8, seed: int = 13,
                      device: str = "cpu") -> dict:
    """线性注意力在**真实前向**上的活性汇总（训练入口与实验电池共用同一口径）。

    两个反事实，都在**同一权重、同一输入**下做：

    1. `rel_dev_vs_full`：把某层的算子换回全因果 softmax 注意力（`Block.forward`
       用同一个块的 ln1/qkv/proj/mlp 权重 = 同权重换算子），相对偏差 > 0 ⇒ 算子
       真的改了计算；
    2. `kernel_mass`：**同一组权重**下两种算子把注意力质量放在多远处
       （`attention_mass_stats`），用于回答契约 R 判据 ③「线性注意力是否天然更依赖
       近处」的**机制侧**证据（与评测期抑制曲线独立）。

    开关关闭时返回 enabled=False 的空块，不产生额外前向。
    """
    if not model.cfg.attn_linear:
        return attn_linear_liveness(model)
    from .data import batch
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    stats, devs, per_layer = [], [], []
    mass = {"linear": [], "full": []}
    L = len(model.blocks)
    devs_by_layer: list[list[float]] = [[] for _ in range(L)]
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, generator)
        x = x.to(device)
        _, aux = model(x, exit_depths=None)
        stats.extend(aux["attn_linear"])
        m_lin = attention_mass_stats(model, x, kind="linear")
        m_full = attention_mass_stats(model, x, kind="full")
        mass["linear"].append(m_lin)
        mass["full"].append(m_full)
        for i, blk in enumerate(model.blocks):
            h_in = aux["states"][i]
            out_lin, tele = blk(h_in)                 # 训练路径（O(T) 分块）
            out_full, _ = Block.forward(blk, h_in)    # 同权重换算子（全注意力）
            assert tele is not None
            dev = float((out_lin - out_full).norm() / out_full.norm())
            devs.append(dev)
            devs_by_layer[i].append(dev)
    per_layer = [{"layer": i + 1,
                  "rel_dev_vs_full": round(sum(d) / len(d), 6),
                  "n_batches": len(d)}
                 for i, d in enumerate(devs_by_layer)]
    live = attn_linear_liveness(model, stats, prob={
        "per_layer": per_layer,
        "rel_dev_vs_full_min": round(min(devs), 6),
        "rel_dev_vs_full_mean": round(sum(devs) / len(devs), 6),
        "alive": bool(min(devs) > 0)})
    live["kernel_mass"] = {
        kind: {key: round(sum(r[key] for r in rows) / len(rows), 4)
               for key in ("mass_mean_dist",) + tuple(
                   f"mass_frac_within_{W}" for W in (32, 64, 128))}
        for kind, rows in mass.items()}
    live["kernel_mass_note"] = (
        "同一组权重、同一输入下两种算子的注意力质量分布（O(T²) 显式打分，评测期"
        "仪器）：mass_mean_dist = 质量加权平均查询-键距离；mass_frac_within_* = "
        "质量落在最近 32/64/128 个位置内的比例。用它看线性核把质量放在多远处")
    return live


@torch.no_grad()
def attention_mass_stats(model: "DepthARModel", tokens: torch.Tensor,
                         kind: str = "linear",
                         windows: tuple[int, ...] = (32, 64, 128)) -> dict:
    """**评测期仪器**：把一次前向的注意力质量分布按距离汇总（逐层 + 全层平均）。

    - `kind="linear"`：线性核权重 `φ(q)·φ(k)`（因果、逐查询归一化）——本臂训练时
      真正用的分布；
    - `kind="full"`：`softmax(q·kᵀ/√dh)`（全因果注意力）——基线臂的分布。

    两者都只用**该模型自己的权重**（同一 `LinearAttnBlock` 的 ln1/qkv），因此
    「线性算子是否更依赖近处」可以直接读出来。返回质量加权平均距离与落在最近
    `windows` 个位置内的质量比例。
    """
    if kind not in ("linear", "full"):
        raise ValueError(f"kind must be 'linear' or 'full', got {kind!r}")
    B, T = tokens.shape
    d = model.cfg.d_model
    _, aux = model(tokens, exit_depths=None)
    dist = (torch.arange(T, device=tokens.device).unsqueeze(1)
            - torch.arange(T, device=tokens.device).unsqueeze(0)).clamp_min(0)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                   device=tokens.device))
    per_layer = []
    for i, blk in enumerate(model.blocks):
        h = aux["states"][i]                     # 该层**真实**输入（同一前向）
        dh = d // blk.n_heads
        q, k, _v = blk.qkv(blk.ln1(h)).chunk(3, dim=-1)
        shape = lambda t: t.view(B, T, blk.n_heads, dh).transpose(1, 2)
        qs, ks = shape(q), shape(k)
        if kind == "linear":
            s = linear_attn_phi(qs) @ linear_attn_phi(ks).transpose(-2, -1)
        else:
            s = qs @ ks.transpose(-2, -1) / dh ** 0.5
        s = s.masked_fill(~causal, 0.0 if kind == "linear" else float("-inf"))
        if kind == "linear":
            w = s / s.sum(-1, keepdim=True)
        else:
            w = s.softmax(-1)
        w = w.float()
        row = {"mass_mean_dist": float((w * dist).sum(-1).mean())}
        for W in windows:
            row[f"mass_frac_within_{W}"] = float(
                (w * (dist < W)).sum(-1).mean())
        per_layer.append(row)
    keys = list(per_layer[0])
    out = {kk: round(sum(r[kk] for r in per_layer) / len(per_layer), 4)
           for kk in keys}
    out.update({"kind": kind, "n_layers": len(per_layer),
                "per_layer": [{kk: round(v, 4) for kk, v in r.items()}
                              for r in per_layer]})
    return out


def window_anchor_keep_mask(T: int, window: int, anchors: int = 0,
                            device="cpu") -> torch.Tensor:
    """训练/评测**共用**的「可见窗口」掩码 `[T, T]`（ticket 32 契约 O）。

    `keep[i, j] = (j ≤ i) ∧ ((i − j) < window ∨ j < anchors)`，对角恒开
    （否则某行被全掩 ⇒ softmax 全 −inf ⇒ NaN，同 `_ragged_mask` 的约定）。
    语义：每个查询只看最近 `window` 个位置，外加最前面的 `anchors` 个锚点位置
    （锚通道恒可见）。

    - `anchors = 0` 时与 `tools/evalkit/retention.recent_keep_mask(T, window)`
      **逐位相同**（同一件仪器的语义，单测钉住）；
    - `window ≥ T` 且 `anchors = 0` 时 = 精确因果掩码（等价于不传掩码的 stock
      SDPA 路径；2614 实测「身份掩码 vs 释放版」逐位相等，单测钉住）。
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if anchors < 0:
        raise ValueError(f"anchors must be >= 0, got {anchors}")
    i, j = torch.meshgrid(torch.arange(T, device=device),
                          torch.arange(T, device=device), indexing="ij")
    keep = (j <= i) & (((i - j) < window) | (j < anchors))
    return keep | torch.eye(T, dtype=torch.bool, device=device)


def window_mask_liveness(T: int, window: int, anchors: int = 0) -> dict:
    """训练期可见窗口的活性量（SCHEMA §10）：被放行的因果 (query, key) 对数与占比。

    `params_added` 恒 0（掩码不引入参数）⇒「干预活着」只能由**掩码本身**证明：
    `alive = visible_frac < 1`（有 key 被真的挡掉）。训练日志每 200 步写同一份
    （掩码是静态的，逐行可核）。
    """
    keep = window_anchor_keep_mask(T, window, anchors)
    n_keep = int(keep.sum().item())
    n_causal = T * (T + 1) // 2
    return {"train_window": int(window), "train_anchors": int(anchors),
            "params_added": 0,
            "n_visible_pairs": n_keep, "causal_pairs": n_causal,
            "visible_frac": round(n_keep / n_causal, 6),
            "dropped_frac": round(1.0 - n_keep / n_causal, 6),
            "alive": n_keep < n_causal}


@torch.no_grad()
def window_mask_probe(model: "DepthARModel", tokens: torch.Tensor) -> dict:
    """训练期可见窗口的一次性功能探针（SCHEMA §10）：**同一批输入**在
    「掩码开 / 掩码关」两次前向的最后一层 logits 最大绝对差。

    `> 0` ⇔ 这个开关真的改了前向，不是死掩码。调用方负责传 `train_window > 0`
    的模型；本函数在两支之间借用 `cfg.train_window`（finally 还原）⇒ 权重、
    dropout 状态、随机数流均不变。
    """
    saved = model.cfg.train_window
    if saved <= 0:
        raise ValueError("window_mask_probe 需要 train_window > 0 的模型")
    was_training = model.training
    model.eval()
    try:
        on = model(tokens)[1]["depth_logits"][-1].float()
        model.cfg.train_window = 0
        off = model(tokens)[1]["depth_logits"][-1].float()
    finally:
        model.cfg.train_window = saved
        model.train(was_training)
    diff = (on - off).abs().max()
    return {"max_abs_logit_diff_masked_vs_open": round(float(diff), 6),
            "alive": bool(diff > 0)}


def topology_liveness(model: "DepthARModel") -> dict:
    """层间拓扑干预的活性量（SCHEMA §10 强制项）。

    - `params_added`：相对 sequential 臂的参数增量（sequential 0 / parallel −L·2d /
      parallel-dual-ln 0 / dense +L(L+1)/2），必须与实测参数量增量逐位一致
      （单测 `test_topology_liveness_params_added_matches_param_delta`）；
    - `params_identical_to_sequential` / `start_bitwise_equals_sequential`：参数账与
      起步等价性（能逐位核对的两个布尔量）；
    - dense 臂：层权 α 的 fro / absmax / 非零占比 / 与 identity 起步的最大偏离
      （α 不动 = 干预死 = 该臂逐位等价基线）；
    - parallel 臂：接线本身无专属参数，活性量 = 参数增量 + 逐层分支增量范数
      （`branch_activity`，在 run.py 用评测数据实测）。
    """
    cfg = model.cfg
    d, L = cfg.d_model, cfg.n_layers
    delta = {"sequential": 0, "parallel": -L * 2 * d, "parallel-dual-ln": 0,
             "dense": L * (L + 1) // 2}[cfg.topology]
    out = {"topology": cfg.topology, "params_added": delta,
           "parameterized": cfg.topology == "dense",
           "params_identical_to_sequential":
               cfg.topology in ("sequential", "parallel-dual-ln"),
           "start_bitwise_equals_sequential": cfg.topology != "parallel"
               and cfg.topology != "parallel-dual-ln"}
    if cfg.topology == "dense":
        a = model.layer_mix.detach().float()
        init = torch.zeros_like(a)
        for k in range(1, L + 1):
            init[k * (k - 1) // 2 + k - 1] = 1.0
        out.update({
            "w_fro": round(float(a.norm()), 6),
            "w_absmax": round(float(a.abs().max()), 6),
            "nonzero_frac": round(float(torch.count_nonzero(a) / a.numel()), 6),
            "max_dev_from_identity_init": round(float((a - init).abs().max()), 6),
            "alpha_by_layer": [
                [round(float(v), 6) for v in a[k * (k - 1) // 2: k * (k + 1) // 2]]
                for k in range(1, L + 1)],
        })
        out["alive"] = out["max_dev_from_identity_init"] > 0
    else:
        out.update({"w_fro": None, "w_absmax": None, "nonzero_frac": None,
                    "alive": cfg.topology != "sequential"})
    return out


def attn_sparse_liveness(model: "DepthARModel",
                         layer_stats: list[dict] | None = None) -> dict:
    """内容 top-k 稀疏注意力的活性块（SCHEMA §10 强制项，契约 J）。

    `params_added` 恒 0：该干预不新增任何参数（单测与实测参数量逐位对照钉住），
    所以「活性」必须来自选择本身，而不是参数账。`layer_stats` = 一次真实 forward
    的 `aux["attn_sparse"]`（每层一个 `sparse_attn_telemetry` 字典）时，汇总：

    - `kept_mean` vs `kept_theory_mean`：实测保留键数是否等于理论
      mean_i min(k, i+1)（= k 真的生效，且掩码恰是「因果 ∩ top-k」）；
    - `kept_count_matches_theory`：上述相等性的布尔量；
    - `sel_in_lastk_frac`：被选键与「最近 k 个键（位置选择基线）」的重合度；
    - `alive` = 保留键数与理论一致（掩码真的在按 k 生效）。
    """
    k = model.cfg.attn_sparse_k
    T = model.cfg.seq_len
    theory = sum(min(k, i + 1) for i in range(T)) / T if k > 0 else 0.0
    out = {"attn_sparse_k": k, "enabled": k > 0, "params_added": 0,
           "parametrized": False,
           "block_python_class": type(model.blocks[0]).__name__,
           "kept_theory_mean": round(theory, 6)}
    if k <= 0:
        out.update({"kept_mean": None, "kept_min": None, "kept_max": None,
                    "kept_frac_of_T": None, "sel_dist_mean": None,
                    "sel_in_lastk_frac": None,
                    "kept_count_matches_theory": None, "alive": False,
                    "alive_note": "开关关闭（attn_sparse_k=0）：无异于基线"})
        return out
    if not layer_stats:
        out.update({"kept_mean": None, "kept_count_matches_theory": None,
                    "alive": None,
                    "alive_note": "未提供 layer_stats（无真实前向的活性样本）"})
        return out
    mean = lambda key: float(torch.stack([s[key] for s in layer_stats]).mean())
    out.update({
        "kept_mean": round(mean("kept_mean"), 6),
        "kept_min": round(float(torch.stack([s["kept_min"]
                                             for s in layer_stats]).min()), 6),
        "kept_max": round(float(torch.stack([s["kept_max"]
                                             for s in layer_stats]).max()), 6),
        "kept_frac_of_T": round(mean("kept_frac_of_T"), 6),
        "sel_dist_mean": round(mean("sel_dist_mean"), 4),
        "sel_in_lastk_frac": round(mean("sel_in_lastk_frac"), 4),
        "n_layers_sampled": len(layer_stats),
    })
    out["kept_count_matches_theory"] = abs(out["kept_mean"] - theory) < 1e-6
    out["alive"] = bool(out["kept_count_matches_theory"])
    out["alive_note"] = (
        "保留键数 == 理论 mean_i min(k,i+1) ⇒ 掩码按 k 生效；"
        "sel_in_lastk_frac ≪ 1 = 选择偏离「最近 k 个键」的位置基线（按内容）")
    return out


@torch.no_grad()
def attn_sparse_probe(model: "DepthARModel", eval_arr, n_batches: int = 2,
                      batch_size: int = 8, seed: int = 13,
                      device: str = "cpu") -> dict:
    """真实前向的 top-k 活性汇总（训练入口与实验电池共用同一口径）。开关关闭时
    返回 enabled=False 的空块，不产生额外前向。"""
    if model.cfg.attn_sparse_k <= 0:
        return attn_sparse_liveness(model)
    from .data import batch
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    stats = []
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, generator)
        _, aux = model(x.to(device), exit_depths=None)
        stats.extend(aux["attn_sparse"])
    return attn_sparse_liveness(model, stats)


class DepthARModel(nn.Module):
    """深度自适应 AR。forward(tokens, exit_depths) → (loss, aux)。

    exit_depths=None 表示全深度（无退出）——评测 depth 曲线与 dense 语义。
    aux: states[k]（depth k 表征）、depth_logits[k]、depth_ce[k]、
    gate_mean[k]、exit_depths。
    """

    def __init__(self, cfg: DepthARConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        d = cfg.d_model
        if cfg.readout_dims is not None and not 0 < cfg.readout_dims <= d:
            raise ValueError(
                f"readout_dims must be in (0, {d}], got {cfg.readout_dims}")
        # Tied E=U：embedding 行 norm ~1（init std = d^-0.5），U = E^T（fp32 头）
        self.embed = nn.Embedding(vocab_size, d)
        nn.init.normal_(self.embed.weight, std=d ** -0.5)
        self.pos_embed = nn.Embedding(cfg.seq_len, d)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        mcfg = ModelConfig(d_model=cfg.d_model, n_layers=cfg.n_layers,
                           n_heads=cfg.n_heads, seq_len=cfg.seq_len,
                           mlp_mult=cfg.mlp_mult, dropout=cfg.dropout,
                           norm_cap=cfg.norm_cap)
        if cfg.topology not in TOPOLOGIES:
            raise ValueError(f"unknown topology {cfg.topology!r}; "
                             f"expected one of {TOPOLOGIES}")
        if cfg.topology != "sequential" and not cfg.dense:
            raise ValueError(
                "层间拓扑干预只在 dense=True（stock decoder-only 形态）下验证"
                f"（单因素）——收到 topology={cfg.topology!r} 且 dense=False")
        if cfg.attn_sparse_k < 0:
            raise ValueError(f"attn_sparse_k must be >= 0, got {cfg.attn_sparse_k}")
        if cfg.attn_sparse_k > 0 and not cfg.dense:
            raise ValueError(
                "内容 top-k 稀疏注意力只在 dense=True（stock decoder-only 形态）下"
                f"验证（单因素）——收到 attn_sparse_k={cfg.attn_sparse_k} 且 "
                "dense=False")
        if cfg.attn_sparse_k > 0 and cfg.topology != "sequential":
            raise ValueError(
                "attn_sparse_k 与层间拓扑互斥（单因素：一次只改一个因素）——"
                f"收到 topology={cfg.topology!r}")
        if cfg.train_window < 0 or cfg.train_anchors < 0:
            raise ValueError("train_window / train_anchors must be >= 0, got "
                             f"{cfg.train_window} / {cfg.train_anchors}")
        if cfg.train_anchors > 0 and cfg.train_window == 0:
            raise ValueError("train_anchors 只在 train_window > 0 时有意义：掩码"
                             "关闭时锚点通道本就全可见（单因素）")
        if cfg.train_window > 0 and not cfg.dense:
            raise ValueError(
                "训练期可见窗口只在 dense=True（stock decoder-only 形态）下验证"
                f"（单因素）——收到 train_window={cfg.train_window} 且 dense=False")
        if cfg.train_window > 0 and cfg.topology != "sequential":
            raise ValueError(
                "train_window 与层间拓扑互斥（单因素：一次只改一个因素）——"
                f"收到 topology={cfg.topology!r}")
        if cfg.attn_sparse_k > 0 and cfg.train_window > 0:
            raise ValueError(
                "attn_sparse_k 与 train_window 互斥（单因素：一次只改一个因素）——"
                f"收到 train_window={cfg.train_window} 且 attn_sparse_k="
                f"{cfg.attn_sparse_k}")
        if cfg.attn_linear and cfg.linear_chunk < 1:
            raise ValueError(f"linear_chunk must be >= 1, got {cfg.linear_chunk}")
        if cfg.attn_linear and not cfg.dense:
            raise ValueError(
                "线性注意力只在 dense=True（stock decoder-only 形态）下验证"
                f"（单因素）——收到 attn_linear=True 且 dense=False")
        if cfg.attn_linear and cfg.topology != "sequential":
            raise ValueError(
                "attn_linear 与层间拓扑互斥（单因素：一次只改一个因素）——"
                f"收到 topology={cfg.topology!r}")
        if cfg.attn_linear and cfg.attn_sparse_k > 0:
            raise ValueError(
                "attn_linear 与 attn_sparse_k 互斥（单因素：一次只改一个注意力"
                f"算子）——收到 attn_sparse_k={cfg.attn_sparse_k}")
        if cfg.attn_linear and cfg.train_window > 0:
            raise ValueError(
                "attn_linear 与 train_window 互斥（单因素：一次只改一个因素）——"
                f"收到 train_window={cfg.train_window}")
        if cfg.n_kv_heads is not None:
            if not 0 < cfg.n_kv_heads <= cfg.n_heads or cfg.n_heads % cfg.n_kv_heads:
                raise ValueError(
                    "n_kv_heads 必须满足 0 < n_kv_heads <= n_heads 且整除 n_heads，"
                    f"收到 n_kv_heads={cfg.n_kv_heads} / n_heads={cfg.n_heads}")
            if cfg.n_kv_heads < cfg.n_heads:
                if not cfg.dense:
                    raise ValueError(
                        "KV 头共享（GQA/MQA）只在 dense=True（stock decoder-only "
                        "形态）下验证（单因素）——收到 n_kv_heads="
                        f"{cfg.n_kv_heads} 且 dense=False")
                if cfg.topology != "sequential":
                    raise ValueError(
                        "n_kv_heads 与层间拓扑互斥（单因素：一次只改一个因素）——"
                        f"收到 topology={cfg.topology!r}")
                for other, on in (("attn_sparse_k", cfg.attn_sparse_k > 0),
                                  ("attn_linear", cfg.attn_linear),
                                  ("train_window", cfg.train_window > 0)):
                    if on:
                        raise ValueError(
                            f"n_kv_heads 与 {other} 互斥（单因素：一次只改一个"
                            "因素）")
        if cfg.attn_sparse_k > 0:
            self.blocks = nn.ModuleList(TopKBlock(mcfg, cfg.attn_sparse_k)
                                        for _ in range(cfg.n_layers))
        elif cfg.attn_linear:
            self.blocks = nn.ModuleList(LinearAttnBlock(mcfg, cfg.linear_chunk)
                                        for _ in range(cfg.n_layers))
        elif cfg.n_kv_heads is not None and cfg.n_kv_heads < cfg.n_heads:
            # KV 头共享（契约 S）：n_kv_heads == n_heads 时走下面的 stock Block，
            # 因此「缺省 / MHA」与 v1 逐位一致
            self.blocks = nn.ModuleList(GQABlock(mcfg, cfg.n_kv_heads)
                                        for _ in range(cfg.n_layers))
        elif cfg.topology == "parallel":
            self.blocks = nn.ModuleList(ParallelBlock(mcfg, share_ln=True)
                                        for _ in range(cfg.n_layers))
        elif cfg.topology == "parallel-dual-ln":
            self.blocks = nn.ModuleList(ParallelBlock(mcfg, share_ln=False)
                                        for _ in range(cfg.n_layers))
        else:
            self.blocks = nn.ModuleList(Block(mcfg) for _ in range(cfg.n_layers))
        # dense 拓扑（契约 E）：层间加权和的层权 α，打包 1-D（层 k 的源 j 索引
        # k(k−1)/2 + j）。初始化为「只取当前层」（α_{k,k−1} = 1、其余 0）⇒ 层 k
        # 输入 = h_{k−1}，起步与 sequential 臂逐位等价（单测钉住）。该点**非鞍点**：
        # ∂L/∂α_kj = ⟨∂L/∂h_in, h_j⟩ 一般非零（α 非全零、h_j 非零），不需要 W4
        # 那种单边零技巧。放在 trunk/gates 之后且不消耗随机数。
        if cfg.topology == "dense":
            self.layer_mix = nn.Parameter(
                torch.zeros(cfg.n_layers * (cfg.n_layers + 1) // 2))
            with torch.no_grad():
                for k in range(1, cfg.n_layers + 1):
                    self.layer_mix[k * (k - 1) // 2 + k - 1] = 1.0
        # 逐通道携带门：gates[k] 作用于 depth k → k+1 的精化 delta
        self.gates = nn.ModuleList(nn.Linear(d, d) for _ in range(cfg.n_layers))
        for g in self.gates:
            nn.init.normal_(g.weight, std=0.02)
            nn.init.constant_(g.bias, cfg.gate_bias_init)
        gate_ids = {id(g) for g in self.gates}
        for m in self.modules():
            if isinstance(m, nn.Linear) and id(m) not in gate_ids:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # proj-fill 逐层 KV 源 adapter：恒等初始化（不消耗随机数，同 seed 下
        # 其余参数初始化与 fill 臂逐位一致），起步 == fill（单测钉住）
        if cfg.proj_fill:
            self.kv_adapters = nn.ModuleList(
                nn.Linear(d, d) for _ in range(cfg.n_layers))
            for a in self.kv_adapters:
                nn.init.zeros_(a.bias)
                with torch.no_grad():
                    a.weight.copy_(torch.eye(d))
        # gated 残差读出（W4 主臂）：W1 ~ N(0, 0.02)、W2 = 0（不消耗"其它参数"的
        # 随机数——建在参数 init 循环之后），起步 transform = 0 ⇒ 严格 == tied
        # 线性（恒等起步，单测钉住），且 ∂L/∂W2 ≠ 0（非鞍点，可学）。
        if cfg.readout_gated:
            self.readout_w1 = nn.Linear(d, d, bias=False)
            self.readout_w2 = nn.Linear(d, d, bias=False)
            nn.init.normal_(self.readout_w1.weight, std=0.02)
            nn.init.zeros_(self.readout_w2.weight)
        # 线性等参数对照读出（W4 契约 B，2611）：P 外、Q 内；Q ~ N(0, 0.02)、
        # **P = 0** ⇒ 起步 transform = P(Qh) = 0，严格 == tied 线性；
        # ∂L/∂P ≠ 0（非鞍点）。同样建在参数 init 循环之后——不消耗其余参数的
        # 随机数，同 seed 下与 A 臂其余参数逐位一致。与 readout_gated 互斥。
        if cfg.readout_linear_resid:
            if cfg.readout_gated:
                raise ValueError(
                    "readout_linear_resid 与 readout_gated 互斥：读出几何"
                    "只允许一个干预（单因素）")
            self.readout_p = nn.Linear(d, d, bias=False)
            self.readout_q = nn.Linear(d, d, bias=False)
            nn.init.normal_(self.readout_q.weight, std=0.02)
            nn.init.zeros_(self.readout_p.weight)

    # ---------- 读出 ----------

    def readout(self, h: torch.Tensor) -> torch.Tensor:
        """Tied unembed，fp32 头（autocast 之外强制 fp32）。

        readout_dims（ticket 25 B 臂）：只读前 r 个通道，tied 表取列切片
        E[:, :r]（unembed 与全部出口头同一切片，机制一致）；workspace 维
        不进 logits。缺省（None）与 r=d 走原表达式，逐位还原 v1。

        readout_gated（W4 主臂）：先做非线性状态变换
        h → h + SiLU(W1 h) ⊙ (W2 h)，再走 tied 读出。变换在**全维度 h** 上
        做（workspace 维经 W1/W2 进入读出），之后才按 readout_dims 切片。

        readout_linear_resid（W4 契约 B 对照臂）：先做纯线性状态变换
        h → h + P·(Q h)（整体 = (I + PQ) h），再走 tied 读出；参数量与 gated
        臂相同（2·d²）但**无任何非线性**。切片语义同 gated。

        缺省（两者皆 False）与上述全维度/切片路径都逐位还原 v1。"""
        with torch.autocast("cuda", enabled=False):
            hf = h.float()
            if self.cfg.readout_gated:
                hf = hf + F.silu(self.readout_w1(hf)) * self.readout_w2(hf)
            elif self.cfg.readout_linear_resid:
                hf = hf + self.readout_p(self.readout_q(hf))
            r = self.cfg.readout_dims
            if r is None or r == self.cfg.d_model:
                return hf @ self.embed.weight.float().T
            U = self.embed.weight.float()[:, :r]
            return hf[..., :r] @ U.T

    def _input_states(self, tokens: torch.Tensor) -> torch.Tensor:
        T = tokens.shape[1]
        h = self.embed(tokens) + self.pos_embed.weight[:T]
        return cap_norm(h, self.cfg.norm_cap)

    # ---------- forward / loss ----------

    def forward(self, tokens: torch.Tensor, exit_depths: torch.Tensor | None = None,
                targets: torch.Tensor | None = None,
                shallow_weight: float = 1.0,
                final_weight: float = 1.0,
                shallow_stopgrad: bool = False,
                dense_aux: bool = False):
        """tokens [B,T] = 干净输入流（无 corruption）；targets 缺省 = tokens
        本身，标准 AR 移位在 loss 内完成（行 i 的 depth-k 读出预测 t_{i+1}，
        仅当该行确实到达 depth k，即 d_i ≥ k）。

        shallow_weight（ticket 23b）：浅层深度 0..L-1 的 CE 权重，最终深度恒
        权重 1；总损失按 (w·L + 1) 归一保持量级。w=1 严格还原 v1 等权平均
        （W0 复现点）；dense 臂只有最终深度 CE，不受此参数影响。

        shallow_stopgrad（ticket 23c，H3）：浅层（depth 0..L-1）读出前对状态
        detach——浅层 CE 只训练 tied U 读出头，梯度不穿 trunk（纯读出）；
        损失值不变，只改梯度通道。

        dense_aux（ticket 23c，H1）：dense 臂开启全深度并行监督（L+1 个 CE
        分量，梯度穿 trunk）；缺省 False 保持 dense 臂仅最终深度 CE。

        final_weight（ticket 24 契约配方）：最终深度 CE 权重（浅层恒
        shallow_weight）；总损失按 (w·L + final_weight) 归一保持量级。
        1.0 严格还原 v1 等权（ticket 23 复现点）；2.0 = 契约的
        「权重均匀 + 最终深度 2×」。"""
        B, T = tokens.shape
        L = self.cfg.n_layers
        h = self._input_states(tokens)
        states = [h]
        gate_means: list[torch.Tensor] = []
        attn_sparse_layers: list[dict] = []
        attn_linear_layers: list[dict] = []
        if not self.cfg.dense:
            if self.cfg.proj_fill and not self.cfg.fill_kv:
                raise ValueError("proj_fill requires fill_kv=True")
            if exit_depths is None:
                exit_depths = torch.full((B, T), L, dtype=torch.long,
                                         device=tokens.device)
            frozen = h
            for k in range(1, L + 1):
                active = exit_depths >= k                       # [B,T]
                g = torch.sigmoid(self.gates[k - 1](h))
                if self.cfg.fill_kv:
                    # 混合深度 KV 源：活跃行用当前表征，已退出行用冻结表征
                    # （proj-fill 臂：冻结表征先经本层线性 adapter remap）
                    kv_src = frozen
                    if self.cfg.proj_fill:
                        kv_src = self.kv_adapters[k - 1](frozen)
                    kv_in = torch.where(active.unsqueeze(-1), h, kv_src)
                    out, _ = self.blocks[k - 1](h, kv_from=kv_in, gate_vec=g)
                else:
                    # no-fill ragged：早退位置在本层 KV 中缺席（原引擎语义，
                    # decode.Decoder per-layer cache 同款）——只有因果键 ∩
                    # 已达层 k 的行参与注意力，无填充。掩码显式 expand 到各
                    # head（CUDA SDPA 内核不广播 bool 掩码的 head 维）
                    mask = self._ragged_mask(exit_depths, k)
                    mask = mask.expand(-1, self.cfg.n_heads, -1, -1)
                    out, _ = self.blocks[k - 1](h, attn_mask=mask,
                                                gate_vec=g)
                h_next = torch.where(active.unsqueeze(-1), out, frozen)
                # 本层退出的位置在此冻结（其 depth-k 表征定格）
                frozen = torch.where((exit_depths == k).unsqueeze(-1),
                                     h_next, frozen)
                h = h_next
                states.append(h)
                gate_means.append(g.detach().float().mean())
        elif self.cfg.topology == "dense":
            # dense 拓扑：每层输入 = 对所有此前层输出的学习加权和（再进 Block）
            for k, blk in enumerate(self.blocks, start=1):
                h, _ = blk(self._layer_mix_input(states, k))
                states.append(h)
        else:  # dense 对照：无门、无退出，普通残差堆叠（sequential/parallel/top-k）
            win_mask = (self._window_mask(B, T, tokens.device)
                        if self.cfg.train_window > 0 else None)
            for blk in self.blocks:
                h, tele = blk(h, attn_mask=win_mask)
                states.append(h)
                if tele is not None:      # 块自带遥测：按块声明的类别归类
                    kind = getattr(blk, "telemetry_kind", None)
                    if kind == "attn_sparse":
                        attn_sparse_layers.append(tele)
                    elif kind == "attn_linear":
                        attn_linear_layers.append(tele)

        targets = tokens if targets is None else targets
        sg = bool(shallow_stopgrad)
        n_states = len(states)
        # H3 stop-grad：浅层读出走 detached 状态（梯度不穿 trunk）；最终深度不变
        depth_logits = [self.readout(s.detach() if sg and k < n_states - 1 else s)
                        for k, s in enumerate(states)]
        # H1 dense+aux：dense 臂也监督全部深度；缺省 dense 仅最终深度
        readouts = depth_logits if (not self.cfg.dense or dense_aux) else depth_logits[-1:]
        loss = tokens.new_zeros(()).float()
        depth_ce: list[torch.Tensor] = []
        w = float(shallow_weight)
        for k, logits in enumerate(readouts):
            if self.cfg.dense:
                mask = torch.ones(tokens.shape[0], tokens.shape[1] - 1,
                                  dtype=torch.bool, device=tokens.device)
            else:
                mask = exit_depths[:, :-1] >= k  # 预测行必须真正到达 depth k
            ce = self._masked_ce(logits[:, :-1], targets[:, 1:], mask)
            depth_ce.append(ce)
            # aux 加权（ticket 23b）：浅层深度乘 w，最终深度乘 final_weight
            # （ticket 24：缺省 1.0 = v1 等权复现点）
            wk = final_weight if k == len(readouts) - 1 else w
            loss = loss + wk * ce
        loss = loss / (w * (len(readouts) - 1) + final_weight)
        aux = {"states": states, "depth_logits": depth_logits, "depth_ce": depth_ce,
               "gate_mean": gate_means, "exit_depths": exit_depths}
        if attn_sparse_layers:
            aux["attn_sparse"] = attn_sparse_layers
        if attn_linear_layers:
            aux["attn_linear"] = attn_linear_layers
        return loss, aux

    def _layer_mix_input(self, states: list[torch.Tensor], k: int) -> torch.Tensor:
        """dense 拓扑：层 k 的输入 = Σ_{j<k} α_kj · h_j（`states[0..k−1]` = depth
        0..k−1 的表征；α 索引 = k(k−1)/2 + j）。α 只在 bf16 autocast 下转成 h 的
        数值类型（保持与 sequential 臂同一计算类型）。identity 起步时
        α = [0,..,0,1] ⇒ 逐项累加后与 h_{k−1} 逐位相等（单测钉住）。"""
        off = k * (k - 1) // 2
        alpha = self.layer_mix[off:off + k].to(states[0].dtype)
        out = alpha[0] * states[0]
        for j in range(1, k):
            out = out + alpha[j] * states[j]
        return out

    @staticmethod
    def _masked_ce(logits: torch.Tensor, targets: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """掩码行平均 CE；掩码全空时贡献 0（不产生 NaN）。"""
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                             targets.reshape(-1), reduction="none")
        m = mask.reshape(-1).float()
        return (ce * m).sum() / m.sum().clamp_min(1.0)

    def _window_mask(self, B: int, T: int, device) -> torch.Tensor:
        """训练期可见窗口掩码 → SDPA 用的 `[B, H, T, T]` bool（CUDA SDPA 的
        bool 掩码不广播 head 维，与 `_ragged_mask` / `retention.masked_logits`
        同款 expand）。掩码由 `window_anchor_keep_mask` 构造 ⇒ 评测侧用同一
        函数即与训练侧逐位一致（单测钉住）。"""
        keep = window_anchor_keep_mask(T, self.cfg.train_window,
                                       self.cfg.train_anchors, device)
        return keep.view(1, 1, T, T).expand(B, self.cfg.n_heads, T, T)

    def _ragged_mask(self, exit_depths: torch.Tensor, k: int) -> torch.Tensor:
        """层 k 的 no-fill 注意力掩码 [B,1,T,T]（bool，True=允许）：
        允许 = 因果（j ≤ i）且 key 行已到达层 k（d_j ≥ k）。对角恒开——
        全掩码行（必为已退出行，输出本就被丢弃）softmax 全 -inf 会产生 NaN
        并经共享 K/V 梯度扩散；开对角对其余行语义零影响。"""
        T = exit_depths.shape[1]
        device = exit_depths.device
        causal = torch.ones(1, 1, T, T, dtype=torch.bool,
                            device=device).tril()               # [1,1,T,T]
        key_reached = exit_depths[:, None, None, :] >= k         # [B,1,1,T]
        allow = causal & key_reached                             # [B,1,T,T]
        allow = allow | torch.eye(T, dtype=torch.bool,
                                  device=device)
        return allow

    def _gated_step(self, k0: int, h: torch.Tensor) -> torch.Tensor:
        """层 k0（0-index）的带门精化：h + g ⊙ delta，KV 取 ln1(h)
        （全深度语义，kv_in == h）。探针/解码复用同一精化定义。"""
        if self.cfg.topology != "sequential":
            raise ValueError(
                f"_gated_step 只定义在 sequential 拓扑上（收到 {self.cfg.topology!r}）："
                "层间拓扑干预只在 dense=True（无退出/无填充）形态下开放")
        out, _ = self.blocks[k0](h)
        g = torch.sigmoid(self.gates[k0](h))
        return h + g * (out - h)


# ---------- 探针 ----------

@torch.no_grad()
def depth_curve(model: DepthARModel, eval_arr, n_batches: int = 20,
                batch_size: int = 16, generator: torch.Generator | None = None,
                device: str = "cpu") -> list[dict]:
    """全深度前向（无退出）的逐深度 bpc + next-token acc：depth-bpc 曲线。
    单调递减 = 精化成立。"""
    from .data import batch
    generator = generator or torch.Generator().manual_seed(2)
    T = model.cfg.seq_len
    nats = [0.0] * (model.cfg.n_layers + 1)
    hits = [0] * (model.cfg.n_layers + 1)
    n = 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, generator)
        x = x.to(device)
        _, aux = model(x, exit_depths=None)
        for k, lg in enumerate(aux["depth_logits"]):
            lg = lg[:, :-1]
            tgt = x[:, 1:]
            nats[k] += F.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                       tgt.reshape(-1), reduction="sum").item()
            hits[k] += (lg.argmax(-1) == tgt).sum().item()
        n += x[:, 1:].numel()
    return [{"depth": k, "bpc": round(nats[k] / n / LN2, 4),
             "acc": round(hits[k] / n, 4)} for k in range(len(nats))]


@torch.no_grad()
def tf_acceptance(model: DepthARModel, eval_arr, n_batches: int = 20,
                  batch_size: int = 16, generator: torch.Generator | None = None,
                  device: str = "cpu") -> list[dict]:
    """出口草稿 + 全栈验证的接受率（teacher-forcing 代理）：全深度前向后，
    depth k 的 argmax 是否等于 depth L 的 argmax（greedy 自 spec 的草稿
    命中率代理；同一 token 被两个深度独立预测）。"""
    from .data import batch
    generator = generator or torch.Generator().manual_seed(3)
    T = model.cfg.seq_len
    L = model.cfg.n_layers
    agree = [0] * L
    n = 0
    model.eval()
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, T, generator)
        x = x.to(device)
        _, aux = model(x, exit_depths=None)
        full = aux["depth_logits"][-1][:, :-1].argmax(-1)
        for k in range(L):
            draft = aux["depth_logits"][k][:, :-1].argmax(-1)
            agree[k] += (draft == full).sum().item()
        n += full.numel()
    return [{"depth": k, "accept": round(agree[k] / n, 4)} for k in range(L)]


@torch.no_grad()
def workspace_probe(model: DepthARModel, eval_arr, n_batches: int = 4,
                    batch_size: int = 8, eps_frac: float = 0.05,
                    delta: float = 0.01, device: str = "cpu",
                    generator: torch.Generator | None = None) -> dict:
    """workspace 探针（ticket 23 判据 2 的可操作化）。

    - **有效读出秩**：logit-Jacobian = tied U（[V,d]）；SVD 奇异值 > 1%
      峰值计为有效秩。null 空间 = d − rank 维 = 读出不可见的携带余量
      （d<V 时结构性为 0）。
    - **null 能量占比**：各深度状态能量落在 null(U) 的比例（模型是否
      在用携带余量存信息）。
    - **扰动敏感度（功能性判据）**：把 mid-depth 状态 h_mid 沿每个 null
      方向 ±ε（ε = 5% 行范数），重跑后半栈，测 depth-L logits 的中位
      |Δ|；> δ 的方向计为 functional（携带信息确实被后续计算读走）。
      同方向**当前行** depth-mid logits 的变化 ≈ 0 是数值对照（null
      正交性自检）。

    判据：sanity-B（d>V，null 维 > 0）workspace 维数 > sanity-C（d<V，
    结构性 0）。
    """
    from .data import batch
    d = model.cfg.d_model
    r = model.cfg.readout_dims or d
    # 有效读出表 = tied 表列切片（与 readout 同一切片）：null 空间 = 切片内
    # SVD null ∪ 结构性 workspace 维（r..d）——与 logits 不可见子空间精确一致
    U = model.embed.weight.detach().float()[:, :r]    # [V, r]
    V = U.shape[0]
    _, S, Vt = torch.linalg.svd(U, full_matrices=True)
    sv = S[: min(V, r)]
    rank = int((sv > sv[0] * 0.01).sum().item())
    null_dim = d - rank
    Nb_slice = Vt[rank:]                              # [r−rank, r]
    pad = torch.zeros(Nb_slice.shape[0], d - r, device=U.device)
    Nb = torch.cat([torch.cat([Nb_slice, pad], dim=1),
                    torch.eye(d, device=U.device)[r:]], dim=0)  # [null_dim, d]
    L = model.cfg.n_layers
    k_mid = L // 2
    generator = generator or torch.Generator().manual_seed(4)
    model.eval()
    energy = [[] for _ in range(L + 1)]
    dir_medians: list[float] = []
    control = float("inf")
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, generator)
        x = x.to(device)
        _, aux = model(x, exit_depths=None)
        if null_dim == 0:
            for k in range(L + 1):
                energy[k].append(0.0)
            continue
        for k in range(L + 1):
            h = aux["states"][k].float()
            null_part = (h @ Nb.T) @ Nb
            frac = (null_part.norm(dim=-1) ** 2) / h.norm(dim=-1).pow(2).clamp_min(1e-12)
            energy[k].append(frac.mean().item())
        h_mid = aux["states"][k_mid].float()
        logits_mid = model.readout(aux["states"][k_mid])
        logits_L = model.readout(aux["states"][-1])
        eps_rows = eps_frac * h_mid.norm(dim=-1, keepdim=True)     # [B,T,1]
        for j in range(null_dim):
            n_j = Nb[j]                                            # [d]
            for sign in (+1.0, -1.0):
                h_p = h_mid + sign * eps_rows * n_j
                hh = h_p
                for gi in range(k_mid, L):
                    hh = model._gated_step(gi, hh)
                dlog = (model.readout(hh) - logits_L).abs().flatten()
                dir_medians.append(dlog.median().item())
                ctrl = (model.readout(h_p) - logits_mid).abs().flatten().max().item()
                control = min(control, ctrl)
    med = sorted(dir_medians)
    return {
        "vocab_size": V, "d_model": d,
        "readout_dims": r if model.cfg.readout_dims is not None else None,
        "readout_rank": rank,
        "null_dim": null_dim,
        "null_energy_frac_by_depth": [round(sum(e) / len(e), 4) for e in energy],
        "perturb_eps_frac": eps_frac,
        "delta": delta,
        "functional_dirs": sum(1 for m_ in med if m_ > delta),
        "median_dlogit_over_dirs": round(med[len(med) // 2], 6) if med else 0.0,
        "max_dlogit_over_dirs": round(med[-1], 6) if med else 0.0,
        "control_same_row_max_dlogit": (round(control, 8)
                                        if control != float("inf") else None),
    }


@torch.no_grad()
def readout_workspace_energy(model: DepthARModel, eval_arr, n_batches: int = 4,
                             batch_size: int = 8, device: str = "cpu",
                             generator: torch.Generator | None = None) -> dict:
    """B 臂探针（ticket 25）：各深度状态能量在「读出切片 h[:r]」与
    「workspace h[r:]」的占比——结构切分后 workspace 维是否真被占用。
    readout_dims=None 时读出占比恒 1（全维度读出，无结构性 workspace）。"""
    from .data import batch
    r = model.cfg.readout_dims or model.cfg.d_model
    generator = generator or torch.Generator().manual_seed(6)
    model.eval()
    read_frac = [[] for _ in range(model.cfg.n_layers + 1)]
    for _ in range(n_batches):
        x, _ = batch(eval_arr, batch_size, model.cfg.seq_len, generator)
        x = x.to(device)
        _, aux = model(x, exit_depths=None)
        for k, s in enumerate(aux["states"]):
            s = s.float()
            f = (s[..., :r].norm(dim=-1) ** 2) / \
                s.norm(dim=-1).pow(2).clamp_min(1e-12)
            read_frac[k].append(f.mean().item())
    rf = [round(sum(v) / len(v), 4) for v in read_frac]
    return {"readout_dims": r, "d_model": model.cfg.d_model,
            "readout_energy_frac_by_depth": rf,
            "workspace_energy_frac_by_depth": [round(1 - v, 4) for v in rf]}


def spec_cost_model(tf_table: list[dict], n_layers: int,
                    draft_depths: tuple[int, ...] = (4, 6)) -> list[dict]:
    """等效验证吞吐估计（ticket 25 契约主口径，纯 FLOPs 模型非墙钟）。

    贪心自 spec（depth-k 出口草稿 + 全栈验证）：每循环草稿成本 k/L（trunk
    单位）+ 验证成本 1；TF 接受率 a_k 下期望产出 token = 1 + a_k（接受出
    2 个：草稿 + 验证红利；拒绝出 1 个修正）→ 等效验证加速 = (1+a_k)/(1+k/L)。
    全深度墙钟 × 加速 = 等效 tok/s（换算基数由调用方并报）。"""
    accept = {row["depth"]: row["accept"] for row in tf_table}
    out = []
    for k in draft_depths:
        a = accept[k]
        out.append({"draft_depth": k, "tf_accept": a,
                    "draft_cost_frac": round(k / n_layers, 4),
                    "expected_tokens_per_cycle": round(1.0 + a, 4),
                    "cost_per_cycle": round(1.0 + k / n_layers, 4),
                    "equivalent_validation_speedup":
                        round((1.0 + a) / (1.0 + k / n_layers), 4)})
    return out


# ---------- 提交与增量解码 ----------

def commit_input(model: DepthARModel, logits: torch.Tensor,
                 prev_state: torch.Tensor, mode: str, pos: int) -> torch.Tensor:
    """把退出深度的预测提交为下一个输入状态 [.., 1, d]（含 pos）。

    soft（默认）：e = Σ p·E[t]，cap_norm(e + pos)——与新鲜 token 同 gauge；
    hard：e = E[argmax]，同上；
    latent（ablation）：直通退出深度状态 + pos，不过 cap（gauge 断裂，
    预期漂移，如实记录）。"""
    pe = model.pos_embed.weight[pos:pos + 1]
    if mode == "latent":
        return prev_state + pe
    if mode == "soft":
        p = logits.float().softmax(-1)
        e = p @ model.embed.weight.float()
    elif mode == "hard":
        e = model.embed.weight.float()[logits.argmax(-1)]
    else:
        raise ValueError(f"unknown commit mode {mode!r}")
    return cap_norm(e + pe, model.cfg.norm_cap)


class DepthARDecoder:
    """增量单序列解码：逐层 KV cache + 冻结态填充。

    缓存每层的 K/V 及其绝对位置；已退出 position 的冻结态与退出深度也
    入缓存。层 k0 的完整 KV = 旧缓存按绝对位置散点 + 冻结态填充
    （depths[p] ≤ k0 的位置用 kv_proj_{k0}(ln1(frozen_p))）——与训练
    forward 的混合深度语义逐位一致
    （tests/test_depth_ar.py::test_decoder_matches_parallel_forward 钉住）。
    """

    def __init__(self, model: DepthARModel):
        if model.cfg.n_kv_heads is not None \
                and model.cfg.n_kv_heads < model.cfg.n_heads:
            raise ValueError(
                "增量解码（KV cache）路径未定义 KV 头共享：GQA/MQA 只在 dense=True"
                "（无退出/无填充）形态下开放，而解码器只用于非 dense 的变深度路径"
                "（单因素 scope）。本契约不主张解码侧收益，故不实现。")
        self.model = model
        L = model.cfg.n_layers
        self.K: list[torch.Tensor | None] = [None] * L   # [1,H,C,dh] per layer
        self.V: list[torch.Tensor | None] = [None] * L
        self.pos_idx: list[torch.Tensor | None] = [None] * L  # 缓存列的绝对位置
        self.frozen: torch.Tensor | None = None          # [1,S,d]
        self.depths: list[int] = []
        self.n = 0                                       # 已缓存 position 数

    def _layer_kv(self, k0: int) -> tuple[torch.Tensor, torch.Tensor]:
        """层 k0（0-index）的完整 K/V [1,H,S,dh]：旧缓存散点 + 冻结填充。"""
        model = self.model
        d, nh = model.cfg.d_model, model.cfg.n_heads
        block = model.blocks[k0]
        S = self.n
        k_all = torch.zeros(1, nh, S, d // nh, device=model.embed.weight.device)
        v_all = torch.zeros_like(k_all)
        if self.K[k0] is not None:
            idx = self.pos_idx[k0]
            k_all[:, :, idx] = self.K[k0]
            v_all[:, :, idx] = self.V[k0]
        depths = torch.tensor(self.depths, device=k_all.device)
        shallow = (depths <= k0).nonzero().flatten()     # 未跑到层 k0 的位置
        if shallow.numel():
            kv = block.qkv(block.ln1(self.frozen[:, shallow]))
            kk, vv = kv[..., d:2 * d], kv[..., 2 * d:]
            shape = lambda t: t.view(1, -1, nh, d // nh).transpose(1, 2)
            k_all[:, :, shallow] = shape(kk)
            v_all[:, :, shallow] = shape(vv)
        return k_all, v_all

    def step(self, inp: torch.Tensor,
             forced_depth: int) -> tuple[torch.Tensor, torch.Tensor]:
        """inp [1,1,d] 已含 pos 的输入状态；forced_depth ∈ 0..L。
        返回（退出深度表征 [1,1,d]，该深度 logits [1,V]）。"""
        model = self.model
        L = model.cfg.n_layers
        d = model.cfg.d_model
        nh = model.cfg.n_heads
        shape = lambda t: t.view(1, 1, nh, d // nh).transpose(1, 2)
        h = inp
        frozen_self = inp if forced_depth == 0 else None  # depth 0 = 输入态本身
        for k0 in range(forced_depth):                    # 跑层 0..forced_depth-1
            block = model.blocks[k0]
            k_all, v_all = self._layer_kv(k0)
            qkv = block.qkv(block.ln1(h))
            q, k_self, v_self = qkv[..., :d], qkv[..., d:2 * d], qkv[..., 2 * d:]
            a = F.scaled_dot_product_attention(shape(q),
                                               torch.cat([k_all, shape(k_self)], dim=2),
                                               torch.cat([v_all, shape(v_self)], dim=2))
            a = a.transpose(1, 2).reshape(1, 1, d)
            y = h + block.proj(a)
            y = y + block.mlp(block.ln2(y))
            g = torch.sigmoid(model.gates[k0](h))
            h = h + g * (y - h)                           # carry 门乘整个 delta
            if k0 == forced_depth - 1:
                frozen_self = h
            self.K[k0] = shape(k_self) if self.K[k0] is None else \
                torch.cat([self.K[k0], shape(k_self)], dim=2)
            self.V[k0] = shape(v_self) if self.V[k0] is None else \
                torch.cat([self.V[k0], shape(v_self)], dim=2)
            pos_t = torch.tensor([self.n], device=h.device)
            self.pos_idx[k0] = pos_t if self.pos_idx[k0] is None else \
                torch.cat([self.pos_idx[k0], pos_t])
        self.frozen = frozen_self if self.frozen is None else \
            torch.cat([self.frozen, frozen_self], dim=1)
        self.depths.append(forced_depth)
        self.n += 1
        return h, model.readout(h)


def token_input(model: DepthARModel, token: int, pos: int) -> torch.Tensor:
    """普通 token 输入状态 [1,1,d]（embed + pos，cap），prefill 用。"""
    e = model.embed.weight.float()[token].view(1, 1, -1)
    pe = model.pos_embed.weight[pos:pos + 1]
    return cap_norm(e + pe, model.cfg.norm_cap)


@torch.no_grad()
def generate(model: DepthARModel, prompt, n_new: int, commit: str = "soft",
             forced_depth: int | None = None) -> tuple[list[int], dict]:
    """贪心生成。forced_depth 给定则所有 position 固定退出深度（出口质量
    测量）；None = 满深度。commit 决定下一输入的提交方式。"""
    model.eval()
    dec = DepthARDecoder(model)
    L = model.cfg.n_layers
    fd = L if forced_depth is None else forced_depth
    prompt = prompt.tolist() if torch.is_tensor(prompt) else list(prompt)
    if len(prompt) + n_new + 1 > model.cfg.seq_len:
        raise ValueError("prompt+n_new exceeds seq_len")
    h = logits = None
    for t in prompt:
        h, logits = dec.step(token_input(model, t, dec.n), fd)
    gen: list[int] = []
    for _ in range(n_new):
        gen.append(int(logits.argmax()))
        if len(gen) >= n_new:
            break
        inp = commit_input(model, logits, h, commit, dec.n)
        h, logits = dec.step(inp, fd)
    return gen, {"commit": commit, "forced_depth": fd, "depths": [fd] * n_new}


@torch.no_grad()
def generation_quality(model: DepthARModel, eval_arr, prompt_len: int = 64,
                       n_new: int = 128, depths: list[int] | None = None,
                       commits: tuple = ("soft", "hard", "latent"),
                       seed: int = 5, device: str = "cpu") -> list[dict]:
    """出口提交测量（判据 3）：每深度 × 每提交方式生成 128 token，报
    合法性（有限/在词表内）、distinct-2（塌缩参考，22b 教训）、生成文本
    在全深度下的重评分 bpc（漂移参考）与出口平均 top-1 概率。"""
    from .data import batch
    generator = torch.Generator().manual_seed(seed)
    n_new = min(n_new, model.cfg.seq_len - prompt_len - 1)  # 位置预算内
    prompt, _ = batch(eval_arr, 1, prompt_len, generator)
    prompt = prompt.to(device)
    L = model.cfg.n_layers
    depths = [0, L // 2, L] if depths is None else depths
    out = []
    model.eval()
    for depth in depths:
        for commit in commits:
            gen, stats = generate(model, prompt[0], n_new, commit=commit,
                                  forced_depth=depth)
            legal = all(0 <= t < model.vocab_size for t in gen)
            # 生成段 distinct-2
            bigrams = list(zip(gen, gen[1:]))
            d2 = len(set(bigrams)) / max(len(bigrams), 1)
            # 重评分：prompt + 生成段的全深度 CE（nats/token → bpc）
            seq = torch.cat([prompt, torch.tensor([gen], device=device)], dim=1)
            _, aux = model(seq[:, :model.cfg.seq_len], exit_depths=None)
            lg = aux["depth_logits"][-1][0, prompt_len - 1:-1]
            tgt = seq[0, prompt_len:prompt_len + lg.shape[0]]
            nats = F.cross_entropy(lg, tgt, reduction="mean").item()
            # 出口平均 top-1 概率：TF 重放 prompt 的前 min(prompt_len, rest) 个位置
            _, aux_p = model(seq[:, :prompt_len], exit_depths=None)
            probs = aux_p["depth_logits"][depth][0].float().softmax(-1)
            top1 = probs[:-1].max(-1).values.mean().item() if depth < prompt_len else None
            out.append({"depth": depth, "commit": commit,
                        "legal": bool(legal and all(torch.isfinite(torch.tensor(gen)))),
                        "distinct2": round(d2, 4),
                        "rescore_bpc": round(nats / LN2, 4),
                        "exit_top1_prob_tf": round(top1, 4) if top1 is not None else None})
    return out
