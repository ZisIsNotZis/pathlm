# 2690 · 长 T 上「远处点的数目」买不买得到质量（`1.2.1` 的支持读数）· 13M · dense

> 本文件是**预登记契约**：判据线在本节写死，跑之前不得修改。判据常数（band / 掩码参数 / 臂清单 /
> 配对表 / 判词规则）住在 `config.json`，与 `run.py` / `train_arms.sh` 一起被
> `python3 tools/prereg_bind.py <本目录> --contract README.md config.json run.py train_arms.sh` 绑住。
> 读数落 `results.json`（实验根）与 `runs/<臂>/results.json`（逐跑），两者都过
> `tools/run_provenance.write_results`（自带 `code_hash_run_py` / `code_hashes`）。

## 1 要回答的问题

`2689` 在 T=2048 上判「同 32 个可见键把位置摊开」相对「取最近 32 个」有没有质量增量。它的
判据③（远处键的边际价值 `D(T)`）只能说明「模型对它读到的那 16 个远处点更不吃它」，**不能**说明
远处信息在长 T 上到底值不值钱——因为 16 个远处点在 2048 长的前缀上间距 ≈ 128，覆盖率
`32/2048 = 1.6%`。所以本实验只改**远处点的数目**（16 → 128，近端窗口 16 不变，每行可见键数
32 → 144）：

> T=2048 上多付 4.5 倍的可见键、把远处点数加到 128（间距 ≈ 16，整行覆盖 7.0%），
> 能不能买到 ≥ 0.014 bpc 的质量？

这一条把 `2689` 的「带内」结论分成两种解释：**远处信息不值钱**（加密度也不买）还是
**32 键太稀、本 k 分辨不出**（加密度就买得到）。

## 2 臂的设计（唯一变量 = 远处点的数目）

| 臂 | 可见集（第 i 行） | 每行键数 |
|---|---|---|
| `dil32-t2048-s{0,1}`（对照，复用 `2689`） | 近端窗口 16 + 16 个摊满前缀的远处点 | `min(32, i+1)` |
| `dil144-t2048-s{0,1}`（实验，本实验训练） | 近端窗口 16 + **128** 个摊满前缀的远处点 | `min(144, i+1)` |

* 两族掩码是**同一条规则**的不同 `n_dil`（`j ∈ {round(t·(P−1)/(n_dil−1))}`），**近端 16 窗口逐位相同**
  （判据②机检 `same_local_window`）⇒ 这一对不是「换一种挑法」，而是「同一挑法下把远处点加密」。
  加密的量化：远处点的平均间距（行平均）`67.24` → `7.94`；对最长的那几行分别是 `≈ 136` → `≈ 16`
  （判据②落盘 `far_spacing_mean_*`）。
  两族的远处点**不互相包含**（`n_dil` 变了，均匀铺开的落点就变了），所以判据只声明
  「同一 local + 远处点更密」，不声明「实验臂 ⊇ 对照臂」。
* 掩码是静态位置掩码，构造在本目录的 `run.py`（`DilatedBlock` 走 `Block.forward(attn_mask=...)`
  的既有掩码路径），**不改** `pathlm/**`、`train_depth_ar.py`、`tools/**`、`tests/**`。

## 3 配方（与 `2689` 的 T=2048 点逐字段一致）

| 字段 | 值 |
|---|---|
| 形态 | stock dense decoder-only（`dense=True`、`topology=sequential`） |
| d / layers / heads / mlp_mult / dropout / norm_cap | 256 / 8 / 8 / 4 / 0.0 / 1.0 |
| T / batch_size / steps / lr / seed | 2048 / 8 / 6000 / 1e-3 / 0 与 1 |
| token 预算 | 98,304,000 / 臂（与 `2689` 的 T=2048 点相等） |
| 数据 | `data/enwik8_full.npz`（前 90M 训练 / 末 10M 评测，字符级 vocab 206） |
| 训练入口 | `train_depth_ar.train()`（同一个训练函数，不复制训练循环） |
| 评测 | 20 批 × bs16、`window_seed=2` 的**配对** clean bpc；**对照臂的读数在本实验仪器上重算**，不是抄 `2689` 的数字 |

## 4 判据线（预登记，跑后不改）

**Δ 方向统一**：**Δ = bpc(16 个远处点) − bpc(128 个远处点)**（对照 − 实验）⇒ **Δ > 0 = 远处点多更好**。

| # | 判据 | 判定线 | 该涨 / 该跌 / 必须不变 |
|---|---|---|---|
| ① **主判（密度）** | 两个 seed 的配对 `Δ` | **0.014 bpc 带**：`≥ +0.014` = 远处点数买得到质量（⇒ `2689` 的固定 k 答案是「本 k 分辨不出」）；`≤ −0.014` = 加密反而更差；`\|Δ\| < 0.014` = 买不到 | 该涨 |
| ② **必须不变：结构** | 实验臂逐行计数 `= min(144, i+1)`、对照臂 `= min(32, i+1)`、两者都 `⊆ causal`、两臂**近端 16 窗口逐位相同** | 机检，落盘 | 必须不变 |
| ③ **必须不变：参数量** | 两臂参数量逐位相等（唯一变量 = 掩码）；转换后逐参数等于 stock 构造 | 机检，落盘 | 必须不变 |
| ④ **等值门（口径未漂移）** | 本实验仪器重算的 `dil32-t2048-s{0,1}` 逐批 bpc 必须与 `2689-longT-layerdil-13m/results.json::per_batch_bpc` **逐位相同** | 不等即中止判决 | 必须不变 |
| ⑤ **入场门（稳定性）** | 每个新训臂 `gnorm_max < 100` 且 `grad_skips == 0` | 破门 ⇒ 该臂作废、不进判据 | 必须不变 |
| ⑥ **申报不进判据** | 墙钟、峰值显存、每行可见键数（32 vs 144 是**成本列**） | 逐臂申报 | — |

**判词规则（`config.json::verdict_rules`）**：两 seed 同向且各自越带 ⇒ 按方向入词（n=2）；
两 seed 不同向 ⇒ 「本 n 下不可排序」；同向但未同时越带 ⇒ 按「买不到质量」入词但标「未同时越带」。
**n=1 不作结论**：只剩一个 seed 时只报读数、不落判词。

**否证线（预登记）**：① 两 seed 都落带内 ⇒ 在 T=2048、这个 k 区间上「远处信息不值钱」——
它与 `2689` 的判据③ 一起构成「长 T 上真实文本不需要远处信息」的两条独立证据；
若 ① 越正带 ⇒ 那条结论收回，改判「本 k（32）分辨不出，远处信息在长 T 上是有价的」。

## 5 预算与失败处置

| 项 | 值 |
|---|---|
| 训练 GPU | 2 支新臂 × 6000 步（T=2048、bs8）≈ 5 min/臂 ⇒ **≈ 10 min**（上限 20 min） |
| 单臂 timeout | 20 min（`train_arms.sh`） |
| 显存 | 每行 144 个可见键、掩码仍是稠密 bool ⇒ 峰值与 `2689` 的 T=2048 臂同量级（逐臂申报） |
| 外部占用 | 卡上另有更高优先级项目（`/home/z/vibe/nomos`）；每臂开跑前查空闲显存，< 8 GB 即排队，不抢不杀 |
| 失败处置 | gnorm 破门 / NaN ⇒ 该臂作废登记「未跑」（不换配方重跑）；等值门失败或结构核查失败 ⇒ 中止判决 |

## 6 产出与复现

本目录：`README.md` / `config.json` / `run.py` / `train_arms.sh` / `results.json` / `analysis.md` /
`prereg_binding.json` / `runs/dil144-t2048-s{0,1}/{model.pt,train_log.jsonl,gate_log.jsonl,results.json}`。

```bash
python3 tools/prereg_bind.py axes/1-time/1.2.1-dilated/experiments/2690-dilated-density-k144-13m \
  --contract README.md config.json run.py train_arms.sh
bash axes/1-time/1.2.1-dilated/experiments/2690-dilated-density-k144-13m/train_arms.sh \
  dil144-t2048-s0 dil144-t2048-s1
python3 axes/1-time/1.2.1-dilated/experiments/2690-dilated-density-k144-13m/run.py --stage eval
```
