# MODEL · 2619 · 统一 KV 预算参照线总图（ticket 36 · U3 线）· 13M

权重（`*.pt`）**不入库**（`.gitignore` 已屏蔽）。本实验**不训练、不新增权重**：所有点
都复用既有 checkpoint（只读），`run.py::load_trunk` / `load_head_arm` 在运行时
**逐臂断言 sha256**（与下表不符直接 `SystemExit`）。

## 0. 被测 checkpoint 一览

| 族 | 臂 | 文件 | 参数 | 体积 | SHA-256 | 训练出处 |
|---|---|---|---|---|---|---|
| trunk | `fp16` / `int8` / `int4` / `win*` | `2610-frozen-trunk-jumphead-13m/trunk.pt` | 13,536,768 | 54,193,675 B | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` | 2610（12k 步，dense） |
| head | `mha` | `2616-gqa-mqa-13m/runs/mha-12k-s0/model.pt` | 13,536,768 | 54,193,675 B | `557453eb12b56a453b29ea279cdbbc0969cfa6bf0dbe1cdbd9965cd6cec62528` | 2616（12k 步，MHA） |
| head | `gqa` | `2616-gqa-mqa-13m/runs/gqa-12k-s0/model.pt` | 12,352,512 | 49,456,651 B | `bd682447c90bc118253a1496b4241bdc711a4103ab8df06f8296a840e7b3bc00` | 2616（12k 步，GQA 2 KV 头） |
| head | `mqa` | `2616-gqa-mqa-13m/runs/mqa-12k-s0/model.pt` | 12,155,136 | 48,667,147 B | `24bdfe7f1549107175e9343444372a6abe3aba7c6ba2423e8ab069a145b1ea97` | 2616（12k 步，MQA 1 KV 头） |

共同架构：`DepthARModel(dense=True)`（12 层 × d256 × 8 头 × mlp6 × seq256，tied 读出、
无 final LN；无门/无退出/仅最终深度 CE）。head 族的唯一差别 = `n_kv_heads`
（`None`/8 = MHA、2 = GQA、1 = MQA），其 `GQABlock` 的 qkv 输出宽度 = `d + 2·n_kv_heads·dh`。

> 注：trunk 与 mha 是**同一配方、同一 seed 的两次独立训练**（2610 vs 2616），
> 参数总量逐位相同但权重不同；两者的 clean bpc 差 0.0016（1.483569 vs 1.482026），
> 是 CUDA 归约顺序导致的 run-to-run 散度。因此本实验的 `damage_bpc` 一律相对**本族**
> 基线（trunk 族 = `fp16`，head 族 = `mha`），未跨族相减。

## 1. 生成命令（复用，本实验不重跑）

```bash
# trunk（2610）
python3 train_depth_ar.py trunk-12k-s0 --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
  --steps 12000 --batch-size 32 --seed 0 --lr 0.001 --eval-batches 24 \
  --out-root axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m

# head 族（2616；四臂串行，本实验只用 mha / gqa / mqa）
python3 run.py --train    # 在 axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/
```

## 2. 本实验新增的派生物（不入库）

| 产物 | 说明 |
|---|---|
| `results.json` | 14 个预算点的 bpc / 召回 / 活性 / 对账（无权重） |
| `chart.png` / `chart.txt` / `table.md` | 图与表 |
| `diagnose_copy_confound.json` | 混杂排查（逐 k 分解 / 毁两端 / 打乱答案） |

**没有新增 `.pt`**：本实验的干预全在推理期（KV 量化往返 / 可见性掩码）或复用了
以不同 `n_kv_heads` 重新训练的既有 checkpoint。
