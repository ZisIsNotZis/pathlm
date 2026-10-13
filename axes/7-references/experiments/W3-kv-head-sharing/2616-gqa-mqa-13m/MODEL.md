# MODEL · 2616 · KV 头共享（GQA / MQA）（ticket 33 契约 S，W3 参照线补齐）· 13M dense

权重（`*.pt`）**不入库**（`.gitignore` 已屏蔽）；本文件给出哈希、体积与生成命令。
登记值来自 `model_hashes.json`（`run.py --train` 训练完即记），评测入口 `run.py::load_arm`
**逐臂断言 sha256**（与登记值不符直接 `SystemExit`）。

## 0. 四臂一览

| 臂 | 干预 | 文件 | 参数 | 体积 | SHA-256 | 训练墙钟 | clean bpc |
|---|---|---|---|---|---|---|---|
| mha | 基线（8 KV 头，= stock `Block`） | `runs/mha-12k-s0/model.pt` | 13,536,768 | 54,193,675 B | `557453eb12b56a453b29ea279cdbbc0969cfa6bf0dbe1cdbd9965cd6cec62528` | 8.44 min | 1.4820 |
| gqa | 2 个 KV 头 | `runs/gqa-12k-s0/model.pt` | 12,352,512 | 49,456,651 B | `bd682447c90bc118253a1496b4241bdc711a4103ab8df06f8296a840e7b3bc00` | 9.02 min | 1.4963 |
| gqa_rep | 2 个 KV 头（**同 seed 重跑**） | `runs/gqa-12k-s0-rep/model.pt` | 12,352,512 | 49,456,651 B | `7fa95a7b5419af5b8f9159acb70cd814588337aa784b9a5e1bd4d0ea9ba6cb47` | 9.00 min | 1.4959 |
| mqa | 1 个 KV 头 | `runs/mqa-12k-s0/model.pt` | 12,155,136 | 48,667,147 B | `24bdfe7f1549107175e9343444372a6abe3aba7c6ba2423e8ab069a145b1ea97` | 9.06 min | 1.4938 |

共同架构：`DepthARModel(dense=True)` —— 主流标准 decoder-only transformer 形态
（12 层 × d256 × 8 头 × mlp6 × seq256，tied 读出、无 final LN；无门/无退出/仅最终深度 CE）。

## 1. 生成命令（逐臂；完整 argv 另存 `runs/<arm>/cmd.txt`）

```bash
# mha（基线）
python3 train_depth_ar.py mha-12k-s0 --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
  --steps 12000 --batch-size 32 --seed 0 --lr 0.001 --eval-batches 24 \
  --out-root <expdir>/runs

# gqa（2 KV 头）
python3 train_depth_ar.py gqa-12k-s0 --d 256 --layers 12 --heads 8 --mlp-mult 6 --dense \
  --steps 12000 --batch-size 32 --seed 0 --lr 0.001 --eval-batches 24 \
  --n-kv-heads 2 --out-root <expdir>/runs

# mqa（1 KV 头）
python3 train_depth_ar.py mqa-12k-s0 ... --n-kv-heads 1 --out-root <expdir>/runs

# gqa 重跑（同 seed 同配方，只换 run 名）
python3 train_depth_ar.py gqa-12k-s0-rep ... --n-kv-heads 2 --out-root <expdir>/runs
```

一键：`python3 run.py --train`（逐臂串行、已存在 `model.pt` 的臂自动跳过）。

## 2. 实现与代码哈希（评测时记录，见 `results.json::code`）

| 文件 | 作用 | SHA-256 |
|---|---|---|
| `pathlm/depth_ar.py` | `GQABlock`（KV 头共享）+ `kv_head_liveness`（实际张量形状） | `9e4ca15d3e1282b3a5826e1f7f3be2828cd598d032c309d9072cc5a853d05cef` |
| `train_depth_ar.py` | `--n-kv-heads` + `build_depthar_config` + 每 200 步活性量 | `b189eb13256d647f687e9190dc456f8d203444534689a37d6fa973822d3eb9c5` |
| `tests/test_depth_ar.py` | 7 条 KV 头共享单测（`-k kv_heads`） | `b5e592dfa65bbea3f11463c8adcafd349c91b44e4e1bcbfdb90af407322bcf4b` |
| `run.py`（本实验入口） | 训练编排 + 评测电池 + 判据 + 三道对账 | `9159d5f9af73906e07a55d5f79195a418cb606c8e0dee5ba26ec836df3942a63` |
| `tools/evalkit/retention.py`（只读复用） | 第二评测路径 `masked_bpc` | `8320f03a2ecf74e32ab29b0f0af0ef610d7cf45116ae5dcdcdeb241c645e6502` |

`reconcile_numbers.py` 会重算这五个文件当前的 sha256 并要求与 `results.json::code` 一致
⇒ 保证「报告 ↔ 结果 ↔ 代码」同源（评测之后不得再改代码）。

## 3. 交叉核对锚（他人权重，本实验只读复用其**记录**，不加载其权重）

| 项 | 值 |
|---|---|
| 文件 | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` |
| 规格 | 同为 `DepthARModel(dense=True)` d256 × 12 层 × 8 头 × mlp6（= 本实验 mha 臂的规格） |
| 参数 | 13,536,768（与本实验 mha 臂**逐位相同**） |
| SHA-256 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |
| 记录 clean bpc | 1.4836（历史）vs 本实验 mha 1.4820（差 0.0016，在 run-to-run 量级内） |
| 用途 | **交叉核对**（说明基线没跑偏）；**不作为判据**，也不进结果表 |
