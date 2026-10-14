# MODEL.md — 2672-fill-n2-13m 的 checkpoint 清单与复现命令

> 本轮全部为 **2k 快照**（2000 步）checkpoint，与 12k 档不可比；sha256 覆盖 `model.pt` 逐字节。12k 档 checkpoint 为**只读复用**（未修改）。

## 1 本实验产出的 6 个 checkpoint

| run | seed | 臂开关 | 步数 | sha256(model.pt) | 文件 |
|---|---|---|---|---|---|
| snap2k-fill-s0 | 0 | fill_kv=True | 2000 | ee544a5515bd28f21dbd741827550125626d49c659530a84f590beda5b179fa6 | model.pt |
| snap2k-nofill-s0 | 0 | fill_kv=False | 2000 | 14d95da827ffa58ac256dc979c94fc941b9f386a18b53159d2b73ca3970b6442 | model.pt |
| snap2k-projfill-s0 | 0 | fill_kv=True + proj_fill | 2000 | d7c5cb922f1f2ef4f4a8eeccbcb9bc808f194d29b8672d9aee11d8025779e573 | model.pt |
| snap2k-fill-s1 | 1 | fill_kv=True | 2000 | 9f14a70af4e54a7d2d2befba90949b39f6861f346fb5b899cb49257cee57e4c8 | model.pt |
| snap2k-nofill-s1 | 1 | fill_kv=False | 2000 | 672d89c897e2d5ddae2c6cce6d11569406a261d58624c078d313b964de44906f | model.pt |
| snap2k-projfill-s1 | 1 | fill_kv=True + proj_fill | 2000 | 03bbc75c32c68cb8f9bd4bb84a464174bd19eaf6d72a7773d1eece384ea61c82 | model.pt |

每个 run 目录同时含 `train_log.jsonl`（逐步 loss）、`gate_log.jsonl`（逐层门）、`results.json`（config/点值/填充语义开关/generation）。

## 2 只读复用的 12k 档 checkpoint

| run | seed | sha256(model.pt) | 路径 |
|---|---|---|---|
| 23*(fill) | 0 | 2b054ab7368757fe79cf36c32bacacb5e7aafe52f718a02f0558dbb3aa5446e6 | `.scratch/23-depth-ar/evidence/23c-H2-w0-12k` |
| 23*(nofill) | 0 | 2bd6f3ee4238e708a8673ad1d470ca8deadea19c62b15e6bc7ca1f0c07c3e9eb | `.scratch/23-depth-ar/evidence/23e-nofill-12k` |
| 23*(projfill) | 0 | bc80c0381d5102d6d5a410fea24c4698b44b790267d789c6f55b74077a477057 | `.scratch/23-depth-ar/evidence/23e-projfill-12k` |

## 3 复现命令（repo 根 = /home/z/vibe/pathlm）

```bash
D=axes/6-predict/6.4-training-fill/experiments/2672-fill-n2-13m

# 训练 6 个快照 run（逐 run 取放 GPU 锁；SCHEMA §20）

STEPS=2000 DEADLINE=$(date -d '+40 min' +%s) bash "$D"/run6.sh

# 或者单臂：STEPS=2000 bash "$D"/run_arm.sh projfill 1

# 等参数 rescore（取锁跑评测）+ config 对平 + 汇聚
bash "$D"/finish.sh

# 文档与墙钟
python3 "$D"/write_docs.py
```

单臂等价的原始终端命令（`run_arm.sh` 内部就是它）：

```bash
python3 -u train_depth_ar.py snap2k-fill-s1 \
  --d 256 --layers 8 --heads 8 --seq-len 256 --mlp-mult 4 \
  --steps 2000 --batch-size 32 --lr 1e-3 --seed 1 --out-root "$D"
# nofill 加 --no-fill-kv；projfill 加 --proj-fill（需 fill_kv=True，缺省）
```

## 4 权重/数据不入库

`model.pt` 共 6 × 27.9 MB 留在实验目录、不进版本库；可由上面的命令在同 seed / 同配置下重训，sha256 用于核对「评测的确实是这一个 checkpoint」。

