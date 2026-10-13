# MODEL · 2638 显式记忆状态质量-代价前沿（13M）

> 权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出 sha256、体积与生成命令。

| 臂 | kind | 窗口 | 通道 | seed | 来源 | sha256 | 体积(B) | 校验 |
|---|---|---|---|---|---|---|---|---|
| `full` | full | 256 | None | 0 | 复用 | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` | 54193675 | OK |
| `win16_s0` | window | 16 | None | 0 | 复用 | `c3f6f08a4fd1dfe9b5c72cb433fe20a2390052b1eda11b8ab9a6d0fe468deca7` | 54193675 | OK |
| `win16_s1` | window | 16 | None | 1 | 复用 | `2bfe167c133c496f42864a191f2fc8833f20ba4ec8f9f4f1dceba43dd84fb162` | 54193675 | OK |
| `win17_s0` | window | 17 | None | 0 | 本实验训练 | `70f0acb92067adba4f796612bab49ead7153c1b4b7e5bb3ea15a5219ab9cdc1c` | 54193675 | OK |
| `win18_s0` | window | 18 | None | 0 | 本实验训练 | `39316ac04f20e158a6833323888990d149bf22c8da76ca04c4aa642cdcfa5913` | 54193675 | OK |
| `win24_s0` | window | 24 | None | 0 | 本实验训练 | `ebd4ef84736355bc163479e702c34b1383708f90d39510756e38b2cf3fcf90d1` | 54193675 | OK |
| `win32_s0` | window | 32 | None | 0 | 本实验训练 | `dccc58c0065d70f0c202424b7f5f40e28929ffb238db3d3a3954d3095e43287b` | 54193675 | OK |
| `win128_s0` | window | 128 | None | 0 | 本实验训练 | `c17a1cf1747ca59e426eeb1378c90e50802e21401d7058e703a766e8cbe287f7` | 54193675 | OK |
| `carry16_s0` | carry | 16 | delta | 0 | 复用 | `d92114822c09f2f7538ea6a59802eed21b1297409fe91f66c031dec9d6c305eb` | 55255619 | OK |
| `carry16_s1` | carry | 16 | delta | 1 | 复用 | `b3d3cdbc57266109b6b461d18c2188c7d1712170f6760b35a48ff9b6170d6843` | 55255619 | OK |

## 生成命令（本实验新训练臂）

```bash
python3 train_depth_ar.py win<K> --d 256 --layers 12 --heads 8 --mlp-mult 6 \
  --seq-len 256 --dense --topology sequential --steps 12000 --batch-size 32 \
  --lr 1e-3 --seed 0 --exit-mode uniform --train-window <K> --train-anchors 0 \
  --eval-batches 4 --carry-write none --out-root axes/1-time/1.1.4.1-explicit-memory/experiments/2638-carry-frontier-13m/train
```

## 复用臂的出处

- `full` = `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`
  （13.54M 全因果主干，12000 步 × bs32 × seq256 = 98.3M token、seed 0、AdamW lr1e-3+cosine；
  配置等值性见 2615/2616 的逐字段对照）。
- `win16_s0/s1`、`carry16_s0/s1` = `.../2634-carry-realtext-13m/train/{rt_none_s0,rt_none_s1,rt_delta_s0,rt_delta_s1}/model.pt`
  （同规格、同 token 预算、同数据）。

## 共享文件指纹（本轮未改任何共享文件；只读引用）

- `pathlm/depth_ar.py`: `1f8f9eb49aac46d3e076f2964b49cfceae8215c13640b65d10a3403ef6c10d7a`
- `train_depth_ar.py`: `218b63179e623d6a938e7cabe85a71c5c4a36f1831c6930e9c65b2e0a6673022`
- `tools/evalkit/retention.py`: `9ef983b70d666460de5d63d5d81c23c685ce708a55dfe1af66fc23187fd12f9a`
- `pathlm/data.py`: `48e23520122a0b4591e71d0e3780dd1f34dc4a82744e14ca3414da8f43ba448a`
- `run.py`: `2023f36d1a9694c8579bfab76c76da5e5b57ab215cfdba46f52f992164615778`
- `config.json`: `eb97327d6523eb2003d3821228223ff60ad62c6ba6941e4c961d693ee4227f55`
