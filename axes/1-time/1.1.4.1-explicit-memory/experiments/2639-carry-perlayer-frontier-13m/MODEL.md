# MODEL · 2639 完整机制（逐层独立状态 + 内容写门 ± γ）质量-代价前沿（13M）

> 权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出 sha256、体积与生成命令。
> 13 臂 = 10 个**复用臂**（eval-only，来源见下）+ 3 个**本实验新训练臂**。

| 臂 | kind | 窗口 | 通道 | 机制 | seed | 来源 | sha256 | 体积(B) | 校验 |
|---|---|---|---|---|---|---|---|---|---|
| `full` | full | 256 | None | — | 0 | 2610 `trunk.pt`（复用） | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` | 54193675 | OK |
| `win16_s0` | window | 16 | None | — | 0 | 2634（复用，eval-only） | `c3f6f08a4fd1dfe9b5c72cb433fe20a2390052b1eda11b8ab9a6d0fe468deca7` | 54193675 | OK |
| `win16_s1` | window | 16 | None | — | 1 | 2634（复用，eval-only） | `2bfe167c133c496f42864a191f2fc8833f20ba4ec8f9f4f1dceba43dd84fb162` | 54193675 | OK |
| `win17_s0` | window | 17 | None | — | 0 | 2638（复用，eval-only） | `70f0acb92067adba4f796612bab49ead7153c1b4b7e5bb3ea15a5219ab9cdc1c` | 54193675 | OK |
| `win18_s0` | window | 18 | None | — | 0 | 2638（复用，eval-only） | `39316ac04f20e158a6833323888990d149bf22c8da76ca04c4aa642cdcfa5913` | 54193675 | OK |
| `win24_s0` | window | 24 | None | — | 0 | 2638（复用，eval-only） | `ebd4ef84736355bc163479e702c34b1383708f90d39510756e38b2cf3fcf90d1` | 54193675 | OK |
| `win32_s0` | window | 32 | None | — | 0 | 2638（复用，eval-only） | `dccc58c0065d70f0c202424b7f5f40e28929ffb238db3d3a3954d3095e43287b` | 54193675 | OK |
| `win128_s0` | window | 128 | None | — | 0 | 2638（复用，eval-only） | `c17a1cf1747ca59e426eeb1378c90e50802e21401d7058e703a766e8cbe287f7` | 54193675 | OK |
| `carry16_s0` | carry | 16 | delta | 单共享槽（c64） | 0 | 2634（复用，eval-only） | `d92114822c09f2f7538ea6a59802eed21b1297409fe91f66c031dec9d6c305eb` | 55255619 | OK |
| `carry16_s1` | carry | 16 | delta | 单共享槽（c64） | 1 | 2634（复用，eval-only） | `b3d3cdbc57266109b6b461d18c2188c7d1712170f6760b35a48ff9b6170d6843` | 55255619 | OK |
| **`carrypl16_s0`** | carry | 16 | delta | **逐层状态 + 内容写门**（γ=1，c256） | 0 | **本实验训练** | `382a74aa0e417b229910cc44f55dcc949c225b3211434df488ee6a72d3036ec3` | 55264963 | OK |
| `c16c256_s0` | carry | 16 | delta | 单共享槽（c256，chunk 匹配对照） | 0 | **本实验训练** | `3e935b33f1cc28ba31b59a7fe7fc82dbe88c3f7dba316a097e82fe3c5a368e5b` | 55255619 | OK |
| `carrypl16d_s0` | carry | 16 | delta | 逐层状态 + 内容写门 + γ=0.99 | 0 | **本实验训练** | `cb1940ee2e97f2311998710870be0c86024ed7d65e2ce54c159fcffbca8fe386` | 55264963 | OK |

`sha256` 全部与磁盘逐位一致（`run.py` 加载时断言，见 `results.json.checkpoints[*].sha256_verified`）。

## 生成命令（本实验新训练臂）

```bash
# pl16（完备机制，γ=1）/ c16（chunk 匹配的单共享槽对照）/ pl16d（完备机制 + γ=0.99）
ARMS="pl16 c16 pl16d" bash axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/train_arms.sh
# 单臂等价命令（逐臂取/放 §20 GPU 锁）：
CARRY_PER_LAYER=1 CARRY_DECAY=1.0 python3 axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/train_pl.py pl16 \
  --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --dense --topology sequential \
  --steps 12000 --batch-size 32 --lr 1e-3 --seed 0 --exit-mode uniform --train-window 16 \
  --train-anchors 0 --eval-batches 4 --carry-write delta --carry-chunk 256 \
  --out-root axes/1-time/1.1.4.1-explicit-memory/experiments/2639-carry-perlayer-frontier-13m/train
```

`train_pl.py` 只把 `train_depth_ar.DepthARConfig` 换成带 E1 缺省（`carry_per_layer`/`carry_decay`）
的子类；共享 runner 与共享引擎**一行未改**（E1 由引擎线交付，见
`.scratch/67-engine-changes/E1-CARRYCHANNEL.md` + `REVIEW-E1.md`）。

## 复用臂的出处

- `full` = `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt`
  （13.54M 全因果主干；12000 步 × bs32 × seq256 = 98.3M token、seed 0、AdamW lr1e-3+cosine）。
- `win16_s0/s1`、`carry16_s0/s1` = `.../2634-carry-realtext-13m/train/{rt_none_s0,rt_none_s1,rt_delta_s0,rt_delta_s1}/model.pt`。
- `win{17,18,24,32,128}_s0` = `.../2638-carry-frontier-13m/train/win{17,18,24,32,128}/model.pt`。
- 复用等值性（`results.json.reuse_equivalence`）：**10/10 复用臂的 clean bpc 与
  `2638/results.json` 逐位相同** ⇒ 换引擎（E1 默认路径逐位不变）后复用臂未被扰动。

## 共享文件指纹（本轴未改任何共享文件；只读引用）

- `pathlm/depth_ar.py`: `2f4b88877028f2acb7fd2b17ddbe145937a207232a2013141255fe98f513f893`
  （= E1 之后的版本；`2638` 记录的是 E1 之前的 `1f8f9eb4…`）
- `train_depth_ar.py`: `218b63179e623d6a938e7cabe85a71c5c4a36f1831c6930e9c65b2e0a6673022`
- `tools/evalkit/retention.py`: `9ef983b70d666460de5d63d5d81c23c685ce708a55dfe1af66fc23187fd12f9a`
- `pathlm/data.py`: `48e23520122a0b4591e71d0e3780dd1f34dc4a82744e14ca3414da8f43ba448a`
- `run.py`: `f04d59da80e9df564bc556c021669ce54a856ec80c041f5d0bde66f21ba0ecf1`
- `config.json`: `28ba94856d40c409868ef4033768fe1fad9743a63ffcc06e0700a89be86cbe14`
- `train_pl.py`: `c63d9386b4cf07d3a71b2ddb2670b4b2dbea27d1135b3772e1cbd2cf3ce54cd8`
- `train_arms.sh`: `dd393d9c4452fc874959bca50d8d1ea38049d69a245bf6fb780732b501216e25`
