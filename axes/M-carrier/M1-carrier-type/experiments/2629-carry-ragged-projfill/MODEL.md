# 2629-carry-ragged-projfill · 权重清单（权重本身不入库；`.gitignore` 覆盖 `*.pt`）

## 一、训练臂权重（本实验自己训练，两臂）

| 臂 | 路径 | 体积（字节） | sha256 | 参数量 | 块类 |
|---|---|---|---|---|---|
| `projfill` | `train/projfill/model.pt` | 57,358,059 | `64cc88ec6c6aadaa725e2e0a3619ddd8afcdc69064df2a35214db2ce17ba5d37` | 14,326,272 | `Block` |
| `projfill_carry` | `train/projfill_carry/model.pt` | 58,420,003 | `9860eb95eee80dcc6351b3268a153cbb90cc0c4fcb2534998c03cd4c0af55b00` | 14,590,568 | `CarryChannelBlock` |

- **哈希口径**：`run.py::load_arm` 逐臂断言 `config.json::checkpoint.sha256`；不一致直接
  `SystemExit`（防「用错权重出数字」）。
- **生成命令**（两臂逐字段一致，唯一变量 = `--carry-write`）：

  ```bash
  # 基准臂（随机退出 + proj-fill，无 carry）
  python3 train_depth_ar.py projfill \
    --d 256 --layers 12 --heads 8 --mlp-mult 6 --seq-len 256 --topology sequential \
    --steps 6000 --batch-size 32 --lr 1e-3 --seed 0 --exit-mode uniform \
    --synth-task assoc-mqar --synth-pairs 8 --synth-queries 8 --synth-gap 14 \
    --proj-fill --final-weight 2.0 \
    --out-root axes/M-carrier/M1-carrier-type/experiments/2629-carry-ragged-projfill/train
  # 目标臂：同上 + --carry-write delta
  ```

  驱动脚本：`train_arms.sh`（按臂取/放 GPU 锁、臂间 65s，§20.1）。
- **权重不入库**：体积与哈希在此登记；需要复现时按上面命令重训（权重来自同一 seed，
  逐次哈希应一致——若不一致，说明实现或库版本变了，必须查）。

## 二、代码指纹（本文件登记；跑离线校验/复核时逐条核对）

| 文件 | sha256 |
|---|---|
| `pathlm/depth_ar.py`（共享引擎，本线**未改**） | `45f946793835f2afadf3a10f727b02ee879682ef6024b23fa179d3351c28c585` |
| `train_depth_ar.py`（共享训练入口，本线**未改**） | `2d1863bbc4403e026d12b597d98a1da2b85f3606110073f2117a7dd1db2a1f35` |
| `ragged_projfill_arms.py`（本线语义层） | `7603549158b34e2199050d4efc94ca0c40e958e82c3e084c3b282cf05e01b308` |
| `run.py`（评测入口） | `49ae02eacd6719dfe9f13e45d09549f22d2267947f9fc43ff995efbfda931a48` |
| `train_arms.sh`（训练驱动） | `ba3ec3053b015625a32d29d1ee89fd11f4897cb470f648af3e9193c781788a01` |
| `reconcile_numbers.py`（对账） | `f09aa6bc299689b388a4fcfab5ef2758874dc0a066ea693720e273101a038411` |
| `mutate.py`（变异验证） | `bbf0578cd559da8cbc349de118f83f57cd0f4c8119e3d8c036ba4eafdb0237ac` |
| `tests/test_carry_ragged_projfill_13m.py` | `7ba46b210dd9b38e406698398a5f55cdddf6e3f4be0296ee48850779a7f13070` |

## 三、共享文件并发写记账（SCHEMA §22）

- 本线**不改**任何共享引擎 / 训练入口文件（`pathlm/depth_ar.py`、`train_depth_ar.py` 的
  sha256 是「读到的版本」，本线未写）。新增逻辑全部放在本实验目录的新模块
  `ragged_projfill_arms.py` + 新单测文件里。
- 上表共享文件哈希是**本线跑评测时**读到的版本；若同波其它线改过这两个文件，哈希会随时间
  变化——本线只声明「我自己没写它们」。
