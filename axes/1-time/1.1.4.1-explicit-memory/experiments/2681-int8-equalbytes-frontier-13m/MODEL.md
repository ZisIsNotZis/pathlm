# MODEL · 2681 int8 档下补 k=60 / k=64 两条纯窗臂（13M）

> 权重（`*.pt`）不入库（`.gitignore` 屏蔽）；本文件给出 sha256、体积与生成命令。
> 13 臂 = 11 个**复用臂**（eval-only，来源见下）+ 2 个**本实验新训练臂**（`win60` / `win64`）。

| 臂 | kind | 窗口 | seed | 是否量化档 | 来源 | sha256 | 体积(B) | 校验 |
|---|---|---|---|---|---|---|---|---|
| `win16_s0` | window | 16 | 0 | fp16 + int8 | 复用（eval-only） | `c3f6f08a4fd1dfe9b5c72cb433fe20a2390052b1eda11b8ab9a6d0fe468deca7` | 54193675 | OK |
| `win16_s1` | window | 16 | 1 | fp16 + int8 | 复用（eval-only） | `2bfe167c133c496f42864a191f2fc8833f20ba4ec8f9f4f1dceba43dd84fb162` | 54193675 | OK |
| `win17_s0` | window | 17 | 0 | fp16 + int8 | 复用（eval-only） | `70f0acb92067adba4f796612bab49ead7153c1b4b7e5bb3ea15a5219ab9cdc1c` | 54193675 | OK |
| `win18_s0` | window | 18 | 0 | fp16 + int8 | 复用（eval-only） | `39316ac04f20e158a6833323888990d149bf22c8da76ca04c4aa642cdcfa5913` | 54193675 | OK |
| `win24_s0` | window | 24 | 0 | fp16 + int8 | 复用（eval-only） | `ebd4ef84736355bc163479e702c34b1383708f90d39510756e38b2cf3fcf90d1` | 54193675 | OK |
| `win32_s0` | window | 32 | 0 | fp16 + int8 | 复用（eval-only） | `dccc58c0065d70f0c202424b7f5f40e28929ffb238db3d3a3954d3095e43287b` | 54193675 | OK |
| `win128_s0` | window | 128 | 0 | fp16 + int8 | 复用（eval-only） | `c17a1cf1747ca59e426eeb1378c90e50802e21401d7058e703a766e8cbe287f7` | 54193675 | OK |
| `carrypl16_s0` | carry | 16 | 0 | 仅 fp16 | 复用（eval-only） | `382a74aa0e417b229910cc44f55dcc949c225b3211434df488ee6a72d3036ec3` | 55264963 | OK |
| `carrypl16d_s0` | carry | 16 | 0 | 仅 fp16 | 复用（eval-only） | `cb1940ee2e97f2311998710870be0c86024ed7d65e2ce54c159fcffbca8fe386` | 55264963 | OK |
| `carry16_s0` | carry | 16 | 0 | 仅 fp16 | 复用（eval-only） | `d92114822c09f2f7538ea6a59802eed21b1297409fe91f66c031dec9d6c305eb` | 55255619 | OK |
| `carry16_s1` | carry | 16 | 1 | 仅 fp16 | 复用（eval-only） | `b3d3cdbc57266109b6b461d18c2188c7d1712170f6760b35a48ff9b6170d6843` | 55255619 | OK |
| `win60_s0` | window | 60 | 0 | fp16 + int8 | **本实验训练** | `b82ed96f08780bad6e759b694a6c8be1f7b022fd8705794f0ac55730302cf1fb` | 54193675 | OK |
| `win64_s0` | window | 64 | 0 | fp16 + int8 | **本实验训练** | `dc5af3e0a3fac881b8f2950d67da3b5211c13271fa820f2f0a92d475f0db402e` | 54193675 | OK |

`sha256` 由 `run.py` 加载时逐臂计算并写进 `results.json::checkpoints[*].sha256`；
复用臂的 sha256 与 `config.json::arms[*].sha256` 逐位相等（不等即拒绝启动）。

## 生成命令（本实验新训练臂）

```bash
bash axes/1-time/1.1.4.1-explicit-memory/experiments/2681-int8-equalbytes-frontier-13m/train_arms.sh
# ⇒ train/win60/model.pt、train/win64/model.pt（各 12k 步 / bs32 / lr 1e-3 / seed 0 / anchors=0）
```
