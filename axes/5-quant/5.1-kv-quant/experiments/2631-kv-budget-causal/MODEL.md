# 2631 · 因果口径 KV 预算图 —— 权重清单（`MODEL.md`）

契约：同目录 `README.md`（补录）+ `config.json::checkpoints`。
本实验**不训练、不新增参数**（`params_added = 0`）；只读复用他人目录的 checkpoint，权重不入库
（`.gitignore` 覆盖 `*.pt`）。下表哈希由本文件写入时**在磁盘上重算**（`sha256sum`），与
`config.json::checkpoints.*.sha256` 逐位相同；`run.py` 在载入时断言同一哈希。

| 角色 | 路径 | 大小 | sha256（磁盘实测 = config 记录） | 生成命令 |
|---|---|---|---|---|
| trunk 族被测模型（fp16 / 量化 / 窗口 / 窗口+锚点 / 窗口或锚点×量化，共 18 臂） | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` | 54 193 675 B | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` | `2610`（13.5M dense decoder-only，12k 步，clean bpc 1.483569）；训练细节见该目录 `MODEL.md` |
| head 族全精度基线 MHA | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/runs/mha-12k-s0/model.pt` | 54 193 675 B | `557453eb12b56a453b29ea279cdbbc0969cfa6bf0dbe1cdbd9965cd6cec62528` | `2616`：d256/L12/H8/mlp6/seq256/seed0，12k 步（`runs/mha-12k-s0/cmd.txt`） |
| 头共享 GQA（2 个 KV 头） | `.../2616-gqa-mqa-13m/runs/gqa-12k-s0/model.pt` | 54 193 675 B | `bd682447c90bc118253a1496b4241bdc711a4103ab8df06f8296a840e7b3bc00` | 同上（`--n-kv-heads 2`；5 臂：gqa / gqa_int8 / gqa_int4 / gqa_win64 / gqa_win128） |
| 头共享 MQA（1 个 KV 头） | `.../2616-gqa-mqa-13m/runs/mqa-12k-s0/model.pt` | 54 193 675 B | `24bdfe7f1549107175e9343444372a6abe3aba7c6ba2423e8ab069a145b1ea97` | 同上（`--n-kv-heads 1`；3 臂：mqa / mqa_int8 / mqa_int4） |

（MHA 只服务 `mha` 一个臂；trunk.pt 服务 18 个臂；18 + 1 + 5 + 3 = 27 = 本实验的臂总数。）

- 头共享族的**注册表**（含各臂配方与哈希）：`axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/model_hashes.json`。
- 复算命令：`sha256sum <上表四个路径>`（本波实测四条全等；`gqa-12k-s0-rep` 是另一条重复 run，
  哈希 `7fa95a7b…`，本实验**未**使用它）。
- 头共享族的参数量混杂（−8.75% / −10.25%）未解，已在 `analysis.md` §8 记为局限；
  本实验只在这些冻结权重上做**推理期存储精度/可见性**的测量，不据此宣称结构优劣。
- 数字与文件的对应：`results.json::points[*].checkpoint` 记每个臂的 run 名与哈希；
  `results.json::code` 记本次参与计算的代码文件 sha256 前 16 位（含本波新增的 `budget_caliber.py`）。
