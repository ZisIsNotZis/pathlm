# 2632 · KV 量化标尺口径重测 —— 权重清单（`MODEL.md`）

契约：同目录 `README.md` + `config.json::parent`（本实验**继承** `2631` 的 checkpoint 定义；
`config.json` 不重复列哈希，运行时由 `run.py` 从父实验 config 载入并由 `2619` 的载入器断言）。
本实验**不训练、不新增参数**（`params_added = 0`）；只读复用他人目录的 checkpoint，权重不入库
（`.gitignore` 覆盖 `*.pt`）。下表哈希由本文件写入时**在磁盘上重算**（`sha256sum`），与
`2631/config.json::checkpoints.*.sha256` 逐位相同。

| 角色 | 路径 | 大小 | sha256（磁盘实测） | 生成命令 |
|---|---|---|---|---|
| trunk 族被测模型（`fp16` / `int8` / `int4` / 窗口族 / 窗口或锚点×量化，共 18 臂） | `axes/H-horizon/H4-head-trunk-relation/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` | 54 193 675 B | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` | `2610`（13.5M dense decoder-only，12k 步，clean bpc 1.483569） |
| head 族全精度基线 MHA | `axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/runs/mha-12k-s0/model.pt` | 54 193 675 B | `557453eb12b56a453b29ea279cdbbc0969cfa6bf0dbe1cdbd9965cd6cec62528` | `2616`：d256/L12/H8/mlp6/seq256/seed0，12k 步 |
| 头共享 GQA（2 个 KV 头） | `.../2616-gqa-mqa-13m/runs/gqa-12k-s0/model.pt` | 49 456 651 B | `bd682447c90bc118253a1496b4241bdc711a4103ab8df06f8296a840e7b3bc00` | 同上（`--n-kv-heads 2`；`gqa` / `gqa_int8` / `gqa_int4` / `gqa_win64` / `gqa_win128`） |
| 头共享 MQA（1 个 KV 头） | `.../2616-gqa-mqa-13m/runs/mqa-12k-s0/model.pt` | 48 667 147 B | `24bdfe7f1549107175e9343444372a6abe3aba7c6ba2423e8ab069a145b1ea97` | 同上（`--n-kv-heads 1`；`mqa` / `mqa_int8` / `mqa_int4`） |

（MHA 只服务 `mha` 一个臂；trunk.pt 服务 18 个臂；18 + 1 + 5 + 3 = 27 = 本实验的臂总数。）

- 头共享族的**注册表**：`axes/W-width/W3-kv-head-sharing/experiments/2616-gqa-mqa-13m/model_hashes.json`。
- 复算命令：`sha256sum <上表四个路径>`（本波实测四条全等）。
- 数字与文件的对应：`results.json::points[*].checkpoint` 记每个臂的 base arm / 模式；
  `results.json::code` 记产生该结果的那一版代码文件 sha256 前 16 位（单一权威：本文件不复列）。
