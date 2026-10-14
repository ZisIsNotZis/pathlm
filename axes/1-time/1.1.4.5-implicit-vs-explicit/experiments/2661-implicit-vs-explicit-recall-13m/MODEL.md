# MODEL.md · 2661-implicit-vs-explicit-recall-13m · 权重清单与脚本指纹

**本实验只做推理，不训练**：全部 6 个权重来自既有实验目录（`2634` / `2639`），
`results.json` 的 `checkpoints` 块在加载时用 sha256 **逐臂断言**；下面的清单与断言值逐位一致。
权重文件 `*.pt` **不入库**（体积 52–55 MB/个，属可重训练产物）；本文件记 sha256 与体积。

## 1. 权重清单（6/6）

| 臂 | 来源 | 机制 | seed | 参数量 | 体积(B) | sha256 |
|---|---|---|---|---|---|---|
| `win16_s0` | `2634-carry-realtext-13m/train/rt_none_s0/model.pt` | 纯窗口 16（implicit） | 0 | 13,536,768 | 54,193,675 | `c3f6f08a4fd1dfe9b5c72cb433fe20a2390052b1eda11b8ab9a6d0fe468deca7` |
| `win16_s1` | `2634-carry-realtext-13m/train/rt_none_s1/model.pt` | 纯窗口 16（implicit） | 1 | 13,536,768 | 54,193,675 | `2bfe167c133c496f42864a191f2fc8833f20ba4ec8f9f4f1dceba43dd84fb162` |
| `carry16_s0` | `2634-carry-realtext-13m/train/rt_delta_s0/model.pt` | 显式（退化：单共享槽 delta，chunk 64） | 0 | 13,801,064 | 55,255,619 | `d92114822c09f2f7538ea6a59802eed21b1297409fe91f66c031dec9d6c305eb` |
| `carry16_s1` | `2634-carry-realtext-13m/train/rt_delta_s1/model.pt` | 显式（退化：单共享槽 delta，chunk 64） | 1 | 13,801,064 | 55,255,619 | `b3d3cdbc57266109b6b461d18c2188c7d1712170f6760b35a48ff9b6170d6843` |
| `carrypl16_s0` | `2639-carry-perlayer-frontier-13m/train/pl16/model.pt` | **显式（完备）**：逐层独立状态 + 内容写门，γ=1，chunk 256 | 0 | 13,801,064 | 55,264,963 | `382a74aa0e417b229910cc44f55dcc949c225b3211434df488ee6a72d3036ec3` |
| `carrypl16d_s0` | `2639-carry-perlayer-frontier-13m/train/pl16d/model.pt` | 显式（完备）+ 衰减 γ=0.99，chunk 256 | 0 | 13,801,064 | 55,264,963 | `cb1940ee2e97f2311998710870be0c86024ed7d65e2ce54c159fcffbca8fe386` |

每臂的 `train_log.jsonl` / `results.json`（训练侧活性量）随权重在来源目录内，`config.json`
的 `train_log` / `train_results_json` 指向它们；`run.py` 只读这些旁证，不重训。

## 2. 脚本指纹（sha256）

| 文件 | sha256 | 说明 |
|---|---|---|
| `run.py` | `941a3ab245be560d178cbce83770f597006c2bd3deb5d1e37d36622dbf1b530a` | 本目录评测入口（fork 自 `2634/run.py`，改臂集与逐臂配置） |
| `config.json` | `5a6856cb6f108941f3a31469bd4458490034f04e7bfdea2a8aee675393284cfb` | 本目录配置（含逐臂 path+sha256 + 预登记记录值 + 带） |
| `realtext_probe.py` | `828225b37f665b01809a2c1014c6bb10db795c0dadffeb5485c578e5ee9df2fb` | 仪器层，自 `2634` **逐字节复制（不改）** |
| `needle_stats.py` | `cd11bb7cdb3b3e60534f102b2df3d396a1a71b8ab0b471776d62b87584d33d4c` | needle 统计层，自 `2634` **逐字节复制（不改）** |

以上 4 个哈希与 `results.json.code_hashes` 逐位一致（`run.py` 落盘时自算）；`run.py` 另记
`pathlm/depth_ar.py`、`tools/evalkit/{retention,carry_channel}.py`、`pathlm/data.py` 的哈希。
