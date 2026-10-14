# WP-2655 · 2.3.1 逐层容量 schedule 最小判决 · 收口报告

> 过程工单（`ticket 61 · 轴 2 深度轴`）。事实文档 = `axes/2-depth/2.3.1-layer-capacity/project.yaml`
> + `experiments/2655-layer-capacity-13m/`。writer = WP-2655 收口 writer（fresh 上下文，唯一写入者）。

## 0 人话摘要

普通 transformer 每层 FFN 一样宽。本实验问：在**总参数、每 token 算力都逐位不变**的前提下，
把 FFN 隐藏宽沿深度重新分配能不能把质量（clean bpc，越低越好）拉好。三臂都训完了：

- **前大后小**（浅层宽 1792 → 深层窄 512）：bpc **1.5431**，比 uniform（**1.5291**）**差 0.0140**，
  正好压在噪声带边界上；逐窗 5/5 都比 uniform 差，方向一致。
- **逐渐放大**（512 → 1792）：bpc **1.5309**，比 uniform 差 0.0018，远在带宽内 ⇒ **分不出**。

两种非均匀分配都**没有比「每层一样宽」更好**（一种略差、一种不可分）。所以「按深度更聪明地
分配有限容量能免费换质量」这个假设，在本尺度（8 层 / d256 / 12k 步 / 仅最终深度 CE 监督）
**没有得到本地证据支持** ⇒ 本项判 **被否证**（预登记否证条件成立）。

## 1 判据与读数（P1–P3）

符号：`Δ = bpc_uniform − bpc_该臂`，**正 = 该臂更好**；噪声带 0.014（限配对 Δ、≥5 窗口）。

| 臂 | clean bpc | Δ（4dp / 全精度） | 逐窗配对 Δ 均值 | 窗间 SE | t | 判词 |
|---|---|---|---|---|---|---|
| `uniform` | 1.5291 | — | — | — | — | 对照（stock） |
| `front-big` | 1.5431 | −0.0140 / −0.0140016 | −0.014002 | 0.003062 | −4.57 | **可排序更差（边界）** |
| `growing` | 1.5309 | −0.0018 / −0.0018288 | −0.001829 | 0.001794 | −1.02 | **不可排序（H0）** |

- 全精度：`uniform` clean_bpc = 1.5290984、`front-big` 1.5431000、`growing` 1.5309272。
- `front-big` 边界：全精度配对窗均 Δ = −0.0140016，`|Δ|` 名义超带 1.6×10⁻⁶（4dp 报数恰为
  −0.0140 = 带界）；5/5 窗口配对 Δ 全负。按预登记 `Δ ≤ −0.014 ⇒ 可分更差` 判「可排序更差」；
  即便按更保守的 `|Δ| ≤ 0.014 ⇒ 不可排序` 读法，也**得不到正面增益**——H1 在任一读法下都被否定。
- P3 逐窗配对 Δ（`uniform − 该臂`，同一 generator seed 同一批序，5 窗口 × 4 批）：
  - `front-big`：[−0.011519, −0.015335, −0.021190, −0.018388, −0.003576]
  - `growing`：[+0.000331, +0.001921, −0.005843, −0.006500, +0.000947]

## 2 三道门（P4–P6，全过）

| 门 | 结果 |
|---|---|
| **P4 等 FFN 预算** | 三臂 Σh = 8192、FFN 参数 4 204 544、总参数 6 962 688、每 token FFN MACs 4 194 304 **逐位相等**；解析式（`Σ(2·h·d+h+d)` / `Σ2·h·d`）与实测互核一致。`results.json:P4.pass=true` |
| **P5 引擎等值** | 显式 uniform schedule == 引擎缺省（`--mlp-mult 4`）：state_dict + 前向**逐位一致**（带必红探针，`smoke_report.json:A`）；训练层旁证：`uniform` 与 2613 sequential-s0 差 +0.0037、与 2641 U4-s0 差 −0.0023（均 <0.005）。`results.json:P5.pass=true` |
| **P6 活性** | 逐层隐藏维 == schedule（三臂 true）；逐层 ‖FFN_out‖/‖FFN_in‖ 全 >0：uniform 0.484–2.014、front-big 0.375–2.410、growing 0.442–1.875 |

## 3 交付物

| 文件 | 状态 |
|---|---|
| `README.md`（预登记契约，P1–P6） | ✅ 判据线未改 |
| `config.json` / `run.py`（`--arms`/`--no-train`/`--aggregate`）/ `smoke.py` + `smoke_report.json` / `test_2655_layer_capacity.py`（8 passed）/ `gpu_lock.py` | ✅ |
| `results_uniform_s0.json` / `results_front-big_s0.json` / `results_growing_s0.json`（逐臂增量） | ✅ 三臂齐全 |
| `results.json`（汇总 P1–P6） | ✅ 三臂，`P4.pass=true`、`P5.pass=true` |
| `analysis.md`（人话摘要 + 三臂表 + 配对 Δ/SE + 三道门 + §12/§13 + C2 两读法 + 跨轴旁证）/ `MODEL.md`（三臂 sha256） | ✅ |
| `runs/<arm>-s0-12k/model.pt`（三臂权重，不入库） | ✅ 见 `MODEL.md` 哈希 |

## 4 命令与真实输出

```text
$ date -Iseconds
2026-10-14T22:55:05+08:00

$ # 三臂完整性（results.json 的 generated_at）
as of 2026-10-14T22:53:04+08:00；arms = {uniform-s0: 1.5291, front-big-s0: 1.5431, growing-s0: 1.5309}

$ python3 -c "import json; d=json.load(open('results.json')); print(d['P4_equal_ffn_budget_check']['pass'], d['P5_engine_equivalence_gate']['pass'])"
True True

$ python3 -c "import json,statistics,math
W={a:json.load(open(f'results_{a}_s0.json'))['eval']['windows']['window_bpc'] for a in ['uniform','front-big','growing']}
for a in ['front-big','growing']:
    d=[u-v for u,v in zip(W['uniform'],W[a])]
    print(a,'mean',round(statistics.mean(d),7),'se',round(statistics.stdev(d)/math.sqrt(5),7))"
front-big mean -0.0140016 se 0.0030623
growing mean -0.0018288 se 0.0017939
```

`run_log.txt`（本窗口）：
```text
{"arm": "uniform",  "status": "ok", "eval": {"clean_bpc": 1.5291}}
{"arm": "front-big","status": "ok", "eval": {"clean_bpc": 1.5431}}
{"arm": "growing",  "status": "ok", "eval": {"clean_bpc": 1.5309}}
```
（此前两次 `gpu_lock_timeout` = 等锁超时、**未消耗 token**，非失败；最终三臂均 `status=ok`。）

## 5 残余缺口（诚实登记）

- **`hourglass`（中段大）臂未列**：本 WP 只做前大后小 / 逐渐放大两种非均匀形态。
- **只重分配 FFN 隐藏宽**：逐层 `d_model` / 注意力头数逐位不变（后者属 `2.3.2` 轴）。
- **n=1（seed=0）**：判词只到「可排序 / 不可排序」二分，不支撑 trend；seed=1 未跑。
- **不外推**：只在本尺度（8 层 / d256 / 12k / 仅最终深度 CE）内判读。
- **跨轴旁证**：`3.1.4 / 2641`（同引擎同配方、等 FFN 预算、n=2）同得 H0；「跨轴证据能否作本项
  终局判据 / 2.3.1 与 3.1.4 是否同题」属框架性取舍，**已上报 Main 待裁决**，本目录不自行合并或改名。

## 6 复核

**独立复核未做**（`SCHEMA.md` §14/§14b：复核排在 writer 停止之后、须 fresh 上下文）。writer 自我
核对：所有报告数字都有 `results_<arm>_s0.json` / `results.json` / `smoke_report.json` / `run_log.txt`
可回溯；判据线与 README 预登记原文逐条一致，未改；未跑出的项一律标「未测 / 未列」。
