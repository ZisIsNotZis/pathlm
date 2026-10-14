# WP-2662 报告：config 生成缺陷修复 + 缺臂补训与评测的后台全流程 driver

- 实验目录：`axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/`（预登记契约 = 同目录 `README.md`）
- 角色/写入者：轴 1 WP worker（fresh 上下文，本目录唯一写入者）；lead = `axis1-lead-4`
- 本报告写入时刻：`2026-10-14T22:58+08:00`

## 0 人话摘要（先看这段）

这次修的是一个「把已经训好的模型白扔」的生成脚本缺陷：`make_config.py` 原本只要有一个新臂没训出来，就**整体不生成配置**，
结果盘上的 `config.json` 里连早已训完的锚点臂（窗口 16 + 锚点 8，seed 0）都没有，评测跑起来也测不到它。
现在改成**逐臂独立判定**：存在的臂全部写入（含逐个校验 sha256），只有没训出来的那个记到 `missing_new_arms` 里并告警，退出码为 0。
重新生成的 `config.json` 已是 4 臂 + `missing_new_arms=["win16a_s1"]`，四个 sha256 与磁盘 `sha256sum` 逐个一致。

另外，lead 发现（并由本 WP 用 CPU dry-run 实测证实）**「只跑评测」这条路走不通**：本实验的判据是跨臂统计，
`run.py` 硬点名了锚点臂的两个 seed，缺 `win16a_s1` 时会在算完质量腿和召回腿之后抛 `KeyError: 'win16a_s1'`，0 读数。
所以交付物改成**一个后台全流程 driver**：先补训缺的第二臂，再重生成配置，最后评测。它**已经在后台跑起来**，
此刻（`22:57:19` 起）正拿着 §20 GPU 锁训练 `win16a_s1`，预计 13 分钟左右训完，随后自动做配置回填与评测。

## 1 `make_config.py` 改了什么（语义一句话）

**`--require-new` 从「任一新臂缺失 ⇒ 整体不生成配置并非零退出」改为「逐臂独立判定：所有已存在且可核 sha256 的臂一律写入 `config.json`，
缺失的新臂记入 `missing_new_arms` 并在 stdout 告警（`--require-new` 时同时打到 stderr），退出码 0」**。

落点（现文件行号，`182` 行）：

| 行 | 改动 |
|---|---|
| 11–13（docstring） | 用一句话重写 `--require-new` 语义：逐臂独立判定、不再整体降级 |
| 74–75 | `def build(require_new: bool)` → `def build()`；`new_arms, missing_new = [], []` |
| 81–82 | 原 `raise SystemExit(f"--require-new：… 尚未训练出来")` → `missing_new.append(run); continue`（只记缺的那臂） |
| 137 | `cfg` 新增显式字段 `"missing_new_arms": missing_new`（紧跟 `"arms"`），无缺臂时为空列表 |
| 161–165 | `--require-new` 的 help 改为「告警同时打到 stderr；不再非零退出」；`cfg = build()` |
| 168–176 | 删掉原「对照 5 臂名单报缺臂」的告警，改以 `missing_new_arms` 为唯一告警源；结尾打印「N 臂，缺 M 个新臂」 |

未改：`run.py`（判据线、臂集、阈值一字未动）、`README.md` 预登记判据、`train_arms.sh`、`run_all.sh`、`pathlm/**`、`tools/**`、`tests/**`、`docs/**`、任何 `project.yaml`。

### 1.1 重生成后的 `arms` 与逐臂 sha256（`sha256sum` 独立复核，逐个一致）

| 臂 | kind | window | anchors | seed | path | sha256（全） |
|---|---|---|---|---|---|---|
| `win16a_s0`（**新臂，已训完**） | window | 16 | 8 | 0 | `…/2662-hard-window-anchor-13m/train/win16a_s0/model.pt` | `5a41ed069ee0ba784187b8d2adc054d334ed418c57ba5ff05fbf6347141cdb2d` |
| `win16_s0`（复用 2634） | window | 16 | 0 | 0 | `axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m/train/rt_none_s0/model.pt` | `c3f6f08a4fd1dfe9b5c72cb433fe20a2390052b1eda11b8ab9a6d0fe468deca7` |
| `win16_s1`（复用 2634） | window | 16 | 0 | 1 | `axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m/train/rt_none_s1/model.pt` | `2bfe167c133c496f42864a191f2fc8833f20ba4ec8f9f4f1dceba43dd84fb162` |
| `full`（复用 2610 主干） | full | 256 | 0 | 0 | `axes/6-predict/6.2-speculative-decoding/experiments/2610-frozen-trunk-jumphead-13m/trunk.pt` | `86953dcb7c9089204184dabff6b3f8629a9d8436ad01f9d1d0000d96eec159d2` |

- `missing_new_arms` = `["win16a_s1"]`（重生成时刻，如实；补训完成后 driver 会重跑本脚本使该字段清空）。
- 复现命令：`python3 make_config.py --require-new`（rc=0；四个 sha256 与 `sha256sum` 逐位相同；`size_bytes` 均为 54193675）。
- 缺陷前后的差别可核：修前盘上 `config.json` 只有 3 臂（`win16_s0/win16_s1/full`，无 `win16a_s0`），修后 4 臂（见上表）；
  旧版行为的过程证据在 `driver.log`（`--require-new：…/win16a_s1/model.pt 尚未训练出来`）。

## 2 结构性依赖的实测证据（为什么不能只跑评测）

CPU dry-run（**未取锁、未动 GPU**，`--config` 指向本目录 `armgap_probe/` 下的 4 臂副本、`device=cpu`）：

```
[    4.2s] 锚点召回腿③：needle v3（8×16，8 针/行，seed=3，归属桶 anchors=8）
Traceback (most recent call last):
  File "run.py", line 394, in main
    pa = ns.recall_on_indices(planted_by_arm[a], idx)
KeyError: 'win16a_s1'
```

- 证据文件：`armgap_probe/probe_med_stdout.txt`（rc=1，未落 `results.json`）；同一份 config 的 ①自检 ②质量腿 全部通过
  （`win16a_s0` bpc=1.601191 / `win16_s0`=1.637986 / `win16_s1`=1.612549 / `full`=1.404325，2×4 小批）。
- 结论：缺臂会在 ③ 召回腿汇总处硬崩（`planted_by_arm` 只含 config 里存在的臂），而 ④ 毁源探针与落盘都还没开始
  ⇒ 排到队也 0 读数。故本轮**不改 `run.py`、不缩臂**，改为「先补训缺臂再评测」。

## 3 新 driver：`run_all_missing.sh`（已后台启动）

- 文件：`axes/1-time/1.1.1-hard-window/experiments/2662-hard-window-anchor-13m/run_all_missing.sh`
- 顺序：① 取锁 → 训练 `win16a_s1` → 放锁 → 臂间 61 s；② `make_config.py --require-new` + 校验 `missing_new_arms` 已空、
  逐臂 sha256 复核；③ 取锁 → `python3 run.py` → 放锁。②若仍缺臂 ⇒ **不排评测**（如实登记，不让 `run.py` 白占 GPU 后崩）。
- 训练 CLI 与 `train_arms.sh` 的 `run_arm` 体**逐字段相同**，只换臂名与 `--seed 1`（机械比对：14 个 flag 全部逐项一致，
  差异仅 `--seed 0→1`、`--steps 12000`（`train_arms.sh` 给 `win16a_s0` 传的就是 12000），`--out-root` 同为 `$EXP/train`）。
- 取/放锁照抄 `run_all.sh`；**未启用**「无 holder_pid 且锁龄 >15 min ⇒ 回收」那条 stale 规则（只回收「holder_pid 存在且进程已死」的锁）。
  等锁上限 **90 min**——本 driver 自持锁，不用 `train_arms.sh` 自带的 20 min 上限（该上限正是 `21:51:57` 丢掉 `win16a_s1` 的原因）。
- 启动命令与状态：
  - 启动：`setsid nohup bash run_all_missing.sh > /dev/null 2>&1 < /dev/null &`（新 session，`sid=2592971`）
  - **driver pid = 2592971**；日志 = `…/2662-hard-window-anchor-13m/driver_all.log`
  - 当前状态（`22:58:38` 采样）：**已取得 §20 锁，正在训练 `win16a_s1`**（训练子进程 pid 2604405，
    起于 `22:57:19`，`gpu_free=21091 MiB`，日志 `train_win16a_s1.log`）；训练完成后自动走 ②③。
  - 我实际等锁时长：driver 从 `22:56:19`（DRIVER START）到 `22:57:19` 拿到锁，共 **60 s**（无退让）。

## 4 为绕过缺臂做过的临时改动（必须已回退的项）

**无。`run.py` 一字未改。** 硬证据：`sha256(run.py) = 7061936b2dc3d3359531c1365d8abe0b274d39c2f5faae665064b9ffe7937260`，
`mtime = 2026-10-14 21:00:46`（早于本会话起始 ~`22:32`；本会话只以只读方式运行过它）。未跑任何 `git` 写命令（未 `add`/未 `commit`）。

留痕但不属于实验产出的临时物：

- `armgap_probe/`（本目录内）：CPU dry-run 的两份 config 副本 + 两份 stdout，只作 §2 的证据，**不进判据、不是实验产物**，可随时删。
- `run_eval_only.sh`：原任务的 eval-only driver，已确认结构性不可用（同 §2），文件头已标注**「已作废，勿启动」**，保留仅为留痕。
- 未缩臂、未缩批数（`config.json` 的 bpc/probe 96×16、needle 64×16 与预登记一致；`armgap_probe/` 的小批只用于 CPU dry-run）。

## 5 锁与队列观测（如实登记，未做任何处置）

- 我的 driver 取锁前，检测到重复 driver `/tmp/2662_drive.sh`（同样的 `win16a_s1` 训练）在 `22:56:57` 释放、`22:56:59` 由 lead 停掉；
  lead 的这句记录写在本目录 `driver_all.log` 里（该文件一度被两个 driver 同时写，现只剩我的 driver 写）。
- `22:58:16` 采样 `/tmp/pathlm_gpu.lock`：`holder_pid=2503753`（**该进程已不存在**）、`name` 为空；
  而我的 driver 在 `22:57:19` 已 `mkdir` 到锁并写入 `holder_pid=2592971` ⇒ 锁目录疑似被第三方取走/重建（我未做任何回收、未动锁目录）。
- 同一时刻 GPU 上除我的训练（2604405，2624 MiB）外，还有别线 `run.py`（2605023，4740 MiB，`WP631` 重试循环）在跑 ⇒ 他线并未互斥。
- 处置：**一概不动**（不杀进程、不删锁、不回收）；这两条只作观测上报，队列互斥归 lead 裁决。

## 6 未做项

1. 本 WP **未**写 `analysis.md` / `MODEL.md` / `project.yaml`：`results.json` 要等 driver ③ 跑完才有读数（预计 `23:10` 之后）。
2. 未跑 `pytest` / `check_tree` / `summary --check`（本 WP 只改一个生成脚本 + 新增一个 driver，未触共享底座；且硬边界禁跑 git）。
3. 未验证 `run.py` 全电池的墙钟与显存（要等 driver ③ 的真跑）。
4. `make_config.py` 的 `--require-new` 不再返回非零码：这是「不整体降级」的直接后果（缺臂改由 `missing_new_arms` 承载）；
   若 lead 要「写盘成功但退出码非零」，是一行改动，报告此处先按字面要求（不丢弃已训好的臂）实现。

## 7 下一步（建议，供 lead 决定）

1. 读 `driver_all.log` 的 `TRAIN done win16a_s1 rc=` 与 `EVAL done rc=` 两行判活；`run_log.txt` + `results.json` 落盘后，
   另派 writer 写 `analysis.md`（六问人话摘要 + 数字表）并把 `missing_new_arms` 清空后的 `config.json` 视为终态。
2. 若 `22:57:19` 起那臂训练在他线并发抢占下失败，重跑 `run_all_missing.sh` 即可（脚本幂等：缺臂照训、config 照回填）。
