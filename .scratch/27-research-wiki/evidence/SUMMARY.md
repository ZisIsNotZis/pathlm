# SUMMARY — ticket 27 · Research Wiki 改造

日期：2026-09-30 ｜ 范围：目录/文档树改造 + 已有研究回填（**未动 Python 代码**）

## 判据验收

| # | 判据 | 结果 | 证据 |
|---|---|---|---|
| J1 | 目录完备：catalog 枚举的每格都有实体目录与 README.md | **PASS** | `check_schema.py`：51 格全部有页 |
| J2 | 模板完备：九节齐全；judged 必须有 judge+evidence | **PASS** | `check_schema.py` 退出 0；25 格已判 |
| J3 | 索引自动、重跑幂等 | **PASS** | `build_wiki.py --check` 干净（0 文件需更新） |
| J4 | 旧内容可回溯：evidence 路径与 commit 锚点真实 | **PASS** | `check_schema.py` 校验全部 evidence 路径存在（含 `archive/legacy-docs/*` 与 `.scratch/*`） |
| J5 | 不破坏现有工程：129 单测全绿 | **PASS** | `pytest tests/ -q` → **129 passed**（102s） |
| J6 | 事实一致：回填数字与归档逐条一致 | **PASS** | 38 个关键数字抽查 37 显式命中 + `−0.007`（Unicode 负号）命中 = **38/38** |

## 状态统计（状态灯）

| 状态 | 格数 | 含义 |
|---|---|---|
| ✅ judged | **25** | 有本地数据 + 判据 + 证据 |
| ◐ partial | 11 | 覆盖了部分选项 |
| ⬜ stub | 8 | 无本地数据（须有预测与判决实验设计） |
| 🏭 reference | 7 | 参照线（必打败的基线，行业已有） |
| 合计 | **51** | 9 个轴目录（A/C/H/W/F/M/G/R/P）+ 51 个格 |

## 交付物清单

**契约层**：`SCHEMA.md`（三级完备 + front-matter + 九节 + 实验目录契约 + scope 声明）、
`CONTRIBUTING.md`（五种贡献 + 7 条硬规则 + 三种审计模板）、`README.md`（门面，重写）、
`AGENTS.md`（更新为新真相来源）。

**目录层**：`catalog.yaml`（51 格枚举 + 逐格本地证据/失败路径/行业口径/依赖）、
`axes/**`（9 个轴 README + 51 个课题页 + 51 个 experiments 目录）、`INDEX.md`（状态灯，自动）。

**方法层**：`docs/design.md`（RAP 设计 + 三条共享底座 + 仓库布局 + 代码迁移计划）、
`docs/methodology.md`（天平：三列账本/方差界/口径/双货币/冒烟+变异/三种审计模板/提交纪律）、
`docs/playbook.md`（9 条已测定律，每条带出处与边界）。

**工具层**：`tools/build_wiki.py`（catalog → 页面/轴页/INDEX，marker 注入保留手写块）、
`tools/check_schema.py`（完备/模板/证据锚点/孤儿目录/生成一致性）、
`tools/backfill_legacy.py`（历史实验映射渲染）、`.github/workflows/wiki.yml`（CI）。

**回填层**：`archive/legacy-experiments.yaml` → **28 个实验条目**渲染到各格子
`experiments/` 下（覆盖 25 个 ticket 的主要判决）。

**归档层**：`archive/README.md`（旧口径四大陷阱警告）、`archive/legacy-docs/`
（11 个旧文档 `git mv` 保留历史）、`.scratch/README.md`（历史工单区说明）。

## 回填映射（ticket → 主格）

| ticket | 主格 | 关键结论（回填） |
|---|---|---|
| 02 M0 | A2 | 校准 PASS / 集成 +0.0008~+0.0015 / direct retry FAIL |
| 03 M1 引擎 | M1 | 27 测试；缓存 bug 修复 |
| 04 M1 runs | A2 | B0 1.5074；R1 −4.6pp |
| 05 followup | A3 | needle v2 窗口内 84.9 / beyond 0.4 |
| 06 M2 | C2 | 元素税定律（shuffle +0.545 … redo +0.044） |
| 07 X1/X3 | A4 | 距离惩罚 −0.0068；假保留陷阱 |
| 08 TTS | G5 | 增益 ∝ 多样性；确定性路径得 0 |
| 09 M3 | C2 | IX1 次可加；shuffle 毒药；bf16 发散 |
| 10 gap | H1 | B2 1.5481 |
| 11 retry 有效性 | F2 | **撤回 6 cell 排序** |
| 12 overnight | H3 | 条件链 deploy 0.547 < 0.560；gauge 修复 |
| 13 四实验 | H5 | Medusa 1.68–1.90×；豁免三路全败 |
| 14 allocator | F3 | **选择性成立** +0.128 vs +0.057 |
| 15 Slider | G1 | 严格占优点 0.635 fwd/tok |
| 16/17/19 规模 | A2/G1 | 税 +0.126 → 渐近 +0.082；门控杠杆消失 |
| 18/21 exit×retry | C6/H4 | 两全证伪；probe 零税 @100M |
| 20 spec 规模 | H1 | 墙钟收益消失；fp 平局翻转 |
| 22 生成质量 | G3 | 三信号监测栈（AUC 0.9815） |
| 23/23b–e | C1/P5/C5/H2 | 四性质成立；w 降权全败；fill 胜；proj-fill 最优 |
| 24/24b | C6 | 税 +0.4563 → 干净归属形态 +0.44 |
| 25 归因 | W1 | 结构切分 −50.4%；税零收缩；**「等智商免费加速」证伪** |

## 未做（明确记录）

- **Python 代码未搬迁**（`pathlm/` / `tests/` / `train_*.py` / `probe_*.py` / `configs/` 原地保留）。
  理由：代码重构与内容重构同时做会破坏历史复现命令与测试；迁移计划见 `docs/design.md` §5。
- **`docs/artifacts.md` 的 25-ticket 证据索引未逐条搬进新结构**：改由
  `archive/legacy-experiments.yaml` + 各格 `experiments/` 承担，原文件仍在 archive。
- **`axes/**` 的手写块全部为空**（页面内容由 catalog 渲染）；如需长篇叙述，
  写在 `<!-- MANUAL:begin/end -->` 之间。

## 复现

```bash
python3 tools/build_wiki.py --check && python3 tools/backfill_legacy.py --check
python3 tools/check_schema.py
python3 -m pytest tests/ -q
```
