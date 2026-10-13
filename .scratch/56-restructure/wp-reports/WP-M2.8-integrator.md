# ticket 56 · M2 · WP-M2.8 整合与收尾（integrator）报告

> 角色：worker，fresh 上下文，**单一写入者**（本轮其它 writer 已停）。语言：中文。
> 时间：2026-10-13T12:0x–12:2x+08:00（`date -Iseconds` 实取）。
> 契约：`docs/structure.md` §12/§13/§15、`docs/research-schema.md`、`.scratch/56-restructure/MAPPING.md`。
> 复核输入：`.scratch/56-restructure/wp-reports/rev-axis1.md` `rev-axis2.md` `rev-axis6.md` `rev-rest.md`。
> 最终 revision：`3861ea54dea99bd4d81a1bafb165322bd6593e81`（main）。
> 提交：`3c03d0b`（退役）→ `a79f207`（更正：归档补上未提交修订）→ `3861ea5`（整合）。

---

## 0 人话摘要（先结论）

7 个轴的迁移已经**合成一个一致的最终状态**：新树（`axes/1-time … 9-backlog`，46 个 `project.yaml`）
与 `docs/structure.md` §12 逐字对上，旧树（9 个 `axes/<旧轴>/`）连同旧 wiki 资产
（`catalog.yaml`/`INDEX.md`/3 个生成式工具/1 个旧测试）**整体退役、只读归档进 `archive/wiki-legacy/`**，
**一条研究证据都没丢**（旧证据 1562 个文件 100% 有 rename 新家，其中 1380 个逐字节相同）。

4 份独立复核报告的发现逐条处置：**改了 6 类确属缺陷者**（迁移后证据里的旧家路径、`6.4` 顶层
status 高报、`docs/structure.md` 的 `1.3`/`6.1` 规范自冲突、`*.md.md` 双扩展名、旧引擎悬空引用
`mental_model.md`、`docs/` 的「净亏」措辞分叉）；**3 类按约束不改并登记未决**（`chain_mode` 默认值
属行为变更、role_map 实名顺延 M3、backfill stub 含测量证据故保留）；`rev-rest` 的两条未列项也顺手处置
（`PRIOR_ART` 指针改指归档、`2.3.1↔3.1.2` 边界核对）。

门全绿：`summary.py --check` / `check_tree.py` / `check_schema.py` 均 OK（46 项），
`pytest 815 passed in 92.75s`（827 → 815，差额恰为退役工具的 12 个用例，无其它失败）。

---

## 1 复核发现逐条处置对照

| # | 发现（来源） | 位置 | 判据 / 依据 | 处置 | 证据 |
|---|---|---|---|---|---|
| 1 | **N1**：已迁入证据内部仍写旧家路径（rev-axis2） | `axes/2-depth/**/experiments/**` 及**全树同类** | MAPPING §0.2 整目录搬新家 ⇒ 内部路径指针应对得上新家；只改路径、**不改测量数字** | **全树路径改写**（§2 详述）：324 文件 / 1474 处替换；改写后新树内**非归档**旧轴引用 = 0 | `3861ea5`；`axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e/MODEL.md:32` 等 |
| 2 | **D2**：backfill 旧 schema stub（rev-axis6） | `axes/6-predict/{6.2,6.4}/experiments/{2512,2513,2521,2523e}-*/README.md` | 任务约束「删除前先确认其内容无测量证据；**有证据者保留并说明**」 | **保留**：4 目录各仅 1 份 README，但正文含测量结论（2512「oracle 链 0.682 ≈ node-1 0.693；deploy 0.547 < direct 0.560」、2521 @100M、2523e 填充税…），且被 `6.2`/`6.4` 的 `experiments` 列为证据 ⇒ 不删。登记未决（§6） | `head -20 axes/6-predict/6.2-speculative-decoding/experiments/2512-overnight-causal-chain/README.md` |
| 3 | **D3**：`6.4` 顶层 status 高报（rev-axis6） | `axes/6-predict/6.4-training-fill/project.yaml` | `docs/research-schema.md` §7「status 只回答做完没有」；milestone 要 13M matched-step **n≥2**，而 WP1=进行中、WP2=未开始、`gaps` 自述「填充语义对照无 n≥2 复本」 | `status: 已完成 → 进行中`、`ended → ""`；`conclusion_review` 与 `change_log` 同步 | `git show 3861ea5 -- axes/6-predict/6.4-training-fill/project.yaml` |
| 4 | **D4**：`chain_mode` 默认 `"latent"` 已被否证（rev-axis6） | `pathlm/config.py:83` | 改默认 = **行为变更** ⇒ 任务要求「只报告不擅改」 | **不改**；登记未决（§6） | `grep -n chain_mode pathlm/config.py`（83 / 140） |
| 5 | **D5** / 55-review **R-7**：`docs/` 措辞分叉（rev-axis6） | `docs/methodology.md`（双货币、记账清单）、`docs/playbook.md`（定律 8/11） | `AGENTS.md` §2.1 一事实一家：6.2 已把 R>1 定为「范式定义性质、不得当判据」，`docs/` 仍叫「净亏」 | 「净亏」→「**代价轴 >1**（`τ≤γ` 的范式定义性质，非亏损发现）」；定律 8 去掉「实为口径事故」变更叙事、改为当前事实句 | `docs/methodology.md:39-43,55`；`docs/playbook.md:85-88,114-116` |
| 6 | **N6**：旧引擎悬空引用（rev-axis2） | `pathlm/{decode,config,slider,depth_ar}.py`、`slider.py`、`probe_frontier.py` | 引用 `docs/mental_model.md`，该文件已归档 ⇒ 悬空 | 11 处全部改指 `archive/legacy-docs/mental_model.md`（**仅注释/字符串**，无行为变更） | `grep -rn mental_model --include=*.py .` |
| 7 | **A-R4**：`*.md.md` 双扩展名（rev-axis1） | `.scratch/doc-log/**` 共 83 个 | ticket 56 §2.4 / 验收 5「`*.md.md` 逐条消失」 | 83 个 `git mv X.md.md → X.md`（零命名冲突）；`find . -name '*.md.md'` = 0 | `3861ea5`（83 个 R100 rename） |
| 8 | `1.3` 规范自冲突（rev-axis1 N4） | `docs/structure.md` §12 vs §13.4 | §12 给 `1.3` 列目录且标「参照」，§13.4 说参照只进 `axes/7-references/README.md` | §12 的 `1.3`、`6.1`「目录」列改为台账 `axes/7-references/README.md`（两目录本就不存在，未建）；§12 自检脚本 OK | §5；`python3 /tmp/check_structure_12.py`（一次性自检） |
| 9 | **role_map 实名**（M1 计划，rev-axis1 N2 / rev-axis2 N3） | 全部 46 份 `project.yaml` | 本轮**不臆造实名** | `docs/research-schema.md` §4 注明「实名在轴 lead 指派时填（M3）」；登记未决（§6） | `docs/research-schema.md:83-84` |
| 10 | **rev-rest N1**：新 SSOT 指向待退役旧文件 | `project.yaml` `sota_baseline` 指 `axes/{W-width,G-policy,F-iteration,H-horizon}/PRIOR_ART*.md` | 退役后指针悬空 | 路径改指 `archive/wiki-legacy/...`，「该文件待 integrator 退役」→「已随旧树退役归档」 | `axes/3-width/3.1.1-.../project.yaml:10` 等 |
| 11 | **rev-rest N2**：`5.1`/`3.2.1` 顶层 `已完成` 与未完 WP 并存 | `axes/5-quant/5.1-kv-quant`、`axes/3-width/3.2.1-readout-reparam` | 复核自判「**可辩护取值**」（结论明确、未完 WP 为后续跟进） | **不改**；登记未决（§6），与 `6.4`（主判据未达）情形不同 | 见 §6 |
| 12 | **rev-rest N3**：`2525-tax-attribution` 主家归属 | `3.1.1`（主家）/`3.2.1`（指针） | MAPPING §1 指定落 `3.1.1` | **维持**（符合 MAPPING）；无文件重复 | §5 证据重复扫描 |
| 13 | `2.3.1 ↔ 3.1.2` 边界待 integrator 统一 | `axes/2-depth/2.3.1-layer-capacity`、`axes/3-width/3.1.2-layer-width` | 两项自述「须由 integrator 核一遍表述不矛盾」 | **已核对：不矛盾**（各按自己轴记一面、互指）。两文件 `gap`/`gaps`/`conclusion_review`/`argument` 由「待统一」改为「已核对」 | `git show 3861ea5 -- axes/3-width/3.1.2-layer-width/project.yaml` |

> 未处置/只报告的 3 项全部落在 §6，并已写进 `.scratch/56-restructure/TICKET.md` §7.6。

---

## 2 证据不丢失核验（N1 路径改写 + 证据迁移）

### 2.1 迁移本身（旧 → 新家）

- **口径**：基线 `ce2610a`（迁移前）→ 各轴迁移提交 → `HEAD`。以 `git diff -M --name-status ce2610a HEAD -- axes`
  的 R（rename）判据核「旧格 experiments 的相对子路径在新家存在」。
- **结果**：旧轴 `<格>/experiments/**` 下 **1562** 个 git 跟踪文件（不含自动索引 `experiments/README.md`）
  → **100% 命中 R-rename，rename 目标 100% 落盘存在**（脚本 `not renamed=0 / dest not on disk=0`）。
  其中 **1380 个 R100（逐字节相同）**，182 个相似度 <100（复核已确认差异为路径指针改写，测量数字零改动）。
- **5 个真删除**：`A10/A3/A4/A9/M2` 的自动生成 `experiments/README.md`（`<!-- AUTO -->` 旧 wiki 索引），
  各新家都新增了自己的索引 ⇒ **非研究证据，不计丢失**。
- **旧轴工作树残留**：退役前旧 9 轴只剩 **85 个文件**（旧课题页 `README.md` / `PRIOR_ART*.md` / 旧实验索引），
  **无任何 experiments 证据**（`find axes/<旧轴> -type f` 逐轴核对）。
- `.scratch/**/evidence/**` 原地未动（迁移提交 0 条 `.scratch` 路径改动）；`archive/` 未动。

### 2.2 路径改写（N1，本 WP 新增）

- **映射构造**：由 `ce2610a→HEAD` 的 `git diff -M` 重命名表建「旧前缀 → 新前缀」字典
  （文件级 + `experiments/` 及更深目录级；含 `<旧格> → <新研究项>` 的**实验目录**级；**排除**「旧格目录本身 → 新研究项目录」
  这一有歧义的一级，避免把旧课题页 README 误指到新研究项目录）。另加「旧轴根 → `archive/wiki-legacy/axes/<旧轴>`」
  兜底；对已归档路径（`archive/dropped/…`）加负向断言不误改。按**最长前缀**替换（1 个歧义键
  `axes/A-mixing/A6-mixing-operator/experiments` 被丢弃，其引用回落到归档兜底）。
- **覆盖**：新树 `axes/1-time…9-backlog/**` + `tools/evalkit/*.py` + `SCHEMA.md` + `pathlm/recon_signal.py` + `docs/**`。
- **规模**：**324 个文件 / 1474 处替换**（`.md 215 / .json 72 / .py 30 / .txt 28 / .yaml 14 / .stdout 7 / .sh 4 / .runlog 3 / .sha256 1`）。
- **验证口径（关键）**：把两边文本里「含 `axes/` 的路径 run」一律置为占位符，再比较**数字多重集**
  （含 `.` 的小数也保留）；`324/324` 文件**完全一致** ⇒ 只有路径 token 变化、**测量数字零改动**。
- **收尾检查**：改写后新树内 **非归档** 旧轴路径引用 = `0`
  （正则 `(?<!wiki-legacy/)(?<!dropped/)axes/(A-mixing|…)` 扫描全部新轴文件）。
- 唯一副作用（已处理）：`6-predict/…/evidence/fix_mtp_nodes_T1/red_before_fix.txt` 3 行是 pytest 失败转录
  （`path:line: ` 行尾带空格），路径改写后成为「新增行」而被 `git diff --check` 判为行尾空白；
  已**仅对那 3 行**去掉行尾空格（内容=原字节 − 1 个空格），现 `git diff --check` 全绿。
- **原始转录日志**（`.txt/.stdout/.runlog` 38 文件）的旧路径也一并改写：取其「全树路径可解析一致」；
  若下游偏好保留原始转录字面，可 revert 这 38 个文件（清单见 §7）。

---

## 3 退役清单（`docs/structure.md` §15）

`git mv` 进 `archive/wiki-legacy/`（**只读保留、不删**，共 91 个文件）：

| 类别 | 条目 |
|---|---|
| 旧轴目录（9 个，85 文件） | `axes/A-mixing` `axes/C-vertical` `axes/H-horizon` `axes/W-width` `axes/M-carrier` `axes/F-iteration` `axes/G-policy` `axes/P-modifiers` `axes/R-references` |
| 旧索引 | `catalog.yaml`、`INDEX.md` |
| 生成式 wiki 工具 | `tools/build_wiki.py`、`tools/check_catalog_strict.py`、`tools/backfill_legacy.py` |
| 退役工具的回归测试 | `tests/test_build_wiki_manual.py`（9 个函数 / 12 用例；`build_wiki` 已不在 `tools/`，import 必失败 ⇒ 随工具退役） |

- 退役后：`find axes -type f` **2510 → 2425**（−85）；`git ls-files axes` **1733 → 1648**（−85）。
- **工作树既有未提交修订**（`catalog.yaml`、`INDEX.md`、`axes/C-vertical/{README,C1-layers-per-token/README}.md`）：
  内容已由新树 `2.1.1` 等承接；本轮**随旧树归档**（提交 `a79f207` 把未提交修订补入 `archive/wiki-legacy/` 的对应文件），
  **未回滚、未丢弃**。
- **声明与磁盘一致**：`docs/structure.md` §15、`AGENTS.md` §2.1/§2.3、`SCHEMA.md`（迁移通告 + 4 处「迁移窗口」）、
  `docs/design.md` §4、`docs/research-schema.md` 顶部、`archive/README.md`（新增「已归档的旧 wiki」节）、
  新树各 README / `project.yaml` 的 `sota_baseline` 中的「待 integrator 退役」均已改述为「已归档」。
- **未停止退役的目录：无**（9 个旧轴全部为「实验证据已迁空、只剩旧页/旧索引」）。

---

## 4 门命令真实输出（单独跑，不并发）

```
$ python3 tools/summary.py --check
OK：46 个 project.yaml 全部符合 docs/research-schema.md。            # exit 0
$ python3 tools/check_tree.py
OK：46 个 project.yaml 全部符合 docs/research-schema.md。            # exit 0
$ python3 tools/check_schema.py     # 兼容入口，转调 check_tree
OK：46 个 project.yaml 全部符合 docs/research-schema.md。            # exit 0
$ python3 -m pytest tests/ -q       # revision 3861ea5
815 passed in 92.75s (0:01:32)                                      # exit 0
$ python3 /tmp/check_structure_12.py     # §12 一致性自检（一次性脚本，未入库）
§12 解析行数: 57 / 参照行: ['1.3','6.1'] / §13.4 声明: {'1.3','6.1'} / OK：§12 行与磁盘/§13.4 一致
$ git diff --check                  # 三个提交的暂存内容均 exit 0
```

- **pytest 827 → 815（−12）**：差额 = 退役工具 `tests/test_build_wiki_manual.py` 的 12 个用例
  （9 个 `def test_`，其中 `test_malformed_manual_block_raises` 参数化 4 例）= 12。**无其它失败/错误**，
  故判「无本次迁移引入的失败」而不是「既有失败」。
- 复核提到的并发 pytest 抢资源问题本轮不存在（见 §4.1）；整套耗时 92.75s，与复核的空载 94–99s 同量级。

### 4.1 环境清理（上一轮 4 份复核的 orphan pytest / GPU）

| 检查 | 清理前 | 清理后 |
|---|---|---|
| `pgrep -af pytest` / `ps -eo pid,cmd \| grep pytest` | **无任何 pytest 进程**（只有 grep 自匹配） | 无 |
| `nvidia-smi` 计算进程 | `llama-server` PID 145968（3908 MiB）、PID 4061254（14666 MiB） | 同上（**未动**） |
| `nvidia-smi` 显存/利用率 | 19228 MiB / 89 % | 19225 MiB / 93 % |

⇒ **本轮无需 kill**：仓库迁移复核的 pytest 已自行退出；GPU 占用者是与本仓**无关**的
`llama-server`（他人进程），按约束**未误杀**。

---

## 5 跨轴一致性结论

| 检查 | 方法 | 结果 |
|---|---|---|
| `id` 全局唯一 | `tools/check_tree.py` | ✅ 46 个 id 无重复 |
| 用词与 §12 逐字一致 | §12 解析脚本比对 `id/title/dir`（55 行节点 + 参照行） | ✅ 0 处不一致（`title`/`dir` 逐字相等） |
| §12 与 §13.4 自洽 | 同上 + 文本断言 | ✅ 参照行 = {`1.3`,`6.1`}，与 §13.4 声明相等 |
| 证据不重复（一格一份家） | 扫全部 `project.yaml` 的 `experiments` 路径 | ✅ 仅 6 条「多研究项同列」：5 条是**共享 `.scratch` 证据锚**（MAPPING §0.4 允许的指针），1 条是 `5.6` 以**指针**引用 `1.1.4.2` 的 `2614…/analysis.md`；**无物理复制** |
| `experiments` 指针无悬空 | `check_tree.py`（路径存在性 + `@sha` 可达） | ✅ 全过 |
| 未覆盖的 experiment 目录 | 扫 `experiments/*` 与 `project.yaml` 差集 | ✅ 仅 5 个：`7-references/experiments/{A2,A8,W3,R7}` 与 `8-engineering/experiments/2523b-…`，由**轴 7/轴 8 台账 README** 登记（无 `project.yaml` 轴，属预期） |
| 根 `INDEX.md` / 旧 `axes/*/README.md` 不再当 SSOT | 扫描新树 | ✅ 旧轴引用（非归档）= 0；`INDEX.md` 已退役。`catalog.yaml` 仅出现在**迁入实验的历史文档**里作过程叙述（如「应同步进 `catalog.yaml`」），不是路径、也不主张现行 SSOT |
| `docs/structure.md` §15 与 `AGENTS.md` §2.3 与磁盘一致 | 人工 + 上述扫描 | ✅ 均改述为「已退役归档于 `archive/wiki-legacy/`」，磁盘 `archive/wiki-legacy/` 存在 91 文件 |

---

## 6 未决（需授权 / 他人接手）

1. **role_map 实名**：46 份 `project.yaml` 的 owner/writer/reviewer/integrator 仍为角色占位；
   按 M1 计划「M2 派发时填实名」未执行，顺延 **M3（轴 lead 指派时）**（`docs/research-schema.md` §4 已注明）。
2. **`pathlm/config.py` `chain_mode` 默认 `"latent"`**：默认选的是已被否证的旧 `T∘T`；改默认 = **行为变更**，
   本 WP **只报告不擅改**，需产品/引擎 owner 定夺（改默认 or 代码标死）。
3. **backfill 旧 schema stub（4 个）**：`6-predict/{6.2,6.4}/experiments/{2512,2513,2521,2523e}-*/README.md`
   含旧 schema front-matter 与旧轴 id，但正文含测量结论 ⇒ 依「有证据者保留」未删；要清须先把结论迁入新契约。
4. **`5.1` / `3.2.1` 顶层 `已完成` 与未完 WP 并存**：复核判「可辩护」，本轮未改；若要统一口径，
   须先定「`status` 与 `work_packages.status` 的对应规则」。
5. **0.014 噪声带的跨轴身份**（55-review P6 记其来源自否）：仍被多轴当可排序阈值；属**框架性取舍**，非 integrator 自决。
6. **原始转录日志路径改写**：`axes/**` 内 38 个 `.txt/.stdout/.runlog` 的旧路径已一并改写为当前路径；
   若下游偏好保留原始转录字面，可 revert 清单见 §7。
7. **`.scratch/54-subproject-schema/` 与 `.scratch/55-review/`**：本仓既有**未跟踪**目录（ticket 54/55 的台账），
   与本 WP 无关，**未纳入提交、保留原状**（连 `commit -a` 也未碰）。

---

## 7 附：可 revert 的转录日志清单（未决 6）

```
axes/1-time/1.1.1-hard-window/experiments/2623-longT-1024/train_all_log.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2616-window192-trained-13m/run_log.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2616-window192-trained-13m/smoke/run_log.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2618-carry-channel-13m/run_log.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2620-capacity-converged/run_log.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2621-carry-under-compression/run_stdout.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2621-carry-under-compression/smoke/smoke_stdout.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2623-composable-arms-13m/run_stdout.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2630-tf4-samecaliber/train_arms.runlog
axes/1-time/1.1.4.1-explicit-memory/experiments/2631-carry-reseed/run_log.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2633-carry-seed2/run_log.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2633-carry-seed2/smoke/log.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2634-carry-realtext-13m/run_log.txt
axes/1-time/1.1.4.1-explicit-memory/experiments/2635-recursive-carrier-13m/run_log.txt
axes/1-time/1.1.4.2-compression/experiments/2614-kv-ablation-retention-13m/run_log.txt
axes/1-time/1.1.4.2-compression/experiments/2615-retention-retrieval-leg-13m/run_log.txt
axes/1-time/1.1.4.3-compression-trigger/experiments/2615-priority-shootout-13m/run_log.txt
axes/1-time/1.1.4.5-implicit-vs-explicit/experiments/2615-window-trained-retention-13m/run_log.txt
axes/1-time/1.2.2-block-sparse/experiments/2616-linear-attention-13m/run_log.txt
axes/1-time/1.2.2-block-sparse/experiments/2616-linear-attention-13m/run_log_assembly.txt
axes/5-quant/5.1-kv-quant/experiments/2614-kv-quant-eval-13m/run_log.txt
axes/5-quant/5.1-kv-quant/experiments/2619-kv-budget-chart/run_log.txt
axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal/charts_only_rerun.runlog
axes/5-quant/5.1-kv-quant/experiments/2631-kv-budget-causal/run_log.txt
axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/run_log.txt
axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/eval.runlog
axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/evidence/fix_mtp_nodes_T1/red_before_fix.txt
axes/6-predict/6.2-speculative-decoding/experiments/2631-spec-three-arms/run_log.txt
axes/7-references/experiments/R7-existing-early-exit/2623-standard-early-exit-13m/run_log.txt
axes/7-references/experiments/W3-kv-head-sharing/2616-gqa-mqa-13m/run_log.txt
axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m/run_log.txt
axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m/smoke/smoke_A.stdout
axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m/smoke/smoke_A_off.stdout
axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m/smoke/smoke_B.stdout
axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m/smoke/smoke_Brep.stdout
axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m/train/A.stdout
axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m/train/B.stdout
axes/9-backlog/9.2-iteration-refinement/experiments/2619-reconstruction-loss-13m/train/B_rep.stdout
```

---

## 8 提交与工作树

| 提交 | 内容 | 规模 |
|---|---|---|
| `3c03d0b` | 退役旧树：9 旧轴 + `catalog.yaml` + `INDEX.md` + 3 工具 + 1 旧测试 → `archive/wiki-legacy/` | 91 rename，0 增删 |
| `a79f207` | **更正提交**：上一条的 rename 未带上工作树未提交修订（`git mv` 只暂存 HEAD blob），本条把 4 个文件的未提交修订补入归档 | 4 文件，+24/−14 |
| `3861ea5` | 复核处置 + N1 全树路径改写 + 退役声明一致性 + 工单 §7 + 11 份报告入库 | 388 文件，+3447/−1443 |

- 暂存区最终为空；除既有未跟踪的 `.scratch/54-subproject-schema/`、`.scratch/55-review/` 外工作树干净。
- 11 份 WP/复核报告已**原样入库**到 `.scratch/56-restructure/wp-reports/`（`diff -r` 与 artifact 目录逐字节一致）。
- 工单：`.scratch/56-restructure/TICKET.md` §7（过程、门输出、未决）。
