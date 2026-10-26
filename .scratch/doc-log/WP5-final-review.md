# WP5 · 元级分层收身工程 · 最终 fresh 复核报告

人话总览：派发方没有自说自话——元级四条机检当场重跑全绿，`pytest` 当场重跑只剩 1 条已登记预存失败（且点名文件与日志工单逐字一致），目录确实是从 46 份 `project.yaml` 逐字生成的。唯一发现的一处事实漂移：`docs/plan.md` 里「现在 3 个 / 3 份 `RESEARCH.md`」是过期计数，磁盘上实际是 9 份（都是研究级派发契约，不违反「RESEARCH.md 非必备件、project.yaml 才是 SSOT」的规矩，只是计数没跟着更新）。以下每条结论都只采信我本轮当场重跑的输出。

---

## 验收 1 · `docs/catalog.md` 是生成物

结论：PASS
复现命令：`python3 tools/build_catalog.py --check && python3 -m pytest tests/test_catalog.py -q`
关键输出：`OK：docs/catalog.md 与 46 个研究项的 project.yaml 逐字节一致（七列：id / 标题 / 目的 / 验收 / 期望产出 / 状态 / 指针）。`；`10 passed`（含改一字符/删一行/加一行三条反向对照，`--check` 手改即红）。另独立脚本核对：`project.yaml 总数 46`、`catalog 行数 46`、`状态不一致或指针不可解析: 无（全部一致且可解析）`。

## 验收 2 · `project.yaml` 是每个研究的 SSOT

结论：PASS（SSOT 与生成物成立；「现在 3 个」是过期计数，见末节裁决项）
复现命令：`python3 tools/build_catalog.py --check; find axes -name project.yaml | wc -l; find axes -name RESEARCH.md | wc -l`
关键输出：目录七列逐字取自 46 份 `project.yaml`；`RESEARCH.md` 实际 `9` 份（不是 plan.md 写的 3），9 份首行均为「研究级契约 / 派发契约 / projected 视图，事实之家是 project.yaml、不复制读数」——规则本身成立，仅计数过期。

## 验收 3 · 元级零研究级读数

结论：PASS
复现命令：`python3 tools/check_meta_layer.py --check`
关键输出：`扫描元级 6 个文档（AGENTS.md, CONTRIBUTING.md, README.md, docs/ledger.md, docs/plan.md, docs/structure.md）：0 处研究级读数；生成物投影不判：docs/catalog.md；白名单 0 条（只放定义性说明）。`（EXIT=0）。另独立 grep 元级文档 `bpc/Δ/experiments 路径` 零命中。

## 验收 4 · `AGENTS.md` 三层契约 + 派发模板；`README.md` 元级重写

结论：PASS
复现命令：`grep -n "### 3.1\|### 3.3\|## 派发：" AGENTS.md; grep -n "元仓库\|一个研究怎么被派发" README.md`
关键输出：`AGENTS.md` 有 `### 3.1 三层各拥有什么、谁可以改`（三层表）与 `### 3.3 派发契约给什么（可复制模板）`（含 `## 派发：<研究项 id / 工作包 id> · <一句话目标>` 模板）；`README.md` 已按元级重写（三层表 + 「目录怎么读」+ 「一个研究怎么被派发与验收」）。

## 验收 5 · 迁移无损 + 台账

结论：PASS
复现命令：`python3 tools/check_ledger.py --check`
关键输出：`台账：18 个原文（其中删除 1 个，其余迁到新家）+ 本次新增 10 个；搬迁前清单 18 个文件；0 处违规。`（EXIT=0）。另独立核对 `2d68eb4` 的 11 对 R100 纯移动：11/11 逐对 blob 哈希相等（字节等价）。

## 验收 6 · 机检按层分工

结论：PASS
复现命令：`cat .github/workflows/wiki.yml | grep -A1 "run:"; python3 tools/summary.py --check`
关键输出：CI（`.github/workflows/wiki.yml`）只跑元级四条（build_catalog / check_meta_layer / check_ledger / check_docs），注释明写「研究级检查由研究级 Agent 自己跑」；研究级机检工具齐备且 `summary.py --check` → `OK：46 个 project.yaml 全部符合 harness/research-schema.md`（EXIT=0），迁走的检查在新层仍生效。

## 验收 7 · 全绿（唯一允许失败 = 已登记预存失败）+ 工作区干净并推送

结论：PASS
复现命令：`python3 -m pytest tests/ -q 2>&1 | tail -5; git status --porcelain; git log -1 --format='%h %s'`
关键输出：`1 failed, 1005 passed in 252.87s`；唯一失败 = `tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`（已登记预存失败）；元级四条机检均 EXIT=0；`git status --porcelain` 为空；`main` 与 `origin/main` 同点（`361cb7a`）。

---

## 未闭环项核对（逐条对 `docs/plan.md` §6）

**① §6「`RESEARCH.md` 现在 3 份（1.1.1 / 1.1.4.1 / 5.5）」——裁决项，计数过期（不下结论）。**
磁盘实况：`find axes -name RESEARCH.md | wc -l` = `9`。除 §6 点名的 3 份外，还有 `1.2.1-dilated`、`1.2.2-block-sparse`、`2.1.3-repeating`、`3.1.5-channel-sparsity`、`3.2.2-per-exit-readout`、`5.4-low-rank` 共 6 份，全是研究级派发契约（首行均声明「事实之家是 project.yaml，本文件不复制读数」）。§6 的**规则**（RESEARCH.md 非必备件、SSOT 是 project.yaml、不挡里程碑）仍成立，只是 `docs/plan.md:24` 与 `:106-108` 的「现在 3 个 / 3 份」计数没随研究级 Agent 接手而更新。这不是里程碑破坏，属元级事实文档的维护滞后。

**② §6「`tools/check_docs.py --check` 当前 0 处违规」——仍成立。**
重跑 `python3 tools/check_docs.py --check` → `0 处违规`（`扫描 504 个文档，检查 9928 处引用`），EXIT=0。

**③ §6「这四处上过红名单但不是缺陷（1.1.1 的 2679 analysis、5.5 的 2657 analysis/results、triage 的 2657 与 2680/2681 键锚点）」——仍成立。**
`axes/1-time/1.1.1-hard-window/experiments/2679-anchor-eval-ablation-13m/analysis.md`、`axes/5-quant/5.5-fp8-head-dim/experiments/2657-fp8-hd128-head-dim/analysis.md` 与 `results.json` 均真实存在；`check_docs` 0 违规即四条锚点解析全部通过。

**④ §6「1.1.4.1 的让位说明曾带仓外路径，已改成不带路径的写法」——仍成立。**
`grep -rnE "/(home|workspace|opt|tmp|mnt|data)/"` 于 `1.1.4.1-explicit-memory/{project.yaml,RESEARCH.md}` → 零命中，无「让位」残留。

**⑤ §6「pytest 剩下逐条归属（964 passed / 6 failed）」——旧计数已过期，现状为 1 预存失败。**
当场重跑 `python3 -m pytest tests/ -q` → `1 failed, 1005 passed in 252.87s`。唯一失败是 `tests/test_mutation_framework.py::test_all_repo_mutate_scripts_use_the_harness`，点名的 4 个脚本为 `2674-mdiffusion-13m` / `2671-padfill-13m` / `2660-latentreg-vs-disc-13m` / `2670-freerun-bpc-13m` 各自的 `mutate.py`，与 `.scratch/63-axis6-predict/NEXT-WP-mutation-harness.md` 第 6–9 行的清单**逐字一致**。§6 所述「剩下 5 条级联（test_all_doc_references_resolve 等）」已随 `check_docs` 0 违规而转绿：重跑 `python3 -m pytest tests/test_repro_inventory.py tests/test_docs_refs.py -q` → `21 passed`（EXIT=0）。

**⑥ §6「`tests/test_repro_inventory.py` 的 2 条已转绿（6 passed）」——仍成立。**
同上，`test_repro_inventory.py + test_docs_refs.py` 合计 `21 passed`，其中 repro_inventory 全绿，与磁盘一致（不是「副本入库集合与真树不一致」那种会重红的情况）。

**残余风险：** `docs/plan.md` 的 `:24`、`:106-108`（RESEARCH.md 计数）与 `:124`（pytest 计数 964/6）三处是过期数字；它们是元级事实文档的维护滞后，需元级 owner 下轮更新，本复核按纪律不改任何事实文档，只在报告点名。
