# ticket 56 · WP-M2.9 审计留痕收口（writer 报告）

> 角色：worker，fresh 上下文，**单一写入者**（收尾小任务）。语言：中文。
> 时间：2026-10-13T12:36:38+08:00（`date -Iseconds` 实取）。
> 契约：独立集成复核 `REVIEW-M2-integration.md`（revision `3861ea5`，PASS）指出的两项 non-blocking 审计可追溯性缺口。
> 提交：`186f1bd2d83d91fe19d4328d66e24255ac277d2c`（main，14 files changed, 3312 insertions）。
> 约束遵守：本 WP 未触碰任何 `axes/**`、测量数字或 `project.yaml`。

---

## 0. 人话摘要（先结论）

集成复核判了 PASS，但发现两处「数据没错、只是后人 clone 仓库看不到完整证据链」的缺口：整合报告和集成复核报告
只在 subagent 的临时目录里、没进仓库；而迁移所依据的问题台账（ticket 54/55 的 `.scratch/` 目录）虽然被
`MAPPING.md` 大量引用，却一直没纳入 git。本 WP 就干这两件事：把两份报告**原样搬进仓库**，把两个台账目录
**显式 `git add` 进 git**，并在工单里补一段 M2 收口的审计记录（时间戳、提交范围、五份复核都 PASS、
1562 个旧证据文件的独立核验口径、四个遗留未决、下一步 M3）。**没有动任何研究内容**（`axes/`、数字、`project.yaml` 全未碰）。
自检：工作树干净、暂存区为空、两道门（`summary --check` / `check_tree`）全绿。

---

## 1. 缺口与处置

| 缺口 | 处置 | 证据 |
|---|---|---|
| ① 整合报告 `WP-M2.8-integrator.md` 只在 artifact 目录 | 原样复制进 `.scratch/56-restructure/wp-reports/` | `diff -q` → SAME |
| ① 集成复核 `REVIEW-M2-integration.md` 只在 artifact 目录 | 同上 | `diff -q` → SAME |
| ② `.scratch/54-subproject-schema/` untracked | 显式路径 `git add` | 2 文件入库 |
| ② `.scratch/55-review/` untracked | 显式路径 `git add` | 9 文件入库 |

- 已存在的 11 份源报告（`wp1…wp7`、`rev-axis1/2/6/rest`）经 `diff -q` 逐字节核对与 artifact 源
  （`…/89c48c00-…/wp-reports/`）**完全一致**，无需复制。
- 现仓库 `.scratch/56-restructure/wp-reports/` = **13 份**（11 源 + 整合报告 + 集成复核）。

## 2. 复制清单（源 → 目标）

| 源（artifact） | 目标（仓库） | 结果 |
|---|---|---|
| `884641c0-…/.scratch/56-restructure/wp-reports/WP-M2.8-integrator.md` | `.scratch/56-restructure/wp-reports/WP-M2.8-integrator.md` | 复制，逐字节 SAME |
| `81e9ae0c-…/.scratch/56-restructure/wp-reports/REVIEW-M2-integration.md` | `.scratch/56-restructure/wp-reports/REVIEW-M2-integration.md` | 复制，逐字节 SAME |
| `89c48c00-…/.scratch/56-restructure/wp-reports/{wp1..wp7}.md` | 同名 | 已存在，`diff -q` SAME（未复制） |
| `89c48c00-…/.scratch/56-restructure/wp-reports/{rev-axis1,rev-axis2,rev-axis6,rev-rest}.md` | 同名 | 已存在，`diff -q` SAME（未复制） |

## 3. 提交内容（14 文件）

- 新增：`.scratch/56-restructure/wp-reports/WP-M2.8-integrator.md`、`…/REVIEW-M2-integration.md`
- 新增：`.scratch/54-subproject-schema/{INVENTORY,TICKET}.md`
- 新增：`.scratch/55-review/{A,C,F,G,H,M,W}-PROBLEMS.md`、`.scratch/55-review/{TICKET,PROPOSED-STRUCTURE}.md`
- 修改：`.scratch/56-restructure/TICKET.md`（追加 §8「WP-M2.9 审计留痕收口」）

`git diff --cached --name-only` 人工核对：14 条，全部在 `.scratch/` 下，无 `axes/**`、无 `project.yaml`。
`git diff --cached --check` → 无空白问题。

## 4. 命令与结果

```
$ git status --short          # commit 前：仅 4 项预期变更（M TICKET.md + 3 个 untracked）
$ git diff --cached --name-only | wc -l    # 14
$ git diff --cached --check                # 无输出（WHITESPACE OK）
$ git commit -F - <<heredoc                # 186f1bd，14 files changed, 3312 insertions(+)
$ git status --short                       # commit 后：空（clean）
$ git diff --cached --name-only | wc -l    # 0
$ python3 tools/summary.py --check         # OK：46 个 project.yaml 全部符合 docs/research-schema.md（exit 0）
$ python3 tools/check_tree.py              # OK：46 个 project.yaml 全部符合 docs/research-schema.md（exit 0）
```

## 5. 遗留未决（已写入 TICKET §8.6）

1. `pathlm/config.py` 的 `chain_mode` 默认 `"latent"`（已被否证）——改默认 = 行为变更，待产品/引擎 owner 定夺。
2. role_map 实名——46 份 `project.yaml` 仍为角色占位，顺延 M3（轴 lead 指派时）填。
3. backfill 旧 schema stub（4 个）——含测量结论，依「有证据者保留」未删，要清须先迁结论。
4. `0.014` 噪声带的跨轴身份——框架性取舍，待出资人裁决。

## 6. 下一步

**M3：按大类 spawn 轴 lead**（见 `.scratch/57-program/TICKET.md`）。
