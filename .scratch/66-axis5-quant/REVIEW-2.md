# 2642 · 第 2 轮独立复核报告（REVIEW-2）

> 复核者：`worker`（**fresh 上下文，不复用第 1 轮**；只读 + 纯 CPU `python3`；**未跑 GPU**）。
> 派单契约：`.scratch/66-axis5-quant/TICKET.md` §7.3；第 1 轮报告 `.scratch/66-axis5-quant/REVIEW-1.md`（仅作参考，不采信其结论）。
> 复核基线（派单指定）= `git rev-parse HEAD` = `d9794b5`（下称「基线」）。**复核期间 HEAD 已前移到 `285eb49`**（并发的
> ticket/证据提交 `1ffce0b`/`b066c48`/`285eb49`，全部只动 `.scratch/66-axis5-quant/**`）；**被复核代码产物在 `d9794b5..HEAD` 之间零改动**
> （`git diff --name-only d9794b5..HEAD -- axes/5-quant/` 为空）⇒ 本节所有判定对基线内容有效，且与当前工作树一致。
> 时间：`2026-10-13T20:22+08:00`。

---

## 0. 人话结论（先读这段）

**结论：PASS（blocker = 0，normal = 0，info = 4）。**

第 1 轮的两条 normal 都**如实改对了**，没有引入新缺陷：

1. **N1（数字写错）已修**：`analysis.md` 里「非支配集 6 臂、其中 5 个是头族」已改为 **4 个**，并点明另外 2 个是
   `int8`（class=`quant`）与 `mha`（class=`baseline`）。我逐条回 `results.json::rows[*].class` 对齐，完全一致。
2. **N2（循环论证门禁）已修**：原来的恒真门禁 `values_unchanged_bitwise_ok` 已改名为
   `readings_passed_through_untouched`，`analysis.md` §1 明确写它「是逐值透传的过程陈述，**不构成独立检验**」；
   `verdict` 改成**只由 G1/G2 决定**（`verdict_basis` 字段自述一致）；死代码（`parent_pairs`/`local_pairs`/
   `value_fingerprint`/`import math`）已删净。
3. **新增的跨产物核对是真检查、不是恒真式**：它把父实验**另一份产物** `2632/table.md`（markdown）与 2642 的
   rows 逐臂对账。我**自己另写解析器**独立重解 27 行，与 rows 零不一致；并用注入扰动实测它**会报红**（见 §3）。
   定位上它是「**2642 未改动读数**」的**独立证据**，但**不能**验证「2632 自身测量正确」——这点项目已在
   `TICKET.md` §7.4 如实登记，我独立复核**同意**。
4. **重跑幂等、指纹自洽、机检全绿、工作树干净**（`exit=0`；三份产物重生成后 sha256 逐字节不变；`summary.py --check` 全绿；
   `git status --short` 空）。

四条 info 都**不影响结论**（容差 5e-5 的固有遮罩边界、独立性强度的措辞、解析器的潜在脆性、`verdict_basis` 缩进），
详见 §4；其中前 2 条项目已自行登记于 `TICKET.md` §7.4，属**确认**而非新发现。

---

## 1. 待核项 1：N1 修复是否真改对 —— **真改对（PASS）**

**判据**：`analysis.md` 应把「原始 bpc 序非支配集 6 臂中头族数」由 5 修正为 **4**，且 4 个是
`mqa_int4`/`gqa_win64`/`mqa_int8`/`mqa`，另 2 个 `int8`/`mha` 的 class 为 `quant`/`baseline`。

**复现命令与证据**：

```bash
grep -n "非支配集\|个是头族\|21 条\|16 臂" axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/analysis.md
python3 - <<'PY'
import json
r=json.load(open("axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/results.json"))
m={row["arm"]:row["class"] for row in r["rows"]}
print(r["pareto"]["raw_arm_order"])
print({a:m[a] for a in r["pareto"]["raw_arm_order"]})
print("raw_dominated:", len(r["pareto"]["raw_dominated_with_dominator"]),
      "strict_over_band:", len(r["pareto"]["strict_over_band"]))
PY
```

实测：

- `analysis.md:126-127` = 「非支配集 = `mqa_int4`, `gqa_win64`, `mqa_int8`, `mqa`, `int8`, `mha`（6 臂）。其中
  **4 个是头族**（`mqa_int4` / `gqa_win64` / `mqa_int8` / `mqa`；另 2 个是 trunk 基线 `mha` 与 trunk×量化 `int8`）」。
- `results.json::pareto.raw_arm_order` = `['mqa_int4','gqa_win64','mqa_int8','mqa','int8','mha']`（6 臂）。
- 逐一 class：`mqa_int4=quant_head`、`gqa_win64=combo_head_window`、`mqa_int8=quant_head`、`mqa=head`
  （四者属头族）；`int8=quant`、`mha=baseline`（**非**头族）。⇒ 头族数 **4**，与文中一致。
- `analysis.md:129` 引用「`raw_dominated_with_dominator`（21 条）」，实测 `len=21`；`analysis.md:132`「越带 = 16 臂」，
  实测 `len(strict_over_band)=16`。逐条对齐。
- 全仓 `grep -rn "5 个是头族" axes/` 无命中（旧错误残留 = 0）。

**判定：修复属实、无新错。**

---

## 2. 待核项 2：N2 修复是否诚实 —— **诚实（PASS）**

### 2.1 门禁改名 + 标注「透传事实、非独立检验」（PASS）

```bash
grep -rn "values_unchanged_bitwise_ok" axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/
grep -n "readings_passed_through_untouched\|非独立检验\|过程陈述" axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/analysis.md
python3 -c "import json;print('readings_passed_through_untouched' in json.load(open('axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/results.json'))['gates'])"
```

- 2642 目录内旧名 `values_unchanged_bitwise_ok` 命中 **0**（仅 `.scratch/66-axis5-quant/{TICKET,REVIEW-1}.md` 的历史叙述里出现，属日志文档，正常）。
- `results.json::gates` 含新键 `readings_passed_through_untouched = true`；
- `analysis.md:56`（§1 门禁表）写明：「本实验对读数是**逐值透传**（不做任何数值变换）⇒ 预登记第 4 条在本层是
  **构造性事实/过程陈述，不构成独立检验**」；`analysis.md:154`（§8 锚点）复述同一限定。措辞明确，未见把透传说成检验。

### 2.2 `verdict` 只由 G1/G2 决定（PASS）

```bash
sed -n '326,336p' axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/frontier_ledger.py
python3 - <<'PY'
import json
g=json.load(open("axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/results.json"))["gates"]
print("G1=",g["G1_independent_recompute_matches_2632"],"G2=",g["G2_all_quantized_arms_use_block_scale"],"verdict=",g["verdict"])
print("basis=",g["verdict_basis"])
PY
```

- 表达式（`frontier_ledger.py:334`）：`"verdict": "PROCEED" if (gate_g1 and gate_g2) else "VOID"` —— **只读 G1/G2**。
- 落盘：`G1=True, G2=True, verdict="PROCEED"`；`verdict_basis` 自述「只由 G1…与 G2…决定；跨文档独立核另列
  （`cross_artifact_table_md`）」，与表达式**一致**。跨产物核对 / 透传事实均**不在** verdict 的条件里。

### 2.3 死代码已删（PASS）

```bash
grep -n "parent_pairs\|local_pairs\|value_fingerprint\|import math\|math\." axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/frontier_ledger.py
```

命中 **0**（NONE FOUND）。四项全部移除。

**判定：N2 三处修复均属实、且如实降级，无过度声称。**

---

## 3. 待核项 3：新增跨产物核对是否真独立 —— **对「2642 未改动读数」是独立证据；对「2632 数值本身正确」仅弱独立**

### 3.1 我**自己**独立重解 `2632/table.md` 表 1（未用 2642 的解析器）

方法：只在 `## 表 1` 与 `## 表 2` 区间内取以 `| \`` 开头、**恰为 13 列**的行，取 `cells[3]`=部署字节、
`cells[7]`=bpc、`cells[10].split()[0]`=copy 召回、`cells[11].strip("[]")`=CI。

```bash
python3 - <<'PY'
import json
md=open("axes/5-quant/5.1-kv-quant/experiments/2632-kv-block-scale/table.md").read().splitlines()
s=e=None
for i,l in enumerate(md):
    if l.startswith("## 表 1"): s=i
    if s is not None and l.startswith("## 表 2"): e=i; break
rows=[]
for l in md[s:e]:
    if not l.startswith("| `"): continue
    c=[x.strip() for x in l.strip().strip("|").split("|")]
    if len(c)!=13: continue
    rows.append((c[0].strip("`"),float(c[3]),float(c[7]),float(c[10].split()[0]),
                 [float(x) for x in c[11].strip("[]").split(",")]))
r=json.load(open("axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/results.json"))
j={x["arm"]:x for x in r["rows"]}
bad=[]
for a,dep,bpc,rec,ci in rows:
    x=j[a]
    if abs(dep-x["deployment_bytes_per_position"])>1e-9: bad.append((a,"deploy"))
    if abs(bpc-x["bpc"])>1e-9: bad.append((a,"bpc"))
    if abs(rec-x["recall_copy"])>5e-5: bad.append((a,"recall"))
    if max(abs(ci[0]-x["recall_copy_ci"][0]),abs(ci[1]-x["recall_copy_ci"][1]))>5e-5: bad.append((a,"ci"))
print("n_rows(independent)=",len(rows),"distinct=",len(set(x[0] for x in rows)),"mismatches=",bad)
PY
```

实测输出：`n_rows(independent)= 27 distinct= 27 mismatches= []`（零不一致）。
抽查（含「4 位小数会变动」的臂）：

| 臂 | md 部署 | json 部署 | md bpc | json bpc | md 召回 | json 召回 | md CI | json CI |
|---|---|---|---|---|---|---|---|---|
| `mqa_int4` | 432 | 432.0 | 1.642750 | 1.64275 | 0.1849 | 0.184896 | [0.1716,0.1990] | [0.171564,0.199014] |
| `int8` | 6528 | 6528.0 | 1.483672 | 1.483672 | 0.1406 | 0.140625 | [0.1288,0.1534] | [0.12878,0.153367] |
| `mha` | 12288 | 12288.0 | 1.482026 | 1.482026 | 0.1530 | 0.152995 | [0.1407,0.1662] | [0.140699,0.166157] |

`mqa_int4` 召回 lower CI 差 |0.1716−0.171564| = 3.6e-5 < 5e-5（四舍五入，非口径差异），与 `run_log.txt:9` 记载的
`max|Δ| = 4.9e-5` 自洽。

**结论：2642 的跨产物核对数值**（27 臂、不一致 0）**可独立复现**。

### 3.2 容差 5e-5 是否合理

- `2632/table.md` 召回/CI 以 `:.4f` 打印 ⇒ 四舍五入误差上界 = 5e-5 ⇒ 容差 5e-5 正好等于理论上界，**合理**。
- **遮罩边界**（实测）：`recall += 4e-5`（< 5e-5）**不被报红**；`recall += 1e-4`（> 5e-5）**报红**。
  即：**存在把 ≤5e-5 的召回篡改静默放过**的可能，但这是 4 位小数显示精度的固有极限 ⇒ 无法更好（除非换个更高精度产物）。
  该残余局限项目已在 `TICKET.md` §7.4 登记。
- **核心量受保护**：部署字节与 bpc 容差 = 1e-9（逐位），实测 `bpc += 1e-3` ⇒ **报红**，`bpc += 1e-10` 亦报红（容差比显示精度更严）。
  ⇒ 主要结论（字节账、bpc 前沿）**不存在容差遮罩**。

### 3.3 该检查能否被伪造 / 是否会静默漏行

- **能必然报红吗**：若 2642 在**重跑**时改动了 rows 的 bpc/召回（bpc 任意改动、召回 >5e-5），逐臂比会命中 ⇒ **必红**。
  （**前提是重跑**：若只事后编辑 `results.json` 输出、不重跑，则文件内嵌的核对结论是陈旧的——这是任何「自报门禁」的
  通性局限，非本条独有；但脚本可**逐字节复现**（见 §4.1），任何输出篡改都会破坏可复现性并被重跑发现。）
- **静默漏行**：解析器对不合格式的行确实 `continue` 跳过，**但** `cross["ok"]` 还要求
  `n_arms_parsed == len(rows)`（=27）⇒ 跳行会让 `n_arms_parsed` < 27、`ok=False`、**门禁报红**。实测：
  删 1 行 ⇒ `n_arms_parsed=26, ok=False`；加 1 行 ⇒ `28, ok=False`。⇒ **不会静默漏行**（风险等级：低）。
- **仍存脆性（info）**：解析器以「`| \`` 开头且 ≥13 列」识别表 1，若 2632 未来新增另一张 13+ 列、`| \`` 开头
  的表，会被误纳入 ⇒ 计数 ≠27 ⇒ 假红（fail-safe 方向，非假绿）。当前实测恰 27，无问题。

### 3.4 独立性判定

- **对「2642 未改动读数」= 独立证据**：比对对象 `2632/table.md` 在 2642 代码/数据流**之外**（不由 2642 生成），
  且 2642 的 rows 来源 `2632/results.json` 受 `input_sha256` 锁死 ⇒ 2642 无从「既改数又改参照」。这正是 N2 想要的
  「比较对象不是同一内存对象」。
- **对「2632 数值本身正确」= 弱独立（不能验证）**：`2632/table.md` 与 `2632/results.json` 由**同一次 `run.py`
  同一份 `points`** 序列化而成（见 `2632-kv-block-scale/run.py:238-247` 表 1 渲染、`:340` 写 table.md）⇒ 二者若同错
  则同显，本核对**不**构成对 2632 测量的独立复核（那是 `2632::reuse` 逐位对照与变异测试的责任）。
- 因此：**不是恒真式**（我实测它能被扰动否证）、**不是全程独立**（共享 2632 上游），正确定位是
  「跨产物一致性 + 2642 防篡改的独立证据」。`analysis.md` §1 与 `TICKET.md` §7.4 的定位与此**一致**，我**同意**。

---

## 4. 待核项 4：修复是否引入新缺陷 —— **无新缺陷（PASS）**

### 4.1 重跑幂等（PASS）

为避免写入被复核目录，我在 `/tmp/sbx2642` 搭镜像（`2632` 与 `axes/6-predict`、`axes/7-references` 软链，
仅 `2642` 目录为可写副本，使 `ROOT` 解析到沙箱）后重跑脚本：

```bash
python3 /tmp/sbx2642/axes/5-quant/5.1-kv-quant/experiments/2642-kv-same-byte-frontier/frontier_ledger.py ; echo exit=$?
sha256sum <real>/results.json <real>/table.md <real>/run_log.txt
sha256sum /tmp/sbx2642/.../{results.json,table.md,run_log.txt}   # 逐字节 diff 亦 SAME
```

实测：

- `exit=0`；日志逐行与 `run_log.txt` 相同（含 `G1 … PASS`、`G2 … PASS`、跨产物核 `不一致 = 0 ⇒ PASS`）。
- 重生成三份产物与仓库版本 **sha256 逐条相同**：
  `results.json ca166b3f…`、`table.md b3f6120e…`、`run_log.txt 93a188a5…`（`diff -q` 亦 SAME）⇒ **幂等**。
- 6 个输入 `sha256` 与 `config.json::inputs_sha256` 独立复算**逐条相符**（含 2 份 2632 产物与 4 个 checkpoint）。

### 4.2 `results.json::code` 指纹（PASS）

`results.json::code` = `{...frontier_ledger.py: 4a929717…c91b}`；磁盘 `sha256sum frontier_ledger.py` = `4a929717…c91b`。**一致。**

### 4.3 `results.json::gates` 每个布尔与 `run_log.txt` 自洽（PASS）

| gate | 值 | run_log 对应行 |
|---|---|---|
| `G1_independent_recompute_matches_2632` / `G1_max_abs_diff_deployment_bytes` | `True` / `0.0` | `G1 … max|Δ部署字节| = 0 ⇒ PASS` |
| `G2_all_quantized_arms_use_block_scale` / `G2_offenders` | `True` / `[]` | `G2 … 非块标尺量化臂 = [] ⇒ PASS` |
| `value_only_caliber_agrees` | `True` | `仅值口径核对：…不一致条目 = 0` |
| `readings_passed_through_untouched` | `True` | `必须不变 (a) 透传事实 = True（非独立检验）` |
| `cross_artifact_table_md_ok` / `n_arms_parsed` / `mismatches` | `True` / `27` / `[]` | `(b) … 解析 27 臂, max|Δ|={…,4.9e-5,…}, 不一致 = 0 ⇒ PASS` |
| `parent_nonquant_bitwise_preserved` / `…delta_bpc` | `True` / `0.0` | `上游独立检验：2632::reuse.nonquant_bitwise_preserved = True（max|Δbpc| = 0.0…）` |
| `verdict` | `PROCEED` | `结论摘要：G1 PASS…`（G1&G2）|

逐条自洽，无矛盾。

### 4.4 机检（PASS）

```bash
python3 tools/summary.py --check   # → OK：46 个 project.yaml 全部符合 docs/research-schema.md；exit=0
```

### 4.5 范围纪律（PASS）

```bash
git status --short            # → 空（工作树干净）
git diff --stat d9794b5..HEAD # → 只动 .scratch/66-axis5-quant/{TICKET.md,evidence/MANIFEST.md}
git diff --name-only d9794b5..HEAD -- axes/5-quant/   # → 空
```

- 基线提交 `d9794b5` 的改动集（`git show --stat`）全部落在 `.scratch/66-axis5-quant/**` 与
  `axes/5-quant/**`，**无越界**（未动 `pathlm/**`、`tests/**`、`tools/**`、其它轴）。
- 工作树**干净**（无残留未提交文件，包括其它轴）。复核期间 HEAD 前移的 3 个提交也**只在** `.scratch/66-axis5-quant/**`，
  未触碰被复核的 `axes/5-quant` 产物 ⇒ 不影响判定。
- 我全程**未** `git add`/`commit`，未改任何被复核文件（重跑在 `/tmp` 沙箱进行）。

---

## 5. 修复是否引入新缺陷 —— 补充（无 blocker/normal）

- 无工具/测试消费这些字段（`grep -rn` 于 `tools/`、`tests/`：无人读 `readings_passed_through_untouched` /
  `cross_artifact_table_md`；`verdict` 命中均为**其它实验**的键）⇒ 改名/改结构**未破坏任何下游**。
- `verdict` 由布尔变字符串、嵌套在 `gates` 内 —— 与 `verdict_basis` 同处 `gates`，JSON 合法、语义自洽（见 info-4）。

---

## 6. 发现汇总

### blocker
- **无**。

### normal
- **无**。

### info（报告级，不阻塞）
- **I1 容差遮罩边界**：召回/CI 容差 5e-5 会把「≤5e-5 的召回篡改」静默放过（实测 `+4e-5` 不红、`+1e-4` 红）。
  固有于 `2632/table.md` 的 4 位小数显示精度，无法更好；**bpc/部署字节为 1e-9 逐位，核心结论不受影响**。
  （`TICKET.md` §7.4 已登记。）
- **I2 独立性措辞**：`analysis.md` 称其为「跨文档独立核」。**对「2642 未改读数」确为独立证据**；但该产物与
  `results.json` 共享 `2632` 同一次 `run.py` 的同一 `points` ⇒ **不构成对 2632 测量正确性的独立复核**。
  定位已在 `TICKET.md` §7.4 写明（「2642 未改动读数的独立证据，不冒充质量结论的新证据」），我**同意**，无需改文。
- **I3 解析器潜在假红（fail-safe）**：以「`| \`` 开头且 ≥13 列」识别表 1；若 2632 未来新增同类表会误纳入 ⇒ 计数 ≠27 ⇒ 假红
  （方向是 fail-safe，非假绿）。当前 27/27 无问题。
- **I4 `verdict_basis` 缩进**：`frontier_ledger.py:334-336` 中 `verdict_basis` 与 `}` 同缩进（8 空格），实际**仍在 `gates` 内**
  （与 `verdict` 同级）；JSON 合法、无下游消费，属排版瑕疵。

---

## 7. 未核项
- **无**。派单要求的所有项均已执行（重跑、指纹、门禁自洽、机检、范围、独立重解、扰动实验）。未跑 GPU（本项纯 CPU 再分析，无需）。
  第 1 轮已 PASS 的手算 G1、口径纪律、同字节分组、参数混杂、Pareto、范围纪律**未重跑**（修复未触碰其逻辑；本轮仅补 §1 的数值对齐）。
