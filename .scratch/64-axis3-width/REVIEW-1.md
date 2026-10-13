> 本文件由 `axis3-lead` **原样落盘**自复核者的最终回复（复核者无写文件工具，全文随回复返回）。
> 复核者 = fresh 独立上下文 subagent（run `e27c6dbf-25a6-44c7-919e-5f8fc780748c`，`reviewer` 角色，**只读**），
> 复核目标 revision = `e6d2db1`（复核期间工作树 == 该提交）。复核者声明的工具限制：无 shell/无写文件工具
> ⇒ T1/T2/T5/T6/T9 的**执行**部分未跑（其报告 §0 L1–L5 与 §3 给出待跑命令）；未跑项**不计入通过**。
> 以下为复核者原文，未作任何删改。

---

# VERDICT: FAIL（窄范围）

一句话总判词：**实验设计、落盘证据、等 FFN 预算声称、门与冻结时序自洽性全部经我独立静态核验成立，预登记阈值规则推出的「总量是旋钮、分配不是（H0）」结论也成立；但 `analysis.md` 与 SSOT `project.yaml` 里有 3 处数字与落盘证据不符（其中 1 处是承载论证"分配效应被种子噪声吃掉"的关键比较），故按派单 blocker 定义判窄范围 FAIL，修完这 3 处（纯文字/数字，不需重跑 GPU）即可复审 PASS。**

范围声明：**不**质疑 7 臂证据本身，**不**要求重训；只要求改正事实文档里的数字与一处越界论证。

---

## 0 工具/环境限制（逐条声明；未跑项一律不算通过）

| # | 限制 | 影响 | 替代核验方式 |
|---|---|---|---|
| L1 | **无 shell / 无执行工具**（只有 read/grep/find/ls/watchdog_diff）⇒ 不能跑 `pytest`、不能跑自写 python、不能 `sha256sum`/`stat`/`git` | T1/T2/T5/T6/T9 的**执行**部分未跑 | 静态读码 + 手算 + 逐字段对账；命令清单见 §3 交 supervisor |
| L2 | 无法读提交内容（`watchdog_diff` 明确不含 committed range）⇒ 不能核 `README.md`/`train_arms.sh`/`test_*.py` 与 `49f697f` 的内容哈希，也不能核 mtime | T6 的哈希/时序断言**未核**（只能核 run_log 内部自洽） | 从 MODEL.md 的哈希清单与 TICKET §7.2.1 交叉读取（**不等于**独立核对） |
| L3 | 未取 `/tmp/pathlm_gpu.lock`、未跑任何训练/评测 | — | 按派单要求 |
| L4 | `runs/**` 被 `.gitignore` 覆盖 ⇒ `grep`/`find` 跳过该目录，只能靠 `read` 逐文件读 | 覆盖度受限（我读了 7/7 个 `runs/*/results.json` 的 `config`+`train` 块，未逐行读满 600 行） | 见 T4 |
| L5 | 我只读、未写；`watchdog_diff` 报工作树相对 `e6d2db1` **无 staged/unstaged 变更** ⇒ 我的核验对象 = `e6d2db1` 工作树 | — | — |

---

## 1 必做检查 T1–T11

### T1 · 门与变异（**未执行**，静态审查）
- 待跑命令（交 supervisor）：`cd axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m && python3 -m pytest test_2641_ffn_ratio.py -q`
- 静态旁证：`gates_report.txt:1-6` 记 `exit=0` / `13 passed in 4.99s`；我**独立数**了测试用例数 = T1 两条 + T2（`sorted(SCHEDULES)` 参数化 4 条 + 等预算 1 + 前缀 1）+ T3 五条 = **13**，与 "13 passed" 一致（只证数目自洽，**不是**我跑过）。
- **门是否有意义（我的独立判断）**：
  - **T3 会红**：可逐条从代码推出——T3a `blk.mlp[1] = nn.ReLU()`（`test_2641_ffn_ratio.py:120-133`）改变前向 ⇒ 与原生 narrow 不同；T3b r=3 与 r=2 权重形状不匹配 ⇒ `load_state_dict` 抛 `RuntimeError`（:139-146）；T3c 少切一层 ⇒ 审计与解析参数不符（:149-165）；`r=0/5/长度不符` ⇒ `narrow_ffn_prefix` 抛 `ValueError`（run.py:96-100, :110）；T3 活性对照要求 trunk 逐位相同（:176-182）。**结论：变异不是恒真摆设。**
  - **T1 本身对"切片语义错"不敏感（见 N5）**：T1a/T1b（:44-58）都先 `sliced.load_state_dict(stock_narrow.state_dict())`，把待检参数**整份覆盖**，而两个模型同类同前向 ⇒ 断言 `torch.equal(a,b)` 由 `a.shape == b.shape` 蕴含。**任何保形的切片错误（取后缀、取偶数行、换层映射）都会通过 T1**。T1a 自己的 docstring（:124-127）已承认这点。
  - 但**证据链没有洞**：前缀语义由 `test_T2_prefix_slice_takes_first_rows_and_cols`（:88-98，同 seed 建全宽模型后比较 `W[:2d]`、`W[:, :2d]`、且断言是新参数对象）钉住，而「Linear 取其前 h 行 ≡ 原生窄 Linear」+ T1 的「同形同接线同权 ⇒ 同前向」合起来确实给出等价性。所以 N5 是**措辞/归属**问题，不是证据缺口。

### T2 · 自算 P1/P2/P4（**脚本未执行**；我用 `read` 取逐臂文件 + 手算复算，等价于该脚本目的）
逐臂原始值（`results_<arm>_s<seed>.json:68` 的 `eval.clean_bpc`）：

| 臂 | clean_bpc | 出处行 |
|---|---|---|
| U4-s0 | 1.5314 | `results_U4_s0.json:68` |
| U2-s0 | 1.5709 | `results_U2_s0.json:68` |
| U2-s1 | 1.5611 | `results_U2_s1.json:68` |
| PYR-s0 | 1.5730 | `results_PYR_s0.json:68` |
| PYR-s1 | 1.5750 | `results_PYR_s1.json:68` |
| APYR-s0 | 1.5644 | `results_APYR_s0.json:68` |
| APYR-s1 | 1.5601 | `results_APYR_s1.json:68` |

- Δ_up = bpc_U2 − bpc_PYR：s0 = 1.5709 − 1.5730 = **−0.0021** ✓；s1 = 1.5611 − 1.5750 = **−0.0139** ✓（与 `results.json::P1_P2_deltas_by_seed` 逐位相同，:104-112）
- Δ_down = bpc_U2 − bpc_APYR：s0 = 1.5709 − 1.5644 = **+0.0065** ✓；s1 = 1.5611 − 1.5601 = **+0.0010** ✓
- 0.014 判定：四个 |Δ| ∈ {0.0021, 0.0139, 0.0065, 0.0010} 全部 `< 0.014` ⇒ 两条 seed 的 `P1_verdict_by_seed` 都应是「不可排序（H0）」，与 `results.json:114-117` 一致 ✓
- P4 同号：up 两值同负 ✓、down 两值同正 ✓ ⇒ `P4_same_sign_across_seeds = true` ✓（`results.json:118`）
- 一一对应：`results.json::source_files` 列 7 个文件、`arms` 块 7 个键，与目录里 7 个 `results_*.json` 一一对应；每臂的 `sum_r/macs_per_token/macs_ffn/params_total/params_ffn/slope_per_1k` 我都与 `results.json::arms` 对了，**逐位相同** ✓（U4-s1 文件确实不存在，聚合器按 `os.path.exists` 自然跳过，P5 用 `("U4",0)` 仍在 ✓）
- 待跑命令（交 supervisor，可与上面互为独立实现）：
  `python3 - <<'PY' …`（重算四个 Δ + 0.014 + 同号；不再赘述，我的算术已给结论）

### T3 · 等 FFN 预算（静态逐位核验）
- `sum_r`：U2 = PYR = APYR = **16**、U4 = **32** = 恰两倍 ✓（`results.json:166-172`；臂文件 `"sum_r": 16/32`）
- `cost.macs_per_token.ffn`：三臂 = **2,097,152** 逐位相等 ✓；U4 = **4,194,304** = **恰两倍** ✓（`results_APYR_s0.json:132`、`results_U4_s0.json:132`、`results.json:173-178`）
- `params.params_ffn`：三臂 = **2,103,296** 逐位相等 ✓；U4 = 4,204,544，而 2×2,103,296 = 4,206,592 ⇒ **"U4 恰为其两倍"对 FFN 参数不成立**（差 2,048 = 8 层 × d=256 的 `lin2` 偏置，不随 r 缩放）→ **N2**
- `cost.ffn_params_analytic == params.params_ffn`：7/7 成立（`ffn_params_analytic_matches_measured: true`，各 `results_*.json:150`；解析式 = Σ(2·r·d² + r·d + d)，Σr=16 ⇒ 2,097,152+4,096+2,048 = 2,103,296 ✓；Σr=32 ⇒ 4,194,304+8,192+2,048 = 4,204,544 ✓）
- 逐层 FFN MACs 分布 == schedule：PYR `[524288, 393216, 262144, 262144, 262144, 131072, 131072, 131072]`（`results_PYR_s0.json:136-145`）== 2·r·256²（r=4→524,288 / 3→393,216 / 2→262,144 / 1→131,072）✓；APYR 为其镜像（`results_APYR_s0.json:136-145`）✓；U2/U4 各层常数 262,144 / 524,288 ✓
- 总 MACs = 8×393,216 + 2,097,152 + 52,736 = **5,295,616** ✓（Σr=16）；U4 = 3,145,728+4,194,304+52,736 = **7,392,768** ✓

### T4 · 臂间配置一致性（静态逐字段对照，7/7）
- `runs/<arm>-s<seed>-12k/results.json` 的 `config` 块（26 字段）在 7 个 run 上**逐字段相同**：`d_model 256 / n_layers 8 / n_heads 8 / seq_len 256 / mlp_mult 4 / dropout 0.0 / norm_cap 1.0 / gate_bias_init -2.0 / dense true / carry_gate true / aux_mode "renorm" / mod_ratio 0.0 / fill_kv true / proj_fill false / readout_dims null / readout_gated false / readout_linear_resid false / readout_perdepth false / readout_wide_shared 0 / topology "sequential" / attn_sparse_k 0 / train_window 0 / train_anchors 0 / attn_linear false / linear_chunk 64 / n_kv_heads null / carry_rule null / carry_chunk 64 / carry_state_dim null / recon_window 0` — **未发现任何不等字段**
- `train` 块：`steps 12000 / batch_size 32 / lr 0.001` 全同，`seed` 按契约分别为 0/1 ✓
- `train_tokens = 98,304,000` 7/7 ✓（= 12000×32×256）；`vocab 206` ✓（`pathlm/data.py:27-30`：前 90M train / 末 10M eval、vocab = 205 独立字节 +1 mask = 206 ✓，各臂同一 `data/enwik8_full.npz`）
- 数据流：`train_depth_ar.py:203` 用独立 `torch.Generator().manual_seed(tcfg["seed"])` ⇒ 同 seed 各臂**同一输入批序列** ✓
- eval 批：`run.py::evaluate` 不传 generator，`pathlm/depth_ar.py:2442` `generator or torch.Generator().manual_seed(2)` ⇒ 臂间同一 eval 批 ✓（= 契约所称 seed 2）
- `code_hash_run_py`：7/7 臂文件 + `results.json` 全为 `cdf48604d32613ecf7ee226105390787c3e841bba442033c1a51483b40fddfbe` ✓ 相同；**但**见 N8（U4-s0/U2-s0 的该哈希是训练后重评写入的，不能单独当作"训练代码冻结"的证据）

### T5 · 初始化控制（**未执行**；给出可判定的静态论证 + 交 supervisor 命令）
- 待跑命令（交 supervisor）：
  ```bash
  cd axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m
  python3 -c "
import torch, run
u4 = run.build_model(run.SCHEDULES['U4'], 206, 0)
for arm in ('U2','PYR','APYR'):
    m = run.build_model(run.SCHEDULES[arm], 206, 0)
    for l in range(8):
        assert torch.equal(u4.blocks[l].qkv.weight, m.blocks[l].qkv.weight)
        assert torch.equal(u4.blocks[l].proj.weight, m.blocks[l].proj.weight)
        r = run.SCHEDULES[arm][l]; h = r*run.D_MODEL
        assert torch.equal(m.blocks[l].mlp[0].weight, u4.blocks[l].mlp[0].weight[:h])
        assert torch.equal(m.blocks[l].mlp[2].weight, u4.blocks[l].mlp[2].weight[:, :h])
print('init-control OK')
"
  ```
- 静态论证（我认为据此该断言必成立）：`run.py::build_model`（:120-131）先 `torch.manual_seed(seed)` 再以**恒定** `mlp_mult=4` 构造，随后 `narrow_ffn_prefix`（:96-119）只做 `lin1.weight = Parameter(W[:h].clone())`、`lin1.bias[:h]`、`lin2.weight = Parameter(W[:, :h].clone())`；RNG 消耗与 schedule 无关 ⇒ trunk 逐位相同、FFN 为 U4 的**逐位前缀**。训练路径同理：`train_depth_ar.py:771` `torch.manual_seed(args.seed)` 在前、`:790` 建模型在后（run.py 用工厂替换，工厂不额外抽 RNG）。
- 已落盘的同类证据：`test_T2_prefix_slice_takes_first_rows_and_cols`（:88-98）与 `test_T3_mutation_wide_arm_differs_from_narrow_arm`（:170-182，U4 vs U2 的 qkv/proj/embed 逐位相同）✓。⇒ 与 `analysis.md` §5.1 的声称一致（**但我未执行复算**，见 L1；另见 I2：§5.1 引的"本次核验脚本"未入库）。

### T6 · 契约未被事后改过（**哈希/mtime 未核**，仅核 run_log 自洽）
- 未核（L2）：`README.md` / `train_arms.sh` / `test_2641_ffn_ratio.py` 与 `49f697f` 的内容哈希；`README.md` mtime < `18:00:37`。TICKET §7.2.1（TICKET.md:158-163）自报 mtime 17:58:24 / 17:59:41 / 18:00:10 且哈希逐位相同，并如实登记「提交 `49f697f` 的 `%cI` = 18:02:28 **晚于**起跑，故不以提交时间作证据」。
- 已核（run_log 自洽，`run_log.txt:1-32`）：
  - 起跑 18:00:37 与 README/analysis 引用的起跑时刻一致 ✓
  - 7 次 `lock acquired` → `lock released` 成对，臂间窗口 **恒 61 s ≥ 60 s** ✓（18:08:29→18:09:30、18:16:20→18:17:21、18:24:38→18:25:39、18:33:16→18:34:17、18:42:09→18:43:10、18:50:54→18:51:55）
  - 7 臂墙钟 6.83–7.87 min（锁窗）对应 `shared_train_results.wall_minutes` 6.73–7.76（analysis §1 的「6.7–7.8 min」= wall_minutes ✓）；峰值 771.1–935.3 MB ✓；末臂 19:00:26 结束、reeval 19:00:42–19:02:52、`results.json::generated_at` 19:07:55 ⇒ **聚合晚于重评** ✓（哈希/字段因此统一）
  - **`nvidia-smi` 只有 "free before"**（run_log.txt:3,7,11,15,19,23,27），无 "after" ⇒ analysis §1「前后读数逐臂见 run_log.txt」不准确 → **N3**

### T7 · 数字保真（逐条核；**3 处不符**）
已在落盘文件中核 **对** 的（节选，全部给出处）：
- §0/§2/§3/§4 的全部 bpc：1.5314 / 1.5709 / 1.5611 / 1.5730 / 1.5750 / 1.5644 / 1.5601（各 `results_*.json:68`）✓；中间深度 1.9727、1.7492（`results_PYR_s0.json:54,59`）、2.3338、2.7887→1.7887（`results_U2_s0.json:54,59`）✓
- 参数/MACs：2,103,296 / 2,097,152 / 5,295,616 / 4,204,544 / 4,194,304 / 6,962,688 / 4,861,440 ✓
- 斜率：0.00401/0.00418/0.00332/0.00270/0.00354/0.00580/0.00178（`results.json:20-90`、各臂 `slope_per_1k_steps`）✓，§2「0.0018–0.0058」✓
- 散布：U2 0.0098、APYR 0.0043、PYR 0.0020 ✓（逐臂 bpc 相减）；§3-P5「四个 run 1.5257/1.5287/1.5298/1.5314，极差 0.0057」= 1.5314−1.5257 ✓，其中 **1.5287 我直接核到锚文件**（`.scratch/23-depth-ar/evidence/23d-dense-B-12k/results.json:85,89`）✓；1.5257/1.5298 只在 `.scratch/…/TICKET.md:30,213` 与 `30-batch3/evidence/REVIEW.md` 可查（二手，未独立核）
- P3 两项：**「每省 1% 每 token 算力 ≈ +0.0014 bpc」**：2,097,152/7,392,768 = 28.368% ⇒ 0.0395/28.368 = 0.001393 ✓；**「−28.4%」**= 28.37% ✓；**「−30.2%」**= 2,101,248/6,962,688 = 30.179% ✓；「2.8× 带」= 0.0395/0.014 = 2.821 ✓（§6 写 2.82 ✓）；§7.2「attention 59.4% / FFN 39.6%」= 3,145,728/5,295,616 = 0.59402 / 0.2097…= 0.39601 ✓（`results_U2_s0.json:135` `ffn_share_of_total_macs = 0.396017` ✓）
- §1「98.3M token / 1.09 epoch」= 98,304,000、÷90M = 1.0923 ✓；「峰值 771–935 MB」✓；「6.7–7.8 min」✓；「gnorm 5.15–7.86」✓（7 臂 `gnorm_monitor.gnorm_max` = 5.15/5.89/6.49/6.73/7.45/7.68/7.86）

**不符（见 findings）**：
- **B1** `analysis.md:32`「四个值…且小于同配置跨 seed 实测散布（0.0098）」——PYR-s1 的 |Δ| = 0.0139 **>** 0.0098（同一句左边就写着 −0.0139）；同错复制到 `project.yaml::conclusion`、`failure_analysis`，并在 `analysis.md:153` 变成「四条等预算读数（±0.007）」（真实范围 −0.0139…+0.0065）。
- **B2** `analysis.md:99`「APYR 则相反（第 6 层 **2.4699**–2.7209）」——落盘 APYR-s1 第 6 层 = **2.4684**（`results_APYR_s1.json:54`），s0 = 2.7209 ✓；2.4699 在**任何**结果文件里都不存在（全仓 grep 只命中 analysis.md 自己）。
- **B3** `analysis.md:140`「= 2·(4−2)·256²·8 + 偏置差(**1,024**)」——2,097,152 + 1,024 = 2,098,176 ≠ 2,101,248；正确余项 = **4,096**（= (4−2)·256·8，`lin1` 偏置；`lin2` 偏置 8d = 2,048 不随 r 变）。

### T8 · 口径纪律（静态扫）
- `axes/3-width/**/project.yaml`（8 份）：`validation` 全部写了「**墙钟不进判据**」；`cost_model` 8/8 只含等效理论量（FLOPs / layer-passes / tokens-per-forward / pipelining 按每层 FLOPs 分布，明确「不用任何时间量」）✓。`3.1.4` 的 `cost_model` 无时间/日期/吞吐项 ✓
- `analysis.md`：§1 的墙钟/显存明确标「只作排程用，不进判据」✓；§2 的 `cost` 附注「不含任何时间量」✓（`results_*.json:147` note 原文可核）；§7.2 提到 pipelining 但只报每层 FLOPs 分布 ✓
- 未发现判据/`cost_model` 中出现墙钟 / 吞吐 / 日期型成本 ⇒ **口径纪律通过**（`cost_model` 无墙钟/日期是 schema 校验项，见 `docs/research-schema.md:150`）

### T9 · 越界扫描（**未执行**，无 git 访问）
- 待跑命令（交 supervisor）：
  `git diff --stat 49f697f..e6d2db1`（要求只出现 `axes/3-width/**` 与 `.scratch/64-axis3-width/**`）、`git status --short`、`python3 tools/summary.py --check`
- 我能给的唯一独立旁证：`watchdog_diff` 报工作树相对 `e6d2db1` **无任何 staged/unstaged 变更、无新增未跟踪文件**（runs/** 被 .gitignore 覆盖故不列出）⇒ 工作树 == 被审 revision ✓。**越界与 `summary.py --check` 未核，不算通过。**

### T10 · 结论是否越证据
- `status: "已完成"` 是 `docs/research-schema.md:48,121` 的合法枚举值，且 `conclusion`/`experiments` 齐备 ✓；`failure_analysis` 非空并按 §7 三分归因 ✓
- **U4 仅 seed 0 如实登记** ✓（`analysis.md:170`、`project.yaml::gaps`、`conclusion` 末尾「U4 锚 n=1」）
- **P4「同号」与「判 H0」不矛盾**：预登记 P4 = 符号可复现、P1 阈值 = |Δ|<0.014，两条判据独立；`analysis.md:84` 也如实写「同号成立 ⇒ 方向可复现；但按预登记只报不可排序」✓。**真正的矛盾在「幅度小于…散布（0.0098）」这句**（B1），它把"同号但幅度小"这一步推理用错数支撑了。
- **「分配不是独立维度」的限定**：`analysis.md:156`（§7.2）与 `project.yaml::failure_analysis`/`gaps` **有**限定（只覆盖同多重集 1120 种排序里的两条单调阶梯）；但 SSOT 的头部 `conclusion` ② 与 `falsification` 用的是无限定的一般式 → **N6**（预登记原文即一般式，故非 blocker，但头部结论应带限定，否则下游 3.1.2 会当成一般定律继承）
- 其余「结论 ≤ 证据」检查通过：判负归因声明「本项未引入新机制 ⇒ innovation 失败不适用」✓；副产品明确标「不作判据」且未接早退口径 ✓

### T11 · 主动证伪（两条最强反假设）
**(a) 共享初始化压住了分配效应（窄臂只能保留前 r·d 个通道；换一次抽样/换通道子集可能出现效应）**
- 事实核：`narrow_ffn_prefix` 确实只取**前缀行/列**（run.py:108-115），故每个窄臂的存活通道恒为 `0..r−1`，且 APYR/PYR 只是把同一多重集换个深度顺序 —— 通道子集这一自由度在全部臂间**恒定**。
- 判定：**现有证据不能排除 (a)**，但能给它一个量级上界。可排除的部分：通道子集选择是 i.i.d. 抽样的（README §11.2 已预登记该混杂），且**同臂换 seed 的实测散布（U2 = 0.0098）本身就包含"通道取值/子集运气"这一分量**；因此"换一个通道子集"能带来的位移量级 ≲ 1e-2 量级，不足以把本实验的四条等预算读数变成 **超过 0.014 的可采纳收益**。不能排除的部分：本设计对"**子集与深度位置的交互**（恰好前几层的头几个通道更有用）"完全不可识别 —— 窄臂的存活通道固定为前缀，使 PYR/APYR 的读数**条件在这一次抽样**上。
- 因此结论应被**弱化为**：「在本尺度、本次共享初始化抽样下，逐层 FFN 分配**不可排序**；不能排除换通道子集/换抽样后出现 ≈0.01–0.014 量级的效应」——而不是无条件的「分配不是独立维度」（这一点与 N6 同向；`analysis.md:32` 的"被种子噪声吃掉"更是**用错数**，见 B1）。
- 最小下一步（可选，非 blocker）：追加 1–2 臂「同 schedule 但把中间通道做一次固定随机置换」的对照（2 臂 × 12k ≈ 15 min GPU），或直接把 PYR/U2 的 seed 数从 2 提到 4（≈30 min GPU）。**该反假设无法由现有证据推翻，故必须写进边界。**

**(b) seed 1 的 Δ_up = −0.0139 已贴 0.014 带边、n=2 ⇒ 不能排除幅度 ≈0.014 的真效应**
- 判定：**不能排除，且分析文档里恰恰用了一个错误的比较来宣称已排除**（B1：它说"四个值…小于 0.0098"，但 0.0139 > 0.0098）。项目自己的方差依据是 `docs/methodology.md:19`「种子散布 ≤0.0138 ⇒ Δ < 0.014 不可排序」——**0.0139 正好等于项目自认的种子散布上界**，所以"贴边"是实质性的，不是措辞。
- 但**决策相关结论仍然稳**：若真效应存在，它在这一 seed 上的**符号是负的**（PYR 更差），即 (b) 能造就的最好结局是"非均匀分配**反向有害**"（预登记 P1 的 `Δ ≤ −0.014` 分支），**而不是"有增益"**；APYR 方向最好也只有 +0.0065 < 0.0098。所以「不采纳非均匀分配」这一决定不受 (b) 威胁。
- 结论应如何弱化：把 §0/§3-P4/`project.yaml` 里"幅度小于同配置跨 seed 散布"改为——「PYR-s1 的 |Δ| = 0.0139 ≈ 项目自认的种子散布上界（0.0138），**不能排除幅度约 0.014 的真实（负向）效应**；按预登记阈值规则仍判不可排序（H0），且没有任何 seed 显示非均匀分配更优」。绝不写"被种子噪声吃掉"。

---

## 2 发现清单

### blocker
- **B1**（承载论证用错数，已进 SSOT）· `analysis.md:32`（§0）、`analysis.md:153`（§7.1）、`analysis.md:94`（§4，弱式）、`axes/3-width/3.1.4-ffn-ratio/project.yaml::conclusion` ② 与 `::failure_analysis` ① · 违反判据 T7/T10（「数字与落盘证据不符」「结论越证据」）· 证据：`results.json:104-112` `delta_up_U2_minus_PYR(s1) = −0.0139`；`results_U2_s1.json:68` = 1.5611、`results_PYR_s1.json:68` = 1.5750 ⇒ |Δ| = 0.0139 **>** 同臂跨 seed 散布 0.0098（1.5709−1.5611）；§7.1 的「±0.007」与 §0/§3 的「−0.0139」自相矛盾 · 复现：`python3 -c "print(1.5611-1.5750, 1.5709-1.5611)"` → `-0.0139 0.0098` · **最小修**：把"且小于同配置跨 seed 实测散布（0.0098）"删掉，改为「四个值全在 0.014 带内；其中 PYR-s1 的 |Δ| = 0.0139 ≈ 项目自认种子散布上界（0.0138）⇒ 不能排除幅度约 0.014 的真实（负向）效应；按预登记阈值规则判 H0，且没有任何 seed 显示非均匀更优」，并把 §7.1 的「(±0.007)」改为「(−0.0139…+0.0065)」
- **B2**（数字与落盘不符）· `analysis.md:99`（§4）· APYR 第 6 层写成 **2.4699**，落盘为 **2.4684**（`results_APYR_s1.json:54`）；s0 = 2.7209 ✓ · 复现：`grep -n '"bpc": 2.46' results_APYR_s1.json`；全仓 grep `2.4699` 只命中 analysis.md · **最小修**：改 2.4699 → 2.4684
- **B3**（算术等式不成立）· `analysis.md:140`（§6.1）· 「2,101,248 = 2·(4−2)·256²·8 + 偏置差(**1,024**)」：2,097,152 + 1,024 = 2,098,176 ≠ 2,101,248；正确余项 = **4,096** = (4−2)·256·8（`lin1` 偏置项；`lin2` 偏置 8d = 2,048 与 r 无关）· 证据：`results.json:81-90`（6,962,688 / 4,861,440）、run.py:145 `ffn_params = Σ(2·r·d² + r·d + d)` · **最小修**：1,024 → 4,096（并可顺手加一句 2,048 的 lin2 偏置不随 r 缩放，见 N2）

### normal
- **N1** · `analysis.md:86`（§3-P6 行）与 `analysis.md:145`（§5.4）「窄层 0.30–0.35、宽层 0.49–2.08」与落盘 `liveness.ffn_out_rel_mag_by_layer` 不符：全臂最小值 0.299338（`results_APYR_s1.json:165`，< 0.30）、PYR-s0 第 8 层（r=1，窄）0.366063（`results_PYR_s0.json:171`，> 0.35）、而 APYR 第 1 层（r=1，窄）1.299022/1.321126（`results_APYR_s1.json:164`/`results_APYR_s0.json:164`）**大于**除首层外的所有宽层 ⇒ 该括注把"幅度随宽度"讲反了，实际主导因素是**层号**（首层幅度在所有臂都最大）。实质声称（逐层幅度全为正 ⇒ 干预活、窄层非死路径）成立（最小 0.299338 > 0）。**最小修**：改为「全臂逐层幅度 ∈ [0.299, 2.081]，首层最大；窄层（r=1）除 APYR 首层外均落 0.299–0.366」
- **N2** · `analysis.md:116`（§5.2）「U4 恰为其两倍」· Σr（32）与每 token FFN MACs（4,194,304 = 2×2,097,152）确实恰两倍，但 **FFN 参数不是**：4,204,544 ≠ 2×2,103,296 = 4,206,592（差 2,048 = 8 层 × d=256 的 `lin2` 偏置，不随 r 缩放）；等预算三臂之间的"逐位相等"不受影响 ✓·另：`2641/README.md` §3 表把「FFN 参数（解析）」列成 2,097,152/4,194,304（不含偏置），而 `results.json::cost.ffn_params_analytic` 是 2,103,296/4,204,544 ⇒ **同一名"解析"量在契约表与分析表里取值不同** · **最小修**：README 该列表头改「FFN 权重量（不含偏置）」，§5.2 补「（FFN 参数含每层 lin2 偏置 8d，故非整倍）」
- **N3** · `analysis.md:99`…（§1 表内）「nvidia-smi memory.free **前后**读数逐臂见 `run_log.txt`」· `train_arms.sh:44-46` 只打 `free before`；`run_log.txt:3,7,11,15,19,23,27` 均有 before、**无 after** · **最小修**：删"前后"或写"取锁前读数"
- **N4** · `analysis.md:31`（§0）与 `analysis.md:99`（§4）报"中间深度"读数时未标 seed，实际是 **seed 0**：PYR-s0 L6 1.9727 vs U2-s0 2.3338（Δ = −0.3611），而 s1 = 1.9729 vs 2.0799（Δ = **−0.1070**，同向但小 3.4×）；APYR L6 相对 U2 为 +0.3871/+0.3885（两 seed 稳定）· **最小修**：标「(seed 0)」或给两 seed
- **N5** · `2641/README.md` §8-T1 与 `analysis.md:110`（§5.1）把「切片实现与引擎原生窄 FFN **实现等价**」归功于 T1；但 `test_2641_ffn_ratio.py:50-56` 先 `load_state_dict` 覆盖全部参数 ⇒ 该断言由「形状相同 + 同类同前向」蕴含，**保形的切片错误（取后缀/取偶行）会通过 T1**（T1a 自己的 docstring :124-127 已承认；T3 的 5 个变异里也只有 T3a 指向 T1，而它变的是激活接线）· 证据链本身完整（T2 前缀测试 :88-98 钉住前缀语义）⇒ 属**归属/措辞**缺陷 · **最小修**：README §8/analysis §5.1 改为「T1 钉形状与接线、T2 钉前缀语义，二者合成给出等价性」，或让 T1 在 `load_state_dict` **之前**比较同 seed 的切片模型与原生窄模型
- **N6** · `project.yaml::conclusion` ② / `::falsification` 用无限定的一般式「逐层 FFN 分配在本尺度不是独立于 FFN 总量的设计维度」，而 `::failure_analysis`、`::gaps`、`analysis.md:156` 都限定「只覆盖同多重集 1120 种排序里的两条单调阶梯」· **最小修**：头部 `conclusion` 补「（限两条单调阶梯；1120 种排序未搜索）」，防 3.1.2 误继承
- **N7** · `MODEL.md:3`「`code_frozen_during_training = True`（各臂记录的 `run.py` 哈希逐位相同）」· 该等号对 **U4-s0/U2-s0 不成立于"训练时"**：它们训练时加载的是 `49f697f` 版（差 1 个派生字段），记录的 `cdf48604…` 是训练后 `--no-train` 重评时写入的（TICKET.md:143-157 已如实登记，analysis §8 也登记）· 事实文档的这句是无条件断言 · **最小修**：加「（U4-s0/U2-s0 的哈希由训练后重评写入，见 analysis §8）」

### info
- **I1** · `run.py::aggregate` 的 `same_sign` 只用 `delta_up`（run.py:334-336），而 P4 判据同时覆盖 Δ_down；本数据两向都同号 ⇒ 无错报，但字段名比判据窄 · 建议改名或同时纳入两向
- **I2** · `analysis.md:111`（§5.1）引「本次核验脚本」做初始化控制，但实验目录内**没有**该脚本（`ls` 无对应文件）⇒ 该断言的**证据工件缺失**（断言本身可核、且被 T2/T3 部分覆盖）· 建议补 5 行 `check_init_control.py` 或直接引用 T2/T3
- **I3** · `MODEL.md` 哈希表「字节」列对非 `model.pt` 文件全空 · 建议删列或补全
- **I4** · `2641/README.md` §12 未列 `python3 -m pytest`；`analysis.md` §9 有 ✓（仅一致性）

---

## 3 交 supervisor 的必跑命令（我未跑，未跑项不得算通过）

```bash
cd axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m
python3 -m pytest test_2641_ffn_ratio.py -q                       # 期望 13 passed（gates_report.txt 自报过，需复核者实跑）
python3 -c "print(round(1.5611-1.5750,4), round(1.5709-1.5611,4))"  # -0.0139 0.0098  → 证明 B1 的"小于 0.0098"为伪
python3 -c "print(2*2103296, 4204544, 2101248-2*(4-2)*256**2*8)"   # 4206592 4204544 4096 → B3/N2
# T5 初始化控制：见 §T5 命令块
git diff --stat 49f697f..e6d2db1 && git status --short            # 越界扫描（应只碰 axes/3-width/** 与 .scratch/64-axis3-width/**）
python3 tools/summary.py --check                                   # 需 OK
sha256sum axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/{README.md,train_arms.sh,test_2641_ffn_ratio.py}
stat -c '%y %n' axes/3-width/3.1.4-ffn-ratio/experiments/2641-ffn-ratio-frontier-13m/{README.md,train_arms.sh,test_2641_ffn_ratio.py}   # mtime 是否全早于 18:00:37
```
（T2 的"自写脚本重算"我已用读文件+手算替代并给出逐项结果；T1/T5 的**执行**必须由具备 shell 的人补跑。）

---

## 4 我应该写的报告文件

`.scratch/64-axis3-width/REVIEW-1.md` —— **我无写文件工具**（只有 read/grep/find/ls/watchdog_diff），故未创建；以上全文即报告内容，请 `axis3-lead` 原样落盘（首行 `VERDICT: FAIL（窄范围）`）。我在本次复核中**未修改任何被审产物**。

修完 B1–B3（可一并处理 N1–N7）后，请按项目制度派**新的** fresh 复核（复核者不得是修改者；`project.yaml::role_map.reviewer` 亦如此规定）。

```acceptance-report
{
  "criteriaSatisfied": [
    {
      "id": "criterion-1",
      "status": "satisfied",
      "evidence": "复核报告全文随本回复返回：首行 VERDICT: FAIL（窄范围）、T1–T11 逐项结论（含每项的我方核验方式与精确命令）、发现 B1–B3/N1–N7/I1–I4（各带 file:line 与复现方式）、工具限制 L1–L5 明确声明未跑项；因无写文件工具，报告未落盘 .scratch/64-axis3-width/REVIEW-1.md，由 axis3-lead 落盘"
    }
  ],
  "changedFiles": [],
  "testsAddedOrUpdated": [],
  "commandsRun": [
    {
      "command": "pytest test_2641_ffn_ratio.py -q",
      "result": "not-run",
      "summary": "无 shell/执行工具（L1）；静态旁证 gates_report.txt 记 exit=0 / 13 passed，且我独立数出 13 个用例，但未执行"
    },
    {
      "command": "自写脚本从 results_<arm>_s<seed>.json 重算 P1/P2/P4",
      "result": "not-run",
      "summary": "未执行代码；改用 read 取 7/7 臂 clean_bpc + 手算，四个 Δ、0.014 判定、同号判定与 results.json 逐位一致"
    },
    {
      "command": "build_model 初始化控制（trunk 逐位 + FFN 前缀）",
      "result": "not-run",
      "summary": "未执行；给出可判定的静态论证（run.py:120-131/96-119、train_depth_ar.py:771,790）与被 T2/T3 测试覆盖的部分，并交 supervisor 精确命令"
    },
    {
      "command": "git diff --stat 49f697f..e6d2db1 / git status --short / summary.py --check / sha256sum / stat mtime",
      "result": "not-run",
      "summary": "无 git/exec 访问；仅 watchdog_diff 证明工作树 == e6d2db1 且无本会话改动。越界扫描、契约哈希与 mtime、summary 校验均未核，不算通过"
    }
  ],
  "validationOutput": [
    "clean_bpc 逐臂（各 results_*.json:68）：U4-s0 1.5314 / U2-s0 1.5709 / U2-s1 1.5611 / PYR-s0 1.5730 / PYR-s1 1.5750 / APYR-s0 1.5644 / APYR-s1 1.5601",
    "手算 Δ_up = −0.0021(s0) / −0.0139(s1)；Δ_down = +0.0065 / +0.0010 —— 与 results.json::P1_P2_deltas_by_seed 逐位一致；|Δ| 全 < 0.014 ⇒ H0；两向同号 ⇒ P4 true",
    "等预算：sum_r 16/16/16、macs_ffn 2,097,152 ×3（U4 4,194,304 恰两倍）、params_ffn 2,103,296 ×3（U4 4,204,544，非整两倍，差 2,048）；ffn_params_analytic == params_ffn 7/7",
    "逐层 FFN MACs == 2·r·256²（PYR [524288,393216,262144,262144,262144,131072,131072,131072]，APYR 镜像，U4 全 524288）",
    "7/7 runs/*/results.json 的 config（26 字段）逐字段相同、train 块仅 seed 不同；train_tokens 98,304,000 ×7；vocab 206；code_hash_run_py 7/7 相同",
    "契约数字不符 3 处：analysis.md:32「小于 0.0098」为伪（0.0139>0.0098）、analysis.md:99「2.4699」应为 2.4684、analysis.md:140「偏置差 1,024」应为 4,096",
    "run_log 自洽：起跑 18:00:37，7 臂锁取放成对、臂间恒 61 s，wall_minutes 6.73–7.76，峰值 771–935 MB"
  ],
  "residualRisks": [
    "我无 shell：T1（pytest 13 passed）、T5（build_model 逐位初始化控制）、T6（三份契约文件与 49f697f 的内容哈希 + README mtime < 18:00:37）、T9（git diff --stat 越界扫描、git status、tools/summary.py --check）均未执行，未跑项不得计为通过；已列出精确命令",
    "T11(a) 反假设（共享前缀初始化压住分配效应 / 换通道子集可能出现效应）不能由现有证据排除，只能给出 ≲1e-2 的量级上界；若要让「分配不是独立维度」成为一般结论，需追加通道置换对照臂或提高 seed 数（≈15–30 min GPU）",
    "T11(b) 成立：PYR-s1 |Δ|=0.0139 ≈ 项目自认种子散布上界 0.0138，不能排除幅度约 0.014 的真实（负向）效应；预登记阈值规则仍给 H0，决策（不采纳非均匀分配）不受威胁，但「分配效应被种子噪声吃掉」这句论证必须删改（B1）",
    "P5 的两个历史锚里，1.5257/1.5298 只在 .scratch 工单/他人复核件中可查到（二手）；我直接核到锚文件的是 1.5287（.scratch/23-depth-ar/evidence/23d-dense-B-12k/results.json:85,89）",
    "修完 B1–B3（建议一并处理 N1–N7）后，按项目制度需派新的 fresh 复核，复核者不得是修改者"
  ],
  "noStagedFiles": true,
  "diffSummary": "本次复核为只读：未修改、未暂存任何文件（watchdog_diff 显示工作树相对 e6d2db1 无 staged/unstaged 变更）；被审交付 = revision e6d2db1 的 2641 实验目录 + axes/3-width/3.1.4-ffn-ratio/project.yaml",
  "reviewFindings": [
    "blocker: analysis.md:32（并复制到 project.yaml::conclusion/failure_analysis、analysis.md:153、:94）- 「四个等预算 Δ 小于同配置跨 seed 散布 0.0098」为伪，PYR-s1 |Δ|=0.0139>0.0098，承载「分配效应被种子噪声吃掉」的论证用错数",
    "blocker: analysis.md:99 - APYR 第 6 层 bpc 写成 2.4699，落盘为 2.4684（results_APYR_s1.json:54）；2.4699 在任何结果文件中都不存在",
    "blocker: analysis.md:140 - 「参数差 2,101,248 = 2·(4−2)·256²·8 + 偏置差(1,024)」等式不成立，余项应为 4,096",
    "normal: analysis.md:86/:145 - liveness 幅度括注「窄层 0.30–0.35、宽层 0.49–2.08」与落盘不符（min 0.299338；PYR-s0 窄层 0.366063；APYR 首层窄层 1.299/1.321 大于多数宽层），且实际主导因素是层号",
    "normal: analysis.md:116 与 2641/README.md §3 - 「U4 恰为其两倍」对 FFN 参数不成立（4,204,544 vs 4,206,592，差 2,048）；README 的「FFN 参数（解析）」与研究项落盘的 ffn_params_analytic 同名不同值",
    "normal: analysis.md:99（§1 表内） - 声称 run_log.txt 有 nvidia-smi memory.free 前后读数，实际只有 before",
    "normal: analysis.md:31/:99 - 中间深度「1.9727 vs 2.3338」等为 seed 0 值未标注（s1 为 1.9729 vs 2.0799，同向但小 3.4×）",
    "normal: 2641/README.md §8 与 analysis.md:110 - 把「实现等价」归功于 T1，而 T1 经 load_state_dict 覆盖参数后对保形切片错误不敏感（前缀语义实际由 T2 承载）",
    "normal: project.yaml::conclusion/falsification - 头部结论用无限定一般式，未带 failure_analysis/gaps 里「只覆盖两条单调阶梯（1120 种排序未搜索）」的限定",
    "normal: MODEL.md:3 - code_frozen_during_training=True 对 U4-s0/U2-s0 无条件成立不实（其哈希由训练后重评写入）",
    "info: run.py:334-336 - P4_same_sign_across_seeds 只由 delta_up 计算，未含 delta_down（本数据无错报）",
    "info: analysis.md:111 - 引用的「本次核验脚本」未入库，初始化控制断言的证据工件缺失",
    "info: MODEL.md 哈希表字节列为空；2641/README.md §12 未列 pytest"
  ],
  "manualNotes": "总判：实验本体（等 FFN 预算逐位成立、7 臂配置一致性、初始化前缀控制的构造性论证、门/变异的存在性与数目自洽、run_log 锁与墙钟自洽、口径无墙钟、结论按预登记阈值规则成立）经独立静态核验通过；判 FAIL（窄范围）只因 analysis.md/project.yaml 里 3 处数字与落盘证据不符（其中 B1 是承载论证用错数，且与同一份文档同句写的 −0.0139 自相矛盾），修完为纯文字/数字改动，无需重训或重评。另：T11(a)(b) 两条反假设都无法被现有证据排除，结论措辞须从「分配不是独立维度」弱化为「在本尺度与本次抽样下不可排序，不能排除 ≈0.01–0.014 量级效应（方向为负或零）」。我无 shell（T1/T5/T6/T9 的执行部分未跑，未跑项不得算通过，命令清单见报告 §3）且无写文件工具，故 .scratch/64-axis3-width/REVIEW-1.md 未由我创建，请原样落盘以上全文。"
}
```