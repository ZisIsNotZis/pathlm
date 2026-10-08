Implemented the M3 wrap-up for `2631-kv-budget-causal`: wrote the judgment doc, and **eliminated the scientific contradiction** by adding a real deployment-caliber (value + scale) byte accounting to the table and answering the "does int8 save memory?" question with numbers, overturning the old price tags where they conflict.

**The contradiction, resolved (the science):**

| 方案 | int8 | int4 | 判读 |
|---|---|---|---|
| 值字节（旧横轴，表里原来那一列） | 6144（−50% 假象） | 3072（−75% 假象） | 把标尺当免费 |
| **(a) 逐值标尺**（严格前缀确定，= 本实验质量数字的同源实现） | **18432**（fp16 的 **1.5×**） | **15360**（**1.25×**） | **量化不省内存，反而多占 50%/25%** |
| **(b) 分块标尺 G=32 + 块内回填**（未实现、未测） | **6528**（省 46.9%） | **3456**（省 71.9%） | 省，但违反前缀确定 ≤ **31 个位置**；staging 0.182 MiB/请求；质量代价未测 |

答案：**「int8 省内存」在部署口径下不成立**（严格因果下是 1.5× fp16）；只有改走「分块标尺 + 块内回填」才成立，且幅度是 46.9%（不是 50%）。10 个量化臂**无一例外**相对同拓扑不量化臂变成 ×1.50(int8)/×1.25(int4)，其中 5 个「无优势」（质量差异不可排序）、5 个「被支配」（还差 0.019–0.054 bpc）。旧账推翻：int8 −50%→+50%、int4 −75%→+25%、GQA×int8 省 87.5%→62.5%（比单用 GQA 的 75% 还差 ⇒ 量化是减分项）；**MQA 纯头共享的 87.5% 成立不变**（不涉及标尺）。

Changed files: 11 (10 in the experiment dir + `tests/test_kv_budget_causal.py`; commit `dec06e7` + 4 message-correction commits `e926971/9a39a45/bb82652/3bd1096/eda1119`).

Validation: `pytest tests/ -q` = **656 passed** (baseline 641 + 15 new); `tools/check_schema.py` = OK; `mutate.py` = **8/8 抓红 + 逐字节还原**（新增 M5–M8 打在 `budget_caliber.py`）；`results.json` 对 `f957f19` 旧字段 **0 处变化**（只新增 `deployment_caliber`/`deployment_bytes`）；`table.md` 主表 27 行去掉新增两列后**逐格一致**；analysis.md 的 27 点表/十对表/块敏感性表/簇表/B2 数全部脚本对账通过。

Open risks/questions:
- **我多写了两个文件**：`README.md`（契约九节，**标注为运行后补录**，判据逐字取自运行前的 `config.json::causal_caliber` + 2621 V1）与 `MODEL.md`（4 个 checkpoint 哈希，本波在磁盘重算全等）。理由是 SCHEMA §3 要求每个实验目录有契约/权重清单，且 `run.py`/`config.json` 原本指向不存在的 `README.md`。若你希望只保留你点名的交付物，可以删（但 `run.py` 的引用要一起改）。
- **(b) 方案需要你/用户裁决**（放弃「过去位置严格前缀确定」是产品级取舍）——我按 `docs/REMEDIATION.md` L1 的「未获批准」处理，**没有**实现它，只在 `analysis.md §9` 记为待决定项，没有停下等你（不阻塞已分配交付物）。
- **旧价签的同步点仍是他人的目录**（我未碰）：`catalog.yaml`（R1 格 `tax/benefit`）、`axes/R-references/R1-quantization/README.md`、`2614/README.md:32`、`2621/README.md:109,167`。
- 召回统计只做了「3 个源位置簇」的簇 bootstrap（存储数据只有这一层粒度）⇒ 区间粒度粗；已在 §6 明说只作稳健性检查。
- 部署内存主张是**解析**的：评测实现立刻反量化，GPU 峰值（462.5 MiB）不随档下降，无法用实测验证。
- 提交信息更正了 4 次（两次是我自己写错命令/计数）。**没有任何代码或证据文件因此改动**；被更正的都是 message 文本。若你要求清理历史（squash），需要你授权（我不会自行改写历史）。
- GPU：`/tmp/pathlm_gpu.lock` **当前被别的线持有**（我未申请也未释放，本任务全程无 GPU 操作）；顺带观察：`2631-carry-reseed`、`2631-spec-three-arms` 目录里有**新鲜在途产物**，说明那两条线正在跑。

Recommended next step: 由主 Agent 决定 (b) 是否立项（立决定 → 加实验：同 27 点 + G 阶梯 + 质量腿），并同步 `catalog.yaml`／R1 格页的三处旧数字；在那之前对外报价只能用「(a) 口径 int8 1.5×、（b）天花板 46.9%/71.9% 且违反前缀确定」这套话。