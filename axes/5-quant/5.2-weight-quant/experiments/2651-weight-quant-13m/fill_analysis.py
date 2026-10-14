#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 `results.json` 生成 `analysis.md`（判读）。

- 本文件是**唯一**写 `analysis.md` 的入口：人话摘要、表、三道对账、失败与局限、下一步。
- 所有数字都从 `results.json` **注入**，本文件不持有任何数字；判决只按预登记的机械规则
  （`max|配对 Δ|` vs `0.014`、`dominates_*`、`at_floor`）读，不做额外判断。
- 兜底：run 未落盘时（本脚本被 watcher 调用），由 watcher 写 `blocked_lock.json`；本脚本拒绝在
  缺少 `results.json` 时写任何东西（fail loud）。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def f(x, n=6):
    return "—" if x is None else f"{x:.{n}f}"


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results.json")
    if not os.path.exists(path):
        print(f"[fail] 缺 {path}：不写任何东西（fail loud）")
        return 2
    rec = json.load(open(path))
    if rec.get("partial", {}).get("partial"):
        print(f"[fail] {path} 是半成品（partial=true，已完成 "
              f"{rec['partial'].get('arms_completed')}）：拒绝据此写 analysis.md（fail loud）")
        return 3
    cfg = rec["config"]
    pts = rec["points"]
    by = {p["arm"]: p for p in pts}
    base = by["fp16"]
    cal = rec.get("caliber", {})
    #: 第二个位置参数 = 输出路径（默认本目录 analysis.md）；仅供自测时写到别处。
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "analysis.md")
    is_cpu = bool(cal.get("is_cpu"))
    inst = rec["instrument"]
    band = rec["band"]
    paired = rec["paired"]
    fa = rec["frontier_analysis"]
    quant = [a for a in cfg["arms"] if a != "fp16"]

    L = []
    L.append("# 2651 · 权重量化最小判决（含标尺字节账）—— 判读（`analysis.md`）")
    L.append("")
    L.append("> 本文件是**当前结论**；契约与判据线在 `README.md` / `config.json::criterion`（运行前写入）；"
             "数字在 `results.json`；表在 `table.md`；过程流水在 `JOURNAL.md`。"
             f"本文件由 `fill_analysis.py` 从 `results.json` 生成（设备口径：`{cal.get('device')}`）。")
    L.append("")

    # ---------------- 人话摘要 ----------------
    L.append("## 0. 人话摘要")
    L.append("")
    L.append("**这条研究在问什么**：同一个 13.5M 的普通解码器变压器（d256/12 层/8 头、只做最后一层"
             "语言建模损失、权重是冻结的），把它的**权重**从 fp16 改存成 int8 或 int4（并额外存标尺，"
             "即每个量化块换算回真实数值用的一个 fp16 系数），质量会掉多少、到底省不省内存？KV 缓存、"
             "激活、偏置、归一化和输出读出**逐位不动**，所以这只是一次单因素测量。")
    L.append("")
    L.append("**为什么要问**：权重量化是部署侧最常被默认的省内存手段，「int8 基本不损、int4 有损」"
             "常被当成经验。我们不做 GPTQ/AWQ 那类二阶补偿，只测**最朴素的对称分组假量化**在这个尺度上"
             "的真实代价，并且把「标尺本身也要占字节」算清楚——标尺不是免费的，这是最常见的账目陷阱。")
    L.append("")
    bpp = {a: by[a]["byte_account"]["scope_linear_all"]["bytes_per_param"] for a in cfg["arms"]}
    L.append("**该跌的（每参数部署字节，含标尺）**："
             + "；".join(f"`{a}` {f(bpp[a])} 字节/参数（÷fp16 = ×"
                         f"{f(by[a]['byte_account']['scope_linear_all']['to_fp16_multiple'], 4)}）"
                         for a in cfg["arms"])
             + "。也就是说 int8 大约把权重面减半、int4 大约减到三分之一（不是四分之一——标尺要钱）。")
    L.append("")
    L.append("**该不跌/可排序的质量代价（clean bpc，越低越好；判定口径 = 5 个窗口上的配对 Δ，"
             "带 = 0.014 bpc）**：" + "；".join(
                 f"`{a}` 与 fp16 的最大配对 Δ = {f(paired[a]['max_abs_delta_bpc'])} ⇒ "
                 f"**{paired[a]['verdict']}**" for a in quant) + "。"
             + (f" `int4_group32` 的越带**只刚越**：`max|配对 Δ| = "
                f"{f(paired['int4_group32']['max_abs_delta_bpc'])}`，超出带 "
                f"{paired['int4_group32']['max_abs_delta_bpc'] - band['in_force_band_bpc']:+.6f}；"
                f"判定用逐窗 `max|Δ|` 聚合（非均值，见 `config.json::criterion`），"
                f"5 窗 Δ 均值 = {f(paired['int4_group32']['mean_delta_bpc'])} **落在带内** ⇒ 该越带判定依赖 max 聚合。"
                if paired["int4_group32"]["verdict"] == "越带" else ""))
    L.append("")
    L.append("**该保持的（合成 copy 召回，把前面出现过的符号串原样搬回来；chance = "
             f"{by['fp16']['chance_uniform']:.4f} = 1/64）**："
             + "；".join(f"`{a}` {p['recall_copy']['accuracy']:.4f}"
                         f"{'（在地板）' if p['recall_copy']['at_floor'] else ''}"
                         for a, p in ((a, by[a]) for a in cfg["arms"]))
             + f"；fp16 的局部先验（只看最近 8 个 token）= {rec['controls']['local_prior_copy']['accuracy']:.4f}，"
               "毁源对照（把要搬的源片段整段换随机符号）见 §3。")
    L.append("")
    # 结论一句话（机械映射，照 README §10 的失败决策树）
    v = {a: paired[a]["verdict"] for a in quant}
    concl = []
    if v["int8_per_channel"] == "不越带" and v["int8_group32"] == "不越带":
        concl.append("**int8 在这一档基本是白赚**：字节减半左右，clean bpc 的配对代价落在 0.014 的噪声带内"
                     "（该带内不可排序）。")
    else:
        concl.append("**int8 也越带**：按契约的失败决策树，判「该尺度该项无收益（在给定字节下）」，如实报、不辩解。")
    if v["int4_group32"] == "越带":
        concl.append("**int4（每 32 个值一把标尺）掉出噪声带**：按契约判「int4 在本规模有损」，回退 int8；"
                     "它省到 " + f(bpp["int4_group32"]) + " 字节/参数，代价是 clean bpc 掉 "
                     f"{by['int4_group32']['bpc'] - base['bpc']:+.6f}（相对 fp16）。")
    else:
        concl.append("**int4（每 32 个值一把标尺）也没越带**：本尺度上它的质量代价不可排序，"
                     "但注意召回腿的读数与限制（§3/§5）。")
    concl.append("**值不值**：把「字节」当钱、把「clean bpc」当货，"
                 + "；".join(
                     f"`{a}` 在 {f(bpp[a])} 字节/参数上"
                     f"{'字节更省且质量不可分辨（**不构成严格支配**）' if fa[k]['dominates_fp16'] else '**不支配**'} fp16"
                     for a, k in (("int8_per_channel", "int8_per_channel_vs_fp16"),
                                  ("int8_group32", "int8_group32_vs_fp16"),
                                  ("int4_group32", "int4_group32_vs_fp16")))
                 + "（`dominates` = 字节不多于 fp16 **且**质量代价落在 0.014 带内、即**不可分辨**，"
                   "**不构成 Pareto 严格改进** ⇒ int4 因越带而不支配 fp16，尽管它最省）；"
                 + f"`int8_per_channel` vs `int8_group32`：前者在 {f(bpp['int8_per_channel'])} "
                   f"字节/参数上字节更省且质量不可分辨（`dominates_group32 = "
                   f"{fa['int8_per_channel_vs_int8_group32']['dominates_group32']}`）"
                   "——本模型行长远大于 32 ⇒ 每行一把标尺比每 32 值一把更省，且质量代价都在带内。")
    L.append("**结论一句话**：" + "".join(concl))
    L.append("")
    L.append("> 术语：`bpc` = 每字符比特数（越低越好）；`标尺（scale）` = 把量化整数换算回实数的"
             "一个 fp16 系数；`配对 Δ` = 同一批评测窗口上 `bpc_臂 − bpc_fp16`（正 = 量化更差）；"
             "`越带` = `max|配对 Δ|` 超过 0.014，即「在本尺度确有可排序的质量代价」（聚合 = 逐窗 `max|Δ|`，"
             "非均值，见 `config.json::criterion`）；"
             "`dominates` = 字节不更多、且质量代价落在 0.014 带内（即**质量不可分辨**，**不构成 Pareto 严格改进**）。")
    L.append("")

    # ---------------- 表 1 前沿主表 ----------------
    L.append("## 1. 前沿主表（每参数部署字节 → 两腿质量）")
    L.append("")
    L.append("| 臂 | 每参数字节（值+标尺） | ×fp16 | 省幅 | clean bpc | Δbpc vs fp16 | "
             "配对 max&#124;Δ&#124;（" + str(len(band["seeds"])) + " 窗） | 与 0.014 带 | "
             "copy 召回 | Wilson 95% | 在地板 | induction | var_recall | 前缀不变（未来/同批） | "
             "changed_frac | rel_err(L1) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in pts:
        dep = p["byte_account"]["scope_linear_all"]
        cp = p["recall_copy"]
        ci = cp["wilson95"]
        L.append(f"| `{p['arm']}` | {f(dep['bytes_per_param'])} | ×{f(dep['to_fp16_multiple'], 4)} | "
                 f"{dep['reduction_vs_fp16']:.1%} | {f(p['bpc'])} | {p['bpc'] - base['bpc']:+.6f} | "
                 f"{f(paired[p['arm']]['max_abs_delta_bpc'])} | **{paired[p['arm']]['verdict']}** | "
                 f"{cp['accuracy']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
                 f"{'是' if cp['at_floor'] else '否'} | {p['recall_induction']['accuracy']:.4f} | "
                 f"{p['recall_var_recall']['accuracy']:.4f} | "
                 f"{p['prefix_invariance']['future_token_past_logits_equal']}/"
                 f"{p['prefix_invariance']['peer_request_logits_equal']} | "
                 f"{p['activity']['changed_frac']:.4f} | {p['activity']['rel_err']:.6f} |")
    L.append("")
    L.append("> 读法：横轴是**部署口径**（值字节 + 标尺字节，标尺 fp16 存储 2 B/把，按参数数加权，"
             "落在全部 60 个 `nn.Linear` 的权重矩阵上）。"
             + ("实测显存峰值 = "
                f"{rec['memory_convention']['measured_gpu_peak_mib']} MiB，**不随档下降**"
                "（实现把假量化权重现算成 fp32 再前向）⇒ 显存只作过程读数。"
                if rec['memory_convention']['measured_gpu_peak_mib'] is not None else
                "显存**未测**：本 run 是 CPU 口径 ⇒ `measured_gpu_peak_mib = null`（不产生 CUDA "
                "显存读数，也不拿别的 run 的读数顶替）；峰值不随档下降是**实现层**性质（假量化权重"
                "现算成 fp32 再前向），本批未测显存 ⇒ 只报解析列。")
             + "判据横轴只用解析字节账（值 + 标尺）。"
             f"`fp16` 臂的 clean bpc = {f(base['bpc'])}，与历史锚（GPU 释放版）"
             f"{inst['anchor_bpc']:.6f} 的差 = {inst['fp16_identity']['abs_diff_vs_anchor']:.2e}"
             f"（device={cal.get('device')}）。")
    L.append("")
    L.append("**字节账逐矩阵**（`scope_linear_all` 的 `per_matrix`，共 "
             f"{len(pts[0]['byte_account']['per_matrix'])} 个矩阵）与三个口径的旁列在 `table.md` 表 2；"
             "`results.json::points[*].byte_account` 是逐矩阵机器可读的权威。按参数数加权的总账："
             + "；".join(f"`{a}` 标尺个数 {by[a]['byte_account']['scope_linear_all']['n_scales']}、"
                         f"标尺字节/参数 {by[a]['byte_account']['scope_linear_all']['scale_bytes'] / max(by[a]['byte_account']['scope_linear_all']['n_params'], 1):.6f}"
                         for a in cfg["arms"]) + "。")
    L.append("")

    # ---------------- 表 2 配对 Δ ----------------
    L.append("## 2. 配对 Δbpc 逐窗口（判据带 = "
             f"{band['in_force_band_bpc']:.3f} bpc，口径 = 配对 Δ、"
             f"{len(band['seeds'])} 窗口）")
    L.append("")
    L.append(f"窗口 = `window_seed ∈ {band['seeds']}`（每窗 "
             f"{cfg['eval']['paired_windows_n_batches']}×{cfg['eval']['batch_size']}）；"
             "`Δ = bpc_臂 − bpc_fp16`（**同一批窗口**）。")
    L.append("")
    L.append("| 臂 | bpc 逐 seed | Δ 逐 seed | Δ 均值 | Δ 区间 | max&#124;Δ&#124; | 判定 |")
    L.append("|---|---|---|---|---|---|---|")
    for p in pts:
        a = p["arm"]
        pp = paired[a]
        L.append(f"| `{a}` | {', '.join(f(x) for x in pp['bpc_by_seed'])} | "
                 f"{', '.join(f'{x:+.6f}' for x in pp['delta_by_seed'])} | "
                 f"{pp['mean_delta_bpc']:+.6f} | [{pp['min_delta_bpc']:+.6f}, {pp['max_delta_bpc']:+.6f}] | "
                 f"{f(pp['max_abs_delta_bpc'])} | **{pp['verdict']}** |")
    L.append("")
    L.append(f"> **主判**（预登记）：每个量化臂的 `max|配对 Δ|` vs "
             f"{band['in_force_band_bpc']:.3f}；≤ 带 ⇒ **不越带**（该带内质量代价不可排序），"
             f"> 带 ⇒ **越带**（本尺度确有可排序的质量代价）。"
             f"本批窗口数 = {band['windows']}，标准口径（≥5 窗口）"
             f"{'满足' if band['standard_met'] else '**不满足**'} ⇒ "
             f"{'在力带 = 0.014 bpc' if band['standard_met'] else '在力带不是 0.014（须按预登记改用单臂跨窗口极差）'}。"
             f"另报的描述性读数（**另一口径、不参与判决**）单臂跨窗口 bpc 极差 = "
             + "；".join(f"`{a}` {rec['single_arm_range_bpc'][a]:.6f}" for a in cfg["arms"]) + "。")
    L.append("")

    # ---------------- 表 3 召回 ----------------
    L.append("## 3. 第二腿：合成长程召回（率类指标只走 Wilson CI，不借 bpc 带）")
    L.append("")
    L.append("| 臂 | copy 汇总 | copy@src0 | copy@src96 | copy@src191 | copy 毁源对照 | "
             "induction | var_recall | chance=1/64 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p in pts:
        ps = p["recall_copy"].get("per_setting") or p["recall_copy"].get("by_position") or {}

        def _at(k):
            d = ps.get(str(k))
            return f"{d['accuracy']:.4f}" if d else "—"

        L.append(f"| `{p['arm']}` | {p['recall_copy']['accuracy']:.4f} | "
                 + " | ".join(_at(k) for k in cfg["tasks"]["copy"]["src_positions"])
                 + f" | {p['recall_copy_corrupt']['accuracy']:.4f} | "
                 f"{p['recall_induction']['accuracy']:.4f} | "
                 f"{p['recall_var_recall']['accuracy']:.4f} | *已知 chance* |")
    L.append("")
    L.append("> 读法：copy 主腿 = 把 `src_pos` 处的 32 个符号原样搬到末尾；`copy 毁源对照` 是同一任务"
             "但源片段整段换随机符号——它说明读数真来自被搬运的信息，而不是局部统计。"
             "若某臂 `at_floor = true`，读作「在该任务上不可分辨」，**不得**写成「机制无效」。"
             f"本次局部先验（只学最近 8 个 token）：copy {rec['controls']['local_prior_copy']['accuracy']:.4f}、"
             f"induction {rec['controls']['local_prior_induction']['accuracy']:.4f}、"
             f"var_recall {rec['controls']['local_prior_var_recall']['accuracy']:.4f}。")
    L.append("")

    # ---------------- 三道对账 ----------------
    L.append("## 4. 三道对账")
    L.append("")
    g = rec["gates"]
    L.append("### 对账① 字节账 ↔ 实现（G3/G4）")
    L.append("")
    L.append("`int8_per_channel` 标尺个数 = `n_out`（每行一把）；`group G` 标尺个数 = `n_out·n_in/G`"
             "（要求 `G | n_in`）；两者在 `wq_arms.activity_of` 内**逐矩阵断言**，对不上直接报错、不出数。"
             f"本次四个臂 `scales_match_analytic` 全 = "
             f"{all(p['activity']['scales_match_analytic'] for p in pts)}，取整误差界（`max_abs_err ≤ 0.5·s_max`）"
             f"违例 = {sum(len(p['activity']['bound_failures']) for p in pts)} 个矩阵；"
             f"G3+G4 门禁 = {g['G3_G4_byte_account_and_bound'] if isinstance(g['G3_G4_byte_account_and_bound'], bool) else {a: v['pass'] for a, v in g['G3_G4_byte_account_and_bound'].items()}}。")
    L.append("")
    L.append("### 对账② `fp16` 臂 ↔ 历史锚 / 独立路径（G2）")
    L.append("")
    fi = inst["fp16_identity"]
    L.append(f"- 主路径 `bpc_of` = {f(fi['bpc_of'])}，独立路径 `retention.masked_bpc` = "
             f"{f(fi['retention_masked_bpc'])}，**逐位相等 = {fi['bitwise_equal']}**"
             f"（本 run 的 device = `{cal.get('device')}`）。")
    L.append(f"- 与 GPU 历史锚 {inst['anchor_bpc']:.6f} 的差 = **{fi['abs_diff_vs_anchor']:.2e}**"
             f"；本口径下「跨设备比锚」是否在力：`{g['G2_nonquant_identity'].get('anchor_caliber_in_force')}`"
             f"（CPU 口径下门禁只要求两条路径逐位相等；锚比较作为**并列读数**报告）。"
             f"若该差 < 1e-6，说明 CPU 与 GPU 口径在本模型/本窗口上事实上重合。")
    L.append(f"- `fp16` 臂的 logits 与 stock 释放前向**逐位相等 = {fi['release_logits_bitwise_equal']}**；"
             "`fp16` 臂**不装任何补丁**（"
             f"`patch_reversibility.fp16.logits_changed_when_patched = "
             f"{inst['patch_reversibility']['fp16']['logits_changed_when_patched']}`）。")
    L.append(f"- 量化臂装上真的改 logits、拆掉**逐位还原**："
             + "；".join(f"`{a}` 改={inst['patch_reversibility'][a]['logits_changed_when_patched']}"
                         f"/还原={inst['patch_reversibility'][a]['logits_restored_bitwise']}"
                         for a in quant)
             + f"；补丁无残留 = {inst['patch_reversibility'][quant[0]]['no_forward_attr_left']}。")
    L.append(f"- `gates.*` 死权重：清零后 logits 逐位不变 = "
             f"{inst['dead_gate_probe']['zeroed_logits_bitwise_equal']}，还原逐位 = "
             f"{inst['dead_gate_probe']['restored_bitwise']}（{inst['dead_gate_probe']['n_dead_modules']} 个模块、"
             f"{inst['dead_gate_probe']['dead_param_count']} 个参数）⇒ 它们只进字节账、不进质量读数。")
    L.append("")
    L.append("### 对账③ 配对 Δ ↔ 带")
    L.append("")
    L.append(f"逐臂 `max|配对 Δ|`：" + "；".join(
        f"`{a}` {f(paired[a]['max_abs_delta_bpc'])}"
        f"（{'越带' if paired[a]['max_abs_delta_bpc'] > band['in_force_band_bpc'] else '不越带'}，"
        f"逐窗符号 {['+' if x > 0 else '-' for x in paired[a]['delta_by_seed']]}）" for a in quant)
        + f" ｜ 在力带 = {band['in_force_band_bpc']:.3f} bpc，口径 = 配对 Δ、{band['windows']} 窗口。"
          "**注意**：单臂跨窗口 bpc 极差（§2 末）是**另一口径**，与 0.014 不得混用。")
    L.append("")

    # ---------------- 失败与局限 ----------------
    L.append("## 5. 失败与局限")
    L.append("")
    L.append("1. **单 seed 冻结 checkpoint**：被测权重是单个 seed 训练出来的 ⇒ 训练侧散布未知，"
             "本实验**不据此宣称结构优劣**，结论只作**这些点**的描述、不外推到别的尺度 / 别的 "
             "`in_features`（`in_features` 更长时 `2/n_in` 更小、`2/G` 不变，本实验不外推该效应）。")
    if is_cpu:
        L.append(f"2. **设备口径（本批）**：本批四个臂**全跑在同一台机器的 CPU 上**（`device=cpu`），"
                 "不取 GPU 锁（SCHEMA §20 不适用：CPU 工作不与 GPU 作业互斥）。"
                 "臂间比较因此是同设备同口径；与 GPU 历史锚的比较是**跨设备**，只作并列读数"
                 f"（实测差 {fi['abs_diff_vs_anchor']:.2e}）。判据（配对 Δ、0.014 带）不受影响："
                 "`batch()` 的取数只依赖 seed/窗口，与设备无关。")
    else:
        L.append("2. **设备口径（本批）**：四个臂同进程、同设备（GPU）跑完，"
                 f"§20 取锁记账见 `results.json::gpu_lock`（wait={rec['gpu_lock'].get('wait_seconds')}s、"
                 f"peak={rec['gpu_lock'].get('peak_mem_mib_this_run')} MiB）。")
    L.append("3. **不做离群保护 / 不做误差补偿**：本实验只做最朴素的对称分组假量化——不含 GPTQ 的"
             "Hessian 补偿、不含 AWQ 的激活感知通道缩放、不含 LLM.int8() 的离群通道留 fp16。"
             "因此 int4 掉得多时，第一个嫌疑是「未做离群保护」，不是「4 bit 本身不行」。")
    L.append("4. **不做组合实验**：权重 × KV × 激活的组合留白是显式声明（`README.md` §2），不是遗漏。")
    L.append("5. **显存**：" + (
        "本批实测峰值 " + f"{rec['memory_convention']['measured_gpu_peak_mib']} MiB 且**不随档下降**"
        "（实现把假量化权重现算成 fp32 再前向）⇒ 显存只作过程读数。"
        if rec['memory_convention']['measured_gpu_peak_mib'] is not None else
        "本批为 **CPU 口径** ⇒ 显存 **未测**（`peak_mem_mib = null`）；"
        "「峰值不随档下降」是**实现层**性质（假量化权重现算成 fp32 再前向），不是本批的读数。")
        + "判据横轴只用解析字节账（值 + 标尺）。")
    L.append("6. **召回是合成任务**：copy / induction / var_recall 是受控合成任务，只说明「在这类把前面"
             "信息原样搬运的任务上是否可分辨」；`at_floor = true` ≠「机制无效」。率类指标不借 0.014 带。")
    L.append("7. **未测的臂**：`int4_per_channel`（解析 0.5 + 2/n_in ≈ 0.508 B/参数）本批**未测**，"
             "见 §6 下一步。")
    L.append("")

    # ---------------- 下一步 ----------------
    L.append("## 6. 下一步（未测项 → 最小实验）")
    L.append("")
    L.append("- **`int4_per_channel`（解析 ≈0.508 B/参数，比 `int4_group32` 还省）**：本批未测。"
             "它回答「int4 的损失是位宽造成的还是标尺粒度造成的」——最小实验 = 加一个臂，"
             "同窗口序跑同一套读数。")
    L.append("- **离群通道保护（LLM.int8() 式）**：若 int4 越带，这是最可能把代价压回带内的改动；"
             "最小实验 = 在 `wq_arms.fake_quantize_weight` 上加一条「amax 超过阈值的通道留 fp16」的"
             "分支，重跑同一套四臂。")
    L.append("- **更细的 int4 分组（G=8/16）**：`int4 group-8` 解析 0.75 B/参数，仍比 int8 省；"
             "回答「标尺粒度能买回多少 int4 的质量」。")
    L.append("- **跨尺度验证**：本结论只在 13.5M / 单 checkpoint 上成立；换 `in_features` 更长的主干"
             "（`2/n_in` 更小、`2/G` 不变）才是外推的必经一步。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("> 生成方式：`python3 fill_analysis.py`（从 `results.json` 注入全部数字；人话与判断映射"
             "由预登记的机械规则固定）。`table.md` 由 `run.py` 直接渲染。过程流水见 `JOURNAL.md`。")
    L.append("")

    out = out_path
    with open(out, "w") as fh:
        fh.write("\n".join(L))
    print(f"[fill] 写 {out} 完成（{len(L)} 行）")
    # 过程流水（JOURNAL 允许日期）：把「判读用哪一版 results.json 生成」记下来，供回溯。
    with open(os.path.join(HERE, "JOURNAL.md"), "a") as jh:
        jh.write(f"- **{__import__('time').strftime('%Y-%m-%dT%H:%M:%S%z')} · `analysis.md` 由 "
                 f"`fill_analysis.py` 从 `results.json` 生成**（device={cal.get('device')}，"
                 f"墙钟读数 {rec['cost']['wall_seconds']}s、"
                 f"fp16 bpc={by['fp16']['bpc']:.6f}、"
                 f"逐臂 max|配对 Δ| = "
                 + " / ".join(f"{a} {paired[a]['max_abs_delta_bpc']:.6f}（{paired[a]['verdict']}）"
                              for a in quant)
                 + "）。\n")
    print("FILL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
