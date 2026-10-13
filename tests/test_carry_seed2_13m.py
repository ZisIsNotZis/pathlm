"""2633-carry-seed2 · 第三种子补点的**可失败单测**（CPU、秒级）。

靶子 = 本实验自己的语义层 `seed3_stats.py` + SCHEMA §9 的复用证据 `equality_vs_2631.py`
+ `config.json` 的臂定义：

① 臂定义：九个臂 = 三规则 × 三种子（前六个复用 2631，后三个本实验新训）；
② 统计层：种子层样本量估计（判据⑥）的算术、t 临界值、Wilson「塌回地板」比例；
③ **复用证据必须可失败**：实现层逐文件 sha256 必须等于 2631 运行时记录、配置层共享字段
   逐字段相等、训练配方 flag 只差 `--seed`/`--out-root` —— 任一不满足即 `reuse_rejected`
   （本组测试直接断言 `verdict == "reuse_ok"`，并另造一个「改了配方」的反例）；
④ `results.json` 存在时（跑完之后）：判据①②③④⑥ 的字段齐备、三 seed 齐备、
   毁源对照阈值口径正确。
"""

import copy
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = os.path.join(ROOT, "axes", "1-time", "1.1.4.1-explicit-memory", "experiments",
                   "2633-carry-seed2")
sys.path.insert(0, ROOT)
sys.path.insert(0, EXP)

import seed3_stats as s3                                           # noqa: E402
import equality_vs_2631 as eqrep                                   # noqa: E402
import reseed_arms as ra                                           # noqa: E402

CFG = json.load(open(os.path.join(EXP, "config.json")))


# ---------------- ① 臂定义 ----------------

def test_arm_grid_is_three_rules_by_three_seeds():
    assert s3.SEEDS3 == (0, 1, 2)
    names = [a["name"] for a in CFG["arms"]]
    assert names == [f"{r}_s{sd}" for sd in (0, 1, 2) for r in ("none", "plain", "delta")]
    assert CFG["new_in_2633"] == ["none_s2", "plain_s2", "delta_s2"]
    assert CFG["reused_from_2631"] == ["none_s0", "plain_s0", "delta_s0",
                                       "none_s1", "plain_s1", "delta_s1"]
    assert s3.arm_name("delta", 2) == "delta_s2"
    # seed=2 是 2631 的 SEEDS 之外的新档：2631 的函数会拒绝，本实验的委派版不会
    with pytest.raises(ValueError):
        ra.arm_name("delta", 2)


def test_arm_names_round_trip_and_delegation():
    for sd in s3.SEEDS3:
        for rule in ra.RULES:
            assert s3.arm_name(rule, sd) == f"{rule}_s{sd}"
    # 重叠档（0/1）必须与 2631 的命名逐字相同（同一口径，不另立一套）
    for sd in ra.SEEDS:
        for rule in ra.RULES:
            assert s3.arm_name(rule, sd) == ra.arm_name(rule, sd)
    with pytest.raises(ValueError):
        s3.arm_name("delta", 3)


# ---------------- ② 统计层（判据⑥） ----------------

def test_seeds_needed_monotone_and_known_value():
    # 算术对账（SCHEMA §12）：σ=24.184pp、Δ=3pp ⇒ n = 2(1.9600+0.8416)²·24.184²/9
    #   = 2·7.8493·584.86/9 = 9182.2/9 ≈ 1020.2 ⇒ ⌈⌉ = 1021
    n = s3.seeds_needed_for_mean_effect(24.184, 3.0)
    assert n == 1021
    # σ 越大、Δ 越小 ⇒ 需要越多
    assert s3.seeds_needed_for_mean_effect(10.0, 3.0) < n
    assert s3.seeds_needed_for_mean_effect(24.184, 1.0) > n


def test_seeds_needed_rejects_bad_input():
    with pytest.raises(ValueError):
        s3.seeds_needed_for_mean_effect(-1.0, 3.0)
    with pytest.raises(ValueError):
        s3.seeds_needed_for_mean_effect(1.0, 0.0)


def test_norm_ppf_matches_known_quantiles():
    assert s3._norm_ppf(0.975) == pytest.approx(1.959964, abs=1e-5)
    assert s3._norm_ppf(0.8) == pytest.approx(0.841621, abs=1e-5)
    assert s3._norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)


def test_mean_ci_t_uses_df2_critical_value():
    out = s3.mean_ci_t([36.478, 35.971, 30.0])
    assert out["n"] == 3 and out["t"] == pytest.approx(4.303)
    assert out["ci95_pp"][0] < out["mean_pp"] < out["ci95_pp"][1]


def test_collapse_wilson_bounds():
    # 三 seed 里 1 个塌回地板
    out = s3.collapse_wilson([False, True, False])
    assert out["k"] == 1 and out["n"] == 3
    assert out["wilson_ci"][0] < 0.333 < out["wilson_ci"][1]
    assert s3.collapse_wilson([True, True])["rate"] == 1.0


def test_judge_seed3_reuses_effect_stable_semantics():
    effects = {}
    for key, vals in (("delta_vs_none", [(sd, 34.9, 34.4) for sd in (0, 1, 2)]),
                      ("delta_vs_plain", [(0, 0.37, -0.15), (1, 34.06, 33.6),
                                          (2, 1.0, 0.2)])):
        per = [{"seed": sd, "effect_pp": v, "paired_boot_ci_pp": [lo, lo + 0.5]}
               for sd, v, lo in vals]
        thr = (ra.DELTA_VS_NONE_MIN_PP if key == "delta_vs_none"
               else ra.DELTA_VS_PLAIN_MIN_PP)
        effects[key] = {"per_seed": per, "stability": ra.effect_stable(per, thr)}
    out = s3.judge_seed3(effects)
    assert out["delta_vs_none"]["stable"] is True
    assert out["delta_vs_plain"]["stable"] is False       # seed0 CI 下界 < 0


# ---------------- ③ 复用证据（SCHEMA §9） ----------------

def test_reuse_equality_report_verdict_ok():
    rep = eqrep.compare(CFG, write=False, recompute_trainer=False)
    assert rep["implementation_equivalent_to_2631_runtime"] is True
    assert rep["differing_fields"] == []
    assert rep["train_recipe_differing_flags"] == []
    assert rep["verdict"] == "reuse_ok"
    # 行为等价必须**两层**成立（复核 P1-3 + 追加要求）：基准是 **2631 运行时版本**
    # （不是 HEAD——HEAD 的字节会被并行线的默认关闭加性改动推动）
    ee = rep["engine_equivalence_working_tree_vs_head"]
    assert "2631 运行时版本" in ee["comparison_base"]
    assert ee["forward_layer_ok"] is True
    assert ee["trainer_layer_ok"] is True, "训练脚本层未证等价 ⇒ 复用声明必须降级"
    assert rep["reuse_scope"].startswith("pathlm/** + train_depth_ar.py")
    # 复用臂的 checkpoint 哈希必须就是 2631 当时落盘的那一份
    assert rep["reused_arm_checkpoint_sha256_at_2631_run"]["delta_s0"]


def test_reuse_equality_rejects_recipe_change():
    """反例：把训练步数改掉（配方不等）⇒ 复用必须被拒。"""
    bad = copy.deepcopy(CFG)
    bad["train_recipe_seed2"]["cmd"] = bad["train_recipe_seed2"]["cmd"].replace(
        "--steps 12000", "--steps 6000")
    rep = eqrep.compare(bad, with_equivalence=False, write=False, recompute_trainer=False)
    assert rep["verdict"] == "reuse_rejected"
    assert "--steps" in rep["train_recipe_differing_flags"]


def test_equality_report_schema_is_pinned_to_consumers():
    """消费方依赖的报告键必须与生产者产出**逐键一致**（复核 P1 的同类事故护栏）。

    `compare()` 返回前自检 `REQUIRED_REPORT_KEYS`；这里再断言冻结清单覆盖了
    `run.py` 真正读到的键，且缺键时 `assert_report_schema` 必须报错（可失败）。
    """
    rep = eqrep.compare(CFG, write=False, recompute_trainer=False)
    assert set(eqrep.REQUIRED_REPORT_KEYS) <= set(rep)
    # run.py 实际读的键
    for key in ("implementation_equivalent_to_2631_runtime", "reuse_scope",
                "differing_fields", "train_recipe_differing_flags"):
        assert key in rep and key in eqrep.REQUIRED_REPORT_KEYS, key
    # 缺键必须报错（不是静默 None / KeyError 深埋）
    with pytest.raises(KeyError, match="缺键"):
        eqrep.assert_report_schema({k: rep[k] for k in rep
                                    if k != "implementation_equivalent_to_2631_runtime"})


def test_run_py_reads_current_keys_not_retired_ones():
    """`run.py`（文档化的复现入口）不得再读已退役的旧键名。"""
    src = open(os.path.join(EXP, "run.py")).read()
    assert "implementation_equivalent_to_2631_runtime" in src
    assert "implementation_identical" not in src
    assert "eqrep.assert_report_schema(equality)" in src


def test_embedded_pointer_matches_authoritative_artifact():
    """`results.json::reuse_equality_vs_2631` 是**指针块**，其标量必须等于权威产物。

    防的正是复核 P2-1：同一份证据两处各存 schema ⇒ 重构后出现相反说法。
    """
    if not os.path.exists(RESULTS):
        pytest.skip("results.json 尚未生成")
    emb = json.load(open(RESULTS))["reuse_equality_vs_2631"]
    art = json.load(open(os.path.join(EXP, "equality_report.json")))
    assert emb["authoritative_artifact"].endswith("equality_report.json")
    for k in ("verdict", "reuse_scope", "implementation_equivalent_to_2631_runtime",
              "differing_fields", "train_recipe_differing_flags"):
        assert emb[k] == art[k], k
    # `engine_equivalence` 必须是**指针 + 标量**（不得整块内嵌 `per_rule` 等细节）
    pe = emb["engine_equivalence"]
    assert pe["authoritative_artifact"].endswith("engine_equivalence.json")
    # 逐档细节不得内嵌：每个值都必须是标量（结构判据，而不是 grep 文本）
    assert all(isinstance(v, (str, bool, int, float)) for v in pe.values()), pe
    assert set(pe) == {"authoritative_artifact", "comparison_base", "forward_layer_ok",
                       "trainer_layer_ok", "all_layers_ok", "coverage",
                       "trainer_n_log_points_compared", "trainer_n_field_diffs", "note"}
    eqrep.assert_pointer_consistent(emb)          # 标量 == 权威产物（逐项）
    # 旧 schema 的痕迹必须消失
    for retired in ("implementation_identical", "head_identical_to_2631",
                    "loss_sequence_head", "n_params_head",
                    "engine_equivalence_working_tree_vs_head"):
        assert retired not in json.dumps(emb)


def test_pointer_consistency_check_is_fail_able():
    """指针块与权威产物不一致时必须报错（不是「只校验 5 个标量」的弱护栏）。"""
    if not os.path.exists(RESULTS):
        pytest.skip("results.json 尚未生成")
    emb = json.load(open(RESULTS))["reuse_equality_vs_2631"]
    eqrep.assert_pointer_consistent(emb)          # 原样应通过
    for mutate in (lambda e: e["engine_equivalence"].__setitem__("forward_layer_ok", False),
                   lambda e: e["engine_equivalence"].__setitem__(
                       "trainer_n_log_points_compared", 0),
                   lambda e: e.__setitem__("reuse_scope", "伪造的范围")):
        bad = json.loads(json.dumps(emb))
        mutate(bad)
        with pytest.raises(ValueError, match="不一致"):
            eqrep.assert_pointer_consistent(bad)


def test_lock_note_consistency_is_enforced(monkeypatch):
    """`lock_note` 必须与 `lock_accounting.mode/lock_taken` 同口径（复核 P1 护栏）。

    两种模式各验一次「构造不出矛盾的结果文件」：派生文案随模式变，且人为制造矛盾时
    落盘前的一致性检查必须报错。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "carry_seed2_run", os.path.join(EXP, "run.py"))
    run_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_mod)
    lock = {"mode": "section_20_lock", "lock_taken": True}
    exc = {"mode": "section_20_1_2_short_job_exception", "lock_taken": False}
    for acct, must, must_not in ((lock, "整程持锁", "短任务插队例外"),
                                 (exc, "§20.1.2 短任务插队例外", "整程持锁")):
        note = run_mod._lock_note(acct)
        assert must in note and must_not not in note
        run_mod._assert_lock_consistency({"lock_note": note, "lock_accounting": acct})
        # 制造矛盾 ⇒ 必须报错（两种方向各一条）
        with pytest.raises(ValueError):
            run_mod._assert_lock_consistency({"lock_note": note,
                                              "lock_accounting": exc if acct is lock else lock})
    # 未记账时必须如实说「未记账」，且不得声称任何口径
    assert "未记账" in run_mod._lock_note(None)
    with pytest.raises(ValueError):
        run_mod._assert_lock_consistency(
            {"lock_note": "评测由 `eval.sh` **整程持锁**", "lock_accounting": None})


def test_committed_lock_notes_match_their_accounting():
    """三个实验的**已入账** results.json 都要满足该不变量（跨实验护栏）。

    判据加强为**逐字**：`cost.lock_note == lock_accounting.derive_note(cost.lock_accounting)`
    ——「由记账派生」不再只对某一格成立（复核 P2-2）。
    """
    for exp in ("2633-carry-seed2", "2634-carry-realtext-13m",
                "2635-recursive-carrier-13m"):
        p = os.path.join(ROOT, "axes", "1-time", "1.1.4.1-explicit-memory", "experiments",
                         exp, "results.json")
        if not os.path.exists(p):
            continue
        mod = _load_exp_module(exp, "lock_accounting")
        cost = json.load(open(p))["cost"]
        acct = cost.get("lock_accounting")
        note = cost.get("lock_note") or ""
        if acct is None:
            assert "整程持锁" not in note and "短任务插队例外" not in note, exp
            continue
        assert note == mod.derive_note(acct), (exp, note, mod.derive_note(acct))
        mod.assert_consistent(cost)                  # 同口径（派生即一致，仍显式核）


def test_record_refreshes_both_fields_and_repairs_contradiction(tmp_path):
    """驱动路径护栏：`record()` 必须**同时**刷新 accounting 与 note，并**修好**矛盾产物。

    （复核 P1 的可复现路径：驱动在 run.py 之后只覆写 accounting ⇒ 若不同时刷 note，就造出
    「note 说例外 / accounting 说持锁」；`record()` 是唯一实现，单测在**临时目录**上验证它。）
    """
    import shutil
    exp_src = os.path.join(ROOT, "axes", "1-time", "1.1.4.1-explicit-memory", "experiments",
                           "2633-carry-seed2")
    shutil.copy(os.path.join(exp_src, "results.json"), tmp_path / "results.json")
    shutil.copy(os.path.join(exp_src, "lock_accounting.json"), tmp_path / "lock_accounting.json")
    mod = _load_exp_module("2633-carry-seed2", "lock_accounting")
    # 先人为制造矛盾：note 说「短任务插队例外」，accounting 说「整程持锁」
    res = json.load(open(tmp_path / "results.json"))
    res["cost"]["lock_accounting"] = {"mode": "section_20_lock", "lock_taken": True}
    res["cost"]["lock_note"] = "评测按 **§20.1.2 短任务插队例外** 执行（纯推理）"
    with pytest.raises(ValueError):
        mod.assert_consistent(res["cost"])            # 矛盾可被抓住（可失败）
    json.dump(res, open(tmp_path / "results.json", "w"))
    # 驱动口径记账 ⇒ 两个字段一起刷新、自检通过
    out = mod.record("section_20_1_2_short_job_exception", 10000, 10000, 42, 0,
                     base_dir=str(tmp_path))
    after = json.load(open(tmp_path / "results.json"))["cost"]
    assert after["lock_accounting"] == out
    assert after["lock_note"] == mod.derive_note(out)
    mod.assert_consistent(after)
    # 反向：整程持锁口径也要一起刷
    out2 = mod.record("section_20_lock", 10000, 9999, 43, 0, base_dir=str(tmp_path))
    after2 = json.load(open(tmp_path / "results.json"))["cost"]
    assert after2["lock_note"] == mod.derive_note(out2)
    assert "整程持锁" in after2["lock_note"] and "短任务插队例外" not in after2["lock_note"]
    assert out2["headroom_ok"] is True and out2["free_gpu_mem_mib_after"] == 9999
    # **跨文件读回**断言（复核 P2-2）：直接对磁盘上两份独立产物核不变量
    mod.assert_cross_file_consistent(str(tmp_path / "lock_accounting.json"),
                                     str(tmp_path / "results.json"))
    # 篡改磁盘上的 note ⇒ 跨文件断言必须抓到（可失败），且 record 会把它修好
    res = json.load(open(tmp_path / "results.json"))
    res["cost"]["lock_note"] = "伪造的措辞：评测按 **§20.1.2 短任务插队例外** 执行"
    json.dump(res, open(tmp_path / "results.json", "w"))
    with pytest.raises(ValueError, match="跨文件不一致"):
        mod.assert_cross_file_consistent(str(tmp_path / "lock_accounting.json"),
                                         str(tmp_path / "results.json"))
    mod.record("section_20_lock", 10000, 9999, 44, 0, base_dir=str(tmp_path))
    mod.assert_cross_file_consistent(str(tmp_path / "lock_accounting.json"),
                                     str(tmp_path / "results.json"))   # 已修好


def test_record_without_results_file_only_writes_accounting(tmp_path):
    """`results.json` 不存在（run.py 崩在最终写盘前）时：只写记账、不留「一边新一边旧」。"""
    mod = _load_exp_module("2633-carry-seed2", "lock_accounting")
    out = mod.record("section_20_lock", 9000, 9000, 10, 1, base_dir=str(tmp_path))
    assert (tmp_path / "lock_accounting.json").exists()
    assert not (tmp_path / "results.json").exists()
    assert json.load(open(tmp_path / "lock_accounting.json")) == out


def test_all_rating_drivers_use_the_single_recorder():
    """六个驱动（每格 `eval.sh` + `eval_concurrent.sh`）都必须走 `lock_accounting.py record`。

    防的是复核 P1 的形态：驱动自己内联覆写 accounting（只改一半 ⇒ 造出矛盾产物）。
    """
    for exp in ("2633-carry-seed2", "2634-carry-realtext-13m",
                "2635-recursive-carrier-13m"):
        for drv in ("eval.sh", "eval_concurrent.sh"):
            p = os.path.join(ROOT, "axes", "1-time", "1.1.4.1-explicit-memory", "experiments",
                             exp, drv)
            if not os.path.exists(p):
                continue
            src = open(p).read()
            assert 'lock_accounting.py" record' in src, (exp, drv)
            assert 'cost"]["lock_accounting"] =' not in src, (exp, drv)   # 不得内联覆写


def test_runtime_version_ledger_covers_every_arm():
    """运行期版本台账（复核 P1-3）：九臂都要有版本分类，且 verdict 与覆盖范围自洽。"""
    p = os.path.join(EXP, "runtime_version_ledger.json")
    if not os.path.exists(p):
        pytest.skip("runtime_version_ledger.json 尚未生成")
    led = json.load(open(p))
    assert set(led["per_arm"]) == {a["name"] for a in CFG["arms"]}
    for name, row in led["per_arm"].items():
        for rel, cell in row["files"].items():
            assert cell.get("version") in (
                "runtime_version", "runtime_version(==working_tree)",
                "working_tree_at_run_time"), (name, rel)
        assert row["started_at"], f"{name} 缺启动时刻（无法判断跑的是哪一版）"
    assert led["verdict"] == "all_arms_behaviorally_equivalent"
    assert "2631 运行时版本" in led["comparison_base"]
    assert "train_depth_ar.py" in led["engine_equivalence_coverage"]
    # 漂移落地上界必须有据（历史观测 mtime 或漂移提交）
    for rel, f in led["code_files"].items():
        if f["drifted_now"]:
            assert f["drift_landed_no_later_than"] or f["drift_commits_after_runtime_version"]
    assert led["drift_snapshot"]["n_bytes"] >= 0


def test_engine_equivalence_per_rule_report_is_complete():
    """三档配置各自都必须给出「参数量相同 + 同权重 logits 逐位相同」的证据。

    这是 `equality_report.json` 里那条**语义级**等价结论的原始来源；缺任何一档就说明
    该档的复用没有实现层证据（本测试会红）。
    """
    import engine_equivalence as ee
    # **只用 CPU 路径**（`with_trainer=False`）：前向层不需要 GPU；训练脚本层的证据由已入账的
    # `engine_equivalence.json::trainer` 提供（生成它才需要 GPU 跑 2×1001 步，≈41 s）。
    r = ee.check(write=False, with_trainer=False)
    assert set(r["per_rule"]) == {"none", "plain", "delta"}
    for rule, v in r["per_rule"].items():
        assert v["n_params_same"], rule
        assert v["params_bitwise_same_at_same_seed"], rule
        assert v["state_dict_keys_same"], rule
        assert v["max_abs_logit_diff_same_weights"] == 0.0, rule


def test_engine_files_behaviorally_unchanged_since_2631_run():
    """§9 实现层判据 = **行为等价**，不是「字节未变」。

    背景（复核追加要求）：并行线对引擎做**默认关闭的加性改动**并入库，`HEAD` 的字节因此
    会前进；「2631 运行时版本」与「当前工作树」的**字节**不必相同，但**开关关闭时行为必须
    逐位不变**。⇒ 判据两件：
      ① 按 2631 记录的 sha256 能在 git 历史里**定位到当时的版本**（否则基准不存在）；
      ② 该版本与当前工作树**两层行为等价**：`pathlm/**` 三档配置逐位 logits +
         `train_depth_ar.py` 同一 run 的日志点逐字段相同（点数 > 1，避免「单点假等价」）。
    """
    rep = eqrep.compare(CFG, write=False, recompute_trainer=False)
    for rel, row in rep["implementation"].items():
        assert row["runtime_version_commit"], f"{rel}: 按记录哈希定位不到 2631 运行时版本"
        assert row["n_matching_commits"] >= 1, rel
        # 字节是否未变只是信息项：允许变，但必须有行为等价证据（下面断言）
        assert "bytes_unchanged_since_2631_run" in row
    assert rep["implementation_equivalent_to_2631_runtime"] is True
    ee = rep["engine_equivalence_working_tree_vs_head"]
    assert "2631 运行时版本" in ee["comparison_base"]
    assert ee["forward_layer_ok"] is True and ee["trainer_layer_ok"] is True
    assert ee["trainer"]["verified"] is True
    assert ee["trainer"]["n_log_points_compared"] >= 2, ee["trainer"]["n_lines"]


def _load_exp_module(exp: str, mod_name: str):
    """按路径加载某个实验目录里的模块（测试用；避免与仓库其它同名模块冲突）。"""
    import importlib.util
    p = os.path.join(ROOT, "axes", "1-time", "1.1.4.1-explicit-memory", "experiments",
                     exp, f"{mod_name}.py")
    spec = importlib.util.spec_from_file_location(f"{exp.replace('-', '_')}_{mod_name}", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


RESULTS = os.path.join(EXP, "results.json")


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="训练/评测尚未完成")
def test_results_criteria_and_seed_coverage():
    r = json.load(open(RESULTS))
    c = r["criteria"]
    for k in ("c1_premise_none_near_chance", "c2_delta_vs_none_stable",
              "c3_delta_vs_plain_stable", "c4_corrupt_control_ok",
              "c5_no_extrapolation", "c6_sample_size_reported"):
        assert k in c, k
    assert len(r["effects"]["delta_vs_plain"]["per_seed"]) == 3
    assert [x["seed"] for x in r["effects"]["delta_vs_plain"]["per_seed"]] == [0, 1, 2]
    assert r["corrupt_control"]["factor"] == 4.0        # 判据④的阈值口径
    assert r["corrupt_control"]["bound"] == pytest.approx(4 * r["premise_check"]["0"]["chance"])
    for name in ("none_s0", "plain_s0", "delta_s0", "none_s1", "plain_s1", "delta_s1",
                 "none_s2", "plain_s2", "delta_s2"):
        assert name in r["main_leg"] and name in r["corrupt_leg"]
    assert r["sample_size_analysis"]["target_min_pp"] == 3.0


@pytest.mark.skipif(not os.path.exists(RESULTS), reason="训练/评测尚未完成")
def test_results_judgement_is_consistent_with_its_own_numbers():
    """判据②③的布尔值必须与逐种子数字自洽（防手工改判）。"""
    r = json.load(open(RESULTS))
    for key, thr in (("delta_vs_none", ra.DELTA_VS_NONE_MIN_PP),
                     ("delta_vs_plain", ra.DELTA_VS_PLAIN_MIN_PP)):
        row = r["effects"][key]["per_seed"]
        expect = all(x["effect_pp"] >= thr and x["paired_boot_ci_pp"][0] > 0
                     for x in row)
        assert row and expect == r["effects"][key]["stability"]["all_seeds_pass"], key
