"""元级隔离：元级作者的文档里不许出现研究级读数（`tools/check_meta_layer.py`）。

钉住七件事，每件都可失败：
1. **当前树是干净的**：根级 `*.md`（`LICENSE.md` 除外）与 `docs/**` 零命中，扫描面不为空；
2. **注入必红**（gate probe）：往 `docs/` 放一份带 `bpc 读数` 的文档 ⇒ 红；带一条逐实验目录路径 ⇒ 红；
   带一行复现三态计数 ⇒ 红；带一个裸读数小数 ⇒ 红；删掉那份文档 ⇒ 恢复绿；
3. **生成物投影有豁免但只有它有**：`docs/catalog.md` 不扫（正文逐字来自 `project.yaml`），
   其余 `docs/*.md` 都扫——豁免不能顺手放给作者写的文档；
4. **白名单只放定义性说明**：白名单条目按「文件 + 片段」匹配，不带数字的定义句本来就不命中，
   带读数的行即使被白名单放行也会在下一条检查里被抓到（见 `test_allowlist_cannot_hide_a_reading`）；
5. **围栏代码块不判**：示例代码里的路径与数字不是散文；
6. **每条规则都说清「为什么这是研究级」**：规则表里不许有空说明；
7. **扫描面自己不能漏文件**：根级 `*.md` 减去写明的非事实清单、加上 `docs/**/*.md`，
   必须与扫描面**逐个相等**——根级新增一份文档而不进清单，禁令对它就是空的。
"""

import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import check_meta_layer  # noqa: E402

PROBE = os.path.join(ROOT, "docs", "probe_meta_layer_tmp.md")


def _write_probe(text):
    with open(PROBE, "w", encoding="utf-8") as fh:
        fh.write(text)


def _drop_probe():
    if os.path.exists(PROBE):
        os.remove(PROBE)


def teardown_function():
    _drop_probe()


def test_current_meta_docs_are_clean(capsys):
    rc = check_meta_layer.main(["--check"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "0 处研究级读数" in out, out
    scanned = int(out.split("扫描元级 ")[1].split(" 个文档")[0])
    assert scanned >= 4, out


def test_scan_covers_root_and_docs_layer():
    files = check_meta_layer.meta_files()
    for rel in ("README.md", "AGENTS.md", "CONTRIBUTING.md", "docs/plan.md", "docs/structure.md",
                "docs/ledger.md"):
        assert rel in files, f"{rel} 没被元级扫描面覆盖：{files}"
    assert "atlas/cost-table.md" not in files, "研究级文档不该进元级扫描面"
    assert "harness/methodology.md" not in files, "工级文档不该进元级扫描面"
    assert "LICENSE.md" not in files, "许可证正文是写明的非事实文档，不进扫描面"


def test_injected_bpc_reading_turns_red():
    _write_probe("# 探针\n\n这条研究的 clean bpc 1.4836，比对照更好。\n")
    assert check_meta_layer.main(["--check"]) == 1
    violations, _ = check_meta_layer.scan()
    assert any(v.rule == "bpc 读数" and v.path.endswith("probe_meta_layer_tmp.md")
               for v in violations), [str(v) for v in violations]


def test_injected_experiment_path_turns_red():
    _write_probe("# 探针\n\n详见 `axes/2-depth/2.1.1-early-exit/experiments/2621-depth-draft-e2e/`。\n")
    assert check_meta_layer.main(["--check"]) == 1
    violations, _ = check_meta_layer.scan()
    assert any(v.rule == "逐实验目录路径" for v in violations), [str(v) for v in violations]


def test_injected_repro_tristate_count_turns_red():
    """三态计数的家是生成物 `atlas/repro.md`：写进元级散文就是第二个家。"""
    _write_probe("# 探针\n\n复现三态：可重算 0 / 只能重训 86 / 不可重建 34。\n")
    assert check_meta_layer.main(["--check"]) == 1
    violations, _ = check_meta_layer.scan()
    assert any(v.rule == "复现三态计数" and v.path.endswith("probe_meta_layer_tmp.md")
               for v in violations), [str(v) for v in violations]


def test_injected_bare_number_reading_turns_red():
    """没有单位、没有百分号的裸差值（`价签 +0.000034`）同样是读数。"""
    _write_probe("# 探针\n\n这一格的价签 +0.000034，按同口径不可排序。\n")
    assert check_meta_layer.main(["--check"]) == 1
    violations, _ = check_meta_layer.scan()
    assert any(v.rule == "裸读数小数" and v.path.endswith("probe_meta_layer_tmp.md")
               for v in violations), [str(v) for v in violations]


def test_removing_the_probe_restores_green():
    _write_probe("# 探针\n\nmemory 1.14 GiB 与 Δ +0.000101。\n")
    assert check_meta_layer.main(["--check"]) == 1
    _drop_probe()
    assert check_meta_layer.main(["--check"]) == 0


def test_generated_projection_is_exempt_and_only_it():
    files = set(check_meta_layer.meta_files())
    assert "docs/catalog.md" not in files, "生成物投影不该被当作作者写的元级文档"
    for rel, why in check_meta_layer.GENERATED.items():
        assert why.strip(), f"{rel} 的豁免没写理由"
    assert set(check_meta_layer.GENERATED) == {"docs/catalog.md"}
    for rel, why in check_meta_layer.NON_FACT_ROOT_DOCS.items():
        assert why.strip(), f"{rel} 的非事实理由不能是空的"
    assert set(check_meta_layer.NON_FACT_ROOT_DOCS) == {"LICENSE.md"}, (
        "非事实清单只该有许可证正文；想把别的文档加进去，先说明它凭什么不是事实文档（并改这条断言）")
    assert all("/" not in rel for rel in check_meta_layer.NON_FACT_ROOT_DOCS), (
        "非事实清单只收根级文档：目录里的文档按层归属判，不能靠一份清单溜出去")
    # 作者写的文档一个都不能漏在扫描面外：**根级 `*.md` 减去写明的非事实清单** + `docs/**/*.md`。
    # 从根级 glob 派生而不是硬编码文件名单：根级新加一份文档时，扫描面必须跟着扩。
    root_md = {os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "*.md"))}
    authored = (root_md - set(check_meta_layer.NON_FACT_ROOT_DOCS)) | {
        os.path.relpath(os.path.join(d, f), ROOT)
        for d, _, fs in os.walk(os.path.join(ROOT, "docs")) for f in fs if f.endswith(".md")}
    assert authored - set(check_meta_layer.GENERATED) == files, (
        "根级或 docs/ 的元级文档没被扫描面覆盖（新文档漏检 = 这条禁令对它是空的）")
    assert "CONTRIBUTING.md" in files, "根级文档必须自动进面：点名的名单漏掉谁，禁令对谁就是空的"


def test_term_tokens_without_numbers_are_allowed():
    """禁令要写出来才有人遵守：不带数字的 `bpc` / `Δ` / `显存` / `墙钟` 是定义句，不判。"""
    text = ("# 定义\n\n元级不写研究级读数：bpc、Δ、显存、墙钟、逐格判词一律只放指针。\n")
    assert check_meta_layer.scan_text("docs/x.md", text) == []


def test_fenced_code_block_is_not_judged():
    text = ("# 示例\n\n```\nclean bpc 1.4836，Δ +0.000101，显存 1.14 GiB\n```\n")
    assert check_meta_layer.scan_text("docs/x.md", text) == []


def test_each_rule_says_why_it_is_research_level():
    assert check_meta_layer.RULES, "规则表不能是空的"
    samples = {"bpc 读数": "1.5 bpc", "Δ 读数": "Δ +0.1",
               "显存/内存读数": "1.14 GiB", "墙钟/延迟读数": "23.2 ms",
               "倍数/比例读数": "2.58×", "百分比读数": "46.9%",
               "逐实验目录路径": "axes/a/experiments/b/",
               "逐格判词计数": "已完成 13",
               "复现三态计数": "可重算 0 / 只能重训 86",
               "裸读数小数": "价签 +0.000034"}
    for name, pattern, why in check_meta_layer.RULES:
        assert name.strip() and why.strip(), f"{name} 没写清为什么这是研究级事实"
        assert name in samples, f"新规则 {name} 没给一条命中样本：这条断言对它就是空的"
        assert pattern.search(samples[name]), name


def test_allowlist_entry_shape_and_effect(monkeypatch):
    """白名单按「文件 + 片段」匹配；只对写明的那个文件生效。"""
    monkeypatch.setattr(check_meta_layer, "ALLOWLIST",
                        (("docs/x.md", "定义：", "定义句，不是读数"),))
    assert check_meta_layer.scan_text("docs/x.md", "定义：bpc 1.4836 在本仓\n") == []
    assert check_meta_layer.scan_text("docs/y.md", "定义：bpc 1.4836 在本仓\n") != []


def test_allowlist_is_not_used_to_hide_readings():
    """当前白名单为空：仓库里没有一条「必须留在元级的读数」。"""
    assert check_meta_layer.ALLOWLIST == (), (
        "白名单只放定义性说明；要放一条读数进去，等于承认它该留在元级，那就该改规则")
