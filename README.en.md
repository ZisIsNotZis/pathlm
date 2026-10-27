**English** | [简体中文](README.md)

<img src="assets/logo.svg" alt="pathlm" width="200">

# PathLM Research Wiki

A research tree over the design space of where a language model spends its compute — one cell per research item, recording who tried it, what came out, why it failed, and how to reproduce it.

**It turns "where should a language model spend its compute" into a machine-checkable research tree: 46 research items sliced by axis, every conclusion carrying its measurement basis, controls, variance band and reproducibility verdict — and a check that turns red the moment the tree drifts.**

This is not a model repository: it does not train, publish, benchmark or assemble combination experiments. What it publishes is **a reviewable research tree plus a measurement discipline**, for two audiences: people changing model architecture who want to know up front whether a change costs quality or comes for free, and people who assign, review and take over work on the tree.

## Up and running in 30 seconds

```bash
git clone git@github.com:ZisIsNotZis/pathlm.git
cd pathlm
tools/bootstrap.sh                       # version gate + dependency gate (Python >= 3.10; meta-layer checks need only PyYAML)
python3 tools/build_catalog.py --check    # meta self-check: catalog and source of truth agree byte for byte
```

The only thing a clean clone cannot install is the experiment stack: `tests/` needs `pip install -r requirements-dev.txt` (pytest / numpy / torch — torch is a multi-GB download). Experiment data has to be fetched by hand (see "What a clean clone cannot rebuild" below).

## What it can do today

- **Research catalog**: `docs/catalog.md` holds one line per research item (id / title / purpose / acceptance / expected output / status / pointer), generated verbatim from 46 `project.yaml` files — hand edits turn the check red.
- **Six research axes plus three side lines**: time · depth · width · experts (MoE) · quantization · prediction, plus references, engineering discipline and backlog (axes 7 and 8 are never greenlit as research items — they keep ledgers and side experiments only).
- **Eleven measured laws**: `atlas/playbook.md` holds the statements and their applicability boundaries; `atlas/cost-table.md` (generated) lines each law up with its evidence and its observed cost.
- **Reproducibility inventory**: `atlas/repro.md` (generated) answers per experiment where the run is, whether its artifacts are in the repo, how to re-run it, whether it is byte-reproducible, and what is missing.
- **Findings index**: `atlas/FINDINGS.md`, one conclusion per line, pointing at homes rather than holding numbers.
- **Research priorities**: `atlas/triage.md` ranks research items by "can this change a decision" and drafts batches.
- **Engine and eval kit**: variable-depth incremental decoding (`pathlm/depth_ar.py`), the eval suite (`eval.py` / `gen_quality.py`), the allocator (`slider.py`) and model parts (`model.py`, `exit_policy.py`, …), with arm configs in `configs/` and entry points in `train_*.py` / `probe_*.py`.
- **Machine checks**: every `--check` under `tools/` is a gate that can fail, and `tests/` pins the discipline down as unit tests.

## How the repository is organised

The repo is cut into **three layers**; a fact has exactly one home, and other layers only carry pointers:

| Layer | Directories | What it owns |
|---|---|---|
| **meta** | repo root + `docs/` | what this repo is, the research catalog, the layer contract and assignment template, milestones and work packages, the migration ledger |
| **research** | `axes/**`, `atlas/**` | each item's facts (`project.yaml` is its source of truth), cross-cell summaries (laws, cost table, reproducibility list, ranking, paper plan) |
| **engineering** | `harness/**`, `tools/**`, `tests/**` | format contracts and measurement discipline, engine capability surface, generators and checks, shared base |

```
README.md / README.en.md   front page (Chinese primary, English mirror)
AGENTS.md                  layer contract, assignment template, collaboration rules (written for agents)
CONTRIBUTING.md            contribution flow and hard rules
docs/                      meta: catalog.md (generated) / plan.md / structure.md / ledger.md
atlas/                     research cross-cell summaries: playbook / cost-table / repro / triage / FINDINGS / design / paper-plan
axes/                      research per item: <axis>/<item>/project.yaml (source of truth)
harness/                   engineering: SCHEMA / research-schema / methodology / engine_capability
tools/                     engineering: generators and --check gates (incl. bootstrap.sh, evalkit/)
tests/                     engineering: shared-base tests
pathlm/ configs/ train_*.py probe_*.py slider.py
                           shared base and historical engine (kept in place; migration plan in atlas/design.md)
.scratch/                  historical tickets and raw evidence (read-only log, not authoritative)
archive/                   legacy project, read-only
assets/                    logo
LICENSE.md                 full AGPL-3.0 text
```

The layers talk in exactly three ways: **pointers** (everything else links to the fact's home), **generated artifacts** (catalog / cost table / reproducibility list are produced verbatim from their sources by `tools/`, and `--check` compares them byte for byte against any hand edit), and **machine checks** (`tools/check_meta_layer.py` fails when a research reading shows up in a meta document, `tools/check_docs.py` fails on unresolvable references or an incomplete per-layer index). Ownership is therefore not a convention — it can turn red.

The life of a research item: the meta layer hands out a contract following the template in `AGENTS.md` (goal / scope / authoritative inputs / deliverables and acceptance / budget / stop condition) → the research layer pre-registers criteria under `axes/<axis>/<item>/`, runs experiments, lands readings and verdicts → cross-cell facts are summarised in `atlas/` → a reviewer with a fresh context inspects it until nothing is left to find.

## Developing and self-checking

```bash
pip install -r requirements-dev.txt        # pytest / numpy / torch (only needed to run tests/)
```

Run the checks for the layer you touched (each is an exit-code gate):

```bash
# meta layer
python3 tools/build_catalog.py --check     # catalog and project.yaml agree byte for byte
python3 tools/check_meta_layer.py --check  # no research readings inside meta documents
python3 tools/check_ledger.py --check      # migration ledger complete, homes exist, deletions justified
python3 tools/check_docs.py --check        # references resolve and all three layer indexes are complete

# research layer
python3 tools/summary.py --check           # item fields / evidence / basis (= tools/check_tree.py)
python3 tools/build_cost_table.py --check  # atlas/cost-table.md matches project.yaml byte for byte
python3 tools/inventory_runs.py --check    # atlas/repro.md matches what is on disk
python3 -m pytest tests/ -q                # shared-base tests
```

Change facts only at their source, then re-run the generator: the catalog and the cost table come from `project.yaml`, the reproducibility list comes from the runs on disk. **Generated files are never hand-edited** — `--check` turns red.

## What a clean clone cannot rebuild

- **Data**: the raw enwik8 bytes (`.tmp/enwik8`) are not committed and are never auto-downloaded; put them there yourself and the cache is generated on first run.
- **Per-run raw artifacts**: `runs/` under each experiment directory is git-ignored; the `analysis.md` / `results.json` that are committed are in the repo. Which experiments are byte-reproducible and what is missing is answered per experiment in `atlas/repro.md`.
- **Weights**: `*.pt` files are derived artifacts; retrain them with the commands in each experiment's `README.md`.
- **GPU surface**: a full `tests/` run needs torch + CUDA; this repo has only been exercised on a single RTX 4090.

## Contributing

- **Report a problem**: open an issue or a PR here. One sentence saying which file, which conclusion, and where you think it is wrong beats a long essay.
- **Before a PR**: run the checks for your layer (the two blocks above) and write numbers the way `harness/methodology.md` asks.
- **Contribution rules**: start at `CONTRIBUTING.md`; format contracts are in `harness/SCHEMA.md` and `harness/research-schema.md`, tree layout in `docs/structure.md`.
- New experiments must bind a contract before training (`python3 tools/prereg_bind.py <experiment dir>`), and the binding is committed.

## Roadmap

**Implemented**

- The three-layer split: the meta layer keeps only the catalog and the contract, research and engineering each own their facts and index; the meta milestone passed all seven acceptance items and a fresh-context review.
- Checks split by layer: four on the meta layer, six on the research layer, all exit-code gates, with `tests/` pinning the discipline.
- A generated catalog over 46 research items plus per-item `project.yaml` sources of truth; measured laws and the cost table.
- The reproducibility inventory (per-experiment three-state verdict) and research prioritisation.
- Verdicts landing across the six axes: hard windows, sparse experts, low-rank compression, repeating depth, inter-layer aggregation, speculative decoding and more (per item in `docs/catalog.md`).
- A paper draft: `atlas/paper.md` follows the claim, structure and ten acceptance items in `atlas/paper-plan.md`; it only moves already-landed facts and recomputes nothing. It ships after the licence decision and a fresh review.

**Planned**

- Turn `pathlm/` into an installable package (`pyproject.toml` + `pip install -e .`), with tests and probes importing from inside the package.
- Extract the eval kit into a shared `tools/evalkit/` base so each research item's experiment directory carries a `run.py` that pulls the balance from it (plan in `atlas/design.md`).
- The paper's release pieces: licence decision (`docs/plan.md` §5) and DOI / tag alignment.
- Fill in the items marked not-started / backlog in the catalog (e.g. external retrieval, weight sharing), each ordered first by the criteria in `atlas/triage.md`. **These need a GPU**: this machine has none and will not get one soon; the route and the registration are in `docs/plan.md` §1.3.

**Where this leads**: turning "where should compute go" from judgement into a cost table with applicability boundaries — so that, before anything touches a GPU, a new idea can be read against the family it belongs to, the cost already measured for that family, and the conditions that must hold for the effect to become measurable at all.

## License

**Code license: AGPL-3.0**, full text in `LICENSE.md` (copyright ZisIsNotZis).

**Documentation and paper licensing is decided separately**: `LICENSE.md` covers code; `atlas/paper-plan.md` aims at public citability, and documentation usually takes a CC BY-style license — not yet settled.
