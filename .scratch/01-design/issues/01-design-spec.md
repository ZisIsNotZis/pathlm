# 01 — Design spec & experiment program

- **Status:** done
- **Type:** design
- **Blocked by:** —

## Issue

Freeze the PathLM architecture from the design session into `docs/design.md` and define the decoupled ablation program in `docs/experiments.md`, plus minimal repo scaffold.

## Acceptance criteria

- [x] Stage pipeline ring, cycles, and tunables written down
- [x] Aggregation principle, training recipe, execution semantics captured
- [x] Metrics battery fixed (all experiments report the same columns)
- [x] 15 main-effect runs + 4 interaction runs + integration run, each with a gate
- [x] Milestones M1–M3 defined
- [x] Scaffold: README, AGENTS.md, WORKSPACE.md, .gitignore, docs/people.md

## Knowhow learned

- Transport and latent retry are inherently coupled (transport without retry is a no-op) — measured as one unit in the C runs.
- Prefix-monotone paths (early exit) admit dense, variance-free supervision; non-monotone paths require global path sampling with per-component CE.
- Including the self estimate (k=0) in the MTP block gives the current token a probability, which becomes the universal control signal for retry/exit/chain.

## Comments

- 2026-09-07 — agent (pi, volc2/glm-5.3-flash) — wrote `docs/design.md` and `docs/experiments.md` from the agreed design discussion; initialized repo scaffold; committed.
