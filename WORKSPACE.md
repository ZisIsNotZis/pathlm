# Workspace

- Status: design phase. Architecture frozen in `docs/design.md`; ablation program in `docs/experiments.md`. No code yet.
- Decisions of record: char-level ASCII tokenizer on **enwik8** (comparability anchor; TinyStories fallback); tied E=U; MTP block + probability heads always on (part of base config); aggregation always inference-time math with detached weights, never inside the training loss; corruption training and repair supervision are one functional unit; approved scale d=256 / 8L / 10–15M / seq 512.
- Open decisions: engine framework (PyTorch custom loop is the working default); M0 probe results may adjust probability-machinery details.
