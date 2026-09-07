# Workspace

- Status: design phase. Architecture frozen in `docs/design.md`; ablation program in `docs/experiments.md`. No code yet.
- Decisions of record: char-level ASCII tokenizer; tied E=U; MTP block + probability heads always on (part of base config); aggregation always inference-time math with detached weights, never inside the training loss; corruption training and repair supervision are one functional unit.
- Open decisions: training dataset (TinyStories char-level proposed, enwik8 alternative); final scale numbers; engine framework for path sampling.
