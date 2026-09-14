# PathLM — agent notes

- Design truth lives in `docs/`: `design.md` is the frozen architecture, `experiments.md` the measurement program, `findings.md` the measured mechanism-by-mechanism conclusions. Do not contradict them; changes go through the user.
- Project-specific knowledge: `WORKSPACE.md`. Tickets: `.scratch/` (Matt Pocock layout).
- The workspace policy at `/home/z/vibe/AGENTS.md` applies.

## Language

- Use Chinese (中文) in conversation with the user, and in all agent-authored
  documents, tickets, evidence READMEs, and commit messages for this project.
- Code identifiers, log strings, and config keys stay in English (code stays
  parseable and greppable); prose around them is Chinese.
- When a doc mixes both, keep one language per document — do not interleave
  paragraph by paragraph.
- User-supplied content is recorded verbatim (L1), whatever language it is in.
