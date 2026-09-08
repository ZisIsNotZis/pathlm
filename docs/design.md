# Path LM — design spec

The frozen architecture of the toy "Path LM": a decoder-only model reinterpreted as a latent sequence model with elastic compute paths. Implementation decisions (frameworks, code layout) are deliberately not here. The measurement program lives in `docs/experiments.md`.

## 1. Summary

Path LM keeps one shared geometry from input embedding to output readout: the residual stream never leaves embedding space. The model refines the current latent in place (a loop over idempotent "enhancer" layers), extrapolates future latents with a fixed set of transform heads extended by chaining, and discretizes lazily through a tied embed/unembed boundary. Training deliberately corrupts inputs (wrong/`[mask]` tokens, noised or pure-noise latents), randomizes the compute path (shuffle/skip/redo/exit), and attaches a calibrated probability to every output — so all redundancy (heads, step-size compositions, retry rounds) becomes voters in one confidence-weighted average.

## 2. The pipeline ring

| Stage | Operation | Tunables |
|---|---|---|
| 0 token | token source: normal / wrong (uniform vocab) / `[mask]` | rate λ, span mode (iid or contiguous) |
| E | embed; tied with U (E = U); input norm capped at c | – |
| 1 perturb | identity / +Gaussian noise / replace with pure-noise latent; or inject a predicted latent (speculative fast path) | rate λ, σ |
| L layers | idempotent enhancer stack | shuffle-locality, p_skip, p_redo, p_exit |
| 2 MTP | N+1 transform heads → (latent_k, prob_k) for positions i…i+N; k=0 = self | N, head size |
| 3 transport | return channel, single choice: direct→1 / linear U·Eᵀ / soft softmax(·Uᵀ)·E / full decode | choice |
| 4 retry | U(latent_0) → if prob_0 low: corrected token → stage 0 | threshold |

Stages form a ring: the exits of 2/3/4 re-enter at 0, 1, or 2. Stage numbering is the canonical reference for paths (e.g. "2→3→1→L→2").

## 3. Cycles — every path is a re-entry

- Latent retry: 2 → 3 → 1 → L → 2. Refine without discretizing; the transport choice defines the flavor. A retry can only add information if the channel transforms the state — direct re-entry re-derives the same fixed point (measured in M0); soft/linear/discrete channels re-embed a corrected or expected token, and that is where repair value lives.
- Token retry: 2 → U → 4 → 0. Discretize, correct, re-enter as a normal token.
- Chain: 2 → 3(direct) → skip-all → 2. Not a new mechanism — an emergent composition of existing tunables (transport=direct, stage-1 identity, all layers skipped); re-enters the MTP block, extending the horizon past N.
- Next position (default): U(latent_1) → sample token → next position's stage 0. Fast path (speculation): latent_1 → next position's stage 1 directly (latent prefill), verified later.

## 4. Stage details

### 0/E/1 — writable input

Corruption elements are first-class: wrong token, `[mask]`, noised embedding, pure-noise latent bypassing the embedding. Embedding norm is capped at c; noise is drawn at a scale comparable to typical embedding norms. Contiguous-span mode corrupts ranges rather than iid positions — the natural model of real text damage at character level.

### L — enhancer stack

Layers are designed as idempotent refinements in the shared geometry, not feature transforms: skipping all of them must leave a usable (if weaker) representation, and repeating must converge. shuffle-locality ∈ [0,1] interpolates between normal order (0) and an exactly uniform random permutation (1) via sort key `(1−locality)·index + locality·U(0,n)` (n = stack size), resampled each pass. p_skip omits a layer, p_redo repeats a layer, p_exit jumps to the stack end.

### 2 — MTP block (always on)

Head k maps latent_i → latent_{i+k}; k=0 is the self estimate, included so the current token carries a probability too. Heads are d×d transforms — no vocab-sized parameters. prob_k is supervised on token identity at position i+k (labels exist under teacher forcing), so confidence is readable without ever calling U. prob_0 is the universal control signal: it gates latent retry, token retry, exit, and chain expansion.

### 3 — transport (single choice)

direct (raw latent) / linear (U·Eᵀ: projection onto the vocab subspace) / soft (softmax(·Uᵀ)·E: re-quantization through the vocab simplex; downstream positions attend to this expected embedding — "repair in context") / full decode (unembed → discrete token → embed).

### 4 — discrete retry

If prob_0 is low, decode the corrected token and re-enter at stage 0. Token-level twin of the latent retry: same "not confident → spend more compute" principle, differing only in whether discretization happens.

## 5. Aggregation principle

Every repeat estimate of the same event is a voter: the N+1 heads, the step-size compositions reaching the same target (T₂ vs T₁∘T₁ vs …), and the retry rounds. Final estimate = confidence-weighted average in token-probability space.

- Compositions collapse to a DP lattice over targets 1…N: node m aggregates its direct head plus all extensions of shorter nodes — O(N²), path enumeration never explicit.
- Chain confidence combines step confidences in log space with a length discount — never a raw product.
- Weights are detached model outputs (or fitted post-hoc); aggregation is never inside the loss.
- Default is token-level mixture (preserves multimodality); latent-level averaging is a cheaper variant.
- A consistency loss (T₂ ≈ T₁∘T₁, etc.) regularizes compositions to agree — this is what makes chaining and aggregation trained rather than hoped for.

## 6. Training recipe

Two classes of path:

- Prefix-monotone (early exit): representations up to the exit point are identical to the full run, so supervise densely — run full depth, apply the MTP block at every depth, CE on every output (LayerSkip-style, variance-free).
- Non-monotone (shuffle, redo, retry, transport): sample the path globally per pass/batch (tokens stay in sync), and give every (latent_k, prob_k) emitted along the traversed path its own CE against ground truth. Each output is trained as a calibrated estimator of its event, independent of the path that produced it.

Loss stack: per-component CE (primary) + consistency loss + optional attention-distance penalty and eviction objectives (X elements). Corruption budget: λ_total split among active elements; conservative start. Calibration is verified, not assumed (ECE gate).

## 7. Execution semantics

Training: retry counts and path knobs are sampled (geometric-style) for coverage; all tokens in a sequence share the sampled path. Inference: per-token adaptivity returns — prob_0 decides retry/exit/chain expansion per position. Expected compute per token is a mixture over paths; the round-count and chain-depth distributions are logged, not assumed.

## 8. Metrics battery (fixed columns for every experiment)

PPL · repair accuracy · ECE (per-component, pooled, per-chain-step) · decode speed per path config · needle-in-haystack · quality-vs-round-count curve · quality-vs-chain-depth curve · path agreement (dispersion across compositions) · weighted-ensemble vs best-single-path · latent-draft accept rate · cosine(predicted, actual latent) per head · token-matched and FLOPs-matched training reported separately.
