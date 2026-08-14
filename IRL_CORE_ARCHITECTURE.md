# Lunaris Dense IRL Core

IRL Core is a simplified causal language-model architecture built beside the
existing IRL-2 implementation. Its narrow hypothesis is that one shared
transition can perform progressively useful computation by repeatedly querying
the same contextual evidence through an evolving latent state.

## Architecture and transition

The model is:

```text
Embedding -> Transformer Prelude -> IRL Core -> Transformer Coda -> LM Head
```

The prelude produces an anchor `A`. The anchor remains unchanged for the whole
recurrent invocation, although gradients still flow through it. The only
explicit recurrent state is `S_t`, initialized with `zeros_like(A)`.

For `t = 0, ..., T - 1`:

```text
A_bar  = RMSNorm(A)
K_A    = W_K(A_bar)                 # once per invocation
V_A    = W_V(A_bar)                 # once per invocation

X_t    = RMSNorm(A + S_t)
Q_t    = W_Q(X_t)
R_t    = W_O CausalGQA(Q_t, K_A, V_A)
U_t    = SwiGLU(RMSNorm(X_t + R_t))
F_t    = R_t + U_t

Lambda = lambda_min
         + (lambda_max - lambda_min) * sigmoid(lambda_logit)
S_t+1  = (1 - Lambda) * S_t + Lambda * F_t
```

`lambda_logit` and `Lambda` have shape `[d_model]`; damping is channel-wise,
not a separate state machine. All query, read, nonlinear-update, norm, and
damping parameters are reused at every recurrent step. The terminal output is
`Y = A + S_T`, which is processed by the coda and normal LM head. Existing
QK-Norm remains enabled in the provided profiles as a numerical-stability
detail around the projected queries and keys.

## Why the anchor is immutable

`A` represents contextual evidence assembled by the Transformer prelude. If
the evidence itself changed during recurrence, changes in recurrent behavior
would conflate evidence mutation with reasoning-state evolution. Holding `A`
fixed makes the mechanism explicit: only `S_t` evolves, while each `Q_t` can
retrieve a different causal view of the same evidence.

The anchor is not detached. Terminal LM gradients train the embedding,
prelude, anchor projections, recurrent transition, coda, and head jointly.

## Why anchor K/V are cached

Keys and values depend only on normalized `A`, so recomputing them at each step
would add large projections without changing the result. IRL Core projects
anchor K/V once, applies RoPE once, and reuses the tensors for all `T` reads.
Queries and attention outputs are recomputed because they depend on `S_t`.

During autoregressive generation, the core has one ordinary model-wide KV-cache
entry. It stores the anchor K/V of prior tokens; current-token anchor K/V are
projected once and appended only after the recurrent invocation. Prelude and
coda layers use the existing cache implementation unchanged.

## Difference from IRL-2

IRL-2 performs attention once inside every dense block and then applies a
shared dense recurrent refinement. Its recurrent iterations do not issue fresh
relational reads. IRL Core instead has one recurrent component in the whole
model, and every iteration forms a new query from `A + S_t` while reading the
same immutable causal anchor K/V.

The ordinary prelude and coda blocks are conventional pre-norm GQA/SwiGLU
Transformer blocks. The original `lunaris_dense_irl2` package and profiles are
left intact.

## Intentionally removed

The architecture has no Z/H/M state decomposition, mutable memory module, slow
strategy state, phase encoder, ACT, adaptive halting, equilibrium loss, ponder
loss, depth sampling, intermediate-state marginalization, or recurrent-state
consistency objective. MTP remains in shared IRL-2 infrastructure but the IRL
Core config rejects nonzero MTP heads and all supplied IRL Core profiles disable
the MTP objective.

## Training regime

Serious profiles use fixed `T_train = T_eval = 6`. An explicit positive
`n_iters` override remains available for diagnostics and generation. Only the
terminal recurrent state receives the normal shifted next-token LM objective.

The production profiles use AdamW, BF16 autocast, global gradient clipping at
1.0, and the existing warmup-stable-decay schedule. The initial schedule has a
2,000-step warmup, 88,000-step stable phase, and 10,000-step cosine warmdown to
10% of peak LR. Peak LR is `3e-4` for the engineering profile and `2e-4` for the
approximately 500M profile. These are conservative starting points; the phase
lengths must be rescaled when the long-run token budget is chosen. Logs include
pre-clip gradient norm and the fraction of optimizer steps that clipped, so a
persistently clipped mature phase is visible rather than silently accepted.

Gradient checkpointing is optional. The `core` policy checkpoints individual
shared recurrent steps while retaining the single projected anchor K/V graph;
the `block` policy checkpoints complete ordinary blocks and the complete core.
The one-MI300X profiles leave it off initially because memory headroom should be
measured before accepting recomputation overhead.

## Complexity

Let `N` be the number of ordinary Transformer blocks and `T` the recurrent
depth. There are `N + T` attention calls per forward, but only `N + 1` key/value
projection pairs: one per ordinary block and one for the whole core. The core
keeps one `[B, L, d_model]` state and no per-prefix slots.

For a global core read, attention mixing remains quadratic in sequence length:
`O(T * B * L^2 * d_model)`. Core projections cost one anchor K/V projection plus
`T` query/output and SwiGLU projections. Ordinary blocks can retain the existing
local/global attention pattern. The implementation uses PyTorch SDPA/GQA and is
structured so a fixed six-step Python loop can be specialized by
`torch.compile`; it adds no custom Triton or ROCm kernels.

## Inference behavior

Evaluation defaults to six iterations. Cache-aware decoding is mathematically
equivalent to full causal evaluation within numerical tolerance because past
anchor K/V and coda K/V are cached, while the current token starts from
`S_0 = 0` and performs the same recurrent transition. Increasing depth beyond
the trained value is diagnostic only and is not expected to improve quality.

## Lightweight diagnostics

Optional diagnostics record per-step state RMS, update RMS, relative update
norm, attention-read RMS, correction/anchor RMS ratio, and adjacent-update
cosine similarity, plus damping minimum/mean/maximum. They are disabled on the
normal training path and are incompatible with activation checkpointing to
avoid retaining diagnostic graphs.

## Known limitations

- Recurrent global reads multiply attention work by six and may still dominate
  long-context throughput despite K/V reuse.
- Fixed-depth training does not establish safe behavior at untrained depths.
- An immutable anchor cannot revise its evidence representation during the
  recurrent phase.
- The architecture has not yet demonstrated downstream quality or scaling
  superiority over a compute-matched Transformer.
- The initial learning-rate schedule and batch sizes require validation against
  the eventual data mixture and token budget.
