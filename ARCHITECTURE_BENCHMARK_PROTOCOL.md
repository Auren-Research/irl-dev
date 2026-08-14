# 135M Controlled Architecture Benchmark Protocol

This benchmark compares one conventional Transformer, the existing IRL-2
architecture, and the qualified simplified IRL Core. It changes no IRL Core
transition or training semantics. The comparison is parameter-matched and
token-matched; it is deliberately not FLOP-matched.

## Frozen architectures

| Model | Total parameters | Recurrent training depth | Attention calls/forward |
| --- | ---: | ---: | ---: |
| Transformer | 132,147,968 | 0 | 16 |
| IRL-2 | 132,197,136 | 2 | 16 |
| IRL Core | 132,149,504 | 6 | 21 |

All models use a 65,536-token vocabulary, width 768, 12 query heads, 4 KV
heads, a 1,536-wide SwiGLU, QK-Norm, RMSNorm, RoPE, tied input/output
embeddings, the `LLLG` local/global policy, a 1,024-token local window, and a
4,096-token sequence length. The Transformer has 16 ordinary blocks. IRL-2
has its existing 16 attention-plus-dense-recurrence blocks and is trained at
its intended fixed T=2 for this profile. IRL Core is byte-for-byte equivalent
at the resolved configuration level to `profiles/irl_core_135m.yaml`: 13
prelude blocks, one shared Core at T=6, and 2 coda blocks.

MTP and every auxiliary recurrent loss are disabled for all three models.
This keeps the LM objective and trainable parameter comparison aligned. The
widest parameter mismatch is about 0.037%, well below the declared 1% limit.

## Data and randomization

One pinned 1B-token Ultra-FineWeb shard and its 65,536-entry tokenizer provide
three contiguous, deterministic, non-overlapping ranges:

- train: 500,000,000 source tokens;
- dev: 4,194,304 source tokens (1,024 sequences);
- confirmation holdout: 4,194,304 source tokens (1,024 sequences).

The preparation manifest records the tokenizer, source, and split SHA-256
digests and exact half-open offsets. All models use data seed 1337 and model
seed 20260814. A preflight hashes the first 2,048 shuffled sequence indices for
every profile and refuses to run unless the hashes match. The holdout is used
only by the frozen evaluator; architecture and optimization settings cannot be
changed after preflight.

## Optimization and milestones

The common effective batch is eight sequences, or 32,768 tokens per optimizer
step. A complete update cannot land on exactly 500,000,000 tokens, so the
trajectory stops below the ceiling at 15,258 steps and 499,974,144 tokens.
Milestones are:

| Nominal milestone | Step | Actual training tokens |
| --- | ---: | ---: |
| 100M | 3,052 | 100,007,936 |
| 250M | 7,629 | 249,987,072 |
| 500M | 15,258 | 499,974,144 |

The shared AdamW recipe uses LR 3e-4, betas (0.9, 0.95), epsilon 1e-8,
weight decay 0.1, global gradient clipping at 1.0, BF16 autocast, chunked CE,
and z-loss coefficient 1e-4 during training. The 15,258-step WSD schedule has
305 warmup steps (about 2%), 13,427 stable steps, and 1,526 cosine warmdown
steps (about 10%) to a 0.1 minimum-LR ratio. This scales the already qualified
conservative recipe to the 500M-token horizon while preserving a meaningful
warmdown and clipping diagnostics.

Stopping at a milestone does not resize the LR schedule. Model, optimizer,
RNG, and absolute data cursor are checkpointed and resumed into the same
trajectory. Architectures are interleaved by milestone so a failure is found
before the remaining compute is consumed.

## Evaluation and uncertainty

Each checkpoint is evaluated on identical sequential dev and holdout batches.
The primary CE uses the trained depth (Transformer T=1 compatibility path,
IRL-2 T=2, IRL Core T=6). IRL dev depth curves use T in
`{1, 2, 4, 6, 8, 12}` without retraining. Keeping depth curves on dev protects
the confirmation holdout from mechanism-oriented inspection.

Pairwise uncertainty uses the 128 aligned holdout batch CE values. Reports
give the paired mean difference and a two-sided normal 95% confidence interval;
raw batch observations are retained. Equal-token quality, parameter count,
estimated training FLOPs/token, cumulative FLOPs, steady throughput,
end-to-end wall time, HBM, clipping, and Core state diagnostics are reported
separately. Token-efficiency interpolation is allowed only when measured
100M--500M points bracket the target CE; no out-of-range extrapolation is used.

## Runtime guard

Each model receives a short real-data BF16 forward/backward qualification on
the MI300X. The orchestrator refuses to begin the long trajectories when the
sum of measured projections plus a 30-minute checkpoint/evaluation allowance
exceeds seven hours. It never increases a model beyond the frozen token cap.
