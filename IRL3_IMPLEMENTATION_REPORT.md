# IRL-3 research-engineering implementation report

## 1. Final architecture

IRL-3 is implemented as a causal Transformer prelude, one shared latent
recurrent computer, gated latent-to-token write-back, and a short Transformer
coda. Teacher-forced training executes an independent compact computer for
every causal prefix in parallel; cached generation executes the identical
program for the newest prefix only.

The joint recurrent state is

\[
S_i^{(t)}=(Z_i^{(t)},H_i^{(t)},M_i^{(t)}),
\]

where (A) is immutable contextual evidence, (Z) is fast workspace, (H)
is slow strategy, (M) is differentiable working memory, and (E) is a
continuous phase/budget/dynamics embedding. One transition performs causal
anchor read, memory and strategy read, latent self-attention and SwiGLU
computation, feature-wise gated workspace update, attention-based memory
erase/add write, periodic slow-state update, and conditional halt evaluation.

Projected anchor K/V are reused across reasoning steps. The final `Z/H/M`
slots are queried by each token and the correction is on the main logit path.
`IRL3Cache` can additionally carry solved reasoning state across autoregressive
token time.

The complete equations, dataflow, causal proof argument, cost model, and
prior-work attribution are in [IRL3_ARCHITECTURE.md](IRL3_ARCHITECTURE.md).

## 2. State transitions

In compact notation:

\[
R_A=\operatorname{Attn}(Z,K_A,V_A),\quad
R_M=\operatorname{Attn}(Z,M,M),\quad
R_H=\operatorname{Attn}(Z,H,H),
\]

\[
\Delta_Z=\operatorname{SelfAttn}(\widetilde Z)
+\operatorname{SwiGLU}(\operatorname{Norm}(\widetilde Z+
\operatorname{SelfAttn}(\widetilde Z))),
\]

\[
Z'=Z+G_Z(Z,\Delta_Z,E)\odot
[\operatorname{Norm}(Z+s_Z\Delta_Z)-Z].
\]

Memory slots query (Z'), then use explicit erase, add, and write gates:

\[
M'=M+w\odot[\operatorname{Norm}((1-e)\odot M+a)-M].
\]

Every configured slow-clock interval, (H) attends to ([Z';M']), applies
slow self-attention and SwiGLU, and updates through its own state-conditioned
gate. Conditional halt hazards form a normalized distribution over recurrent
states; training consumes their differentiable weighted state, while
evaluation can freeze each prefix independently. Explicit fixed budgets remain
available for test-time compute scaling.

## 3. Major implementation decisions

- **Central computer, not per-layer loops.** Fixed prelude layers construct
  strong immutable evidence; one shared state machine concentrates recurrent
  parameters and semantics.
- **Causal parallel prefixes.** Prefix is a batch-like dimension and never
  participates in state attention. This preserves teacher-forcing throughput
  without future leakage.
- **Bounded anchor window.** Contextual anchor states already summarize their
  causal history; recurrent latents re-read a configurable local evidence
  window so cost remains linear in sequence length.
- **Explicit state roles.** Fast slots, slower clocked strategy slots, and
  erase/add working memory have different transitions rather than being
  concatenated copies.
- **Continuous phase conditioning.** Progress, budget, slow clock, residuals,
  memory activity, and halt survival condition the shared program.
- **State-marginalized learned halting.** This is substantially cheaper than
  applying the coda and vocabulary head at every possible stop, but is
  documented as a PonderNet-inspired approximation rather than exact expected
  prediction loss.
- **Weak memory equilibrium.** Terminal normalized residual regularizes
  `Z/H` more strongly than memory, which may still have a useful final write.
- **Scheduled credit horizon.** Full BPTT is supported; deep runs can use an
  explicit final-step horizon with depth/credit warmup.
- **Persistence without dormant-only parameters.** Token-time fusion uses
  cosine gating and recurrent prior projections trained on every batch.
- **IRL-2 retained.** The old package, profiles, trainer, and checkpoints stay
  independently usable for baselines.
- **Portable accelerator path.** Only public PyTorch operations and SDPA/Flex
  paths are used; no CUDA-only extension is introduced.
- **Eager AdamW slots.** Optimizer state is allocated before the first step so
  an early curriculum checkpoint reloads even when some modules have not yet
  entered the graph.

### Pre-qualification measurement corrections

- State-attention FLOPs now count Q/K/V/O projection rows using the actual
  Z/M/H query and key/value slot counts, plus attention mixing, FFNs, all
  recurrent gates/projections, exact slow ticks, write-back, and active MTP
  horizons.
- Training FLOPs distinguish full forward work from the differentiable credit
  horizon and add the selected gradient-checkpoint recomputation explicitly.
- The H equilibrium numerator retains the transition immediately before and
  after the last executed slow tick. A trailing non-tick can no longer make a
  real H update appear to have zero residual.
- Diagnostics expose the slow-tick mask/count, last tick, fast Z/M contraction,
  per-state equilibrium terms, and correction-to-anchor norm ratio.
- The same exported checkpoint is now evaluated in fixed, ACT-soft,
  adaptive-hard, and write-back-off modes. OFF executes the computer but does
  not add its correction to the anchor.
- The parameter-matched IRL-2 run is mandatory. Training budgets were shortened
  to 200 IRL-3 and 60 IRL-2 steps so all controls and reporting fit the bounded
  hardware window without changing the 135M architecture profiles.

## 4. Files added or changed

Core architecture:

- `lunaris_dense_irl3/config.py`
- `lunaris_dense_irl3/diagnostics.py`
- `lunaris_dense_irl3/cache.py`
- `lunaris_dense_irl3/irl3.py`
- `lunaris_dense_irl3/block.py`
- `lunaris_dense_irl3/mtp.py`
- `lunaris_dense_irl3/model.py`
- `lunaris_dense_irl3/generation.py`
- `lunaris_dense_irl3/__init__.py`

Training:

- `lunaris_dense_irl3/training/config.py`
- `lunaris_dense_irl3/training/schedules.py`
- `lunaris_dense_irl3/training/logging.py`
- `lunaris_dense_irl3/training/train.py`
- `lunaris_dense_irl3/training/preflight.py`
- `lunaris_dense_irl3/training/export_checkpoint.py`
- `lunaris_dense_irl3/training/__init__.py`
- `lunaris_dense_irl2/training/optimizers.py` (curriculum-safe optimizer
  checkpoint state and IRL-3 gradient grouping)

Profiles and qualification:

- `profiles/irl3_debug.yaml`
- `profiles/irl3_135m_mi300x_qualification.yaml`
- `training_profiles/irl3_debug_cpu.yaml`
- `training_profiles/irl3_mi300x_smoke.yaml`
- `training_profiles/irl3_135m_1x_mi300x_qualification.yaml`
- `training_profiles/irl2_135m_1x_mi300x_matched.yaml`
- `scripts/benchmark_recurrent_depth.py`
- `scripts/run_irl3_mi300x_qualification.sh`
- `scripts/summarize_irl3_qualification.py`

Tests and documentation:

- `tests/test_irl3_computer.py`
- `tests/test_irl3_model.py`
- `tests/test_irl3_training.py`
- `tests/test_irl3_qualification.py`
- `IRL3_ARCHITECTURE.md`
- `IRL3_MI300X_QUALIFICATION.md`
- `IRL3_IMPLEMENTATION_REPORT.md`
- `README.md`
- `pyproject.toml`

## 5. Tests executed

Final local results:

```text
python -m pytest -q
48 passed
```

The suite includes finite forward/backward, nonzero gradients through every
IRL-3 subsystem, multiple depths, staggered per-prefix halting, no-future
leakage, packed-document isolation, cache parity, persistence, generation,
MTP, gradient checkpointing, BF16 forward/backward, save/load, real CPU
optimizer steps, DCP resume after an early inactive-module checkpoint, export,
meta-device preflight, slot-aware FLOPs accounting, arbitrary slow-clock
equilibrium semantics, same-checkpoint write-back bypass, and qualification
summary controls.

Additional verification:

- an independent 3-step CPU train and resume to step 4 completed;
- every parameter in a model with MTP received a finite nonzero gradient;
- explicitly BF16 cached generation completed on the local CPU build;
- Ruff passed on all new Python and the shared optimizer change;
- `compileall` passed;
- `bash -n scripts/run_irl3_mi300x_qualification.sh` passed;
- editable wheel build/install and all three IRL-3 console entry points passed.

The only emitted test warnings are PyTorch notices that Distributed Checkpoint
is operating in intended single-process mode on the local CPU host.

## 6. Representative parameter counts

Counts use tied input/output embeddings and meta-device construction:

| Profile | Total | Embedding/head | Fixed blocks | IRL-3 computer | MTP | Other |
|---|---:|---:|---:|---:|---:|---:|
| `irl3_debug.yaml` | 746,594 | 49,152 | 333,504 | 298,658 | 65,184 | 96 |
| IRL-3 MI300X | 134,772,866 | 50,331,648 | 61,398,528 | 18,907,778 | 4,134,144 | 768 |
| matched IRL-2 control | 135,153,937 | — | — | — | — | — |

Recurrent depth changes no parameter counts.

## 7. Expected compute and memory

The corrected analytic breakdown estimates approximately **1.254 billion
forward FLOPs per observed token** and **4.025 billion executed training FLOPs
per observed token** for the MI300X profile at depth six with two active MTP
horizons. At 16,384 observed tokens, that is **65.94 TFLOPs per optimizer
step**. The training total includes a two-times differentiable-forward backward
approximation over the four-step recurrent credit horizon and the extra
recurrent-computer forward caused by checkpoint recomputation.

The count includes token Transformer projections/mixing/FFNs, problem
conditioning, anchor K/V, slot-row-aware state attention, anchor mixing,
fast/slow FFNs, priors and gates, exact slow ticks, halting/phase networks,
memory erase/add/write, token write-back, the LM head, and active MTP heads. It
omits norms, nonlinearities, softmax/masking, reductions, elementwise gates,
and launch overhead. Small-slot efficiency makes measured throughput
authoritative; this estimate is not used as a favorable substitute for the
MI300X measurements.

With the single-GPU FP32-master AdamW path, parameters, gradients, and two
optimizer moments have a persistent lower bound of about **2.01 GiB**. This
excludes activations and workspaces. At micro-batch four, context 4096, six
latents, eight state heads, and anchor window 128:

- one FP32 anchor-score tensor per recurrent step is about 0.375 GiB;
- one BF16 fast-state tensor is about 0.094 GiB;
- one BF16 slow-state tensor is about 0.031 GiB;
- the four-step gradient horizon and recurrent checkpoint recomputation bound
  stored recurrence activations;
- the chunked vocabulary projection has bounded transient memory.

Compute and recurrent activations scale approximately linearly with assigned
depth and anchor window. Parameters stay fixed. The actual HBM peak, kernel
occupancy, and tokens/second are deliberately deferred to the MI300X sweep.

## 8. Uncertainties requiring the MI300X

- ROCm gfx942 correctness of all BF16 SDPA/Flex and recurrent backward paths;
- practical efficiency of small state-attention GEMMs;
- true HBM peak at context 4096 and gradient horizon four;
- whether recurrent-computer checkpointing is the best time/memory trade-off;
- stability of state/update RMS and contraction ratios over hundreds of steps;
- whether memory gates saturate or retain useful write activity;
- whether slow-state ticks remain active;
- whether learned halting escapes the minimum-depth solution after delayed
  ponder warmup;
- whether extra fixed inference depth improves, saturates, or degrades CE on a
  short-trained checkpoint;
- whether same-checkpoint write-back OFF measurably changes CE and whether the
  correction-to-anchor ratio is material;
- whether ACT-soft and adaptive-hard differ at matched 4/8/12 budgets;
- whether adaptive per-prefix freezing yields measurable latency savings;
- the interpretable difference from the smaller-token-budget IRL-2 control.

## 9. Exact MI300X commands

The complete bounded run is:

```bash
cd /path/to/irl-dev
timeout --foreground 13800s bash scripts/run_irl3_mi300x_qualification.sh
```

The primary manual train/resume commands are:

```bash
python -m lunaris_dense_irl3.training.train \
  --config training_profiles/irl3_mi300x_smoke.yaml

python -m lunaris_dense_irl3.training.train \
  --config training_profiles/irl3_135m_1x_mi300x_qualification.yaml

python -m lunaris_dense_irl3.training.train \
  --config training_profiles/irl3_135m_1x_mi300x_qualification.yaml \
  --max-steps 205 \
  --resume auto
```

The exact export, fixed/ACT/adaptive/write-back-off benchmarks, guaranteed
IRL-2 control, and summary
commands are encoded in `scripts/run_irl3_mi300x_qualification.sh` and listed
with recovery guidance in
[IRL3_MI300X_QUALIFICATION.md](IRL3_MI300X_QUALIFICATION.md).

## 10. Proposed four-hour schedule

| Window | Work |
|---:|---|
| 0:00-0:15 | runtime/ROCm report and repository tests |
| 0:15-0:30 | data reuse/preparation and both preflight reports |
| 0:30-0:42 | four-step BF16 full-path smoke |
| 0:42-2:22 | 200-step 134.8M IRL-3 architecture run |
| 2:22-2:32 | checkpoint reload, extension, and portable export |
| 2:32-3:02 | guaranteed 60-step matched IRL-2 control and depth sweep |
| 3:02-3:37 | fixed 2/4/6/8/12, ACT/adaptive 4/8/12, and write-back OFF |
| 3:37-3:45 | machine JSON and concise Markdown summary |
| 3:45-3:50 | recovery reserve |

The outer 3h50 timeout leaves a ten-minute recovery margin inside the hard
four-hour allocation. The architecture profile is unchanged; only short-run
training budgets were reduced to make the control and all ablations mandatory.
