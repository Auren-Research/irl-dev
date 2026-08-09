# Lunaris IRL-3

IRL-3 is a research causal language model built around a **stateful latent
recurrent computer**. A compact shared neural program repeatedly reads
immutable contextual evidence, reasons in a fast workspace, reads and writes
working memory, updates slower strategy state, decides how long to continue,
and writes the result back to the token stream.

This branch also retains the complete IRL-2 package as a baseline. IRL-3 is an
experimental implementation, not a pretrained model or a claim that recurrent
compute is superior to static depth.

## Design

The recurrent state is

\[
S_t=(Z_t,H_t,M_t),
\]

with:

- `A`: immutable causal evidence produced by a Transformer prelude;
- `Z`: fast problem-conditioned reasoning slots;
- `H`: fewer slow strategy slots updated on a lower-frequency clock;
- `M`: fixed-capacity differentiable erase/add working memory;
- `E`: continuous phase, budget, residual, memory, and survival conditioning.

Each shared step performs:

```text
read causal anchor + read memory + read strategy
    -> communicate and compute in Z
    -> gated Z update
    -> attention-based memory erase/add write
    -> periodic H update
    -> conditional halt decision
```

Final token queries attend to the solved `Z/H/M` slots before a short
Transformer coda and the LM head. The recurrent computation is therefore on
the primary logit path.

See [IRL3_ARCHITECTURE.md](IRL3_ARCHITECTURE.md) for equations, causal
execution, phase conditioning, halting, equilibrium loss, persistence, cost
analysis, prior-work attribution, and known limitations.

## Repository layout

```text
lunaris_dense_irl3/       IRL-3 model, state machine, cache, generation, trainer
lunaris_dense_irl2/       retained IRL-2 implementation and baseline trainer
profiles/                 model-only profiles
training_profiles/        complete pretraining and qualification profiles
scripts/                  data, accelerator, depth benchmark, qualification
tests/                    IRL-2 and IRL-3 correctness/integration tests
```

Stable Transformer primitives—GQA, RMSNorm, SwiGLU, RoPE, attention backends,
initialization, and layer KV cache—are reused from IRL-2. IRL-2's architecture
and checkpoints remain separately named and loadable.

## Installation

Install a PyTorch build appropriate for the host first, then:

```bash
python -m pip install -e ".[test]"
```

The same public `torch.cuda` API is used for NVIDIA/CUDA and AMD/ROCm. There
are no CUDA-only extension dependencies.

## Minimal use

```python
import torch

from lunaris_dense_irl3 import LunarisIRL3Config, LunarisIRL3ForCausalLM

config = LunarisIRL3Config.debug(vocab_size=512, mtp_num_heads=2)
model = LunarisIRL3ForCausalLM(config)
tokens = torch.randint(0, config.vocab_size, (2, 32))

output = model(
    tokens,
    labels=tokens,
    compute_depth=4,
    halt_mode="act",
    return_logits=False,
    loss_chunk_size=128,
    mtp_loss_weight=0.2,
    mtp_active_heads=2,
    ponder_coefficient=1e-3,
    equilibrium_coefficient=1e-3,
    return_irl3_stats=True,
)
output.loss.backward()

print(output.cross_entropy)
print(output.ponder_loss)
print(output.equilibrium_loss)
print(output.irl3_diagnostics.as_dict())
```

`labels` has the same shape as `input_ids`; the model shifts causal targets
internally. Packed sequences can pass both `position_ids` and `document_ids`.
Anchor reads and all Transformer attention respect document boundaries.

## Test-time compute and generation

An explicit budget is a fixed-depth experiment control:

```python
with torch.no_grad():
    shallow = model(tokens, compute_depth=2).logits
    deeper = model(tokens, compute_depth=8).logits
```

Learned per-prefix halting is available at evaluation:

```python
adaptive = model(
    tokens,
    compute_depth=12,
    halt_mode="adaptive",
    return_irl3_stats=True,
)
print(adaptive.irl3_diagnostics.step_counts)
```

Cached generation can optionally carry final reasoning state from one token to
the next:

```python
generated = model.generate(
    tokens[:, :8],
    max_new_tokens=32,
    temperature=0.0,
    compute_depth=6,
    persist_reasoning_state=True,
)
```

Transformer K/V, projected immutable-anchor K/V, and optional `Z/H/M` state
have separate cache entries.

## Training

IRL-3 has its own strict trainer entry points:

```bash
lunaris-irl3-preflight \
  --config training_profiles/irl3_debug_cpu.yaml \
  --world-size 1

lunaris-irl3-train \
  --config training_profiles/irl3_debug_cpu.yaml

lunaris-irl3-export \
  --checkpoint checkpoints/irl3_debug/step_00000003 \
  --output exports/irl3_debug
```

The trainer reuses the repository's memory-mapped NPY dataset, deterministic
world-size-independent sampler, DDP/FSDP2 setup, BF16/FP16 autocast,
Distributed Checkpoint, JSONL/W&B logger, and regional loss/attention compile
paths. IRL-3-specific schedules control:

- recurrent-depth curriculum and synchronized fixed/uniform/log-uniform depth
  sampling;
- truncated or full recurrent gradient horizon;
- delayed ponder pressure;
- terminal equilibrium regularization;
- MTP weight and active horizons.

Create tiny local shards for the debug profile:

```bash
python - <<'PY'
from pathlib import Path

import numpy as np

rng = np.random.default_rng(1337)
for split in ("train", "validation"):
    path = Path("data/debug") / split
    path.mkdir(parents=True, exist_ok=True)
    tokens = rng.integers(0, 512, 4096, dtype=np.uint16)
    np.save(path / "tokens.npy", tokens)
PY
```

The main objective is

\[
\mathcal L=\mathcal L_{NTP}+\gamma\mathcal L_z
+\alpha\mathcal L_{MTP}+\beta\mathcal L_{ponder}
+\delta\mathcal L_{eq}.
\]

Small profiles:

- `profiles/irl3_debug.yaml`;
- `training_profiles/irl3_debug_cpu.yaml`;
- `training_profiles/irl3_mi300x_smoke.yaml`.

The prepared single-MI300X profile is
`training_profiles/irl3_135m_1x_mi300x_qualification.yaml`. It resolves to
134,772,866 parameters on the current implementation.

## Tests

```bash
python -m pytest -q
```

IRL-3 tests cover:

- shapes and finite forward/backward;
- gradients through anchor reads, latent attention, memory, strategy,
  halting, write-back, and MTP;
- multiple train/inference depths with one parameter set;
- per-prefix adaptive halting;
- causal no-future leakage and packed-document isolation;
- full-versus-cached logit parity;
- optional token-time state persistence;
- gradient checkpointing;
- checkpoint save/reload and generation;
- BF16 when supported by the current device/build;
- training schedules, preflight, optimizer steps, and checkpoint resume.

## MI300X qualification

The hardware run is prepared but has not been executed. With one MI300X and a
hard four-hour allocation:

```bash
timeout --foreground 13800s bash scripts/run_irl3_mi300x_qualification.sh
```

The bounded runner records ROCm/BF16 compatibility, full-path smoke training,
stability, checkpoint resume, fixed-depth scaling, adaptive halting, throughput,
peak HBM, recurrent diagnostics, and—if time remains—a small matched IRL-2
control. It emits machine-readable artifacts and a concise research summary.

See [IRL3_MI300X_QUALIFICATION.md](IRL3_MI300X_QUALIFICATION.md) for the exact
commands, phase caps, artifact schema, acceptance gates, and proposed schedule.

## Important limitations

- No MI300X execution evidence exists yet.
- Working memory is fixed-capacity; there is no sparse expandable episodic
  store in this phase.
- Learned halting uses a state-marginalized approximation rather than an exact
  expected coda/vocabulary loss at every step.
- Hard per-prefix freezing may not translate to proportional dense-kernel
  speedups.
- Persistent state works in inference and remains differentiable when passed
  directly, but the default trainer does not yet carry state across documents.
- The equilibrium loss is a terminal residual regularizer, not proof of
  contraction or an implicit DEQ solver.
- Short qualification depth curves measure path behavior, not converged
  reasoning quality.

## Primary research references

- [HRM-Text](https://arxiv.org/abs/2605.20613)
- [Hierarchical Reasoning Model](https://arxiv.org/abs/2506.21734)
- [Scaling up Test-Time Compute with Latent Reasoning](https://arxiv.org/abs/2502.05171)
- [Universal Transformers](https://arxiv.org/abs/1807.03819)
- [Deep Equilibrium Models](https://arxiv.org/abs/1909.01377)
- [Perceiver](https://arxiv.org/abs/2103.03206)
- [Universal Transformers Need Memory](https://arxiv.org/abs/2604.21999)
- [Depth-Conditioned Looped Transformers for ASR](https://arxiv.org/abs/2606.04678)
- [PonderNet](https://arxiv.org/abs/2107.05407)
- [Neural Turing Machines](https://arxiv.org/abs/1410.5401)
- [Recurrent Memory Transformer](https://arxiv.org/abs/2207.06881)

Ideas adopted from these works and the original IRL-3 composition are
distinguished explicitly in the architecture document.
