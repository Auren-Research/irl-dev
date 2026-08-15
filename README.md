# Lunaris Dense

Lunaris Dense is a research repository for causal decoder-only language models
with shared recurrent refinement. It contains three parameter-matched model
families, reusable training infrastructure, frozen profiles, compact benchmark
evidence, and correctness tests.

## Architectures

| Architecture | Package | Role |
| --- | --- | --- |
| IRL-2 | `lunaris_dense_irl2` | Primary research candidate |
| IRL Core | `lunaris_dense_irl_core` | Simplified recurrent comparison |
| Transformer | `lunaris_dense_transformer` | Conventional baseline |

IRL-2 is the current primary candidate because it achieved the lowest holdout
cross-entropy in the controlled 135M architecture benchmark. This is evidence
for that benchmark setting only; it is not a general scaling or downstream
quality claim.

## Controlled 135M benchmark

All models were parameter-matched and trained on the same 499,974,144-token
trajectory. The final confirmation-holdout results were:

| Model | Parameters | Holdout CE |
| --- | ---: | ---: |
| Transformer | 132,147,968 | 3.611391 |
| IRL-2 | 132,197,136 | **3.594243** |
| IRL Core | 132,149,504 | 3.601472 |

| Paired comparison | CE difference | 95% CI |
| --- | ---: | ---: |
| Core - Transformer | -0.009919 | [-0.010938, -0.008900] |
| IRL-2 - Transformer | -0.017148 | [-0.017951, -0.016345] |
| Core - IRL-2 | +0.007229 | [+0.006318, +0.008140] |

Machine-readable summaries, milestone values, aligned holdout batches, depth
curves, and hardware measurements are in
[`results/architecture_benchmark`](results/architecture_benchmark/).

## Production MI300X profile

`profiles/production_irl2_135m.yaml` preserves the trained two-step IRL-2
recurrence in every layer. The MI300X path uses fused QKV and SwiGLU input
projections, compact RoPE, native RMSNorm, exact FlashAttention, whole-model
`torch.compile`, and a measured 15-local/1-global attention schedule. The model
still has exactly 132,197,136 parameters; the recurrent cell, anchor/state
update, shared internal weights, and learned damping are unchanged.

The full controlled single-MI300X run used the same 499,974,144-token stream as
the frozen benchmark:

| Metric | Frozen IRL-2 | Production IRL-2 | Change |
| --- | ---: | ---: | ---: |
| Holdout CE | 3.594243 | **3.583993** | **-0.010249** |
| Median training tok/s | 93,875 | **193,310** | **2.059x** |
| Peak allocated HBM | 53.10 GiB | **32.02 GiB** | **-39.7%** |
| Estimated FLOPs/token | 1.3967B | **1.3118B** | **-6.08%** |
| Training wall time | 1.492 h | **0.748 h** | **-49.9%** |

The aligned 128-batch holdout comparison gives a paired 95% confidence
interval of [-0.011807, -0.008691] for production minus frozen CE. The reported
production wall time conservatively includes four compilation/resume segments;
the uninterrupted qualification projection is 0.708 h.

The production profile requires a ROCm-compatible `flash-attn` build. Use a
ROCm/PyTorch image that supplies it and verify the environment before training:

```bash
python -c "from flash_attn import flash_attn_func; print('flash-attn ready')"
python -m lunaris_dense_irl2.training.preflight \
  --config training_profiles/production_irl2_135m_1x_mi300x.yaml \
  --require-data
```

Controlled optimization results and the scale recommendation are in
[`results/production_optimization`](results/production_optimization/).

## Installation

Install the PyTorch build appropriate for the local CUDA or ROCm environment,
then install the project:

```bash
python -m pip install -e ".[test]"
```

Python 3.10 or newer is required. NumPy, PyYAML, and PyTorch are the runtime
dependencies.

## Training

Training data consists of one-dimensional integer NPY shards. Configure the
data and checkpoint paths in a YAML profile, then run:

```bash
python -m lunaris_dense_irl2.training.train \
  --config training_profiles/debug_cpu.yaml
```

The same entry point builds IRL-2, IRL Core, or Transformer according to the
profile's `model.architecture_name`. The launch scripts in `scripts/` wrap this
entry point for one node, eight GPUs, or elastic multi-node execution.

### Recreate the frozen 500M-token stream

The controlled run uses revision
`64b5dc936b797823c9dcb780430760f4c0048a84` of
`meryyllebr543/lunaris-ultrafineweb-20b-tokenized`. Only the first source shard
and tokenizer are needed:

```bash
hf download meryyllebr543/lunaris-ultrafineweb-20b-tokenized \
  --type dataset \
  --revision 64b5dc936b797823c9dcb780430760f4c0048a84 \
  --include shard_0000.npy \
  --include lunaris-tokenizer.json \
  --local-dir data/source/ultrafineweb

python scripts/prepare_architecture_benchmark_data.py \
  --source data/source/ultrafineweb/shard_0000.npy \
  --tokenizer data/source/ultrafineweb/lunaris-tokenizer.json \
  --output-root data/architecture_benchmark \
  --train-tokens 500000000 \
  --dev-tokens 4194304 \
  --holdout-tokens 4194304 \
  --sequence-length 4096
```

The resulting training split contains exactly 500,000,000 tokens. The fixed
15,258-step schedule consumes 499,974,144 of them with batch size 8 and sequence
length 4,096. Launch the qualified single-MI300X run with:

```bash
python -m lunaris_dense_irl2.training.train \
  --config training_profiles/production_irl2_135m_1x_mi300x.yaml
```

## Evaluation

Evaluate a benchmark-compatible checkpoint on fixed dev and holdout shards:

```bash
python -m lunaris_dense_irl2.training.architecture_benchmark evaluate \
  --config training_profiles/benchmark_irl2_135m_1x_mi300x.yaml \
  --checkpoint /path/to/checkpoint \
  --model-name irl2 \
  --dev-dir /path/to/dev \
  --holdout-dir /path/to/holdout \
  --output evaluation.json
```

Use `scripts/prepare_architecture_benchmark_data.py` to recreate the frozen
train/dev/holdout split from the source shard and tokenizer.

## Repository structure

```text
lunaris_dense_irl2/          IRL-2 model and shared training infrastructure
lunaris_dense_irl_core/      simplified recurrent architecture
lunaris_dense_transformer/   conventional baseline
profiles/                    model-only YAML profiles
training_profiles/           complete training YAML profiles
scripts/                     reusable data, launch, and hardware utilities
tests/                       regression-focused correctness tests
results/architecture_benchmark/  compact controlled-benchmark evidence
results/production_optimization/ compact MI300X optimization evidence
```

Model and training profiles are intentionally explicit. Review data paths,
hardware topology, precision, and checkpoint settings before a real run.
