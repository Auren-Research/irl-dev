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
```

Model and training profiles are intentionally explicit. Review data paths,
hardware topology, precision, and checkpoint settings before a real run.
