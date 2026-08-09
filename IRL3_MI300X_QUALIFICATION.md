# IRL-3 single-MI300X qualification

This is a bounded architecture qualification for one AMD Instinct MI300X with
192 GB HBM. It is not an attempt to establish converged language-model quality.
The MI300X is not assumed to be available during implementation.

## One-command run

From the AMD Primus PyTorch/ROCm environment at the repository root:

```bash
timeout --foreground 13800s bash scripts/run_irl3_mi300x_qualification.sh
```

The outer timeout is 3 hours 50 minutes, reserving ten minutes inside a hard
four-hour allocation for artifact recovery. The runner is resumable and uses
only complete checkpoints marked `_SUCCESS`.

## What it exercises

1. ROCm device discovery, BF16 matmul, SDPA, and GQA probes.
2. The full repository test suite on the server build.
3. Deterministic data preparation and manifests.
4. A four-step BF16 IRL-3 smoke train.
5. A 136M-parameter, context-4096 train with sampled recurrent budgets.
6. Distributed-checkpoint save, reload, five-step extension, and export.
7. Fixed inference depths 2, 4, 6, 8, and 12 on one checkpoint.
8. Adaptive-halting maximum budgets 4, 8, and 12.
9. Throughput, peak HBM, CE, and recurrent diagnostics at every depth.
10. If at least 40 minutes remain, an approximately parameter- and
    sequence-matched short IRL-2 baseline and its own depth sweep.
11. JSON and Markdown research summaries with explicit interpretation limits.

## Proposed wall-clock schedule

| Elapsed | Phase | Hard phase cap | Purpose |
|---:|---|---:|---|
| 0:00-0:15 | environment, accelerator report, tests | 20 min | reject runtime incompatibility early |
| 0:15-0:35 | download/data reuse or preparation, preflight | 20 min target | verify data IDs and exact configs |
| 0:35-0:50 | four-step smoke | 15 min | forward/backward, MTP, save |
| 0:50-2:50 | 136M IRL-3 run | 120 min | stability and short learning signal |
| 2:50-3:05 | resume and portable export | 15 min | checkpoint correctness |
| 3:05-3:35 | fixed/adaptive depth sweeps | 30 min | test-time compute and halting evidence |
| 3:35-3:50 | optional IRL-2 baseline or report reserve | adaptive | baseline if time permits |
| by 3:50 | summary complete | reserve | preserve artifacts before cutoff |

The runner checks remaining wall time before starting the optional baseline.
All long phases also have local GNU `timeout` caps.

## Main IRL-3 profile

`profiles/irl3_135m_mi300x_qualification.yaml` uses:

- vocabulary 65,536, tied embeddings;
- model width 768;
- 12 fixed Transformer layers: ten prelude and two coda;
- mixed `LLLG` local/global GQA;
- state width 512;
- six fast latents, six memory slots, two slow slots;
- 128-token causal anchor-read window;
- train depth up to six and evaluation support up to 32;
- four-step gradient horizon;
- two MTP horizons;
- BF16 and recurrent-computer checkpointing.

Preflight on the current code is authoritative for the exact parameter count:

```bash
python -m lunaris_dense_irl3.training.preflight \
  --config training_profiles/irl3_135m_1x_mi300x_qualification.yaml \
  --world-size 1 \
  --require-data \
  --output qualification_artifacts/irl3/irl3_preflight.json
```

## Manual phase commands

If the one-command runner must be resumed manually:

```bash
python -m lunaris_dense_irl3.training.train \
  --config training_profiles/irl3_mi300x_smoke.yaml

python -m lunaris_dense_irl3.training.train \
  --config training_profiles/irl3_135m_1x_mi300x_qualification.yaml

python -m lunaris_dense_irl3.training.train \
  --config training_profiles/irl3_135m_1x_mi300x_qualification.yaml \
  --max-steps 245 \
  --resume auto
```

After export, the controlled depth probe is:

```bash
python scripts/benchmark_recurrent_depth.py \
  --architecture irl3 \
  --checkpoint qualification_artifacts/irl3/portable_irl3 \
  --data data/qualification/validation \
  --depths 2,4,6,8,12 \
  --sequence-length 4096 \
  --batch-size 1 \
  --batches 2 \
  --repeats 2 \
  --precision bf16 \
  --halt-mode fixed \
  --output qualification_artifacts/irl3/irl3_fixed_depth.json
```

## Artifacts

Everything is written under `qualification_artifacts/irl3/`:

- `accelerator_report.json` and `rocm_smi.txt`;
- `pytest.txt` and complete `qualification.log`;
- data manifest and IRL-2/IRL-3 preflight JSON;
- smoke, main, and optional baseline JSONL metrics;
- resumable DCP checkpoints and portable exported models;
- `irl3_fixed_depth.json` and `irl3_adaptive_depth.json`;
- optional `irl2_depth.json`;
- `summary.json` and `SUMMARY.md`.

## Acceptance and decision criteria

Required engineering gates:

- the device is reported as gfx942/MI300X and BF16 probes are finite;
- all tests pass in the actual ROCm build;
- full-path loss and gradients remain finite;
- state RMS and update/residual diagnostics do not show runaway growth;
- save, reload, extension, and portable export succeed;
- cached generation and causal tests pass;
- throughput and peak HBM are recorded for all fixed depths.

Research signals, not pass/fail claims:

- whether CE changes monotonically, saturates, or degrades with extra depth;
- whether residual ratios correlate with useful depth;
- whether halt probabilities spread across examples or collapse to the minimum;
- whether memory writes remain active without saturating erase/write gates;
- whether slow-state ticks materially change (H);
- the throughput and HBM slope per extra recurrent step;
- how the short IRL-2 control differs at comparable parameter scale.

A flat or negative depth curve after fewer than four million training tokens is
not enough to reject recurrent test-time scaling. It is enough to reveal a
broken path, unstable dynamics, a collapsed halt policy, or an inefficient
kernel layout that should be fixed before a larger experiment.
