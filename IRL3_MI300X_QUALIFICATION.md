# IRL-3 single-MI300X qualification

This is a bounded architecture qualification for one AMD Instinct MI300X with
192 GB HBM. It is intended to falsify or support implementation and mechanism
hypotheses, not to establish converged language-model quality.

## One-command run

From the AMD Primus PyTorch/ROCm environment at the repository root:

```bash
timeout --foreground 13800s bash scripts/run_irl3_mi300x_qualification.sh
```

The outer timeout is 3 hours 50 minutes, preserving ten minutes inside a hard
four-hour allocation for artifact recovery. The runner is resumable and uses
only distributed checkpoints marked `_SUCCESS`.

## Server requirements

- Linux with one visible gfx942/MI300X accelerator and 192 GB HBM;
- AMD Primus or another ROCm PyTorch environment with Python 3.10 or newer,
  `torch>=2.7`, BF16, SDPA GQA backward, and `torch.version.hip` available;
- Bash, GNU `timeout`, GNU `find` with `-printf`, GNU `date`, `curl`, and normal
  coreutils;
- at least 50 GB of writable local disk for the source shard, prepared data,
  checkpoints, exports, compiler cache, and reports;
- outbound HTTPS access to Hugging Face, or these files pre-staged at
  `data/source/ultrafineweb/`: `shard_0000.npy` and
  `lunaris-tokenizer.json`;
- permission to install the repository editable plus lightweight `numpy`,
  `pytest`, and `PyYAML` dependencies. The bootstrap never replaces the
  server's ROCm PyTorch build.

Set `PYTHON_BIN` if the desired interpreter is not `python3`. The runner sets
`PYTORCH_ROCM_ARCH=gfx942`, `GPU_MAX_HW_QUEUES=2`, and a repository-local
TorchInductor cache unless those variables are already supplied.

## What the run exercises

1. ROCm discovery plus finite BF16 matmul and SDPA/GQA backward probes.
2. The complete repository test suite on the actual server build.
3. Deterministic data preparation, manifests, and both model preflights.
4. A four-step BF16 IRL-3 train through MTP, equilibrium, optimizer, and DCP.
5. A 134.8M-parameter, context-4096 IRL-3 run with sampled recurrent budgets.
6. DCP reload, five-step extension, and portable checkpoint export.
7. A guaranteed parameter/sequence-matched short IRL-2 train and depth sweep.
8. One IRL-3 checkpoint evaluated at fixed depths 2/4/6/8/12.
9. That same checkpoint evaluated with ACT-soft and adaptive-hard budgets
   4/8/12.
10. A same-checkpoint, fixed-depth-six write-back-off control. The computer
    still executes, but its correction is withheld from the coda.
11. CE, latency, throughput, peak HBM, analytic FLOPs, halting distributions,
    Z/H/M dynamics, slow ticks, memory activity, and correction-to-anchor ratio.
12. Machine-readable JSON plus a Markdown report that places soft and hard
    halting side by side and reports OFF-minus-ON CE explicitly.

## Bounded schedule

| Elapsed | Phase | Purpose |
|---:|---|---|
| 0:00-0:15 | environment, accelerator report, tests | reject runtime incompatibility early |
| 0:15-0:30 | data reuse/preparation and preflight | freeze data/config identity |
| 0:30-0:42 | four-step IRL-3 smoke | forward/backward, MTP, save |
| 0:42-2:22 | 200-step IRL-3 run | stability and short learning signal |
| 2:22-2:32 | five-step resume and export | checkpoint correctness |
| 2:32-3:02 | guaranteed 60-step IRL-2 run/export/sweep | matched path control |
| 3:02-3:37 | fixed, ACT, adaptive, and write-back-off probes | mechanism and cost evidence |
| 3:37-3:45 | summary and artifact verification | preserve conclusions before cutoff |
| 3:45-3:50 | recovery reserve | copy or inspect artifacts |

The 200-step IRL-3 budget is 3,276,800 observed training tokens. The guaranteed
IRL-2 control is 60 steps and 983,040 tokens. They are deliberately not
presented as converged or token-matched quality comparisons.

## Main architecture profile

`profiles/irl3_135m_mi300x_qualification.yaml` remains unchanged and uses:

- vocabulary 65,536 with tied embeddings;
- model width 768;
- 12 fixed Transformer layers: ten prelude and two coda;
- mixed `LLLG` local/global GQA;
- state width 512;
- six fast latents, six memory slots, and two slow slots;
- a 128-token causal anchor-read window;
- train depth up to six and evaluation support up to 32;
- a four-step gradient horizon and two MTP horizons;
- BF16 and recurrent-computer checkpointing.

Preflight is authoritative for parameter count and now records the complete
analytic FLOPs component breakdown:

```bash
python -m lunaris_dense_irl3.training.preflight \
  --config training_profiles/irl3_135m_1x_mi300x_qualification.yaml \
  --world-size 1 \
  --require-data \
  --output qualification_artifacts/irl3/irl3_preflight.json
```

At depth six with two active MTP horizons, the corrected estimator reports
approximately 1.254 billion forward FLOPs and 4.025 billion executed training
FLOPs per observed token. The latter includes the four-step credit horizon and
computer checkpoint recomputation. It counts dense linear and attention
matmuls, not normalizations, nonlinearities, softmax/masking, reductions, or
kernel-launch overhead. Measured MI300X time is authoritative.

## Manual inference controls

After portable export, the fixed probe is:

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

Use `--halt-mode act` or `--halt-mode adaptive` with `--depths 4,8,12` for
the soft/hard comparison. Add `--disable-writeback --depths 6 --halt-mode
fixed` for the evaluation-only OFF control. The runner encodes the exact
commands and common data/checkpoint paths.

## Artifacts

Everything is written under `qualification_artifacts/irl3/`:

- `qualification.log`, `pytest.txt`, `accelerator_report.json`, and
  `rocm_smi.txt`;
- `data_manifest.json`, `irl3_preflight.json`, and `irl2_preflight.json`;
- `smoke_metrics.jsonl`, `main_metrics.jsonl`, and
  `irl2_baseline_metrics.jsonl`;
- resumable checkpoints under `checkpoints/`;
- `portable_irl3/` and `portable_irl2/`;
- `irl3_fixed_depth.json`, `irl3_act_depth.json`,
  `irl3_adaptive_depth.json`, and `irl3_writeback_off.json`;
- `irl2_depth.json`;
- `summary.json` and `SUMMARY.md`.

Benchmark JSON schema version 2 records CE samples, median batch latency,
tokens/second, timed and diagnostic peak memory, budget and diagnostic-batch
analytic FLOPs, the evaluation variant, and recurrent diagnostics. The
summarizer rejects IRL-3 input reports that reference different checkpoints.

## Engineering gates and research questions

Required engineering gates:

- gfx942/MI300X and finite BF16 probes are recorded;
- all tests pass on the ROCm build;
- loss and gradients remain finite through full-path training;
- save, reload, five-step extension, export, and benchmark reload succeed;
- no Z/H/M state RMS, update, or residual exhibits runaway growth;
- timed latency, throughput, and peak HBM exist for every requested variant.

The report is designed to answer without selecting only favorable results:

1. Are the recurrent dynamics numerically stable and gradients nonzero?
2. Does same-checkpoint write-back OFF increase, preserve, or reduce CE?
3. Does extra fixed depth improve, saturate, or degrade CE?
4. Do ACT-soft and adaptive-hard differ at the same maximum budgets?
5. Does the halt distribution spread or collapse to a boundary?
6. Are Z/H/M updates, memory writes, and slow ticks active rather than zero?
7. Is the correction-to-anchor ratio material or suppressed?
8. What are the measured compute, latency, throughput, and HBM slopes?
9. How does the short parameter-matched IRL-2 control behave?

Joint residual ratios are clock-confounded because H moves periodically. The
report therefore includes a slow-update mask and a fast Z/M contraction ratio;
the last meaningful H equilibrium residual uses the transition immediately
before and after the most recent slow tick.

A flat or negative curve after roughly three million IRL-3 tokens is not enough
to reject recurrent test-time scaling. It is enough to expose a broken path,
unstable dynamics, collapsed routing, unused write-back, or impractical kernel
cost that should be addressed before a larger experiment.
