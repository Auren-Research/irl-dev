# Single-MI300X Qualification

This run is a bounded hardware and architecture qualification, not a claim of
converged language-model quality.

## Scope

- one AMD Instinct MI300X with 192 GB HBM;
- AMD Primus PyTorch/ROCm runtime;
- one 1B-token Ultra-FineWeb `uint32` NPY shard;
- a non-overlapping 100M-token train subset and 1M-token validation subset;
- the 65,536-token Lunaris BPE tokenizer;
- a 135,153,937-parameter dense model;
- mixed local/global GQA attention;
- two recurrent IRL-2 iterations;
- two auxiliary MTP horizons;
- BF16 single-GPU training.

The main run performs 500 updates with 32,768 tokens per update, for a total of
16,384,000 training tokens. It then reloads the final distributed checkpoint
and extends the run to step 510 to verify accelerator-side resume behavior.

## Why this model size

The existing 120M profile uses a 48,000-token vocabulary and cannot consume the
Ultra-FineWeb token IDs directly. The qualification profile uses the dataset's
full 65,536-token vocabulary, increasing the tied embedding model to
135,153,937 parameters.

The 900M profile is intentionally excluded. A short run at that size would
consume budget without improving the first-pass validation of IRL-2, MTP,
attention, data loading, checkpointing, and ROCm compatibility.

## Run

Execute the following from inside the AMD Primus PyTorch environment:

```bash
bash scripts/run_mi300x_qualification.sh
```

The runner is resumable. Completed downloads and prepared shards are reused,
and the main training configuration resumes from the latest complete
checkpoint.

## Evidence

Generated files are written under `qualification_artifacts/`:

- `accelerator_report.json`: PyTorch/ROCm, device, BF16 matmul, and SDPA/GQA;
- `rocm_smi.txt`: accelerator identity, HBM, utilization, temperature, power;
- `pytest.txt`: unit and integration test results;
- `data_manifest.json`: source range, token range, tokenizer hash, and splits;
- `smoke_metrics.jsonl`: four-step full-path safety run;
- `main_metrics.jsonl`: main training and validation curve;
- `main_config.json`: fully resolved configuration;
- `summary.json` and `SUMMARY.md`: compact qualification results;
- `qualification.log`: complete runner output.

The acceptance conditions are:

1. ROCm detects the MI300X and BF16 probes are finite;
2. all repository tests pass;
3. the GPU smoke test exercises two IRL iterations and two MTP heads;
4. training, auxiliary, gradient, and validation metrics remain finite;
5. training and validation cross-entropy show an initial downward trend;
6. checkpoints save and reload successfully;
7. throughput and peak HBM usage are recorded.

