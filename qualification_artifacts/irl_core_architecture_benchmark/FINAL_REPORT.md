# 135M Controlled Architecture Benchmark

Source commit used for all trajectories: `9a04c391d8d7595b57b13ab04fb2821ba0466554`.

| Model | Params | ~100M CE | ~250M CE | ~500M CE | tok/s | FLOPs/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer | 132,147,968 | 4.287523 | 3.913668 | 3.611391 | 130,770 | 1.057e+09 |
| IRL-2 | 132,197,136 | 4.274335 | 3.900999 | 3.594243 | 93,875 | 1.397e+09 |
| IRL Core | 132,149,504 | 4.285132 | 3.906721 | 3.601472 | 102,955 | 1.387e+09 |

The CE values above are from the frozen confirmation holdout. This benchmark is parameter- and token-matched, not FLOP-matched.

## Final paired comparisons

| Comparison | Delta CE | Paired 95% CI | Batches |
| --- | ---: | ---: | ---: |
| Core-Transformer | -0.009919 | [-0.010938, -0.008900] | 128 |
| IRL2-Transformer | -0.017148 | [-0.017951, -0.016345] | 128 |
| Core-IRL2 | +0.007229 | [+0.006318, +0.008140] | 128 |

## Decision

- IRL Core vs Transformer: `left_better`.
- IRL-2 vs Transformer: `left_better`.
- IRL Core vs IRL-2: `right_better`.
- Core recurrent-time utility emerged by the declared criterion: `True`.
- Core requires `1.312x` estimated training FLOPs/token and `1.267x` measured training wall time relative to the Transformer.
- Core's interpolated token saving to the Transformer-final CE is `1.625%`, but it requires `1.291x` estimated training FLOPs and `1.247x` estimated wall time to reach that CE.
- At the Transformer's final cumulative estimated-FLOP budget, Core is interpolated at CE `3.746828` (`+0.135437` versus the Transformer final CE).
- Scaling decision: `promising_but_do_not_proceed_directly`.

## Explicit benchmark questions

1. **Does IRL Core outperform the Transformer at equal parameters and tokens?** Yes. At the final checkpoint Core is lower by `-0.009919` CE with paired 95% CI `[-0.010938, -0.008900]`.
2. **Does IRL-2 outperform the Transformer?** Yes. IRL-2 is lower by `-0.017148` CE with paired 95% CI `[-0.017951, -0.016345]`.
3. **Does IRL Core outperform IRL-2?** No. Core is higher by `0.007229` CE with paired 95% CI `[0.006318, 0.008140]`.
4. **Does Core recurrent-time utility emerge?** Yes. Dev `CE6-CE1` changes from `-0.021825` at 100M to `-0.030682` at 250M and `-0.049494` at 500M. `CE6-CE2` changes from `-0.006838` to `-0.010048` and `-0.020324`.
5. **What does the Core quality win cost?** At the same 499,974,144 training tokens, Core uses `1.312x` estimated FLOPs/token, `1.267x` training wall time, and runs at `0.787x` Transformer throughput. Peak allocated HBM is `38.33` GiB versus `31.34` GiB.
6. **Where is the advantage strongest?** Parameter and token efficiency. Core saves an interpolated `1.625%` of tokens to the Transformer-final CE, but needs `1.291x` estimated FLOPs and `1.247x` estimated wall time to reach it. IRL-2 saves `2.795%` of tokens but needs `1.284x` estimated FLOPs and `1.350x` estimated wall time.
7. **Does this justify immediate substantial compute on the 501.8M Core candidate?** Not by itself. Core establishes a statistically resolved Transformer win and genuine within-checkpoint recurrent utility, but IRL-2 is significantly better and Core is worse on estimated-FLOP and wall-clock efficiency. The evidence is promising, but it does not clear an unconditional direct-to-scale gate for this specific Core candidate.

## Optimization and mechanism health

Final cumulative clipped-step fractions are `2.346%` (Transformer), `2.314%` (IRL-2), and `2.392%` (Core); mature training was not permanently clipped. Core diagnostics are finite at every milestone. At 500M, damping min/mean/max is `0.7380` / `0.8508` / `0.9057`, state RMS across six steps is `1.4733, 1.7125, 1.7113, 1.6288, 1.6037, 1.6223`, and relative update norm decays to `0.0639`.

The machine-readable summary, paired batch observations, depth curves, clipping data, timings, and plots in this directory are the authoritative evidence. Linear token-efficiency interpolation is reported only when the observed 100M–500M curve brackets the Transformer-at-500M target; no out-of-range extrapolation is used.
