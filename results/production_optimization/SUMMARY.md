# IRL-2 MI300X production optimization

## Outcome

The accepted model preserves the complete IRL-2 nucleus in all 16 layers: a
local anchor/state recurrence, one shared recurrent map across its internal
iterations, learned damping, and two trained refinement steps. At exactly the
same 132,197,136 parameters and 499,974,144-token trajectory, it improves both
quality and efficiency over the frozen IRL-2 result.

| Metric | Frozen IRL-2 | Production IRL-2 | Change |
| --- | ---: | ---: | ---: |
| Holdout CE | 3.594243 | **3.583993** | **-0.010249** |
| Median training tok/s | 93,875 | **193,310** | **2.059x** |
| End-to-end tok/s | 93,105 | **185,727** | **1.995x** |
| Peak allocated HBM | 53.10 GiB | **32.02 GiB** | **-39.7%** |
| Peak reserved HBM | 53.85 GiB | **33.95 GiB** | **-37.0%** |
| Estimated FLOPs/token | 1.3967B | **1.3118B** | **-6.08%** |
| Aggregate training wall | 1.492 h | **0.748 h** | **-49.9%** |
| Clipped-step fraction | 2.314% | **2.287%** | -0.026 pp |

The new wall time is the conservative sum of four logged training segments,
including four lazy compilation starts caused by controlled milestone stops.
The measured uninterrupted projection is 0.708 h. Estimated algorithmic
throughput rises from 131.1 to 253.6 TFLOP/s, or from 10.0% to 19.4% of the
MI300X's published dense BF16 peak; this is an estimated model-FLOP utilization,
not a kernel-counter measurement.

The final holdout comparison uses the same 128 aligned batches. Production
minus frozen CE is -0.010249 with standard error 0.000795 and a normal 95%
paired CI of [-0.011807, -0.008691]; production wins 115/128 batches. The gain
is therefore supported by the fixed evaluation rather than treated as noise.

## Learning curve and recurrence check

| Training tokens | Frozen holdout CE | Production holdout CE | Delta |
| ---: | ---: | ---: | ---: |
| 100,007,936 | 4.274335 | **4.232317** | -0.042018 |
| 249,987,072 | 3.900999 | **3.883779** | -0.017221 |
| 499,974,144 | 3.594243 | **3.583993** | -0.010249 |

The trained depth remains essential. On final dev, T=2 gives CE 3.586515,
whereas T=1 gives 6.504122 and untrained extra iterations give 4.141421 at T=4
and 3.920981 at T=6. The optimized path neither bypasses the recurrence nor
turns it into an ordinary FFN. Layer damping also moved from its 0.9
initialization to a learned 0.676-0.879 range (mean 0.771).

## What changed

Profiling showed excessive launch and memory traffic from separate Q/K/V and
SwiGLU projections, explicit normalization/cast operations, expanded RoPE
tensors, and a dominant local FlexAttention backward. The accepted systems path
therefore uses fused QKV and gate/up projections with strict legacy-checkpoint
migration, native RMSNorm, compact pairwise RoPE, an exact ROCm FlashAttention
window, an 8,192-token loss chunk, and whole-model `torch.compile(mode="default")`.
Default compile matched the best steady speed while avoiding the longer tuning
start and higher reservation of the other tested modes.

The architecture screen then reduced the original four global-attention layers
to one terminal global layer, leaving 15 exact 1,024-token local layers. This
reduced estimated FLOPs/token and improved quality. A two-global version was the
runner-up. Parameter-matched deep/thin and wide/shallow variants were rejected
after 100M-token screens despite the wide model's higher speed, because both had
worse holdout CE. Exact screen values are in `architecture_screens.csv`; kernel
and compile stages are in `systems_screens.csv`.

## Data and reproducibility

The source is
[`meryyllebr543/lunaris-ultrafineweb-20b-tokenized`](https://huggingface.co/datasets/meryyllebr543/lunaris-ultrafineweb-20b-tokenized)
at revision `64b5dc936b797823c9dcb780430760f4c0048a84`. The tokenizer SHA-256 is
`081ed20be80cac199d24b5e683d7cf266a5421213425ba7178895ab707d17a78`.
The non-overlapping train/dev/holdout hashes and exact preparation command are
recorded in `summary.json` and the repository README.

The qualified environment was one AMD Instinct MI300X VF, PyTorch
2.12.0+rocm7.14.0a20260608, HIP 7.14.60850, BF16, and FlashAttention 2.8.3.
Raw final evaluation and qualification records are retained beside this file;
large loser checkpoints, profiler traces, and temporary scripts are not.
The qualified implementation commit is
`cbdcbdfa2b5cf321f00421160200efc5c0b16acf`.

## Research basis and scale decision

The exact attention kernel follows the IO-aware design of
[FlashAttention](https://arxiv.org/abs/2205.14135). The local/global screen was
motivated by the interleaved pattern in the
[Gemma 2 report](https://arxiv.org/abs/2408.00118), but the final one-global
schedule was selected only by the controlled IRL-2 experiments. The deep/thin
candidate was motivated by
[MobileLLM](https://arxiv.org/abs/2402.14905) and rejected here. Compilation and
profiling choices follow the official
[PyTorch compile documentation](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
and [AMD MI300X tuning guidance](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/workload.html).

Recommendation: the architecture and single-MI300X training path are ready for
a scale pilot. Before committing a large multi-GPU budget, qualify the same
graph on the intended 8-GPU MI300X topology, including RCCL overlap, distributed
checkpoint cadence/resume, compiled-graph cache warmup, and per-rank numerical
agreement. No 500M-parameter run was started in this work.
