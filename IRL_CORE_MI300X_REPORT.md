# IRL Core MI300X validation

Date: 2026-08-14

Starting `main`: `1337d3b6aa5dffbb1adf1d1e0583184a48c0cb6d`

Implementation under test: `13305d7f4f1b895168ef51aab17f1456cef797b3`

The simplified architecture passed its local and MI300X correctness gates. The
500M profile fits with substantial HBM headroom, completes BF16 optimizer steps,
and resumes from a model-plus-optimizer checkpoint. These are engineering
qualification results, not evidence of language-model quality.

## Environment

- Container: `rocm` in the supplied AMD Primus environment
- Device: one AMD Instinct MI300X VF, `gfx942:sramecc+:xnack-`
- Visible HBM: 205,822,885,888 bytes (191.69 GiB)
- PyTorch: `2.12.0+rocm7.14.0a20260608`
- HIP runtime reported by PyTorch: `7.14.60850`
- `torch.cuda.is_available()`: true
- BF16 support: true

The preinstalled ROCm/PyTorch stack was used unchanged.

## Exact profiles

| Profile | Parameters | Trainable | IRL Core | T | Attention calls | Est. train FLOPs/token |
|---|---:|---:|---:|---:|---:|---:|
| 135M engineering | 132,149,504 | 132,149,504 | 5,115,008 | 6 | 21 | 1.387B |
| 500M candidate | 501,813,504 | 501,813,504 | 20,060,352 | 6 | 25 | 4.615B |

An attention call is one ordinary prelude/coda attention or one recurrent core
read. The core performs only one anchor K/V projection pair despite its six
attention reads. The FLOPs values are approximate forward-plus-backward
estimates from the repository estimator.

## Full-context BF16 measurements

| Profile / shape | Tokens/s | Step | Forward | Backward | Optimizer | Peak allocated | Reserved | Compile estimate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 135M, B=8, L=4096 | 99,136 | 330.5 ms | 86.2 ms | 240.9 ms | 3.4 ms | 41.26 GiB | 44.35 GiB | 28.33 s |
| 500M, B=1, L=4096 | 27,198 | 150.6 ms | 37.2 ms | 104.7 ms | 8.6 ms | 17.58 GiB | 18.77 GiB | 11.06 s |

The 135M benchmark used two warmup and five measured optimizer steps. Loss fell
from 11.0880 to 10.6042, and every measured core query gradient was finite and
nonzero. Peak HBM left about 150.43 GiB physically available.

The 500M benchmark used two warmup and three measured optimizer steps. Loss
fell from 11.5250 to 10.6876. It left about 174.11 GiB above peak allocated
memory. A 6,022,056,039-byte model-plus-optimizer checkpoint was loaded back,
and the resumed optimizer step completed with finite loss 10.3301. The large
temporary checkpoint was then deleted; only compact results were retained.

The reported compile estimates are first-step overhead relative to warmed step
time and include lazy kernel compilation/autotuning. They are not isolated
compiler CPU-time measurements.

## Shared training-loop sanity run

The real training entry point ran the 135M profile for five steps at full
context with microbatch 2 and four-way accumulation (32,768 tokens/update).
After the compilation-bearing first interval, mean throughput for steps 2-5 was
83,256 tokens/s. Loss moved from 11.2584 to 11.2444 on a temporary synthetic
shard.

Pre-clip gradient norms were 2.763, 1.475, 0.911, 0.589, and 0.506. The newly
logged clipped-step fraction was 100%, 100%, 0%, 0%, and 0%, respectively. This
short startup run therefore exercised the diagnostic and did not remain
permanently clipped. It does not replace monitoring over a mature real-data
run.

## Recurrent-state diagnostic

After the 135M sanity optimization, one bounded T=2/T=6/T=12 diagnostic gave
losses 10.46230, 10.46234, and 10.46233 on the same synthetic batch. No depth
was materially better after only a few updates, but extra depth was not
pathological. At T=6, state RMS approached 0.20386 and update RMS fell from
0.18348 to 0.00024; all values were finite. Mean damping remained 0.90000.

## Compilation behavior

The production runs used the repository's regional compile path. A separate
bounded 135M B=2, L=256 smoke verified that whole-model `torch.compile` can
capture and execute the fixed six-step recurrent loop. It took approximately
165 seconds to compile and then ran at 14,717 tokens/s for that small shape.
PyTorch emitted an existing warning about tracing the LRU-cached local
FlexAttention block-mask helper; no incorrectness was observed. Given the high
one-time cost, whole-model compilation remains disabled in the supplied
production profiles.

## Tests and comparison context

- Local: 40 pytest tests passed; Ruff and `git diff --check` passed.
- MI300X container: the same 40 tests passed, including its real-GPU BF16 test.
- Ruff was not installed in the Primus image; it was not added because lint had
  already passed locally and the accelerator software stack was left untouched.

The direct 135M benchmark is about 5.3x the cited 18.7k tokens/s IRL-3 result;
the shared training loop is about 4.45x. The historical batch, sequence, loss,
and compilation settings were not available here, so this is only engineering
context, not an apples-to-apples architecture or quality comparison. No
verified IRL-2 throughput number was present in the repository, so none is
invented.

Raw compact values are in
`benchmarks/irl_core_mi300x_2026-08-14.json`.
