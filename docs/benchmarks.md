# Benchmarks

Every number here comes from one benchmark suite (`dmt_bench_suite`), run with
the same fixed configuration on one machine:

- **CPU:** an AMD EPYC 9654, with 1 and 8 threads pinned one thread per core
  complex (`OMP_PROC_BIND=close OMP_PLACES='{0}:8:8'`) so each thread has its
  own 32 MB L3 slice. The load average is recorded with every run
  (`bench/results/<config>/plots/summary.md`).
- **GPU:** an NVIDIA L40S.

The rankings below hold for this machine. FDMT speed follows cache and
memory bandwidth, while brute-force DDMT is compute-bound, so another CPU or
GPU can order the engines differently.

GPU times are for **device-resident** input and output, timed with CUDA
events: they do not include copies between host and device. CPU FFTs are
planned with `FFTW_MEASURE` and a saved wisdom file, as a long-running
pipeline would. `FDMT-FFT` and `DDMT-FFT` are the Fourier-domain engines with
fractional delays (`DDMT-FFT` with the NUFFT over the DM axis).

## Setup

Three presets share DM 0–2000 on a piecewise-uniform Levin DM grid (tolerance
1.1), valid mode (streaming) and box smearing. Each figure has a tab per
preset.

`meerkat_lband` (the headline preset, shown first)
: 856–1712 MHz, 4096 channels at 612.48 µs, 4057 DM trials: a typical L-band
  survey setup.

`cryopaf_low`
: 704–1216 MHz, 512 channels at 512 µs, 1545 DM trials: the hard case for
  FDMT (few channels, long dispersion sweeps).

`cryopaf_lband`
: 1100–1714 MHz, 1152 channels at 455.6 µs, 1941 DM trials: the CryoPAF
  production configuration (its FDMTBoxcar search adds one subband level).

`parkes_uwl` (its own section, [below](#parkes-uwl-search))
: 704–4032 MHz, 6656 channels at 64 µs, 4749 DM trials (tolerance 1.6):
  the Parkes UWL search, FDMTBoxcar at subband depth 4.

Timing is steady-state streaming: construction and warm-up calls are
excluded, and each point is the median of 3 runs of one `execute()` on a
32 768-sample block (20.1 s, 16.8 s and 14.9 s of data). Every engine gets
the same DM grid; the FDMT engines round it to integer delays and keep every
trial.

Colour always identifies the algorithm (FDMT blue, FDMT-FFT orange, DDMT
green, SDMT yellow, DDMT-FFT pink). The grey line marks real time: the
duration of the data in one block. Points below it (or above 1× in the
throughput plot) keep up with the telescope.

## Key results

Reference point (32 768 samples, DM 0–2000, float32), time per block:

| engine | `meerkat_lband` CPU | `cryopaf_low` CPU | `cryopaf_lband` CPU | `meerkat_lband` GPU | `cryopaf_low` GPU | `cryopaf_lband` GPU |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| FDMT | 368 ms | 258 ms | 141 ms | 25.9 ms | 21.8 ms | 11.6 ms |
| FDMT-FFT | 439 ms | 320 ms | 183 ms | 44.1 ms | 24.9 ms | 17.5 ms |
| DDMT | 6.69 s | 342 ms | 719 ms | 158 ms | 31.9 ms | 33.8 ms |
| SDMT | 2.35 s | 333 ms | 475 ms | 67.8 ms | 31.7 ms | 12.0 ms |
| DDMT-FFT | 541 ms | 190 ms | 204 ms | 46.7 ms | 14.8 ms | 17.3 ms |

(CPU: 8 threads. FDMT-FFT and DDMT-FFT with fractional delays.)

- **Real time.** Every engine runs faster than real time on both backends,
  by 3–106× on 8 CPU threads and 127–1290× on the GPU, so the choice
  between them is about exactness and memory as much as speed.
- **`meerkat_lband`.** With 4096 channels, FDMT is the fastest engine on
  both backends: 368 ms on 8 CPU threads (54× real time, 18× faster than
  DDMT) and 25.9 ms on the L40S (774×, 6.1× faster than DDMT, 1.8× faster than
  DDMT-FFT).
- **Hard presets for FDMT.** Few channels and long dispersion sweeps (21.8k
  samples on `cryopaf_low`) give every tree level 10⁴–2×10⁴ rows, and every merge
  is a memory access, while DDMT and SDMT reuse each input window from cache
  or shared memory. On `cryopaf_low` the three are within 1.3× on the
  CPU (FDMT 258 ms, SDMT 333 ms, DDMT 342 ms). With more channels per delay,
  FDMT pulls ahead: 5.1× faster than DDMT on `cryopaf_lband`, 18× on
  `meerkat_lband`. DDMT-FFT is the fastest CPU engine on `cryopaf_low` (190 ms),
  while FDMT leads on both L-band presets.
- **On the GPU,** FDMT is 1.5× faster than DDMT and SDMT on `cryopaf_low`, and
  2.9–6.1× faster than DDMT on the L-band presets. Only SDMT on `cryopaf_lband`
  comes close (12.0 ms against 11.6 ms). DDMT-FFT is the fastest GPU
  engine on `cryopaf_low` (14.8 ms).

## Time per block vs number of DM trials

::::{tab-set}
:::{tab-item} meerkat_lband
:sync: meerkat

```{image} ../bench/results/meerkat_lband/plots/light/runtime_vs_ndms.png
:class: only-light
:alt: Time per 32K-sample block against the number of DM trials, for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
```{image} ../bench/results/meerkat_lband/plots/dark/runtime_vs_ndms.png
:class: only-dark
:alt: Time per 32K-sample block against the number of DM trials, for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
:::
:::{tab-item} cryopaf_low
:sync: cryopaf_low

```{image} ../bench/results/cryopaf_low/plots/light/runtime_vs_ndms.png
:class: only-light
:alt: Time per 32K-sample block against the number of DM trials, for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
```{image} ../bench/results/cryopaf_low/plots/dark/runtime_vs_ndms.png
:class: only-dark
:alt: Time per 32K-sample block against the number of DM trials, for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
:::
:::{tab-item} cryopaf_lband
:sync: lband

```{image} ../bench/results/cryopaf_lband/plots/light/runtime_vs_ndms.png
:class: only-light
:alt: Time per 32K-sample block against the number of DM trials, for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
```{image} ../bench/results/cryopaf_lband/plots/dark/runtime_vs_ndms.png
:class: only-dark
:alt: Time per 32K-sample block against the number of DM trials, for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
:::
::::

- **FDMT** cost follows the tree's row count, which grows with the maximum
  delay as well as the trial count.
- **Brute-force DDMT** does `nchans × ndm` additions per output sample, so it
  scales linearly with the DM count.
- **SDMT** computes the same sums as DDMT (integer output bit-identical,
  float equal to rounding) but shares partial sums between DM trials within
  subbands. The saving depends on the grid: dense trials share the most, and
  where the Levin grid gets sparse (`cryopaf_low` from DM 1000) SDMT falls back to
  the DDMT kernel. On `cryopaf_lband` it is the fastest GPU engine at every
  DM range.
- **FDMT-FFT** runs the FDMT tree in the Fourier domain, with each merge's
  shift applied as a fractional phase ramp. Its cost is dominated by the
  per-channel and per-trial FFTs, so it grows slowly with the DM count.
- **DDMT-FFT** is DDMT with exact (unrounded) per-channel delays, applied as
  phase ramps, with a NUFFT over the DM axis. It wins where the grid has many
  trials per channel (`cryopaf_low` at DM 1000–2000).

## Time per block vs block length

::::{tab-set}
:::{tab-item} meerkat_lband
:sync: meerkat

```{image} ../bench/results/meerkat_lband/plots/light/runtime_vs_nsamps.png
:class: only-light
:alt: Time per block against block length for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
```{image} ../bench/results/meerkat_lband/plots/dark/runtime_vs_nsamps.png
:class: only-dark
:alt: Time per block against block length for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
:::
:::{tab-item} cryopaf_low
:sync: cryopaf_low

```{image} ../bench/results/cryopaf_low/plots/light/runtime_vs_nsamps.png
:class: only-light
:alt: Time per block against block length for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
```{image} ../bench/results/cryopaf_low/plots/dark/runtime_vs_nsamps.png
:class: only-dark
:alt: Time per block against block length for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
:::
:::{tab-item} cryopaf_lband
:sync: lband

```{image} ../bench/results/cryopaf_lband/plots/light/runtime_vs_nsamps.png
:class: only-light
:alt: Time per block against block length for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
```{image} ../bench/results/cryopaf_lband/plots/dark/runtime_vs_nsamps.png
:class: only-dark
:alt: Time per block against block length for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
:::
::::

- **FDMT, DDMT and SDMT** are linear in the block length, so their distance to
  the real-time line stays constant as blocks grow. In valid mode the
  streaming history makes the result independent of the block size, so
  choose the block for latency.
- **FDMT-FFT and DDMT-FFT** get cheaper per sample with longer blocks: they
  transform long blocks in overlap-save segments, and the overlap (the
  dispersion sweep) is amortised over more samples.

## Real-time throughput by input width

::::{tab-set}
:::{tab-item} meerkat_lband
:sync: meerkat

```{image} ../bench/results/meerkat_lband/plots/light/throughput_nbits.png
:class: only-light
:alt: Real-time factor against input sample width for FDMT, DDMT, SDMT, FDMT-FFT and DDMT-FFT
```
```{image} ../bench/results/meerkat_lband/plots/dark/throughput_nbits.png
:class: only-dark
:alt: Real-time factor against input sample width for FDMT, DDMT, SDMT, FDMT-FFT and DDMT-FFT
```
:::
:::{tab-item} cryopaf_low
:sync: cryopaf_low

```{image} ../bench/results/cryopaf_low/plots/light/throughput_nbits.png
:class: only-light
:alt: Real-time factor against input sample width for FDMT, DDMT, SDMT, FDMT-FFT and DDMT-FFT
```
```{image} ../bench/results/cryopaf_low/plots/dark/throughput_nbits.png
:class: only-dark
:alt: Real-time factor against input sample width for FDMT, DDMT, SDMT, FDMT-FFT and DDMT-FFT
```
:::
:::{tab-item} cryopaf_lband
:sync: lband

```{image} ../bench/results/cryopaf_lband/plots/light/throughput_nbits.png
:class: only-light
:alt: Real-time factor against input sample width for FDMT, DDMT, SDMT, FDMT-FFT and DDMT-FFT
```
```{image} ../bench/results/cryopaf_lband/plots/dark/throughput_nbits.png
:class: only-dark
:alt: Real-time factor against input sample width for FDMT, DDMT, SDMT, FDMT-FFT and DDMT-FFT
```
:::
::::

The real-time factor is seconds of data processed per second of compute.

- **Packed low-bit input** runs the FDMT tree in narrow integer lanes. On
  the CPU that is 1.2–1.5× faster than float32 at 1–4 bits on one thread
  and 1.2–1.4× on 8 threads (`cryopaf_lband`: 148× against 105× real time at 1
  bit). On the GPU, 1–2-bit input gains on `cryopaf_lband`
  (1587× against 1290×) and loses on `cryopaf_low` and `meerkat_lband`, where the
  float32 path uses the direct bottom and top kernels.
- At 16 bits the tree is mostly float again, so packed input is no faster
  than float32 (on the GPU it is 1.3–1.6× slower).
- **DDMT** sums packed input two samples per instruction in 16-bit lanes.
  **SDMT** shares sums only for float32 and 16-bit input on the GPU; at 1–8
  bits it runs the DDMT kernel.
- **The Fourier-domain engines** transform complex float spectra whatever the
  input width, so their speed hardly depends on it.

## Operation count

::::{tab-set}
:::{tab-item} meerkat_lband
:sync: meerkat

```{image} ../bench/results/meerkat_lband/plots/light/ops_theory.png
:class: only-light
:alt: Operations per output sample against DM trials and against channels, for FDMT, FDMT-FFT and DDMT
```
```{image} ../bench/results/meerkat_lband/plots/dark/ops_theory.png
:class: only-dark
:alt: Operations per output sample against DM trials and against channels, for FDMT, FDMT-FFT and DDMT
```
:::
:::{tab-item} cryopaf_low
:sync: cryopaf_low

```{image} ../bench/results/cryopaf_low/plots/light/ops_theory.png
:class: only-light
:alt: Operations per output sample against DM trials and against channels, for FDMT, FDMT-FFT and DDMT
```
```{image} ../bench/results/cryopaf_low/plots/dark/ops_theory.png
:class: only-dark
:alt: Operations per output sample against DM trials and against channels, for FDMT, FDMT-FFT and DDMT
```
:::
:::{tab-item} cryopaf_lband
:sync: lband

```{image} ../bench/results/cryopaf_lband/plots/light/ops_theory.png
:class: only-light
:alt: Operations per output sample against DM trials and against channels, for FDMT, FDMT-FFT and DDMT
```
```{image} ../bench/results/cryopaf_lband/plots/dark/ops_theory.png
:class: only-dark
:alt: Operations per output sample against DM trials and against channels, for FDMT, FDMT-FFT and DDMT
```
:::
::::

The theoretical cost per output time sample, summed over all DM trials:

- **DDMT**: `nchans × ndm` additions (brute force). SDMT is not drawn: its
  count depends on the DM grid.
- **FDMT**: the plan's exact addition count (`FDMTPlan.total_operations`).
- **FDMT-FFT**: a flop model. It counts forward FFTs of every channel,
  inverse FFTs of every DM row, and one complex multiply-add per tree node
  and frequency bin.

FDMT's advantage grows with both the channel count and the DM count. How
much of it shows up in wall-clock time depends on the platform: FDMT does
fewer operations, but each one is a memory access, so its speed follows the
memory (or cache) bandwidth, while brute force's inner loop is
compute-bound and reuses each input window across a tile of trials.

## Parkes UWL search

`parkes_uwl` is the search the skewed-window path was built for: 6656
channels over 704–4032 MHz at 64 µs, DM 0–2000 (Levin tolerance 1.6, 4749
trials, a 16.2 s sweep), searched by `FDMTBoxcar` at subband depth 4 (the
full band, 2 + 1, 4 + 3, 8 + 7 and 16 + 15 tree and staggered subbands:
218 000 searched rows) with boxcars to 100 ms (wtsp 1.5), within the default
10 GiB memory budget. It is not in the five-engine figures above (FDMT-FFT
needs ~50 GB on it, and brute-force CPU DDMT hours).

The real-time sweep runs the same search at the native 64 µs and
downsampled to 256, 512 and 1024 µs (each with its own DM grid and 100 ms
of widths), on blocks of 2.1 s after the stream is warm:

```{image} ../bench/results/parkes_uwl/plots/light/uwl_realtime.png
:class: only-light
:alt: Real-time factor of the Parkes UWL depth-4 search against time resolution, GPU and 8 CPU threads, uint8 and float32
```
```{image} ../bench/results/parkes_uwl/plots/dark/uwl_realtime.png
:class: only-dark
:alt: Real-time factor of the Parkes UWL depth-4 search against time resolution, GPU and 8 CPU threads, uint8 and float32
```

| time resolution | GPU, uint8 | GPU, float32 | CPU (8 threads), uint8 | CPU (8 threads), float32 |
| ---: | ---: | ---: | ---: | ---: |
| 64 µs (4749 trials) | **1.64×**, 9.4 GiB | **1.15×**, 10.8 GiB | 0.14×, 6.2 GiB | 0.08×, 9.3 GiB |
| 256 µs (2302 trials) | 36×, 3.4 GiB | 38×, 3.5 GiB | **1.13×**, 3.3 GiB | **1.34×**, 3.5 GiB |
| 512 µs (1153 trials) | 161×, 0.7 GiB | 163×, 0.8 GiB | 6.2×, 1.8 GiB | 6.8×, 2.0 GiB |
| 1024 µs (577 trials) | 800×, 0.3 GiB | 940×, 0.3 GiB | 19×, 0.6 GiB | 25×, 0.7 GiB |

(real-time factor = block duration / time per block; working memory
includes the input block.)

- **The GPU searches depth 4 at the native 64 µs faster than real time**
  within 10 GiB, on 8-bit and float32 input. The tree path needed 15 GB for
  the same search.
- **Each halving of the resolution makes the search 4–5× cheaper**: fewer
  samples, half the DM trials, and a tree half as deep in delay. From
  256 µs 8 CPU threads keep up. At high DM the band's bottom channels are
  smeared far beyond 256 µs anyway (24 ms within one 0.5 MHz channel at
  704 MHz and DM 2000), so a coarser search mostly costs sensitivity to
  narrow, low-DM bursts; a real-time system can search at 256–512 µs and
  keep the 64 µs data around triggers.
- **The CPU at 64 µs** is 7–12× slower than real time on 8 threads: fine
  for offline processing of recorded files, not for real time.

The full-band search alone (subband depth 0) takes 0.36 s per 2.1 s block on
the GPU (5.8× real time, 2.1–2.7 GiB). Exact numbers, chunks and the
reference-point runs are in `bench/results/parkes_uwl/plots/summary.md`.

## Coherent hybrid search (CohFDMT)

CohFDMT searches baseband voltages rather than a filterbank, so it has its own
fixed configuration and is not compared with the engines above:

- **Reference configuration:** one GUPPI node (64 × 2.93 MHz at
  1.31–1.50 GHz, int8 FTPRI), t_p = 10 µs (n_p = 30), DM 50–60 pc cm⁻³, and
  the automatic block length.
- **Sweeps:** each one passes through the reference point and varies the
  target resolution, the input bit width, the DM range width (and with it the
  number of coarse coherent trials) or the block length.
- **Real-time factor:** seconds of baseband searched per second of compute,
  counting one stride per block, i.e. excluding the overlap.

```{image} ../bench/results/cfdmt/plots/light/cfdmt_rtf.png
:class: only-light
:alt: CohFDMT real-time factor against t_p, input bits, DM range width and block length
```
```{image} ../bench/results/cfdmt/plots/dark/cfdmt_rtf.png
:class: only-dark
:alt: CohFDMT real-time factor against t_p, input bits, DM range width and block length
```

With DM ranges of a few tens of pc cm⁻³ at L-band a single coherent trial is
enough, and the time goes to the forward transform of the block, shared by
all trials. Each additional coarse trial for a wider DM range adds its own
inverse transforms and one fine FDMT.

## Reproducing

The suite, the runner and the plotting scripts live in `bench/`.
`bench/README.md` gives the steps (build; `run_suite.py` with `--suite main`,
`boxcar` or `cfdmt` per preset; `plot_suite.py`, `plot_uwl.py`,
`plot_cfdmt.py`) and the tips for comparable numbers. The raw results are the
JSON files under `bench/results/<config>/l40s-epyc9654/`.

```{seealso}
- [Fourier-domain variants](pipeline_guide/fourier_variants.md)
- [Streaming and history](pipeline_guide/streaming_and_history.md)
```
