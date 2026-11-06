# Benchmarks

Every number here comes from one benchmark suite (`dmt_bench_suite`), run with
the same fixed configuration on one machine: a GPU worker of a shared
HTCondor cluster, as the code runs in production.

- **CPU:** an AMD EPYC 9334 virtual machine, in an 8-CPU batch slot. Threads
  are not pinned: the VM hides which virtual CPUs share an L3 cache, the slot
  reserves no cores, and other jobs run on the same node. The load average
  is recorded with every run (`bench/results/[<config>/]plots/summary.md`).
- **GPU:** an NVIDIA L40, assigned to the job.

The rankings below hold for this machine. FDMT speed follows cache and
memory bandwidth, while brute-force DDMT is compute-bound, so another CPU or
GPU can order the engines differently. Results from the earlier machines
(an Intel Xeon Gold 6348H and an NVIDIA L40S, dmt 0.7.0) are kept under
`bench/results/[<config>/]other/` and summarised at the end of this page.

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

`cryopaf`
: 704–1216 MHz, 512 channels at 512 µs, 1545 DM trials: the hard case for
  FDMT (few channels, long dispersion sweeps).

`cryopaf_lband`
: 1100–1714 MHz, 1152 channels at 455.6 µs, 1941 DM trials.

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

| engine | `meerkat_lband` CPU | `cryopaf` CPU | `cryopaf_lband` CPU | `meerkat_lband` GPU | `cryopaf` GPU | `cryopaf_lband` GPU |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| FDMT | 251 ms | 200 ms | 110 ms | 27.8 ms | 21.9 ms | 12.4 ms |
| FDMT-FFT | 326 ms | 186 ms | 109 ms | 48.4 ms | 30.7 ms | 19.7 ms |
| DDMT | 3.87 s | 214 ms | 413 ms | 167 ms | 42.9 ms | 37.2 ms |
| SDMT | 1.46 s | 208 ms | 277 ms | 74.9 ms | 42.7 ms | 12.7 ms |
| DDMT-FFT | 303 ms | 108 ms | 112 ms | 46.7 ms | 14.9 ms | 17.3 ms |

(CPU: 8 threads. FDMT-FFT and DDMT-FFT with fractional delays.)

- **Real time.** Every engine runs faster than real time on both backends,
  by 5–155× on 8 CPU threads and 120–1200× on the GPU, so the choice
  between them is about exactness and memory as much as speed.
- **`meerkat_lband`.** With 4096 channels, FDMT is the fastest engine on
  both backends: 251 ms on 8 CPU threads (80× real time, 15× faster than
  DDMT) and 27.8 ms on the L40 (722×, 6× faster than DDMT, 1.7× faster than
  DDMT-FFT).
- **Hard presets for FDMT.** Few channels and long dispersion sweeps (21.8k
  samples on `cryopaf`) give every tree level 10⁴–2×10⁴ rows, and every merge
  is a memory access, while DDMT and SDMT reuse each input window from cache
  or shared memory. On `cryopaf` the three take about the same time on the
  CPU (FDMT 200 ms, DDMT 214 ms, SDMT 208 ms). With more channels per delay,
  FDMT pulls ahead: 3.7× faster than DDMT on `cryopaf_lband`, 15× on
  `meerkat_lband`. DDMT-FFT is the fastest CPU engine on `cryopaf` and
  matches FDMT on `cryopaf_lband`.
- **On the GPU,** FDMT is 2× faster than DDMT and SDMT on `cryopaf`, and
  3–6× faster than DDMT on the L-band presets. Only SDMT on `cryopaf_lband`
  comes close (12.7 ms against 12.4 ms). DDMT-FFT is the fastest GPU
  engine on `cryopaf`.

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
:::{tab-item} cryopaf
:sync: cryopaf

```{image} ../bench/results/plots/light/runtime_vs_ndms.png
:class: only-light
:alt: Time per 32K-sample block against the number of DM trials, for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
```{image} ../bench/results/plots/dark/runtime_vs_ndms.png
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
  where the Levin grid gets sparse (`cryopaf` from DM 1000) SDMT falls back to
  the DDMT kernel. On `cryopaf_lband` it is the fastest GPU engine at every
  DM range.
- **FDMT-FFT** runs the FDMT tree in the Fourier domain, with each merge's
  shift applied as a fractional phase ramp. Its cost is dominated by the
  per-channel and per-trial FFTs, so it grows slowly with the DM count.
- **DDMT-FFT** is DDMT with exact (unrounded) per-channel delays, applied as
  phase ramps, with a NUFFT over the DM axis. It wins where the grid has many
  trials per channel (`cryopaf` at DM 1000–2000).

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
:::{tab-item} cryopaf
:sync: cryopaf

```{image} ../bench/results/plots/light/runtime_vs_nsamps.png
:class: only-light
:alt: Time per block against block length for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
```{image} ../bench/results/plots/dark/runtime_vs_nsamps.png
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
:::{tab-item} cryopaf
:sync: cryopaf

```{image} ../bench/results/plots/light/throughput_nbits.png
:class: only-light
:alt: Real-time factor against input sample width for FDMT, DDMT, SDMT, FDMT-FFT and DDMT-FFT
```
```{image} ../bench/results/plots/dark/throughput_nbits.png
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
  and 1.1–1.4× on 8 threads (`cryopaf`: 117× against 84× real time at 1
  bit). On the GPU, 1–2-bit input gains on the L-band presets
  (`cryopaf_lband`: 1470× against 1215×) and loses on `cryopaf`.
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
:::{tab-item} cryopaf
:sync: cryopaf

```{image} ../bench/results/plots/light/ops_theory.png
:class: only-light
:alt: Operations per output sample against DM trials and against channels, for FDMT, FDMT-FFT and DDMT
```
```{image} ../bench/results/plots/dark/ops_theory.png
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

```{image} ../bench/results/plots/light/cfdmt_rtf.png
:class: only-light
:alt: CohFDMT real-time factor against t_p, input bits, DM range width and block length
```
```{image} ../bench/results/plots/dark/cfdmt_rtf.png
:class: only-dark
:alt: CohFDMT real-time factor against t_p, input bits, DM range width and block length
```

With DM ranges of a few tens of pc cm⁻³ at L-band a single coherent trial is
enough, and the time goes to the forward transform of the block, shared by
all trials. Each additional coarse trial for a wider DM range adds its own
inverse transforms and one fine FDMT.

## Other machines

Earlier runs, kept for comparison only. They used dmt 0.7.0 (2026-10-23),
before the balanced FDMT tree, the staggered subbands and the newer CPU
FDMT and FDMT-FFT code, so only the DDMT and SDMT columns (unchanged since)
compare the hardware. Reference point, float32, median time per block:

| machine | preset | FDMT | FDMT-FFT | DDMT | SDMT | DDMT-FFT |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Intel Xeon Gold 6348H, 8 threads (pinned) | `cryopaf` | 370 ms | 759 ms | 190 ms | 215 ms | 136 ms |
| Intel Xeon Gold 6348H, 8 threads (pinned) | `cryopaf_lband` | 225 ms | 329 ms | 392 ms | 317 ms | 170 ms |
| NVIDIA L40S (shared) | `cryopaf` | 23.0 ms | 25.3 ms | 33.5 ms | 33.3 ms | 14.8 ms |
| NVIDIA L40S (shared) | `cryopaf_lband` | 11.8 ms | 17.1 ms | 34.4 ms | 9.7 ms | 17.3 ms |

The Xeon's two 512-bit FMA units per core make brute-force DDMT
faster there than on the EPYC 9334 (5–12%: 190 ms against 214 ms on `cryopaf`,
392 ms against 413 ms on `cryopaf_lband`).
The L40S ran DDMT and SDMT 8–30% faster than the L40 does now (built with
CUDA 13.0 there, 12.6 here); the L40 keeps its full 2.49 GHz clock under
load, so the gap is not throttling.

## Reproducing

The suite, the runner and the plotting scripts live in `bench/`.
`bench/README.md` gives the per-machine steps (build, `run_suite.py`, commit
the JSON, `plot_suite.py`; for CohFDMT `run_suite.py --suite cfdmt` and
`plot_cfdmt.py`) and the tips for comparable numbers. The raw results are the
JSON files under `bench/results/`.

```{seealso}
- [Fourier-domain variants](pipeline_guide/fourier_variants.md)
- [Streaming and history](pipeline_guide/streaming_and_history.md)
```
