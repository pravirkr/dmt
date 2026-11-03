# Benchmarks

Every number here comes from one benchmark suite (`dmt_bench_suite`), run with
the same fixed configuration on two machines:

- **CPU:** an Intel Xeon Gold 6348H server (8 threads);
- **GPU:** an NVIDIA L40S.

GPU times are for **device-resident** input and output, timed with CUDA
events: they do not include copies between host and device. CPU FFTs are
planned with `FFTW_MEASURE` and a saved wisdom file, as a long-running
pipeline would. `FDMT-FFT` and `DDMT-FFT` are the Fourier-domain engines with
fractional delays (`DDMT-FFT` with the NUFFT over the DM axis).

## Setup

Two presets share DM 0–2000 on a piecewise-uniform Levin DM grid (tolerance
1.1), valid mode (streaming) and box smearing. Each figure has a tab per
preset.

`cryopaf`
: 704–1216 MHz, 512 channels at 512 µs, 1545 DM trials.

`cryopaf_lband`
: 1100–1714 MHz, 1152 channels at 455.6 µs, 1941 DM trials.

Timing is steady-state streaming: construction and warm-up calls are
excluded, and each point is the median of 3 runs of one `execute()` on a
32 768-sample block (16.8 s and 14.9 s of data). Every engine gets the same
DM grid; the FDMT engines round it to integer delays and keep every trial.

Colour always identifies the algorithm (FDMT blue, FDMT-FFT orange, DDMT
green, SDMT yellow, DDMT-FFT pink). The grey line marks real time: the
duration of the data in one block. Points below it (or above 1× in the
throughput plot) keep up with the telescope.

## Key results

- **Real time.** Every engine runs faster than real time on both machines, by
  tens of times on 8 Xeon threads and hundreds of times on the GPU, so the
  choice between them is about exactness and memory as much as speed.
- **Hard presets for FDMT.** Few channels and long dispersion sweeps (21.8k
  samples on `cryopaf`) give every tree level 10⁴–2×10⁴ rows, and every merge
  is a memory access, while DDMT and SDMT reuse each input window from cache
  or shared memory. On the Xeon, brute-force DDMT is 2× faster than FDMT on
  `cryopaf`, while on `cryopaf_lband` (more channels, fewer delays per
  channel) FDMT is 1.7× faster than DDMT. DDMT-FFT is the fastest CPU engine
  on both.
- **On the L40S,** FDMT is 1.45× faster than DDMT and SDMT on `cryopaf` and
  2.9× faster than DDMT on `cryopaf_lband`; DDMT-FFT on `cryopaf` and SDMT on
  `cryopaf_lband` are faster still.

## Time per block vs number of DM trials

::::{tab-set}
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

- **Packed low-bit input** helps FDMT most on the CPU: at 1–4 bits the tree
  runs in narrow integer lanes.
- At 16 bits the tree is mostly float again, so packed input is no faster
  than float32.
- **DDMT** sums packed input two samples per instruction in 16-bit lanes.
  **SDMT** shares sums only for float32 and 16-bit input on the GPU; at 1–8
  bits it runs the DDMT kernel.
- **The Fourier-domain engines** transform complex float spectra whatever the
  input width, so their speed hardly depends on it.

## Operation count

::::{tab-set}
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
