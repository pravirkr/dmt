# Benchmarks

Every engine on one shared node: an AMD EPYC 9654 (8 threads) and an NVIDIA
L40S. The rankings hold for this machine. FDMT speed follows memory bandwidth,
brute-force DDMT is compute-bound, so other hardware can order the engines
differently.

## Setup

| preset | band | channels | sampling | DM trials (0–2000) | 32K-sample block |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `meerkat_lband` | 856–1712 MHz | 4096 | 612.48 µs | 4057 | 20.1 s |
| `cryopaf_lband` | 1100–1714 MHz | 1152 | 455.6 µs | 1941 | 14.9 s |
| `cryopaf_low` | 704–1216 MHz | 512 | 512 µs | 1545 | 16.8 s |

- **Timing.** Steady-state streaming (valid mode, box smearing): the median
  of 3 `execute()` calls on a 32 768-sample block, after warm-up.
- **GPU.** Device-resident input and output, timed with CUDA events (no host
  copies).
- **Grid.** Every engine gets the same piecewise-uniform Levin grid
  (tolerance 1.1); the FDMT engines round it to integer delays and keep every
  trial.
- **Reading the figures.** Colour is the engine. The grey line is real time
  (the duration of one block); below it, or above 1×, keeps up.

:::{dropdown} Measurement details
:icon: info

- **CPU placement.** One thread per core complex (`OMP_PROC_BIND=close
  OMP_PLACES='{8}:8:8'`), so each thread has its own 32 MB L3 slice. The node
  is shared; those core complexes were idle during every run.
- **FFTs.** CPU FFTs are planned with `FFTW_MEASURE` and saved wisdom, as a
  long-running pipeline would. `FDMT-FFT` and `DDMT-FFT` use fractional
  delays (`DDMT-FFT` with the NUFFT over the DM axis).
- **Memory.** Every point records its peak memory in the same run: peak RSS on
  the CPU, device memory in use on the GPU, from construction through the
  timed blocks.
:::

## Key results

```{image} ../bench/results/plots/light/reference_rtf.png
:class: only-light
:alt: Real-time factor of FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT on three presets, on 8 CPU threads and on the GPU
```
```{image} ../bench/results/plots/dark/reference_rtf.png
:class: only-dark
:alt: Real-time factor of FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT on three presets, on 8 CPU threads and on the GPU
```

- **Every engine keeps up** on both backends at the reference point.
- **FDMT's lead grows with the channel count.** It is the fastest engine on
  `meerkat_lband` on both backends and on `cryopaf_lband` on the CPU; on the
  GPU SDMT matches it there.
- **With few channels the engines converge.** On `cryopaf_low` long
  dispersion sweeps make every tree level large and memory-bound, while DDMT
  and SDMT reuse each input window from cache. All five are within about 2× of
  each other, and DDMT-FFT is the fastest on both backends.

:::::{dropdown} Exact numbers (time per block, memory, input widths)
:icon: table

::::{tab-set}
:::{tab-item} meerkat_lband
:sync: meerkat

```{include} ../bench/results/meerkat_lband/plots/summary.md
:start-after: "<!-- results -->"
:end-before: "**FDMTBoxcar**"
```
:::
:::{tab-item} cryopaf_low
:sync: cryopaf_low

```{include} ../bench/results/cryopaf_low/plots/summary.md
:start-after: "<!-- results -->"
:end-before: "**FDMTBoxcar**"
```
:::
:::{tab-item} cryopaf_lband
:sync: lband

```{include} ../bench/results/cryopaf_lband/plots/summary.md
:start-after: "<!-- results -->"
:end-before: "**FDMTBoxcar**"
```
:::
::::
:::::

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

- **FDMT** follows the tree's row count, which grows with the maximum delay
  as well as the trial count.
- **DDMT** is linear in the trial count. **SDMT** shares partial sums within
  subbands: most on dense grids, none where the Levin grid gets sparse
  (`cryopaf_low` above DM 1000), where it runs the DDMT kernel.
- **The Fourier-domain engines** are dominated by their FFTs and grow slowly
  with the trial count. DDMT-FFT wins where there are many trials per channel.

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

- **FDMT, DDMT and SDMT** are linear in the block length, so their distance
  to real time stays constant. Choose the block for latency.
- **The Fourier-domain engines** get cheaper per sample with longer blocks,
  which amortise the overlap (the dispersion sweep).

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

- **CPU.** Packed 1–4-bit input runs the FDMT tree in narrow integer lanes,
  1.1–1.4× faster than float32.
- **GPU.** Only `cryopaf_lband` gains from 1–2-bit input; on the other
  presets float32 uses the faster direct bottom and top kernels.
- **16-bit** input is no faster than float32 (slower on the GPU).
- **DDMT** sums packed input two samples per instruction; **SDMT** shares sums
  only for float32 and 16-bit input on the GPU. **The Fourier-domain engines**
  transform complex floats whatever the input, so their curves are flat.

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

Additions per output sample, summed over all DM trials: `nchans × ndm` for
DDMT, the plan's exact count for FDMT (`FDMTPlan.total_operations`), and a
flop model for FDMT-FFT (channel and row FFTs plus one complex multiply-add
per tree node and bin). SDMT's count depends on the grid and is not drawn.

FDMT's advantage grows with both channels and trials. Less of it shows in
wall-clock time: each FDMT operation is a memory access, while brute force
is compute-bound and reuses its input from cache.

## Parkes UWL search

6656 channels over 704–4032 MHz at 64 µs, DM 0–2000 (4749 trials), searched
by `FDMTBoxcar` at subband depth 4 (about 218 000 searched rows) with
boxcars to 100 ms, within the default 10 GiB memory budget. The sweep repeats
the search downsampled to 256, 512 and 1024 µs, on 2.1 s blocks. The other
engines are not run here (FDMT-FFT needs ~50 GB; CPU DDMT takes hours).

```{image} ../bench/results/parkes_uwl/plots/light/uwl_realtime.png
:class: only-light
:alt: Real-time factor of the Parkes UWL depth-4 search against time resolution, GPU and 8 CPU threads, uint8 and float32
```
```{image} ../bench/results/parkes_uwl/plots/dark/uwl_realtime.png
:class: only-dark
:alt: Real-time factor of the Parkes UWL depth-4 search against time resolution, GPU and 8 CPU threads, uint8 and float32
```

- **The GPU keeps up at the native 64 µs on 8-bit input.** Float32 fits the
  budget only with short chunks and runs just below real time.
- **Each halving of the resolution is 4–5× cheaper.** From 256 µs, 8 CPU
  threads keep up. At high DM the bottom channels are smeared far beyond
  256 µs anyway, so a real-time system can search at 256–512 µs and keep the
  64 µs data around triggers.
- **On the CPU, 64 µs** is for offline processing.

:::{dropdown} Exact numbers
:icon: table

```{include} ../bench/results/parkes_uwl/plots/summary.md
```
:::

## Coherent hybrid search (CohFDMT)

CohFDMT searches baseband voltages, so it has its own configuration: one GUPPI
node (64 × 2.93 MHz at 1.31–1.50 GHz, int8), t_p = 10 µs, DM 50–60 pc cm⁻³.
Each sweep passes through that point and varies one of the target
resolution, input width, DM range or block length. The real-time factor
counts one stride per block (excluding the overlap).

```{image} ../bench/results/cfdmt/plots/light/cfdmt_rtf.png
:class: only-light
:alt: CohFDMT real-time factor against t_p, input bits, DM range width and block length
```
```{image} ../bench/results/cfdmt/plots/dark/cfdmt_rtf.png
:class: only-dark
:alt: CohFDMT real-time factor against t_p, input bits, DM range width and block length
```

At L-band a DM range of a few tens of pc cm⁻³ needs a single coherent trial,
and the time goes to the forward transform shared by all trials. Each extra
coarse trial for a wider range adds its own inverse transforms and one fine
FDMT.

## Reproducing

The suite, runner and plot scripts are in `bench/`; `bench/README.md` gives
the steps. Raw results are the JSON files under
`bench/results/<config>/l40s-epyc9654/`.
