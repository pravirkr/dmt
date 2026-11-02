# Benchmarks

All numbers on this page come from one benchmark suite (`dmt_bench_suite`). Every machine runs it with the same fixed
configuration, and one script turns the results into these plots and the
tables under [Numbers](#numbers). See [Reproducing](#reproducing) to run it yourself.

The results come from two machines:

- an Intel Xeon Gold 6348H server (CPU, 8 threads);
- an NVIDIA L40S GPU (96 MiB L2).

`FDMT-FFT` and `DDMT-FFT` in the plots and the Numbers tables are the
Fourier-domain engines as intended, with **fractional delays**. That is
FDMT-FFT's default, and DDMT-FFT with the NUFFT over the DM axis. The suite
also measures FDMT-FFT with rounded (integer) delays, which is a verification
mode that reproduces FDMT through the Fourier domain, and the exact
brute-force DDMT-FFT. Those two are only listed in the
[Fourier-domain engines](#fourier-domain-engines) section.

The suite plans the CPU FFTs with `FFTW_MEASURE` and a saved wisdom file
(`run_suite.py --fftw-planner`), as a long-running pipeline would.

## Setup

Two presets (`bench/suite/suite_common.hpp`) share DM 0–2000, a
piecewise-uniform Levin DM grid (tolerance 1.1), valid mode (streaming) and
box smearing:

| preset | band | channels | sampling | DM trials | reference block |
| :--- | :--- | ---: | ---: | ---: | :--- |
| `cryopaf` (plots below) | 704–1216 MHz | 512 | 512 µs | 1545 | 32 768 samples (16.8 s) |
| `cryopaf_lband` | 1100–1714 MHz | 1152 | 455.6 µs | 1941 | 32 768 samples (14.9 s) |

| | |
| :--- | :--- |
| DM grid | every engine gets the same Levin grid; the FDMT engines round it to integer delays and keep every trial |
| Timing | steady-state streaming: construction and warm-up calls are excluded; median of 3 runs of one `execute()` per block |
| CPU | 1 and 8 OpenMP threads (FDMT-FFT, DDMT and SDMT at 8 only) |
| GPU | device-resident input and output, CUDA-event timing; `incl. PCIe` = host arrays in and out |

The `parkes_uwl` preset (6656 channels, 704–4032 MHz at 64 µs, 4749 trials)
is deferred: its streaming histories need more than the 10 GB a real-time
search should use (15–16 GB for FDMT on the L40S), and FDMT-FFT's shortest
overlap-save segment (510k samples) does not fit the card.

Colour always identifies the algorithm (FDMT blue, FDMT-FFT orange, DDMT
green, SDMT yellow, DDMT-FFT pink). The grey line marks real time: the
duration of the data in one block. Points below it (or above 1× in the
throughput plot) keep up with the telescope.

## Key results

At the reference points (float32, 32 768-sample blocks; real-time factor in
brackets):

| | Xeon, `cryopaf` | Xeon, `cryopaf_lband` | L40S, `cryopaf` | L40S, `cryopaf_lband` |
| :--- | ---: | ---: | ---: | ---: |
| FDMT | 370 ms (45×) | 225 ms (66×) | 23.0 ms (729×) | 11.8 ms (1263×) |
| FDMT-FFT (fractional delays) | 759 ms (22×) | 329 ms (45×) | 25.3 ms (666×) | 17.1 ms (873×) |
| DDMT (brute force) | 190 ms (87×) | 392 ms (38×) | 33.5 ms (508×) | 34.4 ms (436×) |
| SDMT (exact, shared sums) | 215 ms (76×) | 317 ms (47×) | 33.3 ms (509×) | 9.7 ms (1547×) |
| DDMT-FFT (NUFFT, fractional delays) | 136 ms (124×) | 170 ms (88×) | 14.8 ms (1130×) | 17.3 ms (861×) |
| `FDMTBoxcar` search, ~12 widths up to 100 ms | 592 ms (28×) | 464 ms (32×) | 24.0 ms (699×) | 12.6 ms (1185×) |

(Xeon: 8 threads.)

- Every engine runs faster than real time on both machines, by tens of times
  on 8 Xeon threads (FDMT 7× on one thread) and hundreds of times on the GPU,
  so the choice between them is about exactness and memory as much as speed.
- These presets are hard for a plain FDMT: few channels and long dispersion
  sweeps (21.8k samples on `cryopaf`) give every tree level ~10⁴–2×10⁴ rows,
  and every merge is a memory access, while DDMT and SDMT reuse each input
  window from cache or shared memory. On the Xeon, brute-force DDMT is 2×
  faster than FDMT on `cryopaf` (512 channels), while on `cryopaf_lband`
  (1152 channels, fewer delays per channel) FDMT is 1.7× faster than DDMT and
  1.4× faster than SDMT; DDMT-FFT is the fastest CPU engine on both.
- On the L40S, FDMT runs its tree in chunks whose state fits the L2 cache and
  computes its lowest and highest levels directly
  ([direct levels](pipeline_guide/performance.md#direct-levels-cuda-many-delays-per-channel)),
  2.6–2.7× faster than at the start of this release cycle. It is now 1.45×
  faster than DDMT and SDMT on `cryopaf` and 2.9× faster than DDMT on
  `cryopaf_lband`; DDMT-FFT (NUFFT) on `cryopaf` and SDMT on `cryopaf_lband`
  remain faster.
- On the GPU the fused candidate search costs about 5% beyond the
  transform; on the CPU it is still 1.6–2× the transform (see
  [Candidate search](#candidate-search)).

## Time per block vs number of DM trials

```{image} ../bench/results/plots/light/runtime_vs_ndms.png
:class: only-light
:alt: Time per 32K-sample block against the number of DM trials, for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
```{image} ../bench/results/plots/dark/runtime_vs_ndms.png
:class: only-dark
:alt: Time per 32K-sample block against the number of DM trials, for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```

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

```{image} ../bench/results/plots/light/runtime_vs_nsamps.png
:class: only-light
:alt: Time per block against block length for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```
```{image} ../bench/results/plots/dark/runtime_vs_nsamps.png
:class: only-dark
:alt: Time per block against block length for FDMT, FDMT-FFT, DDMT, SDMT and DDMT-FFT
```

- FDMT, DDMT and SDMT are linear in the block length, so their distance to
  the real-time line stays constant as blocks grow. In valid mode the
  streaming history makes the result independent of the block size, so
  choose the block for latency. FDMT's chunk is fixed by the plan, not the
  block, so long blocks cost no extra memory.
- FDMT-FFT and DDMT-FFT get cheaper per sample with longer blocks: they
  transform long blocks in overlap-save segments, and the overlap (the
  dispersion sweep) is amortised over more samples.

## Real-time throughput by input width

```{image} ../bench/results/plots/light/throughput_nbits.png
:class: only-light
:alt: Real-time factor against input sample width for FDMT, DDMT, SDMT, FDMT-FFT and DDMT-FFT
```
```{image} ../bench/results/plots/dark/throughput_nbits.png
:class: only-dark
:alt: Real-time factor against input sample width for FDMT, DDMT, SDMT, FDMT-FFT and DDMT-FFT
```

The real-time factor is seconds of data processed per second of compute.

- **Packed low-bit input** helps FDMT most: at 1–4 bits the tree runs in
  `uint8`/`uint16` lanes ([integer tree](pipeline_guide/performance.md)). On
  the L40S, 1-bit `cryopaf_lband` runs at 1433× real time against 1263× for
  float32 (packed input does not use the direct levels yet).
- At 16 bits the tree is mostly float again, so packed input is no faster
  than float32.
- **DDMT** sums packed input two samples per instruction in 16-bit lanes.
  **SDMT** shares sums only for float32 and 16-bit input on the GPU; at 1–8
  bits it runs the DDMT kernel.
- **The Fourier-domain engines** transform complex float spectra whatever the
  input width, so their speed hardly depends on it.
- **Host arrays (`incl. PCIe`)** show what a pipeline pays when data starts
  and ends in host memory: the float32 DM–time plane copied back dominates.
  Keeping the downstream search on the GPU ([candidate search](#candidate-search))
  avoids that copy. These points are bound by pageable host-memory copies and
  depend on the host's load.

## Operation count

```{image} ../bench/results/plots/light/ops_theory.png
:class: only-light
:alt: Operations per output sample against DM trials and against channels, for FDMT, FDMT-FFT and DDMT
```
```{image} ../bench/results/plots/dark/ops_theory.png
:class: only-dark
:alt: Operations per output sample against DM trials and against channels, for FDMT, FDMT-FFT and DDMT
```

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

(numbers)=
## Numbers

### `cryopaf`

```{include} ../bench/results/plots/summary.md
```

### `cryopaf_lband`

```{include} ../bench/results/cryopaf_lband/plots/summary.md
```

The `cryopaf_lband` plots are in `bench/results/cryopaf_lband/plots/`.

(candidate-search)=
## Candidate search

`FDMTBoxcar` and `FDMTFFTBoxcar` ([Fused FDMT + Boxcar Search](pipeline_guide/fdmt_boxcar.md))
search the transform for boxcar candidates without writing the DM–time plane.
`dmt_bench --benchmark_filter='^BM_fdmt_boxcar_cuda/'` measures them at each
preset's reference point on noise, with widths up to 100 ms
(`max_boxcar = floor(0.1 s / tsamp)`): wtsp 1.5 gives 12–13 widths, wtsp 1.0
every width (195 on `cryopaf`, 219 on `cryopaf_lband`). L40S, 32 768-sample
blocks, min of 3 (`bench/results/[cryopaf_lband/]l40s/boxcar_cuda.json`):

| | `cryopaf` | `cryopaf_lband` |
| :--- | ---: | ---: |
| `FDMT` (writes the plane) | 22.9 ms | 11.9 ms |
| `FDMT` + separate search of the plane, wtsp 1.5 | 23.4 ms | 12.3 ms |
| `FDMTBoxcar`, wtsp 1.5 | 24.0 ms | 12.6 ms |
| `FDMTBoxcar`, wtsp 1.0 | 28.6 ms | 16.7 ms |
| `FDMTBoxcar`, wtsp 1.5, subband search (`subband_levels=1`) | — | 14.9 ms |
| `FDMTBoxcar`, wtsp 1.5, uint8 input | 104 ms | 40.0 ms |
| `FDMTBoxcar`, wtsp 1.5, host arrays | 28.0 ms | 21.4 ms |
| `FDMT-FFT` (writes the plane) | 25.1 ms | 17.1 ms |
| `FDMTFFTBoxcar`, wtsp 1.5 | 25.0 ms | 16.7 ms |
| `FDMTFFTBoxcar`, wtsp 1.0 | 27.8 ms | 20.7 ms |

- Scoring a dozen widths costs about 5% on top of the transform; every width
  up to 100 ms (~200) about 25–40%. The search needs 0.43–0.78 GB of device
  memory (the direct levels' rings included) against 0.68–0.90 GB for the
  transform and its plane.
- A subband search keeps the tree above level 3 (it reads those levels), so
  it does not get the direct top.
- The packed uint8 search keeps an exact uint64 integer tree and does not use
  the direct levels yet, so it runs at about 3–4.5× the float32 time.
- `FDMTFFTBoxcar` scores each segment's rows as the inverse transform
  produces them, so it also avoids the plane, but its segments are long and
  it holds 2.4–3.9 GB.

Intel Xeon Gold 6348H, `dmt_bench --benchmark_filter='^BM_fdmt_boxcar_stream/'`
(the same width sets; `bench/results/[cryopaf_lband/]xeon-6348h/boxcar_cpu.json`,
min of 3; the transform alone from the suite):

| | `cryopaf`, 8 threads | `cryopaf_lband`, 8 threads | `cryopaf`, 1 thread | `cryopaf_lband`, 1 thread |
| :--- | ---: | ---: | ---: | ---: |
| `FDMT` (writes the plane) | 370 ms | 225 ms | 2.57 s | 1.82 s |
| `FDMTBoxcar`, wtsp 1.5 | 592 ms (28×) | 464 ms (32×) | 3.15 s (5.3×) | 2.69 s (5.5×) |
| `FDMTBoxcar`, wtsp 1.0 | 1.75 s (9.6×) | 2.42 s (6.2×) | 12.7 s (1.3×) | 17.1 s (0.9×) |
| `FDMTBoxcar`, wtsp 1.5, uint8 input | 1.79 s (9.4×) | 1.15 s (13×) | 9.34 s (1.8×) | 6.41 s (2.3×) |

- On the CPU the search is a separate pass over each chunk's root rows, and
  its cost grows with the number of widths: a dozen widths add 60–110% to the
  transform, every width up to 100 ms 4.7–11×. Every case stays faster than
  real time on 8 threads; with every width, one thread falls just short on
  `cryopaf_lband`.

## Fourier-domain engines

`FDMT-FFT` is the engine with fractional delays (the plotted one).
`FDMT-FFT int` is the integer-delay verification mode, which reproduces FDMT.
`DDMT-FFT` uses the NUFFT over the DM axis, and `DDMT-FFT brute` is the exact
per-channel phase rotation. The suite runs `DDMT-FFT brute` at the reference
point only.

Intel Xeon Gold 6348H (8 threads, FFTW MEASURE plans), `cryopaf`,
32 768-sample block vs DM range:

| DM max (trials) | FDMT | FDMT-FFT | DDMT | SDMT | DDMT-FFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 125 (456) | 29.4 ms | 30.4 ms | 43.5 ms | 26.8 ms | 32.0 ms |
| 250 (701) | 55.4 ms | 53.2 ms | 69.3 ms | 46.9 ms | 43.2 ms |
| 500 (978) | 102 ms | 111 ms | 104 ms | 80.5 ms | 61.2 ms |
| 1000 (1260) | 193 ms | 288 ms | 130 ms | 130 ms | 94.5 ms |
| 2000 (1545) | 370 ms | 759 ms | 190 ms | 215 ms | 136 ms |

`cryopaf_lband`:

| DM max (trials) | FDMT | FDMT-FFT | DDMT | SDMT | DDMT-FFT |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 125 (195) | 16.1 ms | 30.7 ms | 34.4 ms | 22.5 ms | 41.2 ms |
| 250 (390) | 33.8 ms | 48.9 ms | 67.1 ms | 45.3 ms | 60.4 ms |
| 500 (776) | 55.7 ms | 71.5 ms | 127 ms | 84.1 ms | 64.6 ms |
| 1000 (1327) | 111 ms | 148 ms | 234 ms | 168 ms | 125 ms |
| 2000 (1941) | 225 ms | 329 ms | 392 ms | 317 ms | 170 ms |

On the Xeon, FDMT-FFT is 2× FDMT at DM 2000 on `cryopaf` (FFT-bound) and
1.5× on `cryopaf_lband`. DDMT-FFT's NUFFT is the fastest engine from DM 250
on `cryopaf` and from DM 1000 on `cryopaf_lband`; below that FDMT leads on
`cryopaf_lband`. The exact brute-force DDMT-FFT takes 398 ms (`cryopaf`) and
856 ms (`cryopaf_lband`) at DM 2000.

NVIDIA L40S (device-resident data, cuFFT), `cryopaf`, 32 768-sample block vs
DM range:

| DM max (trials) | FDMT | FDMT-FFT | DDMT | SDMT | DDMT-FFT | DDMT-FFT brute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 125 (456) | 0.98 ms | 1.65 ms | 1.65 ms | 0.96 ms | 3.03 ms | — |
| 250 (701) | 1.89 ms | 4.31 ms | 2.93 ms | 1.62 ms | 4.58 ms | — |
| 500 (978) | 4.60 ms | 7.22 ms | 6.48 ms | 2.74 ms | 6.65 ms | — |
| 1000 (1260) | 10.7 ms | 12.1 ms | 13.8 ms | 13.8 ms | 8.85 ms | — |
| 2000 (1545) | 23.0 ms | 25.3 ms | 33.5 ms | 33.3 ms | 14.8 ms | 15.0 ms |

`cryopaf_lband`:

| DM max (trials) | FDMT | FDMT-FFT | DDMT | SDMT | DDMT-FFT | DDMT-FFT brute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 125 (195) | 1.06 ms | 1.25 ms | 2.65 ms | 0.88 ms | 4.02 ms | — |
| 250 (390) | 1.51 ms | 1.92 ms | 3.83 ms | 1.66 ms | 4.92 ms | — |
| 500 (776) | 2.78 ms | 6.67 ms | 5.61 ms | 3.11 ms | 7.56 ms | — |
| 1000 (1327) | 5.58 ms | 10.2 ms | 14.5 ms | 5.60 ms | 11.3 ms | — |
| 2000 (1941) | 11.8 ms | 17.1 ms | 34.4 ms | 9.67 ms | 17.3 ms | 27.4 ms |

- **FDMT-FFT** runs the tree as shared-memory programs over bin tiles, so a
  large share of its time is cuFFT. With fractional delays it is within 10%
  of FDMT on `cryopaf` and 1.4× behind on `cryopaf_lband`.
- **DDMT-FFT (NUFFT)** delays every channel exactly. On `cryopaf` at DM 2000
  it is the fastest engine (14.8 ms), as fast as the exact brute-force phase
  rotation there and 2.3× faster than rounded-delay DDMT.

## Coherent hybrid search (CohFDMT)

CohFDMT searches baseband voltages rather than a filterbank, so it has its own
fixed configuration and is **not** compared with the engines above:

- **Reference configuration:** one GUPPI node (64 × 2.93 MHz at
  1.31–1.50 GHz, int8 FTPRI), t_p = 10 µs (n_p = 30), DM 50–60 pc cm⁻³ with
  `dt_step` 16, and the automatic block length.
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

How to read it:

- **One coarse trial.** With DM ranges of a few tens of pc cm⁻³ at L-band, the
  exact multi-subband grid needs a single coherent trial. Time then goes to
  the forward transform of the block, which is run once and shared by all
  trials. Longer blocks raise the useful fraction (`stride / block`) until
  memory becomes the limit.
- **More coarse trials.** For wider DM ranges, each additional trial costs the
  per-channel inverse transforms over the valid window plus one fine FDMT.
- **Measured against FFTW.** On the CPU both regimes run at the speed of FFTW
  and the FDMT. The chirp, detection, normalisation and channel alignment are
  fused into the per-channel inverse-transform pass, and the unpack and the
  channel statistics into the forward pass.
  `dmt_bench --benchmark_filter=cfdmt_(execute|stage)` prints the stage
  timings and the FFTW roofline of the inverse transforms.
- **On the GPU.**
  - The front end runs in L2-sized chunks of subbands.
  - Each coarse trial's coherent stage is one fused kernel: gather times chirp,
    inverse FFT in shared memory, detection, alignment.
  - With several coarse trials, the fine FDMT is the largest per-trial cost
    (DM 50–250: 4.7 ms per trial, against 1.2 ms for the coherent stage).
  - The `host arrays` series include the PCIe upload of the block (about
    130 MB for 8-bit input), overlapped chunk by chunk with the front end.
    They are bound by that copy.
  - `dmt_bench --benchmark_filter=cfdmt_cuda` compares the fused and cuFFT
    coherent stages, and reports the fused kernel's speed against a
    device-to-device copy (`roofline`).
- **CPU series.** Measured on the Intel Xeon Gold 6348H (8 threads).

```{include} ../bench/results/plots/cfdmt_summary.md
```

## Reproducing

The suite, the runner and the plotting script live in `bench/`.
`bench/README.md` gives the per-machine steps (build, `run_suite.py`, commit
the JSON, `plot_suite.py`; for CohFDMT `run_suite.py --suite cfdmt` and
`plot_cfdmt.py`) and the tips for comparable numbers. The
tuning microbenchmarks behind [Performance Tuning](pipeline_guide/performance.md)
and the candidate-search benchmarks are a separate binary (`dmt_bench`).
