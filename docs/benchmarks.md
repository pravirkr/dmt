# Benchmarks

All numbers on this page come from one benchmark suite (`dmt_bench_suite`). Every machine runs it with the same fixed
configuration, and one script turns the results into these plots and the
table at the end. See [Reproducing](#reproducing) to run it yourself.

The results below come from three machines:

- an Apple M1 Pro laptop (8 performance threads), dmt 0.6.0;
- an Intel Xeon Gold 6348H server (8 threads), dmt 0.6.0;
- an NVIDIA L40S GPU, dmt 0.5.0.

0.6.0 rewrote the Fourier-domain engines (`FDMT-FFT`, the new `DDMT-FFT`);
the other engines are unchanged since 0.5.0. The L40S results have not been
re-run with 0.6.0 yet. Its old FDMT-FFT had integer delays only, so the L40S
panels show no FDMT-FFT.

`FDMT-FFT` and `DDMT-FFT` in the plots and the Numbers table are the
Fourier-domain engines as intended, with **fractional delays**. That is
FDMT-FFT's default, and DDMT-FFT with the NUFFT over the DM axis. The suite also measures FDMT-FFT with rounded (integer) delays,
which is a verification mode that reproduces FDMT through the Fourier domain,
and the exact brute-force DDMT-FFT. Those two are only listed in the
[Fourier-domain engines](#fourier-domain-engines) tables.

```{note}
The 0.6.0 Xeon run shared its host with other jobs (load average ~130 of
192 hardware threads). Engines that did not change since 0.5.0 measured
1.5–2x slower than in the quiet 0.5.0 run (load average 8–19): FDMT
82 vs 53 ms, DDMT 1.15 vs 0.60 s at the reference point. Read the Xeon
plots for the ratios between engines, which were measured together. The
Xeon figures quoted in the text for FDMT, DDMT and SDMT are from the quiet
0.5.0 run.
``` The suite now plans
FFTs with `FFTW_MEASURE` and a saved wisdom file (`run_suite.py
--fftw-planner`), as a long-running pipeline would.

## Key results

At the reference point (4096 channels, 16K-sample blocks of 1.34 s, 2049 DM
trials, float32 input):

| | Apple M1 Pro (8 threads, 0.6.0) | Xeon Gold 6348H (8 threads, 0.5.0 run) | NVIDIA L40S (0.5.0) |
| :--- | ---: | ---: | ---: |
| FDMT time per block | 25.5 ms | 52.7 ms | 4.8 ms |
| FDMT real-time factor | 53× | 25× | 280× |
| DDMT (brute force), slower by | 33× | 11× | 4.4× |
| DDMT real-time factor | 1.6× | 2.3× | 63× |
| SDMT (exact, shared sums), slower by | 8× | 3× | 1.8× |
| SDMT real-time factor | 6.5× | 8× | 159× |
| FDMT-FFT (fractional delays), slower by | 3.3× | 1.7× (0.6.0 run) | — |
| DDMT-FFT (exact fractional delays), slower by | 4.3× | 3.3× (0.6.0 run) | — |
| DDMT-FFT, faster than DDMT by | 7.7× | 4.2× (0.6.0 run) | — |
| FDMT real-time factor, 1-bit input | 111× | 71× | 610× |

## Setup

| | |
| :--- | :--- |
| Band | 704–1216 MHz, 4096 channels |
| Sampling | 81.92 µs, `mode="valid"` (streaming), box smearing on |
| DM trials | `dt_max + 1` delay trials (2049 at the reference point); DDMT gets FDMT's exact DM grid |
| Reference point | 16 384 samples per block (1.34 s of data) × 2049 DM trials |
| Timing | steady-state streaming: construction and one warm-up call are excluded; median of 3 runs of one `execute()` per block |
| CPU | 1 and 8 OpenMP threads (FDMT-FFT and DDMT at 8 only) |
| GPU | device-resident input and output, CUDA-event timing; `incl. PCIe` = host arrays in and out |

Colour always identifies the algorithm (FDMT blue, FDMT-FFT orange, DDMT
green, SDMT yellow, DDMT-FFT pink). The grey line marks real time: the duration of the data in one block.
Points below it (or above 1× in the throughput plot) keep up with the
telescope.

## Time per block vs number of DM trials

```{image} ../bench/results/plots/light/runtime_vs_ndms.png
:class: only-light
:alt: Time per 16K-sample block against the number of DM trials, for FDMT, FDMT-FFT and DDMT
```
```{image} ../bench/results/plots/dark/runtime_vs_ndms.png
:class: only-dark
:alt: Time per 16K-sample block against the number of DM trials, for FDMT, FDMT-FFT and DDMT
```

- **FDMT** grows slowly with the DM count: a 16× increase in DM trials (256
  to 4K) costs 6–9× more time on the CPUs (13× on the L40S, where the small
  grids fit the 96 MB L2). The tree's early levels depend on the channel
  count, not the DM count.
- **Brute-force DDMT** does `nchans × ndm` additions per output sample, so it
  scales linearly with the DM count.
  - On the Xeon it stays below the real-time line up to 4K DM trials
    (16× real time at 256 trials, 1.1× at 4K), 6–12× behind FDMT.
  - On the M1 Pro it stays below real time up to ~2K DM trials
    (11× real time at 256 trials, 1.6× at 2049 trials), crossing the line only
    at 4K trials (up from crossing at ~256 trials in 0.3.0).
  - On the L40S it stays well below real time at every DM count tested
    (32–340× real time), 4–7× behind FDMT: the DM-tiled kernels reuse each
    staged input window across up to 64 DM trials.
- **SDMT** computes the same sums as DDMT (integer output bit-identical,
  float equal to rounding) but shares partial sums between DM trials within
  16-channel subbands, so it needs ~7× fewer additions (0.14× the brute-force additions on this
  grid, delivering ~3.5× wall-clock speedup). On the Xeon it runs 3.4–3.7× faster than DDMT at every DM count
  (56× real time at 256 trials, 4× at 4K), 1.8–3.3× behind FDMT. On the M1 Pro
  it runs 3.5–4.5× faster than DDMT (51× real time at 256 trials, 6.2× at 2049 trials).
  On the L40S it runs 2.2–3.5× faster than DDMT (740× real time at
  256 trials, 82× at 4K), 1.6–2.3× behind FDMT. The GPU shares sums through
  a fixed two-level hierarchy (unique 4-channel sums, then unique 16-channel
  sums, per tile of 128 trials) evaluated in shared memory.
  The saving depends on the DM grid: dense grids save the most, while coarse or
  sparse grids fall back to direct sums per subband (on the GPU, to the DDMT
  kernel).
- **FDMT-FFT** runs the FDMT tree in the Fourier domain, with each merge's
  shift applied as a fractional phase ramp. Its cost is dominated by the
  forward and inverse FFTs (4096 + 2049 real transforms per block), so it
  grows slowly with the DM count.
  - On the M1 Pro it is 2.5–5.6× behind FDMT (45 ms at 256 trials, 119 ms
    at 4K), and 11–33× faster than real time.
  - On the Xeon it is 1.7–3.0× behind FDMT. FDMT is memory-bound there, so
    the gap is smaller.
- **DDMT-FFT** is DDMT with exact (unrounded) per-channel delays, applied as
  phase ramps. With the NUFFT over the DM axis it runs 2–10× faster than the
  rounded-delay DDMT on the M1 Pro (57 ms at 256 trials, 162 ms at 4K), and
  1.4–7× on the Xeon. The exact brute-force variant (`DDMT-FFT-brute`) scales
  like DDMT: about 4× slower on NEON, 2–2.6× with the AVX-512 kernel.


## Time per block vs block length

```{image} ../bench/results/plots/light/runtime_vs_nsamps.png
:class: only-light
:alt: Time per block against block length for FDMT, FDMT-FFT and DDMT
```
```{image} ../bench/results/plots/dark/runtime_vs_nsamps.png
:class: only-dark
:alt: Time per block against block length for FDMT, FDMT-FFT and DDMT
```

- FDMT and DDMT are linear in the block length, so their distance to the
  real-time line stays constant as blocks grow. In valid mode the streaming
  history makes the result independent of the block size, so choose the
  block for latency.
- FDMT-FFT and DDMT-FFT (0.6.0) are linear in the block length: they
  transform cache-sized bin tiles, and long blocks in overlap-save segments.
  FDMT-FFT takes 292 ms per 64K-sample block on the M1 Pro and 491 ms on the
  Xeon (1.3× FDMT there). The 0.5.0 engine, with integer delays only, took
  4.0 s and 5.9 s, because its FFT length and working set grew with the block.
- FDMT's time per sample is nearly independent of the block length (within
  ~25% from 4K to 64K samples on the CPUs). On the L40S 4K-sample blocks are
  a third cheaper per sample, because their tree levels fit the 96 MB L2.

## Real-time throughput by input width

```{image} ../bench/results/plots/light/throughput_nbits.png
:class: only-light
:alt: Real-time factor against input sample width for FDMT and DDMT
```
```{image} ../bench/results/plots/dark/throughput_nbits.png
:class: only-dark
:alt: Real-time factor against input sample width for FDMT and DDMT
```

The real-time factor is seconds of data processed per second of compute.

- **Packed low-bit input** is where FDMT gains most. At 1–4 bits the tree
  runs in `uint8`/`uint16` lanes ([integer tree](pipeline_guide/performance.md)).
  - On the CPUs, 1-bit input runs 2.1–2.8× faster than float32: 111× real
    time on the M1 Pro and 71× on the Xeon (8 threads).
  - On the L40S it runs 2.2× faster (610× real time). There the fused kernel
    already keeps the early levels on-chip, which leaves less for the narrow
    tree to save.
- At 16 bits the tree is mostly float again, so packed input is no faster
  than float32.
- **One CPU thread** keeps FDMT above real time at every input width on both
  CPUs: 5–13× on the Xeon and 16–41× on the M1 Pro.
- **Brute-force DDMT** runs above real time on both tested CPUs and GPU:
  - On the Xeon (8 threads) it runs 2.3× real time on float32, 2.6× on 16-bit
    and 4.3–4.7× on 1- to 8-bit input (summed in 16-bit lanes).
  - On the M1 Pro (8 threads) it runs 1.6× real time on float32 (up from
    0.2× in 0.3.0) and 2.5–3.5× on packed integer input.
  - On the L40S it runs 63× real time on float32 input and 92–95× on 1- to
    8-bit input, which is summed two samples per instruction in 16-bit lanes.
- **SDMT** provides a further speedup over direct DDMT (3.5–4.5× on the
  CPUs, 1.7–2.5× on the GPU):
  - On the Xeon it runs 8× real time on float32, 10× on 16-bit and 17–21× on
    1- to 8-bit input.
  - On the M1 Pro it runs 6.7× real time on float32 and 10–13× on 1- to 8-bit input.
  - On the L40S it runs 159× real time on float32 and 165–178× on packed
    input. Its partial sums are 32-bit for every input width, so, unlike
    DDMT, packed input gains little over float.
- **Host arrays (`incl. PCIe`)** show what a pipeline pays when data starts
  and ends in host memory.
  - The packed input crosses the bus cheaply, so the float32 DM-time output
    copied back (134 MB per block here) dominates.
  - 1-bit input then drops from 610× to 84× real time.
  - Keeping the downstream search on the GPU avoids that copy.
  - These points are bound by pageable host-memory copies. The 0.5.0 run
    shared its host with heavy CPU jobs (load average 150–220), and repeated
    runs of them varied by up to 4×; read them as indicative only.

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
  count depends on the DM grid (0.14× DDMT's at the reference point).
- **FDMT**: the plan's exact addition count (`FDMTPlan.total_operations`).
- **FDMT-FFT**: a flop model. It counts forward FFTs of every channel,
  inverse FFTs of every DM row, and one complex multiply-add per tree node
  and frequency bin.

FDMT's advantage grows with both the channel count and the DM count: at the
reference point brute force needs ~270× more operations.

How much of that shows up in wall-clock time depends on the platform:

- **M1 Pro: 33×** for brute-force DDMT (down from 336× in 0.3.0) and **8×** for SDMT.
- **Xeon: 11×.** Its kernels add 16–32 samples per instruction
  from cache-resident windows, 10–16 channels per accumulator pass, so brute
  force's compute-bound inner loop gains more from the CPU than FDMT's
  memory-bound merges.
- **L40S: 4.4×** for DDMT and **1.8×** for SDMT. Brute force suits a GPU
  well: each input window is staged once in shared memory and reused by a
  whole tile of DM trials, so the kernel runs at ~80% of the card's
  shared-memory load rate. FDMT's level-by-level kernels run at ~90% of the
  DRAM bandwidth.

FDMT does fewer operations, but each one is a memory access. So its speed
follows the memory bandwidth, not the peak arithmetic rate.

## Numbers

```{include} ../bench/results/plots/summary.md
```

## Fourier-domain engines

The 0.6.0 Fourier-domain engines on the Apple M1 Pro (8 threads, FFTW MEASURE
plans; from the suite above). `FDMT-FFT` is the engine with fractional
delays (the plotted one). `FDMT-FFT int` is the integer-delay verification
mode, which reproduces FDMT. `DDMT-FFT` uses the NUFFT over the (uniform) DM
axis, and `DDMT-FFT brute` is the exact per-channel phase rotation.

Reference block (16K samples) vs DM trials:

| dt_max (trials) | FDMT | FDMT-FFT int | FDMT-FFT | DDMT | SDMT | DDMT-FFT | DDMT-FFT brute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 (257) | 7.9 ms | 36 ms | 45 ms | 117 ms | 27 ms | 57 ms | 0.41 s |
| 512 (513) | 9.8 ms | 37 ms | 41 ms | 214 ms | 50 ms | 69 ms | 0.78 s |
| 1024 | 14.5 ms | 42 ms | 55 ms | 445 ms | 100 ms | 76 ms | 1.59 s |
| 2048 | 25.5 ms | 57 ms | 83 ms | 851 ms | 206 ms | 111 ms | 3.33 s |
| 4096 | 48 ms | 94 ms | 119 ms | 1.70 s | 392 ms | 162 ms | 7.50 s |

2049 trials vs block length:

| nsamps | FDMT | FDMT-FFT int | FDMT-FFT | DDMT | DDMT-FFT | DDMT-FFT brute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 7.2 ms | 20 ms | 26 ms | 199 ms | 34 ms | 1.13 s |
| 8192 | 13.0 ms | 33 ms | 50 ms | 424 ms | 54 ms | 1.90 s |
| 16384 | 25.7 ms | 68 ms | 94 ms | 1.03 s | 137 ms | 3.90 s |
| 32768 | 62 ms | 141 ms | 155 ms | 2.14 s | 253 ms | 7.51 s |
| 65536 | 118 ms | 242 ms | 292 ms | 3.62 s | 441 ms | 12.6 s |

(The 16K row of this sweep ran while the machine was under other load; the
DM sweep above is the reference.)

- **FDMT-FFT** takes 83 ms at the reference point on the M1 Pro, 5x faster
  than 0.5.0's integer-only engine (415 ms). The integer verification mode
  takes 57 ms, so fractional delays cost 1.1–1.5x. Both are FFT-bound: FFTW
  takes about half the time, and on this machine the FFT alone costs about
  as much as the whole time-domain FDMT. See
  [Fourier-domain variants](pipeline_guide/fourier_variants.md#performance)
  for the breakdown.
- **DDMT-FFT (NUFFT)** applies exact fractional delays 2–10x faster than the
  rounded-delay DDMT, at 1.3–1.9x the cost of FDMT-FFT. Unlike FDMT-FFT's
  tree, it delays every channel exactly.
- **DDMT-FFT on packed input** unpacks straight into its transform rows: 1-bit
  input runs at 11× real time, float32 at 12× (Numbers table).
Intel Xeon Gold 6348H (8 threads, FFTW MEASURE, measured under load; see the
note at the top), reference block vs DM trials:

| dt_max (trials) | FDMT | FDMT-FFT int | FDMT-FFT | DDMT | SDMT | DDMT-FFT | DDMT-FFT brute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 (257) | 23 ms | 62 ms | 70 ms | 171 ms | 44 ms | 123 ms | 0.34 s |
| 512 (513) | 28 ms | 68 ms | 74 ms | 312 ms | 84 ms | 166 ms | 0.70 s |
| 1024 | 45 ms | 82 ms | 96 ms | 569 ms | 164 ms | 159 ms | 1.43 s |
| 2048 | 82 ms | 111 ms | 143 ms | 1.15 s | 319 ms | 274 ms | 2.89 s |
| 4096 | 149 ms | 188 ms | 247 ms | 2.36 s | 631 ms | 333 ms | 6.17 s |

2049 trials vs block length:

| nsamps | FDMT | FDMT-FFT int | FDMT-FFT | DDMT | DDMT-FFT | DDMT-FFT brute |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 20 ms | 35 ms | 50 ms | 251 ms | 77 ms | 1.18 s |
| 8192 | 41 ms | 58 ms | 88 ms | 655 ms | 115 ms | 1.39 s |
| 16384 | 83 ms | 114 ms | 159 ms | 1.29 s | 281 ms | 2.63 s |
| 32768 | 187 ms | 225 ms | 265 ms | 2.27 s | 536 ms | 6.22 s |
| 65536 | 382 ms | 423 ms | 491 ms | 4.36 s | 835 ms | 9.53 s |

- On the Xeon, **FDMT-FFT** is 1.7× FDMT at the reference point and 1.3× at
  64K samples; the integer verification mode is 1.1–1.4× from 2K trials or
  8K samples on (0.5.0: 701 ms, 13× FDMT). **DDMT-FFT (NUFFT)** is 4.2× faster
  than DDMT at the reference point, and its per-bin cost barely depends on the
  input width (5× real time for 1-bit to float32).
- The two machines differ where the FFT meets the tree: on the M1 Pro, FFTW
  alone costs about as much as FDMT; on the Xeon, FDMT is memory-bound, so
  the cache-resident Fourier tree comes close.

## Reproducing

The suite, the runner and the plotting script live in `bench/`.
`bench/README.md` gives the per-machine steps (build, `run_suite.py`, commit
the JSON, `plot_suite.py`) and the tips for comparable numbers. The
tuning microbenchmarks behind [Performance Tuning](pipeline_guide/performance.md)
are a separate binary (`dmt_bench`).
