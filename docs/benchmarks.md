# Benchmarks

All numbers on this page come from one benchmark suite (`dmt_bench_suite`). Every machine runs it with the same fixed
configuration, and one script turns the results into these plots and the
table at the end. See [Reproducing](#reproducing) to run it yourself.

The results below are for three machines (all measured with dmt 0.4.0):

- an Apple M1 Pro laptop (8 performance threads);
- an Intel Xeon Gold 6348H server (8 threads);
- an NVIDIA L40S GPU.

## Key results

At the reference point (4096 channels, 16K-sample blocks of 1.34 s, 2049 DM
trials, float32 input):

| | Apple M1 Pro (8 threads) | Xeon Gold 6348H (8 threads) | NVIDIA L40S |
| :--- | ---: | ---: | ---: |
| FDMT time per block | 25.3 ms | 52.7 ms | 5.5 ms |
| FDMT real-time factor | 53× | 25× | 242× |
| DDMT (brute force), slower by | 37× | 11× | 4× |
| DDMT real-time factor | 1.5× | 2.3× | 63× |
| SDMT (exact, shared sums), slower by | 8× | 3× | — |
| SDMT real-time factor | 6.8× | 8× | — |
| FDMT-FFT, slower by | 16× | 13× | 4× |
| FDMT real-time factor, 1-bit input | 122× | 71× | 387× |

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
green, SDMT yellow). The grey line marks real time: the duration of the data in one block.
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
  to 4K) costs 6–9× more time. The tree's early levels depend on the channel
  count, not the DM count.
- **Brute-force DDMT** does `nchans × ndm` additions per output sample, so it
  scales linearly with the DM count.
  - On the Xeon (0.4.0) it stays below the real-time line up to 4K DM trials
    (16× real time at 256 trials, 1.1× at 4K), 6–12× behind FDMT.
  - On the M1 Pro (0.4.0) it stays below real time up to ~2K DM trials
    (11× real time at 256 trials, 1.6× at 2049 trials), crossing the line only
    at 4K trials (up from crossing at ~256 trials in 0.3.0).
  - On the L40S it stays well below real time at every DM count tested
    (32–340× real time), 3–5× behind FDMT: the DM-tiled kernels reuse each
    staged input window across up to 64 DM trials.
- **SDMT (CPU only)** computes the same sums as DDMT (integer output bit-identical,
  float equal to rounding) but shares partial sums between DM trials within
  16-channel subbands, so it needs ~7× fewer additions (0.14× the brute-force additions on this
  grid, delivering ~3.5× wall-clock speedup). On the Xeon it runs 3.4–3.7× faster than DDMT at every DM count
  (56× real time at 256 trials, 4× at 4K), 1.8–3.3× behind FDMT. On the M1 Pro
  it runs 3.5–4.5× faster than DDMT (51× real time at 256 trials, 6.2× at 2049 trials).
  The saving depends on the DM grid: dense grids save the most, while coarse or
  sparse grids fall back to direct sums per subband.
- **FDMT-FFT** applies the tree's shifts as FFT phase ramps. Its cost is
  dominated by the forward and inverse FFTs, so it is nearly flat in the DM
  count and depends on how well the FFT length factorises.
  - Its gap to FDMT therefore shrinks as DM trials are added: from 60–90×
    slower at 256 DM trials to 7–9× at 4K on the CPUs, and from 21× to 3×
    on the GPU.
  - On the Xeon it is at the real-time limit for small DM counts.
  - It is exact, but it is not a speed alternative to the direct FDMT.

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
- FDMT-FFT grows faster than linearly: its FFT length is the block plus the
  maximum delay, and its working set leaves the caches. On the Xeon it falls
  behind real time at 64K samples.
- FDMT's time per sample is nearly independent of the block length on every
  platform (within ~25% from 4K to 64K samples).

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
  - On the CPUs, 1-bit input runs 2.8× faster than float32: 123× real time
    on the M1 Pro and 70× on the Xeon (8 threads).
  - On the L40S it runs 1.6× faster (387× real time). There the fused kernel
    already keeps the early levels on-chip, which leaves less for the narrow
    tree to save.
- At 16 bits the tree is mostly float again, so packed input is no faster
  than float32.
- **One CPU thread** keeps FDMT above real time at every input width on both
  CPUs: 5–13× on the Xeon and 16–41× on the M1 Pro.
- **Brute-force DDMT** runs above real time on both tested CPUs and GPU:
  - On the Xeon (0.4.0, 8 threads) it runs 2.3× real time on float32, 2.6× on 16-bit
    and 4.3–4.7× on 1- to 8-bit input (summed in 16-bit lanes).
  - On the M1 Pro (0.4.0, 8 threads) it runs 1.6× real time on float32 (up from
    0.2× in 0.3.0) and 2.5–3.5× on packed integer input.
  - On the L40S it runs 63× real time on float32 input and 92–95× on 1- to
    8-bit input, which is summed two samples per instruction in 16-bit lanes.
- **SDMT (CPU only)** provides a further 3.5–4.5× speedup over direct DDMT:
  - On the Xeon it runs 8× real time on float32, 10× on 16-bit and 17–21× on
    1- to 8-bit input.
  - On the M1 Pro it runs 6.2× real time on float32 and 12–16× on 1- to 8-bit input.
- **Host arrays (`incl. PCIe`)** show what a pipeline pays when data starts
  and ends in host memory.
  - The packed input crosses the bus cheaply, so the float32 DM-time output
    copied back (134 MB per block here) dominates.
  - 1-bit input then drops from 387× to 99× real time.
  - Keeping the downstream search on the GPU avoids that copy.

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

- **M1 Pro (0.4.0): 37×** for brute-force DDMT (down from 336× in 0.3.0) and **8×** for SDMT.
- **Xeon (0.4.0): 11×.** Its kernels add 16–32 samples per instruction
  from cache-resident windows, 10–16 channels per accumulator pass, so brute
  force's compute-bound inner loop gains more from the CPU than FDMT's
  memory-bound merges.
- **L40S: 4×.** Brute force suits a GPU well: each input window is staged
  once in shared memory and reused by a whole tile of DM trials, so the
  kernel runs at ~80% of the card's shared-memory load rate.

FDMT does fewer operations, but each one is a memory access. So its speed
follows the memory bandwidth, not the peak arithmetic rate.

## Numbers

```{include} ../bench/results/plots/summary.md
```

## Reproducing

The suite, the runner and the plotting script live in `bench/`.
`bench/README.md` gives the per-machine steps (build, `run_suite.py`, commit
the JSON, `plot_suite.py`) and the tips for comparable numbers. The
tuning microbenchmarks behind [Performance Tuning](pipeline_guide/performance.md)
are a separate binary (`dmt_bench`).
