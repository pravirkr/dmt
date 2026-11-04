# Performance tuning

The tuning measurements below come from the developer microbenchmarks
(`dmt_bench`), each at the fixed geometry stated with it. The cross-algorithm
numbers for the benchmark presets (CryoPAF, CryoPAF L-band and Parkes UWL)
are on the [Benchmarks](../benchmarks.md) page; see also
[the benchmark configurations](https://github.com/pravirkr/dmt/blob/main/bench/README.md).

The FDMT engine (`FDMT`, on the CPU and CUDA backends) needs no tuning by
default. Its two *performance parameters*, the constructor arguments
`fuse_levels` and `int_tree`, default to the fastest settings measured, and
neither changes the result by a single bit (the automatic GPU direct levels,
described below, are the one exception). The *memory parameter*
`chunk_nsamps`, the last constructor argument, bounds the tree state of large plans, also without
changing the result. You will rarely need to set any of them. This page
explains what they do, what you control yourself (input format, threads,
block size) and how memory is allocated.

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
fdmt = FDMT(704.0, 1216.0, 4096, 16384, 8.192e-5, 2048, nthreads=1)
fdmt.fuse_levels       # depth chosen for this plan (fuse_levels=None default)
fdmt = FDMT(704.0, 1216.0, 4096, 16384, 8.192e-5, 2048,
               fuse_levels=4, int_tree=True)  # explicit depth (advanced)
fdmt.chunk_nsamps         # chunk length chosen for this plan (chunk_nsamps=None)
print(fdmt.memory_usage)  # bytes allocated at construction
```
:::
:::{tab-item} C++
:sync: cpp

```cpp
// ..., exec, nbeams, fuse_levels = kFDMTAutoFuse, int_tree = true,
//      chunk_nsamps = 0 (automatic)
dmt::algorithms::FDMT fdmt(704.0F, 1216.0F, 4096, 16384, 8.192e-5F, 2048,
                           0, 1, true, "valid", dmt::Exec::cpu(1), 1,
                           dmt::algorithms::kFDMTAutoFuse, true, 0);
```
:::
::::

## Where the time goes

Every FDMT tree level is one pass of the same loop over rows of the state:

$$\text{out}[t] = \text{tail}[t] + \text{head}[t - d]$$

That is two loads and one store per output sample, for one addition. There is
no arithmetic to optimize: the transform is memory-bandwidth bound, and
the merge loop already runs at the bandwidth limit. An unfused merge level
moves its three streams at over 90% of the bandwidth that a bare
`c[i] = a[i] + b[i]` achieves on the same machine, and on the L40S the
level-by-level kernels run at about 90% of the DRAM bandwidth.

Two consequences follow:

- The early levels dominate (levels 0–3 are ~45% of the time), because
  they have the most rows.
- The only way to go faster is to move fewer bytes. The two defaults below
  do that: one keeps the early levels on-chip (level fusion); the
  other stores them in fewer bytes (the integer tree).

## Level fusion (`fuse_levels`)

Level-$F$ sub-band $g$ depends only on the $2^F$ input channels below it. With
fusion, `execute()` processes one such channel group at a time. It builds the
group's level-0 rows and merges them up to level $F$ on-chip, then writes only
level $F$ to memory. The intermediate levels never make the round trip to
DRAM.

- **CPU:** each thread works on one group in a cache-resident scratch buffer
  holding full rows. The automatic depth (the default) is the deepest $F$
  whose two scratch buffers fit in $\max(36\,\text{MiB} / n_\text{threads},
  8\,\text{MiB})$ per thread (modes `"full"` and `"roll"`; valid mode uses
  time tiles, below). The scratch size follows from the plan
  (rows per channel group × `nsamps`), so the rule uses only the plan and
  the thread count, with no hardware detection. On 1 thread with 4096
  channels and `dt_max=2048`, this gives depths 9 / 7 / 5 at 4K / 16K / 64K
  samples; at 8 threads, 6 / 4 / 2.
- **CUDA:** one thread block handles a (channel group, time tile) pair in
  shared memory. Each fused level is computed over the tile plus a small
  left halo, the sum of the group's merge delays at the fused levels, which
  neighbouring tiles recompute. The automatic depth is the deepest $F$ whose
  tile of at least 128 samples fits a quarter of the SM's shared memory
  (four resident blocks; 24 KiB on an L40S, at most 48 KiB), with a level-0
  halo of at most half a tile.

### Direct levels (CUDA, many delays per channel)

When a band has many delays per channel (the `cryopaf` presets: 512 channels
with a 21.8k-sample sweep), every level holds ~10⁴–2×10⁴ rows, the tile
fusion cannot get past depth 0–1, and each merge streams two whole levels
through L2 and DRAM. There the automatic setting (valid mode, float input)
computes the ends of the tree directly instead:

- **Levels 0–3 from running sums.** A level-3 row is the sum of its 8 leaf
  boxes (channel, box length, accumulated head delay). Per chunk, one kernel
  writes every channel's running sum over the chunk and the samples before it,
  and the next stages a subband's sums in shared memory and adds two reads per
  leaf. Levels 0–2 are never materialised.
- **Trials from level 4.** Each trial is the sum of its level-4 rows at their
  accumulated delays. Every level-4 row keeps a ring of its last samples (as
  many as the largest delay any trial applies to it), read for the samples
  before the chunk. Levels 5 to the root are skipped. This needs the root to
  be the only level read, so a subband search (plan `subband_depth`) keeps
  the tree above level 3, and fusion stops below the lowest level holding
  staggered subbands.

Integer-valued input gives the tree's values exactly. Float input agrees with
the tree within rounding (a few 10⁻⁷ relative), so these engines are no longer
bit-identical with `fuse_levels=0` or across `chunk_nsamps`; explicit fusion
depths still are. Fractional-delay FDMTFFT adds a separate guard and
segmentation error, distinct from this tree contract. Candidate agreement is
threshold-aware: a small numerical difference can split or join an excursion,
and a saturated parallel buffer does not promise the same survivor order.
The streaming history keeps the CPU engine's format: the
skipped levels' tree histories are rebuilt from the running-sum window and
the rings when `save_history()` (or a stepper block) needs them, and after
`load_history()` the tree runs until the window and the rings have refilled.

L40S, 32 768-sample blocks, transform only:

| preset | tree (L2 chunks) | + direct levels 0–3 | + trials from level 4 |
| :--- | ---: | ---: | ---: |
| `cryopaf` | 39.5 ms | 27.1 ms | 22.9 ms |
| `cryopaf_lband` | 16.1 ms | 14.6 ms | 11.9 ms |

Fusion applies to `execute()` only. The stepper (`reset()` / `advance()`)
always runs level by level, so every level stays inspectable. An explicit
depth is clamped to the plan's merge levels (and, on CUDA, reduced until it
fits shared memory). Read the depth actually used with `fdmt.fuse_levels`
(Python) or `get_fuse_levels()` (C++).

### Fusion in practice

:::{dropdown} Why not depth-first?
:icon: info

The standard FDMT walks the tree breadth-first, level by level across the
whole band. That is regular and parallel, but every level writes hundreds of
megabytes to DRAM and reads them back. Computing each DM trial by recursing
down to its leaf channels looks cache-friendly, but the FDMT is a DAG with
overlapping sub-band delay paths: pure recursion either repeats work
exponentially or needs memoization tables that break vectorization and do not
map to GPUs. Near the top of the tree the sub-bands span thousands of channels
and exceed the cache anyway.

Level fusion splits the difference. The first $F$ levels form disjoint binary
sub-trees of $2^F$ channels (for example 8 or 16), each merged depth-first
on-chip, so levels $0 \dots F-1$ never touch DRAM. These early levels carry
~45–50% of the transform's memory traffic. Above level $F$ the engine returns
to breadth-first merges in memory. Both parts are flat loops, with no
recursion, dynamic allocation or redundant computation, so they vectorize
and map onto GPU thread blocks.
:::

Measured on the CPU (float input, valid mode, box smearing, 4096 channels,
`dt_max=2048`):

| machine | samples | threads | unfused | automatic | speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Xeon Gold 6348H | 4096 | 8 | 18.0 ms | 12.2 ms | 1.47× |
| Xeon Gold 6348H | 16 384 | 1 | 394.7 ms | 282.1 ms | 1.40× |
| Xeon Gold 6348H | 16 384 | 8 | 77.0 ms | 53.4 ms | 1.44× |
| Xeon Gold 6348H | 65 536 | 1 | 1584 ms | 1185 ms | 1.34× |
| Xeon Gold 6348H | 65 536 | 8 | 307.3 ms | 244.4 ms | 1.26× |

:::{dropdown} How the automatic rule was fitted
:icon: info

The rule was fitted to fixed-depth sweeps (1, 2, 4 and 8 threads; 4K, 16K
and 64K samples):

- **1 thread:** it picks the best measured depth at 16K and 64K.
- **2–8 threads:** it is within 3% of the best depth, except 4096 samples
  (6–8.5%).
:::

The defaults are sized for x86 server CPUs (Xeon/EPYC): the whole-block
budget of $\max(36\,\text{MiB} / n_\text{threads}, 8\,\text{MiB})$ assumes
an L3 of ~32–36 MiB per socket and 1–8 threads per pipeline instance. Valid
mode's automatic budget is 1.75 MiB per group, described below. Pass an
explicit `fuse_levels` if a sweep on your hardware (`BM_fdmt_fused`, see
[Reproducing](#reproducing-the-numbers)) finds a better depth.

The table and the budget above are for the whole-block fused path, which
modes `"full"` and `"roll"` still use.

**Valid mode: time-tiled fusion.** At high DM the fused groups no longer fit
in cache. For the `cryopaf` preset (512 channels at 512 µs, DM 0–2000) the
plan has about 40 rows per input channel at every level from 0 to 4, and a
whole-chunk level of one 16-channel group is several MB. Mode `"valid"`
therefore walks each group through the chunk in time tiles of 512 or 256
samples:

- Each thread holds one tile of the group's fused levels in two scratch
  buffers.
- A merge whose delay exceeds the tile takes its head samples from its
  tree-history ring, which the previous tile of the same group has just
  written.
- The group's rings therefore stay cache-resident while the thread walks the
  chunk, and only level $F$ goes to main memory, written with streaming
  stores.
- On the CPU, valid mode is independent of the block length, so the output is
  bit-identical to the unfused path (automatic GPU direct levels excepted).

The automatic depth is the deepest $F$ (then the longer tile) whose mean
per-group working set is at most 1.75 MiB: two scratch tiles plus the rings,
in float. It also requires at least two groups per thread. `summary()`
reports the tile.

**Why the FDMT is memory-bound, and why the regime matters.**

- A merge is one addition per 12 bytes moved: two float reads and one write,
  or 16 bytes with the write-allocate read.
- An unfused level is far larger than any cache, so it streams through DRAM
  at the memory bandwidth. A Xeon socket saturates at about 8 threads.
- The brute-force engines reuse every input sample across DM trials in
  cache, and are compute-bound.
- So the FDMT wins by its operation-count advantage divided by that
  per-operation gap. The advantage depends strongly on the configuration:

| preset (DM 0–2000, piecewise Levin grid) | trials | FDMT rows per sample | brute force / FDMT operations |
| :--- | ---: | ---: | ---: |
| `cryopaf`: 512 channels, 704–1216 MHz, 512 µs, tolerance 1.1 | 1545 | 138k | 5.7× |
| `cryopaf_lband`: 1152 channels, 1100–1714 MHz, 455.6 µs, tolerance 1.1 | 1941 | 88k | 25.5× |
| `parkes_uwl`: 6656 channels, 704–4032 MHz, 64 µs, tolerance 1.6 | 4749 | 1864k | 17.0× |

The FDMT's cost is set by the delay range in samples, about
$\log_2 N_\text{chan}$ levels of $\Delta t_\text{max}$ rows. The brute-force
cost is $N_\text{DM}\,N_\text{chan}$. Few channels and a sparse grid
(trials many samples apart, as the Levin grid places them once smearing
dominates) shrink the FDMT's advantage.

Against whole-chunk fusion with plain stores, time-tiled fusion and streaming
stores gave 1.4× on float32 input and 1.1–1.3× on packed input on a Xeon Gold
6348H (8 threads, valid mode, box smearing, high-DM Levin grids), and let the
fused levels scale past 8 threads on one socket.

The big buffers are first-touched in parallel with huge pages requested, so
runs that span sockets are no longer slower than one socket. For the best
throughput, still run one engine per socket, or split the beams across
sockets.

On CUDA (L40S, device-resident, 4096 channels, `dt_max=2048`, valid mode):

| input | samples | unfused | automatic (depth 3) | speedup | best fixed depth |
| :--- | ---: | ---: | ---: | ---: | :--- |
| float | 16 384 | 7.33 ms | 4.74 ms | 1.55× | 4: 4.61 ms |
| float | 32 768 | 14.34 ms | 9.38 ms | 1.53× | 4: 9.08 ms |
| 1-bit, `int_tree` | 16 384 | 2.74 ms | 2.15 ms | 1.27× | 3: 2.15 ms |
| 1-bit, `int_tree` | 32 768 | 5.70 ms | 4.52 ms | 1.26× | 3: 4.52 ms |

(dmt 0.5.0.) Fusion saves less on packed input, because the integer tree has
already made the early levels small. Deeper tiles also cost more there: at
depth 4 the tile shrinks to 160 samples, and 1-bit input becomes 18% *slower*
than at depth 3 (float gains 3%). Depth 3 is within 3% of the best for every
input.

:::{dropdown} Tile size against resident blocks
:icon: info

Smaller tiles at four resident blocks per SM beat larger tiles at two:
the fused kernel is bound by load latency and instruction issue, not by the
halo it recomputes (48 KiB / 384-sample tiles gave 5.17 ms, 24 KiB / 192
samples 4.91 ms, float, before the 0.5.0 level-kernel change).
:::

## Packed low-bit input and the integer tree (`int_tree`)

Digitisers typically deliver 1-, 2-, 4-, 8- or 16-bit samples. Both engines
accept them packed, with no float copy of the waterfall ever made:

- Rows are LSB-first within each byte for fewer than 8 bits (the DDMT
  convention), padded to a whole byte.
- The layout is `(nbeams, nchans, packed_row_bytes(nsamps, nbits))`.
- The output is float. Without box smearing it is identical to `execute()`
  on the same values converted to float.
- With box smearing (the default) each level-0 box mean of $L$ samples is
  rounded to $f(L)$ fractional bits, the smallest $f$ with $3 \cdot 4^f \ge
  50 L$, and held on the plan's common fixed-point scale $2^{-F}$ ($F$ =
  `plan.box_mean_bits`, C++ `get_box_mean_bits()`). The rounding noise is at
  most 2% of the box-mean noise for input with $\sigma \ge 0.5$ LSB (S/N loss
  ≤ 0.5%). Packed input therefore differs from float input with the same
  values by at most $\sum_c 2^{-(f(L_c)+1)}$ per output sample; packed
  results are bit-identical with `int_tree` on and off, and across backends.
  See {ref}`box means <box-mean>`.

With `int_tree=True` (the default), every tree level whose exact value bound
fits is stored as `uint8` or `uint16` instead of float. This halves or
quarters the memory traffic of those levels. The bounds are propagated
exactly through the plan from level 0. With box smearing a level-0 row holds
a rounded box mean, at most about $\text{maxval} \cdot 2^F \sim
4\sqrt{L_{\max}}\,\text{maxval}$ in units of $2^{-F}$, so the integer levels
stay narrow even at high DM. The
bounds stay far below $2^{24}$, and the integer-to-float conversion is an
exact power-of-two scaling, so the result is bit-identical to the float tree
on the same packed input (`int_tree=False`).

::::{tab-set}
:::{tab-item} Python
:sync: python

```python
import numpy as np
from dmtlib import FDMT

nchans, nsamps, nbits = 4096, 16384, 2
fdmt = FDMT(704.0, 1216.0, nchans, nsamps, 8.192e-5, 2048, nthreads=8)

# (nchans, nsamps * nbits / 8) uint8, sample 0 in the low bits of byte 0
packed = np.random.randint(0, 256, size=(nchans, nsamps * nbits // 8),
                           dtype=np.uint8)
dmt = fdmt.execute(packed, nbits)       # float32 (n_dm, n_times)
```
:::
:::{tab-item} C++
:sync: cpp

```cpp
#include "dmt/algorithms/fdmt.hpp"

dmt::algorithms::FDMT fdmt(704.0F, 1216.0F, 4096, 16384, 8.192e-5F, 2048,
                           0, 1, true, "valid", dmt::Exec::cpu(/*nthreads=*/8));
std::vector<float> dmt(fdmt.get_plan().get_dmt_size());
// packed: nchans rows of ceil(nsamps * nbits / 8) bytes
fdmt.execute(std::span<const uint8_t>(packed), /*nbits=*/2, dmt);
```
:::
::::

On the CUDA backend, `execute(packed, nbits)` behaves the same way, and the
host overload copies only the packed bytes over PCIe.

Speedup over float input, both at the default configuration (automatic
fusion, `int_tree` on; 4096 channels, `dt_max=2048`, 16 384 samples):

| input | Xeon 6348H (8 threads) | L40S (device-resident) |
| :--- | ---: | ---: |
| 1-bit | 2.73× | 2.18× |
| 2-bit | 2.58× | 2.02× |
| 4-bit | 2.23× | 1.75× |
| 8-bit | 1.36× | 1.19× |
| 16-bit | 1.00× | 1.01× |

Packed input without the integer tree runs no faster than float input: **the
gain comes from the narrow tree state**. At 16 bits nearly every level stays
float, so the per-channel unpack becomes a small net cost; pass 16-bit data
as float if you already have it in that form. Against the original
level-by-level float path, the default configuration with 1-bit input is
3.9× faster on the Xeon (76.4 → 19.5 ms, 8 threads) and 3.3× on the
L40S (7.33 → 2.20 ms, dmt 0.5.0).

```{note}
Integer tree levels cannot be viewed through the stepper's `view_*` methods,
which throw at such a level. To inspect every level of a packed stepper run,
construct the engine with `int_tree=False`.
```

## Threads, block size and construction

- **Threads** (`nthreads`, CPU): scaling is good up to the point where the
  memory bus saturates. On the Xeon, 8 threads give 5.4× over 1 thread.
- **Block size** (`nsamps`): in valid mode the overlap-save history makes the
  result independent of the block size, so choose it for latency and
  throughput. Throughput is best between a few thousand and ~16K samples,
  where the fused channel groups stay cache-resident. At 65 536 samples the
  fused per-sample cost rises somewhat, while the unfused cost is flat. In
  valid mode the tree state does not grow with the block beyond the chunk
  length (see
  {ref}`below <chunked-execution>`).
- **Construction and memory:** the constructor computes the plan and
  allocates *all* working memory: the tree state (two levels of one chunk),
  the streaming history, and the per-thread fusion/unpack scratch (on CUDA:
  device state, history, the plan and the fused-kernel tables; ~155 ms on an
  L40S host). `execute()`
  then allocates nothing (on the CPU a test enforces this), so a streaming
  pipeline constructs once and calls `execute()` block after block.
  `fdmt.memory_usage` (Python) / `get_memory_usage()` (C++) reports the
  bytes per category and the output buffer each call needs
  (`nbeams * plan.dmt_size` floats; the engine never uses it as scratch);
  `fdmt.summary()` prints the same breakdown together with the plan. To
  multiplex several streams on one engine, use `save_history()` /
  `load_history()` rather than building one engine per stream. On a GPU
  backend, the host-memory `execute()` overloads
  stage the input and output on the device in buffers allocated on the first
  host call and reused afterwards.
- **Multiple beams:** `nbeams > 1` shares one plan across beams. See
  [Multi-beam batching](multibeam.md).

```{tip}
In Python, pass `out=` to `execute()` to reuse the output array too;
otherwise every call allocates and page-faults a new one (~10% of the
runtime at 16K samples).
```

(chunked-execution)=
### Chunked execution (`chunk_nsamps`)

In `mode="valid"`, `execute()` runs each block in chunks of `chunk_nsamps`
samples. The tree state is then two levels of rows × chunk instead of rows ×
`nsamps`. That matters at high DM: the plan has up to ~10⁵ level-0 rows, and
the state of a whole block would be tens of GB while the block's input and
output are a few hundred MB. Valid mode is overlap-save inside the tree.
Packed paths, integer-valued input, explicit fusion depths and the CPU tree
are bit-identical for any chunk; automatic GPU direct levels agree only within
float rounding (see above). Each chunk costs a little extra cross-block
history traffic, a few per cent at a few thousand samples.

- **Default** (`chunk_nsamps=None` in Python, `0` in C++): the longest power
  of two ≥ 2048 whose two state halves (all beams) fit 2 GiB, capped at
  `nsamps`. On a GPU the engine first tries the longest power of two ≥ 256
  whose state halves fit the device's L2 cache, so each merge reads the level
  below from L2 instead of DRAM; it falls back to the 2 GiB rule when even 256
  samples do not fit. On the L40S (96 MiB L2) this picks 512 for `cryopaf`
  and 1024 for `cryopaf_lband`, 1.4× and 1.8× faster than the 2 GiB rule's
  8192 and 16384. Read the value used with `fdmt.chunk_nsamps` (C++
  `get_chunk_nsamps()`); `nsamps` means whole blocks.
- **Explicit** values are rounded up to a multiple of 64 and capped at
  `nsamps`. Pass a longer chunk if memory allows, or a shorter one to save
  more.
- **Full and roll modes:** `"full"` and `"roll"` always run whole blocks; a shorter `chunk_nsamps`
  raises `ValueError` there.
- **Stepper:** the stepper (`reset()` / `advance()` / `view_*`) needs whole blocks:
  construct a stepping engine with `chunk_nsamps=nsamps`. See
  {ref}`stepper rules <stepper-rules>`.

Engine memory for the `cryopaf` preset (512 channels, 704–1216 MHz, 512 µs,
DM 0–2000 in 1545 trials) with 32 768-sample blocks (16.8 s), CPU with 8
threads (`fdmt.memory_usage`); `bench/scripts/bench_chunking.py` adds the
timings and peak RSS on your machine:

| execution | chunk | tree state | tree history | engine total |
| :--- | ---: | ---: | ---: | ---: |
| whole block | 32 768 | 5.46 GiB | 0.11 GiB | 5.63 GiB |
| chunked | 16 384 | 2.73 GiB | 0.11 GiB | 2.90 GiB |
| chunked, automatic | 8192 | 1.36 GiB | 0.11 GiB | 1.54 GiB |
| chunked | 4096 | 0.68 GiB | 0.11 GiB | 0.85 GiB |
| chunked | 2048 | 0.34 GiB | 0.11 GiB | 0.51 GiB |

The output is identical in every row, and the time per block changes little
(the fusion depth adapts to the chunk). The state of a whole block grows with
the block length (and with the DM range, through the rows of the first
levels); the chunked state does not.

## Summary of the performance parameters

| constructor argument | default | applies to | effect |
| :--- | :--- | :--- | :--- |
| `int_tree` | `True` | packed input, CPU + CUDA | uint8/uint16 tree levels where exact; 1.2–2.7× over float for ≤ 8-bit input |
| `fuse_levels` | automatic (`None` in Python, `kFDMTAutoFuse` in C++) | `execute()`, CPU + CUDA | first levels computed on-chip; 1.26–1.47× on the CPU, 1.07–1.55× on CUDA |
| `chunk_nsamps` | automatic (`None` in Python, `0` in C++) | `execute()` in valid mode, CPU + CUDA | tree state of one chunk instead of one block; bounds memory at high DM |

Pass `fuse_levels=0, int_tree=False` to get the original level-by-level float
path, for example to compare against or to benchmark.

## Reproducing the numbers

Build with benchmarks (`-DDMT_BUILD_BENCHMARKS=ON`, Release), then run from
the build directory (the Python script from the repository root):

```bash
# CPU: fusion depth sweep, packed input x int_tree x fusion
./bench/dmt_bench --benchmark_filter='BM_fdmt_fused|BM_fdmt_packed'
# CUDA: device-resident, fusion depth sweep for float and 1-bit input
./bench/dmt_bench --benchmark_filter='BM_fdmt_execute_cuda_packed'
# Plan traffic model and fusion sweep from Python
python bench/scripts/fdmt_cache_benchmark.py --section reuse --nchans 4096 --dt-max 2048
python bench/scripts/fdmt_cache_benchmark.py --section fusion --nchans 4096 --dt-max 2048
```

The `fuse` counter in each benchmark row is the depth actually used.

These are the tuning microbenchmarks (`dmt_bench`). The cross-algorithm,
cross-machine numbers on the [Benchmarks](../benchmarks.md) page come from
the separate published suite (`dmt_bench_suite`); `bench/README.md` describes
how to run it.
