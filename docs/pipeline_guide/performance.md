# Performance Tuning

The FDMT engine (`FDMT`, on the CPU and CUDA backends) is tuned out of the box. Its two
*performance parameters*, the constructor arguments `fuse_levels` and
`int_tree`, default to the fastest settings measured, and neither changes the
result by a single bit. The *memory parameter* `chunk_nsamps`, the last
constructor argument, bounds the tree state of large plans, also without
changing the result. Most users never set any of them. This page explains what
they do, what you control yourself (input format, threads, block size), how
memory is allocated, and why some optimizations that look promising were
measured and rejected.

```python
fdmt = FDMT(704.0, 1216.0, 4096, 16384, 8.192e-5, 2048, nthreads=1)
fdmt.fuse_levels       # depth chosen for this plan (fuse_levels=None default)
fdmt = FDMT(704.0, 1216.0, 4096, 16384, 8.192e-5, 2048,
               fuse_levels=4, int_tree=True)  # explicit depth (advanced)
fdmt.chunk_nsamps         # chunk length chosen for this plan (chunk_nsamps=None)
print(fdmt.memory_usage)  # bytes allocated at construction
```

```cpp
// ..., exec, nbeams, fuse_levels = kFDMTAutoFuse, int_tree = true,
//      chunk_nsamps = 0 (automatic)
dmt::algorithms::FDMT fdmt(704.0F, 1216.0F, 4096, 16384, 8.192e-5F, 2048,
                           0, 1, true, "valid", dmt::Exec::cpu(1), 1,
                           dmt::algorithms::kFDMTAutoFuse, true, 0);
```

---

## 1. Where the time goes

Every FDMT tree level is one pass of the same loop over rows of the state:

$$\text{out}[t] = \text{tail}[t] + \text{head}[t - d]$$

That is two loads and one store per output sample, for one addition. There is
no arithmetic to optimize: **the transform is memory-bandwidth bound**, and
the merge loop already runs at the bandwidth limit. On an M1 Pro (4096
channels, `dt_max=2048`, 16 384 samples, 1 thread), each merge level moves its
three streams at 93–99% of the bandwidth that a bare `c[i] = a[i] + b[i]`
achieves on the same machine:

| level | rows | ms | GB/s | % of `c = a + b` |
| ---: | ---: | ---: | ---: | ---: |
| 0 (init) | 6144 | 12.8 | 52.6 | 69 |
| 1 | 4096 | 11.7 | 69.0 | 90 |
| 2 | 3072 | 8.5 | 71.3 | 93 |
| 3 | 2560 | 6.9 | 73.0 | 95 |
| 4–12 | 2049–2304 | 5.3–6.2 each | 73–76 | 96–99 |

Two consequences follow:

- The **early levels dominate** (levels 0–3 are ~45% of the time), because
  they have the most rows.
- The only way to go faster is to **move fewer bytes**. The two defaults below
  do exactly that. One keeps the early levels on-chip (level fusion); the
  other stores them in fewer bytes (the integer tree).

---

## 2. Level fusion (`fuse_levels`)

Level-$F$ sub-band $g$ depends only on the $2^F$ input channels below it. With
fusion, `execute()` processes one such channel group at a time. It builds the
group's level-0 rows and merges them up to level $F$ on-chip, then writes only
level $F$ to memory. The intermediate levels never make the round trip to
DRAM.

- **CPU:** each thread works on one group in a cache-resident scratch buffer
  holding full rows. The automatic depth (the default) is the deepest $F$
  whose two scratch buffers fit in $\max(36\,\text{MiB} / n_\text{threads},
  5\,\text{MiB})$ per thread. The scratch size follows from the plan
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

Fusion applies to `execute()` only. The stepper (`reset()` / `advance()`)
always runs level by level, so every level stays inspectable. Read the depth
actually used with `fdmt.fuse_levels` (Python) or `get_fuse_levels()` (C++).
An explicit depth is clamped to the plan's merge levels (and, on CUDA, reduced
until it fits shared memory); `fuse_levels` / `get_fuse_levels()` report the
depth actually used.

### Hybrid tree traversal: vertical depth-first meets horizontal breadth-first

Conceptually, the FDMT tree can be traversed in two classical ways:

1. **Horizontal Breadth-First (BFS):** The standard FDMT algorithm traverses the entire band level by level across all channels. While highly parallel and regular, it is cache-unfriendly: every level writes hundreds of megabytes to DRAM and reads them back.
2. **Top-Down Recursive Depth-First (DFS):** Computing a final DM trial by recursively descending to its leaf channels is often proposed as cache-friendly. In practice, pure recursion is deeply flawed for FDMT: because FDMT is a Directed Acyclic Graph (DAG) with overlapping sub-band delay paths, pure recursion either causes exponential redundant calculations or requires dynamic memoization tables that destroy vectorization and fail on GPUs. Furthermore, near the top of the tree, sub-bands span thousands of channels and exceed cache capacity anyway.

Level fusion implements the **best of both worlds**:

- **First $F$ stages (Vertical truncated DFS):** The early channels form strictly disjoint binary sub-trees of size $2^F$ (e.g., 8 or 16 channels). For each sub-tree, all merges from level 0 up to level $F$ are evaluated vertically on-chip inside cache/shared memory. Intermediate levels $0 \dots F-1$ never touch DRAM. Because the early levels dominate data volume (representing ~45–50% of the entire transform's memory traffic), this eliminates the vast majority of memory bus round-trips.
- **Remaining stages (Horizontal BFS):** Once level $F$ is reached, the sub-trees begin merging across wider frequency spans. Here, the engine transitions back to flat, breadth-first global ping-pong merges in memory.

This hybrid structure requires **zero recursion, zero dynamic allocation, and zero redundant computation**: it executes as flat, predictable loops perfectly suited for SIMD vectorization and GPU thread blocks.

Measured on the CPU (float input, valid mode, box smearing, 4096 channels,
`dt_max=2048`):

| machine | samples | threads | unfused | automatic | speedup |
| :--- | ---: | ---: | ---: | ---: | ---: |
| M1 Pro | 4096 | 1 | 29.3 ms | 22.2 ms | 1.32× |
| M1 Pro | 4096 | 8 | 11.6 ms | 6.5 ms | 1.78× |
| M1 Pro | 16 384 | 1 | 91.3 ms | 80.1 ms | 1.14× |
| M1 Pro | 16 384 | 8 | 37.4 ms | 24.6 ms | 1.52× |
| M1 Pro | 65 536 | 8 | 149.9 ms | 114.4 ms | 1.31× |
| Xeon Gold 6348H | 4096 | 8 | 18.0 ms | 12.2 ms | 1.47× |
| Xeon Gold 6348H | 16 384 | 1 | 394.7 ms | 282.1 ms | 1.40× |
| Xeon Gold 6348H | 16 384 | 8 | 77.0 ms | 53.4 ms | 1.44× |
| Xeon Gold 6348H | 65 536 | 1 | 1584 ms | 1185 ms | 1.34× |
| Xeon Gold 6348H | 65 536 | 8 | 307.3 ms | 244.4 ms | 1.26× |

The rule was fitted to fixed-depth sweeps on both machines (1, 2, 4 and 8
threads; 4K, 16K and 64K samples):
- **Xeon, 1 thread:** it picks the best measured depth at 16K and 64K.
- **At 2–8 threads:** it is within 3% of the best depth, except 4096 samples
  on the Xeon (6–8.5%).
- **M1, 1 thread:** it trails the best depth by 5–8%, where the M1's large
  shared L2 favours a shallower depth.

Defaults and automatic budgets are explicitly targeted at **Intel x86 HPC server architectures (Xeon/EPYC)**—the production environment for radio astronomy pipelines. The default memory budget of $\max(36\,\text{MiB} / n_\text{threads}, 8\,\text{MiB})$ is sized for standard ~32–36 MiB L3 cache slices on server sockets with typical 1–8 thread allocations per pipeline instance. Pass an explicit `fuse_levels` if a sweep on your specific hardware (`BM_fdmt_fused`, see [Reproducing](#6-reproducing-the-numbers)) finds a better depth.

The table and the budget above are for the whole-block fused path, which
modes `"full"` and `"roll"` still use.

**Valid mode: time-tiled fusion.** At high DM the fused groups no longer fit
in cache. At 512 µs with 1024 channels and DM 0–2000, the plan has about 22
rows per channel at every level from 0 to 5, and a whole-chunk level of one
16-channel group is several MB. Mode `"valid"` therefore walks each group
through the chunk in **time tiles** of 512 or 256 samples:
- Each thread holds one tile of the group's fused levels in two scratch
  buffers.
- A merge whose delay exceeds the tile takes its head samples from its
  tree-history ring, which the previous tile of the same group has just
  written.
- The group's rings therefore stay cache-resident while the thread walks the
  chunk, and only level $F$ goes to main memory, written with streaming
  stores.
- Valid mode is independent of the block length, so the output is
  bit-identical to the unfused path.

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

| configuration (704–1216 MHz, 512 µs, piecewise Levin grid) | trials | FDMT rows per sample | brute force / FDMT operations |
| :--- | ---: | ---: | ---: |
| 1024 channels, DM 0–2000, tolerance 1.1 (`cryopaf`) | 2501 | 178k | 14× |
| 4096 channels, DM 0–1000, tolerance 1.05 (`cryopaf4k`) | 5183 | 141k | 151× |
| 4096 channels, 81.92 µs, dense dt grid 0–2048 (old suite) | 2049 | 35k | 241× |

The FDMT's cost is set by the delay range in samples, about
$\log_2 N_\text{chan}$ levels of $\Delta t_\text{max}$ rows. The brute-force
cost is $N_\text{DM}\,N_\text{chan}$. Few channels and a sparse grid
(trials many samples apart, as the Levin grid places them once smearing
dominates) shrink the FDMT's advantage.

Time-tiled fusion and streaming stores, measured on a Xeon Gold 6348H at 8
threads, $N = 32768$, valid mode, box smearing. "Before" is whole-chunk
fusion with plain stores:

| configuration | input | before | after | speedup |
| :--- | :--- | ---: | ---: | ---: |
| `cryopaf` | float32 | 685 ms | 485 ms | 1.41× |
| `cryopaf` | 2-bit | 329 ms | 280 ms | 1.18× |
| `cryopaf` | 8-bit | 586 ms | 457 ms | 1.28× |
| `cryopaf4k` | float32 | 493 ms | 346 ms | 1.42× |
| `cryopaf4k` | 2-bit | 252 ms | 232 ms | 1.09× |
| `cryopaf4k` | 8-bit | 421 ms | 325 ms | 1.30× |
| 4096 ch, 81.92 µs, dt 2048, $N = 16384$ | float32 | 52.2 ms | 37.2 ms | 1.40× |

With 16 threads on one socket, `cryopaf` float runs in 357 ms (617 ms
before): the fused levels now scale with the cores.

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
input. Smaller tiles at four resident blocks per SM beat larger tiles at two:
the fused kernel is bound by load latency and instruction issue, not by the
halo it recomputes (48 KiB / 384-sample tiles gave 5.17 ms, 24 KiB / 192
samples 4.91 ms, float, before the 0.5.0 level-kernel change).

---

## 3. Packed low-bit input and the integer tree (`int_tree`)

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

```cpp
#include "dmt/algorithms/fdmt.hpp"

dmt::algorithms::FDMT fdmt(704.0F, 1216.0F, 4096, 16384, 8.192e-5F, 2048,
                           0, 1, true, "valid", dmt::Exec::cpu(/*nthreads=*/8));
std::vector<float> dmt(fdmt.get_plan().get_dmt_size());
// packed: nchans rows of ceil(nsamps * nbits / 8) bytes
fdmt.execute(std::span<const uint8_t>(packed), /*nbits=*/2, dmt);
```

On the CUDA backend, `execute(packed, nbits)` behaves the same way, and the
host overload copies only the packed bytes over PCIe.

Speedup over float input, both at the default configuration (automatic
fusion, `int_tree` on; 4096 channels, `dt_max=2048`, 16 384 samples):

| input | M1 Pro (8 threads) | Xeon 6348H (8 threads) | L40S (device-resident) |
| :--- | ---: | ---: | ---: |
| 1-bit | 2.02× | 2.73× | 2.18× |
| 2-bit | 2.00× | 2.58× | 2.02× |
| 4-bit | 1.77× | 2.23× | 1.75× |
| 8-bit | 1.25× | 1.36× | 1.19× |
| 16-bit | 0.89× | 1.00× | 1.01× |

Packed input without the integer tree runs no faster than float input: the
gain comes from the narrow tree state. At 16 bits nearly every level stays
float, so the per-channel unpack becomes a small net cost; pass 16-bit data
as float if you already have it in that form. Against the original
level-by-level float path, the default configuration with 1-bit input is
**3.2× faster on the M1 Pro** (39.1 → 12.2 ms, 8 threads), **3.9× on the
Xeon** (76.4 → 19.5 ms, 8 threads) and **3.3× on the L40S** (7.33 → 2.20 ms, dmt 0.5.0).

```{note}
Integer tree levels cannot be viewed through the stepper's `view_*` methods,
which throw at such a level. To inspect every level of a packed stepper run,
construct the engine with `int_tree=False`.
```

---

## 4. Threads, block size and construction

- **Threads** (`nthreads`, CPU): scaling is good up to the point where the
  memory bus saturates. On the M1 Pro, 8 threads give 2.3× (unfused) to 3.1×
  (fused) over 1 thread; on the Xeon, 5.4× at 8 threads.
- **Block size** (`nsamps`): in valid mode the overlap-save history makes the
  result independent of the block size, so choose it for latency and
  throughput. Throughput is best between a few thousand and ~16K samples,
  where the fused channel groups stay cache-resident. At 65 536 samples the
  fused per-sample cost rises (M1, 8 threads: 1.16× the 16K cost), while the
  unfused cost is flat. In valid mode the tree state does not grow with the
  block beyond the chunk length (see
  {ref}`below <chunked-execution>`).
- **Construction and memory:** the constructor computes the plan and
  allocates *all* working memory: the tree state (two levels of one chunk),
  the streaming history, and the per-thread fusion/unpack scratch (on CUDA: device state, history,
  the plan and the fused-kernel tables; ~155 ms on an L40S host). `execute()`
  then allocates nothing (on the CPU a test enforces this), so a streaming
  pipeline constructs once and calls `execute()` block after block.
  `fdmt.memory_usage` (Python) / `get_memory_usage()` (C++) reports the
  bytes per category and the output buffer each call needs
  (`nbeams * plan.dmt_size` floats; the engine never uses it as scratch);
  `fdmt.summary()` prints the same breakdown together with the plan. To multiplex several streams on
  one engine, use `save_history()` / `load_history()` rather than building
  one engine per stream. On a GPU backend, the host-memory `execute()` overloads
  stage the input and output on the device in buffers allocated on the first
  host call and reused afterwards. In Python, pass `out=` to `execute()` to
  reuse the output array too; otherwise every call allocates and page-faults
  a new one (~10% of the runtime at 16K samples).
- **Multiple beams:** `nbeams > 1` shares one plan across beams. See
  [Multi-Beam Batching](multibeam.md).

(chunked-execution)=
### Chunked execution (`chunk_nsamps`)

In `mode="valid"`, `execute()` runs each block in chunks of `chunk_nsamps`
samples. The tree state is then two levels of rows × chunk instead of rows ×
`nsamps`. That matters at high DM: the plan has up to ~10⁵ level-0 rows, and
the state of a whole block would be tens of GB while the block's input and
output are a few hundred MB. Valid mode is overlap-save inside the tree, so
the output is **bit-identical for any chunk**. Each chunk costs a little extra
cross-block history traffic, a few per cent at a few thousand samples.

- **Default** (`chunk_nsamps=None` in Python, `0` in C++): the longest power
  of two ≥ 2048 whose two state halves (all beams) fit 2 GiB, capped at
  `nsamps`. Like the fusion depth, it follows from the plan alone, with no
  hardware detection. Read the value used with `fdmt.chunk_nsamps` (C++
  `get_chunk_nsamps()`); `nsamps` means whole blocks.
- **Explicit** values are rounded up to a multiple of 64 and capped at
  `nsamps`. Pass a longer chunk if memory allows, or a shorter one to save
  more.
- `"full"` and `"roll"` always run whole blocks; a shorter `chunk_nsamps`
  raises `ValueError` there.
- The stepper (`reset()` / `advance()` / `view_*`) needs whole blocks:
  construct a stepping engine with `chunk_nsamps=nsamps`. See
  {ref}`Stepper Rules <stepper-rules>`.

Measured on an M1 Pro (8 threads, float input) with the benchmark
configuration (1024 channels, 704–1216 MHz, 512 µs, DM 0–2000 in 2501 trials)
and 32 768-sample blocks (16.8 s), `bench/scripts/bench_chunking.py`:

| execution | chunk | fusion depth (auto) | time per block | tree state | peak memory |
| :--- | ---: | ---: | ---: | ---: | ---: |
| whole block | 32 768 | 0 | 376 ms | 5.56 GiB | 6.20 GiB RSS |
| chunked | 16 384 | 0 | 385 ms | 2.78 GiB | 3.54 GiB RSS |
| chunked, automatic | 8192 | 1 | 358 ms | 1.39 GiB | 2.20 GiB RSS |
| chunked | 4096 | 2 | 396 ms | 0.70 GiB | 1.50 GiB RSS |
| chunked | 2048 | 3 | 411 ms | 0.35 GiB | 1.15 GiB RSS |

The output is identical in every row. The state of a whole block grows with
the block length (and with the DM range, through the rows of the first
levels); the chunked state does not.

## 5. Summary of the performance parameters

| constructor argument | default | applies to | effect |
| :--- | :--- | :--- | :--- |
| `int_tree` | `True` | packed input, CPU + CUDA | uint8/uint16 tree levels where exact; 1.1–2.7× over float for ≤ 8-bit input |
| `fuse_levels` | automatic (`None` in Python, `kFDMTAutoFuse` in C++) | `execute()`, CPU + CUDA | first levels computed on-chip; 1.14–1.78× on the CPU, 1.07–1.38× on CUDA |
| `chunk_nsamps` | automatic (`None` in Python, `0` in C++) | `execute()` in valid mode, CPU + CUDA | tree state of one chunk instead of one block; bounds memory at high DM |

Pass `fuse_levels=0, int_tree=False` to get the original level-by-level float
path, for example to compare against or to benchmark.

---

## 6. Reproducing the numbers

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
