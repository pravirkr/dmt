# Performance Tuning

The FDMT engines (`FDMTCPU`, `FDMTCUDA`) are tuned out of the box. Their two
*performance parameters*, the last constructor arguments `fuse_levels` and
`int_tree`, default to the fastest settings measured, and neither changes the
result by a single bit. Most users never set them. This page explains what
they do, what you control yourself (input format, threads, block size), how
memory is allocated, and why some optimizations that look promising were
measured and rejected.

```python
fdmt = FDMTCPU(704.0, 1216.0, 4096, 16384, 8.192e-5, 2048, nthreads=1)
fdmt.fuse_levels       # depth chosen for this plan (fuse_levels=None default)
fdmt = FDMTCPU(704.0, 1216.0, 4096, 16384, 8.192e-5, 2048,
               fuse_levels=4, int_tree=True)  # explicit depth (advanced)
print(fdmt.memory_usage)  # bytes allocated at construction
```

```cpp
// ..., nthreads, nbeams, fuse_levels = kFDMTAutoFuse, int_tree = true
dmt::algorithms::FDMTCPU fdmt(704.0F, 1216.0F, 4096, 16384, 8.192e-5F, 2048,
                              0, 1, true, "valid", false, 1, 1,
                              dmt::algorithms::kFDMTAutoFuse, true);
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
  tile of at least 256 samples fits in the portable 48 KiB of shared memory.

  The time tile here is not the intra-level cache tiling rejected in
  [section 5](#5-what-was-tried-and-rejected). That tiling reorders one
  level's merges and moves the same bytes. The GPU tile exists only because a
  block's shared memory (≤ 100 KiB) cannot hold whole rows the way a CPU
  cache can, so the fused levels are cut into tiles. It is the same fusion,
  and it removes the intermediate levels' DRAM traffic. The unfused CUDA
  merge kernel is unchanged and untiled.

Fusion applies to `execute()` only. The stepper (`reset()` / `advance()`)
always runs level by level, so every level stays inspectable. Read the depth
actually used with `fdmt.fuse_levels` (Python) or `get_fuse_levels()` (C++).
An explicit depth is clamped to the plan's merge levels (and, on CUDA, reduced
until it fits shared memory), with a warning.

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

Defaults and automatic budgets are explicitly targeted at **Intel x86 HPC server architectures (Xeon/EPYC)**—the production environment for radio astronomy pipelines. The default memory budget of $\max(36\,\text{MiB} / n_\text{threads}, 5\,\text{MiB})$ is sized for standard ~32–36 MiB L3 cache slices on server sockets with typical 1–8 thread allocations per pipeline instance. Pass an explicit `fuse_levels` if a sweep on your specific hardware (`BM_fdmt_fused`, see [Reproducing](#7-reproducing-the-numbers)) finds a better depth.

On CUDA (L40S, device-resident, 4096 channels, `dt_max=2048`, valid mode):

| input | samples | unfused | automatic (depth 3) | speedup | best fixed depth |
| :--- | ---: | ---: | ---: | ---: | :--- |
| float | 16 384 | 7.50 ms | 5.48 ms | 1.37× | 4: 5.34 ms |
| float | 32 768 | 14.70 ms | 10.68 ms | 1.38× | 4: 10.28 ms |
| 1-bit, `int_tree` | 16 384 | 3.69 ms | 3.44 ms | 1.07× | 1: 3.42 ms |
| 1-bit, `int_tree` | 32 768 | 7.42 ms | 6.93 ms | 1.07× | 3: 6.92 ms |

Fusion saves less on packed input, because the integer tree has already made
the early levels small. Deeper tiles also cost more there: at depth 5 the
tile shrinks to 64 samples with a 39-sample halo, and 1-bit input becomes 15%
*slower* than at depth 3. Depth 3 is the one depth within 3% of the best for
every input, so the automatic rule (shared memory <= 48 KiB, tile >= 256 samples,
halo <= half a tile) stops there.

---

## 3. Packed low-bit input and the integer tree (`int_tree`)

Digitisers typically deliver 1-, 2-, 4- or 8-bit samples. Both engines accept
them packed, with no float copy of the waterfall ever made:

- Rows are LSB-first within each byte for fewer than 8 bits (the DDMT
  convention), padded to a whole byte.
- The layout is `(nbeams, nchans, packed_row_bytes(nsamps, nbits))`.
- The output is float, identical to `execute()` on the same values converted
  to float.

With `int_tree=True` (the default), every tree level whose exact value bound
fits is stored as `uint8` or `uint16` instead of float. This halves or
quarters the memory traffic of those levels. The bounds are propagated
exactly through the plan, and they stay far below $2^{24}$, so the result is
bit-identical to the float path.

```python
import numpy as np
from dmtlib import FDMTCPU

nchans, nsamps, nbits = 4096, 16384, 2
fdmt = FDMTCPU(704.0, 1216.0, nchans, nsamps, 8.192e-5, 2048, nthreads=8)

# (nchans, nsamps * nbits / 8) uint8, sample 0 in the low bits of byte 0
packed = np.random.randint(0, 256, size=(nchans, nsamps * nbits // 8),
                           dtype=np.uint8)
dmt = fdmt.execute(packed, nbits)       # float32 (n_dm, n_times)
```

```cpp
#include "dmt/algorithms/fdmt.hpp"

dmt::algorithms::FDMTCPU fdmt(704.0F, 1216.0F, 4096, 16384, 8.192e-5F, 2048,
                              0, 1, true, "valid", false, /*nthreads=*/8);
std::vector<float> dmt(fdmt.get_plan().get_buffer_size());
// packed: nchans * dmt::utils::packed_row_bytes(nsamps, nbits) bytes
fdmt.execute(std::span<const uint8_t>(packed), /*nbits=*/2, dmt);
```

On CUDA, `FDMTCUDA.execute(packed, nbits)` behaves the same way, and the host
overload copies only the packed bytes over PCIe.

Speedup over float input, both at the default configuration (automatic
fusion, `int_tree` on; 4096 channels, `dt_max=2048`, 16 384 samples):

| input | M1 Pro (8 threads) | Xeon 6348H (8 threads) | L40S (device-resident) |
| :--- | ---: | ---: | ---: |
| 1-bit | 2.02× | 2.73× | 1.59× |
| 2-bit | 2.00× | 2.58× | 1.53× |
| 4-bit | 1.77× | 2.23× | 1.43× |
| 8-bit | 1.25× | 1.36× | 1.10× |
| 16-bit | 0.89× | 1.00× | 0.99× |

Packed input without the integer tree runs no faster than float input: the
gain comes from the narrow tree state. At 16 bits nearly every level stays
float, so the per-channel unpack becomes a small net cost; pass 16-bit data
as float if you already have it in that form. Against the original
level-by-level float path, the default configuration with 1-bit input is
**3.2× faster on the M1 Pro** (39.1 → 12.2 ms, 8 threads), **3.9× on the
Xeon** (76.4 → 19.5 ms, 8 threads) and **2.2× on the L40S** (7.50 → 3.44 ms).

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
  unfused cost is flat.
- **Construction and memory:** the constructor computes the plan and
  allocates *all* working memory: the tree state, the streaming history,
  and the per-thread fusion/unpack scratch (on CUDA: device state, history,
  the plan and the fused-kernel tables; ~155 ms on an L40S host). `execute()`
  then allocates nothing (on the CPU a test enforces this), so a streaming
  pipeline constructs once and calls `execute()` block after block.
  `fdmt.memory_usage` (Python) / `get_memory_usage()` (C++) reports the
  bytes per category and the output buffer each call needs; `verbose=True`
  logs the same breakdown at construction. To multiplex several streams on
  one engine, use `save_history()` / `load_history()` rather than building
  one engine per stream. The host-memory `FDMTCUDA.execute()` overloads are
  the one exception: they stage the input and output on the device per call.
- **Multiple beams:** `nbeams > 1` shares one plan across beams. See
  [Multi-Beam Batching](multibeam.md).

---

## 6. Summary of the performance parameters

| constructor argument | default | applies to | effect |
| :--- | :--- | :--- | :--- |
| `int_tree` | `True` | packed input, CPU + CUDA | uint8/uint16 tree levels where exact; 1.1–2.7× over float for ≤ 8-bit input |
| `fuse_levels` | automatic (`None` in Python, `kFDMTAutoFuse` in C++) | `execute()`, CPU + CUDA | first levels computed on-chip; 1.14–1.78× on the CPU, 1.07–1.38× on CUDA |

Pass `fuse_levels=0, int_tree=False` to get the original level-by-level float
path, for example to compare against or to benchmark.

---

## 7. Reproducing the numbers

Build with benchmarks (`-DDMT_BUILD_BENCHMARKS=ON`, Release), then run:

```bash
# CPU: fusion depth sweep, packed input x int_tree x fusion
./bench/dmt_bench --benchmark_filter='BM_fdmt_fused|BM_fdmt_packed'
# CUDA: device-resident, fusion depth sweep for float and 1-bit input
./bench/dmt_bench --benchmark_filter='BM_fdmt_execute_cuda_packed'
# Plan traffic model and fusion sweep from Python
python bench/fdmt_cache_benchmark.py --section reuse --nchans 4096 --dt-max 2048
python bench/fdmt_cache_benchmark.py --section fusion --nchans 4096 --dt-max 2048
```

The `fuse` counter in each benchmark row is the depth actually used.
