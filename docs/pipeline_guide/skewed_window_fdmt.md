# Streaming FDMT on skewed windows (prototype)

A CPU prototype of a valid-mode FDMT that keeps no per-merge tree history
above level 4: `lib/dmt/fdmt_skew.hpp` (`FDMTSkewCpu`,
`compute_skew_windows`, `estimate_skew_memory`), tested by
`tests/cpp/fdmt_skew_cpu_t.cpp` and measured by `bench/fdmt_skew_cpu_b.cpp`
and `bench/fdmt_skew_mem_b.cpp`. It is not wired into the `FDMT` facade.

## The problem

The streaming tree ([streaming and history](streaming_and_history.md))
keeps, at every merge, the last `delay` samples of its head input, so the
tree history is the sum of all merge delays (`plan.tree_history_size`). On a
dense delay grid that grows as $0.755\,\Delta t_{\max}^2$ floats, and on the
Levin grids as trials $\times \Delta t_{\max}$: 10.1 GB for the
`parkes_uwl` preset ($\Delta t_{\max} = 253\,891$, 4749 trials), doubled on
a GPU. The chunk state (two levels of `rows × chunk` floats) is 2.1 GB at
chunk 1024 and 8.5 GB at 4096. UWL therefore does not fit a 10 GB budget at
any chunk length.

## The idea

Walk the plan top-down. A root row is read at delay 0. A merge at level
$l$ with head delay $d$ passes its delay range $[s_{lo}, s_{hi}]$ to its
tail child and $[s_{lo} + d, s_{hi} + d]$ to its head child; a copy passes
its own. Row $r$ then has $s_{lo}(r)$ and $s_{hi}(r)$, the smallest and
largest accumulated head delay over every path to a trial that reads it.
For an output chunk $[T, T + n)$ the row is needed exactly on

$$[T - s_{hi}(r),\; T + n - s_{lo}(r)),$$

its *skewed window*. A parent computed on its own window finds both
children inside theirs, so a chunk needs nothing of the past except the
window overlap $s_{hi} - s_{lo}$, the row's *spread*, which the row keeps as
a short ring: every chunk computes exactly $n$ new samples per row and
carries the last `spread` over. The per-row rings replace the per-merge
histories.

A row's spread is set by how many trials read it. A level-0 row (one
channel at one box length) is read by every trial, so its spread is about
$\Delta t_{\max}$; a row at level 4 and above is read by a narrow DM range
and its spread is tiny (tables below). The two cannot be mixed: a level-4
row evaluated in the past needs its children in the past. So levels 0–3 are
never materialised. A level-4 row is the sum of its 16 leaves (channel, box
length, accumulated delay; the tables of `fdmt_direct.hpp`, which the CUDA
direct path uses), each read as a difference of the channel's running sum.
The running sums are built per level-4 subband and chunk from a raw-input
ring of `nchans × (max_r (s_hi(r) + leaf extent) + chunk)` samples, 1 byte
per sample for packed 8-bit input, so a row can be evaluated at any time
inside the ring, in particular on its own skewed window. This is what the
CUDA direct levels do for a chunk's own time range, with the running-sum
window widened to the ring.

**Exactness.** Levels 0–4 are summed in integer arithmetic (uint32 running
sums, exact box sums, the plan's box-mean recipes for packed input) and
converted once to float at level 4, the tree's own int → float conversion.
The merges above are the tree's additions with the tree's operand order.
So for packed input and for float input holding integers (a filterbank),
while the level-4 row values stay below $2^{24}$ on the plan's scale (true
for 8-bit data on all four presets; 16-bit input is untested) and every
level-0 box sum below $2^{31}$, the output is bit-identical to the
streaming tree engine for any chunk length and any switch level; the unit test checks this on the
`cryopaf_lband` and `meerkat_lband` geometry over three consecutive blocks,
float and packed, with and without box smearing, at levels 3, 4 and 5, and
with uneven chunks, and the hidden `[.skew_preset]` case on the full
presets (DM 0–2000, 32 768-sample blocks). Non-integer float input is summed
in double running sums and agrees within float rounding, as the GPU direct
path does (measured: within $1.5 \times 10^{-7}$ of the output scale on the
`cryopaf_lband` and `meerkat_lband` geometry).

## Window spreads

`dmt_tests "[.skew_report]"` prints these for the four benchmark presets
(DM 0–2000 on the piecewise Levin grid). Largest and mean spread per level,
in samples:

| level | `cryopaf_lband` (Δt 8877) | `meerkat_lband` (Δt 13 876) | `parkes_uwl` (Δt 253 891) |
| ---: | ---: | ---: | ---: |
| 1 | 2325 / 557 | 6134 / 1188 | 127 501 / 3397 |
| 2 | 1333 / 292 | 3114 / 634 | 49 906 / 1695 |
| 3 | 743 / 149 | 1610 / 328 | 30 103 / 837 |
| 4 | 332 / 74 | 870 / 166 | 11 302 / 425 |
| 5 | 168 / 36 | 514 / 83 | 6451 / 230 |
| 6 | 86 / 17 | 278 / 41 | 3641 / 132 |
| 8 | 23 / 2.5 | 66 / 8.9 | 909 / 43 |
| 10 | 2 / 0.0 | 10 / 1.4 | 166 / 8.6 |
| 12 | – | 0 | 0 |

Summed over levels 4 to the root, the overlap a chunk would recompute (or,
here, copies in and out of the rings) relative to the rows' chunk work:

| chunk | `cryopaf_lband` | `meerkat_lband` | `parkes_uwl` |
| ---: | ---: | ---: | ---: |
| 1024 | 2.2% | 4.4% | 19.8% |
| 4096 | 0.6% | 1.1% | 5.0% |

(at level 3: 1.0% / 1.9% / 8.4% at chunk 4096; at level 5: 0.3% / 0.6% /
3.0%). These match the parent analysis.

## Memory

From `estimate_skew_memory` (the engine allocates exactly these), 8
threads, GB = $10^9$ bytes. "ring" is the raw input ring (1 byte per sample
for packed 8-bit input, 4 for float), "state" the two level halves
(`rows × (chunk + spread)` floats at the widest level), "rings" the per-row
window overlaps. The tree engine's figures are its two halves plus its
level-0 and tree history.

| preset, chunk, level | ring | state | rings | total (8-bit) | total (float) | tree state + history |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cryopaf_lband`, 4096, 4 | 0.01 | 0.30 | 0.005 | 0.32 | 0.37 | 0.33 + 0.07 = 0.39 |
| `meerkat_lband`, 4096, 4 | 0.07 | 0.48 | 0.018 | 0.58 | 0.81 | 0.59 + 0.26 = 0.85 |
| `meerkat_lband`, 1024, 4 | 0.06 | 0.13 | 0.018 | 0.22 | 0.41 | 0.15 + 0.26 = 0.41 |
| `parkes_uwl`, 1024, 4 | 1.70 | 2.78 | 0.73 | **5.3** | 10.5 | 2.13 + 10.14 = 12.3 |
| `parkes_uwl`, 2048, 4 | 1.70 | 4.74 | 0.73 | **7.3** | 12.4 | 4.23 + 10.14 = 14.4 |
| `parkes_uwl`, 4096, 4 | 1.72 | 8.66 | 0.73 | 11.2 | 16.5 | 8.54 + 10.14 = 18.7 |
| `parkes_uwl`, 4096, 5 | 1.72 | 7.12 | 0.33 | 9.4 | 14.5 | 8.54 + 10.14 = 18.7 |
| `parkes_uwl`, 4096, 3 | 1.72 | 10.0 | 1.58 | 13.4 | 18.5 | 8.54 + 10.14 = 18.7 |

The history question is answered: at level 4 the 10.1 GB UWL tree history
becomes 0.73 GB of rings, and the cross-chunk state no longer scales with
$\Delta t_{\max}^2$ or trials × $\Delta t_{\max}$. What remains is the
chunk state itself, 240 k level-4 rows times the chunk, which the skewed
windows widen by the mean spread (425 samples at level 4). **`parkes_uwl`
fits under 10 GB with 8-bit input at chunk 1024 (5.3 GB) or 2048 (7.3 GB)**,
and at chunk 4096 only from level 5 (9.4 GB). With float input the ring
alone is 6.8 GB, so UWL needs the packed path.

Measured peak RSS (`dmt_bench_mem`, one engine per process, one block):

| preset, chunk | input | prototype: engine / peak RSS | tree: engine / peak RSS |
| :--- | :--- | ---: | ---: |
| `cryopaf_lband`, 4096 | float | 356 / 781 MiB | 418 / 833 MiB |
| `cryopaf_lband`, 4096 | 8-bit | 309 / 673 MiB | 418 / 725 MiB |
| `meerkat_lband`, 4096 | float | 775 / 1862 MiB | 876 / 1940 MiB |
| `meerkat_lband`, 4096 | 8-bit | 556 / 1259 MiB | 876 / 1555 MiB |
| `parkes_uwl`, 1024 | 8-bit | 5066 / 6667 MiB | 12 269 / 8861 MiB |
| `parkes_uwl`, 2048 | 8-bit | 6944 / 7603 MiB | – (12.3 GB + 2.1 GB of state) |

"engine" is what the engine reports allocating: for the tree that includes
its plan tables, for the prototype only its buffers (its leaf and op
tables, about 150 MB for UWL, are not counted). The RSS includes the
input, the output, the plan tables (0.5 GB for UWL) and the benchmark
process. The tree's UWL RSS is below its allocation because one block
touches only the first 32 768 slots of each history ring; it grows to the
full 12.3 GB over a stream. The prototype's RSS is its whole working set.

## Throughput

`dmt_bench --benchmark_filter="bm_(tree|skew)"`, Apple M5 (10 cores),
32 768-sample blocks at DM 0–2000, steady-state streaming (one block per
iteration after a warm-up block), 8 threads; the tree engine with its
automatic fusion and `int_tree`, at its automatic chunk and at 4096. The
prototype keeps float rows above level 4 (no narrow integer levels), so on
packed input it moves more bytes per merge than the tree.

| preset | input, threads | tree, auto chunk | tree, chunk 4096 | prototype L4, chunk 4096 | L3 | L5 | L4, chunk 1024 |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cryopaf_lband` | float, 8 | 142 ms | 160 ms | 224 ms | 220 ms | 260 ms | – |
| `cryopaf_lband` | 8-bit, 8 | 124 ms | 137 ms | 226 ms | – | – | 247 ms |
| `cryopaf_lband` | float, 1 | – | 404 ms | 657 ms | – | – | – |
| `meerkat_lband` | float, 8 | 345 ms | 369 ms | 538 ms | 494 ms | 720 ms | – |
| `meerkat_lband` | 8-bit, 8 | 287 ms | 318 ms | 510 ms | – | – | 622 ms |
| `meerkat_lband` | float, 1 | – | 902 ms | 1718 ms | – | – | – |

The prototype is 1.4–1.6x slower than the tree at 8 threads. Its phase
timers say where (`meerkat_lband`, float, 8 threads, 538 ms per block):

| phase | ms | note |
| :--- | ---: | :--- |
| ring writes | 13 | |
| running-sum stripes | 38 | serial per channel; 263 ms on one thread |
| level-4 rows from leaves | 210 | 16 leaves x 2 loads per sample per row, from L2 |
| merges, levels 5–12 | 262 | 8 unfused levels through DRAM, ~44 GB per block |

Two things explain the gap, and neither is the skewed windows themselves
(the per-row overlap is copied in and out, about twice the recompute
fraction above: 1.2% of the rows' work for `cryopaf_lband`, 2.2% for
`meerkat_lband` and 10% for UWL at chunk 4096):

- The merges above level 4 run level by level through memory, as the
  unfused tree would. The tree engine fuses levels 1–6/7 in cache-resident
  time tiles, so its levels 5–7 never reach DRAM. Fusion is orthogonal to
  the windows and could be applied above the switch level in the same way
  (the halo per row is its spread).
- Evaluating a level-4 row from its 16 leaves costs 32 loads per sample per
  row, where the tree computes each level-0..3 row once and reuses it
  (about 12 accesses per level-4 row-sample). On a GPU that trade pays
  (shared memory against DRAM, see the direct levels in
  [performance](performance.md)); on the CPU the tree's time-tiled fusion
  already keeps those levels in L1/L2, so the direct bottom is slower.
  Level 3 (8 leaves) is faster than level 4 here and level 5 slower, as
  expected from the leaf count.

On `parkes_uwl` at chunk 1024 (one block, construction included, 8-bit
input), the prototype took 10.6 s against the tree's 13.8 s (9.2 s at
chunk 2048): there the
tree's history traffic dominates, and the ring's 1.7 GB running sums are
rebuilt per chunk (20% of the rows' work) without hurting.

## Limits of the prototype

- One beam, `execute()` only: no stepper, subband views, search sink or
  `save_history` / `load_history`. The state it would save is the raw ring
  plus the rings, a different format from the tree's.
- Levels above the switch level are float. The integer tree (`int_tree`)
  of the packed path is not reproduced above level 4, only its values.
- The two level halves are sized for the widest level including its
  spreads. Because the halves, not the history, now bound UWL, the next
  step for memory is a narrower chunk at the low levels (or running the
  level-4 subbands in groups), and for the GPU the same structure with the
  ring and stripes in device memory.
- Throughput on the CPU needs fusion above the switch level, and a bottom
  that computes each level-0..3 row once (a depth-first walk of each
  level-4 subband's rows in DM order, whose window unions are short)
  instead of per-leaf evaluation. On the GPU the direct bottom is already
  the fast path, so the windows plug into it as they are.
